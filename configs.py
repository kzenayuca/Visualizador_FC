"""Prototipo del motor de raices y de los modelos fisicos.

Solo biblioteca estandar (Python >= 3.10). Los bloques estan
separados como los modulos previstos (modelos, metodos, motor,
analisis) para poder dividirlos despues sin cambiar la logica.
"""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import cosh, isfinite, log, sinh, sqrt
from statistics import median
from time import perf_counter
from typing import Callable, Iterator

Funcion = Callable[[float], float]
PasoBase = tuple[float, float | None, float | None]
R_GAS = 8.314462618        # J/(mol K)
SIGMA_SB = 5.670374419e-8  # W/(m^2 K^4)


# ---------------- Registro de resultados ----------------
@dataclass
class Paso:
    k: int
    x: float
    fx: float
    error_abs: float          # |x_k - x_(k-1)|
    error_rel: float
    error_verd: float | None  # |x_k - x*|, si hay raiz de referencia
    p1: float | None
    p2: float | None


@dataclass
class Resultado:
    raiz: float | None
    convergio: bool
    motivo: str
    historial: list[Paso]
    tiempo_s: float


def evaluar(f: Funcion, x: float) -> float:
    if not isfinite(x):
        raise ArithmeticError("Iterado no finito")
    valor = f(x)
    if not isfinite(valor):
        raise ArithmeticError("Residual no finito")
    return valor


# ---------------- Metodos (patron Strategy) ----------------
class MetodoRaiz(ABC):
    cerrado: bool = False  # True: usa intervalo; False: semillas

    @abstractmethod
    def iterar(
        self, f: Funcion, x0: float, x1: float,
        df: Funcion | None
    ) -> Iterator[PasoBase]:
        pass


class Biseccion(MetodoRaiz):
    cerrado = True

    def iterar(self, f, x0, x1, df):
        a, b = sorted((x0, x1))
        fa, fb = evaluar(f, a), evaluar(f, b)
        if fa == 0:
            yield a, a, b
            return
        if fb == 0:
            yield b, a, b
            return
        if fa * fb > 0:
            raise ValueError("No existe cambio de signo")
        while True:
            c = a + (b - a) / 2
            fc = evaluar(f, c)
            yield c, a, b
            if fc == 0:
                return
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc


class FalsaPosicion(MetodoRaiz):
    cerrado = True

    def iterar(self, f, x0, x1, df):
        a, b = sorted((x0, x1))
        fa, fb = evaluar(f, a), evaluar(f, b)
        if fa == 0:
            yield a, a, b
            return
        if fb == 0:
            yield b, a, b
            return
        if fa * fb > 0:
            raise ValueError("No existe cambio de signo")
        while True:
            denominador = fb - fa
            if denominador == 0:
                raise ArithmeticError("Denominador nulo")
            c = a - fa * (b - a) / denominador
            fc = evaluar(f, c)
            yield c, a, b
            if fc == 0:
                return
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc


class Newton(MetodoRaiz):
    def iterar(self, f, x0, x1, df):
        if df is None:
            raise ValueError("Newton requiere derivada")
        x = x0
        while True:
            fx = evaluar(f, x)
            if fx == 0:
                yield x, x, None
                return
            pendiente = evaluar(df, x)
            if abs(pendiente) < 1e-14:
                raise ArithmeticError("Derivada casi nula")
            nuevo = x - fx / pendiente
            yield nuevo, x, None
            x = nuevo


class Secante(MetodoRaiz):
    def iterar(self, f, x0, x1, df):
        f0, f1 = evaluar(f, x0), evaluar(f, x1)
        while True:
            if f1 == 0:
                yield x1, x0, x1
                return
            denominador = f1 - f0
            if denominador == 0:
                raise ArithmeticError("Denominador nulo")
            x2 = x1 - f1 * (x1 - x0) / denominador
            yield x2, x0, x1
            x0, x1 = x1, x2
            f0, f1 = f1, evaluar(f, x1)


METODOS = [Biseccion(), FalsaPosicion(), Newton(), Secante()]


# ---------------- Motor de resolucion ----------------
def resolver(
    metodo: MetodoRaiz,
    f: Funcion,
    x0: float,
    x1: float,
    df: Funcion | None = None,
    tol_x: float = 1e-9,
    tol_f: float = 1e-10,
    max_iter: int = 200,
    escala_x: float = 1.0,
    escala_f: float = 1.0,
    raiz_ref: float | None = None,
) -> Resultado:
    if tol_x <= 0 or tol_f <= 0 or max_iter < 1:
        raise ValueError("Tolerancias e iteraciones invalidas")
    if escala_x <= 0 or escala_f <= 0:
        raise ValueError("Las escalas deben ser positivas")

    inicio = perf_counter()
    historial: list[Paso] = []
    anterior: float | None = None
    motivo = "Limite de iteraciones"

    try:
        for k, (x, p1, p2) in enumerate(
            metodo.iterar(f, x0, x1, df), start=1
        ):
            fx = evaluar(f, x)
            ea = (
                float("inf") if anterior is None
                else abs(x - anterior)
            )
            er = ea / max(abs(x), 1e-12)
            ev = None if raiz_ref is None else abs(x - raiz_ref)
            historial.append(Paso(k, x, fx, ea, er, ev, p1, p2))

            if fx == 0 or (
                abs(fx) <= tol_f * escala_f
                and ea <= tol_x * max(escala_x, abs(x))
            ):
                return Resultado(
                    x, True, "Convergencia", historial,
                    perf_counter() - inicio
                )
            if k >= max_iter:
                break
            if anterior is not None and x == anterior:
                motivo = "Estancamiento numerico"
                break
            anterior = x

    except (
        ValueError, ArithmeticError,
        OverflowError, ZeroDivisionError
    ) as exc:
        motivo = str(exc)

    raiz = historial[-1].x if historial else None
    return Resultado(
        raiz, False, motivo, historial, perf_counter() - inicio
    )


# ---------------- Modelos fisicos ----------------
@dataclass(frozen=True)
class ModeloFisico:
    nombre: str
    f: Funcion
    df: Funcion | None
    unidad_x: str
    escala_x: float = 1.0   # magnitud tipica de la incognita
    escala_f: float = 1.0   # magnitud tipica del residual


@dataclass(frozen=True)
class Caso:
    nombre: str
    modelo: ModeloFisico
    intervalo: tuple[float, float]
    semillas: tuple[float, float]


def modelo_catenaria(L: float, h: float) -> ModeloFisico:
    if L <= 0 or h <= 0:
        raise ValueError("Parametros fisicos invalidos")

    def f(a: float) -> float:
        if a <= 0:
            raise ValueError("Parametro a fuera del dominio")
        return cosh(L / (2 * a)) - 1 - h / a

    def df(a: float) -> float:
        return (h - (L / 2) * sinh(L / (2 * a))) / (a * a)

    return ModeloFisico("Catenaria", f, df, "m", escala_x=L)


def modelo_van_der_waals(
    P: float, T: float, a_v: float, b: float
) -> ModeloFisico:
    if P <= 0 or T <= 0 or a_v < 0 or b <= 0:
        raise ValueError("Parametros fisicos invalidos")

    def f(V: float) -> float:
        if V <= b:
            raise ValueError("Volumen fuera del dominio (V <= b)")
        return (P + a_v / V**2) * (V - b) - R_GAS * T

    def df(V: float) -> float:
        return P - a_v / V**2 + 2 * a_v * b / V**3

    return ModeloFisico(
        "Van der Waals", f, df, "m3/mol",
        escala_x=b, escala_f=R_GAS * T
    )


def dPdV(V: float, T: float, a_v: float, b: float) -> float:
    """Estabilidad mecanica de vdW: (dP/dV)_T < 0."""
    return -R_GAS * T / (V - b) ** 2 + 2 * a_v / V**3


def modelo_termico(
    Q: float, T_a: float, A: float, h_c: float, eps: float
) -> ModeloFisico:
    if Q <= 0 or T_a <= 0 or A <= 0 or h_c < 0 or not 0 <= eps <= 1:
        raise ValueError("Parametros termicos invalidos")
    hA = h_c * A
    gamma = eps * SIGMA_SB * A

    def f(T: float) -> float:
        if T <= 0:
            raise ValueError("Temperatura fuera del dominio (T <= 0)")
        return hA * (T - T_a) + gamma * (T**4 - T_a**4) - Q

    def df(T: float) -> float:
        return hA + 4 * gamma * T**3

    return ModeloFisico(
        "Equilibrio termico", f, df, "K", escala_x=T_a, escala_f=Q
    )


def casos_base() -> list[Caso]:
    cat = modelo_catenaria(L=10.0, h=2.0)
    vdw = modelo_van_der_waals(4.5e6, 280.0, 0.3640, 4.267e-5)
    ter = modelo_termico(500.0, 293.15, 0.5, 10.0, 0.9)
    return [
        Caso("Catenaria", cat, (5.0, 10.0), (5.0, 10.0)),
        Caso("vdW liquido", vdw, (8.0e-5, 9.5e-5), (7.0e-5, 9.0e-5)),
        Caso("vdW intermedia", vdw, (1.0e-4, 1.8e-4), (1.2e-4, 1.6e-4)),
        Caso("vdW gas", vdw, (2.0e-4, 1.0e-3), (4.5e-4, 6.0e-4)),
        Caso("Termico", ter, (293.15, 600.0), (300.0, 350.0)),
    ]


# ---------------- Analisis ----------------
def buscar_cambios_signo(
    f: Funcion, a: float, b: float, n: int = 2000
) -> list[tuple[float, float]]:
    """Barrido uniforme: subintervalos [x_i, x_i+1] con f*f < 0."""
    salida: list[tuple[float, float]] = []
    previo: tuple[float, float] | None = None
    for i in range(n + 1):
        x = a + i * (b - a) / n
        try:
            fx = evaluar(f, x)
        except (ValueError, ArithmeticError, OverflowError):
            previo = None
            continue
        if fx == 0:
            salida.append((x, x))
        elif previo is not None and previo[1] * fx < 0:
            salida.append((previo[0], x))
        previo = (x, fx)
    return salida


def raiz_referencia(f: Funcion, a: float, b: float) -> float:
    """Biseccion hasta agotar la precision de coma flotante."""
    fa, fb = evaluar(f, a), evaluar(f, b)
    if fa * fb > 0:
        raise ValueError("No existe cambio de signo")
    for _ in range(200):
        c = a + (b - a) / 2
        if c == a or c == b:
            break
        fc = evaluar(f, c)
        if fc == 0:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return a + (b - a) / 2


def orden_convergencia(
    historial: list[Paso], piso_rel: float = 1e-12
) -> list[float]:
    """p_k = ln(e_(k+1)/e_k) / ln(e_k/e_(k-1)) con error verdadero."""
    if not historial:
        return []
    errores = [p.error_verd for p in historial]
    piso = piso_rel * abs(historial[-1].x)
    ordenes: list[float] = []
    for e0, e1, e2 in zip(errores, errores[1:], errores[2:]):
        if None in (e0, e1, e2) or min(e0, e1, e2) <= piso or e0 == e1:
            continue
        ordenes.append(log(e2 / e1) / log(e1 / e0))
    return ordenes


def orden_estimado(historial: list[Paso]) -> float | None:
    ordenes = orden_convergencia(historial)
    return median(ordenes) if ordenes else None


def exportar_csv(resultado: Resultado, ruta: str) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        w = csv.writer(archivo)
        w.writerow([
            "k", "x", "fx", "error_abs", "error_rel",
            "error_verd", "p1", "p2"
        ])
        for p in resultado.historial:
            w.writerow([
                p.k, p.x, p.fx, p.error_abs, p.error_rel,
                p.error_verd, p.p1, p.p2
            ])


def comparar_metodos(
    caso: Caso, raiz_ref: float | None = None, **opciones
) -> dict[str, Resultado]:
    """Ejecuta los cuatro metodos sobre un caso."""
    m = caso.modelo
    if raiz_ref is None:
        raiz_ref = raiz_referencia(m.f, *caso.intervalo)
    resultados = {}
    for metodo in METODOS:
        x0, x1 = caso.intervalo if metodo.cerrado else caso.semillas
        resultados[type(metodo).__name__] = resolver(
            metodo, m.f, x0, x1, m.df,
            escala_x=m.escala_x, escala_f=m.escala_f,
            raiz_ref=raiz_ref, **opciones
        )
    return resultados


def autoprueba() -> None:
    """Verificacion minima con x^2 - 2 en [1, 2]."""
    for metodo in METODOS:
        r = resolver(metodo, lambda x: x * x - 2, 1.0, 2.0,
                     lambda x: 2 * x)
        assert r.convergio and abs(r.raiz - sqrt(2)) < 1e-8, metodo


if __name__ == "__main__":
    autoprueba()
    print(f"{'Caso':15}{'Metodo':14}{'Conv':6}{'Iter':>5}"
          f"{'Err. verd.':>12}{'Orden':>7}{'t [s]':>10}")
    for caso in casos_base():
        for nombre, r in comparar_metodos(caso).items():
            ev = r.historial[-1].error_verd if r.historial else None
            p = (None if nombre == "Biseccion"
                 else orden_estimado(r.historial))
            print(
                f"{caso.nombre:15}{nombre:14}{str(r.convergio):6}"
                f"{len(r.historial):>5}"
                f"{'--' if ev is None else f'{ev:.2e}':>12}"
                f"{'--' if p is None else f'{p:.2f}':>7}"
                f"{r.tiempo_s:>10.6f}"
            )