from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import cosh, isfinite, sinh
from time import perf_counter
from typing import Callable, Iterator

Funcion = Callable[[float], float]
PasoBase = tuple[float, float | None, float | None]


@dataclass
class Paso:
    k: int
    x: float
    fx: float
    error_abs: float
    error_rel: float
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


class MetodoRaiz(ABC):
    @abstractmethod
    def iterar(
        self, f: Funcion, x0: float, x1: float,
        df: Funcion | None
    ) -> Iterator[PasoBase]:
        pass


class Biseccion(MetodoRaiz):
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


def resolver(
    metodo: MetodoRaiz,
    f: Funcion,
    x0: float,
    x1: float,
    df: Funcion | None = None,
    tol_x: float = 1e-9,
    tol_f: float = 1e-10,
    max_iter: int = 200
) -> Resultado:
    if tol_x <= 0 or tol_f <= 0 or max_iter < 1:
        raise ValueError("Tolerancias e iteraciones invalidas")

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
            historial.append(
                Paso(k, x, fx, ea, er, p1, p2)
            )

            if fx == 0 or (
                abs(fx) <= tol_f
                and ea <= tol_x * max(1.0, abs(x))
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
        raiz, False, motivo, historial,
        perf_counter() - inicio
    )


def catenaria(
    a: float, L: float = 10.0, h: float = 2.0
) -> float:
    if a <= 0 or L <= 0 or h <= 0:
        raise ValueError("Parametros fisicos invalidos")
    return cosh(L / (2 * a)) - 1 - h / a


def dcatenaria(
    a: float, L: float = 10.0, h: float = 2.0
) -> float:
    return (
        h - (L / 2) * sinh(L / (2 * a))
    ) / (a * a)


def van_der_waals(
    V: float, P: float, T: float,
    a_v: float, b: float,
    R: float = 8.314462618
) -> float:
    if V <= b or b <= 0 or P <= 0 or T <= 0:
        raise ValueError("Estado fuera del dominio fisico")
    return (P + a_v / V**2) * (V - b) - R * T


def equilibrio_termico(
    T: float, Q: float, T_a: float,
    hA: float, gamma: float
) -> float:
    if T <= 0 or T_a <= 0 or hA < 0 or gamma < 0:
        raise ValueError("Parametros termicos invalidos")
    return (
        hA * (T - T_a)
        + gamma * (T**4 - T_a**4)
        - Q
    )


if __name__ == "__main__":
    metodos = [
        Biseccion(), FalsaPosicion(),
        Newton(), Secante()
    ]
    for metodo in metodos:
        r = resolver(
            metodo, catenaria, 5.0, 10.0, dcatenaria
        )
        print(
            type(metodo).__name__,
            r.convergio, r.raiz,
            len(r.historial),
            round(r.tiempo_s, 6)
        )