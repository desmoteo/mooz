"""Element-wise mathematics on whole signals, on threads.

A detector or a gain curve is one value per sample, five minutes of them 14 million, and numpy
works through each of its operations on one thread, making an array at every step. numexpr
evaluates the whole expression in one pass on several threads, with the same functions and in
the same order, so the samples come out the same (on the band's takes to the last bit) and five
to six times sooner. Short arrays go through numpy: starting the threads would cost more.

The expressions are written out here once, in dB arithmetic's own terms, and used by name.
"""

from __future__ import annotations

import numpy as np

from . import loudness

SHORT = 1 << 15  # elements; below this numpy is as fast
_ne = None       # numexpr, loaded at the first call


def _numexpr():
    global _ne
    if _ne is None:
        import numexpr

        numexpr.set_num_threads(max(1, min(loudness.THREADS, numexpr.detect_number_of_cores())))
        _ne = numexpr
    return _ne


def threads(n: int) -> None:
    """The number of threads from now on (a worker process has its share)."""
    _numexpr().set_num_threads(max(1, n))


def _run(expr: str, **names):
    return _numexpr().evaluate(expr, local_dict=names, global_dict={})


def linear(g_db: np.ndarray) -> np.ndarray:
    """The gain of a curve in dB: 10 ** (g_db / 20)."""
    return 10 ** (g_db / 20) if g_db.size < SHORT else _run("10 ** (g / 20)", g=g_db)


def db(a: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """An amplitude in dB: 20 * log10(a + floor)."""
    return 20 * np.log10(a + floor) if a.size < SHORT else _run("20 * log10(a + f)", a=a, f=floor)


def power_db(p: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """A power in dB: 10 * log10(p + floor)."""
    return 10 * np.log10(p + floor) if p.size < SHORT else _run("10 * log10(p + f)", p=p, f=floor)


def ratio_db(a: np.ndarray, b: np.ndarray, floor: float) -> np.ndarray:
    """One amplitude against another in dB: 20 * log10((a + floor) / (b + floor))."""
    if a.size < SHORT:
        return 20 * np.log10((a + floor) / (b + floor))
    return _run("20 * log10((a + f) / (b + f))", a=a, b=b, f=floor)


def gained(x: np.ndarray, g_db: np.ndarray) -> np.ndarray:
    """x (samples x channels) with a gain curve in dB, one value per sample: x * 10 ** (g_db / 20).
    The curve is worked out in 64 bits; float32 samples stay float32."""
    if x.dtype == np.float32:
        return times(x, linear(g_db))
    if x.size < SHORT:
        return x * (10 ** (g_db / 20))[:, None]
    return _run("x * 10 ** (g / 20)", x=x, g=g_db[:, None])


def times(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """x (samples x channels) times a factor per sample, in x's own precision: x * k[:, None]."""
    k = k.astype(x.dtype, copy=False)[:, None]
    return x * k if x.size < SHORT else _run("x * k", x=x, k=k)


def knee(over: np.ndarray, slope: float, knee_db: float) -> np.ndarray:
    """A soft-knee compressor's curve above its knee's lower edge (dynamics.gain_reduction):
    slope * over past the knee, the parabola slope * (over + knee/2)**2 / (2 knee) inside it."""
    half = knee_db / 2
    if over.size < SHORT:
        return np.where(over > half, slope * over, slope * (over + half) ** 2 / (2 * knee_db))
    return _run("where(over > half, slope * over, slope * (over + half) ** 2 / (2 * knee))", over=over, half=half, slope=slope,
                knee=knee_db)
