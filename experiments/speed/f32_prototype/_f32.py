"""PROTOTYPE, in the scratch folder only: the audio kept in float32, to measure what that gives and costs.

MOOZ_F32 = "chains": the stem chains read their stems as float32 and every effect hands float32 on;
"all": the same for the blend, the bus and the master too; unset: nothing changes (float64, as the program is).
"Audio" is a 2-D array (samples x 1 or 2 channels); per-sample gains and other control signals are 1-D and
stay float64, and so does the arithmetic of the IIR filters (float64 coefficients, the result stored in
float32). Every function of the effect modules casts the audio it is given and the audio it returns, so
arithmetic written with float64 gains inside a function is still done in float64 and rounded when the
function returns: the rounding is that of a float32 program, the speed is a lower bound of one."""

import functools
import inspect
import os

import numpy as np

MODE = os.environ.get("MOOZ_F32", "")
active = MODE == "all"


def _audio(a) -> bool:
    return isinstance(a, np.ndarray) and a.dtype == np.float64 and a.ndim == 2 and a.shape[1] <= 2 and a.shape[0] >= 1024


def down(r):
    if _audio(r):
        return r.astype(np.float32)
    if isinstance(r, tuple):
        return tuple(down(v) for v in r)
    return r


def keep32(f):
    @functools.wraps(f)
    def g(*a, **k):
        if not active:
            return f(*a, **k)
        return down(f(*(down(v) for v in a), **{n: down(v) for n, v in k.items()}))

    return g


def wrap(*modules) -> None:
    for m in modules:
        for name, f in list(vars(m).items()):
            if inspect.isfunction(f) and f.__module__ == m.__name__ and not inspect.isgeneratorfunction(f):
                setattr(m, name, keep32(f))
