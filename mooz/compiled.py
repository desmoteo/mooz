"""Loops over the samples, compiled (numba). Imported at the first call of what uses them:
numba takes a second to load, which a run pays once and `mooz --help` should not. What is
compiled is kept on disk next to this file (or in numba's own cache where that is not writable),
so only the first run after a change compiles."""

from __future__ import annotations

import numba


@numba.njit(cache=True, nogil=True)
def hold(v, r, out):
    """Per row: out[i] = max(v[i], r * out[i - 1]), v floored at 1e-6 (dynamics.release_hold)."""
    for k in range(v.shape[0]):
        e = 0.0
        for i in range(v.shape[1]):
            s = v[k, i] if v[k, i] > 1e-6 else 1e-6
            e *= r
            if s > e:
                e = s
            out[k, i] = e


@numba.njit(cache=True, nogil=True)
def smooth(v, a, out):
    """Per row, a one-pole low-pass: out[i] = (1 - a) * v[i] + a * out[i - 1], from rest
    (dynamics.attack_smooth; the same samples as scipy's lfilter([1 - a], [1, -a]) gave)."""
    b = 1.0 - a
    for k in range(v.shape[0]):
        y = 0.0
        for i in range(v.shape[1]):
            y = b * v[k, i] + a * y
            out[k, i] = y
