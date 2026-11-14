"""Trimming leading/trailing silence and fades."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_opening


def find_bounds(
    x: np.ndarray,
    fs: float,
    integrated_lufs: float,
    threshold_rel_db: float,
    min_active_s: float,
    pre_roll_s: float,
    post_roll_s: float,
) -> tuple[int, int]:
    """Sample range from the first to the last sustained sound, plus margins.

    "Sound" is a 50 ms window louder than integrated_lufs + threshold_rel_db;
    it must last at least min_active_s, so record-button clicks and short
    noises in the silence around the song are ignored.
    """
    win = int(0.05 * fs)
    n = len(x) // win
    if n == 0:
        return 0, len(x)
    lev = 10 * np.log10(np.mean(x[: n * win].reshape(n, win, -1) ** 2, axis=(1, 2)) + 1e-20)
    loud = lev > integrated_lufs + threshold_rel_db
    k = max(1, int(round(min_active_s / 0.05)))
    sustained = np.flatnonzero(binary_opening(loud, structure=np.ones(k, bool)))
    if sustained.size == 0:
        return 0, len(x)
    start = max(0, sustained[0] * win - int(pre_roll_s * fs))
    end = min(len(x), (sustained[-1] + 1) * win + int(post_roll_s * fs))
    return start, end


def _ramp(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)


def fade(x: np.ndarray, fs: float, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    """Raised-cosine fade in and out."""
    y = x.copy()
    n_in = min(len(y), int(fade_in_s * fs))
    if n_in > 1:
        y[:n_in] *= _ramp(n_in)[:, None]
    n_out = min(len(y), int(fade_out_s * fs))
    if n_out > 1:
        y[len(y) - n_out :] *= _ramp(n_out)[::-1, None]
    return y
