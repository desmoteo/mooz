"""Stereo image: mono low end, width and left/right balance."""

from __future__ import annotations

import numpy as np
from scipy import signal


def mid_side(x: np.ndarray, fs: float, mono_below_hz: float, width: float,
             ramp_hz: tuple[float, float] | None = None) -> np.ndarray:
    """Remove the side signal below mono_below_hz (zero-phase) and scale the rest by width.
    ramp_hz = (lo, hi): instead, mono below lo, full width above hi and widening gradually
    (linearly in log frequency) between them, as mix engineers pan (De Man 2017)."""
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])
    if ramp_hz:
        from . import eq

        grid = eq.log_grid(fs)
        lo, hi = ramp_hz
        amp = np.clip(np.log(grid / lo) / np.log(hi / lo), 0, 1)
        side = eq.apply_fir(side[:, None], eq.design_fir(grid, 20 * np.log10(np.maximum(amp, 1e-3)), fs, 8191))[:, 0]
    elif mono_below_hz > 0:
        sos = signal.butter(2, mono_below_hz, "low", fs=fs, output="sos")
        side = side - signal.sosfiltfilt(sos, side)
    if width != 1.0:
        side = side * width
    return np.column_stack([mid + side, mid - side])


def rebalance(x: np.ndarray, strength: float, max_db: float) -> tuple[np.ndarray, float, float]:
    """Pull the left/right energy ratio towards centre.

    Returns the corrected signal, the measured L-R difference and the applied
    correction (both in dB).
    """
    pl, pr = np.mean(x[:, 0] ** 2), np.mean(x[:, 1] ** 2)
    if pl <= 0 or pr <= 0:
        return x, 0.0, 0.0
    diff = float(10 * np.log10(pl / pr))
    corr = float(np.clip(strength * diff, -max_db, max_db))
    g = 10 ** (corr / 40)
    return x * np.array([1 / g, g]), round(diff, 2), round(corr, 2)
