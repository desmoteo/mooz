"""Tonal balance: measure the long-term spectrum, compare it with a target
curve and build one linear-phase FIR from the (smoothed, limited) correction
plus any manual EQ moves.

Spectra are 1/3-octave band powers on a 1/12-octave grid, in dB. Pink noise is
flat on this scale; a well-balanced rock/pop master tilts down by roughly
1.5 dB per octave (the familiar "4.5 dB/oct" slope of a narrow-band analyser).
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

PER_OCT = 12
REPORT_FREQS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def log_grid(fs: float, fmin: float = 20.0) -> np.ndarray:
    n = int(np.floor(PER_OCT * np.log2(0.95 * fs / 2 / fmin))) + 1
    return fmin * 2 ** (np.arange(n) / PER_OCT)


def band_spectrum(x: np.ndarray, fs: float, grid: np.ndarray, bw_oct: float = 1 / 3) -> np.ndarray:
    """Long-term 1/3-octave band power (dB) centred on each grid frequency."""
    psd = None
    for ch in range(x.shape[1]):
        f, p = signal.welch(x[:, ch], fs, nperseg=16384)
        psd = p if psd is None else psd + p
    cum = np.concatenate([[0.0], np.cumsum(psd / x.shape[1])])
    lo = np.searchsorted(f, grid * 2 ** (-bw_oct / 2))
    hi = np.maximum(np.searchsorted(f, grid * 2 ** (bw_oct / 2)), lo + 1)
    return 10 * np.log10(cum[hi] - cum[lo] + 1e-30)


def target_curve(grid: np.ndarray, tilt_db_per_oct: float, hf_rolloff_hz: float, hf_rolloff_db_per_oct: float) -> np.ndarray:
    t = tilt_db_per_oct * np.log2(grid / 1000)
    return t + np.where(grid > hf_rolloff_hz, hf_rolloff_db_per_oct * np.log2(grid / hf_rolloff_hz), 0.0)


def smooth(values: np.ndarray, width_oct: float) -> np.ndarray:
    """Gaussian smoothing along the log-frequency grid (width = FWHM in octaves)."""
    return gaussian_filter1d(values, width_oct * PER_OCT / 2.355, mode="nearest")


def align(curve: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift curve by a constant so its mean over mask matches ref."""
    return curve - np.mean(curve[mask] - ref[mask])


def auto_correction(grid: np.ndarray, measured_db: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (correction_db, target_db aligned to the measurement)."""
    fmin, fmax = cfg["fmin_hz"], cfg["fmax_hz"]
    inside = (grid >= fmin) & (grid <= fmax)
    target = align(
        target_curve(grid, cfg["tilt_db_per_oct"], cfg["hf_rolloff_hz"], cfg["hf_rolloff_db_per_oct"]),
        measured_db,
        inside,
    )
    diff = target - measured_db
    # Out-of-range values (e.g. below a recorder's low-cut) must not leak into
    # the smoothing, so hold the edge values outside the range.
    first, last = np.flatnonzero(inside)[[0, -1]]
    diff[:first], diff[last + 1 :] = diff[first], diff[last]
    corr = np.clip(cfg["strength"] * smooth(diff, cfg["smoothing_oct"]), -cfg["max_cut_db"], cfg["max_boost_db"])
    # Fade the correction out over half an octave beyond the range.
    dist = np.maximum(np.log2(fmin / grid), np.log2(grid / fmax)).clip(0, 0.5) / 0.5
    corr *= 0.5 * (1 + np.cos(np.pi * dist))
    return smooth(corr, 1 / 3), target


def _biquad(kind: str, f0: float, gain_db: float, q: float, fs: float) -> np.ndarray:
    """RBJ Audio EQ Cookbook biquad as one SOS row."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    cw, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    if kind in ("high_pass", "low_pass"):
        s = 1 if kind == "high_pass" else -1
        b = [(1 + s * cw) / 2, -s * (1 + s * cw), (1 + s * cw) / 2]
        den = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        den = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    elif kind in ("low_shelf", "high_shelf"):
        s = 1 if kind == "low_shelf" else -1
        sq = 2 * np.sqrt(a) * alpha
        b = [a * ((a + 1) - s * (a - 1) * cw + sq), s * 2 * a * ((a - 1) - s * (a + 1) * cw), a * ((a + 1) - s * (a - 1) * cw - sq)]
        den = [(a + 1) + s * (a - 1) * cw + sq, -s * 2 * ((a - 1) + s * (a + 1) * cw), (a + 1) + s * (a - 1) * cw - sq]
    else:
        raise ValueError(f"EQ: unknown type {kind!r} (use peak, low_shelf, high_shelf, high_pass or low_pass)")
    return np.array(b + den) / den[0]


def manual_curve(grid: np.ndarray, bands: list[dict], fs: float) -> np.ndarray:
    total = np.zeros_like(grid)
    for b in bands:
        sos = _biquad(b["type"], b["freq_hz"], b.get("gain_db", 0.0), b.get("q", 0.707), fs)[None, :]
        _, h = signal.sosfreqz(sos, worN=grid, fs=fs)
        total += 20 * np.log10(np.abs(h) + 1e-12)
    return total


def design_fir(grid: np.ndarray, gain_db: np.ndarray, fs: float, numtaps: int) -> np.ndarray:
    nyq = fs / 2
    keep = grid < nyq
    f = np.concatenate([[0.0], grid[keep], [nyq]])
    g = np.concatenate([[gain_db[keep][0]], gain_db[keep], [gain_db[keep][-1]]])
    return signal.firwin2(numtaps | 1, f, 10 ** (g / 20), fs=fs)


def fir_response_db(h: np.ndarray, grid: np.ndarray, fs: float) -> np.ndarray:
    _, resp = signal.freqz(h, worN=grid, fs=fs)
    return 20 * np.log10(np.abs(resp) + 1e-12)


def apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Zero-latency linear-phase filtering (odd-length symmetric FIR)."""
    return signal.oaconvolve(x, h[:, None], mode="same", axes=0)


def at_freqs(grid: np.ndarray, curve: np.ndarray, freqs=REPORT_FREQS) -> dict[str, float]:
    return {f"{f:g}": round(float(np.interp(np.log2(f), np.log2(grid), curve)), 1) for f in freqs if f < grid[-1]}
