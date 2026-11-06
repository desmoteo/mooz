"""Loudness (ITU-R BS.1770-4 / EBU R128) and true-peak measurement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

EPS = 1e-12


def k_weighting_sos(fs: float) -> np.ndarray:
    """K-weighting filter (head shelf + RLB high-pass) for any sample rate.

    Derived the same way as libebur128; at 48 kHz it reproduces the
    coefficients published in BS.1770.
    """
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / fs)
    vh = 10 ** (gain_db / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [
        (vh + vb * k / q + k * k) / a0,
        2 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        1.0,
        2 * (k * k - 1) / a0,
        (1 - k / q + k * k) / a0,
    ]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / fs)
    a0 = 1 + k / q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([shelf, highpass])


def _segment_energy(x: np.ndarray, fs: float) -> tuple[np.ndarray, int]:
    """K-weighted energy, summed over channels, per 100 ms segment."""
    z = signal.sosfilt(k_weighting_sos(fs), x, axis=0)
    p = np.einsum("ij,ij->i", z, z)
    seg = int(round(0.1 * fs))
    n = len(p) // seg
    return p[: n * seg].reshape(n, seg).sum(axis=1), seg


def _windows(energy: np.ndarray, seg: int, nseg: int) -> np.ndarray:
    """Mean power of sliding windows nseg segments long, hopping one segment."""
    if len(energy) < nseg:
        return np.empty(0)
    c = np.concatenate([[0.0], np.cumsum(energy)])
    return (c[nseg:] - c[:-nseg]) / (nseg * seg)


def _lufs(power):
    return -0.691 + 10 * np.log10(np.maximum(power, EPS))


def _integrated(mom_power: np.ndarray) -> float:
    p = mom_power[_lufs(mom_power) > -70]
    if p.size == 0:
        return -np.inf
    p = p[_lufs(p) > _lufs(p.mean()) - 10]
    return float(_lufs(p.mean()))


def _lra(st_power: np.ndarray) -> float:
    p = st_power[_lufs(st_power) > -70]
    if p.size == 0:
        return 0.0
    st = _lufs(p[_lufs(p) > _lufs(p.mean()) - 20])
    return float(np.percentile(st, 95) - np.percentile(st, 10))


@dataclass
class Loudness:
    integrated: float  # LUFS
    lra: float  # LU
    momentary: np.ndarray  # LUFS, 400 ms windows every 100 ms
    short_term: np.ndarray  # LUFS, 3 s windows every 100 ms


def measure(x: np.ndarray, fs: float) -> Loudness:
    energy, seg = _segment_energy(x, fs)
    mom = _windows(energy, seg, 4)
    st = _windows(energy, seg, 30)
    return Loudness(_integrated(mom), _lra(st), _lufs(mom), _lufs(st))


def integrated(x: np.ndarray, fs: float) -> float:
    energy, seg = _segment_energy(x, fs)
    return _integrated(_windows(energy, seg, 4))


def active_mask(x: np.ndarray, fs: float, rel_lu: float = -20.0) -> np.ndarray:
    """True for samples inside 400 ms blocks no quieter than integrated + rel_lu."""
    energy, seg = _segment_energy(x, fs)
    mom = _windows(energy, seg, 4)
    if mom.size == 0:
        return np.ones(len(x), bool)
    loud = _lufs(mom) > _integrated(mom) + rel_lu
    seg_active = np.zeros(len(energy), bool)
    for offset in range(4):
        seg_active[offset : offset + len(loud)] |= loud
    mask = np.repeat(seg_active, seg)
    return np.pad(mask, (0, len(x) - len(mask)))


def oversampled_chunks(x: np.ndarray, factor: int, chunk: int = 1 << 17, margin: int = 64):
    """Yield (s, e, a, up): x[a:b] oversampled by factor, where x[s:e] is the chunk
    and up[(s-a)*factor:(e-a)*factor] is its exact (margin-protected) part."""
    n = len(x)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = max(0, s - margin), min(n, e + margin)
        yield s, e, a, signal.resample_poly(x[a:b], factor, 1, axis=0)


def true_peak_envelope(x: np.ndarray, factor: int = 4) -> np.ndarray:
    """Per-sample maximum |x| over channels and oversampled phases (BS.1770 true peak)."""
    env = np.empty(len(x))
    for s, e, a, up in oversampled_chunks(x, factor):
        part = np.abs(up[(s - a) * factor : (e - a) * factor])
        env[s:e] = part.reshape(e - s, -1).max(axis=1)
    np.maximum(env, np.abs(x).max(axis=1), out=env)
    return env


def true_peak_db(x: np.ndarray, factor: int = 4) -> float:
    return float(20 * np.log10(max(true_peak_envelope(x, factor).max(), EPS)))


def channel_balance_db(x: np.ndarray, fs: float) -> float:
    """Left minus right loudness in LU (0 for mono)."""
    if x.shape[1] != 2:
        return 0.0
    return integrated(x[:, :1], fs) - integrated(x[:, 1:], fs)


def summary(x: np.ndarray, fs: float) -> tuple[dict, Loudness]:
    ld = measure(x, fs)
    tp = true_peak_db(x)
    values = {
        "integrated_lufs": ld.integrated,
        "lra_lu": ld.lra,
        "true_peak_dbtp": tp,
        "plr_db": tp - ld.integrated,
        "balance_db": channel_balance_db(x, fs),
    }
    return {k: round(v, 1) + 0.0 for k, v in values.items()}, ld  # + 0.0 drops "-0.0"


def filtfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase filtering along time, one channel at a time (halves the
    temporary memory of scipy's sosfiltfilt on stereo signals)."""
    if x.ndim == 1:
        return signal.sosfiltfilt(sos, x)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = signal.sosfiltfilt(sos, x[:, c])
    return out
