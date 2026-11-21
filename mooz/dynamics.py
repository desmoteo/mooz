"""Compressor, multiband compressor, soft clipper and true-peak limiter.

Gain smoothing follows the "smooth decoupled" peak detector of Giannoulis,
Massberg & Reiss, "Digital Dynamic Range Compressor Design" (JAES 2012),
run on the gain reduction in dB. Its release stage y[n] = max(x[n], r*y[n-1])
is one compiled loop over the samples (numba), so whole songs are processed
without a Python loop. Since the files are processed offline, detectors may
look ahead in time.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from . import expr, stereo
from .loudness import EPS, filtfilt, oversampled_chunks, power, true_peak_envelope


def release_hold(v: np.ndarray, release_ms: float, fs: float, axis: int = -1) -> np.ndarray:
    """Peak-hold a non-negative signal (along axis) with an exponential release:
    y[n] = max(x[n], r*y[n-1]), x floored at 1e-6. One compiled pass over the samples (it was
    a cumulative maximum in the log domain, four times slower and the same to 1e-12 relative)."""
    r = float(np.exp(-1000.0 / (release_ms * fs)))
    rows = np.ascontiguousarray(np.moveaxis(np.asarray(v, dtype=np.float64), axis, -1))
    out = np.empty_like(rows)
    from . import compiled  # (numba: loaded at the first call)

    compiled.hold(rows.reshape(-1, rows.shape[-1]), r, out.reshape(-1, rows.shape[-1]))
    return np.moveaxis(out, -1, axis)


def attack_smooth(v: np.ndarray, attack_ms: float, fs: float, axis: int = -1) -> np.ndarray:
    """One-pole smoothing (along axis) with the time constant attack_ms: one compiled pass (it was
    scipy's lfilter, three times slower and the same to the last bit)."""
    a = float(np.exp(-1000.0 / (attack_ms * fs)))
    rows = np.ascontiguousarray(np.moveaxis(np.asarray(v, dtype=np.float64), axis, -1))
    out = np.empty_like(rows)
    from . import compiled  # (numba: loaded at the first call)

    compiled.smooth(rows.reshape(-1, rows.shape[-1]), a, out.reshape(-1, rows.shape[-1]))
    return np.moveaxis(out, -1, axis)


def envelope(a: np.ndarray, fs: float, attack_ms: float, release_ms: float) -> np.ndarray:
    """Peak envelope follower of a non-negative signal."""
    return attack_smooth(release_hold(a, release_ms, fs), attack_ms, fs)


def gain_reduction(level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    """Static soft-knee compressor curve, as dB of gain reduction (>= 0)."""
    over = level_db - threshold_db
    slope = 1.0 - 1.0 / ratio
    knee = max(knee_db, 1e-6)
    gr = expr.knee(over, slope, knee)
    gr[over < -knee / 2] = 0.0
    return gr


def level_db(x: np.ndarray, fs: float, rms_ms: float) -> np.ndarray:
    """Stereo-linked RMS level over a centred window of rms_ms."""
    p = power(x)
    n = int(round(rms_ms * 1e-3 * fs))
    if n > 1:
        p = np.maximum(uniform_filter1d(p, n), 0.0)  # running mean can dip a hair below zero
    return expr.power_db(p, EPS)


def _reduction(lev: np.ndarray, use, fs: float, threshold_pct: float, ratio: float, knee_db: float,
               attack_ms: float, release_ms: float) -> tuple[np.ndarray, float]:
    """The smoothed gain reduction (dB) of a compressor for the level lev, and its threshold."""
    threshold = float(np.percentile(lev[use][::16], threshold_pct))
    gr = gain_reduction(lev, threshold, ratio, knee_db)
    return attack_smooth(release_hold(gr, release_ms, fs), attack_ms, fs), threshold


def average_reduction(lev: np.ndarray, fs: float, *, threshold_pct: float, ratio: float, knee_db: float, attack_ms: float,
                      release_ms: float, active: np.ndarray | None = None, **_) -> float:
    """compress()'s avg_gr_db for a signal of level lev, without the signal: all a search for the
    threshold needs. The level is the same at every step of such a search, and no gain is applied."""
    use = active if active is not None and active.any() else slice(None)
    gr, _ = _reduction(lev, use, fs, threshold_pct, ratio, knee_db, attack_ms, release_ms)
    return round(float(gr[use].mean()), 2)


def compress(
    x: np.ndarray,
    fs: float,
    *,
    threshold_pct: float,
    ratio: float,
    knee_db: float,
    attack_ms: float,
    release_ms: float,
    rms_ms: float,
    active: np.ndarray | None = None,
    auto_makeup: bool = False,
    level: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compress x. The threshold sits at the threshold_pct percentile of the
    signal's own level (over active samples), so a setting compresses the same
    share of the song regardless of how hot it was recorded. level: x's level
    (level_db with this rms_ms), where the caller has it already.

    auto_makeup restores the original energy, i.e. evens out dynamics without
    changing the long-term level (used per band so the EQ'd balance is kept).
    """
    lev = level_db(x, fs, rms_ms) if level is None else level
    use = active if active is not None and active.any() else slice(None)
    gr, threshold = _reduction(lev, use, fs, threshold_pct, ratio, knee_db, attack_ms, release_ms)
    y = expr.gained(x, -gr)
    makeup = 0.0
    if auto_makeup:
        makeup = float(10 * np.log10((np.sum(x * x) + EPS) / (np.sum(y * y) + EPS)))
        y *= 10 ** (makeup / 20)
    return y, gr, {
        "threshold_db": round(threshold, 2),
        "avg_gr_db": round(float(gr[use].mean()), 2),
        "max_gr_db": round(float(gr.max()), 2),
        "makeup_db": round(makeup, 2),
    }


def split_bands(x: np.ndarray, fs: float, crossovers_hz: list[float]):
    """Yield frequency bands, low to high, that sum back to x exactly.

    Each split is a zero-phase 2nd-order Butterworth low-pass (run forwards and
    backwards, so Linkwitz-Riley-like -6 dB at the crossover) and the remainder.
    """
    rest = x
    for fc in sorted(crossovers_hz):
        sos = signal.butter(2, fc, "low", fs=fs, output="sos")
        low = filtfilt(sos, rest)
        yield low
        rest = rest - low
    yield rest


BAND_PARAMS = ("threshold_pct", "ratio", "knee_db", "attack_ms", "release_ms", "rms_ms")


def multiband(
    x: np.ndarray,
    fs: float,
    crossovers_hz: list[float],
    bands: list[dict],
    active: np.ndarray,
    balance_strength: float = 0.0,
    balance_max_db: float = 0.0,
) -> tuple[np.ndarray, list[dict], dict[str, np.ndarray]]:
    if len(bands) != len(crossovers_hz) + 1:
        raise ValueError(f"multiband: {len(crossovers_hz)} crossovers need {len(crossovers_hz) + 1} bands, got {len(bands)}")
    edges = [0.0, *sorted(crossovers_hz), fs / 2]
    out = np.zeros_like(x)
    stats, curves = [], {}
    for i, cfg in enumerate(bands):
        missing = [k for k in BAND_PARAMS if k not in cfg]
        if missing:
            raise ValueError(f"multiband band {i + 1} is missing {', '.join(missing)}")
    for i, (cfg, band) in enumerate(zip(bands, split_bands(x, fs, crossovers_hz))):
        name = cfg.get("name", f"band{i + 1}")
        st = {"name": name, "range_hz": [edges[i], edges[i + 1]]}
        if x.shape[1] == 2 and balance_strength > 0:
            band, st["balance_measured_db"], st["balance_correction_db"] = stereo.rebalance(
                band, balance_strength, balance_max_db
            )
        y, gr, comp = compress(band, fs, **{k: cfg[k] for k in BAND_PARAMS}, active=active, auto_makeup=True)
        out += y
        stats.append(st | comp)
        curves[name] = gr
    return out, stats, curves


def soft_clip(x: np.ndarray, threshold: float, knee_db: float, factor: int = 4) -> tuple[np.ndarray, dict]:
    """Oversampled clipper: linear below the knee, saturating smoothly to threshold.

    Only the clipping residual goes through the resampler, so untouched parts of
    the signal stay bit-identical.
    """
    k = threshold * 10 ** (-knee_db / 20)
    y = x.copy()
    clipped, max_red = 0, 0.0
    for s, e, a, up in oversampled_chunks(x, factor):
        mag = np.abs(up)
        over = mag > k
        if not over.any():
            continue
        if knee_db > 0:
            tgt = np.where(over, k + (threshold - k) * np.tanh((mag - k) / (threshold - k)), mag)
        else:
            tgt = np.minimum(mag, threshold)
        delta = np.sign(up) * tgt - up
        y[s:e] += signal.resample_poly(delta, 1, factor, axis=0)[s - a : e - a]
        own = slice((s - a) * factor, (e - a) * factor)
        clipped += np.count_nonzero(over[own].reshape(e - s, -1).any(axis=1))
        m = over[own]
        if m.any():
            max_red = max(max_red, float(np.max(20 * np.log10(mag[own][m] / tgt[own][m]))))
    return y, {
        "threshold_dbfs": round(float(20 * np.log10(threshold)), 2),
        "clipped_pct": round(100 * clipped / len(x), 3),
        "max_reduction_db": round(max_red, 2),
    }


def limit(
    x: np.ndarray, fs: float, ceiling_db: float, lookahead_ms: float, release_ms: float, factor: int = 4
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lookahead true-peak limiter.

    The gain needed by every sample is held over the lookahead window and
    ramped in with a moving average of the same length, so the gain is fully
    down by the time each peak arrives; then it is released exponentially.
    """
    need = np.maximum(0.0, expr.db(true_peak_envelope(x, factor), EPS) - ceiling_db)
    la = max(2, int(round(lookahead_ms * 1e-3 * fs)) // 2 * 2)  # even: exact forward window
    held = maximum_filter1d(need, size=la + 1, origin=-(la // 2), mode="constant", cval=0.0)
    c = np.concatenate([np.zeros(la + 1), np.cumsum(held)])
    ramped = (c[la + 1 :] - c[: len(held)]) / (la + 1)
    gr = release_hold(ramped, release_ms, fs)
    y = expr.gained(x, -gr)
    return y, gr, {
        "max_gr_db": round(float(gr.max()), 2),
        "limited_pct": round(100 * float(np.mean(gr > 0.1)), 2),
    }


TRACE_HOP_S = 0.1  # time resolution of gain traces kept for the reports


def decimate(v: np.ndarray, fs: float, reduce=np.max, hop_s: float = TRACE_HOP_S) -> np.ndarray:
    """Reduce a per-sample curve to one value per hop_s (max by default)."""
    hop = int(hop_s * fs)
    n = len(v) // hop
    return reduce(v[: n * hop].reshape(n, hop), axis=1) if n else np.empty(0)
