"""Compressor, multiband compressor, soft clipper and true-peak limiter.

Gain smoothing follows the "smooth decoupled" peak detector of Giannoulis,
Massberg & Reiss, "Digital Dynamic Range Compressor Design" (JAES 2012),
run on the gain reduction in dB. Its release stage y[n] = max(x[n], r*y[n-1])
has the closed form y[n] = max_k r^(n-k) x[k], evaluated below with a cumulative
maximum in the log domain, so whole songs are processed without a Python loop.
Since the files are processed offline, detectors may look ahead in time.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from . import stereo
from .loudness import EPS, filtfilt, oversampled_chunks, true_peak_envelope


def release_hold(v: np.ndarray, release_ms: float, fs: float, axis: int = -1) -> np.ndarray:
    """Peak-hold a non-negative signal (along axis) with an exponential release."""
    c = -1000.0 / (release_ms * fs)  # ln(r)
    shape = [1] * v.ndim
    shape[axis] = v.shape[axis]
    ramp = (c * np.arange(v.shape[axis])).reshape(shape)
    out = np.maximum.accumulate(np.log(np.maximum(v, 1e-6)) - ramp, axis=axis)
    out += ramp
    return np.exp(out)


def attack_smooth(v: np.ndarray, attack_ms: float, fs: float, axis: int = -1) -> np.ndarray:
    a = np.exp(-1000.0 / (attack_ms * fs))
    return signal.lfilter([1 - a], [1, -a], v, axis=axis)


def envelope(a: np.ndarray, fs: float, attack_ms: float, release_ms: float) -> np.ndarray:
    """Peak envelope follower of a non-negative signal."""
    return attack_smooth(release_hold(a, release_ms, fs), attack_ms, fs)


def gain_reduction(level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    """Static soft-knee compressor curve, as dB of gain reduction (>= 0)."""
    over = level_db - threshold_db
    slope = 1.0 - 1.0 / ratio
    knee = max(knee_db, 1e-6)
    gr = np.where(over > knee / 2, slope * over, slope * (over + knee / 2) ** 2 / (2 * knee))
    gr[over < -knee / 2] = 0.0
    return gr


def level_db(x: np.ndarray, fs: float, rms_ms: float) -> np.ndarray:
    """Stereo-linked RMS level over a centred window of rms_ms."""
    p = np.mean(x * x, axis=1)
    n = int(round(rms_ms * 1e-3 * fs))
    if n > 1:
        p = np.maximum(uniform_filter1d(p, n), 0.0)  # running mean can dip a hair below zero
    return 10 * np.log10(p + EPS)


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
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compress x. The threshold sits at the threshold_pct percentile of the
    signal's own level (over active samples), so a setting compresses the same
    share of the song regardless of how hot it was recorded.

    auto_makeup restores the original energy, i.e. evens out dynamics without
    changing the long-term level (used per band so the EQ'd balance is kept).
    """
    lev = level_db(x, fs, rms_ms)
    use = active if active is not None and active.any() else slice(None)
    threshold = float(np.percentile(lev[use][::16], threshold_pct))
    gr = gain_reduction(lev, threshold, ratio, knee_db)
    gr = attack_smooth(release_hold(gr, release_ms, fs), attack_ms, fs)
    y = x * (10 ** (-gr / 20))[:, None]
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
    need = np.maximum(0.0, 20 * np.log10(true_peak_envelope(x, factor) + EPS) - ceiling_db)
    la = max(2, int(round(lookahead_ms * 1e-3 * fs)) // 2 * 2)  # even: exact forward window
    held = maximum_filter1d(need, size=la + 1, origin=-(la // 2), mode="constant", cval=0.0)
    c = np.concatenate([np.zeros(la + 1), np.cumsum(held)])
    ramped = (c[la + 1 :] - c[: len(held)]) / (la + 1)
    gr = release_hold(ramped, release_ms, fs)
    y = x * (10 ** (-gr / 20))[:, None]
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
