"""Loudness (ITU-R BS.1770-4 / EBU R128) and true-peak measurement."""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice

import numpy as np
from scipy import signal

EPS = 1e-12

# Threads for the loops that scipy and numpy run outside the interpreter's lock (the oversampling). They change
# no result: the same pieces are computed, in any order, and put together in theirs. MOOZ_THREADS sets the number
THREADS = max(1, int(os.environ.get("MOOZ_THREADS") or min(8, (os.cpu_count() or 2) // 2)))
_pool: tuple[int, int, ThreadPoolExecutor] | None = None


def pool() -> ThreadPoolExecutor | None:
    """This process's thread pool (None with THREADS = 1). A child process makes its own."""
    global _pool
    if THREADS <= 1:
        return None
    if _pool is None or _pool[:2] != (os.getpid(), THREADS):
        _pool = (os.getpid(), THREADS, ThreadPoolExecutor(THREADS, thread_name_prefix="mooz"))
    return _pool[2]


# numpy reduces along an axis of length 2 one pair at a time, six times slower than the two columns taken as
# arrays. These give the same values as the expressions they stand for, to the last bit.

def mono(x: np.ndarray) -> np.ndarray:
    """The mean of the channels: x.mean(axis=1)."""
    if x.ndim == 2 and x.shape[1] == 2:
        return (x[:, 0] + x[:, 1]) * 0.5
    return x.mean(axis=1)


def peak(x: np.ndarray) -> np.ndarray:
    """The largest magnitude of the channels, per sample: np.abs(x).max(axis=1)."""
    if x.ndim == 2 and x.shape[1] == 2:
        return np.maximum(np.abs(x[:, 0]), np.abs(x[:, 1]))
    return np.abs(x).max(axis=1)


def power(x: np.ndarray) -> np.ndarray:
    """The mean square of the channels, per sample: np.mean(x * x, axis=1)."""
    if x.ndim == 2 and x.shape[1] == 2:
        return (x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]) * 0.5
    return np.mean(x * x, axis=1)


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


def _k_weighted(x: np.ndarray, fs: float) -> np.ndarray:
    """x through the K-weighting filter, its channels on a thread each where there are threads
    (the same samples as the filter along the whole array gives)."""
    sos = k_weighting_sos(fs)
    ex = pool() if x.shape[1] > 1 else None
    if ex is None:
        return signal.sosfilt(sos, x, axis=0)
    z = np.empty(x.shape, dtype=np.result_type(x.dtype, np.float64))

    def channel(c: int) -> None:
        z[:, c] = signal.sosfilt(sos, x[:, c])

    for _ in ex.map(channel, range(x.shape[1])):
        pass
    return z


def _energy(z: np.ndarray, fs: float) -> tuple[np.ndarray, int]:
    """The energy of a K-weighted signal, summed over its channels, per 100 ms segment."""
    p = np.einsum("ij,ij->i", z, z)
    seg = int(round(0.1 * fs))
    n = len(p) // seg
    return p[: n * seg].reshape(n, seg).sum(axis=1), seg


def _segment_energy(x: np.ndarray, fs: float) -> tuple[np.ndarray, int]:
    """K-weighted energy, summed over channels, per 100 ms segment."""
    return _energy(_k_weighted(x, fs), fs)


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


def _measure(energy: np.ndarray, seg: int) -> Loudness:
    mom = _windows(energy, seg, 4)
    st = _windows(energy, seg, 30)
    return Loudness(_integrated(mom), _lra(st), _lufs(mom), _lufs(st))


def measure(x: np.ndarray, fs: float) -> Loudness:
    return _measure(*_segment_energy(x, fs))


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


def _spans(n: int, chunk: int, margin: int) -> list[tuple[int, int, int, int]]:
    return [(s, min(n, s + chunk), max(0, s - margin), min(n, s + chunk + margin)) for s in range(0, n, chunk)]


def oversampled_chunks(x: np.ndarray, factor: int, chunk: int = 1 << 17, margin: int = 64):
    """Yield (s, e, a, up): x[a:b] oversampled by factor, where x[s:e] is the chunk
    and up[(s-a)*factor:(e-a)*factor] is its exact (margin-protected) part. The chunks come
    in order; the next few are being oversampled on other threads meanwhile, so x must not
    change while they are read."""
    up = lambda sp: signal.resample_poly(x[sp[2] : sp[3]], factor, 1, axis=0)  # noqa: E731
    spans = iter(_spans(len(x), chunk, margin))
    ex = pool()
    if ex is None:
        for sp in spans:
            yield sp[0], sp[1], sp[2], up(sp)
        return
    ahead = deque((sp, ex.submit(up, sp)) for sp in islice(spans, THREADS))
    while ahead:
        sp, made = ahead.popleft()
        nxt = next(spans, None)
        if nxt is not None:
            ahead.append((nxt, ex.submit(up, nxt)))
        yield sp[0], sp[1], sp[2], made.result()


def true_peak_envelope(x: np.ndarray, factor: int = 4) -> np.ndarray:
    """Per-sample maximum |x| over channels and oversampled phases (BS.1770 true peak)."""
    def part(sp):
        s, e, a, b = sp
        up = np.abs(signal.resample_poly(x[a:b], factor, 1, axis=0)[(s - a) * factor : (e - a) * factor]).reshape(e - s, -1)
        m = up[:, 0].copy()  # the maximum over the phases and channels, column by column (see mono)
        for j in range(1, up.shape[1]):
            np.maximum(m, up[:, j], out=m)
        return m

    env = np.empty(len(x))
    spans = _spans(len(x), 1 << 17, 64)
    ex = pool()
    for sp, m in zip(spans, ex.map(part, spans) if ex else map(part, spans)):
        env[sp[0] : sp[1]] = m
    np.maximum(env, peak(x), out=env)
    return env


def true_peak_db(x: np.ndarray, factor: int = 4) -> float:
    return float(20 * np.log10(max(true_peak_envelope(x, factor).max(), EPS)))


def channel_balance_db(x: np.ndarray, fs: float) -> float:
    """Left minus right loudness in LU (0 for mono)."""
    if x.shape[1] != 2:
        return 0.0
    return integrated(x[:, :1], fs) - integrated(x[:, 1:], fs)


def summary(x: np.ndarray, fs: float) -> tuple[dict, Loudness]:
    z = _k_weighted(x, fs)  # once, for the loudness and for each channel's (channel_balance_db filters each again)
    ld = _measure(*_energy(z, fs))
    one = lambda c: _integrated(_windows(*_energy(z[:, c : c + 1], fs), 4))  # noqa: E731
    balance = one(0) - one(1) if x.shape[1] == 2 else 0.0
    del z
    tp = true_peak_db(x)
    values = {
        "integrated_lufs": ld.integrated,
        "lra_lu": ld.lra,
        "true_peak_dbtp": tp,
        "plr_db": tp - ld.integrated,
        "balance_db": balance,
    }
    return {k: round(v, 1) + 0.0 for k, v in values.items()}, ld  # + 0.0 drops "-0.0"


def filtfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase filtering along time, channel by channel (scipy's sosfiltfilt on a stereo
    signal needs more temporary memory), the channels on a thread each where there are threads:
    the same samples, a third less time, both channels' temporaries held at once. Where only the
    mono sum of the result is wanted, filter mono(x): half the work, and the same to rounding."""
    if x.ndim == 1:
        return signal.sosfiltfilt(sos, x)
    out = np.empty_like(x)

    def channel(c: int) -> None:
        out[:, c] = signal.sosfiltfilt(sos, x[:, c])

    ex = pool() if x.shape[1] > 1 else None
    for _ in ex.map(channel, range(x.shape[1])) if ex else map(channel, range(x.shape[1])):
        pass
    return out
