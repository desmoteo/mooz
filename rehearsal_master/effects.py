"""Per-track effects and the track chain.

A "track" is one separated stem (vocals, drums, bass, other) or, without
separation, the whole mix. process_track() runs, in order:

  width/balance -> EQ -> dynamic EQ -> resonance suppressor -> compressor
  -> transient shaper -> exciter -> bass enhancer -> gain

EQ is linear-phase with zero latency, so stems stay phase-aligned with each
other (and with their bleed in the other stems) when they are summed again.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from . import dynamics, eq, loudness, stereo
from .loudness import EPS, filtfilt


def _bandpass_sos(freq_hz: float, q: float, fs: float) -> np.ndarray:
    bw_oct = 2 / np.log(2) * np.arcsinh(1 / (2 * q))
    lo, hi = freq_hz * 2 ** (-bw_oct / 2), min(freq_hz * 2 ** (bw_oct / 2), 0.49 * fs)
    return signal.butter(1, [lo, hi], "bandpass", fs=fs, output="sos")


def _taper(f: np.ndarray, fmin: float, fmax: float, width_oct: float = 0.5) -> np.ndarray:
    """1 inside fmin..fmax, raised-cosine to 0 over width_oct outside."""
    with np.errstate(divide="ignore"):
        dist = np.maximum(np.log2(fmin / np.maximum(f, 1e-9)), np.log2(f / fmax)).clip(0, width_oct) / width_oct
    return 0.5 * (1 + np.cos(np.pi * dist))


# --- dynamic EQ -------------------------------------------------------------------------------

def dynamic_eq_shape(grid: np.ndarray, band: dict, cut_db: float, fs: float) -> np.ndarray:
    """Frequency response (dB) of a dynamic EQ band while it cuts by cut_db (< 0: boosts)."""
    _, h = signal.sosfreqz(_bandpass_sos(band["freq_hz"], band["q"], fs), worN=grid, fs=fs)
    return 20 * np.log10(np.maximum(1 - (1 - 10 ** (-cut_db / 20)) * np.abs(h) ** 2, 1e-6))


def dynamic_eq(x: np.ndarray, fs: float, bands: list[dict], active: np.ndarray
               ) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    """Split-band dynamic bell cuts.

    Each band is cut only while its share of the track is more than threshold_db
    above its typical (median) share: a harsh or sibilant moment gets turned
    down, the rest of the time the band is left alone. Silent parts of the
    track (below its activity gate) are never cut.
    """
    stats, traces = [], []
    # Only act while the track is playing: in a stem's silent parts the band
    # is just bleed from other instruments and would otherwise be cut constantly.
    gate = uniform_filter1d(active.astype(float), max(1, int(0.05 * fs))) if active.any() else 1.0
    for b in bands:
        band = filtfilt(_bandpass_sos(b["freq_hz"], b["q"], fs), x)
        band_db = dynamics.level_db(band, fs, 5.0)
        rel = band_db - dynamics.level_db(x, fs, 5.0)
        use = active if active.any() else slice(None)
        if b.get("mode", "cut") == "boost":
            # upward: while the band's share is threshold_db below its usual share (a thin
            # moment), raise it - but only where the band itself carries the instrument
            # (within 20 dB of its loud level), never on bleed or silence
            threshold = float(np.median(rel[use][::16])) - b["threshold_db"]
            present = band_db > np.percentile(band_db[use][::16], 90) - 20
            gr = -((threshold - rel).clip(0) * (1 - 1 / b["ratio"])).clip(0, b["max_boost_db"]) * gate * present
            gr = -dynamics.attack_smooth(dynamics.release_hold(-gr, b.get("release_ms", 120.0), fs),
                                         b.get("attack_ms", 20.0), fs)
            x = x + (10 ** (-gr / 20) - 1)[:, None] * band
            stats.append({"freq_hz": b["freq_hz"], "mode": "boost", "avg_cut_db": round(float(gr[use].mean()), 2),
                          "max_cut_db": round(float(gr.min()), 2)})
            traces.append(dynamics.decimate(gr, fs, np.min))
            continue
        threshold = float(np.median(rel[use][::16])) + b["threshold_db"]
        gr = dynamics.gain_reduction(rel, threshold, b["ratio"], 6.0).clip(0, b["max_cut_db"]) * gate
        gr = dynamics.attack_smooth(dynamics.release_hold(gr, b.get("release_ms", 80.0), fs), b.get("attack_ms", 2.0), fs)
        x = x - (1 - 10 ** (-gr / 20))[:, None] * band
        stats.append({"freq_hz": b["freq_hz"], "avg_cut_db": round(float(gr[use].mean()), 2),
                      "max_cut_db": round(float(gr.max()), 2)})
        traces.append(dynamics.decimate(gr, fs))
    return x, stats, traces


# --- resonance suppressor ---------------------------------------------------------------------

def resonance_suppressor(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, dict, dict]:
    """Spectral dynamic EQ: in every STFT frame, frequencies that stick out of the
    smoothed spectrum by more than threshold_db are pulled down (by strength x the
    excess, at most depth_db). Catches ringing notes and harsh resonances wherever
    they are, without a fixed EQ band. average_ms: the level averaged over that long
    first, so only what sticks out steadily is cut (a room mode under every note, not
    the harmonics of a resolved bass note, which would come down and leave the smear
    between them louder after the make-up gain).

    Make-up gain keeps the track's overall level.
    Returns the signal, stats, and a trace: average and maximum cut per
    frequency bin, and the mean cut across fmin..fmax per 0.1 s."""
    nfft = 8192 if cfg["fmin_hz"] < 200 else 2048
    hop = nfft // 4
    f = np.fft.rfftfreq(nfft, 1 / fs)
    weight = _taper(f, cfg["fmin_hz"], cfg["fmax_hz"])
    lo = max(1, np.searchsorted(f, cfg["fmin_hz"] / 2))
    hi = min(len(f), np.searchsorted(f, cfg["fmax_hz"] * 2) + 1)
    # Gaussian smoothing over log-frequency, as a matrix over the bins in play
    lf = np.log2(f[lo:hi])
    sigma = cfg["smooth_oct"] / 2.355
    smooth = np.exp(-0.5 * ((lf[:, None] - lf[None, :]) / sigma) ** 2)
    smooth /= smooth.sum(axis=1, keepdims=True)
    frame_rate = fs / hop
    chunk, margin = 1 << 20, 8 * nfft
    hop_t = int(dynamics.TRACE_HOP_S * fs)
    cut_sum, cut_max, n_frames = np.zeros(hi - lo), np.zeros(hi - lo), 0
    timeline = np.zeros(len(x) // hop_t + 1)
    counts = np.zeros_like(timeline)
    in_range = weight[lo:hi] > 0.5
    out = np.empty_like(x)
    for s in range(0, len(x), chunk):
        e = min(len(x), s + chunk)
        a, b = max(0, s - margin), min(len(x), e + margin)
        _, _, z = signal.stft(x[a:b].T, fs, nperseg=nfft, noverlap=nfft - hop)
        lev = 10 * np.log10(np.mean(np.abs(z[:, lo:hi]) ** 2, axis=0) + EPS)  # (bins, frames)
        if cfg["average_ms"]:  # only what sticks out over many frames: a room mode, not one note's harmonics
            pw = uniform_filter1d(10 ** (lev / 10), max(1, int(cfg["average_ms"] / 1000 * frame_rate)), axis=1)
            lev = 10 * np.log10(pw + EPS)
        excess = lev - smooth @ lev - cfg["threshold_db"]
        gr = (cfg["strength"] * excess).clip(0, cfg["depth_db"]) * weight[lo:hi, None]
        gr = dynamics.release_hold(gr, cfg["release_ms"], frame_rate, axis=1)
        gr = np.maximum(uniform_filter1d(gr, 3, axis=0), 0.0)  # avoid isolated-bin "musical noise"
        z[:, lo:hi] *= 10 ** (-gr / 20)
        _, y = signal.istft(z, fs, nperseg=nfft, noverlap=nfft - hop)
        y = y.T[: b - a]
        out[s:e] = np.pad(y, ((0, max(0, b - a - len(y))), (0, 0)))[s - a : e - a]
        # statistics over the frames centred inside this chunk's own part
        centres = a + np.arange(gr.shape[1]) * hop
        own = (centres >= s) & (centres < e)
        cut_sum += gr[:, own].sum(axis=1)
        cut_max = np.maximum(cut_max, gr[:, own].max(axis=1, initial=0.0))
        n_frames += int(own.sum())
        np.add.at(timeline, centres[own] // hop_t, gr[in_range][:, own].mean(axis=0))
        np.add.at(counts, centres[own] // hop_t, 1)
    avg = cut_sum / max(n_frames, 1)
    timeline = np.divide(timeline, counts, out=np.zeros_like(timeline), where=counts > 0)
    # Make-up gain restores the track's energy, so the net effect is evening out:
    # frequencies that ring end up lower relative to the rest, not the whole track quieter.
    makeup = float(10 * np.log10((np.mean(x * x) + EPS) / (np.mean(out * out) + EPS)))
    out *= 10 ** (makeup / 20)
    stats = {"avg_cut_db": round(float(avg.mean()), 2), "max_cut_db": round(float(cut_max.max()), 2),
             "makeup_db": round(makeup, 2)}
    return out, stats, {"freqs": f[lo:hi], "avg_cut_db": avg, "max_cut_db": cut_max, "timeline": timeline}


# --- transient shaper -------------------------------------------------------------------------

def _shape(x: np.ndarray, fs: float, attack: float, sustain: float, max_db: float) -> np.ndarray:
    """Gain (dB per sample) of a level-independent attack/sustain shaper."""
    a = np.abs(x).max(axis=1)
    fast = dynamics.envelope(a, fs, 0.5, 30.0)
    slow = dynamics.envelope(a, fs, 15.0, 30.0)
    long = dynamics.envelope(a, fs, 0.5, 300.0)
    onset = 20 * np.log10((fast + EPS) / (slow + EPS)).clip(0)
    tail = 20 * np.log10((long + EPS) / (fast + EPS)).clip(0)
    g = (attack * onset + sustain * tail).clip(-max_db, max_db)
    return dynamics.attack_smooth(g, 1.0, fs)


def transient_shaper(x: np.ndarray, fs: float, attack, sustain, max_db: float, crossovers_hz=()
                     ) -> tuple[np.ndarray, dict, dict]:
    """Level-independent attack/sustain control.

    attack > 0 boosts the onset of each hit (< 0 softens it); sustain < 0
    shortens the decay, which on a room recording mostly means less room.
    With crossovers_hz, each band (e.g. kick / shells / cymbals) is detected
    and shaped on its own, with attack and sustain given per band as lists.
    """
    if not crossovers_hz:
        g = _shape(x, fs, float(attack), float(sustain), max_db)
        stats = {"max_boost_db": round(float(g.max()), 2), "max_cut_db": round(float(-g.min()), 2)}
        return x * (10 ** (g / 20))[:, None], stats, {"boost": dynamics.decimate(g, fs, np.max),
                                                      "cut": dynamics.decimate(g, fs, np.min)}
    n = len(crossovers_hz) + 1
    if len(attack) != n or len(sustain) != n:
        raise ValueError(f"transient: {len(crossovers_hz)} crossovers need {n} attack and sustain values")
    out = np.zeros_like(x)
    boost = cut = None
    bands = []
    edges = [0.0, *sorted(crossovers_hz), fs / 2]
    for i, (band, a, su) in enumerate(zip(dynamics.split_bands(x, fs, crossovers_hz), attack, sustain)):
        g = _shape(band, fs, a, su, max_db)
        out += band * (10 ** (g / 20))[:, None]
        b, c = dynamics.decimate(g, fs, np.max), dynamics.decimate(g, fs, np.min)
        boost = b if boost is None else np.maximum(boost, b)
        cut = c if cut is None else np.minimum(cut, c)
        bands.append({"range_hz": [edges[i], edges[i + 1]], "max_boost_db": round(float(g.max()), 2),
                      "max_cut_db": round(float(-g.min()), 2)})
    stats = {"max_boost_db": max(b["max_boost_db"] for b in bands), "max_cut_db": max(b["max_cut_db"] for b in bands),
             "bands": bands}
    return out, stats, {"boost": boost, "cut": cut}


# --- kick: detection, EQ and kick-triggered ducking -------------------------------------------

def _kick_onsets(x: np.ndarray, fs: float) -> np.ndarray:
    low = filtfilt(signal.butter(2, [40, 150], "bandpass", fs=fs, output="sos"), x).mean(axis=1)
    env = dynamics.envelope(np.abs(low), fs, 1.0, 50.0)
    peaks, _ = signal.find_peaks(env, height=np.percentile(env, 90), distance=int(0.15 * fs))
    return peaks


def kick_hits(x: np.ndarray, fs: float) -> np.ndarray:
    """Start of each kick hit: where the low-band envelope first reaches half of
    the hit's peak (within 40 ms before it)."""
    low = filtfilt(signal.butter(2, [40, 150], "bandpass", fs=fs, output="sos"), x).mean(axis=1)
    env = dynamics.envelope(np.abs(low), fs, 1.0, 50.0)
    peaks, _ = signal.find_peaks(env, height=np.percentile(env, 90), distance=int(0.15 * fs))
    look = int(0.04 * fs)
    starts = []
    for p in peaks:
        seg = env[max(0, p - look) : p + 1]
        below = np.flatnonzero(seg < 0.5 * env[p])
        starts.append(p - len(seg) + 1 + (below[-1] + 1 if below.size else 0))
    return np.array(starts, dtype=int)


def kick_shaper(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, dict, dict]:
    """Shape the drum stem's low band hit by hit: +attack_db over the first
    attack_ms, +punch_db over the first punch_ms of every kick, then -tail_db on
    the boom that rings after it.
    Driven by detected kick hits, so it is precise even when the room makes
    the kick barely stand out from the low-end rumble."""
    hits = kick_hits(x, fs)
    n = len(x)
    ms = lambda v: int(v * 1e-3 * fs)  # noqa: E731
    pm, ts, tl = ms(cfg["punch_ms"]), ms(cfg["tail_start_ms"]), ms(cfg["tail_ms"])
    tmpl = np.zeros(ts + tl)
    tmpl[:pm] = cfg["punch_db"] * np.sin(np.pi * np.arange(pm) / pm) ** 0.5   # quick rise, smooth fall
    am = ms(cfg["attack_ms"])
    if am > 1 and cfg["attack_db"]:  # extra boost on the very start of the hit: a harder onset
        tmpl[:am] += cfg["attack_db"] * np.sin(np.pi * np.arange(am) / am) ** 0.5
    ramp = np.minimum(1, np.arange(tl) / max(1, tl // 4))                        # ease into the tail cut
    tmpl[ts:] = -cfg["tail_db"] * ramp * (1 - np.arange(tl) / tl) ** 0.5
    start = max(0, ms(2))  # begin 2 ms before the detected start
    g = np.zeros(n)
    for h in hits:
        a = max(0, h - start)
        b = min(n, a + len(tmpl))
        seg = tmpl[: b - a]
        g[a:b] = np.where(np.abs(seg) > np.abs(g[a:b]), seg, g[a:b])  # overlapping hits: strongest wins
    g = dynamics.attack_smooth(g, 0.5, fs)
    low = filtfilt(signal.butter(2, cfg["fmax_hz"], "low", fs=fs, output="sos"), x)
    y = x + (10 ** (g / 20) - 1)[:, None] * low
    stats = {"hits": int(len(hits)), "punch_db": cfg["punch_db"], "tail_db": cfg["tail_db"]}
    return y, stats, {"boost": dynamics.decimate(g, fs, np.max), "cut": dynamics.decimate(g, fs, np.min)}


def kick_frequency(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float | None:
    """Kick fundamental: spectral peak (fmin..fmax) of the first 120 ms after each kick hit."""
    w = int(0.12 * fs)
    peaks = [p for p in _kick_onsets(x, fs) if p + w <= len(x)]
    if len(peaks) < 4:
        return None
    spec = np.zeros((1 << 15) // 2 + 1)
    for p in peaks[:400]:
        f, pw = signal.periodogram(x[p : p + w].mean(axis=1) * np.hanning(w), fs, nfft=1 << 15)
        spec += pw
    m = (f >= fmin) & (f <= fmax)
    return float(f[m][np.argmax(spec[m])])


# band each hit source is detected in; the snare on its body only, where its wires' buzz adds
# little, so that buzz is never taken for playing
TRIGGER_BANDS = {"kick": (40.0, 150.0), "snare": (150.0, 400.0), "toms": (70.0, 400.0)}


def hit_trigger(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """0..1 per sample: how strongly a hit is happening in the lo..hi band of x
    (1 within 12 dB of the loudest hits, 0 more than 12 dB below them)."""
    band = filtfilt(signal.butter(2, [lo, hi], "bandpass", fs=fs, output="sos"), x).mean(axis=1)
    env_db = 20 * np.log10(dynamics.envelope(np.abs(band), fs, 1.0, 40.0) + EPS)
    top = np.percentile(env_db, 99)
    return ((env_db - (top - 12)) / 12).clip(0, 1).astype(np.float32)


def kick_trigger(drums: np.ndarray, fs: float) -> np.ndarray:
    return hit_trigger(drums, fs, *TRIGGER_BANDS["kick"])


def _band(x: np.ndarray, fs: float, fmin: float, fmax: float) -> np.ndarray:
    if fmin > 0 and fmax < 0.45 * fs:
        sos = signal.butter(2, [fmin, fmax], "bandpass", fs=fs, output="sos")
    elif fmin > 0:
        sos = signal.butter(2, fmin, "high", fs=fs, output="sos")
    else:
        sos = signal.butter(2, fmax, "low", fs=fs, output="sos")
    return filtfilt(sos, x)


def duck(x: np.ndarray, fs: float, trigger: np.ndarray, fmin_hz: float, fmax_hz: float, depth_db: float,
         release_ms: float) -> tuple[np.ndarray, dict, np.ndarray]:
    """Dip the fmin..fmax band of this track while the trigger source hits: e.g.
    the bass's low end under the kick (so the kick gets through), or the
    snare-wire rattle the kick sets off in the hi-hat stem."""
    gr = trigger.astype(np.float64) * depth_db
    gr = dynamics.attack_smooth(dynamics.release_hold(gr, release_ms, fs), 2.0, fs)
    y = x - (1 - 10 ** (-gr / 20))[:, None] * _band(x, fs, fmin_hz, fmax_hz)
    return y, {"max_db": round(float(gr.max()), 2), "avg_db": round(float(gr.mean()), 2)}, dynamics.decimate(gr, fs)


def hit_gate(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """Turn a drum stem down by depth_db between its own hits (detected in
    fmin..fmax), opening on each hit and closing over release_ms: removes the
    buzz and bleed in the gaps, e.g. snare wires rattling along with the kick."""
    trig = hit_trigger(x, fs, cfg["fmin_hz"], cfg["fmax_hz"]).astype(np.float64)
    # sharp onsets open it fully too: a real hit is impulsive, while the body resonating
    # along with the bass only swells (and opens a level-driven gate only partly)
    band = np.abs(_band(x, fs, cfg["fmin_hz"], cfg["fmax_hz"]).mean(axis=1))
    fast, slow = dynamics.envelope(band, fs, 0.5, 20.0), dynamics.envelope(band, fs, 10.0, 40.0)
    lev = 20 * np.log10(fast + EPS)
    onset = (20 * np.log10((fast + EPS) / (slow + EPS)) > 6.0) & (lev > np.percentile(lev, 99.5) - 24.0)
    open_ = dynamics.release_hold(np.maximum(trig, onset.astype(np.float64)), cfg["release_ms"], fs)
    open_ = np.minimum(open_ * 2.0, 1.0)  # fully open on all but faint hits
    la = max(2, int(0.005 * fs)) // 2 * 2  # look 5 ms ahead so the gate is open before the attack
    open_ = maximum_filter1d(open_, size=la + 1, origin=-(la // 2), mode="nearest")
    open_ = dynamics.attack_smooth(open_, 1.0, fs)
    gr = cfg["depth_db"] * (1 - open_)
    y = x * (10 ** (-gr / 20))[:, None]
    if cfg.get("buzz_extra_db", 0.0) > 0:
        # the wires' buzz band goes further down between hits (it is all buzz there); the hit,
        # its body and its own wire decay (while the gate is open) are untouched
        extra = cfg["buzz_extra_db"] * (1 - open_)
        band = _band(y, fs, *cfg.get("buzz_band_hz", (1200.0, 12000.0)))
        y = y - (1 - 10 ** (-extra / 20))[:, None] * band
    return y, {"avg_db": round(float(gr.mean()), 2), "max_db": round(float(gr.max()), 2)}, dynamics.decimate(gr, fs)


# --- harmonic generation (exciter, bass enhancer) ---------------------------------------------

def _harmonics(src: np.ndarray, fs: float, drive: float, even: float, out_lo: float, out_hi: float,
               amount_db: float) -> np.ndarray:
    """Harmonics of src (no fundamental), band-limited to out_lo..out_hi and scaled
    to amount_db relative to src. The source is level-normalised before the
    waveshaper, so quiet and loud passages get the same amount."""
    # (the running mean can dip a hair below zero numerically: clamp before the sqrt)
    env = np.sqrt(np.maximum(uniform_filter1d(np.mean(src * src, axis=1), max(1, int(0.005 * fs))), 0.0)) + 1e-4
    u = src / env[:, None]
    gen = np.empty_like(u)
    for s, e, a, up in loudness.oversampled_chunks(u, 4):
        # nonlinear part only, with the sign chosen so that what it holds at the source's own
        # frequencies adds in phase (saturation's opposite sign would cancel some of them)
        odd = up - np.tanh(drive * up) / drive
        evn = np.sqrt(up * up + 1e-3)
        g = (1 - even) * odd + even * evn
        gen[s:e] = signal.resample_poly(g, 1, 4, axis=0)[s - a : e - a]
    del u
    gen *= env[:, None]
    src_power = float(np.mean(src * src))
    sos = signal.butter(2, [out_lo, min(out_hi, 0.45 * fs)], "bandpass", fs=fs, output="sos")
    gen = filtfilt(sos, gen)
    gen *= np.sqrt((src_power + EPS) / (float(np.mean(gen * gen)) + EPS)) * 10 ** (amount_db / 20)
    return gen


def exciter(x: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Adds brightness by generating overtones of the upper mids, instead of
    boosting the treble (which would also raise the hiss)."""
    src = filtfilt(signal.butter(2, cfg["freq_hz"], "high", fs=fs, output="sos"), x)
    gen = _harmonics(src, fs, cfg["drive"], cfg["even"], 1.5 * cfg["freq_hz"], 18000.0, cfg["amount_db"])
    del src
    gen += x
    return gen


def bass_enhancer(x: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Psychoacoustic bass: overtones of the lowest notes make them audible on
    small speakers (and make up for a recorder's low-cut)."""
    src = filtfilt(signal.butter(2, cfg["cutoff_hz"], "low", fs=fs, output="sos"), x)
    gen = _harmonics(src, fs, cfg["drive"], cfg["even"], cfg["cutoff_hz"] * 0.8, cfg["cutoff_hz"] * 4,
                     cfg["amount_db"])
    del src
    gen += x
    return gen


def expander(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """Downward expander for a stem's gaps: below (playing level - below_db) the
    level is pushed down further (ratio), by at most range_db. Notes and their
    decay sit above the threshold and are untouched; hiss, hum and bleed between
    notes go down."""
    lev = dynamics.level_db(x, fs, 10.0)
    use = lev > np.percentile(lev, 5)
    playing = float(np.percentile(lev[use][::16], 90))
    thr = playing - cfg["below_db"]
    rng = cfg["range_db"]
    gr = ((thr - lev) * (cfg["ratio"] - 1)).clip(0, rng)
    # smooth the *open* amount: it opens at once (5 ms ahead of each note) and closes slowly
    open_ = dynamics.release_hold(rng - gr + 1e-3, cfg["hold_ms"], fs)
    la = max(2, int(0.005 * fs)) // 2 * 2
    open_ = maximum_filter1d(open_, size=la + 1, origin=-(la // 2), mode="nearest")
    open_ = dynamics.attack_smooth(open_, 2.0, fs)
    gr = (rng - open_).clip(0, rng)
    y = x * (10 ** (-gr / 20))[:, None]
    return y, {"threshold_db": round(thr, 1), "avg_db": round(float(gr.mean()), 2),
               "closed_pct": round(100 * float(np.mean(gr > cfg["range_db"] / 2)), 1)}, dynamics.decimate(gr, fs)


def _hold(mask: np.ndarray, fs: float, before_s: float, after_s: float) -> np.ndarray:
    """1 within [t - after_s, t + before_s] of any True sample (computed at 100 Hz)."""
    step = int(fs / 100)
    n = -(-len(mask) // step)
    m = np.pad(mask, (0, n * step - len(mask))).reshape(n, step).any(axis=1).astype(float)  # any sample in each 10 ms
    a, b = int(after_s * 100), int(before_s * 100)
    k = np.convolve(m, np.ones(a + b + 1), mode="full")[b : b + n] > 0  # window from t-after to t+before
    return np.repeat(k.astype(float), step)[: len(mask)]


def kit_gate(x: np.ndarray, fs: float, cfg: dict, triggers: dict[str, np.ndarray]) -> tuple[np.ndarray, dict, np.ndarray]:
    """For the hi-hat and cymbal stems: while the drummer is not playing (no kick,
    snare or tom hit for hold_s) they only hold the snare wires and cymbals
    resonating with the bass and guitars, and room bleed: turn them down by
    depth_db. The stem's own strong, sharp hits also open it (a count-in on the
    hi-hat, a ride pattern with no kick)."""
    kit = [triggers[k] for k in ("kick", "snare", "toms") if k in triggers]
    # any kit hit within ~11 dB of the song's loudest counts: soft build-ups are playing too
    open_ = _hold(np.max(kit, axis=0) > 0.08, fs, 0.3, cfg["hold_s"]) if kit else np.ones(len(x))
    band = np.abs(_band(x, fs, 2000.0, 12000.0).mean(axis=1))
    fast, slow = dynamics.envelope(band, fs, 0.5, 30.0), dynamics.envelope(band, fs, 20.0, 30.0)
    onset = 20 * np.log10((fast + EPS) / (slow + EPS))
    lev = 20 * np.log10(fast + EPS)
    own = (onset > cfg["own_onset_db"]) & (lev > np.percentile(lev, 99) - 20)
    open_ = np.maximum(open_, _hold(own, fs, 0.05, cfg["own_hold_s"]))
    # smooth: open quickly, close slowly
    open_ = dynamics.attack_smooth(dynamics.release_hold(open_, 300.0, fs), 10.0, fs).clip(0, 1)
    gr = cfg["depth_db"] * (1 - open_)
    y = x * (10 ** (-gr / 20))[:, None]
    stats = {"closed_pct": round(100 * float(np.mean(gr > cfg["depth_db"] / 2)), 1),
             "removed_db": round(float(10 * np.log10((np.mean(y * y) + EPS) / (np.mean(x * x) + EPS))), 2),
             "_gr": gr.astype(np.float32)}  # the whole curve: the original recording follows it ([mix])
    return y, stats, dynamics.decimate(gr, fs)


def saturator(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, dict]:
    """Tube/tape-style warmth: an oversampled, slightly asymmetric tanh curve
    (asymmetry = even harmonics), blended in parallel with `mix`. Drive is
    relative to the track's own level, so it acts the same on any stem.
    engine = "chowtape": CHOW Tape Model instead (plugins.py), fully wet, same loudness;
    if it is not installed, the built-in curve is used."""
    if cfg.get("engine", "builtin") == "chowtape":
        from . import plugins

        if plugins.available(cfg["chowtape"]["path"]):
            return plugins.chowtape(x, fs, cfg["chowtape"])
        print("  note: CHOW Tape Model / pedalboard not installed, using the built-in saturator", file=sys.stderr)
    rms = float(np.sqrt(np.mean(x * x)))
    if rms <= 0:
        return x, {"harmonics_db": None}
    g = 10 ** (cfg["drive_db"] / 20) * 0.3 / rms  # the track's RMS sits at 0.3 * drive on the curve
    b = 0.6 * cfg["even"]
    slope = 1 - np.tanh(b) ** 2  # unity small-signal gain
    resid = np.empty_like(x)
    for s, e, a, up in loudness.oversampled_chunks(x, 4):
        w = (np.tanh(up * g + b) - np.tanh(b)) / (slope * g)
        resid[s:e] = signal.resample_poly(w - up, 1, 4, axis=0)[s - a : e - a]
    resid = filtfilt(signal.butter(2, 20.0, "high", fs=fs, output="sos"), resid)  # asymmetry adds DC
    resid *= cfg["mix"]
    harm = float(10 * np.log10((np.mean(resid * resid) + EPS) / (rms * rms)))
    resid += x
    return resid, {"harmonics_db": round(harm, 1)}


# --- the chain --------------------------------------------------------------------------------

def begin_detail(x: np.ndarray, fs: float) -> dict:
    grid = eq.log_grid(fs)
    return {"grid": grid, "spec_in": eq.band_spectrum(x, fs, grid), "lufs_in": loudness.integrated(x, fs),
            "eq_db": np.zeros_like(grid), "dyn_eq": [], "traces": {}}


def end_detail(detail: dict, x: np.ndarray, fs: float) -> None:
    detail.update(spec_out=eq.band_spectrum(x, fs, detail["grid"]), lufs_out=loudness.integrated(x, fs))


def _ducks(x, fs, cfg, triggers, purpose, stats, detail):
    for dk in cfg["ducks"]:
        trig = (triggers or {}).get(dk["source"])
        if trig is None or dk.get("purpose", "mix") != purpose:
            continue
        x, st, tr = duck(x, fs, trig, dk["fmin_hz"], dk["fmax_hz"], dk["depth_db"], dk["release_ms"])
        stats.setdefault("ducks", []).append(st | {"source": dk["source"], "band_hz": [dk["fmin_hz"], dk["fmax_hz"]],
                                                   "purpose": purpose})
        if detail is not None:
            t = detail["traces"]
            t["duck"] = tr if "duck" not in t else np.maximum(t["duck"], tr)
    return x


def cleanup_track(x: np.ndarray, fs: float, cfg: dict, triggers: dict | None, stats: dict, detail: dict | None,
                  prof: dict | None = None) -> np.ndarray:
    """Clean-up: what a close mic in a studio would not have picked up. The gaps
    between notes (expander), bleed timed to other instruments' hits (clean-up
    ducks), the gaps between the stem's own hits (hit gate), and its content
    while the drummer is not playing (kit gate). prof: the stem's profile (stage 0),
    whose noise floor places the expander threshold (expander.adapt)."""
    ex = cfg["expander"]
    if ex["enabled"]:
        if ex["adapt"] and prof and "snr_db" in prof:
            # just above the measured floor; a floor closer than min_below_db to the playing level is
            # quiet playing, not noise (a stem that never pauses): then the threshold stays as set
            below = prof["snr_db"] - ex["floor_margin_db"]
            if below >= ex["min_below_db"]:
                ex = ex | {"below_db": float(min(below, ex["below_db"]))}
        x, stats["expander"], tr = expander(x, fs, ex)
        stats["expander"]["below_db"] = round(ex["below_db"], 1)
        if detail is not None:
            detail["traces"]["gate"] = tr
    hc = cfg["harmonic_clean"]
    if cfg["note_eq"]["enabled"]:  # the bass note by note: harmonic space and attack (note_eq.py)
        from . import note_eq

        x, stats["note_eq"], stats["_pitch"] = note_eq.note_eq(x, fs, cfg["note_eq"], cfg["bass_fundamental"])
    elif hc["enabled"]:  # pitched stems (the bass): what lies between the note's harmonics (augment.py)
        from . import augment

        pitch = augment.bass_pitch(x, fs, cfg["bass_fundamental"])
        stats["_pitch"] = pitch  # reused by the fundamental (stage 2); not part of the report
        x, stats["harmonic_clean"] = augment.harmonic_clean(x, fs, hc, *pitch)
    x = _ducks(x, fs, cfg, triggers, "cleanup", stats, detail)
    gt = cfg["gate"]
    if gt["enabled"]:
        x, stats["gate"], tr = hit_gate(x, fs, gt)
        if detail is not None:
            t = detail["traces"]
            t["gate"] = tr if "gate" not in t else np.maximum(t["gate"], tr)
    kg = cfg["kit_gate"]
    if kg["enabled"] and triggers:
        x, stats["kit_gate"], tr = kit_gate(x, fs, kg, triggers)
        stats["_kit_gate_gr"] = stats["kit_gate"].pop("_gr")  # for the blend; not part of the report
        if detail is not None:
            t = detail["traces"]
            t["gate"] = tr if "gate" not in t else np.maximum(t["gate"], tr)
    return x


def studio_track(x: np.ndarray, fs: float, cfg: dict, stats: dict, detail: dict | None,
                 prof: dict | None = None, target: dict | None = None, triggers: dict | None = None) -> np.ndarray:
    """The stem as a well-recorded single track: tone (tone match, EQ, dynamic EQ,
    resonances), shape (compressor, kick shaper, transients) and harmonic content
    (saturator, exciter, bass enhancer). No level or placement decisions.
    prof is the stem's profile (stage 0) and target the same instrument's median
    profile over all songs: the tone match and transient amount move the stem
    towards it, so it sounds alike from song to song."""
    from . import augment, profile

    au = cfg["augment"]
    if au["enabled"]:  # drum samples from the drummer's own hits, under the stem (augment.py)
        x, stats["augment"], samples = augment.drum_augment(x, fs, au, triggers)
        if samples:
            stats["_samples"] = samples  # for the stage files; not part of the report
    ca = cfg["cymbal_augment"]
    if ca["enabled"]:  # studio crash / ride / bell hits on the cymbals' hits, by kind (augment.py)
        x, stats["cymbal_augment"], samples = augment.cymbal_augment(x, fs, ca)
        if samples:
            stats["_samples"] = stats.get("_samples", []) + samples
    grid = eq.log_grid(fs)
    bands = list(cfg["eq"])
    k = cfg["kick_eq"]
    if k["enabled"]:
        f0 = kick_frequency(x, fs, k["fmin_hz"], k["fmax_hz"])
        if f0:
            bands.append({"type": "peak", "freq_hz": f0, "gain_db": k["gain_db"], "q": k["q"]})
            stats["kick_hz"] = round(f0, 1)
    curve = eq.manual_curve(grid, bands, fs) if bands else None
    tm = cfg["tone_match"]
    if tm["enabled"] and prof and target and len(prof["tone"]) == len(grid):
        # the target: the band's own median tone for this instrument, blended with the tone of raw
        # studio tracks of it (reference = 0..1)
        band = np.asarray(target["tone"]) if target.get("songs", 0) >= tm["min_songs"] else None
        ref = target.get("reference")
        w = tm["reference"] if ref is not None else 0.0
        if band is None:
            want = ref if w > 0 else None
        else:
            want = band if ref is None else (1 - w) * band + w * ref
        if want is not None:
            match = profile.tone_match_curve(grid, prof["tone"], want, tm)
            stats["tone_match_db"] = eq.at_freqs(grid, match)
            stats["tone_reference"] = round(w if band is not None else 1.0, 2)
            curve = match if curve is None else curve + match
    if curve is not None:
        h = eq.design_fir(grid, curve, fs, 8191)
        x = eq.apply_fir(x, h)
        applied = eq.fir_response_db(h, grid, fs)
        stats["eq_db"] = eq.at_freqs(grid, applied)
        if detail is not None:
            detail["eq_db"] = applied
    active = loudness.active_mask(x, fs)
    if cfg["dynamic_eq"]:
        x, stats["dynamic_eq"], traces = dynamic_eq(x, fs, cfg["dynamic_eq"], active)
        if detail is not None:
            for band, st, tr in zip(cfg["dynamic_eq"], stats["dynamic_eq"], traces):
                detail["dyn_eq"].append({"freq_hz": band["freq_hz"], "trace": tr,
                                         "max_curve": dynamic_eq_shape(grid, band, st["max_cut_db"], fs),
                                         "avg_curve": dynamic_eq_shape(grid, band, st["avg_cut_db"], fs)})
    if cfg["resonance"]["enabled"]:
        x, stats["resonance"], tr = resonance_suppressor(x, fs, cfg["resonance"])
        if detail is not None:
            lg = np.log2(grid)
            detail["resonance"] = {
                "avg_curve": -np.interp(lg, np.log2(tr["freqs"]), tr["avg_cut_db"], left=0, right=0),
                "max_curve": -np.interp(lg, np.log2(tr["freqs"]), tr["max_cut_db"], left=0, right=0),
            }
            detail["traces"]["resonance"] = tr["timeline"]
    c = cfg["compressor"]
    if c["enabled"]:
        x, gr, stats["compressor"] = dynamics.compress(x, fs, **{k: c[k] for k in dynamics.BAND_PARAMS},
                                                       active=active, auto_makeup=True)
        if detail is not None:
            detail["traces"]["compressor"] = dynamics.decimate(gr, fs)
        del gr
    c2 = cfg["compressor2"]
    if c2["enabled"]:  # serial compression: e.g. a fast one catching peaks, then a slow one pinning the level
        x, gr, stats["compressor2"] = dynamics.compress(x, fs, **{k: c2[k] for k in dynamics.BAND_PARAMS},
                                                        active=active, auto_makeup=True)
        if detail is not None:
            tr = dynamics.decimate(gr, fs)
            detail["traces"]["compressor"] = np.maximum(detail["traces"].get("compressor", tr), tr)
        del gr
    ks = cfg["kick_shaper"]
    if ks["enabled"]:
        x, stats["kick_shaper"], tr = kick_shaper(x, fs, ks)
        if detail is not None:
            detail["traces"]["kick_shaper"] = tr
    t = cfg["transient"]
    if t["enabled"]:
        attack = t["attack"]
        if t["adapt"] and prof and target and "punch_db" in prof and "punch_db" in target:
            # duller hits than this instrument usually has: more attack (x2 per 6 dB), sharper: less
            scale = float(np.clip(2 ** ((target["punch_db"] - prof["punch_db"]) / 6), 0.5, 2.0))
            attack = [a * scale for a in attack] if isinstance(attack, list) else attack * scale
            stats["transient_scale"] = round(scale, 2)
        x, stats["transient"], tr = transient_shaper(x, fs, attack, t["sustain"], t["max_db"],
                                                     t["crossovers_hz"])
        if detail is not None:
            detail["traces"]["transient"] = tr
    bf = cfg["bass_fundamental"]
    if bf["enabled"]:  # the bass's own fundamental, pitch-tracked, raised in phase (augment.py)
        x, stats["bass_fundamental"], tr = augment.bass_fundamental(x, fs, bf, stats.get("_pitch"))
        if detail is not None and tr.size > 1:
            detail["traces"]["fundamental"] = tr
    if cfg["saturator"]["enabled"]:
        x, stats["saturator"] = saturator(x, fs, cfg["saturator"])
    if cfg["grit"]["enabled"]:  # parallel band distortion: teeth that cut through the guitars
        from . import space

        x, stats["grit"] = space.grit(x, fs, cfg["grit"])
    if cfg["dual"]["enabled"]:  # split-band bass chain: clean mono lows, driven + chorused highs
        from . import space

        x, stats["dual"] = space.dual(x, fs, cfg["dual"])
    if cfg["exciter"]["enabled"]:
        x = exciter(x, fs, cfg["exciter"])
    if cfg["bass_enhancer"]["enabled"]:
        x = bass_enhancer(x, fs, cfg["bass_enhancer"])
    ne = cfg["note_eq"]
    if ne["enabled"] and ne["after_chain"]:  # again on what the tone match, exciter and transients brought up
        from . import note_eq

        x, stats["note_eq_after_chain"], _ = note_eq.note_eq(x, fs, ne | {"keep_above_hz": ne["after_chain_keep_above_hz"]},
                                                             cfg["bass_fundamental"])
    return x


def mix_track(x: np.ndarray, fs: float, cfg: dict, triggers: dict | None, stats: dict, detail: dict | None) -> np.ndarray:
    """Mix decisions for the stem: making room for other instruments (mix ducks,
    e.g. the bass under the kick), stereo placement (re-centring, width, pan) and
    level (gain)."""
    x = _ducks(x, fs, cfg, triggers, "mix", stats, detail)
    if x.shape[1] == 2:
        if cfg["balance"] > 0:
            x, stats["balance_measured_db"], stats["balance_correction_db"] = stereo.rebalance(x, cfg["balance"], 6.0)
        if cfg["width"] != 1.0:
            # keep the track's power: folding a roomy stem to mono would otherwise lose level
            p_in = float(np.mean(x * x))
            x = stereo.mid_side(x, fs, 0.0, cfg["width"])
            x *= np.sqrt(p_in / (float(np.mean(x * x)) + EPS))
        if cfg["pan"]:
            th = (cfg["pan"] + 1) * np.pi / 4  # constant-power pan law: -1 left, 0 unchanged, +1 right
            x = x * (np.sqrt(2) * np.array([np.cos(th), np.sin(th)]))
    if cfg["gain_db"]:
        x = x * 10 ** (cfg["gain_db"] / 20)
    return x


def process_track(x: np.ndarray, fs: float, cfg: dict, detail: dict | None = None,
                  triggers: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, dict]:
    """The whole chain of one track: clean-up, studio, mix. If detail is a dict, it
    is filled with what the report needs: spectra and loudness before/after, the
    static EQ curve, the dynamic EQ and resonance cut ranges, and gain traces."""
    stats: dict = {}
    if detail is not None:
        detail.update(begin_detail(x, fs))
    x = cleanup_track(x, fs, cfg, triggers, stats, detail)
    x = studio_track(x, fs, cfg, stats, detail)
    stats.pop("_pitch", None)
    x = mix_track(x, fs, cfg, triggers, stats, detail)
    if detail is not None:
        end_detail(detail, x, fs)
    return x, stats
