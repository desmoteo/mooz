"""Reinforcement of the separated stems, as an engineer does with weak recorded tracks.

Drum sample augmentation (kick, snare, toms): samples are made from the drummer's own
cleanest hits in the song (no other drum hitting within the sample's length), aligned
and averaged, in two velocity layers (and, for toms, one per drum, told apart by pitch).
Each detected hit then gets its sample, phase-aligned to the hit (cross-correlation
within +-2 ms) and scaled to the hit's own velocity, blended under the stem. The sound
stays the drummer's; the sample has no bleed and no separation artifacts, and every hit
gets the full attack.

Bass fundamental reinforcement: the bass's pitch is tracked (pYIN) and its fundamental
extracted with a tracking filter (heterodyne to 0 Hz, low-pass, back): it is in phase
with the stem by construction, so it can be raised without beating. Notes whose
fundamental is weak (high notes, a pick, the small cabinet) are filled up towards the
song's usual fundamental-to-note ratio.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import dynamics
from .loudness import EPS, filtfilt

# --- drums -------------------------------------------------------------------------------------


def _band_env(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    band = filtfilt(signal.butter(2, [lo, hi], "bandpass", fs=fs, output="sos"), x).mean(axis=1)
    return dynamics.envelope(np.abs(band), fs, 1.0, 50.0)


def drum_hits(x: np.ndarray, fs: float, lo: float, hi: float, min_gap_s: float = 0.07) -> tuple[np.ndarray, np.ndarray]:
    """Hit starts (where the band envelope first reaches half its peak) and peak levels."""
    env = _band_env(x, fs, lo, hi)
    top = np.percentile(env, 99.5)
    peaks, _ = signal.find_peaks(env, height=max(np.percentile(env, 90), top * 10 ** (-24 / 20)),
                                 prominence=top * 10 ** (-30 / 20), distance=int(min_gap_s * fs))
    look = int(0.03 * fs)
    starts = []
    for p in peaks:
        seg = env[max(0, p - look) : p + 1]
        below = np.flatnonzero(seg < 0.5 * env[p])
        starts.append(p - len(seg) + 1 + (below[-1] + 1 if below.size else 0))
    return np.array(starts, dtype=int), env[peaks]


def _start(smp: np.ndarray, fs: float, lo: float, hi: float) -> int:
    """Where a one-shot "starts" by the same rule as drum_hits (half of its peak envelope), and
    never before its first frame: the filtered envelope of an abrupt sample rises ahead of it."""
    env = _band_env(np.pad(smp, ((int(0.05 * fs), 0), (0, 0))), fs, lo, hi)
    p = int(np.argmax(env))
    seg = env[max(0, p - int(0.03 * fs)) : p + 1]
    below = np.flatnonzero(seg < 0.5 * env[p])
    return max(0, p - len(seg) + 1 + (below[-1] + 1 if below.size else 0) - int(0.05 * fs))


def _pitch(seg: np.ndarray, fs: float) -> float:
    """Dominant frequency 50-400 Hz of a hit (toms: which drum)."""
    n = 1 << 14
    f, p = signal.periodogram(seg.mean(axis=1) * np.hanning(len(seg)), fs, nfft=n)
    m = (f >= 50) & (f <= 400)
    return float(f[m][np.argmax(p[m])])


def _clusters(f: np.ndarray, k_max: int) -> np.ndarray:
    """1-D k-means on log frequency; only as many clusters as are >= 15 % apart."""
    lf = np.log(f)
    best = np.zeros(len(f), int)
    for k in range(2, k_max + 1):
        c = np.quantile(lf, (np.arange(k) + 0.5) / k)
        for _ in range(20):
            lab = np.argmin(np.abs(lf[:, None] - c[None]), axis=1)
            c = np.array([lf[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
        if np.min(np.diff(np.sort(c))) < np.log(1.15) or np.min(np.bincount(lab, minlength=k)) < 4:
            break
        best = lab
    return best


def _lag(ref: np.ndarray, x: np.ndarray, pos: int, max_lag: int, either_polarity: bool = False) -> int:
    """d in -max_lag..max_lag for which x[pos + d :] lines up best with ref (mono); with
    either_polarity, also when inverted (a foreign sample may be miked the other way round)."""
    r = ref.mean(axis=1)
    seg = x[max(0, pos - max_lag) : pos + max_lag + len(r)].mean(axis=1)
    off = pos - max(0, pos - max_lag)
    c = signal.correlate(seg, r, mode="valid")
    return int(np.argmax(np.abs(c) if either_polarity else c)) - off


def drum_augment(x: np.ndarray, fs: float, cfg: dict, others: dict[str, np.ndarray] | None = None
                 ) -> tuple[np.ndarray, dict, list[np.ndarray]]:
    """x plus mix x its own samples on every hit (see the module doc). others: hit
    triggers (0..1 per sample) of the other drums; a hit is clean when those named in
    cfg["clean_of"] are silent around it. Returns the stem, stats and the samples."""
    n = len(x)
    L = int(cfg["length_ms"] / 1000 * fs)
    pre = int(0.002 * fs)
    starts, vel = drum_hits(x, fs, cfg["fmin_hz"], cfg["fmax_hz"])
    starts, vel = starts[(starts > pre) & (starts + L < n)], vel[(starts > pre) & (starts + L < n)]
    lib_cfg = cfg["library"]
    if len(starts) < (3 if lib_cfg["enabled"] else cfg["min_hits"]):
        return x, {"hits": int(len(starts)), "samples": 0}, []
    # clean candidates: no other drum hitting from 20 ms before to the end of the sample,
    # and none of the stem's own hits either; the louder half of the hits
    busy = np.zeros(n, bool)
    for k, t in (others or {}).items():
        if k in cfg["clean_of"]:
            busy |= t > 0.3
    clean = []
    for i, s in enumerate(starts):
        own_next = i + 1 < len(starts) and starts[i + 1] < s + L
        if not own_next and not busy[max(0, s - int(0.02 * fs)) : s + L].any() and vel[i] >= np.median(vel):
            clean.append(i)
    if len(clean) < cfg["min_hits"]:  # busy playing: allow other drums from 40 ms after the hit
        clean = [i for i in range(len(starts)) if vel[i] >= np.median(vel)
                 and not (i + 1 < len(starts) and starts[i + 1] < starts[i] + min(L // 3, int(0.15 * fs)))
                 and not busy[max(0, starts[i] - int(0.01 * fs)) : starts[i] + int(0.04 * fs)].any()]
    if len(clean) < cfg["min_hits"] and not lib_cfg["enabled"]:
        return x, {"hits": int(len(starts)), "samples": 0, "note": "too few clean hits"}, []
    if not cfg["own"]:  # no own samples (hi-hat: an average of noisy hits is a dull blur): the library only
        clean = []
    seg = lambda s: x[s - pre : s - pre + L]  # noqa: E731
    # which drum (toms) and velocity layer each hit belongs to
    groups = np.zeros(len(starts), int)
    if cfg["pitch_clusters"] > 1:
        f = np.array([_pitch(seg(s)[: int(0.15 * fs)], fs) for s in starts])
        groups = _clusters(f, cfg["pitch_clusters"])
    fade = np.ones(L)
    k = int(0.3 * L)
    fade[-k:] = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    max_lag = int(0.002 * fs)
    samples, table = [], {}  # (group, layer) -> sample index
    for g in np.unique(groups):
        cand = [i for i in clean if groups[i] == g]
        if len(cand) < max(4, cfg["min_hits"] // 2):
            continue
        cand = sorted(cand, key=lambda i: -vel[i])[: cfg["max_hits"] * cfg["layers"]]
        layers = np.array_split(sorted(cand, key=lambda i: vel[i]), cfg["layers"]) if len(cand) >= 8 * cfg["layers"] else [cand]
        for li, members in enumerate(layers):
            members = list(members)
            ref = seg(starts[members[-1]])  # the strongest of the layer
            acc = np.zeros_like(ref)
            for i in members:
                lag = _lag(ref[: int(0.03 * fs)], x, starts[i] - pre, max_lag)
                s = seg(starts[i] + lag)
                acc += s / (vel[i] + EPS)
            smp = acc / len(members) * fade[:, None]
            table[(g, li)] = len(samples)
            samples.append(smp)
    if not samples and not lib_cfg["enabled"]:
        return x, {"hits": int(len(starts)), "samples": 0, "note": "too few clean hits"}, []
    lay_edges = {}
    for g in np.unique(groups):
        v = np.sort(vel[groups == g])
        nl = max(li for (gg, li) in table if gg == g) + 1 if any(gg == g for gg, _ in table) else 0
        lay_edges[g] = np.quantile(v, np.arange(1, nl) / nl) if nl > 1 else np.array([])
    out = np.zeros_like(x)
    placed, gains, skipped = 0, [], 0
    floor = np.max(vel) * 10 ** (-cfg["place_above_db"] / 20)
    w = int(0.025 * fs)
    real = []  # hits that pass the checks: the library sample goes there too
    for i, s in enumerate(starts):
        if vel[i] < floor:  # a faint "hit" is bleed of another drum: no sample there
            skipped += 1
            continue
        if not samples:  # no own samples (too few clean hits): only the level check
            real.append(i)
            continue
        g = groups[i]
        if (g, 0) not in table:
            g = min({gg for gg, _ in table}, key=lambda gg: abs(gg - g))
        li = int(np.searchsorted(lay_edges.get(g, np.array([])), vel[i]))
        smp = samples[table.get((g, li), table[(g, 0)])]
        # line it up with this hit, then match its level over the attack (least squares)
        a = s - pre + _lag(smp[:w], x, s - pre, max_lag)
        if a < 0 or a + L > n:
            continue
        hit = x[a : a + w]
        dot = float(np.sum(hit * smp[:w]))
        corr = dot / (np.sqrt(np.sum(hit ** 2) * np.sum(smp[:w] ** 2)) + EPS)
        if corr < cfg["min_corr"]:  # does not look like this drum: something else, leave it alone
            skipped += 1
            continue
        gain = dot / (np.sum(smp[:w] ** 2) + EPS)
        gain = min(gain, 1.5 * vel[i])  # the samples have unit velocity: no runaway on a bleed-heavy hit
        out[a : a + L] += gain * smp
        placed += 1
        gains.append(gain / (vel[i] + EPS))
        real.append(i)
    y = x + cfg["mix"] * out
    st = {"hits": int(len(starts)), "placed": placed, "skipped": skipped, "samples": len(samples),
          "drums": int(len({g for g, _ in table})), "mix": cfg["mix"],
          "fit": round(float(np.median(gains)), 2) if gains else None}
    if lib_cfg["enabled"] and real:
        lib_out, st["library"], lib_samples = _library_layer(x, fs, starts, vel, groups, real, samples, table,
                                                             lib_cfg | {"fmin_hz": cfg["fmin_hz"], "fmax_hz": cfg["fmax_hz"]})
        if lib_out is not None:
            y = y + lib_cfg["mix"] * lib_out
            samples = samples + lib_samples
    st["level_change_db"] = round(float(10 * np.log10((np.mean(y * y) + EPS) / (np.mean(x * x) + EPS))), 1)
    return y, st, samples


_LIBRARY: dict = {}


def _load_library(cfg: dict) -> dict | None:
    """The studio sample library (tools/build_drum_samples.py): index and samples."""
    import json
    import os
    from pathlib import Path

    import soundfile as sf

    d = Path(os.path.expanduser(cfg["dir"]))
    if str(d) not in _LIBRARY:
        idx = d / "index.json"
        if not idx.exists():
            _LIBRARY[str(d)] = None
        else:
            lib = json.loads(idx.read_text())
            for drum, entries in lib["drums"].items():
                for e in entries:
                    for layer in ("loud", "soft"):
                        if layer in e:
                            e[f"_{layer}"] = sf.read(d / drum / e[layer], dtype="float64")[0]
            _LIBRARY[str(d)] = lib
    return _LIBRARY[str(d)]


def _centroid(seg: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Spectral centroid of a hit within lo..hi (hi-hat: how bright)."""
    f, p = signal.periodogram(seg.mean(axis=1) * np.hanning(len(seg)), fs, nfft=1 << 13)
    m = (f >= lo) & (f <= hi)
    return float(np.sum(f[m] * p[m]) / (np.sum(p[m]) + EPS))


def _library_layer(x, fs, starts, vel, groups, real, samples, table, cfg, stops=None):
    """Studio one-shots of this drum, matched to it (nearest pitch, tuned within
    max_pitch_shift; or, with match = "brightness", the nearest spectral centroid, untuned),
    on the same hits as the own samples: loud or soft layer by velocity, aligned to each hit,
    level-matched over the attack (energy; within fmin..fmax with band_level, so bleed in the
    stem does not count), polarity by majority. Optionally: only entries with a spectral
    flatness of at least min_flatness (noise-like: no pitched ping to clash), the samples
    high-passed at hp_hz, and each placed sample cut (10 ms fade) at stops[i] (the next hit:
    no sample sounding over the next one's, which phases)."""
    from fractions import Fraction

    lib = _load_library(cfg)
    entries = (lib or {}).get("drums", {}).get(cfg["drum"], [])
    if cfg.get("min_flatness"):
        entries = [e for e in entries if e.get("flatness", 1.0) >= cfg["min_flatness"]]
    if not entries:
        return None, {"note": "no sample library (tools/build_drum_samples.py)"}, []
    n, pre, w = len(x), int(0.002 * fs), int(0.025 * fs)
    max_lag = int(0.002 * fs)
    chosen, info = {}, []
    for g in np.unique(groups[real]):
        own = [samples[table[k]] for k in sorted(table) if k[0] == g]
        hits_g = [i for i in real if groups[i] == g]
        ref = own[-1] if own else x[starts[hits_g[0]] - pre : starts[hits_g[0]] - pre + int(0.3 * fs)]
        if cfg["match"] == "brightness":  # the song's typical hit (median centroid of its loud hits), untuned
            loud = [i for i in hits_g if vel[i] >= np.median(vel[hits_g])][:200]
            target = float(np.median([_centroid(x[starts[i] : starts[i] + int(0.05 * fs)], fs, cfg["fmin_hz"], cfg["fmax_hz"])
                                      for i in loud]))
            # the library's hats measured the same way (the index's centroid is full-band, bleed included)
            cen = lambda e: _centroid(e["_loud"][: int(0.05 * lib["fs"])][:, None], lib["fs"], cfg["fmin_hz"], cfg["fmax_hz"])  # noqa: E731
            e = min(entries, key=lambda e: abs(np.log(cen(e) / target)))
            r = 1.0
        else:
            target = _pitch(ref[: int(0.15 * fs)], fs)
            e = min(entries, key=lambda e: abs(np.log(e["pitch_hz"] / target)))
            r = float(np.clip(target / e["pitch_hz"], 1 - cfg["max_pitch_shift"], 1 + cfg["max_pitch_shift"]))
        fr = Fraction(1 / r).limit_denominator(64)
        layers = {}
        for layer in ("loud", "soft"):
            if f"_{layer}" in e:
                s1 = signal.resample_poly(e[f"_{layer}"], fr.numerator, fr.denominator)  # tuned: r x the pitch
                if lib["fs"] != fs:
                    s1 = signal.resample_poly(s1, int(fs), int(lib["fs"]))
                if cfg.get("hp_hz"):
                    s1 = filtfilt(signal.butter(4, cfg["hp_hz"], "high", fs=fs, output="sos"), s1)
                s2 = np.column_stack([s1, s1])
                layers[layer] = (s2, _start(s2, fs, cfg["fmin_hz"], cfg["fmax_hz"]))
        chosen[g] = layers
        info.append({"drum": int(g), "sample": e["name"], ("centroid_hz" if cfg["match"] == "brightness" else "pitch_hz"): round(target, 1),
                     "tuned": round(r, 3)})
    med = {g: np.median(vel[[i for i in real if groups[i] == g]]) for g in chosen}
    bandpass = (lambda y: filtfilt(signal.butter(4, [cfg["fmin_hz"], cfg["fmax_hz"]], "bandpass", fs=fs, output="sos"), y)) \
        if cfg["band_level"] else (lambda y: y)
    fl = cfg.get("flatten", 0.0)  # 0: follow each hit's level; 1: every hit the same (the drum's median)
    plan = []
    for i in real:
        g = groups[i]
        layers = chosen[g]
        smp, s0 = layers["loud"] if vel[i] >= med[g] or "soft" not in layers else layers["soft"]
        s = starts[i]
        a = s - s0 + _lag(smp[s0 : s0 + w], x, s, max_lag, True)  # its start on the hit's start, then refined
        if a < 0 or a + len(smp) > n:
            continue
        hit = x[a : a + w]
        dot = float(np.sum(hit * smp[:w]))
        gain = np.sqrt(np.sum(bandpass(x[max(0, a - w) : a + 2 * w])[w if a >= w else a : (w if a >= w else a) + w] ** 2)
                       / (np.sum(bandpass(smp[: 2 * w])[:w] ** 2) + EPS)) if cfg["band_level"] else \
            np.sqrt(np.sum(hit ** 2) / (np.sum(smp[:w] ** 2) + EPS))
        gain *= (med[g] / (vel[i] + EPS)) ** fl  # flattened towards the median hit's level
        if stops is not None:  # cut at the next hit, a 10 ms fade
            m = min(len(smp), max(int(0.02 * fs), stops[i] - a))
            if m < len(smp):
                f = int(0.01 * fs)
                smp = smp[:m].copy()
                smp[-f:] *= np.linspace(1, 0, f)[:, None]
        plan.append((a, gain, smp, dot, g))
    out = np.zeros_like(x)
    for g in chosen:  # polarity: as most hits of this drum say
        sign = 1.0 if sum(np.sign(p[3]) for p in plan if p[4] == g) >= 0 else -1.0
        for a, gain, smp, _, gg in plan:
            if gg == g:
                out[a : a + len(smp)] += sign * gain * smp
    tuned = [layers["loud"][0] for layers in chosen.values()]
    return out, {"placed": len(plan), "mix": cfg["mix"], "samples": info}, tuned


def _flatness(seg: np.ndarray, fs: float, lo: float = 1000.0, hi: float = 8000.0) -> float:
    """Spectral flatness lo..hi (geometric / arithmetic mean of the power): low = tonal (a bell)."""
    f, p = signal.periodogram(seg.mean(axis=1) * np.hanning(len(seg)), fs, nfft=1 << 13)
    m = (f >= lo) & (f <= hi)
    q = p[m] + 1e-20
    return float(np.exp(np.mean(np.log(q))) / np.mean(q))


def cymbal_hits(x: np.ndarray, fs: float, lo: float, hi: float, rise_db: float = 6.0, min_gap_s: float = 0.08
                ) -> tuple[np.ndarray, np.ndarray]:
    """Cymbal hit starts and levels: where the band's level (5 ms frames) rises rise_db within
    30 ms. A crash's long, noisy decay fluctuates by far less, so it is not taken for new hits
    (as peak picking on the envelope did), and quieter hits after a loud one are still found."""
    hop = int(0.005 * fs)
    band = filtfilt(signal.butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), x).mean(axis=1)
    m = len(band) // hop * hop
    db = 10 * np.log10(np.mean(band[:m].reshape(-1, hop) ** 2, axis=1) + EPS)
    prev = np.array([db[max(0, i - 6) : i].min() if i else db[0] for i in range(len(db))])  # the lowest over 30 ms before
    onset = np.flatnonzero((db - prev >= rise_db) & (np.concatenate([[True], (db - prev)[:-1] < rise_db])))
    starts, levels, last = [], [], -10 ** 9
    for i in onset:
        if (i - last) * hop < min_gap_s * fs:
            continue
        last = i
        starts.append(i * hop)
        levels.append(float(np.sqrt(10 ** (db[i : i + 6].max() / 10))))
    return np.array(starts, dtype=int), np.array(levels)


def cymbal_augment(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, dict, list[np.ndarray]]:
    """Studio cymbal one-shots on the cymbals stem's hits, by kind: each hit is a bell
    (tonal: spectral flatness in fmin..fmax under bell_flatness x the stem's median), a crash
    (among the loudest crash_pct % and still within crash_floor_db of its peak all through
    crash_hold_s) or a ride stroke; it gets a single real hit of that kind from the library
    (tools/build_drum_samples.py), the nearest in brightness, untuned, loud or soft by
    velocity, aligned and level-matched in fmin..fmax over the attack (the stem's bleed does
    not count), at mix[kind] under it. Returns the stem, stats and the samples used."""
    n = len(x)
    starts, vel = cymbal_hits(x, fs, cfg["fmin_hz"], cfg["fmax_hz"])
    keep = (starts > int(0.002 * fs)) & (starts + int(0.1 * fs) < n) & (vel >= vel.max(initial=0) * 10 ** (-cfg["place_above_db"] / 20))
    starts, vel = starts[keep], vel[keep]
    if len(starts) < 3:
        return x, {"hits": int(len(starts)), "note": "too few hits"}, []
    env = _band_env(x, fs, cfg["fmin_hz"], cfg["fmax_hz"])
    fl = np.array([_flatness(x[s : s + int(0.08 * fs)], fs, cfg["fmin_hz"], cfg["fmax_hz"]) for s in starts])
    # a crash keeps ringing: over crash_hold_s the band level never falls crash_floor_db under the hit's peak,
    # even with other strokes on top (ride strokes decay between hits: in a busy song the gap to the next hit
    # says nothing)
    floor = []
    a0 = int(0.03 * fs)
    for s in starts:
        seg = env[s : min(n, s + int(cfg["crash_hold_s"] * fs))]
        floor.append(20 * np.log10((seg[a0:].min() if len(seg) > a0 else 0.0) / (seg[:a0].max() + EPS) + EPS))
    floor = np.array(floor)
    kind = np.where(fl < cfg["bell_flatness"] * np.median(fl), "bell",
                    np.where((vel >= np.percentile(vel, 100 - cfg["crash_pct"])) & (floor > -cfg["crash_floor_db"]), "crash", "ride"))
    y = x.copy()
    stats = {"hits": int(len(starts)), "kinds": {k: int(np.sum(kind == k)) for k in ("crash", "ride", "bell")},
             "crash_at_s": [round(float(s) / fs, 1) for s in starts[kind == "crash"]][:40]}
    used = []
    for k in ("crash", "ride", "bell"):
        idx = np.flatnonzero(kind == k)
        if len(idx) == 0 or cfg["mix"][k] <= 0:
            continue
        lcfg = {"dir": cfg["dir"], "drum": k, "match": "brightness", "band_level": True, "fmin_hz": cfg["fmin_hz"],
                "fmax_hz": cfg["fmax_hz"], "max_pitch_shift": 0.0, "flatten": 0.0, "mix": cfg["mix"][k],
                "hp_hz": cfg["hp_hz"], "min_flatness": cfg["min_flatness"].get(k, 0.0)}
        stops = np.append(starts[1:], n)[idx]  # the next hit of any kind
        out, st, smp = _library_layer(x, fs, starts[idx], vel[idx], np.zeros(len(idx), int), list(range(len(idx))), [], {}, lcfg,
                                      stops=stops)
        if out is not None:
            y = y + cfg["mix"][k] * out
            used += smp
        stats[k] = st | {"mix": cfg["mix"][k]}
    stats["level_change_db"] = round(float(10 * np.log10((np.mean(y * y) + EPS) / (np.mean(x * x) + EPS))), 1)
    return y, stats, used


# --- bass ----------------------------------------------------------------------------------------


def pyin(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """pyin's raw output on the bass (f0, voiced flag, voiced probability, frame times),
    as bass_pitch sets it up."""
    import librosa

    mono = x.mean(axis=1)
    sr = 4000
    lo = signal.resample_poly(filtfilt(signal.butter(4, 600, "low", fs=fs, output="sos"), mono), sr, int(fs))
    f0, voiced, prob = librosa.pyin(lo.astype(np.float32), fmin=cfg["fmin_hz"], fmax=cfg["fmax_hz"], sr=sr,
                                    frame_length=2048, hop_length=40, center=True)
    return f0, voiced, prob, np.arange(len(f0)) * 40 / sr


def bass_pitch(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """The bass's pitch, per 10 ms frame (NaN where no note is found), and the frame times:
    pyin on the band below 600 Hz, a 0.5 s window, down to fmin_hz. A long window and a low
    fmin find the low notes even where the room and the recorder lost most of their
    fundamental (on the band's recordings E1/F#1/A1 are found; with a 0.25 s window they
    were read as the octave above)."""
    f0, voiced, prob, t = pyin(x, fs, cfg)
    return np.where(voiced & (prob >= cfg["voiced_prob"]) & np.isfinite(f0), f0, np.nan), t


def harmonic_clean(x: np.ndarray, fs: float, cfg: dict, f0: np.ndarray, t_f: np.ndarray) -> tuple[np.ndarray, dict]:
    """Turn down what lies between the harmonics of the note being played (buzz, bleed,
    separation noise), by up to depth_db, between fmin_hz and fmax_hz, where the pitch is
    known; attacks (no pitch yet) and gaps are left alone. Up to comb_harmonics x f0 the
    harmonics are where the pitch puts them (within tol_cents / tol_hz, plus the string's
    inharmonicity: partial k about 866 stretch k^2 cents sharp), which also tells them from
    tonal bleed; above, where a real string's partials run too sharp to predict, every
    spectral peak standing peak_db over its neighbourhood is kept, whatever its frequency.
    Soft edges; in blocks, so memory does not grow with the song's length."""
    from scipy import ndimage

    nfft, hop = cfg["nfft"], cfg["nfft"] // 8
    floor = 10 ** (-cfg["depth_db"] / 20)
    ok = np.isfinite(f0)
    out = np.empty_like(x)
    block, margin = int(30 * fs), 2 * nfft
    removed = kept = 0.0
    for s in range(0, len(x), block):
        e = min(len(x), s + block)
        a, b = max(0, s - margin), min(len(x), e + margin)
        f, t, Z = signal.stft(x[a:b].T, fs, nperseg=nfft, noverlap=nfft - hop)
        t = t + a / fs
        p = np.interp(t, t_f, np.where(ok, f0, 0.0))
        voiced = (np.interp(t, t_f, ok.astype(float)) > 0.99) & (p > 0)
        band = (f >= cfg["fmin_hz"]) & (f <= cfg["fmax_hz"])
        fb = f[band]
        # the peaks of the (channel-summed) spectrum: local maxima peak_db over the local median
        mag = np.abs(Z).sum(axis=0)[band]
        med = ndimage.median_filter(mag, size=(31, 1), mode="nearest")
        peak = (mag >= med * 10 ** (cfg["peak_db"] / 20)) & (mag >= ndimage.maximum_filter(mag, size=(3, 1)))
        peak = ndimage.binary_dilation(peak, structure=np.ones((5, 1), bool))  # the peak's main lobe
        G = np.ones((len(f), len(t)), dtype=np.float32)
        half = nfft / 2 / fs
        for j in np.nonzero(voiced)[0]:
            # every note sounding within this frame's window (a note change inside it: both notes' harmonics stay)
            w = (t_f >= t[j] - half) & (t_f <= t[j] + half) & ok
            if not w.any():
                continue
            comb = np.zeros(len(fb))
            for q in np.unique(np.round(12 * np.log2(f0[w] / 55.0) * 4) / 4):  # quarter-semitone steps
                pq = 55.0 * 2 ** (q / 12)
                k = np.maximum(1, np.round(fb / pq))
                # a string's partial k runs sharp by about 866 B k^2 cents (inharmonicity B): the tolerance grows with it
                # never under two STFT bins: a harmonic's energy spreads over its window's main lobe
                tol = np.maximum(max(cfg["tol_hz"], 2 * fs / nfft),
                                 fb * (2 ** ((cfg["tol_cents"] + 866 * cfg["stretch"] * k * k) / 1200) - 1))
                d = np.abs(fb - k * pq) / tol
                comb = np.maximum(comb, np.where(d <= 1, 1.0, np.where(d >= 2, floor, floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * (d - 1))))))
            k = np.maximum(1, np.round(fb / p[j]))
            G[band, j] = np.where(k <= cfg["comb_harmonics"], comb, np.where(peak[:, j], 1.0, floor))
        G = signal.convolve2d(G, np.ones((3, 3)) / 9, mode="same", boundary="symm")  # no flutter, soft edges
        y = signal.istft(Z * G[None], fs, nperseg=nfft, noverlap=nfft - hop)[1].T
        y = np.pad(y, ((0, max(0, b - a - len(y))), (0, 0)))[s - a:e - a]
        out[s:e] = y
        removed += float(np.sum((x[s:e] - y) ** 2))
        kept += float(np.sum(x[s:e] ** 2))
    return out, {"removed_db": round(10 * np.log10(removed / (kept + EPS) + EPS), 1), "depth_db": cfg["depth_db"],
                 "pitched_pct": round(100 * float(ok.mean()), 0)}


def bass_fundamental(x: np.ndarray, fs: float, cfg: dict, pitch: tuple[np.ndarray, np.ndarray] | None = None
                     ) -> tuple[np.ndarray, dict, np.ndarray]:
    """x with its fundamental raised by boost_db (see the module doc), extracted in phase
    along the pitch. With restore: where a note's fundamental is weaker than restore_db
    under its 2nd harmonic (low notes whose fundamental the room and the recorder lost:
    on the band's recordings 20 dB under, against about equal on raw studio bass tracks),
    it is brought up to that, in phase; otherwise weak fundamentals are filled towards
    the song's usual fundamental-to-note ratio (fill). pitch: from bass_pitch, if the
    clean-up already tracked it, optionally with the note onsets (then attack_db more
    fundamental at each note's start: punch). Returns the stem, stats and the gain trace (dB of
    fundamental change, for the report)."""
    f0, t_f = pitch[:2] if pitch is not None else bass_pitch(x, fs, cfg)
    onsets = pitch[2] if pitch is not None and len(pitch) > 2 else None
    ok = np.isfinite(f0)
    if ok.sum() < 50:
        return x, {"voiced_pct": round(100 * float(ok.mean()), 0), "note": "no stable pitch"}, np.zeros(1)
    mono = x.mean(axis=1)
    t = np.arange(len(x)) / fs
    lf = np.interp(t, t_f[ok], np.log(f0[ok]))  # held across unvoiced gaps
    f_inst = np.exp(lf)
    mask = np.interp(t, t_f, ok.astype(float))
    mask = dynamics.attack_smooth(dynamics.attack_smooth(mask, 20.0, fs)[::-1], 20.0, fs)[::-1]  # 20 ms fades
    phi = 2 * np.pi * np.cumsum(f_inst) / fs
    lp = signal.butter(4, cfg["bandwidth_hz"], "low", fs=fs, output="sos")

    def demod(k):
        return 2 * (filtfilt(lp, mono * np.cos(k * phi)) - 1j * filtfilt(lp, mono * np.sin(k * phi)))

    c = demod(1)  # the fundamental's complex amplitude
    amp = np.abs(c)
    stats = {"voiced_pct": round(100 * float(ok.mean()), 0), "median_f0_hz": round(float(np.median(f0[ok])), 1),
             "boost_db": cfg["boost_db"]}
    if cfg["restore"]:
        amp2 = np.abs(demod(2))
        target = amp2 * 10 ** (cfg["restore_db"] / 20)
        want = np.maximum(amp * 10 ** (cfg["boost_db"] / 20), target)
        want = np.minimum(want, np.maximum(amp * 10 ** (cfg["max_db"] / 20), target))
        on = (mask > 0.9) & (amp2 > np.percentile(amp2[mask > 0.9], 30)) if (mask > 0.9).any() else mask > 0.9
        if on.any():
            stats["fundamental_re_2nd_db"] = round(float(np.median(20 * np.log10((amp[on] + EPS) / (amp2[on] + EPS)))), 1)
            stats["restored_pct"] = round(100 * float(np.mean(target[on] > amp[on] * 10 ** (cfg["boost_db"] / 20))), 0)
    else:
        env = np.sqrt(np.maximum(filtfilt(signal.butter(2, 10.0, "low", fs=fs, output="sos"), mono * mono), 0)) * np.sqrt(2)
        use = (mask > 0.9) & (env > np.percentile(env, 50))
        ratio = float(np.median(amp[use] / (env[use] + EPS))) if use.any() else 0.0
        fill = cfg["fill"] * ratio * env
        # no fill where there is next to no fundamental: a note the pitch tracker got an octave wrong,
        # or nothing really played there; and never more than max_db in all
        fill = np.where(amp >= ratio * env * 10 ** (-cfg["fill_floor_db"] / 20), fill, 0.0)
        want = np.maximum(amp, fill) * 10 ** (cfg["boost_db"] / 20)
        want = np.minimum(want, amp * 10 ** (cfg["max_db"] / 20))
        stats["fundamental_ratio"] = round(ratio, 2)
    if cfg["attack_db"] and onsets is not None and len(onsets):
        # punch: each note's fundamental attack_db higher at its start (10 ms rise), back over attack_ms (hold a third)
        env = np.zeros(len(x))
        n_a = int(cfg["attack_ms"] / 1000 * fs)
        n_r = int(0.01 * fs)  # a 10 ms rise: switched on at once, the step in the low tone clicked
        shape = np.concatenate([0.5 * (1 - np.cos(np.pi * np.arange(n_r) / n_r)), np.ones(n_a // 3),
                                0.5 * (1 + np.cos(np.pi * np.arange(n_a - n_a // 3) / (n_a - n_a // 3)))])
        for on in onsets:
            i = int(on * fs)
            if 0 <= i < len(x):
                seg = shape[:len(x) - i]
                env[i:i + len(seg)] = np.maximum(env[i:i + len(seg)], seg)
        want = want * 10 ** (cfg["attack_db"] * env / 20)
        stats["attack_db"] = cfg["attack_db"]
    add = (want - amp) * mask * np.cos(phi + np.angle(c))
    y = x + add[:, None]
    gain_db = 20 * np.log10((amp + np.abs(want - amp) * mask + EPS) / (amp + EPS))
    # the fundamental's energy change where the pitch was found (not a mean of dB values: faint
    # notes would dominate it)
    on = mask > 0.5
    stats["avg_gain_db"] = round(float(10 * np.log10(np.sum((amp + (want - amp) * mask)[on] ** 2) / (np.sum(amp[on] ** 2) + EPS))), 1)
    return y, stats, dynamics.decimate(gain_db * mask, fs, np.max)
