"""Per-note dynamic EQ for the bass stem: every note keeps its own harmonics, up to where
they still stand out, and an attack as bright as the song's usual one.

Besides the notes, the separated bass carries smear between their harmonics (bleed, the
separation mask's noise), blobs above a note's highest harmonic, and transients that
start no note (kick and snare bleed). The stage-2 chain (tone match, exciter, transient
shaper) then brings them up as clank. Note by note:

  pitch           pyin's track checked against the spectrum (the first harmonics must
                  stand over what lies between them), octave errors fixed, held across
                  the frames pyin drops. pyin's own confidence is low on a bass whose
                  fundamental the room and the recorder lost: on Stanza OK, 33 % of the
                  frames reach the confidence harmonic_clean needs
  harmonic space  per frame, the highest harmonic still ceil_db over its neighbourhood
                  (a few missing ones tolerated: a pick position's nulls), never rising
                  again once the note's attack is over (a string's partials only decay).
                  The harmonics below it are kept; what lies between them and above it
                  goes down by depth_db, and in the gaps between notes everything above
                  gap_keep_hz goes down by gap_depth_db
  attack          note onsets: a rise of the harmonic band, or a new pitch. A note's
                  first attack_ms are left alone by the above and given to the attack
                  EQ: per note and band, the attack's level against the note's body is
                  moved towards the song's median (clanky notes cut, dull ones lifted a
                  little). Transients that start no note get no attack window, so the
                  harmonic clean-up takes them

All in 30 s blocks, so memory does not grow with the song's length.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .loudness import EPS, filtfilt

BLOCK_S = 30.0


def _blocks(n: int, fs: float, hop: int, margin: int):
    """(s, e, a, b): the block [s, e) and its padded span [a, b), all multiples of hop."""
    block = int(BLOCK_S * fs) // hop * hop
    for s in range(0, n, block):
        e = min(n, s + block)
        yield s, e, max(0, s - margin), min(n, e + margin)


def _frames_of(Z: np.ndarray, s: int, e: int, a: int, hop: int) -> tuple[int, int]:
    """The STFT frames (of a block padded from a) whose centres lie in [s, e)."""
    j0 = (s - a) // hop
    return j0, min(Z.shape[-1], j0 + -(-(e - s) // hop))


def _bands(df: float, nb: int, fmax: float) -> np.ndarray:
    """Third-octave band edges (bins) from 50 Hz to fmax: where the harmonic space is measured."""
    hz = 50.0 * 2 ** (np.arange(0, int(3 * np.log2(fmax / 50.0)) + 1) / 3)
    return np.unique(np.clip(np.round(hz / df).astype(int), 1, nb))


def _centres(c: float, fmax: float, B: float) -> np.ndarray:
    """Where the harmonics of c are: k c, stretched by the string's inharmonicity B."""
    k = np.arange(1, int(fmax / c) + 1)
    return k * c * np.sqrt(1 + B * k * k)


def _hnr(P: np.ndarray, c: float, df: float, edges: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per band, the mean power on the comb of pitch c over the mean power between its
    teeth (dB: about 0 for noise, whatever the band's level), and each band's peak level
    (dB). Bands with no tooth or no gap: NaN."""
    comb = _comb(len(P), df, _centres(c, cfg["fmax_hz"], cfg["stretch"]), cfg, c)
    on, off = comb >= 0.5, comb == 0.0
    e = edges
    pon, non = np.add.reduceat(P * on, e[:-1]), np.add.reduceat(on.astype(float), e[:-1])
    poff, noff = np.add.reduceat(P * off, e[:-1]), np.add.reduceat(off.astype(float), e[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        hnr = 10 * np.log10((pon / non + EPS) / (poff / noff + EPS))
    hnr[(non == 0) | (noff == 0)] = np.nan
    return hnr, 10 * np.log10(np.maximum.reduceat(P, e[:-1]) + EPS)


def _score(hnrs: list[np.ndarray]) -> np.ndarray:
    """How much each candidate pitch looks like the note: its comb's mean contrast over the
    octave bands (80 Hz - 1.3 kHz) where every candidate has teeth and gaps, so they are
    judged on the same ground (an octave too high: the odd harmonics fall between the
    teeth; too low: half of the teeth fall between the harmonics)."""
    H = np.array(hnrs)
    common = np.isfinite(H).all(axis=0)
    if not common.any():
        return np.full(len(hnrs), -np.inf)
    return np.mean(np.clip(H[:, common], -6.0, 30.0), axis=1)


def _ceiling(hnr: np.ndarray, level: np.ndarray, edges: np.ndarray, df: float, cfg: dict) -> float:
    """The note's harmonic space (Hz): up to the last band whose comb stands ceil_db over
    what lies between its teeth, no more than space_range_db under the note's loudest
    band, before more than ceil_gap bands in a row do not."""
    ok = np.nan_to_num(hnr, nan=-99.0) >= cfg["ceil_db"]
    ok &= level >= np.max(level) - cfg["space_range_db"]
    top, miss = -1, 0
    for i, v in enumerate(ok):
        if not np.isfinite(hnr[i]):
            continue  # no harmonic in the band (low bands are narrower than the harmonics' spacing)
        if v:
            top, miss = i, 0
        else:
            miss += 1
            if miss > cfg["ceil_gap"] and top >= 0:
                break
    return float(edges[top + 1] * df) if top >= 0 else 0.0


def _comb(nb: int, df: float, centres: np.ndarray, cfg: dict, c: float, glide: float = 1.0) -> np.ndarray:
    """1 on the harmonics (within tol_cents / tol_hz, never under two bins), a raised-cosine
    edge out to twice that, 0 elsewhere. On the lowest notes, whose harmonics are only a
    few bins apart, the teeth narrow to 0.2 c, the edge to 0.4 c: something stays between.
    glide: the ratio of the highest to the lowest pitch within the frame (a slide, a bend):
    each tooth widens by the half of how far its harmonic travels."""
    f = np.arange(nb) * df
    tol = np.minimum(np.maximum(max(cfg["tol_hz"], 2 * df), centres * (2 ** (cfg["tol_cents"] / 1200) - 1)), 0.2 * c)
    tol = tol + centres * (glide - 1) / 2
    i = np.clip(np.searchsorted(centres, f), 1, len(centres) - 1) if len(centres) > 1 else np.zeros(nb, int)
    cand = [i - 1, i] if len(centres) > 1 else [i]
    d = np.min([np.abs(f - centres[j]) / tol[j] for j in cand], axis=0)
    return np.where(d <= 1, 1.0, np.where(d >= 2, 0.0, 0.5 * (1 + np.cos(np.pi * (d - 1)))))


def _onsets(mono: np.ndarray, fs: float, f_a: np.ndarray, t_a: np.ndarray, cfg: dict
            ) -> tuple[np.ndarray, int]:
    """Note onsets (s): energy onsets (backtracked to where the rise starts) where the
    note's harmonic band (150-800 Hz, above the kick's body) rises by rise_db, or a new
    pitch (not just another octave) begins. The others (drum bleed, a pick scrape) only
    counted."""
    import librosa

    hop = 256
    env = librosa.onset.onset_strength(y=mono.astype(np.float32), sr=fs, hop_length=hop, n_fft=2048,
                                       fmax=1000.0, n_mels=48)
    peak = librosa.onset.onset_detect(onset_envelope=env, sr=fs, hop_length=hop, units="frames")
    on = librosa.frames_to_time(librosa.onset.onset_backtrack(peak, env), sr=fs, hop_length=hop)
    band = filtfilt(signal.butter(2, [150.0, 800.0], "bandpass", fs=fs, output="sos"), mono) ** 2
    cs = np.concatenate([[0.0], np.cumsum(band)])

    def level(t1, t2):
        i1, i2 = np.clip(int(t1 * fs), 0, len(band) - 1), np.clip(int(t2 * fs), 1, len(band))
        return (cs[max(i2, i1 + 1)] - cs[i1]) / max(i2 - i1, 1)

    def pitch(t1, t2):
        sel = (t_a >= t1) & (t_a < t2) & np.isfinite(f_a)
        return float(np.median(f_a[sel])) if sel.any() else np.nan

    keep = []
    for t, p in zip(on, peak):
        after = pitch(t + 0.02, t + 0.15)
        if not np.isfinite(after):
            continue  # no note follows
        before = pitch(t - 0.12, t - 0.02)
        # a new pitch; an octave's jump is the tracker's doubt unless picked (then it rises)
        r = 12 * np.log2(after / before) if np.isfinite(before) else 0.0
        new = not np.isfinite(before) or (abs(r) >= 0.6 and abs(r - 12 * np.round(r / 12)) >= 0.6)
        rise = 10 * np.log10((level(t + 0.01, t + 0.05) + EPS) / (level(t - 0.05, t - 0.01) + EPS))
        if new or rise >= cfg["rise_db"]:
            keep.append((t, env[p]))
    # one onset per note: of those closer than min_gap_s (the window reads a new pitch early), the strongest
    out: list = []
    for t, strength in keep:
        if out and t - out[-1][0] < cfg["min_gap_s"]:
            if strength > out[-1][1]:
                out[-1] = (t, strength)
            continue
        out.append((t, strength))
    return np.array([t for t, _ in out]), len(on) - len(out)


def _attack_eq(y: np.ndarray, fs: float, cfg: dict, onsets: np.ndarray, stats: dict) -> np.ndarray:
    """Per note and band (bands_hz), the attack (its first attack_ms) against the note's
    body (40-160 ms, or up to the next note): moved towards the song's median by
    strength, past tol_db either way; at most max_cut_db down, max_boost_db up. Applied
    with a 5 ms rise before the onset, a hold, and a release_ms raised-cosine fall. A lift
    only where the attack stands 6 dB over the band's level just before it."""
    a = cfg["attack"]
    edges = np.array(a["bands_hz"], float)
    edges = edges[edges < fs / 2]
    nb = len(edges) - 1
    mono = y.mean(axis=1)
    n = len(mono)
    att = a["ms"] / 1000

    def sums(sig):  # cumulative energy for the windows below
        return np.concatenate([[0.0], np.cumsum(sig * sig)])

    def mean(cs, t1, t2):
        i1, i2 = int(np.clip(t1 * fs, 0, n - 1)), int(np.clip(t2 * fs, 1, n))
        return (cs[max(i2, i1 + 1)] - cs[i1]) / max(i2 - i1, 1)

    nxt = np.append(onsets[1:], n / fs)
    full = sums(mono)
    body = np.array([mean(full, t + 0.04, min(t + 0.16, nx - 0.005)) if nx - t > 0.06 else np.nan
                     for t, nx in zip(onsets, nxt)])
    ratio = np.full((len(onsets), nb), np.nan)
    jump = np.full((len(onsets), nb), np.nan)  # the attack over the band's level just before it
    for b in range(nb):
        cs = sums(filtfilt(signal.butter(4, [edges[b], edges[b + 1] * (0.999 if edges[b + 1] >= fs / 2 else 1)],
                                         "bandpass", fs=fs, output="sos"), mono))
        at = np.array([mean(cs, t - 0.005, t + att) for t in onsets])
        ratio[:, b] = 10 * np.log10((at + EPS) / (body + EPS))
        jump[:, b] = 10 * np.log10((at + EPS) / (np.array([mean(cs, t - 0.04, t - 0.01) for t in onsets]) + EPS))
        del cs
    loud = np.isfinite(body) & (body > np.nanpercentile(body, 90) * 10 ** (-30 / 10)) if np.isfinite(body).any() \
        else np.zeros(len(onsets), bool)
    corr = np.zeros((len(onsets), nb))
    if loud.sum() >= 8:
        target = np.nanmedian(ratio[loud], axis=0)
        d = ratio - target
        cut = np.clip(-a["strength"] * (d - a["tol_db"]), -a["max_cut_db"], 0.0)
        # lifted only where the note has an attack of its own in the band (else it would lift noise or bleed)
        lift = np.clip(a["strength"] * (-d - a["tol_db"]), 0.0, a["max_boost_db"]) * (jump >= 6.0)
        corr = np.where(d > a["tol_db"], cut, np.where(d < -a["tol_db"], lift, 0.0))
        corr[~np.isfinite(corr) | ~np.isfinite(body)[:, None]] = 0.0
        stats["attack"] = {"notes": int(len(onsets)), "bands_hz": edges.tolist(),
                           "median_attack_re_body_db": np.round(target, 1).tolist(),
                           "notes_cut_pct": round(100 * float(np.mean((corr < -0.5).any(axis=1))), 0),
                           "notes_lifted_pct": round(100 * float(np.mean((corr > 0.5).any(axis=1))), 0),
                           "cut_db_p90": np.round(np.percentile(corr, 10, axis=0), 1).tolist()}
    else:
        stats["attack"] = {"notes": int(len(onsets)), "note": "too few notes to learn the song's attack"}
        return y
    # per-bin weights: linear in log frequency between the band centres, flat above, fading out an octave below
    nfft, hop = 1024, 256
    f = np.fft.rfftfreq(nfft, 1 / fs)
    cen = np.sqrt(edges[:-1] * edges[1:])
    lf = np.log(np.maximum(f, 1.0))
    W = np.zeros((len(f), nb))
    for b in range(nb):
        e = np.zeros(nb)
        e[b] = 1.0
        W[:, b] = np.interp(lf, np.log(cen), e)
    W *= np.clip(np.log2(np.maximum(f, 1.0) / (cen[0] / 2)), 0, 1)[:, None]
    out = np.empty_like(y)
    for s, e, aa, bb in _blocks(n, fs, hop, 4 * nfft):
        _, _, Z = signal.stft(y[aa:bb].T.astype(np.float32), fs, nperseg=nfft, noverlap=nfft - hop)
        t = (aa + np.arange(Z.shape[-1]) * hop) / fs
        D = np.zeros((nb, len(t)))
        for i in np.nonzero((onsets > t[0] - 0.2) & (onsets < t[-1] + 0.01))[0]:
            on = onsets[i]
            env = np.clip((t - on + 0.005) / 0.005, 0, 1)
            fall = np.clip((t - on - att) / (a["release_ms"] / 1000), 0, 1)
            env = env * 0.5 * (1 + np.cos(np.pi * fall))
            D = np.where(env[None] > 0, np.where(np.abs(corr[i, :, None] * env[None]) > np.abs(D),
                                                 corr[i, :, None] * env[None], D), D)
        G = 10 ** ((W @ D) / 20)
        yb = signal.istft(Z * G[None].astype(np.float32), fs, nperseg=nfft, noverlap=nfft - hop)[1].T
        yb = np.pad(yb, ((0, max(0, bb - aa - len(yb))), (0, 0)))
        out[s:e] = yb[s - aa:e - aa]
    return out


def note_eq(x: np.ndarray, fs: float, cfg: dict, pitch_cfg: dict, parts: dict | None = None
            ) -> tuple[np.ndarray, dict, tuple[np.ndarray, np.ndarray]]:
    """The bass, note by note (see the module doc). pitch_cfg: the pitch tracker's
    settings ([..bass_fundamental]). Returns the stem, stats, and the pitch for the
    fundamental (fundamental_pitch: the checked one, or pyin's as augment.bass_pitch
    gives it) with the note onsets. parts, if a dict, gets "harmonic_space": the stem
    before the attack EQ."""
    from . import augment

    f0, voiced, prob, t_p = augment.pyin(x, fs, pitch_cfg)
    old_pitch = (np.where(voiced & (prob >= pitch_cfg["voiced_prob"]) & np.isfinite(f0), f0, np.nan), t_p)
    vf = voiced & np.isfinite(f0)
    n = len(x)
    mono = x.mean(axis=1)
    nfft = cfg["nfft"]
    hop = nfft // 16
    df = fs / nfft
    nb = int(cfg["fmax_hz"] / df) + 1
    n_fr = (n - 1) // hop + 1
    t_a = np.arange(n_fr) * hop / fs
    margin = 2 * nfft
    # 1. every frame's spectrum (mono, dB, up to fmax_hz)
    L = np.empty((n_fr, nb), np.float32)
    for s, e, a, b in _blocks(n, fs, hop, margin):
        _, _, Z = signal.stft(mono[a:b].astype(np.float32), fs, nperseg=nfft, noverlap=nfft - hop)
        j0, j1 = _frames_of(Z, s, e, a, hop)
        g0 = s // hop
        L[g0:g0 + j1 - j0] = 20 * np.log10(np.abs(Z[:nb, j0:j1]).T + 1e-9)
        del Z
    alive = L.max(axis=1) > np.percentile(L.max(axis=1), 99) - cfg["range_db"]
    # 2. the pitch, checked: pyin's (or, where it dropped the note, the one held from either side)
    #    and its octaves, or the note going on, whichever's harmonics stand out most; and each frame's
    #    harmonic space
    ip = np.clip(np.round(t_a / (t_p[1] - t_p[0])).astype(int), 0, len(f0) - 1)
    idx = np.arange(len(f0))
    prev = np.maximum.accumulate(np.where(vf, idx, -1))
    nxt = np.minimum.accumulate(np.where(vf, idx, len(f0))[::-1])[::-1]
    hold = int(round(cfg["hold_s"] / (t_p[1] - t_p[0])))
    pv = np.full(n_fr, np.nan)
    score = np.full(n_fr, -np.inf)
    ceil_hz = np.zeros(n_fr)
    edges = _bands(df, nb, cfg["fmax_hz"])
    octaves = np.round(np.array([80.0, 160.0, 320.0, 640.0, 1280.0]) / df).astype(int)
    lo_c, hi_c = pitch_cfg["fmin_hz"] * 0.9, 2 * pitch_cfg["fmax_hz"]
    for j in np.nonzero(alive)[0]:
        i = ip[j]
        base = [f0[i]] if vf[i] else [f0[p] for p in (prev[i], nxt[i]) if 0 <= p < len(f0) and abs(p - i) <= hold]
        if not base:
            continue
        P = 10 ** (L[j].astype(np.float64) / 10)
        cands = [(c, bonus) for c0 in base for c, bonus in ((c0, 1.0), (c0 / 2, 0.0), (c0 * 2, 0.0))
                 if lo_c <= c <= hi_c]  # pyin's own reading wins a tie
        if j and np.isfinite(pv[j - 1]):  # the note going on: pyin's long window reads the next one early
            cands.append((pv[j - 1], 1.0))
        if not cands:
            continue
        sc = _score([_hnr(P, c, df, octaves, cfg)[0] for c, _ in cands]) + np.array([b for _, b in cands])
        best = int(np.argmax(sc))
        score[j] = sc[best]
        if sc[best] >= cfg["score_db"]:
            pv[j] = cands[best][0]
            ceil_hz[j] = _ceiling(*_hnr(P, pv[j], df, edges, cfg), edges, df, cfg)
    # isolated checked frames (under 3 in a row) are not a note
    ok = np.isfinite(pv)
    lab, nl = ndimage.label(ok)
    for r, sl in enumerate(ndimage.find_objects(lab)):
        if sl[0].stop - sl[0].start < 3:
            pv[sl] = np.nan
            ok[sl] = False
    # 3. note onsets; the harmonic space never grows again within a note after its attack
    onsets, foreign = _onsets(mono, fs, pv, t_a, cfg)
    att_end = (cfg["attack"]["ms"] + 50.0) / 1000
    seg_start = np.zeros(n_fr, bool)
    seg_start[1:] = ok[1:] & (~ok[:-1] | (np.abs(12 * np.log2(np.where(ok[1:] & ok[:-1], pv[1:] / np.where(ok[:-1], pv[:-1], 1.0), 1.0))) >= 0.6))
    seg_start[0] = ok[0]
    seg_start[np.clip(np.searchsorted(t_a, onsets), 0, n_fr - 1)] |= ok[np.clip(np.searchsorted(t_a, onsets), 0, n_fr - 1)]
    sm = ndimage.median_filter(ceil_hz, size=5, mode="nearest")
    starts = np.nonzero(seg_start)[0]
    for k, j0 in enumerate(starts):
        j1 = starts[k + 1] if k + 1 < len(starts) else n_fr
        run = np.nonzero(~ok[j0:j1])[0]
        j1 = j0 + (run[0] if run.size else j1 - j0)
        ja = j0 + int(att_end * fs / hop)
        if ja < j1:
            ceil_hz[ja:j1] = np.minimum(ceil_hz[ja:j1], np.minimum.accumulate(sm[ja:j1]))
    # 4. the gains: harmonics of every note sounding within union_ms; attacks left alone
    C = np.zeros((n_fr, nb), np.float32)
    f_nb = np.arange(nb) * df
    half = nfft // (2 * hop)  # the frames within a window's reach
    # how far the pitch moves continuously within each window (a slide; a new note's jump is not one)
    # (the range of the pitch curve with the jumps taken out: the tracker's jitter on a held note stays small)
    step = np.diff(np.log(np.where(ok, pv, np.nan)), prepend=np.nan)
    curve = np.cumsum(np.where(np.isfinite(step) & (np.abs(step) < 0.6 * np.log(2) / 12), step, 0.0))
    w = 2 * half + 1
    span = ndimage.maximum_filter1d(curve, size=w) + ndimage.maximum_filter1d(-curve, size=w)
    glide = np.exp(np.minimum(span, np.log(2.0)))
    for j in np.nonzero(ok & (ceil_hz > 0))[0]:
        C[j] = _comb(nb, df, _centres(pv[j], cfg["fmax_hz"], cfg["stretch"]), cfg, pv[j], glide[j]) * (f_nb <= ceil_hz[j])
    u = 2 * int(cfg["union_ms"] / 1000 * fs / hop) + 1
    C = ndimage.maximum_filter1d(C, size=u, axis=0)
    pitched = ndimage.maximum_filter1d(ok.astype(np.uint8), size=u) > 0
    # each note's start left to the attack EQ, below protect_hz (above, the stem's "attack" is only separation
    # noise: let through and cut again it came in bursts on every note), faded in and out over 30 ms
    protect = np.zeros(n_fr)
    fade = 0.03
    for on in onsets:
        a0, a1 = on - 0.02, on + cfg["attack"]["ms"] / 1000 + 0.02
        w = np.clip(np.minimum((t_a - (a0 - fade)) / fade, ((a1 + fade) - t_a) / fade), 0, 1)
        protect = np.maximum(protect, 0.5 * (1 - np.cos(np.pi * w)))
    floor, gap_floor = 10 ** (-cfg["depth_db"] / 20), 10 ** (-cfg["gap_depth_db"] / 20)
    f_all = np.fft.rfftfreq(nfft, 1 / fs)
    low = np.clip((f_all - cfg["fmin_hz"] * 0.8) / (cfg["fmin_hz"] * 0.2), 0, 1)  # 1 = cleaned, 0 = untouched
    gap_keep = np.clip((f_all - cfg["gap_keep_hz"] * 0.8) / (cfg["gap_keep_hz"] * 0.4), 0, 1)
    ka = cfg["keep_above_hz"]  # >0: untouched above (the pass after the chain: the exciter's new harmonics stay)
    keep_hi = np.clip((f_all - ka * 0.8) / (ka * 0.2), 0, 1) if ka > 0 else np.zeros_like(f_all)
    ph = cfg["attack"]["protect_hz"]
    below = 1 - np.clip((f_all - ph) / (ph * 0.5), 0, 1)  # where the attack is protected: 1 up to protect_hz
    out = np.empty_like(x)
    removed = 0.0
    for s, e, a, b in _blocks(n, fs, hop, margin):
        _, _, Z = signal.stft(x[a:b].T.astype(np.float32), fs, nperseg=nfft, noverlap=nfft - hop)
        g = np.clip(a // hop + np.arange(Z.shape[-1]), 0, n_fr - 1)
        Cb = np.zeros((len(f_all), len(g)), np.float32)
        Cb[:nb] = C[g].T
        keep_note = 1 - low[:, None] * (1 - Cb)
        G_note = keep_note + (1 - keep_note) * floor
        G_gap = 1 - gap_keep[:, None] * (1 - gap_floor) * np.ones((1, len(g)))
        G = np.where(pitched[g][None], G_note, G_gap)
        wp = below[:, None] * protect[g][None, :]
        G = G * (1 - wp) + wp
        G = G * (1 - keep_hi[:, None]) + keep_hi[:, None]
        G = signal.convolve2d(G, np.ones((3, 3)) / 9, mode="same", boundary="symm").astype(np.float32)
        yb = signal.istft(Z * G[None], fs, nperseg=nfft, noverlap=nfft - hop)[1].T
        yb = np.pad(yb, ((0, max(0, b - a - len(yb))), (0, 0)))[s - a:e - a]
        out[s:e] = yb
        removed += float(np.sum((x[s:e] - yb) ** 2))
        del Z, G, Cb, G_note, G_gap, keep_note
    stats = {"pitched_pct": round(100 * float(ok.mean()), 0),
             "pyin_confident_pct": round(100 * float(np.isfinite(old_pitch[0]).mean()), 0),
             "median_f0_hz": round(float(np.nanmedian(pv)), 1) if ok.any() else None,
             "median_space_hz": round(float(np.median(ceil_hz[ok])), 0) if ok.any() else None,
             "notes": int(len(onsets)), "foreign_transients": int(foreign),
             "removed_db": round(10 * np.log10(removed / (float(np.sum(x * x)) + EPS) + EPS), 1),
             "depth_db": cfg["depth_db"], "gap_depth_db": cfg["gap_depth_db"]}
    if parts is not None:
        parts["harmonic_space"] = out.copy()
        parts["score"], parts["t"], parts["pitch"], parts["space_hz"], parts["onsets"] = score, t_a, pv, ceil_hz, onsets
    if cfg["attack"]["enabled"] and len(onsets):
        out = _attack_eq(out, fs, cfg, onsets, stats)
    if cfg["fundamental_pitch"] == "checked":
        # the fundamental is raised along this pitch: never an octave under pyin's own reading (nearest one),
        # where a wrong octave would make up a note that was not played
        ref = np.exp(np.interp(t_a, t_p[vf], np.log(f0[vf]))) if vf.any() else np.full(n_fr, np.nan)
        r = 12 * np.log2(pv / ref)
        fp = np.where(np.abs(r + 12) < 0.6, 2 * pv, pv)
        return out, stats, (fp, t_a, onsets)
    return out, stats, (*old_pitch, onsets)
