"""Clean-up around the stems: mains hum removal (on the recording, before
separation), bleed reduction between stems (optional), and level balancing of
stem groups so that e.g. the vocals sit at a similar level in every song."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from .loudness import EPS, filtfilt

# --- mains hum ---------------------------------------------------------------------------------


def _quiet_part(x: np.ndarray, fs: float, below_db: float = 30.0, min_s: float = 1.5) -> np.ndarray | None:
    """100 ms frames at least below_db under the song's loud level (silence, gaps),
    concatenated; None if there are fewer than min_s of them."""
    w = int(0.1 * fs)
    n = len(x) // w
    e = 10 * np.log10(np.mean(x[: n * w].reshape(n, w, -1) ** 2, axis=(1, 2)) + EPS)
    idx = np.flatnonzero(e < np.percentile(e, 90) - below_db)
    if len(idx) * 0.1 < min_s:
        return None
    return np.concatenate([x[i * w : (i + 1) * w] for i in idx])


def detect_hum(x: np.ndarray, fs: float, cfg: dict, lead_in: tuple[int, int] | None = None) -> dict:
    """Mains hum: the harmonics of exactly 50 (or 60) Hz that stand at least
    min_db above their spectral neighbourhood where the band is silent: in the
    recording's lead-in before the music if there is one (the best reference),
    else in moments at least 30 dB below the music. Music is never at exact
    mains multiples for long, so notes are not mistaken for hum; with no silent
    reference at all, nothing is removed."""
    ref, where = None, None
    if lead_in and (lead_in[1] - lead_in[0]) >= 1.5 * fs:
        ref, where = x[lead_in[0] : lead_in[1]], "lead-in"
    if ref is None:
        ref, where = _quiet_part(x, fs), "quiet moments"
    if ref is None:
        return {"base_hz": None, "harmonics": [], "reference": "none (no silent part)"}
    q = ref.mean(axis=1) * np.hanning(len(ref))
    f, p = signal.periodogram(q, fs, nfft=1 << int(np.ceil(np.log2(len(q) * 4))))
    best = (None, [])
    for base in cfg["base_hz"]:
        found = []
        for k in range(1, cfg["max_harmonic"] + 1):
            h = base * k
            if h > 0.45 * fs:
                break
            tol = 0.3 + 0.05 * k  # mains is stable; a few tenths of a Hz of measurement slack
            win = np.abs(f - h) <= tol
            i = np.flatnonzero(win)[np.argmax(p[win])]
            neigh = (np.abs(f - f[i]) > 4) & (np.abs(f - f[i]) < 15)
            prom = 10 * np.log10(p[i] / (np.median(p[neigh]) + EPS) + EPS)
            if prom >= cfg["min_db"]:
                found.append({"k": k, "hz": round(float(f[i]), 2), "prominence_db": round(float(prom), 1)})
        if sum(h["prominence_db"] for h in found) > sum(h["prominence_db"] for h in best[1]):
            best = (base, found)
    base, found = best
    if len(found) < 2:  # a single line could be anything; hum always comes with harmonics
        found = []
    return {"base_hz": base, "harmonics": found, "reference": where}


def remove_hum(x: np.ndarray, fs: float, hum: dict, cfg: dict) -> np.ndarray:
    """Very narrow zero-phase notches at the detected harmonics only."""
    for h in hum["harmonics"]:
        bw = cfg["bandwidth_hz"] * (1.0 if h["k"] == 1 else 1.5)
        b, a = signal.iirnotch(h["hz"], h["hz"] / bw, fs=fs)
        x = filtfilt(signal.tf2sos(b, a), x)
    return x


# --- bleed reduction ---------------------------------------------------------------------------


def refine_stems(mix: np.ndarray, stems: dict[str, np.ndarray], fs: float, power: float, floor_db: float
                 ) -> dict[str, np.ndarray]:
    """Re-estimate every stem as a soft mask on the mix: each time-frequency point
    goes to the stems in proportion to their power raised to `power` (>1 sharpens:
    less bleed, but more 'watery' artifacts if pushed). The stems then sum to the
    mix exactly. floor_db keeps a little of every stem everywhere."""
    names = list(stems)
    nfft, hop = 2048, 512
    out = {n: np.empty_like(stems[n]) for n in names}
    floor = 10 ** (floor_db / 10)
    chunk, margin = 1 << 18, 4 * nfft
    for s in range(0, len(mix), chunk):
        e = min(len(mix), s + chunk)
        a, b = max(0, s - margin), min(len(mix), e + margin)
        _, _, zm = signal.stft(mix[a:b].T, fs, nperseg=nfft, noverlap=nfft - hop)
        pw = []
        for n in names:
            _, _, z = signal.stft(stems[n][a:b].T, fs, nperseg=nfft, noverlap=nfft - hop)
            pw.append(np.mean(np.abs(z) ** 2, axis=0) ** power)
        pw = np.array(pw)
        pw = np.maximum(pw, floor * pw.max(axis=0, keepdims=True))
        pw[:, pw.sum(axis=0) <= 0] = 1.0  # digital silence: split evenly
        masks = pw / (pw.sum(axis=0, keepdims=True) + 1e-300)  # sums to 1: the stems add up to the mix
        for n, m in zip(names, masks):
            _, y = signal.istft(zm * m[None], fs, nperseg=nfft, noverlap=nfft - hop)
            y = y.T[: b - a]
            out[n][s:e] = np.pad(y, ((0, max(0, b - a - len(y))), (0, 0)))[s - a : e - a]
    return out


# --- level balance between songs ---------------------------------------------------------------


def balance_gains(stem_lufs: dict[str, float], total_lufs: float, cfg: dict) -> tuple[dict[str, float], dict]:
    """Gain (dB) per stem so that the groups sit at their target levels relative
    to each other, as they do in the reference songs. Each group's level is
    measured against the (unbalanced) remix; its gain is how far it misses its
    target, minus the typical (median) miss of all groups, so the overall level
    stays where it was (the loudness stage sets it later anyway). Loudness is not
    additive, so the targets are matched as a pattern, not as shares of the total.
    Returns (gains by stem, report by group)."""
    groups, report = {}, {}
    w, priors = cfg.get("prior_weight", 0.0), cfg.get("priors", {})
    for spec in cfg["groups"]:
        members = [m for m in spec["stems"] if stem_lufs.get(m) is not None]
        if members:
            p = sum(10 ** (stem_lufs[m] / 10) for m in members)
            target = spec["lu"]
            if spec["name"] in priors:  # blended with what mix engineers do (published level priors)
                target = (1 - w) * target + w * priors[spec["name"]]
            groups[spec["name"]] = (members, 10 * math.log10(p) - total_lufs,
                                    round(target + cfg.get("offsets", {}).get(spec["name"], 0.0), 2))
    active = {}
    for g, (members, level, target) in groups.items():
        if level < target - cfg["ignore_below_db"]:  # far too quiet: not this instrument, just bleed
            report[g] = {"level_lu": round(level, 1), "target_lu": target, "gain_db": 0.0, "note": "too quiet, left alone"}
        else:
            active[g] = target - level
    offset = float(np.median(list(active.values()))) if active else 0.0
    gains_db = {g: 0.0 for g in groups}
    for g, miss in active.items():
        gains_db[g] = float(np.clip(cfg["strength"] * (miss - offset), -cfg["max_change_db"], cfg["max_change_db"]))
        level, target = groups[g][1], groups[g][2]
        report[g] = {"level_lu": round(level, 1), "target_lu": target, "gain_db": round(gains_db[g], 1),
                     "after_lu": round(level + gains_db[g], 1)}
    gains = {m: gains_db[g] for g, (members, _, _) in groups.items() for m in members}
    for name, l in stem_lufs.items():  # leftover stems with (almost) nothing in them are dropped
        if name not in gains and l is not None and l - total_lufs < cfg["drop_below_lu"]:
            gains[name] = None
    return gains, report


# --- sympathetic snare buzz --------------------------------------------------------------------
#
# The snare's wires and head resonate with the bass, the guitars and the singer, and the drum
# splitter spreads that buzz over the snare, hi-hat, cymbal and tom stems. Two steps:
#   1. learn its spectral fingerprint where it is alone: frames of the hi-hat and cymbal stems
#      while the kick and toms are silent but those stems are not. Every song adds its frames to
#      a small library next to the stem cache, so songs without such moments benefit too.
#   2. remove it: each affected stem's spectrogram is explained by the fixed buzz fingerprint plus
#      free "music" components (non-negative matrix factorisation); the part the fingerprint
#      explains is masked out (at most floor_db), never on the stem's own hits.

_NFFT, _HOP = 2048, 512


def _band_bins(fs: float, band: tuple[float, float]) -> np.ndarray:
    f = np.fft.rfftfreq(_NFFT, 1 / fs)
    return (f >= band[0]) & (f <= band[1])


def _mono_band_mag(x: np.ndarray, fs: float, bins: np.ndarray) -> np.ndarray:
    _, _, z = signal.stft(x.mean(axis=1), fs, nperseg=_NFFT, noverlap=_NFFT - _HOP)
    return np.abs(z[bins]).astype(np.float32)


def _frames(mask: np.ndarray, n: int) -> np.ndarray:
    m = mask[::_HOP][:n]
    return np.pad(m, (0, n - len(m)), mode="edge")


def _nmf(V: np.ndarray, k: int, W_fixed: np.ndarray | None = None, iters: int = 120, learn_w: bool = True, seed: int = 0,
         exclusive: float = 0.0):
    """KL-divergence NMF; columns of W_fixed stay as they are, k free ones are learnt.
    exclusive (0..1) removes that share of each free column's overlap with the fixed
    ones after every update, so the free columns cannot simply learn the fixed sound too."""
    rng = np.random.default_rng(seed)
    Wf = W_fixed if W_fixed is not None else np.zeros((V.shape[0], 0), np.float32)
    W = np.concatenate([Wf, (rng.random((V.shape[0], k)) + 0.1).astype(np.float32)], axis=1)
    H = (rng.random((W.shape[1], V.shape[1])) + 0.1).astype(np.float32)
    nf = Wf.shape[1]
    for _ in range(iters):
        WH = W @ H + 1e-9
        H *= (W.T @ (V / WH)) / (W.sum(axis=0)[:, None] + 1e-9)
        if learn_w and k:
            WH = W @ H + 1e-9
            Wn = W * ((V / WH) @ H.T) / (H.sum(axis=1)[None, :] + 1e-9)
            Wfree = Wn[:, nf:]
            if exclusive and nf:
                B = Wf / (np.linalg.norm(Wf, axis=0, keepdims=True) + 1e-9)
                Wfree = np.maximum(Wfree - exclusive * B @ (B.T @ Wfree), 1e-9)
            W[:, nf:] = Wfree / (Wfree.sum(axis=0, keepdims=True) + 1e-9)
    return W, H


def buzz_frames(stems: dict[str, np.ndarray], idle: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Step 1 for one song: buzz-only spectral frames: kick and toms idle, the stem not
    silent, and not within 300 ms of the stem's own strokes (so hats and cymbals played on
    their own are not learnt as buzz)."""
    bins = _band_bins(fs, cfg["band_hz"])
    out = []
    for name, x in stems.items():
        A = _mono_band_mag(x, fs, bins)
        e = A.sum(axis=0)
        own = own_hits(x, fs, name, 300.0)
        own = np.pad(own, (0, max(0, A.shape[1] - len(own))), mode="edge")[: A.shape[1]] > 0.5
        sel = (_frames(idle, A.shape[1]) & ~own
               & (20 * np.log10(e + 1e-9) > 20 * np.log10(np.percentile(e, 99) + 1e-9) - 45))
        if sel.any():
            out.append(A[:, sel][:, :: max(1, int(sel.sum() // 3000))])  # at most ~3000 frames per stem
    return np.concatenate(out, axis=1) if out else np.zeros((int(bins.sum()), 0), np.float32)


def buzz_fingerprint(library: Path, cfg: dict) -> np.ndarray | None:
    """Fingerprint (spectral templates) from all songs' buzz-only frames in the library."""
    frames = [np.load(f) for f in sorted(library.glob("*.npy"))]
    frames = [f for f in frames if f.size]
    if not frames or sum(f.shape[1] for f in frames) * _HOP / 48000 < cfg["min_learn_s"]:
        return None
    V = np.concatenate(frames, axis=1)
    W, H = _nmf(V, cfg["components"], iters=200)
    # only templates that carry the buzz: one that explains a mere sliver of its energy (it
    # modelled the faint remainder elsewhere in the band) would claim other sounds there
    energy = np.array([float(np.sum(np.outer(W[:, k], H[k]) ** 2)) for k in range(W.shape[1])])
    keep = energy >= 0.1 * energy.sum()
    W = W[:, keep]
    # keep each template's real shape only: a weak tail (e.g. the noise floor elsewhere in the
    # band) would otherwise let it claim other sounds there, such as the hi-hat
    W = np.where(W >= W.max(axis=0, keepdims=True) * 10 ** (-25 / 20), W, 0.0)
    return W / (W.sum(axis=0, keepdims=True) + 1e-9)


def own_hits(x: np.ndarray, fs: float, name: str, window_ms: float) -> np.ndarray:
    """Per STFT frame, 1 within window_ms after the stem's own hits, else 0: the snare's and
    toms' body (150/70-400 Hz peaks), the hi-hat's and cymbals' attacks (4-16 kHz onsets
    within 30 dB of the loudest)."""
    from . import dynamics
    from .effects import _band, _hold

    if name in ("snare", "toms"):
        lo = 150.0 if name == "snare" else 70.0
        env = dynamics.envelope(np.abs(_band(x, fs, lo, 400.0).mean(axis=1)), fs, 1.0, 50.0)
        pk, _ = signal.find_peaks(env, height=np.percentile(env, 92), distance=int(0.1 * fs))
    else:
        band = np.abs(_band(x, fs, 4000.0, 16000.0).mean(axis=1))
        fast, slow = dynamics.envelope(band, fs, 0.5, 20.0), dynamics.envelope(band, fs, 15.0, 20.0)
        onset = 20 * np.log10((fast + EPS) / (slow + EPS))
        lev = 20 * np.log10(fast + EPS)
        cand = (onset > 6.0) & (lev > np.percentile(lev, 99.5) - 40)  # soft strokes too
        pk = np.flatnonzero(cand & ~np.roll(cand, 1))
    hits = np.zeros(len(x), bool)
    hits[pk] = True
    return _hold(hits, fs, 0.01, window_ms / 1000)[::_HOP]


def remove_buzz(x: np.ndarray, fs: float, W_buzz: np.ndarray, cfg: dict, protect: np.ndarray | None = None,
                near: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Step 2: mask out what the buzz fingerprint explains (chunked, stereo), by at most
    floor_db close to the stem's own hits (near = 1 per frame) and by at most between_floor_db
    between them (near = 0), where the stem holds nothing but the buzz and bleed."""
    bins = _band_bins(fs, cfg["band_hz"])
    # the stem's other sounds, kept from becoming buzz-shaped themselves
    W, _ = _nmf(_mono_band_mag(x, fs, bins)[:, ::4], cfg["music_components"], W_buzz, exclusive=cfg["exclusive"])
    nb = W_buzz.shape[1]
    y = np.empty_like(x)
    chunk, margin = 1 << 18, 4 * _NFFT
    floor = 10 ** (cfg["floor_db"] / 20)
    deep = 10 ** (cfg.get("between_floor_db", cfg["floor_db"]) / 20)
    buzz, total = 0.0, 0.0
    for s in range(0, len(x), chunk):
        e = min(len(x), s + chunk)
        a, b = max(0, s - margin), min(len(x), e + margin)
        _, _, z = signal.stft(x[a:b].T, fs, nperseg=_NFFT, noverlap=_NFFT - _HOP)
        V = np.abs(z[:, bins]).mean(axis=0).astype(np.float32)
        _, H = _nmf(V, 0, W, iters=80, learn_w=False)
        B, M = W[:, :nb] @ H[:nb], W @ H + 1e-9
        fl = floor
        if near is not None:
            nn = near[a // _HOP : a // _HOP + B.shape[1]]
            nn = np.pad(nn, (0, B.shape[1] - len(nn)), constant_values=1.0)
            fl = (deep + (floor - deep) * nn)[None, :]
        G = np.clip(1 - cfg["strength"] * B / M, fl, 1.0)
        if protect is not None:  # the stem's own hits keep everything
            p = protect[a // _HOP : a // _HOP + G.shape[1]]
            G = np.maximum(G, np.pad(p, (0, G.shape[1] - len(p)))[None, :])
        G = signal.convolve2d(G, np.ones((3, 3)) / 9, mode="same", boundary="symm")
        z[:, bins] *= G
        _, yy = signal.istft(z, fs, nperseg=_NFFT, noverlap=_NFFT - _HOP)
        yy = yy.T[: b - a]
        y[s:e] = np.pad(yy, ((0, max(0, b - a - len(yy))), (0, 0)))[s - a : e - a]
        own = slice((s - a) // _HOP, (e - a) // _HOP)
        buzz += float(B[:, own].sum())
        total += float(M[:, own].sum())
    sos = signal.butter(2, list(cfg["band_hz"]), "bandpass", fs=fs, output="sos")
    bx, by = filtfilt(sos, x), filtfilt(sos, y)
    removed = float(10 * np.log10((np.mean(by * by) + EPS) / (np.mean(bx * bx) + EPS)))  # in the buzz band
    return y, {"buzz_share_pct": round(100 * buzz / max(total, 1e-9), 1), "removed_db": round(removed, 2)}


# --- vocals: the de-reverb step's two parts, put back together ---------------------------------

def recombine_room(dry_path: Path, room_path: Path, out_path: Path, cfg: dict) -> dict:
    """The vocal as recorded (dry + room: the de-reverb model's two outputs sum back to
    its input), with the room turned down by at most max_reduction_db. The gains follow
    the model's dry/room split, but smoothed over smooth_ms and smooth_bins: the model's
    own dry output changes from frame to frame and bin to bin, which is what sounds
    glitchy. Chunked; returns how much was taken down."""
    import soundfile as sf

    info = sf.info(dry_path)
    fs, n = info.samplerate, info.frames
    nfft, hop = 2048, 512
    floor = 10 ** (-cfg["max_reduction_db"] / 10)
    k = max(1, int(round(cfg["smooth_ms"] / 1000 * fs / hop)))
    chunk, margin = 1 << 20, 8 * nfft
    e_in = e_out = 0.0
    with sf.SoundFile(out_path, "w", fs, info.channels, subtype="FLOAT") as out:
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            a, b = max(0, s - margin), min(n, e + margin)
            dry = sf.read(dry_path, start=a, stop=b, dtype="float32", always_2d=True)[0]
            room = sf.read(room_path, start=a, stop=b, dtype="float32", always_2d=True)[0]
            full = dry + room
            if cfg["max_reduction_db"] > 0:
                _, _, D = signal.stft(dry.T, fs, nperseg=nfft, noverlap=nfft - hop)
                _, _, R = signal.stft(room.T, fs, nperseg=nfft, noverlap=nfft - hop)
                pd, pr = (np.abs(D) ** 2).mean(axis=0), (np.abs(R) ** 2).mean(axis=0)
                if cfg["smooth_bins"] > 1:
                    pd, pr = (uniform_filter1d(p, cfg["smooth_bins"], axis=0) for p in (pd, pr))
                pd, pr = (uniform_filter1d(p, k, axis=1) for p in (pd, pr))
                g = np.sqrt(np.clip(pd / (pd + pr + 1e-20), floor, 1.0)).astype(np.float32)
                _, y = signal.istft((D + R) * g[None], fs, nperseg=nfft, noverlap=nfft - hop)
                y = y.T[: b - a]
                full = np.pad(y, ((0, max(0, b - a - len(y))), (0, 0)))
            seg = full[s - a : e - a]
            ref = (dry + room)[s - a : e - a]
            e_in += float(np.sum(ref.astype(np.float64) ** 2))
            e_out += float(np.sum(seg.astype(np.float64) ** 2))
            out.write(seg)
    return {"room_reduction_db": round(10 * np.log10((e_out + 1e-20) / (e_in + 1e-20)), 1)}
