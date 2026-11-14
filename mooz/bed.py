"""Stage 3: the bed under the hi-hat stem.

While the band plays, the snare's wires rattle along with the PA and the amps, and the drum split
files most of that rattle under the hi-hat: a dark noise (peaking near 3 kHz, 15 dB down by
12 kHz) that sits 15 dB under the hat's hits and fills every gap between them. It is not tied to
any hit, so no gate reaches it, and the hats never stop long enough for the kit gate to close.

The hat is bright, the bed is dark. So each moment of the stem is read as hat + bed (+ the
snare's own burst, which also leaks in), each with a fixed shape in third-octave bands from
1.2 to 16 kHz, learnt from this song: the hat's from the first 10 ms of its hits, the bed's from
the frames with no hit within 50 ms before or after_hit_ms after, the burst's at snare hits with no hat near, where its level against
the raw snare stem is measured too. Per frame the two free parts are fitted (non-negative least
squares in amplitude), the burst is keyed on the snare stem's own envelope, and per bin in
band_hz the stem keeps hat / (hat + bed + burst), at most floor_db down. The first protect_ms of
each hat hit are left alone (the hat masks the bed there), the gain falls back at release_ms,
and above band_hz nothing changes: the hat's top stays as it was.

Nothing is taken when the song gives no hat hits to learn from, no time between hits, or a "bed"
as bright as the hat (an open hat's own decay): then the fit could not tell them apart.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.optimize import nnls

from . import augment

NPER, HOP = 1024, 256
DETECT = {"hihat": (6000.0, 14000.0), "cymbals": (3000.0, 14000.0)}  # where the stem's own hits are found


def _bands(f: np.ndarray, lo: float = 1200.0, hi: float = 16000.0) -> tuple[np.ndarray, np.ndarray]:
    """Third-octave band centres from lo up to hi, and the (bands x bins) summing matrix."""
    centres = lo * 2 ** (np.arange(int(np.log2(hi / lo) * 3) + 1) / 3)
    centres = centres[centres < hi]
    bm = np.array([(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6)) for c in centres]).astype(np.float32)
    return centres, bm


def _spectra(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies and the power spectrogram (bins x frames) of a stereo signal's mono."""
    f, _, z = signal.stft(x.mean(axis=1).astype(np.float32), fs, nperseg=NPER, noverlap=NPER - HOP)
    return f, (np.abs(z) ** 2).astype(np.float32)


def _norm(m: np.ndarray) -> np.ndarray:
    return m / (m.sum(axis=0, keepdims=True) + 1e-30)


def _band_level(x: np.ndarray, fs: float, lo: float, hi: float, frames: np.ndarray, win_s: float = 0.05) -> float:
    """Median level (dB) in lo..hi over the given win_s windows."""
    h = int(win_s * fs)
    b = signal.sosfiltfilt(signal.butter(4, [lo, hi], "band", fs=fs, output="sos"), x.mean(axis=1)) ** 2
    n = len(b) // h
    lev = 10 * np.log10(b[: n * h].reshape(n, h).mean(axis=1) + 1e-20)
    return float(np.median(lev[frames[:n]])) if frames[:n].any() else float("nan")


def remove_bed(x: np.ndarray, fs: float, cfg: dict, key: np.ndarray | None = None, name: str = "hihat"
               ) -> tuple[np.ndarray, dict]:
    """The stem without its bed (see the module doc), and what was done. key: the raw stem whose
    hits leak into this one (the snare), or None for no burst part."""
    lo_b, hi_b = cfg["band_hz"]
    hits, _ = augment.drum_hits(x, fs, *DETECT.get(name, DETECT["hihat"]))
    if len(hits) < cfg["min_hits"]:
        return x, {"note": f"only {len(hits)} hits to learn the hat from: nothing taken"}
    f, p = _spectra(x, fs)
    nfr = p.shape[1]
    centres, bm = _bands(f)
    inb_bands = (centres >= lo_b) & (centres < hi_b)
    pb = bm @ p
    ms = lambda t: int(t / 1000 * fs / HOP)  # noqa: E731  (milliseconds to frames)
    at = lambda s, t: np.clip(s // HOP + ms(t), 0, nfr - 1)  # noqa: E731
    hat_t = np.median(_norm(p[:, np.concatenate([at(hits, 0), at(hits, 5), at(hits, 10)])]), axis=1)
    snares = np.array([], dtype=int)
    if key is not None:
        snares, _ = augment.drum_hits(key, fs, 150.0, 400.0)
    busy = np.zeros(nfr, bool)
    for s in np.concatenate([hits, snares]) // HOP:
        busy[max(0, s - ms(50)) : s + ms(cfg["after_hit_ms"])] = True
    tot = pb[inb_bands].sum(axis=0)
    free = (~busy) & (tot > np.percentile(tot[~busy], 30)) if (~busy).any() else np.zeros(nfr, bool)
    if free.sum() * HOP / fs < cfg["min_between_s"]:
        return x, {"note": f"{free.sum() * HOP / fs:.0f} s between hits: too little to learn the bed from"}
    pick = np.flatnonzero(free)[np.linspace(0, free.sum() - 1, min(4000, free.sum())).astype(int)]
    bed_t = np.median(_norm(p[:, pick]), axis=1)
    bright = lambda t: 10 * np.log10((t[f >= 8000].sum() + 1e-30) / (t[(f >= 2000) & (f < 5000)].sum() + 1e-30))  # noqa: E731
    contrast = bright(hat_t) - bright(bed_t)
    if contrast < cfg["min_contrast_db"]:
        return x, {"note": f"the bed is as bright as the hat ({contrast:.1f} dB): nothing taken", "contrast_db": round(contrast, 1)}
    # the snare's burst in this stem: its shape and its level against the snare stem, at snare hits with no hat near
    leak = np.zeros(nfr, np.float32)
    burst_t, ratio_db, n_alone = bed_t, None, 0
    if key is not None and len(snares):
        alone = np.array([s for s in snares if np.min(np.abs(hits - s)) > 0.15 * fs])
        n_alone = len(alone)
        if n_alone >= 10:
            fk, pk = _spectra(key, fs)
            kb = (bm @ pk)[inb_bands].sum(axis=0)
            del pk
            bf = np.concatenate([at(alone, t) for t in (0, 10, 20, 30)])
            burst_t = np.median(_norm(p[:, bf]), axis=1)
            ratio = float(np.median(tot[bf] / (kb[bf] + 1e-30)))
            ratio_db = round(10 * np.log10(ratio + 1e-30), 1)
            leak = (ratio * kb[: nfr]).astype(np.float32)
            del kb
    w = np.column_stack([bm @ hat_t, bm @ bed_t])
    a = np.sqrt(w / w.sum(axis=0))
    burst_b = bm @ burst_t
    burst_b /= burst_b.sum()
    hh, hb = np.zeros(nfr, np.float32), np.zeros(nfr, np.float32)
    for j in range(nfr):
        r = np.maximum(pb[:, j] - leak[j] * burst_b, 0.0)
        sc = np.sqrt(r.sum()) + 1e-20
        hh[j], hb[j] = nnls(a, np.sqrt(r) / sc)[0] ** 2 * sc ** 2
    del pb
    hat = np.outer(hat_t / w[:, 0].sum(), hh) * 10 ** (cfg["bias_db"] / 10)
    rest = np.outer(bed_t / w[:, 1].sum(), hb) + np.outer(burst_t / (bm @ burst_t).sum(), leak)
    g = np.clip(10 * np.log10((hat + 1e-30) / (hat + rest + 1e-30)), cfg["floor_db"], 0.0).astype(np.float32)
    del hat, rest, p
    inb = (f >= lo_b) & (f < hi_b)
    # smoothed over a third of an octave, so that no single bin is carved out
    sm = np.array([((f >= fi * 2 ** (-1 / 6)) & (f <= fi * 2 ** (1 / 6))) if fi > 0 else (f == 0) for fi in f], dtype=np.float32)
    sm /= sm.sum(axis=1, keepdims=True)
    g = sm @ g
    g[~inb] = 0.0
    if cfg["protect_ms"] > 0:  # the hat's attacks as they are
        for s in hits // HOP:
            g[:, max(0, s - 1) : s + ms(cfg["protect_ms"]) + 1] = 0.0
    rel = np.exp(-HOP / (cfg["release_ms"] / 1000 * fs))
    for j in range(1, nfr):  # rises at once, falls at release_ms
        g[:, j] = np.maximum(g[:, j], rel * g[:, j - 1] + (1 - rel) * g[:, j])
    gain = 10 ** (g / 20)
    del g
    y = np.empty_like(x)
    for ch in range(x.shape[1]):
        _, _, z = signal.stft(x[:, ch].astype(np.float32), fs, nperseg=NPER, noverlap=NPER - HOP)
        _, out = signal.istft(z * gain, fs, nperseg=NPER, noverlap=NPER - HOP)
        y[:, ch] = out[: len(x)]
        del z, out
    n_w = len(x) // int(0.05 * fs)
    hit_w = np.zeros(n_w, bool)
    hit_w[np.clip(hits // int(0.05 * fs), 0, n_w - 1)] = True
    free_w = np.zeros(n_w, bool)
    free_w[np.clip(pick * HOP // int(0.05 * fs), 0, n_w - 1)] = True
    stats = {"hits": int(len(hits)), "between_s": round(float(free.sum() * HOP / fs), 1), "contrast_db": round(float(contrast), 1),
             "snare_alone_hits": int(n_alone), "snare_burst_re_snare_db": ratio_db,
             "between_hits_db": round(_band_level(y, fs, 2000, 8000, free_w) - _band_level(x, fs, 2000, 8000, free_w), 1),
             "hat_hits_db": round(_band_level(y, fs, 2000, 8000, hit_w) - _band_level(x, fs, 2000, 8000, hit_w), 1),
             "removed_db": round(float(10 * np.log10((np.mean((x - y) ** 2) + 1e-20) / (np.mean(x ** 2) + 1e-20))), 1),
             "shapes_re_5k_db": {k: [round(float(v), 1) for v in 10 * np.log10(bm @ t + 1e-30) - 10 * np.log10((bm @ t)[6] + 1e-30)]
                                 for k, t in (("hat", hat_t), ("bed", bed_t), ("burst", burst_t))},
             "band_centres_hz": [int(c) for c in centres]}
    return y, stats
