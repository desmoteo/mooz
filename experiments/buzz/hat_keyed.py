# The snare buzz left in the hi-hat stem, taken down where the hat's own top band says it is not the hat.
# The hat is bright: its hits are flat from 3 to 12.5 kHz. The buzz is dark: it peaks near 3 kHz and is
# about 15 dB down at 12.5 kHz. So, per STFT bin in band_hz, the stem may be at most as loud as the hat's
# own spectrum (learnt from its hits, attack and tail) predicts from the 9-16 kHz level of that moment,
# plus margin_db; above that it is turned down, at most floor_db. Gains are smoothed over 1/3 octave, rise
# at once and fall with release_ms.
# A second way, hat_share: each moment as hat + buzz with fixed shapes, the stem keeps the hat's share.
# Run: uv run python experiments/buzz/hat_keyed.py <stem dir with hihat.wav> <out dir> [start_s end_s]
import sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, ".")
from rehearsal_master import augment

FS = 48000
NPER, HOP = 1024, 256
TOP = (9000.0, 16000.0)


def hat_keyed(x, fs, band_hz=(1200.0, 8000.0), margin_db=6.0, floor_db=-15.0, release_ms=40.0):
    mono = x.mean(axis=1)
    f, _, Z = signal.stft(x.T, fs, nperseg=NPER, noverlap=NPER - HOP)       # (ch, freq, time)
    P = (np.abs(Z) ** 2).mean(axis=0)
    top = P[(f >= TOP[0]) & (f < TOP[1])].sum(axis=0) + 1e-20
    starts, _ = augment.drum_hits(x, fs, 6000.0, 14000.0)
    fr = lambda ms: np.clip((starts + int(ms / 1000 * fs)) // HOP, 0, P.shape[1] - 1)  # noqa: E731
    att = np.median(P[:, np.concatenate([fr(0), fr(10), fr(20)])] / top[np.concatenate([fr(0), fr(10), fr(20)])], axis=1)
    tail = np.median(P[:, np.concatenate([fr(60), fr(90), fr(120)])] / top[np.concatenate([fr(60), fr(90), fr(120)])], axis=1)
    allowed = np.maximum(att, tail)[:, None] * top[None, :] * 10 ** (margin_db / 10)
    g = np.clip(0.5 * np.log10(allowed / (P + 1e-20)) * 20, floor_db, 0.0)   # dB
    inb = (f >= band_hz[0]) & (f < band_hz[1])
    g[~inb] = 0.0
    # smooth over 1/3 octave
    gs = np.empty_like(g)
    for i, fi in enumerate(f):
        k = (f >= fi * 2 ** (-1 / 6)) & (f <= fi * 2 ** (1 / 6)) if fi > 0 else (f == 0)
        gs[i] = g[k].mean(axis=0)
    gs[~inb] = 0.0
    a = np.exp(-HOP / (release_ms / 1000 * fs))
    out = gs.copy()
    for j in range(1, out.shape[1]):
        out[:, j] = np.maximum(gs[:, j], a * out[:, j - 1] + (1 - a) * gs[:, j])  # rise at once, fall slowly
    _, y = signal.istft(Z * 10 ** (out[None] / 20), fs, nperseg=NPER, noverlap=NPER - HOP)
    return y.T[: len(x)], out, f, starts, {"att_minus_top": att, "tail": tail}


def hat_share(x, fs, band_hz=(1200.0, 8000.0), floor_db=-15.0, release_ms=40.0, buzz_from_s=None, bias_db=0.0):
    """Each moment as hat + buzz, with fixed spectral shapes (third-octave bands 1.2-16 kHz): the hat's
    from its hits (attack and tail), the buzz's from the frames with no hat, snare or cymbal hit near
    (from buzz_from_s on). Non-negative least squares per frame; in band_hz the stem keeps the hat's
    share of each bin (Wiener), at most floor_db down; bias_db > 0 leans towards the hat."""
    from scipy.optimize import nnls
    f, _, Z = signal.stft(x.T, fs, nperseg=NPER, noverlap=NPER - HOP)
    P = (np.abs(Z) ** 2).mean(axis=0)
    nfr = P.shape[1]
    centres = 1200.0 * 2 ** (np.arange(13) / 3)
    centres = centres[centres < 16000]
    Bm = np.array([((f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))) for c in centres]).astype(float)
    Pb = Bm @ P                                               # (bands, frames)
    starts, _ = augment.drum_hits(x, fs, 6000.0, 14000.0)
    fr = lambda s, ms: np.clip((s + int(ms / 1000 * fs)) // HOP, 0, nfr - 1)  # noqa: E731
    norm = lambda M: M / (M.sum(axis=0, keepdims=True) + 1e-30)  # noqa: E731
    att = np.median(norm(P[:, np.concatenate([fr(starts, 0), fr(starts, 10)])]), axis=1)
    tail = np.median(norm(P[:, np.concatenate([fr(starts, 90), fr(starts, 130)])]), axis=1)
    busy = np.zeros(nfr, bool)
    for s in starts // HOP:
        busy[max(0, s - 4) : s + 60] = True                    # 300 ms after a hat hit
    if buzz_from_s is not None:
        busy[: int(buzz_from_s * fs) // HOP] = True
    loud = Pb.sum(axis=0) > np.percentile(Pb.sum(axis=0), 30)
    buzz = np.median(norm(P[:, (~busy) & loud]), axis=1)
    W = np.column_stack([Bm @ att, Bm @ tail, Bm @ buzz])
    Wn = W / W.sum(axis=0)
    H = np.zeros((3, nfr))
    sc = np.sqrt(Pb.sum(axis=0)) + 1e-20
    A = np.sqrt(Wn)                                           # fit in amplitude: less dominated by the loudest band
    for j in range(nfr):
        H[:, j] = nnls(A, np.sqrt(Pb[:, j]) / sc[j])[0] ** 2 * sc[j] ** 2
    hat = np.outer(att / W[:, 0].sum(), H[0]) + np.outer(tail / W[:, 1].sum(), H[1])
    bz = np.outer(buzz / W[:, 2].sum(), H[2])
    g = 10 * np.log10((hat * 10 ** (bias_db / 10) + 1e-30) / (hat * 10 ** (bias_db / 10) + bz + 1e-30))
    g = np.clip(g, floor_db, 0.0)
    inb = (f >= band_hz[0]) & (f < band_hz[1])
    g[~inb] = 0.0
    gs = np.empty_like(g)
    for i, fi in enumerate(f):
        k = (f >= fi * 2 ** (-1 / 6)) & (f <= fi * 2 ** (1 / 6)) if fi > 0 else (f == 0)
        gs[i] = g[k].mean(axis=0)
    gs[~inb] = 0.0
    a = np.exp(-HOP / (release_ms / 1000 * fs))
    out = gs.copy()
    for j in range(1, nfr):
        out[:, j] = np.maximum(gs[:, j], a * out[:, j - 1] + (1 - a) * gs[:, j])
    _, y = signal.istft(Z * 10 ** (out[None] / 20), fs, nperseg=NPER, noverlap=NPER - HOP)
    return y.T[: len(x)], starts, {"att": Bm @ att, "tail": Bm @ tail, "buzz": Bm @ buzz, "centres": centres}


def measure(x, y, fs, starts, t0=54.0):
    H = int(0.05 * fs)
    def frames(z, lo, hi):
        b = signal.sosfiltfilt(signal.butter(4, [lo, hi], "band", fs=fs, output="sos"), z.mean(axis=1)) ** 2
        n = len(b) // H
        return 10 * np.log10(b[: n * H].reshape(n, H).mean(axis=1) + 1e-20)
    n = len(x) // H
    t = np.arange(n) * 0.05
    busy = np.zeros(n, bool)
    for s in starts // H:
        busy[max(0, s - 1) : s + 5] = True
    hit = np.zeros(n, bool); hit[starts // H] = True
    tl = np.zeros(n, bool); tl[np.clip(starts // H + 2, 0, n - 1)] = True
    late = t >= t0
    rows = {}
    for lo, hi in ((1200, 2000), (2000, 4000), (4000, 8000), (8000, 16000)):
        fx, fy = frames(x, lo, hi), frames(y, lo, hi)
        rows[f"{lo}-{hi}"] = [float(np.median(fy[k & late]) - np.median(fx[k & late])) for k in (~busy, hit, tl)]
    return rows


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (None, None)
    x, fs = sf.read(src / "hihat.wav", dtype="float32", always_2d=True)
    out.mkdir(parents=True, exist_ok=True)
    variants = {f"ratio_margin{int(m)}": (lambda m=m: hat_keyed(x, fs, margin_db=m)[0:4:3]) for m in (3.0,)}
    for bias in (0.0, 6.0):
        variants[f"share_bias{int(bias)}"] = (lambda bias=bias: hat_share(x, fs, buzz_from_s=a, bias_db=bias)[0:2])
    for name, run in variants.items():
        y, starts = run()
        r = measure(x, y, fs, starts)
        print(f"{name}   change in dB:  between hits / at hat hits (0-50 ms) / hat tails (100-150 ms)")
        for k, v in r.items():
            print(f"   {k:>10} Hz  {v[0]:+6.1f}  {v[1]:+6.1f}  {v[2]:+6.1f}")
        if a is not None:
            i, j = int(a * fs), int(b * fs)
            sf.write(out / f"hihat_{name}.wav", y[i:j], fs, subtype="PCM_24")
            sf.write(out / f"hihat_{name}_removed.wav", (x - y)[i:j], fs, subtype="PCM_24")
    if a is not None:
        sf.write(out / "hihat_raw.wav", x[int(a * fs) : int(b * fs)], fs, subtype="PCM_24")


def hat_share2(x, snare, fs, band_hz=(1200.0, 8000.0), floor_db=-15.0, release_ms=40.0, bias_db=6.0, bed_from_s=None,
               protect_ms=0.0):
    """hat_share with three fixes. The hat's shape is learnt from its attacks only (its tails after
    the band comes in are mostly bed, so a tail template let the bed through, in the tails above all).
    The snare's burst in this stem is a third part, keyed on the raw snare stem: its shape and its
    level re the snare stem are learnt at snare hits with no hat near, and then at every snare hit,
    hat or no hat, that much of the snare's own envelope counts as not-hat. The bed's shape comes
    from the frames with no hit near. Per bin the stem keeps hat / (hat + bed + snare burst).
    protect_ms > 0: nothing is taken in the first protect_ms of each hat hit (the hat masks the bed
    there, and its attack is what must not change); the gain then falls back at release_ms."""
    from scipy.optimize import nnls
    f, _, Z = signal.stft(x.T, fs, nperseg=NPER, noverlap=NPER - HOP)
    P = (np.abs(Z) ** 2).mean(axis=0)
    _, _, Zs = signal.stft(snare.T, fs, nperseg=NPER, noverlap=NPER - HOP)
    Ps = (np.abs(Zs) ** 2).mean(axis=0)
    del Zs
    nfr = P.shape[1]
    centres = 1200.0 * 2 ** (np.arange(13) / 3)
    centres = centres[centres < 16000]
    Bm = np.array([((f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))) for c in centres]).astype(float)
    inb_bands = (centres >= band_hz[0]) & (centres < band_hz[1])
    Pb, Psb = Bm @ P, Bm @ Ps
    hats, _ = augment.drum_hits(x, fs, 6000.0, 14000.0)
    snares, _ = augment.drum_hits(snare, fs, 150.0, 400.0)
    fr = lambda s, ms: np.clip((s + int(ms / 1000 * fs)) // HOP, 0, nfr - 1)  # noqa: E731
    norm = lambda M: M / (M.sum(axis=0, keepdims=True) + 1e-30)  # noqa: E731
    hat_t = np.median(norm(P[:, np.concatenate([fr(hats, 0), fr(hats, 5), fr(hats, 10)])]), axis=1)
    busy = np.zeros(nfr, bool)
    for s in np.concatenate([hats, snares]) // HOP:
        busy[max(0, s - 10) : s + 60] = True
    if bed_from_s is not None:
        busy[: int(bed_from_s * fs) // HOP] = True
    tot = Pb[inb_bands].sum(axis=0)
    free = (~busy) & (tot > np.percentile(tot[~busy], 30))
    bed_t = np.median(norm(P[:, free]), axis=1)
    alone = np.array([s for s in snares if np.min(np.abs(hats - s)) > 0.15 * fs])
    burst_frames = np.concatenate([fr(alone, ms) for ms in (0, 10, 20, 30)])
    burst_t = np.median(norm(P[:, burst_frames]), axis=1)
    ratio = np.median(Pb[inb_bands][:, burst_frames].sum(axis=0) / (Psb[inb_bands][:, burst_frames].sum(axis=0) + 1e-30))
    W = np.column_stack([Bm @ hat_t, Bm @ bed_t])
    Wn = W / W.sum(axis=0)
    A = np.sqrt(Wn)
    burst_b = Bm @ burst_t
    burst_b = burst_b / burst_b.sum()
    leak = ratio * Psb[inb_bands].sum(axis=0)                 # the snare's burst in this stem, per frame
    Hh, Hb = np.zeros(nfr), np.zeros(nfr)
    for j in range(nfr):
        r = np.maximum(Pb[:, j] - leak[j] * burst_b, 0.0)
        sc = np.sqrt(r.sum()) + 1e-20
        h = nnls(A, np.sqrt(r) / sc)[0] ** 2 * sc ** 2
        Hh[j], Hb[j] = h
    hat = np.outer(hat_t / W[:, 0].sum(), Hh) * 10 ** (bias_db / 10)
    rest = np.outer(bed_t / W[:, 1].sum(), Hb) + np.outer(burst_t / (Bm @ burst_t).sum(), leak)
    g = np.clip(10 * np.log10((hat + 1e-30) / (hat + rest + 1e-30)), floor_db, 0.0)
    inb = (f >= band_hz[0]) & (f < band_hz[1])
    g[~inb] = 0.0
    gs = np.empty_like(g)
    for i, fi in enumerate(f):
        k = (f >= fi * 2 ** (-1 / 6)) & (f <= fi * 2 ** (1 / 6)) if fi > 0 else (f == 0)
        gs[i] = g[k].mean(axis=0)
    gs[~inb] = 0.0
    if protect_ms > 0:
        for s in hats // HOP:
            gs[:, max(0, s - 1) : s + int(protect_ms / 1000 * fs / HOP) + 1] = 0.0
    a = np.exp(-HOP / (release_ms / 1000 * fs))
    out = gs.copy()
    for j in range(1, nfr):
        out[:, j] = np.maximum(gs[:, j], a * out[:, j - 1] + (1 - a) * gs[:, j])
    _, y = signal.istft(Z * 10 ** (out[None] / 20), fs, nperseg=NPER, noverlap=NPER - HOP)
    info = {"snare_burst_re_snare_db": round(10 * np.log10(ratio), 1), "snare_alone_hits": len(alone),
            "shapes_re_5k_db": {k: [round(float(v), 1) for v in 10 * np.log10(Bm @ t + 1e-30) - 10 * np.log10((Bm @ t)[6] + 1e-30)]
                                for k, t in (("hat", hat_t), ("bed", bed_t), ("burst", burst_t))}, "centres": [int(c) for c in centres]}
    return y.T[: len(x)], hats, snares, info


def measure2(x, y, snare_hits, fs, hats, t0=54.0):
    """Change y - x (dB) in four bands: between hits, at hat hits, in hat tails, at snare hits with no
    hat within 150 ms (first 50 ms, then 50-200 ms) and at snare hits with a hat within 20 ms."""
    H = int(0.05 * fs)
    n = len(x) // H
    t = np.arange(n) * 0.05
    def frames(z, lo, hi):
        b = signal.sosfiltfilt(signal.butter(4, [lo, hi], "band", fs=fs, output="sos"), z.mean(axis=1)) ** 2
        return 10 * np.log10(b[: n * H].reshape(n, H).mean(axis=1) + 1e-20)
    busy = np.zeros(n, bool)
    for s in np.concatenate([hats, snare_hits]) // H:
        busy[max(0, s - 1) : s + 5] = True
    late = t >= t0
    sel = {"between hits": (~busy) & late}
    for lab, st in (("hat hits", hats), ("hat tails", hats)):
        m = np.zeros(n, bool); m[np.clip(st // H + (2 if lab == "hat tails" else 0), 0, n - 1)] = True; sel[lab] = m & late
    alone = np.array([s for s in snare_hits if np.min(np.abs(hats - s)) > 0.15 * fs])
    together = np.array([s for s in snare_hits if np.min(np.abs(hats - s)) < 0.02 * fs])
    for lab, st, off in (("snare alone 0-50", alone, 0), ("snare alone 50-200", alone, 1), ("snare+hat 0-50", together, 0), ("snare+hat 50-200", together, 1)):
        m = np.zeros(n, bool)
        for k in ([0] if off == 0 else [1, 2, 3]):
            m[np.clip(st // H + k, 0, n - 1)] = True
        sel[lab] = m & late
    bands = ((1200, 2000), (2000, 4000), (4000, 8000), (8000, 16000))
    print(f"   {'':20}" + " ".join(f"{lo}-{hi}".rjust(11) for lo, hi in bands))
    for lab, m in sel.items():
        row = []
        for lo, hi in bands:
            fx, fy = frames(x, lo, hi), frames(y, lo, hi)
            row.append(np.median(fy[m]) - np.median(fx[m]))
        print(f"   {lab:20}" + " ".join(f"{v:+11.1f}" for v in row) + f"   (n={m.sum()})")
