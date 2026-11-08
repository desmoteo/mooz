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
