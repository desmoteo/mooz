"""Original (RoFormer -> DrumSep) kit pieces vs the SAM-drums -> DrumSep ones: alignment,
phase agreement per octave band, and whether combining them helps.

    uv run python experiments/prompt_sep/kit_phase.py [song ...]

Per piece: the lag of the cross-correlation peak (samples) and its sign, broadband
correlation, and per octave band the coherence |sum Sxy|^2 / (sum Sxx sum Syy) and the
mean phase difference of the two (in the frames where the piece plays). Then candidates
for a combined stem: the original, the SAM one, their average, and a band-split average
(averaged only where the two are coherent, the original above):
  leak    energy more than 120 ms after any hit re the energy within 120 ms (kick, snare,
          toms; hits where both versions agree), lower = less bleed and ringing
  buzz    the same for the snare, 150 Hz-6 kHz, hits from the original snare
  mask    phase-safe: the original drums through a soft mask from both estimates' magnitudes
  sum     level of the average re the mean of the two levels: 0 dB = in phase, -3 dB =
          uncorrelated (summing would comb-filter / smear)
Listening files go to processed/experiments/prompt_sep/kit_phase/<song>/.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, read_stem, write  # noqa: E402

PIECES = ["kick", "snare", "toms", "hihat", "cymbals"]
BANDS = [(31, 63), (63, 125), (125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 8000), (8000, 16000), (16000, 22000)]
FS = 48000


def db(x):
    return 10 * np.log10(max(float(x), 1e-20))


def lag(a, b, max_lag=480):
    """Lag of b re a (samples, + = b late) and the sign of the correlation peak, on 30 s."""
    n = min(len(a), 30 * FS)
    s = len(a) // 2 - n // 2
    x, y = a[s:s + n], b[s:s + n]
    c = signal.correlate(y, x, mode="full", method="fft")[n - 1 - max_lag:n + max_lag]
    i = int(np.argmax(np.abs(c)))
    return i - max_lag, float(np.sign(c[i]))


def bands(a, b, span=120 * FS):
    m = max(0, len(a) // 2 - span // 2)  # the middle 2 minutes
    a, b = a[m:m + span].astype(np.float32), b[m:m + span].astype(np.float32)
    f, _, A = signal.stft(a, FS, nperseg=4096, noverlap=3072)
    _, _, B = signal.stft(b, FS, nperseg=4096, noverlap=3072)
    act = (np.abs(A) ** 2).sum(0) > np.percentile((np.abs(A) ** 2).sum(0), 50)  # where it plays
    A, B = A[:, act], B[:, act]
    out = []
    for lo, hi in BANDS:
        k = (f >= lo) & (f < hi)
        sxy = (A[k] * np.conj(B[k])).sum()
        sxx, syy = (np.abs(A[k]) ** 2).sum(), (np.abs(B[k]) ** 2).sum()
        out.append((abs(sxy) ** 2 / (sxx * syy + 1e-30), np.degrees(np.angle(sxy)), db(syy) - db(sxx)))
    return out


def envelope(x, lo=None, hi=None):
    if lo:
        x = signal.sosfilt(signal.butter(4, [lo, hi], "band", fs=FS, output="sos"), x)
    hop = FS // 100
    return np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same")[::hop])


def leak(y, hits_env, lo=None, hi=None):
    pk, _ = signal.find_peaks(hits_env, height=np.percentile(hits_env, 99) * 0.25, distance=8)
    near = np.zeros(len(hits_env), bool)
    for p in pk:
        near[p:p + 12] = True
    e = envelope(y, lo, hi)[:len(near)] ** 2
    return db(e[~near].mean() / e[near].mean())


def band_split(r, s, coh):
    """Average below the first octave band where coherence drops under 0.6, original above."""
    fc = next((lo for (lo, hi), (c, _, _) in zip(BANDS, coh) if lo >= 125 and c < 0.6), None)
    if fc is None:
        return (r + s) / 2, None
    lp = signal.butter(4, fc, "low", fs=FS, output="sos")
    low = lambda x: signal.sosfiltfilt(lp, x, axis=0)  # zero phase: the bands sum back flat
    return low((r + s) / 2) + (r - low(r)), fc


def mask_ensemble(orig: dict, sam: dict, nfft=2048, block=30 * FS, margin=FS) -> dict:
    """Phase-safe combination: each piece = the original drums (sum of the original pieces,
    so the recording's own phase) times a soft mask from the geometric mean of the two
    estimates' magnitudes. The pieces still sum exactly to the original drums."""
    names = list(orig)
    x = sum(orig.values())
    n, out = len(x), {k: np.zeros(x.shape, np.float32) for k in names}
    for a in range(0, n, block):
        lo, hi = max(0, a - margin), min(n, a + block + margin)
        for ch in range(x.shape[1]):
            st = lambda y: signal.stft(y[lo:hi, ch], FS, nperseg=nfft, noverlap=nfft * 3 // 4)[2].astype(np.complex64)
            X = st(x)
            g = {k: np.sqrt(np.abs(st(orig[k])) * np.abs(st(sam[k]))) for k in names}
            tot = sum(g.values()) + 1e-9
            for k in names:
                y = signal.istft(X * (g[k] / tot), FS, nperseg=nfft, noverlap=nfft * 3 // 4)[1][:hi - lo]
                out[k][a:min(n, a + block), ch] = y[a - lo:a - lo + min(block, n - a)]
    return out


def main() -> None:
    songs = sys.argv[1:] or sorted(p.name for p in (OUT / "sam-large").iterdir() if "@" not in p.name
                                   and (p / "2_drums_drumsep" / "_rest.flac").exists())
    for song in songs:
        print(f"\n{song}")
        sam_dir = OUT / "sam-large" / song / "2_drums_drumsep"
        listen = OUT / "kit_phase" / song
        print(f"  {'piece':<8}{'lag':>5}{'sign':>5}{'corr':>6}  coherence / phase deg / SAM-orig dB per octave "
              f"({', '.join(str(lo) for lo, _ in BANDS)} Hz)")
        rows, R, S = [], {}, {}
        for piece in PIECES:
            R[piece] = sf.read(ROOT / "processed" / "stems" / song / f"{piece}.wav", dtype="float32")[0]
            S[piece] = read_stem(sam_dir / f"{piece}.flac")[:len(R[piece])]
        M = mask_ensemble(R, S)
        for piece in PIECES:
            r, s, mk = (R.pop(piece).astype(np.float64), S.pop(piece).astype(np.float64),
                        M.pop(piece).astype(np.float64))
            rm, sm = r.mean(1), s.mean(1)
            lg, sign = lag(rm, sm)
            corr = float(np.sum(rm * sm) / np.sqrt(np.sum(rm ** 2) * np.sum(sm ** 2)))
            coh = bands(rm, sm)
            print(f"  {piece:<8}{lg:>5}{sign:>5.0f}{corr:>6.2f}  "
                  + " ".join(f"{c:.2f}/{p:+.0f}/{d:+.1f}" for c, p, d in coh))
            avg = (r + s) / 2
            split, fc = band_split(r, s, coh)
            sum_db = db(np.mean(avg ** 2)) - (db(np.mean(r ** 2)) + db(np.mean(s ** 2))) / 2
            row = {"piece": piece, "sum": sum_db, "fc": fc,
                   "mask_db": db(np.mean(mk ** 2)) - db(np.mean(r ** 2)),
                   "mask_corr": float(np.sum(mk * r) / np.sqrt(np.sum(mk ** 2) * np.sum(r ** 2)))}
            if piece in ("kick", "snare", "toms"):
                hits = np.sqrt(envelope(rm) * envelope(sm))  # where both versions hit
                for k, y in (("orig", r), ("sam", s), ("avg", avg), ("split", split), ("mask", mk)):
                    row[f"leak_{k}"] = leak(y.mean(1), hits)
                if piece == "snare":
                    he = envelope(rm, 150, 6000)
                    for k, y in (("orig", r), ("sam", s), ("avg", avg), ("split", split), ("mask", mk)):
                        row[f"buzz_{k}"] = leak(y.mean(1), he, 150, 6000)
            rows.append(row)
            write(listen / f"{piece}_a_orig.flac", r, FS)
            write(listen / f"{piece}_b_sam.flac", s, FS)
            write(listen / f"{piece}_c_avg.flac", avg, FS)
            write(listen / f"{piece}_d_split.flac", split, FS)
            write(listen / f"{piece}_e_diff.flac", r - s, FS)  # what they disagree on
            write(listen / f"{piece}_f_mask.flac", mk, FS)
        C = ("orig", "sam", "avg", "split", "mask")
        print(f"  {'piece':<8}{'sum dB':>8}{'split at':>9}{'mask dB':>8}{'corr':>6}   leak {'/'.join(C)} dB        buzz {'/'.join(C)} dB")
        for w in rows:
            lk = "/".join(f"{w['leak_' + k]:.1f}" for k in C) if "leak_orig" in w else ""
            bz = "/".join(f"{w['buzz_' + k]:.1f}" for k in C) if "buzz_orig" in w else ""
            print(f"  {w['piece']:<8}{w['sum']:>8.2f}{str(w['fc'] or '-'):>9}{w['mask_db']:>8.2f}{w['mask_corr']:>6.2f}   {lk:<34}{bz}")


if __name__ == "__main__":
    main()
