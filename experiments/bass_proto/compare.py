"""Before/after the bass clean-up and body changes (and the hi-hat/cymbal tone), on a song's
processed stems: spectra against raw studio tracks, and harmonic vs inharmonic energy."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proto
song = "260928_224856_TrLR Miramare"
st = Path("processed/ensemble/stages") / song / "2_stems"; before = Path("processed/ensemble/_before_bass_fix")
fs = 48000; A, B = 60, 120
rd = lambda f: sf.read(f, dtype="float64", always_2d=True, start=A * fs, stop=B * fs)[0] * 10 ** (12 / 20)
r = json.load(open(f"processed/ensemble/reports/{song}.json"))["tracks"]["bass"]
print("bass report:", {k: r[k] for k in ("harmonic_clean", "bass_fundamental") if k in r})
ref = json.load(open("mooz/references/raw_multitracks.json")); grid = np.array(ref["grid_hz"])
def bands(x, cents):
    f, P = signal.welch(x.mean(1), fs, nperseg=1 << 15)
    return 10 * np.log10(np.array([P[(f >= c / 2 ** (1/6)) & (f < c * 2 ** (1/6))].sum() for c in cents]) + 1e-30)
for name, cents, norm in (("bass", [50, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000], 2),
                          ("hihat", [500, 1000, 2000, 3150, 5000, 8000, 12500, 16000], 4),
                          ("cymbals", [500, 1000, 2000, 3150, 5000, 8000, 12500, 16000], 4)):
    rt = np.interp(np.log(cents), np.log(grid), np.array(ref["instruments"][name]["tone"]))
    print(f"\n{name}: 1/3-octave dB re the {cents[norm]} Hz band   " + " ".join(f"{c:>6d}" for c in cents))
    print(f"  {'studio raw tracks':18s}" + " ".join(f"{v - rt[norm]:6.1f}" for v in rt))
    for tag, f in (("before", before / f"{name}_2_stems.flac"), ("after", st / f"{name}.flac")):
        b = bands(rd(f), cents); print(f"  {tag:18s}" + " ".join(f"{v - b[norm]:6.1f}" for v in b))
# bass harmonicity (pitch from the after-stem)
xa, xb = rd(st / "bass.flac"), rd(before / "bass_2_stems.flac")
f0, tf = proto.pitch(xa, fs, octave_db=99.0)
n = 8192; hop = 480
for tag, x in (("before", xb), ("after", xa)):
    f, t, Z = signal.stft(x.mean(1), fs, nperseg=n, noverlap=n - hop); P = np.abs(Z) ** 2
    f0t = np.interp(t, tf, np.where(np.isfinite(f0), f0, np.nan))
    out = []
    for lo, hi in ((150, 600), (600, 2000), (2000, 5000)):
        band = (f >= lo) & (f < hi); h = i = 0.0
        for j in range(P.shape[1]):
            if not np.isfinite(f0t[j]): continue
            k = f[band] / f0t[j]; near = np.abs(k - np.round(k)) * f0t[j] < np.maximum(4.0, 0.012 * f[band])
            h += P[band, j][near].sum(); i += P[band, j][~near].sum()
        out.append(10 * np.log10(h / (i + 1e-30)))
    print(f"bass harmonic/inharmonic {tag:7s} 150-600 Hz {out[0]:+5.1f} dB  600-2000 Hz {out[1]:+5.1f} dB  2-5 kHz {out[2]:+5.1f} dB")
a, m = proto.harmonic_amps(xa, fs, f0, tf)
lf = np.exp(np.interp(np.arange(len(xa)) / fs, tf[np.isfinite(f0)], np.log(f0[np.isfinite(f0)])))
use = m & (lf < 70) & (a[2] > np.percentile(a[2][m], 30))
print(f"after: low notes' fundamental re 2nd harmonic {np.median(20*np.log10((a[1][use]+1e-9)/(a[2][use]+1e-9))):+.1f} dB (before the change: -22.3 dB on the raw stem)")
