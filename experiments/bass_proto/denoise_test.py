"""Bass stem: noisereduce (spectral gating, stationary / non-stationary) against the
pipeline's harmonic clean-up, on 60 s of a song's raw bass: how much of the bass's own
harmonics each keeps, and how much of what lies between them it takes, per band."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mooz import augment, config
song = "260928_224856_TrLR Miramare"; fs = 48000
t0 = json.load(open(f"processed/reports/{song}.json"))["trim"]["start_s"]
x = sf.read(f"processed/stems_ensemble/{song}/bass.wav", dtype="float64", always_2d=True, start=int((t0 + 60) * fs), stop=int((t0 + 120) * fs))[0]
cfg = config.load(None)["tracks"]["bass"]
pitch = augment.bass_pitch(x, fs, cfg["bass_fundamental"])
f0, tf = pitch
import noisereduce as nr
V = {"none": x,
     "noisereduce, stationary": nr.reduce_noise(y=x.T, sr=fs, stationary=True, prop_decrease=0.9).T,
     "noisereduce, non-stationary": nr.reduce_noise(y=x.T, sr=fs, stationary=False, prop_decrease=0.9).T,
     "harmonic clean (now)": augment.harmonic_clean(x, fs, cfg["harmonic_clean"], *pitch)[0],
     "harmonic clean 60 Hz-4 kHz, 18 dB": augment.harmonic_clean(x, fs, cfg["harmonic_clean"] | {"fmin_hz": 60.0, "fmax_hz": 4000.0, "depth_db": 18.0}, *pitch)[0]}
n = 16384; hop = 2048
f, t, Zx = signal.stft(x.mean(1), fs, nperseg=n, noverlap=n - hop)
f0t = np.interp(t, tf, np.where(np.isfinite(f0), f0, np.nan))
bands = [(40, 150), (150, 600), (600, 2000), (2000, 5000)]
def split(Z):
    P = np.abs(Z) ** 2; out = []
    for lo, hi in bands:
        b = (f >= lo) & (f < hi); h = i = 0.0
        for j in range(P.shape[1]):
            if not np.isfinite(f0t[j]): continue
            k = f[b] / f0t[j]; near = np.abs(k - np.round(k)) * f0t[j] < np.maximum(3.0, 0.01 * f[b])
            h += P[b, j][near].sum(); i += P[b, j][~near].sum()
        out.append((h, i))
    return out
base = split(Zx)
print(f"{'variant':36s} " + "  ".join(f"{lo}-{hi} Hz: harmonics kept / between taken" for lo, hi in bands[:1]) + " ... (dB)")
for k, y in V.items():
    _, _, Z = signal.stft(y.mean(1), fs, nperseg=n, noverlap=n - hop)
    r = split(Z)
    print(f"{k:36s} " + "  ".join(f"{lo}-{hi}: {10*np.log10(h/bh):+5.1f}/{10*np.log10(i/bi):+6.1f}" for (lo, hi), (h, i), (bh, bi) in zip(bands, r, base)))
    if k != "none":
        sf.write(f"processed/experiments/bass_proto/{song}/denoise_{k.split(',')[0].replace(' ', '_')}{'_' + k.split(', ')[1].replace(' ', '_') if ', ' in k else ''}.flac",
                 (y * 10 ** (-12 / 20)).astype(np.float32), fs, subtype="PCM_24")
