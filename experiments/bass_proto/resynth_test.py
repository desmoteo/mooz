"""Bass resynthesis prototype: the harmonic part rebuilt from the pitch track, each harmonic's
amplitude and phase taken along the pitch with a slow filter (the partials' fast flutter, the
"fuzz" of separation masks and distortion sidebands, is left out), against the stem as it is.
Measures the flutter (level modulation 20-200 Hz on each band's envelope re its slow level)."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mooz import augment, config, dynamics
song = "260928_224856_TrLR Miramare"; fs = 48000
OUT = Path("processed/experiments/bass_proto") / song
t0 = json.load(open(f"processed/reports/{song}.json"))["trim"]["start_s"]
x = sf.read(f"processed/stems_ensemble/{song}/bass.wav", dtype="float64", always_2d=True, start=int((t0 + 60) * fs), stop=int((t0 + 120) * fs))[0]
cfg = config.load(None)["tracks"]["bass"]
f0, tf = augment.bass_pitch(x, fs, cfg["bass_fundamental"])
ok = np.isfinite(f0); mono = x.mean(1); t = np.arange(len(mono)) / fs
fi = np.exp(np.interp(t, tf[ok], np.log(f0[ok]))); phi = 2 * np.pi * np.cumsum(fi) / fs
mask = np.interp(t, tf, ok.astype(float))
mask = dynamics.attack_smooth(dynamics.attack_smooth(mask, 20.0, fs)[::-1], 20.0, fs)[::-1]
def resynth(bw=15.0, fmax=2500.0):
    y = np.zeros_like(mono)
    for k in range(1, 80):
        if np.median(k * fi) > fmax: break
        lp = signal.butter(2, max(bw, 0.01 * k * np.median(fi)), "low", fs=fs, output="sos")
        c = 2 * (signal.sosfiltfilt(lp, mono * np.cos(k * phi)) - 1j * signal.sosfiltfilt(lp, mono * np.sin(k * phi)))
        y += np.real(c * np.exp(1j * k * phi)) * (k * fi < fmax)
    return y * mask
r = resynth()
hp = signal.butter(4, 2500, "high", fs=fs, output="sos")
V = {"stem as it is": mono, "resynthesised harmonics (up to 2.5 kHz)": r,
     "resynth + the stem above 2.5 kHz + unpitched moments": r + signal.sosfiltfilt(hp, mono) * mask + mono * (1 - mask)}
def flutter(y, lo, hi):
    b = signal.sosfiltfilt(signal.butter(4, [lo, hi], "band", fs=fs, output="sos"), y)
    env = np.abs(signal.hilbert(b))
    slow = signal.sosfiltfilt(signal.butter(2, 15, "low", fs=fs, output="sos"), env)
    fast = signal.sosfiltfilt(signal.butter(2, [20, 200], "band", fs=fs, output="sos"), env)
    on = slow > np.percentile(slow, 40)
    return 10 * np.log10(np.mean(fast[on] ** 2) / np.mean(slow[on] ** 2))
print("partials' flutter (20-200 Hz level modulation re the slow level, dB; lower = cleaner)")
for k, y in V.items():
    print(f"  {k:48s} " + "  ".join(f"{lo}-{hi} Hz {flutter(y, lo, hi):+6.1f}" for lo, hi in ((60, 250), (250, 800), (800, 2500))) +
          f"   level re stem {10*np.log10(np.mean(y**2)/np.mean(mono**2)):+5.1f} dB")
for k, y, name in ((1, r, "resynth_harmonics_only"), (2, V["resynth + the stem above 2.5 kHz + unpitched moments"], "resynth_blend"),
                   (3, 0.5 * mono + 0.5 * V["resynth + the stem above 2.5 kHz + unpitched moments"], "resynth_half")):
    sf.write(OUT / f"{name}.flac", (np.column_stack([y, y]) * 10 ** (-12 / 20)).astype(np.float32), fs, subtype="PCM_24")
sf.write(OUT / "stem_as_is_60s.flac", (x * 10 ** (-12 / 20)).astype(np.float32), fs, subtype="PCM_24")
print("written", OUT)
