"""Does resynthesising the bass's 0.6-2.5 kHz harmonics bring the separated bass closer to the
true one? Benchmark (Max Spec bass estimates vs the room bass references): SDR, bleedless /
fullness, and the harmonic share 0.6-2.5 kHz, against the stem as it is and the harmonic clean-up."""
import sys
from pathlib import Path
import numpy as np
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sep_bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import evaluate as ev
from bass_tricks import max_spec
from rehearsal_master import augment, config, dynamics
FS = 48000
cfg = config.load(None)["tracks"]["bass"]
hp = ev.config.load(ev.ROOT / "rehearsal.toml")["highpass"]; sos = signal.butter(hp["order"], hp["freq_hz"], "high", fs=FS, output="sos")
def resynth_band(x, f0, tf, lo=600.0, hi=2500.0, bw=15.0):
    mono = x.mean(1); ok = np.isfinite(f0); t = np.arange(len(mono)) / FS
    fi = np.exp(np.interp(t, tf[ok], np.log(f0[ok]))); phi = 2 * np.pi * np.cumsum(fi) / FS
    mask = np.interp(t, tf, ok.astype(float)); mask = dynamics.attack_smooth(dynamics.attack_smooth(mask, 20.0, FS)[::-1], 20.0, FS)[::-1]
    y = np.zeros_like(mono); med = np.median(fi)
    for k in range(1, 120):
        if k * med > hi * 1.3: break
        if k * med < lo * 0.7: continue
        lp = signal.butter(2, max(bw, 0.01 * k * med), "low", fs=FS, output="sos")
        c = 2 * (signal.sosfiltfilt(lp, mono * np.cos(k * phi)) - 1j * signal.sosfiltfilt(lp, mono * np.sin(k * phi)))
        y += np.real(c * np.exp(1j * k * phi))
    bp = signal.butter(4, [lo, hi], "band", fs=FS, output="sos")
    yb = signal.sosfiltfilt(bp, y) * mask
    xb = signal.sosfiltfilt(bp, x, axis=0)
    return x - xb * mask[:, None] + yb[:, None]  # the band replaced where pitched
rows = {}
for d in sorted(p for p in ev.BENCH.iterdir() if (p / "meta.json").exists()):
    ref = signal.sosfilt(sos, ev.read(d / "bass.flac", "float64"), axis=0); n = len(ref)
    sw = ev.read(ev.OUT / "stems/superset" / d.name / "models/1_mix/BS-Roformer-SW/bass.flac", "float64")[:n] * 10 ** 0.6
    xl = ev.load_run("bass_models", d.name)[0]["bass~xlance"].astype(np.float64)[:n]
    cas = ev.read(ev.OUT / "stems/cascade" / d.name / "bass.wav", "float64")[:n]
    est = max_spec([xl, sw, cas]).astype(np.float64)
    pitch = augment.bass_pitch(est, FS, cfg["bass_fundamental"])
    V = {"stem as it is": est, "harmonic clean (now)": augment.harmonic_clean(est, FS, cfg["harmonic_clean"], *pitch)[0],
         "0.6-2.5 kHz resynthesised": resynth_band(est, *pitch)}
    bp = signal.butter(4, [600, 2500], "band", fs=FS, output="sos")
    rb = signal.sosfiltfilt(bp, ref, axis=0)
    for k, y in V.items():
        yb = signal.sosfiltfilt(bp, y, axis=0)
        rows.setdefault(k, []).append([ev.sdr(ref, y), ev.sdr(rb, yb), *ev.bleed_full(ref, y)])
    print(d.name[:28], "  ".join(f"{k.split()[0]} {v[-1][1]:+.2f}" for k, v in rows.items()), flush=True)
print(f"\n{'variant':28s} {'SDR':>6s} {'SDR 0.6-2.5k':>13s} {'bleedless':>9s} {'fullness':>9s}")
for k, v in rows.items():
    m = np.mean(v, axis=0); print(f"{k:28s} {m[0]:6.2f} {m[1]:13.2f} {m[2]:9.2f} {m[3]:9.2f}")
