import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sep_bench"))
import proto
fs = 48000
VAR = {"4096, 40c/8Hz": {}, "8192, 25c/5Hz": {"nfft": 8192, "hop": 1024, "tol_cents": 25.0, "tol_hz": 5.0}}
song = "260928_224856_TrLR Miramare"
d = Path("processed/experiments/bass_proto") / song
x = sf.read(d / "1_raw.flac", dtype="float64", always_2d=True)[0] * 10 ** (12 / 20)
tf, f0 = np.load(d / "pitch.npy")
for k, kw in VAR.items():
    cl = proto.clean(x, fs, f0, tf, **kw)
    print(f"Miramare {k}: " + "  ".join(
        f"{lo}-{hi} Hz {10*np.log10((signal.sosfilt(signal.butter(4,[lo,hi],'band',fs=fs,output='sos'),cl.mean(1))**2).sum()/(signal.sosfilt(signal.butter(4,[lo,hi],'band',fs=fs,output='sos'),x.mean(1))**2).sum()):+.1f} dB"
        for lo, hi in ((150, 600), (600, 2000), (2000, 5000))), flush=True)
    if "8192" in k:
        sf.write(d / "2_cleaned_fine.flac", (cl * 10 ** (-12 / 20)).astype(np.float32), fs, subtype="PCM_24")
        sf.write(d / "2_removed_fine.flac", ((x - cl) * 10 ** (-12 / 20)).astype(np.float32), fs, subtype="PCM_24")
# benchmark safety: SDR and the share of removed energy that is bass
import evaluate as ev
from bass_tricks import max_spec
hp = ev.config.load(ev.ROOT / "rehearsal.toml")["highpass"]; sos = signal.butter(hp["order"], hp["freq_hz"], "high", fs=fs, output="sos")
rows = {k: [] for k in ["none", *VAR]}
for dd in sorted(p for p in ev.BENCH.iterdir() if (p / "meta.json").exists()):
    ref = signal.sosfilt(sos, ev.read(dd / "bass.flac", "float64"), axis=0); n = len(ref)
    sw = ev.read(ev.OUT / "stems/superset" / dd.name / "models/1_mix/BS-Roformer-SW/bass.flac", "float64")[:n] * 10 ** 0.6
    xl = ev.load_run("bass_models", dd.name)[0]["bass~xlance"].astype(np.float64)[:n]
    cas = ev.read(ev.OUT / "stems/cascade" / dd.name / "bass.wav", "float64")[:n]
    est = max_spec([xl, sw, cas]).astype(np.float64)
    fe, t = proto.pitch(est, fs, octave_db=99.0)
    rows["none"].append([ev.sdr(ref, est), np.nan])
    for k, kw in VAR.items():
        y = proto.clean(est, fs, fe, t, **kw); rem = est - y
        rows[k].append([ev.sdr(ref, y), float((rem.ravel() @ ref.ravel()) ** 2 / ((ref.ravel() @ ref.ravel()) * (rem.ravel() @ rem.ravel()) + 1e-20))])
for k, v in rows.items():
    m = np.nanmean(v, axis=0); print(f"benchmark {k:16s} SDR {m[0]:.2f}  bass share of the removed {m[1]:.2f}")
