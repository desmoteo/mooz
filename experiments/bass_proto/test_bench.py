"""Benchmark the prototypes: pitch agreement (separated vs dry studio bass), fundamental vs 2nd
harmonic (studio vs the band), and the clean-up's effect on the benchmark's Max Spec bass."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sep_bench"))
import proto
import evaluate as ev
from bass_tricks import max_spec
FS = 48000
hp = ev.config.load(ev.ROOT / "rehearsal.toml")["highpass"]
sos = signal.butter(hp["order"], hp["freq_hz"], "high", fs=FS, output="sos")

def h12(x, f0, tf):
    a, m = proto.harmonic_amps(x, FS, f0, tf)
    use = m & (a[2] > np.percentile(a[2][m], 30)) if m.any() else m
    return float(np.median(20 * np.log10((a[1][use] + 1e-9) / (a[2][use] + 1e-9)))) if use.sum() > FS else np.nan

agree, agree0, rows, h_studio = [], [], {}, []
for d in sorted(p for p in ev.BENCH.iterdir() if (p / "meta.json").exists()):
    info = json.loads((d / "_tracks/tracks.json").read_text())
    dry = sf.read(d / "_tracks/bass_0.flac", dtype="float64", always_2d=True)[0][info["start"]:]
    ref = signal.sosfilt(sos, ev.read(d / "bass.flac", "float64"), axis=0); n = min(len(ref), len(dry)); ref, dry = ref[:n], dry[:n]
    sw = ev.read(ev.OUT / "stems/superset" / d.name / "models/1_mix/BS-Roformer-SW/bass.flac", "float64")[:n] * 10 ** 0.6
    xl = ev.load_run("bass_models", d.name)[0]["bass~xlance"].astype(np.float64)[:n]
    cas = ev.read(ev.OUT / "stems/cascade" / d.name / "bass.wav", "float64")[:n]
    est = max_spec([xl, sw, cas]).astype(np.float64)
    fd, tf = proto.pitch(dry, FS)
    fe, _ = proto.pitch(est, FS)
    fe0, _ = proto.pitch(est, FS, octave_db=99.0)  # no octave correction
    both = np.isfinite(fd) & np.isfinite(fe)
    if both.sum() > 100:
        c = lambda a: np.abs(1200 * np.log2(a[both] / fd[both]))
        agree.append(np.mean(c(fe) < 60)); agree0.append(np.mean(c(fe0) < 60) if np.isfinite(fe0[both]).all() else np.nan)
    h_studio.append(h12(dry, fd, tf))
    res = {"none": est}
    for name, kw in {"clean 12 dB": {}, "clean 12 dB, tol 60c": {"tol_cents": 60}, "clean 12 dB, no stretch": {"stretch": 0.0},
                     "clean 6 dB": {"depth_db": 6.0}}.items():
        res[name] = proto.clean(est, FS, fe, tf, **kw)
    for name, y in res.items():
        rem = est - y
        bass_share = float((rem.ravel() @ ref.ravel()) ** 2 / ((ref.ravel() @ ref.ravel()) * (rem.ravel() @ rem.ravel()) + 1e-20)) if name != "none" else np.nan
        rows.setdefault(name, []).append([ev.sdr(ref, y), *ev.bleed_full(ref, y), 10 * np.log10((rem ** 2).sum() / (est ** 2).sum() + 1e-20), bass_share])
    print(d.name[:26], f"pitch agreement {agree[-1] if agree else 0:.2f} (uncorrected {agree0[-1] if agree0 else 0:.2f}), studio h1/h2 {h_studio[-1]:+.1f} dB", flush=True)
print(f"\npitch agreement with the dry track (within 60 cents): corrected {np.nanmean(agree):.2f}, uncorrected {np.nanmean(agree0):.2f}")
print(f"studio bass: fundamental re 2nd harmonic, median over songs {np.nanmedian(h_studio):+.1f} dB (per song: {', '.join(f'{v:+.0f}' for v in h_studio)})")
print(f"\n{'clean-up':26s} {'SDR':>6s} {'bleedl':>7s} {'fullns':>7s} {'removed':>8s}  share of the removed that is bass")
for k, v in rows.items():
    m = np.nanmean(v, axis=0); print(f"{k:26s} {m[0]:6.2f} {m[1]:7.2f} {m[2]:7.2f} {m[3]:7.1f} dB  {m[4]:.2f}")
