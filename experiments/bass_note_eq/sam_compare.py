"""The SAM Audio basses (experiments/prompt_sep) and the dockerized audio-separator ones
(prove/run_bass.sh) against the RoFormer ensemble's, on one song: each
source's tone against raw studio bass tracks, how far its notes' harmonics stand over what lies
between them (clarity), its L/R coherence; and each SAM bass through the stage-2 chain with the
fixes (fixes.py's 7_resonance_1s), for listening next to the other chain/ variants:

  sam/<source>.flac            the source as it is, trimmed like the stages, -12 dB
  chain/8_sam_<source>.flac    through the chain

  PYTHONUNBUFFERED=1 uv run python experiments/bass_note_eq/sam_compare.py > processed/logs/bass_sam_compare.log 2>&1
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mooz import config, effects, eq, note_eq, profile, stages  # noqa: E402

song = sys.argv[1] if len(sys.argv) > 1 else "260924_230957_TrLR Stanza OK"
out = ROOT / "processed/experiments/bass_note_eq" / song
w = stages.StageWriter(out, "flac")
rep = json.loads((ROOT / "processed/reports" / f"{song}.json").read_text())
fs = rep["sample_rate"]
start, end = (int(round(rep["trim"][k] * fs)) for k in ("start_s", "end_s"))
P = ROOT / "processed/experiments/prompt_sep"
SOURCES = {  # name -> (file, gain to the real level)
    "roformer_ensemble": (ROOT / "processed/stages" / song / "1_raw/bass.wav", 0.0),
    "xlance_alone": (ROOT / "processed/experiments/bass_single_model" / song / "bass.wav", 0.0),
    "sam_bass_guitar": (P / "sam-large" / song / "1_stems/bass.flac", 12.0),
    "sam_refine_stem": (P / "sam-large-bass" / song / "refine_stem.flac", 12.0),
    "sam_electric_bass": (P / "sam-large-bass" / song / "mix_electric_bass.flac", 12.0),
}
# audio-separator scales its input to a 0.9 peak (the recording peaks over full scale): undone here
_peak = float(np.abs(sf.read(ROOT / f"{song}.WAV", dtype="float32")[0]).max())
DOCKER_GAIN = 20 * np.log10(_peak / 0.9)
for model in ("BS-Roformer-SW", "htdemucs_ft", "hdemucs_mmi", "kuielab_a_bass"):  # prove/run_bass.sh (docker)
    found = sorted((ROOT / "prove" / "bass").glob(f"*([Bb]ass)*{model}*.flac"))
    SOURCES[f"docker_{model}"] = (found[0] if found else ROOT / "prove" / "bass" / f"missing {model}", DOCKER_GAIN)
SOURCES["mega_bass_alone"] = (ROOT / "processed/experiments/bass_single_model" / song / "bass_bs_mega_53stem_bass_mvsep.wav", 0.0)
# through the chain: the SAM ones, or those named on the command line (sam_compare.py "<song>" docker_htdemucs_ft ...)
SAM = sys.argv[2:] or [k for k in SOURCES if k.startswith("sam")]

tcfg = config.load(ROOT / "experiments/bass_note_eq/fixes.toml")["tracks"]["bass"]
tcfg["resonance"] = tcfg["resonance"] | {"average_ms": 1000.0}  # = fixes.py's 7_resonance_1s
stems_dir = (ROOT / "processed/stages" / song / "1_raw").resolve()
p_bass = json.loads((stems_dir / "_profile.json").read_text())["stems"]["bass"]
target = profile.targets(stems_dir.parent, fs).get("bass")
ref = profile.reference_tone("bass", eq.log_grid(fs))
if ref is not None:
    target = (target or {}) | {"reference": ref}

A, B = 30.0, 150.0
cents = [40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 630, 1000, 1600, 2500, 4000]
studio = json.loads((ROOT / "mooz/references/raw_multitracks.json").read_text())
rt = np.interp(np.log(cents), np.log(studio["grid_hz"]), np.array(studio["instruments"]["bass"]["tone"]))
rt -= rt[cents.index(100)]
ana = np.load(out / "raw" / "analysis.npz")
pv = ana["pitch"]
ne_cfg = config.load(None)["tracks"]["bass"]["note_eq"]
nfft, hop = 8192, 512
df = fs / nfft
nb = int(8000 / df) + 1
edges = np.round(np.array([90, 180, 250, 500, 1000, 2000]) / df).astype(int)


def measure(x):
    x2 = x[int(A * fs):int(B * fs)]
    m = x2.mean(axis=1)
    f, Pw = signal.welch(m, fs, nperseg=1 << 15)
    b = 10 * np.log10(np.array([Pw[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum() for c in cents]) + 1e-30)
    b -= b[cents.index(100)]
    _, _, Z = signal.stft(m.astype(np.float32), fs, nperseg=nfft, noverlap=nfft - hop)
    Q = (np.abs(Z[:nb]) ** 2).T.astype(float)
    j0 = int(round(A * fs / hop))
    H = [note_eq._hnr(Q[j], pv[j0 + j], df, edges, ne_cfg)[0] for j in range(0, Q.shape[0], 4)
         if j0 + j < len(pv) and np.isfinite(pv[j0 + j])]
    return {"tone": np.round(b - rt, 1).tolist(), "clarity": np.round(np.nanmedian(H, axis=0), 1).tolist(),
            "lr_corr": round(float(np.corrcoef(x2[:, 0], x2[:, 1])[0, 1]), 2),
            "level_db": round(float(10 * np.log10(np.mean(m ** 2))), 1)}


res, ens = {}, None
for name, (path, g) in SOURCES.items():
    if not path.exists():
        print(f"{name}: missing {path}", flush=True)
        continue
    x = sf.read(path, dtype="float64", always_2d=True, start=start, stop=end)[0] * 10 ** (g / 20)
    if name == "roformer_ensemble":
        ens = x
    res[name] = measure(x)
    if ens is not None and name != "roformer_ensemble":
        n = min(len(ens), len(x))
        res[name]["sdr_vs_ensemble_db"] = round(float(10 * np.log10(np.sum(ens[:n] ** 2) / np.sum((ens[:n] - x[:n]) ** 2))), 1)
    if name in SAM:
        w.write(f"{'docker' if name.startswith('docker') else 'sam'}/{name}", x, fs)
        st: dict = {}
        z = effects.cleanup_track(x, fs, tcfg, None, st, None, p_bass)
        z = effects.studio_track(z, fs, tcfg, st, None, p_bass, target, None)
        w.write(f"chain/8_{name}", z, fs)
        res[name]["chain"] = measure(z)
    print(f"{name}: {res[name]}", flush=True)
old = json.loads((out / "sam_compare.json").read_text()) if (out / "sam_compare.json").exists() else {}
for k, v in old.items():  # keep what an earlier run rendered through the chain
    if k in res and "chain" in v and "chain" not in res[k] and k not in SAM:
        res[k]["chain"] = v["chain"]
(out / "sam_compare.json").write_text(json.dumps(res, indent=1))
print(f"\n{'':28s}" + " ".join(f"{c:>5d}" for c in cents) + "  | clarity 90-180 -250 -500 -1k -2k | L/R  vs ens")
for k, v in res.items():
    for tag, m in (("", v), (" +chain", v.get("chain"))):
        if m:
            print(f"{k + tag:28s}"[:28] + " ".join(f"{d:5.1f}" for d in m["tone"]) + "  |  "
                  + " ".join(f"{c:5.1f}" for c in m["clarity"]) + f" | {m['lr_corr']:4.2f}  {v.get('sdr_vs_ensemble_db', '')}")
