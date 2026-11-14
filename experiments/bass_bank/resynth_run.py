"""The bass rebuilt from its notes (resynth.py) on one song's separated stem, every layer for listening:

  processed/experiments/bass_bank/<song>/resynth/
    0_input  1_fundamental  2_body  3_top  4_attacks  5_fill  6_resynth (the sum)  7_bank_last_time
  measures.json, spectra.png   (FLAC, the stem first brought to a proper level: bass_level.py; trimmed like the stage files)

  PYTHONUNBUFFERED=1 uv run python experiments/bass_bank/resynth_run.py ["<song>"] > processed/logs/bass_resynth.log 2>&1
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments"))
import bass_level  # noqa: E402
import resynth  # noqa: E402
from studio_profile import REGISTERS, by_register, harmonic_db  # noqa: E402

from mooz import config, note_eq, stages  # noqa: E402

song = sys.argv[1] if len(sys.argv) > 1 else "260924_230957_TrLR Stanza OK"
out = ROOT / "processed/experiments/bass_bank" / song / "resynth"
rep = json.loads((ROOT / "processed/reports" / f"{song}.json").read_text())
fs = rep["sample_rate"]
a, b = (int(round(rep["trim"][k] * fs)) for k in ("start_s", "end_s"))
x = sf.read(ROOT / "processed/stages" / song / "1_raw/bass.wav", dtype="float64", always_2d=True, start=a, stop=b)[0]
x = x * bass_level.input_gain(x.mean(axis=1), fs)  # first: the stem at a proper level (bass_level.py)
t0 = time.time()
cfg = config.load(None)["tracks"]["bass"]
_, st, (f0, t_f, onsets) = note_eq.note_eq(x, fs, cfg["note_eq"] | {"enabled": True, "fundamental_pitch": "checked"},
                                           cfg["bass_fundamental"])
print(f"[{time.time() - t0:4.0f}s] pitch: {st['pitched_pct']}%, {len(onsets)} notes", flush=True)
parts = {}
_, stats = resynth.resynth(x, fs, f0, t_f, onsets, None, None, parts)
print(f"[{time.time() - t0:4.0f}s] {stats}", flush=True)
names = {"input": "0_input", "fundamental": "1_fundamental", "body": "2_body", "top": "3_top", "attacks": "4_attacks",
         "fill": "5_fill", "resynth": "6_resynth", "bank_rebalanced": "7_bank_last_time"}
bass_level.write_folder(out, {v: parts[k] for k, v in names.items()}, fs)

studio = json.loads(resynth.STUDIO.read_text())
cents = [40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 630, 1000, 1600, 2500, 4000]
rt = np.interp(np.log(cents), np.log(studio["grid_hz"]), np.array(studio["instruments"]["bass"]["tone"]))
rt -= rt[cents.index(100)]
A_, B_ = 30.0, 250.0


def measure(m):
    s = m[int(A_ * fs):int(B_ * fs)]
    f, P = signal.welch(s, fs, nperseg=1 << 15)
    tone = 10 * np.log10(np.array([P[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum() for c in cents]) + 1e-30)
    tone -= tone[cents.index(100)]
    h = signal.sosfiltfilt(signal.butter(4, 1500, "high", fs=fs, output="sos"), s)
    pk = ndimage.maximum_filter1d(np.abs(h), int(0.002 * fs))
    full = np.sqrt(ndimage.uniform_filter1d(s * s, int(0.03 * fs)) + 1e-20)
    hr = np.sqrt(ndimage.uniform_filter1d(h * h, int(0.03 * fs)) + 1e-20)
    _, nclk = ndimage.label(ndimage.binary_dilation((pk > 6 * hr) & (20 * np.log10(pk / full) > -30), iterations=480))
    db, ff = harmonic_db(np.column_stack([m, m]), fs, f0, t_f)
    return {"tone_re_studio": np.round(tone - rt, 1).tolist(), "clicks_per_min": round(nclk / ((B_ - A_) / 60), 1),
            "harmonics_re_2nd": by_register(db, ff)}


res = {"stats": stats} | {k: measure(parts[k]) for k in ("input", "bank_rebalanced", "resynth")}
(out / "measures.json").write_text(json.dumps(res, indent=1, default=float))
print(f"\n{'':18s}" + " ".join(f"{c:>5d}" for c in cents) + "  clicks/min   (tone: dB re a studio bass, re 100 Hz)")
for k in ("input", "bank_rebalanced", "resynth"):
    print(f"{k:18s}" + " ".join(f"{v:5.1f}" for v in res[k]["tone_re_studio"]) + f"  {res[k]['clicks_per_min']:8.1f}")
prof = json.loads(Path(__file__).with_name("studio_harmonics.json").read_text())["studio_db_re_2nd"]
print("\nharmonics re the 2nd (k = 1..8)")
for i, (lo, hi) in enumerate(zip(REGISTERS[:-1], REGISTERS[1:])):
    print(f"  {lo:3.0f}-{hi:3.0f} Hz  studio  {prof[i][:8] if prof[i] else None}")
    for k in ("input", "resynth"):
        v = res[k]["harmonics_re_2nd"][i]
        print(f"  {'':10s}{k[:7]:8s}{v[:8] if v else None}")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axs = plt.subplots(1, 4, figsize=(28, 6), sharey=True)
for ax, k in zip(axs, ("input", "bank_rebalanced", "resynth", "top")):
    f, t, Z = signal.stft(parts[k][int(60 * fs):int(64 * fs)], fs, nperseg=4096, noverlap=4096 - 256)
    ax.pcolormesh(t + 60, f, 20 * np.log10(np.abs(Z) + 1e-9), vmin=-120, vmax=-25, cmap="magma", shading="auto")
    ax.set_yscale("log")
    ax.set_ylim(30, 8000)
    ax.set_title(k)
plt.tight_layout()
plt.savefig(out / "spectra.png", dpi=55)
print(f"done in {time.time() - t0:.0f}s")
