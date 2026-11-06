"""The bass bank (bank.py) on one song's separated bass stem, for listening, with every step:

  processed/experiments/bass_bank/<song>/<variant>/
    0_input           the separated stem (mono), as the pipeline's stage 1 has it
    1_bank            through the band-pass bank on the notes
    1_bank_removed    what the bank did not pass
    2_rebalanced      the bank, each note's harmonics rebalanced towards a studio bass's
    3_attacks         the attack track: what the bank did not pass, around each note's start
    4_recombined      2 + 3: the result
  variants: medium (+-25 cents, at least +-4 Hz: rings ~40 ms) and tight (+-20 cents, at least +-1.5 Hz)
  measures.json, spectra.png

The stem first brought to a proper level (bass_level.py), every folder at one common gain; trimmed like the stage files.

  PYTHONUNBUFFERED=1 uv run python experiments/bass_bank/run.py ["<song>"] > processed/logs/bass_bank.log 2>&1
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
import bank  # noqa: E402
from studio_profile import REGISTERS, by_register, harmonic_db  # noqa: E402

from rehearsal_master import config, note_eq, stages  # noqa: E402

song = sys.argv[1] if len(sys.argv) > 1 else "260924_230957_TrLR Stanza OK"
VARIANTS = {"medium": {}, "tight": {"width_cents": 20.0, "min_hz": 1.5}}
out = ROOT / "processed/experiments/bass_bank" / song
rep = json.loads((ROOT / "processed/reports" / f"{song}.json").read_text())
fs = rep["sample_rate"]
a, b = (int(round(rep["trim"][k] * fs)) for k in ("start_s", "end_s"))
x = sf.read(ROOT / "processed/stages" / song / "1_raw/bass.wav", dtype="float64", always_2d=True, start=a, stop=b)[0]
x = x * bass_level.input_gain(x.mean(axis=1), fs)  # first: the stem at a proper level (bass_level.py)
t0 = time.time()
cfg = config.load(None)["tracks"]["bass"]
_, st, (f0, t_f, onsets) = note_eq.note_eq(x, fs, cfg["note_eq"] | {"enabled": True, "fundamental_pitch": "checked"},
                                           cfg["bass_fundamental"])
print(f"[{time.time() - t0:4.0f}s] pitch: {st['pitched_pct']}% of the song, {len(onsets)} notes", flush=True)

studio = json.loads((ROOT / "rehearsal_master/references/raw_multitracks.json").read_text())
cents = [40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 630, 1000, 1600, 2500, 4000]
rt = np.interp(np.log(cents), np.log(studio["grid_hz"]), np.array(studio["instruments"]["bass"]["tone"]))
rt -= rt[cents.index(100)]
prof = json.loads(bank.PROFILE.read_text())
A_, B_ = 30.0, 250.0


def measure(m):
    s = m[int(A_ * fs):int(B_ * fs)]
    f, P = signal.welch(s, fs, nperseg=1 << 15)
    tone = 10 * np.log10(np.array([P[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum() for c in cents]) + 1e-30)
    tone -= tone[cents.index(100)]
    hp = signal.butter(4, 1500, "high", fs=fs, output="sos")
    h = signal.sosfiltfilt(hp, s)
    pk = ndimage.maximum_filter1d(np.abs(h), int(0.002 * fs))
    full = np.sqrt(ndimage.uniform_filter1d(s * s, int(0.03 * fs)) + 1e-20)
    hr = np.sqrt(ndimage.uniform_filter1d(h * h, int(0.03 * fs)) + 1e-20)
    _, nclk = ndimage.label(ndimage.binary_dilation((pk > 6 * hr) & (20 * np.log10(pk / full) > -30), iterations=480))
    db, ff = harmonic_db(np.column_stack([m, m]), fs, f0, t_f)
    return {"tone_re_studio": dict(zip(map(str, cents), np.round(tone - rt, 1).tolist())),
            "clicks_per_min": round(nclk / ((B_ - A_) / 60), 1),
            "harmonics_re_2nd_by_register": dict(zip([f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(REGISTERS[:-1], REGISTERS[1:])],
                                                     by_register(db, ff)))}


res = {"pitch": {"pitched_pct": st["pitched_pct"], "notes": len(onsets)}, "studio_target": prof["studio_db_re_2nd"]}
spec = {}
for name, vc in VARIANTS.items():
    parts = {}
    y, stats = bank.bank(x, fs, f0, t_f, onsets, vc, parts)
    print(f"[{time.time() - t0:4.0f}s] {name}: {stats}", flush=True)
    labels = {"input": "0_input", "bank": "1_bank", "bank_removed": "1_bank_removed", "rebalanced": "2_rebalanced",
              "attacks": "3_attacks", "recombined": "4_recombined"}
    bass_level.write_folder(out / name, {lab: parts[k] for k, lab in labels.items() if k != "input" or name == "medium"}, fs)
    res[name] = {"stats": stats} | {k: measure(parts[k]) for k in ("input", "bank", "rebalanced", "recombined")}
    spec[name] = {k: parts[k][int(60 * fs):int(64 * fs)].copy() for k in ("input", "bank", "rebalanced", "recombined")}
    del parts, y
(out / "measures.json").write_text(json.dumps(res, indent=1))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axs = plt.subplots(2, 4, figsize=(28, 11), sharex=True, sharey=True)
for r, name in enumerate(VARIANTS):
    for c, k in enumerate(("input", "bank", "rebalanced", "recombined")):
        f, t, Z = signal.stft(spec[name][k], fs, nperseg=4096, noverlap=4096 - 256)
        axs[r, c].pcolormesh(t + 60, f, 20 * np.log10(np.abs(Z) + 1e-9), vmin=-120, vmax=-25, cmap="magma", shading="auto")
        axs[r, c].set_yscale("log")
        axs[r, c].set_ylim(30, 6000)
        axs[r, c].set_title(f"{name}: {k}")
plt.tight_layout()
plt.savefig(out / "spectra.png", dpi=55)

print(f"\n{'':24s}" + " ".join(f"{c:>5d}" for c in cents) + "  clicks/min  (tone: dB re a studio bass, re 100 Hz)")
for name in VARIANTS:
    for k in ("input", "bank", "rebalanced", "recombined"):
        if k == "input" and name != "medium":
            continue
        m = res[name][k]
        print(f"{name + ' ' + k:24s}" + " ".join(f"{v:5.1f}" for v in m["tone_re_studio"].values()) + f"  {m['clicks_per_min']:8.1f}")
print("\nharmonics re the 2nd (k = 1..6), by the note's fundamental")
for reg, tgt in zip(res["medium"]["input"]["harmonics_re_2nd_by_register"], prof["studio_db_re_2nd"]):
    print(f"  {reg:8s} studio     {tgt[:6] if tgt else None}")
    for name in VARIANTS:
        for k in ("input", "recombined"):
            if k == "input" and name != "medium":
                continue
            v = res[name][k]["harmonics_re_2nd_by_register"][reg]
            print(f"  {'':8s} {name[:3]} {k[:5]:5s} {v[:6] if v else None}")
print(f"done in {time.time() - t0:.0f}s")
