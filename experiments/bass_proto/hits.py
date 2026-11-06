"""Cymbal / hi-hat full hits through the stages: which stage takes their brightness?"""
import json
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
song = "260928_224856_TrLR Miramare"
st = Path("processed/ensemble/stages") / song
rep = json.load(open(f"processed/ensemble/reports/{song}.json")); t0 = rep["trim"]["start_s"]
fs = 48000
BANDS = [(1000, 2000), (2000, 5000), (5000, 10000), (10000, 16000)]
def rd(f, trimmed, g=0.0):
    off = 0 if trimmed else int(t0 * fs)
    n = sf.info(st / "2_stems" / "cymbals.flac").frames
    return sf.read(f, dtype="float64", always_2d=True, start=off, stop=off + n)[0].mean(1) * 10 ** (g / 20)
hop = fs // 100
def env(x, lo, hi):
    y = signal.sosfilt(signal.butter(4, [lo, hi], "band", fs=fs, output="sos"), x)
    n = len(y) // hop * hop
    return np.sqrt((y[:n] ** 2).reshape(-1, hop).mean(1))
for name in ("cymbals", "hihat"):
    stages = {"1 separated": rd(Path("processed/stems_ensemble") / song / f"{name}.wav", False),
              "3 clean-up": rd(st / "3_drum_cleanup" / f"{name}.flac", True, 12),
              "2 processed": rd(st / "2_stems" / f"{name}.flac", True, 12)}
    e = env(stages["1 separated"], 2000, 16000)
    db = 20 * np.log10(e + 1e-9)
    rise = db - np.concatenate([np.full(3, db[0]), db[:-3]])
    pk, _ = signal.find_peaks(db, distance=25)
    full = [p for p in pk if rise[p] > 6 and db[p] > np.percentile(db, 90)]
    print(f"\n== {name}: {len(full)} full hits (loudest 10 %, rising 6 dB)")
    print(f"   {'stage':12s} " + "  ".join(f"{lo//1000}-{hi//1000}k hit/decay" for lo, hi in BANDS) + "   (dB re the separated stem)")
    ref = None
    for k, x in stages.items():
        row = []
        for lo, hi in BANDS:
            ev = env(x, lo, hi)
            hit = np.mean([ev[p:p + 15].mean() for p in full if p + 80 < len(ev)])
            dec = np.mean([ev[p + 30:p + 80].mean() for p in full if p + 80 < len(ev)])
            row.append((20 * np.log10(hit), 20 * np.log10(dec)))
        if ref is None:
            ref = row
        print(f"   {k:12s} " + "  ".join(f"{h - rh:+5.1f}/{d - rd_:+5.1f}  " for (h, d), (rh, rd_) in zip(row, ref)))
print("\nreport:", {n: {k: rep["tracks"][n].get(k) for k in ("dynamic_eq", "transient", "ducks", "kit_gate")} for n in ("cymbals", "hihat")})
