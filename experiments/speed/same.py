# Run from the project root: uv run python experiments/speed/same.py <one output folder> <another>
# Compare two output folders of the same take: the master, the mix and every stage file, sample by sample.
import sys, numpy as np, soundfile as sf
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(p.relative_to(a) for p in a.rglob("*") if p.suffix in (".wav", ".flac") and "samples" not in p.parts)
worst, same, diff = -999.0, 0, []
for r in files:
    if not (b / r).exists(): diff.append((str(r), "missing")); continue
    x, _ = sf.read(a / r, dtype="float64", always_2d=True); y, _ = sf.read(b / r, dtype="float64", always_2d=True)
    if x.shape != y.shape: diff.append((str(r), f"shape {x.shape} vs {y.shape}")); continue
    if np.array_equal(x, y): same += 1; continue
    d = 10 * np.log10(np.mean((x - y) ** 2) / (np.mean(x ** 2) + 1e-30) + 1e-30); worst = max(worst, d); diff.append((str(r), f"{d:.0f} dB re the file"))
print(f"{len(files)} audio files: {same} identical to the sample, {len(diff)} not" + (f" (largest difference {worst:.0f} dB)" if diff else ""))
for r, w in diff[:40]: print("  ", r, w)
