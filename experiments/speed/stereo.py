# How far from mono each stem is: side against mid (dB) and the correlation of the two channels, for every take
# with merged stems. Run from the project root: uv run python experiments/speed/stereo.py <recordings folder>/processed
import sys, numpy as np, soundfile as sf
from pathlib import Path
merged = next((Path(sys.argv[1]) / "stems_ensemble_merged").iterdir())
names = ["kick", "snare", "toms", "hihat", "cymbals", "bass", "guitar", "vocals", "other"]
print(f"{'take':30}" + "".join(f"{n:>13}" for n in names))
for d in sorted(p for p in merged.iterdir() if p.is_dir() and not p.name.startswith("_")):
    row = []
    for n in names:
        f = d / f"{n}.wav"
        if not f.exists(): row.append("      -      "); continue
        x, fs = sf.read(f, dtype="float32", always_2d=True)
        if x.shape[1] == 1 or not x.any(): row.append("   mono/empty"); continue
        m, s = 0.5 * (x[:, 0] + x[:, 1]), 0.5 * (x[:, 0] - x[:, 1])
        row.append(f"{10 * np.log10((np.mean(s * s) + 1e-30) / (np.mean(m * m) + 1e-30)):6.1f} {np.corrcoef(x[:, 0], x[:, 1])[0, 1]:5.2f} ")
    print(f"{d.name:30}" + "".join(row))
