# How two outputs of the same take differ: per audio file the difference over the whole file, its typical 0.4 s
# window and its worst one (a flipped decision shows as a window far above the typical), and every field of the
# reports that is not the same. Run from the project root: uv run python experiments/speed/delta.py <a> <b>
import sys, json, numpy as np, soundfile as sf
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(p.relative_to(a) for p in a.rglob("*") if p.suffix in (".wav", ".flac") and "samples" not in p.parts)
print(f"{'file':44} {'whole':>7} {'typical':>8} {'worst':>7}  at      (dB re the file; windows of 0.4 s, re the file's level there)")
for r in files:
    x, fs = sf.read(a / r, dtype="float64", always_2d=True); y, _ = sf.read(b / r, dtype="float64", always_2d=True)
    if x.shape != y.shape: print(f"{str(r)[-44:]:44} shape {x.shape} vs {y.shape}"); continue
    if np.array_equal(x, y): print(f"{str(r)[-44:]:44}    same"); continue
    d = x - y; h = int(0.4 * fs); n = len(d) // h
    pd = (d[: n * h] ** 2).reshape(n, h, -1).mean(axis=(1, 2)); px = (x[: n * h] ** 2).reshape(n, h, -1).mean(axis=(1, 2))
    live = px > px.max() * 1e-6
    rel = 10 * np.log10((pd[live] + 1e-30) / px[live]); i = int(np.flatnonzero(live)[np.argmax(rel)])
    print(f"{str(r)[-44:]:44} {10*np.log10(d.var()/x.var()+1e-30):7.0f} {np.median(rel):8.0f} {rel.max():7.0f}  {i*0.4:6.1f} s")
ra, rb = (json.loads(next((p / "reports").glob("*.json")).read_text()) for p in (a, b))
out = []
def walk(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in ("outputs", "made_with", "file"): walk(x.get(k), y.get(k), f"{path}/{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{path}[{i}]")
    elif x != y: out.append((path, x, y))
walk(ra, rb)
print(f"\nreport fields that differ: {len(out)}")
for p, x, y in out[:60]: print(f"   {p[:78]:78} {str(x)[:22]:>22} -> {str(y)[:22]}")
