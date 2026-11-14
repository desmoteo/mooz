# What adding the lined-up close stem does to each band, for several shares and for no line-up.
# Run: uv run python experiments/multitrack/merge_comb.py <root> <take name>
import sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, ".")
from mooz import multitrack, loudness
root, take = Path(sys.argv[1]), sys.argv[2]; FS = 48000
rd, cd = root / "processed/stems_ensemble" / take, root / "processed/stems_close" / f"{take} Tr2"
B = ((50,100),(100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6400))
def bandp(x):
    f, p = signal.welch(x, FS, nperseg=8192)
    return np.array([p[(f>=l)&(f<h)].sum() for l,h in B])
for n, (lo, hi) in {"vocals": (200,5000), "bass": (50,700)}.items():
    r = sf.read(rd / f"{n}.wav", dtype="float32", always_2d=True)[0]
    if n == "vocals": r = r + sf.read(rd / "vocal_room.wav", dtype="float32", always_2d=True)[0]
    c = sf.read(cd / f"{n}.wav", dtype="float32", always_2d=True)[0].mean(axis=1)
    rm = r.mean(axis=1)
    d, s, q = multitrack.delay_to_room(rm, c, FS, lo, hi, 250.0)
    st = multitrack._steady(rm, c, FS, lo, hi, 250.0, d, s)
    print(f"\n{n}: delay {1000*d/FS:.2f} ms, polarity {s:+d}, peak {q:.0f}, steady {st}%")
    al = s * multitrack.shifted(c, d, len(rm)); raw = multitrack.shifted(c, 0, len(rm))
    lr = loudness.integrated(r, FS); lc = loudness.integrated(np.column_stack([al, al]), FS)
    print("  share   case        " + " ".join(f"{l}-{h}".rjust(9) for l,h in B) + "   (dB: sum against the two parts' powers)")
    for share in (0.5, 0.7, 0.8):
        g = 10 ** ((lr + 10*np.log10(share/(1-share)) - lc) / 20)
        for label, x in (("lined up", al), ("as it came", raw), ("no polarity", s*al)):
            pr, pc = bandp(rm), bandp(g*x); ps = bandp(rm + g*x)
            print(f"  {share:4.1f}   {label:11} " + " ".join(f"{10*np.log10(a/(b+1e-30)):+9.1f}" for a,b in zip(ps, pr+pc)))
