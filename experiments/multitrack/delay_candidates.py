# Does the whole-take delay add up, against the other delays the same stem's windows suggest?
# For each take, the merged sum against the two parts' powers, per band, for each candidate:
# above 0 dB they add, below they cancel. Run: uv run python experiments/multitrack/delay_candidates.py <root>
import sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, ".")
from rehearsal_master import multitrack, loudness
root = Path(sys.argv[1]); FS = 48000
B = ((50,100),(100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6400))
SHARE = {"vocals": 0.8, "bass": 0.7}
CAND = {"vocals": ((9.6,-1),(10.0,-1),(11.54,-1)), "bass": ((11.05,1),(16.05,1),(40.55,-1))}
BAND = {"vocals": (200,5000), "bass": (50,700)}
def bandp(x):
    f, p = signal.welch(x, FS, nperseg=8192)
    return np.array([p[(f>=l)&(f<h)].sum() for l,h in B])
print(" "*40 + " ".join(f"{l}-{h}".rjust(9) for l,h in B))
for rd in sorted((root/"processed/stems_ensemble").iterdir()):
    take = rd.name; cd = root/"processed/stems_close"/f"{take} Tr2"
    for n in ("vocals","bass"):
        r = sf.read(rd/f"{n}.wav", dtype="float32", always_2d=True)[0]
        if n == "vocals":
            r = r + sf.read(rd/"vocal_room.wav", dtype="float32", always_2d=True)[0]
        c = sf.read(cd/f"{n}.wav", dtype="float32", always_2d=True)[0].mean(axis=1)
        rm = r.mean(axis=1)
        d0, s0, q0 = multitrack.delay_to_room(rm, c, FS, *BAND[n], 250.0)
        st = multitrack._steady(rm, c, FS, *BAND[n], 250.0, d0, s0)
        print(f"{take} {n}: found {1000*d0/FS:.2f} ms pol {s0:+d} peak {q0:.0f} steady {st}%")
        lr = loudness.integrated(r, FS)
        for ms, s in ((1000*d0/FS, s0),) + CAND[n]:
            al = s*multitrack.shifted(c, round(ms*FS/1000), len(rm))
            g = 10**((lr + 10*np.log10(SHARE[n]/(1-SHARE[n])) - loudness.integrated(np.column_stack([al,al]),FS))/20)
            pr, pc, ps = bandp(rm), bandp(g*al), bandp(rm+g*al)
            print(f"   {ms:6.2f} ms {s:+d}".ljust(40) + " ".join(f"{10*np.log10(a/(b+1e-30)):+9.1f}" for a,b in zip(ps,pr+pc)))
