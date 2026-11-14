# What the close track holds and how its stems line up with the room's.
# Run from the project root: uv run python experiments/multitrack/close_vs_room.py <recordings root>
import glob, json, os, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, ".")
from mooz import loudness, separate

root = Path(sys.argv[1])
FS = 48000
BANDS = {"vocals": (200, 5000), "bass": (50, 700), "guitar": (200, 5000)}

def gcc(a, b, fs, lo, hi, max_s=0.25):
    """Delay of a behind b (s), sign and strength: PHAT-weighted cross-spectrum in lo..hi.
    d > 0: a is later than b."""
    n = len(a) + len(b)
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    f = np.fft.rfftfreq(n, 1 / fs)
    C = A * np.conj(B)
    C = C / (np.abs(C) + 1e-20)
    C[(f < lo) | (f > hi)] = 0
    c = np.fft.fftshift(np.fft.irfft(C))
    lags = np.arange(len(c)) - len(c) // 2
    m = np.abs(lags) <= int(max_s * fs)
    i = np.argmax(np.abs(c[m]))
    floor = np.median(np.abs(c[m]))
    return lags[m][i] / fs, np.sign(c[m][i]), float(np.abs(c[m][i]) / (floor + 1e-20))

for close_dir in sorted(glob.glob(str(root / "processed/stems_close/* Tr*"))):
    close_dir = Path(close_dir)
    if not (close_dir / "key.json").exists():
        continue
    take = close_dir.name.rsplit(" ", 1)[0]
    room_dir = root / "processed/stems_ensemble" / take
    if not (room_dir / "key.json").exists():
        continue
    print(f"\n=== {close_dir.name}")
    names = sorted(f.stem for f in close_dir.glob("*.wav") if not f.stem.startswith("_"))
    lv = {n: loudness.integrated(sf.read(close_dir / f"{n}.wav", dtype="float64", always_2d=True)[0], FS) for n in names}
    tr = sum(sf.read(close_dir / f"{n}.wav", dtype="float64", always_2d=True)[0] for n in names)
    tot = loudness.integrated(tr, FS)
    print("  close stems re the sum (LU): " + "  ".join(f"{n} {lv[n]-tot:+.1f}" for n in names) + f"  | sum {tot:.1f} LUFS")
    for name, (lo, hi) in BANDS.items():
        c, fs = sf.read(close_dir / f"{name}.wav", dtype="float64", always_2d=True)
        c = c.mean(axis=1)
        r = sf.read(room_dir / f"{name}.wav", dtype="float64", always_2d=True)[0]
        if name == "vocals":
            r = r + sf.read(room_dir / "vocal_room.wav", dtype="float64", always_2d=True)[0]
        rm = r.mean(axis=1)
        d, s, q = gcc(rm, c, fs, lo, hi)
        # stability: 8 windows of 20 s
        win = int(20 * fs); dd = []
        for k in np.linspace(0, max(0, len(c) - win), 8).astype(int):
            d2, s2, q2 = gcc(rm[k:k+win], c[k:k+win], fs, lo, hi)
            dd.append((round(d2 * 1000, 2), int(s2), round(q2, 0)))
        # coherence after alignment, per octave
        k = int(round(d * fs)) # room later by k
        a = rm[k:] if k >= 0 else rm
        b = c[:len(c)-k] if k >= 0 else c[-k:]
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        f, coh = signal.coherence(a, b, fs, nperseg=4096)
        band = lambda l, h: float(np.mean(coh[(f >= l) & (f < h)]))
        # room stem's and close stem's activity
        la, lb = loudness.integrated(r, fs), loudness.integrated(c[:, None], fs)
        print(f"  {name:7} close {lb:6.1f} LUFS, room {la:6.1f} LUFS | delay room-close {d*1000:+7.2f} ms, polarity {int(s):+d}, peak/floor {q:5.0f}")
        print(f"          windows (ms,pol,q): {dd}")
        print(f"          coherence " + " ".join(f"{l}-{h}:{band(l,h):.2f}" for l, h in ((50,100),(100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6400))))
