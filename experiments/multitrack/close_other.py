# What the close track's "other" stem is: against the close and room stems, by level over time, by
# coherence at the best lag, and by its spectrum. Run: uv run python experiments/multitrack/close_other.py <root>
import glob, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
root = Path(sys.argv[1]); FS = 48000
def env(x, ms=100):
    n = int(ms / 1000 * FS); m = len(x) // n
    return 10 * np.log10(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1) + 1e-20)
def best(a, b):
    n = len(a) + len(b); A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    C = A * np.conj(B); C /= np.abs(C) + 1e-20
    f = np.fft.rfftfreq(n, 1 / FS); C[(f < 150) | (f > 5000)] = 0
    c = np.fft.fftshift(np.fft.irfft(C)); lags = np.arange(len(c)) - len(c) // 2
    m = np.abs(lags) <= int(0.25 * FS); i = np.argmax(np.abs(c[m]))
    return lags[m][i], float(np.abs(c[m][i]) / (np.median(np.abs(c[m])) + 1e-20))
for d in sorted(glob.glob(str(root / "processed/stems_close/* Tr*"))):
    d = Path(d)
    if not (d / "key.json").exists(): continue
    take = d.name.rsplit(" ", 1)[0]; rd = root / "processed/stems_ensemble" / take
    if not (rd / "key.json").exists(): continue
    rc = lambda n: sf.read(d / f"{n}.wav", dtype="float64", always_2d=True)[0].mean(axis=1)
    rr = lambda n: sf.read(rd / f"{n}.wav", dtype="float64", always_2d=True)[0].mean(axis=1)
    o = rc("other"); eo = env(o); act = eo > np.percentile(eo, 95) - 25
    print(f"\n=== {d.name}: other active {100*act.mean():.0f}% of the take; playing level {np.percentile(eo,90):.1f} dB")
    f, p = signal.welch(o, FS, nperseg=8192)
    b = [(50,120),(120,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,8000),(8000,16000)]
    print("  spectrum (dB): " + " ".join(f"{l}-{h}:{10*np.log10(p[(f>=l)&(f<h)].sum()+1e-30):.0f}" for l,h in b))
    for label, s in [(f"close {n}", rc(n)) for n in ("vocals","bass","guitar","drums")] + [(f"room {n}", rr(n)) for n in ("vocals","vocal_room","guitar","bass","other","hihat","cymbals")]:
        e = env(s); m = min(len(e), len(eo)); r = np.corrcoef(e[:m][act[:m]], eo[:m][act[:m]])[0, 1]
        lag, q = best(s, o)
        k = max(lag, 0); j = max(-lag, 0); n = min(len(s) - k, len(o) - j)
        fc, coh = signal.coherence(s[k:k+n], o[j:j+n], FS, nperseg=4096)
        print(f"  vs {label:16} level r {r:+.2f} | lag {lag/FS*1000:+7.2f} ms peak/floor {q:5.0f} | coherence 200-800 {coh[(fc>=200)&(fc<800)].mean():.2f} 800-3200 {coh[(fc>=800)&(fc<3200)].mean():.2f}")
