# How far chains in parallel processes scale on this machine: k chain-like jobs at once, in float64 and float32.
# Run from the project root: uv run python experiments/speed/scale.py   (needs about 1 GB of free memory per job)
import sys, time, numpy as np, soundfile as sf
from multiprocessing import get_context
from scipy import signal
sys.path.insert(0, ".")
F = "new_tracks/drive-download-20261107T141948Z-1-001/processed/stems_ensemble/261105_235855 Biancaneve/bass.wav"
def task(dtype):
    from mooz import dynamics, loudness, eq
    t0 = time.perf_counter()
    x, fs = sf.read(F, dtype=dtype, always_2d=True)
    for lo, hi in ((40, 150), (150, 400), (400, 2000), (2000, 8000), (60, 300), (1000, 4000)) * 2:
        y = loudness.filtfilt(signal.butter(4, [lo, hi], "band", fs=fs, output="sos"), x)
        x = x + 0.01 * y
    a = np.maximum(np.abs(x[:, 0]), np.abs(x[:, 1]))
    for rel in (30.0, 300.0, 100.0, 50.0) * 2:
        e = dynamics.attack_smooth(dynamics.release_hold(a, rel, fs), 0.5, fs)
        x = x * (1.0 / (1.0 + 0.01 * e)).astype(x.dtype)[:, None]
    h = signal.firwin(8191, 0.4).astype(x.dtype)
    for _ in range(2): x = eq.apply_fir(x, h)
    for c in range(2):
        _, _, z = signal.stft(x[:, c], fs, nperseg=1024, noverlap=768); x[:, c] = signal.istft(z, fs, nperseg=1024, noverlap=768)[1][: len(x)]
    return time.perf_counter() - t0
if __name__ == "__main__":
    ctx = get_context("spawn")
    for dtype, ks in (("float64", (1, 2, 4, 6)), ("float32", (1, 2, 4, 6, 8))):
        base = None
        for k in ks:
            with ctx.Pool(k) as pool:
                pool.map(task, [dtype] * k)  # warm: imports, page cache
                t0 = time.perf_counter(); each = pool.map(task, [dtype] * k, chunksize=1); wall = time.perf_counter() - t0
            base = base or wall
            print(f"{dtype}: {k} chains at once: {wall:5.1f} s for all ({np.mean(each):5.1f} s each) -> {k * base / wall:4.1f}x the work of one in the same time", flush=True)
