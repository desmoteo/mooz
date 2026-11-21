import sys, time, numpy as np, soundfile as sf, scipy.fft as sfft
from scipy import signal
sys.path.insert(0, ".")
from mooz import loudness
R = "new_tracks/drive-download-20261107T141948Z-1-001/processed/stems_ensemble/261105_235855 Biancaneve/"
x64, fs = sf.read(R + "bass.wav", dtype="float64", always_2d=True); x64 = x64 + sf.read(R + "hihat.wav", dtype="float64", always_2d=True)[0]
x32 = x64.astype(np.float32)
def db(a, b): return 10 * np.log10(np.mean((a - b) ** 2) / np.mean(b ** 2) + 1e-30)
def best(f, n=3):
    t = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); t = min(t, time.perf_counter() - t0)
    return t, r
def h2(sos, nbins, n, dtype):
    """|H|^2 of the cascade at the bins of an FFT of n points, in real arithmetic (float64, then cast)."""
    c1 = np.cos(np.arange(nbins) * (2 * np.pi / n)); c2 = 2 * c1 * c1 - 1
    out = np.ones(nbins)
    for b0, b1, b2, a0, a1, a2 in sos:
        out *= (b0*b0 + b1*b1 + b2*b2 + 2*(b0*b1 + b1*b2)*c1 + 2*b0*b2*c2) / (a0*a0 + a1*a1 + a2*a2 + 2*(a0*a1 + a1*a2)*c1 + 2*a0*a2*c2)
    return out.astype(dtype)
def whole(x, sos, workers, ring_s=2.0):
    pad = 3 * (2 * len(sos) + 1)
    ext = np.concatenate([2 * x[:1] - x[pad:0:-1], x, 2 * x[-1:] - x[-2:-pad - 2:-1]])
    n = sfft.next_fast_len(len(ext) + int(ring_s * fs), real=True)
    H = h2(sos, n // 2 + 1, n, x.dtype)
    return sfft.irfft(sfft.rfft(ext, n, axis=0, workers=workers) * H[:, None], n, axis=0, workers=workers)[pad:pad + len(x)]
def blocks(x, sos, workers, block=1 << 16, margin=1 << 15):
    """Overlap-save: each block is filtered with margin samples of the signal on both sides for the filter to ring
    in (zero-phase: it rings both ways), all blocks as one batch of FFTs on threads."""
    pad = 3 * (2 * len(sos) + 1)
    ext = np.concatenate([2 * x[:1] - x[pad:0:-1], x, 2 * x[-1:] - x[-2:-pad - 2:-1]])
    n = block + 2 * margin
    nb = -(-len(ext) // block)
    buf = np.zeros((nb * block + 2 * margin, x.shape[1]), x.dtype); buf[margin:margin + len(ext)] = ext
    idx = np.arange(nb)[:, None] * block + np.arange(n)[None, :]
    seg = buf[idx]                                               # (blocks, n, channels)
    H = h2(sos, n // 2 + 1, n, x.dtype)
    y = sfft.irfft(sfft.rfft(seg, n, axis=1, workers=workers) * H[None, :, None], n, axis=1, workers=workers)[:, margin:margin + block]
    return y.reshape(nb * block, x.shape[1])[pad:pad + len(x)]
filters = (("band 2-8 kHz, order 4", signal.butter(4, [2000, 8000], "band", fs=fs, output="sos")),
           ("high-pass 30 Hz, order 4", signal.butter(4, 30, "high", fs=fs, output="sos")),
           ("band 40-150 Hz, order 2", signal.butter(2, [40, 150], "band", fs=fs, output="sos")),
           ("peak 80 Hz, Q 4", np.atleast_2d(np.hstack(signal.iirpeak(80, 4, fs=fs)))),
           ("K-weighting", loudness.k_weighting_sos(fs)))
print(f"signal {len(x64)/fs:.0f} s stereo; times are for both channels; error is against the filter as it is now (float64)")
for name, sos in filters:
    t0, ref = best(lambda: loudness.filtfilt(sos, x64))
    print(f"\n{name}: now {t0:.2f} s")
    for lab, f in (("whole stem, float64, 2 threads", lambda: whole(x64, sos, 2)), ("whole stem, float32, 2 threads", lambda: whole(x32, sos, 2)),
                   ("blocks, float32, 1 thread", lambda: blocks(x32, sos, 1)), ("blocks, float32, 4 threads", lambda: blocks(x32, sos, 4)),
                   ("blocks, float32, 8 threads", lambda: blocks(x32, sos, 8)), ("blocks, float64, 8 threads", lambda: blocks(x64, sos, 8)),
                   ("blocks of 2^14 +- 2^13, float32, 8 threads", lambda: blocks(x32, sos, 8, 1 << 14, 1 << 13))):
        t, y = best(f, 2)
        print(f"   {lab:44} {t:.2f} s  ({t0/t:.1f}x)  error {db(y, ref):.0f} dB")
