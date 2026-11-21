# Precision, threads, compiled loops and file formats, each on a real stem of 5 minutes (docs/SPEED.md).
# Run from the project root: uv run python experiments/speed/bench.py <a folder for temporary files>
import sys, time, numpy as np, soundfile as sf
from pathlib import Path
from scipy import signal
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
sys.path.insert(0, ".")
from mooz import dynamics, loudness, eq
R = Path("new_tracks/drive-download-20261107T141948Z-1-001/processed/stems_ensemble/261105_235855 Biancaneve")
x64, fs = sf.read(R / "bass.wav", dtype="float64", always_2d=True)
x64 = x64 + sf.read(R / "hihat.wav", dtype="float64", always_2d=True)[0]
x32 = x64.astype(np.float32)
print(f"signal: {len(x64)/fs:.0f} s stereo, {x64.nbytes/1e6:.0f} MB as float64")
def t(f, n=3):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); best = min(best, time.perf_counter() - t0)
    return best, r
def db(a, b): return 10 * np.log10(np.mean((a - b) ** 2) / np.mean(b ** 2) + 1e-30)

print("\n1. zero-phase IIR (loudness.filtfilt), per call")
for name, sos in (("band 2-8 kHz, order 4", signal.butter(4, [2000, 8000], "band", fs=fs, output="sos")),
                  ("high-pass 30 Hz, order 4", signal.butter(4, 30, "high", fs=fs, output="sos")),
                  ("peak 80 Hz Q 4 (biquad)", np.atleast_2d(np.hstack(signal.iirpeak(80, 4, fs=fs))))):
    a, r64 = t(lambda: loudness.filtfilt(sos, x64)); b, r32 = t(lambda: loudness.filtfilt(sos.astype(np.float32), x32))
    c, r32b = t(lambda: loudness.filtfilt(sos, x32))
    print(f"   {name:26} float64 {a:.2f} s | float32 {b:.2f} s (dtype out {r32.dtype}, error {db(r32, r64):.0f} dB) | float32 in, float64 coefficients {c:.2f} s ({r32b.dtype}, error {db(r32b, r64):.0f} dB)")
sos = signal.butter(4, [2000, 8000], "band", fs=fs, output="sos")
print("   threads (does scipy release the GIL?): 8 single-channel filters")
ch = [np.ascontiguousarray(x64[:, i % 2]) for i in range(8)]
a, _ = t(lambda: [signal.sosfiltfilt(sos, c) for c in ch], 2)
for k in (2, 4, 8):
    with ThreadPoolExecutor(k) as ex:
        b, _ = t(lambda: list(ex.map(lambda c: signal.sosfiltfilt(sos, c), ch)), 2)
    print(f"      {k} threads: {a/b:.2f}x of sequential ({a:.2f} s -> {b:.2f} s)")
a, _ = t(lambda: signal.sosfilt(sos, x64, axis=0)); b, _ = t(lambda: [signal.sosfilt(sos, np.ascontiguousarray(x64[:, i])) for i in range(2)])
print(f"   one-way sosfilt on the stereo array {a:.2f} s, channel by channel {b:.2f} s")

print("\n2. across the two channels")
a, r1 = t(lambda: x64.mean(axis=1)); b, r2 = t(lambda: 0.5 * (x64[:, 0] + x64[:, 1]))
print(f"   x.mean(axis=1) {a:.3f} s | 0.5*(L+R) {b:.3f} s  ({a/b:.0f}x), same: {np.allclose(r1, r2)}")
a, r1 = t(lambda: np.abs(x64).max(axis=1)); b, r2 = t(lambda: np.maximum(np.abs(x64[:, 0]), np.abs(x64[:, 1])))
print(f"   abs(x).max(axis=1) {a:.3f} s | maximum(|L|,|R|) {b:.3f} s  ({a/b:.0f}x), same: {np.array_equal(r1, r2)}")
a, r1 = t(lambda: np.mean(x64 * x64, axis=1)); b, r2 = t(lambda: 0.5 * (x64[:, 0] ** 2 + x64[:, 1] ** 2))
print(f"   mean(x*x, axis=1) {a:.3f} s | 0.5*(L²+R²) {b:.3f} s  ({a/b:.0f}x)")

print("\n3. envelope follower (dynamics.release_hold), per call")
v = np.abs(x64).max(axis=1)
a, r = t(lambda: dynamics.release_hold(v, 30.0, fs))
import numba
@numba.njit(cache=False, fastmath=False)
def rh(v, r):
    out = np.empty_like(v); e = 1e-6
    for i in range(len(v)):
        e = e * r
        s = v[i] if v[i] > 1e-6 else 1e-6
        if s > e: e = s
        out[i] = e
    return out
rr = np.exp(-1000.0 / (30.0 * fs)); rh(v[:1000], rr)
b, r2 = t(lambda: rh(v, rr)); v32 = v.astype(np.float32); rh(v32[:1000], np.float32(rr)); c, _ = t(lambda: rh(v32, np.float32(rr)))
print(f"   now (log, running maximum, exp) {a:.3f} s | compiled loop float64 {b:.3f} s ({a/b:.0f}x, error {db(r2, r):.0f} dB) | float32 {c:.3f} s")
a, r = t(lambda: dynamics.attack_smooth(v, 0.5, fs)); print(f"   attack_smooth (lfilter) {a:.3f} s")

print("\n4. true peak (4x oversampling), per call")
a, r = t(lambda: loudness.true_peak_envelope(x64), 1)
b, r2 = t(lambda: loudness.true_peak_envelope(x32), 1)
print(f"   now, float64 {a:.2f} s | same code on float32 {b:.2f} s (max difference {np.abs(r-r2).max():.1e})")
h = signal.firwin(2 * 10 * 4 + 1, 1 / 4, window=("kaiser", 5.0)) * 4
def tp_whole(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return np.maximum(np.abs(up).reshape(len(x), 4, -1).max(axis=(1, 2)), np.abs(x).max(axis=1))
c, r3 = t(lambda: tp_whole(x32), 1); print(f"   whole array at once, float32 {c:.2f} s")
def tp_threads(x, k=8):
    n = len(x); cuts = np.linspace(0, n, k + 1).astype(int)
    def part(i):
        a_, b_ = max(0, cuts[i] - 64), min(n, cuts[i + 1] + 64)
        up = signal.resample_poly(x[a_:b_], 4, 1, axis=0)[(cuts[i] - a_) * 4 : (cuts[i + 1] - a_) * 4]
        return np.abs(up).reshape(cuts[i + 1] - cuts[i], -1).max(axis=1)
    with ThreadPoolExecutor(k) as ex: return np.concatenate(list(ex.map(part, range(k))))
d, r4 = t(lambda: tp_threads(x32), 1); print(f"   8 threads over 8 pieces, float32 {d:.2f} s ({a/d:.1f}x of now)")

print("\n5. loudness.integrated (BS.1770), per call")
a, r = t(lambda: loudness.integrated(x64, fs)); b, r2 = t(lambda: loudness.integrated(x32, fs))
print(f"   float64 {a:.2f} s ({r:.3f} LUFS) | float32 {b:.2f} s ({r2:.3f} LUFS)")

print("\n6. FIR EQ (eq.apply_fir, 8191 taps) and STFT round trip")
hh = signal.firwin(8191, 0.3)
a, r = t(lambda: eq.apply_fir(x64, hh), 2); b, r2 = t(lambda: eq.apply_fir(x32, hh.astype(np.float32)), 2)
import scipy.fft
with scipy.fft.set_workers(8): c, _ = t(lambda: eq.apply_fir(x32, hh.astype(np.float32)), 2)
print(f"   FIR float64 {a:.2f} s | float32 {b:.2f} s (error {db(r2, r):.0f} dB) | float32, 8 FFT workers {c:.2f} s")
def rt(x):
    _, _, z = signal.stft(x[:, 0], fs, nperseg=1024, noverlap=768); return signal.istft(z, fs, nperseg=1024, noverlap=768)[1]
a, _ = t(lambda: rt(x64), 2); b, _ = t(lambda: rt(x32), 2); print(f"   STFT+ISTFT one channel float64 {a:.2f} s | float32 {b:.2f} s")

print("\n7. plain array arithmetic on the whole stereo array")
g64 = np.linspace(0.5, 1.0, len(x64)); g32 = g64.astype(np.float32)
a, _ = t(lambda: x64 * g64[:, None]); b, _ = t(lambda: x32 * g32[:, None]); print(f"   x * gain: float64 {a:.3f} s | float32 {b:.3f} s")
a, _ = t(lambda: 10 ** (g64 / 20)); b, _ = t(lambda: 10 ** (g32 / 20)); print(f"   10**(g/20): float64 {a:.3f} s | float32 {b:.3f} s")
a, _ = t(lambda: 20 * np.log10(np.abs(x64[:, 0]) + 1e-9)); b, _ = t(lambda: 20 * np.log10(np.abs(x32[:, 0]) + np.float32(1e-9))); print(f"   20*log10|x|: float64 {a:.3f} s | float32 {b:.3f} s")

print("\n8. files")
import tempfile, os
d = tempfile.mkdtemp(dir=str(Path(sys.argv[1])))
for lab, kw in (("FLAC 24-bit, default level", dict(subtype="PCM_24")), ("FLAC 24-bit, level 0", dict(subtype="PCM_24", compression_level=0.0)),
                ("FLAC 24-bit, level 0.25", dict(subtype="PCM_24", compression_level=0.25))):
    p = f"{d}/a.flac"; a, _ = t(lambda: sf.write(p, x32, fs, format="FLAC", **kw), 1); print(f"   write {lab:28} {a:.2f} s, {os.path.getsize(p)/1e6:.0f} MB")
p = f"{d}/a.wav"; a, _ = t(lambda: sf.write(p, x32, fs, subtype="FLOAT"), 1); print(f"   write WAV float32 {a:.2f} s, {os.path.getsize(p)/1e6:.0f} MB")
a, _ = t(lambda: sf.read(p, dtype="float64", always_2d=True), 2); b, _ = t(lambda: sf.read(p, dtype="float32", always_2d=True), 2); print(f"   read WAV as float64 {a:.2f} s | as float32 {b:.2f} s")
a, _ = t(lambda: sf.read(f"{d}/a.flac", dtype="float32", always_2d=True), 1); print(f"   read FLAC {a:.2f} s")
