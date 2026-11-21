# Faster ways for what tops the profile, each against what the program does now, on a real stem of 5 minutes.
# Run from the project root: uv run --with numexpr python experiments/speed/libs.py
import sys, time, numpy as np, soundfile as sf, numba
from scipy import signal
sys.path.insert(0, ".")
from mooz import loudness, dynamics
R = "new_tracks/drive-download-20261107T141948Z-1-001/processed/stems_ensemble/261105_235855 Biancaneve/"
x, fs = sf.read(R + "bass.wav", dtype="float64", always_2d=True); x = x + sf.read(R + "hihat.wav", dtype="float64", always_2d=True)[0]
def best(f, n=3):
    t = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); t = min(t, time.perf_counter() - t0)
    return t, r
def db(a, b): return 10 * np.log10(np.mean((a - b) ** 2) / np.mean(b ** 2) + 1e-30)
def same(a, b): return "the same to the last bit" if np.array_equal(a, b) else f"{db(a, b):.0f} dB"
print(f"signal: {len(x)/fs:.0f} s stereo, threads for filters {loudness.THREADS}")

# --- 1. zero-phase IIR: one compiled loop over the sections, forward and back, in place of scipy's
@numba.njit(cache=True, nogil=True)
def _cascade(sos, x, zi):                      # transposed direct form II, as scipy's sosfilt
    n = x.shape[0]
    for s in range(sos.shape[0]):
        b0, b1, b2, a1, a2 = sos[s, 0], sos[s, 1], sos[s, 2], sos[s, 4], sos[s, 5]
        z0, z1 = zi[s, 0], zi[s, 1]
        for i in range(n):
            xi = x[i]
            y = b0 * xi + z0
            z0 = b1 * xi - a1 * y + z1
            z1 = b2 * xi - a2 * y
            x[i] = y
def zero_phase(sos, x):
    """scipy.signal.sosfiltfilt for one channel, with the cascade compiled: odd extension, each pass started at rest on its first sample."""
    pad = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    ext = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]])
    zi = signal.sosfilt_zi(sos)
    _cascade(sos, ext, zi * ext[0])
    ext = ext[::-1].copy()
    _cascade(sos, ext, zi * ext[0])
    return ext[::-1][pad:-pad].copy()
def zero_phase_stereo(sos, x, pool):
    out = np.empty_like(x)
    def ch(c): out[:, c] = zero_phase(sos, np.ascontiguousarray(x[:, c]))
    list(pool.map(ch, range(x.shape[1]))) if pool else [ch(c) for c in range(x.shape[1])]
    return out
print("\n1. zero-phase IIR filters (stereo)")
zero_phase(signal.butter(2, 100, "high", fs=fs, output="sos"), x[:1000, 0].copy())
for name, sos in (("band 2-8 kHz, order 4", signal.butter(4, [2000, 8000], "band", fs=fs, output="sos")), ("high-pass 30 Hz, order 4", signal.butter(4, 30, "high", fs=fs, output="sos")),
                  ("band 40-150 Hz, order 2", signal.butter(2, [40, 150], "band", fs=fs, output="sos")), ("K-weighting", loudness.k_weighting_sos(fs))):
    t0, ref = best(lambda: loudness.filtfilt(sos, x)); t1, y = best(lambda: zero_phase_stereo(sos, x, loudness.pool())); t2, y2 = best(lambda: zero_phase_stereo(sos, x, None))
    tm, refm = best(lambda: loudness.filtfilt(sos, x[:, 0])); tn, ym = best(lambda: zero_phase(sos, np.ascontiguousarray(x[:, 0])))
    print(f"   {name:26} now {t0:.2f} s | compiled, a thread per channel {t1:.2f} s ({t0/t1:.1f}x) | compiled, one thread {t2:.2f} s | {same(y, ref)} || one channel: now {tm:.2f} s, compiled {tn:.2f} s ({tm/tn:.1f}x), {same(ym, refm)}")

# --- 2. one-way sosfilt on stereo (the loudness meter's K-weighting) and lfilter (attack smoothing)
print("\n2. one-way filters")
@numba.njit(cache=True, nogil=True)
def _onepole(v, a, out):
    y = 0.0
    for i in range(v.shape[0]):
        y = (1 - a) * v[i] + a * y
        out[i] = y
v = np.maximum(np.abs(x[:, 0]), np.abs(x[:, 1])); out = np.empty_like(v); _onepole(v[:10], 0.5, out[:10])
a = np.exp(-1000.0 / (0.5 * fs))
t0, ref = best(lambda: dynamics.attack_smooth(v, 0.5, fs)); t1, _ = best(lambda: _onepole(v, a, out))
print(f"   attack smoothing (scipy lfilter) {t0:.3f} s | compiled {t1:.3f} s ({t0/t1:.1f}x) | {same(out, ref)}")
ksos = loudness.k_weighting_sos(fs)
def kw(x):
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        col = np.ascontiguousarray(x[:, c]); _cascade(ksos, col, np.zeros((len(ksos), 2))); out[:, c] = col
    return out
t0, ref = best(lambda: signal.sosfilt(ksos, x, axis=0)); t1, y = best(lambda: kw(x))
print(f"   K-weighting, one way, stereo (scipy sosfilt) {t0:.3f} s | compiled {t1:.3f} s ({t0/t1:.1f}x) | {same(y, ref)}")

# --- 3. element-wise mathematics on a whole stem: numpy against numexpr (threads)
print("\n3. element-wise mathematics")
try:
    import numexpr as ne
    g = np.linspace(-12.0, 3.0, len(x))
    for name, f_np, f_ne in (("10 ** (g / 20)", lambda: 10 ** (g / 20), lambda: ne.evaluate("10 ** (g / 20)")),
                             ("20 * log10(|x| + eps)", lambda: 20 * np.log10(np.abs(v) + 1e-12), lambda: ne.evaluate("20 * log10(abs(v) + 1e-12)")),
                             ("tanh(3 x + 0.3)", lambda: np.tanh(3 * x + 0.3), lambda: ne.evaluate("tanh(3 * x + 0.3)")),
                             ("x * gain (stereo)", lambda: x * g[:, None], lambda: ne.evaluate("x * gg", local_dict={"x": x, "gg": g[:, None]}))):
        t0, ref = best(f_np); t1, y = best(f_ne)
        print(f"   {name:24} numpy {t0:.3f} s | numexpr, {ne.detect_number_of_threads()} threads {t1:.3f} s ({t0/t1:.1f}x) | {same(y, ref)}")
except ImportError:
    print("   numexpr not there: run with  uv run --with numexpr")

# --- 4. 4x oversampling: scipy's polyphase filter against libsoxr
print("\n4. oversampling by 4 (stereo)")
import soxr
t0, ref = best(lambda: signal.resample_poly(x, 4, 1, axis=0), 1)
for q in ("HQ", "VHQ"):
    t1, y = best(lambda: soxr.resample(x, fs, 4 * fs, quality=q), 1)
    n = min(len(y), len(ref)); print(f"   scipy resample_poly {t0:.2f} s | soxr {q} {t1:.2f} s ({t0/t1:.1f}x) | {db(y[:n], ref[:n]):.0f} dB from scipy's | peak {np.abs(ref).max():.4f} against {np.abs(y).max():.4f}")
