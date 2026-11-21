import sys, time, numpy as np, soundfile as sf
from scipy import signal
from scipy.fft import next_fast_len
sys.path.insert(0, ".")
t00 = time.perf_counter(); import torch; t_imp = time.perf_counter() - t00
from mooz import loudness, dynamics, eq
R = "new_tracks/drive-download-20261107T141948Z-1-001/processed/stems_ensemble/261105_235855 Biancaneve/"
x, fs = sf.read(R + "bass.wav", dtype="float64", always_2d=True); x = x + sf.read(R + "hihat.wav", dtype="float64", always_2d=True)[0]
dev = torch.device("cuda")
t0 = time.perf_counter(); torch.zeros(1, device=dev); torch.cuda.synchronize(); t_init = time.perf_counter() - t0
print(f"torch import {t_imp:.1f} s, CUDA start {t_init:.1f} s (paid once per process); {torch.cuda.get_device_name(0)}; signal {len(x)/fs:.0f} s stereo")
def db(a, b): return 10 * np.log10(np.mean((a - b) ** 2) / np.mean(b ** 2) + 1e-30)
def timed(f, n=3):
    best = 1e9
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = f(); torch.cuda.synchronize(); best = min(best, time.perf_counter() - t0)
    return best, r
def cpu(f, n=2):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); best = min(best, time.perf_counter() - t0)
    return best, r

print("\n(every GPU time includes sending the samples to the card and bringing the result back)")
for dt, name in ((torch.float64, "float64"), (torch.float32, "float32")):
    print(f"\n--- {name} on the card")
    # 1. true-peak envelope: 4x polyphase upsampling, |.|, max over phases and channels
    h = signal.firwin(2 * 10 * 4 + 1, 1 / 4, window=("kaiser", 5.0)) * 4      # scipy's resample_poly filter
    ht = torch.tensor(h[::-1].copy(), dtype=dt, device=dev)
    def tp():
        xt = torch.as_tensor(x, dtype=dt, device=dev).T.unsqueeze(1)           # (2, 1, n)
        up = torch.zeros(2, 1, xt.shape[-1] * 4, dtype=dt, device=dev); up[..., ::4] = xt
        y = torch.nn.functional.conv1d(up, ht.view(1, 1, -1), padding=len(h) // 2)[..., : xt.shape[-1] * 4]
        return y.abs().view(2, -1, 4).amax(dim=(0, 2)).cpu().numpy()
    c, ref = cpu(lambda: loudness.true_peak_envelope(x), 1); g, got = timed(tp)
    print(f"true-peak envelope: CPU now (8 threads) {c:.2f} s | GPU {g:.3f} s | largest difference {np.abs(got - ref[: len(got)]).max():.1e}")
    # 2. FIR EQ (8191 taps) in the frequency domain
    hh = signal.firwin(8191, 0.3)
    def fir():
        xt = torch.as_tensor(x, dtype=dt, device=dev).T
        n = next_fast_len(xt.shape[-1] + len(hh) - 1, real=True)
        H = torch.fft.rfft(torch.as_tensor(hh, dtype=dt, device=dev), n)
        y = torch.fft.irfft(torch.fft.rfft(xt, n) * H, n)[..., len(hh) // 2 : len(hh) // 2 + xt.shape[-1]]
        return y.T.cpu().numpy()
    c, ref = cpu(lambda: eq.apply_fir(x, hh)); g, got = timed(fir)
    print(f"FIR EQ 8191 taps: CPU {c:.2f} s | GPU {g:.3f} s | error {db(got, ref):.0f} dB")
    # 3. envelope follower as it was first written: log, running maximum, exp
    v = np.maximum(np.abs(x[:, 0]), np.abs(x[:, 1]))
    def follow():
        vt = torch.as_tensor(v, dtype=dt, device=dev)
        ramp = (-1000.0 / (30.0 * fs)) * torch.arange(len(v), dtype=dt, device=dev)
        return torch.exp(torch.cummax(torch.log(vt.clamp_min(1e-6)) - ramp, 0).values + ramp).cpu().numpy()
    c, ref = cpu(lambda: dynamics.release_hold(v, 30.0, fs)); g, got = timed(follow)
    print(f"envelope follower: CPU now (compiled) {c:.3f} s | GPU {g:.3f} s | error {db(got, ref):.0f} dB")
    # 4. zero-phase IIR as one multiplication in the frequency domain (|H|^2), with scipy's padding
    for nm, sos in (("band 2-8 kHz", signal.butter(4, [2000, 8000], "band", fs=fs, output="sos")), ("high-pass 30 Hz", signal.butter(4, 30, "high", fs=fs, output="sos"))):
        pad = 3 * (2 * len(sos) + 1)
        def zp():
            xt = torch.as_tensor(x, dtype=dt, device=dev).T
            ext = torch.cat([2 * xt[:, :1] - xt[:, 1 : pad + 1].flip(-1), xt, 2 * xt[:, -1:] - xt[:, -pad - 1 : -1].flip(-1)], dim=-1)
            n = next_fast_len(ext.shape[-1] + 2 * fs, real=True)               # room for the filter to ring out
            w, Hc = signal.sosfreqz(sos, worN=n // 2 + 1, whole=False)
            H2 = torch.as_tensor(np.abs(Hc) ** 2, dtype=dt, device=dev)
            return torch.fft.irfft(torch.fft.rfft(ext, n) * H2, n)[..., pad : pad + xt.shape[-1]].T.cpu().numpy()
        c, ref = cpu(lambda: loudness.filtfilt(sos, x)); g, got = timed(zp, 2)
        print(f"zero-phase IIR, {nm}: CPU {c:.2f} s | GPU {g:.3f} s (of which the filter's response on the CPU) | error {db(got, ref):.0f} dB")
    # 5. saturator's curve on a 4x oversampled signal
    def sat():
        xt = torch.as_tensor(x, dtype=dt, device=dev)
        return ((torch.tanh(xt * 3.0 + 0.3) - np.tanh(0.3)) / 3.0).cpu().numpy()
    c, ref = cpu(lambda: (np.tanh(x * 3.0 + 0.3) - np.tanh(0.3)) / 3.0); g, got = timed(sat)
    print(f"tanh curve on the whole stem: CPU {c:.2f} s | GPU {g:.3f} s | error {db(got, ref):.0f} dB")
    # 6. STFT and back
    win = torch.hann_window(1024, dtype=dt, device=dev)
    def st():
        xt = torch.as_tensor(x[:, 0], dtype=dt, device=dev)
        z = torch.stft(xt, 1024, 256, window=win, return_complex=True)
        return torch.istft(z, 1024, 256, window=win, length=len(xt)).cpu().numpy()
    c, _ = cpu(lambda: signal.istft(signal.stft(x[:, 0], fs, nperseg=1024, noverlap=768)[2], fs, nperseg=1024, noverlap=768)); g, got = timed(st)
    print(f"STFT and back, one channel: CPU {c:.2f} s | GPU {g:.3f} s | round trip error {db(got, x[:, 0]):.0f} dB")
    print(f"card memory at most {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"); torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
