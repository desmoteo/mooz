"""Prototypes for the bass stem's clean-up and body (before they go into mooz):

  pitch(x)          pyin on the low band, octave-corrected: a weak fundamental (the room and the
                    recorder lose 40-60 Hz) makes pyin report the octave above; when the octave
                    below has its own odd harmonics (f/2 and 3f/2 well above the noise), it is the
                    real note
  harmonics(x, f0)  per-frame amplitudes of the first harmonics, by demodulation along the pitch
  clean(x, f0)      attenuate what lies between the note's harmonics (buzz, bleed, separation
                    noise), 150 Hz - 5 kHz, only where the pitch is known

  uv run python experiments/bass_proto/proto.py
"""

from __future__ import annotations

import numpy as np
from scipy import signal

SR_P = 4000      # pitch analysis rate
HOP_P = 40       # 10 ms


def pitch(x: np.ndarray, fs: int, fmin: float = 30.0, fmax: float = 300.0, prob: float = 0.2,
          octave_db: float = -14.0) -> tuple[np.ndarray, np.ndarray]:
    """(f0 per 10 ms frame, NaN where unvoiced), octave-corrected; and the frame times."""
    import librosa

    mono = x.mean(axis=1) if x.ndim == 2 else x
    lo = signal.resample_poly(signal.sosfiltfilt(signal.butter(4, 600, "low", fs=fs, output="sos"), mono), SR_P, fs)
    f0, v, p = librosa.pyin(lo.astype(np.float32), fmin=fmin, fmax=fmax, sr=SR_P, frame_length=2048, hop_length=HOP_P)
    f0 = np.where(v & (p >= prob), f0, np.nan)
    # octave check on a 0.5 s window spectrum (2 Hz bins) at each voiced frame
    n = 2048
    win = np.hanning(n)
    pad = np.pad(lo, (n // 2, n // 2))
    for i in np.nonzero(np.isfinite(f0))[0]:
        f = f0[i]
        if f / 2 < fmin:
            continue
        seg = pad[i * HOP_P: i * HOP_P + n] * win
        S = np.abs(np.fft.rfft(seg, 4 * n))
        hz = np.fft.rfftfreq(4 * n, 1 / SR_P)
        at = lambda q: S[(hz > q * 0.97) & (hz < q * 1.03)].max()  # noqa: E731
        odd = max(at(f / 2), at(1.5 * f))
        even = max(at(f), at(2 * f))
        if 20 * np.log10(odd / (even + 1e-12) + 1e-12) > octave_db:
            f0[i] = f / 2
    # short glitches: median over 70 ms where voiced
    ok = np.isfinite(f0)
    med = signal.medfilt(np.where(ok, f0, 0.0), 7)
    f0 = np.where(ok & (med > 0), np.where(np.abs(np.log2(f0 / np.maximum(med, 1e-9))) > 0.9, med, f0), f0)
    return f0, np.arange(len(f0)) * HOP_P / SR_P


def harmonic_amps(x: np.ndarray, fs: int, f0: np.ndarray, t_f: np.ndarray, ks=(1, 2, 3), bw: float = 12.0):
    """Amplitude of harmonics k*f0 along the pitch (complex demodulation), per sample; NaN pitch
    frames held (and masked out by the caller)."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    ok = np.isfinite(f0)
    t = np.arange(len(mono)) / fs
    lf = np.interp(t, t_f[ok], np.log(f0[ok]))
    phi = 2 * np.pi * np.cumsum(np.exp(lf)) / fs
    lp = signal.butter(4, bw, "low", fs=fs, output="sos")
    out = {}
    for k in ks:
        c = 2 * (signal.sosfiltfilt(lp, mono * np.cos(k * phi)) - 1j * signal.sosfiltfilt(lp, mono * np.sin(k * phi)))
        out[k] = np.abs(c)
    mask = np.interp(t, t_f, ok.astype(float)) > 0.99
    return out, mask


def clean(x: np.ndarray, fs: int, f0: np.ndarray, t_f: np.ndarray, depth_db: float = 12.0, fmin: float = 150.0,
          fmax: float = 5000.0, tol_cents: float = 40.0, tol_hz: float = 8.0, stretch: float = 3e-4,
          nfft: int = 4096, hop: int = 512) -> np.ndarray:
    """Turn down what lies between the harmonics of the note being played, by up to depth_db,
    fmin..fmax, where the pitch is known. Harmonic k is expected at k*f0*sqrt(1 + stretch*k^2)
    (a bass string's partials run sharp) within tol_cents / tol_hz; a soft mask."""
    f, t, Z = signal.stft(x.T, fs, nperseg=nfft, noverlap=nfft - hop)  # (ch, F, T)
    f0t = np.interp(t, t_f, np.where(np.isfinite(f0), f0, 0.0))
    voiced = np.interp(t, t_f, np.isfinite(f0).astype(float)) > 0.99
    floor = 10 ** (-depth_db / 20)
    band = (f >= fmin) & (f <= fmax)
    G = np.ones((len(f), len(t)), dtype=np.float32)
    fb = f[band]
    for j in np.nonzero(voiced & (f0t > 0))[0]:
        p = f0t[j]
        k = np.maximum(1, np.round(fb / p))
        # nearest stretched harmonic (one Newton step for the stretch)
        k = np.maximum(1, np.round(fb / (p * np.sqrt(1 + stretch * k ** 2))))
        fk = k * p * np.sqrt(1 + stretch * k ** 2)
        tol = np.maximum(tol_hz, fk * (2 ** (tol_cents / 1200) - 1))
        d = np.abs(fb - fk) / tol
        g = np.where(d <= 1, 1.0, np.where(d >= 2, floor, floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * (d - 1)))))
        G[band, j] = g
    # smooth over time (no fluttering): 3 frames
    G = signal.convolve2d(G, np.ones((1, 3)) / 3, mode="same", boundary="symm")
    y = signal.istft(Z * G[None], fs, nperseg=nfft, noverlap=nfft - hop)[1].T
    return np.pad(y, ((0, max(0, len(x) - len(y))), (0, 0)))[: len(x)]
