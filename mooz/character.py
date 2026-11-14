"""Stage 0: characterise the song, to steer the mix stages.

From the trimmed recording and its raw stems we measure what kind of song it
is and how its energy moves over time, and classify it into a style that the
mix stages adapt to (see [character.styles] in defaults.toml):

  wall      loud, dense and steady: distorted guitars, drums playing throughout,
            little contrast (a "wall of sound")
  ballad    sparse: drums playing little or not at all, slow or soft, wide dynamics
  build     a crescendo: the energy rises across the song
  dynamic   strong contrast between quiet and loud sections (verse/chorus)
  steady    none of the above
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import loudness
from .loudness import EPS, filtfilt

HOP_S = 0.1


def _onset_envelope(x: np.ndarray, fs: float) -> np.ndarray:
    """Positive spectral flux, 100 Hz frame rate."""
    nfft, hop = 2048, int(HOP_S * fs / 2)
    _, _, z = signal.stft(x.mean(axis=1), fs, nperseg=nfft, noverlap=nfft - hop)
    mag = np.log1p(np.abs(z) * 100)
    flux = np.maximum(np.diff(mag, axis=1), 0).sum(axis=0)
    return flux


def _tempo(flux: np.ndarray, frame_rate: float) -> float | None:
    """Tempo (BPM, 60..200) from the autocorrelation of the onset envelope."""
    f = flux - flux.mean()
    if not np.any(f):
        return None
    ac = np.correlate(f[: int(60 * frame_rate)], f[: int(60 * frame_rate)], "full")[len(f[: int(60 * frame_rate)]) - 1 :]
    lo, hi = int(frame_rate * 60 / 200), int(frame_rate * 60 / 60)
    if hi >= len(ac):
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(60 * frame_rate / lag)


def _sections(st_lufs: np.ndarray, min_len_s: float = 12.0, jump_lu: float = 3.0) -> list[dict]:
    """Segments of the short-term loudness curve that differ by more than jump_lu."""
    k = int(5 / HOP_S)
    smooth = np.convolve(st_lufs, np.ones(k) / k, mode="same")
    n = int(min_len_s / HOP_S)
    bounds = [0]
    i = n
    while i < len(smooth) - n:
        before, after = np.median(smooth[i - n : i]), np.median(smooth[i : i + n])
        if abs(after - before) >= jump_lu:
            bounds.append(i)
            i += n
        else:
            i += int(1 / HOP_S)
    bounds.append(len(smooth))
    return [{"start_s": round(a * HOP_S, 1), "end_s": round(b * HOP_S, 1),
             "lufs": round(float(np.median(smooth[a:b])), 1)} for a, b in zip(bounds, bounds[1:])]


def characterise(y: np.ndarray, fs: float, stems: dict[str, np.ndarray] | None, triggers: dict | None) -> dict:
    """Metrics describing the song, its style and the reasons for it."""
    ld = loudness.measure(y, fs)
    st = ld.short_term
    st = st[np.isfinite(st) & (st > ld.integrated - 25)]
    lufs = ld.integrated
    p10, p90 = (np.percentile(st, [10, 90]) if st.size else (lufs, lufs))
    thirds = np.array_split(st, 3) if st.size >= 3 else [st, st, st]
    arc = float(np.median(thirds[2]) - np.median(thirds[0])) if st.size >= 3 else 0.0
    t = np.arange(st.size) * HOP_S / 60
    slope = float(np.polyfit(t, st, 1)[0]) if st.size > 10 else 0.0  # LU per minute
    flux = _onset_envelope(y, fs)
    fr = 1 / (HOP_S / 2)
    thr = np.median(flux) + 2 * np.median(np.abs(flux - np.median(flux)))
    peaks, _ = signal.find_peaks(flux, height=thr, distance=int(0.1 * fr))
    onset_rate = len(peaks) / (len(y) / fs)
    tempo = _tempo(flux, fr)
    f, p = signal.welch(y.mean(axis=1), fs, nperseg=8192)
    centroid = float((f * p).sum() / (p.sum() + EPS))
    m = {"integrated_lufs": round(lufs, 1), "lra_lu": round(ld.lra, 1), "contrast_lu": round(float(p90 - p10), 1),
         "arc_lu": round(arc, 1), "slope_lu_per_min": round(slope, 2), "onset_rate_per_s": round(onset_rate, 2),
         "tempo_bpm": round(tempo, 0) if tempo else None, "brightness_hz": round(centroid, 0),
         "sections": _sections(ld.short_term[np.isfinite(ld.short_term)])}
    if triggers:
        from .effects import _hold

        kit = np.max([triggers[k] for k in ("kick", "snare", "toms") if k in triggers], axis=0)
        playing = _hold(kit > 0.08, fs, 0.3, 2.0)  # the drummer is playing: a hit within the last 2 s
        m["drums_playing_pct"] = round(100 * float(np.mean(playing)), 0)
        m["hits_per_s"] = round(float(np.sum(np.diff((kit > 0.5).astype(int)) > 0)) / (len(kit) / fs), 2)
    if stems and "vocals" in stems:
        m["vocals_singing_pct"] = round(100 * float(np.mean(loudness.active_mask(stems["vocals"], fs))), 0)
    guitars = [stems[n] for n in ("guitar", "guitar_left", "guitar_right") if stems and n in stems]
    if guitars:
        g = sum(guitars)
        band = filtfilt(signal.butter(2, [1000, 6000], "bandpass", fs=fs, output="sos"), g).mean(axis=1)
        _, pg = signal.welch(band, fs, nperseg=4096)
        pg = pg[(np.fft.rfftfreq(4096, 1 / fs) > 1000) & (np.fft.rfftfreq(4096, 1 / fs) < 6000)]
        m["guitar_flatness"] = round(float(np.exp(np.mean(np.log(pg + EPS))) / (np.mean(pg) + EPS)), 3)  # ~1 = noisy/distorted
    m["style"], m["why"] = _classify(m)
    return m


def _classify(m: dict) -> tuple[str, str]:
    drums = m.get("drums_playing_pct", 100)
    flat = m.get("guitar_flatness", 0.0)
    tempo = m.get("tempo_bpm") or 120
    if drums < 50 or (tempo < 85 and m["onset_rate_per_s"] < 2.5):
        return "ballad", f"drums playing {drums:.0f}% of the time, tempo {tempo:.0f} BPM, {m['onset_rate_per_s']:.1f} onsets/s"
    if m["arc_lu"] >= 4.0 and m["slope_lu_per_min"] > 0.8:
        return "build", f"energy rises {m['arc_lu']:+.1f} LU from the first to the last third ({m['slope_lu_per_min']:+.1f} LU/min)"
    if m["contrast_lu"] >= 7.0:
        return "dynamic", f"{m['contrast_lu']:.1f} LU between the quiet and loud parts"
    if m["contrast_lu"] <= 5.0 and drums >= 85 and (flat >= 0.25 or m["onset_rate_per_s"] >= 3.0):
        return "wall", f"steady ({m['contrast_lu']:.1f} LU contrast), drums {drums:.0f}% of the time, dense/distorted"
    return "steady", f"{m['contrast_lu']:.1f} LU contrast, no strong build"
