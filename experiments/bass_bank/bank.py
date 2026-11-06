"""A bass as a bank of band-passes on the notes of a 4/5-string bass, rebalanced note by note,
plus its attacks on a track of their own:

  1. bank        one zero-phase band-pass per semitone, from the low B (31 Hz) up to the 8th
                 harmonic of the highest fretted note (G4 x 8: 3.1 kHz), tuned to the band's own
                 tuning (measured from the pitch). Each band passes +-width_cents around its note
                 and rejects what lies between the notes (floor_db): the fundamentals and the main
                 harmonics of every note pass (k = 2, 3, 4, 6, 8, 9 fall within a few cents of a
                 semitone; 5 and 10: 14 cents; 7: 31 cents, mostly out). A band is never narrower
                 than +-min_hz: a narrower one would ring (+-1 Hz: 0.3 s) and smear the notes;
                 so at the bottom, where the notes are under 2 Hz apart, the bands join up and
                 pass everything (selectivity starts about where width_cents equals min_hz).
                 Bands are applied on the whole song's spectrum at once: exact, zero-phase.
  2. rebalance   along the note being played (note_eq's checked, octave-safe pitch): each harmonic,
                 followed along the pitch (track_hz: ~30 ms), gets the gain that brings the note's
                 harmonic profile towards a studio bass's (studio_harmonics.json, by register;
                 relative to its 2nd-4th harmonics, by `strength`, at most +boost_db / +fund_boost_db
                 / -cut_db). Not with the bands' gains: a band a semitone wide at 37 Hz responds over
                 about a second, so a short note got only half its gain. The bands only turn down,
                 where they tell notes apart, what is no harmonic of the note (another instrument on
                 the semitone grid, the last note's ring): other_db; where there is no note,
                 unpitched_db above 250 Hz.
  3. attacks     what the bank does not pass (the attack's noise burst, thump, string and pick),
                 taken back only around each note's start (rise 3 ms before it, attack_ms held,
                 release_ms out), 30 Hz - 8 kHz.
  4. recombined  2 + attack_db x 3.

The input is made mono (the pipeline places the bass in the middle anyway).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

DEFAULTS = {
    "width_cents": 25.0, "edge_cents": 15.0, "min_hz": 4.0, "floor_db": -30.0,
    "lowest_hz": 30.87 / 2 ** (1 / 12), "highest_hz": 392.0 * 8,  # low B (5-string) .. G4 x 8
    "strength": 1.0, "boost_db": 6.0, "fund_boost_db": 12.0, "cut_db": 12.0, "harmonics": 10,
    "other_db": -12.0, "unpitched_db": -6.0, "union_ms": 30.0, "track_hz": 15.0,
    "attack_ms": 30.0, "release_ms": 40.0, "attack_db": 0.0,
}
PROFILE = Path(__file__).with_name("studio_harmonics.json")


def tuning_cents(f0: np.ndarray) -> float:
    """The band's tuning against A = 440 Hz (cents), from the notes' pitch: the circular mean
    of each frame's distance to the nearest semitone."""
    c = 1200 * np.log2(f0[np.isfinite(f0)] / 440.0)
    ang = 2 * np.pi * c / 100
    return float(np.angle(np.mean(np.exp(1j * ang))) * 100 / (2 * np.pi))


def grid(cfg: dict, tune: float) -> np.ndarray:
    """The semitones' frequencies from lowest_hz to highest_hz, in the band's tuning."""
    ref = 440.0 * 2 ** (tune / 1200)
    n = np.arange(np.ceil(12 * np.log2(cfg["lowest_hz"] / ref)), np.floor(12 * np.log2(cfg["highest_hz"] / ref)) + 1)
    return ref * 2 ** (n / 12)


def shapes(f: np.ndarray, centres: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every frequency bin: the comb's gain (1 near a note, floor between), the nearest
    semitone and the distance to it (cents)."""
    floor = 10 ** (cfg["floor_db"] / 20)
    pos = 12 * np.log2(np.maximum(f, 1e-3) / centres[0])  # semitones above the lowest note
    near = np.clip(np.round(pos), 0, len(centres) - 1).astype(int)
    d = np.abs(pos - near) * 100  # cents to the nearest note
    w = np.maximum(cfg["width_cents"], 1200 * np.log2(1 + cfg["min_hz"] / np.maximum(f, 1e-3)))
    e = cfg["edge_cents"]
    A = np.where(d <= w, 1.0, np.where(d >= w + e, floor, floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * (d - w) / e))))
    # the range: nothing under the lowest note's lower quarter-tone or over the highest's upper one
    lo, hi = centres[0] * 2 ** (-1 / 24), centres[-1] * 2 ** (1 / 24)
    A *= np.clip((f - lo * 0.7) / (lo * 0.3), 0, 1) * np.clip((hi * 1.25 - f) / (hi * 0.25), 0, 1)
    return A, near, d


def target_profile(f0: float, prof: dict, k: int) -> np.ndarray:
    """The studio bass's harmonic levels (dB re the 2nd, k = 1..k) for a note of fundamental f0:
    interpolated between the registers' centres (above the last measured one: that one)."""
    edges = np.array(prof["registers_hz"])
    rows = [(np.sqrt(a * b), np.array(r[:k])) for a, b, r in zip(edges[:-1], edges[1:], prof["studio_db_re_2nd"]) if r]
    cs = np.log([c for c, _ in rows])
    R = np.array([r for _, r in rows])
    lf = np.clip(np.log(f0), cs[0], cs[-1])
    return np.array([np.interp(lf, cs, R[:, i]) for i in range(k)])


def harmonic_amps(mono: np.ndarray, fs: float, f0: np.ndarray, t_f: np.ndarray, k: int, step: int) -> np.ndarray:
    """Each harmonic's amplitude along the pitch (rows k = 1..k), every `step` samples."""
    ok = np.isfinite(f0)
    t = np.arange(len(mono)) / fs
    phi = 2 * np.pi * np.cumsum(np.exp(np.interp(t, t_f[ok], np.log(f0[ok])))) / fs
    lp = signal.butter(2, 8.0, "low", fs=fs, output="sos")
    return np.array([np.abs(signal.sosfiltfilt(lp, mono * np.cos(i * phi)) - 1j * signal.sosfiltfilt(lp, mono * np.sin(i * phi)))[::step]
                     for i in range(1, k + 1)])


def bank(x: np.ndarray, fs: float, f0: np.ndarray, t_f: np.ndarray, onsets: np.ndarray, cfg: dict | None = None,
         parts: dict | None = None) -> tuple[np.ndarray, dict]:
    """x (n, ch) -> the recombined bass (n, 2) and stats. f0, t_f: the pitch (NaN: no note),
    onsets: the notes' starts (s). parts, if a dict, gets every intermediate (mono)."""
    cfg = DEFAULTS | (cfg or {})
    prof = json.loads(PROFILE.read_text())
    mono = x.mean(axis=1)
    n = len(mono)
    tune = tuning_cents(f0)
    centres = grid(cfg, tune)
    X = np.fft.rfft(mono)
    f = np.fft.rfftfreq(n, 1 / fs)
    A, _, _ = shapes(f, centres, cfg)
    # each bin belongs to its two nearest notes' bands, crossfaded over the quarter-tone between them (where
    # the bands have joined up at the bottom, a switch there would ring)
    pos = 12 * np.log2(np.maximum(f, 1e-3) / centres[0])
    frac = pos - np.floor(pos)  # 0 at a note, 1 at the next
    xf = np.clip((frac - 0.25) / 0.5, 0, 1)  # share of the upper neighbour, 0.25 .. 0.75 semitone
    xf = 0.5 * (1 - np.cos(np.pi * xf))
    low_note = np.clip(np.floor(pos), 0, len(centres) - 1).astype(int)

    # the gains, per band and 10 ms frame
    step = int(fs // 100)
    t_g = np.arange(0, n, step) / fs
    p = np.interp(t_g, t_f, np.where(np.isfinite(f0), f0, 0.0))
    pitched = (np.interp(t_g, t_f, np.isfinite(f0).astype(float)) > 0.99) & (p > 0)
    K = cfg["harmonics"]
    amps = harmonic_amps(mono, fs, f0, t_f, K, step)[:, :len(t_g)]
    ref = np.sqrt(np.mean(amps[1:4] ** 2, axis=0)) + 1e-12  # the note's 2nd-4th harmonics
    G = np.ones((K, len(t_g)))
    for j in np.nonzero(pitched)[0]:
        tgt = 10 ** (target_profile(p[j], prof, K) / 20)
        tgt /= np.sqrt(np.mean(tgt[1:4] ** 2))
        want = 20 * np.log10(tgt / (amps[:, j] / ref[j] + 1e-9)) * cfg["strength"]
        lim = np.full(K, cfg["boost_db"])
        lim[0] = cfg["fund_boost_db"]
        G[:, j] = 10 ** (np.clip(want, -cfg["cut_db"], lim) / 20)
    G = ndimage.uniform_filter1d(G, 3, axis=1)  # 30 ms
    other, unp = 10 ** (cfg["other_db"] / 20), 10 ** (cfg["unpitched_db"] / 20)
    # the bands' own gains only turn down what is no harmonic of the note, and only where the bands tell the
    # notes apart (a band a semitone wide at the bottom responds over about a second: a note-by-note gain there
    # misses most of a short note; the harmonics are rebalanced below, along the pitch, instead)
    sel_hz = cfg["min_hz"] / (2 ** (cfg["width_cents"] / 1200) - 1)
    gains = np.ones((len(centres), len(t_g)), np.float32)
    for b, c in enumerate(centres):
        if c < 250:
            continue
        k = np.where(pitched, np.round(c / np.where(pitched, p, 1.0)), 0).astype(int)
        reach = max(1200 * np.log2(1 + cfg["min_hz"] / c), cfg["width_cents"], 35.0)
        harm = pitched & (k >= 1) & (k <= K) & (np.abs(1200 * np.log2(c / np.maximum(k, 1) / np.where(pitched, p, 1.0))) < reach)
        gains[b] = np.where(harm, 1.0, np.where(pitched, other if c >= sel_hz else 1.0, unp))
    u = 2 * int(cfg["union_ms"] / 10) + 1  # note changes: the louder of both notes' gains for a moment
    gains = ndimage.uniform_filter1d(ndimage.maximum_filter1d(gains, u, axis=1), 3, axis=1)

    # the bank: every band on its own (exact, zero-phase), static and with its gains
    static = np.zeros(n)
    bal = np.zeros(n)
    t_n = np.arange(n) / fs
    for b in range(len(centres)):
        share = np.where(low_note == b, 1 - xf, 0.0) + np.where(low_note + 1 == b, xf, 0.0)
        sel = share > 0
        if not sel.any():
            continue
        Yb = np.zeros_like(X)
        Yb[sel] = X[sel] * A[sel] * share[sel]
        yb = np.fft.irfft(Yb, n)
        static += yb
        bal += yb * np.interp(t_n, t_g, gains[b]) if (gains[b] != 1).any() else yb
    # the rebalance: each harmonic followed along the pitch (time resolution ~30 ms), (gain - 1) of it added;
    # weighted by the bank's own gain at that harmonic, so a harmonic the bank rejects (the 7th) stays out
    ok = np.isfinite(f0)
    phi = 2 * np.pi * np.cumsum(np.exp(np.interp(t_n, t_f[ok], np.log(f0[ok])))) / fs
    mask = np.interp(t_n, t_f, ok.astype(float))
    lp = signal.butter(2, cfg["track_hz"], "low", fs=fs, output="sos")
    fq = np.fft.rfftfreq(n, 1 / fs)
    # around a note change (a jump of over half a semitone, or a note starting or ending) the followed harmonics
    # sweep from one note to the next: their correction came out as a chirp at 0.5-1.5 kHz; faded out there
    lp_ = np.log(np.where(pitched, p, 1.0))
    jump = np.zeros(len(t_g), bool)
    jump[1:] = (np.abs(np.diff(lp_)) > 0.5 * np.log(2) / 12) | (pitched[1:] != pitched[:-1])
    calm = 1 - ndimage.uniform_filter1d(ndimage.maximum_filter1d(jump.astype(float), 9), 5)
    for i in range(1, K + 1):
        gk = np.interp(t_n, t_g, (G[i - 1] - 1) * np.interp(i * p, fq[::64], A[::64]) * pitched * calm)
        if not np.any(np.abs(gk) > 1e-3):
            continue
        c = 2 * (signal.sosfiltfilt(lp, mono * np.cos(i * phi)) - 1j * signal.sosfiltfilt(lp, mono * np.sin(i * phi)))
        bal += gk * mask * np.real(c * np.exp(1j * i * phi))
        del c
    # the attacks: what the bank does not pass, around each note's start
    env = np.zeros(n)
    rise, hold, rel = int(0.003 * fs), int(cfg["attack_ms"] / 1000 * fs), int(cfg["release_ms"] / 1000 * fs)
    shape = np.concatenate([0.5 * (1 - np.cos(np.pi * np.arange(rise) / rise)), np.ones(hold),
                            0.5 * (1 + np.cos(np.pi * np.arange(rel) / rel))])
    for on in onsets:
        i = int(on * fs) - rise
        if 0 <= i < n:
            seg = shape[:n - i]
            env[i:i + len(seg)] = np.maximum(env[i:i + len(seg)], seg)
    band = signal.butter(2, [30.0, 8000.0], "bandpass", fs=fs, output="sos")
    attacks = signal.sosfiltfilt(band, (mono - static) * env)
    out = bal + 10 ** (cfg["attack_db"] / 20) * attacks
    stats = {"tuning_cents": round(tune, 1), "bands": len(centres),
             "band_hz": [round(float(centres[0]), 1), round(float(centres[-1]), 1)],
             "selective_from_hz": round(float(cfg["min_hz"] / (2 ** (cfg["width_cents"] / 1200) - 1)), 0),
             "pitched_pct": round(100 * float(pitched.mean()), 0), "notes": int(len(onsets)),
             "bank_kept_db": round(10 * np.log10(np.sum(static ** 2) / np.sum(mono ** 2)), 1),
             "attacks_db_re_out": round(10 * np.log10(np.sum(attacks ** 2) / np.sum(out ** 2) + 1e-20), 1)}
    if parts is not None:
        parts.update({"input": mono, "bank": static, "bank_removed": mono - static, "rebalanced": bal,
                      "attacks": attacks, "recombined": out, "gains": gains, "centres": centres, "t_g": t_g, "G": G})
    return np.column_stack([out, out]), stats
