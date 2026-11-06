"""Space and grit, added as sends: plate / room reverb, slapback, tape echo, stereo
delay, chorus; a band-limited parallel distortion ("grit") and a dual (split-band) chain
for the bass.

Every send is set by measurement, not by a fader position: its loudness is put at
level_lu relative to the dry track (integrated loudness), so the same setting gives the
same amount of space on a loud song and a quiet one. De Man (2017) found the total
reverb in good mixes around -14 LU re the mix, and -9 LU already too wet.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import loudness
from .loudness import EPS, filtfilt


def _bands(x: np.ndarray, fs: float, hp: float, lp: float) -> np.ndarray:
    if hp > 0:
        x = filtfilt(signal.butter(2, hp, "high", fs=fs, output="sos"), x)
    if lp < fs / 2:
        x = filtfilt(signal.butter(2, lp, "low", fs=fs, output="sos"), x)
    return x


def reverb_ir(fs: float, rt60_s: float, predelay_ms: float, damping: float, seed: int = 3) -> np.ndarray:
    """A synthetic stereo impulse response: decorrelated noise decaying at rt60 (faster
    in the highs: damping 0..1), after predelay. Plates and small rooms are both
    well modelled by dense, exponentially decaying noise."""
    rng = np.random.default_rng(seed)
    n = int((rt60_s * 1.2 + predelay_ms / 1000) * fs)
    t = np.arange(n) / fs
    ir = np.zeros((n, 2))
    edges = (0.0, 1000.0, 4000.0, fs / 2 * 0.99)
    for lo, hi, k in zip(edges, edges[1:], (0.0, 0.5, 1.0)):
        rt = rt60_s * (1 - damping * 0.6 * k)  # highs die sooner
        noise = rng.normal(0, 1, (n, 2))
        sos = (signal.butter(4, hi, "low", fs=fs, output="sos") if lo == 0
               else signal.butter(4, [lo, hi], "bandpass", fs=fs, output="sos"))
        ir += signal.sosfilt(sos, noise, axis=0) * np.exp(-6.9 * t / rt)[:, None]
    d = int(predelay_ms / 1000 * fs)
    ir = np.concatenate([np.zeros((d, 2)), ir])[:n]
    fade = np.ones(n)
    fade[-int(0.05 * fs):] = np.linspace(1, 0, int(0.05 * fs))
    return ir * fade[:, None] / np.sqrt(np.sum(ir ** 2) / 2)


def _wow(x: np.ndarray, fs: float, depth_ms: float, rate_hz: float, seed: int) -> np.ndarray:
    """Slowly varying delay (tape wow/flutter) by interpolated reading."""
    if depth_ms <= 0:
        return x
    rng = np.random.default_rng(seed)
    t = np.arange(len(x)) / fs
    mod = depth_ms / 1000 * fs * (np.sin(2 * np.pi * rate_hz * t + rng.uniform(0, 6.3))
                                  + 0.3 * np.sin(2 * np.pi * 5.1 * rate_hz * t))
    pos = np.clip(np.arange(len(x)) - depth_ms / 1000 * fs * 1.3 + mod, 0, len(x) - 1)
    return np.column_stack([np.interp(pos, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])])


def _delay(x: np.ndarray, d: int) -> np.ndarray:
    return np.concatenate([np.zeros((d, x.shape[1])), x[: len(x) - d]]) if d > 0 else x


def delay_ms_of(cfg: dict, tempo_bpm: float | None) -> float:
    """delay_beats at the song's measured tempo (clamped to 120..900 ms), else delay_ms."""
    if cfg.get("delay_beats") and tempo_bpm:
        return float(np.clip(60000.0 / tempo_bpm * cfg["delay_beats"], 120.0, 900.0))
    return float(cfg["delay_ms"])


def send(x: np.ndarray, fs: float, cfg: dict, tempo_bpm: float | None = None) -> tuple[np.ndarray, dict]:
    """The wet signal of one send (type plate, room, slap, tape_echo, delay or chorus), at
    level_lu re x. tempo_bpm: the song's, for delays given in beats."""
    kind = cfg["type"]
    src = _bands(x, fs, cfg.get("hp_hz", 200.0), cfg.get("lp_hz", 8000.0))
    if kind in ("plate", "room"):
        ir = reverb_ir(fs, cfg["rt60_s"], cfg.get("predelay_ms", 10.0), cfg.get("damping", 0.5))
        wet = np.column_stack([signal.oaconvolve(src[:, 0], ir[:, 0])[: len(x)],
                               signal.oaconvolve(src[:, 1 % src.shape[1]], ir[:, 1])[: len(x)]])
    elif kind == "slap":
        d = int(cfg["delay_ms"] / 1000 * fs)
        wet = _delay(src, d)
        if wet.shape[1] == 2 and cfg.get("spread_ms", 0):  # a little wider than the dry voice
            wet[:, 1] = _delay(wet[:, [1]], int(cfg["spread_ms"] / 1000 * fs))[:, 0]
    elif kind == "tape_echo":
        # repeats of a tape loop: each one delayed, darker, a little saturated and wobbling
        d = int(cfg["delay_ms"] / 1000 * fs)
        rep, wet = src, np.zeros_like(src)
        for k in range(cfg.get("repeats", 4)):
            rep = _delay(rep, d) * cfg["feedback"] ** (1 if k else 0)
            rep = _bands(rep, fs, cfg.get("hp_hz", 300.0), cfg.get("lp_hz", 3500.0))
            rep = _wow(rep, fs, cfg.get("wow_ms", 0.6), cfg.get("wow_hz", 0.7), seed=k)
            g = 10 ** (cfg.get("drive_db", 6.0) / 20) / (np.sqrt(np.mean(rep ** 2)) + EPS) * 0.3
            rep = np.tanh(rep * g) / g
            wet = wet + rep
    elif kind == "delay":
        # clean digital delay: repeats at the song's tempo, each a little darker, bouncing
        # left and right (ping_pong) or in place
        d = int(delay_ms_of(cfg, tempo_bpm) / 1000 * fs)
        mono = src.mean(axis=1, keepdims=True)
        rep, wet = mono, np.zeros((len(x), 2))
        for k in range(cfg.get("repeats", 5)):
            rep = _delay(rep, d) * (cfg["feedback"] if k else 1.0)
            if k:
                rep = _bands(rep, fs, cfg.get("hp_hz", 200.0), cfg.get("lp_hz", 6000.0))
            if cfg.get("ping_pong", True):
                side = (1.0, 0.25) if k % 2 == 0 else (0.25, 1.0)
            else:
                side = (1.0, 1.0)
            wet += rep * np.array(side)[None, :]
    elif kind == "chorus":
        # micro-pitch doubling: two copies on slowly modulated delays (a few cents up and down),
        # one hard left, one hard right, under the dry voice: wider and bigger, not further away
        mono = src.mean(axis=1, keepdims=True)
        voices = []
        for k, (d_ms, rate) in enumerate(((cfg["delay_ms"], cfg["rate_hz"]), (cfg["delay_ms"] * 1.4, cfg["rate_hz"] * 1.3))):
            v = _delay(mono, int(d_ms / 1000 * fs))
            voices.append(_wow(v, fs, cfg["depth_ms"], rate, seed=10 + k)[:, 0])
        wet = np.column_stack(voices)
    else:
        raise ValueError(f"unknown send type '{kind}' (plate, room, slap, tape_echo, chorus)")
    lx, lw = loudness.integrated(x, fs), loudness.integrated(wet, fs)
    if not (np.isfinite(lx) and np.isfinite(lw)):
        return np.zeros_like(x), {"type": kind}
    gain = 10 ** ((lx + cfg["level_lu"] - lw) / 20)
    st = {"type": kind, "level_lu": cfg["level_lu"]}
    if kind in ("delay", "tape_echo", "slap"):
        st["delay_ms"] = round(delay_ms_of(cfg, tempo_bpm) if kind == "delay" else cfg["delay_ms"], 0)
    return wet * gain, st


def sends(x: np.ndarray, fs: float, cfgs: list[dict], tempo_bpm: float | None = None
          ) -> tuple[np.ndarray | None, list[dict]]:
    """All of a track's sends, summed (None if it has none)."""
    out, stats = None, []
    for c in cfgs:
        if not c.get("enabled", True):
            continue
        w, st = send(x, fs, c, tempo_bpm)
        out = w if out is None else out + w
        stats.append(st)
    return out, stats


def grit(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, dict]:
    """Parallel distortion in a band (bass "teeth"): the track driven hard into a tanh,
    band-passed to band_hz, blended level_db under the track (by RMS)."""
    rms = float(np.sqrt(np.mean(x * x))) + EPS
    g = 10 ** (cfg["drive_db"] / 20) * 0.3 / rms
    dist = np.tanh(x * g)
    dist = filtfilt(signal.butter(4, list(cfg["band_hz"]), "bandpass", fs=fs, output="sos"), dist)
    dist *= rms * 10 ** (cfg["level_db"] / 20) / (float(np.sqrt(np.mean(dist * dist))) + EPS)
    return x + dist, {"drive_db": cfg["drive_db"], "level_db": cfg["level_db"]}


def dual(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, dict]:
    """Dual (split-band, "bi-amp") bass chain. Below crossover_hz the bass stays clean,
    mono and steady (its own compressor); above it, the part that carries the notes'
    character goes through drive and a stereo chorus, then is set high_gain_db against
    its clean self. The two parts sum back to the input exactly when every effect is off."""
    from . import dynamics

    low, high = dynamics.split_bands(x, fs, [cfg["crossover_hz"]])
    low = np.repeat(low.mean(axis=1, keepdims=True), x.shape[1], axis=1)  # the foundation: mono
    st: dict = {"crossover_hz": cfg["crossover_hz"]}
    lc = cfg["low_compressor"]
    if lc["enabled"]:
        low, _, st["low_compressor"] = dynamics.compress(low, fs, **{k: lc[k] for k in dynamics.BAND_PARAMS},
                                                          active=loudness.active_mask(x, fs), auto_makeup=True)
    rms = float(np.sqrt(np.mean(high * high))) + EPS
    if cfg["drive_mix"] > 0:  # amp-like drive on the upper part only: growl without flabby lows
        g = 10 ** (cfg["drive_db"] / 20) * 0.3 / rms
        driven = np.tanh(high * g + 0.15) - np.tanh(0.15)  # a little asymmetric: even harmonics too
        driven = filtfilt(signal.butter(2, cfg["crossover_hz"], "high", fs=fs, output="sos"), driven)
        driven *= rms / (float(np.sqrt(np.mean(driven * driven))) + EPS)
        high = (1 - cfg["drive_mix"]) * high + cfg["drive_mix"] * driven
    if cfg["chorus_mix"] > 0:  # stereo chorus on the upper part: two modulated copies, left and right
        mono = high.mean(axis=1, keepdims=True)
        v = [_wow(_delay(mono, int(d / 1000 * fs)), fs, cfg["chorus_depth_ms"], r, seed=20 + k)[:, 0]
             for k, (d, r) in enumerate(((7.0, cfg["chorus_rate_hz"]), (9.5, cfg["chorus_rate_hz"] * 1.25)))]
        wet = np.column_stack(v) if x.shape[1] == 2 else np.mean(v, axis=0)[:, None]
        high = high + cfg["chorus_mix"] * wet
    high *= 10 ** (cfg["high_gain_db"] / 20)
    return low + high, st | {"drive_mix": cfg["drive_mix"], "chorus_mix": cfg["chorus_mix"],
                             "high_gain_db": cfg["high_gain_db"]}
