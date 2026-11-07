"""Close tracks (Tr<n>) brought into the room's stems, before any chain.

A close track is separated like the room (plan "close"), but only the instruments in [multitrack]
stems are used, and only if it really holds them: a stem less than min_level_lu under its track's
loudest stem is bleed, and stays out. What stays is lined up with the same instrument's room stem,
whole-song and sample-accurate: the room hears a mixer's feed through an amp or a PA, several
metres and sometimes the opposite polarity away (on the band's recordings the voice 10 ms later and
inverted, the bass 11 ms later), and added as they come the two would cancel. It is then set by
loudness against the room stem, close_share of the two parts' loudness, and added to it.

The result is one stem per instrument, with the room's name, so all that follows (profiles,
triggers, chains, balance) works on it as on any stem. The room's own stems stay what the original
recording is made of: the residual and the balance's estimate of the original's share use those.
Merged stems are whole-take and kept in a cache of their own, one folder per set of merge
settings, so that their tones can be the targets of the songs' tone match.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import scipy.fft
import soundfile as sf

from . import cleanup, loudness
from .takes import Take

VERSION = 1


def _read_mono(f: Path) -> np.ndarray:
    return sf.read(f, dtype="float32", always_2d=True)[0].mean(axis=1)


def delay_to_room(room: np.ndarray, close: np.ndarray, fs: float, lo: float, hi: float, max_ms: float
                  ) -> tuple[int, int, float]:
    """How far the close signal must be delayed to line up with the room's (samples, may be
    negative), with which polarity (+1, -1), and how clearly (the correlation peak over its
    surroundings): phase-only cross-correlation in lo..hi, which does not depend on the spectra."""
    n = scipy.fft.next_fast_len(len(room) + len(close), real=True)
    R, C = scipy.fft.rfft(room, n), scipy.fft.rfft(close, n)
    X = R * np.conj(C)
    del R, C
    X /= np.abs(X) + 1e-20
    f = np.fft.rfftfreq(n, 1 / fs)
    X[(f < lo) | (f > hi)] = 0
    c = np.fft.fftshift(scipy.fft.irfft(X, n))
    del X
    lags = np.arange(len(c)) - len(c) // 2
    near = np.abs(lags) <= int(max_ms / 1000 * fs)
    c, lags = c[near], lags[near]
    i = int(np.argmax(np.abs(c)))
    return int(lags[i]), int(np.sign(c[i])), float(np.abs(c[i]) / (np.median(np.abs(c)) + 1e-20))


def _steady(room: np.ndarray, close: np.ndarray, fs: float, lo: float, hi: float, max_ms: float, d: int, sign: int,
            n_win: int = 12, win_s: float = 15.0) -> int | None:
    """In what share (%) of the windows where the close stem plays the same delay and polarity are
    found, within 1 ms. None if it hardly plays. Bleed gives a different answer every time."""
    w = int(win_s * fs)
    if len(close) < w:
        return None
    level = np.array([np.mean(close[k : k + w] ** 2) for k in range(0, len(close) - w + 1, w // 2)]) + 1e-20
    loud = np.max(level) * 10 ** (-20 / 10)
    starts = [k * (w // 2) for k in np.flatnonzero(level >= loud)]
    if not starts:
        return None
    picks = [starts[i] for i in np.linspace(0, len(starts) - 1, min(n_win, len(starts))).astype(int)]
    same = 0
    for k in picks:
        dk, sk, _ = delay_to_room(room[k : k + w], close[k : k + w], fs, lo, hi, max_ms)
        same += abs(dk - d) <= fs / 1000 and sk == sign
    return round(100 * same / len(picks))


def shifted(x: np.ndarray, d: int, n: int) -> np.ndarray:
    """x delayed by d samples (or advanced, d < 0), cut or padded to n samples."""
    if d >= 0:
        y = np.concatenate([np.zeros((d, *x.shape[1:]), x.dtype), x])
    else:
        y = x[-d:]
    return np.pad(y[:n], ((0, max(0, n - len(y))), *((0, 0) for _ in x.shape[1:])))


def choose(levels: dict[str, dict[str, float]], stems: list[str], cfg: dict) -> dict[str, list[str]]:
    """Which close tracks each instrument is taken from. levels: per track, each stem's loudness.
    A stem counts if it is within min_level_lu of its track's loudest (else it is bleed), and, when
    several tracks hold it, if it stands within keep_within_db of where it stands strongest
    (two microphones on two guitars both stay; a vocal mic's bleed of the bass amp does not)."""
    rel = {tr: {s: lv[s] - max(v for v in lv.values() if np.isfinite(v)) for s in stems if s in lv and np.isfinite(lv[s])}
           for tr, lv in levels.items() if any(np.isfinite(v) for v in lv.values())}
    out = {}
    for s in stems:
        have = {tr: r[s] for tr, r in rel.items() if s in r and r[s] >= cfg["min_level_lu"]}
        if have:
            best = max(have.values())
            out[s] = [tr for tr, v in have.items() if v >= best - cfg["keep_within_db"]]
    return out


def _settings_id(cfg: dict) -> str:
    keep = {"multitrack": {k: v for k, v in cfg["multitrack"].items() if k != "room"},
            "vocal_room": cfg["stems"]["vocal_room"], "version": VERSION}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:8]


def merge(take: Take, room_stems: dict[str, Path], close_stems: dict[str, dict[str, Path]], fs: int, cfg: dict,
          cache_root: Path) -> tuple[dict[str, Path], dict[str, Path], Path, dict]:
    """The take's stems with its close tracks in them. Returns (stems, the room's own stems under
    the same names, the folder they are in, what was done)."""
    mcfg = cfg["multitrack"]
    out = cache_root / _settings_id(cfg) / take.name
    src = lambda d: json.loads((d / "key.json").read_text())  # noqa: E731
    key = {"room": src(next(iter(room_stems.values())).parent),
           "close": {tr: src(next(iter(s.values())).parent) for tr, s in close_stems.items()}}
    meta = out / "key.json"
    names = [n for n in room_stems if n != "vocal_room"]
    room_of = {n: out / f"_room_{n}.wav" if n == "vocals" and "vocal_room" in room_stems else room_stems[n] for n in names}
    if meta.exists() and json.loads(meta.read_text()) == key and (out / "_merge.json").exists():
        return {n: out / f"{n}.wav" for n in names}, room_of, out, json.loads((out / "_merge.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    meta.unlink(missing_ok=True)
    report: dict = {}
    if "vocals" in room_stems and "vocal_room" in room_stems:  # the room's vocal as ever: dry + room, the room turned down
        report["vocal_room"] = cleanup.recombine_room(room_stems["vocals"], room_stems["vocal_room"], room_of["vocals"],
                                                      cfg["stems"]["vocal_room"])
    read2 = lambda f: np.repeat(sf.read(f, dtype="float32", always_2d=True)[0], 2, axis=1)  # noqa: E731  (mono, as stereo)
    levels = {tr: {s: loudness.integrated(read2(f), fs) for s, f in st.items()} for tr, st in close_stems.items()}
    use = choose(levels, mcfg["stems"], mcfg)
    report["close"] = {tr: {s: {"level_lu": round(lv[s] - max(v for v in lv.values() if np.isfinite(v)), 1) if np.isfinite(lv[s]) else None,
                            "used": tr in use.get(s, [])} for s in mcfg["stems"] if s in lv} for tr, lv in levels.items()}
    for f in out.glob("*.wav"):
        if f.is_symlink() or not f.stem.startswith("_room_"):
            f.unlink()
    for n in names:
        dst = out / f"{n}.wav"
        if n not in use:  # the room's stem as it is
            dst.symlink_to(os.path.relpath(room_of[n], out))
            continue
        room = sf.read(room_of[n], dtype="float32", always_2d=True)[0]
        room_mono = room.mean(axis=1)
        sum_close = np.zeros_like(room)
        lo, hi = mcfg["align_band_hz"][n]
        for tr in use[n]:
            c = sf.read(close_stems[tr][n], dtype="float32", always_2d=True)[0].mean(axis=1)
            d, sign, peak = delay_to_room(room_mono, c, fs, lo, hi, mcfg["max_delay_ms"])
            steady = _steady(room_mono, c, fs, lo, hi, mcfg["max_delay_ms"], d, sign)
            info = report["close"][tr][n]
            info |= {"delay_ms": round(1000 * d / fs, 2), "polarity": sign, "peak": round(peak, 1), "steady_pct": steady}
            if peak < mcfg["min_peak"]:  # no clear answer: nothing to line up with, so nothing to cancel either
                d, sign = 0, 1
                info["note"] = "not lined up: no clear delay"
            sum_close += (sign * shifted(c, d, len(room)))[:, None]
        lr, lc = loudness.integrated(room, fs), loudness.integrated(sum_close, fs)
        share = mcfg["close_share"][n]
        if np.isfinite(lr) and np.isfinite(lc):
            gain = 10 ** ((lr + 10 * np.log10(share / (1 - share)) - lc) / 20)
        else:  # the room hardly hears it: the close stem alone, as loud as the room's whole recording could not tell
            gain = 1.0
        merged = room + gain * sum_close
        sf.write(dst, merged, fs, subtype="FLOAT")
        report.setdefault("merged", {})[n] = {"tracks": use[n], "close_share": share, "gain_db": round(20 * np.log10(gain), 1),
                                              "lufs": round(loudness.integrated(merged, fs), 1)}
    stems = {n: out / f"{n}.wav" for n in names}
    (out / "_merge.json").write_text(json.dumps(report, indent=1, default=float))
    meta.write_text(json.dumps(key))
    return stems, room_of, out, report
