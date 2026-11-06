"""Stage 0, per stem: a profile of every raw stem, measured before any clean-up.

Measured on the whole cached stem (the separation covers the whole recording),
in 85 ms frames, only where the band is playing (the stems' sum within 25 dB of
its loud level, so talk and tuning between songs do not count):

  level_db      the stem's loud level (90th percentile of the frames where it plays)
  floor_db      its noise floor (5th percentile of all frames: amp hiss, bleed, room)
  snr_db        level - floor
  active_pct    share of the playing time the stem is within 20 dB of its loud level
  tone          its average spectrum while it plays, relative to its own total level
  hf_lf_db      2-8 kHz against 60-250 Hz: bright (pick, open hats) or dark (fingers, toms)
  punch_db      how far its onsets rise above the sustained level (what the transient
                shaper reacts to): sharp hits or soft, room-smeared ones
  kick_locked_db  its level on kick hits against its average: > 0 dB = bleed of the kick,
                or playing with it (bass)

Every song's profiles are kept next to its stems (_profile.json); the median over all
songs is the target that stage 2 matches each stem to (tone match, transient amount),
so the same instrument sounds the same in every song. The measured noise floor sets
the expander threshold of the clean-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import dynamics, eq
from .loudness import EPS

NFFT = 8192
VERSION = 2
VIRTUAL = {"vocals_full": ("vocals", "vocal_room")}  # also profiled: the vocal before de-reverb


def _frames(x: np.ndarray, fs: float, grid: np.ndarray) -> np.ndarray:
    """1/3-octave band power per frame (bands x frames) of a mono signal."""
    f, _, z = signal.stft(x, fs, nperseg=NFFT, noverlap=NFFT // 2, boundary=None, padded=False)
    p = np.abs(z) ** 2
    cum = np.concatenate([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)])
    lo = np.searchsorted(f, grid * 2 ** (-1 / 6))
    hi = np.maximum(np.searchsorted(f, grid * 2 ** (1 / 6)), lo + 1)
    return (cum[hi] - cum[lo]).astype(np.float32)


def _punch(x: np.ndarray, fs: float, where: np.ndarray) -> float:
    """95th percentile of the onset rise (fast / slow envelope, dB) inside `where`."""
    a = np.abs(x)
    fast = dynamics.envelope(a, fs, 0.5, 30.0)
    slow = dynamics.envelope(a, fs, 15.0, 30.0)
    onset = 20 * np.log10((fast + EPS) / (slow + EPS))
    lev = 20 * np.log10(fast + EPS)
    keep = where & (lev > np.percentile(lev[where], 50)) if where.any() else lev > -200
    return float(np.percentile(onset[keep][::8], 95)) if keep.any() else 0.0


def measure(stems: dict[str, Path]) -> dict[str, dict]:
    """Profiles of the stem files (see the module doc)."""
    fs = sf.info(next(iter(stems.values()))).samplerate
    grid = eq.log_grid(fs)
    bands = {}
    total = None
    for name, f in stems.items():
        bands[name] = _frames(sf.read(f, dtype="float32", always_2d=True)[0].mean(axis=1), fs, grid)
        total = bands[name].sum(axis=0) if total is None else total + bands[name].sum(axis=0)
    tot_db = 10 * np.log10(total + EPS)
    band_on = tot_db > np.percentile(tot_db, 90) - 25  # the band is playing
    hop = NFFT // 2
    kick_frames = None
    if "kick" in bands:
        low = 10 * np.log10(bands["kick"][(grid >= 40) & (grid <= 150)].sum(axis=0) + EPS)
        rise = np.diff(low, prepend=low[0])
        kick_frames = band_on & (rise > 6) & (low > np.percentile(low[band_on], 99) - 20)
    for name, parts in VIRTUAL.items():  # not part of the band's total
        if all(p in stems for p in parts):
            bands[name] = _frames(sum(sf.read(stems[p], dtype="float32", always_2d=True)[0] for p in parts).mean(axis=1),
                                  fs, grid)
    out = {}
    for name, b in bands.items():
        lev = 10 * np.log10(b.sum(axis=0) + EPS)
        on = band_on & np.isfinite(lev)
        if not on.any() or np.percentile(lev[on], 90) < -80:  # silent (e.g. no "other" instrument)
            continue
        level = float(np.percentile(lev[on], 90))
        floor = float(np.percentile(lev, 5))
        plays = on & (lev > level - 20)
        if plays.sum() < 10:
            continue
        spec = b[:, plays].mean(axis=1)
        tone = 10 * np.log10(spec + EPS) - 10 * np.log10(spec.sum() + EPS)
        band_db = lambda lo, hi: float(10 * np.log10(spec[(grid >= lo) & (grid < hi)].sum() + EPS))  # noqa: E731
        p = {"level_db": round(level, 1), "floor_db": round(floor, 1), "snr_db": round(level - floor, 1),
             "active_pct": round(100 * float(plays.sum() / max(band_on.sum(), 1)), 0),
             "hf_lf_db": round(band_db(2000, 8000) - band_db(60, 250), 1)}
        if name not in ("vocal_room", "other"):
            parts = VIRTUAL.get(name, (name,))
            x = sum(sf.read(stems[q], dtype="float32", always_2d=True)[0] for q in parts).mean(axis=1)
            where = np.repeat(plays, hop)[: len(x)]
            p["punch_db"] = round(_punch(x, fs, np.pad(where, (0, len(x) - len(where)))), 1)
            del x
        if kick_frames is not None and name != "kick" and kick_frames.any():
            p["kick_locked_db"] = round(float(10 * np.log10(b[:, kick_frames].sum(axis=0).mean() + EPS)
                                              - 10 * np.log10(b[:, plays].sum(axis=0).mean() + EPS)), 1)
        p["tone"] = [round(float(v), 2) for v in tone]
        out[name] = p
    return {"version": VERSION, "fs": fs, "grid_hz": [round(float(g), 2) for g in grid], "stems": out}


def ensure(stems_dir: Path, stems: dict[str, Path] | None = None) -> dict | None:
    """This song's profiles: from _profile.json while it is newer than its stems, else measured."""
    if stems is None:
        stems = {f.stem: f for f in sorted(stems_dir.glob("*.wav")) if not f.stem.startswith("_")}
    if not stems:
        return None
    cache = stems_dir / "_profile.json"
    newest = max(f.stat().st_mtime for f in stems.values())
    if cache.exists() and cache.stat().st_mtime > newest:
        prof = json.loads(cache.read_text())
        if prof.get("version") == VERSION and set(prof["stems"]) <= set(stems) | set(VIRTUAL):
            return prof
    prof = measure(stems)
    cache.write_text(json.dumps(prof))
    return prof


def targets(stems_root: Path, fs: int) -> dict[str, dict]:
    """Per stem, the median over all songs' profiles (every song with cached stems
    under stems_root, profiled now if needed): tone, punch_db, hf_lf_db, and how many
    songs it comes from."""
    profs = []
    for d in sorted(p for p in stems_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if (d / "key.json").exists():
            prof = ensure(d)
            if prof and prof["fs"] == fs:
                profs.append(prof)
    out = {}
    for name in {n for p in profs for n in p["stems"]}:
        ps = [p["stems"][name] for p in profs if name in p["stems"]]
        t = {"songs": len(ps), "tone": np.median([p["tone"] for p in ps], axis=0)}
        for k in ("punch_db", "hf_lf_db"):
            v = [p[k] for p in ps if k in p]
            if v:
                t[k] = float(np.median(v))
        out[name] = t
    return out


REFERENCES = Path(__file__).with_name("references") / "raw_multitracks.json"
REFERENCE_OF = {"kick": "kick", "snare": "snare", "toms": "toms", "hihat": "hihat", "cymbals": "cymbals",
                "bass": "bass", "guitar": "guitar", "guitar_left": "guitar", "guitar_right": "guitar", "vocals": "vocals"}
_refs: dict | None = None


def reference_tone(name: str, grid: np.ndarray) -> np.ndarray | None:
    """The median tone of raw, unprocessed studio tracks of this instrument (built by
    tools/build_raw_references.py from the Cambridge-MT multitracks), on grid."""
    global _refs
    if _refs is None:
        _refs = json.loads(REFERENCES.read_text()) if REFERENCES.exists() else {"instruments": {}}
    r = _refs["instruments"].get(REFERENCE_OF.get(name, ""))
    if r is None:
        return None
    return np.interp(np.log2(grid), np.log2(np.asarray(_refs["grid_hz"])), np.asarray(r["tone"]))


def tone_match_curve(grid: np.ndarray, measured: list[float], target: np.ndarray, cfg: dict) -> np.ndarray:
    """EQ curve (dB on grid) that moves the measured tone part of the way towards the
    target: broad (smooth_oct), limited to max_boost_db / max_cut_db, only inside fmin..fmax, and with
    no overall level change."""
    m = np.asarray(measured)
    diff = np.asarray(target) - m
    inside = (grid >= cfg["fmin_hz"]) & (grid <= cfg["fmax_hz"])
    first, last = np.flatnonzero(inside)[[0, -1]]
    diff = np.concatenate([np.full(first, diff[first]), diff[first : last + 1], np.full(len(diff) - last - 1, diff[last])])
    g = eq.smooth(diff, cfg["smooth_oct"])
    g = np.where(inside, g, 0.0)
    g = eq.smooth(g, 1 / 3)  # soft edges
    g = (cfg["strength"] * g).clip(-cfg["max_cut_db"], cfg["max_boost_db"])
    w = 10 ** (m / 10)  # no level change: the stem's power stays the same
    return g - 10 * np.log10(np.sum(w * 10 ** (g / 10)) / np.sum(w))


def summary(p: dict) -> dict:
    """A profile without its spectrum, for the reports."""
    return {k: v for k, v in p.items() if k != "tone"}
