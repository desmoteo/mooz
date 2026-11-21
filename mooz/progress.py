"""Where a run is, for a front end (the web app, gui/).

With REHEARSAL_PROGRESS set, events go to standard output, one JSON object per line behind
PREFIX; without it nothing is written. The environment carries the switch because a run is
several processes: the command line re-launches itself in a memory-capped scope, and the
separation and the stem chains' workers are children of that. "frac" is of the whole take,
"part" of the stage itself. A "chain" event is one stem's chain (several are made at once):
its clean-up, its studio chain, its place in the mix, done; "kept" as an earlier run left it.

  {"ev": "plan", "units": [{"take": ..., "preset": ...}, ...]}
  {"ev": "prepare", "take": ..., "i": 0, "n": 8}
  {"ev": "separation", "track": ..., "state": "cached" | "start" | "step" | "done", "i": 2, "n": 9, "model": ...}
  {"ev": "unit", "i": 0}
  {"ev": "stage", "key": "stems", "label": "Stem chains", "frac": 0.41, "part": 0.5, "detail": "bass"}
  {"ev": "chain", "stem": "bass", "state": "clean" | "chain" | "mix" | "done" | "kept" | "silent", "frac": 0.3}
  {"ev": "unit_done", "i": 0, "ok": true, "seconds": 251.0, "summary": {...}}
  {"ev": "end", "failed": [...]}
"""

from __future__ import annotations

import json
import os
import sys
import time

PREFIX = "@@"

# One take's stages in order: key, name shown, share of the take's time (%). Measured on
# Miramare of 5 November 2026 (4 minutes, a close track, stems and profiles cached, A/B files:
# 259 s in all, of which the stem chains 168 s, the mix bus 25 s, the character 22 s). A
# separation is not in them: it reports its own steps ("separation" events).
STAGES: list[tuple[str, str, float]] = [
    ("read", "Reading the recording", 1),
    ("prepare", "Trim, rumble and hum", 1),
    ("separate", "Stems", 1),
    ("profile", "Stem profiles", 1),
    ("character", "Character", 8),
    ("stems", "Stem chains", 64),
    ("drum_bus", "Drum bus", 2),
    ("blend", "Blend and balance", 4),
    ("bus", "Mix bus", 10),
    ("master", "Loudness and limiter", 4),
    ("write", "Writing files", 3),
    ("report", "Reports", 1),
]
_TOTAL = sum(w for _, _, w in STAGES)


def enabled() -> bool:
    return bool(os.environ.get("REHEARSAL_PROGRESS"))


def emit(ev: str, **data) -> None:
    if enabled():
        sys.stdout.write(PREFIX + json.dumps({"ev": ev, **data}, default=str) + "\n")
        sys.stdout.flush()


def fraction(key: str, part: float = 0.0) -> float:
    """How much of a take is done when stage key is part (0..1) of the way through."""
    done = 0.0
    for k, _, w in STAGES:
        if k == key:
            return (done + min(max(part, 0.0), 1.0) * w) / _TOTAL
        done += w
    raise KeyError(f"unknown stage '{key}'")


def stage(key: str, part: float = 0.0, detail: str = "") -> None:
    """The take has reached stage key (part of the way through it)."""
    if enabled():
        label = next(name for k, name, _ in STAGES if k == key)
        emit("stage", key=key, label=label, frac=round(fraction(key, part), 4), part=round(min(max(part, 0.0), 1.0), 3), detail=detail)


_phase: list | None = None  # the phase of the chain this process is making: stem, state, from, to, ticks, ticks so far, last said


def chain(stem: str, state: str, frac: float = 0.0, to: float | None = None, ticks: int = 0) -> None:
    """Where one stem's chain is (from whichever process is making it). to and ticks: this phase
    runs from frac to `to` of the chain in that many tick()s, each of which says how far it is."""
    global _phase
    _phase = [stem, state, frac, to, ticks, 0, time.monotonic()] if enabled() and to is not None and ticks > 0 else None
    emit("chain", stem=stem, state=state, frac=frac)


def tick() -> None:
    """One more processor of the chain's phase is through (effects.py calls it after each, switched
    on or not). Said only after one that took a while: a processor that is off is no news."""
    p = _phase
    if p is None:
        return
    p[5] += 1
    now = time.monotonic()
    if now - p[6] > 0.25 and p[5] <= p[4]:
        p[6] = now
        emit("chain", stem=p[0], state=p[1], frac=round(p[2] + (p[3] - p[2]) * p[5] / p[4], 3))
