"""Where a run is, for a front end (the web app, gui/).

With REHEARSAL_PROGRESS set, events go to standard output, one JSON object per line behind
PREFIX; without it nothing is written. The environment carries the switch because a run is
several processes: the command line re-launches itself in a memory-capped scope, and the
separation is a child of that.

  {"ev": "plan", "units": [{"take": ..., "preset": ...}, ...]}
  {"ev": "prepare", "take": ..., "i": 0, "n": 8}
  {"ev": "separation", "track": ..., "state": "cached" | "start" | "step" | "done", "i": 2, "n": 9, "model": ...}
  {"ev": "unit", "i": 0}
  {"ev": "stage", "key": "stems", "label": "Stem chains", "frac": 0.41, "detail": "bass"}
  {"ev": "unit_done", "i": 0, "ok": true, "seconds": 251.0, "summary": {...}}
  {"ev": "end", "failed": [...]}
"""

from __future__ import annotations

import json
import os
import sys

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
        emit("stage", key=key, label=label, frac=round(fraction(key, part), 4), detail=detail)
