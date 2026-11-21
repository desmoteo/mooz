"""What was heard: the verdicts of blind comparisons, kept with the project.

The take's page plays two profiles' masters of a take as A and B without saying which is
which (or a master and the mix it was made from: "pro:mix" is the mix under profile pro); the choice is written to listening.json next to the settings file, where a later look
at the settings (or another person) finds it:

  {"verdicts": [{"id": "19a6f0c2d11", "at": "2026-11-11T10:02:11", "take": "261105_225442 Miramare",
                 "a": "trial-pro", "b": "trial-base",        what was behind A and behind B
                 "choice": "trial-pro",                      a profile, or "same": no difference heard
                 "note": "the snare is less flat", "loop_s": [61.5, 74.0],
                 "masters": {"trial-pro": 1762855000.1, "trial-base": 1762854000.7}}]}

masters: when each of the two was made. A verdict is about those files: once a profile's
master of the take is made again, the verdict is shown as an earlier one and is not counted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .projects import UserError, _replace

FILE = "listening.json"
SAME = "same"
NOTE_MAX = 500
MIX = ":mix"  # after a profile's name: its mix before the master step (<take>_mix), not its master


def _when(results: dict, side: str) -> float | None:
    """When the file a side stands for was made (None: it is not there)."""
    r = results.get(side.removesuffix(MIX))
    if not r or (side.endswith(MIX) and not (r["files"].get("mix_wav") or r["files"].get("mix_mp3"))):
        return None
    return r.get("when")


def _read(cfg_path: Path) -> list[dict]:
    f = cfg_path.with_name(FILE)
    if not f.exists():
        return []
    try:
        verdicts = json.loads(f.read_text())["verdicts"]
        return [v for v in verdicts if isinstance(v, dict) and all(isinstance(v.get(k), str) for k in ("id", "take", "a", "b", "choice"))]
    except (OSError, ValueError, KeyError, TypeError):
        raise UserError(f"{f} cannot be read. Put it right or move it away; the app leaves it as it is.") from None


def _write(cfg_path: Path, verdicts: list[dict]) -> None:
    _replace(cfg_path.with_name(FILE), json.dumps({"verdicts": verdicts}, indent=1) + "\n", backup=False)


def view(cfg_path: Path, takes: list[dict]) -> dict:
    """The verdicts, newest first, each marked stale if one of its masters was made again since
    (or is gone); and per pair of profiles what the others add up to."""
    results = {t["name"]: t["results"] for t in takes}
    out, pairs = [], {}
    for v in reversed(_read(cfg_path)):
        now, was = {p: _when(results.get(v["take"], {}), p) for p in (v["a"], v["b"])}, v.get("masters") or {}
        stale = v["choice"] not in (v["a"], v["b"], SAME) or any(
            now[p] is None or abs(now[p] - (was.get(p) or 0)) > 1.0 for p in (v["a"], v["b"]))
        out.append(v | {"stale": stale})
        if stale:
            continue
        key = tuple(sorted((v["a"], v["b"])))
        pair = pairs.setdefault(key, {"profiles": list(key), "rounds": 0, "wins": {key[0]: 0, key[1]: 0}, SAME: 0, "takes": set()})
        pair["rounds"] += 1
        pair["takes"].add(v["take"])
        if v["choice"] == SAME:
            pair[SAME] += 1
        else:
            pair["wins"][v["choice"]] += 1
    return {"verdicts": out, "pairs": [p | {"takes": len(p["takes"])} for p in pairs.values()],
            "file": str(cfg_path.with_name(FILE))}


def add(cfg_path: Path, body: dict, takes: list[dict]) -> None:
    """One round's verdict. body: take, a, b (the profiles behind A and B), choice (one of them,
    or "same"), note, loop_s."""
    take = next((t for t in takes if t["name"] == body.get("take")), None)
    if take is None:
        raise UserError("That take is not in the project any more. Reload the page.")
    a, b, choice = body.get("a"), body.get("b"), body.get("choice")
    if a == b or not all(isinstance(p, str) and _when(take["results"], p) is not None for p in (a, b)):
        raise UserError("A comparison needs two profiles that both have a master of this take.")
    if choice not in (a, b, SAME):
        raise UserError("The choice is one of the two profiles, or no difference.")
    loop = body.get("loop_s")
    if loop is not None:
        if not (isinstance(loop, list) and len(loop) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in loop)
                and 0 <= loop[0] < loop[1]):
            raise UserError("A loop is two moments of the song, the first before the second.")
        loop = [round(float(x), 1) for x in loop]
    verdicts = _read(cfg_path)
    verdicts.append({"id": f"{time.time_ns() // 1_000_000:x}", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "take": take["name"],
                     "a": a, "b": b, "choice": choice, "note": str(body.get("note") or "").strip()[:NOTE_MAX], "loop_s": loop,
                     "masters": {p: _when(take["results"], p) for p in (a, b)}})
    _write(cfg_path, verdicts)


def remove(cfg_path: Path, vid: str) -> None:
    """A verdict taken back (pressed by mistake)."""
    verdicts = _read(cfg_path)
    left = [v for v in verdicts if v["id"] != vid]
    if len(left) == len(verdicts):
        raise UserError("That verdict is not there any more.")
    _write(cfg_path, left)
