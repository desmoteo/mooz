"""The take being made, on the take's graph: every box of it waiting, at work or done.

The graph of a take's page (stagemap.py) is drawn from the files a run left. A take that is being
made for the first time has none yet, so here the same graph is drawn from what the run will make:
the take's tracks, the stems its separation plans give (or gave, where a cache has them), the stems
the close tracks go into, the kit, the mix and the master. It is the same function that lays it out
(stagemap._graph), given the files' names in place of the files, so a box has the name it will
have on the take's page.

states() then reads the run's progress events as the job holds them (jobs.py) and says where each
box is: "active" with how far (a separation by its models, a stem's chain by its phases, the bus
by its steps), "done", "kept" (as an earlier run left it), "skipped" (a stem with nothing in it),
"stopped" (where a run that failed was). A box it does not name is waiting.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from .. import config, progress
from . import projects, stagemap
from .projects import UserError

STAGES = [k for k, _, _ in progress.STAGES]  # read, prepare, separate, profile, character, stems, drum_bus, blend, bus, master, write, report
SEPARATION_LAYERS = 3  # the layers a run that only separates gets to: recording, separation, stems, merged


CLEAN_SHARE = 0.3  # of a chain's progress, the part that is its clean-up (pipeline._chain says so)


def _outputs(scfg: dict) -> set[str]:
    """The stems a separation plan ends with, worked out from its steps (separate._worker_roformer)."""
    from .. import separate  # (kept out of the app's start)

    if scfg["backend"] != "roformer":
        return {"vocals", "drums", "bass", "other"}
    have = {"mix"}
    for step in separate.model_steps(scfg):
        if "combine" in step:
            if any(n in have for n in step["inputs"]):
                have -= set(step["inputs"])
                have.add(step["output"])
        elif step["input"] in have:
            if step["input"] != "mix":
                have.discard(step["input"])
            for m in separate.members(step):
                have |= {v for v in m["outputs"].values() if v}
    return {n for n in have if n != "mix" and not n.startswith("_")}


def plan(cfg_path: Path, take_name: str, profile: str, separate_only: bool = False) -> dict:
    """The graph of what a run of this take makes: stagemap's layers, boxes and arrows, no box to play."""
    from .. import multitrack, separate

    take = next((t for t in projects.find_takes(cfg_path) if t.name == take_name), None)
    if take is None:
        raise UserError(f"There is no take '{take_name}' in this project.")
    try:
        cfg = config.load(cfg_path, None if profile in (projects.DEFAULT, "") else profile)
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        raise UserError(projects._message(e)) from None
    scfg = cfg["stems"]
    out = cfg["output"]["stems_dir"]

    def stems_of(folder: Path, plan_cfg: dict) -> set[str]:  # as a separation left them, or as its plan says
        return set(stagemap._stems(folder)) if (folder / "key.json").exists() else _outputs(plan_cfg)

    raw = stems_of(take.root / (out + separate.cache_name(scfg)) / take.name, scfg) if scfg["enabled"] else set()
    close_cfg = scfg | {"plan": cfg["multitrack"]["plan"], "backend": "roformer"}
    close = {tr: stems_of(take.root / (out + separate.cache_name(close_cfg)) / f"{take.name} {tr}", close_cfg) for tr in take.close}
    report: dict = {}
    merged: set[str] = set()
    if take.close and raw:
        try:  # what the merge said of it, if it ran with these settings
            word = take.root / (out + separate.cache_name(scfg) + "_merged") / multitrack._settings_id(cfg) / take.name / "_merge.json"
            report = {"multitrack": json.loads(word.read_text())}
            merged = set(report["multitrack"].get("merged") or {})
        except (OSError, json.JSONDecodeError, KeyError):
            merged = {s for s in cfg["multitrack"]["stems"] if s in raw and any(s in c for c in close.values())}
    ps = scfg["pan_split"]
    if ps["enabled"] and ps["input"] in raw:  # one stem as two, by where the amps stood
        raw = (raw - {ps["input"]}) | {n for n in ps["outputs"].values() if n}
    kit_names = set(cfg["drum_bus"]["stems"]) if cfg["drum_bus"]["enabled"] else set()
    names = sorted(raw, key=lambda n: (n in kit_names, stagemap.ORDER.index(n) if n in stagemap.ORDER else len(stagemap.ORDER), n))
    kit = [n for n in names if n in kit_names]
    no_chain = {"vocal_room"} if scfg["vocal_room"]["recombine"] and {"vocals", "vocal_room"} <= raw else set()
    original = cfg["mix"]["original_share"] > 0 and bool(names)
    rows = ([{"key": "original", "label": "recording", "kit": False}] if original else []) + [
        {"key": n, "label": n.replace("_", " "), "kit": n in kit} for n in names]
    ids: list[tuple[str, str]] = [("recording/room", take.room.stem.rsplit("_", 1)[-1] if take.close or take.mixdown else "recording")]
    ids += [(f"recording/{tr}", tr) for tr in take.close]
    if take.mixdown is not None:
        ids.append(("recording/mixdown", take.mixdown.stem.rsplit("_", 1)[-1]))
    for n in names:
        ids.append((f"separated/{n}", n))
        if n in merged:
            ids.append((f"separated/{n}/room", "room"))
        ids += [(f"separated/{n}/{tr}", tr) for tr in close if n in close[tr]]
        if n in kit:
            ids.append((f"cleaned/{n}", n))
        if n not in no_chain:
            ids.append((f"chain/{n}", n))
            if n not in kit:
                ids.append((f"premix/{n}", n))
    for tr, files in close.items():  # a close track's stems that are no stem of the room's: boxes of their own all the same
        ids += [(f"separated/{n}/{tr}", n) for n in files if n not in names]
    if kit:
        ids += [("drum_bus/drum_bus", "drum bus"), ("premix/drum_bus", "drum bus")]
    if original:
        ids.append(("premix/original", "recording"))
    if names:
        ids += [("mix/mix", "mix"), ("master/master", "master")]
    nodes = [{"id": i, "col": i.split("/", 1)[0], "label": label} for i, label in ids]
    g = stagemap._graph(take, rows, names, kit, close, merged, nodes, [], report)
    for b in g["boxes"]:
        b["play"] = None
    if separate_only:
        keep = {b["id"] for b in g["boxes"] if b["layer"] <= SEPARATION_LAYERS}
        g = {"layers": g["layers"][: SEPARATION_LAYERS + 1], "boxes": [b for b in g["boxes"] if b["id"] in keep],
             "arrows": [a for a in g["arrows"] if a["from"] in keep and a["to"] in keep]}
    return g


def states(graph: dict, take_name: str, unit: dict | None, separated: dict) -> dict[str, dict]:
    """Where each box of the take's graph is, from the job's view of the run: unit, the take's unit
    if it has begun (jobs.Job._unit), separated, every track's separation so far by its cache's name."""
    boxes = {b["id"]: b for b in graph["boxes"]}
    out: dict[str, dict] = {}

    def put(box: str, state: str, frac: float | None = None, note: str = "") -> None:
        if box in boxes and out.get(box, {}).get("state") != "skipped":
            out[box] = {"state": state} | ({"frac": round(min(max(frac, 0.0), 1.0), 3)} if frac is not None else {}) | ({"note": note} if note else {})

    key = (unit or {}).get("key") or ""
    at = STAGES.index(key) if key in STAGES else -1
    over = bool(unit) and unit["state"] == "done"
    for b in boxes:
        if b.startswith("src/"):
            put(b, "done", note="recorded")
    # the separations, and the stems each one gives
    through = []
    for b in [b for b in boxes if b.startswith("sep/")]:
        tr = b.split("/", 1)[1]
        mine = [x for x in boxes if x.startswith("stem/")] if tr == "room" else [x for x in boxes if x.startswith(f"close/{tr}/")]
        s = separated.get(take_name if tr == "room" else f"{take_name} {tr}")
        if s and s["state"] in ("start", "step"):
            n = s.get("n") or 0
            put(b, "active", (s.get("i", 0) / n) if n else 0.0, f"model {s.get('i', 0) + 1} of {n}: {s.get('model', '')}" if n else "starting")
            through.append(False)
            continue
        if s or at >= STAGES.index("profile") or over:
            kept = bool(s) and s["state"] == "cached"
            put(b, "kept" if kept else "done", note="as separated before" if kept else "")
            for x in mine:
                put(x, "kept" if kept else "done")
            through.append(True)
        else:
            through.append(False)
    if (through and all(through)) or at >= STAGES.index("profile") or over:
        for b in boxes:
            if b.startswith("merged/"):
                put(b, "done")
    # the stems' chains: each where its own events say
    for stem, c in ((unit or {}).get("chains") or {}).items():
        clean, chain = f"cleaned/{stem}", f"chain/{stem}"
        phase, frac = c.get("state"), float(c.get("frac") or 0.0)
        if phase == "silent":
            for b in (clean, chain, f"premix/{stem}"):
                if b in boxes:
                    out[b] = {"state": "skipped", "note": "nothing in it"}
        elif phase in ("kept", "done"):
            for b in (clean, chain):
                put(b, phase, note="as an earlier run left it" if phase == "kept" else "")
        elif phase == "clean":  # (the clean-up is the first CLEAN_SHARE of a chain's progress: pipeline._chain)
            put(clean if clean in boxes else chain, "active", frac / CLEAN_SHARE if clean in boxes else frac, "clean-up")
        else:  # "chain", "mix"
            put(clean, "done")
            put(chain, "active", (frac - CLEAN_SHARE) / (1 - CLEAN_SHARE) if clean in boxes else frac,
                "its place in the mix" if phase == "mix" else "studio chain")
    past = lambda stage: at > STAGES.index(stage) or over  # noqa: E731
    if past("stems"):
        for b in boxes:
            if b.startswith(("cleaned/", "chain/")) and out.get(b, {}).get("state") not in ("kept", "skipped"):
                put(b, "done")
    part = float((unit or {}).get("part") or 0.0)
    detail = (unit or {}).get("detail") or ""
    for stage, mine in (("drum_bus", [b for b in boxes if b == "drum_bus/drum_bus"]), ("blend", [b for b in boxes if b.startswith("premix/")]),
                        ("bus", [b for b in boxes if b == "mix/mix"])):
        for b in mine:
            if key == stage and not over:
                put(b, "active", part if stage == "bus" else None, detail if stage == "bus" else "")
            elif past(stage):
                put(b, "done")
    for b in [b for b in boxes if b.startswith("master/")]:
        if key in ("master", "write") and not over:
            put(b, "active", note="loudness and limiter" if key == "master" else "writing the files")
        elif past("write"):
            put(b, "done")
    if over and unit.get("kept"):  # already mastered with these settings: nothing was made again
        out = {b: {"state": "kept", "note": "up to date"} for b in boxes}
    if unit and unit["state"] in ("failed", "cancelled"):
        out = {b: ({"state": "stopped"} if s["state"] == "active" else s) for b, s in out.items()}
    return out
