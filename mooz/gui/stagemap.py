"""A take's files laid out along the pipeline, for the take's page: what was recorded, the
separated stems, every stage file the run wrote (stages.py), the mix and the master.

Nothing is computed here. The page plays the files the command line left (the stage files
in <output dir>/stages/<take>/, the stem caches they link to, the masters) and shows the
figures of the take's report next to them.

  columns   the pipeline's stages, in the order the audio goes through them
  rows      the recording itself, then the stems (the kit's last, next to each other)
  nodes     one file each: its column, the rows it belongs to ("*": the whole band), whether it
            is the whole recording or already trimmed, and how far it is from its real level
  extras    files with no place in the grid: every separation model's own output
  graph     the same files as boxes and arrows: who feeds whom, the close tracks' stems as
            boxes of their own, the separations as steps
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from .. import config, stages
from . import projects
from .projects import UserError

AUDIO = (".flac", ".wav", ".mp3")
KIT = ("kick", "snare", "toms", "hihat", "cymbals")
ORDER = ("vocals", "backing_vocals", "vocal_room", "guitar", "guitar_left", "guitar_right", "bass", "other", *KIT, "drums")

COLUMNS = [
    ("recording", "Recording", "The take as the recorder wrote it, whole: the trim comes later."),
    ("separated", "Separated", "The stems the chains start from, untouched and not trimmed yet."),
    ("cleaned", "Cleaned", "The kit's stems after the clean-up (snare buzz, bleed, gates), and what it took out."),
    ("chain", "Stem chain", "Each stem after its own chain: tone, dynamics, samples under the drums. No level or place in the mix yet."),
    ("drum_bus", "Drum bus", "The kit mixed as one instrument."),
    ("premix", "In the mix", "Every part as it goes into the mix, balanced and placed. Together they are the mix."),
    ("mix", "Mix", "After the mix bus, before loudness and the limiter."),
    ("master", "Master", "The finished file."),
]
# the graph's layers, left to right
LAYERS = ["Recording", "Separation", "Stems", "Merged", "Cleaned", "Stem chain", "Drum bus", "In the mix", "Mix", "Master"]


def _order(name: str) -> tuple:
    return (ORDER.index(name) if name in ORDER else len(ORDER), name)


def _stems(folder: Path) -> dict[str, Path]:
    """A folder's audio files by name; the ones starting with _ are working files."""
    out: dict[str, Path] = {}
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in AUDIO and f.is_file() and not f.name.startswith("_"):
                out.setdefault(f.stem, f)
    return out


def _audio(folder: Path, name: str, suffixes: tuple[str, ...] = AUDIO) -> Path | None:
    return next((folder / f"{name}{s}" for s in suffixes if (folder / f"{name}{s}").is_file()), None)


def _raw_dir(take, cfg: dict, stage: Path) -> Path:
    """The room track's stems: the stage folder's link, or without stage files the cache itself."""
    if (stage / "1_raw").is_dir():
        return stage / "1_raw"
    from .. import separate  # only for the cache's name; kept out of the app's start

    cache = take.root / (cfg["output"]["stems_dir"] + separate.cache_name(cfg["stems"])) / take.name
    return cache if (cache / "key.json").exists() else stage / "1_raw"  # a separation in flight is not listed


def build(cfg_path: Path, take_name: str, profile: str) -> dict:
    found = projects.find_takes(cfg_path)
    take = next((t for t in found if t.name == take_name), None)
    if take is None:
        raise UserError(f"There is no take '{take_name}' in this project.")
    try:
        cfg = config.load(cfg_path, None if profile == projects.DEFAULT else profile)
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        raise UserError(projects._message(e)) from None
    out_dir = take.root / cfg["output"]["dir"]
    stage = out_dir / "stages" / take.name
    try:
        report = json.loads((out_dir / "reports" / f"{take.name}.json").read_text())
    except (OSError, json.JSONDecodeError):
        report = {}
    nodes: list[dict] = []
    extras: list[dict] = []
    staged = stages.GAIN_DB

    def rel(f: Path) -> str | None:
        try:
            return str(f.relative_to(take.root))
        except ValueError:  # an output folder outside the take's: not served
            return None

    def node(col: str, key: str, rows, label: str, f: Path, what: str, *, whole: bool = False, level: float = 0.0,
             sync: bool = True, main: bool = True, group: str | None = None, **more) -> None:
        if rel(f) is None:
            return
        n = {"id": f"{col}/{key}", "col": col, "rows": rows, "label": label, "what": what, "file": rel(f), "whole": whole,
             "level_db": level, "sync": sync, "main": main} | more
        if group is None:
            nodes.append(n)
        else:
            extras.append(n | {"group": group})

    # what was recorded
    zoom = bool(take.close or take.mixdown)
    node("recording", "room", "*", take.room.stem.rsplit("_", 1)[-1] if zoom else "recording", take.room,
         "the room track: the recording the pipeline works on" if zoom else "the recording the pipeline works on", whole=True)
    for name, f in take.close.items():
        node("recording", name, "*", name, f, "a close track: its voice and bass go into the room's stems", whole=True, main=False)
    if take.mixdown is not None:
        node("recording", "mixdown", "*", take.mixdown.stem.rsplit("_", 1)[-1], take.mixdown,
             "the recorder's own mix-down of its tracks (not used)", whole=True, main=False)

    # the stems and the stage files
    raw_dir = _raw_dir(take, cfg, stage)
    raw = _stems(raw_dir)
    merged = {n: f for n, f in _stems(stage / "1_merged").items() if not f.is_symlink()}  # a link: the room's stem as it is
    close = {d.name.removeprefix("1_raw_"): _stems(d) for d in sorted(stage.glob("1_raw_*")) if d.is_dir()}
    cleaned, chain = _stems(stage / "3_drum_cleanup"), _stems(stage / "2_stems")
    samples, premix = _stems(stage / "2_stems" / "samples"), _stems(stage / "5_premix")
    kit_names = set((report.get("tracks") or {}).get("drum_bus", {}).get("stems") or KIT)
    names = sorted(set(chain) | set(raw) | set(merged),
                   key=lambda n: (n in kit_names, ORDER.index(n) if n in ORDER else len(ORDER), n))
    kit = [n for n in names if n in kit_names]
    rows = [{"key": "original", "label": "recording", "kit": False}] if "original" in premix else []
    rows += [{"key": n, "label": n.replace("_", " "), "kit": n in kit} for n in names]

    # With close tracks every pad of the Separated column says what it is: "merged" or "room" for what the chain
    # starts from, then what it is made of. The parts of a merged stem are at their level in it (as the merge's
    # report has it), so they compare by ear with "Levels lined up"; a close stem left out is as its file is.
    mt = report.get("multitrack") or {}
    for n in names:
        was = (mt.get("merged") or {}).get(n, {})
        if n in merged:
            node("separated", n, [n], n, merged[n], "the close track's stem with a little of the room's: what the chain starts from",
                 whole=True, short="merged")
            mine = _audio(stage / "1_merged", f"_room_{n}")  # the room's voice as the merge takes it: dry part and room sound
            if mine or n in raw:
                node("separated", f"{n}/room", [n], "room", mine or raw[n],
                     "the room's voice as it goes into the merge: the dry part and the room's sound, the room turned down" if mine
                     else "the room track's stem, at its level in the merged one", whole=True, main=False,
                     level=-float(was.get("room_gain_db", 0.0)))
            if mine and n in raw:
                node("separated", f"{n}/raw", "*", f"{n}, the dry part", raw[n], "the room's stem as the separation gave it, without its room sound",
                     whole=True, group="The room track's stems")
        elif n in raw:
            node("separated", n, [n], n, raw[n], "the room track's stem, as the separation gave it" if close else "as the separation models gave it",
                 whole=True, **({"short": "room"} if close else {}))
        for tr, files in close.items():
            if n in files:
                info = (mt.get("close") or {}).get(tr, {}).get(n) or {}
                used = bool(info.get("used", n in merged))
                under = f", {abs(info['level_lu']):g} LU under its track's loudest stem" if info.get("level_lu") is not None else ""
                node("separated", f"{n}/{tr}", [n], tr, files[n],
                     f"the close track {tr}'s stem, at its level in the merged one" if used
                     else f"the close track {tr}'s stem: not used, it is bleed there{under}",
                     whole=True, main=False, unused=not used, level=-float(was.get("gain_db", 0.0)) if used else 0.0)
        if n in cleaned:
            node("cleaned", n, [n], n, cleaned[n], "after the clean-up", level=staged)
        if f"{n}_removed" in cleaned:
            node("cleaned", f"{n}/removed", [n], "taken out", cleaned[f"{n}_removed"], "what the clean-up took out", level=staged, main=False)
        if n in chain:
            node("chain", n, [n], n, chain[n], "after its chain", level=staged)
        for k, f in samples.items():
            if k.rsplit("_", 1)[0] == n:
                i = k.rsplit("_", 1)[1]
                node("chain", f"{n}/sample{i}", [n], i, f, f"sample {i} laid under its hits, on its own (not at its level in the mix)",
                     level=staged, sync=False, main=False)
    for tr, files in close.items():
        for n, f in files.items():
            if n not in names:
                node("separated", f"{n}/{tr}", "*", n, f, f"from the close track {tr}", whole=True, group=f"Close track {tr}")
    bus = _audio(stage, "4_drum_bus")
    if bus:
        node("drum_bus", "drum_bus", kit or "*", "drum bus", bus, "the kit mixed as one instrument", level=staged,
             group=None if kit else "Stage files")
    for n, f in premix.items():
        if n == "original":
            node("premix", n, ["original"], "recording", f, "the recording itself, at its share under the stems", level=staged)
        elif n == "drum_bus" and kit:
            node("premix", n, kit, "drum bus", f, "as it goes into the mix: balanced and placed", level=staged)
        elif n in names:
            node("premix", n, [n], n, f, "as it goes into the mix: balanced and placed", level=staged)
        else:
            node("premix", n, "*", n, f, "as it goes into the mix", level=staged, group="Stage files")
    mix = _audio(stage, "5_mix")
    if mix:
        node("mix", "mix", "*", "mix", mix, "the mix before loudness and the limiter", level=staged)
    master = _audio(out_dir, take.name, (".wav", ".mp3"))  # the WAV first: an MP3 starts a few ms late
    if master:
        node("master", "master", "*", "master", master, "the finished file")
    ab = _audio(out_dir, f"{take.name}_AB", (".wav", ".mp3"))
    if ab:
        node("master", "ab", "*", "A/B", ab, "the recording and the master in turns, at the same loudness", main=False)

    # every separation model's own output, as each separation listed them
    for track, folder in [("room", raw_dir)] + [(tr, stage / f"1_raw_{tr}") for tr in close]:
        try:
            listed = json.loads((folder / "models" / "index.json").read_text())["files"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        for e in listed:
            f = folder / "models" / str(e.get("file", ""))
            if f.is_file() and f.suffix.lower() in AUDIO:
                source = str(e.get("input", "")).replace("+", ", ").replace("_", " ").strip().replace("  ", " ")
                node("models", f"{track}/{e['file']}", "*", f"{f.parent.name}: {f.stem.lstrip('_')}", f,
                     f"one model's own output, weight {e.get('weight', 1.0):g} in the stem", whole=True, level=float(e.get("level_db", 0.0)),
                     group=f"Separation of {'the room track' if track == 'room' else track}, step {e.get('step', '')}: from the {source}",
                     track=track)

    roots = sorted({t.root.resolve() for t in found})
    return {"take": projects._take_info(take) | {"root": roots.index(take.root.resolve())}, "profile": profile,
            "output_dir": cfg["output"]["dir"], "stage_dir": rel(stage) if stage.is_dir() else None,
            "stages_setting": cfg["output"]["stages"], "mastered": bool(master),
            "columns": [{"key": k, "title": t, "about": a} for k, t, a in COLUMNS], "rows": rows, "nodes": nodes, "extras": extras,
            "graph": _graph(take, rows, names, kit, close, set(merged), nodes, extras, report),
            "trim_s": float((report.get("trim") or {}).get("start_s") or 0.0), "stage_gain_db": staged, "report": report}


def _graph(take, rows: list[dict], names: list[str], kit: list[str], close: dict[str, dict], merged: set[str],
           nodes: list[dict], extras: list[dict], report: dict) -> dict:
    """The take's files as boxes and arrows. A box is a file to play ("play": its node above) or a
    step with no file of its own (the separations); layer is its column, lane its height. An arrow
    says that one goes into the other, with the share or the gain where the report has it."""
    have = {n["id"] for n in nodes} | {n["id"] for n in extras}
    mt, tracks = report.get("multitrack") or {}, report.get("tracks") or {}
    boxes: list[dict] = []
    arrows: list[dict] = []

    def box(gid: str, label: str, layer: int, lane: float, play: str | None = None, **more) -> str:
        boxes.append({"id": gid, "label": label.replace("_", " "), "layer": layer, "lane": lane, "play": play if play in have else None} | more)
        return gid

    def arrow(a: str, b: str, label: str = "", **more) -> None:
        arrows.append({"from": a, "to": b, "label": label} | more)

    # the close tracks on top, each with its separation and its stems
    lane = 0.0
    for tr in take.close:
        files = sorted(close.get(tr, {}), key=_order)
        mid = lane + (max(len(files), 1) - 1) / 2
        src = box(f"src/{tr}", tr, 0, mid, f"recording/{tr}")
        if files:
            sep = box(f"sep/{tr}", "separation", 1, mid, step=True, track=tr, what=f"The close track {tr} split into its instruments.")
            arrow(src, sep)
        for i, stem in enumerate(files):
            info = (mt.get("close") or {}).get(tr, {}).get(stem) or {}
            unused = not info.get("used", stem in merged)
            arrow(sep, box(f"close/{tr}/{stem}", stem, 2, lane + i, f"separated/{stem}/{tr}", unused=unused), unused=unused)
        lane += max(len(files), 1)
    if take.close:
        lane += 0.6
    # the room track under them: a lane per row of the grid
    at = {r["key"]: lane + i for i, r in enumerate(rows)}
    middle = lambda keys: sum(at[k] for k in keys) / len(keys) if keys else lane  # noqa: E731
    room = box("src/room", next((n["label"] for n in nodes if n["id"] == "recording/room"), "recording"), 0,
               at.get("original", middle(names)), "recording/room")
    tail: dict[str, str] = {}
    of_room = {n: next((i for i in (f"separated/{n}/raw", f"separated/{n}/room", f"separated/{n}") if i in have
                        and (n in merged or i == f"separated/{n}")), None) for n in names}
    if any(of_room.values()):
        sep = box("sep/room", "separation", 1, middle(names), step=True, track="room", what="The room track split into its stems by the separation models.")
        arrow(room, sep)
    for n in names:
        if of_room[n]:
            tail[n] = box(f"stem/{n}", n, 2, at[n], of_room[n])
            arrow(sep, tail[n])
        if n in merged:
            was = (mt.get("merged") or {}).get(n, {})
            share = was.get("close_share")
            m = box(f"merged/{n}", n, 3, at[n], f"separated/{n}")
            if n in tail:
                arrow(tail[n], m, f"{100 * (1 - share):.0f} %" if share else "")
            for tr in was.get("tracks") or [t for t in close if n in close[t]]:
                if n in close.get(tr, {}):
                    arrow(f"close/{tr}/{n}", m, f"{100 * share:.0f} %" if share else "")
            tail[n] = m
    for n in names:
        for layer, col in ((4, "cleaned"), (5, "chain")):
            if f"{col}/{n}" in have:
                here = box(f"{col}/{n}", n, layer, at[n], f"{col}/{n}")
                if n in tail:
                    arrow(tail[n], here)
                tail[n] = here
    if "vocal_room" in tail and "vocals" in names and "chain/vocal_room" not in have:  # the voice's room sound goes back into the voice
        to = next((i for i in ("merged/vocals", "cleaned/vocals", "chain/vocals") if any(b["id"] == i for b in boxes)), None)
        if to:
            arrow(tail.pop("vocal_room"), to, "its room")
    bus = None
    if "drum_bus/drum_bus" in have and kit:
        bus = box("drum_bus/drum_bus", "drum bus", 6, middle(kit), "drum_bus/drum_bus")
        for k in kit:
            if k in tail:
                arrow(tail[k], bus)
    parts = []
    for n in nodes:
        if n["col"] != "premix":
            continue
        name = n["id"].split("/", 1)[1]
        gain = (tracks.get(name) or {}).get("balance_gain_db")
        label = f"{gain:+.1f} dB" if isinstance(gain, (int, float)) else ""
        if name == "original":
            share = (tracks.get("blend") or {}).get("original_share")
            parts.append(box(n["id"], "recording", 7, at["original"], n["id"]))
            arrow(room, parts[-1], f"{100 * share:.0f} %" if share else "")
        elif name == "drum_bus":
            parts.append(box(n["id"], "drum bus", 7, middle(kit), n["id"]))
            for src in [bus] if bus else [tail[k] for k in kit if k in tail]:
                arrow(src, parts[-1], label)
        elif name in at:
            parts.append(box(n["id"], name, 7, at[name], n["id"]))
            if name in tail:
                arrow(tail[name], parts[-1], label)
    centre = middle([r["key"] for r in rows])
    if "mix/mix" in have:
        box("mix/mix", "mix", 8, centre, "mix/mix")
        for part in parts:
            arrow(part, "mix/mix")
    if "master/master" in have:
        box("master/master", "master", 9, centre, "master/master")
        if "mix/mix" in have:
            arrow("mix/mix", "master/master")
        if "master/ab" in have:
            box("master/ab", "A/B", 9, centre + 1.2, "master/ab")
            arrow("master/master", "master/ab")
    if take.mixdown is not None:  # the recorder's own mix-down: recorded, not used
        box("src/mixdown", next((n["label"] for n in nodes if n["id"] == "recording/mixdown"), "mix-down"), 0,
            max([b["lane"] for b in boxes] + [lane]) + 1.2, "recording/mixdown", unused=True)
    return {"layers": LAYERS, "boxes": boxes, "arrows": arrows}
