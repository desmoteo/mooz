"""What the web app knows of the files: the list of projects, and for a project its takes,
their results, its settings and its profiles.

A project is a settings file (rehearsal.toml) and the recordings its [input] paths name. A
profile is a preset: a built-in one, or presets/<name>.toml next to the settings file (the
project's own, or its version of a built-in one); "default" is the project's settings with
no preset. Nothing here is the app's own format
but the list of projects: everything else is what the command line reads and writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from pathlib import Path

import soundfile as sf
import tomlkit
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import InlineTable

from .. import config, takes

DEFAULT = "default"  # the profile with no preset
PROFILE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
SERVED = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".png": "image/png",
          ".json": "application/json"}

NEW_PROJECT = """\
# Settings for the recordings in this folder. List only what you want to change;
# every option, with an explanation, is in mooz/defaults.toml.
# Order of precedence: defaults < this file < the profile (preset) < [files."..."].

[input]
paths = ["."]              # the recordings: this folder
presets = [""]             # the profiles every take is mastered with ("" = no preset)
"""

NEW_PROFILE = """\
# {name}: what this profile changes, on top of the project's settings.
# Every option is explained in mooz/defaults.toml.

[output]
dir = "processed/{name}"   # its own folder, so it does not overwrite another profile's masters
"""

# The settings the form shows: the ones the README's tuning table sends people to most.
# (key, label, kind, help[, min, max, step] or [, choices])
KNOBS: list[tuple[str, list[tuple]]] = [
    ("Loudness and blend", [
        ("loudness.target_lufs", "Loudness target", "number",
         "LUFS. −14 is where streaming and phones play; −16 keeps more punch.", -24.0, -8.0, 0.5),
        ("mix.original_share", "Original under the mix", "number",
         "The untouched recording's share of the loudness. 0.25 is a quarter; 0 is the stems alone.", 0.0, 0.5, 0.05),
        ("stems.vocal_room.max_reduction_db", "Vocal room turned down", "number",
         "dB at most. 0 leaves the voice as recorded.", 0.0, 18.0, 1.0),
        ("character.adapt", "Adapt the mix to each song's style", "bool",
         "Off: one chain for every song, so the songs sound alike."),
    ]),
    ("Levels in every song", [
        ("balance.offsets.vocals", "Vocals", "number", "dB on top of the learnt target.", -9.0, 9.0, 0.5),
        ("balance.offsets.guitars", "Guitars", "number", "dB on top of the learnt target.", -9.0, 9.0, 0.5),
        ("balance.offsets.bass", "Bass", "number", "dB on top of the learnt target.", -9.0, 9.0, 0.5),
        ("balance.offsets.drums", "Drums", "number", "dB on top of the learnt target.", -9.0, 9.0, 0.5),
    ]),
    ("Levels inside the kit", [
        ("drum_bus.balance.offsets.kick", "Kick", "number", "dB against the rest of the kit.", -9.0, 9.0, 0.5),
        ("drum_bus.balance.offsets.snare", "Snare", "number", "dB against the rest of the kit.", -9.0, 9.0, 0.5),
        ("drum_bus.balance.offsets.toms", "Toms", "number", "dB against the rest of the kit.", -9.0, 9.0, 0.5),
        ("drum_bus.balance.offsets.hihat", "Hi-hat", "number", "dB against the rest of the kit.", -9.0, 9.0, 0.5),
        ("drum_bus.balance.offsets.cymbals", "Cymbals", "number", "dB against the rest of the kit.", -9.0, 9.0, 0.5),
    ]),
    ("Guitars and close tracks", [
        ("tracks.guitar.width", "Guitar width", "number", "0 is mono, 1 as recorded.", 0.0, 1.5, 0.05),
        ("tracks.guitar.balance", "Guitars pulled to the centre", "number", "0 leaves the two amps where they stood, 1 centres them.", 0.0, 1.0, 0.05),
        ("multitrack.close_share.vocals", "Close voice against the room's", "number",
         "With a close track: its share of the merged voice. 0.9 adds a tenth of the room's; lower is roomier.", 0.0, 1.0, 0.05),
        ("multitrack.close_share.bass", "Close bass against the room's", "number",
         "With a close track: its share of the merged bass. 0.9 adds a tenth of the room's.", 0.0, 1.0, 0.05),
    ]),
    ("Files", [
        ("output.artist", "Artist", "text", "Written in the MP3's tags."),
        ("output.album", "Album", "text", "{date} becomes the recording's date."),
        ("output.mp3", "Write an MP3 next to the WAV", "bool", ""),
        ("output.stages", "Stage files", "choice", "Every stage of a take, for listening. About 0.5 GB per take.",
         ["flac", "wav", "mp3", "off"]),
    ]),
]


class UserError(Exception):
    """Something the person can fix: shown as it is."""


# ---- the list of projects ------------------------------------------------------------------

def default_registry() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "mooz" / "projects.json"


class Registry:
    """The known projects: a name and a settings file each, in one small JSON file."""

    def __init__(self, path: Path | None = None):
        self.path = path or default_registry()

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text())["projects"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return []

    def _write(self, projects: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"projects": projects}, indent=1))
        os.replace(tmp, self.path)

    def list(self) -> list[dict]:
        return [p | {"exists": Path(p["config"]).exists(), "folder": str(Path(p["config"]).parent)} for p in self._read()]

    def get(self, pid: str) -> dict:
        for p in self.list():
            if p["id"] == pid:
                return p
        raise UserError("That project is not in the list any more.")

    def add(self, name: str, cfg_path: Path) -> dict:
        cfg_path = cfg_path.resolve()
        pid = hashlib.sha1(str(cfg_path).encode()).hexdigest()[:10]
        projects = [p for p in self._read() if p["id"] != pid]
        projects.append({"id": pid, "name": name.strip() or cfg_path.parent.name, "config": str(cfg_path)})
        self._write(projects)
        return self.get(pid)

    def remove(self, pid: str) -> None:
        self._write([p for p in self._read() if p["id"] != pid])

    def adopt(self, folder: Path) -> dict | None:
        """The project in folder, added to the list if it has a settings file and is not there yet."""
        cfg_path = (folder / "rehearsal.toml").resolve()
        if not cfg_path.exists():
            return None
        known = next((p for p in self.list() if p["config"] == str(cfg_path)), None)
        return known or self.add(cfg_path.parent.name, cfg_path)


# ---- folders and takes ---------------------------------------------------------------------

def browse(path: str | None) -> dict:
    """The folders inside path, for choosing where the recordings are."""
    here = Path(path).expanduser() if path else Path.home()
    if not here.is_dir():
        raise UserError(f"Not a folder: {here}")
    here = here.resolve()
    dirs, audio = [], 0
    try:
        for entry in sorted(here.iterdir(), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": str(entry)})
            elif entry.suffix.lower() in takes.AUDIO_SUFFIXES:
                audio += 1
    except PermissionError:
        raise UserError(f"No permission to read {here}") from None
    return {"path": str(here), "parent": str(here.parent) if here.parent != here else None, "dirs": dirs,
            "audio_files": audio, "has_settings": (here / "rehearsal.toml").exists()}


def _skip(cfg: dict) -> set[str]:
    """Folders a search for recordings leaves out: our own output (as the command line does)."""
    return {Path(cfg["output"]["dir"]).parts[0], Path(cfg["output"]["stems_dir"]).parts[0],
            Path(cfg["chains"]["dir"]).parts[0], "experiments", "prove", ".venv"}


def find_takes(cfg_path: Path | None, folder: Path | None = None) -> list[takes.Take]:
    """The takes of a settings file's [input] paths, or of folder when there is no file yet."""
    cfg = config.load(cfg_path)
    here = cfg_path.parent if cfg_path else folder
    paths = [here / p for p in cfg["input"]["paths"]] or [here]
    try:
        return takes.discover(paths, _skip(cfg), cfg["multitrack"]["room"])
    except (FileNotFoundError, ValueError) as e:
        raise UserError(str(e)) from None


def _take_info(take: takes.Take) -> dict:
    try:
        info = sf.info(take.room)
        seconds = round(info.frames / info.samplerate, 1)
    except Exception:  # an unreadable file is still listed: the run will say what is wrong with it
        seconds = None
    tracks = ([take.room.stem.rsplit("_", 1)[-1]] if take.close or take.mixdown else ["stereo"]) + list(take.close)
    if take.mixdown is not None:
        tracks.append("TrLR")
    return {"name": take.name, "title": take.title, "date": take.date, "seconds": seconds, "tracks": tracks,
            "room": str(take.room), "room_file": _inside(take.room, take.root)}


def _inside(f: Path, root: Path) -> str | None:
    """f as the browser asks for it: relative to its take's folder."""
    try:
        return str(f.relative_to(root))
    except ValueError:
        return None


def preview(folder: str) -> dict:
    """What creating a project in folder would find, before anything is written."""
    here = Path(folder).expanduser()
    if not here.is_dir():
        raise UserError(f"Not a folder: {here}")
    here = here.resolve()
    cfg_path = here / "rehearsal.toml"
    exists = cfg_path.exists()
    try:
        found = find_takes(cfg_path if exists else None, here)
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        raise UserError(f"{cfg_path}: {_message(e)}") from None
    return {"folder": str(here), "has_settings": exists, "takes": [_take_info(t) for t in found]}


def create(registry: Registry, name: str, folder: str, presets: list[str]) -> dict:
    """A project in folder: its rehearsal.toml written (an existing one is kept as it is), and listed."""
    here = Path(folder).expanduser()
    if not here.is_dir():
        raise UserError(f"Not a folder: {here}")
    cfg_path = here.resolve() / "rehearsal.toml"
    if not cfg_path.exists():
        doc = tomlkit.parse(NEW_PROJECT)
        doc["input"]["presets"] = _to_presets(presets or [DEFAULT], None)
        try:
            cfg_path.write_text(tomlkit.dumps(doc))
        except OSError as e:
            raise UserError(f"Cannot write {cfg_path}: {e.strerror}") from None
    return registry.add(name, cfg_path)


# ---- a project: takes and results ----------------------------------------------------------

def _message(e: Exception) -> str:
    return e.args[0] if isinstance(e, KeyError) and e.args else str(e)


def _to_presets(names: list[str], cfg_path: Path | None) -> list[str]:
    out = []
    for n in names:
        if n != DEFAULT:
            try:
                config.preset_file(n, cfg_path)
            except KeyError as e:
                raise UserError(_message(e)) from None
        out.append("" if n == DEFAULT else n)
    return out


def _first_comment(path: Path) -> str:
    """A profile's description: the first sentence of the comment its file opens with."""
    lines = []
    for line in path.read_text().splitlines():
        text = line.lstrip("#").strip()
        if not line.startswith("#") or not text:
            break
        lines.append(text)
    return " ".join(lines).split(". ")[0].rstrip(".:") + "." if lines else ""


def _builtin(name: str) -> bool:
    return (config.PRESETS_DIR / f"{name}.toml").exists()


def profiles(cfg_path: Path) -> list[dict]:
    """Every profile the project can use, with where its masters go. builtin: one of that name comes
    with the program; own: the project has its file (its version, when both)."""
    out = []
    for name in [DEFAULT] + config.presets(cfg_path):
        p = {"name": name, "builtin": True, "own": False, "description": "The project's settings, with no preset on top."}
        if name != DEFAULT:
            f = config.preset_file(name, cfg_path)
            p["builtin"], p["own"] = _builtin(name), f.parent != config.PRESETS_DIR
            p["description"] = _first_comment(f)
        try:
            p["output_dir"] = config.load(cfg_path, None if name == DEFAULT else name)["output"]["dir"]
        except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
            p["error"] = _message(e)
        out.append(p)
    return out


def _result(take: takes.Take, out_dir: str) -> dict | None:
    """What a profile made of a take, from its report and the files next to it."""
    base = take.root / out_dir
    report = base / "reports" / f"{take.name}.json"
    files = {kind: base / f"{take.name}{suffix}" for kind, suffix in
             (("mp3", ".mp3"), ("wav", ".wav"), ("ab_mp3", "_AB.mp3"), ("ab_wav", "_AB.wav"))}
    files |= {"figure": base / "reports" / f"{take.name}.png", "tracks_figure": base / "reports" / f"{take.name}_tracks.png"}
    files = {k: str(f.relative_to(take.root)) for k, f in files.items() if f.exists()}
    if not files and not report.exists():
        return None
    out: dict = {"files": files}
    try:
        rep = json.loads(report.read_text())
        out |= {"lufs": rep["processed"]["integrated_lufs"], "lra_lu": rep["processed"]["lra_lu"],
                "true_peak_dbtp": rep["processed"]["true_peak_dbtp"], "original_lra_lu": rep["original"]["lra_lu"],
                "limited_pct": rep["finalize"]["limiter"]["limited_pct"], "when": report.stat().st_mtime,
                "trim_s": float((rep.get("trim") or {}).get("start_s") or 0.0)}
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    return out


def detail(project: dict) -> dict:
    """The project's page: its takes, its profiles, and each take's result under each profile."""
    cfg_path = Path(project["config"])
    out = dict(project) | {"takes": [], "profiles": [], "run": [DEFAULT], "roots": []}
    if not cfg_path.exists():
        return out | {"error": f"{cfg_path} is gone. Remove the project from the list, or put the file back."}
    try:
        cfg = config.load(cfg_path)
        found = find_takes(cfg_path)
        out["profiles"] = profiles(cfg_path)
    except (UserError, KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        return out | {"error": f"{cfg_path.name}: {_message(e)}"}
    out["run"] = [p or DEFAULT for p in (cfg["input"]["presets"] or [""])]
    roots = sorted({t.root.resolve() for t in found})
    out["roots"] = [str(r) for r in roots]
    for take in found:
        info = _take_info(take) | {"root": roots.index(take.root.resolve()), "results": {}}
        for p in out["profiles"]:
            if "output_dir" in p:
                res = _result(take, p["output_dir"])
                if res:
                    info["results"][p["name"]] = res
        out["takes"].append(info)
    from . import listening  # it uses this module's helpers

    try:
        out["listening"] = listening.view(cfg_path, out["takes"])
    except UserError as e:
        out["listening"] = {"verdicts": [], "pairs": [], "error": str(e)}
    return out


def served_file(roots: list[str], index: int, rel: str) -> Path:
    """A project's file for the browser: only inside the project's folders, only audio, figures
    and reports."""
    if not 0 <= index < len(roots):
        raise UserError("No such folder in this project.")
    root = Path(roots[index]).resolve()
    target = (root / rel).resolve()
    if not target.is_relative_to(root) or target.suffix.lower() not in SERVED or not target.is_file():
        raise UserError("No such file in this project.")
    return target


# ---- settings ------------------------------------------------------------------------------

def _get(d: dict, key: str):
    for k in key.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _groups(cfg: dict, base: dict, raw: dict) -> list[dict]:
    """The form's fields: each knob's value in cfg, its value without the file being edited
    ("default": the defaults for the project's settings, the project's for a profile), and
    whether that file (raw) sets it."""
    groups = []
    for title, knobs in KNOBS:
        fields = []
        for key, label, kind, help_, *rest in knobs:
            f = {"key": key, "label": label, "kind": kind, "help": help_, "value": _get(cfg, key),
                 "default": _get(base, key), "set": _get(raw, key) is not None}
            if kind == "number":
                f["min"], f["max"], f["step"] = rest
            elif kind == "choice":
                f["choices"] = rest[0]
            fields.append(f)
        groups.append({"title": title, "fields": fields})
    return groups


def settings(cfg_path: Path) -> dict:
    """The project's settings for the form: every knob with its value, the default, and whether
    the file sets it; and the file's text."""
    text = cfg_path.read_text()
    defaults = config.load(None)
    try:
        raw = tomllib.loads(text)
        cfg = config.load(cfg_path)
        error = None
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        raw, cfg, error = {}, defaults, _message(e)
    return {"toml": text, "groups": _groups(cfg, defaults, raw), "error": error,
            "run": [p or DEFAULT for p in (cfg["input"]["presets"] or [""])], "defaults_file": str(config.DEFAULTS_PATH)}


def _knob(key: str) -> tuple:
    for _, knobs in KNOBS:
        for k in knobs:
            if k[0] == key:
                return k
    raise UserError(f"'{key}' is not a setting of the form.")


def _typed(key: str, value):
    """A value from the form as the settings file wants it."""
    _, label, kind, _, *rest = _knob(key)
    if kind == "bool":
        if not isinstance(value, bool):
            raise UserError(f"{label}: on or off.")
        return value
    if kind == "text":
        return str(value)
    if kind == "choice":
        if value not in rest[0]:
            raise UserError(f"{label}: one of {', '.join(rest[0])}.")
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise UserError(f"{label}: a number is needed.") from None
    lo, hi, _ = rest
    if not lo <= number <= hi:
        raise UserError(f"{label}: between {lo:g} and {hi:g}.")
    return number


def _set(doc, key: str, value) -> None:
    node = doc
    *parents, leaf = key.split(".")
    for k in parents:
        if k not in node:
            node[k] = tomlkit.inline_table() if isinstance(node, InlineTable) else tomlkit.table()
        node = node[k]
    node[leaf] = value


def _unset(doc, key: str) -> None:
    """The key taken out of the file; a table left with nothing in it (no comment either) goes too."""
    chain = [doc]
    *parents, leaf = key.split(".")
    for k in parents:
        if k not in chain[-1]:
            return
        chain.append(chain[-1][k])
    if leaf in chain[-1]:
        del chain[-1][leaf]
    for node, parent, k in zip(reversed(chain[1:]), reversed(chain[:-1]), reversed(parents)):
        if len(node) == 0 and (isinstance(node, InlineTable) or not node.as_string().strip()):
            del parent[k]
        else:
            break


def _check(cfg_path: Path, text: str) -> None:
    """Raises UserError unless text loads as this project's settings, as the command line would load it."""
    tmp = cfg_path.with_name(f".{cfg_path.name}.check")
    try:
        tmp.write_text(text)
        cfg = config.load(tmp)
        for preset in cfg["input"]["presets"]:
            if preset:
                config.load(tmp, preset)
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        raise UserError(_message(e).replace(f"{tmp}.", "").replace(str(tmp), cfg_path.name)) from None
    except OSError as e:
        raise UserError(f"Cannot write in {cfg_path.parent}: {e.strerror}") from None
    finally:
        tmp.unlink(missing_ok=True)


def _replace(path: Path, text: str, backup: bool = True) -> None:
    if path.exists() and path.read_text() == text:
        return
    if backup and path.exists():
        path.with_name(path.name + ".bak").write_text(path.read_text())
    tmp = path.with_name(f".{path.name}.new")
    tmp.write_text(text)
    os.replace(tmp, path)


def _with_values(text: str, values: dict | None):
    """The form's values on a file's text, comments kept (None takes the key out)."""
    try:
        doc = tomlkit.parse(text)
    except TOMLKitError as e:
        raise UserError(str(e)) from None
    for key, value in (values or {}).items():
        if value is None:
            _knob(key)
            _unset(doc, key)
        else:
            _set(doc, key, _typed(key, value))
    return doc


def save_settings(cfg_path: Path, values: dict | None = None, run: list[str] | None = None, toml: str | None = None) -> None:
    """The settings file rewritten: the whole text (toml), or the form's values on the file as it
    is, comments kept (a value of None takes the key out: back to the default). Checked before
    it is written; the previous file is kept as rehearsal.toml.bak."""
    if toml is None:
        doc = _with_values(cfg_path.read_text(), values)
        if run is not None:
            if not run:
                raise UserError("Choose at least one profile to run.")
            _set(doc, "input.presets", _to_presets(run, cfg_path))
        toml = tomlkit.dumps(doc)
    _check(cfg_path, toml)
    _replace(cfg_path, toml)


# ---- profiles ------------------------------------------------------------------------------

def _profile_view(cfg_path: Path, name: str, text: str) -> dict:
    """A profile for its editor: the text, and the form's fields with the project's values as
    what each goes back to."""
    out = {"name": name, "text": text, "groups": [], "error": None}
    try:
        base, layer = config.load(cfg_path), tomllib.loads(text)
        out["groups"] = _groups(config.with_layer(base, f"profile {name}", layer), base, layer)
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        out["error"] = _message(e).replace(f"profile {name}.", "")
    return out


def read_profile(cfg_path: Path, name: str) -> dict:
    if name == DEFAULT:
        raise UserError("The default profile is the project's settings: change them in the form.")
    try:
        f = config.preset_file(name, cfg_path)
    except KeyError as e:
        raise UserError(_message(e)) from None
    return _profile_view(cfg_path, name, f.read_text()) | {"builtin": _builtin(name), "own": f.parent != config.PRESETS_DIR}


def new_profile(cfg_path: Path, name: str, copy_of: str | None = None) -> dict:
    """Where a new profile starts (nothing is written yet): an empty one with its own output
    folder, or a copy of another."""
    _profile_name(name)
    if name in config.presets(cfg_path):
        raise UserError(f"There is a profile '{name}' already: edit it, or choose another name.")
    if not copy_of or copy_of == DEFAULT:
        text = NEW_PROFILE.format(name=name)
    else:
        doc = tomlkit.parse(read_profile(cfg_path, copy_of)["text"])
        _set(doc, "output.dir", f"processed/{name}")
        text = f"# {name}: a copy of {copy_of}.\n" + tomlkit.dumps(doc)
    return _profile_view(cfg_path, name, text) | {"builtin": False, "own": False, "new": True}


def _profile_name(name: str) -> None:
    if not PROFILE_NAME.match(name) or name == DEFAULT:
        raise UserError("A profile's name: lower-case letters, digits, - and _, starting with a letter or digit.")


def save_profile(cfg_path: Path, name: str, text: str, values: dict | None = None) -> None:
    """A profile written as presets/<name>.toml next to the settings file: the text, with the
    form's values on it. Checked on top of the project's settings first. Under a built-in
    one's name it is the project's version of it: the settings loader looks there first, and
    the built-in file is never touched."""
    _profile_name(name)
    if values:
        text = tomlkit.dumps(_with_values(text, values))
    try:
        layer = tomllib.loads(text)
        config.with_layer(config.load(cfg_path), f"profile {name}", layer)
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        raise UserError(_message(e).replace(f"profile {name}.", "")) from None
    target = cfg_path.parent / "presets" / f"{name}.toml"
    try:
        target.parent.mkdir(exist_ok=True)
        _replace(target, text, backup=False)
    except OSError as e:
        raise UserError(f"Cannot write {target}: {e.strerror}") from None


def delete_profile(cfg_path: Path, name: str) -> None:
    """The project's file of a profile removed. Its version of a built-in one: back to the
    built-in."""
    target = cfg_path.parent / "presets" / f"{name}.toml"
    if not PROFILE_NAME.match(name) or not target.exists():
        raise UserError(f"'{name}' is not one of the project's own profiles.")
    cfg = config.load(cfg_path)
    if name in cfg["input"]["presets"] and not _builtin(name):
        raise UserError(f"'{name}' is one of the profiles that run by default. Untick it in the settings first.")
    target.unlink()
