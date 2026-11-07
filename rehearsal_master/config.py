"""Settings: packaged defaults, deep-merged with the layers that override them.

Order of precedence (each layer overrides the previous):

  defaults.toml < rehearsal.toml < --preset < the song's style < [files."..."] < command line

load() resolves the first three (plus the command line) once; for_file() adds the song's
style and its [files] sections, then puts the command line back on top. Keys are validated
against the defaults (a typo is an error); tables merge, lists replace. Track chains are
expanded last, from every layer's raw [tracks] tables: track_defaults, then the track a
track is based_on, then its own settings, so a per-song change to [tracks.guitar] also
reaches guitar_left and guitar_right."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

DEFAULTS_PATH = Path(__file__).with_name("defaults.toml")
PRESETS_DIR = Path(__file__).with_name("presets")
_PRIVATE = ("_layers", "_cli", "files")  # kept in the resolved settings for for_file(); not settings


def presets() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.toml"))


def _merge(base: dict, over: dict, path: str = "") -> dict:
    """Merge over into base; every key must already exist (catches typos)."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        name = f"{path}.{key}" if path else key
        if key not in out:
            raise KeyError(f"unknown setting '{name}' (all settings are listed in {DEFAULTS_PATH})")
        if isinstance(out[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"setting '{name}' must be a table")
            out[key] = _merge(out[key], value, name)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _deep(a: dict, b: dict) -> dict:
    """b merged into a, tables recursively (unvalidated: tracks are checked when expanded)."""
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _layer_tracks(raw: dict, over: dict) -> dict:
    """Combine track overrides per track and key (validated later, when expanded); nested
    tables merge, so a preset can change e.g. augment.library.mix and keep the rest."""
    out = copy.deepcopy(raw)
    for name, t in over.items():
        out[name] = _deep(out.get(name, {}), t)
    return out


def _resolve(layers: list[tuple[str, dict]]) -> dict:
    """The defaults with every layer merged in order; tracks expanded at the end."""
    with open(DEFAULTS_PATH, "rb") as f:
        cfg = tomllib.load(f)
    raw = cfg.pop("tracks")
    for name, layer in layers:
        raw = _layer_tracks(raw, layer.get("tracks", {}))
        if "track_defaults" in layer:
            cfg["track_defaults"] = _merge(cfg["track_defaults"], layer["track_defaults"], f"{name}: track_defaults")
        cfg = _merge(cfg, {k: v for k, v in layer.items() if k not in ("tracks", "track_defaults")}, name)

    def expand(track: str) -> dict:
        over = dict(raw[track])
        base = over.pop("based_on", "")
        start = expand(base) if base else cfg["track_defaults"]
        return _merge(start, over, f"tracks.{track}")

    cfg["tracks"] = {track: expand(track) for track in raw}
    return cfg


def _layer_name(name: str, path: str) -> str:
    return f'{name} ({path})' if path else name


def load(path: Path | None, preset: str | None = None, overrides: dict | None = None) -> dict:
    """defaults.toml, then the user's file, then a preset (presets/<name>.toml), then the
    command line's overrides. The per-song [files] sections are validated here and applied,
    together with the song's style, by for_file()."""
    layers: list[tuple[str, dict]] = []
    files: dict = {}
    if path is not None:
        with open(path, "rb") as f:
            user = tomllib.load(f)
        files = user.pop("files", {})
        layers.append((str(path), user))
    if preset:
        pf = PRESETS_DIR / f"{preset}.toml"
        if not pf.exists():
            raise KeyError(f"unknown preset '{preset}' (available: {', '.join(presets()) or 'none'})")
        with open(pf, "rb") as f:
            layers.append((f"preset {preset}", tomllib.load(f)))
    cli = overrides or {}
    cfg = _resolve(layers + [("command line", cli)])
    for key, over in files.items():  # validate per-song sections up front
        _resolve(layers + [(f'files."{key}"', over)])
    cfg["_layers"], cfg["_cli"], cfg["files"] = layers, cli, files
    return cfg


def for_file(cfg: dict, filename: str, style: str | None = None) -> dict:
    """Settings for one song: the loaded layers, then the song's style table
    ([character.styles.<style>], if given), then every [files."..."] section whose key
    appears in the file name (case-insensitive), then the command line again; tracks are
    expanded last, so per-song track changes reach the tracks based on them."""
    layers = list(cfg["_layers"])
    if style is not None and style in cfg["character"]["styles"]:
        layers.append((f"character.styles.{style}", cfg["character"]["styles"][style]))
    for key, over in cfg["files"].items():
        if key.lower() in filename.lower():
            layers.append((f'files."{key}"', over))
    out = _resolve(layers + [("command line", cfg["_cli"])])
    out["_layers"], out["_cli"], out["files"] = cfg["_layers"], cfg["_cli"], cfg["files"]
    return out
