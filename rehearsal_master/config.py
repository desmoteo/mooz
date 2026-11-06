"""Settings: packaged defaults, deep-merged with an optional user TOML file."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

DEFAULTS_PATH = Path(__file__).with_name("defaults.toml")
PRESETS_DIR = Path(__file__).with_name("presets")


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


def load(path: Path | None, preset: str | None = None) -> dict:
    """defaults.toml, then the user's file, then a preset (presets/<name>.toml): the
    preset's style wins over general settings; per-song [files] sections still apply last."""
    with open(DEFAULTS_PATH, "rb") as f:
        cfg = tomllib.load(f)
    layers, files = [], {}
    if path is not None:
        with open(path, "rb") as f:
            user = tomllib.load(f)
        files = user.pop("files", {})
        layers.append(("rehearsal.toml", user))
    if preset:
        pf = PRESETS_DIR / f"{preset}.toml"
        if not pf.exists():
            raise KeyError(f"unknown preset '{preset}' (available: {', '.join(presets()) or 'none'})")
        with open(pf, "rb") as f:
            layers.append((f"preset {preset}", tomllib.load(f)))
    # Track chains: every layer's track settings first, then each track = track_defaults + the
    # track it is based_on (if any) + its own settings, so e.g. a change to [tracks.guitar]
    # also reaches guitar_left / guitar_right.
    raw = cfg.pop("tracks")
    for _, layer in layers:
        raw = _layer_tracks(raw, layer.get("tracks", {}))
        if "track_defaults" in layer:
            cfg["track_defaults"] = _merge(cfg["track_defaults"], layer["track_defaults"], "track_defaults")

    def expand(name: str) -> dict:
        over = dict(raw[name])
        base = over.pop("based_on", "")
        start = expand(base) if base else cfg["track_defaults"]
        return _merge(start, over, f"tracks.{name}")

    tracks = {name: expand(name) for name in raw}
    for _, layer in layers:
        cfg = _merge(cfg, {k: v for k, v in layer.items() if k not in ("tracks", "track_defaults")})
    cfg["tracks"] = tracks
    for key, over in files.items():  # validate per-song sections up front
        _merge(cfg, over, f'files."{key}"')
    cfg["files"] = files
    return cfg


def for_file(cfg: dict, filename: str) -> dict:
    """Settings for one file: apply every [files."..."] section whose key
    appears in the file name (case-insensitive)."""
    out = {k: v for k, v in cfg.items() if k != "files"}
    for key, over in cfg["files"].items():
        if key.lower() in filename.lower():
            out = _merge(out, over, f'files."{key}"')
    return out
