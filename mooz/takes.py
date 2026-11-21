"""What was recorded: a take is one or more files from the same moment.

Zoom recorders write one file per track, named <YYMMDD>_<HHMMSS>_Tr<id>.WAV, optionally in a
folder per take ("261105_225442 Miramare/"):

  TrMic     the built-in stereo microphone: the room
  Tr<n>     input <n>: an external microphone or a mixer (the close tracks)
  TrLR      on recorders that also write TrMic, the recorder's own mix-down of all of them
            (measured on the band's recordings: 1.413 TrMic + 1.000 Tr2, residual -149 dB);
            without a TrMic it is the stereo pair itself

So the room track is TrMic when there is one, else TrLR ([multitrack] room). Any other audio file
is a take of its own, with only a room track.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff"}
_TRACK = re.compile(r"^(?P<base>\d{6}_\d{6})_Tr(?P<track>LR|Mic|\d+)(?P<rest>[ _-].*)?$")
_FOLDER = re.compile(r"^(?P<base>\d{6}_\d{6})(?:[ _-]+(?P<title>.*))?$")


@dataclass
class Take:
    name: str                # names its outputs and caches; unique among the takes
    root: Path               # outputs and caches go under this folder
    room: Path               # the stereo recording of the whole band
    close: dict[str, Path] = field(default_factory=dict)  # "Tr2" -> file
    mixdown: Path | None = None  # the recorder's own mix-down (TrLR next to a TrMic), if any
    title: str = ""
    date: str = ""


def _date(base: str) -> str:
    return f"20{base[:2]}-{base[2:4]}-{base[4:6]}"


def single(path: Path) -> Take:
    """A file on its own: the whole take is that one recording."""
    from . import export

    title, date = export.song_info(path)
    return Take(path.stem, path.parent, path, title=title, date=date)


def _from_group(base: str, files: dict[str, Path], rest: str, room: str) -> Take | None:
    """The take of one group of same-moment Zoom files. room: "auto" (TrMic if there is one,
    else TrLR), or the track to use."""
    close = {f"Tr{k}": p for k, p in sorted(files.items(), key=lambda kv: kv[0]) if k.isdigit()}
    use = room.removeprefix("Tr") if room != "auto" else ("Mic" if "Mic" in files else "LR")
    if use not in files:
        return None
    mixdown = files.get("LR") if use != "LR" else None
    folder = next(iter(files.values())).parent
    m = _FOLDER.match(folder.name)
    if m and m.group("base") == base:  # a folder per take: its name says the song, the outputs go one level up
        title = (m.group("title") or "").strip()
        return Take(folder.name, folder.parent, files[use], close, mixdown, title or base, _date(base))
    title = rest.strip(" _-")
    return Take(f"{base} {title}".strip(), folder, files[use], close, mixdown, title or base, _date(base))


def discover(paths: list[Path], skip_dirs: set[str], room: str = "auto") -> list[Take]:
    """The takes in the given files and folders (folders are searched at any depth, but for the
    ones named in skip_dirs: our own output). A Zoom file named alone brings its whole take."""
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES
                            and not skip_dirs & set(f.relative_to(p).parts[:-1]))
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"not found: {p}")
    groups: dict[tuple[Path, str], tuple[dict[str, Path], str]] = {}
    takes: list[Take] = []
    for f in found:
        if f.stem.endswith(("_AB", "_mix")):  # our own files, should they lie among the recordings
            continue
        m = _TRACK.match(f.stem)
        if not m:
            takes.append(single(f))
            continue
        if m.group("rest"):  # 260924_230957_TrLR Stanza OK.WAV: a stereo file named after its song, on its own
            takes.append(single(f))
            continue
        for g in sorted(f.parent.glob(f"{m.group('base')}_Tr*")):  # the whole take, whichever file was named
            mg = _TRACK.match(g.stem)
            if mg and not mg.group("rest") and g.suffix.lower() in AUDIO_SUFFIXES:
                groups.setdefault((f.parent, m.group("base")), ({}, ""))[0][mg.group("track")] = g
    for (_, base), (files, rest) in sorted(groups.items(), key=lambda kv: kv[0][1]):
        take = _from_group(base, files, rest, room)
        if take is None:
            raise ValueError(f"take {base}: no {room} track among {', '.join(sorted(files))}")
        takes.append(take)
    names = [t.name for t in takes]
    if len(set(names)) != len(names):
        raise ValueError(f"two takes share a name: {sorted(n for n in set(names) if names.count(n) > 1)}")
    return sorted(takes, key=lambda t: t.name)


def mixdown_formula(take: Take, seconds: float = 30.0) -> dict | None:
    """What the recorder's mix-down is made of, measured: least squares of TrLR on the other
    tracks over seconds from the middle of the take. None if there is no mix-down."""
    if take.mixdown is None:
        return None
    info = sf.info(take.mixdown)
    n = int(seconds * info.samplerate)
    a = max(0, info.frames // 2 - n // 2)
    read = lambda p: sf.read(p, dtype="float64", always_2d=True, start=a, stop=a + n)[0]  # noqa: E731
    lr, room = read(take.mixdown), read(take.room)
    close = {k: read(p) for k, p in take.close.items()}
    out = {}
    worst = -np.inf
    for c in range(lr.shape[1]):
        names = [("room", room[:, c % room.shape[1]])] + [(k, v[:, 0]) for k, v in close.items()]
        A = np.column_stack([v for _, v in names])
        w = np.linalg.lstsq(A, lr[:, c], rcond=None)[0]
        worst = max(worst, 10 * np.log10(np.sum((lr[:, c] - A @ w) ** 2) / (np.sum(lr[:, c] ** 2) + 1e-30) + 1e-30))
        out[c] = {k: round(float(g), 3) for (k, _), g in zip(names, w)}
    return {"gains": out[0], "residual_db": round(float(worst), 1)}
