"""Each stem's chain, kept: a change to the mix or the master does not run the stem chains again.

The stem chains are two thirds of a take's time, and most changes leave them alone: a bus
setting, the balance, the loudness, another stem's chain. So what one stem's chain gave is
kept under a key made of everything that chain reads, and the next run with the same key
takes it from there, whichever preset asks:

  <recordings>/<[chains] dir>/<take>/<stem>/<key>/
      proc.wav       the processed stem, 32-bit float: the very samples the drum bus and the mix get
      meta.json      what the key is made of, and the stem's part of the report
      data.npz       the arrays: the tracks report's curves, the kit gate's gain
      taken.wav      what the bed clean-up took from the stem (the original follows it)
      stages/        the stem's stage files, as written

The key holds the code (every module that can change the sound, and the libraries' versions),
the stem's file (name, size and time; its content if this run made it), the trim, the files the
hit triggers come from, the stem's whole chain, its profile and its target, the clean-ups that
apply to it and the sample libraries it uses. A master made from kept chains is the same,
sample for sample, as one made without them.

Per stem the last [chains] keep results stay (two presets that differ in one chain need two
of that stem, one of the others); results of other code are removed, they can never be used.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf

# Modules that only read settings, report or write files: a change to them leaves every kept chain valid.
# pipeline.py is the stage order, of which only the part that makes the chains counts: it names those
# functions itself (uses), so work on the mix bus, the master and the files written keeps the chains.
_NOT_SOUND = {"cli.py", "report.py", "export.py", "progress.py", "pipeline.py"}
_ND = "__nd__"


def _code() -> str:
    h = hashlib.blake2b(digest_size=8)
    for f in sorted(Path(__file__).parent.glob("*.py")):
        if f.name not in _NOT_SOUND:
            h.update(f.name.encode())
            h.update(f.read_bytes())
    h.update(f"numpy {np.__version__} scipy {scipy.__version__}".encode())
    return h.hexdigest()


# Read when the run starts, with the modules themselves: what a run keeps is named after the code it runs.
_MODULES = _code()
_USED: list[str] = []


def uses(function):
    """Named by pipeline.py as it loads: this function of its is part of what makes a chain (its
    source goes into the code's name). Returns the function, to be used as a decorator."""
    import inspect

    _USED.append(inspect.getsource(function))
    return function


def code() -> str:
    """One name for the code that makes a chain: a change to any of it and nothing kept is used."""
    return hashlib.blake2b((_MODULES + "".join(sorted(_USED))).encode(), digest_size=8).hexdigest()


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=12).hexdigest()


def ident(path: Path, work: Path | None = None) -> list:
    """What tells a file from another one: name, size and time, as the separation cache does. A
    file this run made itself (in work: the recombined vocal) is new every time, so by its samples
    (not its bytes: a float WAV's header holds the time it was written)."""
    if work is not None and work in path.parents:
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        return ["samples", fs, list(x.shape), _digest(x.tobytes())]
    s = path.stat()
    return [path.parent.name, path.name, s.st_size, s.st_mtime_ns]


def library(folder: str) -> list | None:
    """A sample library as part of a key: its index, which changes when the library is built again."""
    idx = Path(os.path.expanduser(folder)) / "index.json"
    return ident(idx) if idx.exists() else None


def _plain(o):
    """For the key: arrays by their content, numbers as numbers."""
    if isinstance(o, np.ndarray):
        return {_ND: _digest(np.ascontiguousarray(o).tobytes()), "shape": list(o.shape), "dtype": str(o.dtype)}
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} in a chain's key")


def _pack(o, arrays: dict):
    """o as JSON, its arrays set aside in arrays (data.npz)."""
    if isinstance(o, np.ndarray):
        k = f"a{len(arrays)}"
        arrays[k] = o
        return {_ND: k}
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _pack(v, arrays) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_pack(v, arrays) for v in o]
    return o


def _unpack(o, arrays):
    if isinstance(o, dict):
        if set(o) == {_ND}:
            return arrays[o[_ND]]
        return {k: _unpack(v, arrays) for k, v in o.items()}
    if isinstance(o, list):
        return [_unpack(v, arrays) for v in o]
    return o


def _link(src: Path, dst: Path) -> None:
    """dst is src: a second name for the same file where the disk allows it, else a copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


@dataclass
class Kept:
    """One stem's chain as an earlier run left it."""
    proc: Path  # in this run's work folder
    stats: dict
    detail: dict
    kit_gr: np.ndarray | None
    taken: np.ndarray | None
    stages: dict[str, Path]  # stage name -> file


@dataclass
class Chains:
    """One take's kept chains. root: <recordings>/<[chains] dir>/<take>."""
    root: Path
    keep: int = 3
    reused: list[str] = field(default_factory=list)
    made: list[str] = field(default_factory=list)

    @staticmethod
    def key(parts: dict) -> tuple[str, str]:
        """(key, what it is made of as JSON) for everything one stem's chain reads."""
        text = json.dumps({"code": code()} | parts, sort_keys=True, default=_plain)
        return _digest(text.encode()), text

    def character(self, parts: dict, make) -> dict:
        """The song's character (stage 0), kept like a chain: it reads the recording and a few
        raw stems, and no settings. parts: what it reads; make: works it out."""
        f = self.root / "_character" / f"{self.key(parts)[0]}.json"
        try:
            return json.loads(f.read_text())
        except (OSError, ValueError):
            pass
        out = json.loads(json.dumps(make(), default=float))  # as it will be read back
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            for old in f.parent.glob("*.json"):  # one recording, one trim: the last one is the one in use
                old.unlink(missing_ok=True)
            f.write_text(json.dumps(out))
        except OSError:
            pass
        return out

    def has(self, stem: str, key: str) -> bool:
        """Whether load() would find this chain: its files are all there. Nothing is read or marked used."""
        d = self.root / stem / key
        try:
            m = json.loads((d / "meta.json").read_text())
            files = [d / "data.npz", d / "proc.wav", *(d / "stages" / f for f in m["stages"].values())]
            return all(f.exists() for f in files + ([d / "taken.wav"] if m["taken"] else []))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def load(self, stem: str, key: str, work: Path) -> Kept | None:
        """The stem's chain under key, if it is kept: its processed file linked into work, the
        rest in memory, so nothing here depends on the kept files afterwards."""
        d = self.root / stem / key
        meta = d / "meta.json"
        try:
            m = json.loads(meta.read_text())
            os.utime(meta)  # used now: the last to go when the stem's results are thinned
            with np.load(d / "data.npz") as z:
                arrays = {k: z[k] for k in z.files}
            taken = sf.read(d / "taken.wav", dtype="float32", always_2d=True)[0] if m["taken"] else None
            stages = {rel: d / "stages" / f for rel, f in m["stages"].items()}
            if not all(f.exists() for f in stages.values()):
                return None
            proc = work / f"_proc_{stem}.wav"
            _link(d / "proc.wav", proc)
        except (OSError, ValueError, KeyError, sf.LibsndfileError):  # not there, or half there: made again
            return None
        self.reused.append(stem)
        return Kept(proc, _unpack(m["stats"], arrays), _unpack(m["detail"], arrays),
                    arrays.get("kit_gr"), taken, stages)

    def store(self, stem: str, key: str, made_of: str, proc: Path, fs: int, stats: dict, detail: dict,
              kit_gr: np.ndarray | None, taken: np.ndarray | None, stages: dict[str, Path]) -> None:
        """Keep what the stem's chain gave. proc: its processed file (linked, not moved). A full
        disk or a folder that cannot be written costs the next run its time, never this run its
        master: said, and left."""
        self.made.append(stem)
        final = self.root / stem / key
        tmp = self.root / stem / f".tmp-{os.getpid()}-{key}"
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            _link(proc, tmp / "proc.wav")
            arrays: dict = {}
            m = {"code": code(), "stats": _pack(stats, arrays), "detail": _pack(detail, arrays),
                 "taken": taken is not None, "stages": {}, "made_of": json.loads(made_of)}
            if kit_gr is not None:
                arrays["kit_gr"] = kit_gr
            np.savez_compressed(tmp / "data.npz", **arrays)
            if taken is not None:
                sf.write(tmp / "taken.wav", taken, fs, subtype="FLOAT")
            for rel, f in stages.items():
                name = rel.replace("/", "__") + f.suffix
                _link(f, tmp / "stages" / name)
                m["stages"][rel] = name
            (tmp / "meta.json").write_text(json.dumps(m, default=float))
            shutil.rmtree(final, ignore_errors=True)  # a half-kept one found by load(), if any
            os.rename(tmp, final)
        except OSError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  note: the {stem} chain could not be kept for the next run ({e})", file=sys.stderr)
            return
        self._thin(stem, key)

    def _thin(self, stem: str, current: str) -> None:
        """Of this stem's kept results, the last `keep` used stay; those of other code go, and
        so does what a run that was stopped left half written (a day old)."""
        used = []
        for d in (self.root / stem).iterdir():
            meta = d / "meta.json"
            try:
                if d.name.startswith(".tmp-"):
                    if time.time() - d.stat().st_mtime > 86400:
                        shutil.rmtree(d, ignore_errors=True)
                    continue
                if d.name != current and json.loads(meta.read_text()).get("code") != code():
                    shutil.rmtree(d, ignore_errors=True)
                    continue
                used.append((meta.stat().st_mtime, d.name != current, d))
            except (OSError, ValueError):
                shutil.rmtree(d, ignore_errors=True)
        for _, _, d in sorted(used, key=lambda u: (u[1], -u[0]))[max(1, self.keep):]:
            shutil.rmtree(d, ignore_errors=True)
