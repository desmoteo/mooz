"""Writing every stage of the pipeline for listening:

  stages/<song>/character.json      stage 0: what kind of song it is, and why
  stages/<song>/1_raw/              stage 1: link to the separated stems (whole recording)
  stages/<song>/2_stems/<stem>      stage 2: every stem cleaned and studio-processed
  stages/<song>/3_drum_cleanup/     stage 3: drum stems after the clean-up (buzz, bleed, gates),
                                    and <stem>_removed: what the clean-up took out
  stages/<song>/4_drum_bus          stage 4: the drum kit mixed as one instrument
  stages/<song>/5_premix/           stage 5: every part as it goes into the mix; they add up to it
  stages/<song>/5_mix               stage 5: the full mix before mastering (loudness/limiter)

All stage files share one fixed gain offset (GAIN_DB), so stems and buses keep
their real levels relative to each other (they line up in a DAW) and the raw
drum peaks, which go above full scale, do not clip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

GAIN_DB = -12.0


class StageWriter:
    def __init__(self, root: Path | None, fmt: str):
        self.root = root if fmt != "off" else None
        self.fmt = fmt
        self.written: list[str] = []
        self.files: dict[str, Path] = {}  # stage name -> the file it is in
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def __bool__(self) -> bool:
        return self.root is not None

    def write(self, rel: str, x: np.ndarray, fs: int) -> None:
        if not self.root:
            return
        path = self.root / f"{rel}.{'flac' if self.fmt == 'flac' else 'wav'}"
        path.parent.mkdir(parents=True, exist_ok=True)
        y = x * 10 ** (GAIN_DB / 20)
        peak = float(np.abs(y).max()) if y.size else 0.0
        if peak >= 1.0:  # should not happen with the offset; never clip a listening copy
            y = y / (peak * 1.0001)
        if self.fmt == "mp3" and shutil.which("ffmpeg"):
            tmp = path.with_suffix(".tmp.wav")
            sf.write(tmp, y.astype(np.float32), fs, subtype="FLOAT")
            path = path.with_suffix(".mp3")
            path.unlink(missing_ok=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-codec:a", "libmp3lame", "-b:a", "256k",
                            str(path)], check=True)
            tmp.unlink()
        else:
            # never written through: the file there may be a kept chain's, under a second name (adopt)
            path.unlink(missing_ok=True)
            sf.write(path, y, fs, subtype="PCM_24", format="FLAC" if self.fmt == "flac" else "WAV")
        self.written.append(rel)
        self.files[rel] = path

    def adopt(self, rel: str, src: Path) -> None:
        """A stage file an earlier run wrote (chaincache.py), put in its place: a second name for
        the same file where the disk allows it, else a copy."""
        if not self.root:
            return
        path = self.root / f"{rel}{src.suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        try:
            os.link(src, path)
        except OSError:
            shutil.copy2(src, path)
        self.written.append(rel)
        self.files[rel] = path

    def json(self, rel: str, data: dict) -> None:
        if self.root:
            (self.root / f"{rel}.json").write_text(json.dumps(data, indent=2, default=float))

    def link_raw(self, stems_dir: Path, name: str = "1_raw") -> None:
        if not self.root:
            return
        link = self.root / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(stems_dir, self.root), target_is_directory=True)

    def readme(self, song: str, info: dict) -> None:
        if not self.root:
            return
        lines = [f"Stages of {song}", "",
                 f"All files are {GAIN_DB:+.0f} dB from their real level (one offset for all, so they line up).", "",
                 "character.json     stage 0: the song's character, style and the reasons",
                 "1_raw/             stage 1: the separated stems, untouched (whole recording, not trimmed)",
                 "1_raw/models/      stage 1: every separation model's own outputs, before averaging (12 dB down,",
                 "                   see models/index.json), <step>_<input>/<model>/<stem>.flac and <step>_<input>/ensemble/",
                 "1_raw_Tr<n>/       stage 1: a close track's separated stems (if the take has close tracks)",
                 "1_merged/          stage 1: the stems the chains start from: the room's, with the close stems lined up and added",
                 "2_stems/           stage 2: each stem cleaned and studio-processed (no level/placement yet)",
                 "3_drum_cleanup/    stage 3: drum stems after the clean-up, and *_removed = what it took out",
                 "4_drum_bus         stage 4: the drum kit mixed as one instrument",
                 "5_premix/          stage 5: every part of the mix as it goes in, balanced and placed, the original recording too;",
                 "                   together they are the mix before the bus chain, so a DAW can remix them",
                 "5_mix              stage 5: the full mix before loudness and limiting (the master is next to stages/)",
                 "", json.dumps(info, indent=2, default=float)]
        (self.root / "README.txt").write_text("\n".join(lines))
