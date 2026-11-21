"""Writing WAV/MP3 files and the loudness-matched A/B comparison."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

# Zoom recorders name files YYMMDD_HHMMSS_<track>[ name].WAV
_ZOOM_NAME = re.compile(r"^(\d{2})(\d{2})(\d{2})_\d{6}_[A-Za-z0-9]+(?:[ _-]+(.*))?$")


def song_info(path: Path) -> tuple[str, str]:
    """(title, date) from the file name; date is '' if it isn't a Zoom-style name."""
    m = _ZOOM_NAME.match(path.stem)
    if not m:
        return path.stem, ""
    yy, mm, dd, name = m.groups()
    return (name or path.stem).strip(), f"20{yy}-{mm}-{dd}"


def write_wav(path: Path, x: np.ndarray, fs: int, bit_depth: int) -> None:
    if not np.isfinite(x).all():
        raise ValueError(f"refusing to write {path.name}: the processed audio contains NaN/Inf samples")
    if bit_depth == 32:
        sf.write(path, x.astype(np.float32), fs, subtype="FLOAT")
        return
    if bit_depth not in (16, 24):
        raise ValueError(f"output.bit_depth must be 16, 24 or 32, not {bit_depth}")
    lsb = 2.0 ** -(bit_depth - 1)
    rng = np.random.default_rng(0)
    dithered = x + (rng.random(x.shape) - rng.random(x.shape)) * lsb  # TPDF, +-1 LSB
    sf.write(path, np.clip(dithered, -1.0, 1.0 - lsb), fs, subtype=f"PCM_{bit_depth}")


def _mp3_command(wav: Path, mp3: Path, bitrate: str, tags: dict[str, str]) -> list[str]:
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", bitrate]
    for key, value in tags.items():
        if value:
            cmd += ["-metadata", f"{key}={value}"]
    return [*cmd, str(mp3)]


def write_mp3(wav: Path, mp3: Path, bitrate: str, tags: dict[str, str]) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    subprocess.run(_mp3_command(wav, mp3, bitrate, tags), check=True)
    return True


def start_mp3(wav: Path, mp3: Path, bitrate: str, tags: dict[str, str], started: list) -> bool:
    """write_mp3 without the wait: the encoder is started and added to `started`, and the run goes
    on (a master's MP3 takes four seconds). finish(started) waits for them. False without ffmpeg."""
    if shutil.which("ffmpeg") is None:
        return False
    cmd = _mp3_command(wav, mp3, bitrate, tags)
    started.append((cmd, subprocess.Popen(cmd)))
    return True


def finish(started: list) -> None:
    """Wait for the encoders start_mp3 started; one that failed raises, as write_mp3 would have."""
    failed = [(cmd, proc.returncode) for cmd, proc in started if proc.wait() != 0]
    started.clear()
    if failed:
        raise subprocess.CalledProcessError(failed[0][1], failed[0][0])


def ab_mix(original: np.ndarray, processed: np.ndarray, fs: int, gain_orig_db: float, gain_proc_db: float, segment_s: float) -> np.ndarray:
    """Alternate original/processed every segment_s seconds (starting with the
    original), with 30 ms crossfades; both are pre-scaled to equal loudness."""
    n = len(processed)
    t = np.arange(n) / fs
    square = (np.floor(t / segment_s) % 2).astype(float)  # 0 = original, 1 = processed
    k = max(1, int(0.03 * fs))
    mix = np.convolve(square, np.ones(k) / k, mode="same")
    a = original[:n] * 10 ** (gain_orig_db / 20)
    b = processed * 10 ** (gain_proc_db / 20)
    return a * (1 - mix)[:, None] + b * mix[:, None]
