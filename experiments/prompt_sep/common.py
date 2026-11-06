"""Shared by the SAM Audio and FlowSep runners: paths, file format, windowed overlap-add."""

from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "processed" / "experiments" / "prompt_sep"
LEVEL = 10 ** (-12 / 20)  # files are written 12 dB down, like stages/


def write(path: Path, y: np.ndarray, fs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(y * LEVEL, -1, 1), fs, subtype="PCM_24")


def read_stem(path: Path) -> np.ndarray:
    return sf.read(path, dtype="float32")[0] / LEVEL


def read_input(path: Path, excerpt):
    info = sf.info(path)
    start, frames = 0, -1
    if excerpt:
        start, frames = int(excerpt[0] * info.samplerate), int(excerpt[1] * info.samplerate)
    return sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)


def song_dirs(args_songs, excerpt, model_dir: str):
    songs = [Path(s) for s in args_songs] or sorted((HERE / "inputs").glob("*.wav"))
    for s in songs:
        s = s if s.exists() else HERE / "inputs" / f"{s.stem}.wav"
        tag = s.stem + (f"@{excerpt[0]:g}-{excerpt[0] + excerpt[1]:g}s" if excerpt else "")
        yield s, tag, OUT / model_dir / tag


def overlap_add(x: np.ndarray, win: int, hop: int, run) -> np.ndarray:
    """run(segment (m, 2), index) -> (m, 2), on windows of `win` samples every `hop`,
    cross-faded with a raised cosine over the overlap."""
    n, ov = len(x), win - hop
    starts = list(range(0, max(n - ov, 1), hop))
    out = np.zeros((n, x.shape[1]), np.float64)
    wsum = np.zeros(n, np.float64)
    fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov))
    for i, s in enumerate(starts):
        seg = x[s:s + win]
        m = len(seg)
        y = run(seg, i)
        w = np.ones(m)
        if i > 0:
            w[:min(ov, m)] = fade[:min(ov, m)]
        if i < len(starts) - 1 and m == win:
            w[-ov:] = fade[::-1]
        out[s:s + m] += y * w[:, None]
        wsum[s:s + m] += w
    return out / np.maximum(wsum, 1e-9)[:, None]
