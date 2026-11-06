"""The bass experiments' level: their first step, and the level their listening files are written at.

The separated bass stem sits about -18 LUFS, and the experiments used to write it the stage files' way
(12 dB down, mono): -31..-37 LUFS, too quiet to judge. Now:

  input_gain(x, fs)   the gain that first brings the stem to TARGET_LUFS (as it will be heard: mono on
                      both channels), its true peak at most TP_MAX_DB
  write_folder(...)   a folder's files at one common gain (comparisons stay fair: a louder version is
                      louder), lowered only if one of them would peak over TP_MAX_DB; level.json says how
  relevel(folder)     the same for results written the old way (12 dB down)

  uv run python experiments/bass_level.py      re-levels every bass experiment folder written the old way

Only numpy/scipy/soundfile and rehearsal_master.loudness: usable from the AudioSR / UniverSR environments too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rehearsal_master import loudness  # noqa: E402

TARGET_LUFS = -16.0
TP_MAX_DB = -1.0
OLD_LEVEL_DB = -12.0  # the stage files' offset the experiments used to write with


def _dual(x: np.ndarray) -> np.ndarray:
    return np.column_stack([x, x]) if x.ndim == 1 else x


def input_gain(x: np.ndarray, fs: float) -> float:
    """Linear gain bringing x (mono, as heard on both channels, or stereo) to TARGET_LUFS, true peak <= TP_MAX_DB."""
    y = _dual(x)
    g_db = TARGET_LUFS - loudness.integrated(y, fs)
    g_db = min(g_db, TP_MAX_DB - loudness.true_peak_db(y))
    return float(10 ** (g_db / 20))


def write_folder(folder: Path, files: dict[str, np.ndarray], fs: int, note: str = "") -> float:
    """Write {name: signal} as 24-bit FLAC (mono -> both channels) at one common gain: 1, lowered only so
    that no file's true peak exceeds TP_MAX_DB. Returns that gain (dB)."""
    folder.mkdir(parents=True, exist_ok=True)
    tp = max(loudness.true_peak_db(_dual(v)) for v in files.values())
    g_db = min(0.0, TP_MAX_DB - tp)
    g = 10 ** (g_db / 20)
    lufs = {}
    for name, v in files.items():
        y = _dual(v) * g
        sf.write(folder / f"{name}.flac", np.clip(y, -1, 1), fs, subtype="PCM_24")
        lufs[name] = round(loudness.integrated(y, fs), 1)
    (folder / "level.json").write_text(json.dumps(
        {"target_lufs": TARGET_LUFS, "true_peak_max_db": TP_MAX_DB, "folder_gain_db": round(g_db, 2),
         "lufs": lufs, "note": note or "the input brought to the target first; every file here at one common gain"},
        indent=1))
    return g_db


def relevel(folder: Path, reference: str | None = None, ref_gain: float | None = None, cap_each: bool = False) -> dict | None:
    """Re-level a folder written the old way (12 dB down): one common gain, from `reference` (the folder's
    input) or given as ref_gain (linear, re the old files), so the reference reaches TARGET_LUFS, lowered if
    any file would peak over TP_MAX_DB (cap_each: only that file, by as much as it needs; for folders of
    very different renders, where one would hold all the others down); with neither, each file on its own.
    One file in memory at a time.
    Rewrites the files in place; level.json records each file's gain re the old file."""
    lj = folder / "level.json"
    if lj.exists():
        return None  # already at the new level
    files = sorted(folder.glob("*.flac"))
    if not files:
        return None
    read = lambda f: sf.read(f, dtype="float64", always_2d=True)  # noqa: E731
    if ref_gain is None and reference and (folder / f"{reference}.flac").exists():
        x, fs = read(folder / f"{reference}.flac")
        ref_gain = input_gain(x, fs)
    gains, tps = {}, {}
    for f in files:
        x, fs = read(f)
        gains[f.stem] = ref_gain if ref_gain is not None else input_gain(x, fs)
        tps[f.stem] = loudness.true_peak_db(x * gains[f.stem])
    caps = {k: 10 ** (min(0.0, TP_MAX_DB - (v if cap_each else max(tps.values()))) / 20) for k, v in tps.items()}
    out = {}
    for f in files:
        x, fs = read(f)
        cap = caps[f.stem]
        y = x * gains[f.stem] * cap
        sf.write(f, np.clip(y, -1, 1), fs, subtype="PCM_24")
        out[f.stem] = {"gain_db_re_old_file": round(20 * np.log10(gains[f.stem] * cap), 2),
                       "lufs": round(loudness.integrated(y, fs), 1)}
    lj.write_text(json.dumps({"target_lufs": TARGET_LUFS, "true_peak_max_db": TP_MAX_DB,
                              "reference": reference if ref_gain is not None else "each file on its own",
                              "peak_cap": "each file on its own" if cap_each else "common",
                              "files": out}, indent=1))
    return out


def main() -> None:
    song = sys.argv[1] if len(sys.argv) > 1 else "260924_230957_TrLR Stanza OK"
    P = ROOT / "processed/experiments"
    jobs = []
    bb = P / "bass_bank" / song
    for d in sorted(bb.glob("*/")):  # medium and resynth have their 0_input; tight is levelled like medium
        jobs.append((d, "0_input"))
    for d in sorted((P / "bass_sr" / song).glob("*/")):
        jobs.append((d, "0_input"))
    nq = P / "bass_note_eq" / song
    jobs += [(nq / "raw", "0_input"), (nq / "post", "0_input"), (nq / "chain", "0_now*"),
             (nq / "sam", None), (nq / "docker", None), (P / "bass_single_model" / song, None)]
    for d, ref in jobs:
        g = None
        if d.name == "tight" and (bb / "medium/level.json").exists():  # no input of its own: medium's gain
            g = 10 ** (json.loads((bb / "medium/level.json").read_text())["files"]["0_input"]["gain_db_re_old_file"] / 20)
            ref = "0_input (medium's)"
        each = ref.endswith("*") if ref else False  # chain/: 24 very different renders, capped one by one
        r = relevel(d, ref.rstrip("*") if ref else ref, g, cap_each=each)
        if r:
            print(f"{d.relative_to(P)}: " + ", ".join(f"{k} {v['lufs']} LUFS" for k, v in list(r.items())[:4])
                  + (" ..." if len(r) > 4 else ""), flush=True)


if __name__ == "__main__":
    main()
