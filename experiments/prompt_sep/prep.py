"""Write the separation input of each song: the whole recording with the pipeline's own
rumble filter and hum notches (the same signal the RoFormer chain separates, so the new
stems line up sample for sample with processed/stems/). Read-only use of rehearsal_master.

    uv run python experiments/prompt_sep/prep.py [song.WAV ...]
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from rehearsal_master import cleanup, config, loudness, trim

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "inputs"


def prepare(path: Path, cfg: dict) -> Path:
    out = OUT / f"{path.stem}.wav"
    if out.exists():
        return out
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    x = x[:, :2]
    before, _ = loudness.summary(x, fs)
    t = cfg["trim"]
    start, _ = trim.find_bounds(x, fs, before["integrated_lufs"], t["threshold_rel_db"],
                                t["min_active_s"], t["pre_roll_s"], t["post_roll_s"])
    if cfg["highpass"]["enabled"]:
        hp = cfg["highpass"]
        x = signal.sosfilt(signal.butter(hp["order"], hp["freq_hz"], "high", fs=fs, output="sos"), x, axis=0)
    if cfg["hum"]["enabled"]:
        lead_in = (int(0.5 * fs), start - int(0.2 * fs)) if start > int(2.2 * fs) else None
        hum = cleanup.detect_hum(x, fs, cfg["hum"], lead_in)
        if hum["harmonics"]:
            x = cleanup.remove_hum(x, fs, hum, cfg["hum"])
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(out, x.astype(np.float32), fs, subtype="FLOAT")
    return out


def main() -> None:
    cfg = config.load(ROOT / "rehearsal.toml")
    files = [Path(a) for a in sys.argv[1:]] or sorted(ROOT.glob("*.WAV"))
    for f in files:
        print(prepare(f, config.for_file(cfg, f.name)), flush=True)


if __name__ == "__main__":
    main()
