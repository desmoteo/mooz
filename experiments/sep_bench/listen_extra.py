"""Listening test on the band's own recordings for what the benchmark cannot score well:
the tambourine (tamburello) and the harmonica, from MVSep's 53-stem BS-RoFormer and
yolkispaliks' percussion MelBand-RoFormer (candidates.py), run on the whole recording as
the pipeline separates it (rumble filter and hum notches: experiments/prompt_sep/inputs/).

Writes processed/experiments/new_models/<song>/<model>/<stem>.flac (12 dB down, as the
stage files), via the pipeline's own separation code.

  uv run python experiments/sep_bench/listen_extra.py [song-substring ...]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import custom_models  # noqa: E402

from rehearsal_master import config, separate  # noqa: E402

INPUTS = ROOT / "experiments" / "prompt_sep" / "inputs"
OUT = ROOT / "processed" / "experiments" / "new_models"
STEP = {"input": "mix", "models": [
    {"model": "bs_mega_53stem_full_mvsep.ckpt", "segment_size": 512, "chunk_s": 20.0, "outputs": {
        "tambourine": "tambourine", "percussion": "percussion", "harmonica": "harmonica"}},
    {"model": "mbr_percussion_yolkispaliks.ckpt", "outputs": {"percussions": "percussion_yolkispaliks"}},
]}


def main() -> None:
    songs = sorted(INPUTS.glob("*.wav"))
    if sys.argv[1:]:
        songs = [f for f in songs if any(a.lower() in f.name.lower() for a in sys.argv[1:])]
    scfg = config.load(ROOT / "rehearsal.toml")["stems"]
    scfg = dict(scfg, plan="listen", plans={"listen": {"steps": [STEP]}}, cache=True, keep_models=True,
                custom_models=scfg["custom_models"] | custom_models())
    for f in songs:
        x, fs = sf.read(f, dtype="float32", always_2d=True)
        work = OUT / f.stem / "_cache"
        separate.separate(x, fs, scfg, work, {"listen": f.name, "step": STEP})
        for d in (work / "models" / "1_mix").iterdir():  # <song>/<model>/<stem>.flac, the cache's WAVs dropped
            dst = OUT / f.stem / d.name
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(d, dst)
        shutil.copy(work / "models" / "index.json", OUT / f.stem / "index.json")
        print(f.stem, sorted(p.name for p in (OUT / f.stem).iterdir()), flush=True)


if __name__ == "__main__":
    main()
