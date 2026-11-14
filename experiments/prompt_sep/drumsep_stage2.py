"""Stage 2 for the prompt-based drums: the pipeline's own MDX23C DrumSep model (read-only
use of mooz.separate) on SAM's drums stem, since SAM gives back the whole kit
for every kit-piece prompt. Writes 2_drums_drumsep/ next to 2_drums/, plus _rest.flac
(the drums minus the pieces: the tamburello, if anywhere, and what DrumSep missed).

    uv run python experiments/prompt_sep/drumsep_stage2.py processed/experiments/prompt_sep/sam-large/<song> ...
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from mooz import config, separate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_stem, write  # noqa: E402

STEP = {"model": "MDX23C-DrumSep-aufr33-jarredou.ckpt", "input": "mix",
        "outputs": {"kick": "kick", "snare": "snare", "toms": "toms", "hh": "hihat", "ride": "cymbals", "crash": "cymbals"}}


def main() -> None:
    scfg = dict(config.load(ROOT / "rehearsal.toml")["stems"], plan="drumsep", plans={"drumsep": {"steps": [STEP]}}, cache=False)
    for d in map(Path, sys.argv[1:]):
        x = read_stem(d / "1_stems" / "drums.flac").astype(np.float64)
        fs = sf.info(d / "1_stems" / "drums.flac").samplerate
        work = d / "_drumsep_work"
        stems = separate.separate(x, fs, scfg, work, {"drumsep": STEP["model"]})
        out = d / "2_drums_drumsep"
        rest = x.copy()
        for name, f in stems.items():
            y = sf.read(f, dtype="float64", always_2d=True)[0]
            write(out / f"{name}.flac", y, fs)
            rest -= y
        write(out / "_rest.flac", rest, fs)
        shutil.rmtree(work)
        print(d.name, sorted(stems), flush=True)


if __name__ == "__main__":
    main()
