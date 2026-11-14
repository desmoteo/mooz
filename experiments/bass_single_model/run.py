"""Reference: the bass of one recording from the single best bass model alone (xlance's
BS-RoFormer, bs_bass_xlancer.ckpt: SDR 11.0 on the benchmark, SW 10.7; processed/logs/bass_models.log),
on the original WAV as it is: no rumble filter, no hum notches, no ensemble, no processing.

  processed/experiments/bass_single_model/<song>/bass.wav          as separated (float, whole recording)
  processed/experiments/bass_single_model/<song>/bass_stage.flac   the same, -12 dB and trimmed like the
                                                                   stage files (lines up with stages/<song>/)

  uv run python experiments/bass_single_model/run.py ["<song>.WAV"] [model.ckpt]
  (another model: its files get the model's name, e.g. bass_bs_mega_53stem_bass_mvsep.wav)
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mooz import config, separate, stages  # noqa: E402

wav = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "260924_230957_TrLR Stanza OK.WAV")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "bs_bass_xlancer.ckpt"
tag = "" if MODEL == "bs_bass_xlancer.ckpt" else "_" + Path(MODEL).stem
out = ROOT / "processed/experiments/bass_single_model" / wav.stem
x, fs = sf.read(wav, dtype="float32", always_2d=True)
steps = [{"input": "mix", "model": MODEL, "outputs": {"bass": "bass"}}]
scfg = config.load(ROOT / "rehearsal.toml")["stems"]
scfg = dict(scfg, backend="roformer", plan="single", plans={"single": {"steps": steps}}, cache=True, keep_models=False)
stems = separate.separate(x, fs, scfg, out / f"_cache{tag}", {"file": wav.name, "steps": steps, "untouched": True})
shutil.copy(stems["bass"], out / f"bass{tag}.wav")
bass = sf.read(out / f"bass{tag}.wav", dtype="float64", always_2d=True)[0]
rep = ROOT / "processed/reports" / f"{wav.stem}.json"
if rep.exists():
    trim = json.loads(rep.read_text())["trim"]
    bass = bass[int(round(trim["start_s"] * fs)):int(round(trim["end_s"] * fs))]
stages.StageWriter(out, "flac").write(f"bass{tag}_stage", bass, fs)
print(f"{out / f'bass{tag}.wav'}: {len(x) / fs:.1f} s, peak {20 * np.log10(np.abs(bass).max()):.1f} dBFS")
