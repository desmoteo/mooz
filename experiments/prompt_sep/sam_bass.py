"""SAM Audio (large) for the bass only, two new ways (the plain "bass guitar" on the mix is
sam-large/<song>/1_stems/bass.flac, from sam_sep.py):

  refine_stem.flac         prompt "bass guitar" on the RoFormer ensemble's bass stem (stems cache):
                           SAM only has to keep the bass of what is already mostly bass, and
                           regenerates it without the separation's clank and smear
  mix_electric_bass.flac   prompt "electric bass guitar" on the mix

Both fed in mono (the same signal on L and R): the bass is mono, and SAM generates each channel
on its own (its plain stereo bass: L/R correlation 0.37). Outputs (-12 dB, whole recording, like
the stems cache) in processed/experiments/prompt_sep/sam-large-bass/<song>/.

    . ./env.sh && .venv-sam/bin/python sam_bass.py [song]
"""

import sys
import time

import numpy as np
import soundfile as sf
import torch

from common import HERE, OUT, read_input, write
from sam_sep import load, separate

song = sys.argv[1] if len(sys.argv) > 1 else "260924_230957_TrLR Stanza OK"
d = OUT / "sam-large-bass" / song
stem = HERE.parents[1] / "processed" / "stems_ensemble" / song / "bass.wav"
mix, fs = read_input(HERE / "inputs" / f"{song}.wav", None)
bass, fs2 = sf.read(stem, dtype="float32", always_2d=True)
assert fs == fs2 == 48000 and len(bass) == len(mix)
mono = lambda x: np.repeat(x.mean(axis=1, keepdims=True), 2, axis=1)  # noqa: E731
jobs = {"refine_stem": (mono(bass), "bass guitar"), "mix_electric_bass": (mono(mix), "electric bass guitar")}
t0 = time.time()
model, proc = load("large")
model._c2 = model._get_audio_features(torch.zeros(1, 1, proc.audio_hop_length, device="cuda")).shape[-1]
print(f"loaded in {time.time() - t0:.0f}s", flush=True)
for name, (x, prompt) in jobs.items():
    if (d / f"{name}.flac").exists():
        print(f"{name}: exists", flush=True)
        continue
    t = time.time()
    y = separate(model, proc, x, fs, prompt)
    write(d / f"{name}.flac", y, fs)
    print(f"{name} '{prompt}': {time.time() - t:.0f}s, L/R corr {np.corrcoef(y[:, 0], y[:, 1])[0, 1]:.2f}", flush=True)
print(f"done in {time.time() - t0:.0f}s", flush=True)
