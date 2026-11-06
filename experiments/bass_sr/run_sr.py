"""Bass stem -> downsampled hard -> AudioSR (haoheliu/audiosr_basic, versatile audio super-resolution, to
48 kHz) regenerates everything above the new Nyquist. Several downsampling levels, on an excerpt:

  processed/experiments/bass_sr/<song>/<label>/
    0_input        the separated stem (mono), the excerpt
    1_down         downsampled to <rate> (back at 48 kHz, so it lines up): what AudioSR gets
    2_audiosr      AudioSR's output as it is
    3_merged       the stem as it is below the cutoff, AudioSR's above (crossover at 0.9 x the cutoff)
  labels: sr<rate> per rate, and "orig" (AudioSR on the stem itself, no downsampling); resynth_*: the same on
  experiments/bass_bank's rebuilt bass (--source resynth)

AudioSR runs on 10.24 s chunks (its training length), 2.56 s overlap, raised-cosine crossfades.
First step: the stem brought to a proper level (bass_level.py: -16 LUFS, true peak <= -1 dBFS); each folder
written at one common gain (level.json). Runs in its own environment (AudioSR pins old numpy/librosa):

  . experiments/bass_sr/env.sh   (HF cache in the project)
  experiments/bass_sr/.venv-audiosr/bin/python experiments/bass_sr/run_sr.py [--song S] [--start 40 --dur 60]
      [--rates 2000,4000,8000,16000,orig] [--steps 50] [--guidance 3.5] [--tag ""]
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import bass_level  # noqa: E402

FS = 48000
CHUNK_S, OVERLAP_S = 10.24, 2.56
CUTOFF = [None]  # the cutoff AudioSR is told (Hz), per run


def downsample(x: np.ndarray, rate: int) -> np.ndarray:
    """To `rate` and back to 48 kHz: nothing above rate/2 left (polyphase, Kaiser window)."""
    from math import gcd

    g = gcd(rate, FS)
    lo = signal.resample_poly(x, rate // g, FS // g, window=("kaiser", 8.0))
    return lo, signal.resample_poly(lo, FS // g, rate // g, window=("kaiser", 8.0))[:len(x)]


def audiosr_chunks(model, x: np.ndarray, rate: int, args) -> np.ndarray:
    """AudioSR on x (given at `rate`), chunk by chunk, crossfaded; 48 kHz out, len(x)*48000/rate."""
    from audiosr import super_resolution

    n_out = int(round(len(x) * FS / rate))
    win, hop = int(CHUNK_S * rate), int((CHUNK_S - OVERLAP_S) * rate)
    ov_out = int(OVERLAP_S * FS)
    fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov_out))
    out, wsum = np.zeros(n_out + FS), np.zeros(n_out + FS)
    starts = list(range(0, max(len(x) - (win - hop), 1), hop))
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate(starts):
            seg = x[s:s + win]
            f = Path(tmp) / f"c{i}.wav"
            sf.write(f, seg.astype(np.float32), rate, subtype="FLOAT")
            y = np.asarray(super_resolution(model, str(f), seed=args.seed, guidance_scale=args.guidance,
                                            ddim_steps=args.steps, latent_t_per_second=12.8)).reshape(-1)
            a = int(round(s * FS / rate))
            m = min(len(y), int(round(len(seg) * FS / rate)), len(out) - a)
            y = y[:m]
            w = np.ones(m)
            if i > 0:
                w[:min(ov_out, m)] = fade[:min(ov_out, m)]
            if i < len(starts) - 1:
                w[-ov_out:] = fade[::-1][-min(ov_out, m):] if m >= ov_out else w[-ov_out:]
            out[a:a + m] += y * w
            wsum[a:a + m] += w
    return (out / np.maximum(wsum, 1e-9))[:n_out]


def load_model():
    """audiosr.build_model, but the 6.2 GB checkpoint memory-mapped on the CPU and freed before the model
    goes to the GPU (build_model loads it straight onto the GPU next to the model: out of memory on 8 GB)."""
    import gc

    import torch
    from audiosr.latent_diffusion.models.ddpm import LatentDiffusion
    from audiosr.pipeline import default_audioldm_config, download_checkpoint

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = default_audioldm_config("basic")
    config["model"]["params"]["device"] = dev
    model = LatentDiffusion(**config["model"]["params"])
    ckpt = torch.load(download_checkpoint("basic"), map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt
    gc.collect()
    model.eval()
    model = model.to(dev)
    print(f"model on {dev}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M params, "
          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="260924_230957_TrLR Stanza OK")
    ap.add_argument("--start", type=float, default=40.0)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--rates", default="2000,4000,8000,16000,orig")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--source", default="stem", choices=["stem", "resynth"],
                    help="stem: the separated bass; resynth: experiments/bass_bank's rebuilt bass (6_resynth)")
    a = ap.parse_args()
    import torch

    rep = json.loads((ROOT / "processed/reports" / f"{a.song}.json").read_text())
    if a.source == "stem":
        t0 = int(round((rep["trim"]["start_s"] + a.start) * FS))
        x = sf.read(ROOT / "processed/stages" / a.song / "1_raw/bass.wav", dtype="float64", always_2d=True,
                    start=t0, stop=t0 + int(a.dur * FS))[0].mean(axis=1)
    else:  # already trimmed, -12 dB
        t0 = int(round(a.start * FS))
        x = sf.read(ROOT / "processed/experiments/bass_bank" / a.song / "resynth/6_resynth.flac", dtype="float64",
                    always_2d=True, start=t0, stop=t0 + int(a.dur * FS))[0].mean(axis=1)
    x = x * bass_level.input_gain(x, FS)  # first: the stem at a proper level (bass_level.py)
    out = ROOT / "processed/experiments/bass_sr" / a.song
    # AudioSR works on a normalised level: give it a healthy one, put the stem's level back afterwards
    peak = float(np.abs(x).max())
    xs = x * (0.5 / peak)
    t = time.time()
    import torchaudio

    def _load(path, *args, **kw):  # torchaudio >= 2.9 reads through torchcodec (not installed): soundfile instead
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _load
    # AudioSR guesses the input's bandwidth (98.5 % of the energy below it); on a bass that is under 1 kHz,
    # which it takes for "empty" (then a filter at Nyquist: crash). Given here: the downsampling's cutoff, or
    # (no downsampling) its own guess without that rule
    import audiosr.pipeline as ap_
    from audiosr.lowpass import lowpass
    from audiosr.utils import _locate_cutoff_freq

    def _prep(dl_output):
        wf, sr_ = dl_output["waveform"], dl_output["sampling_rate"]
        c = CUTOFF[0] or _locate_cutoff_freq(dl_output["stft"], percentile=0.985) / 1024 * 24000
        c = float(np.clip(c, 300.0, 23000.0))
        y = lowpass(wf.numpy().squeeze(), highcut=c, fs=sr_, order=8, _type="butter")
        y = torch.FloatTensor(y.copy()).unsqueeze(0)[..., :wf.size(-1)]
        return {"waveform_lowpass": torch.nn.functional.pad(y, (0, wf.size(-1) - y.size(-1)))}

    ap_.lowpass_filtering_prepare_inference = _prep
    model = load_model()
    print(f"AudioSR loaded in {time.time() - t:.0f}s", flush=True)
    for r in a.rates.split(","):
        label = ("resynth_" if a.source == "resynth" else "") + ("orig" if r == "orig" else f"sr{int(r)}") + a.tag
        d = out / label
        t = time.time()
        if r == "orig":
            lo, back, rate, cutoff = xs, xs, FS, None
            CUTOFF[0] = None
        else:
            rate = int(r)
            lo, back = downsample(xs, rate)
            cutoff = rate / 2
            CUTOFF[0] = cutoff
        y = audiosr_chunks(model, lo, rate, a)[:len(x)]
        y = np.pad(y, (0, len(x) - len(y)))
        files = {"0_input": x, "1_down": back * peak / 0.5, "2_audiosr": y * peak / 0.5}
        if cutoff:
            xo = 0.9 * cutoff
            lpf = signal.butter(8, xo, "low", fs=FS, output="sos")
            hpf = signal.butter(8, xo, "high", fs=FS, output="sos")
            files["3_merged"] = signal.sosfiltfilt(lpf, x) + signal.sosfiltfilt(hpf, y * peak / 0.5)
        bass_level.write_folder(d, files, FS)
        print(f"{label}: {time.time() - t:.0f}s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
