"""Prompt-based stems with FlowSep-hive (AlayaLab), on the prepared inputs.

FlowSep works at 16 kHz on a 64-band mel spectrogram up to 8 kHz and resynthesises the
target with a BigVGAN vocoder: its stems have nothing above 8 kHz. They are resampled to
48 kHz only so that they line up with the other stems. The model flows from the mixture's
latent to the target's on 10.24 s windows (here overlapping by 2.24 s and cross-faded);
each window is normalised to 0.5 peak on the way in and scaled back on the way out.
L and R go through as a batch of two with the same window gain.

    .venv-flowsep/bin/python flowsep_sep.py [--stage 1|2|all] [--excerpt START DUR] [song ...]
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

from common import HERE, overlap_add, read_input, read_stem, song_dirs, write

sys.path.insert(0, str(HERE / "Hive" / "models" / "flowsep"))
from latent_diffusion.util import instantiate_from_config  # noqa: E402

FS = 16000
CHUNK_IN = 163840  # 10.24 s, the model's window
HOP = 128000  # 8 s
STAGE1 = {  # prompts named after the Hive ontology's classes
    "vocals": "singing",
    "bass": "bass guitar",
    "guitars": "electric guitar",
    "drums": "drum kit",
    "other": "harmonica",
}
STAGE2 = {  # the ontology has drum kit, cymbal and tambourine, not the other pieces
    "kick": "kick drum",
    "snare": "snare drum",
    "toms": "tom-tom drum",
    "hihat": "hi-hat",
    "cymbals": "cymbal",
    "percussion": "tambourine",
}


def load():
    snap = glob.glob(str(Path(os.environ["HF_HUB_CACHE"]) / "models--AlayaLab--FlowSep-hive" / "snapshots" / "*"))[0]
    cfg = yaml.load(open(f"{snap}/config.yaml"), Loader=yaml.FullLoader)
    here = os.getcwd()
    os.chdir(snap)  # the config names vae.ckpt relative to the working directory
    try:
        model = instantiate_from_config(cfg["model"])
    finally:
        os.chdir(here)
    sd = torch.load(f"{snap}/flowsep_hive.ckpt", map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    model.load_state_dict(sd, strict=True)
    del sd
    import utilities.audio as Audio
    p = cfg["preprocessing"]
    stft = Audio.stft.TacotronSTFT(p["stft"]["filter_length"], p["stft"]["hop_length"], p["stft"]["win_length"],
                                   p["mel"]["n_mel_channels"], p["audio"]["sampling_rate"],
                                   p["mel"]["mel_fmin"], p["mel"]["mel_fmax"])
    frames = int(p["audio"]["duration"] * p["audio"]["sampling_rate"] / p["stft"]["hop_length"])
    return model.eval().cuda(), stft, frames


def mel(wav: np.ndarray, stft, frames: int) -> torch.Tensor:
    import utilities.audio as Audio
    m, _, _ = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav), stft)
    m = torch.FloatTensor(m.T)
    if len(m) < frames:
        m = torch.nn.functional.pad(m, (0, 0, 0, frames - len(m)))
    return m[:frames]


def separate(model, stft, frames, x16: np.ndarray, prompt: str, scratch: Path) -> np.ndarray:
    """x16: (n, 2) at 16 kHz -> the prompted source, (n, 2) at 16 kHz."""

    def run(seg, i):
        m = len(seg)
        seg = seg - seg.mean(0)
        peak = float(np.abs(seg).max()) + 1e-8
        g = 0.5 / peak
        w = np.zeros((2, CHUNK_IN), np.float32)
        w[:, :m] = (seg * g).T
        mels = torch.stack([mel(w[c], stft, frames) for c in range(2)])
        batch = {
            "fname": ["L.wav", "R.wav"], "text": [prompt] * 2, "caption": [prompt] * 2,
            "waveform": torch.zeros(2, 1, CHUNK_IN), "log_mel_spec": torch.zeros(2, frames, 64),
            "sampling_rate": torch.tensor([FS] * 2), "label_vector": torch.zeros(2, 527),
            "stft": torch.zeros(2, frames, 512), "mixed_waveform": torch.from_numpy(w[:, None]),
            "mixed_mel": mels,
        }
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        torch.manual_seed(0)
        with torch.no_grad():
            y = model.generate_sample([batch], name=str(scratch), unconditional_guidance_scale=1.0,
                                      ddim_steps=20, n_gen=1, save=False, save_mixed=False)
        y = np.asarray(y if isinstance(y, np.ndarray) else y.cpu().numpy()).reshape(2, -1)
        return y[:, :m].T / g

    return overlap_add(x16, CHUNK_IN, HOP, run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("songs", nargs="*")
    ap.add_argument("--stage", default="all", choices=["1", "2", "all"])
    ap.add_argument("--excerpt", nargs=2, type=float, metavar=("START", "DUR"))
    ap.add_argument("--only", help="comma-separated stems to (re)do")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "flowsep_scratch"

    t0 = time.time()
    model, stft, frames = load()
    print(f"loaded FlowSep-hive in {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    up = lambda y: torchaudio.functional.resample(torch.from_numpy(y.T.astype(np.float32)), FS, 48000).numpy().T
    down = lambda y: torchaudio.functional.resample(torch.from_numpy(y.T.copy()), 48000, FS).numpy().T

    for song, tag, d in song_dirs(a.songs, a.excerpt, "flowsep-hive"):
        x, fs = read_input(song, a.excerpt)
        assert fs == 48000
        print(f"{tag}: {len(x) / fs:.0f}s", flush=True)
        write(d / "0_mix.flac", x, fs)
        fit = lambda y: np.pad(y, ((0, max(0, len(x) - len(y))), (0, 0)))[:len(x)]
        if a.stage in ("1", "all"):
            x16 = down(x)
            for stem, prompt in STAGE1.items():
                if only and stem not in only:
                    continue
                t = time.time()
                write(d / "1_stems" / f"{stem}.flac", fit(up(separate(model, stft, frames, x16, prompt, scratch))), fs)
                print(f"  1 {stem:<10} '{prompt}'  {time.time() - t:.0f}s", flush=True)
            parts = [d / "1_stems" / f"{s}.flac" for s in STAGE1]
            if all(p.exists() for p in parts):
                write(d / "1_stems" / "_missed.flac", x - sum(read_stem(p) for p in parts), fs)
        if a.stage in ("2", "all"):
            drums16 = down(read_stem(d / "1_stems" / "drums.flac"))
            for piece, prompt in STAGE2.items():
                if only and piece not in only:
                    continue
                t = time.time()
                write(d / "2_drums" / f"{piece}.flac", fit(up(separate(model, stft, frames, drums16, prompt, scratch))), fs)
                print(f"  2 {piece:<10} '{prompt}'  {time.time() - t:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
