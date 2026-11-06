"""Prompt-based stems with Meta's SAM Audio (text prompts), on the prepared inputs.

Stage 1 splits the mix (vocals, bass, guitars, drums with percussion, other); stage 2
splits stage 1's drums into kit pieces. Stereo: L and R go through the model as a batch
of two with the same starting noise, so the two generated channels agree. Long songs are
cut into 20 s windows that overlap by 4 s and are cross-faded.

Outputs go to processed/experiments/prompt_sep/sam-<size>/<song>/ as 24-bit FLAC at
-12 dB (like stages/), so they line up with the RoFormer stems in a DAW.

    .venv-sam/bin/python sam_sep.py --model base [--stage 1|2|all] [--excerpt START DUR] [song ...]
"""

import argparse
import glob
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from common import overlap_add, read_input, read_stem, song_dirs, write
from sam_audio import SAMAudioProcessor
from sam_audio.model.config import SAMAudioConfig

mm = importlib.import_module("sam_audio.model.model")

STAGE1 = {  # stem -> text prompt (lowercase noun phrases, as the model was trained)
    "vocals": "vocals",
    "bass": "bass guitar",
    "guitars": "electric guitar",
    "drums": "drums",
    "other": "harmonica",
}
STAGE2 = {  # kit piece -> prompt, separated from stage 1's drums
    "kick": "kick drum",
    "snare": "snare drum",
    "toms": "tom-tom drum",
    "hihat": "hi-hat",
    "cymbals": "cymbals",
    "percussion": "tambourine",
}
WIN_S, OVERLAP_S = 20.0, 4.0


class _NoVision(torch.nn.Module):
    """Stands in for the 671M-parameter video encoder: only its width is used for audio."""

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim


def _dit_bf16(cfg):
    """The diffusion transformer in bf16 (its RoPE and timestep tables stay fp32)."""
    torch.set_default_dtype(torch.bfloat16)
    try:
        return _DiT(cfg)
    finally:
        torch.set_default_dtype(torch.float32)


_DiT = mm.DiT
mm.DiT = _dit_bf16
mm.PerceptionEncoder = _NoVision


def load(size: str, spans: bool = False):
    repo = f"facebook/sam-audio-{size}"
    snap = glob.glob(str(Path(os.environ["HF_HUB_CACHE"]) / f"models--facebook--sam-audio-{size}" / "snapshots" / "*"))
    if not snap:
        from huggingface_hub import snapshot_download
        snap = [snapshot_download(repo)]
    snap = snap[0]
    cfg = json.load(open(f"{snap}/config.json"))
    cfg.update(visual_ranker=None, text_ranker=None)  # unused here
    if not spans:
        cfg["span_predictor"] = None
    model = mm.SAMAudio(SAMAudioConfig(**cfg))
    sd = torch.load(f"{snap}/checkpoint.pt", mmap=True, weights_only=True, map_location="cpu")
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("vision_encoder.")}, strict=True)
    del sd
    model.eval()
    # the text encoder runs on the CPU (once per prompt); everything else on the GPU
    text_encoder = model.text_encoder
    del model.text_encoder
    span_predictor = getattr(model, "span_predictor", None)
    if span_predictor is not None:  # too big for the GPU next to the model: CPU, once per window
        del model.span_predictor
    model.cuda()
    object.__setattr__(model, "text_encoder", _CpuText(text_encoder))
    if span_predictor is not None:
        def predict(**kw):
            with torch.inference_mode():
                return span_predictor(**{k: v.cpu() if torch.is_tensor(v) else v for k, v in kw.items()})
        object.__setattr__(model, "span_predictor", predict)
    dit_forward = model.transformer.forward

    def forward(*a, **k):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return dit_forward(*a, **k).float()

    model.transformer.forward = forward

    # The 48 kHz codec's activations are large: run it one signal at a time, and decode
    # only the targets (the model also generates the residual, which is not used here).
    codec = model.audio_codec
    encode, decode = codec.forward, codec.decode

    def encode_chunked(w, step=50, margin=25):
        hop = codec.hop_length
        w = codec._pad(w)
        F, parts = w.shape[-1] // hop, []
        for f0 in range(0, F, step):
            a, b = max(0, f0 - margin), min(F, f0 + step + margin)
            z = encode(w[..., a * hop:b * hop])
            parts.append(z[..., f0 - a:min(F, f0 + step) - a])
        return torch.cat(parts, -1)

    codec.forward = lambda w: torch.cat([encode_chunked(w[i:i + 1]) for i in range(len(w))])

    def decode_chunked(z, step=50, margin=25):
        """Decode 2 s of latents at a time with 1 s of context on each side (the decoder is
        convolutional, so the trimmed pieces join seamlessly)."""
        hop, T, parts = codec.hop_length, z.shape[-1], []
        for t0 in range(0, T, step):
            a, b = max(0, t0 - margin), min(T, t0 + step + margin)
            y = decode(z[..., a:b])
            parts.append(y[..., (t0 - a) * hop:(min(T, t0 + step) - a) * hop])
            torch.cuda.empty_cache()
        return torch.cat(parts, -1)

    def decode_targets(z):
        out = None
        for i in range(0, len(z), 2):  # rows alternate target, residual
            y = decode_chunked(z[i:i + 1])
            if out is None:
                out = y.new_zeros((len(z),) + y.shape[1:])
            out[i] = y[0]
        return out

    codec.decode = decode_targets
    return model, SAMAudioProcessor.from_pretrained(snap)


class _CpuText:
    def __init__(self, enc):
        self.enc, self.cache = enc, {}

    def __call__(self, texts):
        key = tuple(texts)
        if key not in self.cache:
            with torch.inference_mode():
                f, m = self.enc(list(texts))
            self.cache[key] = (f.cuda(), m.cuda())
        return self.cache[key]


def separate(model, proc, x: np.ndarray, fs: int, prompt: str, seed: int = 0, spans: bool = False) -> np.ndarray:
    """x: (n, 2) at 48 kHz -> the prompted source, (n, 2)."""
    assert fs == proc.audio_sampling_rate

    def run(seg, i):
        chans = [torch.from_numpy(np.ascontiguousarray(seg[:, c], np.float32))[None] for c in range(2)]
        batch = proc(audios=chans, descriptions=[prompt, prompt]).to("cuda")
        g = torch.Generator(device="cuda").manual_seed(seed + i)
        noise = torch.randn(1, int(batch.sizes.max()), model._c2, device="cuda", generator=g)
        with torch.inference_mode():
            res = model.separate(batch, noise=noise.expand(2, -1, -1).contiguous(), predict_spans=spans)
        return np.stack([res.target[c].float().cpu().numpy().reshape(-1)[:len(seg)] for c in range(2)], 1)

    return overlap_add(x, int(WIN_S * fs), int((WIN_S - OVERLAP_S) * fs), run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("songs", nargs="*")
    ap.add_argument("--model", default="base", choices=["small", "base", "large", "small-tv", "base-tv", "large-tv"])
    ap.add_argument("--stage", default="all", choices=["1", "2", "all"])
    ap.add_argument("--excerpt", nargs=2, type=float, metavar=("START", "DUR"))
    ap.add_argument("--only", help="comma-separated stems to (re)do")
    ap.add_argument("--kit-from-mix", action="store_true", help="stage 2 prompts the mix, not stage 1's drums")
    ap.add_argument("--spans", action="store_true", help="let the span predictor find where each prompt sounds")
    ap.add_argument("--tag", default="", help="suffix for the output folder (variants of one model)")
    ap.add_argument("--prompt", action="append", default=[], metavar="STEM=TEXT", help="override a prompt")
    a = ap.parse_args()
    for o in a.prompt:
        stem, text = o.split("=", 1)
        (STAGE1 if stem in STAGE1 else STAGE2)[stem] = text
    only = set(a.only.split(",")) if a.only else None

    t0 = time.time()
    model, proc = load(a.model, a.spans)
    model._c2 = model._get_audio_features(torch.zeros(1, 1, proc.audio_hop_length, device="cuda")).shape[-1]
    print(f"loaded sam-audio-{a.model} in {time.time() - t0:.0f}s, "
          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    for song, tag, d in song_dirs(a.songs, a.excerpt, f"sam-{a.model}{a.tag}"):
        x, fs = read_input(song, a.excerpt)
        print(f"{tag}: {len(x) / fs:.0f}s", flush=True)
        if a.stage in ("1", "all"):
            write(d / "0_mix.flac", x, fs)
            for stem, prompt in STAGE1.items():
                if only and stem not in only:
                    continue
                t = time.time()
                y = separate(model, proc, x, fs, prompt, spans=a.spans)
                write(d / "1_stems" / f"{stem}.flac", y, fs)
                print(f"  1 {stem:<10} '{prompt}'  {time.time() - t:.0f}s", flush=True)
            parts = [d / "1_stems" / f"{s}.flac" for s in STAGE1]
            if all(p.exists() for p in parts):
                rest = x - sum(read_stem(p) for p in parts)
                write(d / "1_stems" / "_missed.flac", rest, fs)  # the mix minus the five stems
        if a.stage in ("2", "all"):
            drums = x if a.kit_from_mix else read_stem(d / "1_stems" / "drums.flac")
            if a.stage == "2":
                write(d / "0_mix.flac", x, fs)
            for piece, prompt in STAGE2.items():
                if only and piece not in only:
                    continue
                t = time.time()
                y = separate(model, proc, drums, fs, prompt, spans=a.spans)
                write(d / "2_drums" / f"{piece}.flac", y, fs)
                print(f"  2 {piece:<10} '{prompt}'  {time.time() - t:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
