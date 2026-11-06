"""Extra models to try (from the noblebarkrr/mvsepless_resources mirror of MSST checkpoints),
in the [stems.custom_models] form: each checkpoint is pinned by the SHA-256 of its Git-LFS
object, each config by the SHA-256 of the copy first downloaded, and loaded as plain tensors
only. `uv run python experiments/sep_bench/candidates.py` downloads them and checks that each
loads and separates a few seconds of audio.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASE = "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/"
# file in the mirror -> the start of its checkpoint's SHA-256 (the full one is checked against the mirror)
MODELS = {
    "bs_roformer/bs_mega_53stem_full_mvsep": "c62820893b",
    "mel_band_roformer/mbr_lead_rhythm_guitar_listra92": "b3c47bca33",
    "mel_band_roformer/mbr_percussion_yolkispaliks": "8fede38c91",
    "bs_roformer/bs_drums_xlancer": "70c01700b7",
    "bs_roformer/bs_drums2_xlancer": "7d53b4d663",
    "bs_roformer/bs_drums_gilliaaan": "f60c437025",
    "bs_roformer/bs_bass_xlancer": "c409465e3d",
    "bs_roformer/bs_bass_beatloo_labs": "57ab7828b3",
    "bs_roformer/bs_vocals_large1_unwa": "3d1334565b",
    "bs_roformer/bs_voc_hyperace2_unwa": "54cf516f62",
}
PINS = Path(__file__).with_name("candidates.json")  # full hashes, written on first download


def custom_models() -> dict:
    """{checkpoint file name: spec} for separate.py's custom-model loader."""
    import json

    pins = json.loads(PINS.read_text())
    return {f"{Path(p).name}.ckpt": {"arch": "MDXC", "config": f"{Path(p).name}_config.yaml",
                                     "url": f"{BASE}{p}.ckpt", "config_url": f"{BASE}{p}_config.yaml",
                                     "sha256": pins[p]["sha256"], "config_sha256": pins[p]["config_sha256"]}
            for p in MODELS if p in pins}


def pin() -> None:
    """Full checkpoint hashes from the mirror's file list (checked against the short ones
    above), config hashes from a first download."""
    import hashlib
    import json
    import urllib.request

    from rehearsal_master.separate import MODEL_DIR

    tree = {f["path"]: f for f in json.load(urllib.request.urlopen(
        "https://huggingface.co/api/models/noblebarkrr/mvsepless_resources/tree/main?recursive=1")) if f["type"] == "file"}
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    for p, short in MODELS.items():
        sha = tree[f"{p}.ckpt"]["lfs"]["oid"]
        assert sha.startswith(short), p
        cfg = MODEL_DIR / f"{Path(p).name}_config.yaml"
        if not cfg.exists():
            urllib.request.urlretrieve(f"{BASE}{p}_config.yaml", cfg)
        pins.setdefault(p, {"sha256": sha, "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()})
    PINS.write_text(json.dumps(pins, indent=1))


def check() -> None:
    """Download each model, load it and separate 10 s of noise: does audio-separator run it?"""
    import logging

    import numpy as np
    import soundfile as sf
    from audio_separator.separator import Separator

    from rehearsal_master import separate

    work = ROOT / "processed" / "experiments" / "sep_bench" / "_check"
    work.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(0).standard_normal((441000, 2)).astype(np.float32) * 0.05
    sf.write(work / "in.wav", x, 44100, subtype="FLOAT")
    sep = Separator(log_level=logging.ERROR, model_file_dir=str(separate.MODEL_DIR), output_dir=str(work),
                    output_format="WAV", use_soundfile=True, sample_rate=44100)
    for name, spec in custom_models().items():
        try:
            separate._load(sep, name, {name: spec})
            outs = sep.separate(str(work / "in.wav"))
            stems = sorted(separate._STEM_IN_NAME.search(Path(f).name).group(1) for f in outs)
            print(f"ok    {name}: {len(stems)} outputs {stems[:8]}{' ...' if len(stems) > 8 else ''}", flush=True)
        except (Exception, SystemExit) as e:  # noqa: BLE001  (audio-separator exits on a failed load)
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        for f in work.glob("*_(*).wav"):
            f.unlink()


if __name__ == "__main__":
    pin()
    check()
