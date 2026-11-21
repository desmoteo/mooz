# What fewer overlap windows cost the separation: 60 s of a take through two Roformer models at overlap 8, 4 and 2,
# the time and how far the stems move from overlap 8. Run from the project root: uv run python experiments/speed/overlap.py <work folder>
import sys, time, logging, numpy as np, soundfile as sf
from pathlib import Path
sys.path.insert(0, ".")
from mooz import separate, config
from audio_separator.separator import Separator
import torch
S = Path(sys.argv[1]); work = S / "ov"; (work / "out").mkdir(parents=True, exist_ok=True)
cfg = config.load(None)["stems"]
src = next(Path("new_tracks/drive-download-20261107T141948Z-1-001/261105_235855 Biancaneve").glob("*TrMic*"))
x, fs = sf.read(src, dtype="float32", always_2d=True, start=100 * 48000, stop=160 * 48000)
sr = 44100
piece = work / "piece.wav"; sf.write(piece, separate._resample(x, fs, sr).astype(np.float32), sr, subtype="FLOAT")
sep = Separator(log_level=logging.ERROR, model_file_dir=str(separate.MODEL_DIR), output_dir=str(work / "out"), output_format="WAV",
                use_soundfile=True, sample_rate=sr, use_autocast=True, normalization_threshold=1.0, amplification_threshold=0.0)
for model in ("BS-Roformer-SW.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"):
    ref = {}
    for ov in (8, 4, 2):
        sep.arch_specific_params["MDXC"].update(segment_size=256, override_model_segment_size=False, pitch_shift=0, overlap=ov)
        separate._load(sep, model, cfg.get("custom_models", {}))
        for old in (work / "out").glob("*"): old.unlink()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        files = sep.separate(str(piece)); dt = time.time() - t0
        outs = {}
        for f in files:
            f = Path(f) if Path(f).is_absolute() else work / "out" / f
            m = separate._STEM_IN_NAME.search(f.name); outs[(m.group(1) if m else f.stem).lower()] = sf.read(f, dtype="float32", always_2d=True)[0]
        if ov == 8: ref = outs
        diff = {k: round(float(10 * np.log10(np.mean((v[: len(ref[k])] - ref[k][: len(v)]) ** 2) / (np.mean(ref[k] ** 2) + 1e-20) + 1e-20)), 1) for k, v in outs.items() if np.mean(ref[k] ** 2) > 1e-9}
        print(f"{model[:28]:28} overlap {ov}: {dt:5.1f} s for 60 s of audio, GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" + ("" if ov == 8 else f" | difference from overlap 8, dB re each stem: {diff}"), flush=True)
