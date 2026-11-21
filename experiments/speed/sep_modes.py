# The separation models under the precisions and the compilation audio-separator offers, and bfloat16: the time for
# 30 s of a take and how far each stem is from the float32 one. Overlap as the program has it (the library's 8).
# Run from the project root: uv run python experiments/speed/sep_modes.py <work folder> [model ...]
import sys, time, logging, functools, numpy as np, soundfile as sf
from pathlib import Path


def main():
    import torch
    from mooz import separate, config
    from audio_separator.separator import Separator
    S = Path(sys.argv[1]); work = S / "sepmodes"; (work / "out").mkdir(parents=True, exist_ok=True)
    cfg = config.load(None)["stems"]
    src = next(Path("new_tracks/drive-download-20261107T141948Z-1-001/261105_235855 Biancaneve").glob("*TrMic*"))
    x, fs = sf.read(src, dtype="float32", always_2d=True, start=100 * 48000, stop=130 * 48000)
    sr = 44100
    piece = work / "piece.wav"; sf.write(piece, separate._resample(x, fs, sr).astype(np.float32), sr, subtype="FLOAT")
    models = sys.argv[2:] or ["BS-Roformer-SW.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"]
    variants = [("float32", dict(use_autocast=False), None), ("autocast float16 (as now)", dict(use_autocast=True), None),
                ("native float16", dict(use_autocast=False, use_native_fp16=True), None),
                ("autocast float16, compiled", dict(use_autocast=True, use_torch_compile=True), None),
                ("native float16, compiled", dict(use_autocast=False, use_native_fp16=True, use_torch_compile=True), None),
                ("autocast bfloat16", dict(use_autocast=True), torch.bfloat16)]
    import torch.amp.autocast_mode as am
    plain_autocast = am.autocast
    for model in models:
        ref = None
        print(f"\n{model}")
        for name, flags, cast in variants:
            am.autocast = plain_autocast if cast is None else functools.partial(plain_autocast, dtype=cast)
            try:
                sep = Separator(log_level=logging.ERROR, model_file_dir=str(separate.MODEL_DIR), output_dir=str(work / "out"), output_format="WAV",
                                use_soundfile=True, sample_rate=sr, normalization_threshold=1.0, amplification_threshold=0.0, **flags)
                sep.arch_specific_params["MDXC"].update(segment_size=256, override_model_segment_size=False, pitch_shift=0)
                separate._load(sep, model, cfg.get("custom_models", {}))
                inst = sep.model_instance
                took = []
                for run in range(2 if flags.get("use_torch_compile") else 1):  # the first compiled run compiles
                    for old in (work / "out").glob("*"): old.unlink()
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                    files = sep.separate(str(piece)); torch.cuda.synchronize(); took.append(time.time() - t0)
                outs = {}
                for f in files:
                    f = Path(f) if Path(f).is_absolute() else work / "out" / f
                    m = separate._STEM_IN_NAME.search(f.name); outs[(m.group(1) if m else f.stem).lower()] = sf.read(f, dtype="float32", always_2d=True)[0]
                if ref is None: ref = outs
                diff = {k: round(float(10 * np.log10(np.mean((v[: len(ref[k])] - ref[k][: len(v)]) ** 2) / (np.mean(ref[k] ** 2) + 1e-20) + 1e-20))) for k, v in outs.items() if np.mean(ref[k] ** 2) > 1e-9}
                what = f"precision {getattr(inst, 'effective_precision', '?')}, compiled {getattr(inst, 'effective_torch_compile', '?')}"
                print(f"   {name:28} {took[-1]:5.1f} s" + (f" (first run, compiling: {took[0]:.0f} s)" if len(took) > 1 else "") + f" | GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB | {what}"
                      + ("" if name == "float32" else f" | from float32, dB re each stem: {diff}"), flush=True)
                del sep, inst
            except Exception as e:
                print(f"   {name:28} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
            torch.cuda.empty_cache()
    am.autocast = plain_autocast


if __name__ == "__main__":
    main()
