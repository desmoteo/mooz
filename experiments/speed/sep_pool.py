# All the models of the plans kept loaded together, their blocks compiled: the GPU memory each one takes, and for
# each the first run (which compiles) against the second. Run it twice with the same cache folder to see what a
# compilation kept on disk saves:
#   TORCHINDUCTOR_CACHE_DIR=<folder> uv run python experiments/speed/sep_pool.py <work folder>
import sys, time, logging, os, numpy as np, soundfile as sf
from pathlib import Path


def main():
    t_start = time.time()
    import torch
    from mooz import separate, config
    from audio_separator.separator import Separator
    S = Path(sys.argv[1]); work = S / "seppool"; (work / "out").mkdir(parents=True, exist_ok=True)
    cfg = config.load(None)["stems"]
    src = next(Path("new_tracks/drive-download-20261107T141948Z-1-001/261105_235855 Biancaneve").glob("*TrMic*"))
    x, fs = sf.read(src, dtype="float32", always_2d=True, start=100 * 48000, stop=120 * 48000)
    piece = work / "piece.wav"; sf.write(piece, separate._resample(x, fs, 44100).astype(np.float32), 44100, subtype="FLOAT")
    models = []
    for plan in ("ensemble", "close"):
        for step in cfg["plans"][plan]["steps"]:
            if not step.get("enabled", True): continue
            for m in ([step] if "model" in step else step.get("models", [])):
                if m.get("model") and m["model"] not in models: models.append(m["model"])
    print(f"cache folder {os.environ.get('TORCHINDUCTOR_CACHE_DIR')} | torch loaded in {time.time() - t_start:.1f} s | GPU free {torch.cuda.mem_get_info()[0] / 2**30:.1f} of {torch.cuda.mem_get_info()[1] / 2**30:.1f} GB")
    pool = {}
    for name in models:
        t0 = time.time(); before = torch.cuda.memory_allocated()
        sep = Separator(log_level=logging.ERROR, model_file_dir=str(separate.MODEL_DIR), output_dir=str(work / "out"), output_format="WAV",
                        use_soundfile=True, sample_rate=44100, normalization_threshold=1.0, amplification_threshold=0.0, use_autocast=True, use_torch_compile=True)
        sep.arch_specific_params["MDXC"].update(segment_size=256, override_model_segment_size=False, pitch_shift=0)
        # (separate._load frees the separator's previous model and the GPU's cache: here every model has a separator of its own)
        separate._load(sep, name, cfg.get("custom_models", {}))
        pool[name] = sep
        print(f"   loaded {name[:44]:44} in {time.time() - t0:4.1f} s | its weights {(torch.cuda.memory_allocated() - before) / 2**30:.2f} GB | all so far {torch.cuda.memory_allocated() / 2**30:.2f} GB | compiled {getattr(sep.model_instance, 'effective_torch_compile', '?')}", flush=True)
    for name, sep in pool.items():
        took = []
        for _ in range(2):
            for old in (work / "out").glob("*"): old.unlink()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); sep.separate(str(piece)); torch.cuda.synchronize(); took.append(time.time() - t0)
        print(f"   ran    {name[:44]:44} first {took[0]:5.1f} s, then {took[1]:5.1f} s: {took[0] - took[1]:5.1f} s of compiling | GPU at most {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"all {len(pool)} loaded and run: {time.time() - t_start:.0f} s | GPU free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
