"""Source separation into stems, with two backends:

- "demucs": Demucs (vocals / drums / bass / other).
- "roformer": a chain of models run through audio-separator (BS-/Mel-RoFormer,
  MDX23C, Demucs, ...), one of the plans in [stems.plans] (defaults.toml). Each step
  separates one stem further, e.g. mix -> vocals / drums / bass / guitar / other
  -> kick / snare / toms / hihat / cymbals. A step may run several models on the same
  input: each stem is then the weighted average of the models that produce it (an
  ensemble). Every model's own outputs are kept in <cache>/models/ for listening.

Separation runs in a process of its own (its memory is released before the rest of
the pipeline runs) and the stems are cached as float WAVs in
<output dir>/stems/<song>/, reused while you tune the rest of the chain. They are
also handy for remixing in a DAW.

That process stays for as long as there is something to separate, and takes one source
after another (a take's room track, its close tracks, the next take's): its models stay
loaded, each in a separator of its own, and their repeated blocks compiled ([stems]
compile: twice as fast). A model was loaded for every source before, 13 times for a take
with a close track, and compiling at each load would have eaten the gain. The code it
compiles is kept on disk (COMPILED_DIR), so a later run compiles in half the time.
release() ends the process: the pipeline calls it before the stem chains, which want
the memory.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import progress

CHUNK_S = 60.0  # seconds per model call
OVERLAP_S = 5.0  # crossfaded between consecutive chunks
MODEL_DIR = Path.home() / ".cache" / "mooz" / "models"
COMPILED_DIR = Path.home() / ".cache" / "mooz" / "compiled"  # what torch.compile makes of the models' blocks (TORCHINDUCTOR_CACHE_DIR)
GPU_ROOM_GB = 2.0  # free memory wanted on the card before one more model is loaded next to the others


def available(backend: str = "demucs") -> bool:
    """Is the backend installed? Checked without importing it: PyTorch alone would
    add hundreds of MB to this process while the separation subprocess runs."""
    from importlib.util import find_spec

    return all(find_spec(m) is not None for m in ("torch", "audio_separator" if backend == "roformer" else "demucs"))


def _short(model: str) -> str:
    """'model_bs_roformer_ep_317_sdr_12.9755.ckpt' -> 'bs_roformer'."""
    name = re.sub(r"(_ep_\d+)?(_sdr_[\d.]+)?$", "", Path(model).stem)
    return re.sub(r"^model_", "", name)


def describe(cfg: dict) -> str:
    if cfg["backend"] == "roformer":
        out = " → ".join(f"{s['combine']}({', '.join(i.lstrip('_') for i in s['inputs'])})" if "combine" in s else
                         "+".join(_short(m["model"]) for m in members(s)) + f" ({s['input'].lstrip('_')})"
                         for s in model_steps(cfg))
        out = f"{cfg['plan']}: {out}"
        return out + (f" + pan split ({cfg['pan_split']['input']})" if cfg.get("pan_split", {}).get("enabled") else "")
    return cfg["model"]


def cache_name(cfg: dict) -> str:
    """Folder name suffix of a plan's stem cache: each plan keeps its own stems (the
    "cascade" plan the original folder), so switching plans never discards stems."""
    return "" if cfg["backend"] != "roformer" or cfg["plan"] == "cascade" else f"_{cfg['plan']}"


def members(step: dict) -> list[dict]:
    """The models of a step: {model, outputs, weight, weights} each. A step is either one
    model (model + outputs) or an ensemble (models = [...])."""
    ms = step.get("models") or [{"model": step["model"], "outputs": step["outputs"]}]
    return [{"weight": 1.0, "weights": {}} | m for m in ms]


def max_spec(ests: list[np.ndarray], nfft: int = 4096, hop: int = 1024, block: int = 1 << 20) -> np.ndarray:
    """Per STFT bin, the estimate with the largest magnitude, with its phase (UVR's "Max Spec"
    ensemble): what any of several models kept of a stem, e.g. the bass's upper partials,
    which each model files elsewhere at different times. In blocks with margins, so memory
    does not grow with the song's length."""
    n = min(len(e) for e in ests)
    out = np.empty((n, ests[0].shape[1]), dtype=np.float32)
    margin = 4 * nfft
    for s in range(0, n, block):
        e = min(n, s + block)
        a, b = max(0, s - margin), min(n, e + margin)
        S = np.stack([signal.stft(x[a:b].T, nperseg=nfft, noverlap=nfft - hop)[2].astype(np.complex64) for x in ests])
        pick = np.take_along_axis(S, np.abs(S).argmax(axis=0)[None], axis=0)[0]
        y = signal.istft(pick, nperseg=nfft, noverlap=nfft - hop)[1].T
        out[s:e] = np.pad(y, ((0, max(0, b - a - len(y))), (0, 0)))[s - a:e - a]
    return out


def combine(parts: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """Weighted average of several models' estimates of one stem."""
    total = sum(w for w, _ in parts)
    if total <= 0:
        raise ValueError("ensemble weights of a stem must add up to more than 0")
    n = min(len(y) for _, y in parts)
    return sum(w * y[:n] for w, y in parts) / total


def _resample(x: np.ndarray, fs_from: int, fs_to: int) -> np.ndarray:
    if fs_from == fs_to:
        return x
    r = Fraction(fs_to, fs_from)
    return signal.resample_poly(x, r.numerator, r.denominator, axis=0)


def model_steps(cfg: dict) -> list[dict]:
    """The enabled steps of the chosen plan (what the separation cache depends on)."""
    if cfg["plan"] not in cfg["plans"]:
        raise KeyError(f"stems.plan: unknown plan '{cfg['plan']}' (available: {', '.join(cfg['plans'])})")
    return [s for s in cfg["plans"][cfg["plan"]]["steps"] if s.get("enabled", True) and not s.get("split")]


def split(stems: dict[str, Path], pcfg: dict, cache_dir: Path) -> dict[str, Path]:
    """The pan split ([stems.pan_split]), run on the cached stems: its outputs are
    written as _split_<name>.wav next to them and replace the input stem."""
    if not pcfg["enabled"] or pcfg["input"] not in stems:
        return stems
    stems = dict(stems)
    src = stems.pop(pcfg["input"])
    print(f"  stems: pan split of {pcfg['input']}", file=sys.stderr)
    for part, f in _pan_split(src, cache_dir, sf.info(src).samplerate).items():
        name = pcfg["outputs"].get(part)
        if name is None:
            f.unlink()
            continue
        stems[name] = f.rename(cache_dir / f"_split_{name}.wav")
    return stems


def _stems_in(cache_dir: Path) -> dict[str, Path]:
    return {f.stem: f for f in sorted(cache_dir.glob("*.wav")) if not f.stem.startswith("_")}


AGAIN = False  # --separate-again: kept stems are not used, every source is separated once more (once a run)
_made: set[str] = set()  # the caches this run has made: those it uses, also with AGAIN (a second profile, the take's own turn)


def separate(x: np.ndarray, fs: int, cfg: dict, cache_dir: Path, key: dict) -> dict[str, Path]:
    """Stem files for x (same length, rate and channels), from cache when key matches."""
    meta = cache_dir / "key.json"
    asked_again = AGAIN and str(cache_dir.resolve()) not in _made
    if cfg["cache"] and not asked_again and meta.exists() and json.loads(meta.read_text()) == key:
        stems = _stems_in(cache_dir)
        if stems and all(sf.info(f).frames == len(x) for f in stems.values()):
            print("  stems: using cache", file=sys.stderr)
            progress.emit("separation", track=cache_dir.name, state="cached")
            return stems
    print(f"  stems: separating ({cfg['backend']}: {describe(cfg)}) ...", file=sys.stderr)
    progress.emit("separation", track=cache_dir.name, state="start")
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta.unlink(missing_ok=True)
    for old in cache_dir.glob("*.wav"):  # a different backend/plan gives different stems
        old.unlink()
    shutil.rmtree(cache_dir / "models", ignore_errors=True)
    inp = cache_dir / "_input.wav"
    job = cache_dir / "_job.json"
    sf.write(inp, x.astype(np.float32), fs, subtype="FLOAT")
    job.write_text(json.dumps({"input": str(inp), "out_dir": str(cache_dir), "cfg": cfg}))
    try:
        _ask(job)
    finally:
        inp.unlink(missing_ok=True)
        job.unlink(missing_ok=True)
    meta.write_text(json.dumps(key))
    _made.add(str(cache_dir.resolve()))
    progress.emit("separation", track=cache_dir.name, state="done")
    return _stems_in(cache_dir)


# --- the process that separates ----------------------------------------------------------------
# One for the whole run: asked for one source after another over its standard input (a job file's
# path per line), answering on a pipe of its own, since its standard output carries the progress
# events and its standard error what it prints. A job that fails ends it (its GPU may not be in a
# state to go on); the next source starts another.

_server: dict | None = None


def _ask(job: Path) -> None:
    """Have the separating process do this job (started if there is none); raises as a failed
    subprocess does if the job fails or the process is gone."""
    global _server
    if _server is None or _server["proc"].poll() is not None:
        release()
        back, forth = os.pipe()
        env = os.environ | ({} if "TORCHINDUCTOR_CACHE_DIR" in os.environ else {"TORCHINDUCTOR_CACHE_DIR": str(COMPILED_DIR)})
        cmd = [sys.executable, "-m", "mooz.separate", "--serve", str(forth)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, pass_fds=(forth,), env=env, text=True)
        os.close(forth)
        _server = {"proc": proc, "answers": os.fdopen(back, "r"), "cmd": cmd}
    proc = _server["proc"]
    try:
        proc.stdin.write(f"{job}\n")
        proc.stdin.flush()
        answer = _server["answers"].readline().strip()
    except (BrokenPipeError, OSError, ValueError):
        answer = ""
    if answer != "ok":  # it said so on its standard error; it has ended, or is ending
        cmd = _server["cmd"]
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
        release()
        raise subprocess.CalledProcessError(code or 1, cmd)


def release() -> None:
    """End the separating process, if there is one: its models and the memory they hold go with
    it. Nothing is lost but the time to load them again."""
    global _server
    if _server is None:
        return
    server, _server = _server, None
    proc = server["proc"]
    try:
        proc.stdin.close()  # no more jobs: it ends when it has read this
    except OSError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    server["answers"].close()


atexit.register(release)


def _serve(answers_fd: int) -> None:
    """In the separating process: one job per line of standard input, until it closes."""
    import traceback

    if sys.platform.startswith("linux"):  # it goes when the run goes, however the run goes
        try:
            import ctypes
            import signal as signals

            ctypes.CDLL("libc.so.6").prctl(1, signals.SIGTERM)  # PR_SET_PDEATHSIG
        except OSError:
            pass
    answers = os.fdopen(answers_fd, "w")
    models = _Models()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            job = json.loads(Path(line.strip()).read_text())
            if job["cfg"]["backend"] == "roformer":
                _worker_roformer(Path(job["input"]), Path(job["out_dir"]), job["cfg"], models)
            else:
                models.clear()  # (the other backend wants the card to itself)
                _worker_demucs(Path(job["input"]), Path(job["out_dir"]), job["cfg"])
        except BaseException:
            traceback.print_exc()
            answers.write("failed\n")
            answers.flush()
            sys.exit(1)
        answers.write("ok\n")
        answers.flush()


class _Models:
    """The models this process has loaded, each in a separator of its own, the last used last."""

    def __init__(self) -> None:
        self.loaded: dict[tuple, object] = {}

    def clear(self) -> None:
        import gc

        import torch

        self.loaded.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get(self, m: dict, cfg: dict, out: Path, sr: int):
        """The separator with model m loaded as the step asks, writing to out. Loaded at its first
        use and kept; the least recently used ones go when the card has less than GPU_ROOM_GB free
        for a new one (on the CPU, where a model is in main memory, only the last one stays)."""
        import gc
        import logging

        import torch
        from audio_separator.separator import Separator

        on_gpu = cfg["device"] != "cpu" and torch.cuda.is_available()
        compiled = bool(cfg.get("compile", True)) and on_gpu
        # segment_size (STFT frames): shorter model windows than its config's, for models too big
        # for the GPU at their own (e.g. 53 stems at 20 s); the separator is told at load time
        # pitch_shift (semitones): the input is shifted down by this much before the model and the
        # outputs back up (UVR's "pitch change"), e.g. as a second pass to average with the first
        key = (m["model"], m.get("segment_size", 256), "segment_size" in m, m.get("pitch_shift", 0), cfg["device"], compiled)
        sep = self.loaded.pop(key, None)
        out.mkdir(parents=True, exist_ok=True)
        if sep is None:
            while self.loaded and (not on_gpu or torch.cuda.mem_get_info()[0] < GPU_ROOM_GB * 2 ** 30):
                del self.loaded[next(iter(self.loaded))]
                gc.collect()
                if on_gpu:
                    torch.cuda.empty_cache()
            sep = Separator(log_level=logging.ERROR, model_file_dir=str(MODEL_DIR), output_dir=str(out), output_format="WAV",
                            use_soundfile=True, sample_rate=sr, use_autocast=cfg["device"] != "cpu", use_torch_compile=compiled,
                            normalization_threshold=1.0, amplification_threshold=0.0)
            sep.arch_specific_params["MDXC"].update(segment_size=key[1], override_model_segment_size=key[2], pitch_shift=key[3])
            _load(sep, m["model"], cfg.get("custom_models", {}))
        self.loaded[key] = sep
        sep.output_dir = sep.model_instance.output_dir = str(out)  # this job's folder
        return sep


# --- Demucs ---------------------------------------------------------------------------------

def _worker_demucs(inp: Path, out_dir: Path, cfg: dict) -> None:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(cfg["model"])
    model.eval()
    sr = model.samplerate
    x, fs = sf.read(inp, dtype="float32", always_2d=True)
    n_out, mono = len(x), x.shape[1] == 1
    if mono:
        x = np.repeat(x, 2, axis=1)
    x = _resample(x, fs, sr).astype(np.float32)
    ref = x.mean(axis=1)
    mean, std = float(ref.mean()), float(ref.std()) + 1e-8
    del ref

    tmp = {name: sf.SoundFile(out_dir / f"_{name}.wav", "w", sr, 2, subtype="FLOAT") for name in model.sources}
    chunk, ov = int(CHUNK_S * sr), int(OVERLAP_S * sr)
    fade_in = np.linspace(0, 1, ov, dtype=np.float32)
    tail = None  # previous chunk's overlap region, (sources, ov, 2)
    pos = 0
    while pos < len(x):
        end = min(len(x), pos + chunk)
        seg = torch.from_numpy(np.ascontiguousarray(((x[pos:end] - mean) / std).T))
        with torch.no_grad():
            out = apply_model(model, seg[None], device=device, shifts=cfg["shifts"], split=True,
                              overlap=cfg["overlap"], progress=False)[0]
        out = (out.cpu().numpy() * std + mean).transpose(0, 2, 1)  # (sources, time, 2)
        if tail is not None:
            k = min(ov, out.shape[1])
            out[:, :k] = tail[:, :k] * (1 - fade_in[:k, None]) + out[:, :k] * fade_in[:k, None]
        last = end >= len(x)
        keep = out.shape[1] if last else out.shape[1] - ov
        for i, name in enumerate(model.sources):
            tmp[name].write(out[i, :keep])
        tail = None if last else out[:, keep:].copy()
        pos = end if last else end - ov
        print(f"  stems: {min(end, len(x)) / sr:5.0f}s / {len(x) / sr:.0f}s", file=sys.stderr, flush=True)
        progress.emit("separation", track=out_dir.name, state="step", i=min(end, len(x)), n=len(x), model=cfg["model"])
    for f in tmp.values():
        f.close()
    del x, model
    if device == "cuda":
        torch.cuda.empty_cache()
    for name in tmp:
        _finish(out_dir / f"_{name}.wav", out_dir / f"{name}.wav", sr, fs, n_out, mono)


def _finish(src: Path, dst: Path, sr: int, fs: int, n_out: int, mono: bool, gain: float = 1.0) -> None:
    """Back to the original rate, length, level and channel count."""
    y, _ = sf.read(src, dtype="float32", always_2d=True)
    y = (_resample(y, sr, fs) * gain).astype(np.float32)
    y = np.pad(y, ((0, max(0, n_out - len(y))), (0, 0)))[:n_out]
    if mono:
        y = y.mean(axis=1, keepdims=True)
    sf.write(dst, y, fs, subtype="FLOAT")
    src.unlink()


# --- RoFormer & co. via audio-separator -------------------------------------------------------

_STEM_IN_NAME = re.compile(r"_\(([^)]+)\)")


def _add_to(path: Path, y: np.ndarray, sr: int) -> None:
    """Write y to path, or add it to what is already there (several outputs -> one stem)."""
    if path.exists():
        y = y + sf.read(path, dtype="float32", always_2d=True)[0][: len(y)]
    sf.write(path, y, sr, subtype="FLOAT")


def _separate_chunked(sep, src: Path, work: Path, sr: int, chunk_s: float = CHUNK_S,
                      wanted: set[str] | None = None) -> dict[str, Path | None]:
    """Run the loaded model on src in chunk_s pieces with OVERLAP_S crossfades,
    so memory does not grow with song length (the separator holds every output of a
    piece: many-stem models need shorter pieces). Returns {model stem name: file}; outputs
    not in `wanted` are not read (None)."""
    info = sf.info(src)
    chunk, ov = int(chunk_s * sr), int(OVERLAP_S * sr)
    fade_in = np.linspace(0, 1, ov, dtype=np.float32)[:, None]
    writers: dict[str, sf.SoundFile] = {}
    tails: dict[str, np.ndarray] = {}
    outdir = work / "out"
    pos = 0
    while pos < info.frames:
        end = min(info.frames, pos + chunk)
        piece = work / "_chunk.wav"
        sf.write(piece, sf.read(src, start=pos, stop=end, dtype="float32", always_2d=True)[0], sr, subtype="FLOAT")
        for old in outdir.glob("*"):
            old.unlink()
        last = end >= info.frames
        for f in sep.separate(str(piece)):
            f = Path(f) if Path(f).is_absolute() else outdir / f
            m = _STEM_IN_NAME.search(f.name)
            name = (m.group(1) if m else f.stem).lower()
            if wanted is not None and name not in wanted:
                writers.setdefault(name, None)
                f.unlink()
                continue
            y = sf.read(f, dtype="float32", always_2d=True)[0]
            y = np.pad(y, ((0, max(0, (end - pos) - len(y))), (0, 0)))[: end - pos]
            if np.abs(y).max() >= 0.999:
                print(f"  stems: warning: {name} may have been rescaled by audio-separator", file=sys.stderr)
            if name not in writers:
                writers[name] = sf.SoundFile(work / f"_raw_{name}.wav", "w", sr, y.shape[1], subtype="FLOAT")
            if name in tails:
                k = min(ov, len(y))
                y[:k] = tails[name][:k] * (1 - fade_in[:k]) + y[:k] * fade_in[:k]
            keep = len(y) if last else len(y) - ov
            writers[name].write(y[:keep])
            tails[name] = y[keep:].copy()
        piece.unlink()
        pos = end if last else end - ov
    for w in writers.values():
        if w is not None:
            w.close()
    return {name: (work / f"_raw_{name}.wav" if w is not None else None) for name, w in writers.items()}


def _pan_histogram(x: np.ndarray, sr: int, nfft: int, hop: int):
    """Energy-weighted histogram of per-bin left/right level difference (dB, + = left), 200 Hz - 5 kHz."""
    edges = np.linspace(-20, 20, 81)
    hist = np.zeros(len(edges) - 1)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    band = (f > 200) & (f < 5000)
    for s in range(0, len(x), 1 << 20):
        _, _, z = signal.stft(x[s : s + (1 << 20)].T, sr, nperseg=nfft, noverlap=nfft - hop)
        L, R = np.abs(z[0, band]), np.abs(z[1, band])
        ild = 20 * np.log10((L + 1e-9) / (R + 1e-9))
        hist += np.histogram(ild.clip(-19.9, 19.9), bins=edges, weights=L * L + R * R)[0]
    return hist / max(hist.sum(), 1e-20), 0.5 * (edges[1:] + edges[:-1])


def _find_split(hist: np.ndarray, centres: np.ndarray, min_gap_db: float = 8.0) -> float | None:
    """Level difference between the two main clusters, or None if there is only one."""
    h = np.convolve(hist, np.ones(5) / 5, mode="same")
    peaks, _ = signal.find_peaks(h, distance=int(min_gap_db / (centres[1] - centres[0])))
    if len(peaks) < 2:
        return None
    a, b = sorted(peaks[np.argsort(h[peaks])[-2:]])
    valley = a + int(np.argmin(h[a : b + 1]))
    # well apart (two amps in different places), a dip between them, and the quieter one
    # at least a tenth of the louder one
    if (centres[b] - centres[a] < min_gap_db or h[valley] > 0.8 * min(h[a], h[b])
            or min(h[a], h[b]) < 0.1 * max(h[a], h[b])):
        return None
    return float(centres[valley])


def _pan_split(src: Path, work: Path, sr: int) -> dict[str, Path]:
    """Split a stereo stem into its two most separated positions (e.g. two guitar amps
    at different places in the room) with soft time-frequency masks that sum back to
    the input exactly. Returns {"left": ..., "right": ...} or {"both": ...}."""
    x = sf.read(src, dtype="float32", always_2d=True)[0]
    nfft, hop = 2048, 512
    hist, centres = _pan_histogram(x, sr, nfft, hop)
    split = _find_split(hist, centres)
    if x.shape[1] != 2 or split is None:
        print("  stems:   no two distinct positions found, kept as one stem", file=sys.stderr)
        out = work / "_raw_both.wav"
        sf.write(out, x, sr, subtype="FLOAT")
        return {"both": out}
    print(f"  stems:   split at {split:+.1f} dB left/right", file=sys.stderr)
    left = np.empty_like(x)
    chunk, margin = 1 << 20, 4 * nfft
    for s in range(0, len(x), chunk):
        e = min(len(x), s + chunk)
        a, b = max(0, s - margin), min(len(x), e + margin)
        _, _, z = signal.stft(x[a:b].T, sr, nperseg=nfft, noverlap=nfft - hop)
        ild = 20 * np.log10((np.abs(z[0]) + 1e-9) / (np.abs(z[1]) + 1e-9))
        m = 1 / (1 + np.exp(-(ild - split) / 1.5))  # soft: 1.5 dB transition width
        _, y = signal.istft(z * m, sr, nperseg=nfft, noverlap=nfft - hop)
        y = y.T[: b - a]
        left[s:e] = np.pad(y, ((0, max(0, b - a - len(y))), (0, 0)))[s - a : e - a]
    out_l, out_r = work / "_raw_left.wav", work / "_raw_right.wav"
    sf.write(out_l, left, sr, subtype="FLOAT")
    sf.write(out_r, x - left, sr, subtype="FLOAT")  # the complement: both sum to the input exactly
    return {"left": out_l, "right": out_r}


KEEP_DB = -12.0  # kept model outputs are written this far below the recording's level (as the stage files)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_custom(name: str, spec: dict) -> tuple[Path, Path]:
    """A model audio-separator does not list ([stems.custom_models]): downloaded once, its
    checksum checked, and opened as plain tensors (a checkpoint is a pickle: anything that
    is not only weights is refused)."""
    import urllib.request

    import torch

    files = []
    for url, fname, sha in ((spec["url"], name, spec["sha256"]), (spec["config_url"], spec["config"], spec["config_sha256"])):
        path = MODEL_DIR / fname
        if not path.exists():
            print(f"  stems:   downloading {fname}", file=sys.stderr, flush=True)
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
        if _sha256(path) != sha:
            path.unlink()
            raise ValueError(f"{fname}: checksum mismatch (deleted; check [stems.custom_models])")
        files.append(path)
    torch.load(files[0], map_location="cpu", weights_only=True)  # raises on anything but tensors
    return files[0], files[1]


def _load(sep, model: str, custom: dict) -> None:
    """Load a model into a separator, including the [stems.custom_models] ones (settings such
    as the segment size are taken at load time: a model loaded otherwise is another separator's)."""
    import torch

    if model in custom:
        ckpt, yaml = _fetch_custom(model, custom[model])
        spec = custom[model]
        from audio_separator.separator.roformer.parameter_validator import ParameterValidator

        lo, hi = ParameterValidator.PARAMETER_RANGES["num_stems"]  # 16 at most; MVSep's mega model has 53
        ParameterValidator.PARAMETER_RANGES["num_stems"] = (lo, max(hi, 64))
        orig, orig_load = sep.download_model_files, torch.load
        sep.download_model_files = lambda f: (f, spec["arch"], f"custom: {f}", str(ckpt), str(yaml)) if f == model else orig(f)
        # memory-mapped: the checkpoint is paged in from disk, not copied into RAM next to the model's
        # own weights (a 1.4 GB checkpoint would otherwise need ~3 GB while loading)
        torch.load = lambda *a, **k: orig_load(*a, **({"mmap": True} | k))
        try:
            sep.load_model(model_filename=model)
        finally:
            sep.download_model_files, torch.load = orig, orig_load
    else:
        sep.load_model(model_filename=model)


def _keep(src: Path, dst: Path, sr: int, fs: int, n_out: int, mono: bool, gain: float) -> float:
    """A model's output, for listening: original rate and length, KEEP_DB below the
    recording's level (lower if it would clip), 24-bit FLAC. Returns the level in dB."""
    y, _ = sf.read(src, dtype="float32", always_2d=True)
    y = _resample(y, sr, fs) * gain * 10 ** (KEEP_DB / 20)
    y = np.pad(y, ((0, max(0, n_out - len(y))), (0, 0)))[:n_out]
    if mono:
        y = y.mean(axis=1, keepdims=True)
    peak = float(np.abs(y).max())
    extra = min(1.0, 0.99 / peak) if peak > 0 else 1.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, (y * extra).astype(np.float32), fs, subtype="PCM_24")
    return KEEP_DB + 20 * np.log10(extra)


def _worker_roformer(inp: Path, out_dir: Path, cfg: dict, models: _Models | None = None) -> None:
    """One source through the plan's steps. models: this process's loaded models, kept from one
    source to the next (a process that does one job has its own)."""
    models = models if models is not None else _Models()
    sr = 44100  # what these models are trained on
    x, fs = sf.read(inp, dtype="float32", always_2d=True)
    n_out, mono = len(x), x.shape[1] == 1
    if mono:
        x = np.repeat(x, 2, axis=1)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    for old in work.glob("*.wav"):
        old.unlink()
    current = {"mix": work / "_mix.wav"}  # stem name -> file at 44.1 kHz
    # audio-separator rescales any output peaking above 1.0, which would break the
    # remix sums: work 12 dB below full scale and undo that at the end.
    pre = 0.25 / max(float(np.abs(x).max()), 1e-9)
    sf.write(current["mix"], (_resample(x, fs, sr) * pre).astype(np.float32), sr, subtype="FLOAT")
    del x

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    keep = cfg.get("keep_models", True)
    kept: list[dict] = []
    n_models = sum(len(members(step)) for step in model_steps(cfg) if "combine" not in step)  # for the progress events
    n_run = 0
    for i, step in enumerate(model_steps(cfg)):  # (split steps run later, on the cached stems)
        if "combine" in step:  # merge stems the earlier steps made, e.g. several models' bass
            have = [n for n in step["inputs"] if n in current]
            if not have:
                print(f"  stems: step {i + 1} skipped, none of {step['inputs']}", file=sys.stderr)
                continue
            print(f"  stems: step {i + 1}: {step['combine']} of {', '.join(have)} -> {step['output']}", file=sys.stderr, flush=True)
            ys = [sf.read(current.pop(n), dtype="float32", always_2d=True)[0] for n in have]
            if step["combine"] != "max_spec":
                raise ValueError(f"stems: unknown combine '{step['combine']}' (max_spec)")
            y = max_spec(ys) if len(ys) > 1 else ys[0]
            del ys
            dst = work / f"{step['output']}.wav"
            sf.write(dst, y, sr, subtype="FLOAT")
            current[step["output"]] = dst
            if keep:
                rel = Path(f"{i + 1}_combine") / step["combine"] / f"{step['output']}.flac"
                kept.append({"file": str(rel), "step": i + 1, "input": "+".join(have), "model": step["combine"],
                             "stem": step["output"], "level_db": round(_keep(dst, out_dir / "models" / rel, sr, fs, n_out, mono, 1 / pre), 2)})
            for n in have:
                (work / f"{n}.wav").unlink(missing_ok=True)
            continue
        src = step["input"]
        if src not in current:
            print(f"  stems: step {i + 1} skipped, no '{src}' stem", file=sys.stderr)
            continue
        src_path = current[src]
        if src != "mix":  # consumed: move it aside, its outputs may reuse the name (e.g. de-reverb)
            del current[src]
            src_path = src_path.rename(work / f"_in{i}.wav")
        parts: dict[str, list[tuple[float, Path, str]]] = {}
        ms = members(step)
        for j, m in enumerate(ms):
            print(f"  stems: step {i + 1}: {m['model']} on {src}", file=sys.stderr, flush=True)
            progress.emit("separation", track=out_dir.name, state="step", i=n_run, n=n_models, model=_short(m["model"]))
            n_run += 1
            t0 = time.time()
            sep = models.get(m, cfg, work / "out", sr)
            mapping = {k.lower(): v for k, v in m["outputs"].items()}
            raw = _separate_chunked(sep, src_path, work, sr, m.get("chunk_s", CHUNK_S), set(mapping))
            mine: dict[str, Path] = {}
            for model_name, f in raw.items():
                name = mapping.get(model_name)
                if name is None:  # unmapped output: left to the residual
                    if len(raw) <= 8:
                        print(f"  stems:   ({model_name} not used)", file=sys.stderr)
                    continue
                else:
                    mine[name] = work / f"_m{j}_{name}.wav"
                    _add_to(mine[name], sf.read(f, dtype="float32", always_2d=True)[0], sr)
                f.unlink()
            for name, f in mine.items():
                w = float(m["weights"].get(name, m["weight"]))
                if w > 0:
                    parts.setdefault(name, []).append((w, f, m["model"]))
                if keep:
                    rel = Path(f"{i + 1}_{src.lstrip('_')}") / _short(m["model"]) / f"{name}.flac"
                    kept.append({"file": str(rel), "step": i + 1, "input": src, "model": m["model"], "stem": name,
                                 "weight": w, "level_db": round(_keep(f, out_dir / "models" / rel, sr, fs, n_out, mono, 1 / pre), 2)})
            print(f"  stems:   done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        for name, ps in parts.items():
            dst = work / f"{name}.wav"
            if len(ps) == 1:
                y = sf.read(ps[0][1], dtype="float32", always_2d=True)[0]
            else:
                y = combine([(w, sf.read(f, dtype="float32", always_2d=True)[0]) for w, f, _ in ps])
                print(f"  stems:   {name}: average of {len(ps)} models", file=sys.stderr, flush=True)
                if keep:
                    rel = Path(f"{i + 1}_{src.lstrip('_')}") / "ensemble" / f"{name}.flac"
                    tmp = work / "_ens.wav"
                    sf.write(tmp, y, sr, subtype="FLOAT")
                    kept.append({"file": str(rel), "step": i + 1, "input": src, "model": "ensemble", "stem": name,
                                 "weights": {mod: w for w, _, mod in ps},
                                 "level_db": round(_keep(tmp, out_dir / "models" / rel, sr, fs, n_out, mono, 1 / pre), 2)})
                    tmp.unlink()
            _add_to(dst, y, sr)  # a stem an earlier step made (e.g. the vocals) gets this step's part added
            current[name] = dst
        for f in work.glob("_m*_*.wav"):
            f.unlink()
        if src != "mix":
            src_path.unlink()
    for name, path in current.items():
        if name == "mix" or name.startswith("_"):
            continue
        _finish(path, out_dir / f"{name}.wav", sr, fs, n_out, mono, gain=1 / pre)
    if kept:
        (out_dir / "models" / "index.json").write_text(json.dumps(
            {"note": f"every model's own outputs, {-KEEP_DB:.0f} dB below the recording's level (level_db); "
                     "the folders are <step>_<input>/<model>/", "files": kept}, indent=1))
    for leftover in work.rglob("*"):
        if leftover.is_file():
            leftover.unlink()
    for d in sorted(work.rglob("*"), reverse=True) + [work]:
        if d.is_dir():
            d.rmdir()


if __name__ == "__main__":
    if sys.argv[1] == "--serve":  # the run's separating process: jobs on standard input, answers on this pipe
        _serve(int(sys.argv[2]))
    else:  # one job, then gone
        job = json.loads(Path(sys.argv[1]).read_text())
        worker = _worker_roformer if job["cfg"]["backend"] == "roformer" else _worker_demucs
        worker(Path(job["input"]), Path(job["out_dir"]), job["cfg"])
