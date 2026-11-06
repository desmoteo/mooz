"""The same test as run_sr.py with UniverSR (woongzip1/universr-audio: flow matching in the complex STFT
domain, no vocoder; speech, music and sound effects; 8/12/16/24 kHz -> 48 kHz), on the same excerpt:

  processed/experiments/bass_sr/<song>/universr_[resynth_]sr<rate>/
    0_input  1_down (what UniverSR gets)  2_universr (as it is)  3_merged (input below 0.9 x the cutoff, UniverSR above)

  . experiments/bass_sr/env.sh
  experiments/bass_sr/.venv-universr/bin/python experiments/bass_sr/run_universr.py [--source stem|resynth]
      [--rates 8000,12000] [--steps 4] [--guidance 1.5]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from run_sr import FS, ROOT, bass_level, downsample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="260924_230957_TrLR Stanza OK")
    ap.add_argument("--start", type=float, default=40.0)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--rates", default="8000,12000")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--source", default="stem", choices=["stem", "resynth"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import torch
    from universr import UniverSR

    rep = json.loads((ROOT / "processed/reports" / f"{a.song}.json").read_text())
    if a.source == "stem":
        t0 = int(round((rep["trim"]["start_s"] + a.start) * FS))
        x = sf.read(ROOT / "processed/stages" / a.song / "1_raw/bass.wav", dtype="float64", always_2d=True,
                    start=t0, stop=t0 + int(a.dur * FS))[0].mean(axis=1)
    else:
        t0 = int(round(a.start * FS))
        x = sf.read(ROOT / "processed/experiments/bass_bank" / a.song / "resynth/6_resynth.flac", dtype="float64",
                    always_2d=True, start=t0, stop=t0 + int(a.dur * FS))[0].mean(axis=1)
    x = x * bass_level.input_gain(x, FS)  # first: the stem at a proper level (bass_level.py)
    peak = float(np.abs(x).max())
    xs = x * (0.5 / peak)
    t = time.time()
    model = UniverSR.from_pretrained("woongzip1/universr-audio", device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"UniverSR loaded in {time.time() - t:.0f}s", flush=True)
    out = ROOT / "processed/experiments/bass_sr" / a.song
    for r in (int(v) for v in a.rates.split(",")):
        t = time.time()
        label = "universr_" + ("resynth_" if a.source == "resynth" else "") + f"sr{r}" + a.tag
        lo, back = downsample(xs, r)
        # 10 s at a time (a minute at once: out of memory on 8 GB), 2 s overlaps, raised-cosine crossfades
        win, hop, ov = 10 * r, 8 * r, 2 * FS
        fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov))
        n_out = int(round(len(lo) * FS / r))
        acc, wsum = np.zeros(n_out + FS), np.zeros(n_out + FS)
        starts = list(range(0, max(len(lo) - (win - hop), 1), hop))
        for i, s0 in enumerate(starts):
            with torch.no_grad():
                yc = model.enhance(lo[s0:s0 + win].astype(np.float32), input_sr=r, ode_steps=a.steps,
                                   guidance_scale=a.guidance, seed=42 + i)
            yc = yc.squeeze().float().cpu().numpy()
            b0 = int(round(s0 * FS / r))
            m = min(len(yc), len(acc) - b0)
            w = np.ones(m)
            if i > 0:
                w[:ov] = fade
            if i < len(starts) - 1:
                w[-ov:] = fade[::-1]
            acc[b0:b0 + m] += yc[:m] * w
            wsum[b0:b0 + m] += w
            torch.cuda.empty_cache()
        y = (acc / np.maximum(wsum, 1e-9))[:len(x)]
        y = np.pad(y, (0, len(x) - len(y))) * peak / 0.5
        d = out / label
        xo = 0.9 * r / 2
        merged = (signal.sosfiltfilt(signal.butter(8, xo, "low", fs=FS, output="sos"), x)
                  + signal.sosfiltfilt(signal.butter(8, xo, "high", fs=FS, output="sos"), y))
        bass_level.write_folder(d, {"0_input": x, "1_down": back * peak / 0.5, "2_universr": y, "3_merged": merged}, FS)
        print(f"{label}: {time.time() - t:.0f}s, GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
