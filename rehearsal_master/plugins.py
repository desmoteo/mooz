"""Third-party plugins, hosted headless with Spotify's pedalboard.

CHOW Tape Model (Chowdhury DSP, GPL-3.0): a physical model of a reel-to-reel tape
machine (hysteresis, head loss, head bump). Install it once (Linux):

  curl -LO https://github.com/jatinchowdhury18/AnalogTapeModel/releases/download/v2.11.4/ChowTapeModel-Linux-x64-2.11.4.deb
  dpkg-deb -x ChowTapeModel-Linux-x64-2.11.4.deb chowtape && mkdir -p ~/.vst3
  cp -r chowtape/usr/lib/vst3/CHOWTapeModel.vst3 ~/.vst3/

The hysteresis model shifts phase (about 20 degrees in the mids, 100 at 50 Hz), so
it is always run fully wet: blended with the dry signal it would cancel part of the
low end. Its amount is set with drive / saturation instead of a mix.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import signal

CHOWTAPE = "~/.vst3/CHOWTapeModel.vst3"


def available(path: str = CHOWTAPE) -> bool:
    from importlib.util import find_spec

    return find_spec("pedalboard") is not None and Path(os.path.expanduser(path)).exists()


def chowtape(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, dict]:
    """x through CHOW Tape Model; cfg: the chowtape table. The tape reacts to level like a
    real machine, so the signal is first brought to an operating level (level_db RMS, the
    "0 VU" calibration; input_gain_db hits the tape harder or softer than that) and put
    back at its own level afterwards: the same settings act the same on every song."""
    import pedalboard

    p = pedalboard.load_plugin(os.path.expanduser(cfg["path"]))
    settings = {
        "input_gain": cfg["input_gain_db"], "output_gain": 0.0, "dry_wet": 100.0,
        "tape_on_off": True, "tape_drive": cfg["drive"], "tape_saturation": cfg["saturation"],
        "tape_bias": cfg["bias"], "tape_speed": cfg["speed_ips"], "loss_on_off": cfg["loss"],
        "tone_on_off": False, "compression_on_off": False, "wow_flutter_on_off": False,
        "degrade_on_off": False, "chew_on_off": False, "input_filters_on_off": False,
        "oversampling_factor_render": float(cfg["oversampling"]), "oversampling_render_like_real_time": False,
    }
    for k, v in settings.items():
        if k not in p.parameters:
            raise KeyError(f"CHOW Tape Model has no parameter '{k}'")
        setattr(p, k, v)
    rms = float(np.sqrt(np.mean(x * x))) + 1e-20
    cal = 10 ** (cfg["level_db"] / 20) / rms
    # one call for the whole song: pedalboard's chunked streaming does not give the same result
    y = p.process(np.ascontiguousarray((x * cal).T, dtype=np.float32), fs).T.astype(np.float64)
    # its oversampling filters delay it by a few samples: line it up again (it may sit next to
    # other stems that did not go through it)
    a, b = x.mean(axis=1)[: int(30 * fs)], y.mean(axis=1)[: int(30 * fs)]
    lags = range(-8, 65)
    lag = max(lags, key=lambda d: float(np.dot(a[: len(a) - max(d, 0)], b[max(d, 0) :]) if d >= 0
                                         else np.dot(a[-d:], b[: len(b) + d])))
    if lag > 0:
        y = np.concatenate([y[lag:], np.zeros((lag, y.shape[1]))])
    elif lag < 0:
        y = np.concatenate([np.zeros((-lag, y.shape[1])), y[:lag]])
    gain = rms / (float(np.sqrt(np.mean(y * y))) + 1e-20)
    # harmonics: the share of the output that is not a (phase-shifted) copy of the input
    a_, b_ = x.mean(axis=1)[::2], y.mean(axis=1)[::2]
    _, coh = signal.coherence(a_, b_, fs / 2, nperseg=4096)
    _, pyy = signal.welch(b_, fs / 2, nperseg=4096)
    harm = float(10 * np.log10(np.sum((1 - coh) * pyy) / (np.sum(pyy) + 1e-30) + 1e-12))
    return y * gain, {"engine": "chowtape", "drive": cfg["drive"], "saturation": cfg["saturation"],
                      "harmonics_db": round(harm, 1), "delay_samples": lag}
