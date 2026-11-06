"""UVR's tricks applied to the bass stem, on the benchmark (after bass_models.py):

  pitch     BS-RoFormer SW also run 2 semitones down and up (UVR's "pitch change": the input is
            resampled, separated, resampled back), the passes averaged with the normal one
  max spec  per time-frequency bin the estimate with the largest magnitude (UVR's "Max Spec"
            ensemble: fuller, more bleed), of SW, xlance's bass model and SW on the vocal model's
            instrumental (the cascade plan), full band or only above 300 Hz (the upper partials
            SW loses; below, SW's own estimate)

Scores as bass_models.py: SDR, bleedless/fullness, level kept per band.

  uv run python experiments/sep_bench/bass_tricks.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
from bass_models import BANDS, FS, band_db  # noqa: E402

SHIFTS = [-2, 2]
NF, HOP = 4096, 1024


def max_spec(ests: list[np.ndarray], above_hz: float = 0.0) -> np.ndarray:
    """Per STFT bin the estimate with the largest magnitude (with its phase); below above_hz
    the first estimate as it is."""
    S = np.stack([signal.stft(e.T, FS, nperseg=NF, noverlap=NF - HOP)[2] for e in ests])  # (k, ch, F, T)
    pick = np.take_along_axis(S, np.abs(S).argmax(axis=0)[None], axis=0)[0]
    f = np.fft.rfftfreq(NF, 1 / FS)
    pick[:, f < above_hz] = S[0][:, f < above_hz]
    y = signal.istft(pick, FS, nperseg=NF, noverlap=NF - HOP)[1].T
    n = len(ests[0])
    return np.pad(y, ((0, max(0, n - len(y))), (0, 0)))[:n]


def main():
    hp = ev.config.load(ev.ROOT / "rehearsal.toml")["highpass"]
    sos = signal.butter(hp["order"], hp["freq_hz"], "high", fs=FS, output="sos")
    res: dict[str, list] = {}
    for d in sorted(p for p in ev.BENCH.iterdir() if (p / "meta.json").exists()):
        mix = signal.sosfilt(sos, ev.read(d / "mix.flac", "float64"), axis=0)
        step = {"input": "mix", "models": [{"model": "BS-Roformer-SW.ckpt", "pitch_shift": s, "outputs": {"bass": f"bass~sw{s:+d}"}}
                                           for s in SHIFTS]}
        ev.separate_run("bass_pitch", [step], mix, FS, d.name)
        ref = signal.sosfilt(sos, ev.read(d / "bass.flac", "float64"), axis=0)
        n = len(ref)
        sw = ev.read(ev.OUT / "stems/superset" / d.name / "models/1_mix/BS-Roformer-SW/bass.flac", "float64")[:n] * 10 ** 0.6
        shifted = [y.astype(np.float64)[:n] for y in ev.load_run("bass_pitch", d.name)[0].values()]
        xl = ev.load_run("bass_models", d.name)[0]["bass~xlance"].astype(np.float64)[:n]
        cas = ev.read(ev.OUT / "stems/cascade" / d.name / "bass.wav", "float64")[:n]
        cands = {
            "SW": sw,
            "SW pitch -2 alone": shifted[0], "SW pitch +2 alone": shifted[1],
            "SW + pitch -2/+2 averaged": (sw + shifted[0] + shifted[1]) / 3,
            "max spec SW, xlance, cascade": max_spec([sw, xl, cas]),
            "max spec >300 Hz SW, xlance, cascade": max_spec([sw, xl, cas], 300.0),
            "max spec >300 Hz SW, pitch -2/+2": max_spec([sw, *shifted], 300.0),
        }
        for k, y in cands.items():
            res.setdefault(k, []).append([ev.sdr(ref, y), *ev.bleed_full(ref, y), *(band_db(y, ref, lo, hi) for lo, hi in BANDS)])
        print(d.name, "  ".join(f"{k.split()[0]} {v[-1][0]:5.2f}" for k, v in list(res.items())[:4]), flush=True)
    print(f"\n{'variant':38s} {'SDR':>6s} {'bleedl':>7s} {'fullns':>7s}  level re true bass: " + "  ".join(f"{lo}-{hi}" for lo, hi in BANDS))
    for k, v in res.items():
        m = np.mean(v, axis=0)
        print(f"{k:38s} {m[0]:6.2f} {m[1]:7.2f} {m[2]:7.2f}  " + "  ".join(f"{x:+8.1f}" for x in m[3:]))


if __name__ == "__main__":
    main()
