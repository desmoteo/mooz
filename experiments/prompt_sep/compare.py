"""Measure prompt-based stems against the current RoFormer stems of the same stretch.

    uv run python experiments/prompt_sep/compare.py processed/experiments/prompt_sep/<model>/<song@a-bs>

Per stem: level re the mix, agreement with the RoFormer stem (correlation and an SDR
taking the RoFormer stem as reference), stereo correlation, top-octave level (12-20 kHz
re 1-4 kHz). For the snare: the buzz, the level between hits re the hits.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
LEVEL = 10 ** (-12 / 20)
ROFORMER = {  # new stem -> cached RoFormer stems that hold the same instrument
    "vocals": ["vocals", "vocal_room"], "bass": ["bass"], "guitars": ["guitar"],
    "drums": ["kick", "snare", "toms", "hihat", "cymbals"], "other": ["other"],
    "kick": ["kick"], "snare": ["snare"], "toms": ["toms"], "hihat": ["hihat"], "cymbals": ["cymbals"],
}


def db(x):
    return 10 * np.log10(max(float(x), 1e-20))


def band_db(x, fs, lo, hi):
    f, p = signal.welch(x.mean(1), fs, nperseg=8192)
    return db(p[(f >= lo) & (f < hi)].sum())


def snare_buzz(y, ref, fs):
    """Energy more than 120 ms after any snare hit (hits from the RoFormer snare) re
    the energy within 120 ms of the hits, 150 Hz-6 kHz."""
    sos = signal.butter(4, [150, 6000], "band", fs=fs, output="sos")
    hop = int(0.01 * fs)
    env = lambda s: np.sqrt(np.convolve(signal.sosfilt(sos, s.mean(1)) ** 2, np.ones(hop) / hop, "same")[::hop])
    e_ref = env(ref)
    peaks, _ = signal.find_peaks(e_ref, height=np.percentile(e_ref, 99) * 0.25, distance=8)
    near = np.zeros(len(e_ref), bool)
    for p in peaks:
        near[p:p + 12] = True
    e = env(y) ** 2
    return db(e[~near].mean() / e[near].mean()), len(peaks)


def main() -> None:
    d = Path(sys.argv[1])
    m = re.match(r"(.+)@([\d.]+)-([\d.]+)s$", d.name)
    song, a, b = (m.group(1), float(m.group(2)), float(m.group(3))) if m else (d.name, 0, None)
    mix, fs = sf.read(d / "0_mix.flac", dtype="float64")
    mix /= LEVEL
    s0, n = int(a * fs), len(mix)
    cache = ROOT / "processed" / "stems" / song
    ref = lambda names: sum(sf.read(cache / f"{x}.wav", start=s0, frames=n, dtype="float64")[0] for x in names)
    mix_db = db(np.mean(mix ** 2))
    rows = []
    for stage, prefix in (("1_stems", "1"), ("2_drums", "2"), ("2_drums_drumsep", "2ds")):
        for p in sorted((d / stage).glob("*.flac")):
            name = p.stem
            y = sf.read(p, dtype="float64")[0] / LEVEL
            row = {"stem": f"{prefix} {name}", "level_db": db(np.mean(y ** 2)) - mix_db,
                   "lr_corr": float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]),
                   "top_db": band_db(y, fs, 12000, 20000) - band_db(y, fs, 1000, 4000)}
            if name in ROFORMER:
                r = ref(ROFORMER[name])
                row["rof_level_db"] = db(np.mean(r ** 2)) - mix_db
                row["corr"] = float(np.sum(y * r) / np.sqrt(np.sum(y * y) * np.sum(r * r)))
                row["sdr_vs_rof"] = db(np.sum(r * r)) - db(np.sum((y - r) ** 2))
                row["rof_top_db"] = band_db(r, fs, 12000, 20000) - band_db(r, fs, 1000, 4000)
            if name == "snare":
                r = ref(["snare"])
                row["buzz_db"], hits = snare_buzz(y, r, fs)
                row["rof_buzz_db"], _ = snare_buzz(r, r, fs)
                row["hits"] = hits
            rows.append(row)
    keys = ["level_db", "rof_level_db", "corr", "sdr_vs_rof", "lr_corr", "top_db", "rof_top_db", "buzz_db", "rof_buzz_db"]
    print(f"{'stem':<14}" + "".join(f"{k:>13}" for k in keys))
    for r in rows:
        print(f"{r['stem']:<14}" + "".join(f"{r[k]:>13.2f}" if k in r else f"{'':>13}" for k in keys))
    print(f"mix lr_corr {np.corrcoef(mix[:, 0], mix[:, 1])[0, 1]:.2f}")
    (d / "compare.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
