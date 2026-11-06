"""The per-note dynamic EQ on the separation benchmark (experiments/sep_bench): the band's
plan's bass (Max Spec of xlance, SW and SW on the instrumental) against the true room bass.
Per variant: SDR, bleedless / fullness, the share of what it took out that was bass
(0 = only bleed and noise), and the level kept per band against the stem as it was.

  PYTHONUNBUFFERED=1 uv run python experiments/bass_note_eq/bench.py > processed/logs/bass_note_eq_bench.log 2>&1
"""

import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "sep_bench"))
import evaluate as ev  # noqa: E402
from bass_tricks import max_spec  # noqa: E402

from rehearsal_master import augment, config, note_eq  # noqa: E402

FS = 48000
BANDS = [(90, 250), (250, 600), (600, 1500), (1500, 4000)]
cfg = config.load(ROOT / "rehearsal.toml")["tracks"]["bass"]
ne = cfg["note_eq"] | {"enabled": True}
VARIANTS = {
    "harmonic_clean (now)": lambda x, pitch: augment.harmonic_clean(x, FS, cfg["harmonic_clean"], *pitch)[0],
    "note EQ: harmonic space": lambda x, pitch: note_eq.note_eq(x, FS, ne | {"attack": ne["attack"] | {"enabled": False}},
                                                                cfg["bass_fundamental"])[0],
    "note EQ: + attack EQ": lambda x, pitch: note_eq.note_eq(x, FS, ne, cfg["bass_fundamental"])[0],
}


def band_db(a, b, lo, hi):
    sos = signal.butter(4, [lo, hi], "band", fs=FS, output="sos")
    e = lambda z: float((signal.sosfilt(sos, z.mean(axis=1)) ** 2).sum()) + 1e-20  # noqa: E731
    return 10 * np.log10(e(a) / e(b))


hp = ev.config.load(ev.ROOT / "rehearsal.toml")["highpass"]
sos = signal.butter(hp["order"], hp["freq_hz"], "high", fs=FS, output="sos")
rows = {k: [] for k in ["stem as separated", *VARIANTS]}
for d in sorted(p for p in ev.BENCH.iterdir() if (p / "meta.json").exists()):
    ref = signal.sosfilt(sos, ev.read(d / "bass.flac", "float64"), axis=0)
    n = len(ref)
    sw = ev.read(ev.OUT / "stems/superset" / d.name / "models/1_mix/BS-Roformer-SW/bass.flac", "float64")[:n] * 10 ** 0.6
    xl = ev.load_run("bass_models", d.name)[0]["bass~xlance"].astype(np.float64)[:n]
    cas = ev.read(ev.OUT / "stems/cascade" / d.name / "bass.wav", "float64")[:n]
    est = max_spec([xl, sw, cas]).astype(np.float64)
    del sw, xl, cas
    pitch = augment.bass_pitch(est, FS, cfg["bass_fundamental"])
    line = []
    for k in rows:
        y = est if k == "stem as separated" else VARIANTS[k](est, pitch)
        rem = (est - y).ravel()
        share = float((rem @ ref.ravel()) ** 2 / ((ref.ravel() @ ref.ravel()) * (rem @ rem) + 1e-20)) if rem.any() else np.nan
        bl, fu = ev.bleed_full(ref, y)
        rows[k].append([ev.sdr(ref, y), bl, fu, share, *(band_db(y, est, lo, hi) for lo, hi in BANDS)])
        line.append(f"{rows[k][-1][0]:.2f}")
    print(f"{d.name[:40]:40s} SDR " + " / ".join(line), flush=True)
print(f"\n{'':26s}{'SDR':>7s}{'bleedless':>10s}{'fullness':>9s}{'bass in removed':>16s}"
      + "".join(f"{f'{lo}-{hi}':>11s}" for lo, hi in BANDS) + "   (dB kept per band)")
for k, v in rows.items():
    m = np.nanmean(np.array(v, float), axis=0)
    print(f"{k:26s}{m[0]:7.2f}{m[1]:10.2f}{m[2]:9.2f}{m[3]:16.2f}" + "".join(f"{x:11.1f}" for x in m[4:]))
d_sdr = np.array([r[0] for r in rows["note EQ: + attack EQ"]]) - np.array([r[0] for r in rows["harmonic_clean (now)"]])
print(f"\nnote EQ vs harmonic_clean: SDR {d_sdr.mean():+.2f} dB on average, better in {int((d_sdr > 0).sum())}/{len(d_sdr)} songs")
