"""MVSep Mega 53 stems' bass (its single-stem split, bs_mega_53stem_bass_mvsep.ckpt) on the room
benchmark, against BS-RoFormer SW, xlance's bass model (bass_models.py's run) and the band's plan
(Max Spec of xlance, SW and SW on the instrumental); and that plan with Mega's bass as a fourth
estimate. Scores as bass_models.py: SDR, bleedless / fullness, level kept per band.

  PYTHONUNBUFFERED=1 uv run python experiments/sep_bench/bass_mega.py > processed/logs/bass_mega.log 2>&1
"""

import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
from bass_models import BANDS, FS, band_db  # noqa: E402
from bass_tricks import max_spec  # noqa: E402

MEGA = "bs_mega_53stem_bass_mvsep.ckpt"


def main():
    hp = ev.config.load(ev.ROOT / "rehearsal.toml")["highpass"]
    sos = signal.butter(hp["order"], hp["freq_hz"], "high", fs=FS, output="sos")
    names = ["sw", "xlance", "mega", "plan (max spec)", "plan + mega"]
    res = {k: [] for k in names}
    for d in sorted(p for p in ev.BENCH.iterdir() if (p / "meta.json").exists()):
        mix = signal.sosfilt(sos, ev.read(d / "mix.flac", "float64"), axis=0)
        ev.separate_run("bass_mega", [{"input": "mix", "models": [{"model": MEGA, "outputs": {"bass": "bass~mega"}}]}],
                        mix, FS, d.name)
        ref = signal.sosfilt(sos, ev.read(d / "bass.flac", "float64"), axis=0)
        n = len(ref)
        est = {"sw": ev.read(ev.OUT / "stems/superset" / d.name / "models/1_mix/BS-Roformer-SW/bass.flac", "float64")[:n] * 10 ** 0.6,
               "xlance": ev.load_run("bass_models", d.name)[0]["bass~xlance"].astype(np.float64)[:n],
               "mega": ev.load_run("bass_mega", d.name)[0]["bass~mega"].astype(np.float64)[:n]}
        cas = ev.read(ev.OUT / "stems/cascade" / d.name / "bass.wav", "float64")[:n]
        est["plan (max spec)"] = max_spec([est["xlance"], est["sw"], cas]).astype(np.float64)
        est["plan + mega"] = max_spec([est["xlance"], est["sw"], cas, est["mega"]]).astype(np.float64)
        for k, y in est.items():
            res[k].append([ev.sdr(ref, y), *ev.bleed_full(ref, y), *(band_db(y, ref, lo, hi) for lo, hi in BANDS)])
        print(d.name[:36].ljust(36), "  ".join(f"{k} {v[-1][0]:5.2f}" for k, v in res.items()), flush=True)
    print(f"\n{'model':16s} {'SDR':>6s} {'bleedl':>7s} {'fullns':>7s}  level re true bass per band: "
          + "  ".join(f"{lo}-{hi}" for lo, hi in BANDS))
    for k, v in res.items():
        m = np.mean(v, axis=0)
        print(f"{k:16s} {m[0]:6.2f} {m[1]:7.2f} {m[2]:7.2f}  " + "  ".join(f"{x:+8.1f}" for x in m[3:]))
    dm = np.array([r[0] for r in res["mega"]]) - np.array([r[0] for r in res["xlance"]])
    dp = np.array([r[0] for r in res["plan + mega"]]) - np.array([r[0] for r in res["plan (max spec)"]])
    print(f"\nmega vs xlance: {dm.mean():+.2f} dB SDR, better in {int((dm > 0).sum())}/{len(dm)} songs")
    print(f"plan + mega vs plan: {dp.mean():+.2f} dB SDR, better in {int((dp > 0).sum())}/{len(dp)} songs")


if __name__ == "__main__":
    main()
