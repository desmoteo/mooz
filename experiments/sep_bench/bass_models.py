"""Bass candidates against BS-RoFormer SW on the benchmark: SDR, bleedless/fullness, the
level kept per octave (the upper partials are what SW loses), and SW averaged with each
(weights fitted, checked leave-one-song-out).

  uv run python experiments/sep_bench/bass_models.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

MODELS = {"beatloo": "bs_bass_beatloo_labs.ckpt", "xlance": "bs_bass_xlancer.ckpt"}
BANDS = [(31, 125), (125, 500), (500, 1000), (1000, 2000), (2000, 4000)]
FS = 48000


def band_db(est, ref, lo, hi):
    sos = signal.butter(4, [lo, hi], "band", fs=FS, output="sos")
    e = lambda x: float((signal.sosfilt(sos, x.mean(1)) ** 2).sum())
    return 10 * np.log10(e(est) / e(ref))


def main():
    hp = ev.config.load(ev.ROOT / "rehearsal.toml")["highpass"]
    sos = signal.butter(hp["order"], hp["freq_hz"], "high", fs=FS, output="sos")
    res = {k: [] for k in ["sw", *MODELS]}
    gram = {k: ([], [], []) for k in MODELS}
    for d in sorted(p for p in ev.BENCH.iterdir() if (p / "meta.json").exists()):
        mix = signal.sosfilt(sos, ev.read(d / "mix.flac", "float64"), axis=0)
        step = {"input": "mix", "models": [{"model": m, "outputs": {"bass": f"bass~{k}"}} for k, m in MODELS.items()]}
        ev.separate_run("bass_models", [step], mix, FS, d.name)
        ref = signal.sosfilt(sos, ev.read(d / "bass.flac", "float64"), axis=0)
        est = {"sw": ev.read(ev.OUT / "stems/superset" / d.name / "models/1_mix/BS-Roformer-SW/bass.flac", "float64") * 10 ** 0.6}
        models, _ = ev.load_run("bass_models", d.name)
        est |= {k: models[f"bass~{k}"].astype(np.float64) for k in MODELS}
        for k, y in est.items():
            y = y[: len(ref)]
            res[k].append([ev.sdr(ref, y), *ev.bleed_full(ref, y), *(band_db(y, ref, lo, hi) for lo, hi in BANDS)])
        for k in MODELS:
            Y = np.stack([est["sw"][: len(ref)].ravel(), est[k][: len(ref)].ravel()])
            gram[k][0].append(Y @ Y.T), gram[k][1].append(Y @ ref.ravel()), gram[k][2].append(float(ref.ravel() @ ref.ravel()))
        print(d.name, "  ".join(f"{k} {v[-1][0]:5.2f}" for k, v in res.items()), flush=True)
    print(f"\n{'model':8s} {'SDR':>6s} {'bleedl':>7s} {'fullns':>7s}  level re true bass per band: " + "  ".join(f"{lo}-{hi}" for lo, hi in BANDS))
    for k, v in res.items():
        m = np.mean(v, axis=0)
        print(f"{k:8s} {m[0]:6.2f} {m[1]:7.2f} {m[2]:7.2f}  " + "  ".join(f"{x:+8.1f}" for x in m[3:]))
    for k, (G, b, r) in gram.items():
        w = ev.fit(G, b, r)
        loo = np.mean([ev.song_sdr(G[i], b[i], r[i], ev.fit(G[:i] + G[i + 1:], b[:i] + b[i + 1:], r[:i] + r[i + 1:])) for i in range(len(G))])
        print(f"SW + {k}: weights {w.round(2)}  in-sample {np.mean([ev.song_sdr(g, bb, rr, w) for g, bb, rr in zip(G, b, r)]):.2f}  loo {loo:.2f}")


if __name__ == "__main__":
    main()
