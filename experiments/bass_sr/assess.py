"""Assess the AudioSR runs (run_sr.py) on the excerpt, against the stem and a studio bass:

  tone       1/3-octave levels re a studio bass's long-term tone (both re 100 Hz)
  harmonic   how far the notes' harmonics stand over what lies between them, per band (dB, about 0 for
             noise): whether what AudioSR made up above the cutoff follows the notes (a string) or not
  kept       how much AudioSR changed the band it was given (its raw output re the input, below 0.9 x the
             cutoff; SDR, dB: higher = more faithful)
  clicks     impulsive >1.5 kHz events standing out of the bass by more than -30 dB, per minute
  spectra.png  4 s of each

  uv run python experiments/bass_sr/assess.py [--song S] [--start 40]
"""

import argparse
import re
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rehearsal_master import config, note_eq  # noqa: E402

FS = 48000
LEVEL = 10 ** (-12 / 20)
CENTS = [40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000]
HBANDS = [(90, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def rd(p: Path) -> np.ndarray:
    return sf.read(p, dtype="float64", always_2d=True)[0].mean(axis=1) / LEVEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="260924_230957_TrLR Stanza OK")
    ap.add_argument("--start", type=float, default=40.0)
    a = ap.parse_args()
    base = ROOT / "processed/experiments/bass_sr" / a.song
    studio = json.loads((ROOT / "rehearsal_master/references/raw_multitracks.json").read_text())
    rt = np.interp(np.log(CENTS), np.log(studio["grid_hz"]), np.array(studio["instruments"]["bass"]["tone"]))
    rt -= rt[CENTS.index(100)]
    an = np.load(ROOT / "processed/experiments/bass_note_eq" / a.song / "raw/analysis.npz")
    ne = config.load(None)["tracks"]["bass"]["note_eq"]
    nfft, hop = 8192, 512
    df = FS / nfft
    nb = int(12000 / df) + 1
    edges = np.round(np.array([e for b in HBANDS for e in b][::2] + [HBANDS[-1][1]]) / df).astype(int)
    j_off = int(round(a.start * FS / hop))

    def tone(x):
        f, P = signal.welch(x, FS, nperseg=1 << 15)
        b = 10 * np.log10(np.array([P[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum() for c in CENTS]) + 1e-30)
        return b - b[CENTS.index(100)] - rt

    def harmonic(x):
        _, _, Z = signal.stft(x.astype(np.float32), FS, nperseg=nfft, noverlap=nfft - hop)
        P = (np.abs(Z[:nb]) ** 2).T.astype(float)
        H = []
        for j in range(0, P.shape[0], 4):
            c = an["pitch"][j_off + j] if j_off + j < len(an["pitch"]) else np.nan
            if np.isfinite(c):
                H.append(note_eq._hnr(P[j], c, df, edges, ne)[0])
        return np.nanmedian(H, axis=0)

    def clicks(x):
        h = signal.sosfiltfilt(signal.butter(4, 1500, "high", fs=FS, output="sos"), x)
        pk = ndimage.maximum_filter1d(np.abs(h), int(0.002 * FS))
        full = np.sqrt(ndimage.uniform_filter1d(x * x, int(0.03 * FS)) + 1e-20)
        hr = np.sqrt(ndimage.uniform_filter1d(h * h, int(0.03 * FS)) + 1e-20)
        _, n = ndimage.label(ndimage.binary_dilation((pk > 6 * hr) & (20 * np.log10(pk / full) > -30), iterations=480))
        return n / (len(x) / FS / 60)

    rows, spec = {}, {}
    dirs = sorted(d for d in base.iterdir() if d.is_dir() and any(d.glob("2_*.flac")))
    x0 = rd(dirs[0] / "0_input.flac")
    rows["stem (input)"] = (tone(x0), harmonic(x0), np.nan, clicks(x0))
    spec["stem (input)"] = x0
    for d in dirs:
        y = rd(d / "3_merged.flac") if (d / "3_merged.flac").exists() else rd(next(d.glob("2_*.flac")))
        raw, xin = rd(next(d.glob("2_*.flac"))), rd(d / "0_input.flac")
        if "resynth_" in d.name:
            rows.setdefault("resynth (input)", (tone(xin), harmonic(xin), np.nan, clicks(xin)))
            spec.setdefault("resynth (input)", xin)
        m_ = re.search(r"sr(\d+)", d.name)
        rate = int(m_.group(1)) if m_ else None
        kept = np.nan
        if rate:
            lp = signal.butter(8, 0.9 * rate / 2, "low", fs=FS, output="sos")
            u, v = signal.sosfiltfilt(lp, xin), signal.sosfiltfilt(lp, raw)
            kept = 10 * np.log10(np.sum(u ** 2) / np.sum((u - v) ** 2))
        rows[d.name] = (tone(y), harmonic(y), kept, clicks(y))
        spec[d.name] = y
    print(f"{'':22s}{'tone re a studio bass (dB, re 100 Hz)':^{6 * len(CENTS)}s}   harmonic (dB over between) per band"
          f"{'':4s}kept  clicks")
    labels = ["90-250", "250-500", "500-1k", "1k-2k", "2k-4k", "4k-8k"]
    print(f"{'':22s}" + "".join(f"{c:>6d}" for c in CENTS) + "   " + " ".join(f"{b:>9s}" for b in labels) + "   SDR  /min")
    for k, (t, h, kept, c) in rows.items():
        print(f"{k:22s}" + "".join(f"{v:6.1f}" for v in t) + "   " + " ".join(f"{v:9.1f}" for v in h) + f"  {kept:5.1f} {c:5.1f}")
    (base / "assess.json").write_text(json.dumps({k: {"tone_re_studio": np.round(t, 1).tolist(), "harmonic_db": np.round(h, 1).tolist(),
                                                       "kept_sdr_db": None if np.isnan(kept) else round(float(kept), 1),
                                                       "clicks_per_min": round(c, 1)} for k, (t, h, kept, c) in rows.items()},
                                                  indent=1))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(spec)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 10), sharey=True)
    for ax, (k, y) in zip(axs.ravel(), spec.items()):
        f, t, Z = signal.stft(y[int(20 * FS):int(24 * FS)], FS, nperseg=2048, noverlap=2048 - 128)
        ax.pcolormesh(t + a.start + 20, f, 20 * np.log10(np.abs(Z) + 1e-9), vmin=-130, vmax=-20, cmap="magma", shading="auto")
        ax.set_yscale("log")
        ax.set_ylim(30, 20000)
        ax.set_title(k)
    plt.tight_layout()
    plt.savefig(base / "spectra.png", dpi=50)


if __name__ == "__main__":
    main()
