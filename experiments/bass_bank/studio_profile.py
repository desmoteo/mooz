"""How loud each harmonic of a bass note is, relative to its 2nd harmonic, on real studio bass
tracks: the dry bass tracks of the separation benchmark (12 Cambridge-MT multitracks,
experiments/sep_bench), by register (the note's fundamental). The target bank.py rebalances
towards. The same on a band song's separated stem, for comparison.

  uv run python experiments/bass_bank/studio_profile.py   -> experiments/bass_bank/studio_harmonics.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rehearsal_master import augment, config  # noqa: E402

FS = 48000
K = 10
REGISTERS = [30.0, 45.0, 60.0, 80.0, 110.0, 160.0, 420.0]  # fundamental, Hz: band edges


def harmonic_db(x: np.ndarray, fs: int, f0: np.ndarray, t_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per 10 ms where the pitch is known and the note is loud: each harmonic's level re the 2nd
    (dB, rows k = 1..K) and the fundamental; by demodulation along the pitch (8 Hz)."""
    mono = x.mean(axis=1)
    ok = np.isfinite(f0)
    t = np.arange(len(mono)) / fs
    f = np.exp(np.interp(t, t_f[ok], np.log(f0[ok])))
    on = np.interp(t, t_f, ok.astype(float)) > 0.99
    phi = 2 * np.pi * np.cumsum(f) / fs
    lp = signal.butter(2, 8.0, "low", fs=fs, output="sos")
    dec = slice(0, None, fs // 100)
    A = np.array([np.abs(signal.sosfiltfilt(lp, mono * np.cos(k * phi)) - 1j * signal.sosfiltfilt(lp, mono * np.sin(k * phi)))[dec]
                  for k in range(1, K + 1)])
    use = on[dec] & (A[1] > np.percentile(A[1][on[dec]], 40)) & (f[dec] * K < fs / 2)
    return 20 * np.log10((A[:, use] + 1e-12) / (A[1, use] + 1e-12)), f[dec][use]


def by_register(db: np.ndarray, f: np.ndarray) -> list:
    out = []
    for lo, hi in zip(REGISTERS[:-1], REGISTERS[1:]):
        m = (f >= lo) & (f < hi)
        out.append(np.round(np.median(db[:, m], axis=1), 1).tolist() if m.sum() > 200 else None)
    return out


def main() -> None:
    sys.path.insert(0, str(ROOT / "experiments" / "sep_bench"))
    import evaluate as ev

    pcfg = config.load(None)["tracks"]["bass"]["bass_fundamental"]
    dbs, fs_ = [], []
    for d in sorted(p for p in ev.BENCH.iterdir() if (p / "meta.json").exists()):
        info = json.loads((d / "_tracks/tracks.json").read_text())
        dry = sf.read(d / "_tracks/bass_0.flac", dtype="float64", always_2d=True)[0][info["start"]:]
        f0, t_f = augment.bass_pitch(dry, FS, pcfg)
        db, f = harmonic_db(dry, FS, f0, t_f)
        dbs.append(db), fs_.append(f)
        print(f"{d.name[:36]:36s} {len(f) / 100:5.0f} s of notes, median f0 {np.median(f):5.1f} Hz, "
              f"k=1..6 re 2nd: {np.round(np.median(db, axis=1)[:6], 1).tolist()}", flush=True)
    studio = by_register(np.concatenate(dbs, axis=1), np.concatenate(fs_))
    song = "260924_230957_TrLR Stanza OK"
    an = np.load(ROOT / "processed/experiments/bass_note_eq" / song / "raw" / "analysis.npz")
    rep = json.loads((ROOT / "processed/reports" / f"{song}.json").read_text())
    a, b = (int(round(rep["trim"][k] * FS)) for k in ("start_s", "end_s"))
    x = sf.read(ROOT / "processed/stages" / song / "1_raw/bass.wav", dtype="float64", always_2d=True, start=a, stop=b)[0]
    db, f = harmonic_db(x, FS, an["pitch"], an["t"])
    band = by_register(db, f)
    out = {"k": list(range(1, K + 1)), "registers_hz": REGISTERS, "studio_db_re_2nd": studio, song: band}
    (Path(__file__).with_name("studio_harmonics.json")).write_text(json.dumps(out, indent=1))
    print(f"\nharmonic level re the 2nd (dB), k = 1..{K}")
    for (lo, hi), s, m in zip(zip(REGISTERS[:-1], REGISTERS[1:]), studio, band):
        print(f"  f0 {lo:3.0f}-{hi:3.0f} Hz  studio {s}\n  {'':13s}  band   {m}")


if __name__ == "__main__":
    main()
