"""The per-note dynamic EQ (mooz/note_eq.py) on one song's bass, for listening:

  raw/        on the separated stem (stage 1), against the clean-up the pipeline uses now
              (harmonic_clean): the output, stage A alone (harmonic space), what each took out
  chain/      the whole stage-2 bass chain (clean-up + studio) as it runs now, and with the note EQ
              in place of harmonic_clean: what the band hears in the mix
  post/       on the finished stage-2 stem (2_stems/bass.flac), as a post-process
  spectra.png before / after / removed, a few seconds of each

All FLAC, -12 dB like the stage files, aligned with stages/<song>/2_stems. Writes nothing in
stages/ itself.

  PYTHONUNBUFFERED=1 uv run python experiments/bass_note_eq/run.py ["<song>"] > processed/logs/bass_note_eq.log 2>&1
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mooz import augment, config, effects, eq, note_eq, profile, stages  # noqa: E402

song = sys.argv[1] if len(sys.argv) > 1 else "260924_230957_TrLR Stanza OK"
SEE = (60.0, 64.0)  # the spectrogram's seconds
st_dir = ROOT / "processed/stages" / song
out = ROOT / "processed/experiments/bass_note_eq" / song
w = stages.StageWriter(out, "flac")
rep = json.loads((ROOT / "processed/reports" / f"{song}.json").read_text())
fs = rep["sample_rate"]
start, end = (int(round(rep["trim"][k] * fs)) for k in ("start_s", "end_s"))
cfg = config.load(ROOT / "rehearsal.toml")
bass = cfg["tracks"]["bass"]
ne_cfg = bass["note_eq"] | {"enabled": True}
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB] {msg}", flush=True)


def bands(x, y):
    from scipy import signal

    row = {}
    for lo, hi in ((30, 90), (90, 250), (250, 600), (600, 1500), (1500, 4000), (4000, 12000)):
        sos = signal.butter(4, [lo, hi], "band", fs=fs, output="sos")
        e = lambda z: float((signal.sosfilt(sos, z.mean(axis=1)) ** 2).sum()) + 1e-20  # noqa: E731
        row[f"{lo}-{hi}"] = round(10 * np.log10(e(y) / e(x)), 1)
    return row


index = {"song": song, "note_eq": ne_cfg, "files": {}, "stats": {}, "dB change per band": {}}

# --- raw: the separated stem ---
raw = sf.read(st_dir / "1_raw" / "bass.wav", dtype="float64", always_2d=True, start=start, stop=end)[0]
log(f"raw stem: {len(raw) / fs:.1f} s")
w.write("raw/0_input", raw, fs)
parts = {}
y, stats, _ = note_eq.note_eq(raw, fs, ne_cfg, bass["bass_fundamental"], parts)
log(f"note EQ: {stats}")
w.write("raw/1_note_eq", y, fs)
w.write("raw/1a_harmonic_space", parts["harmonic_space"], fs)
w.write("raw/1_note_eq_removed", raw - y, fs)
w.write("raw/1a_harmonic_space_removed", raw - parts["harmonic_space"], fs)
w.write("raw/1b_attack_eq_change", y - parts["harmonic_space"], fs)
index["stats"]["raw note_eq"] = stats
index["dB change per band"]["raw: harmonic space (A)"] = bands(raw, parts["harmonic_space"])
index["dB change per band"]["raw: attack EQ (B)"] = bands(parts["harmonic_space"], y)
np.savez_compressed(out / "raw" / "analysis.npz", t=parts["t"], pitch=parts["pitch"], space_hz=parts["space_hz"],
                    score=parts["score"], onsets=parts["onsets"])
a, b = (int(v * fs) for v in SEE)
spec_raw = (raw[a:b].copy(), y[a:b].copy())
del y, parts
pitch = augment.bass_pitch(raw, fs, bass["bass_fundamental"])
old, st_old = augment.harmonic_clean(raw, fs, bass["harmonic_clean"], *pitch)
log(f"harmonic_clean (now): {st_old}")
w.write("raw/2_harmonic_clean_now", old, fs)
w.write("raw/2_harmonic_clean_now_removed", raw - old, fs)
index["stats"]["raw harmonic_clean"] = st_old
index["dB change per band"]["raw: harmonic_clean (now)"] = bands(raw, old)
del old

# --- chain: clean-up + studio, as the pipeline runs them ---
stems_dir = (st_dir / "1_raw").resolve()
prof = json.loads((stems_dir / "_profile.json").read_text()) if (stems_dir / "_profile.json").exists() else None
p_bass = prof["stems"].get("bass") if prof else None
targets = profile.targets(stems_dir.parent, fs)
target = targets.get("bass")
ref = profile.reference_tone("bass", eq.log_grid(fs))
if ref is not None:
    target = (target or {}) | {"reference": ref}
for tag, tcfg in (("now", bass), ("note_eq", bass | {"note_eq": ne_cfg})):
    st: dict = {}
    z = effects.cleanup_track(raw, fs, tcfg, None, st, None, p_bass)
    z = effects.studio_track(z, fs, tcfg, st, None, p_bass, target, None)
    st.pop("_pitch", None)
    w.write(f"chain/{'1_note_eq' if tag == 'note_eq' else '0_now'}", z, fs)
    index["stats"][f"chain {tag}"] = st
    if tag == "now":
        ref2 = sf.read(st_dir / "2_stems" / "bass.flac", dtype="float64", always_2d=True)[0] * 10 ** (12 / 20)
        n = min(len(ref2), len(z))
        index["chain_now_vs_stage_file_db"] = round(10 * np.log10(np.sum((ref2[:n] - z[:n]) ** 2) / np.sum(ref2[:n] ** 2)), 1)
        log(f"chain as now: matches 2_stems/bass.flac to {index['chain_now_vs_stage_file_db']} dB")
        del ref2
    else:
        index["dB change per band"]["chain: note EQ vs now"] = bands(prev, z)
    prev = z
    log(f"chain {tag} done")
del raw, prev, z

# --- post: on the finished stage-2 stem ---
proc = sf.read(st_dir / "2_stems" / "bass.flac", dtype="float64", always_2d=True)[0] * 10 ** (12 / 20)
parts = {}
y, stats, _ = note_eq.note_eq(proc, fs, ne_cfg, bass["bass_fundamental"], parts)
log(f"post: {stats}")
w.write("post/0_input", proc, fs)
w.write("post/1_note_eq", y, fs)
w.write("post/1a_harmonic_space", parts["harmonic_space"], fs)
w.write("post/1_note_eq_removed", proc - y, fs)
index["stats"]["post note_eq"] = stats
index["dB change per band"]["post: harmonic space (A)"] = bands(proc, parts["harmonic_space"])
index["dB change per band"]["post: attack EQ (B)"] = bands(parts["harmonic_space"], y)
spec_post = (proc[a:b].copy(), y[a:b].copy())
del proc, y, parts

# --- spectrograms ---
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import signal  # noqa: E402

fig, axs = plt.subplots(2, 3, figsize=(24, 11), sharex=True, sharey=True)
for r, (name, (xi, yi)) in enumerate((("separated stem", spec_raw), ("stage-2 stem", spec_post))):
    for c, (what, sig) in enumerate((("before", xi), ("note EQ", yi), ("removed", xi - yi))):
        f, t, Z = signal.stft(sig.mean(axis=1), fs, nperseg=2048, noverlap=2048 - 128)
        ax = axs[r, c]
        ax.pcolormesh(t + SEE[0], f, 20 * np.log10(np.abs(Z) + 1e-9), vmin=-130, vmax=-30, cmap="magma", shading="auto")
        ax.set_yscale("log")
        ax.set_ylim(30, 12000)
        ax.set_title(f"{name}: {what}")
axs[1, 0].set_xlabel("s")
axs[0, 0].set_ylabel("Hz")
plt.tight_layout()
plt.savefig(out / "spectra.png", dpi=60)
index["files"] = sorted(str(p.relative_to(out)) for p in out.rglob("*.flac"))
(out / "index.json").write_text(json.dumps(index, indent=2, default=float))
log(f"done: {out}")
print(json.dumps(index["dB change per band"], indent=1))
