# Run from the project root: uv run python experiments/buzz/buzz_lift.py
"""Does the chain lift the between-hit floor of the hats and cymbals (where the buzz lives)?
Per song: the 1.2-12 kHz level of the stem between its own strokes while the kit plays,
against the level on its strokes, before (1_raw) and after the chain (2_stems), and after
the drum bus (4_drum_bus, all pieces together)."""
import json, glob, sys, pathlib
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, ".")
from rehearsal_master import effects, cleanup
FR, LO, HI = 0.01, 1200.0, 12000.0
def frames_db(x, fs):
    b = signal.sosfilt(signal.butter(4, [LO, HI], "bandpass", fs=fs, output="sos"), x, axis=0)
    n = int(FR * fs); m = len(b) // n
    return 10 * np.log10(np.mean(b[: m * n].reshape(m, n, -1) ** 2, axis=(1, 2)) + 1e-20)
def to_frames(mask, fs):
    n = int(FR * fs); m = len(mask) // n
    return mask[: m * n].reshape(m, n).mean(axis=1) > 0.5
print(f"{'song':16} {'stem':8} {'hits raw':>9} {'between raw':>12} {'gap raw':>8} | {'hits 2_stems':>12} {'between':>8} {'gap':>6} | {'lift of between re hits (dB)':>28}")
for rep_f in sorted(glob.glob("processed/trial-blend/reports/*.json")):
    rep = json.load(open(rep_f)); title = rep["title"]; song = pathlib.Path(rep["file"]).stem
    st = pathlib.Path("processed/trial-blend/stages") / song
    fs = 48000; a, b = int(rep["trim"]["start_s"] * fs), int(rep["trim"]["end_s"] * fs)
    raw = lambda n: sf.read(st / "1_raw" / f"{n}.wav", dtype="float32", start=a, stop=b, always_2d=True)[0]
    trig = {k: effects.hit_trigger(raw(k), fs, *effects.TRIGGER_BANDS[k]) for k in ("kick", "snare", "toms")}
    playing = to_frames(effects._hold(np.max(list(trig.values()), axis=0) > 0.08, fs, 0.3, 2.0) > 0.5, fs)
    for name in ("hihat", "cymbals"):
        x0 = raw(name)
        own = cleanup.own_hits(x0, fs, name, 300.0)
        own_f = to_frames(np.pad(np.repeat(own, 512)[: len(x0)], (0, max(0, len(x0) - len(own) * 512))) > 0.5, fs)
        m = min(len(playing), len(own_f)); hit_f, gap_f = (own_f & playing)[:m], (~own_f & playing)[:m]
        l0 = frames_db(x0, fs)[:m]
        x1 = sf.read(st / "2_stems" / f"{name}.flac", dtype="float32", always_2d=True)[0]
        l1 = frames_db(x1, fs)[:m]
        h0, g0 = np.median(l0[hit_f]), np.median(l0[gap_f]); h1, g1 = np.median(l1[hit_f]), np.median(l1[gap_f])
        print(f"{title[:16]:16} {name:8} {h0:9.1f} {g0:12.1f} {h0-g0:8.1f} | {h1:12.1f} {g1:8.1f} {h1-g1:6.1f} | {(h0-g0)-(h1-g1):+28.1f}")
