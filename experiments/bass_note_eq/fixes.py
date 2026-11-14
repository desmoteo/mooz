"""The four bass fixes (fixes.toml), one on top of the other, through the stage-2 bass chain,
for listening, each with its tone against raw studio bass tracks, its punch and its mud:

  chain/2_lows    the fundamental restored along note_eq's checked pitch, to -1.5 dB re the 2nd harmonic
  chain/3_punch   + 3 dB more fundamental at each note's start
  chain/4_mud     + 200 Hz +1.5 dB, 315 Hz -2 dB
  chain/5_clang   + 1.6 kHz -3 dB, no tone-match boost above 900 Hz, note_eq again after the chain
                  (= fixes.toml; the kick duck, 3.5 -> 2 dB, acts in the mix only)

  chain/6_no_resonance  + the resonance suppressor off (it flattened each note's harmonics)
  chain/7_resonance_1s  5_clang with the suppressor on the level averaged over 1 s instead
  chain/9a_no_clicks    7 with the click fixes (code) and without the clean-up after the chain
  chain/9b_punch        + low band (<500 Hz) transient: attack 0.4 -> 0.6, sustain 0 -> -0.4
  chain/9c_less_boom    + 100 Hz low shelf +1.5 -> +0.5 dB
  chain/9d_brighter     + exciter +2 dB
  chain/9e_more_punch   + low band transient: attack 0.8, sustain -0.8
  chain/9f_brighter_more  + exciter +4 dB more
  chain/9g_clean_mids   + the clean-up again after the chain, below 1 kHz only
  chain/9h_punch_fundamental  + 5 dB (was 3) more fundamental at each note's start

(chain/0_now and chain/1_note_eq come from run.py.) FLAC at -12 dB in
processed/experiments/bass_note_eq/<song>/chain/, measurements in fixes.json.

  PYTHONUNBUFFERED=1 uv run python experiments/bass_note_eq/fixes.py > processed/logs/bass_note_eq_fixes.log 2>&1
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mooz import config, effects, eq, profile, stages  # noqa: E402

song = sys.argv[1] if len(sys.argv) > 1 else "260924_230957_TrLR Stanza OK"
st_dir = ROOT / "processed/stages" / song
out = ROOT / "processed/experiments/bass_note_eq" / song
w = stages.StageWriter(out, "flac")
rep = json.loads((ROOT / "processed/reports" / f"{song}.json").read_text())
fs = rep["sample_rate"]
start, end = (int(round(rep["trim"][k] * fs)) for k in ("start_s", "end_s"))
now = config.load(ROOT / "rehearsal.toml")["tracks"]["bass"]
final = config.load(ROOT / "experiments/bass_note_eq/fixes.toml")["tracks"]["bass"]
t0 = time.time()

# the steps, each on top of the one before; the last must be fixes.toml's bass
steps = {}
c = copy.deepcopy(now)
c["note_eq"] = c["note_eq"] | {"enabled": True, "fundamental_pitch": "checked"}
c["bass_fundamental"] = c["bass_fundamental"] | {"restore_db": -1.5}
steps["2_lows"] = copy.deepcopy(c)
c["bass_fundamental"] = c["bass_fundamental"] | {"attack_db": 3.0}
steps["3_punch"] = copy.deepcopy(c)
c["eq"] = [b for b in final["eq"] if b.get("freq_hz") != 1600.0]
steps["4_mud"] = copy.deepcopy(c)
c["eq"] = final["eq"]
c["tone_match"] = final["tone_match"]
c["note_eq"] = c["note_eq"] | {"after_chain": True}
steps["5_clang"] = copy.deepcopy(c)
assert {k: v for k, v in c.items() if k != "ducks"} == {k: v for k, v in final.items() if k != "ducks"}, "steps != fixes.toml"
# beyond fixes.toml: the resonance suppressor took each note's harmonics (they stick out of its half-octave
# smoothing) down by up to 6 dB, then the make-up gain raised everything, the smear between them too
c["resonance"] = c["resonance"] | {"enabled": False}
steps["6_no_resonance"] = copy.deepcopy(c)
# instead: the suppressor on the level averaged over 1 s (the room's box under every note, not the notes' harmonics)
c["resonance"] = c["resonance"] | {"enabled": True, "average_ms": 1000.0}
steps["7_resonance_1s"] = copy.deepcopy(c)
# round 2 (the user on 5/7: clicks, high-frequency artifacts, a little boomy, poor punch, dull). In code: the punch's
# fundamental now rises over 10 ms (it switched on at once: 28 -> 150 clicks/min), and the clean-up leaves each
# note's start alone only below attack.protect_hz (above, the separation noise came in bursts on every note)
c["note_eq"] = c["note_eq"] | {"after_chain": False}  # it took the exciter's new harmonics away (dull) and made bursts
steps["9a_no_clicks"] = copy.deepcopy(c)
c["transient"] = c["transient"] | {"attack": [0.6, 0.15], "sustain": [-0.4, 0.0]}  # below 500 Hz: the start up, the ring down
steps["9b_punch"] = copy.deepcopy(c)
c["eq"] = [b | {"gain_db": 0.5} if b["type"] == "low_shelf" else b for b in c["eq"]]  # 100 Hz shelf +1.5 -> +0.5 dB
steps["9c_less_boom"] = copy.deepcopy(c)
c["exciter"] = c["exciter"] | {"amount_db": c["exciter"]["amount_db"] + 2.0}  # on the cleaned notes now: +2 dB
steps["9d_brighter"] = copy.deepcopy(c)
c["transient"] = c["transient"] | {"attack": [0.8, 0.15], "sustain": [-0.8, 0.0]}  # 9b's move barely registered
steps["9e_more_punch"] = copy.deepcopy(c)
c["exciter"] = c["exciter"] | {"amount_db": c["exciter"]["amount_db"] + 4.0}  # 1-4 kHz still 15-20 dB under a studio bass
steps["9f_brighter_more"] = copy.deepcopy(c)
# without the pass after the chain the box came back (clarity 250-500 Hz 20.5 -> 14.4 dB): again, but only below 1 kHz
c["note_eq"] = c["note_eq"] | {"after_chain": True, "after_chain_keep_above_hz": 1000.0}
steps["9g_clean_mids"] = copy.deepcopy(c)
c["bass_fundamental"] = c["bass_fundamental"] | {"attack_db": 5.0}  # punch: the transient shaper barely moved it
steps["9h_punch_fundamental"] = copy.deepcopy(c)
only = sys.argv[2:] or list(steps)  # e.g. fixes.py "<song>" 6_no_resonance: the others measured from their files

stems_dir = (st_dir / "1_raw").resolve()
p_bass = json.loads((stems_dir / "_profile.json").read_text())["stems"]["bass"]
target = profile.targets(stems_dir.parent, fs).get("bass")
ref = profile.reference_tone("bass", eq.log_grid(fs))
if ref is not None:
    target = (target or {}) | {"reference": ref}
raw = sf.read(st_dir / "1_raw" / "bass.wav", dtype="float64", always_2d=True, start=start, stop=end)[0]

# measurements, on the loud part
A, B = 30.0, 250.0
cents = [40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 630, 1000, 1600, 2500, 4000]
studio = json.loads((ROOT / "mooz/references/raw_multitracks.json").read_text())
rt = np.interp(np.log(cents), np.log(studio["grid_hz"]), np.array(studio["instruments"]["bass"]["tone"]))
rt = rt - rt[cents.index(100)]
onsets = np.load(out / "raw" / "analysis.npz")["onsets"]
onsets = onsets[(onsets > A + 0.5) & (onsets < B - 0.5)] - A


def measure(x):
    x = x[int(A * fs):int(B * fs)].mean(axis=1)
    f, P = signal.welch(x, fs, nperseg=1 << 15)
    b = 10 * np.log10(np.array([P[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum() for c in cents]) + 1e-30)
    b -= b[cents.index(100)]
    lo = signal.sosfiltfilt(signal.butter(4, [40, 120], "band", fs=fs, output="sos"), x) ** 2
    mi = signal.sosfiltfilt(signal.butter(4, [150, 400], "band", fs=fs, output="sos"), x) ** 2
    e = lambda y, a, b2: np.array([y[int((t + a) * fs):int((t + b2) * fs)].mean() for t in onsets])  # noqa: E731
    return {"re studio, dB (re 100 Hz)": dict(zip(map(str, cents), np.round(b - rt, 1).tolist())),
            "punch_db": round(float(np.median(10 * np.log10(e(lo, 0, .06) / e(lo, .15, .35)))), 1),
            "mud_db": round(float(np.median(10 * np.log10(e(mi, .1, .35) / e(lo, .1, .35)))), 1)}


res = {}
for name in ("0_now", "1_note_eq", *(k for k in steps if k not in only)):
    res[name] = measure(sf.read(out / "chain" / f"{name}.flac", dtype="float64", always_2d=True)[0] * 10 ** (12 / 20))
for name, tcfg in ((k, v) for k, v in steps.items() if k in only):
    st: dict = {}
    z = effects.cleanup_track(raw, fs, tcfg, None, st, None, p_bass)
    z = effects.studio_track(z, fs, tcfg, st, None, p_bass, target, None)
    st.pop("_pitch", None)
    w.write(f"chain/{name}", z, fs)
    res[name] = measure(z) | {"bass_fundamental": st.get("bass_fundamental")}
    print(f"[{time.time() - t0:5.0f}s] {name}: {res[name]}", flush=True)
(out / "fixes.json").write_text(json.dumps(res, indent=1, default=float))
res = {k: res[k] for k in ("0_now", "1_note_eq", *steps)}
print(f"\n{'':15s}" + " ".join(f"{c:>5d}" for c in cents) + "   punch   mud   (tone: dB re a studio bass, both re 100 Hz)")
for k, v in res.items():
    print(f"{k:15s}" + " ".join(f"{d:5.1f}" for d in v["re studio, dB (re 100 Hz)"].values())
          + f"   {v['punch_db']:+5.1f} {v['mud_db']:+5.1f}")
