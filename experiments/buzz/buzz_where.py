# Run from the project root: uv run python experiments/buzz/buzz_where.py <folder for buzz_where.json>
"""Where the snare buzz lands: per song and stem, the 1.2-12 kHz level in buzz-only frames
(kit idle, snare idle, not near the stem's own strokes, band playing) against the stem's own
hits and the floor; its share per stem; how much the master keeps of it; HPSS on the hats."""
import json, glob, sys, pathlib
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, ".")
from mooz import effects, cleanup, loudness
FR = 0.01  # frame, s
LO, HI = 1200.0, 12000.0

def frames_db(x, fs, lo=LO, hi=HI):
    sos = signal.butter(4, [lo, hi], "bandpass", fs=fs, output="sos") if lo > 0 else None
    b = signal.sosfilt(sos, x, axis=0) if sos is not None else x
    n = int(FR * fs); m = len(b) // n
    return 10 * np.log10(np.mean(b[: m * n].reshape(m, n, -1) ** 2, axis=(1, 2)) + 1e-20)

def to_frames(mask, fs):
    n = int(FR * fs); m = len(mask) // n
    return mask[: m * n].reshape(m, n).mean(axis=1) > 0.5

out = {}
for rep_f in sorted(glob.glob("processed/trial-blend/reports/*.json")):
    rep = json.load(open(rep_f)); title = rep["title"]
    d = pathlib.Path("processed/stems_ensemble") / pathlib.Path(rep["file"]).stem
    fs = sf.info(d / "kick.wav").samplerate
    a, b = int(rep["trim"]["start_s"] * fs), int(rep["trim"]["end_s"] * fs)
    rd = lambda n: sf.read(d / f"{n}.wav", dtype="float32", start=a, stop=b, always_2d=True)[0]
    trig = {k: effects.hit_trigger(rd(k), fs, *effects.TRIGGER_BANDS[k]) for k in ("kick", "snare", "toms")}
    kit_idle = to_frames(effects._hold(np.maximum(trig["kick"], trig["toms"]) > 0.08, fs, 0.3, 2.0) < 0.5, fs)
    snare_idle = to_frames(effects._hold(trig["snare"] > 0.08, fs, 0.3, 0.6) < 0.5, fs)
    # the exciters: bass, guitar, voice
    exc = sum(rd(n) for n in ("bass", "guitar", "vocals", "vocal_room"))
    exc_db = frames_db(exc, fs, 0, 0)
    band_on = exc_db > np.percentile(exc_db, 90) - 25
    silent = exc_db < np.percentile(exc_db, 90) - 45
    m = min(len(kit_idle), len(snare_idle), len(band_on))
    kit_idle, snare_idle, band_on, silent, exc_db = (v[:m] for v in (kit_idle, snare_idle, band_on, silent, exc_db))
    row = {"frames_s": m * FR, "kit_idle_pct": 100 * kit_idle.mean(), "buzz_frames_pct": None}
    stems = {}
    buzz_pow = {}
    for name in ("hihat", "cymbals", "toms", "snare", "kick", "guitar", "vocals", "other", "bass"):
        x = rd(name)
        lev = frames_db(x, fs)[:m]
        if name in ("hihat", "cymbals", "snare", "toms"):
            own = cleanup.own_hits(x, fs, name, 300.0)  # per 512-hop frame
            own_s = np.repeat(own, 512)[: len(x)]
            own_f = to_frames(np.pad(own_s, (0, len(x) - len(own_s)))> 0.5, fs)[:m]
        else:
            own_f = np.zeros(m, bool)
        buzz_only = kit_idle & snare_idle & band_on & ~own_f & (lev > -200)
        if name == "hihat":
            row["buzz_frames_pct"] = 100 * buzz_only.mean()
            row["buzz_in_kit_gate_closed_pct"] = 100 * (kit_idle & buzz_only).sum() / max(buzz_only.sum(), 1)
        floor = np.median(lev[silent]) if silent.any() else np.nan
        lb = np.median(lev[buzz_only]) if buzz_only.any() else np.nan
        lown = np.median(lev[own_f]) if own_f.any() else np.nan
        lplay = np.median(lev[~kit_idle & band_on]) if (~kit_idle & band_on).any() else np.nan
        r = np.corrcoef(lev[buzz_only], exc_db[buzz_only])[0, 1] if buzz_only.sum() > 50 else np.nan
        stems[name] = {"floor": floor, "buzz": lb, "own_hits": lown, "while_kit_plays": lplay, "r_exciters": r}
        buzz_pow[name] = 10 ** (lb / 10) if np.isfinite(lb) else 0.0
    tot = sum(buzz_pow.values())
    for n in stems: stems[n]["share_pct"] = 100 * buzz_pow[n] / tot
    # the master and the recording in the buzz-only frames (both at the master's loudness)
    y, _ = sf.read(rep["outputs"]["wav"], dtype="float32", always_2d=True)
    x0 = sf.read(rep["file"], dtype="float32", start=a, stop=b, always_2d=True)[0]
    g = 10 ** ((loudness.integrated(y, fs) - loudness.integrated(x0, fs)) / 20)
    ly, lx = frames_db(y, fs)[:m], frames_db(x0 * g, fs)[:m]
    fy, fx = frames_db(y, fs, 0, 0)[:m], frames_db(x0 * g, fs, 0, 0)[:m]
    bo = kit_idle & snare_idle & band_on
    row["master_vs_recording_buzzband_db"] = float(np.median(ly[bo] - lx[bo]))
    row["master_vs_recording_full_db"] = float(np.median(fy[bo] - fx[bo]))
    row["stems"] = stems
    out[title] = row
    print(f"\n{title}: {row['frames_s']:.0f} s, kit idle {row['kit_idle_pct']:.0f}% of the time, buzz-only frames {row['buzz_frames_pct']:.1f}% "
          f"(of which inside kit-idle: {row['buzz_in_kit_gate_closed_pct']:.0f}%); master vs recording in those frames: buzz band {row['master_vs_recording_buzzband_db']:+.1f} dB, full band {row['master_vs_recording_full_db']:+.1f} dB")
    print(f"  {'stem':8} {'floor':>7} {'buzz':>7} {'own':>7} {'play':>7} {'buzz-floor':>10} {'own-buzz':>8} {'share%':>7} {'r(exc)':>7}")
    for n, s in stems.items():
        print(f"  {n:8} {s['floor']:7.1f} {s['buzz']:7.1f} {s['own_hits']:7.1f} {s['while_kit_plays']:7.1f} {s['buzz']-s['floor']:10.1f} {s['own_hits']-s['buzz']:8.1f} {s['share_pct']:7.1f} {s['r_exciters']:7.2f}")
json.dump(out, open(sys.argv[1] + "/buzz_where.json", "w"), indent=1, default=float)

# HPSS on the hi-hat stem of one song: does the buzz sit in the sustained part, the hats in the percussive?
import librosa
for title, song in (("Vita", "260928_231324_TrLR_Vita"), ("Stanza OK", "260924_230957_TrLR Stanza OK")):
    d = pathlib.Path("processed/stems_ensemble") / song
    rep = json.load(open(f"processed/trial-blend/reports/{song}.json"))
    fs = 48000; a = int(rep["trim"]["start_s"] * fs); b = min(int(rep["trim"]["end_s"] * fs), a + 150 * fs)
    rd = lambda n: sf.read(d / f"{n}.wav", dtype="float32", start=a, stop=b, always_2d=True)[0]
    hat = rd("hihat"); mono = hat.mean(axis=1)
    trig = {k: effects.hit_trigger(rd(k), fs, *effects.TRIGGER_BANDS[k]) for k in ("kick", "snare", "toms")}
    kit_idle = to_frames(effects._hold(np.maximum(trig["kick"], trig["toms"]) > 0.08, fs, 0.3, 2.0) < 0.5, fs)
    snare_idle = to_frames(effects._hold(trig["snare"] > 0.08, fs, 0.3, 0.6) < 0.5, fs)
    exc = sum(rd(n) for n in ("bass", "guitar", "vocals", "vocal_room")); exc_db = frames_db(exc, fs, 0, 0)
    band_on = exc_db > np.percentile(exc_db, 90) - 25
    own = cleanup.own_hits(hat, fs, "hihat", 300.0); own_f = to_frames(np.pad(np.repeat(own, 512)[: len(hat)], (0, max(0, len(hat) - len(own) * 512))) > 0.5, fs)
    m = min(len(kit_idle), len(snare_idle), len(band_on), len(own_f))
    buzz_only = (kit_idle & snare_idle & band_on & ~own_f)[:m]; hits = (own_f & band_on)[:m]
    Z = librosa.stft(mono, n_fft=2048, hop_length=512)
    for margin in (1.0, 2.0, 4.0):
        H, P = librosa.decompose.hpss(Z, margin=margin)
        h = librosa.istft(H, hop_length=512, length=len(mono)); p = librosa.istft(P, hop_length=512, length=len(mono))
        def e(x, mask):
            lev = frames_db(x[:, None], fs)[:m]; return 10 * np.log10(np.mean(10 ** (lev[mask] / 10)) + 1e-20)
        print(f"\nHPSS {title} (first 150 s), margin {margin}: buzz-only frames: sustained part {e(h, buzz_only) - e(mono, buzz_only):+.1f} dB, percussive {e(p, buzz_only) - e(mono, buzz_only):+.1f} dB re the stem; "
              f"hat hits: sustained {e(h, hits) - e(mono, hits):+.1f} dB, percussive {e(p, hits) - e(mono, hits):+.1f} dB")
