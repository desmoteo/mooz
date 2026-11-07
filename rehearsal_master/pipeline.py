"""The processing chain, run in order by process_file().

  high-pass -> hum removal -> separate (whole recording) -> trim -> [bleed reduction ->
  per-track chain -> drum bus -> sum + the original recording under it -> group balance]
  -> stereo (mono bass) -> tonal EQ -> multiband compression -> saturation -> glue compression
  -> loudness gain -> clipper -> true-peak limiter -> fades
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import character, cleanup, config, dynamics, effects, eq, export, loudness, profile, separate, space, stages, stereo, trim


def _finalize(x: np.ndarray, fs: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Gain to the loudness target, clip, limit. Iterates because clipping and
    limiting take a little loudness away."""
    target, ceiling = cfg["loudness"]["target_lufs"], cfg["loudness"]["ceiling_dbtp"]
    ccfg, lcfg = cfg["clipper"], cfg["limiter"]
    gain_db = target - loudness.integrated(x, fs)
    clip_stats = None
    for iteration in range(1, 6):
        y = x * 10 ** (gain_db / 20)
        if ccfg["enabled"]:
            y, clip_stats = dynamics.soft_clip(y, 10 ** ((ceiling + ccfg["headroom_db"]) / 20), ccfg["knee_db"], ccfg["oversample"])
        y, gr, lim_stats = dynamics.limit(y, fs, ceiling, lcfg["lookahead_ms"], lcfg["release_ms"], lcfg["oversample"])
        err = target - loudness.integrated(y, fs)
        if abs(err) < 0.05 or iteration == 5:
            break
        gain_db += err
    # Gain changes inside the limiter can leave a hair of inter-sample overshoot.
    tp = loudness.true_peak_db(y, lcfg["oversample"])
    if tp > ceiling + 0.01:
        y, gr2, _ = dynamics.limit(y, fs, ceiling, lcfg["lookahead_ms"], lcfg["release_ms"], lcfg["oversample"])
        gr = gr + gr2
    return y, gr, {
        "gain_db": round(gain_db, 2),
        "iterations": iteration,
        "clipper": clip_stats,
        "limiter": lim_stats | {"max_gr_db": round(float(gr.max()), 2)},
    }


def _ride(out: np.ndarray, v: np.ndarray, fs: int, cfg: dict) -> tuple[np.ndarray, dict]:
    """Vocal rider: like an engineer on the vocal fader. Where the band (guitars, cymbals)
    gets dense, the words lose ground against it: measured as the voice-to-band ratio in
    the words' band (1-4 kHz) over window_ms, the vocal is raised where that ratio falls
    below its usual (median) value, and lowered a little where it stands out, only while
    the singer sings. v: the vocal as it is in out."""
    lo, hi = cfg["band_hz"]
    sos = signal.butter(2, [lo, hi], "bandpass", fs=fs, output="sos")
    vb = loudness.filtfilt(sos, v)
    ab = loudness.filtfilt(sos, out - v)
    lv = dynamics.level_db(vb, fs, cfg["window_ms"])
    la = dynamics.level_db(ab, fs, cfg["window_ms"])
    sing = lv > np.percentile(lv[::64], 90) - 20
    if sing.mean() < 0.05:
        return out, {"note": "hardly any singing"}
    r = lv - la
    med = float(np.median(r[sing][::64]))
    g = np.clip(cfg["strength"] * (med - r), -cfg["max_down_db"], cfg["max_up_db"]) * sing
    k = max(1, int(cfg["smooth_ms"] / 1000 * fs))
    from scipy.ndimage import uniform_filter1d

    g = uniform_filter1d(uniform_filter1d(g, k), k)  # a slow hand on the fader
    out = out + (10 ** (g / 20) - 1)[:, None] * v
    return out, {"avg_up_db": round(float(g[g > 0].mean()) if (g > 0).any() else 0.0, 1),
                 "max_up_db": round(float(g.max()), 1), "max_down_db": round(float(-g.min()), 1),
                 "riding_up_pct": round(100 * float(np.mean(g > 1.0)), 0)}


def _mix_tone(x: np.ndarray, fs: int, grid: np.ndarray) -> np.ndarray:
    """Long-term 1/3-octave spectrum relative to its own total level (dB)."""
    s = eq.band_spectrum(x, fs, grid)
    return s - 10 * np.log10(np.sum(10 ** (s / 10)))


def _consistency(y: np.ndarray, fs: int, cfg: dict, out_dir: Path, song: str) -> tuple[np.ndarray, dict]:
    """The same band should sound alike in every song: the mix's long-term tone is moved
    part of the way towards the median tone of this preset's songs. The tones are kept in
    <out dir>/_mixtone/ (measured before this correction); songs processed before the
    library existed are added from their stage-5 files."""
    cc = cfg["consistency"]
    grid = eq.log_grid(fs)
    lib = out_dir / "_mixtone"
    lib.mkdir(parents=True, exist_ok=True)
    for f in (out_dir / "stages").glob("*/5_mix.*"):
        entry = lib / f"{f.parent.name}.json"
        if f.parent.name != song and not entry.exists():
            x, fs_x = sf.read(f, dtype="float32", always_2d=True)
            if fs_x == fs:
                entry.write_text(json.dumps({"fs": fs, "tone": [round(float(v), 2) for v in _mix_tone(x, fs, grid)]}))
    tone = _mix_tone(y, fs, grid)
    (lib / f"{song}.json").write_text(json.dumps({"fs": fs, "tone": [round(float(v), 2) for v in tone]}))
    tones = [d["tone"] for d in (json.loads(f.read_text()) for f in lib.glob("*.json")) if d["fs"] == fs]
    if len(tones) < cc["min_songs"]:
        return y, {"songs": len(tones)}
    curve = profile.tone_match_curve(grid, list(tone), np.median(tones, axis=0), cc)
    y = eq.apply_fir(y, eq.design_fir(grid, curve, fs, 8191))
    return y, {"songs": len(tones), "eq_db": eq.at_freqs(grid, curve)}


def _drum_bus(files: dict[str, Path], lufs: dict[str, float], fs: int, cfg: dict) -> tuple[np.ndarray | None, dict]:
    """Stage 4: the kit as one instrument. Its pieces are balanced against each other
    (kit targets), then the bus gets its EQ, glue compression, parallel ("New York")
    compression and saturation."""
    dcfg = cfg["drum_bus"]
    members = [n for n in dcfg["stems"] if n in files]
    if not dcfg["enabled"] or not members:
        return None, {}
    read = lambda n: sf.read(files[n], dtype="float64", always_2d=True)[0]  # noqa: E731
    bus = sum(read(n) for n in members)
    stats: dict = {"stems": members, "unbalanced_lufs": round(loudness.integrated(bus, fs), 1)}
    bcfg = dcfg["balance"]
    gains: dict = {}
    if bcfg["enabled"]:
        gains, stats["balance"] = cleanup.balance_gains({n: lufs.get(n) for n in members},
                                                        loudness.integrated(bus, fs), bcfg)
        for n, g in gains.items():
            if n in members and g != 0.0:  # None: dropped (too quiet to be that drum)
                bus += (-1.0 if g is None else 10 ** (g / 20) - 1.0) * read(n)
        stats["gains_db"] = {n: ("dropped" if g is None else round(g, 1)) for n, g in gains.items() if g != 0.0}
    if dcfg["eq"]:
        grid = eq.log_grid(fs)
        bus = eq.apply_fir(bus, eq.design_fir(grid, eq.manual_curve(grid, dcfg["eq"], fs), fs, 8191))
    rm = dcfg["room"]
    shells = [n for n in rm["stems"] if n in members]
    if rm["enabled"] and shells:
        # a shared room for the shells (snare, toms): the clean-up took theirs away; the bus
        # compressors below then glue (and the styles' parallel compression crushes) it with the kit
        src = sum(read(n) * (0.0 if gains.get(n, 0.0) is None else 10 ** (gains.get(n, 0.0) / 20)) for n in shells)
        wet, _ = space.send(src, fs, rm | {"type": "room", "level_lu": 0.0})
        lb, lw = loudness.integrated(bus, fs), loudness.integrated(wet, fs)
        if np.isfinite(lb) and np.isfinite(lw):
            bus = bus + wet * 10 ** ((lb + rm["level_lu"] - lw) / 20)
            stats["room"] = {"stems": shells, "level_lu": rm["level_lu"], "rt60_s": rm["rt60_s"]}
        del src, wet
    active = loudness.active_mask(bus, fs)
    c = dcfg["compressor"]
    if c["enabled"]:
        bus, _, stats["compressor"] = dynamics.compress(bus, fs, **{k: c[k] for k in dynamics.BAND_PARAMS},
                                                        active=active, auto_makeup=True)
    pc = dcfg["parallel"]
    if pc["enabled"] and pc["mix"] > 0:
        crushed, _, st = dynamics.compress(bus, fs, **{k: pc[k] for k in dynamics.BAND_PARAMS}, active=active,
                                           auto_makeup=True)
        if pc["drive_db"] > 0:  # trashy: the crushed copy driven into saturation too
            crushed, _ = effects.saturator(crushed, fs, {"engine": "builtin", "drive_db": pc["drive_db"],
                                                         "even": 0.3, "mix": 1.0})
        bus = bus + pc["mix"] * crushed
        stats["parallel"] = st | {"mix": pc["mix"]}
    if dcfg["saturation"]["enabled"]:
        bus, stats["saturation"] = effects.saturator(bus, fs, dcfg["saturation"])
    return bus, stats


STEM_GAIN_MAX_DB = 12.0  # the most the balance moves the processed part of a group


def _balance(out: np.ndarray, original: np.ndarray, g_orig: float, processed_files: dict[str, Path], raw_of,
             stem_lufs: dict[str, float], fs: int, bcfg: dict) -> tuple[dict[str, float | None], dict]:
    """Group balance in the blend. Each stem is measured as it will be heard: its processed
    part plus its share of the original (its raw stem at the blend's gain). The groups' misses
    give the change each group needs (cleanup.balance_gains). The original's share does not
    move, so the processed part gets the gain that yields that change once both add up in
    phase: (1 + r) 10^(change/20) - r, r being the original's amplitude share of the group.
    The gains are applied to out in place; returns them per processed stem (None: dropped)
    and the report by group, with each group's level measured again afterwards."""
    total = loudness.integrated(out + original, fs)
    in_blend, in_orig = {}, {}
    for name, f in processed_files.items():
        r = raw_of(name)
        r *= g_orig
        in_orig[name] = loudness.integrated(r, fs)
        r += sf.read(f, dtype="float64", always_2d=True)[0]
        in_blend[name] = loudness.integrated(r, fs)
        del r
    change, report = cleanup.balance_gains(in_blend, total, bcfg)
    groups = {spec["name"]: [m for m in spec["stems"] if m in processed_files] for spec in bcfg["groups"]}
    gains: dict[str, float | None] = dict(change)  # stems in no group: 0.0, or None (dropped)
    power = lambda levels, members: sum(10 ** (levels[m] / 10) for m in members if np.isfinite(levels[m]))  # noqa: E731
    for gname, members in groups.items():
        if not members:
            continue
        delta = change[members[0]]
        pp, po = power(stem_lufs, members), power(in_orig, members)
        r = float(np.sqrt(po / pp)) if pp > 0 else 0.0
        lin = (1 + r) * 10 ** (delta / 20) - r  # the original's share cannot be turned down with the stems
        gain = float(np.clip(20 * np.log10(lin) if lin > 0 else -STEM_GAIN_MAX_DB, -STEM_GAIN_MAX_DB, STEM_GAIN_MAX_DB))
        for m in members:
            gains[m] = gain
        report[gname]["stem_gain_db"] = round(gain, 1)
        report[gname]["original_share_db"] = round(20 * np.log10(r), 1) if r > 0 else None
    for name, g in gains.items():
        if g != 0.0:
            out += (-1.0 if g is None else 10 ** (g / 20) - 1.0) * sf.read(processed_files[name], dtype="float64", always_2d=True)[0]
    total = loudness.integrated(out + original, fs)
    for gname, members in groups.items():
        if gname not in report or not members:
            continue
        acc = None
        for m in members:
            a = raw_of(m)
            a *= g_orig
            if gains[m] is not None:
                a += sf.read(processed_files[m], dtype="float64", always_2d=True)[0] * 10 ** (gains[m] / 20)
            acc = a if acc is None else acc + a
            del a
        report[gname]["after_lu"] = round(loudness.integrated(acc, fs) - total, 1)
        del acc
    return gains, report


def _tracks(y: np.ndarray, fs: int, cfg: dict, stems: dict[str, Path] | None, start: int, end: int,
            work: Path, stems_dir: Path, details: dict, stage: stages.StageWriter, profiles: dict | None = None,
            targets: dict | None = None, tempo_bpm: float | None = None) -> tuple[np.ndarray, dict]:
    """Stages 1-5 for the stems: (3) drum clean-up, (2) studio processing of every
    stem, (4) the drum bus, (5) the sum with the original recording under it and the
    level balance. Without stems the "mix" chain runs on y. work: this run's folder for
    intermediate files; stems_dir: the song's separation cache (the buzz library sits next
    to it). details collects per-track data for the tracks report; stage writes each
    stage for listening. profiles: each stem's stage-0 profile; targets: each
    instrument's median profile."""
    profiles, targets = profiles or {}, targets or {}
    mix_lufs = loudness.integrated(y, fs)
    if stems is None:
        details["mix"] = {}
        out, st = effects.process_track(y, fs, cfg["tracks"]["mix"], details["mix"])
        return out, {"mix": st}
    read = lambda f: sf.read(f, dtype="float64", always_2d=True, start=start, stop=end)[0]  # noqa: E731
    scfg = cfg["stems"]
    rcfg = scfg["refine"]
    if rcfg["enabled"]:  # bleed reduction: stems re-estimated as sharpened masks on the mix
        refined = cleanup.refine_stems(y, {n: read(f).astype(np.float32) for n, f in stems.items()}, fs,
                                       rcfg["power"], rcfg["floor_db"])
        stems = {}
        for n, a in refined.items():
            stems[n] = work / f"_refined_{n}.wav"
            sf.write(stems[n], a, fs, subtype="FLOAT")
        del refined
        read = lambda f: sf.read(f, dtype="float64", always_2d=True)[0]  # noqa: E731  (already trimmed)
    original = y.copy()  # the recording as prepared: blended under the processed stems in stage 5
    total = float(np.mean(y**2)) + 1e-20
    stats, processed_files, stem_lufs = {}, {}, {}
    # Hit triggers for the ducks and gates, from the unprocessed kick/snare/tom stems (or the
    # whole drum stem with the demucs backend); those stems are processed first.
    sources = {src: (src if src in stems else "drums") for src in effects.TRIGGER_BANDS}
    triggers = {src: effects.hit_trigger(read(stems[n]), fs, *effects.TRIGGER_BANDS[src])
                for src, n in sources.items() if n in stems}
    if "vocals" in stems:  # the singer's words (1-4 kHz), for the unmasking ducks of the guitars
        triggers["vocals"] = effects.hit_trigger(read(stems["vocals"]), fs, 1000.0, 4000.0)
    # Stage 3 (drums), step 1: learn the snare buzz fingerprint (this song's buzz-only frames
    # join the library of all songs), so step 2 can remove it from the drum stems below.
    zcfg = cfg["buzz"]
    w_buzz = None
    if zcfg["enabled"] and any(n in stems for n in zcfg["stems"]):
        library = stems_dir.parent / "_buzz"
        library.mkdir(parents=True, exist_ok=True)
        idle_src = [triggers[k] for k in ("kick", "toms") if k in triggers]
        if idle_src:
            idle = effects._hold(np.max(idle_src, axis=0) > 0.08, fs, 0.3, 2.0) < 0.5
            np.save(library / f"{stems_dir.name}.npy",
                    cleanup.buzz_frames({n: read(stems[n]) for n in ("hihat", "cymbals") if n in stems}, idle, fs, zcfg))
        w_buzz = cleanup.buzz_fingerprint(library, zcfg)
        stats["buzz"] = {"learnt_from_songs": len(list(library.glob("*.npy"))), "fingerprint": w_buzz is not None}
    drum_names = set(cfg["drum_bus"]["stems"]) | {"drums"}
    kit_gr = None  # the kit gate's gain reduction (dB per sample), open when any gated stem is open
    for name in sorted(stems, key=lambda n: n not in sources.values()):
        raw = read(stems[name])
        y -= raw
        tcfg = cfg["tracks"].get(name, cfg["track_defaults"])
        if not np.isfinite(loudness.integrated(raw, fs)):
            # nothing in it (below the loudness gate everywhere), e.g. no "other" instrument in this
            # song: no chain, left out of the tracks report (it is in the original's share)
            stats[name] = {"silent": True}
            continue
        st: dict = {}
        detail = details[name] = effects.begin_detail(raw, fs)
        stem = raw
        if w_buzz is not None and name in zcfg["stems"]:  # stage 3, step 2: remove the buzz
            protect = None
            if name in ("snare", "toms"):  # never on the stem's own hits
                protect = cleanup.own_hits(stem, fs, name, zcfg["protect_ms"])
            # deeper between the stem's own hits, where it holds nothing but buzz and bleed
            near = cleanup.own_hits(stem, fs, name, zcfg["own_hit_ms"][name])
            zc = zcfg | {"floor_db": zcfg["stem_floor_db"][name], "between_floor_db": zcfg["stem_between_floor_db"][name]}
            stem, st["buzz"] = cleanup.remove_buzz(stem, fs, w_buzz, zc, protect, near)
        prof, target = profiles.get(name), targets.get(name)
        ref = profile.reference_tone(name, eq.log_grid(fs))
        if ref is not None:
            target = (target or {}) | {"reference": ref}
        if prof:
            st["profile"] = profile.summary(prof)
        stem = effects.cleanup_track(stem, fs, tcfg, triggers, st, detail, prof)
        g = st.pop("_kit_gate_gr", None)
        if g is not None:
            kit_gr = g if kit_gr is None else np.minimum(kit_gr, g)
        if name in drum_names and stage:
            stage.write(f"3_drum_cleanup/{name}", stem, fs)
            stage.write(f"3_drum_cleanup/{name}_removed", raw - stem, fs)
        del raw
        others = {k: v for k, v in triggers.items() if k != name}
        stem = effects.studio_track(stem, fs, tcfg, st, detail, prof, target, others)  # stage 2
        stage.write(f"2_stems/{name}", stem, fs)
        st.pop("_pitch", None)  # the bass's pitch track, shared by its clean-up and fundamental
        for i, smp in enumerate(st.pop("_samples", [])):  # the drum samples made from this song
            stage.write(f"2_stems/samples/{name}_{i + 1}", smp / (np.abs(smp).max() + 1e-9) * 0.25, fs)
        stem = effects.mix_track(stem, fs, tcfg, triggers, st, detail)
        if tcfg["sends"]:  # its space (plate, slapback, tape echo), set relative to the dry track
            wet, st["sends"] = space.sends(stem, fs, tcfg["sends"], tempo_bpm)
            if wet is not None:
                stem = stem + wet
                del wet
        effects.end_detail(detail, stem, fs)
        stats[name] = st
        st["lufs_in"] = round(detail["lufs_in"], 1)
        lufs_out = detail["lufs_out"]
        st["lufs_out"] = round(lufs_out, 1) if np.isfinite(lufs_out) else None
        stem_lufs[name] = lufs_out if np.isfinite(lufs_out) else None
        processed_files[name] = work / f"_proc_{name}.wav"
        sf.write(processed_files[name], stem.astype(np.float32), fs, subtype="FLOAT")
        del stem
    del triggers
    # Stage 4: the drum kit as one instrument
    bus, stats["drum_bus"] = _drum_bus(processed_files, stem_lufs, fs, cfg)
    kit_members = stats["drum_bus"].get("stems", []) if bus is not None else []
    if bus is not None:
        stage.write("4_drum_bus", bus, fs)
        for n in stats["drum_bus"]["stems"]:
            processed_files.pop(n).unlink(missing_ok=True)
            stem_lufs.pop(n, None)
        processed_files["drum_bus"] = work / "_proc_drum_bus.wav"
        sf.write(processed_files["drum_bus"], bus.astype(np.float32), fs, subtype="FLOAT")
        stem_lufs["drum_bus"] = loudness.integrated(bus, fs)
        stats["drum_bus"]["lufs_out"] = round(stem_lufs["drum_bus"], 1)
        del bus
    # Stage 5: everything together. The processed stems are summed; the original recording goes
    # under them at [mix] original_share of the two parts' loudness (what the separation and the
    # clean-up took out comes back at that level, nothing else); the groups (vocals, guitars,
    # bass, drums) are balanced so that they sit at their targets in that blend.
    residual_lufs = loudness.integrated(y, fs)  # what the separation missed (mix - stems): reported, not added
    stats["residual"] = {"level_db": round(float(10 * np.log10((np.mean(y**2) + 1e-20) / total)), 1),
                         "lufs": round(residual_lufs, 1) if np.isfinite(residual_lufs) else None}
    del y
    out = np.zeros_like(original)
    for f in processed_files.values():
        out += sf.read(f, dtype="float64", always_2d=True)[0]
    unbalanced = loudness.integrated(out, fs)
    share = cfg["mix"]["original_share"]
    if not 0.0 <= share <= 0.5:
        raise ValueError(f"mix.original_share must be between 0 and 0.5, not {share}")
    orig_lufs = loudness.integrated(original, fs)
    g_orig, level_lu = 0.0, None
    if share > 0 and np.isfinite(orig_lufs) and np.isfinite(unbalanced):
        level_lu = 10 * np.log10(share / (1 - share))  # the original's loudness against the processed stems
        g_orig = 10 ** ((unbalanced + level_lu - orig_lufs) / 20)
    original *= g_orig
    stats["blend"] = {"original_share": share, "level_lu": None if level_lu is None else round(level_lu, 2),
                      "gain_db": None if g_orig == 0 else round(20 * np.log10(g_orig), 1),
                      "processed_lufs": round(unbalanced, 1),
                      "original_lufs": None if g_orig == 0 else round(orig_lufs + 20 * np.log10(g_orig), 1)}
    band = cfg["mix"]["kit_gate_band_hz"]
    if g_orig > 0 and band and kit_gr is not None and kit_gr.max() > 0:
        # the original follows the kit gate in the snare buzz's band: while the drummer is not playing,
        # the buzz the gate took from the hat and cymbal stems must not come back through the blend
        gr = kit_gr.astype(np.float64)
        original -= (1 - 10 ** (-gr / 20))[:, None] * effects._band(original, fs, band[0], band[1])
        stats["blend"]["kit_gate"] = {"band_hz": list(band), "max_db": round(float(gr.max()), 1),
                                      "closed_pct": round(100 * float(np.mean(gr > gr.max() / 2)), 1)}
        del gr
    del kit_gr
    bcfg = cfg["balance"]
    gains: dict[str, float | None] = {}
    if bcfg["enabled"]:
        raw_of = lambda n: sum(read(stems[m]) for m in (kit_members if n == "drum_bus" else [n]))  # noqa: E731
        gains, stats["balance"] = _balance(out, original, g_orig, processed_files, raw_of, stem_lufs, fs, bcfg)
        for name, g in gains.items():
            if g != 0.0:
                (stats["drum_bus"] if name == "drum_bus" else stats[name])["balance_gain_db"] = (
                    "dropped" if g is None else round(g, 1))
    out += original
    del original
    rd = cfg["rider"]
    voc = [n for n in rd["stems"] if n in processed_files]
    if rd["enabled"] and voc:
        g_of = lambda n: gains.get(n) or 0.0  # noqa: E731
        v = sum(sf.read(processed_files[n], dtype="float64", always_2d=True)[0] * 10 ** (g_of(n) / 20) for n in voc)
        out, stats["rider"] = _ride(out, v, fs, rd)
        del v
    stats["remix"] = {"mix_lufs": round(mix_lufs, 1), "unbalanced_lufs": round(unbalanced, 1),
                      "remix_lufs": round(loudness.integrated(out, fs), 1)}
    return out, stats


def process_file(path: Path, cfg: dict, *, analyze_only: bool = False, ab: bool = False) -> dict:
    """One recording through the whole chain; returns the report. cfg: from config.for_file()."""
    if analyze_only:
        return _process(path, cfg, None, True, ab)
    # this run's intermediate files live under the output folder (the preset's), never in the
    # shared stems cache, so two presets can run on one song at the same time
    work = path.parent / cfg["output"]["dir"] / ".work" / path.stem
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        return _process(path, cfg, work, False, ab)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            work.parent.rmdir()  # .work itself, when no other run is using it
        except OSError:
            pass


def _process(path: Path, cfg: dict, work: Path | None, analyze_only: bool, ab: bool) -> dict:
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    if x.shape[1] > 2:
        print(f"  note: {x.shape[1]} channels, using the first two", file=sys.stderr)
        x = x[:, :2]
    stereo_in = x.shape[1] == 2
    title, date = export.song_info(path)
    rep: dict = {"file": path.name, "title": title, "date": date, "sample_rate": fs, "channels": x.shape[1],
                 "duration_s": round(len(x) / fs, 2)}
    curves: dict = {}

    before, before_ld = loudness.summary(x, fs)
    rep["original"] = before

    # Trim
    tcfg = cfg["trim"]
    start, end = 0, len(x)
    if tcfg["enabled"]:
        start, end = trim.find_bounds(x, fs, before["integrated_lufs"], tcfg["threshold_rel_db"],
                                      tcfg["min_active_s"], tcfg["pre_roll_s"], tcfg["post_roll_s"])
    if tcfg["start_s"] >= 0:
        start = int(tcfg["start_s"] * fs)
    if tcfg["end_s"] >= 0:
        end = min(len(x), int(tcfg["end_s"] * fs))
    if end <= start:
        raise ValueError(f"trim: nothing left between {start / fs:.2f}s and {end / fs:.2f}s")
    rep["trim"] = {"start_s": round(start / fs, 2), "end_s": round(end / fs, 2)}
    grid = eq.log_grid(fs)
    spec_before = eq.band_spectrum(x[start:end], fs, grid)

    # Rumble filter and mains hum, on the whole recording: that is also what gets separated
    # (so changing the trim never needs a new separation)
    full = x
    if cfg["highpass"]["enabled"]:
        hp = cfg["highpass"]
        full = signal.sosfilt(signal.butter(hp["order"], hp["freq_hz"], "high", fs=fs, output="sos"), full, axis=0)
    if cfg["hum"]["enabled"]:
        lead_in = (int(0.5 * fs), start - int(0.2 * fs)) if start > int(2.2 * fs) else None  # skip the button click
        rep["hum"] = cleanup.detect_hum(full, fs, cfg["hum"], lead_in)
        if rep["hum"]["harmonics"]:
            full = cleanup.remove_hum(full, fs, rep["hum"], cfg["hum"])
    del x
    y = full[start:end].copy()

    out_dir = path.parent / cfg["output"]["dir"]
    details: dict = {}
    # The tonal-balance EQ corrects the room and recorder, so by default it is
    # measured on the recording itself: deliberate per-track moves (kick boost,
    # cymbal shelf, vocal presence) are then kept instead of being "corrected" back.
    pre_tracks_spec = eq.band_spectrum(y, fs, grid) if cfg["eq"]["measure_before_tracks"] else None
    stems = None
    scfg = cfg["stems"]
    if not analyze_only and scfg["enabled"]:
        if separate.available(scfg["backend"]):
            st_ = path.stat()
            key = {"file": path.name, "size": st_.st_size, "mtime": st_.st_mtime, "whole_file": True,
                   "highpass": cfg["highpass"], "hum": cfg["hum"],
                   "stems": {k: scfg[k] for k in ("backend", "model", "shifts", "overlap")}
                   | {"steps": separate.model_steps(scfg)}}
            stems = separate.separate(full, fs, scfg, path.parent / (cfg["output"]["stems_dir"] + separate.cache_name(scfg))
                                      / path.stem, key)
        else:
            print("  note: stem separation not installed (uv sync --group stems); processing the whole mix",
                  file=sys.stderr)
    del full
    stems_dir = path.parent / (cfg["output"]["stems_dir"] + separate.cache_name(scfg)) / path.stem  # one per plan
    profiles, targets = {}, {}
    if stems:
        # Stage 0, per stem: every raw stem profiled before any clean-up; the medians over
        # all songs are what stage 2 matches each stem to
        prof = profile.ensure(stems_dir, stems)
        targets = profile.targets(stems_dir.parent, fs)
        ps = scfg["pan_split"]
        stems = separate.split(stems, ps, stems_dir)  # (optional) the two guitars, by position
        for n in ps["outputs"].values() if ps["enabled"] else ():  # they share their source's profile
            if n in stems and n not in prof["stems"] and ps["input"] in prof["stems"]:
                prof["stems"][n] = prof["stems"][ps["input"]]
                if ps["input"] in targets:
                    targets[n] = targets[ps["input"]]
        vr = scfg["vocal_room"]
        if vr["recombine"] and {"vocals", "vocal_room"} <= set(stems):
            out = work / "_vocals_recombined.wav"
            rep["vocal_room"] = cleanup.recombine_room(stems["vocals"], stems.pop("vocal_room"), out, vr)
            stems["vocals"] = out
            for d in (prof["stems"], targets):  # its profile: the recombined vocal's
                if "vocals_full" in d:
                    d["vocals"] = d["vocals_full"]
        profiles = prof["stems"]
        rep["stem_profiles"] = {n: profile.summary(p) for n, p in profiles.items() if n in stems}
    stage = stages.StageWriter(out_dir / "stages" / path.stem if not analyze_only else None, cfg["output"]["stages"])
    # Stage 0: what kind of song this is; the mix stages adapt to its style
    ccfg = cfg["character"]
    if ccfg["enabled"]:
        read = lambda n: sf.read(stems[n], dtype="float64", always_2d=True, start=start, stop=end)[0]  # noqa: E731
        raw = {n: read(n) for n in ("vocals", "guitar", "guitar_left", "guitar_right") if stems and n in stems}
        trig = ({k: effects.hit_trigger(read(n), fs, *effects.TRIGGER_BANDS[k])
                 for k, n in (("kick", "kick"), ("snare", "snare"), ("toms", "toms")) if n in stems} if stems else None)
        rep["character"] = character.characterise(y, fs, raw, trig)
        del raw, trig
        style = rep["character"]["style"]
        if ccfg["adapt"] and style in ccfg["styles"]:  # the style's table, under the song's own settings
            cfg = config.for_file(cfg, path.name, style)
        stage.json("character", rep["character"] | {"stems": rep.get("stem_profiles", {})})
    if stems:
        stage.link_raw(stems_dir)
    if not analyze_only:
        y, rep["tracks"] = _tracks(y, fs, cfg, stems, start, end, work, stems_dir, details, stage, profiles, targets,
                                   rep.get("character", {}).get("tempo_bpm"))

    # Stereo
    if cfg["stereo"]["enabled"] and stereo_in:
        y = stereo.mid_side(y, fs, cfg["stereo"]["mono_below_hz"], cfg["stereo"]["width"],
                            cfg["stereo"]["mono_ramp_hz"] or None)

    # Tonal EQ
    ecfg = cfg["eq"]
    eq_curve = np.zeros_like(grid)
    target = None
    if ecfg["enabled"]:
        measured = pre_tracks_spec if pre_tracks_spec is not None else eq.band_spectrum(y, fs, grid)
        auto, target = eq.auto_correction(grid, measured, ecfg)
        eq_curve = auto + eq.manual_curve(grid, ecfg["manual"], fs)
        h = eq.design_fir(grid, eq_curve, fs, ecfg["fir_taps"])
        eq_curve = eq.fir_response_db(h, grid, fs)  # what is actually applied
        rep["eq_db"] = eq.at_freqs(grid, eq_curve)
        if not analyze_only:
            y = eq.apply_fir(y, h)

    if analyze_only:
        return rep

    # Dynamics
    active = loudness.active_mask(y, fs)
    mcfg = cfg["multiband"]
    if mcfg["enabled"]:
        y, rep["multiband"], band_gr = dynamics.multiband(
            y, fs, mcfg["crossovers_hz"], mcfg["bands"], active,
            mcfg["balance_strength"] if stereo_in else 0.0, mcfg["balance_max_db"])
        curves["multiband"] = {k: dynamics.decimate(v, fs) for k, v in band_gr.items()}
    scfg_ = cfg["saturation"]
    if scfg_["enabled"]:
        y, rep["saturation"] = effects.saturator(y, fs, scfg_)
    gcfg = cfg["glue"]
    if gcfg["enabled"]:
        params = {k: gcfg[k] for k in dynamics.BAND_PARAMS}
        if gcfg["target_gr_db"] > 0:  # the threshold that gives this much gain reduction while playing
            lo_, hi_ = 5.0, 99.5
            for _ in range(7):
                mid_ = 0.5 * (lo_ + hi_)
                _, _, st_ = dynamics.compress(y, fs, **(params | {"threshold_pct": mid_}), active=active)
                lo_, hi_ = (mid_, hi_) if st_["avg_gr_db"] > gcfg["target_gr_db"] else (lo_, mid_)
            params["threshold_pct"] = round(0.5 * (lo_ + hi_), 1)
        y, gr, rep["glue"] = dynamics.compress(y, fs, **params, active=active)
        rep["glue"]["threshold_pct"] = params["threshold_pct"]
        curves["glue"] = dynamics.decimate(gr, fs)

    if cfg["consistency"]["enabled"]:
        y, rep["consistency"] = _consistency(y, fs, cfg, out_dir, path.stem)
    stage.write("5_mix", y, fs)  # stage 5, before loudness and limiting

    # Loudness, clipper, limiter, fades
    y, lim_gr, rep["finalize"] = _finalize(y, fs, cfg)
    curves["limiter"] = dynamics.decimate(lim_gr, fs)
    y = trim.fade(y, fs, tcfg["fade_in_s"], tcfg["fade_out_s"])

    after, after_ld = loudness.summary(y, fs)
    rep["processed"] = after

    # Outputs
    ocfg = cfg["output"]
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{path.stem}.wav"
    export.write_wav(wav, y, fs, ocfg["bit_depth"])
    rep["outputs"] = {"wav": str(wav)}
    if stage:
        stage.readme(path.stem, {"style": rep.get("character", {}).get("style"), "files": stage.written})
        rep["outputs"]["stages"] = str(stage.root)
    tags = {"title": title, "artist": ocfg["artist"], "album": ocfg["album"].format(date=date or "").strip(),
            "date": date[:4]}
    if ocfg["mp3"]:
        mp3 = out_dir / f"{path.stem}.mp3"
        if export.write_mp3(wav, mp3, ocfg["mp3_bitrate"], tags):
            rep["outputs"]["mp3"] = str(mp3)
        else:
            print("  note: ffmpeg not found, skipping MP3", file=sys.stderr)
    if ab:
        # Level both versions so the original's peaks still fit under the ceiling.
        level = min(after["integrated_lufs"], cfg["loudness"]["ceiling_dbtp"] - before["plr_db"])
        raw = sf.read(path, dtype="float64", always_2d=True, start=start, stop=end)[0][:, : y.shape[1]]
        mix = export.ab_mix(raw, y, fs, level - before["integrated_lufs"], level - after["integrated_lufs"],
                            ocfg["ab_segment_s"])
        ab_path = out_dir / f"{path.stem}_AB.wav"
        export.write_wav(ab_path, mix, fs, 24)
        rep["outputs"]["ab"] = str(ab_path)
        rep["ab_level_lufs"] = round(level, 1)
        if ocfg["mp3"]:
            ab_mp3 = ab_path.with_suffix(".mp3")
            ab_tags = tags | {"title": f"{title} (A/B: original first, switches every {ocfg['ab_segment_s']:g} s)"}
            if export.write_mp3(ab_path, ab_mp3, ocfg["mp3_bitrate"], ab_tags):
                rep["outputs"]["ab_mp3"] = str(ab_mp3)
    if ocfg["report"]:
        from . import report  # matplotlib is only imported when needed

        rdir = out_dir / "reports"
        rdir.mkdir(exist_ok=True)
        spec_after = eq.band_spectrum(y, fs, grid)
        png = rdir / f"{path.stem}.png"
        report.plot(png, rep, grid, spec_before, spec_after, target, eq_curve, curves,
                    before_ld.short_term, after_ld.short_term, cfg)
        rep["outputs"]["report"] = str(png)
        if details:
            tracks_png = rdir / f"{path.stem}_tracks.png"
            report.plot_tracks(tracks_png, rep, details, cfg)
            rep["outputs"]["tracks_report"] = str(tracks_png)
        (rdir / f"{path.stem}.json").write_text(json.dumps(rep, indent=2, default=float))
    return rep
