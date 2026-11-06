"""Command line: rehearsal-master [files or folders] [options]"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from . import config, pipeline

AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff"}


def _collect(paths: list[Path], out_dir_name: str) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_SUFFIXES and f.is_file())
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"not found: {p}")
    # never re-process our own output (any preset's folder inside the output folder)
    top = Path(out_dir_name).parts[0]
    return [f for f in files if top not in f.parent.parts and not f.stem.endswith("_AB")]


def _row(label: str, s: dict) -> str:
    return (f"  {label:<10} {s['integrated_lufs']:7.1f} LUFS  LRA {s['lra_lu']:4.1f}  "
            f"peak {s['true_peak_dbtp']:+5.1f} dBTP  PLR {s['plr_db']:4.1f}  L-R {s['balance_db']:+4.1f} dB")


def _print(rep: dict) -> None:
    t = rep["trim"]
    print(f"  kept {t['start_s']:.1f}s .. {t['end_s']:.1f}s of {rep['duration_s']:.1f}s")
    hum = rep.get("hum")
    if hum:
        if hum["harmonics"]:
            freqs = ", ".join(f"{h['hz']:.0f}" for h in hum["harmonics"])
            print(f"  hum       {hum['base_hz']:g} Hz mains, notched at {freqs} Hz (seen in the {hum['reference']})")
        else:
            print("  hum       none found")
    if "vocal_room" in rep:
        print(f"  vocals    de-reverb parts recombined, room down {-rep['vocal_room']['room_reduction_db']:.1f} dB (smoothed)")
    ch = rep.get("character")
    if ch:
        print(f"  character {ch['style']}: {ch['why']}")
    print(_row("original", rep["original"]))
    if "processed" in rep:
        print(_row("processed", rep["processed"]))
    tracks = rep.get("tracks", {})
    remix = tracks.get("remix")
    if remix:
        print(f"  remix     mix {remix['mix_lufs']:.1f} LUFS -> {remix['remix_lufs']:.1f} LUFS before the bus chain")
    for name, st in tracks.items():
        if name == "drum_bus":
            if st:
                kit = ", ".join(f"{k} {v:+.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in st.get("gains_db", {}).items())
                comp = st.get("compressor", {})
                par = st.get("parallel")
                room = st.get("room")
                print(f"  [drum bus] kit balance: {kit or 'unchanged'} · glue GR avg {comp.get('avg_gr_db', 0):.1f}"
                      + (f" · parallel mix {par['mix']:.2f}" if par else "")
                      + (f" · room ({'+'.join(room['stems'])}) at {room['level_lu']:+.0f} LU" if room else ""))
            continue
        if name in ("remix", "balance", "buzz", "rider"):
            continue
        parts = []
        if st.get("silent"):
            print(f"  [{name}] silent: nothing to process, added back as it is")
            continue
        if remix and st.get("lufs_in") is not None and st.get("lufs_out") is not None:
            parts.append(f"{st['lufs_in'] - remix['mix_lufs']:+.1f} -> {st['lufs_out'] - remix['remix_lufs']:+.1f} LU re mix")
        if "level_db" in st:
            parts.append("added back unprocessed")
        pr = st.get("profile")
        if pr:
            parts.append(f"SNR {pr['snr_db']:.0f} dB, hf/lf {pr['hf_lf_db']:+.0f}"
                         + (f", punch {pr['punch_db']:.1f}" if "punch_db" in pr else ""))
        if "tone_match_db" in st:
            tm = st["tone_match_db"]
            parts.append("tone match " + " ".join(f"{f}:{g:+.1f}" for f, g in tm.items() if abs(g) >= 0.5)
                         if any(abs(g) >= 0.5 for g in tm.values()) else "tone match ~0")
        if "expander" in st:
            parts.append(f"expander at -{st['expander']['below_db']:.0f} dB")
        if "transient_scale" in st:
            parts.append(f"attack x{st['transient_scale']:.2f}")
        if "balance_correction_db" in st:
            parts.append(f"balance {st['balance_correction_db']:+.1f}")
        for b in st.get("dynamic_eq", []):
            parts.append(f"dyn-EQ {b['freq_hz']:g} Hz avg {-b['avg_cut_db']:+.1f}/max {-b['max_cut_db']:+.1f}")
        if "resonance" in st:
            parts.append(f"resonances avg -{st['resonance']['avg_cut_db']:.1f}/max -{st['resonance']['max_cut_db']:.1f}")
        if "compressor" in st:
            c = st["compressor"]
            parts.append(f"comp GR avg {c['avg_gr_db']:.1f}/max {c['max_gr_db']:.1f}, make-up {c['makeup_db']:+.1f}")
        au = st.get("augment")
        if au:
            parts.append(f"own samples on {au['placed']}/{au['hits']} hits (mix {au['mix']:g})"
                         if au.get("placed") else f"no own samples ({au.get('note', 'too few hits')})")
            lib = au.get("library")
            if lib and lib.get("placed"):
                names = ", ".join(f"{d['sample'][:24]} x{d['tuned']:.2f}" for d in lib["samples"])
                parts.append(f"studio samples on {lib['placed']} hits (mix {lib['mix']:g}: {names})")
        bf = st.get("bass_fundamental")
        if bf:
            parts.append(f"fundamental +{bf['avg_gain_db']:.1f} dB (pitch found {bf['voiced_pct']:.0f}%)"
                         if "avg_gain_db" in bf else f"fundamental: {bf.get('note')}")
        for snd in st.get("sends", []):
            parts.append(f"{snd['type'].replace('_', ' ')} at {snd['level_lu']:+.0f} LU"
                         + (f" ({snd['delay_ms']:.0f} ms)" if snd.get("delay_ms") and snd["type"] != "slap" else ""))
        if "dual" in st:
            d = st["dual"]
            parts.append(f"dual chain: clean mono <{d['crossover_hz']:g} Hz, above drive {d['drive_mix']:g} / chorus {d['chorus_mix']:g}")
        if "grit" in st:
            parts.append(f"grit {st['grit']['level_db']:+.0f} dB")
        if "kick_hz" in st:
            parts.append(f"kick EQ @ {st['kick_hz']:.0f} Hz")
        if "kick_shaper" in st:
            parts.append(f"kick shaper on {st['kick_shaper']['hits']} hits")
        for d in st.get("ducks", []):
            lo, hi = d["band_hz"]
            band = f"<{hi:g}" if lo <= 0 else f"{lo / 1000:g}-{hi / 1000:g}k"
            parts.append(f"ducked {band} on {d['source']} max {d['max_db']:.1f}")
        if "gate" in st:
            parts.append(f"gate avg -{st['gate']['avg_db']:.1f}")
        if "buzz" in st:
            parts.append(f"buzz removed {st['buzz']['removed_db']:+.1f} dB of its 1.2-12 kHz")
        if "kit_gate" in st:
            parts.append(f"kit gate closed {st['kit_gate']['closed_pct']:.0f}% of the song")
        if "saturator" in st and st["saturator"]["harmonics_db"] is not None:
            parts.append(f"saturation {st['saturator']['harmonics_db']:+.0f} dB harmonics")
        if "transient" in st:
            parts.append(f"transient {st['transient']['max_boost_db']:+.1f}/{-st['transient']['max_cut_db']:+.1f}")
        print(f"  [{name}] " + " · ".join(parts or ["unchanged"]))
    rider = tracks.get("rider")
    if rider and "avg_up_db" in rider:
        print(f"  rider     vocals up {rider['avg_up_db']:.1f} dB on average where the band is dense "
              f"(max +{rider['max_up_db']:.1f}, -{rider['max_down_db']:.1f}; up >1 dB {rider['riding_up_pct']:.0f}% of the time)")
    for g, b in tracks.get("balance", {}).items():
        note = b.get("note") or f"{b['level_lu']:+.1f} -> {b['after_lu']:+.1f} LU (target {b['target_lu']:+.1f}), gain {b['gain_db']:+.1f} dB"
        print(f"  balance   {g:<8} {note}")
    if "eq_db" in rep:
        print("  EQ        " + "  ".join(f"{f}:{g:+.1f}" for f, g in rep["eq_db"].items()))
    for b in rep.get("multiband", []):
        bal = f"  balance {b['balance_correction_db']:+.1f}" if "balance_correction_db" in b else ""
        print(f"  {b['name']:<9} GR avg {b['avg_gr_db']:.1f} / max {b['max_gr_db']:.1f} dB{bal}")
    if "saturation" in rep:
        sat = rep["saturation"]
        print(f"  {'saturate':<9} {'CHOW Tape' if sat.get('engine') == 'chowtape' else 'built-in'}, "
              f"harmonics {sat['harmonics_db']:+.0f} dB re mix")
    if rep.get("consistency", {}).get("eq_db"):
        print(f"  {'tone':<9} towards the {rep['consistency']['songs']} songs' median: "
              + "  ".join(f"{f}:{g:+.1f}" for f, g in rep["consistency"]["eq_db"].items()))
    if "glue" in rep:
        print(f"  {'glue':<9} GR avg {rep['glue']['avg_gr_db']:.1f} / max {rep['glue']['max_gr_db']:.1f} dB"
              f" (threshold at the {rep['glue']['threshold_pct']:g}th percentile)")
    if "finalize" in rep:
        f = rep["finalize"]
        if f["clipper"]:
            c = f["clipper"]
            print(f"  {'clipper':<9} {c['clipped_pct']:.2f}% of samples, max {c['max_reduction_db']:.1f} dB")
        print(f"  {'limiter':<9} {f['limiter']['limited_pct']:.1f}% of time, max {f['limiter']['max_gr_db']:.1f} dB")
    for kind, p in rep.get("outputs", {}).items():
        print(f"  -> {kind}: {p}")


def _run_in_memory_scope() -> None:
    """Re-launch inside a memory-capped systemd scope, so that if processing ever
    runs out of memory only this program is stopped, not the desktop session.
    REHEARSAL_MEMORY_MAX sets the cap (default 4G); "off" disables it."""
    limit = os.environ.get("REHEARSAL_MEMORY_MAX", "4G")
    if os.environ.get("_REHEARSAL_IN_SCOPE") or limit == "off" or not sys.platform.startswith("linux"):
        return
    if not shutil.which("systemd-run") or not os.environ.get("XDG_RUNTIME_DIR"):
        return
    os.environ["_REHEARSAL_IN_SCOPE"] = "1"
    os.execvp("systemd-run", ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={limit}",
                              "-p", "MemorySwapMax=0", sys.executable, "-m", "rehearsal_master.cli", *sys.argv[1:]])


def _learn_balance(reports: Path, cfg: dict) -> None:
    """Median level of every balance group (re the unbalanced remix), and of every
    kit group (re the unbalanced drum bus), over all reports, leaving out songs
    more than 2.5 robust standard deviations away."""
    import json
    import math

    runs = []
    for f in sorted(reports.glob("*.json")):
        runs.append(json.loads(f.read_text()))

    def level(t: dict, stems: list[str], ref: float) -> float | None:
        p = sum(10 ** (t[m]["lufs_out"] / 10) for m in stems if t.get(m, {}).get("lufs_out") is not None)
        return 10 * math.log10(p) - ref if p > 0 else None

    mix, kit = {}, {}
    for r in runs:
        t = r.get("tracks", {})
        ref = t.get("remix", {}).get("unbalanced_lufs")
        if ref is not None:
            for g in cfg["balance"]["groups"]:
                if g["name"] == "drums" and "drum_bus" in t and t["drum_bus"].get("lufs_out") is not None:
                    v = t["drum_bus"]["lufs_out"] - ref
                else:
                    v = level(t, g["stems"], ref)
                if v is not None:
                    mix.setdefault(g["name"], []).append((r["title"], v))
        kref = t.get("drum_bus", {}).get("unbalanced_lufs")
        if kref is not None:
            for g in cfg["drum_bus"]["balance"]["groups"]:
                v = level(t, g["stems"], kref)
                if v is not None:
                    kit.setdefault(g["name"], []).append((r["title"], v))
    _print_targets("[balance]", cfg["balance"]["groups"], mix)
    _print_targets("[drum_bus.balance]", cfg["drum_bus"]["balance"]["groups"], kit)


def _print_targets(section: str, groups: list[dict], levels: dict) -> None:
    import json
    import statistics as st

    print(f"{section}          # learnt from", len({s for v in levels.values() for s, _ in v}), "songs")
    print("groups = [")
    for g in groups:
        vals = levels.get(g["name"], [])
        if not vals:
            continue
        x = [v for _, v in vals]
        med = st.median(x)
        mad = st.median([abs(v - med) for v in x]) * 1.4826 or 1.0
        kept = [(s, v) for s, v in vals if abs(v - med) <= 2.5 * mad]
        out = [s for s, v in vals if abs(v - med) > 2.5 * mad]
        target = st.median([v for _, v in kept])
        note = f"  # left out: {', '.join(out)}" if out else ""
        print(f'  {{ name = "{g["name"]}", stems = {json.dumps(g["stems"])}, lu = {target:.1f} }},{note}')
    print("]")


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        _run_in_memory_scope()
    ap = argparse.ArgumentParser(prog="rehearsal-master", description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, default=[Path(".")], help="WAV files or folders (default: current folder)")
    ap.add_argument("-c", "--config", type=Path, help="settings file (default: ./rehearsal.toml if present)")
    ap.add_argument("--target", type=float, metavar="LUFS", help="override loudness.target_lufs")
    ap.add_argument("--analyze", action="store_true", help="only measure and show the planned EQ, write nothing")
    ap.add_argument("--ab", action="store_true", help="also write <name>_AB.wav, alternating original/processed at equal loudness")
    ap.add_argument("--no-mp3", action="store_true", help="skip MP3 export")
    ap.add_argument("--preset", metavar="NAME", help=f"mixing preset on top of the defaults ({', '.join(config.presets())})")
    ap.add_argument("--learn-balance", action="store_true",
                    help="print balance targets learnt from the songs' reports (median, outliers removed), then exit")
    args = ap.parse_args(argv)

    cfg_path = args.config or (Path("rehearsal.toml") if Path("rehearsal.toml").exists() else None)
    cfg = config.load(cfg_path, args.preset)
    if args.target is not None:
        cfg["loudness"]["target_lufs"] = args.target
    if args.no_mp3:
        cfg["output"]["mp3"] = False

    if args.learn_balance:
        _learn_balance(Path(".") / cfg["output"]["dir"] / "reports", cfg)
        return
    files = _collect(args.paths, cfg["output"]["dir"])
    if not files:
        sys.exit("no audio files found")
    print(f"settings: {cfg_path or 'built-in defaults'}" + (f" + preset {args.preset}" if args.preset else ""))
    for f in files:
        print(f"\n{f.name}")
        t0 = time.time()
        rep = pipeline.process_file(f, config.for_file(cfg, f.name), analyze_only=args.analyze, ab=args.ab)
        _print(rep)
        print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
