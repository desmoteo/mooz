"""Command line: mooz [files or folders] [options]"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import config, pipeline, progress, takes


def _row(label: str, s: dict) -> str:
    return (f"  {label:<10} {s['integrated_lufs']:7.1f} LUFS  LRA {s['lra_lu']:4.1f}  "
            f"peak {s['true_peak_dbtp']:+5.1f} dBTP  PLR {s['plr_db']:4.1f}  L-R {s['balance_db']:+4.1f} dB")


def _print(rep: dict) -> None:
    t = rep["trim"]
    print(f"  kept {t['start_s']:.1f}s .. {t['end_s']:.1f}s of {rep['duration_s']:.1f}s")
    mt = rep.get("multitrack")
    if mt:
        md = mt.get("recorder_mixdown")
        print(f"  room      {mt['room_track']}" + (f" (the recorder's mix-down is {' + '.join(f'{v:.3f} {k}' for k, v in md['gains'].items())}, "
                                                   f"residual {md['residual_db']:.0f} dB)" if md else ""))
        for tr, stems in mt.get("close", {}).items():
            for s, i in stems.items():
                if i["used"]:
                    m = mt["merged"][s]
                    print(f"  {tr:<9} {s}: {i['level_lu']:+.1f} LU in the track, lined up {i['delay_ms']:+.1f} ms, polarity {i['polarity']:+d} "
                          f"(steady {i['steady_pct']}%), {100 * m['close_share']:.0f}% of the merged stem")
                else:
                    print(f"  {tr:<9} {s}: {i['level_lu'] if i['level_lu'] is not None else '-'} LU in the track: bleed, left out")
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
    kept = rep.get("chains")
    if kept and kept["reused"]:
        print(f"  chains    {len(kept['reused'])} kept from an earlier run"
              + (f", made again: {', '.join(kept['made'])}" if kept["made"] else ", none made again"))
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
        if name in ("remix", "balance", "buzz", "rider", "residual", "blend"):
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
        if "bed" in st:
            b = st["bed"]
            parts.append(f"bed: {b['note']}" if "note" in b else
                         f"bed between hits {b['between_hits_db']:+.1f} dB, at the hits {b['hat_hits_db']:+.1f} (2-8 kHz)")
        if "kit_gate" in st:
            parts.append(f"kit gate closed {st['kit_gate']['closed_pct']:.0f}% of the song")
        if "saturator" in st and st["saturator"]["harmonics_db"] is not None:
            parts.append(f"saturation {st['saturator']['harmonics_db']:+.0f} dB harmonics")
        if "transient" in st:
            parts.append(f"transient {st['transient']['max_boost_db']:+.1f}/{-st['transient']['max_cut_db']:+.1f}")
        print(f"  [{name}] " + " · ".join(parts or ["unchanged"]))
    blend = tracks.get("blend")
    if blend and blend.get("gain_db") is not None:
        print(f"  original  {100 * blend['original_share']:.0f}% of the loudness: {blend['level_lu']:+.1f} LU under the "
              f"stems (gain {blend['gain_db']:+.1f} dB)")
    res = tracks.get("residual")
    if res:
        print(f"  residual  what the separation missed: {res['level_db']:+.1f} dB re the mix (in the original's share)")
    rider = tracks.get("rider")
    if rider and "avg_up_db" in rider:
        print(f"  rider     vocals up {rider['avg_up_db']:.1f} dB on average where the band is dense "
              f"(max +{rider['max_up_db']:.1f}, -{rider['max_down_db']:.1f}; up >1 dB {rider['riding_up_pct']:.0f}% of the time)")
    for g, b in tracks.get("balance", {}).items():
        note = b.get("note") or (f"{b['level_lu']:+.1f} -> {b['after_lu']:+.1f} LU (target {b['target_lu']:+.1f}), "
                                 f"{b['gain_db']:+.1f} dB wanted" + (f", {b['stem_gain_db']:+.1f} dB on the stems"
                                                                      if "stem_gain_db" in b else ""))
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


MEMORY_FLOOR_GB = 4  # the least a run may use, whatever the machine has free


def memory_limit() -> str:
    """How much memory a run may use, as systemd writes it ("15G"), or "off". MOOZ_MEMORY_MAX
    sets it (REHEARSAL_MEMORY_MAX: the older name). By default it is what the machine can spare
    when the run starts: four fifths of the memory free now, at most half of all there is, at
    least 4G. A take needs about 0.5 GB per minute on top of 1.3 GB (4.1 GB for one of 5:40), and
    a fixed 4G stopped such takes half-way."""
    named = os.environ.get("MOOZ_MEMORY_MAX") or os.environ.get("REHEARSAL_MEMORY_MAX")
    if named:
        return named
    try:
        with open("/proc/meminfo") as f:
            kb = {line.split(":")[0]: int(line.split()[1]) for line in f if line.startswith(("MemTotal:", "MemAvailable:"))}
        gb = min(0.5 * kb["MemTotal"], 0.8 * kb["MemAvailable"]) / 2 ** 20
    except (OSError, KeyError, ValueError, IndexError):
        gb = MEMORY_FLOOR_GB
    return f"{max(MEMORY_FLOOR_GB, int(gb))}G"


def _with_huge_pages(env) -> dict:
    """env with glibc's allocator told to ask for huge pages (GLIBC_TUNABLES, glibc 2.35 or later;
    ignored elsewhere), unless the variable is set already. A run allocates and frees whole stems
    all the time (229 MB each for five minutes); in 4 KiB pages that was 7.4 million page faults
    a take and a fifth of its processor time, in 2 MB pages a quarter of that (docs/SPEED.md).
    The allocator reads the variable when the process starts: hence a start of its own."""
    return dict(env) | ({} if "GLIBC_TUNABLES" in env else {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=1"}) | {"_MOOZ_TUNED": "1"}


def _run_with_huge_pages() -> None:
    """Where no memory scope starts the run again (no systemd, or the limit off): start it again
    here, once, for the same variable."""
    if os.environ.get("_MOOZ_TUNED") or "GLIBC_TUNABLES" in os.environ or not sys.platform.startswith("linux"):
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, "-m", "mooz.cli", *sys.argv[1:]], _with_huge_pages(os.environ))


def _run_in_memory_scope() -> None:
    """Run again inside a memory-capped systemd scope and wait for it, so that if processing ever
    runs out of memory only this program is stopped, not the desktop session. The system stops
    such a run with a kill, which leaves it no word to say: so the one that waits says it, to the
    terminal and to a front end (progress event "stopped")."""
    limit = memory_limit()
    if os.environ.get("_REHEARSAL_IN_SCOPE") or limit == "off" or not sys.platform.startswith("linux"):
        return
    if not shutil.which("systemd-run") or not os.environ.get("XDG_RUNTIME_DIR"):
        return
    unit = f"mooz-{os.getpid()}-{int(time.time())}"
    child = subprocess.Popen(["systemd-run", "--user", "--scope", "--quiet", "--unit", unit, "-p", f"MemoryMax={limit}",
                              "-p", "MemorySwapMax=0", sys.executable, "-m", "mooz.cli", *sys.argv[1:]],
                             env=_with_huge_pages(os.environ | {"_REHEARSAL_IN_SCOPE": "1"}))
    # Ctrl-C and the web app's Cancel reach the run itself (the same process group); a SIGTERM sent to this one
    # alone is passed on. The run takes the first and ignores the rest, so its clean-up is not cut short
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda signum, frame: child.poll() is None and child.send_signal(signal.SIGTERM))
    code = child.wait()
    if code in (-signal.SIGKILL, 128 + signal.SIGKILL):
        why = ""
        try:
            why = subprocess.run(["systemctl", "--user", "show", "-p", "Result", "--value", f"{unit}.scope"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        if why == "oom-kill":
            progress.emit("stopped", why="memory", limit=limit)
            print(f"\nmooz: the system stopped the run: it needed more memory than the {limit} it may use. Takes already "
                  "finished are kept: run it again and only the others are made. MOOZ_MEMORY_MAX sets the limit "
                  "(for example MOOZ_MEMORY_MAX=24G, or off).", file=sys.stderr, flush=True)
        else:
            progress.emit("stopped", why="killed", limit=limit)
            print("\nmooz: the run was killed from outside. Takes already finished are kept: run it again and only "
                  "the others are made.", file=sys.stderr, flush=True)
    subprocess.run(["systemctl", "--user", "reset-failed", f"{unit}.scope"], capture_output=True)  # a stopped scope stays listed
    sys.exit(code if code >= 0 else 128 - code)


def _code() -> str:
    """One name for the program as it is: every module's source."""
    h = hashlib.blake2b(digest_size=8)
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def made_with(take: takes.Take, cfg: dict) -> str:
    """One name for what a take's master is made with, as far as it is known before it is made:
    the program, the take's settings (the song's own sections in, its style not yet), and the
    recorder's files (name, size, time). Written into the take's report; a take whose report
    has this name and whose files are there is up to date and is not made again. Not in it: what
    the other takes bring (the tone targets are medians over all of them), which --again is for."""
    files = [take.room, *take.close.values()] + ([take.mixdown] if take.mixdown else [])
    settings = {k: v for k, v in cfg.items() if k not in ("_layers", "_cli", "files", "input", "chains")}
    parts = {"code": _code(), "settings": settings,
             "files": [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]}
    return hashlib.blake2b(json.dumps(parts, sort_keys=True, default=str).encode(), digest_size=12).hexdigest()


def up_to_date(take: takes.Take, cfg: dict, stamp: str, ab: bool) -> dict | None:
    """The take's report, if its master was made with stamp and every file it names is there
    (and the A/B file, if one is asked for)."""
    out = take.root / cfg["output"]["dir"]
    try:
        rep = json.loads((out / "reports" / f"{take.name}.json").read_text())
    except (OSError, ValueError):
        return None
    outputs = rep.get("outputs") or {}
    if rep.get("made_with") != stamp or "wav" not in outputs or (ab and "ab" not in outputs):
        return None
    for kind, path in outputs.items():
        name = Path(path).name
        if not any((d / name).exists() for d in (out, out / "reports", out / "stages")):
            return None
    return rep


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
                measured = t.get("balance", {}).get(g["name"], {}).get("level_lu")  # as heard, in the blend
                if measured is not None:
                    v = measured
                elif g["name"] == "drums" and "drum_bus" in t and t["drum_bus"].get("lufs_out") is not None:
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


def _summary(rep: dict) -> dict:
    """What a front end shows of a finished take."""
    out = {"outputs": rep.get("outputs", {})}
    for side in ("original", "processed"):
        if side in rep:
            out[side] = {k: rep[side][k] for k in ("integrated_lufs", "lra_lu", "true_peak_dbtp")}
    if "finalize" in rep:
        out["limited_pct"] = rep["finalize"]["limiter"]["limited_pct"]
    return out


def _stop(signum, frame) -> None:
    """SIGTERM (the web app's Cancel) leaves as an exit, so work folders are removed on the way out.
    Only the first one: a second (the same Cancel, passed on by the process that waits for this
    one) must not cut that clean-up short."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.exit(143)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        _run_in_memory_scope()
        _run_with_huge_pages()
        signal.signal(signal.SIGTERM, _stop)
    ap = argparse.ArgumentParser(prog="mooz", description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path,
                    help="recordings: files or folders (default: [input] paths in the settings, else the current folder)")
    ap.add_argument("-c", "--config", type=Path,
                    help="settings file (default: rehearsal.toml in the recordings' folder, else in the current one)")
    ap.add_argument("--target", type=float, metavar="LUFS", help="override loudness.target_lufs")
    ap.add_argument("--analyze", action="store_true", help="only measure and show the planned EQ, write nothing")
    ap.add_argument("--ab", action="store_true", help="also write <name>_AB.wav, alternating original/processed at equal loudness")
    ap.add_argument("--no-mp3", action="store_true", help="skip MP3 export")
    ap.add_argument("--again", action="store_true",
                    help="master again also the takes that are up to date (made with these very settings, recordings and "
                         "program, all their files there); they are left as they are otherwise")
    ap.add_argument("--separate-again", action="store_true",
                    help="separate every track of the chosen takes again instead of using its kept stems (the slow part: "
                         "minutes per track), then master them, up to date or not; with --separate-only, only separate")
    ap.add_argument("--no-chain-cache", action="store_true",
                    help="run every stem's chain again and keep none for the next run ([chains] enabled = false)")
    ap.add_argument("--preset", metavar="NAME", action="append",
                    help=f"mixing preset on top of the defaults ({', '.join(config.presets())}, or one of the project's own in "
                         "presets/ next to its settings file); several: every take once per preset; 'default': no preset")
    ap.add_argument("--separate-only", action="store_true", help="only separate every track into the cache, then exit")
    ap.add_argument("--learn-balance", action="store_true",
                    help="print balance targets learnt from the songs' reports (median, outliers removed), then exit")
    args = ap.parse_args(argv)

    first = args.paths[0] if args.paths else Path(".")
    folder = first if first.is_dir() else first.parent
    cfg_path = args.config or next((d / "rehearsal.toml" for d in (folder, Path(".")) if (d / "rehearsal.toml").exists()), None)
    overrides: dict = {}
    if args.target is not None:
        overrides["loudness"] = {"target_lufs": args.target}
    if args.no_mp3:
        overrides["output"] = {"mp3": False}
    if args.no_chain_cache:
        overrides["chains"] = {"enabled": False}
    if args.separate_again:  # new stems under a master that stays would be a master of other stems: made again too
        pipeline.separate.AGAIN = True
        args.again = True
    base = config.load(cfg_path, None, overrides)
    paths = args.paths
    if not paths:
        here = cfg_path.parent if cfg_path else Path(".")
        paths = [here / p for p in base["input"]["paths"]] or [Path(".")]
    presets = (["" if p == "default" else p for p in args.preset] if args.preset
               else (base["input"]["presets"] or [""]))

    if args.learn_balance:
        where = paths[0] if paths[0].is_dir() else paths[0].parent  # the recordings' folder
        _learn_balance(where / config.load(cfg_path, presets[0] or None, overrides)["output"]["dir"] / "reports", base)
        return
    # never re-process our own output, whichever folder it is in
    skip = {Path(base["output"]["dir"]).parts[0], Path(base["output"]["stems_dir"]).parts[0],
            Path(base["chains"]["dir"]).parts[0], "experiments", "prove", ".venv"}
    try:
        found = takes.discover(paths, skip, base["multitrack"]["room"])
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    if not found:
        sys.exit("no audio files found")
    failed = []
    unit = 0  # the take and preset being worked on, as numbered in the plan (progress.py)
    if args.separate_only:
        progress.emit("plan", units=[{"take": t.name, "preset": ""} for t in found], separate_only=True)
        for take in found:
            print(f"\n{take.name}: room {take.room.name}" + "".join(f", {k} {p.name}" for k, p in take.close.items()), flush=True)
            progress.emit("unit", i=unit)
            t0 = time.time()
            err = None
            try:
                pipeline.separate_take(take, config.for_file(base, take.name))
            except Exception as e:
                traceback.print_exc()
                failed.append(take.name)
                err = f"{type(e).__name__}: {e}"
                print(f"  FAILED: {err}", file=sys.stderr)
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
            progress.emit("unit_done", i=unit, ok=err is None, seconds=round(time.time() - t0, 1), error=err)
            unit += 1
    else:
        progress.emit("plan", units=[{"take": t.name, "preset": p} for p in presets for t in found], separate_only=False)
        for preset in presets:
            cfg = config.load(cfg_path, preset or None, overrides)
            print(f"settings: {cfg_path or 'built-in defaults'}" + (f" + preset {preset}" if preset else ""))
            # what is up to date is left as it is: a run that was stopped goes on where it stopped
            stamps = {t.name: made_with(t, config.for_file(cfg, t.name)) for t in found} if not args.analyze else {}
            kept = {} if args.again or args.analyze else {
                t.name: rep for t in found if (rep := up_to_date(t, config.for_file(cfg, t.name), stamps[t.name], args.ab))}
            if not args.analyze:  # all the takes' stems first: each song's tone targets are then the same, whatever the order
                ahead = [t for t in found if t.close and t.name not in kept]
                for k, take in enumerate(ahead):
                    progress.emit("prepare", take=take.name, i=k, n=len(ahead))
                    try:
                        pipeline.separate_take(take, config.for_file(cfg, take.name))
                    except Exception as e:  # reported again below, when this take's turn comes
                        print(f"  {take.name}: {type(e).__name__}: {e}", file=sys.stderr)
                pipeline.separate.release()  # every take's stems are there: the separating process is not needed any more
            for take in found:
                print(f"\n{take.name}" + (f" [{preset}]" if preset else ""))
                progress.emit("unit", i=unit)
                unit += 1
                if take.name in kept:
                    print("  up to date: mastered with these settings, recordings and program (--again makes it again)", flush=True)
                    progress.emit("unit_done", i=unit - 1, ok=True, seconds=0.0, summary=_summary(kept[take.name]), kept=True)
                    continue
                t0 = time.time()
                try:
                    rep = pipeline.process_file(take, config.for_file(cfg, take.name), analyze_only=args.analyze, ab=args.ab)
                except Exception as e:  # one song's failure does not stop the others
                    traceback.print_exc()
                    print(f"  FAILED after {time.time() - t0:.0f}s: {type(e).__name__}: {e}", file=sys.stderr)
                    failed.append(f"{take.name}" + (f" [{preset}]" if preset else ""))
                    progress.emit("unit_done", i=unit - 1, ok=False, seconds=round(time.time() - t0, 1),
                                  error=f"{type(e).__name__}: {e}")
                    continue
                _print(rep)
                report = take.root / cfg["output"]["dir"] / "reports" / f"{take.name}.json"
                if not args.analyze and report.exists():  # the report is the last file written: now it says what made it
                    rep["made_with"] = stamps[take.name]
                    report.write_text(json.dumps(rep, indent=2, default=float))
                peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
                print(f"  ({time.time() - t0:.0f}s; the run has used {peak_gb:.1f} GB of memory at most)", flush=True)
                progress.emit("unit_done", i=unit - 1, ok=True, seconds=round(time.time() - t0, 1), summary=_summary(rep),
                              peak_gb=round(peak_gb, 1))
    progress.emit("end", failed=failed)
    if failed:
        sys.exit(f"\n{len(failed)} failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
