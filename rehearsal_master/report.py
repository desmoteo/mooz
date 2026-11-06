"""Per-song report figure: spectrum, EQ, loudness over time, gain reduction."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ORIGINAL, PROCESSED = "#eb6834", "#2a78d6"
FREQ_TICKS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
HOP_S = 0.1  # time resolution of loudness and gain-reduction curves

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 9,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left"})


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=7.5, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _freq_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(30, 20000)
    ax.xaxis.set_major_locator(FixedLocator(FREQ_TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda f, _: f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"))
    ax.set_xlabel("Hz", color=MUTED, fontsize=7.5, loc="right")


def _normalise(grid: np.ndarray, spec: np.ndarray) -> np.ndarray:
    band = (grid >= 100) & (grid <= 10000)
    return spec - spec[band].mean()


def _label(ax, x, y, text, color, dy) -> None:
    ax.annotate(text, (x, y), xytext=(0, dy), textcoords="offset points", ha="center",
                va="bottom" if dy > 0 else "top", fontsize=7.5, color=INK2,
                bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": color, "lw": 0.8})


def _fmt(v: float, unit: str) -> str:
    v = round(v, 1) + 0.0
    return f"{v:+.1f}".replace("-", "−") + unit if unit in (" dBTP", " dB") else f"{v:.1f}".replace("-", "−") + unit


def plot(png: Path, rep: dict, grid, spec_before, spec_after, target, eq_curve, curves: dict,
         short_term_before: np.ndarray, short_term_after: np.ndarray, cfg: dict) -> None:
    rows = [(k, v) for k, v in curves.get("multiband", {}).items()]
    if "glue" in curves:
        rows.append(("glue", curves["glue"]))
    rows.append(("limiter", curves["limiter"]))

    fig = plt.figure(figsize=(11, 4.6 + 0.62 * len(rows) + 2.4), dpi=130, facecolor=SURFACE)
    gs = fig.add_gridspec(2 + len(rows), 2, height_ratios=[3.3, 2.5] + [0.62] * len(rows),
                          width_ratios=[1.6, 1], hspace=0.55, wspace=0.18,
                          left=0.065, right=0.985, top=0.9, bottom=0.04)

    o, p = rep["original"], rep["processed"]
    style = rep.get("character", {}).get("style")
    fig.text(0.065, 0.965, f"{rep['title']}" + (f"  ·  {rep['date']}" if rep["date"] else "")
             + (f"  ·  {style}" if style else ""),
             fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.065, 0.935,
             f"loudness {_fmt(o['integrated_lufs'], ' LUFS')} → {_fmt(p['integrated_lufs'], ' LUFS')}    "
             f"true peak {_fmt(o['true_peak_dbtp'], ' dBTP')} → {_fmt(p['true_peak_dbtp'], ' dBTP')}    "
             f"loudness range {_fmt(o['lra_lu'], ' LU')} → {_fmt(p['lra_lu'], ' LU')}    "
             f"left−right {_fmt(o['balance_db'], ' dB')} → {_fmt(p['balance_db'], ' dB')}",
             fontsize=8.5, color=INK2, va="top")

    # 1. Spectrum
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    _freq_axis(ax)
    keep = (grid >= 30) & (grid <= 20000)
    b, a = _normalise(grid, spec_before), _normalise(grid, spec_after)
    ax.plot(grid[keep], b[keep], color=ORIGINAL, lw=1.4, label="original")
    ax.plot(grid[keep], a[keep], color=PROCESSED, lw=1.4, label="processed")
    if target is not None:
        t = _normalise(grid, target)
        ax.plot(grid[keep], t[keep], color=INK2, lw=1.0, ls=(0, (4, 3)), label="target balance")
    ax.set_title("Tonal balance (1/3-octave band level, aligned 100 Hz–10 kHz)")
    ax.set_ylabel("dB", color=MUTED, fontsize=7.5, loc="top")
    lo = np.percentile(np.concatenate([b[keep], a[keep]]), 1)
    ax.set_ylim(max(lo - 3, -40), max(b[keep].max(), a[keep].max()) + 6)
    i250 = np.argmin(np.abs(grid - 250))
    if b[i250] >= a[i250]:
        _label(ax, 250, b[i250], "original", ORIGINAL, 6)
        _label(ax, 250, a[i250], "processed", PROCESSED, -6)
    else:
        _label(ax, 250, a[i250], "processed", PROCESSED, 6)
        _label(ax, 250, b[i250], "original", ORIGINAL, -6)
    ax.legend(loc="lower left", frameon=False, fontsize=7.5, labelcolor=INK2)

    # 2. EQ curve
    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    _freq_axis(ax)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(grid[keep], eq_curve[keep], color=PROCESSED, lw=1.4)
    lim = max(3.0, np.abs(eq_curve[keep]).max() + 1)
    ax.set_ylim(-lim, lim)
    ax.set_title("EQ applied")
    ax.set_ylabel("dB", color=MUTED, fontsize=7.5, loc="top")

    # 3. Short-term loudness over time
    start = rep["trim"]["start_s"]
    dur = rep["trim"]["end_s"] - start
    ax_t = fig.add_subplot(gs[1, :])
    _style(ax_t)
    tb = np.arange(len(short_term_before)) * HOP_S + 1.5 - start
    ta = np.arange(len(short_term_after)) * HOP_S + 1.5
    ax_t.plot(tb, short_term_before, color=ORIGINAL, lw=1.2, label="original")
    ax_t.plot(ta, short_term_after, color=PROCESSED, lw=1.2, label="processed")
    tgt = cfg["loudness"]["target_lufs"]
    ax_t.axhline(tgt, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax_t.annotate(f"target {_fmt(tgt, ' LUFS')}", (0, tgt), xytext=(4, 3), textcoords="offset points",
                  ha="left", va="bottom", fontsize=7.5, color=INK2,
                  bbox={"boxstyle": "round,pad=0.1", "fc": SURFACE, "ec": "none"})
    vis = np.concatenate([short_term_before[(tb >= 0) & (tb <= dur)], short_term_after])
    vis = vis[np.isfinite(vis) & (vis > -70)]
    ax_t.set_ylim(max(vis.min() - 2, tgt - 30), vis.max() + 3)
    ax_t.set_xlim(0, dur)
    ax_t.set_title("Short-term loudness (3 s window)")
    ax_t.set_ylabel("LUFS", color=MUTED, fontsize=7.5, loc="top")
    ax_t.legend(loc="lower center", frameon=False, ncol=2, fontsize=7.5, labelcolor=INK2)

    # 4. Gain reduction, one row per stage, shared scale
    stats = {b["name"]: b for b in rep.get("multiband", [])}
    if "glue" in rep:
        stats["glue"] = rep["glue"]
    stats["limiter"] = rep["finalize"]["limiter"] | {"avg_gr_db": float(curves["limiter"].mean())}
    gr_max = max(6.0, max(float(c.max()) for _, c in rows))
    for i, (name, c) in enumerate(rows):
        ax = fig.add_subplot(gs[2 + i, :], sharex=ax_t)
        _style(ax)
        ax.grid(False)
        ax.spines["bottom"].set_visible(i == len(rows) - 1)
        t = np.arange(len(c)) * HOP_S
        ax.fill_between(t, 0, -c, color=PROCESSED, lw=0, alpha=0.85)
        ax.set_ylim(-gr_max, 0.3)
        ax.set_yticks([0, -round(gr_max)])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
        s = stats.get(name, {})
        ax.text(1.0, 0.04, f"{name}   avg {s.get('avg_gr_db', 0):.1f} · max {s.get('max_gr_db', 0):.1f} dB",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
        if i == 0:
            ax.set_title("Gain reduction (dB) by stage")
        if i < len(rows) - 1:
            ax.tick_params(labelbottom=False, bottom=False)
        else:
            ax.set_xlabel("seconds", color=MUTED, fontsize=7.5, loc="right")
    ax_t.tick_params(labelbottom=True)

    fig.savefig(png, facecolor=SURFACE)
    plt.close(fig)


# --- tracks report: per-stem EQ, dynamics and remix levels -----------------------------------

# Effect identity colours, in the palette's fixed categorical order.
COMP, DYN, RES, TRANS, DUCK, KICK, GATE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"
BEFORE, AFTER = MUTED, INK


def _num(v: float, fmt: str = "+.1f") -> str:
    return format(round(v, 1) + 0.0, fmt).replace("-", "−")


def _hz(f: float) -> str:
    return f"{f / 1000:g} kHz" if f >= 1000 else f"{f:g} Hz"


def _describe(t: dict) -> str:
    """One-line summary of a track chain's settings."""
    parts = []
    if t["gain_db"]:
        parts.append(f"gain {_num(t['gain_db'])} dB")
    if t["width"] != 1.0:
        parts.append("mono" if t["width"] == 0 else f"width {t['width']:g}")
    if t["balance"] > 0:
        parts.append("re-centred")
    if t["augment"]["enabled"]:
        parts.append(f"own-hit samples at {t['augment']['mix']:g}")
    if t["bass_fundamental"]["enabled"]:
        parts.append(f"fundamental +{t['bass_fundamental']['boost_db']:g} dB")
    if t["tone_match"]["enabled"]:
        ref = t["tone_match"]["reference"]
        parts.append(f"tone match {t['tone_match']['strength']:.0%} to its usual tone"
                     + (f" ({ref:.0%} raw studio tracks)" if ref else ""))
    for b in t["eq"]:
        kind = {"high_pass": "HP", "low_pass": "LP", "low_shelf": "low shelf", "high_shelf": "high shelf"}.get(b["type"])
        parts.append(f"{kind} {_hz(b['freq_hz'])}" if kind in ("HP", "LP") else
                     f"{(kind + ' ') if kind else ''}{_num(b['gain_db'])} dB @ {_hz(b['freq_hz'])}")
    for b in t["dynamic_eq"]:
        parts.append(f"dyn-EQ {_hz(b['freq_hz'])} (max +{b['max_boost_db']:g} when thin)" if b.get("mode") == "boost"
                     else f"dyn-EQ {_hz(b['freq_hz'])} (max −{b['max_cut_db']:g})")
    if t["resonance"]["enabled"]:
        r = t["resonance"]
        parts.append(f"resonances {_hz(r['fmin_hz'])}–{_hz(r['fmax_hz'])}")
    if t["compressor"]["enabled"]:
        parts.append(f"comp {t['compressor']['ratio']:g}:1")
    if t["kick_eq"]["enabled"]:
        parts.append(f"kick EQ {_num(t['kick_eq']['gain_db'])} dB at the kick's pitch")
    if t["kick_shaper"]["enabled"]:
        k = t["kick_shaper"]
        parts.append(f"kick shaper +{k['punch_db']:g} dB punch / \u2212{k['tail_db']:g} dB boom")
    for d in t["ducks"]:
        band = f"<{_hz(d['fmax_hz'])}" if d["fmin_hz"] <= 0 else f"{_hz(d['fmin_hz'])}\u2013{_hz(d['fmax_hz'])}"
        parts.append(f"ducks {band} by {d['depth_db']:g} dB on {d['source']}")
    if t["gate"]["enabled"]:
        parts.append(f"gate \u2212{t['gate']['depth_db']:g} dB between hits")
    if t["kit_gate"]["enabled"]:
        parts.append(f"\u2212{t['kit_gate']['depth_db']:g} dB while the drums are silent")
    if t["saturator"]["enabled"]:
        parts.append(f"saturator {t['saturator']['drive_db']:g} dB drive, {t['saturator']['mix']:.0%} mix")
    tr = t["transient"]
    if tr["enabled"]:
        if tr["crossovers_hz"]:
            edges = [_hz(f) for f in tr["crossovers_hz"]]
            names = [f"<{edges[0]}"] + [f"{a}\u2013{b}" for a, b in zip(edges, edges[1:])] + [f">{edges[-1]}"]
            parts.append("transient " + ", ".join(f"{n} att {_num(a)}/sus {_num(su)}"
                                                  for n, a, su in zip(names, tr["attack"], tr["sustain"])))
        else:
            parts.append(f"transient att {_num(tr['attack'])}/sus {_num(tr['sustain'])}")
    if t["exciter"]["enabled"]:
        parts.append(f"exciter {_hz(t['exciter']['freq_hz'])}")
    if t["bass_enhancer"]["enabled"]:
        parts.append(f"bass enhancer <{_hz(t['bass_enhancer']['cutoff_hz'])}")
    return " · ".join(parts) or "unchanged"


def _wrap(text: str, width: int = 70) -> str:
    lines, cur = [], ""
    for part in text.split(" · "):
        if cur and len(cur) + len(part) + 3 > width:
            lines.append(cur)
            cur = part
        else:
            cur = f"{cur} · {part}" if cur else part
    return "\n".join(lines + [cur])


def plot_tracks(png: Path, rep: dict, details: dict[str, dict], cfg: dict) -> None:
    names = list(details)
    tstats = rep.get("tracks", {})
    remix = tstats.get("remix")
    tcfg = cfg["tracks"]
    blocks = [_wrap(_describe(tcfg.get(n, cfg["track_defaults"])), 78) for n in names]
    line_in = 0.15  # height of one 7.5 pt text line, inches
    header_in = max(2.0, line_in * sum(b.count("\n") + 1 for b in blocks) + 0.12 * len(names) + 0.3)
    row_in, gap_in, top_in, bottom_in = 2.2, 0.75, 1.2, 0.35
    fig_h = top_in + bottom_in + header_in + 1.0 + len(names) * (row_in + gap_in)
    fig = plt.figure(figsize=(13, fig_h), dpi=120, facecolor=SURFACE)
    gs = fig.add_gridspec(1 + len(names), 3, height_ratios=[header_in + 1.0 - gap_in] + [row_in] * len(names),
                          width_ratios=[1.25, 1, 2.1], hspace=gap_in / row_in, wspace=0.2, left=0.06, right=0.985,
                          top=1 - top_in / fig_h, bottom=bottom_in / fig_h)

    from .separate import describe

    sep = f"separated with {describe(cfg['stems'])}" if remix else "no separation: one chain on the whole mix"
    res = tstats.get("residual")
    fig.text(0.06, 1 - 0.25 / fig.get_figheight(),
             f"{rep['title']}  ·  per-track processing" + (f"  ·  {rep['date']}" if rep["date"] else ""),
             fontsize=13, fontweight="bold", color=INK, va="top")
    sub = sep + (f"\nresidual (what separation missed) {_num(res['level_db'])} dB re mix, added back unprocessed"
                 if res else "")
    fig.text(0.06, 1 - 0.55 / fig.get_figheight(), sub, fontsize=8.5, color=INK2, va="top", linespacing=1.5)

    # Remix levels: each stem's loudness relative to the whole, before and after its chain
    head = gs[0, :2].subgridspec(2, 1, height_ratios=[header_in, 1.0])
    ax = fig.add_subplot(head[0])
    _style(ax)
    if remix:
        src = {**{n: tstats[n] for n in names}, **({"residual": res} if res else {})}
        rows = [n for n in src if src[n].get("lufs_in") is not None and src[n].get("lufs_out") is not None]
        y = np.arange(len(rows))[::-1]
        before = np.array([src[n]["lufs_in"] - remix["mix_lufs"] for n in rows])
        bg = lambda n: src[n].get("balance_gain_db", 0.0) if isinstance(src[n].get("balance_gain_db", 0.0), float) else 0.0  # noqa: E731
        after = np.array([src[n]["lufs_out"] + bg(n) - remix["remix_lufs"] for n in rows])
        for yi, b, a in zip(y, before, after):
            ax.plot([b, a], [yi, yi], color=AXIS, lw=2, solid_capstyle="round", zorder=1)
        ax.scatter(before, y, s=46, color=BEFORE, zorder=2, label="original mix", edgecolor=SURFACE, lw=1.5)
        ax.scatter(after, y, s=46, color=AFTER, zorder=3, label="after track chains", edgecolor=SURFACE, lw=1.5)
        for yi, b, a in zip(y, before, after):
            ax.annotate(f"{_num(a)} LU", (max(a, b), yi), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y, rows)
        ax.tick_params(axis="y", labelsize=8, labelcolor=INK)
        lo, hi = min(before.min(), after.min()), max(before.max(), after.max())
        ax.set_xlim(lo - 2, hi + 5)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title("Remix: each track's loudness relative to the mix")
        ax.set_xlabel(f"LU relative to the mix   (mix {_num(remix['mix_lufs'], '.1f')} LUFS \u2192 remix "
                      f"{_num(remix['remix_lufs'], '.1f')} LUFS, before the bus chain)", color=MUTED, fontsize=7.5, loc="right")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=7.5,
                  labelcolor=INK2, borderaxespad=0.3)
    else:
        ax.axis("off")

    # Chain settings, one block per track
    ax = fig.add_subplot(gs[0, 2].subgridspec(2, 1, height_ratios=[header_in, 1.0])[0])
    ax.axis("off")
    ax.set_title("Chains", loc="left")
    ax_h = ax.get_position().height * fig_h  # inches
    yy = 1.0
    for n, body in zip(names, blocks):
        ax.text(0, yy, n, transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK, va="top")
        ax.text(0.16, yy, body, transform=ax.transAxes, fontsize=7.5, color=INK2, va="top", linespacing=1.35)
        yy -= (line_in * (body.count("\n") + 1) + 0.12) / ax_h

    # Effect legend, shared by all rows
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([], [], color=BEFORE, lw=1.4, label="before chain"),
               Line2D([], [], color=AFTER, lw=1.4, label="after chain"),
               Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 2)), label="static EQ"),
               Patch(color=DYN, alpha=0.55, label="dynamic EQ"),
               Patch(color=RES, alpha=0.55, label="resonance suppressor"),
               Patch(color=COMP, alpha=0.85, label="compressor"),
               Patch(color=TRANS, alpha=0.85, label="transient shaper"),
               Line2D([], [], color=DUCK, lw=1.4, label="ducking"),
               Line2D([], [], color=GATE, lw=1.4, label="hit gate"),
               Patch(color=KICK, alpha=0.85, label="kick shaper")]
    top_row = gs[1, 0].get_position(fig)
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, top_row.y1 + 0.42 / fig_h),
               ncol=len(handles), frameon=False, fontsize=7.5, labelcolor=INK2, handlelength=1.8, columnspacing=1.4)

    share = None
    for i, n in enumerate(names):
        d, st = details[n], tstats.get(n, {})
        grid = d["grid"]
        keep = (grid >= 30) & (grid <= 20000)

        # spectrum before/after
        ax = fig.add_subplot(gs[1 + i, 0])
        _style(ax)
        _freq_axis(ax)
        ax.plot(grid[keep], d["spec_in"][keep], color=BEFORE, lw=1.3)
        ax.plot(grid[keep], d["spec_out"][keep], color=AFTER, lw=1.3)
        top = max(d["spec_in"][keep].max(), d["spec_out"][keep].max())
        ax.set_ylim(top - 50, top + 4)
        lev = f"{_num(d['lufs_in'], '.1f')} → {_num(d['lufs_out'], '.1f')} LUFS"
        ax.set_title(f"{n}   {lev}")
        ax.set_ylabel("dB", color=MUTED, fontsize=7.5, loc="top")

        # EQ: static curve + range of the dynamic processors
        ax = fig.add_subplot(gs[1 + i, 1])
        _style(ax)
        _freq_axis(ax)
        ax.axhline(0, color=AXIS, lw=0.8)
        static = d["eq_db"]
        lows = [static[keep].min(), -3.0]
        for b in d["dyn_eq"]:
            ax.fill_between(grid[keep], static[keep], (static + b["max_curve"])[keep], color=DYN, alpha=0.3, lw=0)
            ax.plot(grid[keep], (static + b["avg_curve"])[keep], color=DYN, lw=1.2)
            lows.append((static + b["max_curve"])[keep].min())
        if "resonance" in d:
            r = d["resonance"]
            ax.fill_between(grid[keep], static[keep], (static + r["max_curve"])[keep], color=RES, alpha=0.3, lw=0)
            ax.plot(grid[keep], (static + r["avg_curve"])[keep], color=RES, lw=1.2)
            lows.append((static + r["max_curve"])[keep].min())
        ax.plot(grid[keep], static[keep], color=INK2, lw=1.4, ls=(0, (4, 2)))
        ax.set_ylim(min(lows) - 1, max(3.0, static[keep].max() + 1.5))
        ax.set_title("EQ")
        ax.set_title("line = average, area = deepest cut", loc="right", fontsize=7, fontweight="normal", color=INK2)
        ax.set_ylabel("dB", color=MUTED, fontsize=7.5, loc="top")

        # dynamics over time
        ax = fig.add_subplot(gs[1 + i, 2], sharex=share)
        share = share or ax
        _style(ax)
        ax.axhline(0, color=AXIS, lw=0.8)
        tr = d["traces"]
        t = lambda v: np.arange(len(v)) * HOP_S  # noqa: E731
        lo, hi = -3.0, 1.0
        if "transient" in tr:
            ax.fill_between(t(tr["transient"]["boost"]), tr["transient"]["cut"], tr["transient"]["boost"],
                            color=TRANS, alpha=0.55, lw=0)
            lo, hi = min(lo, tr["transient"]["cut"].min()), max(hi, tr["transient"]["boost"].max())
        if "kick_shaper" in tr:
            ax.fill_between(t(tr["kick_shaper"]["boost"]), tr["kick_shaper"]["cut"], tr["kick_shaper"]["boost"],
                            color=KICK, alpha=0.6, lw=0)
            lo, hi = min(lo, tr["kick_shaper"]["cut"].min()), max(hi, tr["kick_shaper"]["boost"].max())
        if "compressor" in tr:
            ax.fill_between(t(tr["compressor"]), 0, -tr["compressor"], color=COMP, alpha=0.85, lw=0)
            lo = min(lo, -tr["compressor"].max())
        for b in d["dyn_eq"]:
            ax.plot(t(b["trace"]), -b["trace"], color=DYN, lw=0.9)
            lo = min(lo, -b["trace"].max())
        if "resonance" in tr:
            ax.plot(t(tr["resonance"]), -tr["resonance"], color=RES, lw=0.9)
            lo = min(lo, -tr["resonance"].max())
        if "duck" in tr:
            ax.plot(t(tr["duck"]), -tr["duck"], color=DUCK, lw=0.9)
            lo = min(lo, -tr["duck"].max())
        if "gate" in tr:
            ax.plot(t(tr["gate"]), -tr["gate"], color=GATE, lw=0.9)
            lo = min(lo, -tr["gate"].max())
        ax.set_ylim(lo - 1, hi + 1)
        ax.set_xlim(0, rep["trim"]["end_s"] - rep["trim"]["start_s"])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
        bits = []
        if "compressor" in st:
            c = st["compressor"]
            bits.append(f"comp avg {c['avg_gr_db']:.1f} while playing / max {c['max_gr_db']:.1f}, "
                        f"make-up {_num(c['makeup_db'])} dB")
        for b in st.get("dynamic_eq", []):
            bits.append(f"dyn-EQ {_hz(b['freq_hz'])} avg {b['avg_cut_db']:.1f} / max {b['max_cut_db']:.1f}")
        if "resonance" in st:
            r = st["resonance"]
            bits.append(f"resonances avg {r['avg_cut_db']:.1f} / max {r['max_cut_db']:.1f}, make-up {_num(r['makeup_db'])} dB")
        if st.get("augment", {}).get("samples"):
            bits.append(f"samples on {st['augment']['placed']} hits")
        if "avg_gain_db" in st.get("bass_fundamental", {}):
            bits.append(f"fundamental +{st['bass_fundamental']['avg_gain_db']:.1f} dB")
        if "kick_hz" in st:
            bits.append(f"kick at {st['kick_hz']:.0f} Hz")
        if "expander" in st:
            bits.append(f"expander at −{st['expander']['below_db']:.0f} dB")
        if "kick_shaper" in st:
            bits.append(f"{st['kick_shaper']['hits']} kick hits shaped")
        for d in st.get("ducks", []):
            bits.append(f"ducking on {d['source']} max {d['max_db']:.1f} dB")
        if "gate" in st:
            bits.append(f"gate avg {st['gate']['avg_db']:.1f} dB")
        if "kit_gate" in st:
            bits.append(f"kit gate closed {st['kit_gate']['closed_pct']:.0f}%")
        if st.get("saturator", {}).get("harmonics_db") is not None:
            bits.append(f"saturation {_num(st['saturator']['harmonics_db'], '+.0f')} dB")
        if "transient" in st:
            bits.append(f"transient {_num(st['transient']['max_boost_db'])} / {_num(-st['transient']['max_cut_db'])}"
                        + (f" (attack ×{st['transient_scale']:.2f})" if "transient_scale" in st else ""))
        ax.set_title("Dynamics over time (dB)")
        half = (len(bits) + 1) // 2
        ax.set_title("  ·  ".join(bits[:half]) + ("\n" + "  ·  ".join(bits[half:]) if bits[half:] else "")
                     if bits else "no dynamics", loc="right", fontsize=7, fontweight="normal", color=INK2)
        if i == len(names) - 1:
            ax.set_xlabel("seconds", color=MUTED, fontsize=7.5, loc="right")

    fig.savefig(png, facecolor=SURFACE)
    plt.close(fig)
