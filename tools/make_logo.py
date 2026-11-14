"""Build the Mooz (Master Out Of Zoom) logo files in docs/.

The mark: two woven rings (the "oo", the recorder's stereo pair) with a waveform running
through them, uneven bars on the left (the raw room recording), a smooth even sine on the
right (the mastered track). Text is converted to outlines (URW Gothic, from the urw-base35
fonts) so the files render the same everywhere, including GitHub, which shows SVGs as images
and cannot load fonts.

  mooz-mark.svg         icon on its tile, 48 px and up
  mooz-mark-small.svg   simplified icon for 16-32 px (favicon)
  mooz.svg              square tile with the name and tagline
  mooz-banner.svg       README header, transparent, follows the viewer's light/dark scheme
  mooz-banner-light.svg / mooz-banner-dark.svg   the same, fixed, for GitHub's <picture>
  mooz-avatar.png       full-bleed 512 px square for avatar uploads (GitHub rounds it itself)
  mooz-social.png       1280 x 640 social preview (GitHub: Settings > General > Social preview)

  uv run --with fonttools --with cairosvg python tools/make_logo.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

DOCS = Path(__file__).resolve().parents[1] / "docs"
FONTS = Path("/usr/share/fonts/opentype/urw-base35")
DEMI = TTFont(FONTS / "URWGothic-Demi.otf")
BOOK = TTFont(FONTS / "URWGothic-Book.otf")

TAGLINE = ["MASTER", "OUT", "OF", "ZOOM"]

# mark geometry, in a 256 box centred on (128, 128)
R, RING_W = 58, 14
CA, CB = (98, 128), (158, 128)
GAP = 5  # clearance cut around whatever passes over a ring
MARK_TOP, MARK_BOTTOM = 128 - R - RING_W / 2, 128 + R + RING_W / 2
MARK_LEFT, MARK_RIGHT = CA[0] - R - RING_W / 2, CB[0] + R + RING_W / 2

THEMES = {
    "dark": {
        "word": "#ffffff", "tag": "#c9c0e6", "raw": "#a99fd3", "clean": "#ffffff",
        "ring0": "#ffa53c", "ring1": "#ff4d8d",
    },
    "light": {
        "word": "#1d1533", "tag": "#5d5180", "raw": "#9186ba", "clean": "#1d1533",
        "ring0": "#f58a1f", "ring1": "#e8336f",
    },
}


def num(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


# ---- text ---------------------------------------------------------------------------------

def advance(font: TTFont, s: str, size: float, track: float) -> float:
    cmap, hmtx = font.getBestCmap(), font["hmtx"]
    k = size / font["head"].unitsPerEm
    return sum(hmtx[cmap[ord(c)]][0] * k for c in s) + track * size * (len(s) - 1)


def text_path(font: TTFont, s: str, x: float, y: float, size: float, track: float = 0.0) -> str:
    """Outline of `s`, baseline at y, starting at x; track is extra spacing in em."""
    gs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    k = size / font["head"].unitsPerEm
    pen = SVGPathPen(gs, ntos=num)
    for c in s:
        name = cmap[ord(c)]
        gs[name].draw(TransformPen(pen, (k, 0, 0, -k, x, y)))
        x += hmtx[name][0] * k + track * size
    return pen.getCommands()


def cap_height(font: TTFont, size: float) -> float:
    return font["OS/2"].sCapHeight * size / font["head"].unitsPerEm


def wordmark(x: float, baseline: float, size: float) -> tuple[str, float]:
    return text_path(DEMI, "Mooz", x, baseline, size), advance(DEMI, "Mooz", size, 0)


def tagline(x: float, baseline: float, width: float, track: float = 0.14) -> tuple[str, str, float]:
    """MASTER OUT OF ZOOM set to exactly `width`; returns (initials, rest, size).
    The initials M-O-O-Z get the ring colours."""
    full = " ".join(TAGLINE)
    size = width / (advance(BOOK, full, 1, 0) + track * (len(full) - 1))
    lead, rest = [], []
    for i, word in enumerate(TAGLINE):
        lead.append(text_path(BOOK, word[0], x, baseline, size, track))
        tail = word[1:] + (" " if i < len(TAGLINE) - 1 else "")
        x += advance(BOOK, word[0], size, track) + track * size
        rest.append(text_path(BOOK, tail, x, baseline, size, track))
        x += advance(BOOK, tail, size, track) + track * size
    return " ".join(lead), " ".join(rest), size


# ---- mark ---------------------------------------------------------------------------------

def raw_bars() -> list[tuple[float, float]]:
    """Uneven bars, (x, half height), kept inside the left ring so they never cut it."""
    room = R - RING_W / 2 - GAP - 3  # 3 = half the bar width
    shape = [0.55, 0.95, 0.5, 1.0, 0.35]
    out = []
    for i, f in enumerate(shape):
        x = 60 + 8 * i
        out.append((x, f * math.sqrt(room**2 - (x - CA[0]) ** 2)))
    return out


def sine_d(x0: float, x1: float, amp: float, cycles: float) -> str:
    pts = []
    for i in range(121):
        x = x0 + (x1 - x0) * i / 120
        pts.append((x, 128 - amp * math.sin((x - x0) / (x1 - x0) * cycles * 2 * math.pi)))
    return "M" + " L".join(f"{num(x)} {num(y)}" for x, y in pts)


def bars_d(bars: list[tuple[float, float]]) -> str:
    return " ".join(f"M{num(x)} {num(128 - h)}V{num(128 + h)}" for x, h in bars)


def ring_mask(pid: str, other: tuple[int, int], under_at: float, wave: list[tuple[str, float]],
              ring_w: float) -> str:
    """Mask for one ring: cut where the other ring passes over it (near y = under_at) and
    where the waveform crosses it, so the weave works on any background."""
    cuts = [
        f'<circle cx="{other[0]}" cy="{other[1]}" r="{R}" stroke-width="{num(ring_w + 2 * GAP)}" '
        f'clip-path="url(#{pid}-near{"T" if under_at < 128 else "B"})"/>'
    ] if ring_w else []
    cuts += [f'<path d="{d}" stroke-width="{num(w + 2 * GAP)}"/>' for d, w in wave]
    return (f'<mask id="{pid}-m{"A" if other == CB else "B"}" maskUnits="userSpaceOnUse" '
            f'x="0" y="0" width="256" height="256">'
            f'<rect width="256" height="256" fill="#fff"/>'
            f'<g fill="none" stroke="#000" stroke-linecap="round" stroke-linejoin="round">'
            f'{"".join(cuts)}</g></mask>')


def mark(pid: str, small: bool = False) -> tuple[str, str]:
    """(defs, body) of the mark in the 256 box. Colours come from CSS classes."""
    iy = math.sqrt(R**2 - ((CB[0] - CA[0]) / 2) ** 2)
    top, bottom = 128 - iy, 128 + iy
    if small:
        ring_w = 22
        wave = [(sine_d(40, 216, 30, 1.5), 18.0)]
        weave = False
        body_wave = (f'<path class="clean" d="{wave[0][0]}" fill="none" stroke-width="18" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        ring_w = RING_W
        bars = raw_bars()
        wave = [(bars_d(bars), 6.0), (sine_d(100, 218, 30, 1.5), 8.0)]
        weave = True
        body_wave = (f'<g fill="none" stroke-linecap="round" stroke-linejoin="round">'
                     f'<path class="raw" d="{wave[0][0]}" stroke-width="6"/>'
                     f'<path class="clean" d="{wave[1][0]}" stroke-width="8"/></g>')
    near = "".join(
        f'<clipPath id="{pid}-near{tag}"><circle cx="128" cy="{num(y)}" r="{num(ring_w + 8)}"/></clipPath>'
        for tag, y in (("T", top), ("B", bottom))
    )
    defs = (
        f'<linearGradient id="{pid}-ring" x1="{CA[0] - R}" y1="0" x2="{CB[0] + R}" y2="0" '
        f'gradientUnits="userSpaceOnUse"><stop offset="0" class="ring0"/>'
        f'<stop offset="1" class="ring1"/></linearGradient>'
        + near
        # A (left) passes over B at the top crossing, B over A at the bottom
        + ring_mask(pid, CB, bottom, wave, ring_w if weave else 0)
        + ring_mask(pid, CA, top, wave, ring_w if weave else 0)
    )
    body = (
        f'<g fill="none" stroke="url(#{pid}-ring)" stroke-width="{ring_w}">'
        f'<circle cx="{CA[0]}" cy="{CA[1]}" r="{R}" mask="url(#{pid}-mA)"/>'
        f'<circle cx="{CB[0]}" cy="{CB[1]}" r="{R}" mask="url(#{pid}-mB)"/></g>'
        + body_wave
    )
    return defs, body


# ---- documents ----------------------------------------------------------------------------

def css(theme: str | None) -> str:
    def rules(t: dict[str, str]) -> str:
        return (f".word{{fill:{t['word']}}}.tag{{fill:{t['tag']}}}"
                f".raw{{stroke:{t['raw']}}}.clean{{stroke:{t['clean']}}}"
                f".ring0{{stop-color:{t['ring0']}}}.ring1{{stop-color:{t['ring1']}}}"
                f".init{{fill:url(#accent)}}")
    if theme:
        return rules(THEMES[theme])
    return (rules(THEMES["light"])
            + "@media (prefers-color-scheme: dark){" + rules(THEMES["dark"]) + "}")


ACCENT = ('<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">'
          '<stop offset="0" class="ring0"/><stop offset="1" class="ring1"/></linearGradient>')

TILE = ('<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#34255c"/><stop offset="1" stop-color="#150f27"/>'
        '</linearGradient>')


def tile(rounded: bool) -> str:
    if not rounded:
        return '<rect width="256" height="256" fill="url(#bg)"/>'
    # faint inner edge so the tile still has a border on GitHub's near-black dark theme
    return ('<rect width="256" height="256" rx="56" fill="url(#bg)"/>'
            '<rect x="1" y="1" width="254" height="254" rx="55" fill="none" '
            'stroke="#ffffff" stroke-opacity="0.09" stroke-width="2"/>')


def doc(w: float, h: float, label: str, style: str, defs: str, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {num(w)} {num(h)}" '
            f'role="img" aria-label="{label}">\n<title>{label}</title>\n'
            f'<style>{style}</style>\n<defs>{defs}</defs>\n{body}\n</svg>\n')


def icon(small: bool, rounded: bool = True) -> str:
    defs, body = mark("m", small)
    if small:
        # bigger in the tile: the small icon is all about presence
        body = f'<g transform="translate(128 128) scale(1.12) translate(-128 -128)">{body}</g>'
    return doc(256, 256, "Mooz", css("dark"), TILE + defs, tile(rounded) + "\n" + body)


def square() -> str:
    defs, body = mark("m")
    s = 0.68
    word_size = 58
    word_w = advance(DEMI, "Mooz", word_size, 0)
    mark_h = (MARK_BOTTOM - MARK_TOP) * s
    cap = cap_height(DEMI, word_size)
    gap1, gap2 = 16, 12
    # tagline size depends on the wordmark width; lay out once to know its cap height
    _, _, tag_size = tagline(0, 0, word_w)
    tag_cap = cap_height(BOOK, tag_size)
    total = mark_h + gap1 + cap + gap2 + tag_cap
    y0 = (256 - total) / 2
    mark_cy = y0 + mark_h / 2
    base = y0 + mark_h + gap1 + cap
    x = 128 - word_w / 2
    word, _ = wordmark(x, base, word_size)
    lead, rest, _ = tagline(x, base + gap2 + tag_cap, word_w)
    out = (tile(True)
           + f'\n<g transform="translate(128 {num(mark_cy)}) scale({s}) translate(-128 -128)">{body}</g>'
           + f'\n<path class="word" d="{word}"/>'
           + f'\n<path class="init" d="{lead}"/><path class="tag" d="{rest}"/>')
    return doc(256, 256, "Mooz: Master Out Of Zoom", css("dark"), TILE + ACCENT + defs, out)


def banner(theme: str | None) -> str:
    defs, body = mark("m")
    s = 0.9
    word_size = 112
    word_w = advance(DEMI, "Mooz", word_size, 0)
    cap = cap_height(DEMI, word_size)
    _, _, tag_size = tagline(0, 0, word_w)
    tag_cap = cap_height(BOOK, tag_size)
    gap_text = 0.3 * cap
    text_h = cap + gap_text + tag_cap
    mark_w = (MARK_RIGHT - MARK_LEFT) * s
    mark_h = (MARK_BOTTOM - MARK_TOP) * s
    pad, gap = 12, 34
    h = max(mark_h, text_h) + 2 * pad
    w = pad + mark_w + gap + word_w + pad
    cy = h / 2
    mark_x = pad + (128 - MARK_LEFT) * s
    tx = pad + mark_w + gap
    base = cy - text_h / 2 + cap
    word, _ = wordmark(tx, base, word_size)
    lead, rest, _ = tagline(tx, base + gap_text + tag_cap, word_w)
    out = (f'<g transform="translate({num(mark_x)} {num(cy)}) scale({s}) translate(-128 -128)">{body}</g>'
           f'\n<path class="word" d="{word}"/>'
           f'\n<path class="init" d="{lead}"/><path class="tag" d="{rest}"/>')
    return doc(w, h, "Mooz: Master Out Of Zoom", css(theme), ACCENT + defs, out)


SOCIAL_LINE = "Rehearsal recordings in, mixed and mastered tracks out"


def social(w: int = 1280, h: int = 640) -> str:
    """The dark banner on the tile's background at link-preview size, with one line under it.
    Sites crop the edges of these images, so everything sits well inside the frame."""
    defs, body = mark("m")
    s = 1.45
    word_size = 176
    word_w = advance(DEMI, "Mooz", word_size, 0)
    cap = cap_height(DEMI, word_size)
    _, _, tag_size = tagline(0, 0, word_w)
    tag_cap = cap_height(BOOK, tag_size)
    gap_text = 0.3 * cap
    text_h = cap + gap_text + tag_cap
    mark_w = (MARK_RIGHT - MARK_LEFT) * s
    mark_h = (MARK_BOTTOM - MARK_TOP) * s
    gap = 52
    lock_w = mark_w + gap + word_w
    lock_h = max(mark_h, text_h)
    line_size, line_gap = 34, 64
    line_cap = cap_height(BOOK, line_size)
    line_w = advance(BOOK, SOCIAL_LINE, line_size, 0.02)
    top = (h - (lock_h + line_gap + line_cap)) / 2
    cy = top + lock_h / 2
    x0 = (w - lock_w) / 2
    mark_x = x0 + (128 - MARK_LEFT) * s
    tx = x0 + mark_w + gap
    base = cy - text_h / 2 + cap
    word, _ = wordmark(tx, base, word_size)
    lead, rest, _ = tagline(tx, base + gap_text + tag_cap, word_w)
    line = text_path(BOOK, SOCIAL_LINE, (w - line_w) / 2, top + lock_h + line_gap + line_cap,
                     line_size, 0.02)
    out = (f'<rect width="{w}" height="{h}" fill="url(#bg)"/>'
           f'\n<g transform="translate({num(mark_x)} {num(cy)}) scale({s}) translate(-128 -128)">{body}</g>'
           f'\n<path class="word" d="{word}"/>'
           f'\n<path class="init" d="{lead}"/><path class="tag" d="{rest}"/>'
           f'\n<path class="tag" d="{line}"/>')
    return doc(w, h, "Mooz: Master Out Of Zoom", css("dark"), TILE + ACCENT + defs, out)


def main() -> None:
    files = {
        "mooz-mark.svg": icon(small=False),
        "mooz-mark-small.svg": icon(small=True),
        "mooz.svg": square(),
        "mooz-banner.svg": banner(None),
        "mooz-banner-light.svg": banner("light"),
        "mooz-banner-dark.svg": banner("dark"),
    }
    for name, svg in files.items():
        (DOCS / name).write_text(svg)
        print(f"{name}: {len(svg.encode()) / 1024:.1f} KB")
    import cairosvg
    cairosvg.svg2png(bytestring=icon(small=False, rounded=False).encode(),
                     write_to=str(DOCS / "mooz-avatar.png"), output_width=512, output_height=512)
    print("mooz-avatar.png")
    cairosvg.svg2png(bytestring=social().encode(), write_to=str(DOCS / "mooz-social.png"),
                     output_width=1280, output_height=640)
    print("mooz-social.png")


if __name__ == "__main__":
    main()
