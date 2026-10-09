"""Pictograms of the main-text figures, one definition for TikZ (Fig. 1) and matplotlib (Figs. 2-5).

Every glyph is a list of (path, filled) pairs in a 10 x 10 box with the y axis pointing up. A path is SVG path data
restricted to absolute M, L, C and Z, so the same string drives TikZ's svg.path library and the matplotlib parser
below; circles and arcs are written as cubic Beziers. The glyphs are schematic everyday objects drawn for this paper
(shield, sword, card, hat, die, chip, book, lens, pin, hourglass), with no resemblance to any product mark.

Sizes: 2.5 mm inline next to 7 pt text, 3 mm as a plot marker; stroke 0.45 pt in ink.

matplotlib: glyph_box() returns a DrawingArea, add_glyph() places one at data, axes or figure coordinates, and
GlyphHandle with legend_handler_map() puts a glyph into a legend, glyph_marker() gives a path marker.
TikZ: tikz_pic_styles() returns the pic definitions for the options of a tikzpicture; write_template() refreshes them
between the glyph markers of fig1_overview.tex.in (run `python code/glyphs.py` after changing a glyph).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

INK = "#333333"          # stroke colour, black!80
STROKE_PT = 0.45
MM = 72.0 / 25.4         # points per millimetre
K = 0.5523               # cubic Bezier constant for a quarter circle


def _f(x):
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def circle(cx, cy, r):
    k = K * r
    p = [(cx + r, cy), (cx + r, cy + k), (cx + k, cy + r), (cx, cy + r), (cx - k, cy + r), (cx - r, cy + k),
         (cx - r, cy), (cx - r, cy - k), (cx - k, cy - r), (cx, cy - r), (cx + k, cy - r), (cx + r, cy - k), (cx + r, cy)]
    s = f"M {_f(p[0][0])} {_f(p[0][1])}"
    for i in range(1, 13, 3):
        s += " C " + " ".join(f"{_f(x)} {_f(y)}" for x, y in p[i:i + 3])
    return s + " Z"


def rrect(x0, y0, x1, y1, r):
    k = K * r
    return (f"M {_f(x0 + r)} {_f(y0)} L {_f(x1 - r)} {_f(y0)} C {_f(x1 - r + k)} {_f(y0)} {_f(x1)} {_f(y0 + r - k)} {_f(x1)} {_f(y0 + r)}"
            f" L {_f(x1)} {_f(y1 - r)} C {_f(x1)} {_f(y1 - r + k)} {_f(x1 - r + k)} {_f(y1)} {_f(x1 - r)} {_f(y1)}"
            f" L {_f(x0 + r)} {_f(y1)} C {_f(x0 + r - k)} {_f(y1)} {_f(x0)} {_f(y1 - r + k)} {_f(x0)} {_f(y1 - r)}"
            f" L {_f(x0)} {_f(y0 + r)} C {_f(x0)} {_f(y0 + r - k)} {_f(x0 + r - k)} {_f(y0)} {_f(x0 + r)} {_f(y0)} Z")


def poly(*pts, close=False):
    s = f"M {_f(pts[0][0])} {_f(pts[0][1])}" + "".join(f" L {_f(x)} {_f(y)}" for x, y in pts[1:])
    return s + (" Z" if close else "")


def _rays(cx, cy, r0, r1, n, start=90):
    out = []
    for i in range(n):
        a = math.radians(start + 360 * i / n)
        out.append(poly((cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r1 * math.cos(a), cy + r1 * math.sin(a))))
    return out


def _head_shoulders():
    return [(circle(3.9, 7.0, 1.75), False),
            ("M 0.6 1.0 C 0.6 3.7 2.0 4.7 3.9 4.7 C 5.8 4.7 7.2 3.7 7.2 1.0", False)]


def _chip(inner):
    body = [(poly((2.6, 2.6), (7.4, 2.6), (7.4, 7.4), (2.6, 7.4), close=True), False)]
    pins = [(poly((0.7, y), (2.6, y)), False) for y in (3.7, 5.0, 6.3)] + \
           [(poly((7.4, y), (9.3, y)), False) for y in (3.7, 5.0, 6.3)]
    return body + pins + inner


GLYPHS: dict[str, list[tuple[str, bool]]] = {
    # games
    # shield with a sword held point up: the blade narrows to a tip above the rim, the guard sits low
    "battle": [("M 1.6 8.4 L 8.4 8.4 L 8.4 5.2 C 8.4 2.9 6.5 1.4 5.0 0.6 C 3.5 1.4 1.6 2.9 1.6 5.2 Z", False),
               (poly((4.35, 3.0), (4.35, 8.0), (5.0, 9.8), (5.65, 8.0), (5.65, 3.0)), False),
               (poly((3.1, 3.0), (6.9, 3.0)), False), (poly((5.0, 3.0), (5.0, 1.3)), False)],
    "combined": [(poly((1.7, 1.7), (8.6, 8.6)), False), (poly((1.6, 4.2), (4.2, 1.6)), False),
                 (poly((8.4, 1.6), (1.7, 8.3)), False), (poly((1.5, 5.7), (1.5, 8.5), (4.3, 8.5)), False)],
    "rps": [(circle(1.5, 5.0, 1.35), True), (poly((3.7, 3.7), (6.3, 3.7), (6.3, 6.3), (3.7, 6.3), close=True), False),
            (poly((7.5, 6.4), (8.5, 3.6), (9.5, 6.4)), False)],
    "hanabi": [(rrect(2.0, 0.5, 8.0, 9.5, 0.9), False), (circle(5.0, 5.0, 0.75), True)] +
              [(r, False) for r in _rays(5.0, 5.0, 1.55, 2.75, 5)],
    "overcooked": [(poly((2.6, 0.9), (7.4, 0.9), (7.4, 2.9), (2.6, 2.9), close=True), False),
                   ("M 2.6 2.9 C 0.9 3.3 0.5 6.3 2.6 6.7 C 2.7 8.9 4.0 9.5 5.0 9.5 C 6.0 9.5 7.3 8.9 7.4 6.7 "
                    "C 9.5 6.3 9.1 3.3 7.4 2.9", False)],
    "smac": [(circle(2.0, 7.7, 1.15), False), (circle(2.0, 2.3, 1.15), False), (circle(4.4, 5.0, 1.15), False),
             (poly((6.0, 5.0), (9.6, 5.0)), False), (poly((8.1, 6.4), (9.6, 5.0), (8.1, 3.6)), False)],
    "die": [(rrect(1.0, 1.0, 9.0, 9.0, 1.6), False), (circle(3.2, 6.8, 0.95), True), (circle(5.0, 5.0, 0.95), True),
            (circle(6.8, 3.2, 0.95), True)],
    # information
    "bubble": [("M 2.6 3.0 L 2.1 0.8 L 4.7 3.0 L 7.5 3.0 C 8.33 3.0 9.0 3.67 9.0 4.5 L 9.0 7.6 C 9.0 8.43 8.33 9.1 7.5 9.1 "
                "L 2.5 9.1 C 1.67 9.1 1.0 8.43 1.0 7.6 L 1.0 4.5 C 1.0 3.67 1.67 3.0 2.5 3.0 Z", False),
               (poly((2.8, 7.3), (7.2, 7.3)), False), (poly((2.8, 5.0), (6.0, 5.0)), False)],
    "eye": [("M 0.5 5.0 C 2.6 8.5 7.4 8.5 9.5 5.0 C 7.4 1.5 2.6 1.5 0.5 5.0 Z", False), (circle(5.0, 5.0, 1.3), True)],
    "known": _head_shoulders() + [(poly((6.6, 6.5), (7.7, 5.2), (9.7, 8.0)), False)],
    "new": _head_shoulders() + [(poly((8.3, 5.1), (8.3, 8.9)), False), (poly((6.4, 7.0), (10.0, 7.0)), False)],
    # selectors
    "chip": _chip([]),
    "chip20b": _chip([(circle(5.0, 5.0, 0.8), False)]),
    "chip120b": _chip([(circle(5.0, 5.0, 1.55), False)]),
    "chipqwen": _chip([(poly((5.0, 6.6), (6.6, 5.0), (5.0, 3.4), (3.4, 5.0), close=True), False)]),
    "lever": [(poly((0.8, 1.0), (6.4, 1.0), (6.4, 7.6), (0.8, 7.6), close=True), False),
              (poly((2.0, 3.4), (5.2, 3.4), (5.2, 5.6), (2.0, 5.6), close=True), False),
              (poly((6.4, 3.0), (8.6, 3.0), (8.6, 7.6)), False), (circle(8.6, 8.5, 0.95), True)],
    "books": [(poly((0.8, 1.0), (2.7, 1.0), (2.7, 8.3), (0.8, 8.3), close=True), False),
              (poly((3.3, 1.0), (5.2, 1.0), (5.2, 9.0), (3.3, 9.0), close=True), False),
              (poly((5.8, 1.0), (7.6, 1.0), (9.5, 7.5), (7.8, 8.0), close=True), False)],
    "lens": [(poly((0.6, 8.6), (7.6, 8.6)), False), (poly((0.6, 6.6), (3.4, 6.6)), False), (circle(5.8, 5.4, 2.3), False),
             (poly((7.45, 3.75), (9.6, 1.4)), False)],
    "rule": [(poly((4.0, 8.8), (7.4, 5.0), (4.0, 1.2), (0.6, 5.0), close=True), False), (poly((7.4, 5.0), (9.8, 5.0)), False),
             (poly((8.6, 6.1), (9.8, 5.0), (8.6, 3.9)), False)],
    "net": [(circle(2.0, 5.0, 1.25), False), (circle(8.0, 8.2, 1.25), False), (circle(8.0, 1.8, 1.25), False),
            (poly((3.1, 5.6), (6.9, 7.6)), False), (poly((3.1, 4.4), (6.9, 2.4)), False)],
    "pin": [(circle(5.0, 7.0, 2.3), False), (poly((3.2, 4.4), (6.8, 4.4)), False), (poly((5.0, 4.4), (5.0, 0.5)), False)],
    "hourglass": [(poly((2.4, 9.3), (7.6, 9.3)), False), (poly((2.4, 0.7), (7.6, 0.7)), False),
                  (poly((3.0, 9.3), (7.0, 9.3), (5.0, 5.0), close=True), False),
                  (poly((5.0, 5.0), (7.0, 0.7), (3.0, 0.7), close=True), True)],
}


# ------------------------------------------------------------------------------------------------------------------
# parser: SVG path data (absolute M, L, C, Z) -> vertices and codes
# ------------------------------------------------------------------------------------------------------------------
_TOK = re.compile(r"[MLCZ]|-?\d*\.?\d+")


def parse(d):
    """[(command, [numbers])] of an absolute M/L/C/Z path."""
    out, cmd, nums = [], None, []
    for t in _TOK.findall(d):
        if t in "MLCZ":
            if cmd:
                out.append((cmd, nums))
            cmd, nums = t, []
        else:
            nums.append(float(t))
    if cmd:
        out.append((cmd, nums))
    return out


def to_mpl_path(d, scale=1.0, dx=0.0, dy=0.0):
    from matplotlib.path import Path as MPath
    verts, codes = [], []
    start = None
    for cmd, n in parse(d):
        if cmd == "M":
            start = (n[0] * scale + dx, n[1] * scale + dy)
            verts.append(start); codes.append(MPath.MOVETO)
            for i in range(2, len(n), 2):
                verts.append((n[i] * scale + dx, n[i + 1] * scale + dy)); codes.append(MPath.LINETO)
        elif cmd == "L":
            for i in range(0, len(n), 2):
                verts.append((n[i] * scale + dx, n[i + 1] * scale + dy)); codes.append(MPath.LINETO)
        elif cmd == "C":
            for i in range(0, len(n), 6):
                for j in range(3):
                    verts.append((n[i + 2 * j] * scale + dx, n[i + 2 * j + 1] * scale + dy)); codes.append(MPath.CURVE4)
        elif cmd == "Z":
            verts.append(start); codes.append(MPath.CLOSEPOLY)
    return MPath(verts, codes)


def glyph_marker(name):
    """One matplotlib Path of a glyph, centred on the origin, for marker= (drawn unfilled: mfc="none")."""
    from matplotlib.path import Path as MPath
    return MPath.make_compound_path(*[to_mpl_path(d, 0.1, -0.5, -0.5) for d, _ in GLYPHS[name]])


def glyph_artists(name, size_pt, color=INK, lw=STROKE_PT, x0=0.0, y0=0.0):
    """PathPatches of a glyph whose 10-unit box spans size_pt points, lower left corner at (x0, y0) points."""
    from matplotlib.patches import PathPatch
    s = size_pt / 10.0
    arts = []
    for d, filled in GLYPHS[name]:
        p = to_mpl_path(d, s, x0, y0)
        arts.append(PathPatch(p, facecolor=color if filled else "none", edgecolor=color, lw=lw,
                              capstyle="round", joinstyle="round"))
    return arts


def glyph_box(names, size_mm=2.5, color=INK, gap_mm=0.5, lw=STROKE_PT):
    """A DrawingArea holding one glyph or a row of glyphs (names: str or list), in points."""
    from matplotlib.offsetbox import DrawingArea
    names = [names] if isinstance(names, str) else list(names)
    sz, gap = size_mm * MM, gap_mm * MM
    da = DrawingArea(len(names) * sz + (len(names) - 1) * gap, sz, 0, 0)
    for i, n in enumerate(names):
        for a in glyph_artists(n, sz, color, lw, x0=i * (sz + gap)):
            da.add_artist(a)
    return da


def add_glyph(target, names, xy, xycoords="data", size_mm=2.5, align=(0.5, 0.5), color=INK, zorder=6, gap_mm=0.5,
              clip=False):
    """Place a glyph (or a row of glyphs) with its box aligned at xy; target is an Axes or a Figure."""
    from matplotlib.offsetbox import AnnotationBbox
    if xycoords == "figure fraction" and hasattr(target, "transFigure"):
        # the transform object, not the string: savefig(bbox_inches="tight") moves transFigure, the string misses it
        xycoords = target.transFigure
    ab = AnnotationBbox(glyph_box(names, size_mm, color, gap_mm), xy, xycoords=xycoords, frameon=False,
                        box_alignment=align, pad=0, annotation_clip=clip, zorder=zorder)
    if hasattr(target, "add_artist"):
        target.add_artist(ab)
    return ab


class GlyphHandle:
    """Legend proxy for a glyph or a row of glyphs; pass handler_map=legend_handler_map() to legend()."""

    def __init__(self, names, size_mm=2.5):
        self.names = [names] if isinstance(names, str) else list(names)
        self.size_mm = size_mm


def _handler():
    from matplotlib.legend_handler import HandlerBase

    class GlyphHandler(HandlerBase):
        def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
            sz = orig.size_mm * MM
            arts = []
            x = -xdescent
            yc = -ydescent + height / 2.0
            for n in orig.names:
                for a in glyph_artists(n, sz, x0=x, y0=yc - sz / 2.0):
                    a.set_transform(trans)
                    arts.append(a)
                x += sz + 0.5 * MM
            return arts

    return GlyphHandler()


def legend_handler_map():
    return {GlyphHandle: _handler()}


# ------------------------------------------------------------------------------------------------------------------
# TikZ
# ------------------------------------------------------------------------------------------------------------------
def tikz_pic_styles(names=None, prefix="gl"):
    """Pic definitions `gl-<name>` for the options of a tikzpicture, for the given glyph names (default: all). Each pic
    is centred on its position and its box is \\glsize wide (key `gl size`, default 2.5 mm); stroke 0.45 pt in
    black!80. svg.path reads the path data in points, so the scope scales by size/10pt."""
    lines = [f"  {prefix} size/.store in=\\glsize, {prefix} size=2.5mm,",
             f"  {prefix} ink/.style={{line width=0.45pt, draw=black!80, line cap=round, line join=round}},",
             f"  {prefix} dot/.style={{{prefix} ink, fill=black!80}},"]
    for name, parts in GLYPHS.items():
        if names is not None and name not in names:
            continue
        body = []
        for d, filled in parts:
            body.append(f"\\draw[{prefix} {'dot' if filled else 'ink'}] svg {{{d}}};")
        lines.append(f"  pics/{prefix}-{name}/.style={{code={{\\begin{{scope}}[shift={{(-0.5*\\glsize,-0.5*\\glsize)}}, "
                     f"scale=\\glsize/10pt]" + " ".join(body) + "\\end{scope}}},")
    return "\n".join(lines)


TEMPLATE = Path(__file__).resolve().parent / "fig1_overview.tex.in"
BEGIN, END = "% glyph pics: begin", "% glyph pics: end"


def write_template(path=TEMPLATE):
    """Refresh the pic definitions between the markers, for the glyphs the template uses (keeps main.tex short)."""
    s = path.read_text(encoding="utf-8")
    i, j = s.index(BEGIN), s.index(END)
    used = set(re.findall(r"\{gl-([a-z0-9]+)\}", s[:i] + s[j:]))
    assert used <= set(GLYPHS), used - set(GLYPHS)
    s = s[:i + len(BEGIN)] + "\n" + tikz_pic_styles(used) + "\n" + s[j:]
    path.write_text(s, encoding="utf-8", newline="\n")
    print("glyph pics written into", path.name)


if __name__ == "__main__":
    write_template()
