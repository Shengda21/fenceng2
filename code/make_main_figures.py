"""Main-text figures, drawn from analysis/master.json and analysis/extras.json, and for the briefing family from
analysis/briefing (read at run time, never copied).

fig_headroom_map   : H_D against the type count M, with the briefing family's full-state headroom and prefix ceiling
                     (column width)
fig_legibility     : capture by encoding (zero-shot) and by number of examples, and the reward gain of a briefing over
                     the first moves for gpt-oss-120b (column width)
fig_curves         : capture curves of the learners against two in-context captures, and the sandbox crossover horizon
                     against the predicted scale (column width)

Colour rule, shared with make_fig_briefing.py: the three language-model families own the three colour-blind-safe hues
of COL in every figure; every non-LLM selector is neutral (grey or black) and told apart by line style and marker
(STY); game families carry no colour, only a glyph (glyphs.py) and a marker shape. Text is set in ink. Every text span
is at least 7 pt at print size; the badges use BADGE_FS.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glyphs as G  # noqa: E402

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
args = ap.parse_args()
ROOT = Path(args.root).resolve()
D = json.load(open(ROOT / "analysis" / "master.json", encoding="utf-8"))
M = D["master"]
EX = json.load(open(ROOT / "analysis" / "extras.json", encoding="utf-8"))
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": "#222222",
    "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
})
# language-model families: Okabe-Ito orange, blue and bluish green (colour-blind-safe); nothing else is coloured
COL = {"gpt-oss-20b": "#E69F00", "gpt-oss-120b": "#0072B2", "qwen3.8-27b": "#009E73"}
INK = "#222222"
# series neutrals: black, dark and light grey only (a mid grey would sit at the lightness of the green and collapse
# with it under deuteranopia); REF is for unlabelled reference lines, never for a series
DARK, LIGHT, REF = "#404040", "#b8b8b8", "#999999"
# non-LLM selectors: neutral, distinguished by line style and marker
STY = {
    "linucb": dict(color=DARK, ls="-", lw=1.0),
    "lints": dict(color=DARK, ls=(0, (4.0, 1.6)), lw=1.0),
    "ctxucb": dict(color=DARK, ls=(0, (5.0, 1.4, 1.2, 1.4)), lw=1.0),
    "mucb": dict(color=DARK, ls=(0, (1.0, 1.3)), lw=1.15),
    "plastic": dict(color="#000000", ls="-", lw=2.4),
    "fewshot__5": dict(color=LIGHT, ls=(3.0, (3.0, 3.0)), lw=1.1),
    "ppo": dict(color=LIGHT, ls="-", lw=1.0),
    "text_linucb": dict(color=LIGHT, ls=(0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)), lw=1.25),
}
MARK = {"linucb": "s", "lints": "o"}
BADGE_FS = 7.2       # the C1-C3 badges; every other text span is 7 pt or more
BRIEF = ROOT / "analysis" / "briefing"


@functools.lru_cache(maxsize=None)
def _stage0():
    return json.load(open(BRIEF / "all_types.json", encoding="utf-8"))


def briefing_group(M, sigma=0, kind="react"):
    """The every-type-deployed group of the briefing family (all_types.json) at M types, sharpness 0.8."""
    K = 10 if M == 18 else 5
    return _stage0()["groups"][f"sandbox-traits-M{M}-K{K}-sharp0.8-noise{sigma:g}-{kind}"]


def llm(cell, model, mode="single", nshot=0):
    e = M.get(cell)
    if not e:
        return None
    for v in e["arms"].values():
        m = v["meta"]
        if m["arm"] == "llm" and m["model"] == model and m["mode"] == mode and m["nshot"] == nshot:
            return v
    return None


# ------------------------------------------------------------------------------------------------------------
# shared furniture: panel letters, badges, glyph labels, hand-laid legend rows, output
# ------------------------------------------------------------------------------------------------------------
def letter(ax, ch, dx=-0.02, dy=1.0):
    ax.text(dx, dy, ch, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="left", va="bottom")


def badge(fig, tag, x=0.992, y=0.994, ax=None, ha="right", va="top"):
    """A condition badge (C1-C3) at BADGE_FS; at the figure's top right unless an axes position is given."""
    kw = dict(fontsize=BADGE_FS, fontweight="bold", ha=ha, va=va, color=INK,
              bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#222222", lw=0.6))
    return (ax.text(x, y, tag, transform=ax.transAxes, **kw) if ax is not None else fig.text(x, y, tag, **kw))


def glyph_label(target, names, text, xy, xycoords="data", align=(0.0, 0.5), offset=(0, 0), fs=7, size_mm=2.5,
                sep=2.0, zorder=6, text_kw=None):
    """A glyph (or a row of glyphs) followed by ink text, kept together in points."""
    parts = [G.glyph_box(names, size_mm)] if names else []
    parts.append(TextArea(text, textprops=dict(fontsize=fs, color=INK, linespacing=1.05, **(text_kw or {}))))
    box = HPacker(children=parts, align="center", pad=0, sep=sep)
    ab = AnnotationBbox(box, xy, xycoords=xycoords, xybox=offset, boxcoords="offset points", frameon=False,
                        box_alignment=align, pad=0, annotation_clip=False, zorder=zorder)
    target.add_artist(ab)
    return ab


def legend_rows(fig, rows, x0, y0, fs=7, dy_pt=9.5, handle_pt=15.0, pad_pt=2.5, gap_pt=7.0, glyph_mm=2.5):
    """Legend laid out by hand in rows, in figure coordinates: (x0, y0) is the left end of the first row's centre line.
    Items: ("glyph", names) a category glyph; ("line", Line2D kwargs, label) a line sample, with an optional
    'marker' drawn once at its middle; ("marker", kwargs, label) a marker sample; ("text", label); ("skip", points)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_figwidth() * 72.0, fig.get_figheight() * 72.0
    right = 0.0
    for i, row in enumerate(rows):
        x, y = x0 * W, y0 * H - i * dy_pt
        for it in row:
            kind = it[0]
            if kind == "skip":
                x += it[1]
                continue
            if kind == "glyph":
                n = 1 if isinstance(it[1], str) else len(it[1])
                G.add_glyph(fig, it[1], (x / W, y / H), xycoords="figure fraction", size_mm=glyph_mm, align=(0.0, 0.5))
                x += n * glyph_mm * G.MM + (n - 1) * 0.5 * G.MM + pad_pt
                continue
            label = it[-1]
            if kind in ("line", "marker"):
                kw = dict(it[1])
                mk = {k: kw.pop(k) for k in list(kw) if k.startswith("marker") or k in ("ms", "mfc", "mec", "mew")}
                color = kw.pop("color", INK)
                if kind == "line":
                    fig.add_artist(Line2D([x / W, (x + handle_pt) / W], [y / H, y / H], transform=fig.transFigure,
                                          solid_capstyle="butt", color=color, **kw))
                if mk:
                    fig.add_artist(Line2D([(x + handle_pt / 2) / W], [y / H], transform=fig.transFigure, ls="",
                                          color=color, **mk))
                x += handle_pt + pad_pt
            if not label:
                continue
            t = fig.text(x / W, y / H, label, fontsize=fs, va="center", ha="left", color=INK)
            x += t.get_window_extent(r).width * 72.0 / fig.dpi + gap_pt
        right = max(right, x - gap_pt)
    return right / W


def save(fig, name):
    """Write PDF and PNG. The width is the full canvas, so a column figure prints at exactly 100% and every font at its
    set size; the height is cropped to the content. Anything that would stick out sideways is reported."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    extra = list(fig.get_default_bbox_extra_artists())
    for a in fig.axes:
        extra += [c for c in a.get_children() if isinstance(c, AnnotationBbox)]
    tb = fig.get_tightbbox(fig.canvas.get_renderer(), bbox_extra_artists=extra)
    W = fig.get_figwidth()
    if tb.x0 < -0.003 or tb.x1 > W + 0.003:
        print(f"WARNING {name}: content spans x {tb.x0:.3f}..{tb.x1:.3f} in, canvas 0..{W:.3f} in")
    kw = dict(bbox_inches=Bbox([[0.0, tb.y0 - 0.02], [W, tb.y1 + 0.02]]))
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None}, **kw)
    fig.savefig(FIG / f"{name}.png", dpi=300, **kw)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------------------------
# figure 2: headroom map
# ------------------------------------------------------------------------------------------------------------
# the four headline sandbox cells of the master table; the other four differ only in payoff noise and read the same
# H_D on every seed list (asserted below), so plotting them would only stack identical markers
HEADLINE_SANDBOX = {(3, 0.8, 0.0), (3, 0.6, 0.5), (6, 0.8, 0.0), (6, 0.6, 0.5)}
# single-type slice and the four-partner Overcooked cell: values of Section SI E3 and of the setup paragraph of the
# manuscript (diagnostic multi-window probe; SMAC reads at or below zero and is drawn at zero; Overcooked reads 0
# under its deployment executor and above 28 without the overrides, a lower bound drawn at 28.2)
FIRST_SLICE = {"hanabi": 0.19, "magent": 0.145, "smac": 0.0, "overcooked": 0.0, "overcooked_bare": 28.0}
OVERCOOKED_TYPED = 0.0


def headroom_map():
    pts = []  # (M, H_D, H_D_ci, calibration H_D, calibration H_D_ci, domain)
    twins = {}
    refs = sorted((c, e) for c, e in M.items() if any(v["meta"]["arm"] == "fixed" for v in e["arms"].values()))
    for c, e in refs:
        m = e["meta"]
        unc = EX["A3_uncertainty"]["cells"].get(c, {})
        dep = unc.get("deployment", {})
        cal = unc.get("calibration", {})
        lock = cal.get("H_D", (e.get("lock") or {}).get("H_D") if e.get("lock") else (e.get("closed") or {}).get("H_D"))
        if m["domain"] == "sandbox":
            twins.setdefault((m["M"], m["sharpness"]), []).append((e["reference"]["H_D"], lock))
            if (m["M"], m["sharpness"], float(m["reward_noise"])) not in HEADLINE_SANDBOX:
                continue
        pts.append((m["M"], e["reference"]["H_D"], dep.get("H_D_ci"), lock, cal.get("H_D_ci"), m["domain"]))
    for k, v in twins.items():
        assert len(v) == 2 and abs(v[0][0] - v[1][0]) < 1e-9 and abs(v[0][1] - v[1][1]) < 1e-9, ("noise twins differ", k, v)
    assert sum(p[5] == "sandbox" for p in pts) == 4
    fig, ax = plt.subplots(figsize=(3.5, 1.645))
    ax.set_yscale("symlog", linthresh=0.5, linscale=1.25)
    jit = {"battle": -0.12, "combined": 0.0, "sandbox": 0.12}
    shape = {"battle": "o", "combined": "s", "sandbox": "D"}
    # briefing family, every type deployed, s=0.8, sigma=0: the full-state headroom (exact) and beside it the prefix
    # ceiling in reward, i.e. the share of H_D that an exact reader of the other agent's first moves can reach at most
    # (prefix_ceiling_bayes_capture) times H_D. The axis breaks between 9 and 18.
    x_of = {9: 8.95, 18: 9.85}
    for i, Mb in enumerate((9, 18)):
        ex = briefing_group(Mb)["exact"]
        h, cap = ex["H_D"], ex["prefix_ceiling_bayes_capture"]
        x = x_of[Mb]
        ax.plot([x + 0.12, x + 0.12], [h, cap * h], color=INK, lw=0.7, ls=(0, (1, 1.2)), zorder=2)
        ax.plot([x - 0.12], [h], ls="", marker="*", ms=6.5, color=INK, mec=INK, zorder=4)
        ax.plot([x + 0.12], [cap * h], ls="", marker="_", ms=7, mew=1.6, color=INK, zorder=4)
        ax.annotate(f"{cap:.2f}", (x + 0.12, cap * h), xytext=(0, -3.5), textcoords="offset points", fontsize=7,
                    color=INK, va="top", ha="center")
    for Mc, h, hci, lock, lci, dom in pts:
        x = Mc + jit[dom]
        yerr = None if not hci else [[max(h - hci[0], 0)], [max(hci[1] - h, 0)]]
        ax.errorbar([x], [h], yerr=yerr, fmt=shape[dom], ms=3.4, color=INK, capsize=1.5, elinewidth=0.65,
                    markeredgewidth=0.6, zorder=3)
        if lock is not None:
            lx = x + 0.055
            lyerr = None if not lci else [[max(lock - lci[0], 0)], [max(lci[1] - lock, 0)]]
            ax.errorbar([lx], [lock], yerr=lyerr, fmt=shape[dom], ms=3.4, mfc="white", mec=INK, color=INK,
                        capsize=1.5, elinewidth=0.6, markeredgewidth=0.75, zorder=2)
    # M = 1: the single-type slice, a 3-mm glyph per task at its value, two side by side where the values coincide
    for name, x, y in (("hanabi", 0.69, FIRST_SLICE["hanabi"]), ("die", 1.07, FIRST_SLICE["magent"]),
                       ("smac", 0.69, FIRST_SLICE["smac"]), ("overcooked", 1.07, FIRST_SLICE["overcooked"]),
                       ("overcooked", 0.88, 28.2)):
        G.add_glyph(ax, name, (x, y), size_mm=3.0, zorder=5)
    # the Overcooked cell with four partner types, H_D = 0, beside the typed cells at M = 4
    G.add_glyph(ax, "overcooked", (4.34, OVERCOOKED_TYPED), size_mm=3.0, zorder=5)
    glyph_label(ax, None, "Overcooked, four partner types: 0", (4.62, OVERCOOKED_TYPED))
    # the bare-executor reading sits under its hat; the zero readings and the slice tasks are keyed by their glyphs
    ax.text(0.53, 4.6, "Overcooked,\nno overrides:\n$\\geq 28$", fontsize=7, va="center", ha="left", color=INK,
            linespacing=1.0)
    glyph_label(ax, "hanabi", "Hanabi 0.19", (1.28, 0.17), offset=(0, 5.0), sep=1.5)
    glyph_label(ax, "die", "MAgent 0.15", (1.28, 0.17), offset=(0, -4.6), sep=1.5)
    glyph_label(ax, ["smac", "overcooked"], "SMAC, Overcooked: 0", (1.28, 0.0), offset=(0, -9.0), sep=1.5)
    # family labels: a glyph before ink text; the marker shape keys the family in the plot
    glyph_label(ax, "battle", "battle: one\ncounter fits all", (4.32, 0.42), align=(0.0, 0.0))
    glyph_label(ax, "combined", "combined arms: two regimes", (4.2, 1.75), align=(0.0, 0.0))
    glyph_label(ax, "rps", "sandbox: each type needs its own throw", (2.62, 36.0), align=(0.0, 0.0))
    # the axis label shares its row with the legend; the caption names M the co-player type count
    ax.set_xlabel("type count $M$", ha="left")
    ax.xaxis.set_label_coords(-0.02, -0.165)
    ax.set_ylabel("headroom $H_D$ (reward)")
    ax.set_xticks([1, 3, 4, 6, 8, x_of[9], x_of[18]])
    ax.set_xticklabels(["1", "3", "4", "6", "8", "9", "18"])
    ax.set_xlim(0.42, 10.35)
    xb = (x_of[9] + x_of[18]) / 2
    for dx in (-0.09, 0.09):
        ax.plot([xb + dx - 0.07, xb + dx + 0.07], [-0.07, 0.07], transform=ax.get_xaxis_transform(), color="black",
                lw=0.6, clip_on=False, zorder=6)
    ax.plot([xb - 0.09, xb + 0.09], [0, 0], transform=ax.get_xaxis_transform(), color="white", lw=1.6, clip_on=False,
            zorder=5)
    ax.set_yticks([0, 0.5, 1, 3, 10, 30])
    ax.set_yticklabels(["0", "0.5", "1", "3", "10", "30"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.4, 50)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    fig.subplots_adjust(left=0.15, right=0.995, top=0.96, bottom=0.2)
    legend_rows(fig, [[("marker", dict(marker="*", ms=6.5, color=INK), "briefing game"),
                       ("marker", dict(marker="_", ms=7, mew=1.6, color=INK), "prefix ceiling")]],
                0.5, 0.035, handle_pt=8.0)
    badge(fig, "C1", x=0.985, y=0.035, ax=ax, va="bottom")
    save(fig, "fig_headroom_map")


# ------------------------------------------------------------------------------------------------------------
# figure 3: legibility (encoding) and examples
# ------------------------------------------------------------------------------------------------------------
def panel_title(ax, glyphs, text, y=1.04, fs=7.5):
    """Panel title led by a game glyph, centred over the axes."""
    glyph_label(ax, glyphs, text, (0.5, y), xycoords="axes fraction", align=(0.5, 0.0), fs=fs)


def legibility():
    encs = ["semantic", "numeric", "opaque", "relabel", "swapdesc"]
    panels = [("combined", "combined base4", lambda enc: f"combined-base4-{enc}", "combined-base4-semantic"),
              ("combined", "combined traits8", lambda enc: f"combined-traits8-{enc}", "combined-traits8-semantic"),
              ("rps", "sandbox $M{=}3$, $s{=}0.8$, $\\sigma{=}0$", lambda enc: f"sandbox-M3-K3-sharp0.8-noise0-{enc}",
               "sandbox-M3-K3-sharp0.8-noise0-nonllm")]
    models = ("gpt-oss-20b", "gpt-oss-120b", "qwen3.8-27b")
    dodge = {"gpt-oss-20b": -0.17, "gpt-oss-120b": 0.0, "qwen3.8-27b": 0.17}
    # axes placed by hand: a and b on top, c below a, and below b the examples panel d beside the briefing panel e
    fig = plt.figure(figsize=(3.5, 3.495))
    left, right, top, bottom = 0.15, 0.985, 0.91, 0.27
    aw = (right - left) / 2.36
    ah = (top - bottom) / 3.05
    xb = left + 1.36 * aw
    ew, egap = 0.50 / 3.5, 0.32 / 3.5
    axes = [fig.add_axes([left, top - ah, aw, ah]), fig.add_axes([xb, top - ah, aw, ah]),
            fig.add_axes([left, bottom, aw, ah]), fig.add_axes([xb, bottom, right - ew - egap - xb, ah]),
            fig.add_axes([right - ew, bottom, ew, ah])]
    for pi, (ax, (gl, title, cell_of, lcell)) in enumerate(zip(axes[:3], panels)):
        for key, ls in (("plastic", ":"), ("linucb", "--")):
            v = M[lcell]["arms"].get(key)
            if v:
                ax.axhline(v["capture"], color=DARK, ls=ls, lw=0.8, zorder=1)
        for model in models:
            for i, enc in enumerate(encs):
                v = llm(cell_of(enc), model)
                if not v:
                    continue
                lo, hi = v["capture_ci"]
                ax.errorbar([i + dodge[model]], [v["capture"]], yerr=[[v["capture"] - lo], [hi - v["capture"]]],
                            color=COL[model], marker="o", ms=2.8, capsize=0, elinewidth=0.8, ls="", zorder=3)
        ctrl = EX["A5_opaque_controls"]["controls"].get(cell_of("opaque"), {}).get("one_hot_controls", {})
        if ctrl.get("tabular_ucb1_signature"):
            ax.scatter([encs.index("opaque") - 0.36], [ctrl["tabular_ucb1_signature"]["capture"]], marker="x", s=22,
                       color="#000000", linewidths=0.9, zorder=5)
        if ctrl.get("library_lookup_signature"):
            ax.scatter([encs.index("opaque") + 0.36], [ctrl["library_lookup_signature"]["capture"]], marker="+", s=26,
                       color="#000000", linewidths=0.9, zorder=5)
        ax.set_xticks(range(len(encs)))
        ax.set_xticklabels(encs, rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.55, len(encs) - 0.45)
        panel_title(ax, gl, title)
        if pi < 2:
            ax.set_ylim(-6.0, 1.3)
            ax.set_yticks([-6, -4, -2, 0])
        else:
            # the sandbox panel on its own scale, so the relabel / swapdesc contrast is visible
            ax.set_ylim(-0.8, 1.12)
            ax.set_yticks([-0.5, 0, 0.5, 1])
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="#999999", lw=0.4, zorder=0)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("capture $c$", fontsize=7, labelpad=2)
    # panel d: examples per type in combined arms, M = 4, with 95% intervals; the few-shot classifier beside them
    ax = axes[3]
    cell = "combined-base4-semantic"
    for model in models:
        for xi, n in enumerate((0, 1, 5)):
            v = llm(cell, model, "single", n)
            if not v:
                continue
            lo, hi = v["capture_ci"]
            ax.errorbar([xi + dodge[model]], [v["capture"]], yerr=[[v["capture"] - lo], [hi - v["capture"]]],
                        color=COL[model], marker="o", ms=2.8, capsize=0, elinewidth=0.8, ls="",
                        zorder=4 if model == "qwen3.8-27b" else 3)
    for xi, n in ((1, 1), (2, 5), (3, 20)):
        v = M[cell]["arms"].get(f"fewshot__{n}")
        if v:
            lo, hi = v["capture_ci"]
            ax.errorbar([xi + 0.36 if xi < 3 else xi], [v["capture"]], yerr=[[v["capture"] - lo], [hi - v["capture"]]],
                        color=DARK, marker="s", ms=2.8, mfc="white", capsize=0, elinewidth=0.8, ls="", zorder=2)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xlim(-0.4, 3.4); ax.set_xticklabels(["0", "1", "5", "20"], fontsize=7)
    ax.set_xlabel("examples\nper type $N$", fontsize=7)
    panel_title(ax, "combined", "base4", fs=7)
    ax.set_ylim(-6.0, 1.3)
    ax.set_yticks([-6, -4, -2, 0])
    ax.tick_params(labelsize=7)
    ax.axhline(0, color="#999999", lw=0.4, zorder=0)
    # panel e: what a briefing adds over the first moves, gpt-oss-120b zero-shot (single), in the briefing game with
    # M=18, sigma=0, every type deployed: the paired difference in reward per episode, briefing minus prefix-only
    # encoding, with its 95% interval, for briefings in known and in new wordings
    bx = axes[4]
    g18 = briefing_group(18)
    for i, w in enumerate(("briefing-sameword", "briefing-newword")):
        v = g18["briefing_minus_semantic"][f"{w}/llm__gpt-oss-120b__single__nshot0"]
        bx.errorbar([i], [v["diff"]], yerr=[[v["diff"] - v["ci"][0]], [v["ci"][1] - v["diff"]]], color=COL["gpt-oss-120b"],
                    marker="o", ms=2.8, capsize=0, elinewidth=0.8, ls="")
    h18 = g18["exact"]["H_D"]
    bx.axhline(h18, color=REF, ls=(0, (1, 1.5)), lw=0.8)
    bx.text(1.35, h18, "$H_D$", fontsize=7, ha="right", va="bottom", color=INK)
    bx.axhline(0, color="#999999", lw=0.4)
    bx.set_xlim(-0.35, 1.35)
    bx.set_ylim(-1.0, 21.5)
    bx.set_yticks([0, 10, 20])
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["known", "new"], fontsize=7)
    bx.set_xlabel("wording", fontsize=7)
    bx.set_ylabel("reward gain", fontsize=7, labelpad=1)
    bx.set_title("briefing vs\nfirst moves", fontsize=7, pad=3)
    bx.tick_params(labelsize=7)
    for a, ch, dx in zip(axes, "abcde", (-0.3, -0.2, -0.3, -0.45, -0.62)):
        letter(a, ch, dx=dx, dy=1.04)
    # legend: the three models with their family chips, then the reference lines and the opaque-token controls
    mk = lambda m: dict(marker="o", ms=2.8, color=COL[m])  # noqa: E731
    legend_rows(fig, [
        [("marker", mk("gpt-oss-20b"), ""), ("glyph", "chip20b"), ("text", "gpt-oss-20b"),
         ("marker", mk("gpt-oss-120b"), ""), ("glyph", "chip120b"), ("text", "gpt-oss-120b"),
         ("marker", mk("qwen3.8-27b"), ""), ("glyph", "chipqwen"), ("text", "Qwen3.8-27B")],
        [("glyph", "lever"), ("line", dict(color=DARK, ls="--", lw=0.8), "LinUCB"),
         ("glyph", "books"), ("line", dict(color=DARK, ls=":", lw=0.8), "PLASTIC"),
         ("marker", dict(marker="s", ms=2.8, mfc="white", mec=DARK), "few-shot classifier")],
        [("marker", dict(marker="x", ms=4.0, color="#000000", mew=0.9), "bandit, same token"),
         ("marker", dict(marker="+", ms=4.6, color="#000000", mew=0.9), "library, same token")],
    ], 0.035, 0.085, handle_pt=8.0, gap_pt=6.0)
    badge(fig, "C2")
    save(fig, "fig_legibility")


# ------------------------------------------------------------------------------------------------------------
# figure 4: capture curves and crossover horizons
# ------------------------------------------------------------------------------------------------------------
def curves():
    fig = plt.figure(figsize=(3.5, 3.62))
    ax = fig.add_axes([0.14, 0.49, 0.826, 0.315])
    bx = fig.add_axes([0.14, 0.085, 0.826, 0.25])
    learners = ["ppo", "mucb", "ctxucb", "linucb", "lints", "plastic", "fewshot__5"]   # drawing order, bottom first
    # panel a: combined base4 paired curves; the library learners and Qwen with one example sit at 1 from the first
    # episode: PLASTIC drawn wide in black, the five-shot classifier and Qwen dashed on top with alternating dashes
    cell, ref = "combined-base4-semantic", "combined-base4-once"
    for z, k in enumerate(learners):
        v = M[cell]["arms"].get(k) or M[ref]["arms"].get(k)
        if v and "curve_paired" in v:
            ax.plot(range(1, len(v["curve_paired"]) + 1), v["curve_paired"], zorder=2 + z, **STY[k])
    v1 = llm(cell, "qwen3.8-27b", "single", 1)
    v5 = llm(cell, "gpt-oss-120b", "single", 5)
    ax.axhline(v1["capture"], color=COL["qwen3.8-27b"], ls=(0, (3.0, 3.0)), lw=1.1, zorder=12)
    ax.axhline(v5["capture"], color=COL["gpt-oss-120b"], ls=(0, (3.0, 1.5)), lw=1.1, zorder=12)
    ax.set_xlim(1, 100)
    ax.set_xlabel("deployment episodes $T$ (combined arms, $M{=}4$, paired)", fontsize=7)
    ax.set_ylabel("capture $c(T)$", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-2.6, 1.45)
    ax.set_yticks([-2, -1, 0, 1])
    ax.axhline(0, color="#999999", lw=0.4, zorder=0)
    G.add_glyph(fig, "combined", (0.062, 0.806), xycoords="figure fraction", size_mm=2.5, align=(0.0, 0.0))
    # panel b: population crossover horizon of the two linear bandits against zero-shot gpt-oss-120b in the eight
    # sandbox cells, against the predicted sample-complexity scale; the two bandits are offset by 4.5% in x (LinTS right), a learner
    # that has not caught the model by episode 300 is drawn hollow at 300
    rows = []
    for c, e in M.items():
        if not (c.startswith("sandbox") and c.endswith("-nonllm")):
            continue
        v = llm(c.replace("-nonllm", "-semantic"), "gpt-oss-120b")
        if not v or not v.get("crossover"):
            continue
        pred = e["closed"]["predicted_T_star"]
        for key in ("lints", "linucb"):
            t = (v["crossover"].get(key) or {}).get("T_star_population")
            rows.append((key, pred, t))
    off = {"lints": 1.045, "linucb": 0.955}
    fill = {"lints": "#000000", "linucb": DARK}
    for key, pred, t in rows:
        x = pred * off[key]
        if t is None:
            bx.plot([x], [300], ls="", marker=MARK[key], ms=4.0, mfc="white", mec=fill[key], mew=0.9, zorder=3)
        else:
            bx.plot([x], [t], ls="", marker=MARK[key], ms=4.0, color=fill[key], mec=fill[key], zorder=4)
    lo, hi = 5.0, 95.0
    bx.plot([lo, hi], [lo, hi], color=REF, lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    bx.text(30, 24, "as predicted", fontsize=7, color=INK, ha="left", va="top")
    bx.axhline(300, color="#cccccc", lw=0.5, zorder=0)
    bx.text(12.6, 300, "not caught by 300", fontsize=7, color=INK, ha="left", va="center",
            bbox=dict(fc="white", ec="none", pad=0.6))
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlim(5.0, 95.0); bx.set_ylim(3.0, 560.0)
    bx.set_xticks([5, 10, 20, 50]); bx.set_xticklabels(["5", "10", "20", "50"])
    bx.set_yticks([3, 10, 30, 100, 300]); bx.set_yticklabels(["3", "10", "30", "100", "300"])
    bx.xaxis.set_minor_locator(NullLocator()); bx.yaxis.set_minor_locator(NullLocator())
    bx.set_xlabel("predicted scale $MK\\sigma_{\\mathrm{eff}}^2/\\Delta_{\\min}^2$ (Monte Carlo)", fontsize=7)
    bx.set_ylabel("crossover $T^\\star$", fontsize=7)
    bx.tick_params(labelsize=7)
    panel_title(bx, "rps", "sandbox, eight cells: when a bandit catches gpt-oss-120b", y=1.03, fs=7)
    letter(ax, "a", dx=-0.13, dy=1.0)
    letter(bx, "b", dx=-0.13, dy=1.0)
    # legend by category glyph: the bandits, the library learners, PPO, the two in-context captures
    s = lambda k, **extra: {**STY[k], **extra}  # noqa: E731
    legend_rows(fig, [
        [("glyph", "lever"), ("line", s("linucb", marker="s", ms=3.4), "LinUCB"),
         ("line", s("lints", marker="o", ms=3.4), "linear Thompson sampling (LinTS)")],
        [("skip", 7.1 + 2.5), ("line", s("ctxucb"), "binned bandit (CtxUCB)"), ("line", s("mucb"), "blind bandit (M-UCB)")],
        [("glyph", "books"), ("line", s("plastic"), "PLASTIC"), ("line", s("fewshot__5", ls=(0, (3.0, 3.0))), "five-shot classifier"),
         ("glyph", "net"), ("line", s("ppo"), "PPO")],
        [("glyph", "chipqwen"), ("line", dict(color=COL["qwen3.8-27b"], ls=(0, (3.0, 3.0)), lw=1.1), "Qwen, one example per type"),
         ("glyph", "chip120b"), ("line", dict(color=COL["gpt-oss-120b"], ls=(0, (3.0, 1.5)), lw=1.1), "120b, five per type")],
    ], 0.012, 0.975, handle_pt=13.0, gap_pt=6.0)
    badge(fig, "C3")
    save(fig, "fig_curves")


if __name__ == "__main__":
    headroom_map()
    legibility()
    curves()
