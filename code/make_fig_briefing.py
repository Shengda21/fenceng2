"""Figure of the briefing section, drawn from the three briefing analyses: the nested layout of this repository (analysis/briefing/, names
all_types.json, new_agents.json, task_prompt.json) or a flat analysis/ layout with the earlier file names, selected by
--root (default: this repository).

fig_briefing : M=18 types, sigma 0, briefings in new words (double column)
  a  known against new agents: for twelve headline readers, the capture with most agents known (every type deployed,
     all_types.json), against four new types and against eight new types (held-out-only deployments,
     new_agents.json; prompt, example and reasoning variants from task_prompt.json; the factorized reader with the
     simulator composer from fcr_sym.json and with the fitted composer from fcr.json), with 95% intervals; a short
     black tick per condition marks the strongest tested non-LLM arm of that deployment with both composers among the
     candidates, a ring an arm that passes its pre-registered test against it (verdict yes, and a primary test or a
     secondary test that survives Holm correction, recomputed by briefing_c3)
  b  population capture curves of the library-free learners to T=300 against three language-model arms on the same
     scale, against four new types
Every plotted value is checked against numbers_briefing.json (the rows of Table 2, written to latex/tables/ in a
working tree or analysis/tables/ in this repository) where that file has it.
Re-run after any of the three analyses changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_main_figures import (COL, DARK, INK, REF, STY, G, badge, legend_rows, letter, plt,  # noqa: E402
                               save)
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default=None, help="briefing analyses: a working tree or this repository (see module docstring)")
_args = _ap.parse_args()

_HERE_ROOT = Path(__file__).resolve().parents[1]
_src_root = Path(_args.root) if _args.root else (_HERE_ROOT if (_HERE_ROOT / "analysis" / "briefing").is_dir() else Path("<LOCAL_PATH>"))  # the else is unused here
_briefing = _src_root / "analysis" / "briefing"
SRC, _OLD = (_briefing, False) if _briefing.is_dir() else (_src_root / "analysis", True)
FN = ("stage0.json", "stage0b.json", "stage0c.json") if _OLD else ("all_types.json", "new_agents.json", "task_prompt.json")
S0, S0B, S0C = (json.load(open(SRC / f, encoding="utf-8")) for f in FN)
# the factorized compositional reader: analysis/briefing/fcr.json in this repository, analysis/fcr.json in the paper tree
_fcr = next(p for p in (_HERE_ROOT / "analysis" / "briefing" / "fcr.json", _HERE_ROOT / "analysis" / "fcr.json")
            if p.is_file())
FCR = json.load(open(_fcr, encoding="utf-8"))
from briefing_c3 import C3, load_fcr_sym  # noqa: E402
FS = load_fcr_sym(_HERE_ROOT)
C3V = C3(S0, S0B, S0C, FS)
NUMB = _HERE_ROOT / ("analysis" if (_HERE_ROOT / "analysis" / "briefing").is_dir() else "latex") / "tables" / "numbers_briefing.json"
NUMB = json.load(open(NUMB, encoding="utf-8")) if NUMB.is_file() else {}
P0 = "sandbox-traits-M18-K10-sharp0.8-noise0-react"      # four new types (held-out deployment) / every type (all_types)
R8 = "sandbox-traits-M18-K10-sharp0.8-noise0-react8"     # eight new types
NEW = "briefing-newword"
# deployment -> (analysis, its file name, task_prompt part, group, numbers_briefing tag)
DEP = {"known": (S0, FN[0], "s0", P0, "full"), "four": (S0B, FN[1], "s0b", P0, "react4"),
       "eight": (S0B, FN[1], "s0b", R8, "react8")}
CONDS = ("known", "four", "eight")
BANDITS = ("linucb", "lints", "ctxucb", "mucb")
# rows of panel a: (arm key, category glyph, first label line, second label line, long reasoning). Labels match
# Table 2's naming (short model tags, "task"/"bare", N examples per known type) so the two figures read as one system.
ROWS = [("text_bow", "lens", "TF-IDF", None, False),
        ("text_embed", "lens", "embedding", None, False),
        ("scripted_text", "lens", "keyword rule", None, False),
        ("sym_full", "lens", "FCR, simulator composer", None, False),
        ("fcr_best", "lens", "FCR, fitted composer", None, False),
        ("bandit", "lever", "strongest bandit", None, False),
        ("llm__gpt-oss-120b__single__nshot0__v2", "chip120b", "120b, task, 0/type", None, False),
        ("llm__gpt-oss-120b__single__nshot1__v2", "chip120b", "120b, task, 1/type", None, False),
        ("llm__gpt-oss-20b__single__nshot1__v2", "chip20b", "20b, task, 1/type", None, False),
        ("llm__qwen3.8-27b__single__nshot1", "chipqwen", "Qwen, bare, 1/type", None, False),
        ("llm__qwen3.8-27b__single__nshot1__v2", "chipqwen", "Qwen, task, 1/type", None, False),
        ("llm__qwen3.8-27b__single__nshot1__v2-think-long", "chipqwen", "Qwen, task thinking, 1/type", None, True)]


def variant(key):
    parts = key.split("__")
    return parts[4] if key.startswith("llm__") and len(parts) > 4 else None


def find(dep, cell, key):
    """(record, source path) of an arm: bare-prompt arms from all_types/new_agents, variants from task_prompt."""
    d, fname, part, group, _ = DEP[dep]
    if variant(key):
        return (S0C[part].get(group, {}).get("arms", {}).get(f"{cell}/{key}"),
                f"{FN[2]}:{part}.{group}.arms.{cell}/{key}")
    return d["groups"][group]["cells"].get(cell, {}).get("arms", {}).get(key), f"{fname}:groups.{group}.cells.{cell}.arms.{key}"


def _primary_variant():
    out = set()
    for k in S0C["decision"]["primary_0c"]:
        part, model = k.split("/")
        out.add(("s0" if part == "s0" else "s0b", P0 if part in ("s0", "s0b-react4") else R8, f"llm__{model}__single__nshot0__v2"))
    return out


PRIMARY_VARIANT = _primary_variant()


def test_status(dep, cell, key):
    """P: the pre-registered primary test; H: a secondary test that survives Holm correction; S: otherwise (the rule
    of Table 2), against the strongest tested non-LLM arm with both composers among the candidates (briefing_c3)."""
    d, _, part, group, _ = DEP[dep]
    if variant(key):
        prim = (part, group, key) in PRIMARY_VARIANT and cell == NEW
    else:
        pc = d["P_E"]["primary_cell"]
        pg = next(g for g, v in d["groups"].items() if v["meta"]["M"] == pc["M"] and v["meta"]["reward_noise"] == pc["reward_noise"]
                  and v["meta"]["kind"] == pc["kind"])
        prim = (group == pg and cell == pc["cell"] and f"__{pc['mode']}__nshot{pc['nshot']}" in key
                and (pc.get("model") is None or f"__{pc['model']}__" in key))
    a = C3V.get(part, group, cell, key)
    assert a["prim"] == prim, (dep, cell, key)
    return a["status"]


def strongest_bandit(dep):
    recs = [(k, find(dep, "nonllm", k)[0]) for k in BANDITS]
    return max(((k, v) for k, v in recs if v), key=lambda kv: kv[1]["mean"])[0]


def cell_of(key):
    return "nonllm" if key in BANDITS else NEW


PLOTTED = []   # (row label, condition, arm, capture, ci, passes, source) for the notes


FCR_DEP = {"known": f"all/{P0}/newword", "four": f"heldout/{P0}/newword", "eight": f"heldout/{R8}/newword"}


def fcr_best(dep):
    """The strongest of the nine non-privileged FCR arms of a deployment, by headline mean (the rule of Table 2)."""
    arms = FCR["deployments"][FCR_DEP[dep]]["arms"]
    names = list(FCR["meta"]["fcr_arms"])
    return max(names, key=lambda k: (arms[k]["mean"], -names.index(k)))


def fcr_point(row_key, dep):
    """A factorized-reader row: the simulator composer (sym_*, fcr_sym.json) or a fitted-composer arm (fcr.json)."""
    key = fcr_best(dep) if row_key == "fcr_best" else row_key
    src_json, pre = (FS, "sym") if key.startswith("sym_") else (FCR, "fcr")
    rec = src_json["deployments"][FCR_DEP[dep]]["arms"][key]
    tab = NUMB.get(f"tab_briefing:{DEP[dep][4]}:{pre}/{key}")
    if tab is not None:   # the row is in Table 2: the figure must agree with it
        assert abs(tab["capture"] - rec["capture"]) < 1e-12, (dep, key)
    PLOTTED.append((row_key, dep, key, rec["capture"], rec["capture_ci"], rec.get("n"), False, "--",
                    f"tab_briefing:{DEP[dep][4]}:{pre}/{key}" if tab is not None else
                    f"{'fcr_sym' if pre == 'sym' else 'fcr'}.json:{FCR_DEP[dep]}.{key}"))
    return key, rec, False


def point(row_key, dep):
    if row_key.startswith(("fcr_", "sym_")):
        return fcr_point(row_key, dep)
    key = strongest_bandit(dep) if row_key == "bandit" else row_key
    cell = cell_of(key)
    rec, src = find(dep, cell, key)
    if rec is None:
        return None
    is_llm = key.startswith("llm__")
    status = test_status(dep, cell, key) if is_llm else "--"
    pays = C3V.get(DEP[dep][2], DEP[dep][3], cell, key)["pays"] if is_llm else None
    passes = is_llm and pays == "yes" and status in ("P", "H")
    tab = NUMB.get(f"tab_briefing:{DEP[dep][4]}:{cell}/{key}")
    if tab is not None:   # the row is in Table 2: the figure must agree with it
        assert abs(tab["capture"] - rec["capture"]) < 1e-12, (dep, key)
        if is_llm:
            assert tab["test"] == status and tab["pays"] == pays, (dep, key, tab["test"], status)
    PLOTTED.append((row_key, dep, key, rec["capture"], rec["capture_ci"], rec.get("n"), passes, status,
                    f"tab_briefing:{DEP[dep][4]}:{cell}/{key}" if tab is not None else src))
    return key, rec, passes


def strongest_nonllm(dep):
    """The strongest tested non-LLM arm of a deployment with both composers among the candidates, and its capture."""
    _, _, part, group, tag = DEP[dep]
    s = C3V.strongest(part, group)
    if s.startswith(("sym_", "fcr_")):
        rec = (FS if s.startswith("sym_") else FCR)["deployments"][FCR_DEP[dep]]["arms"][s]
    else:
        cell, key = s.replace("briefing:", NEW + "/").replace("nonllm:", "nonllm/").split("/")
        rec, _ = find(dep, cell, key)
    tab = NUMB.get(f"tab_briefing:{tag}:" + next(iter(NUMB[k]["strongest_nonllm"] for k in NUMB
                                                       if k.startswith(f"tab_briefing:{tag}:"))))
    if tab is not None:   # Table 2 marks the same arm with its dagger
        assert abs(tab["capture"] - rec["capture"]) < 1e-12, (dep, s)
    return s, rec["capture"]


def population_capture(rec):
    """An arm's mean reward on the population scale of the learners' curves (exact V* and H_D of the cell)."""
    gb = S0B["groups"][P0]
    mean = rec["mean"] if "mean" in rec else rec["margin"] + gb["reference"]["V_star"]
    return (mean - gb["exact"]["V_star"]) / gb["exact"]["H_D"]


def row_label(ax, y, glyph, line1, line2, long_reasoning):
    text = line1 if not line2 else f"{line1}\n{line2}"
    parts = [G.glyph_box(glyph, 2.5), TextArea(text, textprops=dict(fontsize=7, color=INK, linespacing=1.0))]
    if long_reasoning:
        parts.append(G.glyph_box("hourglass", 2.5))
    box = HPacker(children=parts, align="center", pad=0, sep=2.0)
    ax.add_artist(AnnotationBbox(box, (0.004, y), xycoords=("figure fraction", "data"), frameon=False,
                                 box_alignment=(0.0, 0.5), pad=0, annotation_clip=False))


def figure():
    fig = plt.figure(figsize=(7.0, 3.28))
    ax = fig.add_axes([0.262, 0.105, 0.355, 0.77])
    bx = fig.add_axes([0.708, 0.40, 0.284, 0.46])
    off = {"known": 0.31, "four": 0.0, "eight": -0.31}
    style = {"known": dict(marker="o", ms=4.3, mew=0.9), "four": dict(marker="o", ms=4.3, mew=0.9),
             "eight": dict(marker="D", ms=3.7, mew=0.9)}
    XMIN, XMAX = -1.06, 1.1
    ref = {dep: strongest_nonllm(dep) for dep in CONDS}
    ys = []
    n_nonllm = sum(1 for r in ROWS if not r[0].startswith("llm__"))
    for i, (rk, glyph, l1, l2, longr) in enumerate(ROWS):
        y = -(i + (0.45 if i >= n_nonllm else 0.0))
        ys.append(y)
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f2f2f2", lw=0, zorder=0)
        row_label(ax, y, glyph, l1, l2, longr)
        pts = []
        for dep in CONDS:
            got = point(rk, dep)
            if got is None:
                continue
            key, rec, passes = got
            col = COL[key.split("__")[1]] if key.startswith("llm__") else DARK
            yy = y + off[dep]
            c = rec["capture"]
            # the strongest tested non-LLM arm of this deployment, as a short tick on this condition's line
            ax.plot([ref[dep][1]] * 2, [yy - 0.13, yy + 0.13], color="#000000", lw=1.0, solid_capstyle="butt", zorder=2)
            if c < XMIN + 0.04:
                ax.plot([XMIN + 0.03], [yy], ls="", marker="<", ms=4.0, color=col, zorder=4)
                pts.append((XMIN + 0.03, yy))
                continue
            lo, hi = rec["capture_ci"]
            ax.plot([max(lo, XMIN), min(hi, XMAX)], [yy, yy], color=col, lw=0.75, solid_capstyle="butt", zorder=3)
            mfc = "white" if dep == "known" else col
            ax.plot([c], [yy], ls="", color=col, mfc=mfc, mec=col, zorder=5, **style[dep])
            if passes:
                ax.plot([c], [yy], ls="", marker="o", ms=7.6, mfc="none", mec="#000000", mew=0.7, zorder=6)
            pts.append((c, yy))
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#c4c4c4", lw=0.6, zorder=1)
    # the keyword rule against new agents lies off the axis, at the same value for four and eight new types
    kw4, kw8 = (find(dep, NEW, "scripted_text")[0]["capture"] for dep in ("four", "eight"))
    assert f"{kw4:.2f}" == f"{kw8:.2f}"
    ax.text(XMIN + 0.09, ys[2] - 0.155, f"both {kw4:.2f}".replace("-", "\u2212"), fontsize=7, color=INK, va="center", ha="left")
    ax.axvline(0, color="#999999", lw=0.5, zorder=1)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(ys[-1] - 0.62, 0.62)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["\u22121", "\u22120.5", "0", "0.5", "1"])
    ax.set_xlabel("capture $c$", fontsize=7)
    ax.tick_params(labelsize=7, length=2)
    # panel a legend: the three conditions, the reference tick, the ring, the hourglass
    legend_rows(fig, [
        [("marker", dict(marker="o", ms=4.3, mfc="white", mec=INK, mew=0.9), "most agents known"),
         ("marker", dict(marker="o", ms=4.3, color=INK), "four new types"),
         ("marker", dict(marker="D", ms=3.7, color=INK), "eight new types")],
        [("marker", dict(marker="|", ms=7.0, mew=1.0, color="#000000"), "strongest tested non-LLM"),
         ("marker", dict(marker="o", ms=7.6, mfc="none", mec="#000000", mew=0.7), "passes its pre-registered test"),
         ("glyph", "hourglass"), ("text", "long reasoning")],
    ], 0.012, 0.965, handle_pt=9.0, gap_pt=8.0)
    # panel b: the library-free learners against four new types, on the population scale
    for k, cell in (("linucb", "nonllm"), ("lints", "nonllm"), ("ctxucb", "nonllm"), ("text_linucb", NEW)):
        curve = S0B["groups"][P0]["cells"][cell]["arms"][k]["curve_population"]
        bx.plot(range(1, len(curve) + 1), curve, zorder=3, **STY[k])
    lines = [(S0C["s0b"][P0]["arms"].get(f"{NEW}/llm__gpt-oss-120b__single__nshot0__v2"), "gpt-oss-120b", (0, (7.0, 2.0))),
             (S0B["groups"][P0]["cells"][NEW]["arms"].get("llm__qwen3.8-27b__single__nshot1"), "qwen3.8-27b", (0, (1.0, 1.3))),
             (S0C["s0b"][P0]["arms"].get(f"{NEW}/llm__qwen3.8-27b__single__nshot1__v2"), "qwen3.8-27b", (0, (3.0, 1.5)))]
    for rec, model, ls in lines:
        bx.axhline(population_capture(rec), color=COL[model], ls=ls, lw=1.1, zorder=4)
    t_pred = S0B["groups"][P0]["reference_0b"]["predicted_T_star"]
    bx.axvline(t_pred, color=REF, ls=(0, (1, 2)), lw=0.8, zorder=2)
    bx.text(t_pred * 0.9, 1.19, "predicted $T^\\star$", fontsize=7, ha="right", va="top", color=INK)
    # prefix ceiling of this deployment (exact, the four new types): the first moves reveal all of its headroom
    ceiling = S0B["groups"][P0]["exact"]["prefix_ceiling_bayes_capture"]
    bx.axhline(ceiling, color=DARK, ls=(0, (5.0, 1.5, 1.2, 1.5)), lw=0.8, zorder=2)
    bx.text(1.12, ceiling + 0.03, "prefix ceiling", fontsize=7, ha="left", va="bottom", color=INK)
    bx.axhline(0, color="#999999", lw=0.4)
    bx.set_xscale("log")
    bx.set_xlim(1, 300)
    bx.set_ylim(-1.5, 1.24)
    bx.set_xticks([1, 10, 100]); bx.set_xticklabels(["1", "10", "100"])
    bx.set_xlabel("deployment episodes $T$ (log)", fontsize=7)
    bx.set_ylabel("capture $c(T)$, population", fontsize=7)
    bx.tick_params(labelsize=7, length=2)
    s = lambda k: dict(STY[k])  # noqa: E731
    legend_rows(fig, [
        [("glyph", "lever"), ("line", s("linucb"), "LinUCB"), ("line", s("ctxucb"), "binned bandit (CtxUCB)")],
        [("skip", 7.1 + 2.5), ("line", s("lints"), "linear Thompson sampling (LinTS)")],
        [("glyph", "lens"), ("line", s("text_linucb"), "text LinUCB")],
        [("glyph", "chip120b"), ("line", dict(color=COL["gpt-oss-120b"], ls=(0, (7.0, 2.0)), lw=1.1), "120b, task prompt, zero-shot")],
        [("glyph", "chipqwen"), ("text", "Qwen, one example per known type:")],
        [("skip", 7.1 + 2.5), ("line", dict(color=COL["qwen3.8-27b"], ls=(0, (1.0, 1.3)), lw=1.1), "bare prompt"),
         ("line", dict(color=COL["qwen3.8-27b"], ls=(0, (3.0, 1.5)), lw=1.1), "task prompt")],
    ], 0.66, 0.245, handle_pt=13.0, gap_pt=6.0, dy_pt=9.2)
    letter(ax, "a", dx=-0.72, dy=1.0)
    letter(bx, "b", dx=-0.2, dy=1.0)
    badge(fig, "C3")
    save(fig, "fig_briefing")


if __name__ == "__main__":
    figure()
    for r in PLOTTED:
        rk, dep, key, c, ci, n, passes, status, src = r
        print(f"{rk:48s} {dep:6s} {key:50s} {c:+.3f} [{ci[0]:+.2f},{ci[1]:+.2f}] n={n} {status} ring={passes}  {src}")
