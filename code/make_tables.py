"""Generate the LaTeX tables of the manuscript and supplement from analysis/master.json.

Outputs (latex/tables/): tab_master.tex (main text), tab_sandbox.tex (main text), tab_ledger.tex (prediction ledger, main),
si_master_full.tex (every cell x every arm), si_pertype.tex, si_encoding.tex, si_nshot.tex, si_crossover.tex, si_heldout.tex,
and numbers.json (every number quoted in prose, keyed) for the check_numbers gate.
"""
from __future__ import annotations

import json
import math
import re as _re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from briefing_c3 import C3, load_fcr_sym  # noqa: E402

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
args = ap.parse_args()
ROOT = Path(args.root).resolve()
D = json.load(open(ROOT / "analysis" / "master.json", encoding="utf-8"))
M = D["master"]
H = D["heldout"]
EX = json.load(open(ROOT / "analysis" / "extras.json", encoding="utf-8"))
OUT = ROOT / "analysis" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
NUM: dict = OrderedDict()
MEASURE = ROOT / "analysis" / "tab_master_width.txt"


def f(x, d=2, plus=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.{d}f}" if plus else f"{x:.{d}f}"
    return s


def ci(v, key="capture", d=2):
    lo, hi = v[f"{key}_ci"]
    return f"${f(v[key], d)}\\,[{f(lo, d)},{f(hi, d)}]$"


def ci_raw(value, interval, d=2, plus=False, math=True):
    if value is None or not interval or interval[0] is None or interval[1] is None:
        return "--"
    s = f"{f(value, d, plus)}\\,[{f(interval[0], d, plus)},{f(interval[1], d, plus)}]"
    return f"${s}$" if math else s


def point_ci(v, key, d=2):
    return ci_raw(v.get(key), v.get(f"{key}_ci"), d=d)


def grey_ci(lo, hi, d=2, plus=False):
    """A '[lo,hi]' interval with each bound set in math mode, for twomain's grey secondary line: a negative bound
    then prints a math minus, not a text hyphen, like the main (black) values above it."""
    return f"[${f(lo, d, plus)}$,${f(hi, d, plus)}$]"


def tex(s: str) -> str:
    return str(s).replace("_", "\\_")


MODELS = [("gpt-oss-20b", "20b"), ("gpt-oss-120b", "120b"), ("qwen3.8-27b", "Qwen")]


def llm(cell, model, mode="single", nshot=0):
    e = M.get(cell)
    if not e:
        return None
    for k, v in e["arms"].items():
        m = v["meta"]
        if m["arm"] == "llm" and m["model"] == model and m["mode"] == mode and m["nshot"] == nshot:
            return v
    return None


def arm(cell, key):
    e = M.get(cell)
    return e["arms"].get(key) if e else None


def best_of(cell, keys):
    cands = [(k, arm(cell, k)) for k in keys if arm(cell, k) and not math.isnan(arm(cell, k)["mean"])]
    if not cands:
        return None, None
    k, v = max(cands, key=lambda kv: kv[1]["mean"])
    return k, v


# ------------------------------------------------------------------------------------------------------------
# main master table
# ------------------------------------------------------------------------------------------------------------
GROUPS = [
    # label, reference cell, learner/LLM cell, M
    ("MAgent battle, base4", "battle-base4-pool8-semantic-single", "battle-base4-pool8-semantic-single"),
    ("MAgent battle, traits8", "battle-traits8-pool8-semantic-single", "battle-traits8-pool8-semantic-single"),
    ("combined arms, base4", "combined-base4-once", "combined-base4-semantic"),
    ("combined arms, traits8", "combined-traits8-once", "combined-traits8-semantic"),
    ("combined arms, base4, atten.", "combined-base4-delta06-once", "combined-base4-delta06-semantic"),
    ("sandbox $M{=}3$, $s{=}0.8$, $\\sigma{=}0$", "sandbox-M3-K3-sharp0.8-noise0-nonllm", "sandbox-M3-K3-sharp0.8-noise0-semantic"),
    ("sandbox $M{=}3$, $s{=}0.6$, $\\sigma{=}0.5$", "sandbox-M3-K3-sharp0.6-noise0.5-nonllm", "sandbox-M3-K3-sharp0.6-noise0.5-semantic"),
    ("sandbox $M{=}6$, $s{=}0.8$, $\\sigma{=}0$", "sandbox-M6-K3-sharp0.8-noise0-nonllm", "sandbox-M6-K3-sharp0.8-noise0-semantic"),
    ("sandbox $M{=}6$, $s{=}0.6$, $\\sigma{=}0.5$", "sandbox-M6-K3-sharp0.6-noise0.5-nonllm", "sandbox-M6-K3-sharp0.6-noise0.5-semantic"),
]
LIB = ["plastic", "fewshot__5", "fewshot__1", "fewshot__3", "fewshot__20"]
BANDITS = ["linucb", "lints", "ctxucb", "mucb"]


def pays(v, strongest_diff):
    """The pre-registered criterion: margin lower bound > 0.5 and paired difference vs strongest non-LLM lower bound > 0."""
    if v is None:
        return "--"
    ok1 = v["margin_lo95"] > 0.5
    ok2 = strongest_diff is not None and strongest_diff["lo95"] > 0
    return "yes" if ok1 and ok2 else ("fixed only" if ok1 else "no")


def short_label(label):
    """(first line, second line) of a master-table cell label: domain, then variant."""
    out = (label.replace("MAgent battle", "battle")
                .replace("combined arms", "combined"))
    if out.startswith("sandbox $M{=}"):
        return "sandbox", out.split("$, ", 1)[1]
    head, _, tail = out.partition(", ")
    return head, tail


def best_nshot(cell):
    best = None
    for e in M.get(cell, {}).get("arms", {}).values():
        m = e["meta"]
        if m["arm"] == "llm" and m.get("nshot", 0) > 0:
            if best is None or e["capture"] > best["capture"]:
                best = e
    return best


def strongest_zero_shot(cell):
    cands = [llm(cell, m) for m, _ in MODELS]
    cands = [v for v in cands if v]
    return max(cands, key=lambda v: v["capture"]) if cands else None


def model_tag(v):
    if not v:
        return "--"
    m = v["meta"]
    return {"gpt-oss-20b": "20b", "gpt-oss-120b": "120b", "qwen3.8-27b": "Qwen"}.get(m.get("model"), m.get("model", "--"))


def nshot_tag(v):
    if not v:
        return "--"
    m = v["meta"]
    return f"{model_tag(v)} $N{{=}}{m.get('nshot')}$"


def twomain(main, sub=""):
    sub = sub or r"\phantom{--}"
    return r"\begin{tabular}[t]{@{}c@{}}" + main + r"\\{\color{black!60}" + sub + r"}\end{tabular}"


def twolabel(main, sub=""):
    sub = sub or r"\phantom{--}"
    return r"\begin{tabular}[t]{@{}l@{}}" + main + r"\\" + sub + r"\end{tabular}"


rows = []
for idx, (label, ref, cell) in enumerate(GROUPS):
    e = M[ref]
    r = e["reference"]
    Mcount = e["meta"]["M"]
    to = arm(ref, "type_oracle")
    sc = arm(ref, "scripted")
    ppo = arm(ref, "ppo")
    # learners live in the learner cell (real games) or in the reference cell (sandbox)
    lcell = cell if any(arm(cell, k) for k in BANDITS) else ref
    bk, bv = best_of(lcell, BANDITS)
    lkk, lv = best_of(lcell, LIB)
    l20, l120, lq = (llm(cell, m) for m, _ in MODELS)
    best_n = best_nshot(cell)
    unc = EX["A3_uncertainty"]["cells"].get(ref, {})
    hd_ci = (unc.get("deployment") or {}).get("H_D_ci")
    judge = strongest_zero_shot(cell)
    vs = judge.get("vs_strongest") if judge else None
    strongest = e.get("strongest_nonllm") or M[cell].get("strongest_nonllm")
    def cap(v):
        return "--" if v is None else f"${f(v['capture'])}$"
    bandit_name = {"linucb": "LinUCB", "lints": "LinTS", "ctxucb": "CtxUCB", "mucb": "M-UCB"}.get(bk, bk)
    lib_name = {"plastic": "PLASTIC", "fewshot__5": "few-shot 5", "fewshot__1": "few-shot 1", "fewshot__3": "few-shot 3", "fewshot__20": "few-shot 20"}.get(lkk, lkk)
    if idx in (2, 5):
        rows.append(r"\midrule")
    row = " & ".join([
        twolabel(*short_label(label)),
        twomain(str(Mcount)),
        twomain(f"${f(r['V_star'])}$"),
        twomain(f"${f(r['H_D'])}$", grey_ci(hd_ci[0], hd_ci[1]) if hd_ci else "--"),
        twomain(cap(to)),
        twomain(cap(sc)),
        twomain(cap(bv), tex(bandit_name)),
        twomain(cap(lv), tex(lib_name)),
        twomain(cap(l20)),
        twomain(cap(l120)),
        twomain(cap(lq)),
        twomain(cap(best_n), nshot_tag(best_n)) if best_n else twomain("--"),
        twomain(f"${f(vs['diff'], plus=True)}$" if vs else "--", grey_ci(vs['ci'][0], vs['ci'][1], plus=True) if vs else "--"),
        twomain(pays(judge, vs) if judge else "--"),
    ])
    rows.append(row + " \\\\")
    NUM[f"master:{label}"] = {"M": Mcount, "V_star": r["V_star"], "best_fixed": r["best_fixed"], "H_D": r["H_D"], "H_D_ci": hd_ci, "H_type": r["H_type"], "share": r["share"],
                              "type_oracle": to["capture"] if to else None, "scripted": sc["capture"] if sc else None,
                              "best_bandit": (bk, bv["capture"] if bv else None), "best_library": (lkk, lv["capture"] if lv else None),
                              "ppo": ppo["capture"] if ppo else None,
                              "llm": {m: {"capture": (llm(cell, m) or {}).get("capture"), "capture_ci": (llm(cell, m) or {}).get("capture_ci"),
                                          "margin": (llm(cell, m) or {}).get("margin"), "margin_ci": (llm(cell, m) or {}).get("margin_ci"), "margin_lo95": (llm(cell, m) or {}).get("margin_lo95"),
                                          "vs_strongest": (llm(cell, m) or {}).get("vs_strongest"), "vs_strongest_arm": (llm(cell, m) or {}).get("vs_strongest_arm"),
                                          "fallback": (llm(cell, m) or {}).get("fallback_rate")} for m, _ in MODELS},
                              "best_nshot": {"meta": best_n["meta"], "capture": best_n["capture"], "capture_ci": best_n["capture_ci"], "margin_lo95": best_n["margin_lo95"], "vs_strongest": best_n.get("vs_strongest")} if best_n else None,
                              "zero_shot_judge": {"meta": judge["meta"], "capture": judge["capture"], "vs_strongest": vs} if judge else None}

header = r"""\begin{table*}[t]
\centering
\caption{The master table: each slot occupant's capture in the first-moves cells on the headline seeds $0$--$99$, four
of the eight sandbox cells among them.}
\label{tab:master}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\newcommand{\hd}[2]{\begin{tabular}[b]{@{}c@{}}#1\\#2\end{tabular}}
\begin{tabular}{@{}lccccccccccccc@{}}
\toprule
\textbf{Cell} & $M$ & $\Vstar$ & $\HD$ & \hd{type}{oracle} & scripted & \hd{best}{bandit} & \hd{best}{library} &
\texttt{20b} & \texttt{120b} & \texttt{Qwen} & \hd{best}{$N$-shot} & \hd{LLM$-$}{non-LLM} & \hd{pays}{(reward)} \\
\midrule
"""
body = "\n".join(rows)
footer = r"""
\bottomrule
\end{tabular}
\par\vspace{2pt}
\parbox{\textwidth}{\scriptsize Grey: $95\%$ bootstrap intervals and the arm behind each best value. \texttt{20b},
\texttt{120b}, \texttt{Qwen}: zero-shot \texttt{gpt-oss-20b}, \texttt{gpt-oss-120b} and \texttt{Qwen3.8-27B} with the
semantic encoding. Best $N$-shot: the strongest arm with labelled episodes in the prompt. LLM$-$non-LLM: the paired
difference of the strongest zero-shot model against the strongest tested non-LLM arm; pays: the pre-registered reward
criterion, \emph{fixed only} when the model clears $\Vstar$ but not that arm. Atten.: counter execution attenuated.}
\end{table*}
"""
(OUT / "tab_master.tex").write_text(header + body + footer, encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------
# sandbox crossover table (main text): per sandbox cell, predicted scale, 120b capture, T* of each learner
# ------------------------------------------------------------------------------------------------------------
SB = [c for c in M if c.startswith("sandbox-") and c.endswith("-nonllm")]
def cell_label(c):
    m = M[c]["meta"]
    return f"$M{{=}}{m['M']}$, $s{{=}}{m['sharpness']}$, $\\sigma{{=}}{m['reward_noise']:g}$"
def tstar_str(t, key="T_star", tmax_key="T_max"):
    if t is None:
        return "--"
    val = t.get(key)
    if val is None:
        return f"$>{t.get(tmax_key, 100)}$"
    return f"${val}$"

lines = [r"""\begin{table}[ht]
\centering
\caption{Sandbox: the identity against its Monte Carlo reference, and the crossover horizons.}
\label{tab:sandbox}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
\textbf{Cell} & $\widehat\HD$ & pop. & $\frac{MK\sigma^2}{\Delta^2}$ & $\capf_{\texttt{120b}}$ & LinTS & LinUCB & CtxUCB & M-UCB \\
\midrule"""]
for c in sorted(SB, key=lambda c: (M[c]["meta"]["M"], -M[c]["meta"]["sharpness"], M[c]["meta"]["reward_noise"])):
    e = M[c]
    sib = c.replace("-nonllm", "-semantic")
    v = llm(sib, "gpt-oss-120b")
    cross = (v or {}).get("crossover", {})
    assert abs(e["reference"]["H_D"] - e["closed_headline"]["H_D"]) < 0.005, c
    lines.append(" & ".join([cell_label(c), f(e["reference"]["H_D"]), f(e["closed"]["H_D"]),
                             f(e["closed"]["predicted_T_star"], 1), ci(v) if v else "--",
                             tstar_str(cross.get("lints")), tstar_str(cross.get("linucb")), tstar_str(cross.get("ctxucb")), tstar_str(cross.get("mucb"))]) + " \\\\")
    NUM[f"sandbox:{c}"] = {"H_D_probe": e["reference"]["H_D"], "H_D_closed_headline": e["closed_headline"]["H_D"], "H_D_closed_pop": e["closed"]["H_D"],
                           "V_star_probe": e["reference"]["V_star"], "V_star_closed_headline": e["closed_headline"]["V_star"],
                           "pred_T": e["closed"]["predicted_T_star"], "snr": e["closed"]["snr"], "Delta_min": e["closed"]["Delta_min"], "sigma": e["closed"]["sigma_eff"],
                           "llm120": {"capture": v["capture"], "ci": v["capture_ci"], "margin_lo95": v["margin_lo95"], "vs_strongest": v.get("vs_strongest"), "vs_strongest_arm": v.get("vs_strongest_arm")} if v else None,
                           "llm20": (lambda w: {"capture": w["capture"], "ci": w["capture_ci"]} if w else None)(llm(sib, "gpt-oss-20b")),
                           "qwen": (lambda w: {"capture": w["capture"], "ci": w["capture_ci"]} if w else None)(llm(sib, "qwen3.8-27b")),
                           "crossover_vs_120b": {k: {"T_star": t.get("T_star"), "lo": t.get("T_star_at_llm_hi"), "hi": t.get("T_star_at_llm_lo"), "pop": t.get("T_star_population")} for k, t in cross.items()},
                           "learners": {k: {"capture": a["capture"], "ci": a["capture_ci"], "T_50": a.get("T_50"), "T_80": a.get("T_80"), "T_50_pop": a.get("T_50_population"), "T_80_pop": a.get("T_80_population")}
                                        for k, a in e["arms"].items() if "curve_paired" in a},
                           "scripted": e["arms"]["scripted"]["capture"], "type_oracle": e["arms"]["type_oracle"]["capture"]}
lines.append(r"""\bottomrule
\end{tabular}
\par\vspace{2pt}
\parbox{\linewidth}{\scriptsize $\widehat\HD$: the probe's headroom on the headline seeds; pop.: the population value
over $20{,}000$ seeds; $\frac{MK\sigma^2}{\Delta^2}$: the sample-complexity scale
$MK\sigma_{\mathrm{eff}}^2/\Delta_{\min}^2$; $\capf_{\texttt{120b}}$: the zero-shot capture of \texttt{gpt-oss-120b};
learner columns: the first episode after which the learner's cumulative capture stays at or above it, $>100$ when that
has not happened by the end.}
\end{table}""")
(OUT / "tab_sandbox.tex").write_text("\n".join(lines), encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------
# supplement: full per-cell tables
# ------------------------------------------------------------------------------------------------------------
SCR = r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.08}"
MODEL_LONG = {"gpt-oss-20b": r"\texttt{gpt-oss-20b}", "gpt-oss-120b": r"\texttt{gpt-oss-120b}", "qwen3.8-27b": r"\texttt{Qwen3.8-27B}"}
MODE_LONG = {"single": "", "hypothesis_first": "hypothesis-first", "deliberate": "deliberate"}
ARM_LONG = {"best_fixed": "best fixed", "linucb": "LinUCB", "lints": "LinTS", "ctxucb": "CtxUCB", "mucb": "M-UCB",
            "scripted": "scripted", "plastic": "PLASTIC", "ppo": "PPO", "random": "random",
            "fewshot__1": "few-shot 1", "fewshot__3": "few-shot 3", "fewshot__5": "few-shot 5", "fewshot__20": "few-shot 20",
            "linucb_signature": "LinUCB", "lints_signature": "LinTS", "ctxucb_signature": "CtxUCB",
            "tabular_ucb1_signature": "tabular UCB1", "library_lookup_signature": "library lookup"}
ENC_LONG = {"semantic": "", "numeric": "numeric", "opaque": "opaque", "permuted": "permuted", "relabel": "relabel",
            "swapdesc": "swapdesc", "nonllm": "", "once": "", "single": ""}


def readable_cell(c, with_enc=True):
    """Readable cell name: domain, types, and any encoding other than semantic."""
    if c.startswith("sandbox-"):
        m = _re.match(r"sandbox-M(\d+)-K\d+-sharp([0-9.]+)-noise([0-9.]+)-(\w+)", c)
        Mv, s, n, enc = m.groups()
        base = f"sandbox, $M{{=}}{Mv}$, $s{{=}}{s}$, $\\sigma{{=}}{float(n):g}$"
    else:
        parts = c.split("-")
        dom = "battle" if parts[0] == "battle" else "combined"
        types = parts[1]
        base = f"{dom}, {types}"
        if "delta06" in c:
            base += ", atten."
        if "heldout" in c:
            base += ", held-out " + tex(c.split("heldout-")[1])
        enc = parts[-1] if parts[-1] in ENC_LONG else ""
        if c.endswith("-single"):
            enc = parts[3] if len(parts) > 3 else ""
    extra = ENC_LONG.get(enc, "") if with_enc else ""
    return base + (f", {extra}" if extra else "")


def arm_label(key):
    """llm__gpt-oss-120b__single__nshot0 -> gpt-oss-120b, zero-shot; learners by name."""
    if key.startswith("llm__"):
        _, model, mode, nshot = key.split("__")
        n = int(nshot.replace("nshot", "").replace("-", "") or 0)
        bits = [MODEL_LONG.get(model, tex(model))]
        if MODE_LONG.get(mode):
            bits.append(MODE_LONG[mode])
        bits.append("zero-shot" if n == 0 else f"$N{{=}}{n}$")
        return ", ".join(bits)
    return ARM_LONG.get(key, tex(key))


def ci2(value, interval, plus=False):
    return ci_raw(value, interval, d=2, plus=plus)

TYPE_PART = {"hold": "h", "push": "p", "screen": "sc", "standoff": "st", "melee_first": "m", "ranged_first": "r",
             "tight": "t", "spread": "s", "focus": "f", "spray": "y"}
TYPE_BASE = {"MELEE_RUSH": "MR", "RANGED_STANDOFF": "RS", "MIXED_ADVANCE": "MA", "KITE": "KITE", "RUSH": "RUSH",
             "TURTLE": "TURTLE", "FLANK": "FLANK", "PAPER_LOVER": "PL", "ROCK_LOVER": "RL", "SCISSORS_LOVER": "SL",
             "BEST_RESPONDER": "BR", "FLIPPER_L": "FL", "GULLIBLE": "GU"}


def abbr_note(t):
    a = type_abbr(t)
    return "" if a == tex(t) else f" ({a})"


def type_abbr(t):
    if "-" in t:
        return "-".join(TYPE_PART.get(x, x) for x in t.split("-"))
    return TYPE_BASE.get(t, tex(t))


def arm_name(meta):
    a = meta["arm"]
    if a == "fixed":
        return "fixed:" + tex(meta["tau"])
    if a == "fewshot":
        return f"few-shot {meta['N']}"
    if a == "llm":
        return f"\\texttt{{{tex(meta['model'])}}} {meta['mode'].replace('_', '-')} $N{{=}}{meta['nshot']}$"
    return {"linucb": "LinUCB", "lints": "LinTS", "ctxucb": "CtxUCB", "mucb": "M-UCB", "plastic": "PLASTIC", "ppo": "PPO", "type_oracle": "type oracle",
            "scripted": "scripted", "random": "random"}.get(a, tex(a))

# one body convention for every generated table: \scriptsize, 3pt column separation, rows at 1.05
SMALL = r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}"
TINY = r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}"
ENDG = r"\endgroup"
RAG = r">{\raggedright\arraybackslash}"

full = [r"\section*{Every cell, every arm}"]
for c, e in M.items():
    r = e["reference"]
    hdi = ((EX["A3_uncertainty"]["cells"].get(c) or {}).get("deployment") or {}).get("H_D_ci")
    full.append(f"\\subsection*{{{readable_cell(c)}}}")
    full.append(f"$\\Vstar={f(r['V_star'])}$ ({tex(r['best_fixed'])}), $V_{{\\mathrm{{default}}}}={f(r['V_default'])}$, $\\HD={f(r['H_D'])}$" +
                (f" $[{f(hdi[0])},{f(hdi[1])}]$" if hdi else "") +
                f", type-oracle headroom ${f(r['H_type'])}$, share ${f(r['share'])}$; types: " +
                ", ".join(f"{tex(t)} {n}" for t, n in r["type_shares"].items()) + ".")
    full.append(SMALL)
    full.append(r"\begin{longtable}{@{}lrrrrr@{}}\toprule arm & $n$ & mean & margin vs $\Vstar$ & capture & fallback \\ \midrule \endhead")
    for k, v in e["arms"].items():
        if v["n"] == 0:
            full.append(f"{arm_name(v['meta'])} & 0 & -- & -- & -- & -- \\\\")
            continue
        full.append(f"{arm_name(v['meta'])} & {v['n']} & {f(v['mean'])} & {ci(v, 'margin')} & {ci(v)} & {f(v['fallback_rate'])} \\\\")
    full.append(r"\bottomrule\end{longtable}" + ENDG)
(OUT / "si_master_full.tex").write_text("\n".join(full), encoding="utf-8")

# per-type misreading tables for the semantic real-game cells and one sandbox cell
pt = [r"\section*{Per-type readings}"]
for c in ["battle-base4-pool8-semantic-single", "combined-base4-semantic", "combined-traits8-semantic", "sandbox-M3-K3-sharp0.8-noise0-semantic"]:
    e = M[c]
    ref = M[e["meta"]["ref"]]["reference"]
    types = list(ref["per_type"].keys())
    pt.append(f"\\subsection*{{{readable_cell(c)}}}")
    pt.append("Best counter and gap per type: " + "; ".join(f"{tex(t)}{abbr_note(t)}: {tex(i['best_tau'])} ({f(i['best_value'])}), $\\Delta={f(i['Delta'])}$, share {f(i['share'])}" for t, i in ref["per_type"].items()) + ".")
    pt.append(TINY)
    pt.append(r"\begin{longtable}{@{}l" + "r" * len(types) + r"r@{}}\toprule arm & " + " & ".join(type_abbr(t) for t in types) + r" & $\mathbb{E}[\rho]$ \\ \midrule \endhead")
    pt.append(r"\multicolumn{" + str(len(types) + 2) + r"}{@{}l@{}}{\emph{Entries are $p_{\mathrm{mis}}$/loss.}}\\")
    for k, v in e["arms"].items():
        if v["meta"]["arm"] in ("fixed", "random") or v["n"] == 0:
            continue
        cells = " & ".join(f"{f(v['per_type'][t]['p_mis'])}/{f(v['per_type'][t]['cost'], 1)}" if t in v["per_type"] else "--" for t in types)
        pt.append(f"{arm_name(v['meta'])} & {cells} & {f(v['expected_regret_pertype'])} \\\\")
    pt.append(r"\bottomrule\end{longtable}" + ENDG)
(OUT / "si_pertype.tex").write_text("\n".join(pt), encoding="utf-8")

# encoding table (LLM single N=0 by encoding, plus LinUCB/PLASTIC)
enc = [r"\section*{Encodings}", SMALL,
       r"\begin{longtable}{@{}" + RAG + r"p{0.22\linewidth}lrrrrr@{}}\toprule group & encoding & \texttt{20b} & \texttt{120b} & \texttt{Qwen} & LinUCB & PLASTIC \\ \midrule \endhead"]
for c, e in M.items():
    if e["meta"].get("held_out") or c.endswith("-once") or c.endswith("-nonllm"):
        continue
    if not any(v["meta"]["arm"] == "llm" for v in e["arms"].values()):
        continue
    l = {m: llm(c, m) for m, _ in MODELS}
    lu, pl = arm(c, "linucb"), arm(c, "plastic")
    plastic_cell = "--" if c in ("combined-base4-opaque", "combined-traits8-opaque") else (f(pl['capture']) if pl and pl['n'] else "--")
    enc.append(f"{readable_cell(c, with_enc=False)} & {e['meta']['encoding']} & " + " & ".join(ci(l[m]) if l[m] else "--" for m, _ in MODELS) +
               f" & {f(lu['capture']) if lu and lu['n'] else '--'} & {plastic_cell} \\\\")
    NUM[f"encoding:{c}"] = {m: {"capture": l[m]["capture"], "ci": l[m]["capture_ci"], "fallback": l[m]["fallback_rate"]} if l[m] else None for m, _ in MODELS}
enc.append(r"\bottomrule\end{longtable}" + ENDG)
(OUT / "si_encoding.tex").write_text("\n".join(enc), encoding="utf-8")

# n-shot / mode table
ns = [r"\section*{In-context examples and prompt form (semantic encoding)}", TINY,
      r"\begin{longtable}{@{}" + RAG + r"p{0.21\linewidth}llrrp{0.26\linewidth}r@{}}\toprule cell & model & mode & $N$ & capture & vs strongest non-LLM & fallback \\ \midrule \endhead"]
for c, e in M.items():
    if e["meta"]["encoding"] != "semantic" or e["meta"].get("held_out"):
        continue
    for k, v in e["arms"].items():
        m = v["meta"]
        if m["arm"] != "llm":
            continue
        vs = v.get("vs_strongest")
        ns.append(f"{readable_cell(c)} & \\texttt{{{tex(m['model'])}}} & {m['mode'].replace('_', '-')} & {m['nshot']} & {ci(v)} & " +
                  (f"${f(vs['diff'])}\\,[{f(vs['ci'][0])},{f(vs['ci'][1])}]$ ({tex(v['vs_strongest_arm'])})" if vs else "--") + f" & {f(v['fallback_rate'])} \\\\")
        NUM[f"nshot:{c}:{m['model']}:{m['mode']}:{m['nshot']}"] = {"capture": v["capture"], "ci": v["capture_ci"], "margin": v["margin"], "margin_lo95": v["margin_lo95"], "vs": vs, "vs_arm": v.get("vs_strongest_arm"), "mean": v["mean"]}
ns.append(r"\bottomrule\end{longtable}" + ENDG)
(OUT / "si_nshot.tex").write_text("\n".join(ns), encoding="utf-8")

# crossover table (all LLM arms, all learners)
cr = [r"\section*{Crossover horizons}",
      r"Each entry reads \emph{learner}: paired horizon, and in the sandbox a second horizon on the population scale; $>T$ means the learner had not caught the model by episode $T$.",
      TINY,
      r"\begin{longtable}{@{}" + RAG + r"p{0.20\linewidth}" + RAG + r"p{0.14\linewidth}r" + RAG + r"p{0.44\linewidth}@{}}\toprule cell & LLM arm & capture & $\Tstar$ per learner \\ \midrule \endhead"]
for c, e in M.items():
    for k, v in e["arms"].items():
        if v["meta"]["arm"] != "llm" or not v.get("crossover"):
            continue
        lc = M[v.get("crossover_learner_cell", c)]["arms"]
        parts = []
        for lk, t in v["crossover"].items():
            nm = arm_name(lc[lk]["meta"])
            s = f"{nm}: {t['T_star'] if t['T_star'] is not None else '>' + str(t['T_max'])}"
            if "T_star_population" in t:
                s += f" / {t['T_star_population'] if t['T_star_population'] is not None else '>' + str(t['T_max_population'])}"
            parts.append(s)
        cr.append(f"{readable_cell(c)} & {arm_name(v['meta'])} & {f(v['capture'])} & {'; '.join(parts)} \\\\")
cr.append(r"\bottomrule\end{longtable}" + ENDG)
(OUT / "si_crossover.tex").write_text("\n".join(cr), encoding="utf-8")

# held-out table
ho = [r"\section*{Held-out types}", SMALL,
      r"\begin{longtable}{@{}" + RAG + r"p{0.17\linewidth}" + RAG + r"p{0.14\linewidth}rr" + RAG + r"p{0.17\linewidth}rr@{}}\toprule held-out type & arm & unseen & seen & best counter & $p_{\mathrm{mis}}$ unseen & $p_{\mathrm{mis}}$ seen \\ \midrule \endhead"]
for c, h in H.items():
    for k, r in h["arms"].items():
        ho.append(f"{tex(h['held_out'])} ({f(h['share'])}) & {arm_name(M[c]['arms'][k]['meta'])} & {f(r['unseen_mean'])} & {f(r['seen_mean'])} & {f(r['best_counter_value'])} ({tex(r['best_counter'])}) & {f(r['unseen_p_mis'])} & {f(r['seen_p_mis'])} \\\\")
        NUM[f"heldout:{h['held_out']}:{k}"] = r
ho.append(r"\bottomrule\end{longtable}" + ENDG)
(OUT / "si_heldout.tex").write_text("\n".join(ho), encoding="utf-8")


# calibration/deployment readings, inference cost, cost-adjusted margins, verdict robustness, opaque controls
# (readable labels throughout; widths budgeted for the one-column supplement at \scriptsize)
import re as _re

# --- calibration and deployment -------------------------------------------------------------------------------
base_calib = ["battle-base4-pool8-semantic-single", "battle-traits8-pool8-semantic-single", "combined-base4-once",
              "combined-traits8-once", "combined-base4-delta06-once"]
sandbox_calib = sorted([c for c in EX["A3_uncertainty"]["cells"] if c.startswith("sandbox-")],
                       key=lambda c: (M[c]["meta"]["M"], -M[c]["meta"]["sharpness"], M[c]["meta"]["reward_noise"]))
heldout_calib = sorted([c for c in EX["A3_uncertainty"]["cells"] if "heldout" in c])
cal = [SCR,
       r"\begin{longtable}{@{}" + RAG + r"p{0.16\linewidth}rrrrrr@{}}\toprule cell & $\Vstar$ dep. & $\Vstar$ cal. & "
       r"$\HD$ dep. & $\HD$ cal. & $\HD$ dep.$-$cal. & $\HD^{\mathrm{ps}}$ dep./cal. \\ \midrule \endhead"]
for c in base_calib + sandbox_calib + heldout_calib:
    item = EX["A3_uncertainty"]["cells"][c]
    dep, ca, diff = item["deployment"], item["calibration"], item["deployment_minus_calibration"]
    label = readable_cell(c)
    if "heldout" in c:
        label += r"$^{\dagger}$"
    cal.append(" & ".join([label, point_ci(dep, "V_star"), point_ci(ca, "V_star"), point_ci(dep, "H_D"),
                           point_ci(ca, "H_D"), point_ci(diff, "H_D"),
                           f"${f(dep.get('H_D_ps'))}$ / ${f(ca.get('H_D_ps'))}$"]) + r" \\")
    NUM[f"A3:{c}"] = item
cal.append(r"\bottomrule\end{longtable}" + ENDG)
cal.append(r"{\scriptsize dep.: deployment seeds $0$--$99$; cal.: calibration seeds; ps: post-stratified to equal type shares. $^{\dagger}$The calibration of a held-out cell excludes one type by design.}")
cal.append("")
cal.append(SCR)
cal.append(r"\begin{longtable}{@{}" + RAG + r"p{0.16\linewidth}" + RAG + r"p{0.78\linewidth}@{}}\toprule cell & "
           r"type counts, calibration (deployment) \\ \midrule \endhead")
for c in ["battle-base4-pool8-semantic-single", "battle-traits8-pool8-semantic-single", "combined-base4-once",
          "combined-traits8-once"]:
    tc = EX["A3_uncertainty"]["cells"][c]["calibration"]["type_counts"]
    td = EX["A3_uncertainty"]["cells"][c]["deployment"]["type_counts"]
    cal.append(f"{readable_cell(c)} & " + "; ".join(f"{tex(k)} {v} ({td.get(k, 0)})" for k, v in tc.items()) + r" \\")
cal.append(r"\bottomrule\end{longtable}" + ENDG)
(OUT / "si_calib.tex").write_text("\n".join(cal), encoding="utf-8")


# --- inference cost -------------------------------------------------------------------------------------------
def file_summary(model, headline, build=None):
    vals = [v for v in EX["A1_inference_cost"]["llm_files"].values() if v["model"] == model
            and (not headline or (v["cell"].endswith("-semantic") and v["mode"] == "single" and v["nshot"] == 0))
            and (build is None or "+".join(v.get("system_fingerprint", [])) == build)]
    dec = sum(v["decisions"] for v in vals) or 1

    def uniq(key):
        out = []
        for v in vals:
            for x in v.get(key, []):
                if x not in out:
                    out.append(x)
        return out
    return {"fallback": sum(v["fallback_rate"] * v["decisions"] for v in vals) / dec,
            "fingerprint": uniq("system_fingerprint"), "temperature": uniq("temperature"),
            "max_tokens": uniq("max_tokens"), "extra_body": uniq("extra_body")}


cost = [r"\section*{Inference cost}", SCR,
        r"\begin{longtable}{@{}llrrrrrrr@{}}\toprule model & arms & decisions & calls & prompt tokens & "
        r"completion tokens & total tokens & latency (s) & fallback \\ & & & & med./p95 & med./p95 & med./p95 & "
        r"med./p95 & rate \\ \midrule \endhead"]
serving = {}
# the GPU behind each serving build: the recorded runs used one H800; the sandbox example arms were rerun after
# deployment on one RTX PRO 6000 (vLLM prints a different fingerprint for each model and build)
SERVING_GPU = {"vllm-0.28.0-71c30221": "H800", "vllm-0.28.0-7fec5b1f": "H800", "vllm-0.28.0-ac20b7d2": "H800",
               "vllm-0.28.0-fe5f1b4f": "RTX PRO 6000", "vllm-0.28.0-c2f2d62b": "RTX PRO 6000",
               "vllm-0.28.0-c41b102e": "RTX PRO 6000"}
AGG = EX["A1_inference_cost"]["aggregates"]
cost_rows = [("headline_semantic_single_nshot0_by_model", model, stats, "headline", True, None)
             for model, stats in AGG["headline_semantic_single_nshot0_by_model"].items()]
for key, stats in sorted(AGG["by_serving_build"].items(),
                         key=lambda kv: (kv[0].split("|")[0], SERVING_GPU.get(kv[0].split("|")[1], "~"))):
    model, build = key.split("|")
    gpu = SERVING_GPU.get(build, build)
    cost_rows.append((f"by_serving_build:{build}", model, stats, "all, H800" if gpu == "H800" else f"examples, {gpu}",
                      False, build))
for model in AGG["all_cells_by_model"]:
    serving[model] = file_summary(model, False)
    NUM[f"A1:all_cells_by_model:{model}"] = {"aggregate": AGG["all_cells_by_model"][model], **serving[model]}
for group_key, model, stats, group_name, headline, build in cost_rows:
    fs = file_summary(model, headline, build)
    tok = stats["tokens_per_decision"]
    lat = stats["latency_per_decision_s"]["latency_s"]
    cost.append(" & ".join([
        MODEL_LONG.get(model, tex(model)), group_name, str(stats["decisions"]), str(stats["calls"]),
        f"{tok['prompt']['median']:.0f}/{tok['prompt']['p95']:.0f}",
        f"{tok['completion']['median']:.0f}/{tok['completion']['p95']:.0f}",
        f"{tok['total']['median']:.0f}/{tok['total']['p95']:.0f}",
        f"{f(lat['median'])}/{f(lat['p95'])}", f(fs["fallback"], 3)]) + r" \\")
    NUM[f"A1:{group_key}:{model}"] = {"aggregate": stats, **fs}
cost.append(r"\bottomrule\end{longtable}" + ENDG)
cost.append(r"{\scriptsize headline: the zero-shot single-step semantic arms of the master table; all, H800: every "
            r"arm served on one H800 GPU; examples, RTX PRO 6000: the sandbox arms with labelled examples and their "
            r"zero-shot controls, served on one RTX PRO 6000 GPU. Latency depends on the GPU; tokens do not.}")
cost.append("")
cost.append(SCR)


def _uniq(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def server_words(fingerprints):
    """'vllm-0.28.0-<hash>' -> 'vLLM 0.28.0'; the per-build fingerprints are listed in the README"""
    return _uniq([f"vLLM {m.group(1)}" if (m := _re.match(r"vllm-(\d+(?:\.\d+)*)", x)) else tex(x) for x in fingerprints])


def reasoning_words(extra_bodies):
    """the recorded request extras, in plain words"""
    out = []
    for x in extra_bodies:
        try:
            d = json.loads(x) if isinstance(x, str) else x
        except ValueError:
            d = None
        if isinstance(d, dict) and "reasoning_effort" in d:
            out.append(f"{d['reasoning_effort']} reasoning effort")
        elif isinstance(d, dict) and "enable_thinking" in d.get("chat_template_kwargs", {}):
            out.append("thinking mode " + ("on" if d["chat_template_kwargs"]["enable_thinking"] else "off"))
        else:
            out.append(tex(str(x)))
    return _uniq(out)


cost.append(r"\begin{longtable}{@{}lll" + RAG + r"p{0.24\linewidth}ll@{}}\toprule model & server & reasoning setting & "
            r"GPU & temperature & max tokens \\ \midrule \endhead")
for model, fs in serving.items():
    gpus = _uniq([SERVING_GPU.get(x, "--") for x in fs["fingerprint"]])
    cost.append(" & ".join([MODEL_LONG.get(model, tex(model)), ", ".join(server_words(fs["fingerprint"])),
                            ", ".join(reasoning_words(fs["extra_body"])) or "--", ", ".join(gpus),
                            ", ".join(str(x) for x in fs["temperature"]), ", ".join(str(x) for x in fs["max_tokens"])])
                + r" \\")
cost.append(r"\bottomrule\end{longtable}" + ENDG)
mb = EX["A1_inference_cost"]["nonllm_microbench"]
cost.append("")
cost.append(SCR)
cost.append(r"\begin{longtable}{@{}lrr@{}}\toprule non-LLM selector & median ms per decision & p95 ms per decision \\ "
            r"\midrule \endhead")
for k, v in mb.items():
    if k == "platform":
        continue
    cost.append(f"{ARM_LONG.get(k, tex(k))} & {f(v['median_ms_per_select_update'], 3)} & "
                f"{f(v['p95_ms_per_select_update'], 3)} \\\\")
cost.append(r"\bottomrule\end{longtable}" + ENDG)
cost.append(r"{\scriptsize Decision time: one select-and-update operation on the stored feature vectors, on one core of "
            r"a desktop Intel CPU.}")
NUM["A1:nonllm_microbench"] = mb
vals_med = [v["median_ms_per_select_update"] for k, v in mb.items() if k != "platform"]
vals_p95 = [v["p95_ms_per_select_update"] for k, v in mb.items() if k != "platform"]
NUM["A1:nonllm_microbench_range"] = {"median_ms_min": min(vals_med), "median_ms_max": max(vals_med),
                                     "p95_ms_min": min(vals_p95), "p95_ms_max": max(vals_p95)}
(OUT / "si_cost.tex").write_text("\n".join(cost), encoding="utf-8")

# --- cost-adjusted margins ------------------------------------------------------------------------------------
breakeven = [r"\section*{Cost-adjusted margins}",
             r"$D$: paired lead in reward per episode at $T=100$; $t$: thousands of tokens per episode; "
             r"$\lambda^\star=D/t$ in reward per thousand tokens.", ""]
for family, pred in [("the best fixed strategy", lambda r: r["comparator"] == "best_fixed"),
                     ("the library-free learners", lambda r: r["comparator"] != "best_fixed")]:
    breakeven.append(f"\\subsection*{{Against {family}}}")
    breakeven.append(SCR)
    breakeven.append(r"\begin{longtable}{@{}" + RAG + r"p{0.21\linewidth}" + RAG + r"p{0.17\linewidth}lrrrr@{}}\toprule "
                     r"cell & language model & against & $D$ [CI] & $t$ & $\lambda^\star$ [CI] & reward/GPU-s \\ "
                     r"\midrule \endhead")
    for r0 in [r for r in EX["A2_cost_adjusted"]["comparisons"] if pred(r)]:
        breakeven.append(" & ".join([
            readable_cell(r0["cell"]), arm_label(r0["arm"]), ARM_LONG.get(r0["comparator"], tex(r0["comparator"])),
            ci2(r0["D"], r0["ci"]), f(r0["tokens_per_episode"] / 1000, 3),
            ci2(r0["lambda_star_reward_per_1k_tokens"], r0["lambda_star_ci"]), f(r0["reward_per_gpu_second"])]) + r" \\")
    breakeven.append(r"\bottomrule\end{longtable}" + ENDG)
NUM["A2:comparisons"] = EX["A2_cost_adjusted"]["comparisons"]
pos_fixed = [r["lambda_star_reward_per_1k_tokens"] for r in EX["A2_cost_adjusted"]["comparisons"]
             if r["comparator"] == "best_fixed" and r["D"] > 0]
pos_learn = [r["lambda_star_reward_per_1k_tokens"] for r in EX["A2_cost_adjusted"]["comparisons"]
             if r["comparator"] != "best_fixed" and r["D"] > 0]
NUM["A2:lambda_positive_ranges"] = {"best_fixed": {"min": min(pos_fixed), "max": max(pos_fixed)},
                                    "library_free_learners": {"min": min(pos_learn), "max": max(pos_learn)}}
(OUT / "si_breakeven.tex").write_text("\n".join(breakeven), encoding="utf-8")

# --- robustness of the verdict --------------------------------------------------------------------------------
non_ex = EX["A4_pays_robustness"]["non_excluded"]
robust_rows = [("never above" if round(r0["hi95"], 2) == 0 else "within noise", r0) for r0 in non_ex]
total = sum(EX["A4_pays_robustness"]["counts"].values())
robust_counts = {"excluded": total - len(non_ex),
                 "never above": sum(1 for cls, _ in robust_rows if cls == "never above"),
                 "within noise": sum(1 for cls, _ in robust_rows if cls == "within noise"), "total": total}
NUM["robust_excluded"] = robust_counts["excluded"]
NUM["robust_never_above"] = robust_counts["never above"]
NUM["robust_within_noise"] = robust_counts["within noise"]
NUM["robust_total"] = robust_counts["total"]
rob = [SCR,
       r"\begin{longtable}{@{}" + RAG + r"p{0.22\linewidth}" + RAG + r"p{0.25\linewidth}lrlr@{}}\toprule cell & "
       r"language model & class & difference [CI] & strongest non-LLM & captures \\ \midrule \endhead"]
for cls, r0 in robust_rows:
    rob.append(" & ".join([readable_cell(r0["cell"]), arm_label(r0["arm"]), cls,
                           ci2(r0["diff"], r0["ci"]), ARM_LONG.get(r0["strongest_arm"], tex(r0["strongest_arm"])),
                           f"{f(r0['llm_capture'])} / {f(r0['strongest_capture'])}"]) + r" \\")
rob.append(r"\bottomrule\end{longtable}" + ENDG)
(OUT / "si_robust.tex").write_text("\n".join(rob), encoding="utf-8")


# --- opaque controls ------------------------------------------------------------------------------------------
def cap_cell(v):
    return ci_raw(v.get("capture"), v.get("capture_ci"), d=2) if v else "--"


def cap_point(v):
    return f"${f(v['capture'])}$" if v else "--"


opaque = [r"\section*{Information-matched controls under the opaque encoding}",
          r"Captures at $T=100$ on the headline seeds. In combined arms the signature is one-to-one with the type; the "
          r"token-only arms read it alone, the feature learners read it as a one-hot prefix to their features, and "
          r"the library learners fitted on the plain features are not applicable.", "", SCR,
          r"\begin{longtable}{@{}llrr@{}}\toprule input & selector & base4, $M{=}4$ & traits8, $M{=}8$ \\ "
          r"\midrule \endhead"]
b4 = EX["A5_opaque_controls"]["controls"]["combined-base4-opaque"]
t8 = EX["A5_opaque_controls"]["controls"]["combined-traits8-opaque"]
for c, block in (("combined-base4-opaque", b4), ("combined-traits8-opaque", t8)):
    for k, v in block["one_hot_controls"].items():
        NUM[f"A5:{c}:{k}"] = {"capture": v["capture"], "capture_ci": v["capture_ci"]}
    llm_caps = [v["capture"] for v in block["recorded"]["opaque_llm"].values()]
    NUM[f"A5:{c}:opaque_llm_capture_range"] = {"min": min(llm_caps), "max": max(llm_caps)}
for k in ["linucb_signature", "lints_signature", "ctxucb_signature", "tabular_ucb1_signature", "library_lookup_signature"]:
    opaque.append(f"token only & {ARM_LONG[k]} & {cap_cell(b4['one_hot_controls'].get(k))} & "
                  f"{cap_cell(t8['one_hot_controls'].get(k))} \\\\")
opaque.append(r"\midrule")
for k in ["linucb", "lints", "ctxucb"]:
    opaque.append(f"features and token & {ARM_LONG[k]} & {cap_cell(b4['recorded']['feature_learners'].get(k))} & "
                  f"{cap_cell(t8['recorded']['feature_learners'].get(k))} \\\\")
for k in ["plastic", "fewshot__5"]:
    opaque.append(f"features and token & {ARM_LONG[k]} & n/a & n/a \\\\")
opaque.append(r"\midrule")
for k in ["llm__gpt-oss-20b__single__nshot0", "llm__gpt-oss-120b__single__nshot0", "llm__qwen3.8-27b__single__nshot0"]:
    opaque.append(f"token and descriptions & {arm_label(k)} & {cap_cell(b4['recorded']['opaque_llm'].get(k))} & "
                  f"{cap_cell(t8['recorded']['opaque_llm'].get(k))} \\\\")
opaque.append(r"\bottomrule\end{longtable}" + ENDG)
opaque.append("")
opaque.append(r"In the sandbox the signature is the same for every type, so every input below carries no type "
              r"information; point captures at $T=100$.")
opaque.append("")
opaque.append(SCR)
opaque.append(r"\begin{longtable}{@{}" + RAG + r"p{0.25\linewidth}rrrrrr@{}}\toprule cell & LinUCB & LinTS & CtxUCB & "
              r"tabular UCB1 & library lookup & language models \\ \midrule \endhead")
for c in sorted([x for x in EX["A5_opaque_controls"]["controls"] if x.startswith("sandbox-")],
                key=lambda x: (M[x]["meta"]["M"], -M[x]["meta"]["sharpness"], M[x]["meta"]["reward_noise"])):
    block = EX["A5_opaque_controls"]["controls"][c]
    ctr = block["one_hot_controls"]
    llm_caps = [v["capture"] for v in block["recorded"]["opaque_llm"].values()]
    opaque.append(" & ".join([readable_cell(c, with_enc=False)] +
                             [cap_point(ctr.get(k)) for k in ["linucb_signature", "lints_signature", "ctxucb_signature",
                                                              "tabular_ucb1_signature", "library_lookup_signature"]] +
                             [f"${f(min(llm_caps))}$ to ${f(max(llm_caps))}$"]) + r" \\")
opaque.append(r"\bottomrule\end{longtable}" + ENDG)
(OUT / "si_opaque.tex").write_text("\n".join(opaque), encoding="utf-8")


# --- checks run after deployment (analysis/checks.json) -------------------------------------------------------
CK = json.load(open(ROOT / "analysis" / "checks.json", encoding="utf-8"))
FRESH_LABEL = {"battle-base4": "battle-base4-pool8-semantic-single", "battle-traits8": "battle-traits8-pool8-semantic-single",
               "combined-base4": "combined-base4-once", "combined-traits8": "combined-traits8-once",
               "combined-base4-delta06": "combined-base4-delta06-once"}
fr = [SCR,
      r"\begin{longtable}{@{}" + RAG + r"p{0.17\linewidth}rrrrr@{}}\toprule cell & $\Vstar$ fresh & $\HD$ fresh & "
      r"$\HD$ dep. & $\HD$ cal. & $\HD^{\mathrm{ps}}$ fresh/dep./cal. \\ \midrule \endhead"]
for cell, ref in FRESH_LABEL.items():
    r0 = CK["fresh_seeds"][cell]
    dep, ca = r0["deployment"], r0["calibration"]
    fr.append(" & ".join([readable_cell(ref), ci2(r0["V_star"], r0["V_star_ci"]), ci2(r0["H_D"], r0["H_D_ci"]),
                          ci2(dep["H_D"], dep["H_D_ci"]), f"${f(ca['H_D'])}$",
                          f"${f(r0['H_D_ps'])}$ / ${f(dep['H_D_ps'])}$ / ${f(ca['H_D_ps'])}$"]) + r" \\")
    NUM[f"fresh:{cell}"] = r0
fr.append(r"\bottomrule\end{longtable}" + ENDG)
fr.append(r"{\scriptsize fresh: $500$ seeds, $5000$--$5499$, used by neither calibration nor deployment; dep.: deployment seeds $0$--$99$; "
          r"cal.: calibration seeds $1000$--$1039$; ps: post-stratified to equal type shares.}")
(OUT / "si_fresh.tex").write_text("\n".join(fr), encoding="utf-8")

exs = [SCR,
       r"\begin{longtable}{@{}" + RAG + r"p{0.2\linewidth}lrrrrrr@{}}\toprule cell & model & \multicolumn{2}{c}{zero-shot} & "
       r"$N{=}1$ & $N{=}5$ & $N{=}1-N{=}0$ & $N{=}5-N{=}0$ \\ & & H800 & RTX & & & \multicolumn{2}{c}{paired, RTX} \\ "
       r"\midrule \endhead"]
def _sandbox_order(c):
    m = _re.match(r"sandbox-M(\d+)-K\d+-sharp([0-9.]+)-noise([0-9.]+)", c)
    return int(m.group(1)), -float(m.group(2)), float(m.group(3))


MODEL_ORDER = {"gpt-oss-120b": 0, "gpt-oss-20b": 1, "qwen3.8-27b": 2}
for key, r0 in sorted(CK["sandbox_examples"].items(),
                      key=lambda kv: (_sandbox_order(kv[0].split("|")[0]), MODEL_ORDER[kv[0].split("|")[1]])):
    c, model = key.split("|")
    cp, dd = r0["capture"], r0["diff"]
    exs.append(" & ".join([readable_cell(c + "-semantic"), MODEL_LONG[model]] +
                          [f"${f(cp[k])}$" for k in ("N0_recorded", "N0_rerun", "N1", "N5")] +
                          [ci2(dd[k]["mean"], dd[k]["ci"]) for k in ("N1-N0_rerun", "N5-N0_rerun")]) + r" \\")
    NUM[f"examples:{key}"] = r0
exs.append(r"\bottomrule\end{longtable}" + ENDG)
exs.append(r"{\scriptsize Captures on seeds $0$--$99$. H800: the zero-shot arm served on one H800 GPU; RTX: the same arm "
           r"served on the RTX PRO 6000 that serves the example arms, a serving control. Differences are in units of $\HD$, paired by seed, "
           r"with $95\%$ bootstrap intervals.}")
(OUT / "si_examples.tex").write_text("\n".join(exs), encoding="utf-8")
same = [r0["diff"]["N0_rerun-N0_recorded"] for r0 in CK["sandbox_examples"].values()]
NUM["examples:rerun_vs_recorded_ci_contains_zero"] = sum(1 for d in same if d["ci"][0] <= 0 <= d["ci"][1])
NUM["examples:n_arms"] = len(same)

strongest_all = {r0["strongest"] for r0 in CK["sandbox_extension"].values()}
one_strongest = len(strongest_all) == 1
vs_head = (f"vs {ARM_LONG.get(next(iter(strongest_all)), next(iter(strongest_all)))}" if one_strongest
           else "vs strongest stateless")
ext = [SCR,
       r"\begin{longtable}{@{}" + RAG + r"p{0.2\linewidth}" + RAG + r"p{0.13\linewidth}rrrr" + RAG
       + r"p{0.12\linewidth}@{}}\toprule cell & \texttt{gpt-oss-120b} & capture & margin over $\Vstar$ & " + vs_head
       + r" & vs scripted & verdict \\ \midrule \endhead"]
EXT_ORDER = {"semantic, zero-shot": 0, "numeric, zero-shot": 1, "relabel, zero-shot": 2, "semantic, N=5": 3}
for key, r0 in sorted(CK["sandbox_extension"].items(),
                      key=lambda kv: (_sandbox_order(kv[0].split(": ")[0]), EXT_ORDER[kv[0].split(": ")[1]])):
    base, label = key.split(": ")
    ext.append(" & ".join([readable_cell(base + "-semantic"), label.replace("N=5", "$N{=}5$"), f"${f(r0['capture'])}$",
                           ci2(r0["margin"], r0["margin_ci"]),
                           ci2(r0["diff"], r0["diff_ci"])
                           + ("" if one_strongest else f" ({ARM_LONG.get(r0['strongest'], tex(r0['strongest']))})"),
                           ci2(r0["diff_scripted"], r0["diff_scripted_ci"]), f"{r0['verdict']}; {r0['class']}"]) + r" \\")
    NUM[f"extension:{key}"] = r0
ext.append(r"\bottomrule\end{longtable}" + ENDG)
ext.append(r"{\scriptsize $500$ seeds: the headline seeds $0$--$99$ and the extension seeds $320$--$719$. "
           r"Stateless non-LLM arms only, because learners that update across episodes have no pooled counterpart; "
           r"the strongest is chosen by mean. Differences in reward per episode with $95\%$ bootstrap intervals; verdict: "
           r"the pre-registered criterion against the strongest stateless arm, then the robustness class.}")
(OUT / "si_extension.tex").write_text("\n".join(ext), encoding="utf-8")
NUM["extension:n_excluded"] = sum(1 for r0 in CK["sandbox_extension"].values() if r0["class"] == "excluded")
NUM["extension:n_arms"] = len(CK["sandbox_extension"])
NUM["linux_rerun"] = {k: v for k, v in CK["linux_rerun"].items() if k != "per_arm_first_run"}
NUM["superseded_audit"] = CK["superseded_audit"]
NUM["A0:prompt_example_audit"] = EX["A0_integrity"]["prompt_example_audit"]

# battle: no zero-shot model clears the best fixed strategy (the C2 verdict of the text), range of the margin lower bounds
_blo = [v["margin_lo95"] for k0, e0 in NUM.items() if k0.startswith("master:MAgent battle") for v in e0["llm"].values()]
NUM["battle:llm_margin_lo95"] = {"min": min(_blo), "max": max(_blo), "n_arms": len(_blo)}

# --- figure 1: overview, counts filled into the TikZ template ----------------------------------------------------
fig1_arms = [v for e in M.values() for v in e["arms"].values() if v["meta"].get("arm") == "llm" and "vs_strongest" in v]
fig1_clear = [v for v in fig1_arms if v["margin_lo95"] > 0.5]
fig1_beat = [v for v in fig1_clear if v["vs_strongest"]["lo95"] > 0]
fig1_refs = {e["meta"].get("ref", c) for c, e in M.items() if any(v["meta"].get("arm") == "llm" and "vs_strongest" in v for v in e["arms"].values())}
fig1_head = [c for c in fig1_refs if EX["A3_uncertainty"]["cells"][c]["deployment"]["H_D_ci"][0] > 0]


def fig1_briefing_counts(root=None):
    """Counts of the briefing cells for Fig. 1c, read at run time from the briefing analyses.

    BK: every type deployed, so most agents are known (all_types.json, plus the prompt and example variants of its
    M=18 cell in task_prompt.json:s0). BN: only held-out combinations deployed, so every agent is new
    (new_agents.json, plus task_prompt.json:s0b). Arms are every LLM arm with a verdict, each run once: the
    pre-registered family of the file (P_E.verdicts) plus the variant arms (key with a fifth field, v2...) of
    task_prompt.json; the bare-prompt copies that task_prompt.json re-bootstraps are not counted twice. A cell has
    headroom when its exact H_D > 0. An arm clears C2 when its verdict is 'yes' or 'fixed only' (margin lower bound >
    0.5). Verdicts are read against the strongest tested non-LLM arm with the factorized readers among the candidates,
    both composers, fitted and simulator (fcr_sym.json, through briefing_c3.C3). An arm passes C3 when its verdict is
    'yes' and it is a pre-registered primary test or survives Holm correction in its secondary family, recomputed on
    the p-values against that arm; BEAT_UNCORR counts every 'yes'; the _REL counts are those against the original
    candidates alone.

    `root` is the nested layout of this repository (analysis/briefing/, names all_types.json/new_agents.json/
    task_prompt.json) or a flat analysis/ layout with the earlier file names; the default is this repository."""
    root = root or Path("<LOCAL_PATH>")  # unused here: analysis/briefing/ is always present
    briefing = root / "analysis" / "briefing"
    src, names = (briefing, ("all_types.json", "new_agents.json", "task_prompt.json")) if briefing.is_dir() else \
        (root / "analysis", ("stage0.json", "stage0b.json", "stage0c.json"))
    s0 = json.load(open(src / names[0], encoding="utf-8"))
    s0b = json.load(open(src / names[1], encoding="utf-8"))
    s0c = json.load(open(src / names[2], encoding="utf-8"))
    dec = s0c["decision"]
    c3 = C3(s0, s0b, s0c, load_fcr_sym(ROOT))

    def variant(k):
        return k.split("__")[4] if k.startswith("llm__") and k.count("__") >= 4 else None

    out, passing, allarms = {}, {}, {}
    for tag, d, part in (("BK", s0, "s0"), ("BN", s0b, "s0b")):
        # (group, arm key, verdict, passes C3 after correction, verdict and C3 against the original candidates alone)
        arms = [(a["group"], a["key"], a["pays"], a["c3"], a["pays_release"], a["c3_release"])
                for a in c3.arms.values() if a["part"] == part]
        out[f"{tag}_CELLS"] = len(d["groups"])
        out[f"{tag}_HEAD"] = sum(1 for g in d["groups"].values() if g["exact"]["H_D"] > 0)
        out[f"{tag}_ARMS"] = len(arms)
        out[f"{tag}_CLEAR"] = sum(1 for a in arms if a[2] in ("yes", "fixed only"))
        out[f"{tag}_BEAT"] = sum(1 for a in arms if a[3])
        out[f"{tag}_BEAT_UNCORR"] = sum(1 for a in arms if a[2] == "yes")
        out[f"{tag}_BEAT_REL"] = sum(1 for a in arms if a[5])
        out[f"{tag}_BEAT_UNCORR_REL"] = sum(1 for a in arms if a[4] == "yes")
        out[f"{tag}_ZS_BEAT"] = sum(1 for a in arms if a[3] and a[1].split("__")[3] == "nshot0")   # zero-shot arms
        out[f"{tag}_OPEN_CELLS"] = len({a[0] for a in arms if a[3]})
        passing[tag] = [a[1] for a in arms if a[3]]
        allarms[tag] = arms
    # the sentences of the box under Fig. 1c follow the counts, so that a refreshed analysis cannot leave them wrong
    nominal = [a[1] for a in allarms["BK"] if a[2] == "yes" and not a[3]]
    if out["BK_BEAT"] == 0 and nominal and all("think-long" in k for k in nominal):
        box_b = ("no arm beats the strongest tested cheap reader after correction; "
                 + ("only a long-reasoning arm passes" if len(nominal) == 1 else "only long-reasoning arms pass")
                 + " uncorrected.")
    elif out["BK_BEAT"] == 0:
        box_b = "no arm beats the strongest tested cheap reader."
    else:
        box_b = f"{out['BK_BEAT']} arms beat the strongest tested cheap reader."
        print("WARNING fig1: an arm beats the strongest tested cheap reader with most agents known; check the box of Fig. 1c")
    # new agents, sigma = 0: four new types (react) and eight new types (react8)
    g4, g8 = "sandbox-traits-M18-K10-sharp0.8-noise0-react", "sandbox-traits-M18-K10-sharp0.8-noise0-react8"
    zs = "llm__gpt-oss-120b__single__nshot0__v2"
    zs120_four = any(a[0] == g4 and a[1] == zs and a[3] for a in allarms["BN"])
    zs120_eight = [a[2] for a in allarms["BN"] if a[0] == g8 and a[1] == zs and not a[3]] == ["fixed only"]
    qwen1 = all(any(a[0] == g and a[1].startswith("llm__qwen3.8-27b__single__nshot1") and a[3] for a in allarms["BN"])
                for g in (g4, g8))
    if zs120_four and zs120_eight and qwen1:
        box_c = ("gpt-oss-120b with the task prompt pays zero-shot against four new types, clears only $\\Vstar$ "
                 "against eight; Qwen, one example per known type, pays in both.")
    else:
        box_c = f"{out['BN_BEAT']} arms pay."
        print("WARNING fig1: the named new-agents configurations no longer pass; check the box of Fig. 1c")
    out["BOX_B"], out["BOX_C"] = box_b, box_c
    return out


FIG1_BW = 0.84   # cm, the longest bar of a column in Fig. 1c
fig1_b = fig1_briefing_counts(ROOT)
fig1_sub = {"N_CELLS": len(fig1_refs), "N_HEAD": len(fig1_head), "N_ARMS": len(fig1_arms), "N_CLEAR": len(fig1_clear),
            "N_BEAT": len(fig1_beat), "N_PAYS": len(fig1_beat), **fig1_b}
# bar lengths per column (A no briefing, B briefing with most agents known, C briefing with new agents): C1 the share
# of cells with headroom, C2 and C3 the arms that remain after the condition as a share of the column's arms
for col, (arms, clear, beat, cells, head) in {"A": (len(fig1_arms), len(fig1_clear), len(fig1_beat), len(fig1_refs), len(fig1_head)),
                                              "B": (fig1_b["BK_ARMS"], fig1_b["BK_CLEAR"], fig1_b["BK_BEAT"], fig1_b["BK_CELLS"], fig1_b["BK_HEAD"]),
                                              "C": (fig1_b["BN_ARMS"], fig1_b["BN_CLEAR"], fig1_b["BN_BEAT"], fig1_b["BN_CELLS"], fig1_b["BN_HEAD"])}.items():
    fig1_sub[f"W{col}1"] = f"{FIG1_BW * head / cells:.3f}"
    fig1_sub[f"W{col}2"] = f"{FIG1_BW * clear / arms:.3f}"
    fig1_sub[f"W{col}3"] = f"{FIG1_BW * beat / arms:.3f}"
# panel c states the regimes in words; the counts behind them sit in the first sentence of the Results, so the
# words are checked here: no first-moves arm passes C3 and none clears C2 in battle; with a briefing and most agents
# known C3 closes for every arm after correction; with new agents some arms, not all, pass
_battle_c2 = [v for c0, e0 in M.items() if c0.startswith("battle") for v in e0["arms"].values()
              if v["meta"].get("arm") == "llm" and v.get("margin_lo95") is not None and v["margin_lo95"] > 0.5]
if fig1_sub["N_BEAT"] or _battle_c2 or fig1_b["BK_BEAT"] or not (0 < fig1_b["BN_BEAT"] < fig1_b["BN_CLEAR"]):
    print("WARNING fig1: the regime box of panel c no longer matches the counts; check fig1_overview.tex.in")
fig1_tex = (Path(__file__).resolve().parent / "fig1_overview.tex.in").read_text(encoding="utf-8")
for k0, v0 in fig1_sub.items():
    fig1_tex = fig1_tex.replace(f"<<{k0}>>", str(v0))
assert "<<" not in fig1_tex
(OUT / "fig_overview.tex").write_text(fig1_tex, encoding="utf-8", newline="\n")
NUM["fig1:counts"] = {k0: v0 for k0, v0 in fig1_sub.items() if not k0.startswith(("W", "BOX"))}
NUM["fig1:box"] = {k0: v0 for k0, v0 in fig1_sub.items() if k0.startswith("BOX")}
NUM["fig1:briefing_rule"] = " ".join(fig1_briefing_counts.__doc__.split())

json.dump(NUM, open(OUT / "numbers.json", "w", encoding="utf-8"), indent=1, default=float)
print("tables written to", OUT)
for k in ("master:MAgent battle, base4", "master:combined arms, base4", "master:combined arms, traits8", "master:sandbox $M{=}3$, $s{=}0.8$, $\\sigma{=}0$"):
    print(k, json.dumps(NUM[k], default=float)[:700])
