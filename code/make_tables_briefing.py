"""Generate the LaTeX tables of the briefing section from the three briefing analyses.

Inputs: the nested layout of this repository (analysis/briefing/, names all_types.json, new_agents.json,
task_prompt.json, bare_prompt_first_format.json) or a flat analysis/ layout with the earlier file names, selected by
--root (default: this repository).
all_types.json: every type deployed; new_agents.json: held-out combinations only; task_prompt.json: prompt, example
and reasoning variants; bare_prompt_first_format.json: the fallback rate of the bare-prompt arms under their first output format. Re-run
after any of them changes.

Outputs: tab_briefing.tex (main text), si_briefing_design.tex, si_briefing_prereg.tex, si_briefing_cells.tex,
si_briefing_cost.tex, si_briefing_classic.tex, si_briefing_pertype.tex, and numbers_briefing.json (every number the
section quotes, keyed, in the schema of numbers.json) for check_numbers.py --extra, written to latex/tables/ in a
working tree or analysis/tables/ in this repository.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from briefing_c3 import C3, load_fcr_sym  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default=None, help="briefing analyses: a working tree or this repository (see module docstring)")
_args = _ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]  # this script's own tree
_release = (ROOT / "analysis" / "briefing").is_dir()
_src_root = Path(_args.root) if _args.root else (ROOT if _release else Path("<LOCAL_PATH>"))  # the else is unused here
_briefing = _src_root / "analysis" / "briefing"
SRC, _OLD = (_briefing, False) if _briefing.is_dir() else (_src_root / "analysis", True)
S0 = json.load(open(SRC / ("stage0.json" if _OLD else "all_types.json"), encoding="utf-8"))
S0B = json.load(open(SRC / ("stage0b.json" if _OLD else "new_agents.json"), encoding="utf-8"))
S0C = json.load(open(SRC / ("stage0c.json" if _OLD else "task_prompt.json"), encoding="utf-8"))
S0R1 = json.load(open(SRC / ("stage0_run1.json" if _OLD else "bare_prompt_first_format.json"), encoding="utf-8"))


def _first_file(*cands):
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(cands[0])


# the factorized compositional reader (fcr.json) and the calibration-seed verdict check (calib_verdict.json): in this
# repository under analysis/briefing/ and analysis/calib_verdict/, in the paper tree directly under analysis/
FCR = json.load(open(_first_file(ROOT / "analysis" / "briefing" / "fcr.json", ROOT / "analysis" / "fcr.json"),
                     encoding="utf-8"))
CAL = json.load(open(_first_file(ROOT / "analysis" / "calib_verdict" / "calib_verdict.json",
                                 ROOT / "analysis" / "calib_verdict.json"), encoding="utf-8"))
# the factorized reader with the simulator composer (fcr_sym.json) and its calibration-only check of the composer
# (fcr_sym_calib.json): in this repository code/briefing/fcr_sym/calib/sym_calib_check.json, in the paper
# tree a copy of it next to fcr.json
FS = load_fcr_sym(ROOT)
FSC = json.load(open(_first_file(ROOT / "code" / "briefing" / "fcr_sym" / "calib" / "sym_calib_check.json",
                                 ROOT / "analysis" / "fcr_sym_calib.json"), encoding="utf-8"))
# every language-model verdict against the strongest tested non-LLM arm with both composers among the candidates
C3V = C3(S0, S0B, S0C, FS)
OUT = ROOT / "analysis" / "tables" if _release else ROOT / "latex" / "tables"
NUM: dict = OrderedDict()

P0 = "sandbox-traits-M18-K10-sharp0.8-noise0-react"     # M=18, sigma 0, react4 held out
R8 = "sandbox-traits-M18-K10-sharp0.8-noise0-react8"    # M=18, sigma 0, react8 held out (held-out-only deployment)
NEW, SAME = "briefing-newword", "briefing-sameword"
MODELS = [("gpt-oss-120b", "120b"), ("gpt-oss-20b", "20b"), ("qwen3.8-27b", "Qwen")]
TAG = dict(MODELS)
MODEL_LONG = {"gpt-oss-20b": r"\texttt{gpt-oss-20b}", "gpt-oss-120b": r"\texttt{gpt-oss-120b}", "qwen3.8-27b": r"\texttt{Qwen3.8-27B}"}
POOL_ORDER = ["ROCK", "PAPER", "SCISSORS", "OUTPACE", "MIRROR_BEAT", "LAG_R", "LAG_P", "LAG_S", "SWITCH_OM", "SWITCH_MO"]
NONLLM_ORDER = ["type_oracle", "scripted", "fewshot__1", "plastic", "linucb", "lints", "ctxucb", "mucb", "ppo", "random",
                "scripted_text", "text_bow", "text_embed", "text_linucb"]
BANDITS = ["linucb", "lints", "ctxucb", "mucb"]
# which file holds the bare-prompt arms and which task_prompt part holds the variants, per deployment
DEPLOY = {"full": (S0, "all_types.json", "s0"), "heldout": (S0B, "new_agents.json", "s0b")}


def f(x, d=2, plus=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.{d}f}" if plus else f"{x:.{d}f}"
    return s.replace("-", "+" if plus else "") if float(s) == 0 else s


def tex(s: str) -> str:
    return str(s).replace("_", "\\_")


def grey(s):
    return r"{\color{black!60}" + s + "}"


def grey_ci(lo, hi, d=2, plus=False):
    """A grey '[lo,hi]' interval with each bound set in math mode, so a negative bound prints a math minus, not a
    text hyphen, exactly as the main (black) values already do."""
    return grey(f"[${f(lo, d, plus)}$,${f(hi, d, plus)}$]")


def cap_ci(a, d=2):
    """Capture in math and its 95% interval in grey text, as on the second lines of the master table."""
    if a is None or a.get("capture") is None:
        return "--"
    lo, hi = a["capture_ci"]
    return f"${f(a['capture'], d)}$\\," + grey_ci(lo, hi, d)


def ci_math(v, lo_hi, d=2, plus=False):
    if v is None or not lo_hi or lo_hi[0] is None:
        return "--"
    return f"${f(v, d, plus)}\\,[{f(lo_hi[0], d, plus)},{f(lo_hi[1], d, plus)}]$"


# ------------------------------------------------------------------------------------------------------------------
# accessors and labels
# ------------------------------------------------------------------------------------------------------------------
def variant_of(key):
    return key.split("__")[4] if key.startswith("llm__") and key.count("__") >= 4 else None


def find(deploy, group, cell, key):
    """(record, json path) of an arm: bare-prompt arms from all_types/new_agents, variants and matched readers from task_prompt."""
    d, fname, part = DEPLOY[deploy]
    if variant_of(key) or "_matched__" in key:
        rec = S0C[part].get(group, {}).get("arms", {}).get(f"{cell}/{key}")
        return rec, f"task_prompt.json:{part}.{group}.arms.{cell}/{key}"
    if key == "prefix_bayes_ceiling":
        return d["groups"][group]["prefix_bayes_ceiling"], f"{fname}:groups.{group}.prefix_bayes_ceiling"
    rec = d["groups"][group]["cells"].get(cell, {}).get("arms", {}).get(key)
    return rec, f"{fname}:groups.{group}.cells.{cell}.arms.{key}"


VARIANT = {None: "bare", "v2": "task", "v2-shuf": "task shuffled", "v2-med": "task medium",
           "v2-high": "task high 4096", "v2-high-long": "task high", "v2-think": "task thinking 4096",
           "v2-think-long": "task thinking"}
NONLLM = {"type_oracle": "type oracle", "prefix_bayes_ceiling": "prefix ceiling", "scripted": "prefix rule",
          "fewshot__1": "few-shot 1", "plastic": "PLASTIC", "linucb": "LinUCB", "lints": "LinTS", "ctxucb": "CtxUCB",
          "mucb": "M-UCB", "ppo": "PPO", "random": "random", "scripted_text": "keyword rule", "text_bow": "TF-IDF",
          "text_embed": "embedding", "text_linucb": "text LinUCB"}


def arm_label(key, short=True):
    """llm__gpt-oss-120b__single__nshot1__v2 -> 120b task N=1; non-LLM arms by name."""
    if key.startswith("llm__"):
        parts = key.split("__")
        model, mode, nshot = parts[1], parts[2], int(parts[3].replace("nshot", ""))
        bits = [r"\texttt{" + TAG[model] + "}" if short else MODEL_LONG[model], VARIANT[variant_of(key)]]
        if mode == "hypothesis_first":
            bits.append("hf" if short else "hypothesis-first")
        return " ".join(bits) + f" $N{{=}}{nshot}$"
    if "_matched__nshot" in key:
        base, n = key.split("_matched__nshot")
        return f"{NONLLM[base]} on the $N{{=}}{n}$ rows"
    if key.startswith("fixed__"):
        return "fixed " + tex(key[7:])
    return NONLLM.get(key, tex(key))


def known_types(group):
    """Types that calibration has seen in a group: M minus the held-out set (the same set in all_types and new_agents)."""
    meta = (S0B["groups"].get(group) or S0["groups"][group])["meta"]
    return meta["M"] - len(meta["held_out"])


def examples_in_words(key, group):
    """The examples of an N-shot arm per known type: N=0 none, N=1 one per known type, N=42 (fourteen known types) and
    N=30 (ten) three; checked against the group's known-type count."""
    n = int(key.split("__")[3].replace("nshot", ""))
    k = known_types(group)
    per = n if n <= 1 else n // k
    assert n <= 1 or per * k == n, (key, group, k)
    return f"{per}/type"


def table2_label(key, group):
    """Arm label of Table 2, with the examples counted per known type: 120b task, 1/type."""
    parts = key.split("__")
    bits = [r"\texttt{" + TAG[parts[1]] + "}", VARIANT[variant_of(key)]] + (["hf"] if parts[2] == "hypothesis_first" else [])
    return " ".join(bits) + ", " + examples_in_words(key, group)


def best_by(cands, field):
    cands = [(k, v) for k, v in cands if v is not None and v.get(field) is not None]
    return max(cands, key=lambda kv: kv[1][field]) if cands else (None, None)


# ------------------------------------------------------------------------------------------------------------------
# test status: P = the pre-registered primary test, H = secondary and surviving Holm, S = secondary
# ------------------------------------------------------------------------------------------------------------------
def _primary_group(d):
    pc = d["P_E"]["primary_cell"]
    for gname, g in d["groups"].items():
        m = g["meta"]
        if m["M"] == pc["M"] and m["reward_noise"] == pc["reward_noise"] and m["kind"] == pc["kind"]:
            return gname
    raise KeyError("primary cell")


C_PRIMARY = set()
for _k in S0C["decision"]["primary_0c"]:
    _part, _model = _k.split("/")
    C_PRIMARY.add(("s0" if _part == "s0" else "s0b", P0 if _part in ("s0", "s0b-react4") else R8,
                   f"llm__{_model}__single__nshot0__v2"))


def test_status(deploy, group, cell, key, release=False):
    """P, H or S of a language-model arm in its briefing family (briefing_c3): judged against the strongest tested
    non-LLM arm with the factorized readers, both composers, among the candidates, or with release=True against the
    original candidates alone. The primary tests are those of the briefing pre-registration (C_PRIMARY and P_E)."""
    if not key.startswith("llm__"):
        return "--"
    d, _, part = DEPLOY[deploy]
    a = C3V.get(part, group, cell, key)
    if variant_of(key):
        prim = (part, group, key) in C_PRIMARY and cell == NEW
    else:
        pc = d["P_E"]["primary_cell"]
        model = pc.get("model")
        prim = (group == _primary_group(d) and cell == pc["cell"] and f"__{pc['mode']}__nshot{pc['nshot']}" in key
                and (model is None or f"__{model}__" in key))
    assert a is not None and a["prim"] == prim, (deploy, group, cell, key)
    return a["status_release" if release else "status"]


# one verdict convention for Table 2 and the supplement's briefing and FCR tables: a primary test (P) is judged by the
# pre-registered criterion uncorrected; a secondary test (S) after Holm correction within its family, so a secondary arm
# whose uncorrected verdict is yes but which fails Holm shows NOMINAL. The internal status keeps H (secondary, surviving
# Holm) apart from S (secondary, not surviving) for numbers_briefing.json; the tables print P or S.
NOMINAL = r"yes\textsuperscript{n}"


def shown_verdict(pays, status):
    """The verdict as printed: yes, fixed only or no, with NOMINAL for a secondary arm that passes only uncorrected."""
    return NOMINAL if (pays == "yes" and status == "S") else (pays or "--")


def shown_test(status):
    return "P" if status == "P" else "S"


# ------------------------------------------------------------------------------------------------------------------
# main table: tab_briefing
# ------------------------------------------------------------------------------------------------------------------
def row(deploy, group, cell, key, label=None):
    rec, path = find(deploy, group, cell, key)
    if rec is None:
        return None
    c3 = C3V.get(DEPLOY[deploy][2], group, cell, key) if key.startswith("llm__") else None
    return {"label": label or arm_label(key), "key": f"{cell}/{key}" if cell else key, "rec": rec, "path": path,
            "test": test_status(deploy, group, cell, key), "c3": c3}


FCR_ARMS = list(FCR["meta"]["fcr_arms"])          # the nine non-privileged FCR arms
FCR_SHORT = {"fcr_full": "TF-IDF traits", "fcr_full_pairwise": "pairwise", "fcr_full_embed": "embedding",
             "fcr_n_bare": "1/type bare rows", "fcr_n_bare_pairwise": "1/type bare rows, pairwise",
             "fcr_n_bare_embed": "1/type bare rows, embedding", "fcr_n_task": "1/type task rows",
             "fcr_n_task_pairwise": "1/type task rows, pairwise", "fcr_n_task_embed": "1/type task rows, embedding",
             "fcr_oracle": "trait oracle", "fcr_oracle_pairwise": "trait oracle, pairwise"}
FCR_LONG = {"fcr_full": "FCR-full, TF-IDF, main effects", "fcr_full_pairwise": "FCR-full, TF-IDF, pairwise",
            "fcr_full_embed": "FCR-full, embedding, main effects", "fcr_n_bare": "FCR-N bare rows, TF-IDF, main effects",
            "fcr_n_bare_pairwise": "FCR-N bare rows, TF-IDF, pairwise",
            "fcr_n_bare_embed": "FCR-N bare rows, embedding, main effects",
            "fcr_n_task": "FCR-N task rows, TF-IDF, main effects", "fcr_n_task_pairwise": "FCR-N task rows, TF-IDF, pairwise",
            "fcr_n_task_embed": "FCR-N task rows, embedding, main effects",
            "fcr_oracle": r"trait oracle, main effects$^{\S}$", "fcr_oracle_pairwise": r"trait oracle, pairwise$^{\S}$"}
FCR_PRIMARY_DEP = OrderedDict([("most agents known", f"all/{P0}/newword"), ("new agents, react4", f"heldout/{P0}/newword"),
                               ("new agents, react8", f"heldout/{R8}/newword")])


def fcr_dep(deploy, group):
    """Key of a briefing deployment in fcr.json: every type deployed (all/) or held-out types only (heldout/)."""
    return f"{'all' if deploy == 'full' else 'heldout'}/{group}/newword"


def fcr_best(dk):
    """The strongest of the nine non-privileged FCR arms of a deployment, by headline mean (ties: earlier arm)."""
    arms = FCR["deployments"][dk]["arms"]
    return max(FCR_ARMS, key=lambda k: (arms[k]["mean"], -FCR_ARMS.index(k)))


SYM_ARMS = [k for k, v in FS["meta"]["sym_arms"].items()]          # the six non-privileged simulator-composer arms
SYM_SHORT = {"sym_full": "simulator, TF-IDF traits", "sym_full_embed": "simulator, embedding",
             "sym_n_bare": "simulator, 1/type bare rows", "sym_n_bare_embed": "simulator, 1/type bare rows, embedding",
             "sym_n_task": "simulator, 1/type task rows", "sym_n_task_embed": "simulator, 1/type task rows, embedding",
             "sym_oracle": "trait oracle, simulator"}
SYM_LONG = {"sym_full": "FCR-sym-full, TF-IDF", "sym_full_embed": "FCR-sym-full, embedding",
            "sym_n_bare": "FCR-sym-N bare rows, TF-IDF", "sym_n_bare_embed": "FCR-sym-N bare rows, embedding",
            "sym_n_task": "FCR-sym-N task rows, TF-IDF", "sym_n_task_embed": "FCR-sym-N task rows, embedding",
            "sym_oracle": r"trait oracle, simulator$^{\S}$"}


def fcr_rows(deploy, group):
    """The FCR rows of Table 2: the simulator composer fitted to every known briefing (FCR-sym-full), the strongest
    fitted-composer variant (the strongest of the nine non-privileged FCR arms) and the simulator composer on the bare
    prompt's one-example rows (FCR-sym-N)."""
    dk = fcr_dep(deploy, group)
    best = fcr_best(dk)
    sym = FS["deployments"][dk]["arms"]
    # short labels: Table 2 is one column wide at \scriptsize, so a label wider than about 78 pt overflows it
    return [{"label": "FCR, simulator", "key": "sym/sym_full", "rec": sym["sym_full"],
             "path": f"fcr_sym.json:deployments.{dk}.arms.sym_full", "test": "--"},
            {"label": "FCR, fitted", "key": f"fcr/{best}", "rec": FCR["deployments"][dk]["arms"][best],
             "path": f"fcr.json:deployments.{dk}.arms.{best}", "test": "--"},
            {"label": "FCR-N, simulator", "key": "sym/sym_n_bare", "rec": sym["sym_n_bare"],
             "path": f"fcr_sym.json:deployments.{dk}.arms.sym_n_bare", "test": "--"}]


def strongest_row_key(deploy, group):
    """Row key in Table 2 of the strongest tested non-LLM arm with both composers among the candidates."""
    s = C3V.strongest(DEPLOY[deploy][2], group)
    if s.startswith("sym_"):
        return "sym/" + s
    if s.startswith("fcr_"):
        return "fcr/" + s
    return s.replace("briefing:", NEW + "/").replace("nonllm:", "nonllm/")


def nonllm_rows(deploy, group):
    d = DEPLOY[deploy][0]
    g = d["groups"][group]
    # the type oracle collects the whole headroom in all three deployments; the caption says so instead of three rows
    to = find(deploy, group, "nonllm", "type_oracle")[0]
    assert abs(to["capture"] - 1.0) < 1e-9 and min(to["capture_ci"]) > 1.0 - 1e-9, (deploy, group)
    rows = [row(deploy, group, "", "prefix_bayes_ceiling"),
            row(deploy, group, "nonllm", "scripted"), row(deploy, group, "nonllm", "fewshot__1")]
    bk, _ = best_by([(k, find(deploy, group, "nonllm", k)[0]) for k in BANDITS], "mean")
    rows.append(row(deploy, group, "nonllm", bk, f"bandit: {NONLLM[bk]}"))
    rows += [row(deploy, group, NEW, k) for k in ["scripted_text", "text_bow", "text_embed", "text_linucb"]]
    # the original strongest arm, which the fitted composer alone leaves unchanged in every briefing deployment
    rel = g["strongest_nonllm"]["newword"]
    assert FCR["deployments"][fcr_dep(deploy, group)]["strongest"]["with_fcr"].replace("briefing:", NEW + "/").replace(
        "nonllm:", "nonllm/") == rel, (deploy, group)
    strongest = strongest_row_key(deploy, group)
    rows += fcr_rows(deploy, group)
    if strongest not in [r["key"] for r in rows]:
        c, k = strongest.split("/")
        assert c in ("nonllm", NEW), strongest
        rows.append(row(deploy, group, c, k))
    for r in rows:
        if r["key"] == strongest:
            r["label"] += r"$^{\dagger}$"
    assert sum(r["label"].endswith(r"$^{\dagger}$") for r in rows) == 1, (deploy, group)
    return rows, strongest


def best_zero_shot_bare(group, model):
    k, _ = best_by([(f"llm__{model}__{m}__nshot0", find("full", group, NEW, f"llm__{model}__{m}__nshot0")[0])
                    for m in ("single", "hypothesis_first")], "capture")
    return k


def best_any(group, model):
    """A model's strongest arm in a held-out-only cell over prompts, modes and examples, at the full seed count."""
    n = S0B["groups"][group]["reference"]["seeds_n"]
    keys = [k for k in S0B["groups"][group]["cells"][NEW]["arms"] if f"__{model}__" in k]
    keys += [ck.split("/", 1)[1] for ck, v in S0C["s0b"][group]["arms"].items()
             if ck.startswith(NEW + "/") and f"__{model}__" in ck and variant_of(ck.split("/", 1)[1]) and v["n"] == n]
    k, _ = best_by([(k, find("heldout", group, NEW, k)[0]) for k in keys], "capture")
    return k


def llm_rows(deploy, group):
    if deploy == "full":
        keys = [best_zero_shot_bare(group, m) for m, _ in MODELS]
        keys += ["llm__gpt-oss-120b__single__nshot0__v2", "llm__gpt-oss-120b__single__nshot1__v2",
                 "llm__qwen3.8-27b__single__nshot1", "llm__qwen3.8-27b__single__nshot1__v2"]
    else:
        keys = ["llm__gpt-oss-120b__single__nshot0", "llm__gpt-oss-120b__single__nshot0__v2",
                "llm__gpt-oss-120b__single__nshot1__v2", "llm__qwen3.8-27b__single__nshot1",
                "llm__qwen3.8-27b__single__nshot1__v2", best_any(group, "gpt-oss-20b")]
    return [r for r in (row(deploy, group, NEW, k, table2_label(k, group)) for k in keys) if r]


TAB_GROUPS = [("full", P0, r"most agents known"),
              ("heldout", P0, r"new agents, four new types"),
              ("heldout", R8, r"new agents, eight new types")]
TAB_ID = {0: "full", 1: "react4", 2: "react8"}

body = []
for gi, (deploy, group, title) in enumerate(TAB_GROUPS):
    nrows, strongest = nonllm_rows(deploy, group)
    lrows = llm_rows(deploy, group)
    if gi:
        body.append(r"\midrule")
    body.append(r"\multicolumn{7}{@{}l}{\emph{" + title + r"}}\\")
    for r in nrows + [None] + lrows:
        if r is None:
            body.append(r"\addlinespace[1pt]")
            continue
        a = r["rec"]
        is_llm = r["key"].split("/", 1)[-1].startswith("llm__")
        c3 = r.get("c3")
        # language-model rows: the difference and the verdict against the strongest arm with both composers among the
        # candidates (briefing_c3); the capture and its interval are the original ones
        vs = c3["vs"] if c3 else a.get("vs_strongest")
        pays = c3["pays"] if c3 else a.get("pays")
        # main value, then its secondary value in grey on the same line: capture and interval, difference and its
        # lower bound, verdict and test
        # each pair sits in two columns joined by a thin space, so main values align on the decimal point
        lo, hi = a["capture_ci"]
        body.append(" & ".join([
            r["label"], f"${f(a['capture'])}$", grey_ci(lo, hi),
            f"${f(vs['diff'], plus=True)}$" if (is_llm and vs) else "--",
            grey(f"${f(vs['lo95'], plus=True)}$") if (is_llm and vs) else "",
            shown_verdict(pays, r["test"]) if is_llm else "--",
            grey(shown_test(r["test"])) if is_llm else ""]) + r" \\")
        NUM[f"tab_briefing:{TAB_ID[gi]}:{r['key']}"] = OrderedDict(
            label=r["label"], src=r["path"], n=a.get("n"), capture=a.get("capture"), capture_ci=a.get("capture_ci"),
            vs_strongest=(OrderedDict(diff=vs["diff"], ci=vs["ci"], lo95=vs["lo95"], hi95=vs["hi95"],
                                      one_sided_bounds=[vs["lo95"], vs["hi95"]]) if (is_llm and vs) else None),
            vs_strongest_arm=(c3["strongest"] if c3 else a.get("vs_strongest_arm")) if is_llm else None,
            pays=pays if is_llm else None, test=r["test"],
            verdict_shown=shown_verdict(pays, r["test"]) if is_llm else None, strongest_nonllm=strongest,
            **({"pays_release": c3["pays_release"], "test_release": c3["status_release"],
                "margin_lo95": c3["margin_lo95"]} if c3 else {}),
            **({"counter_accuracy": a.get("counter_accuracy"), "trait_accuracy": a.get("trait_accuracy")}
               if r["key"].startswith(("fcr/", "sym/")) else {}))

fb_after = OrderedDict()
for name, d in (("after", S0), ("before", S0R1)):
    seen = {}
    for gname, g in d["groups"].items():
        base = gname.rsplit("-", 1)[0]
        for cname, c in g["cells"].items():
            for k, v in c["arms"].items():
                if not k.startswith("llm__"):
                    continue
                shared = (cname in ("briefing-sameword", "briefing_prefix-sameword") and k.endswith("nshot0")) or cname == "semantic"
                seen[(base if shared else gname, cname, k)] = (v["n"], v["fallback_rate"])
    calls = sum(n for n, _ in seen.values())
    fb_after[name] = OrderedDict(arms=len(seen), episodes=calls, fallbacks=round(sum(n * r for n, r in seen.values())),
                                 rate=sum(n * r for n, r in seen.values()) / calls,
                                 max_arm_rate=max(r for _, r in seen.values()))
NUM["fallback"] = OrderedDict(src="all_types.json and bare_prompt_first_format.json: groups.*.cells.*.arms.llm__*.fallback_rate, "
                                  "shared runs counted once", **fb_after)

_K4, _K8 = known_types(P0), known_types(R8)
caption = (r"""\caption{Who reads a scouting briefing: the main arms at $M{=}18$, no payoff noise, new types briefed in
new words.}""")
# the column definitions the abbreviations need, under the table; the statements about the deployments and
# the type oracle are in the text of the Results, tied by the quotes T2_ORACLE and T2_PREFIX_R4 below
tab2_note = (r"""\par\vspace{2pt}
\parbox{\linewidth}{\scriptsize Grey: the $95\%$ interval of the capture; the one-sided $95\%$ lower bound of $\Delta$,
the paired reward difference against the strongest tested non-LLM arm$^{\dagger}$; the test, P primary and judged
uncorrected, S secondary and judged after Holm correction in its family. hf: names the type first. $n$/type: $n$
labelled briefings per known type; bare, 0/type with most agents known: each model's best zero-shot bare-prompt arm.
FCR, fitted: the strongest regression composer; FCR-N: fitted to the bare prompt's examples.}""")
header = r"""\begin{table}[!t]
\centering
""" + caption + r"""
\label{tab:briefing}
\scriptsize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.0}
\newcommand{\hd}[2]{\begin{tabular}[b]{@{}c@{}}#1\\#2\end{tabular}}
\begin{tabular}{@{}lr@{\,}lr@{\,}ll@{\,}l@{}}
\toprule
\textbf{Arm} & \multicolumn{2}{c}{\hd{capture}{$\capf$\,\color{black!60}[95\%]}} & \multicolumn{2}{c}{\hd{$\Delta$}{\color{black!60}lo}} &
\multicolumn{2}{c@{}}{\hd{verdict}{\color{black!60}test}} \\
\midrule
"""
footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_briefing.tex").write_text(header + "\n".join(body) + footer.replace(r"\end{tabular}", r"\end{tabular}" + "\n" + tab2_note), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------------
# numbers quoted in the section
# ------------------------------------------------------------------------------------------------------------------
def gname(M, sigma, kind):
    K = 10 if M == 18 else 5
    return f"sandbox-traits-M{M}-K{K}-sharp0.8-noise{sigma:g}-{kind}"


for M in (18, 9):
    g = S0["groups"][gname(M, 0, "react")]
    NUM[f"lock:M{M}"] = OrderedDict(src=f"all_types.json:groups.{gname(M, 0, 'react')}.exact and .prefix_bayes_ceiling",
                                    V_star=g["exact"]["V_star"], H_D=g["exact"]["H_D"], share=g["exact"]["share"],
                                    prefix_ceiling_capture=g["exact"]["prefix_ceiling_bayes_capture"],
                                    prefix_ceiling_class_capture=g["exact"]["prefix_ceiling_class_capture"],
                                    measured=OrderedDict(capture=g["prefix_bayes_ceiling"]["capture"],
                                                         capture_ci=g["prefix_bayes_ceiling"]["capture_ci"]))
for gn, g in S0B["groups"].items():
    NUM[f"lock_heldout:{gn}"] = OrderedDict(src=f"new_agents.json:groups.{gn}.exact, .reference, .reference_0b",
                                            deploy_types=len(g["reference_0b"]["deployment_types"]),
                                            V_star=g["reference"]["V_star"], H_D=g["reference"]["H_D"],
                                            H_D_exact=g["exact"]["H_D"], share=g["exact"]["share"],
                                            prefix_ceiling_capture=g["exact"]["prefix_ceiling_bayes_capture"],
                                            predicted_T_star=g["reference_0b"]["predicted_T_star"],
                                            strongest_nonllm=g["strongest_nonllm"]["newword"])

readers = OrderedDict()
for k in ["text_bow", "scripted_text", "text_embed"]:
    vals = {gn: g["cells"][NEW]["arms"][k]["capture"] for gn, g in S0B["groups"].items()}
    readers[k] = OrderedDict(min=min(vals.values()), max=max(vals.values()), argmin=min(vals, key=vals.get),
                             argmax=max(vals, key=vals.get), per_cell=vals)
NUM["heldout_readers_newword"] = OrderedDict(src="new_agents.json:groups.*.cells.briefing-newword.arms.{text_bow,scripted_text,text_embed}.capture",
                                             **readers)

bend = OrderedDict()
for gn, g in S0B["groups"].items():
    if "-M18-" not in gn or "diag" in gn:
        continue
    bend[gn] = OrderedDict((k, OrderedDict(capture=g["cells"]["nonllm"]["arms"][k]["capture"],
                                           capture_ci=g["cells"]["nonllm"]["arms"][k]["capture_ci"],
                                           population_T300=g["cells"]["nonllm"]["arms"][k]["capture_at_population"]["300"],
                                           T_50=g["cells"]["nonllm"]["arms"][k]["T_50"]))
                           for k in ("linucb", "lints"))
best_end = [max(v["linucb"]["capture"], v["lints"]["capture"]) for v in bend.values()]
m9 = OrderedDict()
for gn, g in S0B["groups"].items():
    if "-M9-" not in gn:
        continue
    sk = g["strongest_nonllm"]["newword"]
    sa = g["cells"][sk.split("/")[0]]["arms"][sk.split("/")[1]]
    q = g["cells"][NEW]["arms"]["llm__qwen3.8-27b__single__nshot1"]
    m9[gn] = OrderedDict(strongest=sk, strongest_capture=sa["capture"], strongest_capture_ci=sa["capture_ci"],
                         strongest_T_50=sa.get("T_50"), predicted_T_star=g["reference_0b"]["predicted_T_star"],
                         qwen_n1_capture=q["capture"], qwen_n1_capture_ci=q["capture_ci"], qwen_n1_pays=q["pays"])
NUM["heldout_M9"] = OrderedDict(src="new_agents.json:groups.<M9>.strongest_nonllm.newword and "
                                    "cells.briefing-newword.arms.llm__qwen3.8-27b__single__nshot1", per_cell=m9,
                                strongest_min=min(v["strongest_capture"] for v in m9.values()),
                                strongest_max=max(v["strongest_capture"] for v in m9.values()),
                                qwen_n1_min=min(v["qwen_n1_capture"] for v in m9.values()),
                                qwen_n1_max=max(v["qwen_n1_capture"] for v in m9.values()),
                                strongest_T_50_min=min(v["strongest_T_50"] for v in m9.values()),
                                strongest_T_50_max=max(v["strongest_T_50"] for v in m9.values()),
                                T_star_min=min(v["predicted_T_star"] for v in m9.values()),
                                T_star_max=max(v["predicted_T_star"] for v in m9.values()))

NUM["bandits_T300_M18_react"] = OrderedDict(src="new_agents.json:groups.<M18 react4/react8>.cells.nonllm.arms.{linucb,lints}.capture "
                                                "(300 episodes, paired)", per_cell=bend,
                                            best_min=min(best_end), best_max=max(best_end))

NUM["holm"] = OrderedDict(
    src="all_types.json:P_E; new_agents.json:P_E; task_prompt.json:decision",
    full=OrderedDict(entries=S0["P_E"]["secondary_counts"]["llm_arm_entries"], yes=S0["P_E"]["secondary_counts"]["pays_yes"],
                     holm=len(S0["P_E"]["holm_surviving"]["pays"]), primary_holds=S0["P_E"]["primary_holds"]),
    heldout=OrderedDict(entries=S0B["P_E"]["secondary_counts"]["llm_arm_entries"], yes=S0B["P_E"]["secondary_counts"]["pays_yes"],
                        holm=len(S0B["P_E"]["holm_surviving"]["pays"]), primary_holds=S0B["P_E"]["primary_holds"]),
    variants=OrderedDict(entries=S0C["decision"]["secondary_entries"], yes=len(S0C["decision"]["secondary_uncorrected_yes"]),
                         holm=len(S0C["decision"]["holm_surviving"]), primary_yes=S0C["decision"]["primary_yes"],
                         verdict=S0C["decision"]["verdict"], holm_surviving=S0C["decision"]["holm_surviving"],
                         uncorrected_yes=S0C["decision"]["secondary_uncorrected_yes"]))
NUM["primary_tests"] = OrderedDict(
    full={m: OrderedDict(capture=v["capture"], capture_ci=v["capture_ci"], margin_lo95=v["margin_lo95"],
                         vs_strongest=v["vs_strongest"], vs_strongest_arm=v["vs_strongest_arm"], pays=v["pays"],
                         vs_strongest_held_out=v.get("vs_strongest_held_out")) for m, v in S0["P_E"]["primary"].items()},
    heldout={m: OrderedDict(capture=v["capture"], capture_ci=v["capture_ci"], margin_lo95=v["margin_lo95"],
                            vs_strongest=v["vs_strongest"], vs_strongest_arm=v["vs_strongest_arm"], pays=v["pays"])
             for m, v in S0B["P_E"]["primary"].items()},
    variants={k: OrderedDict(capture=v["capture"], capture_ci=v["capture_ci"], margin_lo95=v["margin_lo95"],
                             vs_strongest=v["vs_strongest"], vs_strongest_arm=v["vs_strongest_arm"], pays=v["pays"],
                             p_pays=v["p_pays"]) for k, v in S0C["decision"]["primary_0c"].items()},
    src="all_types.json:P_E.primary; new_agents.json:P_E.primary; task_prompt.json:decision.primary_0c")

bms = OrderedDict()
for M in (18, 9):
    g = S0["groups"][gname(M, 0, "react")]
    for w in (SAME, NEW):
        for m, _ in MODELS:
            k = f"{w}/llm__{m}__single__nshot0"
            v = g["briefing_minus_semantic"][k]
            bms[f"M{M}:{w}:{m}"] = OrderedDict(diff=v["diff"], ci=v["ci"], lo95=v["lo95"], hi95=v["hi95"])
NUM["briefing_minus_semantic"] = OrderedDict(src="all_types.json:groups.<M sigma0 react>.briefing_minus_semantic.<wording>/llm__<model>__single__nshot0",
                                             **bms)

NUM["classic"] = OrderedDict(src="task_prompt.json:classic.P_N and classic.cells", **{
    m: OrderedDict(mean=v["mean_capture_diff"], ci=v["mean_ci"], width=v["stage0_width"], material=v["material"],
                   cells_beyond_width=v["cells_beyond_width"], ci_excludes_0=v["ci_excludes_0"])
    for m, v in S0C["classic"]["P_N"].items()})

COST_KEYS = ["mean_completion_tokens", "p90_completion_tokens", "mean_latency_s", "p90_latency_s", "length_finish_rate",
             "fallback_rate", "mean_prompt_tokens", "episodes"]


def cost_rows():
    """Every reasoning arm of task_prompt.json with its cost, and the default-effort arm of each model in the full-type cell."""
    out = []
    for m, _ in MODELS:
        for part, group in (("s0", P0), ("s0b", P0), ("s0b", R8)):
            for ck, v in S0C[part][group]["arms"].items():
                cell, k = ck.split("/", 1)
                var = variant_of(k)
                if f"__{m}__" not in k or "__single__" not in k or not v.get("cost"):
                    continue
                base = var == "v2" and part == "s0" and cell == NEW and "__nshot0__" in k
                if base or var in ("v2-med", "v2-high", "v2-high-long", "v2-think", "v2-think-long"):
                    out.append((m, var, part, group, cell, k, v))
    order = {"v2": 0, "v2-med": 1, "v2-high": 2, "v2-high-long": 3, "v2-think": 4, "v2-think-long": 5}
    return sorted(out, key=lambda r: ([mm for mm, _ in MODELS].index(r[0]), order[r[1]], r[2], r[3] != P0, r[4] != NEW, r[5]))


NUM["cost"] = OrderedDict(src="task_prompt.json:<part>.<group>.arms.<cell>/<key>.cost")
for m, var, part, group, cell, k, v in cost_rows():
    NUM["cost"][f"{part}:{group}:{cell}/{k}"] = OrderedDict(n=v["n"], capture=v["capture"], capture_ci=v["capture_ci"],
                                                             **{c: v["cost"].get(c) for c in COST_KEYS})
PL = S0C["predictions"]["P_L"]
NUM["cost:P_L"] = OrderedDict(src="task_prompt.json:predictions.P_L", capture=PL["capture"], capture_ci=PL["capture_ci"],
                              token_ratio=PL["token_ratio"], think_tokens=PL["think_cost"]["mean_completion_tokens"],
                              base_tokens=PL["base_cost"]["mean_completion_tokens"], n=PL["think_cost"]["episodes"])

c_pred = S0C["predictions"]
NUM["variants:contrasts"] = OrderedDict(
    src="task_prompt.json:predictions.{P_J,P_K,P_M,s0b_n1} and s0b.<group>.contrasts",
    P_J=OrderedDict(diff=c_pred["P_J"]["contrast"]["capture_diff"], ci=c_pred["P_J"]["contrast"]["capture_ci"],
                    threshold=c_pred["P_J"]["threshold"], holds=c_pred["P_J"]["holds"]),
    P_K=OrderedDict(diff=c_pred["P_K"]["contrast"]["capture_diff"], ci=c_pred["P_K"]["contrast"]["capture_ci"],
                    holds=c_pred["P_K"]["holds"]),
    P_M={m: OrderedDict(diff=v["n1"]["value_diff"], ci=v["n1"]["value_ci"], lo95=v["n1"]["value_lo95"],
                        capture_diff=OrderedDict(diff=v["n1"]["capture_diff"], ci=v["n1"]["capture_ci"]),
                        nmany=OrderedDict(diff=v["n42_reported"]["value_diff"], ci=v["n42_reported"]["value_ci"],
                                          capture_diff=OrderedDict(diff=v["n42_reported"]["capture_diff"],
                                                                   ci=v["n42_reported"]["capture_ci"])))
         for m, v in c_pred["P_M"]["models"].items()},
    heldout_v2_minus_v1={f"{s}:{m}": OrderedDict(diff=v["v2_minus_v1"]["capture_diff"], ci=v["v2_minus_v1"]["capture_ci"])
                         for s, by in c_pred["s0b_n1"].items() for m, v in by.items()},
    heldout_120b_n0_v2_minus_v1=OrderedDict(
        diff=S0C["s0b"][P0]["contrasts"][NEW]["gpt-oss-120b"]["v2_minus_v1_single_n0"]["capture_diff"],
        ci=S0C["s0b"][P0]["contrasts"][NEW]["gpt-oss-120b"]["v2_minus_v1_single_n0"]["capture_ci"]))

# ------------------------------------------------------------------------------------------------------------------
# supplement: design
# ------------------------------------------------------------------------------------------------------------------
SCR = r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.08}"
ENDG = r"\endgroup"
RAG = r">{\raggedright\arraybackslash}"


def pool(M):
    g = S0["groups"][gname(M, 0, "react")]
    taus = [k[7:] for k in g["cells"]["nonllm"]["arms"] if k.startswith("fixed__")]
    return sorted(taus, key=lambda t: POOL_ORDER.index(t) if t in POOL_ORDER else len(POOL_ORDER))


def heldout_sets():
    sets = OrderedDict()
    for d in (S0, S0B):
        for g in d["groups"].values():
            m = g["meta"]
            sets.setdefault(m["held_out_set"], (m["M"], m["held_out"]))
    return OrderedDict(sorted(sets.items(), key=lambda kv: SET_ORDER.index(kv[0]) if kv[0] in SET_ORDER else 99))


SET_ORDER = ["react2", "diag3", "react4", "react8", "diag6"]


def ho_order(gn):
    m = S0B["groups"][gn]["meta"]
    return (-m["M"], SET_ORDER.index(m["held_out_set"]) if m["held_out_set"] in SET_ORDER else 99, m["reward_noise"])


design = [SCR, r"\begin{longtable}{@{}" + RAG + r"p{0.17\linewidth}" + RAG + r"p{0.78\linewidth}@{}}\toprule item & design \\ \midrule \endhead"]
design += [
    r"family & Each opponent type combines a favourite throw (rock, paper or scissors) with a reactivity: habit plays the "
    r"favourite with probability $s$, beat-last plays the throw that beats the player's previous throw with probability "
    r"$s$, and copy-last repeats it. At $M{=}18$ a timing trait is added: stable types keep their behaviour, flip types "
    r"change it at the halfway round. $s{=}" + f"{S0['groups'][P0]['meta']['sharpness']:g}" + r"$; reward noise "
    r"$\sigma\in\{0,0.5\}$ is common to every arm on a seed. \\",
    r"episode & $30$ rounds of rock-paper-scissors; the default strategy ROCK plays rounds $1$--$5$, then the selector "
    r"picks one pool strategy for rounds $6$--$30$. \\",
    r"pool & $M{=}9$, $K{=}" + str(len(pool(9))) + r"$: " + ", ".join(tex(p) for p in pool(9)) + r". $M{=}18$, $K{=}" +
    str(len(pool(18))) + r"$: " + ", ".join(tex(p) for p in pool(18)) + r". Every pool strategy is the best response "
    r"to at least one type. \\",
    r"briefing & One sentence per trait in natural language, with six hand-written phrasings per trait value; it "
    r"describes the opponent and never names a strategy or a counter. Phrasings $0$--$2$ are seen in calibration; "
    r"held-out types are briefed in phrasings $0$--$2$ under seen words or $3$--$5$ under new words. \\",
]
for name, (M, types) in heldout_sets().items():
    design.append(f"held out, {tex(name)} & $M{{={M}}}$: " + ", ".join(tex(t) for t in types) + r". \\")
nF = S0["groups"][P0]["reference"]["seeds_n"]
nH = S0B["groups"][P0]["reference"]["seeds_n"]
design += [
    r"deployment & Most agents known: seeds $0$--$" + str(nF - 1) + r"$, types drawn uniformly over all $M$, calibration "
    r"on the seen types only. New agents: $" + str(nH) + r"$ seeds drawn uniformly over the held-out set, with $\Vstar$ "
    r"and $\HD$ recomputed on that population. Learners run $300$ episodes. \\",
    r"text readers & keyword rule: a hand-written rule over the seen phrasings; TF-IDF: word and character n-grams with "
    r"logistic regression, fitted once on every seen phrasing of every seen type with the best response estimated on "
    r"calibration data as label; embedding: a small sentence-embedding model with a nearest-neighbour or logistic classifier on the same "
    r"rows; text LinUCB: LinUCB on the embedded briefing, learning online without offline fitting; matched TF-IDF "
    r"and embedding: refitted on exactly the labelled rows of the $N$-shot prompt. \\",
    r"factorized reader & One logistic classifier per trait on the TF-IDF features of the briefing, or on its "
    r"sentence embedding, then a composer. FCR, the fitted composer: a ridge regression of each pool strategy's "
    r"calibration value on the one-hot traits, with main effects or pairwise interactions, picks the strategy of highest "
    r"predicted value. FCR-sym, the simulator composer: every pool strategy plays the scripted opponent of the predicted "
    r"traits in the game simulator under the deployment's episode protocol, $" + str(FS["meta"]["composer"]["rollouts"]) +
    r"$ rollouts each on simulator seeds $" + f"{FS['meta']['composer']['seed_base']:,}".replace(",", "{,}") +
    r"+r$ shared by every strategy and tuple, and the highest mean is played, ties going to the earliest pool strategy. "
    r"FCR-full and FCR-sym-full are fitted to every known phrasing of every known type, FCR-N and FCR-sym-N to exactly "
    r"the labelled rows of a one-example prompt; the trait oracle feeds the true traits to a composer and is a privileged "
    r"diagnostic. \\",
    r"prompts & bare: the pool docs, the default and the briefing, without the game, the objective or whose pool it is; "
    r"task: states the rules and payoffs, the objective, that the pool lists the player's own strategies and what the "
    r"default does, and asks for \emph{Final:} followed by one pool name. $N{=}1$ gives one labelled briefing per known "
    r"type, $" + str(known_types(P0)) + r"$ examples with " + str(known_types(P0)) + r" known types and $" +
    str(known_types(R8)) + r"$ with " + str(known_types(R8)) + r"; $N{=}" + str(3 * known_types(P0)) + r"$ and $N{=}" +
    str(3 * known_types(R8)) + r"$ give three per known type; the examples come in shuffled order. \\",
    r"reasoning & default: low effort for \texttt{gpt-oss}, thinking off for \texttt{Qwen3.8-27B}, $1024$ completion "
    r"tokens; medium effort with $4096$ tokens; high effort and \texttt{Qwen} thinking with $16{,}384$ tokens in a "
    r"$32{,}768$-token context; high effort and thinking with $4096$ tokens measure truncation. \\",
    r"\bottomrule\end{longtable}" + ENDG, "",
    SCR,
    r"\begin{longtable}{@{}" + RAG + r"p{0.24\linewidth}rrrrrr@{}}\toprule deployment & types & $\Vstar$ & $\HD$ & "
    r"share & prefix ceiling & predicted $\Tstar$ \\ \midrule \endhead",
]
for M in (9, 18):
    g = S0["groups"][gname(M, 0, "react")]
    design.append(f"most agents known, $M{{={M}}}$ & {M} & ${f(g['exact']['V_star'])}$ & ${f(g['exact']['H_D'])}$ & "
                  f"${f(g['exact']['share'])}$ & ${f(g['exact']['prefix_ceiling_bayes_capture'])}$ & -- \\\\")
for gn in sorted(S0B["groups"], key=ho_order):
    g = S0B["groups"][gn]
    m = g["meta"]
    design.append(f"new agents, {tex(m['held_out_set'])}, $\\sigma{{={m['reward_noise']:g}}}$ & "
                  f"{len(g['reference_0b']['deployment_types'])} & ${f(g['exact']['V_star'])}$ & ${f(g['exact']['H_D'])}$ & "
                  f"${f(g['exact']['share'])}$ & ${f(g['exact']['prefix_ceiling_bayes_capture'])}$ & "
                  f"${f(g['reference_0b']['predicted_T_star'], 1)}$ \\\\")
design.append(r"\bottomrule\end{longtable}" + ENDG)
design.append(r"{\scriptsize Exact values over the deployed types. Prefix ceiling: the capture of the best average counter "
              r"given the five-round prefix, which bounds every reader of the prefix. Predicted $\Tstar$: the sample-complexity "
              r"scale $n K/\mathrm{snr}^2$ for the deployments of new agents.}")
(OUT / "si_briefing_design.tex").write_text("\n".join(design), encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------------
# supplement: prediction ledger
# ------------------------------------------------------------------------------------------------------------------
G0 = S0["groups"]
G0B = S0B["groups"]
react0 = [g for g in G0.values() if g["meta"]["kind"] == "react"]


def hitword(ok, partial=False):
    return "partial" if partial else ("hit" if ok else "miss")


ceil_hi = {M: max(g["prefix_bayes_ceiling"]["capture_ci"][1] for g in G0.values() if g["meta"]["M"] == M) for M in (9, 18)}
share = {M: next(g["exact"]["share"] for g in G0.values() if g["meta"]["M"] == M) for M in (9, 18)}
pa_ok = all(v < 1 for v in ceil_hi.values()) and round(share[18], 6) >= round(share[9], 6)
pb_ok = all(g["predictions"]["P_B"]["all_at_or_below_ceiling"] and g["predictions"]["P_B"]["scripted_reactive_negative"]
            for g in G0.values()) and all(g["predictions"]["P_B"]["bayes_top2_is_habit_R_P"] for g in G0.values() if g["meta"]["M"] == 9)
pc_fail = sorted({(g["meta"]["M"], k.split("/")[1]) for g in react0 for k, v in g["predictions"]["P_C"].items()
                  if (k.startswith("sameword/") and not v["within_noise"]) or
                  (k in ("newword/scripted_text", "newword/text_bow") and not v["held_out_below_0.5"])})
pd_cells = [gn for gn, g in G0.items() if g["predictions"]["P_D"]["holds"]]
pf18 = [g["predictions"]["P_F"]["holds"] for g in G0.values() if g["meta"]["M"] == 18]
pf9 = [g["predictions"]["P_F"]["holds"] for g in G0.values() if g["meta"]["M"] == 9]
emb18 = G0[P0]["predictions"]["P_C"]["sameword/text_embed"]["held_out_capture"]
lin9 = max(g["predictions"]["P_F"]["arms"]["nonllm/linucb"]["capture_at"]["100"] for g in G0.values() if g["meta"]["M"] == 9)
pd_best = max(((gn, k, v) for gn in pd_cells for k, v in G0[gn]["predictions"]["P_D"]["arms"].items() if v["holds"]),
              key=lambda t: t[2]["capture"])
pe = S0["P_E"]["primary"]["gpt-oss-120b"]
pg_ok = [g["predictions"]["P_G"]["holds"] for g in G0B.values()]
ph_ok = [g["predictions"]["P_H"]["holds"] for g in G0B.values()]
pi_ok = [g["predictions"]["P_I"]["within_noise"] for g in G0B.values() if g["predictions"].get("P_I")]
pi0 = G0B[P0]["predictions"]["P_I"]
pe_b = S0B["P_E"]["primary"]["gpt-oss-120b"]
R = readers
pm = c_pred["P_M"]["models"]
pn = S0C["classic"]["P_N"]

ledger = [
    ("most agents known", None),
    ("P-A", "Both families pass the heterogeneity gate; the full-state $\\HD$ exceeds the prefix-only ceiling; the $M{=}18$ "
            "share is no smaller than the $M{=}9$ share.", hitword(pa_ok),
     f"ceiling upper bounds ${f(ceil_hi[18])}$ ($M{{=}}18$) and ${f(ceil_hi[9])}$ ($M{{=}}9$); share ${share[18]:.4f}$ in both"),
    ("P-B", "Every prefix reader stays at or below the prefix ceiling; the prefix-Bayes losses fall on the minority habit "
            "types at $M{=}9$; the prefix rule is negative on reactive types.", hitword(pb_ok),
     f"prefix rule on reactive types ${f(G0[P0]['predictions']['P_B']['scripted_reactive_capture'])}$ ($M{{=}}18$), "
     f"${f(G0[gname(9, 0, 'react')]['predictions']['P_B']['scripted_reactive_capture'])}$ ($M{{=}}9$)"),
    ("P-C", "Under seen words the text readers read held-out combinations as well as seen ones; under new words the keyword "
            "rule and TF-IDF fall below $0.5$.", hitword(not pc_fail, partial=bool(pc_fail)),
     "fails only for " + "; ".join(f"{NONLLM[k]} at $M{{={M}}}$" for M, k in pc_fail) +
     f" under seen words, held-out capture ${f(emb18)}$" if pc_fail else "every clause holds"),
    ("P-D", "Some model reaches capture $0.5$ on the briefing, no worse on held-out combinations, and significantly above "
            "its prefix-only reading.", hitword(bool(pd_cells)),
     f"holds in {len(pd_cells)} of {len(G0)} cells; best {arm_label(pd_best[1].split('/')[1])}"
     f"{' under seen words' if pd_best[1].startswith(SAME) else ''} ${f(pd_best[2]['capture'])}$, "
     f"briefing minus prefix lower bound ${f(pd_best[2]['briefing_minus_semantic']['lo95'])}$"),
    ("P-E", "Under new words some language-model arm pays; primary: \\texttt{120b}, \\texttt{20b} and \\texttt{Qwen} "
            "zero-shot with most agents known.", hitword(S0["P_E"]["primary_holds"]),
     f"\\texttt{{120b}} ${f(pe['capture'])}$, $\\Delta={f(pe['vs_strongest']['diff'], plus=True)}$ against TF-IDF; "
     f"{S0['P_E']['secondary_counts']['pays_yes']} of {S0['P_E']['secondary_counts']['llm_arm_entries']} arms pay"),
    ("P-F", "At $M{=}18$ no library-free bandit reaches capture $0.5$ by $T{=}300$; at $M{=}9$ LinTS or LinUCB does by "
            "$T{=}100$.", hitword(all(pf18) and all(pf9), partial=all(pf18) != all(pf9)),
     f"$M{{=}}18$ holds in {sum(pf18)} of {len(pf18)} cells; $M{{=}}9$ in {sum(pf9)} of {len(pf9)}, best LinUCB "
     f"${f(lin9)}$ at $T{{=}}100$"),
    ("new agents", None),
    ("P-E$'$", "Primary: the verdict of \\texttt{120b} zero-shot under the bare prompt, four combinations, $\\sigma{=}0$.",
     "verdict " + pe_b["pays"],
     f"${f(pe_b['capture'])}$; $\\Delta={f(pe_b['vs_strongest']['diff'], plus=True)}$ against "
     f"{NONLLM[pe_b['vs_strongest_arm'].split('/')[1]]}; secondary {S0B['P_E']['secondary_counts']['pays_yes']} of "
     f"{S0B['P_E']['secondary_counts']['llm_arm_entries']} pass uncorrected, {len(S0B['P_E']['holm_surviving']['pays'])} survive "
     f"Holm"),
    ("P-G", "TF-IDF and the keyword rule stay below $0.5$ under new words.", hitword(all(pg_ok)),
     f"{sum(pg_ok)} of {len(pg_ok)} cells; TF-IDF ${f(R['text_bow']['min'])}$ to ${f(R['text_bow']['max'])}$, keyword "
     f"${f(R['scripted_text']['min'])}$ to ${f(R['scripted_text']['max'])}$, embedding ${f(R['text_embed']['min'])}$ to "
     f"${f(R['text_embed']['max'], plus=True)}$"),
    ("P-H", "A library-free bandit crosses capture $0.5$ within $300$ episodes exactly where the predicted $\\Tstar$ is "
            "below $300$.", hitword(all(ph_ok), partial=0 < sum(ph_ok) < len(ph_ok)),
     f"{sum(ph_ok)} of {len(ph_ok)} cells; no bandit crosses at $M{{=}}18$, where the best ends at "
     f"${f(min(best_end))}$ to ${f(max(best_end))}$"),
    ("P-I", "\\texttt{120b} facing only new agents matches its reading of the same types when most agents are known.",
     hitword(all(pi_ok)), f"{sum(pi_ok)} of {len(pi_ok)} cells; primary cell difference "
                          f"{ci_math(pi0['diff'], pi0['diff_ci'], plus=True)}"),
    ("prompt, examples and reasoning", None),
    ("P-J", "The task prompt raises \\texttt{120b} zero-shot with most agents known by at least $" +
     f"{c_pred['P_J']['threshold']:+.3f}" + "$.", hitword(c_pred["P_J"]["holds"]),
     "difference " + ci_math(c_pred["P_J"]["contrast"]["capture_diff"], c_pred["P_J"]["contrast"]["capture_ci"], plus=True)),
    ("P-K", "Under the task prompt one example per seen type does not lower \\texttt{120b}.", hitword(c_pred["P_K"]["holds"]),
     "$N{=}1$ minus $N{=}0$ " + ci_math(c_pred["P_K"]["contrast"]["capture_diff"], c_pred["P_K"]["contrast"]["capture_ci"], plus=True)),
    ("P-L", "\\texttt{Qwen} thinking reaches capture $0.5$ at more than ten times the completion tokens.",
     hitword(PL["holds"]) + (", capture clause holds" if PL["holds_capture"] and not PL["holds_cost"] else ""),
     f"capture {ci_math(PL['capture'], PL['capture_ci'])} on {PL['think_cost']['episodes']} seeds; token ratio "
     f"${PL['token_ratio']:.1f}$"),
    ("P-M", "With the same labelled rows, TF-IDF falls below each model at $N{=}1$.", hitword(c_pred["P_M"]["holds"]),
     "; ".join(f"\\texttt{{{TAG[m]}}} ${f(v['n1']['value_diff'], plus=True)}$ (lo ${f(v['n1']['value_lo95'], plus=True)}$)"
               for m, v in pm.items())),
    ("P-N", "The task prompt moves the zero-shot rows of the first-moves sandbox by more than their interval width.",
     "no material change" if not any(v["material"] for v in pn.values()) else "material change",
     "; ".join(f"\\texttt{{{TAG[m]}}} ${f(v['mean_capture_diff'], d=3, plus=True)}$ against ${f(v['stage0_width'], d=3)}$"
               for m, v in pn.items())),
]
led = [SCR, r"\begin{longtable}{@{}l" + RAG + r"p{0.40\linewidth}" + RAG + r"p{0.12\linewidth}" + RAG +
       r"p{0.36\linewidth}@{}}\toprule & prediction & result & key number \\ \midrule \endhead"]
for item in ledger:
    if item[1] is None:
        led.append(r"\multicolumn{4}{@{}l}{\emph{" + item[0] + r"}}\\")
        continue
    led.append(" & ".join(item) + r" \\")
    NUM[f"ledger:{item[0]}"] = OrderedDict(result=item[2], key=item[3])
led.append(r"\bottomrule\end{longtable}" + ENDG)
led.append(r"{\scriptsize Each prediction is frozen before its deployment episodes run, apart from a ten-seed pilot before "
           r"P-A to P-F; the dated pre-registration and "
           r"its amendment record are distributed with the artifact. $\Delta$: paired difference in reward per episode; lo: "
           r"its one-sided $95\%$ lower bound.}")
(OUT / "si_briefing_prereg.tex").write_text("\n".join(led), encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------------
# supplement: every cell, every arm
# ------------------------------------------------------------------------------------------------------------------
CELL_NAME = {"nonllm": "prefix readers and fixed strategies", "semantic": "prefix only", NEW: "briefing, new words",
             SAME: "briefing, seen words", "briefing_prefix-newword": "briefing and prefix, new words",
             "briefing_prefix-sameword": "briefing and prefix, seen words"}


def col_label(g):
    m = g["meta"]
    return f"{tex(m['held_out_set'])}, $\\sigma{{={m['reward_noise']:g}}}$"


def two(a, single=False):
    """Capture and interval, on one line or stacked when the table has many columns."""
    if a is None:
        return "--"
    lo, hi = a["capture_ci"]
    if single:
        return f"${f(a['capture'])}$\\," + grey_ci(lo, hi)
    return (r"\begin{tabular}[t]{@{}r@{}}$" + f(a["capture"]) + r"$\\" + grey_ci(lo, hi) + r"\end{tabular}")


def arm_order(k):
    return (NONLLM_ORDER.index(k) if k in NONLLM_ORDER else len(NONLLM_ORDER) + (1 if k.startswith("fixed__") else 0), k)


def grid(title, d, groups, note):
    single = len(groups) <= 4
    out = [f"\\subsection*{{{title}}}", SCR,
           r"\begin{longtable}{@{}" + RAG + (r"p{0.30\linewidth}" if single else r"p{0.24\linewidth}") + "r" * len(groups) +
           r"@{}}\toprule arm & " +
           " & ".join(col_label(d["groups"][g]) for g in groups) + r" \\ \midrule \endhead"]
    cells = OrderedDict()
    for g in groups:
        for c, block in d["groups"][g]["cells"].items():
            for k in block["arms"]:
                cells.setdefault(c, [])
                if k not in cells[c]:
                    cells[c].append(k)
    out.append(r"\emph{prefix ceiling} & " + " & ".join(two(d["groups"][g]["prefix_bayes_ceiling"], single) for g in groups) + r" \\")
    for c in [x for x in ["nonllm", "semantic", SAME, NEW, "briefing_prefix-sameword", "briefing_prefix-newword"] if x in cells]:
        out.append(r"\multicolumn{" + str(len(groups) + 1) + r"}{@{}l}{\emph{" + CELL_NAME[c] + r"}}\\")
        for k in sorted(cells[c], key=arm_order):
            out.append(arm_label(k, short=False) + " & " +
                       " & ".join(two(d["groups"][g]["cells"].get(c, {}).get("arms", {}).get(k), single) for g in groups) + r" \\")
    out.append(r"\bottomrule\end{longtable}" + ENDG)
    out.append(r"{\scriptsize " + note + "}")
    return out


cells_tex = []
order0 = lambda M: [gname(M, s, k) for s in (0, 0.5) for k in ("react", "diag")]
cells_tex += grid("Most agents known, $M{=}18$", S0, order0(18),
                  f"Captures with $95\\%$ bootstrap intervals on seeds $0$--${nF - 1}$; learners at $T{{=}}100$ on the paired "
                  "scale. The prefix ceiling is evaluated on the realised prefixes.")
cells_tex += grid("Most agents known, $M{=}9$", S0, order0(9), "As above.")
order0b18 = [gname(18, s, k) for s in (0, 0.5) for k in ("react", "react8", "diag")]
order0b9 = [gname(9, s, k) for s in (0, 0.5) for k in ("react", "diag")]
cells_tex += grid("New agents, $M{=}18$", S0B, order0b18,
                  f"Captures with $95\\%$ bootstrap intervals on {nH} seeds of new agents; learners over $300$ episodes.")
cells_tex += grid("New agents, $M{=}9$", S0B, order0b9, "As above.")

parts = [("s0", P0, "most known"), ("s0b", P0, "four new"), ("s0b", R8, "eight new")]
keys = []
for part, group, _ in parts:
    for ck in S0C[part][group]["arms"]:
        k = ck.split("/", 1)[1]
        if (variant_of(k) or "_matched__" in k) and ck not in keys:
            keys.append(ck)
keys.sort(key=lambda ck: (ck.split("/")[0] != NEW, "_matched__" in ck, ck))
c_tab = [r"\subsection*{Prompt, examples and reasoning, $M{=}18$, $\sigma{=}0$}", SCR,
         r"\begin{longtable}{@{}" + RAG + r"p{0.34\linewidth}" + "rr" * len(parts) + r"@{}}\toprule arm & " +
         " & ".join(r"\multicolumn{2}{c}{" + t + "}" for *_, t in parts) + r" \\ & " +
         " & ".join(r"$n$ & capture" for _ in parts) + r" \\ \midrule \endhead"]
cur = None
for ck in keys:
    c, k = ck.split("/", 1)
    if c != cur:
        c_tab.append(r"\multicolumn{" + str(1 + 2 * len(parts)) + r"}{@{}l}{\emph{" + CELL_NAME[c] + r"}}\\")
        cur = c
    cols = []
    for part, group, _ in parts:
        a = S0C[part][group]["arms"].get(ck)
        cols += [str(a["n"]) if a else "--", two(a, single=True)]
    c_tab.append(arm_label(k, short=False) + " & " + " & ".join(cols) + r" \\")
c_tab.append(r"\bottomrule\end{longtable}" + ENDG)
c_tab.append(r"{\scriptsize Most known: seeds $0$--$" + str(S0C["s0"][P0]["reference"]["seeds_n"] - 1) +
             r"$; four and eight new: $" + str(S0C['s0b'][P0]['reference']['seeds_n']) + r"$ seeds of new agents, or seeds "
             r"$0$--$99$ for the arms with a $16{,}384$-token budget. The bare-prompt arms of these cells are in the tables "
             r"above.}")
cells_tex += c_tab
(OUT / "si_briefing_cells.tex").write_text("\n".join(cells_tex), encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------------
# supplement: cost of the reasoning arms
# ------------------------------------------------------------------------------------------------------------------
WHERE = {("s0", P0): "most known", ("s0b", P0): "four new", ("s0b", R8): "eight new"}
cost = [SCR, r"\begin{longtable}{@{}l" + RAG + r"p{0.22\linewidth}" + RAG + r"p{0.13\linewidth}rrrrrrrr@{}}\toprule model & arm & "
        r"deployment & $n$ & capture & \multicolumn{2}{c}{completion tokens} & \multicolumn{2}{c}{latency (s)} & length & "
        r"fallback \\ & & & & & mean & p90 & mean & p90 & stops & \\ \midrule \endhead"]
for m, var, part, group, cell, k, v in cost_rows():
    cs = v["cost"]
    where = WHERE[(part, group)] + (", seen words" if cell == SAME else "")
    cost.append(" & ".join([MODEL_LONG[m], arm_label(k, short=False).split("} ", 1)[1], where, str(v["n"]),
                            f"${f(v['capture'])}$", f"{cs['mean_completion_tokens']:,.1f}", f"{cs['p90_completion_tokens']:,.0f}",
                            f"{cs['mean_latency_s']:.1f}", f"{cs['p90_latency_s']:.1f}", f(cs.get("length_finish_rate"), 3),
                            f(cs.get("fallback_rate"), 2)]) + r" \\")
cost.append(r"\bottomrule\end{longtable}" + ENDG)
cost.append(r"{\scriptsize Per-episode means and 90th percentiles over the recorded calls; length stops: share of calls that "
            r"end at the token limit; fallback: share of episodes that fall back to the default after three attempts. "
            r"\texttt{gpt-oss-20b} at high effort is not run. Latency is measured on one RTX PRO 6000 under concurrent load.}")
(OUT / "si_briefing_cost.tex").write_text("\n".join(cost), encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------------
# supplement: classic sandbox under the two prompts
# ------------------------------------------------------------------------------------------------------------------
import re as _re


def classic_label(c):
    m = _re.match(r"sandbox-M(\d+)-K\d+-sharp([0-9.]+)-noise([0-9.]+)", c)
    return f"$M{{={m.group(1)}}}$, $s{{={m.group(2)}}}$, $\\sigma{{={float(m.group(3)):g}}}$"


cl = [SCR, r"\begin{longtable}{@{}l" + "rrr" * len(MODELS) + r"@{}}\toprule cell & " +
      " & ".join(r"\multicolumn{3}{c}{" + MODEL_LONG[m] + "}" for m, _ in MODELS) + r" \\ & " +
      " & ".join("bare & task & difference" for _ in MODELS) + r" \\ \midrule \endhead"]
for c, e in sorted(S0C["classic"]["cells"].items(), key=lambda kv: kv[0]):
    vals = []
    for m, _ in MODELS:
        x = e["models"][m]
        vals += [f"${f(x['v1']['capture'])}$", f"${f(x['v2']['capture'])}$",
                 ci_math(x["v2_minus_v1"]["capture_diff"], x["v2_minus_v1"]["capture_ci"], plus=True)]
    cl.append(classic_label(c) + " & " + " & ".join(vals) + r" \\")
cl.append(r"\midrule")
cl.append("mean of the 8 cells & " + " & ".join(
    f"\\multicolumn{{3}}{{r}}{{{ci_math(pn[m]['mean_capture_diff'], pn[m]['mean_ci'], d=3, plus=True)}, width ${f(pn[m]['stage0_width'], 3)}$}}"
    for m, _ in MODELS) + r" \\")
cl.append(r"\bottomrule\end{longtable}" + ENDG)
cl.append(r"{\scriptsize Zero-shot captures of the first-moves sandbox on seeds $0$--$99$ against its fixed-strategy reference; "
          r"bare: the bare prompt, as in the master table; task: the task prompt; difference: paired by seed with a $95\%$ bootstrap interval. "
          r"Width: the $95\%$ interval width of the model's zero-shot capture with most agents known in the briefing "
          r"game, the pre-registered threshold for a material change.}")
(OUT / "si_briefing_classic.tex").write_text("\n".join(cl), encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------------
# supplement: per-type hit rates, four combinations held out, held-out-only deployment
# ------------------------------------------------------------------------------------------------------------------
PT_ARMS = [("heldout", NEW, "llm__gpt-oss-120b__single__nshot0", "bare, new words"),
           ("heldout", SAME, "llm__gpt-oss-120b__single__nshot0", "bare, seen words"),
           ("heldout", NEW, "llm__gpt-oss-120b__single__nshot0__v2", "task, new words"),
           ("heldout", NEW, "llm__gpt-oss-120b__single__nshot1__v2", "task $N{=}1$"),
           ("heldout", NEW, "llm__qwen3.8-27b__single__nshot1", "bare $N{=}1$"),
           ("heldout", NEW, "llm__qwen3.8-27b__single__nshot1__v2", "task $N{=}1$")]
ref_pt = S0B["groups"][P0]["reference"]["per_type"]
types = list(ref_pt.keys())
pt = [SCR, r"\begin{longtable}{@{}" + RAG + r"p{0.22\linewidth}r" + "r" * len(PT_ARMS) + r"@{}}\toprule & & "
      r"\multicolumn{4}{c}{\texttt{gpt-oss-120b}} & \multicolumn{2}{c}{\texttt{Qwen3.8-27B}} \\ held-out type (best response) "
      r"& $n$ & " + " & ".join(lab for *_, lab in PT_ARMS) + r" \\ \midrule \endhead"]
NUM["pertype:react4"] = OrderedDict(src="new_agents.json:groups.<react4 sigma0>.cells.<cell>.arms.<key>.per_type.<type>.p_mis "
                                        "(hit = 1 - p_mis); task_prompt.json strata.held_out.hit_rate where per_type is absent")
recs = [find(dep, P0, c, k) for dep, c, k, _ in PT_ARMS]
for t in types:
    cells = []
    for (rec, _), (dep, c, k, lab) in zip(recs, PT_ARMS):
        p = (rec or {}).get("per_type", {}).get(t)
        cells.append(f"${f(1 - p['p_mis'])}$" if p else "--")
        if p:
            NUM["pertype:react4"][f"{c}/{k}:{t}"] = 1 - p["p_mis"]
    pt.append(f"{tex(t)} ({tex(ref_pt[t]['best_tau'])}) & {ref_pt[t]['n']} & " + " & ".join(cells) + r" \\")
pt.append(r"\midrule")
pooled = []
for (rec, _), (dep, c, k, lab) in zip(recs, PT_ARMS):
    h = ((rec or {}).get("strata") or {}).get("held_out", {})
    pooled.append(f"${f(h.get('hit_rate'))}$" if h.get("hit_rate") is not None else "--")
    NUM["pertype:react4"][f"{c}/{k}:all"] = OrderedDict(hit_rate=h.get("hit_rate"), hit_ci=h.get("hit_ci"))
pt.append(f"all held-out types & {S0B['groups'][P0]['reference']['seeds_n']} & " + " & ".join(pooled) + r" \\")
pt.append(r"\bottomrule\end{longtable}" + ENDG)
pt.append(r"{\scriptsize Hit rate: the share of episodes whose chosen strategy scores within $0.5$ of the type's best "
          r"response. New agents: only the four held-out combinations are deployed, at $M{=}18$, $\sigma{=}0$. Per-type hit "
          r"rates are recorded for the bare-prompt arms; the task-prompt arms report the pooled rate.}")
(OUT / "si_briefing_pertype.tex").write_text("\n".join(pt), encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------------
# supplement: the factorized compositional reader (fcr.json)
# ------------------------------------------------------------------------------------------------------------------
FD = FCR["deployments"]


def fam_status(dk, ak, release=False):
    """Test status, in its briefing family, of a language-model arm of fcr.json: dk 'all/<group>/newword' or
    'heldout/<group>/newword', ak '<cell>:<arm key>', the arm read in the new-wording cell; release=True: against the
    original candidates alone, which the fitted composer leaves unchanged."""
    scope, group, _ = dk.split("/")
    return test_status("full" if scope == "all" else "heldout", group, NEW, ak.split(":", 1)[1], release=release)


PRIM_LLM = OrderedDict(FCR["meta"]["primary_llm_arms"])          # label -> "<cell>:<key>"
PRIM_LLM["120b task 1/type"] = "task:llm__gpt-oss-120b__single__nshot1__v2"   # paired with FCR-N task rows only
LLM_TEX = {"120b task 0/type": r"\texttt{120b} task, 0/type", "Qwen bare 1/type": r"\texttt{Qwen} bare, 1/type",
           "Qwen task 1/type": r"\texttt{Qwen} task, 1/type", "120b task 1/type": r"\texttt{120b} task, 1/type"}


def arm_name_any(k):
    """Readable name of an arm key of fcr.json or fcr_sym.json ('nonllm:linucb', 'briefing:text_bow', 'fcr_full',
    'sym_full', ...)."""
    if k in FCR_LONG:
        return FCR_LONG[k]
    if k in SYM_LONG:
        return SYM_LONG[k]
    base = k.split(":", 1)[-1].split("/", 1)[-1]
    return NONLLM.get(base, tex(base))


def fcr_cap(a, release=False):
    src = a.get("release") if release and a.get("release") else a
    lo, hi = src["capture_ci"]
    return f"${f(src['capture'])}$\\," + grey_ci(lo, hi)


def lo_s(x):
    return f"${f(x, plus=True)}$"


fcr_tex = []
# (1) leave-one-combination-out cross-validation inside calibration
COND_NAME = {"full": "all known briefings", "n_bare": "1/type bare rows", "n_task": "1/type task rows"}
fcr_tex += [r"\subsubsection*{Leave-one-combination-out cross-validation inside calibration}", SCR,
            r"\begin{longtable}{@{}llrlrrrrr@{}}"
            r"\caption{FCR cross-validation inside calibration, end to end.}\label{tab:fcr-cv}\\"
            r"\toprule setting & condition & rows & C & \multicolumn{3}{c}{TF-IDF} & embedding & whole-type \\ "
            r"& & & & tuple & counter, main & counter, pairwise & tuple & TF-IDF counter \\ \midrule \endfirsthead"
            r"\toprule setting & condition & rows & C & \multicolumn{3}{c}{TF-IDF} & embedding & whole-type \\ "
            r"& & & & tuple & counter, main & counter, pairwise & tuple & TF-IDF counter \\ \midrule \endhead"]
fcr_cv_num = OrderedDict()
for s, cv in FCR["calibration"]["loco_cv_summary"].items():
    for cond in ("full", "n_bare", "n_task"):
        c = cv["conditions"].get(cond)
        if not c:
            continue
        tf, em = c["tfidf"], c["embed"]
        Cs = "/".join(f"{tf['C'][t]:g}" for t in ("favourite", "reactivity", "timing") if t in tf["C"])
        fcr_tex.append(" & ".join([tex(s), COND_NAME[cond], str(c["n_rows"]), Cs,
                                   f"${f(tf['end_to_end']['main']['tuple'])}$",
                                   f"${f(tf['end_to_end']['main']['counter'])}$", f"${f(tf['end_to_end']['pairwise']['counter'])}$",
                                   f"${f(em['end_to_end']['main']['tuple'])}$",
                                   f"${f(c['whole_type_tfidf_loco_counter_accuracy'])}$"]) + r" \\")
        fcr_cv_num[f"{s}:{cond}"] = OrderedDict(rows=c["n_rows"], tuple=tf["end_to_end"]["main"]["tuple"],
                                                counter_main=tf["end_to_end"]["main"]["counter"],
                                                counter_pairwise=tf["end_to_end"]["pairwise"]["counter"],
                                                embed_tuple=em["end_to_end"]["main"]["tuple"],
                                                whole_type_counter=c["whole_type_tfidf_loco_counter_accuracy"])
fcr_tex.append(r"\bottomrule\end{longtable}" + ENDG)
fcr_tex.append(r"{\scriptsize C: the chosen inverse regularisation of the favourite, reaction and timing classifiers. "
               r"Tuple: the share of left-out rows whose three traits are all read correctly; counter: the share whose "
               r"counter, picked by the composer from the predicted traits, is the best response estimated on calibration "
               r"data.\par}")
fcr_tex += ["", SCR, r"\begin{longtable}{@{}llrrr@{}}"
            r"\caption{The FCR composer given the true traits, inside calibration.}\label{tab:fcr-composer}\\"
            r"\toprule setting & composer & $\alpha$ & left out & fit \\ \midrule \endfirsthead"
            r"\toprule setting & composer & $\alpha$ & left out & fit \\ \midrule \endhead"]
for s, cv in FCR["calibration"]["loco_cv_summary"].items():
    for form, name in (("main", "main effects"), ("pairwise", "pairwise")):
        cp = cv["composer"][form]
        fcr_tex.append(f"{tex(s)} & {name} & ${cp['alpha']:g}$ & ${f(cp['loco_counter_accuracy'])}$ & "
                       f"${f(cp['train_counter_accuracy_seen'])}$ \\\\")
        fcr_cv_num[f"{s}:composer:{form}"] = OrderedDict(alpha=cp["alpha"], loco=cp["loco_counter_accuracy"],
                                                         fit=cp["train_counter_accuracy_seen"])
fcr_tex.append(r"\bottomrule\end{longtable}" + ENDG)
fcr_tex.append(r"{\scriptsize $\alpha$: the chosen ridge penalty; left out: the counter accuracy on the left-out "
               r"combination; fit: the counter accuracy on the known types it was fitted to.\par}")

# (1b) the simulator composer inside calibration (fcr_sym_calib.json): run before any of its deployment choices
SC = FSC["summary"]
assert SC["lock_ties_at_top_seen"] == 0
assert max(SC["max_abs_dev_mean_vs_value_table_sigma0"], SC["max_abs_dev_difference_vs_value_table"]) < 1e-13
_CP = FS["meta"]["composer"]
_SEED = f"{_CP['seed_base']:,}".replace(",", "{,}")
_EPS = "10^{" + str(int(round(math.log10(_CP["tie_eps"])))) + "}"
fcr_tex += ["", (
    f"The simulator composer averages each pool strategy's return over ${_CP['rollouts']}$ rollouts on the simulator "
    f"seeds ${_SEED}+r$, $r=0,\\dots,{_CP['rollouts'] - 1}$, shared by every strategy and every trait "
    f"tuple, and treats strategies within ${_EPS}$ of the maximum as tied. Inside calibration, before any of "
    f"its deployment choices, fed the true traits of each "
    f"known type, it reproduces the best response of the calibration estimate in ${SC['checks'] - SC['mismatches']}$ "
    f"of ${SC['checks']}$ checks over the ${SC['locks']}$ calibration estimates, with no tie at the top and a smallest "
    f"gap of ${SC['min_simulated_gap_best_second']:.1f}$ reward between the best and the second-best strategy; its "
    r"simulated values match the calibration value tables to within $10^{-13}$ reward."), ""]


def c3_of(dk, ak):
    """The briefing_c3 record of a language-model arm of fcr.json or fcr_sym.json ('<cell>:<key>' in deployment dk)."""
    scope, group, _ = dk.split("/")
    return C3V.get("s0" if scope == "all" else "s0b", group, NEW, ak.split(":", 1)[1])


def any_name(k):
    return SYM_LONG[k] if k in SYM_LONG else arm_name_any(k)


def any_name_inline(k):
    """The arm name inside a sentence: FCR-sym-full (TF-IDF traits) rather than FCR-sym-full, TF-IDF."""
    n = any_name(k)
    return "{} ({} traits)".format(*n.split(", ", 1)) if k in SYM_LONG and ", " in n else n


# (2) the three primary deployments
fcr_tex += [r"\subsubsection*{The three primary deployments}", SCR,
            r"\begin{longtable}{@{}" + RAG + r"p{0.32\linewidth}rlrrrl@{}}"
            r"\caption{The factorized reader in the three primary deployments, $M{=}18$, no payoff noise, new types briefed "
            r"in new words.}"
            r"\label{tab:fcr-primary}\\"
            r"\toprule arm & \multicolumn{2}{c}{capture [95\%]} & counter & tuple & $\Delta$ (lo) & verdict \\ \midrule "
            r"\endfirsthead"
            r"\toprule arm & \multicolumn{2}{c}{capture [95\%]} & counter & tuple & $\Delta$ (lo) & verdict \\ \midrule \endhead"]
for lab, dk in FCR_PRIMARY_DEP.items():
    dep, sdep = FD[dk], FS["deployments"][dk]
    ref, st = dep["reference"], dep["strongest"]
    fin = sdep["strongest"]["after_with_fcr_sym"]
    assert sdep["strongest"]["release"] == st["release"] == st["with_fcr"], dk
    fin_name = any_name_inline(fin)
    where = (f"{fin_name}, with or without the factorized readers among the candidates" if fin == st["release"]
             else f"{any_name(st['release'])} without the factorized readers or with the fitted composer alone; "
                  f"{fin_name} with both composers")
    fcr_tex.append(r"\multicolumn{7}{@{}p{0.97\linewidth}@{}}{\emph{" + lab + r"}: " +
                   f"{ref['seeds_n']} seeds, $\\Vstar={f(ref['V_star'])}$, "
                   f"$\\HD={f(ref['H_D'])}$; strongest tested non-LLM arm: {where}" + r"}\\")
    order = (["briefing:text_bow", st["online_bandit"]] + FCR_ARMS + SYM_ARMS
             + ["fcr_oracle", "fcr_oracle_pairwise", "sym_oracle"])
    for k in order:
        a = sdep["arms"][k] if k.startswith("sym_") else dep["arms"][k]
        name = any_name(k) if k != st["online_bandit"] else "strongest bandit: " + arm_name_any(k)
        if k == "briefing:text_bow":
            name = "TF-IDF, whole type"
        rel = not k.startswith(("fcr", "sym"))
        lo, hi = (a["release"] if rel and a.get("release") else a)["capture_ci"]
        cap = (a["release"] if rel and a.get("release") else a)["capture"]
        ta = (a.get("trait_accuracy") or {}).get("tuple")
        fcr_tex.append(" & ".join([name, f"${f(cap)}$", grey_ci(lo, hi), f"${f(a['counter_accuracy'])}$",
                                   f"${f(ta)}$" if ta is not None else "--", "--", "--"]) + r" \\")
    for an, ak in PRIM_LLM.items():
        L = dep["llm"][ak]
        a3 = c3_of(dk, ak)
        fcr_tex.append(" & ".join([LLM_TEX[an], f"${f(L['capture'])}$",
                                   grey_ci(L["capture_ci"][0], L["capture_ci"][1]),
                                   f"${f(L['counter_accuracy'])}$", "--",
                                   f"${f(a3['vs']['diff'], plus=True)}$ " + grey(f"(${f(a3['vs']['lo95'], plus=True)}$)"),
                                   f"{shown_verdict(a3['pays_release'], a3['status_release'])} $\\to$ "
                                   f"{shown_verdict(a3['pays'], a3['status'])}"]) + r" \\")
fcr_tex.append(r"\bottomrule\end{longtable}" + ENDG)
fcr_tex.append(r"{\scriptsize FCR: the fitted composer; FCR-sym: the simulator composer. Counter: the share of episodes on "
               r"the type's best counter; tuple: the share whose three traits are all read correctly; $\Delta$: the paired "
               r"reward difference against the strongest tested non-LLM arm with both composers among the candidates, "
               r"with its one-sided $95\%$ lower bound; verdict: the pre-registered criterion without $\to$ with the "
               r"fifteen non-privileged factorized arms among the candidates; yes\textsuperscript{n}: only the uncorrected "
               r"criterion passes, for an arm that is a secondary test in its briefing family. $^{\S}$Privileged: given "
               r"the true traits, never a candidate.\par}")

# (3) primary tests: the fitted composer's, then the simulator composer's
TEST_TEX = {"T1 pays with FCR: 120b task 0/type": r"T1 \texttt{120b} task, 0/type",
            "T1 pays with FCR: Qwen bare 1/type": r"T1 \texttt{Qwen} bare, 1/type",
            "T1 pays with FCR: Qwen task 1/type": r"T1 \texttt{Qwen} task, 1/type",
            "T2 FCR-full - TF-IDF whole-type": r"T2 FCR-full $-$ TF-IDF, whole type",
            "T3 Qwen bare 1/type - fcr_n_bare": r"T3 \texttt{Qwen} bare, 1/type $-$ FCR-N bare rows",
            "T3 Qwen task 1/type - fcr_n_task": r"T3 \texttt{Qwen} task, 1/type $-$ FCR-N task rows",
            "T3 120b task 1/type - fcr_n_task": r"T3 \texttt{120b} task, 1/type $-$ FCR-N task rows",
            "T1 pays with FCR-sym: 120b task 0/type": r"T1$'$ \texttt{120b} task, 0/type",
            "T1 pays with FCR-sym: Qwen bare 1/type": r"T1$'$ \texttt{Qwen} bare, 1/type",
            "T1 pays with FCR-sym: Qwen task 1/type": r"T1$'$ \texttt{Qwen} task, 1/type",
            "T2' FCR-sym-full - TF-IDF whole-type": r"T2$'$ FCR-sym-full $-$ TF-IDF, whole type",
            "T3' Qwen bare 1/type - sym_n_bare": r"T3$'$ \texttt{Qwen} bare, 1/type $-$ FCR-sym-N bare rows",
            "T3' Qwen task 1/type - sym_n_task": r"T3$'$ \texttt{Qwen} task, 1/type $-$ FCR-sym-N task rows",
            "T3' 120b task 1/type - sym_n_task": r"T3$'$ \texttt{120b} task, 1/type $-$ FCR-sym-N task rows"}
TESTS_HEAD = (r"\toprule deployment & test & estimate & lo & $p$ & holds & Holm \\ \midrule \endfirsthead"
              r"\toprule deployment & test & estimate & lo & $p$ & holds & Holm \\ \midrule \endhead")


def tests_rows(src, final):
    """Rows of a primary-test table; T1 shows the verdict against the original candidates and against the extended set
    (final=True: both composers; final=False: the fitted composer, which changes none)."""
    rows, t3s = [], []
    for lab, tests in src["primary_tests"].items():
        dk = FCR_PRIMARY_DEP[lab]
        for tk, t in tests.items():
            if tk.startswith("T1"):
                a3 = c3_of(dk, src["meta"]["primary_llm_arms"][tk.split(": ", 1)[1]])
                assert t["release_pays"] == a3["pays_release"] and (not final or t["pays"] == a3["pays"]), tk
                after = shown_verdict(t["pays"], a3["status"] if final else a3["status_release"])
                est = (f"{shown_verdict(t['release_pays'], a3['status_release'])} $\\to$ {after}, against "
                       f"{any_name_inline(t['strongest'])}")
                lo, p = t["vs_lo95"], t["p"]
            else:
                est = f"${f(t['diff'], plus=True)}$ ({f(t['capture_diff'], plus=True)})"
                lo, p = t["lo95"], t["p"]
                if tk.startswith("T3"):
                    t3s.append(t)
            rows.append(" & ".join([lab, TEST_TEX[tk], est, lo_s(lo), f"${p:.4f}$", "yes" if t["holds"] else "no",
                                    "yes" if t["holm_within_primary"] else "no"]) + r" \\")
    return rows, t3s


fcr_tex += [r"\subsubsection*{Primary tests}", SCR,
            r"\begin{longtable}{@{}l" + RAG + r"p{0.31\linewidth}" + RAG + r"p{0.24\linewidth}rrcc@{}}"
            r"\caption{The primary tests of the fitted composer.}"
            r"\label{tab:fcr-tests}\\" + TESTS_HEAD]
_rows, _t3 = tests_rows(FCR, final=False)
t3 = [t["diff"] for t in _t3]
fcr_tex += _rows + [r"\bottomrule\end{longtable}" + ENDG]
fcr_tex.append(r"{\scriptsize holds: the test passes uncorrected; Holm: it survives Holm correction within these $21$ "
               r"tests. T1: the pre-registered verdict without $\to$ with the nine fitted-composer arms among the "
               r"candidates, yes\textsuperscript{n} as in Table~\ref{tab:fcr-primary}; T2: FCR-full minus whole-type "
               r"TF-IDF; T3: a one-example language-model arm minus FCR-N fitted to the same labelled rows. Differences in "
               r"reward per episode, capture in parentheses.\par}")
fcr_tex += ["", SCR,
            r"\begin{longtable}{@{}l" + RAG + r"p{0.31\linewidth}" + RAG + r"p{0.24\linewidth}rrcc@{}}"
            r"\caption{The primary tests of the simulator composer.}\label{tab:fcr-sym-tests}\\" + TESTS_HEAD]
_rows, _t3s = tests_rows(FS, final=True)
fcr_tex += _rows + [r"\bottomrule\end{longtable}" + ENDG]
fcr_tex.append(r"{\scriptsize holds and Holm as in Table~\ref{tab:fcr-tests}. T1$'$: the pre-registered verdict without "
               r"$\to$ with all fifteen non-privileged factorized arms among the candidates, in the convention of the "
               r"briefing families; T2$'$: FCR-sym-full minus whole-type TF-IDF; T3$'$: a one-example language-model arm "
               r"minus FCR-sym-N fitted to the same labelled rows. Differences in reward per episode, capture in "
               r"parentheses.\par}")

# (4) outcome letters: relative to the fitted composer, then to the simulator composer
fcr_tex += [r"\subsubsection*{Outcome letters}", SCR,
            r"\begin{longtable}{@{}ll" + RAG + r"p{0.13\linewidth}c" + RAG + r"p{0.25\linewidth}" + RAG +
            r"p{0.17\linewidth}@{}}"
            r"\caption{The pre-registered outcome of each primary language-model arm $L$ relative to the fitted composer.}"
            r"\label{tab:fcr-letters}\\"
            r"\toprule deployment & arm $L$ & verdict with the FCR & letter & decisive contrast & other oracle form \\ "
            r"\midrule \endfirsthead"
            r"\toprule deployment & arm $L$ & verdict with the FCR & letter & decisive contrast & other oracle form \\ "
            r"\midrule \endhead"]
letters = OrderedDict()
for lab, o in FCR["outcomes"].items():
    dk = FCR_PRIMARY_DEP[lab]
    for an, v in o["per_arm"].items():
        ak = FCR["meta"]["primary_llm_arms"][an]
        if "llm_minus_oracle_star" in v:
            star = v["oracle_star"]
            other = "fcr_oracle_pairwise" if star == "fcr_oracle" else "fcr_oracle"
            vo = FD[dk]["llm"][ak]["vs"][other]
            dec = (f"$-$ {FCR_SHORT[star]}: ${f(v['llm_minus_oracle_star']['diff'], plus=True)}$ "
                   f"({f(v['llm_minus_oracle_star']['lo95'], plus=True)})")
            oth = f"{'D' if vo['lo95'] > 0 else 'C'}: ${f(vo['diff'], plus=True)}$ ({f(vo['lo95'], plus=True)})"
            letter = v["letter"]
        elif "llm_minus_fcr_n_star" in v:
            dec = (f"$-$ FCR-N, {FCR_SHORT[v['fcr_n_star']]}: ${f(v['llm_minus_fcr_n_star']['diff'], plus=True)}$ "
                   f"({f(v['llm_minus_fcr_n_star']['lo95'], plus=True)})")
            oth, letter = "--", v["letter"]
        else:
            dec, oth, letter = "zero-shot: no sample-matched FCR-N", "--", "closed"
        letters[f"{lab}:{an}"] = letter
        star_mark = r"$^{\star}$" if an == o["L_star"] else ""
        fcr_tex.append(" & ".join([lab, LLM_TEX[an] + star_mark,
                                   shown_verdict(v["updated_pays"], fam_status(dk, ak, release=True)),
                                   letter, dec, oth]) + r" \\")
fcr_tex.append(r"\bottomrule\end{longtable}" + ENDG)
fcr_tex.append(r"{\scriptsize Verdict: T1, uncorrected, yes\textsuperscript{n} as in Table~\ref{tab:fcr-primary}. "
               r"Decisive contrast: $L$ minus the trait oracle in the composer form of the strongest FCR-full variant when "
               r"the gate is open, $L$ minus the strongest FCR-N variant built from $L$'s rows when it is closed. Other "
               r"oracle form: the letter with the other composer form of the trait oracle. Differences in reward per "
               r"episode, one-sided $95\%$ lower bounds in parentheses. $^{\star}$The arm whose letter is the deployment's "
               r"headline letter, fixed in advance as the primary arm with the highest capture there.\par}")
fcr_tex += ["", SCR,
            r"\begin{longtable}{@{}ll" + RAG + r"p{0.12\linewidth}cc" + RAG + r"p{0.34\linewidth}@{}}"
            r"\caption{The outcome of each primary language-model arm $L$ relative to the simulator composer, beside its "
            r"letter relative to the fitted composer.}"
            r"\label{tab:fcr-sym-letters}\\"
            r"\toprule deployment & arm $L$ & verdict & \multicolumn{2}{c}{letter} & decisive contrast, simulator \\ "
            r"& & & fitted & simulator & \\ \midrule \endfirsthead"
            r"\toprule deployment & arm $L$ & verdict & \multicolumn{2}{c}{letter} & decisive contrast, simulator \\ "
            r"& & & fitted & simulator & \\ \midrule \endhead"]
sym_letters = OrderedDict()
for lab, o in FS["outcomes"].items():
    dk = FCR_PRIMARY_DEP[lab]
    for an, v in o["per_arm"].items():
        a3 = c3_of(dk, FS["meta"]["primary_llm_arms"][an])
        assert v["after_pays"] == a3["pays"], (lab, an)
        if "llm_minus_oracle_star" in v:
            assert v["oracle_star"] == "sym_oracle"
            dec = (f"$-$ trait oracle, simulator: ${f(v['llm_minus_oracle_star']['diff'], plus=True)}$ "
                   f"({f(v['llm_minus_oracle_star']['lo95'], plus=True)})")
            letter = v["letter"]
        elif "llm_minus_n_star" in v:
            dec = (f"$-$ {any_name(v['n_star'])}: ${f(v['llm_minus_n_star']['diff'], plus=True)}$ "
                   f"({f(v['llm_minus_n_star']['lo95'], plus=True)}); $-$ {any_name(v['sym_n_star'])}: "
                   f"${f(v['llm_minus_sym_n_star']['diff'], plus=True)}$ ({f(v['llm_minus_sym_n_star']['lo95'], plus=True)})")
            letter = v["letter"]
        else:
            dec, letter = "zero-shot: no sample-matched reader", "closed"
        assert letter in ("A", "B", "C", "closed"), letter       # D cannot occur with the simulator composer
        sym_letters[f"{lab}:{an}"] = letter
        star_mark = r"$^{\star}$" if an == o["L_star"] else ""
        fcr_tex.append(" & ".join([lab, LLM_TEX[an] + star_mark, shown_verdict(a3["pays"], a3["status"]),
                                   letters[f"{lab}:{an}"], letter, dec]) + r" \\")
fcr_tex.append(r"\bottomrule\end{longtable}" + ENDG)
fcr_tex.append(r"{\scriptsize Verdict: with both composers among the candidates. Decisive contrast: $L$ minus "
               r"trait-oracle-sym when the gate is open, $L$ minus the strongest of the five sample-matched factorized "
               r"readers built from $L$'s rows, three with the fitted and two with the simulator composer, when it is "
               r"closed. Differences in reward per episode, one-sided $95\%$ lower bounds in parentheses. $^{\star}$As in "
               r"Table~\ref{tab:fcr-letters}.\par}")

# (5) the trait-oracle decomposition, both composers
fcr_tex += [r"\subsubsection*{Parsing and composition}", SCR,
            r"\begin{longtable}{@{}l" + RAG + r"p{0.17\linewidth}" + "r" * 8 + r"@{}}"
            r"\caption{The trait-oracle decomposition of FCR-full and FCR-sym-full.}"
            r"\label{tab:fcr-decomp}\\"
            r"\toprule deployment & variant & capture & oracle & parsing & composition & \multicolumn{2}{c}{episodes} & "
            r"\multicolumn{2}{c}{regret} \\ & & & & loss & loss & right, wrong & wrong, wrong & composition & parsing \\ "
            r"\midrule \endfirsthead"
            r"\toprule deployment & variant & capture & oracle & parsing & composition & \multicolumn{2}{c}{episodes} & "
            r"\multicolumn{2}{c}{regret} \\ & & & & loss & loss & right, wrong & wrong, wrong & composition & parsing \\ "
            r"\midrule \endhead"]
decomp_num = OrderedDict()
for lab, dk in FCR_PRIMARY_DEP.items():
    for k in ("fcr_full", "fcr_full_pairwise", "fcr_full_embed", "sym_full", "sym_full_embed"):
        if k.startswith("sym_"):
            dd = FS["deployments"][dk]["decomposition"][k]
            es, rs = dd["episode_shares"], dd["regret_shares"]
            rw = 1.0 - sum(es.values())
            assert abs(rw) < 1e-9 and dd["composition_loss"] == 0.0 and abs(rs["traits wrong, counter wrong (parsing)"] - 1) < 1e-9
            vals = [0.0, es["traits wrong, counter wrong (parsing)"], 0.0, rs["traits wrong, counter wrong (parsing)"]]
            name = SYM_SHORT[k]
        else:
            dd = FD[dk]["decomposition"][k]
            es, rs = dd["episode_shares"], dd["regret_shares"]
            vals = [es["traits right, counter wrong (composition)"], es["traits wrong, counter wrong (parsing)"],
                    rs["traits right, counter wrong (composition)"], rs["traits wrong, counter wrong (parsing)"]]
            name = FCR_SHORT[k]
        fcr_tex.append(" & ".join([lab, name, f"${f(dd['capture'])}$", f"${f(dd['oracle_capture'])}$",
                                   f"${f(dd['parsing_loss'])}$", f"${f(dd['composition_loss'])}$"] +
                                  [f"${f(x)}$" for x in vals]) + r" \\")
        decomp_num[f"{lab}:{k}"] = dd
fcr_tex.append(r"\bottomrule\end{longtable}" + ENDG)
fcr_tex.append(r"{\scriptsize Oracle: the trait oracle of the same composer. Parsing loss: oracle capture minus reader "
               r"capture; composition loss: one minus oracle capture. Episodes: the shares with all traits right but the "
               r"counter wrong, and with traits and counter wrong; regret: the share of the reader's regret on those two "
               r"kinds of episode.\par}")
# trait accuracy by stratum, including the known-wording controls of the new-agents deployments
STRATA = [("most agents known, known types, known words", f"all/{P0}/newword", "seen"),
          ("most agents known, new types, new words", f"all/{P0}/newword", "held_out"),
          ("four new types, new words", f"heldout/{P0}/newword", "held_out"),
          ("eight new types, new words", f"heldout/{R8}/newword", "held_out"),
          ("four new types, known words", f"heldout/{P0}/sameword", "held_out"),
          ("eight new types, known words", f"heldout/{R8}/sameword", "held_out")]
fcr_tex += ["", SCR, r"\begin{longtable}{@{}" + RAG + r"p{0.32\linewidth}rrrrrrrr@{}}"
            r"\caption{Trait accuracy of the TF-IDF classifiers fitted to every known briefing, by stratum, and the "
            r"counter accuracy of three composers on their traits.}"
            r"\label{tab:fcr-strata}\\"
            r"\toprule stratum & episodes & favourite & reaction & timing & tuple & \multicolumn{3}{c}{counter} \\ "
            r"& & & & & & main & pairwise & simulator \\ \midrule \endfirsthead"
            r"\toprule stratum & episodes & favourite & reaction & timing & tuple & \multicolumn{3}{c}{counter} \\ "
            r"& & & & & & main & pairwise & simulator \\ \midrule \endhead"]
strata_num = OrderedDict()
for name, dk, s in STRATA:
    a = FD[dk]["arms"]["fcr_full"]["strata"][s]
    pw = FD[dk]["arms"]["fcr_full_pairwise"]["strata"][s]
    sy = FS["deployments"][dk]["arms"]["sym_full"]["strata"][s]
    assert sy["n"] == a["n"] and abs(sy["tuple"] - a["tuple"]) < 1e-12, (name, dk)    # the same parser
    fcr_tex.append(" & ".join([name, str(a["n"]), f"${f(a['favourite'])}$", f"${f(a['reactivity'])}$",
                               f"${f(a['timing'])}$", f"${f(a['tuple'])}$", f"${f(a['counter_accuracy'])}$",
                               f"${f(pw['counter_accuracy'])}$", f"${f(sy['counter_accuracy'])}$"]) + r" \\")
    strata_num[name] = OrderedDict(n=a["n"], tuple=a["tuple"], counter_main=a["counter_accuracy"],
                                   counter_pairwise=pw["counter_accuracy"], counter_simulator=sy["counter_accuracy"])
fcr_tex.append(r"\bottomrule\end{longtable}" + ENDG)
fcr_tex.append(r"{\scriptsize Known words: the same new types briefed in calibration phrasings. Counter: the main-effects, "
               r"the pairwise and the simulator composer.\par}")
fcr_tex.append("")
S1, S2 = FCR["secondary"]["S1_pays_with_fcr"], FCR["secondary"]["S2_fcr_full_minus_tfidf"]
S1s, S2s = FS["secondary"]["S1_pays_with_fcr_sym"], FS["secondary"]["S2_sym_full_minus_tfidf"]
n_strongest_fcr = sum(1 for d in FD.values() if d["strongest"]["with_fcr"].startswith("fcr"))
n_strongest_sym = sum(1 for d in FS["deployments"].values() if d["strongest"]["after_with_fcr_sym"].startswith("sym_"))
assert n_strongest_sym == len(FS["strongest_changes"])
assert S2["holm_surviving"] and all("-M18-" in x and "-diag/newword" in x and x.startswith("heldout/")
                                    for x in S2["holm_surviving"]), S2["holm_surviving"]
assert S2s["holm_surviving"] and all(x.endswith("-diag/sameword") for x in S2s["holm_surviving"]), S2s["holm_surviving"]
fcr_tex.append(r"The secondary families are judged with Holm's correction at $\alpha{=}0.05$. S1, the verdict of every "
               r"other language-model "
               f"arm with the fitted composer among the candidates: {S1['release_pays_yes']} of {S1['tests']} pass "
               f"uncorrected without it and {S1['updated_pays_yes_uncorrected']} with it, {len(S1['holm_surviving'])} "
               f"survive Holm, and {len(S1['lost_by_fcr_uncorrected'])} verdict changes; the strongest tested non-LLM arm "
               f"is a fitted-composer arm in {n_strongest_fcr} of the {len(FD)} briefing deployments. S2, FCR-full minus "
               f"whole-type TF-IDF in the {S2['tests']} other deployments: {len(S2['uncorrected_lo95_gt0'])} lower bounds "
               f"above zero, {len(S2['holm_surviving'])} surviving Holm, both in the deployments of six new types in new "
               r"words, where the main-effects composer gives every new type the best average strategy. S1$'$, the same "
               f"arms with both composers among the candidates: {S1s['after_pays_yes_uncorrected']} of {S1s['tests']} "
               f"pass uncorrected, {len(S1s['changed_uncorrected'])} verdict changes, and {len(S1s['holm_surviving_after'])} "
               f"survive Holm in this family against {len(S1s['holm_surviving_before'])} with the fitted composer alone; "
               f"the strongest tested non-LLM arm is a simulator-composer arm in {n_strongest_sym} of the "
               f"{len(FS['deployments'])} deployments, all with most agents known and new words. S2$'$, FCR-sym-full minus "
               f"whole-type TF-IDF in the {S2s['tests']} other deployments: {len(S2s['uncorrected_lo95_gt0'])} lower "
               f"bounds above zero, {len(S2s['holm_surviving'])} surviving Holm, all in known words in the deployments "
               r"whose held-out set holds habitual types, whose counter depends on the favourite throw.")

# (6) the predictions of the simulator composer
fcr_tex += [r"\subsubsection*{Predictions of the simulator composer}", SCR,
            r"\begin{longtable}{@{}l" + RAG + r"p{0.56\linewidth}c" + RAG + r"p{0.26\linewidth}@{}}"
            r"\caption{The predictions frozen with the simulator composer, in their pre-registered wording, and their "
            r"outcomes.}\label{tab:fcr-sym-ledger}\\"
            r"\toprule & prediction & holds & outcome \\ \midrule \endfirsthead"
            r"\toprule & prediction & holds & outcome \\ \midrule \endhead"]


def _claim(s):
    """A frozen prediction as printed: a long parenthetical aside becomes a comma-delimited phrase (the style rule)."""
    s = _re.sub(r" \(([^()]{38,})\)", r", \1,", s).replace(",,", ",").replace(" > 0", " $>0$")
    return tex(s).replace(">=", r"$\ge$").replace("<=", r"$\le$").replace(" - ", r" $-$ ").replace("'", r"$'$")


PL_SYM = FS["predictions_ledger"]
for pid, v in PL_SYM.items():
    val = v["value"]
    if isinstance(val, dict):
        val = "; ".join(f"{k0}: {v0}" for k0, v0 in val.items())
    val = val.replace("'", "")
    if v.get("mismatches"):
        val += "; not as predicted: " + "; ".join(
            f"{TAG[k0.split('__')[1]]} {VARIANT[variant_of(k0.split(':', 1)[1])]}, "
            f"{examples_in_words(k0.split(':', 1)[1], k0.split('/')[1])} at react4, {m0['observed']}"
            for k0, m0 in v["mismatches"].items())
    mark = r"$^{a}$" if pid == "P-SYM-4" else ""
    fcr_tex.append(" & ".join([pid + mark, _claim(v["claim"]), "yes" if v["holds"] else "no",
                               tex(val).replace("{", "").replace("}", "")]) + r" \\")
fcr_tex.append(r"\bottomrule\end{longtable}" + ENDG)
fcr_tex.append(r"{\scriptsize $^{a}$The frozen text counts sixteen deployments in known words; there are eighteen, "
               r"and the frozen check reads all of them.\par}")
(OUT / "si_briefing_fcr.tex").write_text("\n".join(fcr_tex), encoding="utf-8")

# numbers of the FCR that the main text quotes
ok_dep = FD[FCR_PRIMARY_DEP["most agents known"]]
r4n, r8n = FD[FCR_PRIMARY_DEP["new agents, react4"]], FD[FCR_PRIMARY_DEP["new agents, react8"]]
r4s = FD[f"heldout/{P0}/sameword"]
best_tau = {t: v["best_tau"] for t, v in S0B["groups"][P0]["reference"]["per_type"].items()}
orc_choice = {t: max(c, key=c.get) for t, c in r4s["arms"]["fcr_oracle"]["choices_held_out"].items()}
orc_right = sum(orc_choice[t] == best_tau[t] for t in best_tau)
edge4 = FD[FCR_PRIMARY_DEP["new agents, react4"]]["llm"]["briefing:llm__qwen3.8-27b__single__nshot1"]["vs"]["fcr_oracle"]
edge8p = r8n["llm"]["briefing:llm__qwen3.8-27b__single__nshot1"]["vs"]["fcr_oracle_pairwise"]
edgekp = ok_dep["llm"]["task:llm__qwen3.8-27b__single__nshot1__v2"]["vs"]["fcr_oracle_pairwise"]
NUM["fcr"] = OrderedDict(
    src="fcr.json (the pre-registered factorized compositional reader, fitted composer)",
    freeze=FCR["meta"]["freeze"]["frozen_utc"],
    FCR_DEPLOYMENTS=len(FD), FCR_STRONGEST_IN=n_strongest_fcr, FCR_S1_TESTS=S1["tests"], FCR_S1_YES=S1["release_pays_yes"],
    FCR_S1_YES_WITH=S1["updated_pays_yes_uncorrected"], FCR_S1_HOLM=len(S1["holm_surviving"]),
    FCR_VERDICT_CHANGES=len(S1["lost_by_fcr_uncorrected"]),
    FCR_KNOWN_SEEN_TUPLE=ok_dep["arms"]["fcr_full"]["strata"]["seen"]["tuple"],
    FCR_R4_KNOWNWORDS_TUPLE=r4s["arms"]["fcr_full"]["trait_accuracy"]["tuple"],
    FCR_R4_KNOWNWORDS_COUNTER=r4s["arms"]["fcr_full"]["counter_accuracy"],
    FCR_R4_KNOWNWORDS_CAPTURE=r4s["arms"]["fcr_full"]["capture"],
    FCR_R4_KNOWNWORDS_ORACLE_RIGHT=orc_right, FCR_R4_NEW_TYPES=len(best_tau),
    FCR_R4_KNOWNWORDS_TFIDF_CAPTURE=r4s["arms"]["briefing:text_bow"]["capture"],
    FCR_NEWWORD_TUPLE=[ok_dep["arms"]["fcr_full"]["strata"]["held_out"]["tuple"],
                       r4n["arms"]["fcr_full"]["strata"]["held_out"]["tuple"],
                       r8n["arms"]["fcr_full"]["strata"]["held_out"]["tuple"]],
    FCR_T3_MIN=min(t3), FCR_T3_MAX=max(t3), FCR_T3_ALL_HOLD=all(
        t["holds"] for tests in FCR["primary_tests"].values() for k, t in tests.items() if k.startswith("T3")),
    FCR_ORACLE_PAIR_KNOWN=ok_dep["arms"]["fcr_oracle_pairwise"]["capture"],
    FCR_ORACLE_PAIR_R8=r8n["arms"]["fcr_oracle_pairwise"]["capture"],
    FCR_ORACLE_R4=r4n["arms"]["fcr_oracle"]["capture"], FCR_ORACLE_PAIR_R4=r4n["arms"]["fcr_oracle_pairwise"]["capture"],
    FCR_EDGE_R4=OrderedDict(diff=edge4["diff"], ci=edge4["ci"], lo95=edge4["lo95"]),
    FCR_EDGE_PAIR_R8=OrderedDict(diff=edge8p["diff"], ci=edge8p["ci"], lo95=edge8p["lo95"]),
    FCR_EDGE_PAIR_KNOWN=OrderedDict(diff=edgekp["diff"], ci=edgekp["ci"], lo95=edgekp["lo95"]),
    letters=letters, loco_cv=fcr_cv_num, decomposition=decomp_num, strata=strata_num)


# numbers of the simulator composer that the main text quotes
def miss_regret(dep, rec):
    """Mean regret per episode on a wrong counter: H_D (1 - capture) / (1 - counter accuracy), exact because an episode on
    the best counter has zero regret."""
    return dep["reference"]["H_D"] * (1 - rec["capture"]) / (1 - rec["counter_accuracy"])


SD = FS["deployments"]
s_ok, s_r4, s_r8 = (SD[FCR_PRIMARY_DEP[k]] for k in ("most agents known", "new agents, react4", "new agents, react8"))
qtask = C3V.get("s0", P0, NEW, "llm__qwen3.8-27b__single__nshot1__v2")
qlong = C3V.get("s0", P0, NEW, "llm__qwen3.8-27b__single__nshot1__v2-think-long")
assert qtask["pays_release"] == "yes" and qtask["pays"] == "fixed only" and qtask["strongest"] == "sym_full"
assert qlong["pays"] == "yes" and qlong["status_release"] == "H" and qlong["status"] == "S"
oracle_caps = [d["arms"]["sym_oracle"]["capture"] for d in SD.values()]
bandit = {k: d["strongest"]["online_bandit"] for k, d in (("r4", s_r4), ("r8", s_r8))}
assert bandit["r4"] == "nonllm:linucb"
t3s_lo = [t["lo95"] for t in _t3s]
fam_c3 = C3V.family_counts()
NUM["fcr_sym"] = OrderedDict(
    src="fcr_sym.json (the pre-registered factorized reader with the simulator composer) and fcr_sym_calib.json",
    freeze=FS["meta"]["freeze"]["frozen_utc"],
    FS_DEPLOYMENTS=len(SD), FS_LLM_ARMS=sum(len(d["llm"]) for d in SD.values()),
    FS_CHANGED_UNCORR=len(FS["changed_verdicts"]), FS_HOLM_LOST=len(S1s["lost_holm"]),
    FS_STRONGEST_IN=n_strongest_sym,
    FS_KNOWN=OrderedDict(capture=s_ok["arms"]["sym_full"]["capture"], capture_ci=s_ok["arms"]["sym_full"]["capture_ci"],
                         counter=s_ok["arms"]["sym_full"]["counter_accuracy"]),
    FS_R4=OrderedDict(capture=s_r4["arms"]["sym_full"]["capture"], capture_ci=s_r4["arms"]["sym_full"]["capture_ci"],
                      counter=s_r4["arms"]["sym_full"]["counter_accuracy"],
                      tuple=s_r4["arms"]["sym_full"]["trait_accuracy"]["tuple"]),
    FS_R8=OrderedDict(capture=s_r8["arms"]["sym_full"]["capture"], capture_ci=s_r8["arms"]["sym_full"]["capture_ci"],
                      counter=s_r8["arms"]["sym_full"]["counter_accuracy"],
                      tuple=s_r8["arms"]["sym_full"]["trait_accuracy"]["tuple"]),
    FS_MISS_REGRET=OrderedDict(known=miss_regret(s_ok, s_ok["arms"]["sym_full"]),
                               r4=miss_regret(s_r4, s_r4["arms"]["sym_full"]), r8=miss_regret(s_r8, s_r8["arms"]["sym_full"])),
    BANDIT_MISS_REGRET=OrderedDict(r4=miss_regret(s_r4, s_r4["comparators"][bandit["r4"]]),
                                   r8=miss_regret(s_r8, s_r8["comparators"][bandit["r8"]]), arms=bandit),
    FS_ORACLE_MIN=min(oracle_caps), FS_ORACLE_MAX=max(oracle_caps),
    FS_QWEN_TASK=OrderedDict(diff=qtask["vs"]["diff"], ci=qtask["vs"]["ci"], lo95=qtask["vs"]["lo95"],
                             margin_lo95=qtask["margin_lo95"], capture=qtask["sym"]["capture"]),
    FS_QWEN_LONG=OrderedDict(lo95=qlong["vs"]["lo95"], p=qlong["p"], lo95_release=qlong["sym"]["release"]["vs_strongest"]["lo95"],
                             p_release=qlong["p_release"]),
    FS_T3_LO_MIN=min(t3s_lo), FS_T3_LO_MAX=max(t3s_lo), FS_T3_DIFF_MIN=min(t["diff"] for t in _t3s),
    FS_T3_DIFF_MAX=max(t["diff"] for t in _t3s), FS_T3_ALL_HOLD=all(t["holds"] for t in _t3s),
    FS_CALIB_CHECKS=SC["checks"], FS_CALIB_MISMATCHES=SC["mismatches"], FS_CALIB_LOCKS=SC["locks"],
    families_with_both_composers=fam_c3, letters=sym_letters,
    predictions=OrderedDict((k, v["holds"]) for k, v in PL_SYM.items()))

# ------------------------------------------------------------------------------------------------------------------
# supplement: the verdict read on calibration seeds (calib_verdict.json)
# ------------------------------------------------------------------------------------------------------------------
def cal_cell(c):
    dom, name = c.split("/", 1)
    if dom == "sandbox":
        m = _re.match(r"sandbox-M(\d+)-K\d+-sharp([0-9.]+)-noise([0-9.]+)", name)
        return f"sandbox $M{{={m.group(1)}}}$, $s{{={m.group(2)}}}$, $\\sigma{{={float(m.group(3)):g}}}$"
    if dom == "briefing":
        return f"briefing $M{{={_re.search(r'-M(\d+)-', name).group(1)}}}$, known words"
    parts = name.split("-")
    return f"{parts[0]} {parts[1]}" + (", atten." if "delta06" in name else "")


def cal_arm(k, dom):
    parts = k.split("__")
    model, mode, n = parts[1], parts[2], int(parts[3].replace("nshot", ""))
    bits = [r"\texttt{" + TAG[model] + "}"]
    if dom.startswith("briefing"):
        bits.append("task" if len(parts) > 4 else "bare")
    bits += {"hypothesis_first": ["hf"], "deliberate": ["three turns"]}.get(mode, [])
    return " ".join(bits) + f" $N{{={n}}}$"


CAL_NAME = dict(NONLLM, scripted="scripted rule", fewshot__5="few-shot 5", fewshot__20="few-shot 20")


def cal_comp(k):
    return CAL_NAME.get(k.split("/")[-1], tex(k))


yn = lambda b: "yes" if b else "no"  # noqa: E731
cv_tex = [SCR, r"\begin{longtable}{@{}" + RAG + r"p{0.25\linewidth}rlrr" + RAG + r"p{0.16\linewidth}r" + RAG +
          r"p{0.13\linewidth}@{}}"
          r"\caption{Reference quantities of the read half B of each cell.}\label{tab:calib-ref}\\"
          r"\toprule cell & $n_B$ & best fixed & $V^\star_B$ & $H_B$ & strongest on B & capture & at deployment \\ "
          r"\midrule \endfirsthead"
          r"\toprule cell & $n_B$ & best fixed & $V^\star_B$ & $H_B$ & strongest on B & capture & at deployment \\ "
          r"\midrule \endhead"]
for c, v in CAL["cells"].items():
    deps = sorted({a["deployment"]["vs_strongest_arm"] for a in v["arms"].values()})
    cv_tex.append(" & ".join([cal_cell(c), str(v["n_B"]), tex(v["best_fixed_B"]), f"${f(v['V_star_B'])}$",
                              f"${f(v['H_B'])}$", cal_comp(v["strongest_nonllm_B"]), f"${f(v['strongest_capture_B'])}$",
                              ", ".join(cal_comp(x) for x in deps)]) + r" \\")
cv_tex.append(r"\bottomrule\end{longtable}" + ENDG)
cv_tex.append(r"{\scriptsize $n_B$: episodes on B; $V^\star_B$, $H_B$: the best fixed strategy's mean and the headroom on "
              r"the B episodes; strongest on B: the strongest non-LLM arm fitted on half A and read on B, with its capture; "
              r"at deployment: the strongest non-LLM arm at deployment.\par}")
cv_tex += ["", SCR, r"\begin{longtable}{@{}" + RAG + r"p{0.22\linewidth}" + RAG + r"p{0.15\linewidth}rcccc" + "rrr" +
           r"c@{}}"
           r"\caption{The C2 and C3 verdicts of every arm at deployment and on the read half B.}"
           r"\label{tab:calib-agree}\\"
           r"\toprule cell & arm & $n_B$ & \multicolumn{2}{c}{C2} & \multicolumn{2}{c}{C3} & \multicolumn{2}{c}{B lower bound} "
           r"& P2 & agree \\ & & & dep. & B & dep. & B & margin & $\Delta$ & & \\ \midrule \endfirsthead"
           r"\toprule cell & arm & $n_B$ & \multicolumn{2}{c}{C2} & \multicolumn{2}{c}{C3} & \multicolumn{2}{c}{B lower bound} "
           r"& P2 & agree \\ & & & dep. & B & dep. & B & margin & $\Delta$ & & \\ \midrule \endhead"]
for c, v in CAL["cells"].items():
    for k, a in v["arms"].items():
        B, Dp = a["B"], a["deployment"]
        both = a["agree_C2"] and a["agree_C3"]
        cv_tex.append(" & ".join([cal_cell(c), cal_arm(k, c), str(B["n"]), yn(Dp["C2"]), yn(B["C2"]), yn(Dp["C3"]),
                                  yn(B["C3"]), lo_s(B["margin_lo95"]), lo_s(B["vs_lo95"]),
                                  f"${a['power']['p_agree_C2']:.2f}$", "yes" if both else r"\textbf{no}"]) + r" \\")
cv_tex.append(r"\bottomrule\end{longtable}" + ENDG)
cv_tex.append(r"{\scriptsize dep.: deployment. B lower bound: the one-sided $95\%$ lower bounds on B of the margin over the "
              r"best fixed strategy and of the paired difference $\Delta$ against the strongest non-LLM arm. P2: the power "
              r"reading. Agree: both the C2 and the C3 verdict on B match deployment.\par}")
SM = CAL["summary"]
cv_tex += ["", SCR, r"\begin{longtable}{@{}" + RAG + r"p{0.22\linewidth}" + RAG + r"p{0.15\linewidth}rrlrrr@{}}"
           r"\caption{The arms whose C2 verdict on B differs from deployment.}\label{tab:calib-disagree}\\"
           r"\toprule cell & arm & \multicolumn{2}{c}{deployment} & \multicolumn{3}{c}{read half B} & P2 \\ "
           r"& & capture & margin lo & capture [95\%] & margin & margin lo & \\ \midrule \endfirsthead"
           r"\toprule cell & arm & \multicolumn{2}{c}{deployment} & \multicolumn{3}{c}{read half B} & P2 \\ "
           r"& & capture & margin lo & capture [95\%] & margin & margin lo & \\ \midrule \endhead"]
dis_rows = []
for d in SM["disagreements"]:
    a = CAL["cells"][d["cell"]]["arms"][d["arm"]]
    B, Dp = a["B"], a["deployment"]
    assert d["C2_dep"] and not d["C2_B"] and B["margin"] > 0.5, d
    dis_rows.append(OrderedDict(cell=d["cell"], arm=d["arm"], B_margin=B["margin"], B_margin_lo95=B["margin_lo95"],
                                P2=a["power"]["p_agree_C2"], n_B=B["n"],
                                dep_inside_B_ci=B["capture_ci"][0] <= Dp["capture"] <= B["capture_ci"][1]))
    cv_tex.append(" & ".join([cal_cell(d["cell"]), cal_arm(d["arm"], d["cell"]), f"${f(Dp['capture'])}$",
                              lo_s(Dp["margin_lo95"]), f"${f(B['capture'])}$\\," +
                              grey_ci(B["capture_ci"][0], B["capture_ci"][1]),
                              lo_s(B["margin"]), lo_s(B["margin_lo95"]), f"${a['power']['p_agree_C2']:.2f}$"]) + r" \\")
cv_tex.append(r"\bottomrule\end{longtable}" + ENDG)
_in = [r for r in dis_rows if r["dep_inside_B_ci"]]
_out = [r for r in dis_rows if not r["dep_inside_B_ci"]]
# the supplement names these rows: zero-shot gpt-oss-120b in the two three-type sandbox cells at sharpness 0.6
assert _out and all(r["arm"] == "llm__gpt-oss-120b__single__nshot0" and "-M3-K3-sharp0.6-" in r["cell"] for r in _out), _out
cv_tex.append("\n" + f"The margin on B lies between ${f(min(r['B_margin'] for r in dis_rows))}$ and "
              f"${f(max(r['B_margin'] for r in dis_rows))}$ in every row, on ${min(r['n_B'] for r in dis_rows)}$ to "
              f"${max(r['n_B'] for r in dis_rows)}$ episodes. In {len(_in)} rows the deployment capture lies inside B's "
              f"$95\\%$ interval and P2 is at most ${max(r['P2'] for r in _in):.2f}$; in the other {len(_out)} it lies "
              f"outside, with P2 at most ${max(r['P2'] for r in _out):.2f}$.")
c3_pass_B = sum(1 for v in CAL["cells"].values() for a in v["arms"].values() if a["B"]["C3"])
c3_pass_D = sum(1 for v in CAL["cells"].values() for a in v["arms"].values() if a["deployment"]["C3"])
cv_tex.append(f"Over the {len(CAL['cells'])} cells and {SM['arms']} language-model arms, C3 agrees for "
              f"{SM['agree_C3']} of {SM['arms']} arms, passing for {c3_pass_B} on B and {c3_pass_D} at deployment; C2 "
              f"agrees for {SM['agree_C2']}, against {SM['expected_C2_agreements_from_power']:.1f} expected from the power "
              f"reading. Exact replay holds on every B episode of every arm, with {SM['replay_mismatches_llm']} mismatches "
              f"for the language-model arms and {SM['replay_mismatches_comparators']} for the comparators.")
(OUT / "si_calib_verdict.tex").write_text("\n".join(cv_tex), encoding="utf-8")
dirs = {(d["C2_dep"], d["C2_B"]) for d in SM["disagreements"]}
NUM["calib"] = OrderedDict(
    src="calib_verdict.json (the pre-registered calibration-seed verdict check)",
    CALIB_CELLS=len(CAL["cells"]), CALIB_ARMS=SM["arms"], CALIB_C3_AGREE=SM["agree_C3"], CALIB_C2_AGREE=SM["agree_C2"],
    CALIB_C2_EXPECTED=SM["expected_C2_agreements_from_power"], CALIB_C2_DISAGREE=len(SM["disagreements"]),
    CALIB_C3_PASS_B=c3_pass_B, CALIB_C3_PASS_DEP=c3_pass_D, CALIB_ONE_DIRECTION=dirs == {(True, False)},
    CALIB_DISAGREE_B_MARGIN_MIN=min(r["B_margin"] for r in dis_rows),
    CALIB_DISAGREE_B_MARGIN_MAX=max(r["B_margin"] for r in dis_rows),
    CALIB_DISAGREE_P2_MAX=max(r["P2"] for r in dis_rows), disagreements=dis_rows)

# every number the main text quotes from the two checks, with the exact text it is quoted as; check_numbers.py
# --extra verifies that each text occurs in main.tex
FN_ = NUM["fcr"]
FS_ = NUM["fcr_sym"]
CN_ = NUM["calib"]
_var = FS_["families_with_both_composers"]["variants"]
# the variant arms that survive Holm with both composers among the candidates are all against new agents
assert all(a["part"] == "s0b" for a in C3V.arms.values() if a["family"] == "variants" and a["holm"])
assert FS_["FS_ORACLE_MIN"] == 1.0 and FS_["FS_T3_ALL_HOLD"]
assert FS_["BANDIT_MISS_REGRET"]["arms"]["r4"] == "nonllm:linucb"
# the prefix ceiling against four new types (Table 2, Fig. 5b): the first moves reveal the whole headroom
_pr4 = NUM["tab_briefing:react4:prefix_bayes_ceiling"]
assert round(_pr4["capture"], 2) == 1.0 and min(_pr4["capture_ci"]) > 0.995, _pr4
_prefix_r4_quote = "the first moves reveal the whole headroom against four new types"
NUM["quotes:main"] = OrderedDict([
    ("FS_STRONGEST_IN", f"${FS_['FS_STRONGEST_IN']}$ of the ${FS_['FS_DEPLOYMENTS']}$ briefing deployments"),
    ("FS_KNOWN", f"${f(FS_['FS_KNOWN']['capture'])}\\,[{f(FS_['FS_KNOWN']['capture_ci'][0])},"
                 f"{f(FS_['FS_KNOWN']['capture_ci'][1])}]$, the strongest tested non-LLM arm"),
    ("FS_QWEN_TASK_CAPTURE", f"collects ${f(FS_['FS_QWEN_TASK']['capture'])}$, clearing"),
    ("FS_QWEN_TASK_MARGIN", f"by a lower bound of ${f(FS_['FS_QWEN_TASK']['margin_lo95'], plus=True)}$ reward"),
    ("FS_QWEN_TASK_LO", f"lower bound against that reader is ${f(FS_['FS_QWEN_TASK']['lo95'], plus=True)}$"),
    ("HOLM_VARIANTS", f"${_var['entries']}$ variant arms, ${_var['holm']}$ survive Holm correction, all with new agents"),
    ("FS_TUPLE", f"${f(FS_['FS_R4']['tuple'])}$ and ${f(FS_['FS_R8']['tuple'])}$ of episodes against four and eight"),
    ("FS_MISS_REGRET", f"about ${FS_['FS_MISS_REGRET']['r4']:.0f}$ reward, against about "
                       f"${FS_['BANDIT_MISS_REGRET']['r4']:.0f}$ for LinUCB"),
    ("FS_NEW_CAPTURE", f"collects ${f(FS_['FS_R4']['capture'])}$ and ${f(FS_['FS_R8']['capture'])}$, below the bandits"),
    ("FS_ORACLE", f"Given the true traits it collects all of the headroom in all ${FS_['FS_DEPLOYMENTS']}$"),
    ("FCR_R4_KNOWNWORDS_ORACLE_RIGHT", f"right counter for ${FN_['FCR_R4_KNOWNWORDS_ORACLE_RIGHT']}$ of the "
                                       f"${FN_['FCR_R4_NEW_TYPES']}$"),
    ("FS_T3_LO", f"lower bounds of ${FS_['FS_T3_LO_MIN']:.1f}$ to ${FS_['FS_T3_LO_MAX']:.1f}$ reward"),
    ("CALIB_C3", f"${CN_['CALIB_C3_AGREE']}$ of ${CN_['CALIB_ARMS']}$"),
    ("CALIB_C2", f"${CN_['CALIB_C2_AGREE']}$ of the ${CN_['CALIB_ARMS']}$"),
    ("CALIB_CELLS", f"${CN_['CALIB_CELLS']}$ cells"),
    ("CALIB_C2_EXPECTED", f"about ${round(CN_['CALIB_C2_EXPECTED'])}$"),
    # nonllm_rows asserts that the type oracle collects 1.00 of the headroom in all three deployments of Table 2
    ("T2_ORACLE", "the type oracle collects all of the headroom in every deployment"),
    ("T2_PREFIX_R4", _prefix_r4_quote),
])
assert FN_["FCR_STRONGEST_IN"] == 0 and FN_["FCR_VERDICT_CHANGES"] == 0 and FN_["FCR_T3_ALL_HOLD"]
assert CN_["CALIB_C3_AGREE"] == CN_["CALIB_ARMS"] and CN_["CALIB_ONE_DIRECTION"]

with open(OUT / "numbers_briefing.json", "w", encoding="utf-8") as fh:
    json.dump(NUM, fh, indent=1, default=float)
print("tables written to", OUT)
for k, v in NUM.items():
    if k.startswith("tab_briefing:"):
        vs = v["vs_strongest"]
        print(f"{k:90s} {v['capture']:+.3f} {v['capture_ci']} " + (f"d={vs['diff']:+.2f} lo={vs['lo95']:+.2f}" if vs else "") +
              f" {v['pays']} {v['test']}")
