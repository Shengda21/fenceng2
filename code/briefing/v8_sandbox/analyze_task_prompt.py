"""Task-prompt analysis: per-arm capture and the paper's verdict (reusing analyze_briefing), paired contrasts
(v2 - v1, N1 - N0, reasoning - default, LLM - information-matched text_bow), cost per call (tokens, latency, reasoning text,
truncation), misreading rates (describe-the-opponent, ROCK over-selection), the classic-sandbox check, predictions P-J..P-N
and the decision rule.

usage (from the repository root, after code/briefing/assemble_task_prompt.py --root .):
  python code/briefing/v8_sandbox/analyze_task_prompt.py --root .
writes analysis/briefing/task_prompt.json and analysis/briefing/task_prompt.ledger.txt"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

import analyze_briefing as a0
from sandbox.briefing import resolve_held_out_set
from sandbox.game import parse_trait

MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b"]
# primary cell of the all-types deployment (M18 sigma0 react briefing-newword, single N=0): width of each model's 95% capture CI (analysis/all_types.json)
STAGE0_WIDTH = {"gpt-oss-120b": 0.758 - 0.425, "gpt-oss-20b": 0.286 - (-0.176), "qwen3.8-27b": -0.102 - (-0.559)}
# Audit (AUDIT_LLM_CONFIG.md, cf.py): 120b single N=0 in the primary cell, 0.606 -> 0.673 if every frame error became correct
PJ_GAIN = 0.067
PL_CAPTURE, PL_TOKEN_RATIO = 0.5, 10.0
S0_GROUP = "sandbox-traits-M18-K10-sharp0.8-noise0-react"
S0B_GROUPS = ["sandbox-traits-M18-K10-sharp0.8-noise0-react", "sandbox-traits-M18-K10-sharp0.8-noise0-react8"]
CLASSIC = [f"sandbox-M{M}-K3-sharp{s}-noise{n}" for M in (3, 6) for s in ("0.6", "0.8") for n in ("0", "0.5")]
LAG_DESCRIBES = {"ROCK": "LAG_S", "PAPER": "LAG_R", "SCISSORS": "LAG_P"}  # the LAG whose doc reads like this favourite's flip opponent
TEXT_KEYS = ("text_bow", "text_embed", "scripted_text", "text_bow_matched", "text_embed_matched")


def read_raw(path_stem: Path) -> list[dict]:
    for p, opener in ((path_stem.with_name(path_stem.name + ".jsonl"), open), (path_stem.with_name(path_stem.name + ".jsonl.gz"), gzip.open)):
        if p.exists():
            with opener(p, "rt", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
    return []


def describe_target(typ: str) -> str | None:
    fav, react, timing = parse_trait(typ)
    if react == "habit":
        return fav if timing == "stable" else LAG_DESCRIBES[fav]
    if react == "beat_last" and timing == "stable":
        return "MIRROR_BEAT"
    return None


def cost(recs: list[dict]) -> dict:
    pt, ct, lat, chars, n_calls, with_reasoning, length, att, rules = [], [], [], [], 0, 0, 0, 0, Counter()
    for r in recs:
        e = r["provenance"][0].get("extra", {})
        u = e.get("usage") or {}
        pt.append(u.get("prompt_tokens") or 0)
        ct.append(u.get("completion_tokens") or 0)
        lat.append(e.get("call_latency_s") or 0.0)
        att += int((e.get("attempts") or 1) > 1)
        rules[e.get("parse_rule", "v1")] += 1
        calls = e.get("calls") or []
        n_calls += len(calls)
        with_reasoning += sum(bool(c.get("reasoning")) for c in calls)
        length += sum(c.get("finish_reason") == "length" for c in calls)
        chars.append(sum(len(c.get("reasoning") or "") for c in calls))
    n = max(1, len(recs))
    return {"episodes": len(recs), "mean_prompt_tokens": float(np.mean(pt)) if pt else None, "mean_completion_tokens": float(np.mean(ct)) if ct else None,
            "p90_completion_tokens": float(np.percentile(ct, 90)) if ct else None, "mean_latency_s": float(np.mean(lat)) if lat else None,
            "p90_latency_s": float(np.percentile(lat, 90)) if lat else None, "calls_logged": n_calls,
            "reasoning_stored_rate": with_reasoning / n_calls if n_calls else None, "mean_reasoning_chars": float(np.mean(chars)) if n_calls else None,
            "length_finish_rate": length / n_calls if n_calls else None, "retried_rate": att / n, "parse_rules": dict(rules),
            "fallback_rate": sum(r.get("fallback_count", 0) > 0 for r in recs) / n}


def misreading(ref: a0.Reference, rows: list[dict]) -> dict:
    ch = a0.by_seed(rows, "tau")
    desc = [(ch.get(s), describe_target(t)) for s, t in zip(ref.seeds, ref.types) if describe_target(t)]
    rock_correct = sum("ROCK" in ref.per_type[t]["good_set"] for t in ref.types)
    rock_chosen = sum(ch.get(s) == "ROCK" for s in ref.seeds)
    rock_wrong = sum(ch.get(s) == "ROCK" and "ROCK" not in ref.per_type[t]["good_set"] for s, t in zip(ref.seeds, ref.types))
    return {"describe_n": len(desc), "describe_rate": float(np.mean([a == b for a, b in desc])) if desc else None,
            "rock_chosen": rock_chosen, "rock_correct_episodes": rock_correct,
            "rock_ratio": rock_chosen / rock_correct if rock_correct else None, "rock_wrong_share": rock_wrong / rock_chosen if rock_chosen else None}


def paired(ref: a0.Reference, x: dict, y: dict, B: int, rng) -> dict | None:
    """Paired by seed over the reference's headline seeds: value difference and capture difference (V* cancels; H_D is
    re-computed inside every replicate, as in analyze_briefing.arm_stats)."""

    mask = np.array([s in x and s in y for s in ref.seeds])
    if not mask.any():
        return None
    seeds = [s for s in ref.seeds if s in x and s in y]
    d = np.array([x[s] - y[s] for s in seeds])
    F = ref.F[:, mask]
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    Fb = F[:, idx]
    H_b = Fb.max(axis=0).mean(axis=1) - Fb.mean(axis=2).max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cap_b = np.where(H_b > 0, d[idx].mean(axis=1) / H_b, np.nan)
    cap_b = cap_b[np.isfinite(cap_b)]
    H = float(F.max(axis=0).mean() - F.mean(axis=1).max())
    val_b = d[idx].mean(axis=1)
    return {"n": n, "partial": bool(not mask.all()), "value_diff": float(d.mean()), "value_ci": [a0.pct(val_b, 2.5), a0.pct(val_b, 97.5)],
            "value_lo95": a0.pct(val_b, 5), "capture_diff": float(d.mean() / H) if H > 0 else None,
            "capture_ci": [a0.pct(cap_b, 2.5), a0.pct(cap_b, 97.5)], "capture_lo95": a0.pct(cap_b, 5), "capture_hi95": a0.pct(cap_b, 95)}


def keyfind(arms: dict, cell: str, model: str, mode: str, nshot, variant):
    """Arm key in a cell; nshot='many' matches the v2 N-shot arm with 3 renderings per seen type (nshot42 or nshot30)."""

    for key, a in arms.get(cell, {}).items():
        m = a["meta"]
        if m.get("arm") != "llm" or m.get("model") != model or m.get("mode") != mode or m.get("variant") != variant:
            continue
        if (nshot == "many" and m["nshot"] > 1) or m["nshot"] == nshot:
            return key
    return None


def matched_key(arms: dict, cell: str, name: str, many: bool):
    for key, a in arms.get(cell, {}).items():
        if a["meta"].get("arm") == name and ((a["meta"]["nshot"] > 1) if many else a["meta"]["nshot"] == 1):
            return key
    return None


def family(cells_dir: Path, deploy: str, headline_n: int, B: int) -> tuple[dict, dict]:
    a0.HEADLINE_N = headline_n
    a0.DEPLOY["types"] = deploy
    groups = a0.load(cells_dir)
    res = a0.analyse(groups, B)
    out = OrderedDict()
    for gname, g in groups.items():
        if "nonllm" not in g["cells"]:
            continue
        held = resolve_held_out_set(g["kind"], g["M"])
        ref = a0.Reference(g["cells"]["nonllm"]["arms"], held)
        rng = np.random.default_rng(20261005)
        entry = res["groups"][gname]
        arms = {cell: c["arms"] for cell, c in g["cells"].items() if cell != "nonllm"}
        info = OrderedDict()
        for cell, c in g["cells"].items():
            if cell == "nonllm":
                continue
            for key, arm in c["arms"].items():
                meta = arm["meta"]
                if meta["arm"] != "llm" and meta["arm"] not in TEXT_KEYS:
                    continue
                st = entry["cells"][cell]["arms"][key]
                row = {k: st.get(k) for k in ("n", "capture", "capture_ci", "capture_lo95", "margin", "margin_lo95", "pays", "pays_margin0",
                                              "p_pays", "vs_strongest", "vs_strongest_arm", "vs_strongest_held_out", "vs_strongest_held_out_arm", "fallback_rate")}
                row["meta"] = meta
                if "strata" in st:
                    row["strata"] = {s: {k: st["strata"][s].get(k) for k in ("n", "capture", "hit_rate")} for s in ("seen", "held_out", "habit", "reactive")}
                if meta["arm"] == "llm":
                    row["cost"] = cost(read_raw(cells_dir / c["dir"] / key))
                    row["misreading"] = misreading(ref, arm["rows"])
                info[f"{cell}/{key}"] = row
        vals = {(cell, key): a0.by_seed(a["rows"]) for cell, c in arms.items() for key, a in c.items()}

        def pair(cell, kx, ky):
            if kx is None or ky is None:
                return None
            return paired(ref, vals[(cell, kx)], vals[(cell, ky)], B, rng)

        contrasts = OrderedDict()
        for cell in arms:
            cc = OrderedDict()
            for m in MODELS:
                k = lambda mode, n, v: keyfind(arms, cell, m, mode, n, v)  # noqa: E731
                c = OrderedDict()
                c["v2_minus_v1_single_n0"] = pair(cell, k("single", 0, "v2"), k("single", 0, None))
                c["v2_minus_v1_hf_n0"] = pair(cell, k("hypothesis_first", 0, "v2"), k("hypothesis_first", 0, None))
                c["v2_minus_v1_single_n1"] = pair(cell, k("single", 1, "v2"), k("single", 1, None))
                c["n1_minus_n0_v2"] = pair(cell, k("single", 1, "v2"), k("single", 0, "v2"))
                c["nmany_minus_n0_v2"] = pair(cell, k("single", "many", "v2"), k("single", 0, "v2"))
                c["nmany_minus_n1_v2"] = pair(cell, k("single", "many", "v2"), k("single", 1, "v2"))
                c["hf_minus_single_v2"] = pair(cell, k("hypothesis_first", 0, "v2"), k("single", 0, "v2"))
                c["shuf_minus_fixed_v2"] = pair(cell, k("single", 0, "v2-shuf"), k("single", 0, "v2"))
                for v in ("v2-med", "v2-high", "v2-think", "v2-high-long", "v2-think-long"):
                    c[f"{v}_minus_v2_n0"] = pair(cell, k("single", 0, v), k("single", 0, "v2"))
                c["v2-think_minus_v2_n1"] = pair(cell, k("single", 1, "v2-think"), k("single", 1, "v2"))
                c["v2-think-long_minus_v2_n1"] = pair(cell, k("single", 1, "v2-think-long"), k("single", 1, "v2"))
                c["v2-high-long_minus_v1_n0"] = pair(cell, k("single", 0, "v2-high-long"), k("single", 0, None))
                c["v2-think-long_minus_v1_n1"] = pair(cell, k("single", 1, "v2-think-long"), k("single", 1, None))
                for name in ("text_bow_matched", "text_embed_matched"):
                    c[f"n1_v2_minus_{name}_n1"] = pair(cell, k("single", 1, "v2"), matched_key(arms, cell, name, False))
                    c[f"nmany_v2_minus_{name}_nmany"] = pair(cell, k("single", "many", "v2"), matched_key(arms, cell, name, True))
                c["n1_v2_minus_text_bow_full"] = pair(cell, k("single", 1, "v2"), "text_bow" if "text_bow" in arms[cell] else None)
                c["nmany_v2_minus_text_bow_full"] = pair(cell, k("single", "many", "v2"), "text_bow" if "text_bow" in arms[cell] else None)
                cc[m] = {kk: vv for kk, vv in c.items() if vv is not None}
            contrasts[cell] = cc
        out[gname] = {"meta": entry["meta"], "reference": {"V_star": ref.V_star, "H_D": ref.H_D, "best_fixed": ref.taus[ref.best_i], "seeds_n": len(ref.seeds)},
                      "strongest_nonllm": entry["strongest_nonllm"], "arms": info, "contrasts": contrasts}
    return out, res


def classic_family(cells_dir: Path, B: int) -> dict:
    rng = np.random.default_rng(20261006)
    out = OrderedDict()
    reps = {m: [] for m in MODELS}
    for stem in CLASSIC:
        fixed = {}
        for f in sorted((cells_dir / f"{stem}-nonllm").glob("fixed__*.jsonl*")):
            fixed[f.name.split("__", 1)[1].split(".")[0]] = a0.by_seed(a0.read_rows(f))
        if not fixed:
            continue
        seeds = sorted(set.intersection(*[set(v) for v in fixed.values()]))
        seeds = [s for s in seeds if s < 1000][:100]
        taus = list(fixed)
        F = np.array([[fixed[t][s] for s in seeds] for t in taus])
        idx = rng.integers(0, len(seeds), size=(B, len(seeds)))
        Fb = F[:, idx]
        Vb = Fb.mean(axis=2).max(axis=0)
        Hb = Fb.max(axis=0).mean(axis=1) - Vb
        V, H = float(F.mean(axis=1).max()), float(F.max(axis=0).mean() - F.mean(axis=1).max())
        cell = OrderedDict(seeds_n=len(seeds), V_star=V, H_D=H, best_fixed=taus[int(F.mean(axis=1).argmax())], models=OrderedDict())

        def stats(A):
            cap_b = (A[idx].mean(axis=1) - Vb) / Hb
            return {"mean": float(A.mean()), "capture": float((A.mean() - V) / H), "capture_ci": [a0.pct(cap_b, 2.5), a0.pct(cap_b, 97.5)]}

        for m in MODELS:
            rows = {}
            for tag, name in (("v1", f"llm__{m}__single__nshot0"), ("v2", f"llm__{m}__single__nshot0__v2")):
                recs = read_raw(cells_dir / f"{stem}-semantic" / name)
                if recs:
                    rows[tag] = recs
            entry = OrderedDict()
            arr = {}
            for tag, recs in rows.items():
                v = {}
                for r in recs:
                    v.setdefault(int(r["seed"]), float(r["total_reward"]))
                if all(s in v for s in seeds):
                    arr[tag] = np.array([v[s] for s in seeds])
                    entry[tag] = stats(arr[tag])
                    entry[tag]["cost"] = cost(recs)
            if "v1" in arr and "v2" in arr:
                d = arr["v2"] - arr["v1"]
                dcap_b = d[idx].mean(axis=1) / Hb
                entry["v2_minus_v1"] = {"capture_diff": float(d.mean() / H), "capture_ci": [a0.pct(dcap_b, 2.5), a0.pct(dcap_b, 97.5)],
                                        "value_diff": float(d.mean())}
                reps[m].append(dcap_b)
            cell["models"][m] = entry
        out[stem] = cell
    pn = OrderedDict()
    for m in MODELS:
        diffs = [c["models"][m]["v2_minus_v1"]["capture_diff"] for c in out.values() if "v2_minus_v1" in c["models"].get(m, {})]
        if not diffs:
            continue
        mean_b = np.mean(np.stack(reps[m]), axis=0)
        mean = float(np.mean(diffs))
        pn[m] = {"cells": len(diffs), "mean_capture_diff": mean, "mean_ci": [a0.pct(mean_b, 2.5), a0.pct(mean_b, 97.5)],
                 "stage0_width": STAGE0_WIDTH[m], "material": abs(mean) > STAGE0_WIDTH[m],
                 "cells_beyond_width": sum(abs(x) > STAGE0_WIDTH[m] for x in diffs),
                 "ci_excludes_0": bool(a0.pct(mean_b, 2.5) > 0 or a0.pct(mean_b, 97.5) < 0)}
    return {"cells": out, "P_N": pn}


def get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def predictions(s0: dict, s0b: dict, classic: dict) -> dict:
    g = s0.get(S0_GROUP, {})
    c = get(g, "contrasts", "briefing-newword") or {}
    arms = get(g, "arms") or {}
    pj = get(c, "gpt-oss-120b", "v2_minus_v1_single_n0")
    pk = get(c, "gpt-oss-120b", "n1_minus_n0_v2")
    think_normal = arms.get("briefing-newword/llm__qwen3.8-27b__single__nshot0__v2-think")
    think_long = arms.get("briefing-newword/llm__qwen3.8-27b__single__nshot0__v2-think-long")
    # the long-context arm replaces the 4096-budget arm when Qwen thinking hit that cap
    think = think_normal if think_normal and think_normal.get("n") == 100 else think_long
    base = arms.get("briefing-newword/llm__qwen3.8-27b__single__nshot0__v2")
    ratio = None
    if think and base and base["cost"]["mean_completion_tokens"]:
        ratio = think["cost"]["mean_completion_tokens"] / base["cost"]["mean_completion_tokens"]
    pm = OrderedDict()
    for m in MODELS:
        n1 = get(c, m, "n1_v2_minus_text_bow_matched_n1")
        nm = get(c, m, "nmany_v2_minus_text_bow_matched_nmany")
        pm[m] = {"n1": n1, "holds_n1": bool(n1 and n1["value_lo95"] > 0), "n42_reported": nm}
    out = OrderedDict()
    out["P_J"] = {"contrast": pj, "threshold": PJ_GAIN, "holds": bool(pj and pj["capture_diff"] is not None and pj["capture_diff"] >= PJ_GAIN and pj["capture_lo95"] > 0)}
    out["P_K"] = {"contrast": pk, "holds": bool(pk and pk["capture_ci"][1] >= 0)}
    out["P_L"] = {"arm_used": think and think["meta"].get("variant"), "capture": think and think["capture"], "capture_ci": think and think["capture_ci"], "token_ratio": ratio,
                  "think_cost": think and think["cost"], "base_cost": base and base["cost"],
                  "holds_capture": bool(think and think["capture"] >= PL_CAPTURE), "holds_cost": bool(ratio is not None and ratio > PL_TOKEN_RATIO)}
    out["P_L"]["holds"] = out["P_L"]["holds_capture"] and out["P_L"]["holds_cost"]
    out["P_M"] = {"models": pm, "holds": all(v["holds_n1"] for v in pm.values()) if pm else False}
    out["P_N"] = classic.get("P_N")
    out["s0b_n1"] = s0b_n1(s0b)
    out["s0_reasoning"] = {m: {v: (lambda a: a and {kk: a.get(kk) for kk in ("n", "capture", "capture_ci", "pays", "p_pays", "cost")})(
        arms.get(f"briefing-newword/llm__{m}__single__nshot0__{v}")) for v in ("v2", "v2-med", "v2-high", "v2-high-long", "v2-think", "v2-think-long")}
        for m in MODELS}
    return out


def s0b_n1(s0b: dict) -> dict:
    """In the held-out-only cells: does v2 change gpt-oss-120b N=1, and does Qwen N=1 still pass?"""

    out = OrderedDict()
    for gname in S0B_GROUPS:
        g = s0b.get(gname)
        if not g:
            continue
        arms, c = g["arms"], g["contrasts"].get("briefing-newword", {})
        kind = "react8" if gname.endswith("react8") else "react4"
        e = OrderedDict()
        for m in ("gpt-oss-120b", "qwen3.8-27b"):
            row = OrderedDict()
            for tag, suffix in (("v1", ""), ("v2", "__v2"), ("v2-think-long", "__v2-think-long")):
                a = arms.get(f"briefing-newword/llm__{m}__single__nshot1{suffix}")
                if a:
                    row[tag] = {kk: a.get(kk) for kk in ("n", "capture", "capture_ci", "pays", "pays_margin0", "p_pays", "vs_strongest_arm")}
                    row[tag]["vs_strongest"] = a.get("vs_strongest")
            row["v2_minus_v1"] = get(c, m, "v2_minus_v1_single_n1")
            if m == "qwen3.8-27b":
                row["think_long_minus_v1"] = get(c, m, "v2-think-long_minus_v1_n1")
            e[m] = row
        out[kind] = e
    return out


def decision(s0: dict, s0b: dict) -> dict:
    prim = OrderedDict()
    for m in MODELS:
        prim[f"s0/{m}"] = get(s0, S0_GROUP, "arms", f"briefing-newword/llm__{m}__single__nshot0__v2")
    prim["s0b-react4/gpt-oss-120b"] = get(s0b, S0B_GROUPS[0], "arms", "briefing-newword/llm__gpt-oss-120b__single__nshot0__v2")
    primary_ids = {id(v) for v in prim.values() if v}
    entries = {}
    for fam, data in (("s0", s0), ("s0b", s0b)):
        for gname, g in data.items():
            for key, row in g["arms"].items():
                # secondary family: every other task-prompt LLM arm in a newword cell (sameword arms are wording controls)
                if (row["meta"].get("arm") == "llm" and row["meta"].get("variant") and row.get("p_pays") is not None
                        and key.startswith("briefing-newword/") and id(row) not in primary_ids):
                    entries[f"{fam}/{gname}/{key}"] = row
    h = a0.holm({k: v["p_pays"] for k, v in entries.items()}) if entries else {}
    surviving = [k for k, ok in h.items() if ok]
    primary_yes = [k for k, v in prim.items() if v and v.get("pays") == "yes"]
    uncorrected = [k for k, v in entries.items() if v.get("pays") == "yes"]
    if primary_yes or surviving:
        verdict = "configuration-driven"
    elif uncorrected:
        verdict = "model-driven, with uncorrected configuration signals"
    else:
        verdict = "model-driven"
    return {"primary_0c": {k: (v and {kk: v.get(kk) for kk in ("capture", "capture_ci", "margin_lo95", "vs_strongest", "vs_strongest_arm", "pays", "p_pays")}) for k, v in prim.items()},
            "primary_yes": primary_yes, "secondary_entries": len(entries), "secondary_uncorrected_yes": uncorrected,
            "holm_surviving": surviving, "verdict": verdict}


def ledger(res: dict) -> str:
    lines = []
    P = res["predictions"]
    f = lambda x: "n/a" if x is None else f"{x:+.3f}" if isinstance(x, float) else str(x)  # noqa: E731
    pj = P["P_J"]["contrast"] or {}
    lines.append(f"P-J 120b v2-v1 single N0 (s0 newword): dcap {f(pj.get('capture_diff'))} CI {pj.get('capture_ci')} threshold +{PJ_GAIN} -> holds={P['P_J']['holds']}")
    pk = P["P_K"]["contrast"] or {}
    lines.append(f"P-K 120b v2 N1-N0: dcap {f(pk.get('capture_diff'))} CI {pk.get('capture_ci')} -> holds={P['P_K']['holds']}")
    pl = P["P_L"]
    lines.append(f"P-L qwen think N0 ({pl.get('arm_used')}): capture {f(pl['capture'])} token ratio {f(pl['token_ratio'])} -> holds={pl['holds']}")
    for m, v in P["P_M"]["models"].items():
        n1 = v["n1"] or {}
        lines.append(f"P-M {m}: LLM v2 N1 - text_bow_matched N1 value {f(n1.get('value_diff'))} lo95 {f(n1.get('value_lo95'))} -> {v['holds_n1']}")
    for m, v in (P["P_N"] or {}).items():
        lines.append(f"P-N {m}: mean dcap over {v['cells']} classic cells {f(v['mean_capture_diff'])} CI {v['mean_ci']} width {v['stage0_width']:.3f} material={v['material']}")
    for kind, e in (P.get("s0b_n1") or {}).items():
        for m, row in e.items():
            parts = [f"{t} cap {f((row.get(t) or {}).get('capture'))} {(row.get(t) or {}).get('pays')}" for t in ("v1", "v2", "v2-think-long") if row.get(t)]
            dv = row.get("v2_minus_v1") or {}
            lines.append(f"s0b {kind} {m} N1: " + "; ".join(parts) + f"; v2-v1 dcap {f(dv.get('capture_diff'))} CI {dv.get('capture_ci')}")
    d = res["decision"]
    lines.append(f"Decision: {d['verdict']}; primary yes {d['primary_yes']}; Holm survivors {len(d['holm_surviving'])}/{d['secondary_entries']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="repository root: --cells defaults to build/briefing_cells, --out to analysis/briefing/task_prompt.json")
    ap.add_argument("--cells", default=None, help="assemble_task_prompt.py output (with s0/, s0b/, classic/)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--B", type=int, default=10_000)
    a = ap.parse_args()
    if a.root:
        a.cells = a.cells or str(Path(a.root) / "build" / "briefing_cells")
        a.out = a.out or str(Path(a.root) / "analysis" / "briefing" / "task_prompt.json")
    if not a.cells:
        ap.error("--cells or --root is required")
    cells = Path(a.cells)
    s0, _ = family(cells / "s0", "all", 100, a.B) if (cells / "s0").is_dir() else ({}, None)
    s0b, _ = family(cells / "s0b", "heldout", 300, a.B) if (cells / "s0b").is_dir() else ({}, None)
    classic = classic_family(cells / "classic", a.B) if (cells / "classic").is_dir() else {}
    res = OrderedDict(s0=s0, s0b=s0b, classic=classic)
    res["predictions"] = predictions(s0, s0b, classic)
    res["decision"] = decision(s0, s0b)
    out = Path(a.out or "analysis/task_prompt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    text = ledger(res)
    out.with_suffix(".ledger.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    print("written", out)


if __name__ == "__main__":
    main()
