#!/usr/bin/env python3
"""
Derived quantities: the deployment-headroom table.

For every audited cell this computes, on each reported scale,

    V_default        return of the domain default (constant tau_default)
    V_const_star     return of the best measured constant skill
    dVmax            = V_const_star - V_default            (constant-recoverable gap)
    Delta_def        = contextual default gap (per-window oracle, where measurable)
    H_D              = Delta_def - dVmax                    (deployment headroom)
                       or a lower bound  H_D >= V_sel - V_const_star
                       supplied by the strongest measured state-dependent selector
    margin(x)        = V_x - V_const_star  for each selector x  (deployment margin)
    capture(x)       = margin(x) / H_D_lower_bound           (upper bound on the
                       fraction of available headroom that selector x collected)

Inputs: results_all.json (recomputed from raw logs by code/analyze.py) plus the
probe readings that live only in the supplementary tables, listed in
PROBES below with their provenance.

Run:  python code/derive_headroom.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_all.json")
OUT = os.path.join(ROOT, "headroom_table.json")

with open(RES, "r", encoding="utf-8") as f:
    R = json.load(f)

# ---------------------------------------------------------------------------
# Probe readings that are not recomputable from the released per-episode logs.
# Provenance is recorded per entry; every value matches the supplementary
# material.
# ---------------------------------------------------------------------------
PROBES = {
    # Probe 1: exhaustive constant-tau sweep (separate batch, original route).
    # Values as reported in the shipped paired analysis of the same batch,
    # data/experiments_phase3/paired_analysis_diagnostic_cloud.json.
    "constant_sweep": {
        "hanabi_small2p": {"best_tau": "PLAY_KNOWN", "reward": 5.20, "success": 1.00,
                           "default_tau": "DISCARD_OLDEST",
                           "default_reward": 0.00, "default_success": 0.00},
        "smac_3m": {"best_tau": "focus_fire", "reward": 15.5342, "success": 0.66,
                    "default_tau": "focus_fire",
                    "default_reward": 15.5342, "default_success": 0.66},
        "smac_8m": {"best_tau": "focus_fire", "reward": 17.2769, "success": 0.66,
                    "default_tau": "focus_fire",
                    "default_reward": 17.2769, "default_success": 0.66},
        "smac_MMM": {"best_tau": "focus_fire", "reward": 10.5281, "success": 0.12,
                     "default_tau": "focus_fire",
                     "default_reward": 10.5281, "default_success": 0.12},
        "magent_m20": {"best_tau": "ATTACK_FORWARD", "reward": 22.3558, "success": 0.99,
                       "default_tau": "ATTACK_FORWARD",
                       "default_reward": 22.3558, "default_success": 0.99},
    },
    # Probe 2: per-window oracle, contextual default gap Delta_def.
    # Cloneable engines only (Hanabi, MAgent). SI "Oracle contextual default gap".
    "oracle": {
        "hanabi_small2p": {"N": 50, "success": 1.00, "reward": None},
        "magent_m20": {"N": 60, "success": 0.02, "reward": 1.85},
    },
}

SOURCE = {
    "constant_sweep": "data/experiments_phase3/paired_analysis_diagnostic_cloud.json"
                      " (best_const per cell, original serving batch)",
    "oracle": "ver.4 supplementary, 'Oracle contextual default gap'"
              " (per-window branching on the two cloneable engines)",
}


def fmt(x, nd=2):
    return None if x is None else round(float(x), nd)


table = {"diagnostic_cells": {}, "overcooked": {}, "hla": {}, "_source": SOURCE}

# The five matched diagnostic cells are not derived here: their ledger is
# assembled by code/ledger.py and code/forest_rows.py, which keep the
# comparator convention of the manuscript. This file covers the Overcooked and
# HLA blocks that the figures read.

# ---------------------------------------------------------------------------
# 2. Overcooked deployment interface
#
# Under the deployment interface (shared I1/I2 overrides, proposal resolver) the
# locked constant pair is get_onion,get_dish; the exhaustive 64-pair probe under
# the bare mapper scores 0% on all four layouts, so no constant pair is competent
# absent the overrides.  H_D is bounded below by the strongest measured
# state-dependent selector minus the best measured constant.
# ---------------------------------------------------------------------------
OCARMS = {"llm_task": "LLM", "scripted_task": "scripted",
          "ctx_linucb_task": "LinUCB", "random_task": "random",
          "fixed_task": "constant"}
for layout, arms in R["overcooked_deployment"].items():
    if layout.startswith("_") or layout in ("calib", "gate_run"):
        continue
    row = {"arms": {}}
    # The released logs show 20 task assignments per 400-step episode for the
    # llm / LinUCB / random / constant arms (timestamped 0, 20, ..., 380) and 40
    # for the scripted state machine (a running decision counter).  The schedule
    # is therefore verified matched only across the first four arms, and the
    # headroom lower bound is taken over those; the scripted machine is carried
    # as an engineering reference at an unmatched, finer decision cadence.
    MATCHED = ("ctx_linucb_task", "llm_task", "random_task")
    for scale, fld in (("success", "success_rate"), ("reward", "mean_reward")):
        vconst = arms["fixed_task"][fld]
        vsel = max(arms[a][fld] for a in MATCHED if a in arms)
        hd_lb = max(0.0, vsel - vconst)
        entry = {"V_const_star": fmt(vconst),
                 "H_D_lower_bound": fmt(hd_lb),
                 "H_D_status": "lower bound = best matched-cadence selector - best constant",
                 "unmatched_reference_scripted": fmt(arms["scripted_task"][fld]),
                 "scripted_margin": fmt(arms["scripted_task"][fld] - vconst)}
        for a, nm in OCARMS.items():
            if a not in arms or a == "fixed_task":
                continue
            m = arms[a][fld] - vconst
            entry[f"margin_{nm}"] = fmt(m)
            entry[f"capture_{nm}"] = (None if hd_lb <= 0 else fmt(m / hd_lb, 3))
        row[scale] = entry
    for a in arms:
        row["arms"][a] = {"n": arms[a]["n"],
                          "success_rate": fmt(arms[a]["success_rate"], 3),
                          "mean_reward": fmt(arms[a]["mean_reward"])}
    table["overcooked"][layout] = row

# ---------------------------------------------------------------------------
# 3. HLA external transfer
#     arms: <layout>_full      = native selector with the LLM prior
#           <layout>_scoreonly = native selector, prior removed (matched router-off)
#           <layout>_constant_*= locked best constant macro-action
#           <layout>_linucb    = zero-inference contextual bandit
# ---------------------------------------------------------------------------
main = R["hla"].get("main", {})
calib = R["hla"].get("calib", {})
for layout in ("ring", "partition"):
    arms = {k: v for k, v in main.items() if k.startswith(layout + "_")}
    if not arms:
        continue
    const_key = next((k for k in arms if "constant" in k), None)
    row = {"arms": {k: {"n": v["n"], "success_rate": fmt(v.get("success_rate"), 3),
                        "mean_score": fmt(v.get("mean_score"))} for k, v in arms.items()},
           "locked_constant": const_key}
    # full constant sweep on the calibration seeds
    sweep = {k: fmt(v.get("mean_score")) for k, v in calib.items()
             if k.startswith(layout + "_constant_")}
    row["calibration_constant_sweep"] = dict(sorted(
        sweep.items(), key=lambda kv: -(kv[1] if kv[1] is not None else -1e9)))
    for scale, fld in (("success", "success_rate"), ("score", "mean_score")):
        if const_key is None:
            continue
        vconst = arms[const_key][fld]
        cands = [arms[k][fld] for k in arms if k != const_key]
        vsel = max(cands) if cands else vconst
        hd_lb = max(0.0, vsel - vconst)
        entry = {"V_const_star": fmt(vconst),
                 "H_D_lower_bound": fmt(hd_lb),
                 "H_D_status": "lower bound = best measured selector - locked constant"}
        for k in arms:
            if k == const_key:
                continue
            nm = k[len(layout) + 1:]
            m = arms[k][fld] - vconst
            entry[f"margin_{nm}"] = fmt(m)
            entry[f"capture_{nm}"] = (None if hd_lb <= 0 else fmt(m / hd_lb, 3))
        row[scale] = entry
    table["hla"][layout] = row

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(table, f, indent=1, ensure_ascii=False)

# ---------------------------------------------------------------------------
# console report
# ---------------------------------------------------------------------------
print("=" * 78)
print("DEPLOYMENT HEADROOM TABLE")
print("=" * 78)
print("\n-- five matched diagnostic cells --")
for cell, row in table["diagnostic_cells"].items():
    print(f"\n{cell}  (default={row['default']}, best constant={row['best_constant']})")
    for scale in ("success", "reward"):
        s = row[scale]
        print(f"   {scale:8s} Vdef={s['V_default']!s:>8} V*={s['V_const_star']!s:>8} "
              f"dVmax={s['dVmax']!s:>7} Ddef={s['Delta_def_oracle']!s:>6} "
              f"H_D={s['H_D']!s:>6}  margin_LLM={s.get('margin_llm')!s:>7} "
              f"{s.get('margin_llm_ci','')}")

print("\n-- Overcooked deployment interface --")
for layout, row in table["overcooked"].items():
    print(f"\n{layout}")
    for scale in ("success", "reward"):
        s = row[scale]
        print(f"   {scale:8s} V*const={s['V_const_star']!s:>8}  H_D>={s['H_D_lower_bound']!s:>8}"
              f"   LLM={s['margin_LLM']!s:>8} ({s['capture_LLM']!s:>6})"
              f"   scripted={s['margin_scripted']!s:>8} ({s['capture_scripted']!s:>6})"
              f"   LinUCB={s['margin_LinUCB']!s:>8} ({s['capture_LinUCB']!s:>6})")

print("\n-- HLA external transfer --")
for layout, row in table["hla"].items():
    print(f"\n{layout}  locked constant = {row['locked_constant']}")
    for scale in ("success", "score"):
        if scale not in row:
            continue
        s = row[scale]
        keys = [k for k in s if k.startswith("margin_")]
        parts = " ".join(f"{k[7:]}={s[k]!s}({s['capture_' + k[7:]]!s})" for k in keys)
        print(f"   {scale:8s} V*const={s['V_const_star']!s:>8}  H_D>={s['H_D_lower_bound']!s:>8}  {parts}")

print(f"\nwrote {OUT}")
