#!/usr/bin/env python3
"""
Recompute every number that appears in the paper
directly from the raw per-episode logs, and write them to results_all.json.

Run:  python code/analyze.py
"""
import json
import os
import glob
import math
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results_all.json")

B = 10000


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mean_ci(x, z=1.96):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(x.size) if x.size > 1 else 0.0
    return m, m - z * se, m + z * se


def paired_bootstrap(a, b, B=B, seed=0):
    """Paired bootstrap CI for mean(a) - mean(b); a, b aligned by seed."""
    rng = np.random.default_rng(20260825 + seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    n = a.size
    if n == 0:
        return dict(diff=float("nan"), lo=float("nan"), hi=float("nan"),
                    hi_one_sided=float("nan"), n=0)
    d = a - b
    idx = rng.integers(0, n, size=(B, n))
    reps = d[idx].mean(axis=1)
    return dict(
        diff=float(d.mean()),
        lo=float(np.percentile(reps, 2.5)),
        hi=float(np.percentile(reps, 97.5)),
        hi_one_sided=float(np.percentile(reps, 95.0)),
        n=int(n),
    )


def summarise(rows, key_success="success", key_reward="total_reward"):
    n = len(rows)
    k = sum(1 for r in rows if r.get(key_success))
    rew = [float(r.get(key_reward, 0.0)) for r in rows]
    m, lo, hi = mean_ci(rew)
    wlo, whi = wilson(k, n)
    return dict(
        n=n,
        successes=k,
        success_rate=(k / n if n else float("nan")),
        success_ci=[wlo, whi],
        mean_reward=m,
        reward_ci=[lo, hi],
        reward_sd=float(np.std(rew, ddof=1)) if n > 1 else 0.0,
    )


def align_by_seed(rows_a, rows_b, field):
    """Return (va, vb, seeds) aligned on the shared seed set."""
    da, db = {}, {}
    for r in rows_a:
        da.setdefault(r.get("seed", r.get("episode")), r)
    for r in rows_b:
        db.setdefault(r.get("seed", r.get("episode")), r)
    seeds = sorted(set(da) & set(db))
    va = [float(bool(da[s][field])) if isinstance(da[s][field], bool)
          else float(da[s][field]) for s in seeds]
    vb = [float(bool(db[s][field])) if isinstance(db[s][field], bool)
          else float(db[s][field]) for s in seeds]
    return va, vb, seeds


results = {}

# ----------------------------------------------------------------------------
# 1. Matched diagnostic cells: hier_v2 (LLM) vs hier_no_llm (default)
# ----------------------------------------------------------------------------
P3 = os.path.join(DATA, "experiments_phase3", "phase3_results")
CELLS = {
    "hanabi_small2p": ("hanabi_small2p_hierv2", "hanabi_small2p_hiernollm"),
    "smac_3m": ("smac_3m_hierv2", "smac_3m_hiernollm"),
    "smac_8m": ("smac_8m_hierv2", "smac_8m_hiernollm"),
    "smac_MMM": ("smac_MMM_hierv2", "smac_MMM_hiernollm"),
    "magent_m20": ("magent_m20_hierv2", "magent_m20_hiernollm"),
}

diag = {}
for i, (cell, (dllm, dnol)) in enumerate(CELLS.items()):
    pa = glob.glob(os.path.join(P3, dllm, "*.json"))
    pb = glob.glob(os.path.join(P3, dnol, "*.json"))
    if not pa or not pb:
        diag[cell] = {"error": f"missing dirs {dllm} / {dnol}"}
        continue
    ra, rb = load_json(pa[0]), load_json(pb[0])
    entry = {
        "llm_arm": summarise(ra),
        "default_arm": summarise(rb),
        "replan_every": ra[0].get("replan_every"),
        "files": [os.path.relpath(pa[0], ROOT), os.path.relpath(pb[0], ROOT)],
    }
    for j, (field, name) in enumerate((("total_reward", "reward"),
                                       ("success", "success"))):
        va, vb, seeds = align_by_seed(ra, rb, field)
        entry[f"paired_{name}"] = paired_bootstrap(va, vb, seed=10 * i + j)
        entry[f"paired_{name}"]["n_shared_seeds"] = len(seeds)
    dec = defaultdict(int)
    for r in ra:
        dec[str(r.get("decomposition", ""))[:48]] += 1
    entry["llm_emissions"] = dict(sorted(dec.items(), key=lambda kv: -kv[1])[:6])
    dec0 = defaultdict(int)
    for r in rb:
        dec0[str(r.get("decomposition", ""))[:48]] += 1
    entry["default_emissions"] = dict(sorted(dec0.items(), key=lambda kv: -kv[1])[:4])
    diag[cell] = entry
results["diagnostic_cells"] = diag

# ----------------------------------------------------------------------------
# 2. SMAC 8m exhaustive constant ordered-pair sweep
# ----------------------------------------------------------------------------
const_dir = os.path.join(P3, "smac_8m_e2_constant")
const = {}
if os.path.isdir(const_dir):
    for p in sorted(glob.glob(os.path.join(const_dir, "*.json"))):
        rows = load_json(p)
        tag = os.path.basename(p).replace("smac_8m_constant_", "").rsplit("_", 2)[0]
        const[tag] = summarise(rows)
results["smac_8m_constant_pairs"] = const

# ----------------------------------------------------------------------------
# 3. Overcooked injected task-level deployment comparison
#
# Arm assembly. The non-LLM arms are single files under main_run/ (seeds 0-99).
# The LLM arm was collected on a key-rotated serving path in segments under
# main_run_rotated/; overlapping segments resolve first-file-wins in path order,
# so the later "fill_*" batches contribute only seeds the base batches missed.
# This reproduces the shipped oc_stats_results.json exactly.
# ----------------------------------------------------------------------------
OC = os.path.join(DATA, "oc_audit")
OC_LAYOUTS = ["cramped_room", "coordination_ring",
              "forced_coordination", "asymmetric_advantages"]


def oc_llm_arm():
    lay = defaultdict(dict)
    files = [p for p in sorted(glob.glob(
        os.path.join(OC, "main_run_rotated", "**", "*.json"), recursive=True))
        if "manifest" not in os.path.basename(p)]
    for p in reversed(files):            # reversed + overwrite == first-wins
        for r in load_json(p):
            r["_src"] = os.path.relpath(p, ROOT)
            lay[r["layout"]][r["seed"]] = r
    return {k: list(v.values()) for k, v in lay.items()}, files


def oc_other_arms():
    lay = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(OC, "main_run", "*.json"))):
        if "manifest" in os.path.basename(p):
            continue
        rows = load_json(p)
        if not isinstance(rows, list) or not rows:
            continue
        arm = rows[0].get("arm", "?")
        if arm == "llm_task":
            continue
        lay[rows[0].get("layout", "?")][arm] = rows
    return lay


llm_arm, llm_files = oc_llm_arm()
other = oc_other_arms()

oc, oc_paired = {}, {}
for k_lay, layout in enumerate(OC_LAYOUTS):
    arms = dict(other.get(layout, {}))
    if layout in llm_arm:
        arms["llm_task"] = llm_arm[layout]
    oc[layout] = {}
    for arm, rows in sorted(arms.items()):
        s = summarise(rows)
        calls = [r.get("llm_calls", 0) for r in rows]
        if any(calls):
            s["llm_calls_total"] = int(sum(calls))
            s["llm_parse_fallbacks_total"] = int(
                sum(r.get("llm_parse_fallbacks", 0) for r in rows))
            lat = [r.get("llm_latency_sum", 0.0) for r in rows]
            s["llm_latency_sum_total_s"] = float(sum(lat))
        s["episodes_with_error"] = sum(1 for r in rows if r.get("error"))
        oc[layout][arm] = s
    if "llm_task" not in arms:
        continue
    oc_paired[layout] = {}
    for k_arm, (arm, rows) in enumerate(sorted(arms.items())):
        if arm == "llm_task":
            continue
        e = {}
        for j, (field, name) in enumerate((("total_reward", "reward"),
                                           ("success", "success"))):
            va, vb, seeds = align_by_seed(arms["llm_task"], rows, field)
            e[name] = paired_bootstrap(va, vb, seed=100 + 20 * k_lay + 2 * k_arm + j)
            e[name]["n_shared_seeds"] = len(seeds)
        oc_paired[layout][f"llm_minus_{arm}"] = e

for sub in ("calib", "gate_run"):
    d = os.path.join(OC, sub)
    if not os.path.isdir(d):
        continue
    bucket = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if "manifest" in os.path.basename(p):
            continue
        rows = load_json(p)
        if isinstance(rows, list) and rows:
            bucket[rows[0].get("layout", "?")][
                rows[0].get("arm", rows[0].get("method", "?"))] = summarise(rows)
    oc[sub] = {k: dict(v) for k, v in bucket.items()}

oc["_llm_arm_source_files"] = [os.path.relpath(p, ROOT) for p in llm_files]
results["overcooked_deployment"] = oc
results["overcooked_paired"] = oc_paired

for name, fn in (("overcooked_shipped_stats", "oc_stats_results.json"),
                 ("overcooked_calibration_lock", "calibration_lock_task.json")):
    p = os.path.join(OC, fn)
    if os.path.exists(p):
        results[name] = load_json(p)

probe = {}
for d in sorted(glob.glob(os.path.join(OC, "taskprobe_*"))) + \
         sorted(glob.glob(os.path.join(OC, "taskstage0_*"))) + \
         sorted(glob.glob(os.path.join(OC, "stage0_*"))):
    tag = os.path.basename(d)
    bucket = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
        if "manifest" in os.path.basename(p):
            continue
        rows = load_json(p)
        if isinstance(rows, list) and rows:
            key = f"{rows[0].get('layout','?')}|{rows[0].get('arm', rows[0].get('method','?'))}"
            bucket[key] = summarise(rows)
    if bucket:
        probe[tag] = bucket
results["overcooked_probes"] = probe

# ----------------------------------------------------------------------------
# 4. HLA external transfer audit
# ----------------------------------------------------------------------------
HLA = os.path.join(DATA, "hla_audit", "out")


def hla_summarise(rows):
    if not rows:
        return {"n": 0}

    def pick(r, *names):
        for nm in names:
            if nm in r:
                return r[nm]
        return None

    scores = [float(pick(r, "score", "total_reward", "reward") or 0.0) for r in rows]
    out = {"n": len(rows)}
    m, lo, hi = mean_ci(scores)
    out.update(mean_score=m, score_ci=[lo, hi])
    succ = [pick(r, "success") for r in rows]
    if all(s is not None for s in succ):
        k = sum(1 for s in succ if s)
    else:
        k = sum(1 for s in scores if s > 0)
        out["success_def"] = "score>0"
    wlo, whi = wilson(k, len(rows))
    out.update(successes=k, success_rate=k / len(rows), success_ci=[wlo, whi])
    for fld in ("llm_calls", "n_calls", "prompt_tokens", "completion_tokens"):
        vals = [r.get(fld) for r in rows if isinstance(r.get(fld), (int, float))]
        if vals:
            out[f"{fld}_total"] = float(sum(vals))
    return out


hla = {}
for sub in ("main", "stage0", "calib"):
    d = os.path.join(HLA, sub)
    if not os.path.isdir(d):
        continue
    bucket = {}
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        rows = [r for r in load_jsonl(p) if isinstance(r, dict)]
        key = os.path.basename(p)[:-6]
        bucket[key] = hla_summarise(rows)
        if rows:
            bucket[key]["fields"] = sorted(rows[0].keys())
    hla[sub] = bucket

mainrows = {}
d = os.path.join(HLA, "main")
if os.path.isdir(d):
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        mainrows[os.path.basename(p)[:-6]] = [
            r for r in load_jsonl(p) if isinstance(r, dict)]
hla_paired = {}
for layout in ("ring", "partition"):
    keys = [k for k in mainrows if k.startswith(layout)]
    ref = next((k for k in keys if "full" in k or "llm" in k), None)
    if ref is None or not mainrows.get(ref):
        continue
    hla_paired[layout] = {}
    for j, k in enumerate(sorted(keys)):
        if k == ref:
            continue
        e = {}
        for f, nm in (("score", "score"), ("success", "success")):
            if f not in mainrows[ref][0] or not mainrows[k] or f not in mainrows[k][0]:
                continue
            va, vb, seeds = align_by_seed(mainrows[ref], mainrows[k], f)
            e[nm] = paired_bootstrap(va, vb, seed=500 + j)
            e[nm]["n_shared_seeds"] = len(seeds)
        hla_paired[layout][f"{ref}_minus_{k}"] = e
hla["paired"] = hla_paired

lockf = os.path.join(HLA, "calibration_lock.json")
if os.path.exists(lockf):
    hla["calibration_lock"] = load_json(lockf)
results["hla"] = hla


# ----------------------------------------------------------------------------
def clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


with open(OUT, "w", encoding="utf-8") as f:
    json.dump(clean(results), f, indent=1, ensure_ascii=False)
print(f"wrote {OUT}")
for k in results:
    print(" ", k)
