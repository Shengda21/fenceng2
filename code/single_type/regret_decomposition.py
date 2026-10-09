#!/usr/bin/env python3
"""
Empirical decomposition of the matched router effect on the bit-deterministic
engines, plus schedule-cost readings.

Theorem 1 writes the deployment margin as  H_D - rho_Pi + residual.  On an
engine whose episode is a deterministic function of the seed, the second term is
directly observable: whenever the router emits the default atom in every window
the trajectory is bit-identical to the matched default arm and contributes
exactly zero, so the whole matched difference is carried by the episodes in
which the router departed.  This script measures

    p_dev   fraction of episodes in which the router's emission differed
            from the default arm's,
    ell     mean paired difference conditional on departure,
    check   p_dev * ell  against the unconditional matched difference,

and reports how much of each cell's effect is bit-identical (hence structurally
zero) versus genuinely carried by router departures.

It also recomputes the LLM call-cost table from the raw call logs and the
per-turn Hanabi schedule arm.

Run:  python code/regret_decomposition.py
"""
import json
import os
import glob
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P3 = os.path.join(ROOT, "data", "experiments_phase3")
RES = os.path.join(P3, "phase3_results")
LOGS = os.path.join(P3, "logs")
OUT = os.path.join(ROOT, "regret_decomposition.json")
B = 10000

CELLS = {
    "hanabi_small2p": ("hanabi_small2p_hierv2", "hanabi_small2p_hiernollm"),
    "smac_3m": ("smac_3m_hierv2", "smac_3m_hiernollm"),
    "smac_8m": ("smac_8m_hierv2", "smac_8m_hiernollm"),
    "smac_MMM": ("smac_MMM_hierv2", "smac_MMM_hiernollm"),
    "magent_m20": ("magent_m20_hierv2", "magent_m20_hiernollm"),
}


def atoms(dec):
    """Split a recorded decomposition into the atoms the router emitted."""
    s = str(dec)
    for sep in (";", "|", ","):
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [s.strip()]


def emissions_match(a, b):
    """True when every atom the router emitted equals the default's atom.

    Arms may log a different number of windows (the MAgent diagnostic schedule
    records two calls per episode against the default arm's single entry), so
    the test is set-equality against the default atom rather than string
    equality of the whole trace.
    """
    da = set(atoms(a))
    db = set(atoms(b))
    return len(db) == 1 and da == db


def load(d):
    p = glob.glob(os.path.join(RES, d, "*.json"))
    return json.load(open(p[0], encoding="utf-8")) if p else []


def boot(d, seed=0):
    rng = np.random.default_rng(20260825 + seed)
    d = np.asarray(d, float)
    if d.size == 0:
        return None
    reps = d[rng.integers(0, d.size, size=(B, d.size))].mean(axis=1)
    return dict(mean=float(d.mean()), lo=float(np.percentile(reps, 2.5)),
                hi=float(np.percentile(reps, 97.5)), n=int(d.size))


out = {"decomposition": {}, "cost": {}, "hanabi_schedule": {}}

for i, (cell, (da, db)) in enumerate(CELLS.items()):
    A = {r["seed"]: r for r in load(da)}
    Bm = {r["seed"]: r for r in load(db)}
    seeds = sorted(set(A) & set(Bm))
    if not seeds:
        continue
    same, diff = [], []
    same_bit = 0
    for s in seeds:
        a, b = A[s], Bm[s]
        d = float(a["total_reward"]) - float(b["total_reward"])
        if emissions_match(a.get("decomposition"), b.get("decomposition")):
            same.append(d)
            if abs(d) < 1e-12:
                same_bit += 1
        else:
            diff.append(d)
    n = len(seeds)
    p_dev = len(diff) / n
    row = {
        "n": n,
        "n_router_matched_default": len(same),
        "n_router_matched_default_bit_identical": same_bit,
        "n_router_departed": len(diff),
        "p_departure": round(p_dev, 4),
        "mean_diff_when_matched": (round(float(np.mean(same)), 4) if same else None),
        "mean_diff_when_departed": (round(float(np.mean(diff)), 4) if diff else None),
        "unconditional_mean_diff": round(
            float(np.mean(same + diff)), 4),
        "reconstructed_p_times_ell": (
            round(p_dev * float(np.mean(diff)), 4) if diff else 0.0),
    }
    row["conditional_ci"] = boot(diff, seed=i) if diff else None
    row["engine_bit_deterministic_on_matched"] = (
        None if not same else round(same_bit / len(same), 3))
    # what the router actually emitted
    row["router_emission_counts"] = dict(
        Counter(str(A[s].get("decomposition"))[:40] for s in seeds).most_common(5))
    row["default_emission"] = str(Bm[seeds[0]].get("decomposition"))[:40]
    out["decomposition"][cell] = row

# ---------------------------------------------------------------------------
# call-cost table recomputed from the raw call logs
# ---------------------------------------------------------------------------
for p in sorted(glob.glob(os.path.join(LOGS, "llmcalls_*.jsonl"))):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        continue
    key = os.path.basename(p)[len("llmcalls_"):-6]

    def col(*names):
        for nm in names:
            v = [r[nm] for r in rows if isinstance(r.get(nm), (int, float))]
            if v:
                return np.asarray(v, float)
        return np.asarray([], float)

    lat = col("latency", "latency_s", "elapsed", "duration")
    pt = col("prompt_tokens", "ptok")
    ct = col("completion_tokens", "ctok")
    retries = [r for r in rows if r.get("retries") or r.get("retry")]
    out["cost"][key] = {
        "calls": len(rows),
        "lat_p50": (round(float(np.percentile(lat, 50)), 2) if lat.size else None),
        "lat_p95": (round(float(np.percentile(lat, 95)), 2) if lat.size else None),
        "prompt_tokens_mean": (round(float(pt.mean()), 1) if pt.size else None),
        "prompt_tokens_total": (int(pt.sum()) if pt.size else None),
        "completion_tokens_p50": (round(float(np.percentile(ct, 50)), 1) if ct.size else None),
        "completion_tokens_total": (int(ct.sum()) if ct.size else None),
        "retry_rate": round(len(retries) / len(rows), 4),
        "fields": sorted(rows[0].keys()),
    }

# ---------------------------------------------------------------------------
# Hanabi replan-schedule arm (per-turn), against the same matched default
# ---------------------------------------------------------------------------
per = {r["seed"]: r for r in load("hanabi_small2p_hierv2_perturn")}
one = {r["seed"]: r for r in load("hanabi_small2p_hierv2")}
dflt = {r["seed"]: r for r in load("hanabi_small2p_hiernollm")}
if per:
    seeds = sorted(set(per) & set(one) & set(dflt))
    out["hanabi_schedule"] = {
        "n_perturn_completed": len(per),
        "n_shared_seeds": len(seeds),
        "perturn_mean_reward": round(
            float(np.mean([per[s]["total_reward"] for s in per])), 4),
        "perturn_success_rate": round(
            float(np.mean([1.0 if per[s]["success"] else 0.0 for s in per])), 4),
        "onecall_mean_reward_same_seeds": round(
            float(np.mean([one[s]["total_reward"] for s in seeds])), 4),
        "perturn_minus_default": boot(
            [per[s]["total_reward"] - dflt[s]["total_reward"] for s in seeds], seed=91),
        "perturn_minus_onecall": boot(
            [per[s]["total_reward"] - one[s]["total_reward"] for s in seeds], seed=92),
    }
    pc = out["cost"].get("hanabi_hierv2_perturn", {})
    if pc.get("calls") and len(per):
        out["hanabi_schedule"]["calls_per_episode_perturn"] = round(
            pc["calls"] / len(per), 2)
    oc = out["cost"].get("hanabi_hierv2", {})
    if oc.get("calls") and len(one):
        out["hanabi_schedule"]["calls_per_episode_onecall"] = round(
            oc["calls"] / len(one), 2)

json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("=" * 78)
print("MATCHED-EFFECT DECOMPOSITION  (router departed vs reproduced the default)")
print("=" * 78)
for cell, r in out["decomposition"].items():
    ci = r["conditional_ci"]
    cis = f"[{ci['lo']:+.2f},{ci['hi']:+.2f}]" if ci else ""
    print(f"{cell:16s} n={r['n']:3d}  matched={r['n_router_matched_default']:3d} "
          f"(bit-identical {r['engine_bit_deterministic_on_matched']}) "
          f"departed={r['n_router_departed']:3d} p={r['p_departure']:.2f}")
    print(f"{'':16s}   effect|matched={r['mean_diff_when_matched']!s:>8}  "
          f"effect|departed={r['mean_diff_when_departed']!s:>9} {cis}  "
          f"p*ell={r['reconstructed_p_times_ell']!s:>8}  total={r['unconditional_mean_diff']}")
print()
print("=" * 78)
print("CALL COST (recomputed from raw call logs)")
print("=" * 78)
for k, v in out["cost"].items():
    print(f"{k:28s} calls={v['calls']:5d} p50={v['lat_p50']!s:>6}s p95={v['lat_p95']!s:>6}s "
          f"ptok={v['prompt_tokens_mean']!s:>7} ctok_p50={v['completion_tokens_p50']!s:>7} "
          f"retry={v['retry_rate']}")
print()
print("HANABI SCHEDULE ARM")
print(json.dumps(out["hanabi_schedule"], indent=1))
print(f"\nwrote {OUT}")
