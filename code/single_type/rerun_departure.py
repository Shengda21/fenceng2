#!/usr/bin/env python3
"""Departure decomposition on the re-run.

Theorem 1 writes the deployment margin as H_D - rho_Pi + residual. Where the
episode is a deterministic function of the seed, the regret term is observable:
if the router emits the default in every window the trajectory is bit-identical
to the matched default arm and contributes exactly zero, so the whole matched
difference is carried by the episodes in which the router departed.

Run:  python code/rerun_departure.py
"""
import glob
import json
import math
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "data", "rerun2026", "out")
OUT = os.path.join(ROOT, "rerun_departure.json")
B = 10000

CELLS = [("hanabi", "small_2p", "DISCARD_OLDEST", "Hanabi-Small"),
         ("smac", "3m", "focus_fire", "SMAC 3m"),
         ("smac", "8m", "focus_fire", "SMAC 8m"),
         ("smac", "MMM", "focus_fire", "SMAC MMM"),
         ("magent", "m20", "ATTACK_FORWARD", "MAgent m20")]


def rows_of(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def atoms(dec):
    s = str(dec)
    for sep in (";", "|", ","):
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [s.strip()]


def boot(d, seed=0):
    if not d:
        return None
    r = random.Random(20260826 + seed)
    n = len(d)
    reps = sorted(sum(r.choice(d) for _ in range(n)) / n for _ in range(B))
    return {"mean": sum(d) / n, "lo": reps[int(0.025 * B)],
            "hi": reps[int(0.975 * B)], "n": n}


res = {}
for i, (dom, cell, dflt, label) in enumerate(CELLS):
    lp = glob.glob(f"{OUT_DIR}/llm/{dom}_{cell}_llm_t0.5_*.json")
    fp = f"{OUT_DIR}/probe1/{dom}_{cell}_fixed_default.json"
    if not lp or not os.path.exists(fp):
        continue
    A = {r["seed"]: r for r in rows_of(lp[0])}
    Bm = {r["seed"]: r for r in rows_of(fp)}
    seeds = sorted(set(A) & set(Bm))

    same, diff, same_bit = [], [], 0
    for s in seeds:
        d = float(A[s]["total_reward"]) - float(Bm[s]["total_reward"])
        reproduced = set(atoms(A[s].get("decomposition"))) == {dflt}
        if reproduced:
            same.append(d)
            if abs(d) < 1e-12:
                same_bit += 1
        else:
            diff.append(d)

    n = len(seeds)
    p = len(diff) / n if n else 0.0
    ell = sum(diff) / len(diff) if diff else 0.0
    res[f"{dom}/{cell}"] = {
        "label": label, "n": n,
        "n_reproduced": len(same), "n_departed": len(diff),
        "p_departure": p,
        "bit_identical_share": (same_bit / len(same)) if same else None,
        "effect_when_reproduced": (sum(same) / len(same)) if same else None,
        "effect_when_departed": ell,
        "effect_when_departed_ci": boot(diff, i),
        "p_times_ell": p * ell,
        "unconditional_effect": sum(same + diff) / n if n else None,
        "router_emissions": dict(Counter(str(A[s].get("decomposition"))[:44]
                                         for s in seeds).most_common(5)),
    }

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)

hdr = (f"{'cell':14s} {'n':>4} {'p_dep':>6} {'bit-id':>7} {'eff|repr':>9} "
       f"{'eff|departed':>26} {'p*ell':>8} {'total':>8}")
print(hdr)
print("-" * len(hdr))
for k, r in res.items():
    ci = r["effect_when_departed_ci"]
    bi = "---" if r["bit_identical_share"] is None else f"{r['bit_identical_share']:.2f}"
    er = "---" if r["effect_when_reproduced"] is None else f"{r['effect_when_reproduced']:+.2f}"
    print(f"{r['label']:14s} {r['n']:4d} {r['p_departure']:6.2f} {bi:>7} {er:>9} "
          f"{ci['mean']:+8.2f} [{ci['lo']:+6.2f},{ci['hi']:+6.2f}] "
          f"{r['p_times_ell']:+8.2f} {r['unconditional_effect']:+8.2f}")
print(f"\nwrote {OUT}")
