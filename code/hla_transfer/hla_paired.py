#!/usr/bin/env python3
"""Paired comparisons for the HLA external transfer audit, on both scales.

The released HLA logs carry the native episode score in the field `reward` and
the binary outcome in `success`; every arm runs the same 100 held-out seeds, so
all comparisons are paired by seed.

Run:  python code/hla_paired.py
"""
import json
import os
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAIN = os.path.join(ROOT, "data", "hla_audit", "out", "main")
CALIB = os.path.join(ROOT, "data", "hla_audit", "out", "calib")
OUT = os.path.join(ROOT, "hla_paired.json")
B = 10000


def load_jsonl(p):
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if isinstance(r, dict):
                    rows.append(r)
    return rows


def paired(a, b, seed=0):
    rng = np.random.default_rng(20260825 + seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    reps = d[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return dict(diff=float(d.mean()),
                lo=float(np.percentile(reps, 2.5)),
                hi=float(np.percentile(reps, 97.5)),
                up95=float(np.percentile(reps, 95.0)),
                n=int(n))


def by_seed(rows, field):
    return {r["seed"]: (float(r[field]) if not isinstance(r[field], bool)
                        else float(r[field])) for r in rows}


arms = {os.path.basename(p)[:-6]: load_jsonl(p)
        for p in sorted(glob.glob(os.path.join(MAIN, "*.jsonl")))}

out = {}
for layout in ("ring", "partition"):
    keys = sorted(k for k in arms if k.startswith(layout + "_"))
    out[layout] = {"arms": {}, "paired": {}}
    for k in keys:
        rows = arms[k]
        sc = [float(r["reward"]) for r in rows]
        su = [1.0 if r["success"] else 0.0 for r in rows]
        out[layout]["arms"][k] = {
            "n": len(rows),
            "success_rate": round(float(np.mean(su)), 4),
            "mean_score": round(float(np.mean(sc)), 4),
            "score_sd": round(float(np.std(sc, ddof=1)), 4),
            "n_decisions_total": int(sum(r.get("n_decisions", 0) for r in rows)),
            "commands_total": int(sum(len(r.get("commands", [])) if isinstance(r.get("commands"), list) else r.get("commands", 0) for r in rows)),
        }
    # every ordered contrast against the two references that matter:
    # the matched router-off arm (scoreonly) and the locked constant.
    refs = [k for k in keys if "scoreonly" in k or "constant" in k]
    for j, ref in enumerate(refs):
        for i, k in enumerate(keys):
            if k == ref:
                continue
            e = {}
            for f, nm in (("reward", "score"), ("success", "success")):
                da, db = by_seed(arms[k], f), by_seed(arms[ref], f)
                seeds = sorted(set(da) & set(db))
                e[nm] = paired([da[s] for s in seeds], [db[s] for s in seeds],
                               seed=17 * j + i)
                e[nm]["n_shared_seeds"] = len(seeds)
            out[layout]["paired"][f"{k}_minus_{ref}"] = e

# full constant sweep on the calibration seeds, both scales
for layout in ("ring", "partition"):
    sweep = {}
    for p in sorted(glob.glob(os.path.join(CALIB, f"{layout}_*.jsonl"))):
        rows = load_jsonl(p)
        if not rows:
            continue
        sweep[os.path.basename(p)[:-6]] = {
            "n": len(rows),
            "mean_score": round(float(np.mean([r["reward"] for r in rows])), 3),
            "success_rate": round(float(np.mean(
                [1.0 if r["success"] else 0.0 for r in rows])), 3),
        }
    out[layout]["calibration_sweep"] = dict(sorted(
        sweep.items(), key=lambda kv: -kv[1]["mean_score"]))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

for layout in out:
    print("=" * 72)
    print(layout.upper())
    for k, v in out[layout]["arms"].items():
        print(f"  {k:34s} n={v['n']:3d} succ={v['success_rate']:.2f} "
              f"score={v['mean_score']:8.2f} decisions={v['n_decisions_total']}")
    print("  -- paired --")
    for k, v in out[layout]["paired"].items():
        s, b = v["score"], v["success"]
        print(f"  {k:56s} dSCORE={s['diff']:+8.2f} [{s['lo']:+.2f},{s['hi']:+.2f}] "
              f"dSUCC={b['diff']:+.3f} [{b['lo']:+.3f},{b['hi']:+.3f}]")
    print("  -- calibration constant sweep (top 6) --")
    for k, v in list(out[layout]["calibration_sweep"].items())[:6]:
        print(f"  {k:40s} n={v['n']:3d} score={v['mean_score']:8.2f} succ={v['success_rate']:.2f}")
print(f"\nwrote {OUT}")
