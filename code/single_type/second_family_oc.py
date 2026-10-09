#!/usr/bin/env python3
"""The second router family in the Overcooked contest.

This is the half of the second-family run that matters most. On the diagnostic
cells H_D reads zero, so Theorem 3 caps every router alike and a second model
can only confirm a prediction. Overcooked is where the bound is vacuous and only
a contest decides, so it is where "was the first reading model-specific?" is a
real question rather than a rhetorical one.

Comparators come from the released per-episode records. The first-family LLM arm
is assembled by the same first-file-wins tiling analyze.py uses, so the number
this is paired against is the number the manuscript reports.

The arms are read from out/oc3 (enable_thinking:false). The guard is the sidecar
provenance log (data/rerun2026/out/oc3/<layout>/provenance/<layout>.jsonl),
written by the same shim that instruments all three of the repo's independent
LLM call sites. Its `source` field ("model" vs "give_up") is read directly
rather than inferred from whether an emission matches the fallback pair, because
on Hanabi/MAgent (and in principle here) a genuine answer and a give-up default
can be textually identical. A cell whose logged give-up rate exceeds the
threshold is not analyzed.

Run:  python code/second_family_oc.py [--boot 10000]
"""
import argparse
import glob
import json
import os
import random
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OC = os.path.join(ROOT, "data", "oc_audit")
NEW = os.path.join(ROOT, "data", "rerun2026", "out", "oc3")
DEST = os.path.join(ROOT, "second_family_oc.json")

LAYOUTS = ["cramped_room", "coordination_ring",
           "forced_coordination", "asymmetric_advantages"]


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def rows_file(d):
    """The results file in a directory, not the run manifest beside it."""
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if "manifest" in os.path.basename(p):
            continue
        v = load(p)
        if isinstance(v, list) and v:
            return v
    return None


def provenance_stats(layout):
    """Give-up rate and latency straight from the sidecar log, not inferred."""
    f = os.path.join(NEW, layout, "provenance", f"{layout}.jsonl")
    if not os.path.exists(f):
        return None
    recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    give_up = sum(1 for r in recs if r["source"] == "give_up")
    lat = [r["elapsed"] for r in recs if r["source"] == "model" and "elapsed" in r]
    return {
        "n_calls": len(recs), "give_up": give_up,
        "give_up_rate": give_up / len(recs) if recs else None,
        "median_latency": sorted(lat)[len(lat) // 2] if lat else None,
    }


def first_family_llm():
    """First-file-wins tiling in path order, as analyze.py does it."""
    lay = defaultdict(dict)
    files = [p for p in sorted(glob.glob(
        os.path.join(OC, "main_run_rotated", "**", "*.json"), recursive=True))
        if "manifest" not in os.path.basename(p)]
    for p in reversed(files):
        for r in load(p):
            lay[r["layout"]][r["seed"]] = r
    return {k: v for k, v in lay.items()}


def other_arms():
    lay = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(OC, "main_run", "*.json"))):
        if "manifest" in os.path.basename(p):
            continue
        rows = load(p)
        if not isinstance(rows, list) or not rows:
            continue
        arm = rows[0].get("arm") or rows[0].get("method")
        if arm == "llm_task":
            continue
        lay[rows[0]["layout"]][arm] = {r["seed"]: r for r in rows}
    return lay


def paired_boot(a, b, seeds, key, B, seed):
    diffs = [float(a[s][key]) - float(b[s][key]) for s in seeds]
    n = len(diffs)
    rng = random.Random(seed)
    reps = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(B))
    return {"diff": sum(diffs) / n, "lo": reps[int(0.025 * B)],
            "hi": reps[min(int(0.975 * B), B - 1)],
            "up95": reps[min(int(0.95 * B), B - 1)], "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--max-giveup", type=float, default=0.02)
    args = ap.parse_args()

    ff = first_family_llm()
    others = other_arms()
    report = {}

    for lay in LAYOUTS:
        rows = rows_file(os.path.join(NEW, lay))
        if not rows:
            continue
        m2 = {r["episode"]: r for r in rows}

        prov = provenance_stats(lay)
        if prov is None:
            raise SystemExit(
                f"FAIL {lay}: no provenance/{lay}.jsonl found -- this layout's "
                f"episode records cannot be trusted without it")
        if prov["give_up_rate"] is not None and prov["give_up_rate"] > args.max_giveup:
            raise SystemExit(
                f"FAIL {lay}: give-up rate {100*prov['give_up_rate']:.1f}% "
                f"exceeds {100*args.max_giveup:.0f}% -- do not analyze this "
                f"layout's contrasts")

        arms = dict(others.get(lay, {}))
        if lay in ff:
            arms["llm_task_family1"] = ff[lay]

        entry = {
            "layout": lay,
            "n": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "reward": statistics.mean(r["total_reward"] for r in rows),
            "success": statistics.mean(bool(r["success"]) for r in rows),
            "provenance": prov,
            "median_minutes": statistics.median(r["duration"] / 60 for r in rows),
            "contrasts": {},
        }

        for i, (name, other) in enumerate(sorted(arms.items())):
            shared = sorted(set(m2) & set(other))
            if len(shared) != len(m2) or len(shared) != len(other):
                entry["contrasts"][name] = {
                    "skipped": f"seed sets differ ({len(m2)} vs {len(other)}, "
                               f"{len(shared)} shared)"}
                continue
            entry["contrasts"][name] = {
                "reward": paired_boot(m2, other, shared, "total_reward",
                                      args.boot, 3000 + i),
                "success": paired_boot(m2, other, shared, "success",
                                       args.boot, 4000 + i),
            }
        report[lay] = entry

    json.dump(report, open(DEST, "w", encoding="utf-8"), indent=1)

    def f(d):
        return (f"{d['diff']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]"
                if d and "diff" in d else d.get("skipped", "--")[:28])

    print(f"{'layout':24s} {'rew':>7s} {'succ':>5s} {'give_up':>8s} "
          f"{'vs LinUCB':>22s} {'vs family 1':>22s} {'vs constant':>22s}")
    for lay, e in report.items():
        c = e["contrasts"]
        p = e["provenance"]
        print(f"{lay:24s} {e['reward']:7.1f} {e['success']:5.2f} "
              f"{100*p['give_up_rate']:7.1f}% "
              f"{f(c.get('ctx_linucb_task', {}).get('reward', {})):>22s} "
              f"{f(c.get('llm_task_family1', {}).get('reward', {})):>22s} "
              f"{f(c.get('fixed_task', {}).get('reward', {})):>22s}")
    print(f"\nwrote {os.path.relpath(DEST, ROOT)}")


if __name__ == "__main__":
    main()
