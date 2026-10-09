#!/usr/bin/env python3
"""The second router family, paired against the first and against the constants.

Reads data/rerun2026/out/llm3/*.json (the qwen3.5, enable_thinking:false arms)
and pairs them, by environment seed, against the arms already in the release:
the matched default, the best constant from the Probe-1 sweep, and the
gpt-oss-120b router.

enable_thinking:false keeps qwen3.5's Overcooked-scale latency well under a
second. Every call carries a sidecar provenance record
(data/rerun2026/out/llm3/provenance/*.jsonl) written by the same shim regardless
of which of the three call sites issued it, with an explicit source field:
"model" (a real answer, possibly after retries) or "give_up" (every attempt
exhausted). That field is what makes the checks below checks of fact rather than
inference from the row format, which cannot make this distinction on its own.

Two things are checked before any contrast is reported:

  * give-up rate, read directly from the sidecar log, not inferred from
    whether an emission equals the domain default (a real answer and a
    give-up can be textually identical, e.g. Hanabi and MAgent).
  * seed alignment.  A paired bootstrap over mismatched seed lists is not a
    paired bootstrap.  The seed sets must be identical or the cell is skipped.

Run:  python code/second_family.py [--boot 10000]
"""
import argparse
import glob
import json
import os
import random
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTD = os.path.join(ROOT, "data", "rerun2026", "out")
LEDGER = os.path.join(ROOT, "rerun_ledger.json")
DEST = os.path.join(ROOT, "second_family.json")

# cell -> probe1 file stem
CELLS = {
    "hanabi/small_2p": "hanabi_small_2p",
    "magent/m20": "magent_m20",
    "smac/3m": "smac_3m",
    "smac/8m": "smac_8m",
    "smac/MMM": "smac_MMM",
}


def rows(path):
    """seed -> row, for one arm."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    return {r["seed"]: r for r in d["rows"]}


def provenance_stats(stem):
    """Give-up rate and latency straight from the sidecar log, not inferred."""
    f = os.path.join(OUTD, "llm3", "provenance", f"{stem}_full3.jsonl")
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


def paired_boot(a, b, seeds, key, B, seed):
    """Paired bootstrap of mean(a) - mean(b) over shared seeds.

    One seeded stream per comparison, so endpoints are reproducible and do not
    shift when other comparisons are added or reordered.
    """
    diffs = [float(a[s][key]) - float(b[s][key]) for s in seeds]
    n = len(diffs)
    point = sum(diffs) / n
    rng = random.Random(seed)
    reps = []
    for _ in range(B):
        reps.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    reps.sort()
    lo = reps[int(0.025 * B)]
    hi = reps[min(int(0.975 * B), B - 1)]
    up95 = reps[min(int(0.95 * B), B - 1)]
    return {"diff": point, "lo": lo, "hi": hi, "up95": up95, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--max-giveup", type=float, default=0.02)
    args = ap.parse_args()

    led = json.load(open(LEDGER, encoding="utf-8"))
    report = {}

    for cell, stem in CELLS.items():
        m2_files = glob.glob(os.path.join(OUTD, "llm3", f"{stem}_llm_*_full3.json"))
        if not m2_files:
            continue
        m2 = rows(m2_files[0])

        prov = provenance_stats(stem)
        if prov is None:
            raise SystemExit(
                f"FAIL {cell}: no provenance/{stem}_full3.jsonl found -- this "
                f"cell's episode records cannot be trusted without it "
                f"(see the docstring: an emission and a give-up can be "
                f"textually identical)")
        if prov["give_up_rate"] is not None and prov["give_up_rate"] > args.max_giveup:
            raise SystemExit(
                f"FAIL {cell}: give-up rate {100*prov['give_up_rate']:.1f}% "
                f"exceeds {100*args.max_giveup:.0f}% -- do not analyze this "
                f"cell's contrasts")

        emis = [str(r.get("decomposition")) for r in m2.values()]
        errs = sum(1 for r in m2.values() if r.get("error"))
        distinct = len(set(emis))

        # --- comparators, all from the released per-episode records ---
        best_by_reward = led[cell]["best_constant"]["by_reward"]
        cmp_paths = {
            "matched_default": f"{stem}_fixed_default.json",
            "best_constant": f"{stem}_const-{best_by_reward}.json",
        }
        arms = {}
        for name, fn in cmp_paths.items():
            arms[name] = rows(os.path.join(OUTD, "probe1", fn))
        f1 = glob.glob(os.path.join(OUTD, "llm", f"{stem}_llm_t0.5_t05.json"))
        arms["gpt_oss_120b"] = rows(f1[0]) if f1 else None

        entry = {
            "label": led[cell]["label"],
            "n": len(m2),
            "errors": errs,
            "distinct_emissions": distinct,
            "emission_top": sorted(
                {e: emis.count(e) for e in set(emis)}.items(),
                key=lambda kv: (-kv[1], str(kv[0])))[:5],  # ties by name, so the output does not depend on hash seed
            "provenance": prov,
            "best_constant_by_reward": best_by_reward,
            "median_duration_s": statistics.median(
                r["duration"] for r in m2.values()),
            "contrasts": {},
        }

        for i, (name, other) in enumerate(arms.items()):
            if other is None:
                continue
            shared = sorted(set(m2) & set(other))
            if len(shared) != len(m2) or len(shared) != len(other):
                entry["contrasts"][name] = {
                    "skipped": f"seed sets differ "
                               f"({len(m2)} vs {len(other)}, {len(shared)} shared)"}
                continue
            entry["contrasts"][name] = {
                "success": paired_boot(m2, other, shared, "success",
                                       args.boot, 1000 + i),
                "reward": paired_boot(m2, other, shared, "total_reward",
                                      args.boot, 2000 + i),
            }
        report[cell] = entry

    json.dump(report, open(DEST, "w", encoding="utf-8"), indent=1)

    def f(d, nd=3):
        return (f"{d['diff']:+.{nd}f} [{d['lo']:+.{nd}f},{d['hi']:+.{nd}f}]"
                if d and "diff" in d else "--")

    print(f"{'cell':14s} {'n':>4s} {'give_up':>8s} {'med_s':>7s}  "
          f"{'vs best constant (reward)':>30s}  {'vs gpt-oss-120b (reward)':>30s}")
    for cell, e in report.items():
        c = e["contrasts"]
        p = e["provenance"]
        print(f"{e['label']:14s} {e['n']:4d} "
              f"{100*p['give_up_rate']:7.1f}% {p['median_latency']:7.2f}  "
              f"{f(c.get('best_constant', {}).get('reward')):>30s}  "
              f"{f(c.get('gpt_oss_120b', {}).get('reward')):>30s}")
        print(f"   emissions: {e['emission_top']}")
    print(f"\nwrote {os.path.relpath(DEST, ROOT)}")


if __name__ == "__main__":
    main()
