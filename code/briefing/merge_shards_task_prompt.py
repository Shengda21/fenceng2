"""Check and merge GPU seed shards into one file per arm.

usage: merge_shards_task_prompt.py [--arms tasks/arms_0c.json] [--models m1,m2] [--check-only] [--remap /root/oghp/stage0c=LOCAL]
For each arm (one cell x one LLM configuration) in arms_0c.json: every shard must exist with exactly its seeds, in order.
Complete arms are concatenated in shard order (= the deployment seed order), episodes renumbered 0..T-1, and written to
runs/<group>/<cell>/<file>.jsonl. The report (merge_report_0c.json next to arms_0c.json, or --report) lists missing or short
shards and every arm's fallback rate, finish_reason=length rate and parse rules."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def parse_range(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def read(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arm_status(arm: dict, remap) -> dict:
    rows, problems = [], []
    for sh in arm["shards"]:
        recs = read(remap(sh["out"]))
        want = parse_range(sh["seeds"])
        if recs is None:
            problems.append({"seeds": sh["seeds"], "problem": "missing"})
            continue
        got = [r["seed"] for r in recs]
        if got != want:
            problems.append({"seeds": sh["seeds"], "problem": f"seeds {len(got)}/{len(want)}"})
            continue
        rows.extend(recs)
    calls = [c for r in rows for c in r["provenance"][0].get("extra", {}).get("calls", [])]
    return {
        "complete": not problems and len(rows) == arm["T"],
        "problems": problems,
        "episodes": len(rows),
        "fallback_rate": sum(r.get("fallback_count", 0) > 0 for r in rows) / max(1, len(rows)),
        "length_finish_rate": sum(c.get("finish_reason") == "length" for c in calls) / max(1, len(calls)),
        "calls": len(calls),
        "parse_rules": dict(Counter(r["provenance"][0].get("extra", {}).get("parse_rule") for r in rows)),
        "_rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="/root/oghp/stage0c/tasks/arms_0c.json")
    ap.add_argument("--models", default=None, help="comma list; default all")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--remap", default=None, help="OLD=NEW path prefix (local tests)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--allow-partial", action="store_true", help="also merge arms with missing shards (long pass stopped at its cap); "
                    "the merged file holds the complete shards only and the report marks it partial")
    a = ap.parse_args()
    old, new = (a.remap.split("=", 1) if a.remap else ("", ""))
    remap = (lambda p: p.replace(old, new, 1)) if a.remap else (lambda p: p)
    arms = json.load(open(a.arms, encoding="utf-8"))
    models = set(a.models.split(",")) if a.models else None
    report, merged = {}, 0
    for key, arm in arms.items():
        if models and arm["model"] not in models:
            continue
        st = arm_status(arm, remap)
        rows = st.pop("_rows")
        report[key] = st
        st["partial_merged"] = bool(a.allow_partial and not st["complete"] and rows)
        if (st["complete"] or st["partial_merged"]) and not a.check_only:
            first = remap(arm["shards"][0]["out"])
            out = first.replace("/runs_shards/", "/runs/").rsplit("__seeds", 1)[0] + ".jsonl"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as handle:
                for i, r in enumerate(rows):
                    r["episode"] = i
                    handle.write(json.dumps(r, sort_keys=True) + "\n")
            merged += 1
    path = a.report or os.path.join(os.path.dirname(a.arms), "merge_report_0c.json")
    json.dump(report, open(path, "w", encoding="utf-8"), indent=1)
    incomplete = {k: v["problems"] for k, v in report.items() if not v["complete"]}
    over = {k: round(v["fallback_rate"], 3) for k, v in report.items() if v["fallback_rate"] > 0.05}
    print(json.dumps({"arms": len(report), "merged": merged, "incomplete": len(incomplete), "over_5pct_fallback": over,
                      "max_length_finish_rate": max((v["length_finish_rate"] for v in report.values()), default=0)}, indent=1))
    if incomplete:
        print("INCOMPLETE", json.dumps(incomplete)[:2000])
    sys.exit(0 if not incomplete else 2)


if __name__ == "__main__":
    main()
