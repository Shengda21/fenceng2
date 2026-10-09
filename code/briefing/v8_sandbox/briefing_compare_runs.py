"""First run vs the rerun: fallback rate and capture per model x mode (LLM arms, every group)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def collect(res: dict) -> dict:
    out = {}
    for g, e in res["groups"].items():
        for c, cc in e["cells"].items():
            for k, a in cc["arms"].items():
                if a["meta"]["arm"] == "llm" and a.get("n"):
                    out[(g, c, k)] = a
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run1")
    parser.add_argument("rerun")
    args = parser.parse_args()
    a, b = collect(json.load(open(args.run1, encoding="utf-8"))), collect(json.load(open(args.rerun, encoding="utf-8")))
    agg = defaultdict(lambda: {"fb1": [], "fb2": [], "c1": [], "c2": []})
    for key in sorted(set(a) & set(b)):
        m = b[key]["meta"]
        row = agg[(m["model"], m["mode"])]
        row["fb1"].append(a[key]["fallback_rate"])
        row["fb2"].append(b[key]["fallback_rate"])
        row["c1"].append(a[key]["capture"])
        row["c2"].append(b[key]["capture"])
    print("| model | mode | arms | fallback run 1 (mean / max) | fallback rerun (mean / max) | mean capture run 1 | mean capture rerun |")
    print("|---|---|---|---|---|---|---|")
    for (model, mode), r in sorted(agg.items()):
        mean = lambda v: sum(v) / len(v)
        print(f"| {model} | {mode} | {len(r['fb1'])} | {mean(r['fb1']):.3f} / {max(r['fb1']):.2f} | {mean(r['fb2']):.3f} / {max(r['fb2']):.2f} | {mean(r['c1']):.2f} | {mean(r['c2']):.2f} |")


if __name__ == "__main__":
    main()
