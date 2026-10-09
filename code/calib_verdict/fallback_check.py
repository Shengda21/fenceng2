"""Fallback rate of every LLM arm on its B episodes (arms above 5% are flagged).

usage: python code/calib_verdict/fallback_check.py [<records folder>]   (default: data/calib_verdict)
"""

import collections
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "calib_verdict")
fb = collections.defaultdict(lambda: [0, 0])
err = collections.Counter()
for f in glob.glob(os.path.join(root, "runs", "**", "llm__*"), recursive=True):
    if f.endswith(".log") or f.endswith(".meta.json"):
        continue
    rel = os.path.relpath(f, os.path.join(root, "runs")).replace(os.sep, "/")
    key = rel.rsplit("/", 1)[0] + "/" + rel.rsplit("/", 1)[1].split("__b1")[0].replace(".jsonl", "").replace(".json", "")
    if f.endswith(".jsonl"):
        recs = [json.loads(line) for line in open(f, encoding="utf-8")]
    else:
        d = json.load(open(f, encoding="utf-8"))
        recs = d["rows"] if isinstance(d, dict) else d
    for r in recs:
        if r.get("phase", "B") != "B":
            continue
        if r.get("total_reward") is None:
            err[key] += 1
        p = (r.get("provenance") or [{}])[0]
        x = p.get("fallback") or bool(r.get("fallback_count", 0)) or any(dd.get("fallback") for dd in (r.get("decisions") or []))
        fb[key][0] += bool(x)
        fb[key][1] += 1
over = []
for k, (a, n) in sorted(fb.items()):
    flag = " <-- above 5%" if a / n > 0.05 else ""
    if flag:
        over.append(k)
    print(f"{k:100s} {a:3d}/{n:3d} {a / n:.3f}{flag}")
print("arms", len(fb), "episode errors", dict(err), "above 5%:", over)
