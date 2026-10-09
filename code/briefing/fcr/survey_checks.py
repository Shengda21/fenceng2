"""Consistency checks on the locks, calibration rows, N-shot rows and exact replay (read only; no FCR prediction is computed here).

1. Lock identity: for each (M, held-out set), the locks of sigma 0 and 0.5 and of both deployments carry the same text rows,
   value table and best responses.
2. Calibration rows: text_rows = seen types x every combination of phrasings 0-2; the lock's best response of every seen
   type is the argmax of its value row.
3. N-shot rows: the labelled examples parsed from every recorded prompt of the bare-prompt and task-prompt N=1 arms equal
   the rows rebuilt by CalibSetting.rows_n_bare / rows_n_task (same texts, same labels, same set in every episode).
4. Exact replay: every recorded episode of every non-fixed arm in the briefing deployments returns exactly the value of the
   fixed strategy it chose on that seed.

usage: python code/briefing/fcr/survey_checks.py [--root <repository root>]  -> analysis/briefing/fcr_survey_checks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

import fcr_common as fc


def lock_identity(root: Path) -> dict:
    sig = OrderedDict()
    for p in sorted((root / "data" / "briefing" / "locks").glob("*/lock_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        key = fc.setting_name(d["config"]["M"], fc.CELL_RE.match(p.stem[len("lock_"):] + "-nonllm").group(6))
        blob = json.dumps([d["text_rows"], d["value_table"], d["best_response"], d["held_out"]], sort_keys=True)
        sig.setdefault(key, {})[str(p.relative_to(root)).replace("\\", "/")] = hashlib.sha256(blob.encode()).hexdigest()
    return {k: {"locks": v, "identical": len(set(v.values())) == 1} for k, v in sig.items()}


def calib_rows(root: Path) -> dict:
    out = OrderedDict()
    for name in fc.SETTINGS:
        s = fc.CalibSetting(root, name)
        n_axes = len(fc.TRAITS[s.M])
        argmax_ok = all(max(s.pool, key=lambda tau: (s.values_seen[t][tau], tau)) == s.best_response[t] for t in s.seen)
        gaps = []
        for t in s.seen:
            v = sorted(s.values_seen[t].values())
            gaps.append(v[-1] - v[-2])
        per_type = Counter(r["type"] for r in s.text_rows)
        out[name] = {
            "lock": s.lock_path, "held_out": s.held_out, "n_seen": len(s.seen), "n_text_rows": len(s.text_rows),
            "rows_per_seen_type": sorted(set(per_type.values())), "expected_rows_per_type": 3 ** n_axes,
            "phrasings_in_rows": sorted({v for r in s.text_rows for v in r["variants"]}),
            "seen_argmax_equals_best_response": bool(argmax_ok), "seen_min_gap": float(min(gaps)),
            "n_bare_rows": len(s.rows_n_bare()), "n_task_rows": len(s.rows_n_task()),
            "n_bare_variants": sorted({tuple(r["variants"]) for r in s.rows_n_bare()}),
        }
    return out


BARE_EX = re.compile(r"Observation: (.*?)\nStrategy: ([A-Z_]+)")
TASK_EX = re.compile(r"Opponent: (.*?)\nBest strategy for you: ([A-Z_]+)")


def nshot_check(root: Path) -> dict:
    out = OrderedDict()
    data = root / "data" / "briefing"
    for deploy, base in (("all", data / "all"), ("heldout", data / "heldout"), ("task_prompt/all", data / "task_prompt" / "all"),
                         ("task_prompt/heldout", data / "task_prompt" / "heldout")):
        for cell in sorted(p for p in base.iterdir() if p.is_dir()):
            m = fc.CELL_RE.match(cell.name)
            if not m or m.group(7) not in ("briefing", "briefing_prefix"):
                continue
            s = fc.CalibSetting(root, fc.setting_name(int(m.group(2)), m.group(6)))
            for f in fc.arm_files(cell):
                meta = fc.parse_arm(f.name)
                if meta["arm"] != "llm" or meta["nshot"] != 1:
                    continue
                task = str(meta.get("variant", "")).startswith("v2")
                want = sorted((r["text"], s.best_response[r["type"]]) for r in (s.rows_n_task() if task else s.rows_n_bare()))
                sets, n = Counter(), 0
                for rec in fc.read_raw(f):
                    prompt = rec["provenance"][0]["extra"]["prompt"]
                    got = sorted((a.strip(), b) for a, b in (TASK_EX if task else BARE_EX).findall(prompt))
                    sets[json.dumps(got)] += 1
                    n += 1
                key = f"{deploy}/{cell.name}/{meta['key']}"
                only = json.loads(next(iter(sets))) if len(sets) == 1 else None
                out[key] = {"episodes": n, "distinct_example_sets": len(sets), "n_examples": len(want),
                            "equals_rebuilt_rows": bool(only is not None and [tuple(x) for x in only] == want)}
    return out


def replay_check(root: Path) -> dict:
    data = root / "data" / "briefing"
    groups = OrderedDict()
    for deploy in ("all", "heldout"):
        for cell in sorted(p for p in (data / deploy).iterdir() if p.is_dir()):
            m = fc.CELL_RE.match(cell.name)
            if m:
                groups.setdefault((deploy, f"{m.group(1)}-{m.group(6)}"), []).append(cell)
    for sub in ("all", "heldout"):
        for cell in sorted(p for p in (data / "task_prompt" / sub).iterdir() if p.is_dir()):
            m = fc.CELL_RE.match(cell.name)
            groups.setdefault((sub, f"{m.group(1)}-{m.group(6)}"), []).append(cell)
    total, bad, arms = 0, 0, 0
    per_group = OrderedDict()
    for (deploy, stem), cells in groups.items():
        nonllm = [c for c in cells if c.name.endswith("-nonllm")][0]
        fixed = {}
        for f in fc.arm_files(nonllm):
            meta = fc.parse_arm(f.name)
            if meta["arm"] == "fixed":
                fixed[meta["tau"]] = fc.by_seed(fc.read_rows(f))
        g_total, g_bad, g_missing = 0, 0, 0
        for c in cells:
            for f in fc.arm_files(c):
                meta = fc.parse_arm(f.name)
                if meta["arm"] == "fixed":
                    continue
                arms += 1
                for r in fc.read_rows(f):
                    if r["seed"] not in fixed.get(r["tau"], {}):
                        g_missing += 1
                        continue
                    g_total += 1
                    g_bad += int(abs(fixed[r["tau"]][r["seed"]] - r["value"]) > 1e-9)
        per_group[f"{deploy}/{stem}"] = {"episodes_checked": g_total, "mismatches": g_bad, "seed_or_tau_not_in_fixed_arms": g_missing}
        total += g_total
        bad += g_bad
    return {"non_fixed_arm_files": arms, "episodes_checked": total, "mismatches": bad, "per_group": per_group}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(fc.RELEASE_DEFAULT))
    ap.add_argument("--out", default=None, help="default: <root>/analysis/briefing/fcr_survey_checks.json")
    a = ap.parse_args()
    root = Path(a.root)
    res = OrderedDict(lock_identity=lock_identity(root), calibration_rows=calib_rows(root), nshot_rows=nshot_check(root),
                      exact_replay=replay_check(root))
    ns = res["nshot_rows"]
    res["summary"] = {
        "locks_identical_per_setting": all(v["identical"] for v in res["lock_identity"].values()),
        "nshot_arms_checked": len(ns), "nshot_arms_matching": sum(v["equals_rebuilt_rows"] for v in ns.values()),
        "nshot_prompts_checked": sum(v["episodes"] for v in ns.values()),
        "replay_episodes": res["exact_replay"]["episodes_checked"], "replay_mismatches": res["exact_replay"]["mismatches"],
    }
    fc.write_json(Path(a.out) if a.out else root / fc.ANALYSIS / "fcr_survey_checks.json", res)
    print(json.dumps(res["summary"], indent=1))
    print(json.dumps(res["calibration_rows"], indent=1, default=str))


if __name__ == "__main__":
    main()
