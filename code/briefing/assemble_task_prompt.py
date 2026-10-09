"""Assemble the analysis cells of the task-stating prompt: reference cells (all-types deployment, held-out-only deployment and
the classic sandbox) are copied and the merged task-stating-prompt runs are overlaid. Nothing in the source trees is modified.

usage (repository):  python code/briefing/assemble_task_prompt.py --root .
  reads data/briefing/task_prompt/{all,heldout,classic}, data/briefing/all, data/briefing/heldout and
  data/raw_episodes/sandbox; writes build/briefing_cells (a working copy, removed and rebuilt on every run).
usage (run tree): python assemble_task_prompt.py --runs-0c results_0c/runs --out results_0c/cells --release-raw <root>/data/raw_episodes/sandbox
  [--stage0-cells results/final/cells] [--s0b-cells results_0b/final/runs]
Output: <out>/s0/<cell>, <out>/s0b/<cell>, <out>/classic/<cell>; a manifest of every copied file with its source."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "sandbox-traits-M18-K10-sharp0.8-noise0"
S0_CELLS = [f"{BASE}-react-nonllm", f"{BASE}-react-briefing-newword", f"{BASE}-react-briefing-sameword"]
S0B_CELLS = [f"{BASE}-{k}-{c}" for k in ("react", "react8") for c in ("nonllm", "briefing-newword")]
CLASSIC = [f"sandbox-M{M}-K3-sharp{s}-noise{n}" for M in (3, 6) for s in ("0.6", "0.8") for n in ("0", "0.5")]


def copy_tree(src: Path, dst: Path, manifest: list, pattern: str = "*.jsonl*") -> int:
    if not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob(pattern)):
        shutil.copy2(f, dst / f.name)
        manifest.append({"dst": str(dst / f.name), "src": str(f)})
        n += 1
    return n


def main() -> None:
    tree = HERE.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="repository root (package-relative inputs and output)")
    ap.add_argument("--runs-0c", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stage0-cells", default=None)
    ap.add_argument("--s0b-cells", default=None)
    ap.add_argument("--release-raw", default=None)
    a = ap.parse_args()
    group_dir = {"s0": "s0", "s0b": "s0b", "classic": "classic"}  # sub-directory of --runs-0c holding each group's runs
    if a.root:
        root = Path(a.root)
        data = root / "data" / "briefing"
        a.runs_0c = a.runs_0c or str(data / "task_prompt")
        a.out = a.out or str(root / "build" / "briefing_cells")
        a.stage0_cells = a.stage0_cells or str(data / "all")
        a.s0b_cells = a.s0b_cells or str(data / "heldout")
        a.release_raw = a.release_raw or str(root / "data" / "raw_episodes" / "sandbox")
        group_dir = {"s0": "all", "s0b": "heldout", "classic": "classic"}
    if not (a.runs_0c and a.out and a.release_raw):
        ap.error("--root, or --runs-0c, --out and --release-raw, are required")
    runs, out = Path(a.runs_0c), Path(a.out)
    if (out / "manifest_0c.json").exists():
        shutil.rmtree(out)  # a previous assembly: rebuild from scratch
    stage0_cells = Path(a.stage0_cells) if a.stage0_cells else tree / "results" / "final" / "cells"
    s0b = Path(a.s0b_cells) if a.s0b_cells else next((p for p in (tree / "results_0b" / "final" / "runs", tree / "results_0b" / "runs", tree / "results_0b" / "cpu" / "runs") if p.is_dir()), None)
    manifest, counts = [], {}
    for cell in S0_CELLS:
        counts[f"s0/{cell}"] = copy_tree(stage0_cells / cell, out / "s0" / cell, manifest)
        counts[f"s0/{cell}"] += copy_tree(runs / group_dir["s0"] / cell, out / "s0" / cell, manifest)
    for cell in S0B_CELLS:
        counts[f"s0b/{cell}"] = copy_tree(s0b / cell, out / "s0b" / cell, manifest) if s0b else 0
        counts[f"s0b/{cell}"] += copy_tree(runs / group_dir["s0b"] / cell, out / "s0b" / cell, manifest)
    rel = Path(a.release_raw)
    for stem in CLASSIC:
        counts[f"classic/{stem}-nonllm"] = copy_tree(rel / f"{stem}-nonllm", out / "classic" / f"{stem}-nonllm", manifest, "fixed__*.jsonl.gz")
        counts[f"classic/{stem}-semantic"] = copy_tree(rel / f"{stem}-semantic", out / "classic" / f"{stem}-semantic", manifest, "llm__*__single__nshot0.jsonl.gz")
        counts[f"classic/{stem}-semantic"] += copy_tree(runs / group_dir["classic"] / f"{stem}-semantic", out / "classic" / f"{stem}-semantic", manifest)
    (out / "manifest_0c.json").write_text(json.dumps({"s0b_source": str(s0b), "counts": counts, "files": manifest}, indent=1), encoding="utf-8")
    print(json.dumps({"s0b_source": str(s0b), "counts": counts}, indent=1))


if __name__ == "__main__":
    main()
