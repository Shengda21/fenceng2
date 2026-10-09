"""Task lists for the held-out-only deployment ("every opponent is new"), 300 seeds for every arm.

Cells: M18 x {react (react4), react8, diag (diag6)} x sigma{0, 0.5}; M9 x {react (react2), diag (diag3)} x sigma{0, 0.5}.
Non-LLM arms in `-nonllm`; text readers under newword and (control) sameword; LLM: three models x {single N0, single N1,
hypothesis_first N0} under newword, plus gpt-oss-120b single N0 under sameword. No run is shared across cells (the
deployment types depend on the held-out set)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "v8_lib"), str(HERE / "v8_sandbox")]

import make_jobs  # noqa: E402
from sandbox.game import strategy_pool  # noqa: E402

make_jobs.ROOT = "/root/oghp/stage0b"
make_jobs.RESULTS = "/root/oghp/stage0b/runs"
CELLS = [(18, 10, kind, noise) for kind in ("react", "react8", "diag") for noise in (0.0, 0.5)] + \
        [(9, 5, kind, noise) for kind in ("react", "diag") for noise in (0.0, 0.5)]
NONLLM = ["type_oracle", "scripted", "fewshot:1", "plastic", "linucb", "lints", "ctxucb", "mucb", "ppo"]
TEXT = ["scripted_text", "text_bow", "text_embed", "text_linucb"]
LLM_NEWWORD = [("single", 0), ("single", 1), ("hypothesis_first", 0)]
PHASES = {"gpu_phase1_tasks.txt": ["gpt-oss-120b"], "gpu_phase2_tasks.txt": ["gpt-oss-20b", "qwen3.8-27b"]}
T = 300
SEEDS = "0-99,120-319"


def tasks(T_: int = T, seeds: str = SEEDS, llm: bool = True) -> list[dict]:
    out = []
    for M, K, kind, noise in CELLS:
        common = ["--family", "traits", "--M", str(M), "--K", str(K), "--sharpness", "0.8", "--reward-noise", f"{noise:g}",
                  "--delta", "1.0", "--H", "30", "--k", "5", "--schedule", "single", "--features", "basic",
                  "--seeds", seeds, "--calib-seeds", make_jobs.CALIB, "--held-out-set", kind, "--deploy-types", "heldout"]

        def add(kind_, encoding, wording, arm, nshot=0, model=None):
            cell = make_jobs.cell_name(M, K, 0.8, noise, kind, "nonllm" if encoding == "semantic" else encoding, None if encoding == "semantic" else wording)
            argv = common + ["--encoding", encoding, "--wording", wording, "--arm", arm, "--T", str(T_)]
            if model:
                argv += ["--temperature", "0.5", "--max-tokens", str(make_jobs.MAX_TOKENS), "--nshot", str(nshot),
                         "--extra-body-json", json.dumps(make_jobs.EXTRA_BODY[model])]
            out.append({"kind": kind_, "M": M, "model": model, "cell": cell, "file": make_jobs.arm_file(arm, nshot), "arm": arm,
                        "encoding": encoding, "wording": wording, "nshot": nshot, "argv": argv})

        for arm in [f"fixed:{tau}" for tau in strategy_pool(K, "traits")] + NONLLM:
            add("cpu", "semantic", "newword", arm)
        for wording in ("newword", "sameword"):
            for arm in TEXT:
                add("cpu", "briefing", wording, arm)
        if not llm:
            continue
        for model in make_jobs.MODELS:
            for mode, nshot in LLM_NEWWORD:
                add("gpu", "briefing", "newword", f"llm:{model}:{mode}", nshot, model)
        add("gpu", "briefing", "sameword", "llm:gpt-oss-120b:single", 0, "gpt-oss-120b")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(HERE.parent / "tasks_0b"))
    parser.add_argument("--check-prompts", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_list = tasks()
    cpu = [make_jobs.command(t) for t in task_list if t["kind"] == "cpu"]
    (out_dir / "cpu_tasks.txt").write_text("\n".join(cpu) + "\n", encoding="utf-8", newline="\n")
    summary = {"cpu_tasks": len(cpu), "cpu_episodes": T * len(cpu)}
    for name, models in PHASES.items():
        lines = [make_jobs.command(t) for t in task_list if t["kind"] == "gpu" and t["model"] in models]
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        summary[name] = {"tasks": len(lines), "prompts": T * len(lines)}
    summary["gpu_tasks_by_model"] = dict(Counter(t["model"] for t in task_list if t["kind"] == "gpu"))
    smoke = [make_jobs.command(dict(t, argv=[a if a not in (str(T), SEEDS) else {str(T): "10", SEEDS: "0-9"}[a] for a in t["argv"]]))
             .replace("/root/oghp/stage0b/runs/", "/root/oghp/stage0b/smoke_runs/") for t in task_list if t["kind"] == "cpu"]
    (out_dir / "cpu_smoke_tasks.txt").write_text("\n".join(smoke) + "\n", encoding="utf-8", newline="\n")
    if args.check_prompts:
        pc = make_jobs.check_prompts([dict(t, argv=[a if a != str(T) else "20" for a in t["argv"]]) for t in task_list])
        summary["prompt_check"] = {"max_budget_needed": pc["max_budget_needed"], "over_limit": pc["over_limit"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
