"""Write the task lists for the all-types deployment (CPU non-LLM arms, GPU LLM arms per phase), the map that copies shared
runs into every cell that uses them, and check prompt lengths against the server limit.

Runs whose inputs do not depend on the held-out kind sit in a `base` scope and are copied into both kinds' cells."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "v8_lib"), str(HERE / "v8_sandbox")]

from sandbox.game import strategy_pool  # noqa: E402

ROOT = "/root/oghp/stage0"
PYTHON = "/root/envs/oghp/bin/python"
RESULTS = "/root/oghp/stage0/runs"
ENDPOINTS = {
    "gpt-oss-120b": "http://127.0.0.1:8002/v1",
    "gpt-oss-20b": "http://127.0.0.1:8000/v1",
    "qwen3.8-27b": "http://127.0.0.1:8001/v1",
}
EXTRA_BODY = {
    "gpt-oss-120b": {"reasoning_effort": "low"},
    "gpt-oss-20b": {"reasoning_effort": "low"},
    "qwen3.8-27b": {"chat_template_kwargs": {"enable_thinking": False}},
}
PHASES = {"gpu_phase1_tasks.txt": ["gpt-oss-120b"], "gpu_phase2_tasks.txt": ["gpt-oss-20b", "qwen3.8-27b"]}
MAX_MODEL_LEN = 8192
MAX_TOKENS = 1024
TEMPLATE_OVERHEAD = 200

BASE_CELLS = [(9, 5, 0.8, 0.0), (9, 5, 0.8, 0.5), (18, 10, 0.8, 0.0), (18, 10, 0.8, 0.5)]
KINDS = ["react", "diag"]
SHARED_NONLLM = ["type_oracle", "random", "scripted", "linucb", "lints", "mucb"]
KIND_NONLLM = ["fewshot:1", "plastic", "ctxucb", "ppo"]
SHARED_SAMEWORD = ["scripted_text", "text_linucb"]
KIND_SAMEWORD = ["text_bow", "text_embed"]
KIND_NEWWORD = ["scripted_text", "text_linucb", "text_bow", "text_embed"]
LONG_ARMS = {"fewshot:1", "plastic", "linucb", "lints", "ctxucb", "mucb", "ppo", "text_bow", "text_embed", "text_linucb"}
MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b"]
MODES = ["single", "hypothesis_first"]
SHARED_LLM = [("briefing", "sameword", 0), ("briefing_prefix", "sameword", 0), ("semantic", None, 0)]
KIND_LLM = [("briefing", "newword", 0), ("briefing_prefix", "newword", 0), ("briefing", "sameword", 1), ("briefing", "newword", 1)]
SEEDS = "0-99,120-319"
CALIB = "100-119"


def cell_name(M, K, s, noise, scope, encoding, wording=None) -> str:
    tail = f"-{wording}" if wording else ""
    return f"sandbox-traits-M{M}-K{K}-sharp{s:g}-noise{noise:g}-{scope}-{encoding}{tail}"


def targets(cell: str) -> list[str]:
    return [cell.replace("-base-", f"-{kind}-") for kind in KINDS] if "-base-" in cell else [cell]


def arm_file(arm: str, nshot: int = 0) -> str:
    if arm.startswith("fixed:"):
        return "fixed__" + arm.split(":", 1)[1]
    if arm.startswith("fewshot:"):
        return "fewshot__" + arm.split(":", 1)[1]
    if arm.startswith("llm:"):
        _, model, mode = arm.split(":")
        return f"llm__{model}__{mode}__nshot{nshot}"
    return arm


def tasks(T_short: int = 100, T_long: int = 300, seeds: str = SEEDS, llm: bool = True) -> list[dict]:
    out = []
    for M, K, s, noise in BASE_CELLS:
        def common(scope):
            return ["--family", "traits", "--M", str(M), "--K", str(K), "--sharpness", f"{s:g}", "--reward-noise", f"{noise:g}",
                    "--delta", "1.0", "--H", "30", "--k", "5", "--schedule", "single", "--features", "basic",
                    "--seeds", seeds, "--calib-seeds", CALIB, "--held-out-set", "react" if scope == "base" else scope]

        def cpu(scope, encoding, wording, arm):
            T = T_long if arm in LONG_ARMS else T_short
            argv = common(scope) + ["--encoding", encoding, "--wording", wording, "--arm", arm, "--T", str(T)]
            cell = cell_name(M, K, s, noise, scope, "nonllm", None) if encoding == "semantic" else cell_name(M, K, s, noise, scope, encoding, wording)
            out.append({"kind": "cpu", "M": M, "scope": scope, "cell": cell, "file": arm_file(arm), "arm": arm, "argv": argv})

        def gpu(scope, model, mode, encoding, wording, nshot):
            arm = f"llm:{model}:{mode}"
            argv = common(scope) + ["--encoding", encoding, "--wording", wording or "sameword", "--arm", arm, "--T", str(T_short),
                                    "--temperature", "0.5", "--max-tokens", str(MAX_TOKENS), "--nshot", str(nshot),
                                    "--extra-body-json", json.dumps(EXTRA_BODY[model])]
            out.append({"kind": "gpu", "M": M, "scope": scope, "model": model, "cell": cell_name(M, K, s, noise, scope, encoding, wording),
                        "file": arm_file(arm, nshot), "arm": arm, "encoding": encoding, "wording": wording, "nshot": nshot, "argv": argv})

        for arm in [f"fixed:{tau}" for tau in strategy_pool(K, "traits")] + SHARED_NONLLM:
            cpu("base", "semantic", "sameword", arm)
        for arm in SHARED_SAMEWORD:
            cpu("base", "briefing", "sameword", arm)
        for kind in KINDS:
            for arm in KIND_NONLLM:
                cpu(kind, "semantic", "sameword", arm)
            for arm in KIND_SAMEWORD:
                cpu(kind, "briefing", "sameword", arm)
            for arm in KIND_NEWWORD:
                cpu(kind, "briefing", "newword", arm)
        if not llm:
            continue
        for model in MODELS:
            for mode in MODES:
                for encoding, wording, nshot in SHARED_LLM:
                    gpu("base", model, mode, encoding, wording, nshot)
                for kind in KINDS:
                    for encoding, wording, nshot in KIND_LLM:
                        gpu(kind, model, mode, encoding, wording, nshot)
    return out


def shared_map(task_list: list[dict]) -> dict:
    return {f"{t['cell']}/{t['file']}.jsonl": [f"{c}/{t['file']}.jsonl" for c in targets(t["cell"])] for t in task_list}


def command(task: dict) -> str:
    out = f"{RESULTS}/{task['cell']}/{task['file']}.jsonl"
    env = f"PYTHONPATH={ROOT}/v8_lib:{ROOT}/v8_sandbox"
    if task["kind"] == "gpu":
        env = f"LLM_API_BASE={ENDPOINTS[task['model']]} LLM_MODEL={task['model']} LLM_API_KEY=EMPTY " + env
    argv = " ".join(shlex.quote(a) for a in task["argv"] + ["--out", out])
    return f"mkdir -p $(dirname {out}) && {env} {PYTHON} {ROOT}/v8_sandbox/run_sandbox.py {argv}"


def check_prompts(task_list: list[dict]) -> dict:
    from dry_run_prompts import prompt_tokens, render

    seen, rows = {}, []
    for task in task_list:
        if task["kind"] != "gpu":
            continue
        key = (task["cell"], task["nshot"])
        if key not in seen:
            argv = list(task["argv"])
            argv[argv.index("--arm") + 1] = "llm:stub:hypothesis_first"
            T = int(argv[argv.index("--T") + 1])
            lengths = [prompt_tokens(m) for m in render(argv, T)]
            seen[key] = {"cell": task["cell"], "nshot": task["nshot"], "prompts": len(lengths), "max_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths)}
        need = seen[key]["max_tokens"] + TEMPLATE_OVERHEAD + MAX_TOKENS
        rows.append({"file": f"{task['cell']}/{task['file']}", "max_prompt_tokens": seen[key]["max_tokens"], "budget_needed": need, "over": need > MAX_MODEL_LEN})
    return {"configs": list(seen.values()), "over_limit": [r for r in rows if r["over"]], "max_budget_needed": max(r["budget_needed"] for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(HERE.parent / "tasks"))
    parser.add_argument("--check-prompts", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_list = tasks()
    cpu = [command(t) for t in task_list if t["kind"] == "cpu"]
    (out_dir / "cpu_tasks.txt").write_text("\n".join(cpu) + "\n", encoding="utf-8", newline="\n")
    summary = {"cpu_tasks": len(cpu), "cpu_episodes": sum(int(t["argv"][t["argv"].index("--T") + 1]) for t in task_list if t["kind"] == "cpu")}
    for name, models in PHASES.items():
        lines = [command(t) for t in task_list if t["kind"] == "gpu" and t["model"] in models]
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        prompts = sum(int(t["argv"][t["argv"].index("--T") + 1]) for t in task_list if t["kind"] == "gpu" and t["model"] in models)
        summary[name] = {"tasks": len(lines), "prompts": prompts}
    summary["gpu_tasks_by_model"] = dict(Counter(t["model"] for t in task_list if t["kind"] == "gpu"))
    smap = shared_map(task_list)
    (out_dir / "shared_map.json").write_text(json.dumps(smap, indent=1), encoding="utf-8")
    summary["assembled_files"] = sum(len(v) for v in smap.values())
    if args.check_prompts:
        summary["prompt_check"] = check_prompts(task_list)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "prompt_check"}, indent=2))
    if args.check_prompts:
        pc = summary["prompt_check"]
        print(json.dumps({"max_budget_needed": pc["max_budget_needed"], "over_limit": pc["over_limit"]}, indent=1))


if __name__ == "__main__":
    main()
