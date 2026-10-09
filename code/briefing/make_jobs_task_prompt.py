"""Task lists for the task-stating prompt (prompt v2): shuffled pool order, v2 N-shot (1 and 3 renderings per seen type),
higher reasoning, and information-matched text readers.

Groups and run roots on the server (ROOT = /root/oghp/stage0c):
  s0       all-types deployment: M18 sigma0 react (react4), briefing newword and sameword, seeds 0-99.
  s0b      held-out-only cells: M18 sigma0 react4 and react8, briefing newword, 300 seeds (0-99,120-319).
  classic  the classic sandbox: M{3,6} K3 s{0.6,0.8} sigma{0,0.5}, semantic, zero-shot, seeds 0-99.
GPU tasks are seed shards (`--seeds lo-hi`) written to runs_shards/<group>/<cell>/<file>__seeds<lo>-<hi>.jsonl and merged by
merge_shards_task_prompt.py into runs/<group>/<cell>/<file>.jsonl; LLM selectors keep no state across episodes and the v2 shuffles
are seeded by the episode seed, so a merged file equals an unsharded run up to server-side sampling."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "v8_lib"), str(HERE / "v8_sandbox")]

from run_sandbox import parse_range  # noqa: E402
from sandbox.briefing import resolve_held_out_set  # noqa: E402
from sandbox.game import trait_types  # noqa: E402

ROOT = "/root/oghp/stage0c"
PYTHON = "/root/envs/oghp/bin/python"
ENDPOINTS = {"gpt-oss-120b": "http://127.0.0.1:8002/v1", "gpt-oss-20b": "http://127.0.0.1:8000/v1", "qwen3.8-27b": "http://127.0.0.1:8001/v1"}
MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b"]
GPTOSS = ["gpt-oss-120b", "gpt-oss-20b"]
QWEN = ["qwen3.8-27b"]
BASE_BODY = {"gpt-oss-120b": {"reasoning_effort": "low"}, "gpt-oss-20b": {"reasoning_effort": "low"},
             "qwen3.8-27b": {"chat_template_kwargs": {"enable_thinking": False}}}
# variant suffix -> (extra_body override or None for the base body, max_tokens, extra argv)
VARIANTS = {
    "v2": (None, 1024, ["--timeout-s", "300"]),
    "v2-shuf": (None, 1024, ["--timeout-s", "300", "--pool-order", "shuffled"]),
    "v2-med": ({"reasoning_effort": "medium"}, 4096, ["--timeout-s", "900"]),
    "v2-high": ({"reasoning_effort": "high"}, 4096, ["--timeout-s", "900"]),
    "v2-think": ({"chat_template_kwargs": {"enable_thinking": True}}, 4096, ["--timeout-s", "900"]),
}
# Arms for groups s0 and s0b: (mode, renderings per seen type, variant, models, tier)
# tier "primary" = the primary tests; "long" = high-reasoning arms (run early, small shards).
ARMS_A = [
    ("single", 0, "v2", MODELS, "primary"),
    ("hypothesis_first", 0, "v2", MODELS, "main"),
    ("single", 1, "v2", MODELS, "main"),
    ("single", 3, "v2", MODELS, "main"),
    ("single", 0, "v2-shuf", MODELS, "main"),
    ("single", 0, "v2-med", GPTOSS, "long"),
    ("single", 0, "v2-high", GPTOSS, "long"),
    ("single", 0, "v2-think", QWEN, "long"),
    ("single", 1, "v2-think", QWEN, "long"),
]
MATCHED = ["text_bow_matched", "text_embed_matched"]
CLASSIC_CELLS = [(M, s, n) for M in (3, 6) for s in (0.6, 0.8) for n in (0.0, 0.5)]
S0_SEEDS, S0B_SEEDS = "0-99,120-319", "0-99,120-319"
CALIB = "100-119"
SHARD = {"main": 25, "primary": 25, "long": 10}
PHASES = OrderedDict([("gpu_phase1_tasks.txt", ["gpt-oss-120b"]), ("gpu_phase2_tasks.txt", ["gpt-oss-20b", "qwen3.8-27b"])])


def traits_common(kind: str, noise: float, deploy: str, seeds: str) -> list[str]:
    argv = ["--family", "traits", "--M", "18", "--K", "10", "--sharpness", "0.8", "--reward-noise", f"{noise:g}", "--delta", "1.0",
            "--H", "30", "--k", "5", "--schedule", "single", "--features", "basic", "--seeds", seeds, "--calib-seeds", CALIB,
            "--held-out-set", kind]
    return argv + (["--deploy-types", "heldout"] if deploy == "heldout" else [])


def n_seen(kind: str) -> int:
    return len(trait_types(18)) - len(resolve_held_out_set(kind, 18))


def nshot_label(renderings: int, kind: str) -> int:
    """File label: 0 and 1 mean N=0 and N=1 (one example per seen type); 3 renderings per seen type is named by its total
    count (42 at react4, 30 at react8)."""

    return renderings if renderings <= 1 else renderings * n_seen(kind)


def shards(seeds: str, T: int, size: int) -> list[str]:
    lst = parse_range(seeds)[:T]
    out = []
    for i in range(0, len(lst), size):
        chunk = lst[i:i + size]
        assert chunk == list(range(chunk[0], chunk[-1] + 1)), "shard must be a contiguous seed range"
        out.append(f"{chunk[0]}-{chunk[-1]}")
    return out


def llm_argv(model: str, mode: str, renderings: int, variant: str) -> list[str]:
    body, max_tokens, extra = VARIANTS[variant]
    argv = ["--arm", f"llm:{model}:{mode}", "--temperature", "0.5", "--max-tokens", str(max_tokens), "--nshot", str(renderings),
            "--extra-body-json", json.dumps(body or BASE_BODY[model]), "--prompt-version", "v2"] + extra
    if renderings > 0:
        argv += ["--example-order", "shuffled"]
    return argv


def tasks() -> list[dict]:
    out = []
    groups = [("s0", "react", "newword", "all", S0_SEEDS, 100), ("s0", "react", "sameword", "all", S0_SEEDS, 100),
              ("s0b", "react", "newword", "heldout", S0B_SEEDS, 300), ("s0b", "react8", "newword", "heldout", S0B_SEEDS, 300)]
    for group, kind, wording, deploy, seeds, T in groups:
        cell = f"sandbox-traits-M18-K10-sharp0.8-noise0-{kind}-briefing-{wording}"
        common = traits_common(kind, 0.0, deploy, seeds) + ["--encoding", "briefing", "--wording", wording]
        for mode, renderings, variant, models, tier in ARMS_A:
            for model in models:
                file = f"llm__{model}__{mode}__nshot{nshot_label(renderings, kind)}__{variant}"
                for sh in shards(seeds, T, SHARD[tier]):
                    argv = [a if a != seeds else sh for a in common] + llm_argv(model, mode, renderings, variant) + ["--T", str(len(parse_range(sh)))]
                    out.append({"kind": "gpu", "group": group, "cell": cell, "file": file, "shard": sh, "model": model, "tier": tier,
                                "variant": variant, "mode": mode, "nshot": nshot_label(renderings, kind), "T_arm": T, "argv": argv})
        for name in MATCHED:
            for renderings in (1, 3):
                file = f"{name}__nshot{nshot_label(renderings, kind)}"
                argv = common + ["--arm", f"{name}:{renderings}", "--T", str(T)]
                out.append({"kind": "cpu", "group": group, "cell": cell, "file": file, "shard": None, "T_arm": T, "argv": argv})
    for M, s, noise in CLASSIC_CELLS:
        cell = f"sandbox-M{M}-K3-sharp{s:g}-noise{noise:g}-semantic"
        common = ["--M", str(M), "--K", "3", "--sharpness", f"{s:g}", "--reward-noise", f"{noise:g}", "--delta", "1.0", "--H", "30",
                  "--k", "5", "--schedule", "single", "--features", "basic", "--encoding", "semantic", "--seeds", S0_SEEDS,
                  "--calib-seeds", CALIB]
        for model in MODELS:
            file = f"llm__{model}__single__nshot0__v2"
            for sh in shards(S0_SEEDS, 100, SHARD["main"]):
                argv = [a if a != S0_SEEDS else sh for a in common] + llm_argv(model, "single", 0, "v2") + ["--T", str(len(parse_range(sh)))]
                out.append({"kind": "gpu", "group": "classic", "cell": cell, "file": file, "shard": sh, "model": model, "tier": "main",
                            "variant": "v2", "mode": "single", "nshot": 0, "T_arm": 100, "argv": argv})
    return out


def out_path(task: dict) -> str:
    if task["kind"] == "cpu":
        return f"{ROOT}/runs/{task['group']}/{task['cell']}/{task['file']}.jsonl"
    return f"{ROOT}/runs_shards/{task['group']}/{task['cell']}/{task['file']}__seeds{task['shard']}.jsonl"


def command(task: dict) -> str:
    out = out_path(task)
    env = f"PYTHONPATH={ROOT}/v8_lib:{ROOT}/v8_sandbox"
    if task["kind"] == "gpu":
        env = f"LLM_API_BASE={ENDPOINTS[task['model']]} LLM_MODEL={task['model']} LLM_API_KEY=EMPTY " + env
    argv = " ".join(shlex.quote(a) for a in task["argv"] + ["--out", out])
    return f"mkdir -p $(dirname {out}) && {env} {PYTHON} {ROOT}/v8_sandbox/run_sandbox.py {argv}"


def order_key(task: dict):
    """Primary tests first (s0 newword, then s0b react4), then the long reasoning shards (longest first helps the makespan),
    then the remaining arms by group priority."""

    gprio = {("s0", "newword"): 0, ("s0b", "react"): 1, ("s0", "sameword"): 2, ("s0b", "react8"): 3, ("classic", None): 4}
    wording = task["cell"].rsplit("-", 1)[-1] if task["group"] == "s0" else None
    kind = "react8" if "-react8-" in task["cell"] else "react"
    g = gprio[(task["group"], wording)] if task["group"] != "s0b" else gprio[("s0b", kind)]
    primary = task["tier"] == "primary" and g <= 1
    return (0 if primary else 1 if task["tier"] == "long" else 2, g)


def check_prompts(task_list: list[dict]) -> dict:
    """Render every distinct (cell, arm config) once over its whole seed list with the stub and budget the largest prompt."""

    from dry_run_prompts import prompt_tokens, render

    seen, rows = {}, []
    for t in task_list:
        if t["kind"] != "gpu":
            continue
        key = (t["group"], t["cell"], t["mode"], t["nshot"], t["variant"])
        if key not in seen:
            argv = list(t["argv"])
            argv[argv.index("--arm") + 1] = f"llm:stub:{t['mode']}"
            full = S0B_SEEDS if t["group"] == "s0b" else S0_SEEDS
            argv[argv.index("--seeds") + 1] = full
            n = t["T_arm"] if t["nshot"] else min(t["T_arm"], 100)
            lengths = [prompt_tokens(m) for m in render(argv, n)]
            seen[key] = {"group": t["group"], "cell": t["cell"], "mode": t["mode"], "nshot": t["nshot"], "variant": t["variant"],
                         "prompts_rendered": len(lengths), "max_prompt_tokens": max(lengths), "mean_prompt_tokens": sum(lengths) / len(lengths)}
        max_tokens = int(t["argv"][t["argv"].index("--max-tokens") + 1])
        need = seen[key]["max_prompt_tokens"] + 200 + max_tokens
        rows.append({"task": out_path(t), "budget_needed": need, "over": need > 8192})
    return {"configs": list(seen.values()), "max_budget_needed": max(r["budget_needed"] for r in rows), "over_limit": [r for r in rows if r["over"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE.parent / "tasks_0c"))
    ap.add_argument("--check-prompts", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_list = tasks()
    cpu = [t for t in task_list if t["kind"] == "cpu"]
    (out_dir / "cpu_tasks.txt").write_text("\n".join(command(t) for t in cpu) + "\n", encoding="utf-8", newline="\n")
    summary = OrderedDict(cpu_tasks=len(cpu), cpu_episodes=sum(t["T_arm"] for t in cpu))
    gpu = sorted([t for t in task_list if t["kind"] == "gpu"], key=order_key)
    for name, models in PHASES.items():
        sel = [t for t in gpu if t["model"] in models]
        (out_dir / name).write_text("\n".join(command(t) for t in sel) + "\n", encoding="utf-8", newline="\n")
        summary[name] = {"tasks": len(sel), "prompts": sum(len(parse_range(t["shard"])) for t in sel)}
    arms = OrderedDict()
    for t in gpu:
        k = f"{t['group']}/{t['cell']}/{t['file']}"
        arms.setdefault(k, {"group": t["group"], "cell": t["cell"], "file": t["file"], "model": t["model"], "variant": t["variant"],
                            "mode": t["mode"], "nshot": t["nshot"], "tier": t["tier"], "T": t["T_arm"], "shards": []})
        arms[k]["shards"].append({"seeds": t["shard"], "out": out_path(t), "cmd": command(t)})
    (out_dir / "arms_0c.json").write_text(json.dumps(arms, indent=1), encoding="utf-8")
    summary["gpu_arms"] = len(arms)
    summary["prompts_by_model"] = dict(Counter({m: sum(len(parse_range(t["shard"])) for t in gpu if t["model"] == m) for m in MODELS}))
    summary["prompts_by_group"] = dict(Counter({g: sum(len(parse_range(t["shard"])) for t in gpu if t["group"] == g) for g in ("s0", "s0b", "classic")}))
    summary["prompts_by_variant"] = {v: sum(len(parse_range(t["shard"])) for t in gpu if t["variant"] == v) for v in VARIANTS}
    summary["prompts_total"] = sum(len(parse_range(t["shard"])) for t in gpu)
    if a.check_prompts:
        summary["prompt_check"] = check_prompts(task_list)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "prompt_check"}, indent=2))
    if a.check_prompts:
        pc = summary["prompt_check"]
        print(json.dumps({"max_budget_needed": pc["max_budget_needed"], "over_limit": len(pc["over_limit"])}, indent=1))


if __name__ == "__main__":
    main()
