"""Long-context pass of the task-stating prompt: --max-model-len 32768 serving, max_tokens 16384.

Arms (v2 single, seeds 0-99, cells s0 = M18 sigma0 react4 newword full-type deployment, s0b react4 newword held-out-only):
  gpt-oss-120b reasoning_effort=high N=0                 -> __v2-high-long
  qwen3.8-27b enable_thinking N=0 and N=1                  -> __v2-think-long
Writes tasks_0c/long_tasks_120b.txt, long_tasks_qwen.txt and arms_0c_long.json. Shards of 5 seeds, interleaved by seed so a
deadline stop leaves both cells with similar seed counts."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import make_jobs_task_prompt as mj
from run_sandbox import parse_range

HERE = Path(__file__).resolve().parent
mj.VARIANTS.update({
    "v2-high-long": ({"reasoning_effort": "high"}, 16384, ["--timeout-s", "1800"]),
    "v2-think-long": ({"chat_template_kwargs": {"enable_thinking": True}}, 16384, ["--timeout-s", "1800"]),
})
CELLS = [("s0", "react", "all"), ("s0b", "react", "heldout")]
ARMS = [("gpt-oss-120b", 0, "v2-high-long"), ("qwen3.8-27b", 0, "v2-think-long"), ("qwen3.8-27b", 1, "v2-think-long")]
SEEDS = "0-99"
SHARD = 5


def tasks() -> list[dict]:
    out = []
    for sh in mj.shards(SEEDS, 100, SHARD):
        for model, renderings, variant in ARMS:
            for group, kind, deploy in CELLS:
                cell = f"sandbox-traits-M18-K10-sharp0.8-noise0-{kind}-briefing-newword"
                common = mj.traits_common(kind, 0.0, deploy, sh) + ["--encoding", "briefing", "--wording", "newword"]
                file = f"llm__{model}__single__nshot{mj.nshot_label(renderings, kind)}__{variant}"
                argv = common + mj.llm_argv(model, "single", renderings, variant) + ["--T", str(len(parse_range(sh)))]
                out.append({"kind": "gpu", "group": group, "cell": cell, "file": file, "shard": sh, "model": model, "tier": "long",
                            "variant": variant, "mode": "single", "nshot": mj.nshot_label(renderings, kind), "T_arm": 100, "argv": argv})
    return out


def main() -> None:
    out_dir = HERE.parent / "tasks_0c"
    t = tasks()
    for name, model in (("long_tasks_120b.txt", "gpt-oss-120b"), ("long_tasks_qwen.txt", "qwen3.8-27b")):
        (out_dir / name).write_text("\n".join(mj.command(x) for x in t if x["model"] == model) + "\n", encoding="utf-8", newline="\n")
    arms = OrderedDict()
    for x in t:
        k = f"{x['group']}/{x['cell']}/{x['file']}"
        arms.setdefault(k, {"group": x["group"], "cell": x["cell"], "file": x["file"], "model": x["model"], "variant": x["variant"],
                            "mode": "single", "nshot": x["nshot"], "tier": "long", "T": 100, "shards": []})
        arms[k]["shards"].append({"seeds": x["shard"], "out": mj.out_path(x), "cmd": mj.command(x)})
    (out_dir / "arms_0c_long.json").write_text(json.dumps(arms, indent=1), encoding="utf-8")
    print(json.dumps({"tasks_120b": sum(x["model"] == "gpt-oss-120b" for x in t), "tasks_qwen": sum(x["model"] == "qwen3.8-27b" for x in t),
                      "arms": list(arms)}, indent=1))


if __name__ == "__main__":
    main()
