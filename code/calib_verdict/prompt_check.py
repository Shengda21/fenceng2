"""Instrument check: the runner's prompts equal the ones stored in the deployment records.

(1) Sandbox and briefing: rebuild, with the setup of this repository and the calibration seeds 100-119, the prompt of
    deployment seed 0 for one arm of each prompt family, and compare it with the prompt stored in the deployment record.
(2) MAgent: the observation text of every B prompt must equal the decision-time text the lock recorded for that
    calibration seed; N-shot examples must be A rows only.
Run with no PYTHONPATH; it switches between code/oghp and code/briefing itself.
usage: python code/calib_verdict/prompt_check.py [--root <repository root>]  -> analysis/calib_verdict/prompt_check.json
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

from calib_lib import DATA, OUT, REL

HERE = Path(__file__).resolve().parent
RES = DATA / "runs"

CHILD = r'''
import sys, json, gzip
sys.argv = ["x"] + json.loads(sys.argv[1])
rel_rec = sys.argv.pop()
import run_sandbox as rs
classic = not hasattr(rs, "build_parser")
sys.path.insert(0, r"%s")
import calib_run_sandbox as cr
parser = cr.classic_parser() if classic else rs.build_parser()
args = parser.parse_args(sys.argv[1:])
cfg, calib, enc, sel, lock = cr.classic_setup(args) if classic else rs.setup(args)
from sandbox.env import SandboxEnv
from v8lib.runner import _encode_if_needed, _visible_ctx
rec = json.loads(gzip.open(rel_rec, "rt", encoding="utf-8").readline())
seed = int(rec["seed"])
env = SandboxEnv(cfg)
if hasattr(sel, "set_episode_seed"):
    sel.set_episode_seed(seed)
ctx = _visible_ctx(_encode_if_needed(env.reset(seed), enc), sel)
msgs = sel._messages(ctx)
mine = msgs[-1]["content"]
ex = (rec.get("provenance") or [{}])[0].get("extra") or {}
theirs = ex.get("prompt")
sys_mine = msgs[0]["content"]
sys_theirs = ex.get("system_prompt")
print(json.dumps({"seed": seed, "type": rec["type"], "equal": mine == theirs, "system_equal": (sys_theirs is None) or (sys_mine == sys_theirs),
                  "mine_head": mine[:160], "theirs_head": (theirs or "")[:160]}))
'''

CASES = [
    ("classic N=0", "oghp", ["--M", "3", "--sharpness", "0.8", "--reward-noise", "0", "--encoding", "semantic", "--arm", "llm:gpt-oss-120b:single",
                             "--max-tokens", "256", "--nshot", "0", "--calib-seeds", "100-119"],
     REL / "data/episodes/sandbox/sandbox-M3-K3-sharp0.8-noise0-semantic/llm__gpt-oss-120b__single__nshot0.jsonl.gz"),
    ("classic N=5", "oghp", ["--M", "6", "--sharpness", "0.8", "--reward-noise", "0", "--encoding", "semantic", "--arm", "llm:qwen3.8-27b:single",
                             "--max-tokens", "256", "--nshot", "5", "--calib-seeds", "100-119"],
     REL / "data/episodes/sandbox/sandbox-M6-K3-sharp0.8-noise0-semantic/llm__qwen3.8-27b__single__nshot5.jsonl.gz"),
    ("briefing bare N=0", "briefing", ["--family", "traits", "--M", "18", "--K", "10", "--sharpness", "0.8", "--reward-noise", "0", "--held-out-set", "react",
                                       "--encoding", "briefing", "--wording", "sameword", "--arm", "llm:gpt-oss-120b:single", "--max-tokens", "1024",
                                       "--nshot", "0", "--calib-seeds", "100-119"],
     REL / "data/briefing/all/sandbox-traits-M18-K10-sharp0.8-noise0-react-briefing-sameword/llm__gpt-oss-120b__single__nshot0.jsonl.gz"),
    ("briefing bare hf N=0", "briefing", ["--family", "traits", "--M", "9", "--K", "5", "--sharpness", "0.8", "--reward-noise", "0", "--held-out-set", "react",
                                          "--encoding", "briefing", "--wording", "sameword", "--arm", "llm:qwen3.8-27b:hypothesis_first", "--max-tokens", "1024",
                                          "--nshot", "0", "--calib-seeds", "100-119"],
     REL / "data/briefing/all/sandbox-traits-M9-K5-sharp0.8-noise0-react-briefing-sameword/llm__qwen3.8-27b__hypothesis_first__nshot0.jsonl.gz"),
    ("briefing bare N=1", "briefing", ["--family", "traits", "--M", "18", "--K", "10", "--sharpness", "0.8", "--reward-noise", "0", "--held-out-set", "react",
                                       "--encoding", "briefing", "--wording", "sameword", "--arm", "llm:qwen3.8-27b:single", "--max-tokens", "1024",
                                       "--nshot", "1", "--calib-seeds", "100-119"],
     REL / "data/briefing/all/sandbox-traits-M18-K10-sharp0.8-noise0-react-briefing-sameword/llm__qwen3.8-27b__single__nshot1.jsonl.gz"),
    ("briefing task N=1", "briefing", ["--family", "traits", "--M", "18", "--K", "10", "--sharpness", "0.8", "--reward-noise", "0", "--held-out-set", "react",
                                       "--encoding", "briefing", "--wording", "sameword", "--arm", "llm:gpt-oss-120b:single", "--max-tokens", "1024",
                                       "--nshot", "1", "--prompt-version", "v2", "--timeout-s", "300", "--example-order", "shuffled", "--calib-seeds", "100-119"],
     REL / "data/briefing/task_prompt/all/sandbox-traits-M18-K10-sharp0.8-noise0-react-briefing-sameword/llm__gpt-oss-120b__single__nshot1__v2.jsonl.gz"),
]


def main():
    results = {"rebuild_vs_release": {}, "magent_text_vs_lock": {}}
    for name, copy, argv, rec in CASES:
        lib = REL / "code" / copy
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(lib / "v8_lib"), str(lib / "v8_sandbox")]), "LLM_API_BASE": "http://127.0.0.1:1/v1"}
        p = subprocess.run([sys.executable, "-B", "-c", CHILD % str(HERE / "server"), json.dumps(argv + [str(rec)])], capture_output=True, text=True, env=env)
        line = (p.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            results["rebuild_vs_release"][name] = json.loads(line)
        except Exception:
            results["rebuild_vs_release"][name] = {"error": p.stderr[-800:]}
        print(name, results["rebuild_vs_release"][name])
    for cell_dir in sorted((RES / "magent").iterdir()):
        table = json.load(open(DATA / "locks" / f"calib_table_{cell_dir.name}.json", encoding="utf-8"))
        text = {e["seed"]: e["text"] for e in table["episodes"]}
        A = set(table["A"])
        for f in sorted(cell_dir.glob("llm__*.json")):
            d = json.load(open(f, encoding="utf-8"))
            bad = 0
            for r in d["rows"]:
                prompt = (r["provenance"][0].get("extra") or {}).get("prompt") or ""
                if text[int(r["seed"])] not in prompt:
                    bad += 1
            ex_seeds = sorted({int(x["seed"]) for x in d["meta"].get("nshot_examples", [])})
            results["magent_text_vs_lock"][f"{cell_dir.name}/{f.stem}"] = {"episodes": len(d["rows"]), "text_mismatch": bad,
                                                                           "example_seeds": ex_seeds, "examples_in_A": all(s in A for s in ex_seeds)}
    for k, v in results["magent_text_vs_lock"].items():
        print(k, v)
    (OUT / "prompt_check.json").write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
