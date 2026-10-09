"""Write the task lists of the calibration split (one shell command per line).

Outputs (in data/calib_verdict/tasks):
  cpu_tasks.txt        fixed-strategy sweeps, references and every non-LLM comparator on B (fit on A)
  gpu_120b.txt         LLM arms served by gpt-oss-120b (port 8002)
  gpu_20b_qwen.txt     LLM arms served by gpt-oss-20b (port 8000) and qwen3.8-27b (port 8001)
  smoke_tasks.txt      one LLM call per prompt family on one A episode (pipeline check, not read)
  arms.json            the arm list per cell with the configuration of each arm
The analysis outputs are read to pick the per-cell best N-shot arm of the master table, exactly as
code/make_tables.py does (first arm of maximal capture among LLM arms with N >= 1 in the cell's arm order).
The commands use the server layout: /root/calib/code/release/{oghp,briefing} are code/{oghp,briefing} of this repository,
/root/calib/code/calib_run_sandbox.py is code/calib_verdict/server/calib_run_sandbox.py, /root/calib/locks is
data/calib_verdict/locks.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

REL = Path(__file__).resolve().parents[3]  # the repository root
OUT = REL / "data" / "calib_verdict" / "tasks"
R = "/root/calib"
PY = "/root/envs/oghp/bin/python"
PP_SANDBOX = f"PYTHONPATH={R}/code/release/oghp/v8_lib:{R}/code/release/oghp/v8_sandbox"
PP_BRIEF = f"PYTHONPATH={R}/code/release/briefing/v8_lib:{R}/code/release/briefing/v8_sandbox"
EMBED = "OGHP_EMBED_MODEL=/root/data-disk/models/bge-small-en-v1.5"
SERVE = {"gpt-oss-120b": 8002, "gpt-oss-20b": 8000, "qwen3.8-27b": 8001}
EXTRA = {"gpt-oss-120b": '{"reasoning_effort": "low"}', "gpt-oss-20b": '{"reasoning_effort": "low"}',
         "qwen3.8-27b": '{"chat_template_kwargs": {"enable_thinking": false}}'}

SB_A = list(range(100, 120))[0::2]
SB_B = list(range(100, 120))[1::2]
MG_A = list(range(1000, 1040))[0::2]
MG_B = list(range(1000, 1040))[1::2]


def csv(xs):
    return ",".join(str(x) for x in xs)


def llm_env(model):
    return f"LLM_API_BASE=http://127.0.0.1:{SERVE[model]}/v1 LLM_MODEL={model} LLM_API_KEY=EMPTY"


def cmd(out, body):
    return f"mkdir -p $(dirname {out}) && {body} > {out}.log 2>&1"


# ---------------------------------------------------------------- cells
SANDBOX = []
for M in (3, 6):
    for s, n in ((0.8, 0), (0.8, 0.5), (0.6, 0), (0.6, 0.5)):
        SANDBOX.append({"cell": f"sandbox-M{M}-K3-sharp{s:g}-noise{n:g}", "M": M, "s": s, "noise": n})
BRIEF = [
    {"cell": "sandbox-traits-M18-K10-sharp0.8-noise0-react-sameword", "M": 18, "K": 10},
    {"cell": "sandbox-traits-M9-K5-sharp0.8-noise0-react-sameword", "M": 9, "K": 5},
]
MAGENT = {
    "battle-base4": {"runner": "run_rs_magent.py", "master": "battle-base4-pool8-semantic-single",
                     "args": "--type-set base4 --M 4 --encoding semantic --schedule single --delta-scale 1.0 --type-hints --blue scripted --k 20 --map-size 20 --max-cycles 100 --permutation-seed 0 --default HOLD_POSITION --pool pool8"},
    "battle-traits8": {"runner": "run_rs_magent.py", "master": "battle-traits8-pool8-semantic-single",
                       "args": "--type-set traits8 --M 8 --encoding semantic --schedule single --delta-scale 1.0 --type-hints --blue scripted --k 20 --map-size 20 --max-cycles 100 --permutation-seed 0 --default HOLD_POSITION --pool pool8"},
    "combined-base4": {"runner": "run_rs_combined.py", "master": "combined-base4-semantic",
                       "args": "--type-set base4 --M 4 --encoding semantic --schedule single --delta-scale 1.0 --type-hints --blue scripted --k 20 --map-size 16 --max-cycles 120 --n-melee 6 --n-ranged 6 --permutation-seed 0 --default HOLD_LINE --pool combined_pool6"},
    "combined-traits8": {"runner": "run_rs_combined.py", "master": "combined-traits8-semantic",
                         "args": "--type-set traits8 --M 8 --encoding semantic --schedule single --delta-scale 1.0 --type-hints --blue scripted --k 20 --map-size 16 --max-cycles 120 --n-melee 6 --n-ranged 6 --permutation-seed 0 --default HOLD_LINE --pool combined_pool6"},
    "combined-base4-delta06": {"runner": "run_rs_combined.py", "master": "combined-base4-delta06-semantic",
                               "args": "--type-set base4 --M 4 --encoding semantic --schedule single --delta-scale 0.6 --type-hints --blue scripted --k 20 --map-size 16 --max-cycles 120 --n-melee 6 --n-ranged 6 --permutation-seed 0 --default HOLD_LINE --pool combined_pool6"},
}
POOLS = {"battle": ["ATTACK_FORWARD", "HOLD_POSITION", "SPREAD_OUT", "RETREAT", "FOCUS_FIRE", "FLANK", "KITE", "RETREAT_REAL"],
         "combined": ["ALL_ATTACK", "HOLD_LINE", "RANGED_FIRST", "MELEE_FIRST", "SCREEN_KITE", "FLANK_RANGED"]}
BANDITS = ("linucb", "lints", "ctxucb", "mucb")


def master():
    return json.load(open(REL / "analysis" / "master.json", encoding="utf-8"))["master"]


def best_nshot(M, cell):
    best = None
    for k, e in M.get(cell, {}).get("arms", {}).items():
        m = e["meta"]
        if m["arm"] == "llm" and m.get("nshot", 0) > 0:
            if best is None or e["capture"] > best[1]["capture"]:
                best = (k, e)
    return best


def arm_key(model, mode, nshot, variant=None):
    return f"llm__{model}__{mode}__nshot{nshot}" + (f"__{variant}" if variant else "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Mst = master()
    arms = {"sandbox": {}, "briefing": {}, "magent": {}}
    cpu, gpu = [], {"120b": [], "20b_qwen": []}
    smoke = []

    def gpu_add(model, line):
        gpu["120b" if model == "gpt-oss-120b" else "20b_qwen"].append(line)

    # ------------------------------------------------ first-moves sandbox
    for c in SANDBOX:
        cell = c["cell"]
        base = f"--M {c['M']} --sharpness {c['s']:g} --reward-noise {c['noise']:g} --K 3 --delta 1.0 --H 30 --k 5 --schedule single --features basic"
        ab = f"--a-seeds {csv(SB_A)}"
        runner = f"{PY} {R}/code/calib_run_sandbox.py"
        llm_cell = cell + "-semantic"
        chosen = [("gpt-oss-20b", "single", 0), ("gpt-oss-120b", "single", 0), ("qwen3.8-27b", "single", 0)]
        bn = best_nshot(Mst, llm_cell)
        bn_meta = bn[1]["meta"]
        chosen.append((bn_meta["model"], bn_meta["mode"], int(bn_meta["nshot"])))
        arms["sandbox"][cell] = {"llm_cell": llm_cell, "ref_cell": cell + "-nonllm", "best_nshot_release": bn[0],
                                 "llm_arms": [arm_key(*a) for a in chosen]}
        for model, mode, n in chosen:
            key = arm_key(model, mode, n)
            for b in SB_B:
                out = f"{R}/runs/sandbox/{cell}/{key}__b{b}.jsonl"
                body = (f"{llm_env(model)} {PP_SANDBOX} {runner} {base} --encoding semantic --arm llm:{model}:{mode} "
                        f"--max-tokens 256 --nshot {n} --extra-body-json {shlex.quote(EXTRA[model])} {ab} --b-seeds {b} --out {out}")
                gpu_add(model, cmd(out, body))
            if model == "gpt-oss-120b" and n == 0 and not any("sandbox" in s for s in smoke):
                out = f"{R}/smoke/sandbox/{cell}/{key}.jsonl"
                smoke.append(cmd(out, f"{llm_env(model)} {PP_SANDBOX} {runner} {base} --encoding semantic --arm llm:{model}:{mode} "
                                      f"--max-tokens 256 --nshot {n} --extra-body-json {shlex.quote(EXTRA[model])} --a-seeds {csv(SB_A[1:])} --b-seeds {SB_A[0]} --out {out}"))
        nonllm = [f"fixed:{t}" for t in ("ROCK", "PAPER", "SCISSORS")] + ["type_oracle", "random", "scripted", "fewshot:1", "fewshot:5",
                                                                       "fewshot:20", "plastic", "ppo", *BANDITS]
        arms["sandbox"][cell]["comparators"] = ["scripted", "fewshot__1", "fewshot__5", "fewshot__20", "plastic", "ppo", *BANDITS]
        for a in nonllm:
            key = a.replace(":", "__")
            out = f"{R}/runs/sandbox/{cell}/{key}.jsonl"
            extra = " --warmup" if a in BANDITS else (" --ppo-on-a" if a == "ppo" else "")
            cpu.append(cmd(out, f"{PP_SANDBOX} {runner} {base} --encoding semantic --arm {a} {ab} --b-seeds {csv(SB_B)}{extra} --out {out}"))

    # ------------------------------------------------ briefing game, most agents known, known wording
    for c in BRIEF:
        cell = c["cell"]
        base = (f"--family traits --M {c['M']} --K {c['K']} --sharpness 0.8 --reward-noise 0 --delta 1.0 --H 30 --k 5 "
                f"--schedule single --features basic --held-out-set react --wording sameword")
        ab = f"--a-seeds {csv(SB_A)}"
        runner = f"{EMBED} {PP_BRIEF} {PY} {R}/code/calib_run_sandbox.py"
        chosen = [("gpt-oss-120b", "single", 0, None), ("gpt-oss-20b", "single", 0, None), ("qwen3.8-27b", "hypothesis_first", 0, None),
                  ("qwen3.8-27b", "single", 0, None), ("qwen3.8-27b", "single", 1, None)]
        if c["M"] == 18:
            chosen += [("gpt-oss-120b", "single", 0, "v2"), ("gpt-oss-120b", "single", 1, "v2"), ("qwen3.8-27b", "single", 1, "v2")]
        arms["briefing"][cell] = {"llm_arms": [arm_key(m, md, n, v) for m, md, n, v in chosen],
                                  "comparators_prefix": ["scripted", "fewshot__1", "plastic", "ppo", *BANDITS],
                                  "comparators_text": ["scripted_text", "text_bow", "text_embed", "text_linucb"],
                                  "secondary_text": ["text_bow_A", "text_embed_A"]}
        for model, mode, n, v in chosen:
            key = arm_key(model, mode, n, v)
            v2 = ""
            if v == "v2":
                v2 = " --prompt-version v2 --timeout-s 300" + (" --example-order shuffled" if n > 0 else "")
            for b in SB_B:
                out = f"{R}/runs/briefing/{cell}/{key}__b{b}.jsonl"
                body = (f"{llm_env(model)} {runner} {base} --encoding briefing --arm llm:{model}:{mode} --temperature 0.5 --max-tokens 1024 "
                        f"--nshot {n} --extra-body-json {shlex.quote(EXTRA[model])}{v2} {ab} --b-seeds {b} --out {out}")
                gpu_add(model, cmd(out, body))
            if c["M"] == 18 and model == "gpt-oss-120b" and n == 1:
                out = f"{R}/smoke/briefing/{cell}/{key}.jsonl"
                smoke.append(cmd(out, f"{llm_env(model)} {runner} {base} --encoding briefing --arm llm:{model}:{mode} --temperature 0.5 --max-tokens 1024 "
                                      f"--nshot {n} --extra-body-json {shlex.quote(EXTRA[model])}{v2} --a-seeds {csv(SB_A[1:])} --b-seeds {SB_A[0]} --out {out}"))
            if c["M"] == 18 and model == "gpt-oss-120b" and n == 0 and v is None:
                out = f"{R}/smoke/briefing/{cell}/{key}.jsonl"
                smoke.append(cmd(out, f"{llm_env(model)} {runner} {base} --encoding briefing --arm llm:{model}:{mode} --temperature 0.5 --max-tokens 1024 "
                                      f"--nshot {n} --extra-body-json {shlex.quote(EXTRA[model])} --a-seeds {csv(SB_A[1:])} --b-seeds {SB_A[0]} --out {out}"))
        pool = 10 if c["M"] == 18 else 5
        from importlib import util  # noqa: F401  (pool names come from the game module below)
        import sys
        sys.path.insert(0, str(REL / "code" / "briefing" / "v8_sandbox")); sys.path.insert(0, str(REL / "code" / "briefing" / "v8_lib"))
        from sandbox.game import strategy_pool
        taus = strategy_pool(pool, "traits")
        for a in [f"fixed:{t}" for t in taus] + ["type_oracle", "random", "scripted", "fewshot:1", "plastic", "ppo", *BANDITS]:
            key = a.replace(":", "__")
            out = f"{R}/runs/briefing/{cell}/nonllm/{key}.jsonl"
            extra = " --warmup" if a in BANDITS else (" --ppo-on-a" if a == "ppo" else "")
            cpu.append(cmd(out, f"{runner} {base} --encoding semantic --arm {a} {ab} --b-seeds {csv(SB_B)}{extra} --out {out}"))
        for a in ["scripted_text", "text_bow", "text_embed", "text_linucb", "text_bow_A", "text_embed_A"]:
            out = f"{R}/runs/briefing/{cell}/briefing/{a}.jsonl"
            extra = " --warmup" if a == "text_linucb" else ""
            cpu.append(cmd(out, f"{runner} {base} --encoding briefing --arm {a} {ab} --b-seeds {csv(SB_B)}{extra} --out {out}"))

    # ------------------------------------------------ MAgent battle and combined arms
    for cell, spec in MAGENT.items():
        mcell = spec["master"]
        dom = "battle" if cell.startswith("battle") else "combined"
        runner = f"{PY} {R}/code/release/oghp/experiments0106b/scripts/rs/{spec['runner']}"
        lockA = f"{R}/locks/lockA_{cell}.json"
        lockAB = f"{R}/locks/lockAB_{cell}.json"
        rel_arms = Mst[mcell]["arms"]
        chosen = []
        for model in ("gpt-oss-20b", "gpt-oss-120b", "qwen3.8-27b"):
            k = next((kk for kk, e in rel_arms.items() if e["meta"]["arm"] == "llm" and e["meta"]["model"] == model
                      and e["meta"]["mode"] == "single" and e["meta"]["nshot"] == 0), None)
            if k:
                chosen.append((model, "single", 0))
        bn = best_nshot(Mst, mcell)
        if bn:
            chosen.append((bn[1]["meta"]["model"], bn[1]["meta"]["mode"], int(bn[1]["meta"]["nshot"])))
        if cell == "combined-base4":
            chosen.append(("qwen3.8-27b", "single", 1))      # one labelled episode per type (ties the master pick)
        if cell == "combined-traits8":
            chosen.append(("qwen3.8-27b", "hypothesis_first", 0))  # zero-shot C2 pass in a real game
        chosen = list(dict.fromkeys(chosen))
        comps = [k for k, e in rel_arms.items() if e["meta"]["arm"] in ("mucb", "linucb", "lints", "ctxucb", "plastic", "ppo", "scripted", "fewshot")]
        arms["magent"][cell] = {"master_cell": mcell, "best_nshot_release": bn[0] if bn else None,
                                "llm_arms": [arm_key(*a) for a in chosen], "comparators": comps}
        for model, mode, n in chosen:
            key = arm_key(model, mode, n)
            out = f"{R}/runs/magent/{cell}/{key}.json"
            body = (f"{llm_env(model)} {runner} {spec['args']} --arm llm:{model}:{mode} --T {len(MG_B)} --lock {lockA} "
                    f"--nshot {n} --temperature 0.5 --max-tokens 256 --extra-body-json {shlex.quote(EXTRA[model])} --out {out}")
            gpu_add(model, cmd(out, body))
        if cell == "combined-base4":
            out = f"{R}/smoke/magent/{cell}/llm__gpt-oss-120b__single__nshot0.json"
            smoke.append(cmd(out, f"{llm_env('gpt-oss-120b')} {runner} {spec['args']} --arm llm:gpt-oss-120b:single --T 1 --lock {R}/locks/smoke_{cell}.json "
                                  f"--nshot 0 --temperature 0.5 --max-tokens 256 --extra-body-json {shlex.quote(EXTRA['gpt-oss-120b'])} --out {out}"))
        for tau in POOLS[dom]:
            out = f"{R}/runs/magent/{cell}/fixed__{tau}.json"
            cpu.append(cmd(out, f"{runner} {spec['args']} --arm fixed:{tau} --T {len(MG_B)} --lock {lockA} --out {out}"))
        for a in comps + ["type_oracle", "random"]:
            if a == "ppo":
                continue  # not in any LLM cell's comparator set (combined PPO lives in the instrument cell)
            arm_arg = a.replace("__", ":")
            out = f"{R}/runs/magent/{cell}/{a}.json"
            if a in BANDITS:
                cpu.append(cmd(out, f"{runner} {spec['args']} --arm {arm_arg} --T {len(MG_A) + len(MG_B)} --lock {lockAB} --out {out}"))
            else:
                cpu.append(cmd(out, f"{runner} {spec['args']} --arm {arm_arg} --T {len(MG_B)} --lock {lockA} --out {out}"))

    (OUT / "cpu_tasks.txt").write_text("\n".join(cpu) + "\n", encoding="utf-8", newline="\n")
    (OUT / "gpu_120b.txt").write_text("\n".join(gpu["120b"]) + "\n", encoding="utf-8", newline="\n")
    (OUT / "gpu_20b_qwen.txt").write_text("\n".join(gpu["20b_qwen"]) + "\n", encoding="utf-8", newline="\n")
    (OUT / "smoke_tasks.txt").write_text("\n".join(smoke) + "\n", encoding="utf-8", newline="\n")
    (OUT / "arms.json").write_text(json.dumps(arms, indent=1), encoding="utf-8", newline="\n")
    print("cpu", len(cpu), "gpu120", len(gpu["120b"]), "gpu20q", len(gpu["20b_qwen"]), "smoke", len(smoke))
    print(json.dumps({d: {c: v["llm_arms"] for c, v in arms[d].items()} for d in arms}, indent=1))


if __name__ == "__main__":
    main()
