"""usage: fallback_rerun_task_prompt.py <tag> <P> --models m1[,m2] [--arms tasks/arms_0c.json]

Rerun rule for the sharded arms, applied once before the servers stop:
  - a missing or short shard is rerun (missing data, not a selection);
  - an arm (one cell x one LLM configuration, 100 or 300 episodes) whose fallback rate exceeds 5% is rerun in full, every
    shard, and the rerun result is used whatever it is.
First attempts are kept under runs_first_attempt/; the log goes to fallback_rerun_0c_<tag>.json."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from merge_shards_task_prompt import arm_status, read

S = "/root/oghp/stage0c"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("P", type=int)
    ap.add_argument("--models", required=True)
    ap.add_argument("--arms", default=f"{S}/tasks/arms_0c.json")
    a = ap.parse_args()
    arms = json.load(open(a.arms, encoding="utf-8"))
    models = set(a.models.split(","))
    todo, log = [], []
    for key, arm in arms.items():
        if arm["model"] not in models:
            continue
        st = arm_status(arm, lambda p: p)
        st.pop("_rows")
        bad = {p["seeds"] for p in st["problems"]}
        if st["fallback_rate"] > 0.05:
            shards = list(arm["shards"])
            reason = f"fallback {st['fallback_rate']:.3f}"
        else:
            shards = [s for s in arm["shards"] if s["seeds"] in bad]
            reason = "missing/short"
        if shards:
            log.append({"arm": key, "reason": reason, "fallback_before": st["fallback_rate"], "shards": [s["seeds"] for s in shards]})
            todo.extend(shards)

    def rerun(sh):
        if os.path.exists(sh["out"]):
            keep = sh["out"].replace("/runs_shards/", "/runs_first_attempt/")
            os.makedirs(os.path.dirname(keep), exist_ok=True)
            shutil.copyfile(sh["out"], keep)
        logdir = f"{S}/cmds/rerun_{a.tag}"
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, os.path.basename(sh["out"]) + ".log"), "w") as logf:
            code = subprocess.run(["bash", "-c", sh["cmd"]], stdout=logf, stderr=subprocess.STDOUT).returncode
        recs = read(sh["out"])
        return {"seeds": sh["seeds"], "exit": code, "episodes": None if recs is None else len(recs)}

    with ThreadPoolExecutor(max_workers=a.P) as pool:
        results = list(pool.map(rerun, todo))
    for entry in log:
        st = arm_status(arms[entry["arm"]], lambda p: p)
        st.pop("_rows")
        entry["fallback_after"] = st["fallback_rate"]
        entry["complete_after"] = st["complete"]
    json.dump({"tag": a.tag, "reruns": log, "shard_results": results}, open(f"{S}/fallback_rerun_0c_{a.tag}.json", "w"), indent=1)
    print(json.dumps({"tag": a.tag, "arms_rerun": len(log), "shards_rerun": len(todo),
                      "still_incomplete": [e["arm"] for e in log if not e["complete_after"]]}, indent=1))


if __name__ == "__main__":
    main()
