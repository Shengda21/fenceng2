#!/bin/bash
# Run the held-back Qwen v2-think shards under the normal serving (20b + Qwen, max-model-len 8192, still up after the chain; only
# if Qwen thinking fits the 4096 budget), then the failure and fallback reruns and merge.
S=/root/oghp/stage0c
PY=/root/envs/oghp/bin/python
cd $S
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY - <<'EOF'
import json, os
arms = json.load(open("/root/oghp/stage0c/tasks/arms_0c_think.json"))
lines = [sh["cmd"] for a in arms.values() for sh in a["shards"] if not os.path.exists(sh["out"])]
open("/root/oghp/stage0c/tasks/think_normal_tasks.txt", "w").write("\n".join(lines) + "\n")
print("think shards to run", len(lines))
EOF
D=cmds/gpu2think; rm -rf $D; mkdir -p $D
i=0; while IFS= read -r line; do i=$((i+1)); printf '%s\n' "$line" > $D/task_$(printf %04d $i).cmd; done < tasks/think_normal_tasks.txt
: > status_gpu2think.txt
T0=$(date +%s)
ls $D/task_*.cmd | xargs -P 64 -I{} bash -c 'bash {} > {}.log 2>&1; echo "$? {}" >> status_gpu2think.txt'
echo "gpu2think: $(grep -c '^0 ' status_gpu2think.txt)/$(wc -l < status_gpu2think.txt) ok in $(( $(date +%s) - T0 )) s"
if grep -qv '^0 ' status_gpu2think.txt; then bash $S/task_prompt_rerun.sh gpu2think 64; fi
$PY $S/fallback_rerun_task_prompt.py gpu2think 64 --models qwen3.8-27b --arms $S/tasks/arms_0c_think.json
$PY $S/merge_shards_task_prompt.py --arms $S/tasks/arms_0c_think.json --report $S/merge_report_think.json
echo "THINK_NORMAL_DONE $(date -u)"
