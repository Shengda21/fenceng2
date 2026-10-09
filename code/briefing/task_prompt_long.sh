#!/bin/bash
# usage: task_prompt_long.sh <with_qwen: 0|1>
# Long-context pass. Capped at 90 GPU-minutes from start: tasks that would start
# after the cap are skipped and running shards are stopped at the cap; complete shards are merged (partial arms marked).
S=/root/oghp/stage0c
PY=/root/envs/oghp/bin/python
WITH_QWEN=${1:-0}
cd $S
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
rm -f $S/LONG_PASS_DONE $S/LONG_DEADLINE_HIT
START=$(date +%s); DEADLINE=$((START + 90*60)); echo $DEADLINE > $S/long_deadline.txt
echo "long pass start $(date -u) deadline $(date -u -d @$DEADLINE) with_qwen=$WITH_QWEN"

( while [ $(date +%s) -lt $DEADLINE ]; do sleep 15; [ -f $S/LONG_PASS_DONE ] && exit 0; done
  echo "DEADLINE_HIT $(date -u)" | tee $S/LONG_DEADLINE_HIT; pkill -f "run_sandbox.py.*lon[g]__seeds" ) &
WATCH=$!

stop_vllm() {
pkill -f "vllm serv[e]"; sleep 10
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if ! pgrep -f "vllm serv[e]" >/dev/null && [ "$used" -lt 2000 ]; then break; fi
  sleep 5
done
echo "vllm stopped; gpu memory used ${used} MiB"
}

run_list() {  # <task list> <tag> <P>; every task checks the cap before it starts
L=$1; TAG=$2; P=$3; D=cmds/$TAG
rm -rf $D; mkdir -p $D
i=0; while IFS= read -r line; do i=$((i+1))
  printf '[ $(date +%%s) -lt %s ] || { echo SKIPPED_AT_CAP; exit 3; }\n%s\n' "$DEADLINE" "$line" > $D/task_$(printf %04d $i).cmd; done < $L
head -1 $L | sed -E 's/--T [0-9]+/--T 1/; s/--seeds [0-9]+-[0-9]+/--seeds 0-0/; s#/root/oghp/stage0c/runs_shards/#/root/oghp/stage0c/smoke_long/#g' > $D/smoke.cmd
if ! bash $D/smoke.cmd > $D/smoke.log 2>&1; then echo SMOKE_FAILED_$TAG; tail -20 $D/smoke.log; return 1; fi
out=$(grep -o -- '--out [^ ]*' $D/smoke.cmd | cut -d' ' -f2)
$PY -c "
import json; r=[json.loads(l) for l in open('$out')][0]; e=r['provenance'][0]['extra']; c=e['calls'][-1]
print('LONG_SMOKE $TAG tau', r['provenance'][0]['tau'], 'parse', e['parse_rule'], 'field', c['reasoning_field'], 'chars', len(c['reasoning'] or ''),
      'ct', c['completion_tokens'], 'finish', c['finish_reason'], 'latency', round(c['latency_s'], 1), 'calls', len(e['calls']))"
: > status_$TAG.txt
T0=$(date +%s)
ls $D/task_*.cmd | xargs -P $P -I{} bash -c 'bash {} > {}.log 2>&1; echo "$? {}" >> status_'$TAG'.txt'
echo "$TAG: $(grep -c '^0 ' status_$TAG.txt)/$(wc -l < status_$TAG.txt) ok, $(grep -c '^3 ' status_$TAG.txt) skipped at cap, in $(( $(date +%s) - T0 )) s"
}

# ---------------- gpt-oss-120b, reasoning_effort=high ----------------
stop_vllm
bash $S/serve_long.sh /root/data-disk/models/gpt-oss-120b gpt-oss-120b 8002 0.92 TIKTOKEN_ENCODINGS_BASE=/root/tikcache || { echo LONG_ABORTED_SERVE120; kill $WATCH; exit 1; }
run_list $S/tasks/long_tasks_120b.txt long120 40 || { echo LONG_ABORTED_SMOKE120; kill $WATCH; exit 1; }
if [ $(( DEADLINE - $(date +%s) )) -gt 900 ]; then
  timeout $(( DEADLINE - $(date +%s) )) $PY $S/fallback_rerun_task_prompt.py long120 40 --models gpt-oss-120b --arms $S/tasks/arms_0c_long.json
fi
$PY $S/merge_shards_task_prompt.py --arms $S/tasks/arms_0c_long.json --models gpt-oss-120b --allow-partial --report $S/merge_report_long120.json
echo "LONG120_DONE $(date -u)"

# ---------------- qwen3.8-27b, enable_thinking (only if it hit the 4096 cap) ----------------
if [ "$WITH_QWEN" = "1" ] && [ $(date +%s) -lt $DEADLINE ]; then
  stop_vllm
  bash $S/serve_long.sh /root/data-disk/models/Qwen3.8-27B-FP8 qwen3.8-27b 8001 0.92 VLLM_USE_DEEP_GEMM=0 || { echo LONG_ABORTED_SERVEQ; kill $WATCH; exit 1; }
  run_list $S/tasks/long_tasks_qwen.txt longq 64 || { echo LONG_ABORTED_SMOKEQ; kill $WATCH; exit 1; }
  if [ $(( DEADLINE - $(date +%s) )) -gt 900 ]; then
    timeout $(( DEADLINE - $(date +%s) )) $PY $S/fallback_rerun_task_prompt.py longq 64 --models qwen3.8-27b --arms $S/tasks/arms_0c_long.json
  fi
  $PY $S/merge_shards_task_prompt.py --arms $S/tasks/arms_0c_long.json --models qwen3.8-27b --allow-partial --report $S/merge_report_longq.json
  echo "LONGQ_DONE $(date -u)"
fi
touch $S/LONG_PASS_DONE
# the high-reasoning shards that ran in phase 1 (s0 newword, 4096 tokens): merged as the truncation measurement
$PY $S/merge_shards_task_prompt.py --arms $S/tasks/arms_0c_high.json --report $S/merge_report_high4096.json | head -8
echo "long pass GPU minutes: $(( ($(date +%s) - START) / 60 ))"
mkdir -p $S/logs && cp /root/jobs/serve_*_long.log /root/jobs/s0c_long.log /root/jobs/s0c_chain.log /root/jobs/s0c_think_normal.log $S/logs/ 2>/dev/null
cd $S && tar -czf /root/jobs/s0c_final.tgz runs runs_first_attempt runs_shards status_*.txt fallback_rerun_0c_*.json merge_report_*.json cmds logs validation tasks long_deadline.txt LONG_DEADLINE_HIT 2>/dev/null
sha256sum /root/jobs/s0c_final.tgz | tee /root/jobs/s0c_final.sha256
ls -la /root/jobs/s0c_final.tgz
echo "LONG_PASS_ALL_DONE $(date -u)"
