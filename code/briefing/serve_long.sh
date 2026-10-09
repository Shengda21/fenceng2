#!/bin/bash
# usage: serve_long.sh <model_dir> <served_name> <port> <gpu_util> [extra env assignments...]
# Same as /root/jobs/serve.sh but with --max-model-len 32768 (long-context pass).
MODEL=$1; NAME=$2; PORT=$3; UTIL=$4; shift 4
export VLLM_USE_FLASHINFER_SAMPLER=0
for kv in "$@"; do export "$kv"; done
V=/root/envs/vllm/bin
nohup $V/vllm serve "$MODEL" --served-model-name "$NAME" --port "$PORT" --host 127.0.0.1 --max-model-len 32768 \
  --gpu-memory-utilization "$UTIL" --max-num-seqs 128 --attention-backend TRITON_ATTN > /root/jobs/serve_${NAME}_long.log 2>&1 < /dev/null &
for i in $(seq 1 180); do
  a=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models)
  if [ "$a" = "200" ]; then echo "SERVE_OK $NAME (max-model-len 32768) after $((i*10)) s"; break; fi
  if grep -qE "Engine core initialization failed|Traceback|CUDA error" /root/jobs/serve_${NAME}_long.log; then echo "SERVE_FAILED $NAME"; tail -30 /root/jobs/serve_${NAME}_long.log; exit 1; fi
  sleep 10
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
