#!/bin/bash
# usage: run_tasks.sh <task_file> <parallel> <tag>
# Runs every line of the task file in its own temporary working directory, <parallel> at a time; writes
# "<exit> <line-number>" per task to /root/calib/status_<tag>.txt and a DONE marker to the log.
TASKS=$1; P=$2; TAG=$3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p /root/calib/tmp
: > /root/calib/status_$TAG.txt
echo "START $TAG $(date -u +%FT%TZ) $(grep -c . $TASKS) tasks, P=$P"
nl -ba -w1 -s$'\t' "$TASKS" | grep -P '\t.' | xargs -d '\n' -P "$P" -I{} bash -c '
  n="${1%%$'"'"'\t'"'"'*}"; line="${1#*$'"'"'\t'"'"'}"
  d=$(mktemp -d -p /root/calib/tmp); cd "$d" && bash -c "$line"; rc=$?
  echo "$rc $n" >> /root/calib/status_'"$TAG"'.txt' _ {}
echo "DONE $TAG $(date -u +%FT%TZ) ok=$(grep -c '^0 ' /root/calib/status_$TAG.txt) fail=$(grep -vc '^0 ' /root/calib/status_$TAG.txt)"
