#!/bin/bash
# usage: task_prompt_gpu.sh <task list> <tag> <P>; a --T 2 smoke of the first line, then xargs -P P with status_<tag>.txt
cd /root/oghp/stage0c
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
L=$1; TAG=$2; P=$3; D=cmds/$TAG
rm -rf $D; mkdir -p $D
i=0; while IFS= read -r line; do i=$((i+1)); printf '%s\n' "$line" > $D/task_$(printf %04d $i).cmd; done < $L
head -1 $L | sed -E 's/--T [0-9]+/--T 2/; s#/root/oghp/stage0c/runs_shards/#/root/oghp/stage0c/smoke_gpu/#g' > $D/smoke.cmd
if ! bash $D/smoke.cmd > $D/smoke.log 2>&1; then echo SMOKE_FAILED_$TAG; tail -20 $D/smoke.log; exit 1; fi
echo "smoke ok: $(tail -3 $D/smoke.log | tr '\n' ' ')"
: > status_$TAG.txt
START=$(date +%s)
ls $D/task_*.cmd | xargs -P $P -I{} bash -c 'bash {} > {}.log 2>&1; echo "$? {}" >> status_'$TAG'.txt'
echo "$TAG: $(grep -c '^0 ' status_$TAG.txt)/$(wc -l < status_$TAG.txt) ok in $(( $(date +%s) - START )) s"
echo DONE_$TAG
