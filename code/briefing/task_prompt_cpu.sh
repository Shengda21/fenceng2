#!/bin/bash
# usage: task_prompt_cpu.sh <task list> <tag> [P=16]; one .cmd per line, xargs -P P, exit code per task in status_<tag>.txt
cd /root/oghp/stage0c
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
L=$1; TAG=$2; P=${3:-16}; D=cmds/$TAG
rm -rf $D; mkdir -p $D
i=0; while IFS= read -r line; do i=$((i+1)); printf '%s\n' "$line" > $D/task_$(printf %04d $i).cmd; done < $L
: > status_$TAG.txt
START=$(date +%s)
ls $D/task_*.cmd | xargs -P $P -I{} bash -c 'bash {} > {}.log 2>&1; echo "$? {}" >> status_'$TAG'.txt'
echo "$TAG: $(grep -c '^0 ' status_$TAG.txt)/$(wc -l < status_$TAG.txt) ok in $(( $(date +%s) - START )) s"
echo DONE_$TAG
