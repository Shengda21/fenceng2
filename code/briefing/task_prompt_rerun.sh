#!/bin/bash
# usage: task_prompt_rerun.sh <tag> <P>; reruns the .cmd files of a tag that exited non-zero, appends to status_<tag>.txt
cd /root/oghp/stage0c
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TAG=$1; P=${2:-20}
grep -v '^0 ' status_$TAG.txt | awk '{print $2}' > failed_$TAG.txt
grep '^0 ' status_$TAG.txt > status_$TAG.tmp; mv status_$TAG.tmp status_$TAG.txt
echo "rerunning $(wc -l < failed_$TAG.txt)"
cat failed_$TAG.txt | xargs -P $P -I{} bash -c 'bash {} > {}.log 2>&1; echo "$? {}" >> status_'$TAG'.txt'
echo "$TAG: $(grep -c '^0 ' status_$TAG.txt)/$(wc -l < status_$TAG.txt) ok"
echo DONE_RERUN_$TAG
