#!/bin/bash
# gpt-oss-120b arms (server already up on port 8002), then the CPU tasks run in parallel from their own job.
bash /root/jobs/run_tasks.sh /root/calib/tasks/gpu_120b.txt 96 gpu120
