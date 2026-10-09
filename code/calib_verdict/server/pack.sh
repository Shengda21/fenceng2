#!/bin/bash
# Pack every record, prompt/completion log, task status and serving log of the run, with per-file sha256.
cd /root/calib
mkdir -p meta
cp /root/jobs/serve_gpt-oss-120b.log /root/jobs/serve_gpt-oss-20b.log /root/jobs/serve_qwen3.8-27b.log meta/ 2>/dev/null
cp /root/jobs/go120.log /root/jobs/go20q.log /root/jobs/gocpu.log /root/jobs/serve120.log /root/jobs/smoke.log meta/ 2>/dev/null
cp /root/calib/status_*.txt meta/
{ date -u; nvidia-smi; /root/envs/vllm/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)";
  /root/envs/oghp/bin/python -c "import numpy,scipy,sklearn,magent2,pettingzoo,torch;print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__,'magent2',magent2.__version__ if hasattr(magent2,'__version__') else '?','pettingzoo',pettingzoo.__version__,'torch',torch.__version__)";
  uname -a; } > meta/env_info.txt 2>&1
find runs smoke meta -type f | sort > meta/FILES.txt
find runs smoke meta -type f ! -name SHA256SUMS | sort | xargs sha256sum > meta/SHA256SUMS
tar -czf /root/calib_results.tgz runs smoke meta locks tasks
sha256sum /root/calib_results.tgz > /root/calib_results.tgz.sha256
echo "files $(wc -l < meta/FILES.txt) $(du -sh /root/calib_results.tgz)"
cat /root/calib_results.tgz.sha256
echo PACK_DONE
