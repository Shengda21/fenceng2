#!/bin/bash
# Serve gpt-oss-120b alone (port 8002, util 0.92) with the sm120 flags, then one hello check.
pkill -f "vllm serv[e]"; sleep 5
bash /root/jobs/serve.sh /root/data-disk/models/gpt-oss-120b gpt-oss-120b 8002 0.92 TIKTOKEN_ENCODINGS_BASE=/root/tikcache || exit 1
cat > /root/jobs/hello120.py <<'PY'
import time
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="none")
t0 = time.time()
r = c.chat.completions.create(model="gpt-oss-120b", messages=[{"role": "user", "content": "Say hello in exactly five words."}],
                              max_tokens=256, temperature=0.5, extra_body={"reasoning_effort": "low"})
print("latency", round(time.time() - t0, 2), "reply", repr(r.choices[0].message.content), "fp", r.system_fingerprint)
PY
/root/envs/oghp/bin/python /root/jobs/hello120.py
echo SERVE120_DONE
