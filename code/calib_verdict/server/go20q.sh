#!/bin/bash
# Wait for the 120b task run to finish, then serve gpt-oss-20b + qwen3.8-27b together and run their arms.
while ! grep -q "^DONE gpu120" /root/jobs/go120.log; do sleep 10; done
echo "120b run finished $(date -u +%FT%TZ)"
pkill -f "vllm serv[e]"; sleep 20
bash /root/jobs/serve_both.sh || { echo SERVE_BOTH_FAILED; exit 1; }
cat > /root/jobs/hello_both.py <<'PY'
import time
from openai import OpenAI
for port, model, eb in ((8000, "gpt-oss-20b", {"reasoning_effort": "low"}), (8001, "qwen3.8-27b", {"chat_template_kwargs": {"enable_thinking": False}})):
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="none")
    t0 = time.time()
    r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say hello in exactly five words."}], max_tokens=256, temperature=0.5, extra_body=eb)
    print(model, "latency", round(time.time() - t0, 2), "reply", repr(r.choices[0].message.content), "fp", r.system_fingerprint)
PY
/root/envs/oghp/bin/python /root/jobs/hello_both.py
bash /root/jobs/run_tasks.sh /root/calib/tasks/gpu_20b_qwen.txt 112 gpu20q
