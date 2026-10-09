# Extracted from audit/llm_client.py (class GatewaySampler: __init__, _one_call, _parse, sample_prior) and audit/sync_agent.py (_select_macro), the language-model arm of the HLA transfer.
"""HLA transfer: the sampled prior over macro actions.

At each macro decision the host system builds its own prompts: a rules text and a
situation text. I do not have those two texts. They come from the host system's
prompt builder (Em_prompt_ep), which is not part of this repository, so the examples
file shows them as placeholders. Everything this file adds around them is verbatim:
the instruction appended to the situation, the system message, the sampling settings,
the parser and the smoothing.

The model cannot return log-probabilities here, so the prior over the available
macro actions is estimated from M independent completions.
"""
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# The runs set the model name to gpt-oss-120b and called an OpenAI-compatible chat
# completions endpoint. The address and the keys come from the environment here.
API_BASE = os.environ.get("LLM_API_URL", "")
MODEL = "gpt-oss-120b"


def _load_keys():
    return [k.strip() for k in os.environ.get("LLM_GATEWAY_KEYS", "").split(",") if k.strip()]


class GatewaySampler:
    def __init__(self, m_samples=10, temperature=1.0, max_workers=8, max_retries=5):
        self.keys = _load_keys()
        self.m = m_samples
        self.temperature = temperature
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.max_retries = max_retries
        self._k = 0
        self._klock = threading.Lock()
        self.stats = {"calls": 0, "retries": 0, "parse_fail": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "latency_sum": 0.0}
        self._slock = threading.Lock()

    def _next_key(self):
        with self._klock:
            k = self.keys[self._k % len(self.keys)]
            self._k += 1
            return k

    def _one_call(self, messages, max_tokens=700):
        for attempt in range(self.max_retries):
            key = self._next_key()
            t0 = time.time()
            try:
                r = requests.post(
                    API_BASE,
                    headers={"Authorization": "Bearer " + key,
                             "Content-Type": "application/json"},
                    json={"model": MODEL, "messages": messages,
                          "temperature": self.temperature, "max_tokens": max_tokens},
                    timeout=90)
                if r.status_code != 200:
                    raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:200]))
                js = r.json()
                content = js["choices"][0]["message"]["content"] or ""
                usage = js.get("usage", {})
                with self._slock:
                    self.stats["calls"] += 1
                    self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.stats["completion_tokens"] += usage.get("completion_tokens", 0)
                    self.stats["latency_sum"] += time.time() - t0
                return content
            except Exception:
                with self._slock:
                    self.stats["retries"] += 1
                time.sleep(1.5 * (attempt + 1))
        return None

    @staticmethod
    def _parse(text, available):
        if text is None:
            return None
        t = text.strip().strip('"').strip("'").strip(".").strip()
        low = t.lower()
        for a in available:
            if low == a.lower():
                return a
        hits = [a for a in available if a.lower() in low]
        if len(hits) == 1:
            return hits[0]
        if hits:
            # choose the earliest mention in the text
            hits.sort(key=lambda a: low.find(a.lower()))
            return hits[0]
        return None

    def sample_prior(self, native_prompts, available, chosen_so_far):
        """Return (logp_hat dict over available, diag dict).

        native_prompts: the host system's prompt structure
        [[game_rules, "Ok."], [situation_text, "My actions are: ..."]].
        The host's own scoring of a prefilled assistant turn is replaced by an
        explicit single-action instruction.
        """
        q0 = native_prompts[0][0]
        situation = native_prompts[1][0]
        acted = ("Actions you have already taken, in order: " + chosen_so_far + ".\n") \
            if chosen_so_far else ""
        user = (situation + "\n" + acted +
                "Available actions right now: " + ", ".join(available) + ".\n" +
                "Reply with exactly one action, copied verbatim from the available list. "
                "No explanation, no punctuation, nothing else.")
        messages = [{"role": "system", "content": q0},
                    {"role": "user", "content": user}]
        futures = [self.pool.submit(self._one_call, messages) for _ in range(self.m)]
        counts = {a: 0 for a in available}
        n_ok = 0
        for f in futures:
            a = self._parse(f.result(), available)
            if a is None:
                with self._slock:
                    self.stats["parse_fail"] += 1
                continue
            counts[a] += 1
            n_ok += 1
        # additive smoothing over the available set
        beta = 0.5
        denom = n_ok + beta * len(available)
        logp = {a: math.log((counts[a] + beta) / denom) for a in available}
        return logp, {"counts": counts, "n_ok": n_ok}


# --------------------------------------------------------------------- call site
# The host builds (prompts, choices, available_moves, prob_base) and the sampled
# prior replaces the host's own language-model likelihood:
#
#     chosen_so_far = ", ".join(m["task"] for m in mov_his)
#     prior, diag = router.prior(prompts, am, chosen_so_far)
#     final = np.array([prior[a] - pb[a] for a in am], dtype=float)
#     move_name = am[int(np.argmax(final))]
#
# Each decision makes M = 10 calls (--m-samples 10) at temperature 1.0 and
# max_tokens 700; an episode has about 16 decisions, so about 160 calls.
