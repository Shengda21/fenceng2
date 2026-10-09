# Extracted from scripts/run_magent_sweep.py (META_TASKS, call_llm, get_positions, build_magent_prompt), scripts/hier_v2.py (parse_magent_meta_v2) and scripts/run_magent_sweep_v2.py (run_episode_v2, the call site).
"""MAgent battle (map_size 20, 12 vs 12): the meta-task prompt.

The selector sees the live unit counts, the step, the two centroids and their
distance, and names one of four meta-tasks for the red team. A fixed executor
carries it out until the next call.
"""
import math
import os
import time
from typing import Any

import requests

# ---------------------------------------------------------------- call settings
# The runs set LLM_MODEL to gpt-oss-120b and LLM_API_BASE to an OpenAI-compatible
# server through the environment; the address and the key are not part of this file.
LLM_API_BASE = os.environ.get("LLM_API_BASE", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "30"))

META_TASKS = ["ATTACK_FORWARD", "HOLD_POSITION", "SPREAD_OUT", "RETREAT"]
DEFAULT_META = "ATTACK_FORWARD"


def call_llm(prompt: str) -> str:
    if not LLM_API_KEY:
        return DEFAULT_META
    body: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "60")),
    }
    eff = os.environ.get("LLM_REASONING_EFFORT")
    if eff or "nvidia" in LLM_API_BASE.lower():
        body["reasoning_effort"] = eff or "low"
    backoff = [0, 5, 15, 30]
    last_err: Exception | None = None
    for wait in backoff:
        if wait:
            time.sleep(wait)
        try:
            resp = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}",
                         "Content-Type": "application/json"},
                json=body,
                timeout=LLM_TIMEOUT,
            )
            if resp.status_code == 429:
                last_err = RuntimeError(f"429 {resp.text[:80]}")
                continue
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
            return (content or DEFAULT_META).strip()
        except Exception as e:
            last_err = e
            continue
    print(f"  [LLM-FAIL] {last_err}", flush=True)
    return DEFAULT_META


# ------------------------------------------------------------------- positions
def get_positions(env) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Red and blue unit positions read from the underlying magent2 environment."""
    red_pos, blue_pos = {}, {}
    try:
        raw = env.unwrapped
        if hasattr(raw, "env"):
            handles = raw.env.get_handles()
            r_pos = raw.env.get_pos(handles[0])
            b_pos = raw.env.get_pos(handles[1])
            r_ids = sorted([a for a in env.agents if a.startswith("red_")],
                           key=lambda x: int(x.split('_')[1]))
            b_ids = sorted([a for a in env.agents if a.startswith("blue_")],
                           key=lambda x: int(x.split('_')[1]))
            for i, p in enumerate(r_pos):
                if i < len(r_ids):
                    red_pos[r_ids[i]] = (int(p[0]), int(p[1]))
            for i, p in enumerate(b_pos):
                if i < len(b_ids):
                    blue_pos[b_ids[i]] = (int(p[0]), int(p[1]))
    except Exception:
        pass
    return red_pos, blue_pos


# ------------------------------------------------------------------- the prompt
def build_magent_prompt(red_pos: dict, blue_pos: dict, step: int, max_cycles: int) -> str:
    n_red = len(red_pos)
    n_blue = len(blue_pos)
    if n_red and n_blue:
        rxs = [p[0] for p in red_pos.values()]
        rys = [p[1] for p in red_pos.values()]
        bxs = [p[0] for p in blue_pos.values()]
        bys = [p[1] for p in blue_pos.values()]
        red_centroid = (sum(rxs) / n_red, sum(rys) / n_red)
        blue_centroid = (sum(bxs) / n_blue, sum(bys) / n_blue)
        dist = math.hypot(red_centroid[0] - blue_centroid[0], red_centroid[1] - blue_centroid[1])
    else:
        red_centroid = blue_centroid = (0, 0)
        dist = 0
    return (
        "You are a meta-planner for the RED team in MAgent battle (a multi-agent skirmish).\n"
        f"Red alive: {n_red}, Blue alive: {n_blue}, step: {step}/{max_cycles}.\n"
        f"Red centroid: {red_centroid}, Blue centroid: {blue_centroid}, centroid_dist: {dist:.1f}.\n"
        "Pick exactly ONE meta-task for the team this phase:\n"
        "  ATTACK_FORWARD - move toward nearest enemy and attack adjacent ones (default offensive)\n"
        "  HOLD_POSITION  - stay and only attack adjacent enemies (when outnumbering blue locally)\n"
        "  SPREAD_OUT     - spread away from allies; attack adjacent enemies (avoid clustering)\n"
        "  RETREAT        - hold (a pacifist last resort; rarely useful)\n"
        "Heuristic: if red >= blue and dist < 5 → ATTACK_FORWARD; if red < blue → SPREAD_OUT (avoid being surrounded).\n"
        "Output ONLY the meta-task token. Example: ATTACK_FORWARD\n"
        "Answer:"
    )


# ---------------------------------------------------------------------- parsing
MAGENT_DEFAULT_META = "ATTACK_FORWARD"
MAGENT_VALID_METAS = {"ATTACK_FORWARD", "HOLD_POSITION", "SPREAD_OUT", "RETREAT"}


def parse_magent_meta_v2(text: str) -> str:
    if not text:
        return MAGENT_DEFAULT_META
    t = text.upper()
    for m in MAGENT_VALID_METAS:
        if m in t:
            return m
    return MAGENT_DEFAULT_META


# --------------------------------------------------------------------- call site
# Selector calls per episode: one at the start, then one after every 99 steps
# (replan_every=99), with max_cycles=100, so the second call is made only if the
# episode is still running after step 99. The text is read with the parser above.
# The temperature-controlled router runs sent the same prompt through a copy of
# call_llm_v2 (see hanabi.py) that takes the temperature as an argument, with
# default="ATTACK_FORWARD", max_tokens=80 and temperature=0.5.
def select_meta(env, step: int, max_cycles: int = 100) -> str:
    rp, bp = get_positions(env)
    text = call_llm(build_magent_prompt(rp, bp, step, max_cycles))
    return parse_magent_meta_v2(text)
