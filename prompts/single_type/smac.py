# Extracted from scripts/run_smac_sweep.py (call_llm, get_tactical_plan), scripts/hier_v2.py (parse_smac_tactics_v2, default and valid tactics) and scripts/run_smac_sweep_v2.py (run_episode_v2, the call site).
"""SMAC (3m, 8m, MMM): the tactic prompt.

The selector sees the map name, both team sizes and the health of every unit, and
names a tactic. A fixed executor carries the tactic out until the next call. The
prompt is written in Chinese and was sent as written; strings are not translated.
"""
import os
from typing import Any

import requests

# ---------------------------------------------------------------- call settings
# The runs set LLM_MODEL to gpt-oss-120b and LLM_API_BASE to an OpenAI-compatible
# server through the environment; the address and the key are not part of this file.
LLM_API_BASE = os.environ.get("LLM_API_BASE", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "30"))


def call_llm(prompt: str) -> str:
    """OpenAI-compatible call; if the answer is in reasoning_content, that is used."""
    if not LLM_API_KEY:
        return "focus_fire ; kite"
    body: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "400")),
    }
    eff = os.environ.get("LLM_REASONING_EFFORT")
    if eff or "nvidia" in LLM_API_BASE.lower():
        body["reasoning_effort"] = eff or "low"
    try:
        resp = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content")
        if not content:
            content = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return (content or "focus_fire ; kite").strip()
    except Exception as e:
        print(f"  [LLM-FAIL] {e}", flush=True)
        return "focus_fire ; kite"


# ------------------------------------------------------------------- the prompt
def get_tactical_plan(env_wrap, ally_health: list[float],
                      enemy_health: list[float], map_name: str) -> str:
    n_agents = env_wrap.n_agents
    n_enemies = len(enemy_health)
    prompt = (
        f"你是星际争霸 II 战术指挥官。地图: {map_name}。"
        f"我方 {n_agents} 单位 vs 敌方 {n_enemies} 单位。\n"
        f"我方血量: {[f'{h:.0%}' for h in ally_health]}\n"
        f"敌方血量: {[f'{h:.0%}' for h in enemy_health]}\n\n"
        "可用战术: focus_fire / spread_fire / kite / retreat / advance\n"
        "DSL 组合: 'A ; B' 顺序; 'A | B' 并行\n"
        "只输出 DSL 表达式（不超过两个战术）："
    )
    return call_llm(prompt)


# ---------------------------------------------------------------------- parsing
SMAC_DEFAULT_TACTIC = "focus_fire"
SMAC_VALID_TACTICS = {"focus_fire", "spread_fire", "kite", "retreat", "advance"}


def parse_smac_tactics_v2(plan: str) -> list[str]:
    """Tactics named in the reply; only the part before the first ";" is read."""
    plan = (plan or "").strip().replace("(", "").replace(")", "")
    if not plan:
        return [SMAC_DEFAULT_TACTIC]
    if ";" in plan:
        plan = plan.split(";")[0].strip()
    raw = []
    for tok in plan.split("|"):
        tok = tok.strip().lower()
        for t in SMAC_VALID_TACTICS:
            if t in tok:
                raw.append(t)
                break
    return raw if raw else [SMAC_DEFAULT_TACTIC]


# --------------------------------------------------------------------- call site
# Selector calls per episode: one at the start and one after every 20 steps
# (replan_every=20). Unit i follows tactics[i % len(tactics)].
def select_tactics(env_wrap, map_name: str) -> list[str]:
    state = env_wrap.get_state()
    ally_h = [u.health / 100.0 if u.is_alive else 0.0 for u in state.ally_units]
    enemy_h = [u.health / 100.0 if u.is_alive else 0.0 for u in state.enemy_units]
    plan_str = get_tactical_plan(env_wrap, ally_h, enemy_h, map_name)
    return parse_smac_tactics_v2(plan_str)
