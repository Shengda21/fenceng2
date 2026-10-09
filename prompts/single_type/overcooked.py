# Extracted from scripts/run_overcooked_sweep.py (VALID_TASKS, call_llm, parse_plan, _facility_reach, build_plan_prompt), scripts/llm_rotate.py (call_llm_rotating), scripts/task_resolve.py (VALID_TASKS) and scripts/run_task_deployment.py (the llm_task branch of refresh_tasks).
"""Overcooked (four layouts, two chefs): the task-pair prompt of the llm_task arm.

The planner reads the pot, what each chef holds and which facilities each chef can
reach, and names one high-level task per chef. The tasks are resolved and carried out
by a fixed executor (BFS path finding and counter handover) for the next 20 steps.
"""
import itertools
import os
import time
from typing import Any, Optional

import requests

# ---------------------------------------------------------------- call settings
# The runs set LLM_MODEL to gpt-oss-120b and LLM_API_BASE to an OpenAI-compatible
# server through the environment; the address and the keys are not part of this file.
LLM_API_BASE = os.environ.get("LLM_API_BASE", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "30"))

VALID_TASKS = {
    "get_onion", "put_onion", "cook", "get_dish", "plate_soup",
    "serve_soup", "drop_held_item", "stay",
}
DEFAULT_TASKS = ["get_onion", "get_onion"]

# The deployment script checks the parsed list against this list (named VALID_TASKS
# in task_resolve.py); it holds the same eight task names.
RESOLVER_VALID_TASKS = [
    "get_onion",
    "put_onion",
    "get_dish",
    "plate_soup",
    "serve_soup",
    "cook",
    "drop_held_item",
    "stay",
]


def call_llm(prompt: str) -> str:
    if not LLM_API_KEY:
        return "get_onion | get_onion"
    body: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "120")),
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
        return (content or "get_onion | get_onion").strip()
    except Exception as e:
        print(f"  [LLM-FAIL] {e}", flush=True)
        return "get_onion | get_onion"


# The runs that spread the load over several keys sent the same request body
# through this caller. It tries up to three keys, waits and retries on a 429 or an
# error, and counts a call as an API failure when every attempt fails. A failed call
# returns FAIL_DEFAULT, and the failure is recorded apart from a parse fallback.
LLM_TIMEOUT_ROTATING = float(os.environ.get("LLM_TIMEOUT", "45"))

_keys = [k.strip() for k in os.environ.get("LLM_API_KEYS", "").split(",") if k.strip()]
if not _keys and os.environ.get("LLM_API_KEY"):
    _keys = [os.environ["LLM_API_KEY"]]
_key_cycle = itertools.cycle(_keys) if _keys else None

FAIL_DEFAULT = "get_onion | get_onion"


def call_llm_rotating(prompt: str) -> tuple[str, int, bool]:
    """Returns (raw_text, n_attempts, api_failed)."""
    if _key_cycle is None:
        return FAIL_DEFAULT, 0, True
    body: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "120")),
    }
    max_attempts = min(3, max(1, len(_keys)))
    last_err = None
    for attempt in range(1, max_attempts + 1):
        key = next(_key_cycle)
        try:
            resp = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json=body,
                timeout=LLM_TIMEOUT_ROTATING,
            )
            if resp.status_code == 429:
                last_err = RuntimeError("429 rate limited")
                time.sleep(2 * attempt)
                continue
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content")
            if not content:
                content = msg.get("reasoning_content") or msg.get("reasoning") or ""
            text = (content or "").strip()
            if text:
                return text, attempt, False
            last_err = RuntimeError("empty completion")
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1 * attempt)
    print(f"  [LLM-API-FAILED after {max_attempts} keys] {last_err}", flush=True)
    return FAIL_DEFAULT, max_attempts, True


# ---------------------------------------------------------------------- parsing
def parse_plan(plan: str) -> list[str]:
    """Extract the next task of each of the two chefs from the reply.

    Accepted forms: 'get_onion | put_onion'; a single task; '... ; ...' (the first
    part is kept); 'chef_0:X | chef_1:Y'. The result always has length 2.
    """
    p = plan.strip().replace("(", "").replace(")", "")
    # take the last non-empty line (a reasoning model may ramble before it)
    lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
    if lines:
        p = lines[-1]
    # a semicolon keeps the first part
    if ";" in p:
        p = p.split(";")[0].strip()

    # supports 'chef_0:X | chef_1:Y'
    raw_tasks: list[str] = []
    for tok in p.split("|"):
        tok = tok.strip()
        if ":" in tok:
            tok = tok.split(":", 1)[1].strip()
        # strip characters that are not part of a task name
        tok = tok.split()[0] if tok else ""
        tok = tok.strip(",.;:'\"")
        if tok in VALID_TASKS:
            raw_tasks.append(tok)

    if not raw_tasks:
        return list(DEFAULT_TASKS)
    while len(raw_tasks) < 2:
        raw_tasks.append(raw_tasks[-1])
    return raw_tasks[:2]


def _is_valid_task_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(task, str) and task in RESOLVER_VALID_TASKS for task in value)
    )


def parser_used_default_fallback(parsed_tasks: object) -> bool:
    """Detect ``parse_plan``'s default result plus defensive invalid results."""
    if not _is_valid_task_list(parsed_tasks):
        return True
    return parsed_tasks == DEFAULT_TASKS


def _parse_validated_plan(plan_text: str) -> tuple[list[str], bool]:
    parsed_tasks = parse_plan(plan_text)
    parse_fallback = parser_used_default_fallback(parsed_tasks)
    if not _is_valid_task_list(parsed_tasks):
        return list(DEFAULT_TASKS), True
    return list(parsed_tasks), parse_fallback


# ------------------------------------------------------------------- the prompt
def _facility_reach(mapper: "OvercookedActionMapper", idx: int) -> str:
    """Summary of whether chef idx can reach each of the four facility types."""
    pos = mapper._get_pos(idx)
    onion = list(mapper.mdp.get_onion_dispenser_locations())
    dish = list(mapper.mdp.get_dish_dispenser_locations())
    pot = list(mapper.mdp.get_pot_locations())
    serve = list(mapper.mdp.get_serving_locations())

    def reach(targets):
        for t in targets:
            if mapper._bfs(pos, lambda p, t=t: p in mapper._neighbors(t)):
                return True
        return False

    return (
        f"reach[onion]={reach(onion)} dish={reach(dish)} "
        f"pot={reach(pot)} serve={reach(serve)}"
    )


def build_plan_prompt(layout: str, env_wrap: "OvercookedEnvWrapper",
                      mapper: Optional["OvercookedActionMapper"] = None) -> str:
    state = env_wrap.get_state()
    pot_locs = env_wrap._mdp.get_pot_locations() if env_wrap._mdp else []
    pots = state.objects.get("soups", []) if state else []
    pot_summary = "empty"
    if pot_locs and pots:
        target = pot_locs[0]
        for pp in pots:
            if pp["position"] == target:
                s = str(pp.get("state", ""))
                n_onions = s.count("'onion'")
                ready = "ready:True" in s
                cooking = "cooking:True" in s
                pot_summary = f"onions={n_onions} cooking={cooking} ready={ready}"
                break

    held = []
    for p in state.players:
        held.append(p.get("held_object") or "empty")

    reach0 = _facility_reach(mapper, 0) if mapper else ""
    reach1 = _facility_reach(mapper, 1) if mapper else ""

    return (
        f"You are an Overcooked task planner. Layout: {layout}.\n"
        f"Pot status: {pot_summary}.\n"
        f"Chef 0 holds: {held[0] if len(held) > 0 else 'empty'}. {reach0}\n"
        f"Chef 1 holds: {held[1] if len(held) > 1 else 'empty'}. {reach1}\n"
        "Note: if a chef has reach=False for a facility, they CANNOT do "
        "tasks needing it directly; they must drop items on a shared counter "
        "for the other chef to pick up (handover).\n"
        "Pick the best next high-level task for each chef.\n"
        "Available tasks: get_onion, put_onion, cook, get_dish, plate_soup, "
        "serve_soup, drop_held_item, stay.\n"
        "Output ONLY two task names separated by ' | ' (chef_0_task | chef_1_task).\n"
        "Example: get_onion | get_dish\n"
        "Answer:"
    )


# --------------------------------------------------------------------- call site
# Planner calls per episode: one every 20 steps (replan_every=20) over 400 steps,
# which is 20 calls. Chef i follows tasks[i % len(tasks)]. A reply that gives no
# valid task falls back to DEFAULT_TASKS and is counted as a parse fallback.
def select_tasks(layout: str, env_wrap, mapper) -> tuple[list[str], bool]:
    prompt = build_plan_prompt(layout, env_wrap, mapper)
    if os.environ.get("LLM_API_KEYS"):
        raw_text, n_attempts, api_failed = call_llm_rotating(prompt)
    else:
        raw_text = call_llm(prompt)
    return _parse_validated_plan(raw_text)
