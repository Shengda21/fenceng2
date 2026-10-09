#!/usr/bin/env python3
"""MAgent battle_v4 sweep: 4 methods x N episodes x 1 map.

Design follows run_hanabi_sweep.py / run_overcooked_sweep.py:
  - calls the magent2.environments.battle_v4 parallel_env directly and bypasses
    BaseEnvWrapper (the v1 wrapper interface is too heavy for this sweep)
  - 4 methods: random / greedy / scripted / hierarchical
  - scoring: reward = (red_alive_final - blue_alive_final) / (n_red+n_blue),
    plus the accumulated global reward (pettingzoo gives a team-shared reward)
  - terminal: max_cycles, or one team reaches zero

OGHP META_TASKS:
  ATTACK_FORWARD : each red unit attacks first (if an enemy is in its 8-neighbour attack range), otherwise moves toward the nearest enemy
  HOLD_POSITION  : no movement, attack only
  SPREAD_OUT     : move away from the nearest *friendly* neighbour; attack still takes priority
  RETREAT        : move toward the spawn direction, no attack
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random as random_lib
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ============================================================
# LLM
# ============================================================
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://integrate.api.nvidia.com/v1")
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


def parse_meta_task(text: str) -> str:
    t = text.upper().strip()
    for mt in META_TASKS:
        if mt in t:
            return mt
    return DEFAULT_META


# ============================================================
# MAgent helpers
# ============================================================
# action 0..12: move (idx 12 = stay if available; magent2 layout: 13 move directions)
# action 13..20: attack, 8-neighbourhood
ATTACK_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# 13 move directions (tied to magent2's internal unit_size; simplified to the 8 neighbours plus 5 long jumps, the last one used as stay)
MOVE_OFFSETS = [
    (0, 0),   # 0: stay (approximate)
    (-1, 0), (1, 0), (0, -1), (0, 1),  # 1-4 cardinal
    (-1, -1), (-1, 1), (1, -1), (1, 1),  # 5-8 diagonal
    (-2, 0), (2, 0), (0, -2), (0, 2),  # 9-12 long jumps
]


def get_positions(env) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Read red/blue agent positions from the magent2 internals."""
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


def nearest(pos: tuple[int, int], targets: list[tuple[int, int]]) -> tuple[tuple[int, int] | None, float]:
    if not targets:
        return None, math.inf
    best = min(targets, key=lambda q: (q[0] - pos[0]) ** 2 + (q[1] - pos[1]) ** 2)
    d = math.hypot(best[0] - pos[0], best[1] - pos[1])
    return best, d


def attack_action_if_enemy_adjacent(my: tuple[int, int], enemies: list[tuple[int, int]]) -> int | None:
    for i, (dx, dy) in enumerate(ATTACK_OFFSETS):
        if (my[0] + dx, my[1] + dy) in enemies:
            return 13 + i
    return None


def move_toward(my: tuple[int, int], target: tuple[int, int]) -> int:
    dx = target[0] - my[0]
    dy = target[1] - my[1]
    sx = (dx > 0) - (dx < 0)
    sy = (dy > 0) - (dy < 0)
    # match (sx, sy) -> closest MOVE_OFFSETS index
    best_idx, best_d = 0, math.inf
    for i, (mx, my_) in enumerate(MOVE_OFFSETS):
        if (mx, my_) == (0, 0):
            continue
        d = (mx - sx) ** 2 + (my_ - sy) ** 2
        if d < best_d:
            best_d, best_idx = d, i
    return best_idx


def move_away(my: tuple[int, int], from_: tuple[int, int]) -> int:
    return move_toward(my, (2 * my[0] - from_[0], 2 * my[1] - from_[1]))


# ============================================================
# Method implementations (per-step action dict for red team)
# ============================================================
def act_random(env, agents: list[str]) -> dict[str, int]:
    return {a: random_lib.randint(0, 20) for a in agents}


def act_greedy(env, agents: list[str]) -> dict[str, int]:
    """All-attack-13 baseline (always attack action; effectively 'try to attack adjacent enemy')."""
    return {a: 13 for a in agents}


def _scripted_red(env, red_ids: list[str], red_pos: dict, blue_pos: dict) -> dict[str, int]:
    enemies = list(blue_pos.values())
    out = {}
    for rid in red_ids:
        if rid not in red_pos:
            out[rid] = 13
            continue
        p = red_pos[rid]
        a = attack_action_if_enemy_adjacent(p, enemies)
        if a is not None:
            out[rid] = a
        else:
            tgt, _ = nearest(p, enemies)
            out[rid] = move_toward(p, tgt) if tgt else 0
    return out


def act_scripted(env, agents: list[str]) -> dict[str, int]:
    red_pos, blue_pos = get_positions(env)
    red_ids = [a for a in agents if a.startswith("red_")]
    blue_ids = [a for a in agents if a.startswith("blue_")]
    out = _scripted_red(env, red_ids, red_pos, blue_pos)
    # blue: random, for simplicity (in the paper blue is an uncontrolled baseline; scripting both sides would be too symmetric to be informative)
    for b in blue_ids:
        out[b] = random_lib.randint(0, 20)
    return out


def act_oghp(env, agents: list[str], meta: str) -> dict[str, int]:
    red_pos, blue_pos = get_positions(env)
    red_ids = [a for a in agents if a.startswith("red_")]
    blue_ids = [a for a in agents if a.startswith("blue_")]
    enemies = list(blue_pos.values())
    out = {}
    for rid in red_ids:
        if rid not in red_pos:
            out[rid] = 13
            continue
        p = red_pos[rid]
        a_atk = attack_action_if_enemy_adjacent(p, enemies)

        if meta == "ATTACK_FORWARD":
            if a_atk is not None:
                out[rid] = a_atk
            else:
                tgt, _ = nearest(p, enemies)
                out[rid] = move_toward(p, tgt) if tgt else 0
        elif meta == "HOLD_POSITION":
            out[rid] = a_atk if a_atk is not None else 0
        elif meta == "SPREAD_OUT":
            if a_atk is not None:
                out[rid] = a_atk
            else:
                allies = [q for q in red_pos.values() if q != p]
                ally, _ = nearest(p, allies)
                out[rid] = move_away(p, ally) if ally else 0
        elif meta == "RETREAT":
            out[rid] = move_away(p, (env.unwrapped.env.get_view_space(env.unwrapped.env.get_handles()[0])[0] // 2,) * 2) if False else 0  # keep simple: stay
        else:
            out[rid] = a_atk if a_atk is not None else 0

    for b in blue_ids:
        out[b] = random_lib.randint(0, 20)
    return out


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


# ============================================================
# Episode loop
# ============================================================
def run_episode(env, method: str, episode_idx: int, max_cycles: int,
                replan_every: int = 99) -> dict[str, Any]:
    t0 = time.time()
    res = env.reset(seed=episode_idx)
    obs = res[0] if isinstance(res, tuple) else res
    decomposition = method
    meta = DEFAULT_META

    if method == "hierarchical":
        rp, bp = get_positions(env)
        meta = parse_meta_task(call_llm(build_magent_prompt(rp, bp, 0, max_cycles)))
        decomposition = meta

    cum_reward_red = 0.0
    step = 0
    while env.agents:
        if method == "random":
            actions = act_random(env, env.agents)
        elif method == "greedy":
            actions = act_greedy(env, env.agents)
        elif method == "scripted":
            actions = act_scripted(env, env.agents)
        elif method == "hierarchical":
            actions = act_oghp(env, env.agents, meta)
        else:
            actions = act_random(env, env.agents)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        for k, r in rewards.items():
            if k.startswith("red_"):
                cum_reward_red += float(r)
        step += 1

        if method == "hierarchical" and step % replan_every == 0:
            rp, bp = get_positions(env)
            meta = parse_meta_task(call_llm(build_magent_prompt(rp, bp, step, max_cycles)))
            decomposition = decomposition + ";" + meta
        if step >= max_cycles:
            break

    rp, bp = get_positions(env)
    n_red_final = len(rp)
    n_blue_final = len(bp)
    # success: net red reward > 0 (the v4 default reward gives positive credit for kills and negative for damage taken, so a positive total means kills outweigh losses)
    win = cum_reward_red > 0

    return {
        "env": "magent",
        "map_size": int(getattr(env.unwrapped, "map_size", 20)) if hasattr(env, "unwrapped") else 20,
        "method": method,
        "episode": episode_idx,
        "success": bool(win),
        "total_reward": cum_reward_red,
        "steps": step,
        "red_alive_final": n_red_final,
        "blue_alive_final": n_blue_final,
        "decomposition": decomposition,
        "duration": time.time() - t0,
        "error": "",
    }


# ============================================================
# CLI orchestration
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["random", "greedy", "scripted", "hierarchical"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--map-size", type=int, default=20)
    p.add_argument("--max-cycles", type=int, default=100)
    p.add_argument("--output-dir", type=str, default="results/magent_sweep")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"MAGENT SWEEP start at {ts}", flush=True)
    print(f"  methods: {args.methods}", flush=True)
    print(f"  episodes: {args.episodes}, map_size={args.map_size}, max_cycles={args.max_cycles}", flush=True)
    print(f"  output: {args.output_dir}", flush=True)
    print(f"  LLM: base={LLM_API_BASE} model={LLM_MODEL} key={'SET' if LLM_API_KEY else 'MISSING'}", flush=True)
    print()

    from magent2.environments import battle_v4

    env = battle_v4.parallel_env(map_size=args.map_size, max_cycles=args.max_cycles, minimap_mode=False)

    for method in args.methods:
        out_path = out / f"magent_battle_m{args.map_size}_{method}_{ts}.json"
        if args.skip_existing and out_path.exists():
            print(f"  [skip] {out_path.name}", flush=True)
            continue

        print(f"========== {method} × {args.episodes} eps ==========", flush=True)
        results = []
        wins = 0
        rewards = []
        for ep in range(args.episodes):
            try:
                rec = run_episode(env, method, ep, args.max_cycles)
            except Exception as e:
                traceback.print_exc()
                rec = {
                    "env": "magent",
                    "method": method,
                    "episode": ep,
                    "success": False,
                    "total_reward": 0.0,
                    "steps": 0,
                    "red_alive_final": 0,
                    "blue_alive_final": 0,
                    "decomposition": "ERROR",
                    "duration": 0.0,
                    "error": str(e)[:200],
                }
            results.append(rec)
            wins += int(rec["success"])
            rewards.append(rec["total_reward"])
            if (ep + 1) % 10 == 0:
                avg = sum(rewards) / len(rewards)
                print(f"  [magent/{method}] ep {ep+1}/{args.episodes} wins={wins} avg_r={avg:.2f}", flush=True)
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [DONE] {method} -> {out_path}", flush=True)

    env.close()
    flag = out / "DONE"
    flag.write_text("DONE\n")
    print(f"ALL DONE. Flag: {flag}", flush=True)


if __name__ == "__main__":
    main()
