"""Scripted BLUE/RED tactics for the MAgent combined_arms regime harness."""

from __future__ import annotations

import hashlib
import math
from itertools import product
from typing import Any

import numpy as np


MELEE_MOVE_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
MELEE_ATTACK_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RANGED_MOVE_OFFSETS = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
]
RANGED_ATTACK_OFFSETS = RANGED_MOVE_OFFSETS[1:]

BASE_TYPES = ["MELEE_RUSH", "RANGED_STANDOFF", "MIXED_ADVANCE", "KITE"]
TRAIT_TYPES = [
    f"{advance}-{ranged_stance}-{focus}"
    for advance, ranged_stance, focus in product(
        ["push", "hold"], ["standoff", "screen"], ["melee_first", "ranged_first"]
    )
]
TYPE_SETS = {"base4": BASE_TYPES, "traits8": TRAIT_TYPES}

COMBINED_POOL6 = (
    "ALL_ATTACK",
    "HOLD_LINE",
    "RANGED_FIRST",
    "MELEE_FIRST",
    "SCREEN_KITE",
    "FLANK_RANGED",
)
POOL_SETS = {"combined_pool6": COMBINED_POOL6}

TYPE_DOCS = {
    "MELEE_RUSH": "Melee units charge ahead while ranged units stay on the back line and only fire if the fight reaches them.",
    "RANGED_STANDOFF": "Ranged units hold at range two and shoot while melee units form a forward screen.",
    "MIXED_ADVANCE": "Melee and ranged units advance in step; ranged units close to range two and fire before melee contact.",
    "KITE": "Ranged units alternate firing with stepping back while melee units screen the nearest threats.",
}
TYPE_SEMANTICS = {
    "MELEE_RUSH": "isolated melee-first pressure with ranged units anchored behind the charge",
    "RANGED_STANDOFF": "ranged standoff fire behind a melee screen",
    "MIXED_ADVANCE": "coordinated whole-force advance with ranged units firing at range two",
    "KITE": "ranged fire-and-step-back pressure behind a melee screen",
}

for _name in TRAIT_TYPES:
    _advance, _stance, _focus = _name.split("-")
    TYPE_DOCS[_name] = (
        f"{_advance} movement, ranged {_stance} positioning, and {_focus.replace('_', ' ')} target priority."
    )
    TYPE_SEMANTICS[_name] = (
        f"{_advance} movement with ranged {_stance} behavior and {_focus.replace('_', ' ')} focus"
    )


def describe_type(name: str) -> str:
    if name == "random":
        return "Units choose deterministic pseudo-random legal low-level actions without a stable team tactic."
    if name not in TYPE_DOCS:
        raise ValueError(f"unknown BLUE type: {name}")
    return TYPE_DOCS[name]


def semantic_descriptor(name: str) -> str:
    if name == "random":
        return "unstable random low-level movement without persistent class-specific behavior"
    if name not in TYPE_SEMANTICS:
        raise ValueError(f"unknown BLUE type: {name}")
    return TYPE_SEMANTICS[name]


def act_blue(env, agents: list[str], blue_type: str, rng, step: int, state: dict[str, Any]) -> dict[str, int]:
    if blue_type == "random":
        return {a: int(rng.integers(0, action_space_n(a))) for a in agents if is_blue(a)}

    pos = get_combined_positions(env)
    blue_ids = sorted([a for a in agents if is_blue(a)], key=agent_sort_key)
    red_items = sorted(pos["red"].items(), key=lambda kv: agent_sort_key(kv[0]))
    red_melee = [(a, p) for a, p in red_items if is_melee(a)]
    red_ranged = [(a, p) for a, p in red_items if is_ranged(a)]

    if blue_type == "MELEE_RUSH":
        return _blue_melee_rush(blue_ids, pos, red_melee, red_ranged)
    if blue_type == "RANGED_STANDOFF":
        return _blue_standoff(blue_ids, pos, red_melee, red_ranged, advance=False, focus="melee_first")
    if blue_type == "MIXED_ADVANCE":
        return _blue_mixed_advance(blue_ids, pos, red_melee, red_ranged)
    if blue_type == "KITE":
        return _blue_kite(blue_ids, pos, red_melee, red_ranged, step, state)
    if blue_type in TRAIT_TYPES:
        advance, stance, focus = blue_type.split("-")
        if stance == "standoff":
            return _blue_standoff(blue_ids, pos, red_melee, red_ranged, advance=(advance == "push"), focus=focus)
        return _blue_trait_screen(blue_ids, pos, red_melee, red_ranged, advance=(advance == "push"), focus=focus)
    raise ValueError(f"unknown BLUE type: {blue_type}")


def act_red_strategy(env, agents: list[str], tau: str, step: int = 0, state: dict[str, Any] | None = None) -> dict[str, int]:
    if state is None:
        state = {}
    pos = get_combined_positions(env)
    red_ids = sorted([a for a in agents if is_red(a)], key=agent_sort_key)
    blue_items = sorted(pos["blue"].items(), key=lambda kv: agent_sort_key(kv[0]))
    blue_melee = [(a, p) for a, p in blue_items if is_melee(a)]
    blue_ranged = [(a, p) for a, p in blue_items if is_ranged(a)]

    if tau == "ALL_ATTACK":
        return _all_attack(red_ids, pos, blue_melee, blue_ranged)
    if tau == "HOLD_LINE":
        return _hold_line(red_ids, pos, blue_melee, blue_ranged)
    if tau == "RANGED_FIRST":
        return _priority_attack(red_ids, pos, blue_ranged, blue_melee)
    if tau == "MELEE_FIRST":
        return _priority_attack(red_ids, pos, blue_melee, blue_ranged)
    if tau == "SCREEN_KITE":
        return _screen_kite(red_ids, pos, blue_melee, blue_ranged, step, state)
    if tau == "FLANK_RANGED":
        return _flank_ranged(red_ids, pos, blue_melee, blue_ranged, step)
    return _hold_line(red_ids, pos, blue_melee, blue_ranged)


def episode_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) * 1000 + 71)


def keyed_uniform(episode_seed: int, absolute_step: int, agent_id: str, purpose: str) -> float:
    digest = hashlib.sha256(
        f"combined:{int(episode_seed)}:{int(absolute_step)}:{agent_id}:{purpose}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") >> 11
    return value / float(1 << 53)


def keyed_int(episode_seed: int, absolute_step: int, agent_id: str, purpose: str, high: int) -> int:
    if high <= 0:
        raise ValueError("high must be positive")
    digest = hashlib.sha256(
        f"combined:{int(episode_seed)}:{int(absolute_step)}:{agent_id}:{purpose}".encode("utf-8")
    ).digest()
    return int(int.from_bytes(digest[:8], "big") % int(high))


def get_combined_positions(env) -> dict[str, dict[str, tuple[int, int]]]:
    out = {"red": {}, "blue": {}, "redmelee": {}, "redranged": {}, "bluemelee": {}, "blueranged": {}}
    if hasattr(env, "positions"):
        for aid, p in getattr(env, "positions", {}).items():
            pp = (int(p[0]), int(p[1]))
            if aid.startswith("red"):
                out["red"][aid] = pp
            if aid.startswith("blue"):
                out["blue"][aid] = pp
            for prefix in ("redmelee", "redranged", "bluemelee", "blueranged"):
                if aid.startswith(prefix + "_"):
                    out[prefix][aid] = pp
        return out

    try:
        raw = env.unwrapped
        handles = raw.env.get_handles()
        prefixes = ["redmelee", "redranged", "bluemelee", "blueranged"]
        for handle, prefix in zip(handles[:4], prefixes):
            ids = sorted([a for a in env.agents if a.startswith(prefix + "_")], key=agent_sort_key)
            for aid, p in zip(ids, raw.env.get_pos(handle)):
                pp = (int(p[0]), int(p[1]))
                out[prefix][aid] = pp
                out["red" if aid.startswith("red") else "blue"][aid] = pp
    except Exception:
        pass
    return out


def action_space_n(agent_id: str) -> int:
    return 9 if is_melee(agent_id) else 25


def is_red(agent_id: str) -> bool:
    return agent_id.startswith("red")


def is_blue(agent_id: str) -> bool:
    return agent_id.startswith("blue")


def is_melee(agent_id: str) -> bool:
    return "melee_" in agent_id


def is_ranged(agent_id: str) -> bool:
    return "ranged_" in agent_id


def attack_base(agent_id: str) -> int:
    return 5 if is_melee(agent_id) else 13


def attack_offsets(agent_id: str) -> list[tuple[int, int]]:
    return MELEE_ATTACK_OFFSETS if is_melee(agent_id) else RANGED_ATTACK_OFFSETS


def move_offsets(agent_id: str) -> list[tuple[int, int]]:
    return MELEE_MOVE_OFFSETS if is_melee(agent_id) else RANGED_MOVE_OFFSETS


def attack_specific(agent_id: str, my: tuple[int, int], target: tuple[int, int] | None) -> int | None:
    if target is None:
        return None
    dx, dy = int(target[0] - my[0]), int(target[1] - my[1])
    offsets = attack_offsets(agent_id)
    if (dx, dy) in offsets:
        return attack_base(agent_id) + offsets.index((dx, dy))
    return None


def attack_any(agent_id: str, my: tuple[int, int], enemies: list[tuple[int, int]]) -> int | None:
    for i, off in enumerate(attack_offsets(agent_id)):
        if (my[0] + off[0], my[1] + off[1]) in enemies:
            return attack_base(agent_id) + i
    return None


def attack_target(agent_id: str, my: tuple[int, int], enemies: list[tuple[str, tuple[int, int]]]) -> int | None:
    for _eid, ep in enemies:
        atk = attack_specific(agent_id, my, ep)
        if atk is not None:
            return atk
    return None


def move_toward(agent_id: str, my: tuple[int, int], target: tuple[int, int] | tuple[float, float] | None) -> int:
    if target is None:
        return 0
    dx = float(target[0]) - float(my[0])
    dy = float(target[1]) - float(my[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0
    sx = (dx > 0) - (dx < 0)
    sy = (dy > 0) - (dy < 0)
    best_idx, best_d = 0, math.inf
    for i, (mx, my_) in enumerate(move_offsets(agent_id)):
        if i == 0:
            continue
        d = (mx - sx) ** 2 + (my_ - sy) ** 2
        if d < best_d:
            best_idx, best_d = i, d
    return best_idx


def move_away(agent_id: str, my: tuple[int, int], from_: tuple[int, int] | tuple[float, float] | None) -> int:
    if from_ is None:
        return 0
    return move_toward(agent_id, my, (2 * my[0] - float(from_[0]), 2 * my[1] - float(from_[1])))


def nearest(pos: tuple[int, int], targets: list[tuple[int, int]] | list[tuple[str, tuple[int, int]]]):
    if not targets:
        return None, math.inf
    points = [x[1] if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) else x for x in targets]
    best = min(points, key=lambda q: (float(q[0]) - pos[0]) ** 2 + (float(q[1]) - pos[1]) ** 2)
    return best, dist(pos, best)


def centroid(points) -> tuple[float, float]:
    vals = list(points.values()) if isinstance(points, dict) else list(points)
    if not vals:
        return (0.0, 0.0)
    arr = np.asarray(vals, dtype=float)
    c = arr.mean(axis=0)
    return (float(c[0]), float(c[1]))


def dist(a, b) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def agent_sort_key(agent_id: str) -> tuple[int, int]:
    prefix_order = {"redmelee": 0, "redranged": 1, "bluemelee": 2, "blueranged": 3}
    prefix = agent_id.rsplit("_", 1)[0]
    try:
        idx = int(agent_id.rsplit("_", 1)[1])
    except Exception:
        idx = abs(hash(agent_id)) % (2**31)
    return (prefix_order.get(prefix, 99), idx)


def _blue_melee_rush(blue_ids, pos, red_melee, red_ranged):
    out = {}
    enemies = red_melee + red_ranged
    enemy_points = [p for _, p in enemies]
    for bid in blue_ids:
        p = pos["blue"].get(bid)
        if p is None:
            continue
        atk = attack_any(bid, p, enemy_points)
        if atk is not None:
            out[bid] = atk
        elif is_melee(bid):
            out[bid] = move_toward(bid, p, nearest(p, enemy_points)[0])
        else:
            tgt, d = nearest(p, enemy_points)
            out[bid] = move_away(bid, p, tgt) if d < 2.0 else 0
    return out


def _blue_standoff(blue_ids, pos, red_melee, red_ranged, advance: bool, focus: str):
    out = {}
    primary = red_ranged if focus == "ranged_first" else red_melee
    enemies = primary + (red_melee if focus == "ranged_first" else red_ranged)
    enemy_points = [p for _, p in enemies]
    red_ctr = centroid(pos["red"])
    ranged_ctr = centroid(pos["blueranged"])
    for bid in blue_ids:
        p = pos["blue"].get(bid)
        if p is None:
            continue
        atk = attack_target(bid, p, enemies)
        if atk is not None:
            out[bid] = atk
            continue
        tgt, d = nearest(p, enemy_points)
        if is_ranged(bid):
            if d < 2.0:
                out[bid] = move_away(bid, p, tgt)
            elif d > 2.0 and advance:
                out[bid] = move_toward(bid, p, tgt)
            else:
                out[bid] = 0
        else:
            screen = (ranged_ctr[0] + np.sign(red_ctr[0] - ranged_ctr[0]), ranged_ctr[1])
            out[bid] = attack_any(bid, p, enemy_points) or move_toward(bid, p, screen)
    return out


def _blue_mixed_advance(blue_ids, pos, red_melee, red_ranged):
    out = {}
    enemies = red_melee + red_ranged
    points = [p for _, p in enemies]
    melee_ctr = centroid(pos["bluemelee"])
    for bid in blue_ids:
        p = pos["blue"].get(bid)
        if p is None:
            continue
        atk = attack_any(bid, p, points)
        if atk is not None:
            out[bid] = atk
            continue
        tgt, d = nearest(p, points)
        if is_ranged(bid):
            if d > 2.0:
                out[bid] = move_toward(bid, p, tgt)
            elif d < 2.0:
                out[bid] = move_away(bid, p, tgt)
            else:
                out[bid] = move_toward(bid, p, melee_ctr)
        else:
            out[bid] = move_toward(bid, p, tgt)
    return out


def _blue_kite(blue_ids, pos, red_melee, red_ranged, step, state):
    out = {}
    enemies = red_melee + red_ranged
    points = [p for _, p in enemies]
    cooldown = state.setdefault("blue_kite_backstep", {})
    for bid in blue_ids:
        p = pos["blue"].get(bid)
        if p is None:
            continue
        tgt, d = nearest(p, points)
        if is_ranged(bid) and cooldown.pop(bid, False):
            out[bid] = move_away(bid, p, tgt)
            continue
        atk = attack_any(bid, p, points)
        if atk is not None:
            out[bid] = atk
            if is_ranged(bid):
                cooldown[bid] = True
        elif is_ranged(bid):
            out[bid] = move_toward(bid, p, tgt) if d > 2.0 else move_away(bid, p, tgt)
        else:
            out[bid] = 0 if d <= 1.5 else move_toward(bid, p, tgt)
    return out


def _blue_trait_screen(blue_ids, pos, red_melee, red_ranged, advance: bool, focus: str):
    out = {}
    primary = red_ranged if focus == "ranged_first" else red_melee
    enemies = primary + (red_melee if focus == "ranged_first" else red_ranged)
    points = [p for _, p in enemies]
    for bid in blue_ids:
        p = pos["blue"].get(bid)
        if p is None:
            continue
        atk = attack_target(bid, p, enemies)
        if atk is not None:
            out[bid] = atk
        elif is_ranged(bid):
            tgt, d = nearest(p, points)
            out[bid] = move_toward(bid, p, tgt) if advance and d > 2.0 else 0
        else:
            tgt, d = nearest(p, points)
            out[bid] = move_toward(bid, p, tgt) if advance or d > 1.5 else 0
    return out


def _all_attack(red_ids, pos, blue_melee, blue_ranged):
    out = {}
    enemies = blue_melee + blue_ranged
    points = [p for _, p in enemies]
    enemy_ctr = centroid(points)
    for rid in red_ids:
        p = pos["red"].get(rid)
        if p is None:
            continue
        atk = attack_any(rid, p, points)
        out[rid] = atk if atk is not None else move_toward(rid, p, enemy_ctr)
    return out


def _hold_line(red_ids, pos, blue_melee, blue_ranged):
    out = {}
    enemies = blue_melee + blue_ranged
    points = [p for _, p in enemies]
    red_ranged_ctr = centroid(pos["redranged"])
    blue_ctr = centroid(pos["blue"])
    for rid in red_ids:
        p = pos["red"].get(rid)
        if p is None:
            continue
        atk = attack_any(rid, p, points)
        if atk is not None:
            out[rid] = atk
        elif is_melee(rid):
            screen = (red_ranged_ctr[0] + np.sign(blue_ctr[0] - red_ranged_ctr[0]), red_ranged_ctr[1])
            out[rid] = move_toward(rid, p, screen)
        else:
            out[rid] = 0
    return out


def _priority_attack(red_ids, pos, primary, secondary):
    out = {}
    enemies = primary + secondary
    points = [p for _, p in enemies]
    for rid in red_ids:
        p = pos["red"].get(rid)
        if p is None:
            continue
        atk = attack_target(rid, p, enemies)
        out[rid] = atk if atk is not None else move_toward(rid, p, nearest(p, [p for _, p in primary] or points)[0])
    return out


def _screen_kite(red_ids, pos, blue_melee, blue_ranged, step, state):
    out = {}
    enemies = blue_melee + blue_ranged
    points = [p for _, p in enemies]
    backstep = state.setdefault("red_screen_kite_backstep", {})
    for rid in red_ids:
        p = pos["red"].get(rid)
        if p is None:
            continue
        tgt, d = nearest(p, points)
        atk = attack_any(rid, p, points)
        if is_ranged(rid):
            if backstep.pop(rid, False):
                out[rid] = move_away(rid, p, tgt)
            elif atk is not None:
                out[rid] = atk
                backstep[rid] = True
            else:
                out[rid] = move_toward(rid, p, tgt) if d > 2.0 else move_away(rid, p, tgt)
        else:
            out[rid] = atk if atk is not None else (0 if d <= 1.5 else move_toward(rid, p, tgt))
    return out


def _flank_ranged(red_ids, pos, blue_melee, blue_ranged, step):
    out = {}
    enemies = blue_melee + blue_ranged
    points = [p for _, p in enemies]
    blue_ctr = np.asarray(centroid(pos["blue"]), dtype=float)
    red_ctr = np.asarray(centroid(pos["red"]), dtype=float)
    red_ranged_ctr = centroid(pos["redranged"])
    direction = blue_ctr - red_ctr
    lateral = np.asarray([-direction[1], direction[0]], dtype=float)
    norm = float(np.linalg.norm(lateral))
    lateral = np.asarray([0.0, 1.0]) if norm == 0.0 else lateral / norm
    ranged = [rid for rid in red_ids if is_ranged(rid)]
    split = max(1, len(ranged) // 2)
    for rid in red_ids:
        p = pos["red"].get(rid)
        if p is None:
            continue
        priority = blue_ranged + blue_melee
        atk = attack_target(rid, p, priority)
        if atk is not None:
            out[rid] = atk
        elif is_ranged(rid):
            i = ranged.index(rid)
            side = -1.0 if i < split else 1.0
            if step % 3 < 2:
                out[rid] = move_toward(rid, p, (p[0], p[1] + 3.0 * side))
            else:
                out[rid] = move_toward(rid, p, tuple((blue_ctr + 3.0 * side * lateral).tolist()))
        else:
            screen = (red_ranged_ctr[0] + np.sign(blue_ctr[0] - red_ranged_ctr[0]), red_ranged_ctr[1])
            out[rid] = move_toward(rid, p, screen)
    return out
