"""Scripted BLUE tactics for the MAgent regime-switching harness."""

from __future__ import annotations

import hashlib
from itertools import product
from typing import Any

import numpy as np

from scripts.run_magent_sweep import (
    attack_action_if_enemy_adjacent,
    get_positions,
    move_away,
    move_toward,
    nearest,
)


BASE_TYPES = ["RUSH", "TURTLE", "KITE", "FLANK"]
TRAIT_TYPES = [
    f"{advance}-{formation}-{fire}"
    for advance, formation, fire in product(
        ["push", "hold"], ["tight", "spread"], ["focus", "spray"]
    )
]
TYPE_SETS = {"base4": BASE_TYPES, "traits8": TRAIT_TYPES}

TYPE_DOCS = {
    "RUSH": "Every unit attacks adjacent enemies, otherwise moves toward the nearest enemy.",
    "TURTLE": "Units hold a loose defensive line, re-form if displaced, and attack adjacent enemies without advancing.",
    "KITE": "Units hit adjacent enemies, then back away on the next step; otherwise they close only to short range.",
    "FLANK": "The team splits into two wings that move toward offset enemy centroids and attack when adjacent.",
}

TYPE_SEMANTICS = {
    "RUSH": "direct advance with immediate adjacent attacks",
    "TURTLE": "defensive holding with line repair and adjacent attacks only",
    "KITE": "hit-and-step-back pressure that avoids staying adjacent",
    "FLANK": "two-wing lateral movement toward offset enemy positions",
}

for _name in TRAIT_TYPES:
    _advance, _formation, _fire = _name.split("-")
    TYPE_DOCS[_name] = (
        f"{_advance} movement with {_formation} formation control and {_fire} adjacent-target selection."
    )
    TYPE_SEMANTICS[_name] = (
        f"{_advance} movement, {_formation} formation control, and {_fire} adjacent-target selection"
    )


def describe_type(name: str) -> str:
    """Return a one-sentence plain-language description for a BLUE type."""

    if name == "random":
        return "Units choose random legal low-level actions without a stable team tactic."
    if name not in TYPE_DOCS:
        raise ValueError(f"unknown BLUE type: {name}")
    return TYPE_DOCS[name]


def semantic_descriptor(name: str) -> str:
    """Return the behavioral concept text used by prompt encodings."""

    if name == "random":
        return "unstable random low-level movement without a persistent team pattern"
    if name not in TYPE_SEMANTICS:
        raise ValueError(f"unknown BLUE type: {name}")
    return TYPE_SEMANTICS[name]


def act_blue(env, agents: list[str], blue_type: str, rng, step: int, state: dict[str, Any]) -> dict[str, int]:
    """Return low-level actions for BLUE agents under a named scripted tactic."""

    if blue_type == "random":
        return {a: int(rng.integers(0, 21)) for a in agents if a.startswith("blue_")}

    red_pos, blue_pos = get_positions(env)
    blue_ids = sorted([a for a in agents if a.startswith("blue_")], key=_agent_sort_key)
    red_items = sorted(red_pos.items(), key=lambda kv: _agent_sort_key(kv[0]))
    enemies = [p for _, p in red_items]
    out: dict[str, int] = {}

    if blue_type in BASE_TYPES:
        if blue_type == "RUSH":
            return _rush(blue_ids, blue_pos, enemies)
        if blue_type == "TURTLE":
            return _turtle(blue_ids, blue_pos, enemies, state)
        if blue_type == "KITE":
            return _kite(blue_ids, blue_pos, enemies, step, state)
        if blue_type == "FLANK":
            return _flank(blue_ids, blue_pos, enemies)

    if blue_type in TRAIT_TYPES:
        advance, formation, fire = blue_type.split("-")
        if advance == "hold" and formation == "tight":
            return _turtle(blue_ids, blue_pos, enemies, state)
        if advance == "push" and formation == "spread":
            return _flank(blue_ids, blue_pos, enemies, fire=fire, red_items=red_items)
        focus_target = _focus_target(blue_ids, blue_pos, red_items) if fire == "focus" else None
        for bid in blue_ids:
            p = blue_pos.get(bid)
            if p is None:
                out[bid] = 0
                continue
            atk = _attack_specific(p, focus_target) if focus_target is not None else None
            if atk is None:
                atk = attack_action_if_enemy_adjacent(p, enemies)
            if atk is not None:
                out[bid] = atk
                continue
            if advance == "hold" and formation == "spread":
                tgt, dist = nearest(p, enemies)
                out[bid] = move_away(p, tgt) if tgt is not None and dist <= 2.0 else 0
                continue
            if formation == "tight" and step % 2 == 1 and len(blue_pos) > 1:
                out[bid] = move_toward(p, _centroid(list(blue_pos.values())))
            elif formation == "spread" and step % 2 == 1 and len(blue_pos) > 1:
                ally, _ = nearest(p, [q for q in blue_pos.values() if q != p])
                out[bid] = move_away(p, ally) if ally else 0
            elif advance == "push":
                tgt, _ = nearest(p, enemies)
                out[bid] = move_toward(p, tgt) if tgt else 0
            else:
                out[bid] = 0
        return out

    raise ValueError(f"unknown BLUE type: {blue_type}")


def episode_rng(seed: int) -> np.random.Generator:
    """Deterministic per-episode RNG used by scripted BLUE."""

    return np.random.default_rng(int(seed) * 1000 + 7)


def keyed_uniform(episode_seed: int, absolute_step: int, agent_id: str, purpose: str) -> float:
    """Stable U[0,1) variate keyed by exogenous event identity."""

    digest = hashlib.sha256(
        f"{int(episode_seed)}:{int(absolute_step)}:{agent_id}:{purpose}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") >> 11
    return value / float(1 << 53)


def keyed_int(episode_seed: int, absolute_step: int, agent_id: str, purpose: str, high: int) -> int:
    """Stable integer in [0, high) keyed by exogenous event identity."""

    if high <= 0:
        raise ValueError("high must be positive")
    digest = hashlib.sha256(
        f"{int(episode_seed)}:{int(absolute_step)}:{agent_id}:{purpose}".encode("utf-8")
    ).digest()
    return int(int.from_bytes(digest[:8], "big") % int(high))


def _rush(blue_ids, blue_pos, enemies):
    out = {}
    for bid in blue_ids:
        p = blue_pos.get(bid)
        if p is None:
            out[bid] = 0
            continue
        atk = attack_action_if_enemy_adjacent(p, enemies)
        if atk is not None:
            out[bid] = atk
        else:
            tgt, _ = nearest(p, enemies)
            out[bid] = move_toward(p, tgt) if tgt else 0
    return out


def _turtle(blue_ids, blue_pos, enemies, state):
    out = {}
    anchors = state.setdefault("turtle_anchor", {})
    for bid in blue_ids:
        p = blue_pos.get(bid)
        if p is None:
            out[bid] = 0
            continue
        anchors.setdefault(bid, p)
        atk = attack_action_if_enemy_adjacent(p, enemies)
        if atk is not None:
            out[bid] = atk
        elif _dist(p, anchors[bid]) > 1.5:
            out[bid] = move_toward(p, anchors[bid])
        else:
            out[bid] = 0
    return out


def _kite(blue_ids, blue_pos, enemies, step, state):
    out = {}
    cooldown = state.setdefault("kite_cooldown", {})
    for bid in blue_ids:
        p = blue_pos.get(bid)
        if p is None:
            out[bid] = 0
            continue
        if cooldown.pop(bid, False):
            tgt, _ = nearest(p, enemies)
            out[bid] = move_away(p, tgt) if tgt else 0
            continue
        atk = attack_action_if_enemy_adjacent(p, enemies)
        if atk is not None:
            out[bid] = atk
            cooldown[bid] = True
            continue
        tgt, dist = nearest(p, enemies)
        out[bid] = move_toward(p, tgt) if tgt and dist > 2.0 else 0
    return out


def _flank(blue_ids, blue_pos, enemies, fire="spray", red_items=None):
    out = {}
    if not enemies:
        return {bid: 0 for bid in blue_ids}
    focus_target = _focus_target(blue_ids, blue_pos, red_items or []) if fire == "focus" else None
    enemy_ctr = np.asarray(_centroid(enemies), dtype=float)
    own_ctr = np.asarray(_centroid(list(blue_pos.values())), dtype=float)
    direction = enemy_ctr - own_ctr
    lateral = np.asarray([-direction[1], direction[0]], dtype=float)
    norm = float(np.linalg.norm(lateral))
    if norm == 0.0:
        lateral = np.asarray([0.0, 1.0])
    else:
        lateral = lateral / norm
    split = max(1, len(blue_ids) // 2)
    for i, bid in enumerate(blue_ids):
        p = blue_pos.get(bid)
        if p is None:
            out[bid] = 0
            continue
        atk = _attack_specific(p, focus_target) if focus_target is not None else None
        if atk is None:
            atk = attack_action_if_enemy_adjacent(p, enemies)
        if atk is not None:
            out[bid] = atk
            continue
        side = -1.0 if i < split else 1.0
        target = tuple((enemy_ctr + side * 3.0 * lateral).tolist())
        out[bid] = move_toward(p, target)
    return out


def _focus_target(blue_ids, blue_pos, red_items):
    for rid, pos in red_items:
        for bid in blue_ids:
            p = blue_pos.get(bid)
            if p is not None and max(abs(pos[0] - p[0]), abs(pos[1] - p[1])) <= 1:
                return pos
    return None


def _attack_specific(my, target):
    if target is None:
        return None
    dx, dy = int(target[0] - my[0]), int(target[1] - my[1])
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if (dx, dy) in offsets:
        return 13 + offsets.index((dx, dy))
    return None


def _centroid(points):
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    ctr = arr.mean(axis=0)
    return (float(ctr[0]), float(ctr[1]))


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _agent_sort_key(agent_id: str) -> int:
    try:
        return int(agent_id.split("_")[-1])
    except Exception:
        return abs(hash(agent_id)) % (2**31)
