"""Shared utilities for the Overcooked replay-selection harness."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # overcooked-ai 1.1.0 imports np.Inf on NumPy 2.x

ROOT = Path(__file__).resolve().parents[2]
V8_ROOT = ROOT.parent.parent / "v8_lib"
for path in (ROOT, V8_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.envs.overcooked.wrapper import (  # noqa: E402
    OvercookedConfig,
    OvercookedEnvWrapper,
    OvercookedState,
)
from scripts.run_overcooked_sweep import (  # noqa: E402
    OvercookedActionMapper,
    scripted_pick_task,
)

TASKS = {
    "get_onion",
    "put_onion",
    "cook",
    "get_dish",
    "plate_soup",
    "serve_soup",
    "drop_held_item",
    "stay",
}


class RSOvercookedActionMapper(OvercookedActionMapper):
    """Mapper extensions for the replay-selection harness that preserve the get_action API."""

    def _object_action(self, idx: int, target: tuple[int, int]) -> int:
        pos = self._get_pos(idx)
        if target in self._neighbors(pos):
            return self._interact_or_turn(idx, target)
        return self._move_to(idx, target)

    def _nearest_reachable_object(self, idx: int, kind: str) -> tuple[int, int] | None:
        pos = self._get_pos(idx)
        candidates = []
        for target in object_positions(self.env, kind):
            path = self._bfs(pos, lambda p, t=target: p in self._neighbors(t))
            if path:
                candidates.append((target, len(path)))
        if not candidates:
            return None
        return sorted(candidates, key=lambda x: (x[1], x[0]))[0][0]

    def _soup_state_by_pot(self) -> dict[tuple[int, int], dict[str, Any]]:
        out = {
            tuple(p): {"onions": 0, "cooking": False, "ready": False, "exists": False}
            for p in self.mdp.get_pot_locations()
        }
        state = self.env._current_state
        if not state:
            return out
        for obj in state.objects.get("soups", []):
            loc = tuple(obj["position"])
            text = str(obj.get("state", ""))
            out[loc] = {
                "onions": text.count("'onion'"),
                "cooking": "cooking:True" in text,
                "ready": "ready:True" in text,
                "exists": True,
            }
        return out

    def _nearest_reachable_pot(self, idx: int, predicate) -> tuple[int, int] | None:
        pos = self._get_pos(idx)
        states = self._soup_state_by_pot()
        candidates = []
        for pot, state in states.items():
            if not predicate(state):
                continue
            path = self._bfs(pos, lambda p, t=pot: p in self._neighbors(t))
            if path:
                candidates.append((pot, len(path)))
        if not candidates:
            return None
        return sorted(candidates, key=lambda x: (x[1], x[0]))[0][0]

    def get_action(self, idx: int, task: str, use_handover: bool = True) -> int:
        held = held_item(self, idx)

        if task == "serve_soup" and held == "empty":
            soup = self._nearest_reachable_object(idx, "soup")
            if soup is not None:
                return self._object_action(idx, soup)

        if task == "drop_held_item" and held != "empty":
            target = nearest_shared_counter_to_pot(self, require_empty=True)
            if target is not None:
                return self._object_action(idx, target)

        if task == "put_onion" and held == "onion":
            target = self._nearest_reachable_pot(
                idx, lambda s: not s["ready"] and not s["cooking"] and s["onions"] < 3
            )
            if target is not None:
                return self._object_action(idx, target)

        if task == "cook":
            target = self._nearest_reachable_pot(
                idx, lambda s: s["onions"] >= 3 and not s["ready"] and not s["cooking"]
            )
            if target is not None:
                return self._object_action(idx, target)

        if task == "plate_soup" and held == "dish":
            target = self._nearest_reachable_pot(idx, lambda s: s["ready"])
            if target is not None:
                return self._object_action(idx, target)

        return super().get_action(idx, task, use_handover=use_handover)


def make_env(layout: str, horizon: int) -> tuple[OvercookedEnvWrapper, RSOvercookedActionMapper]:
    patch_motion_planner_cache()
    env = OvercookedEnvWrapper(OvercookedConfig(layout_name=layout, horizon=horizon))
    env.reset()
    return env, RSOvercookedActionMapper(env)


def patch_motion_planner_cache() -> None:
    """Avoid writes to read-only overcooked_ai_py package planner cache."""

    try:
        from overcooked_ai_py.planning.planners import MotionPlanner
    except Exception:
        return
    if getattr(MotionPlanner, "_wp3_no_save_patch", False):
        return

    def compute_mp_no_save(filename, mdp, counter_goals):
        del filename
        return MotionPlanner(mdp, counter_goals)

    MotionPlanner.compute_mp = staticmethod(compute_mp_no_save)
    MotionPlanner._wp3_no_save_patch = True


def pot_state(env_wrap: OvercookedEnvWrapper, mapper: OvercookedActionMapper) -> dict[str, Any]:
    state = env_wrap._current_state
    pots = list(mapper.mdp.get_pot_locations())
    target = pots[0] if pots else None
    out = {"loc": target, "onions": 0, "cooking": False, "ready": False, "exists": False}
    if not state or target is None:
        return out
    for obj in state.objects.get("soups", []):
        if tuple(obj.get("position")) == tuple(target):
            text = str(obj.get("state", ""))
            out.update(
                {
                    "onions": text.count("'onion'"),
                    "cooking": "cooking:True" in text,
                    "ready": "ready:True" in text,
                    "exists": True,
                }
            )
            break
    return out


def held_item(mapper: OvercookedActionMapper, idx: int) -> str:
    held = mapper._get_held(idx)
    if not held:
        return "empty"
    low = str(held).lower()
    if "soup" in low:
        return "soup"
    if "dish" in low:
        return "dish"
    if "onion" in low:
        return "onion"
    return low


def occupied_positions(mapper: OvercookedActionMapper) -> set[tuple[int, int]]:
    return mapper._occupied_obj_pos()


def object_positions(env_wrap: OvercookedEnvWrapper, kind: str) -> list[tuple[int, int]]:
    state = env_wrap._current_state
    if not state:
        return []
    key = {"onion": "onions", "dish": "dishes", "soup": "soups"}.get(kind, kind)
    return [tuple(o["position"]) for o in state.objects.get(key, [])]


def reachable_object_positions(
    env_wrap: OvercookedEnvWrapper,
    mapper: OvercookedActionMapper,
    idx: int,
    kind: str,
) -> list[tuple[int, int]]:
    del env_wrap
    pos = mapper._get_pos(idx)
    out: list[tuple[int, int]] = []
    for p in object_positions(mapper.env, kind):
        if mapper._bfs(pos, lambda x, p=p: x in mapper._neighbors(p)):
            out.append(p)
    return sorted(out, key=lambda p: manhattan(pos, p))


def nearest_shared_counter_to_pot(
    mapper: OvercookedActionMapper, require_empty: bool = True
) -> tuple[int, int] | None:
    shared = mapper._shared_counters()
    if require_empty:
        occupied = occupied_positions(mapper)
        shared = [c for c in shared if c not in occupied]
    pots = list(mapper.mdp.get_pot_locations())
    anchor = pots[0] if pots else mapper._get_pos(0)
    if not shared:
        return None
    return sorted(shared, key=lambda c: (manhattan(c, anchor), c))[0]


def can_reach_facility(mapper: OvercookedActionMapper, idx: int, targets: list[tuple[int, int]]) -> bool:
    pos = mapper._get_pos(idx)
    return any(mapper._bfs(pos, lambda p, t=t: p in mapper._neighbors(t)) for t in targets)


def nearest_facility_distance(mapper: OvercookedActionMapper, idx: int, targets: list[tuple[int, int]]) -> float:
    pos = mapper._get_pos(idx)
    best = math.inf
    for target in targets:
        path = mapper._bfs(pos, lambda p, t=target: p in mapper._neighbors(t))
        if path:
            best = min(best, len(path) - 1)
    return float(best if math.isfinite(best) else 999.0)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def i1_held_task(mapper: OvercookedActionMapper, idx: int, soup_task: str = "serve_soup") -> str | None:
    held = held_item(mapper, idx)
    if held == "soup":
        return soup_task
    if held == "dish":
        return "plate_soup"
    if held == "onion":
        return "put_onion"
    return None


def urgent_i2_task(env_wrap: OvercookedEnvWrapper, mapper: OvercookedActionMapper) -> str | None:
    ps = pot_state(env_wrap, mapper)
    if ps["ready"] or ps["cooking"]:
        return "get_dish"
    if ps["onions"] >= 3 and not ps["cooking"]:
        return "cook"
    return None


def sync_wrapper_state(env_wrap: OvercookedEnvWrapper) -> None:
    if env_wrap._env is not None and getattr(env_wrap._env, "state", None) is not None:
        old_score = env_wrap._current_state.score if env_wrap._current_state else 0.0
        env_wrap._current_state = OvercookedState.from_overcooked_state(
            env_wrap._env.state, env_wrap.oc_config.layout_name
        )
        env_wrap._current_state.timestep = getattr(env_wrap._env.state, "timestep", env_wrap._step_count)
        env_wrap._current_state.score = old_score


def clone_env_wrapper(env_wrap: OvercookedEnvWrapper) -> OvercookedEnvWrapper:
    clone = OvercookedEnvWrapper(
        OvercookedConfig(layout_name=env_wrap.oc_config.layout_name, horizon=env_wrap.oc_config.horizon)
    )
    clone._env = env_wrap._env.copy()
    clone._base_env = clone._env
    clone._mdp = env_wrap._mdp
    clone._step_count = env_wrap._step_count
    clone._episode_count = env_wrap._episode_count
    clone._current_state = OvercookedState.from_overcooked_state(
        clone._env.state, clone.oc_config.layout_name
    )
    clone._current_state.timestep = env_wrap._current_state.timestep if env_wrap._current_state else 0
    clone._current_state.score = env_wrap._current_state.score if env_wrap._current_state else 0.0
    return clone


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
