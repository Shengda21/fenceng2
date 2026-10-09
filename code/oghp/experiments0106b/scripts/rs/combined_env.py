"""MAgent combined_arms regime-switching runner with a deterministic fake adapter."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from v8lib.context import Context, redact_hidden_context
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "v8_lib"))
    from v8lib.context import Context, redact_hidden_context

from scripts.rs.combined_types import (
    MELEE_ATTACK_OFFSETS,
    MELEE_MOVE_OFFSETS,
    POOL_SETS,
    RANGED_ATTACK_OFFSETS,
    RANGED_MOVE_OFFSETS,
    TYPE_SETS,
    act_blue,
    act_red_strategy,
    agent_sort_key,
    attack_base,
    attack_offsets,
    centroid,
    describe_type,
    dist,
    episode_rng,
    get_combined_positions,
    is_blue,
    is_melee,
    is_ranged,
    is_red,
    keyed_int,
    keyed_uniform,
    move_offsets,
    semantic_descriptor,
)

FEATURE_NAMES = [
    "blue_centroid_velocity_toward_red",
    "blue_melee_advance_fraction",
    "blue_ranged_fire_fraction",
    "blue_ranged_retreat_fraction",
    "blue_melee_screen_fraction",
    "engagement_distance",
    "blue_melee_mean_distance",
    "blue_ranged_mean_distance",
    "blue_alive",
    "red_alive",
    "step",
]


def feature_fn(raw: dict) -> np.ndarray:
    events = raw.get("events", [])
    start_blue = raw.get("start_blue_pos", {})
    cur_blue = raw.get("blue_pos", {})
    cur_red = raw.get("red_pos", {})
    start_red = raw.get("start_red_pos", {})
    step = int(raw.get("step", 0))
    n_steps = max(1, int(raw.get("window_steps", step)))
    velocity = (_centroid_gap(start_red, start_blue) - _centroid_gap(cur_red, cur_blue)) / n_steps

    denom_melee = sum(max(1, int(e.get("blue_melee_count", 0))) for e in events) or 1
    denom_ranged = sum(max(1, int(e.get("blue_ranged_count", 0))) for e in events) or 1
    melee_advance = sum(int(e.get("blue_melee_advanced", 0)) for e in events) / denom_melee
    ranged_fire = sum(int(e.get("blue_ranged_fired", 0)) for e in events) / denom_ranged
    ranged_retreat = sum(int(e.get("blue_ranged_retreated", 0)) for e in events) / denom_ranged
    melee_screen = sum(int(e.get("blue_melee_screened", 0)) for e in events) / denom_melee

    melee_dist = _mean_distance(raw.get("blue_melee_pos", {}), cur_red)
    ranged_dist = _mean_distance(raw.get("blue_ranged_pos", {}), cur_red)
    return np.asarray(
        [
            velocity,
            melee_advance,
            ranged_fire,
            ranged_retreat,
            melee_screen,
            _centroid_gap(cur_red, cur_blue),
            melee_dist,
            ranged_dist,
            float(len(cur_blue)),
            float(len(cur_red)),
            float(step),
        ],
        dtype=float,
    )


def semantic_template(raw: dict, concept_mapping: dict[str, str] | None = None) -> str:
    # Observation text is derived from measured features only. The hidden type's descriptor must never appear
    # here (that would hand the label to the selector); type descriptions live in the separate hint list, and
    # the `permuted` intervention permutes that hint dictionary (concept_mapping is kept for API compatibility).
    vals = dict(zip(FEATURE_NAMES, feature_fn(raw)))
    pressure = "closing on our line" if vals["blue_centroid_velocity_toward_red"] > 0.1 else "holding its distance"
    melee_adv = vals["blue_melee_advance_fraction"]
    melee = "most of its melee units have been advancing" if melee_adv > 0.5 else (
        "some of its melee units have been advancing" if melee_adv > 0.15 else "its melee units have barely moved forward")
    fire = "heavy ranged fire" if vals["blue_ranged_fire_fraction"] > 0.35 else "little ranged fire"
    kite = "and its ranged units step back often after firing" if vals["blue_ranged_retreat_fraction"] > 0.25 else "and its ranged units rarely step back"
    return (
        f"Over the first {int(vals.get('step', 0)) or 'observed'} steps the enemy has been {pressure}; {melee} "
        f"(advance fraction {melee_adv:.2f}); we have seen {fire} {kite}. "
        f"Centroid distance is {vals['engagement_distance']:.1f}; "
        f"enemy melee/ranged mean distances are {vals['blue_melee_mean_distance']:.1f}/{vals['blue_ranged_mean_distance']:.1f}."
    )


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    type1: str
    type2: str | None = None
    switch_step: int | None = None

    @property
    def key(self) -> str:
        return f"{self.seed}:{self.type1}:{self.type2 or ''}:{self.switch_step if self.switch_step is not None else ''}"


class CombinedArmsRS:
    """Decision-window environment wrapper for combined_arms_v6."""

    def __init__(
        self,
        map_size: int = 16,
        max_cycles: int = 120,
        n_melee: int = 6,
        n_ranged: int = 6,
        type_set: str | list[str] | tuple[str, ...] = "base4",
        pool: str | list[str] | tuple[str, ...] = "combined_pool6",
        default: str = "HOLD_LINE",
        k: int = 20,
        switch: str | None = None,
        delta_scale: float = 1.0,
        held_out: str | None = None,
        env_factory: Callable[[], Any] | None = None,
        force_fake: bool = False,
        forced_type: str | None = None,
        forced_type2: str | None = None,
        switch_step: int | None = None,
        encoder: Callable[[dict, Context], Context] | None = None,
    ) -> None:
        if switch == "single":
            switch = None
        if switch not in {None, "mid", "block20"}:
            raise ValueError("switch must be one of None, 'mid', or 'block20'")
        self.map_size = int(map_size)
        self.max_cycles = int(max_cycles)
        self.n_melee = int(n_melee)
        self.n_ranged = int(n_ranged)
        self.type_names = _resolve_type_set(type_set)
        self.pool = _resolve_pool(pool)
        self.default = default
        self.k = int(k)
        self.switch = switch
        self.delta_scale = float(delta_scale)
        self.held_out = held_out
        self.env_factory = env_factory
        self.force_fake = bool(force_fake)
        self.forced_type = forced_type
        self.forced_type2 = forced_type2
        self.switch_step = switch_step
        self._episode_switch_step = switch_step
        self.encoder = encoder
        self.env = None
        self.seed = 0
        self.episode_index = 0
        self.step = 0
        self.window_index = 0
        self.blue_type: str | None = None
        self.blue_type2: str | None = None
        self._blue_state: dict[str, Any] = {}
        self._red_state: dict[str, Any] = {}
        self._blue_rng = episode_rng(0)
        self._done = False
        self._prefix_reward = 0.0
        self._total_reward = 0.0
        self._window_start_step = 0
        self._obs_start_red: dict[str, tuple[int, int]] = {}
        self._obs_start_blue: dict[str, tuple[int, int]] = {}
        self._events: list[dict[str, int]] = []
        self.last_raw: dict[str, Any] | None = None
        self.action_trace: list[dict[str, Any]] = []
        self.rng_trace: list[dict[str, Any]] = []

    def set_encoder(self, encoder: Callable[[dict, Context], Context] | None) -> None:
        self.encoder = encoder

    def reset(self, seed: int, episode_index: int | None = None, episode_spec: EpisodeSpec | dict[str, Any] | None = None) -> Context:
        if episode_spec is not None:
            spec = _coerce_episode_spec(episode_spec)
            seed = spec.seed
            old_forced_type, old_forced_type2, old_switch_step = self.forced_type, self.forced_type2, self.switch_step
            self.forced_type = spec.type1
            self.forced_type2 = spec.type2
            self.switch_step = spec.switch_step
        else:
            old_forced_type = old_forced_type2 = old_switch_step = None
        self.seed = int(seed)
        self.episode_index = int(seed if episode_index is None else episode_index)
        self.step = 0
        self.window_index = 0
        self._done = False
        self._prefix_reward = 0.0
        self._total_reward = 0.0
        self.action_trace = []
        self.rng_trace = []
        self._episode_switch_step = self.switch_step
        self._blue_state = {}
        self._red_state = {}
        self._blue_rng = episode_rng(seed)
        self.blue_type, self.blue_type2 = self._sample_types(seed)
        self.env = self._make_env()
        self.env.reset(seed=seed)
        self._begin_observation_window()
        while self.step < min(self.k, self.max_cycles) and not self._done:
            self._prefix_reward += self._step(self.default)
        self._total_reward += self._prefix_reward
        ctx = self._make_context()
        self._window_start_step = self.step
        self._begin_observation_window()
        if episode_spec is not None:
            self.forced_type, self.forced_type2, self.switch_step = old_forced_type, old_forced_type2, old_switch_step
        return ctx

    def decide_steps(self) -> list[int]:
        steps = [self.k]
        if self.switch == "mid":
            steps.append(self.k + 40)
        return steps

    def step_window(self, tau: str) -> tuple[Context | None, float, bool, dict]:
        if self._done or self.step >= self.max_cycles:
            self._done = True
            return None, 0.0, True, {"step_start": self.step, "step_end": self.step}
        start = self.step
        next_decision = self._next_decision_after_current()
        reward = 0.0
        while not self._done and self.step < self.max_cycles:
            if next_decision is not None and self.step >= next_decision:
                break
            reward += self._step(tau)
        self._total_reward += reward
        done = self._done or self.step >= self.max_cycles
        self._done = done
        info = {
            "step_start": start,
            "step_end": self.step,
            "type": self.blue_type,
            "type2": self.blue_type2,
            "state_summary": self.state_summary(),
        }
        if done:
            return None, float(reward), True, info
        self.window_index += 1
        ctx = self._make_context()
        self._window_start_step = self.step
        self._begin_observation_window()
        return ctx, float(reward), False, info

    def replay(self, seed: int, decide_fn: Callable[..., str]) -> dict[str, Any]:
        ctx = self.reset(seed)
        total = float(self._prefix_reward)
        decisions: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        done = self._done
        while not done:
            tau = _call_decide(decide_fn, ctx, self)
            fallback = tau not in self.pool
            if fallback:
                tau = self.default
            decisions.append({"step": self.step, "tau": tau, "fallback": fallback})
            prov = getattr(decide_fn, "last_provenance", None)
            if isinstance(prov, dict):
                provenance.append(dict(prov))
            ctx, reward, done, info = self.step_window(tau)
            total += float(reward)
        pos = get_combined_positions(self.env)
        return {
            "episode": int(seed),
            "seed": int(seed),
            "type": self.blue_type,
            "type2": self.blue_type2,
            "decisions": decisions,
            "total_reward": float(total),
            "success": bool(total > 0),
            "steps": int(self.step),
            "red_alive_final": len(pos["red"]),
            "blue_alive_final": len(pos["blue"]),
            "provenance": provenance,
        }

    def clone(self) -> "CombinedArmsRS":
        if not isinstance(self.env, FakeParallelCombinedArms):
            raise RuntimeError("CombinedArmsRS.clone is only supported by the fake adapter")
        return copy.deepcopy(self)

    def state_summary(self) -> dict[str, Any]:
        pos = get_combined_positions(self.env)
        return {
            "step": self.step,
            "red_alive": len(pos["red"]),
            "blue_alive": len(pos["blue"]),
            "red_melee_alive": len(pos["redmelee"]),
            "red_ranged_alive": len(pos["redranged"]),
            "blue_melee_alive": len(pos["bluemelee"]),
            "blue_ranged_alive": len(pos["blueranged"]),
            "total_reward": self._total_reward,
        }

    def _make_env(self):
        if self.env_factory is not None:
            return self.env_factory()
        if not self.force_fake:
            try:
                from magent2.environments import combined_arms_v6

                env = combined_arms_v6.parallel_env(
                    map_size=self.map_size,
                    max_cycles=self.max_cycles,
                    minimap_mode=False,
                    extra_features=False,
                )
                _install_fixed_combined_map(env, self.n_melee, self.n_ranged)
                return env
            except Exception:
                pass
        return FakeParallelCombinedArms(
            map_size=max(16, self.map_size),
            max_cycles=self.max_cycles,
            n_melee=self.n_melee,
            n_ranged=self.n_ranged,
        )

    def _sample_types(self, seed: int) -> tuple[str, str | None]:
        if self.forced_type is not None:
            first = self.forced_type
        else:
            first = _sample_type_for_seed(self.type_names, seed, self.switch, self.held_out)
        second = None
        if self.switch == "mid":
            if self.forced_type2 is not None:
                second = self.forced_type2
            else:
                choices = _eligible_types(self.type_names, _held_out_for_seed(seed, self.held_out))
                rng = np.random.default_rng(seed)
                _ = rng.choice(choices)
                second = str(rng.choice(choices))
        return first, second

    def _active_blue_type(self) -> str:
        switch_step = 50 if self._episode_switch_step is None else int(self._episode_switch_step)
        if self.switch == "mid" and self.blue_type2 is not None and self.step >= switch_step:
            return self.blue_type2
        return str(self.blue_type)

    def _next_decision_after_current(self) -> int | None:
        for decision_step in self.decide_steps():
            if decision_step > self.step:
                return decision_step
        return None

    def _begin_observation_window(self) -> None:
        pos = get_combined_positions(self.env)
        self._obs_start_red = dict(pos["red"])
        self._obs_start_blue = dict(pos["blue"])
        self._events = []

    def _step(self, tau: str) -> float:
        agents = list(getattr(self.env, "agents", []))
        red_ids = [a for a in agents if is_red(a)]
        blue_ids = [a for a in agents if is_blue(a)]
        before = get_combined_positions(self.env)
        chosen = act_red_strategy(self.env, agents, tau, self.step, self._red_state)
        hold = act_red_strategy(self.env, agents, "HOLD_LINE", self.step, self._red_state)
        actions: dict[str, int] = {}
        red_gates = {}
        for rid in red_ids:
            gate = keyed_uniform(self.seed, self.step, rid, "red_delta_gate")
            self.rng_trace.append(
                {"seed": self.seed, "step": self.step, "agent_id": rid, "purpose": "red_delta_gate", "value": float(gate)}
            )
            red_gates[rid] = float(gate)
            actions[rid] = int(chosen.get(rid, 0) if gate <= self.delta_scale else hold.get(rid, 0))
        active_blue = self._active_blue_type()
        if active_blue == "random":
            blue_actions = {}
            for bid in blue_ids:
                val = keyed_int(self.seed, self.step, bid, "blue_random_action", 9 if is_melee(bid) else 25)
                self.rng_trace.append(
                    {"seed": self.seed, "step": self.step, "agent_id": bid, "purpose": "blue_random_action", "value": int(val)}
                )
                blue_actions[bid] = int(val)
        else:
            blue_actions = act_blue(self.env, blue_ids, active_blue, self._blue_rng, self.step, self._blue_state)
        actions.update(blue_actions)
        self.action_trace.append(
            {
                "seed": self.seed,
                "step": self.step,
                "tau": tau,
                "active_type": active_blue,
                "actions": {k: int(v) for k, v in sorted(actions.items(), key=lambda kv: agent_sort_key(kv[0]))},
                "red_gates": red_gates,
            }
        )
        _, rewards, _, _, _ = self.env.step(actions)
        after = get_combined_positions(self.env)
        self._events.append(_blue_event(before, after, actions))
        self.step += 1
        if not getattr(self.env, "agents", []) or self.step >= self.max_cycles:
            self._done = True
        return float(sum(float(r) for k, r in rewards.items() if is_red(k)))

    def _make_context(self) -> Context:
        raw = self._raw_state()
        self.last_raw = raw
        ctx = Context(
            pool=list(self.pool),
            features=feature_fn(raw),
            text=semantic_template(raw),
            episode_index=self.episode_index,
            window_index=self.window_index,
            default=self.default,
            type_label=self._active_blue_type(),
            extra={
                "seed": self.seed,
                "step": self.step,
                "type": self.blue_type,
                "type2": self.blue_type2,
                "feature_names": list(FEATURE_NAMES),
                "raw": raw,
            },
        )
        if self.encoder is not None:
            return self.encoder(raw, ctx)
        return ctx

    def _raw_state(self) -> dict[str, Any]:
        pos = get_combined_positions(self.env)
        return {
            "seed": self.seed,
            "step": self.step,
            "window_index": self.window_index,
            "type": self._active_blue_type(),
            "type_label": self._active_blue_type(),
            "type1": self.blue_type,
            "type2": self.blue_type2,
            "switch_step": self._episode_switch_step if self.switch == "mid" else None,
            "red_pos": dict(pos["red"]),
            "blue_pos": dict(pos["blue"]),
            "red_melee_pos": dict(pos["redmelee"]),
            "red_ranged_pos": dict(pos["redranged"]),
            "blue_melee_pos": dict(pos["bluemelee"]),
            "blue_ranged_pos": dict(pos["blueranged"]),
            "start_red_pos": dict(self._obs_start_red),
            "start_blue_pos": dict(self._obs_start_blue),
            "events": list(self._events),
            "window_steps": max(1, self.step - self._window_start_step),
        }


class FakeParallelCombinedArms:
    """Small deterministic in-memory subset of the combined_arms_v6 parallel API."""

    def __init__(self, map_size: int = 16, max_cycles: int = 120, n_melee: int = 6, n_ranged: int = 6) -> None:
        if map_size < 16:
            raise AssertionError("size of map must be at least 16")
        self.map_size = int(map_size)
        self.max_cycles = int(max_cycles)
        self.n_melee = int(n_melee)
        self.n_ranged = int(n_ranged)
        self.unwrapped = self
        self.env = _FakeCombinedCore(self)
        self.agents: list[str] = []
        self.positions: dict[str, tuple[int, int]] = {}
        self.hp: dict[str, float] = {}
        self.step_count = 0

    def reset(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        centre = self.map_size // 2
        melee_offsets = np.arange(self.n_melee) - (self.n_melee - 1) / 2
        ranged_offsets = np.arange(self.n_ranged) - (self.n_ranged - 1) / 2
        self.positions = {}
        self.hp = {}
        for i, off in enumerate(melee_offsets):
            jitter = int(rng.integers(-1, 2))
            self.positions[f"bluemelee_{i}"] = (3 + jitter, int(round(centre + off)))
            self.positions[f"redmelee_{i}"] = (self.map_size - 4 - jitter, int(round(centre + off)))
            self.hp[f"bluemelee_{i}"] = 10.0
            self.hp[f"redmelee_{i}"] = 10.0
        for i, off in enumerate(ranged_offsets):
            jitter = int(rng.integers(-1, 2))
            self.positions[f"blueranged_{i}"] = (2 + jitter, int(round(centre + off)))
            self.positions[f"redranged_{i}"] = (self.map_size - 3 - jitter, int(round(centre + off)))
            self.hp[f"blueranged_{i}"] = 3.0
            self.hp[f"redranged_{i}"] = 3.0
        self.step_count = 0
        self._refresh_agents()
        return {a: np.zeros(1, dtype=float) for a in self.agents}

    def step(self, actions: dict[str, int]):
        live_before = list(self.agents)
        rewards = {a: -0.005 for a in live_before}
        killed: set[str] = set()
        for actor in sorted(live_before, key=agent_sort_key):
            if actor in killed or actor not in self.positions:
                continue
            action = int(actions.get(actor, 0))
            if action >= attack_base(actor):
                rewards[actor] = rewards.get(actor, 0.0) - 0.1
                target = self._attack_target(actor, action)
                if target is not None and target not in killed:
                    if actor_team(actor) != actor_team(target):
                        rewards[actor] = rewards.get(actor, 0.0) + 0.2
                    self.hp[target] -= 2.0
                    if self.hp[target] <= 0:
                        killed.add(target)
                        if actor_team(actor) != actor_team(target):
                            rewards[actor] = rewards.get(actor, 0.0) + 5.0
                        rewards[target] = rewards.get(target, 0.0) - 0.1
        for dead in killed:
            self.positions.pop(dead, None)
            self.hp.pop(dead, None)
        occupied = dict(self.positions)
        for actor in sorted(live_before, key=agent_sort_key):
            if actor not in self.positions:
                continue
            action = int(actions.get(actor, 0))
            offsets = move_offsets(actor)
            if action <= 0 or action >= len(offsets):
                continue
            dx, dy = offsets[action]
            x, y = self.positions[actor]
            nxt = (int(np.clip(x + dx, 1, self.map_size - 2)), int(np.clip(y + dy, 1, self.map_size - 2)))
            if nxt not in occupied.values():
                self.positions[actor] = nxt
                occupied[actor] = nxt
        for aid in list(self.hp):
            cap = 10.0 if is_melee(aid) else 3.0
            self.hp[aid] = min(cap, self.hp[aid] + 0.1)
        self.step_count += 1
        self._refresh_agents()
        done = (
            self.step_count >= self.max_cycles
            or not any(is_red(a) for a in self.agents)
            or not any(is_blue(a) for a in self.agents)
        )
        if done:
            terminations = {a: True for a in live_before}
            truncations = {a: self.step_count >= self.max_cycles for a in live_before}
            self.agents = []
        else:
            terminations = {a: False for a in self.agents}
            truncations = {a: False for a in self.agents}
        obs = {a: np.zeros(1, dtype=float) for a in self.agents}
        infos = {a: {} for a in live_before}
        return obs, rewards, terminations, truncations, infos

    def _attack_target(self, actor: str, action: int) -> str | None:
        idx = int(action) - attack_base(actor)
        offsets = attack_offsets(actor)
        if idx < 0 or idx >= len(offsets):
            return None
        dx, dy = offsets[idx]
        x, y = self.positions[actor]
        target_pos = (x + dx, y + dy)
        for agent_id, pos in self.positions.items():
            if pos == target_pos:
                return agent_id
        return None

    def _refresh_agents(self) -> None:
        self.agents = sorted(self.positions, key=agent_sort_key)


class _FakeCombinedCore:
    def __init__(self, outer: FakeParallelCombinedArms) -> None:
        self.outer = outer

    def get_handles(self):
        return ["redmelee", "redranged", "bluemelee", "blueranged"]

    def get_pos(self, handle):
        prefix = str(handle)
        vals = [self.outer.positions[a] for a in sorted(self.outer.positions, key=agent_sort_key) if a.startswith(prefix + "_")]
        return np.asarray(vals, dtype=int)

    def get_view_space(self, handle):
        return ((self.outer.map_size, self.outer.map_size, 1),)


def actor_team(agent_id: str) -> str:
    return "red" if is_red(agent_id) else "blue"


def _install_fixed_combined_map(env, n_melee: int, n_ranged: int) -> None:
    def generate_map():
        raw = env.unwrapped.env
        handles = raw.get_handles()
        size = env.unwrapped.map_size
        positions = _fixed_positions(size, n_melee, n_ranged)
        for handle, key in zip(handles[:4], ["redmelee", "redranged", "bluemelee", "blueranged"]):
            raw.add_agents(handle, method="custom", pos=positions[key])

    env.unwrapped.generate_map = generate_map


def _fixed_positions(map_size: int, n_melee: int, n_ranged: int) -> dict[str, list[list[int]]]:
    centre = map_size // 2
    def ys(n):
        return [int(round(centre + off)) for off in (np.arange(n) - (n - 1) / 2)]

    return {
        "bluemelee": [[3, y] for y in ys(n_melee)],
        "blueranged": [[2, y] for y in ys(n_ranged)],
        "redmelee": [[map_size - 4, y] for y in ys(n_melee)],
        "redranged": [[map_size - 3, y] for y in ys(n_ranged)],
    }


def _blue_event(before, after, actions):
    red_before = before["red"]
    red_after = after["red"]
    blue_before = before["blue"]
    blue_after = after["blue"]
    red_ctr_before = centroid(red_before)
    red_ctr_after = centroid(red_after)
    melee_count = ranged_count = 0
    melee_advanced = ranged_fired = ranged_retreated = melee_screened = 0
    ranged_ctr = centroid(after["blueranged"])
    for bid, p0 in blue_before.items():
        if bid not in blue_after:
            continue
        p1 = blue_after[bid]
        if is_melee(bid):
            melee_count += 1
            if dist(p1, red_ctr_after) < dist(p0, red_ctr_before):
                melee_advanced += 1
            if after["blueranged"] and dist(p1, red_ctr_after) <= dist(ranged_ctr, red_ctr_after) + 0.75:
                melee_screened += 1
        elif is_ranged(bid):
            ranged_count += 1
            if int(actions.get(bid, 0)) >= attack_base(bid):
                ranged_fired += 1
            if dist(p1, red_ctr_after) > dist(p0, red_ctr_before):
                ranged_retreated += 1
    return {
        "blue_melee_count": melee_count,
        "blue_ranged_count": ranged_count,
        "blue_melee_advanced": melee_advanced,
        "blue_ranged_fired": ranged_fired,
        "blue_ranged_retreated": ranged_retreated,
        "blue_melee_screened": melee_screened,
    }


CALIBRATION_SEED_FLOOR = 1000  # seeds >= 1000 are calibration seeds; the held-out type is excluded only there


def _held_out_for_seed(seed: int, held_out: str | None) -> str | None:
    """The held-out type never appears on calibration seeds but does appear on deployment seeds (< 1000)."""
    return held_out if int(seed) >= CALIBRATION_SEED_FLOOR else None


def _sample_type_for_seed(type_names: list[str], seed: int, switch: str | None, held_out: str | None) -> str:
    choices = _eligible_types(type_names, _held_out_for_seed(seed, held_out))
    rng_seed = (int(seed) // 20) if switch == "block20" else int(seed)
    rng = np.random.default_rng(rng_seed)
    return str(rng.choice(choices))


def _eligible_types(type_names: list[str], held_out: str | None) -> list[str]:
    out = [t for t in type_names if t != held_out]
    if not out:
        raise ValueError("held_out removed every available type for calibration seeds")
    return out


def _resolve_type_set(type_set: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(type_set, str):
        if type_set not in TYPE_SETS:
            raise ValueError(f"unknown type_set: {type_set}")
        return list(TYPE_SETS[type_set])
    return [str(t) for t in type_set]


def _resolve_pool(pool: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(pool, str):
        if pool not in POOL_SETS:
            raise ValueError(f"unknown pool: {pool}")
        return list(POOL_SETS[pool])
    return [str(t) for t in pool]


def _coerce_episode_spec(spec: EpisodeSpec | dict[str, Any]) -> EpisodeSpec:
    if isinstance(spec, EpisodeSpec):
        return spec
    return EpisodeSpec(
        seed=int(spec["seed"]),
        type1=str(spec.get("type1", spec.get("type"))),
        type2=None if spec.get("type2") is None else str(spec.get("type2")),
        switch_step=None if spec.get("switch_step") is None else int(spec.get("switch_step")),
    )


def _centroid_gap(red_pos, blue_pos) -> float:
    if not red_pos or not blue_pos:
        return 0.0
    return dist(centroid(red_pos), centroid(blue_pos))


def _mean_distance(src: dict[str, tuple[int, int]], targets: dict[str, tuple[int, int]]) -> float:
    if not src or not targets:
        return 0.0
    target_points = list(targets.values())
    vals = []
    for p in src.values():
        vals.append(min(dist(p, q) for q in target_points))
    return float(np.mean(vals)) if vals else 0.0


def _call_decide(decide_fn, ctx: Context, env: CombinedArmsRS) -> str:
    try:
        return str(decide_fn(ctx, env))
    except TypeError:
        return str(decide_fn(ctx))
