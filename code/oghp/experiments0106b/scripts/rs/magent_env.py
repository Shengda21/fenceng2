"""MAgent regime-switching episode runner with a deterministic fake adapter."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from v8lib.context import Context
except ImportError:  # pragma: no cover - used when v8lib is not installed editable
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "v8_lib"))
    from v8lib.context import Context

from scripts.run_magent_sweep import (
    ATTACK_OFFSETS,
    MOVE_OFFSETS,
    act_oghp,
    attack_action_if_enemy_adjacent,
    get_positions,
    move_away,
    move_toward,
    nearest,
)
from scripts.rs.magent_types import TYPE_SETS, act_blue, episode_rng, keyed_int, keyed_uniform, semantic_descriptor

POOL4 = ("ATTACK_FORWARD", "HOLD_POSITION", "SPREAD_OUT", "RETREAT")
POOL8 = POOL4 + ("FOCUS_FIRE", "FLANK", "KITE", "RETREAT_REAL")
POOL_SETS = {"pool4": POOL4, "pool8": POOL8}

FEATURE_NAMES = [
    "blue_centroid_velocity_toward_red",
    "blue_dispersion",
    "blue_dispersion_change",
    "engagement_distance",
    "blue_fraction_advanced",
    "blue_fraction_attacked",
    "blue_alive",
    "red_alive",
    "step",
]


def feature_fn(raw: dict) -> np.ndarray:
    """Convert raw MAgentRS state into the fixed numeric feature vector."""

    events = raw.get("events", [])
    start_blue = raw.get("start_blue_pos", {})
    cur_blue = raw.get("blue_pos", {})
    cur_red = raw.get("red_pos", {})
    start_red = raw.get("start_red_pos", {})
    step = int(raw.get("step", 0))

    start_gap = _centroid_gap(start_red, start_blue)
    cur_gap = _centroid_gap(cur_red, cur_blue)
    n_steps = max(1, int(raw.get("window_steps", step)))
    velocity = (start_gap - cur_gap) / n_steps

    start_disp = _dispersion(start_blue)
    cur_disp = _dispersion(cur_blue)
    denom = sum(max(1, int(e.get("blue_count", 0))) for e in events) or 1
    advanced = sum(int(e.get("advanced", 0)) for e in events) / denom
    attacked = sum(int(e.get("attacked", 0)) for e in events) / denom

    return np.asarray(
        [
            velocity,
            cur_disp,
            cur_disp - start_disp,
            cur_gap,
            advanced,
            attacked,
            float(len(cur_blue)),
            float(len(cur_red)),
            float(step),
        ],
        dtype=float,
    )


def semantic_template(raw: dict, concept_mapping: dict[str, str] | None = None) -> str:
    """Describe observed BLUE behavior without naming the hidden type."""

    vals = dict(zip(FEATURE_NAMES, feature_fn(raw)))
    typ = str(raw.get("type_label", raw.get("type", "")))
    concept_type = (concept_mapping or {}).get(typ, typ)
    try:
        concept = semantic_descriptor(concept_type)
    except ValueError:
        concept = "behavior that must be inferred from the current movement window"
    speed = vals["blue_centroid_velocity_toward_red"]
    dispersion = vals["blue_dispersion"]
    spread_change = vals["blue_dispersion_change"]
    gap = vals["engagement_distance"]
    advanced = vals["blue_fraction_advanced"]
    fired = vals["blue_fraction_attacked"]
    formation = "tight" if dispersion < 2.0 else "wide"
    spread_phrase = "spreading out" if spread_change > 0.15 else "compressing" if spread_change < -0.15 else "holding shape"
    pace = "advancing quickly" if speed > 0.35 else "edging forward" if speed > 0.05 else "not advancing"
    move_quant = "almost every" if advanced > 0.75 else "many" if advanced > 0.4 else "few"
    fire_quant = "many" if fired > 0.4 else "some" if fired > 0.1 else "few"
    # Feature-derived text only: the hidden type's descriptor (`concept`) must not appear in the observation.
    del concept
    return (
        f"The enemy force is {pace} in a {formation} formation and is now {gap:.1f} cells away. "
        f"{move_quant.capitalize()} enemy units have been moving toward us while {fire_quant} have fired. "
        f"Their spacing is {spread_phrase}, with {int(vals['blue_alive'])} enemies still active."
    )


@dataclass
class ReplayResult:
    context: Context | None
    reward: float
    done: bool
    info: dict


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    type1: str
    type2: str | None = None
    switch_step: int | None = None

    @property
    def key(self) -> str:
        return f"{self.seed}:{self.type1}:{self.type2 or ''}:{self.switch_step if self.switch_step is not None else ''}"


class MAgentRS:
    """Decision-window environment wrapper for MAgent BLUE type switching."""

    def __init__(
        self,
        map_size: int = 20,
        max_cycles: int = 100,
        type_set: str | list[str] | tuple[str, ...] = "base4",
        pool: str | list[str] | tuple[str, ...] = "pool8",
        default: str = "HOLD_POSITION",
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
        self._red_rng = np.random.default_rng(0)
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
        self._red_rng = np.random.default_rng(seed * 1000 + 17)
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
        red_pos, blue_pos = get_positions(self.env)
        return {
            "episode": int(seed),
            "seed": int(seed),
            "type": self.blue_type,
            "type2": self.blue_type2,
            "decisions": decisions,
            "total_reward": float(total),
            "success": bool(total > 0),
            "steps": int(self.step),
            "red_alive_final": len(red_pos),
            "blue_alive_final": len(blue_pos),
            "provenance": provenance,
        }

    def clone(self) -> "MAgentRS":
        if not isinstance(self.env, FakeParallelMAgent):
            raise RuntimeError("MAgentRS.clone is only supported by the fake adapter")
        return copy.deepcopy(self)

    def state_summary(self) -> dict[str, Any]:
        red_pos, blue_pos = get_positions(self.env)
        return {
            "step": self.step,
            "red_alive": len(red_pos),
            "blue_alive": len(blue_pos),
            "total_reward": self._total_reward,
        }

    def _make_env(self):
        if self.env_factory is not None:
            return self.env_factory()
        if not self.force_fake:
            try:
                from magent2.environments import battle_v4

                return battle_v4.parallel_env(
                    map_size=self.map_size,
                    max_cycles=self.max_cycles,
                    minimap_mode=False,
                )
            except Exception:
                pass
        return FakeParallelMAgent(map_size=self.map_size, max_cycles=self.max_cycles)

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
                choices = _eligible_types(self.type_names, seed, self.held_out)
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
        red_pos, blue_pos = get_positions(self.env)
        self._obs_start_red = dict(red_pos)
        self._obs_start_blue = dict(blue_pos)
        self._events = []

    def _step(self, tau: str) -> float:
        agents = list(getattr(self.env, "agents", []))
        red_ids = [a for a in agents if a.startswith("red_")]
        blue_ids = [a for a in agents if a.startswith("blue_")]
        before_red, before_blue = get_positions(self.env)
        chosen = act_red_strategy(self.env, agents, tau, self.step, self._red_state)
        hold = act_red_strategy(self.env, agents, "HOLD_POSITION", self.step, self._red_state)
        actions: dict[str, int] = {}
        red_gates = {}
        for rid in red_ids:
            gate = keyed_uniform(self.seed, self.step, rid, "red_delta_gate")
            self.rng_trace.append(
                {
                    "seed": self.seed,
                    "step": self.step,
                    "agent_id": rid,
                    "purpose": "red_delta_gate",
                    "value": float(gate),
                }
            )
            red_gates[rid] = float(gate)
            use_tau = gate <= self.delta_scale
            actions[rid] = int(chosen.get(rid, 0) if use_tau else hold.get(rid, 0))
        active_blue = self._active_blue_type()
        if active_blue == "random":
            blue_actions = {}
            for bid in blue_ids:
                val = keyed_int(self.seed, self.step, bid, "blue_random_action", 21)
                self.rng_trace.append(
                    {
                        "seed": self.seed,
                        "step": self.step,
                        "agent_id": bid,
                        "purpose": "blue_random_action",
                        "value": int(val),
                    }
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
                "actions": {k: int(v) for k, v in sorted(actions.items())},
                "red_gates": red_gates,
            }
        )
        _, rewards, _, _, _ = self.env.step(actions)
        after_red, after_blue = get_positions(self.env)
        self._events.append(_blue_event(before_red, before_blue, after_red, after_blue, actions))
        self.step += 1
        if not getattr(self.env, "agents", []) or self.step >= self.max_cycles:
            self._done = True
        return float(sum(float(r) for k, r in rewards.items() if k.startswith("red_")))

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
        red_pos, blue_pos = get_positions(self.env)
        return {
            "seed": self.seed,
            "step": self.step,
            "window_index": self.window_index,
            "type": self._active_blue_type(),
            "type_label": self._active_blue_type(),
            "type1": self.blue_type,
            "type2": self.blue_type2,
            "switch_step": self._episode_switch_step if self.switch == "mid" else None,
            "red_pos": dict(red_pos),
            "blue_pos": dict(blue_pos),
            "start_red_pos": dict(self._obs_start_red),
            "start_blue_pos": dict(self._obs_start_blue),
            "events": list(self._events),
            "window_steps": max(1, self.step - self._window_start_step),
        }


class FakeParallelMAgent:
    """Small deterministic in-memory subset of the battle_v4 parallel API."""

    def __init__(self, map_size: int = 20, max_cycles: int = 100, n_per_team: int = 6) -> None:
        self.map_size = int(map_size)
        self.max_cycles = int(max_cycles)
        self.n_per_team = int(n_per_team)
        self.unwrapped = self
        self.env = _FakeCore(self)
        self.agents: list[str] = []
        self.positions: dict[str, tuple[int, int]] = {}
        self.hp: dict[str, int] = {}
        self.step_count = 0

    def reset(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        centre = self.map_size // 2
        offsets = np.arange(self.n_per_team) - (self.n_per_team - 1) / 2
        self.positions = {}
        self.hp = {}
        for i, off in enumerate(offsets):
            jitter = int(rng.integers(-1, 2))
            self.positions[f"blue_{i}"] = (3 + jitter, int(round(centre + off)))
            self.positions[f"red_{i}"] = (self.map_size - 4 - jitter, int(round(centre + off)))
            self.hp[f"blue_{i}"] = 5
            self.hp[f"red_{i}"] = 5
        self.step_count = 0
        self._refresh_agents()
        return {a: np.zeros(1, dtype=float) for a in self.agents}

    def step(self, actions: dict[str, int]):
        live_before = list(self.agents)
        rewards = {a: 0.0 for a in live_before}
        killed: set[str] = set()
        for actor in sorted(live_before, key=_agent_sort_key):
            if actor in killed or actor not in self.positions:
                continue
            action = int(actions.get(actor, 0))
            if action < 13:
                continue
            target = self._attack_target(actor, action)
            if target is None or target in killed:
                continue
            self.hp[target] -= 1
            if self.hp[target] <= 0:
                killed.add(target)
                if actor.startswith("red_"):
                    rewards[actor] = rewards.get(actor, 0.0) + 1.0
                else:
                    for rid in [a for a in live_before if a.startswith("red_")]:
                        rewards[rid] = rewards.get(rid, 0.0) - 1.0 / max(1, self.n_per_team)
        for dead in killed:
            self.positions.pop(dead, None)
            self.hp.pop(dead, None)
        occupied = dict(self.positions)
        for actor in sorted(live_before, key=_agent_sort_key):
            if actor not in self.positions:
                continue
            action = int(actions.get(actor, 0))
            if action <= 0 or action >= 13:
                continue
            dx, dy = MOVE_OFFSETS[action]
            x, y = self.positions[actor]
            nxt = (int(np.clip(x + dx, 0, self.map_size - 1)), int(np.clip(y + dy, 0, self.map_size - 1)))
            if nxt not in occupied.values():
                self.positions[actor] = nxt
                occupied[actor] = nxt
        self.step_count += 1
        self._refresh_agents()
        done = self.step_count >= self.max_cycles or not any(a.startswith("red_") for a in self.agents) or not any(a.startswith("blue_") for a in self.agents)
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
        idx = action - 13
        if idx < 0 or idx >= len(ATTACK_OFFSETS):
            return None
        dx, dy = ATTACK_OFFSETS[idx]
        x, y = self.positions[actor]
        target_pos = (x + dx, y + dy)
        prefix = "blue_" if actor.startswith("red_") else "red_"
        for agent_id, pos in self.positions.items():
            if agent_id.startswith(prefix) and pos == target_pos:
                return agent_id
        return None

    def _refresh_agents(self) -> None:
        self.agents = sorted(self.positions, key=lambda a: (0 if a.startswith("red_") else 1, _agent_sort_key(a)))


class _FakeCore:
    def __init__(self, outer: FakeParallelMAgent) -> None:
        self.outer = outer

    def get_handles(self):
        return ["red", "blue"]

    def get_pos(self, handle):
        prefix = "red_" if handle == "red" else "blue_"
        vals = [
            self.outer.positions[a]
            for a in sorted(self.outer.positions, key=_agent_sort_key)
            if a.startswith(prefix)
        ]
        return np.asarray(vals, dtype=int)

    def get_view_space(self, handle):
        return ((self.outer.map_size, self.outer.map_size, 1),)


def _sample_type_for_seed(type_names: list[str], seed: int, switch: str | None, held_out: str | None) -> str:
    choices = _eligible_types(type_names, seed, held_out)
    rng_seed = (int(seed) // 20) if switch == "block20" else int(seed)
    rng = np.random.default_rng(rng_seed)
    return str(rng.choice(choices))


CALIBRATION_SEED_FLOOR = 1000  # seeds >= 1000 are calibration seeds; the held-out type is excluded only there


def _eligible_types(type_names: list[str], seed: int, held_out: str | None) -> list[str]:
    """The held-out type never appears on calibration seeds but does appear on deployment seeds (< 1000)."""
    out = list(type_names)
    if held_out is not None and int(seed) >= CALIBRATION_SEED_FLOOR:
        out = [t for t in out if t != held_out]
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


def act_red_strategy(env, agents: list[str], tau: str, step: int = 0, state: dict[str, Any] | None = None) -> dict[str, int]:
    """Return low-level RED actions for an expanded regime-switching strategy."""

    if tau in POOL4:
        return _act_oghp_red_only(env, agents, tau)
    if state is None:
        state = {}

    red_pos, blue_pos = get_positions(env)
    red_ids = sorted([a for a in agents if a.startswith("red_")], key=_agent_sort_key)
    blue_items = sorted(blue_pos.items(), key=lambda kv: _agent_sort_key(kv[0]))
    enemies = [p for _, p in blue_items]

    if tau == "FOCUS_FIRE":
        return _act_focus_fire(red_ids, red_pos, blue_items)
    if tau == "FLANK":
        return _act_flank(red_ids, red_pos, enemies)
    if tau == "KITE":
        return _act_kite(red_ids, red_pos, enemies, step, state)
    if tau == "RETREAT_REAL":
        return _act_retreat_real(red_ids, red_pos, enemies)
    return act_oghp(env, agents, "HOLD_POSITION")


def _act_oghp_red_only(env, agents: list[str], meta: str) -> dict[str, int]:
    """RED-only half of act_oghp, kept action-identical for the original tactics."""

    red_pos, blue_pos = get_positions(env)
    red_ids = [a for a in agents if a.startswith("red_")]
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
            out[rid] = 0
        else:
            out[rid] = a_atk if a_atk is not None else 0
    return out


def redact_hidden_context(ctx: Context) -> Context:
    """Remove hidden type labels recursively before non-oracle selectors see a context."""

    hidden_keys = {"type", "type_label", "type1", "type2", "blue_type", "blue_type2"}
    return replace(ctx, type_label=None, extra=_redact_value(ctx.extra, hidden_keys))


def _redact_value(value: Any, hidden_keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            k: _redact_value(v, hidden_keys)
            for k, v in value.items()
            if str(k) not in hidden_keys
        }
    if isinstance(value, list):
        return [_redact_value(v, hidden_keys) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v, hidden_keys) for v in value)
    return value


def _coerce_episode_spec(spec: EpisodeSpec | dict[str, Any]) -> EpisodeSpec:
    if isinstance(spec, EpisodeSpec):
        return spec
    return EpisodeSpec(
        seed=int(spec["seed"]),
        type1=str(spec["type1"]),
        type2=None if spec.get("type2") is None else str(spec.get("type2")),
        switch_step=None if spec.get("switch_step") is None else int(spec.get("switch_step")),
    )


def _act_focus_fire(red_ids, red_pos, blue_items):
    out = {}
    if not blue_items:
        return {rid: 0 for rid in red_ids}
    red_ctr = _centroid(red_pos)
    blue_ctr = _centroid(dict(blue_items))
    focus = _lowest_adjacent_enemy(red_pos, blue_items)
    enemies = [pos for _, pos in blue_items]
    for rid in red_ids:
        p = red_pos.get(rid)
        if p is None:
            out[rid] = 0
            continue
        atk = _attack_specific(p, focus)
        if atk is None:
            atk = attack_action_if_enemy_adjacent(p, enemies)
        if atk is not None:
            out[rid] = atk
        elif _dist(p, red_ctr) > 2.0:
            out[rid] = move_toward(p, red_ctr)
        else:
            out[rid] = move_toward(p, blue_ctr)
    return out


def _act_flank(red_ids, red_pos, enemies):
    out = {}
    if not enemies:
        return {rid: 0 for rid in red_ids}
    own_ctr = np.asarray(_centroid(red_pos), dtype=float)
    enemy_ctr = np.asarray(_centroid({str(i): p for i, p in enumerate(enemies)}), dtype=float)
    direction = enemy_ctr - own_ctr
    lateral = np.asarray([-direction[1], direction[0]], dtype=float)
    norm = float(np.linalg.norm(lateral))
    lateral = np.asarray([0.0, 1.0]) if norm == 0.0 else lateral / norm
    for i, rid in enumerate(red_ids):
        p = red_pos.get(rid)
        if p is None:
            out[rid] = 0
            continue
        atk = attack_action_if_enemy_adjacent(p, enemies)
        if atk is not None:
            out[rid] = atk
            continue
        side = -1.0 if i % 2 == 0 else 1.0
        target = tuple((enemy_ctr + side * 3.0 * lateral).tolist())
        out[rid] = move_toward(p, target)
    return out


def _act_kite(red_ids, red_pos, enemies, step, state):
    out = {}
    last_attack = state.setdefault("red_kite_last_attack", {})
    for rid in red_ids:
        p = red_pos.get(rid)
        if p is None:
            out[rid] = 0
            continue
        tgt, dist = nearest(p, enemies)
        atk = attack_action_if_enemy_adjacent(p, enemies)
        if atk is not None and not last_attack.get(rid, False):
            out[rid] = atk
            last_attack[rid] = True
        elif tgt is not None and (atk is not None or dist <= 2.0):
            out[rid] = move_away(p, tgt)
            last_attack[rid] = False
        else:
            out[rid] = move_toward(p, tgt) if tgt is not None else 0
            last_attack[rid] = False
    return out


def _act_retreat_real(red_ids, red_pos, enemies):
    out = {}
    for rid in red_ids:
        p = red_pos.get(rid)
        if p is None:
            out[rid] = 0
            continue
        tgt, _ = nearest(p, enemies)
        out[rid] = move_away(p, tgt) if tgt is not None else 0
    return out


def _lowest_adjacent_enemy(red_pos, blue_items):
    for _bid, bpos in blue_items:
        for rpos in red_pos.values():
            if max(abs(bpos[0] - rpos[0]), abs(bpos[1] - rpos[1])) <= 1:
                return bpos
    return None


def _attack_specific(my, target):
    if target is None:
        return None
    dx, dy = int(target[0] - my[0]), int(target[1] - my[1])
    if (dx, dy) in ATTACK_OFFSETS:
        return 13 + ATTACK_OFFSETS.index((dx, dy))
    return None


def _blue_event(before_red, before_blue, after_red, after_blue, actions):
    red_ctr_before = _centroid(before_red)
    red_ctr_after = _centroid(after_red)
    advanced = 0
    attacked = 0
    count = 0
    for bid, p0 in before_blue.items():
        if bid not in after_blue:
            continue
        count += 1
        p1 = after_blue[bid]
        if _dist(p1, red_ctr_after) < _dist(p0, red_ctr_before):
            advanced += 1
        if int(actions.get(bid, 0)) >= 13:
            attacked += 1
    return {"blue_count": count, "advanced": advanced, "attacked": attacked}


def _centroid(pos: dict[str, tuple[int, int]]) -> tuple[float, float]:
    if not pos:
        return (0.0, 0.0)
    arr = np.asarray(list(pos.values()), dtype=float)
    ctr = arr.mean(axis=0)
    return (float(ctr[0]), float(ctr[1]))


def _centroid_gap(red_pos, blue_pos) -> float:
    if not red_pos or not blue_pos:
        return 0.0
    return _dist(_centroid(red_pos), _centroid(blue_pos))


def _dispersion(pos: dict[str, tuple[int, int]]) -> float:
    if not pos:
        return 0.0
    ctr = np.asarray(_centroid(pos), dtype=float)
    arr = np.asarray(list(pos.values()), dtype=float)
    return float(np.mean(np.linalg.norm(arr - ctr, axis=1)))


def _dist(a, b) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _agent_sort_key(agent_id: str) -> int:
    try:
        return int(agent_id.split("_")[-1])
    except Exception:
        return abs(hash(agent_id)) % (2**31)


def _call_decide(decide_fn, ctx: Context, env: MAgentRS) -> str:
    try:
        return str(decide_fn(ctx, env))
    except TypeError:
        return str(decide_fn(ctx))
