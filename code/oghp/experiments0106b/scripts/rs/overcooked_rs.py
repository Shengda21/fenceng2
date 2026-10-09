"""Overcooked replay-selection environment."""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V8_ROOT = ROOT.parent.parent / "v8_lib"
for path in (ROOT, V8_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from v8lib.context import Context

try:
    from .oc_common import (
        clone_env_wrapper,
        held_item,
        make_env,
        nearest_facility_distance,
        pot_state,
        seed_everything,
    )
    from .overcooked_roles import ROLE_FULL, ROLES, role_task, role_uses_partner_server
    from .overcooked_types import COUNTER_PASSER, PARTNER_TYPES, partner_task
except ImportError:  # pragma: no cover - direct script fallback
    from oc_common import (  # type: ignore
        clone_env_wrapper,
        held_item,
        make_env,
        nearest_facility_distance,
        pot_state,
        seed_everything,
    )
    from overcooked_roles import ROLE_FULL, ROLES, role_task, role_uses_partner_server  # type: ignore
    from overcooked_types import COUNTER_PASSER, PARTNER_TYPES, partner_task  # type: ignore

FEATURE_NAMES = [
    "partner_pickups_onion",
    "partner_pickups_dish",
    "partner_pickups_soup",
    "partner_drops",
    "partner_serves",
    "partner_mean_dist_pot",
    "partner_mean_dist_serve",
    "partner_mean_dist_onion",
    "partner_stay_frac",
    "partner_holds_empty",
    "partner_holds_onion",
    "partner_holds_dish",
    "partner_holds_soup",
    "pot_onions",
    "pot_cooking",
    "pot_ready",
    "step",
]


def feature_vector(raw: dict[str, Any]) -> np.ndarray:
    held = raw.get("partner_held", "empty")
    pot = raw.get("pot", {})
    return np.asarray(
        [
            raw.get("partner_pickups", {}).get("onion", 0),
            raw.get("partner_pickups", {}).get("dish", 0),
            raw.get("partner_pickups", {}).get("soup", 0),
            raw.get("partner_drops", 0),
            raw.get("partner_serves", 0),
            raw.get("partner_mean_dist_pot", 999.0),
            raw.get("partner_mean_dist_serve", 999.0),
            raw.get("partner_mean_dist_onion", 999.0),
            raw.get("partner_stay_frac", 0.0),
            1.0 if held == "empty" else 0.0,
            1.0 if held == "onion" else 0.0,
            1.0 if held == "dish" else 0.0,
            1.0 if held == "soup" else 0.0,
            pot.get("onions", 0),
            1.0 if pot.get("cooking") else 0.0,
            1.0 if pot.get("ready") else 0.0,
            raw.get("step", 0),
        ],
        dtype=float,
    )


def semantic_template(raw: dict[str, Any]) -> str:
    pickups = raw.get("partner_pickups", {})
    dominant = max(("onion", "dish", "soup"), key=lambda k: pickups.get(k, 0))
    pot = raw.get("pot", {})
    held = raw.get("partner_held", "empty")
    sentence1 = (
        f"Your partner has spent the first {raw.get('step', 0)} steps carrying "
        f"{dominant}s most often, with onion pickups={pickups.get('onion', 0)}, "
        f"dish pickups={pickups.get('dish', 0)}, soup pickups={pickups.get('soup', 0)}, "
        f"serves={raw.get('partner_serves', 0)}, and drops={raw.get('partner_drops', 0)}."
    )
    sentence2 = (
        f"The partner is currently holding {held}; the pot has {pot.get('onions', 0)} onions, "
        f"cooking={bool(pot.get('cooking'))}, ready={bool(pot.get('ready'))}, "
        f"and the selector must choose one role for chef 0."
    )
    return sentence1 + " " + sentence2


@dataclass
class PartnerStats:
    pickups: dict[str, int] = field(default_factory=lambda: {"onion": 0, "dish": 0, "soup": 0})
    drops: int = 0
    serves: int = 0
    stays: int = 0
    dist_pot: list[float] = field(default_factory=list)
    dist_serve: list[float] = field(default_factory=list)
    dist_onion: list[float] = field(default_factory=list)


class OvercookedRS:
    """Decision-window environment around deterministic Overcooked partner types."""

    def __init__(
        self,
        layout: str,
        ptype_set: tuple[str, ...] = PARTNER_TYPES,
        pool: tuple[str, ...] = ROLES,
        default: str = ROLE_FULL,
        k: int = 40,
        horizon: int = 400,
        switch: str | None = None,
        held_out: str | None = None,
    ) -> None:
        self.layout = layout
        self.ptype_set = tuple(ptype_set)
        self.pool = list(pool)
        self.default = default
        self.k = int(k)
        self.horizon = int(horizon)
        self.switch = switch
        self.held_out = held_out
        self.schedule = "cadence20" if switch == "cadence20" else ("mid" if switch == "mid" else "single")

        self.env_wrap, self.mapper = make_env(layout, horizon)
        self.rng = np.random.default_rng(0)
        self.seed = 0
        self.episode_index = 0
        self.window_index = 0
        self.step = 0
        self.total_reward = 0.0
        self.done = False
        self.ptype = self.ptype_set[0]
        self.ptype2: str | None = None
        self.current_ptype = self.ptype
        self.stats = PartnerStats()
        self.decisions: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []

    def reset(self, seed: int) -> Context:
        self.seed = int(seed)
        self.episode_index = int(seed)
        self.window_index = 0
        self.step = 0
        self.total_reward = 0.0
        self.done = False
        self.decisions = []
        self.trace = []
        self.stats = PartnerStats()
        seed_everything(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.env_wrap.reset(seed=self.seed)

        choices = list(self.ptype_set)
        if self.held_out:
            if 0 <= self.seed <= 99:
                self.ptype = self.held_out
            else:
                choices = [p for p in choices if p != self.held_out] or choices
                self.ptype = str(self.rng.choice(choices))
        else:
            self.ptype = str(self.rng.choice(choices))
        self.ptype2 = str(self.rng.choice(choices)) if self.switch == "mid" else None
        self.current_ptype = self.ptype

        first = self._decision_steps()[0]
        self._advance_until(first, self.default)
        return self._context()

    def step_window(self, tau: str) -> tuple[Context | None, float, bool, dict]:
        if self.done:
            return None, 0.0, True, self._info()
        chosen = tau if tau in self.pool else self.default
        self.decisions.append({"step": self.step, "tau": chosen, "fallback": chosen != tau})
        start_reward = self.total_reward
        next_decision = self._next_decision_after(self.step)
        target = next_decision if next_decision is not None else self.horizon
        self._advance_until(target, chosen)
        reward = self.total_reward - start_reward
        self.window_index = len(self.decisions)
        ctx = None if self.done or next_decision is None else self._context()
        return ctx, float(reward), self.done or next_decision is None, self._info()

    def replay(self, seed: int, decide_fn: Callable[[Context], str]) -> dict[str, Any]:
        ctx = self.reset(seed)
        done = False
        while not done:
            tau = decide_fn(ctx)
            ctx, _, done, _ = self.step_window(tau)
            if ctx is None:
                done = True
        return self._row()

    def clone(self) -> "OvercookedRS":
        out = object.__new__(OvercookedRS)
        out.layout = self.layout
        out.ptype_set = self.ptype_set
        out.pool = list(self.pool)
        out.default = self.default
        out.k = self.k
        out.horizon = self.horizon
        out.switch = self.switch
        out.held_out = self.held_out
        out.schedule = self.schedule
        out.env_wrap = clone_env_wrapper(self.env_wrap)
        out.mapper = type(self.mapper)(out.env_wrap)
        out.rng = np.random.default_rng()
        out.rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        out.seed = self.seed
        out.episode_index = self.episode_index
        out.window_index = self.window_index
        out.step = self.step
        out.total_reward = self.total_reward
        out.done = self.done
        out.ptype = self.ptype
        out.ptype2 = self.ptype2
        out.current_ptype = self.current_ptype
        out.stats = copy.deepcopy(self.stats)
        out.decisions = copy.deepcopy(self.decisions)
        out.trace = copy.deepcopy(self.trace)
        return out

    def close(self) -> None:
        self.env_wrap.close()

    def _decision_steps(self) -> list[int]:
        if self.schedule == "cadence20":
            return list(range(20, self.horizon, 20))
        if self.schedule == "mid":
            return [self.k, 240]
        return [self.k]

    def _next_decision_after(self, step: int) -> int | None:
        for ds in self._decision_steps():
            if ds > step:
                return ds
        return None

    def _advance_until(self, target_step: int, tau: str) -> None:
        target_step = min(target_step, self.horizon)
        while self.step < target_step and not self.env_wrap.is_done():
            if self.switch == "mid" and self.ptype2 is not None and self.step >= 200:
                self.current_ptype = self.ptype2
            self._advance_one(tau)
        self.done = self.env_wrap.is_done() or self.step >= self.horizon

    def _advance_one(self, tau: str) -> None:
        before_held = held_item(self.mapper, 1)
        before_pos = self.mapper._get_pos(1)
        partner = partner_task(self.env_wrap, self.mapper, self.current_ptype, self.step, self.rng)
        own = role_task(
            self.env_wrap,
            self.mapper,
            tau,
            self.step,
            partner_serves=self.current_ptype == "SERVER" or role_uses_partner_server(tau),
        )
        actions = {
            "chef_0": self.mapper.get_action(0, own, use_handover=True),
            "chef_1": self.mapper.get_action(1, partner, use_handover=True),
        }
        self._record_distances()
        if actions["chef_1"] == 4:
            self.stats.stays += 1
        result = self.env_wrap.step(actions)
        reward = float(result.total_reward())
        self.total_reward += reward
        after_held = held_item(self.mapper, 1)
        self._record_partner_transition(before_held, after_held, reward)
        self.trace.append(
            {
                "step": self.step,
                "ptype": self.current_ptype,
                "role": tau,
                "our_task": own,
                "partner_task": partner,
                "partner_pos": before_pos,
                "partner_held_before": before_held,
                "partner_held_after": after_held,
                "reward": reward,
            }
        )
        self.step += 1

    def _record_distances(self) -> None:
        self.stats.dist_pot.append(nearest_facility_distance(self.mapper, 1, list(self.mapper.mdp.get_pot_locations())))
        self.stats.dist_serve.append(
            nearest_facility_distance(self.mapper, 1, list(self.mapper.mdp.get_serving_locations()))
        )
        self.stats.dist_onion.append(
            nearest_facility_distance(self.mapper, 1, list(self.mapper.mdp.get_onion_dispenser_locations()))
        )

    def _record_partner_transition(self, before: str, after: str, reward: float) -> None:
        if before == "empty" and after in self.stats.pickups:
            self.stats.pickups[after] += 1
        elif before != "empty" and after == "empty":
            if reward > 0 and before == "soup":
                self.stats.serves += 1
            else:
                self.stats.drops += 1

    def _raw_features(self) -> dict[str, Any]:
        n = max(1, self.step)
        raw = {
            "type_label": self.current_ptype,
            "ptype": self.ptype,
            "ptype2": self.ptype2,
            "step": self.step,
            "partner_pickups": dict(self.stats.pickups),
            "partner_drops": self.stats.drops,
            "partner_serves": self.stats.serves,
            "partner_mean_dist_pot": float(np.mean(self.stats.dist_pot)) if self.stats.dist_pot else 999.0,
            "partner_mean_dist_serve": float(np.mean(self.stats.dist_serve)) if self.stats.dist_serve else 999.0,
            "partner_mean_dist_onion": float(np.mean(self.stats.dist_onion)) if self.stats.dist_onion else 999.0,
            "partner_stay_frac": self.stats.stays / n,
            "partner_held": held_item(self.mapper, 1),
            "pot": pot_state(self.env_wrap, self.mapper),
        }
        return raw

    def _context(self) -> Context:
        raw = self._raw_features()
        return Context(
            pool=list(self.pool),
            features=feature_vector(raw),
            text=semantic_template(raw),
            episode_index=self.episode_index,
            window_index=self.window_index,
            default=self.default,
            type_label=self.current_ptype,
            extra={"raw": raw, "layout": self.layout, "seed": self.seed},
        )

    def _info(self) -> dict[str, Any]:
        return {
            "success": self.total_reward > 0,
            "total_reward": self.total_reward,
            "steps": self.step,
            "ptype": self.ptype,
            "ptype2": self.ptype2,
            "decisions": copy.deepcopy(self.decisions),
            "trace": self.trace,
        }

    def _row(self, provenance: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "episode": self.episode_index,
            "seed": self.seed,
            "ptype": self.ptype,
            "ptype2": self.ptype2,
            "decisions": copy.deepcopy(self.decisions),
            "total_reward": float(self.total_reward),
            "success": bool(self.total_reward > 0),
            "steps": self.step,
            "provenance": provenance or [],
        }


def mask_context(ctx: Context) -> Context:
    extra = copy.deepcopy(ctx.extra)
    raw = extra.get("raw")
    if isinstance(raw, dict):
        raw.pop("type_label", None)
        raw.pop("ptype", None)
        raw.pop("ptype2", None)
    return replace(ctx, type_label=None, extra=extra)


def run_episode_with_selector(
    env: OvercookedRS,
    seed: int,
    selector: Any,
    encoder: Any | None = None,
) -> dict[str, Any]:
    ctx = env.reset(seed)
    if encoder is not None:
        ctx = encoder(ctx.extra["raw"], ctx)
    selector.reset_episode(ctx if getattr(selector, "reads_type_label", False) else mask_context(ctx))
    provenance = []
    done = False
    start = time.perf_counter()
    while not done:
        visible = ctx if getattr(selector, "reads_type_label", False) else mask_context(ctx)
        tau = selector.select(visible)
        if getattr(selector, "last_provenance", None):
            provenance.append(copy.deepcopy(selector.last_provenance))
        next_ctx, reward, done, info = env.step_window(tau)
        selector.update(visible, tau, reward, done, info)
        if next_ctx is None:
            break
        if encoder is not None:
            next_ctx = encoder(next_ctx.extra["raw"], next_ctx)
        ctx = next_ctx
        ctx.window_index = len(env.decisions)
    row = env._row(provenance=provenance)
    row["provenance_latency_s"] = float(sum(p.get("latency_s", 0.0) for p in provenance))
    row["wall_s"] = float(time.perf_counter() - start) if provenance else 0.0
    return row
