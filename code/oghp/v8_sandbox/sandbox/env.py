"""Seed-deterministic typed repeated game environment."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from v8lib.context import Context
from v8lib.encoders import make_encoder

from sandbox.game import (
    ACTIONS,
    all_types,
    beats,
    describe_type,
    expected_payoff,
    opponent_dist,
    payoff_matrix,
    strategy_dist,
    strategy_pool,
)


@dataclass(frozen=True)
class SandboxConfig:
    M: int = 3
    K: int = 3
    delta: float = 1.0
    sharpness: float = 0.8
    reward_noise: float = 0.0
    H: int = 30
    k: int = 5
    schedule: str = "single"
    default: str = "ROCK"
    L: int = 5
    switch: str | None = None
    encoding: str = "semantic"
    features: str = "basic"
    held_out: str | None = None
    episode_index: int = 0

    def normalized(self) -> "SandboxConfig":
        schedule = "per_round" if self.schedule == "mid" else self.schedule
        switch = "mid" if self.schedule == "mid" else self.switch
        if schedule not in {"single", "per_round"}:
            raise ValueError("schedule must be single, per_round, or mid")
        if self.features not in {"basic", "rich"}:
            raise ValueError("features must be basic or rich")
        if not 0.0 <= float(self.sharpness) <= 1.0:
            raise ValueError("sharpness must be in [0, 1]")
        if float(self.reward_noise) < 0.0:
            raise ValueError("reward_noise must be non-negative")
        if self.default not in strategy_pool(self.K):
            raise ValueError("default strategy must be in the pool")
        return replace(self, schedule=schedule, switch=switch)


class SandboxEnv:
    """Episode environment compatible with `v8lib.runner.run_deployment`."""

    def __init__(self, config: SandboxConfig | None = None, forced_type: str | None = None):
        self.config = (config or SandboxConfig()).normalized()
        self.pool = strategy_pool(self.config.K)
        self.A = payoff_matrix(self.config.delta)
        self.forced_type = forced_type
        self.rng = np.random.default_rng(0)
        self.reward_rng = np.random.default_rng(1)
        self.seed = 0
        self.type_label = ""
        self.round_index = 0
        self.window_index = 0
        self.our_history: list[str] = []
        self.opp_history: list[str] = []
        self._pending_reward = 0.0
        self._done = False

    def reset(self, seed: int) -> Context:
        self.rng = np.random.default_rng(int(seed))
        self.reward_rng = np.random.default_rng(int(seed) + 1_000_003)
        self.seed = int(seed)
        self.type_label = self.forced_type or self._draw_type()
        self.round_index = 0
        self.window_index = 0
        self.our_history = []
        self.opp_history = []
        self._pending_reward = 0.0
        self._done = False
        if self.config.schedule == "single":
            for _ in range(min(self.config.k, self.config.H)):
                self._pending_reward += self._play_round(self.config.default)
        return self._context()

    def step_window(self, tau: str) -> tuple[Context | None, float, bool, dict]:
        if self._done:
            raise RuntimeError("episode is already done")
        if tau not in self.pool:
            tau = self.config.default
        reward = self._pending_reward
        self._pending_reward = 0.0
        rounds = 1 if self.config.schedule == "per_round" else self.config.H - self.round_index
        for _ in range(max(0, rounds)):
            reward += self._play_round(tau)
        self.window_index += 1
        self._done = self.round_index >= self.config.H
        info = {
            "type": self.type_label,
            "success": bool(reward > 0.0),
            "round": self.round_index,
            "raw": self._raw_observation(),
        }
        return (None if self._done else self._context(), float(reward), self._done, info)

    def clone(self) -> "SandboxEnv":
        return copy.deepcopy(self)

    def replay(self, seed: int, decide_fn: Callable[[Context], str]) -> float:
        ctx = self.reset(seed)
        total = 0.0
        done = False
        while not done:
            tau = decide_fn(ctx)
            ctx, reward, done, _ = self.step_window(tau)
            total += float(reward)
        return total

    def _draw_type(self) -> str:
        labels = all_types(self.config.M)
        if self.config.held_out and 100 <= self.seed <= 119:
            labels = [label for label in labels if label != self.config.held_out]
        return str(self.rng.choice(labels))

    def _active_type(self) -> str:
        if self.config.switch == "mid" and self.round_index == self.config.H // 2:
            labels = all_types(self.config.M)
            self.type_label = str(self.rng.choice(labels))
        return self.type_label

    def _play_round(self, strategy: str) -> float:
        typ = self._active_type()
        opp_probs = opponent_dist(
            typ,
            self.round_index,
            self.our_history,
            L=self.config.L,
            sharpness=self.config.sharpness,
        )
        reward = expected_payoff(strategy, opp_probs, self.A)
        if self.config.reward_noise > 0.0:
            reward += float(self.reward_rng.normal(0.0, self.config.reward_noise))
        our_action = str(self.rng.choice(ACTIONS, p=strategy_dist(strategy)))
        opp_action = str(self.rng.choice(ACTIONS, p=opp_probs))
        self.our_history.append(our_action)
        self.opp_history.append(opp_action)
        self.round_index += 1
        return reward

    def _raw_observation(self) -> dict:
        counts = {action: int(self.opp_history.count(action)) for action in ACTIONS}
        n_prev = max(0, len(self.opp_history) - 1)
        br = 0
        cp = 0
        for i in range(1, len(self.opp_history)):
            prev_ours = self.our_history[i - 1]
            br += int(self.opp_history[i] == beats(prev_ours))
            cp += int(self.opp_history[i] == prev_ours)
        common = max(ACTIONS, key=lambda a: (counts[a], a))
        run = 0
        for action in reversed(self.opp_history):
            if action != common:
                break
            run += 1
        return {
            "counts": counts,
            "br_fraction": float(br / n_prev) if n_prev else 0.0,
            "copy_fraction": float(cp / n_prev) if n_prev else 0.0,
            "last_best_response": bool(
                len(self.opp_history) >= 2 and self.opp_history[-1] == beats(self.our_history[-2])
            ),
            "last_copy": bool(len(self.opp_history) >= 2 and self.opp_history[-1] == self.our_history[-2]),
            "common_run": int(run),
            "round": int(self.round_index),
            "hidden_type_label": self.type_label,
            "hidden_type_description": describe_type(self.type_label),
            "our_history": list(self.our_history),
            "opp_history": list(self.opp_history),
        }

    def _context(self) -> Context:
        raw = self._raw_observation()
        return Context(
            pool=list(self.pool),
            features=feature_vector(raw, mode=self.config.features),
            text=semantic_template(raw),
            episode_index=self.config.episode_index,
            window_index=self.window_index,
            default=self.config.default,
            type_label=self.type_label,
            extra={"raw": raw},
        )


def basic_feature_vector(raw: dict) -> np.ndarray:
    counts = raw.get("counts", {})
    round_idx = max(1.0, float(raw.get("round", 0)))
    return np.array(
        [
            float(counts.get("ROCK", 0)) / round_idx,
            float(counts.get("PAPER", 0)) / round_idx,
            float(counts.get("SCISSORS", 0)) / round_idx,
            float(raw.get("br_fraction", 0.0)),
            float(raw.get("copy_fraction", 0.0)),
            float(raw.get("common_run", 0)) / round_idx,
            float(raw.get("round", 0)) / 30.0,
        ],
        dtype=float,
    )


def rich_feature_vector(raw: dict) -> np.ndarray:
    counts = raw.get("counts", {})
    common = max(ACTIONS, key=lambda a: (counts.get(a, 0), a))
    common_hot = np.array([float(action == common) for action in ACTIONS], dtype=float)
    copy = bool(raw.get("last_copy", False))
    br = bool(raw.get("last_best_response", False))
    return np.concatenate(
        [
            common_hot,
            np.array([float(copy), float(not copy), float(br), float(not br)], dtype=float),
            basic_feature_vector(raw),
        ]
    )


def feature_vector(raw: dict, mode: str = "basic") -> np.ndarray:
    if mode == "basic":
        return basic_feature_vector(raw)
    if mode == "rich":
        return rich_feature_vector(raw)
    raise ValueError("mode must be basic or rich")


BASIC_FEATURE_NAMES = [
    "opp_rock",
    "opp_paper",
    "opp_scissors",
    "best_response_fraction",
    "copy_fraction",
    "common_run_fraction",
    "round_over_30",
]

RICH_FEATURE_NAMES = [
    "most_common_rock",
    "most_common_paper",
    "most_common_scissors",
    "last_copy_true",
    "last_copy_false",
    "last_best_response_true",
    "last_best_response_false",
] + BASIC_FEATURE_NAMES


def semantic_template(raw: dict) -> str:
    counts = raw.get("counts", {})
    pieces = [f"{name.lower()} {counts.get(name, 0)} times" for name in ACTIONS]
    copy = raw.get("copy_fraction", 0.0)
    br = raw.get("br_fraction", 0.0)
    return (
        f"So far the opponent has thrown {', '.join(pieces)}. "
        f"It copied your previous move {copy:.0%} of the time and best-responded {br:.0%} of the time."
    )


def make_sandbox_encoder(config: SandboxConfig):
    cfg = config.normalized()
    return make_encoder(
        "semantic" if cfg.encoding in {"relabel", "swapdesc"} else cfg.encoding,
        semantic_template,
        lambda raw: feature_vector(raw, mode=cfg.features),
        all_types(cfg.M),
        permutation_seed=17,
        keep_numeric=True,
        feature_names=RICH_FEATURE_NAMES if cfg.features == "rich" else BASIC_FEATURE_NAMES,
    )


def make_env(config: SandboxConfig | None = None, forced_type: str | None = None) -> SandboxEnv:
    return SandboxEnv(config=config, forced_type=forced_type)
