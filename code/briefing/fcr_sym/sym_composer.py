"""Symbolic (model-based) counter composer for FCR-sym.

Given a trait tuple z = (favourite, reactivity[, timing]), the composer instantiates the scripted opponent that the
sandbox code defines for z (`sandbox.game.trait_dist`, reached through `sandbox.env.SandboxEnv(forced_type=type_of(z))`),
plays every pool strategy against it over the deployment's episode protocol (the default strategy for the first k rounds,
then the chosen strategy to round H, sharpness and reward noise sigma of the cell), and returns the strategy with the
highest mean simulated return. It is the builder-owned simulator the probes assume (`sandbox/probes.py`,
`replay_fixed_sweep`): every rollout is `SandboxEnv.replay(seed, lambda ctx: tau)`.

Settings:
  - R = 64 rollouts per (z, tau), on simulator seeds SIM_SEED_BASE + r, r = 0..63. The same seeds serve every strategy and
    every z (common random numbers). SIM_SEED_BASE = 900,000,000 is disjoint from every deployment seed (< 1,000), the
    calibration seeds (100-119) and the lock seeds (10,000-10,017 and 20,000-20,017), and so are the derived reward-noise
    streams (seed + 1,000,003) and briefing streams (seed + 7,777,777).
  - Choice: argmax of the 64-rollout mean; strategies within TIE_EPS = 1e-9 of the maximum are tied, and a tie goes to the
    earliest strategy in pool order (the FCR composer's rule).
  - Inputs: z, the game rules of the cell (M, K, delta, sharpness, H, k, schedule, default strategy, reward noise) and the
    pool. No calibration label, value table, held-out label, deployment seed or deployment return enters.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # code/briefing/fcr_sym
for _p in (HERE.parent / "v8_sandbox", HERE.parent / "v8_lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sandbox.env import SandboxConfig, SandboxEnv  # noqa: E402
from sandbox.game import strategy_pool, trait_types  # noqa: E402

SIM_ROLLOUTS = 64
SIM_SEED_BASE = 900_000_000
TIE_EPS = 1e-9
TRAIT_NAMES = {9: ("favourite", "reactivity"), 18: ("favourite", "reactivity", "timing")}

# The game rules of the traits family (run_sandbox.py defaults; every briefing lock stores the same values).
GAME_RULES = {"delta": 1.0, "sharpness": 0.8, "H": 30, "k": 5, "schedule": "single", "default": "ROCK"}


def type_of(z, M: int) -> str:
    return "-".join(z[: len(TRAIT_NAMES[int(M)])])


def all_tuples(M: int) -> list[tuple]:
    from sandbox.game import parse_trait

    return [tuple(parse_trait(t)[: len(TRAIT_NAMES[int(M)])]) for t in trait_types(int(M))]


class SymbolicComposer:
    """z -> argmax_tau of the simulated mean return of tau against the scripted opponent of z.

    One instance per game cell (M, K, reward noise). Results are memoised per z; memoisation does not change the function.
    """

    def __init__(self, M: int, K: int, reward_noise: float, rollouts: int = SIM_ROLLOUTS, seed_base: int = SIM_SEED_BASE,
                 rules: dict | None = None):
        rules = dict(GAME_RULES if rules is None else rules)
        self.M, self.K, self.reward_noise = int(M), int(K), float(reward_noise)
        self.rollouts, self.seed_base = int(rollouts), int(seed_base)
        self.rules = rules
        self.cfg = SandboxConfig(M=self.M, K=self.K, delta=float(rules["delta"]), sharpness=float(rules["sharpness"]),
                                 reward_noise=self.reward_noise, H=int(rules["H"]), k=int(rules["k"]),
                                 schedule=str(rules["schedule"]), default=str(rules["default"]), family="traits").normalized()
        self.pool = strategy_pool(self.K, "traits")
        self.seeds = [self.seed_base + r for r in range(self.rollouts)]
        self._cache: dict = {}

    def simulate(self, z) -> dict:
        """Per-strategy returns over the R rollouts against the scripted opponent of z."""

        z = tuple(z)
        if z in self._cache:
            return self._cache[z]
        typ = type_of(z, self.M)
        G = np.zeros((len(self.pool), self.rollouts))
        for j, tau in enumerate(self.pool):
            for r, seed in enumerate(self.seeds):
                env = SandboxEnv(self.cfg, forced_type=typ)
                G[j, r] = env.replay(int(seed), lambda _ctx, tau=tau: tau)
        means = G.mean(axis=1)
        top = float(means.max())
        tied = [self.pool[j] for j in range(len(self.pool)) if means[j] >= top - TIE_EPS]
        choice = tied[0]
        order = np.argsort(-means, kind="stable")
        second = float(means[order[1]]) if len(order) > 1 else top
        # with deterministic pool strategies and common noise, the per-rollout difference to the chosen strategy is constant
        jc = self.pool.index(choice)
        D = G - G[jc]
        res = {
            "type": typ,
            "choice": choice,
            "tied": tied,
            "means": OrderedDict((tau, float(means[j])) for j, tau in enumerate(self.pool)),
            "rollout_sd": OrderedDict((tau, float(G[j].std())) for j, tau in enumerate(self.pool)),
            "gap_best_second": float(top - second) if not len(tied) > 1 else 0.0,
            "max_dev_of_difference_across_rollouts": float(np.max(np.abs(D - D.mean(axis=1, keepdims=True)))),
        }
        self._cache[z] = res
        return res

    def choose(self, zs) -> list[str]:
        return [self.simulate(z)["choice"] for z in zs]

    def describe(self) -> dict:
        return {"M": self.M, "K": self.K, "reward_noise": self.reward_noise, "rollouts": self.rollouts,
                "seed_base": self.seed_base, "tie_eps": TIE_EPS, "rules": self.rules, "pool": list(self.pool)}
