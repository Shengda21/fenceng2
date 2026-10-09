"""Fixed-strategy and per-window oracle probes for selector experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

import numpy as np

from v8lib.context import Context


@dataclass(frozen=True)
class BestFixedResult:
    tau: str | None
    mean: float
    tie_set: list[str]
    separated: bool

    def __iter__(self) -> Iterator:
        yield self.tau
        yield self.mean

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return (self.tau, self.mean) == other
        return super().__eq__(other)


class EpisodeEnv(Protocol):
    def reset(self, seed: int) -> Context: ...

    def step_window(self, tau: str) -> tuple[Context | None, float, bool, dict]: ...

    def clone(self) -> "EpisodeEnv": ...


def fixed_strategy_sweep(
    make_env: Callable[[], EpisodeEnv], pool: list[str], seeds
) -> dict[str, dict]:
    """Roll one full episode per strategy and seed."""

    out = {}
    seed_list = list(seeds)
    for tau in pool:
        per_seed = []
        for seed in seed_list:
            env = make_env()
            env.reset(int(seed))
            total, done = 0.0, False
            while not done:
                _, reward, done, _ = env.step_window(tau)
                total += float(reward)
            per_seed.append(total)
        out[tau] = {"mean": float(np.mean(per_seed)), "per_seed": per_seed}
    return out


def best_fixed(results: dict[str, dict], tol: float = 1e-9) -> BestFixedResult:
    """Return the best fixed strategy, value, and unresolved tie set."""

    if not results:
        raise ValueError("results must not be empty")
    means = {tau: float(row["mean"]) for tau, row in results.items()}
    top_mean = max(means.values())
    tied = {tau for tau, mean in means.items() if top_mean - mean <= float(tol)}
    top_tau = max(tied)
    top_values = results[top_tau].get("per_seed")
    if top_values is not None:
        top_arr = np.asarray(top_values, dtype=float)
        for tau, row in results.items():
            if tau == top_tau:
                continue
            vals = row.get("per_seed")
            if vals is None:
                continue
            arr = np.asarray(vals, dtype=float)
            if arr.shape == top_arr.shape:
                diffs = top_arr - arr
                if float(np.mean(diffs)) <= float(tol) or float(np.min(diffs)) <= float(tol):
                    tied.add(tau)
    tie_set = sorted(tied)
    separated = len(tie_set) == 1
    return BestFixedResult(top_tau if separated else None, top_mean, tie_set, separated)


def oracle_h_d(delta_def: float, placebo: float, v_star: float, v_default: float) -> float:
    """Compute the placebo-corrected decision heterogeneity value."""

    return float((delta_def - placebo) - (v_star - v_default))


def per_window_oracle(
    make_env: Callable[[], EpisodeEnv],
    pool: list[str],
    default: str,
    seeds,
    placebo: bool = False,
) -> dict:
    """Branch each window from a default rollout and estimate oracle lift."""

    seed_list = list(seeds)
    deltas, by_seed = _oracle_deltas(make_env, pool, default, seed_list, placebo=placebo)
    delta_def = float(np.mean([sum(xs) for xs in by_seed])) if by_seed else 0.0
    if placebo:
        placebo_delta = delta_def
    else:
        _, placebo_by_seed = _oracle_deltas(make_env, pool, default, seed_list, placebo=True)
        placebo_delta = float(np.mean([sum(xs) for xs in placebo_by_seed])) if placebo_by_seed else 0.0
    sweep = fixed_strategy_sweep(make_env, pool, seed_list)
    _, v_star = best_fixed(sweep)
    v_default = float(sweep[default]["mean"])
    return {
        "Delta_def": delta_def,
        "placebo": placebo_delta,
        "per_window": deltas,
        "per_seed": by_seed,
        "V_star": v_star,
        "V_default": v_default,
        "H_D": oracle_h_d(delta_def, placebo_delta, v_star, v_default),
    }


def _oracle_deltas(make_env, pool, default, seeds, placebo: bool):
    by_seed = []
    flattened = []
    for seed in seeds:
        env = make_env()
        env.reset(int(seed))
        seed_deltas = []
        done = False
        while not done:
            state = env.clone()
            branch_values = []
            branch_pool = [default for _ in pool] if placebo else pool
            for tau in branch_pool:
                branch = state.clone()
                total, b_done = 0.0, False
                _, reward, b_done, _ = branch.step_window(tau)
                total += float(reward)
                while not b_done:
                    _, reward, b_done, _ = branch.step_window(default)
                    total += float(reward)
                branch_values.append(total)
            default_branch = state.clone()
            default_total, d_done = 0.0, False
            while not d_done:
                _, reward, d_done, _ = default_branch.step_window(default)
                default_total += float(reward)
            delta = float(max(branch_values) - default_total)
            seed_deltas.append(delta)
            flattened.append(delta)
            _, _, done, _ = env.step_window(default)
        by_seed.append(seed_deltas)
    return flattened, by_seed
