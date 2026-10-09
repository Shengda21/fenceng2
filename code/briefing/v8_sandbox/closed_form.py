"""Closed-form style Monte Carlo references for the E0 sandbox."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace

import numpy as np

from sandbox.env import SandboxConfig, SandboxEnv, deployment_types
from sandbox.game import (
    ACTIONS,
    action_index,
    all_types,
    best_response_to_dist,
    opponent_dist,
    payoff_matrix,
    strategy_dist,
    strategy_pool,
)


def compute_closed_forms(
    config: SandboxConfig,
    samples: int = 100_000,
    seed_start: int = 0,
) -> dict:
    """Estimate fixed and oracle values by deterministic Monte Carlo seeds."""

    cfg = config.normalized()
    pool = strategy_pool(cfg.K, cfg.family)
    seeds = list(range(int(seed_start), int(seed_start) + int(samples)))
    fixed = {tau: [] for tau in pool}
    oracle_values = []
    default_values = []
    for seed in seeds:
        base = SandboxEnv(cfg)
        base.reset(seed)
        for tau in pool:
            env = base.clone()
            total, done = 0.0, False
            while not done:
                _, reward, done, _ = env.step_window(tau)
                total += float(reward)
            fixed[tau].append(total)

        env = base.clone()
        default_env = base.clone()
        oracle_total = 0.0
        default_total = 0.0
        done = False
        while not done:
            candidates = []
            for tau in pool:
                branch = env.clone()
                _, reward, _, _ = branch.step_window(tau)
                candidates.append((float(reward), tau))
            tau = max(candidates, key=lambda item: (item[0], item[1]))[1]
            _, reward, done, _ = env.step_window(tau)
            oracle_total += float(reward)
            _, d_reward, d_done, _ = default_env.step_window(cfg.default)
            default_total += float(d_reward)
            if d_done and not done:
                raise RuntimeError("default and oracle rollouts diverged in horizon")
        oracle_values.append(oracle_total)
        default_values.append(default_total)

    V_tau = {tau: float(np.mean(vals)) for tau, vals in fixed.items()}
    best_tau = max(pool, key=lambda tau: (V_tau[tau], tau))
    V_star = float(V_tau[best_tau])
    oracle = float(np.mean(oracle_values))
    V_default = float(V_tau[cfg.default])
    gap = delta_min(cfg)
    sigma = sigma_eff(cfg, samples=min(int(samples), 20_000), seed_start=seed_start + 50_000_000)
    snr_value = snr(gap, sigma)
    return {
        "method": "deterministic Monte Carlo over seeded episodes",
        "samples": int(samples),
        "V_tau": V_tau,
        "best_tau": best_tau,
        "V_star": V_star,
        "V_default": V_default,
        "per_decision_oracle": oracle,
        "H_D": float(oracle - V_star),
        "Delta_min": float(gap),
        "sigma_eff": float(sigma),
        "snr": float(snr_value),
        "predicted_T_star": float(predicted_T_star(cfg.K, len(deployment_types(cfg)), snr_value)),
        "predicted_T_star_known_type": float(predicted_T_star(cfg.K, len(deployment_types(cfg)), snr_value, method="known_type")),
    }


def delta_min(config: SandboxConfig) -> float:
    cfg = config.normalized()
    if cfg.family == "traits":
        rounds = max(1, cfg.H - cfg.k)
        values = type_values(cfg)
        gaps = [float(np.diff(np.sort(list(values[t].values()))[-2:])[0]) / rounds for t in deployment_types(cfg)]
        return float(min(gaps))
    pool = strategy_pool(cfg.K)
    A = payoff_matrix(cfg.delta)
    gaps = []
    for typ in all_types(cfg.M):
        if typ.endswith("_LOVER"):
            dist = opponent_dist(typ, cfg.k, [cfg.default] * cfg.k, L=cfg.L, sharpness=cfg.sharpness)
            gaps.append(best_response_to_dist(pool, dist, A)[2])
        elif typ == "FLIPPER_L":
            dist = opponent_dist(typ, cfg.k, [cfg.default] * cfg.k, L=cfg.L, sharpness=cfg.sharpness)
            gaps.append(best_response_to_dist(pool, dist, A)[2])
        elif typ == "BEST_RESPONDER":
            dist = opponent_dist(typ, cfg.k, [cfg.default] * cfg.k, L=cfg.L, sharpness=cfg.sharpness)
            gaps.append(best_response_to_dist(pool, dist, A)[2])
        elif typ == "GULLIBLE":
            dist = opponent_dist(typ, cfg.k, [cfg.default] * cfg.k, L=cfg.L, sharpness=cfg.sharpness)
            gaps.append(best_response_to_dist(pool, dist, A)[2])
    return float(min(gaps))


def sigma_eff(config: SandboxConfig, samples: int = 20_000, seed_start: int = 0) -> float:
    """Monte Carlo per-round reward standard deviation under best responses."""

    cfg = config.normalized()
    if cfg.family == "traits":
        return trait_sigma_eff(cfg, samples=samples, seed_start=seed_start)
    pool = strategy_pool(cfg.K)
    labels = all_types(cfg.M)
    A = payoff_matrix(cfg.delta)
    rng = np.random.default_rng(int(seed_start))
    rewards = np.empty(int(samples), dtype=float)
    for i in range(int(samples)):
        typ = labels[int(rng.integers(0, len(labels)))]
        round_index = cfg.k if cfg.schedule == "single" else int(rng.integers(0, max(1, cfg.H)))
        our_history = [cfg.default] * max(0, round_index)
        dist = opponent_dist(typ, round_index, our_history, L=cfg.L, sharpness=cfg.sharpness)
        tau = best_response_to_dist(pool, dist, A)[0]
        our_action = str(rng.choice(ACTIONS, p=strategy_dist(tau)))
        opp_action = str(rng.choice(ACTIONS, p=dist))
        reward = float(A[action_index(our_action), action_index(opp_action)])
        if cfg.reward_noise > 0.0:
            reward += float(rng.normal(0.0, cfg.reward_noise))
        rewards[i] = reward
    return float(np.std(rewards, ddof=0))


def type_values(config: SandboxConfig) -> dict[str, dict[str, float]]:
    cfg = replace(config.normalized(), reward_noise=0.0, held_out=None, deploy_types="all")
    out = {}
    for typ in all_types(cfg.M, cfg.family):
        base = SandboxEnv(cfg, forced_type=typ)
        base.reset(0)
        out[typ] = {}
        for tau in strategy_pool(cfg.K, cfg.family):
            env = base.clone()
            total, done = 0.0, False
            while not done:
                _, reward, done, _ = env.step_window(tau)
                total += float(reward)
            out[typ][tau] = total
    return out


def best_response_table(values: dict[str, dict[str, float]]) -> dict[str, str]:
    return {typ: max(v, key=lambda tau: (v[tau], tau)) for typ, v in values.items()}


def trait_sigma_eff(cfg: SandboxConfig, samples: int = 20_000, seed_start: int = 0) -> float:
    labels = deployment_types(cfg)
    best = best_response_table(type_values(cfg))
    A = payoff_matrix(cfg.delta)
    rng = np.random.default_rng(int(seed_start))
    rewards = np.empty(int(samples), dtype=float)
    for i in range(int(samples)):
        typ = labels[int(rng.integers(0, len(labels)))]
        round_index = int(rng.integers(cfg.k, max(cfg.k + 1, cfg.H)))
        ours = [cfg.default] * cfg.k
        for r in range(cfg.k, round_index):
            ours.append(ACTIONS[int(np.argmax(strategy_dist(best[typ], ours, r, cfg.H)))])
        dist = opponent_dist(typ, round_index, ours, L=cfg.L, sharpness=cfg.sharpness, H=cfg.H)
        our_action = str(rng.choice(ACTIONS, p=strategy_dist(best[typ], ours, round_index, cfg.H)))
        opp_action = str(rng.choice(ACTIONS, p=dist))
        reward = float(A[action_index(our_action), action_index(opp_action)])
        if cfg.reward_noise > 0.0:
            reward += float(rng.normal(0.0, cfg.reward_noise))
        rewards[i] = reward
    return float(np.std(rewards, ddof=0))


def prefix_likelihoods(cfg: SandboxConfig, types: list[str]) -> tuple[list[tuple[str, ...]], np.ndarray]:
    seqs = list(itertools.product(ACTIONS, repeat=cfg.k))
    hist = [cfg.default] * cfg.k
    dists = {t: [opponent_dist(t, r, hist[:r], L=cfg.L, sharpness=cfg.sharpness, H=cfg.H) for r in range(cfg.k)] for t in types}
    P = np.array([[np.prod([dists[t][r][action_index(a)] for r, a in enumerate(seq)]) for t in types] for seq in seqs])
    return seqs, P


def dominant_prefix_action(cfg: SandboxConfig, typ: str) -> str:
    hist = [cfg.default] * cfg.k
    rounds = range(1, cfg.k) if cfg.k > 1 else range(cfg.k)
    mean = np.mean([opponent_dist(typ, r, hist[:r], L=cfg.L, sharpness=cfg.sharpness, H=cfg.H) for r in rounds], axis=0)
    return ACTIONS[int(np.argmax(mean))]


def exact_reference(config: SandboxConfig) -> dict:
    """Exact type-averaged values for the traits family: rewards are expected payoffs, so each type-strategy value is deterministic."""

    cfg = config.normalized()
    pool = strategy_pool(cfg.K, cfg.family)
    types = deployment_types(cfg)
    values = type_values(cfg)
    V = np.array([[values[t][tau] for tau in pool] for t in types])
    V_tau = {tau: float(V[:, j].mean()) for j, tau in enumerate(pool)}
    best_tau = max(pool, key=lambda tau: (V_tau[tau], tau))
    V_star = V_tau[best_tau]
    oracle = float(V.max(axis=1).mean())
    H_D = oracle - V_star
    V_min = min(V_tau.values())
    seqs, P = prefix_likelihoods(cfg, types)
    scores = (P / len(types)) @ V
    bayes = float(scores.max(axis=1).sum())
    choice = [pool[int(j)] for j in scores.argmax(axis=1)]
    per_type_bayes = {t: float(sum(P[n, i] * values[t][choice[n]] for n in range(len(seqs)))) for i, t in enumerate(types)}
    classes: dict[str, list[str]] = {}
    for t in types:
        classes.setdefault(dominant_prefix_action(cfg, t), []).append(t)
    class_value, class_best = 0.0, {}
    for sig, members in classes.items():
        idx = [types.index(t) for t in members]
        means = V[idx].mean(axis=0)
        class_best[sig] = pool[int(np.argmax(means))]
        class_value += len(idx) / len(types) * float(means.max())
    return {
        "method": "exact enumeration over types; prefix ceilings over all 3^k prefix sequences under the default",
        "V_tau": V_tau,
        "best_tau": best_tau,
        "V_star": V_star,
        "V_default": V_tau[cfg.default],
        "V_min": V_min,
        "per_decision_oracle": oracle,
        "H_D": H_D,
        "share": H_D / (V_star - V_min) if V_star > V_min else float("nan"),
        "best_response": best_response_table(values),
        "values": values,
        "prefix_ceiling_bayes": bayes,
        "prefix_ceiling_bayes_capture": (bayes - V_star) / H_D if H_D > 0 else float("nan"),
        "prefix_ceiling_bayes_per_type_regret": {t: max(values[t].values()) - per_type_bayes[t] for t in types},
        "prefix_bayes_choice": {" ".join(seq): tau for seq, tau in zip(seqs, choice)},
        "prefix_classes": classes,
        "prefix_class_best": class_best,
        "prefix_ceiling_class": class_value,
        "prefix_ceiling_class_capture": (class_value - V_star) / H_D if H_D > 0 else float("nan"),
    }


def snr(Delta_min: float, sigma: float) -> float:
    if float(Delta_min) <= 0:
        return 0.0
    if float(sigma) <= 1e-12:
        return float("inf")
    return float(Delta_min) / float(sigma)


def predicted_T_star(K: int, M: int, snr_value: float, method: str = "linucb_tabular") -> float:
    if float(snr_value) <= 0:
        return float("inf")
    multiplier = 1.0 if method in {"known_type", "per_type"} else float(M)
    return float(multiplier * int(K) / (float(snr_value) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="classic", choices=["classic", "traits"])
    parser.add_argument("--M", type=int, default=3, choices=[3, 6, 9, 18])
    parser.add_argument("--K", type=int, default=3, choices=[3, 5, 10])
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=0.8)
    parser.add_argument("--reward-noise", type=float, default=0.0)
    parser.add_argument("--H", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--schedule", default="single", choices=["single", "per_round", "mid"])
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = SandboxConfig(
        M=args.M,
        K=args.K,
        delta=args.delta,
        sharpness=args.sharpness,
        reward_noise=args.reward_noise,
        H=args.H,
        k=args.k,
        schedule=args.schedule,
        family=args.family,
    )
    start = time.perf_counter()
    out = compute_closed_forms(cfg, samples=args.samples)
    if args.family == "traits":
        out["runtime_s"] = time.perf_counter() - start
        start = time.perf_counter()
        out["exact"] = exact_reference(cfg)
        out["exact"]["runtime_s"] = time.perf_counter() - start
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
