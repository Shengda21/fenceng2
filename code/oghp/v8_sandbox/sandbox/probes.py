"""Replay-based probes alongside v8lib clone-based probes."""

from __future__ import annotations

import numpy as np


def replay_fixed_sweep(make_env, pool: list[str], seeds) -> dict[str, dict]:
    out = {}
    for tau in pool:
        vals = []
        for seed in seeds:
            env = make_env()
            vals.append(env.replay(int(seed), lambda _ctx, tau=tau: tau))
        out[tau] = {"mean": float(np.mean(vals)), "per_seed": vals}
    return out


def replay_oracle(make_env, pool: list[str], default: str, seeds) -> dict:
    totals = []
    default_vals = []
    for seed in seeds:
        env = make_env()
        ctx = env.reset(int(seed))
        done = False
        total = 0.0
        default_total = 0.0
        while not done:
            vals = []
            for tau in pool:
                branch = env.clone()
                _, reward, _, _ = branch.step_window(tau)
                vals.append((reward, tau))
            tau = max(vals, key=lambda x: (x[0], x[1]))[1]
            d_branch = env.clone()
            _, d_reward, _, _ = d_branch.step_window(default)
            default_total += float(d_reward)
            ctx, reward, done, _ = env.step_window(tau)
            total += float(reward)
        totals.append(total)
        default_vals.append(default_total)
    return {"oracle_values": totals, "default_values": default_vals, "mean": float(np.mean(totals))}
