"""Calibration artifacts for few-shot and PLASTIC-style selectors."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from sandbox.briefing import all_renderings
from sandbox.env import SandboxConfig, SandboxEnv, feature_vector, make_sandbox_encoder
from sandbox.game import ACTIONS, all_types, beats, best_response_to_dist, opponent_dist, payoff_matrix, strategy_pool


def canonical_best_response(config: SandboxConfig) -> dict[str, str]:
    cfg = replace(config.normalized(), reward_noise=0.0)
    pool = strategy_pool(cfg.K, cfg.family)
    A = payoff_matrix(cfg.delta)
    out = {}
    for typ in all_types(cfg.M, cfg.family):
        env = SandboxEnv(cfg, forced_type=typ)
        env.reset(10_000 + all_types(cfg.M, cfg.family).index(typ))
        vals = {}
        for tau in pool:
            branch = env.clone()
            _, reward, _, _ = branch.step_window(tau)
            vals[tau] = reward
        out[typ] = max(pool, key=lambda tau: (vals[tau], tau))
    return out


def value_table(config: SandboxConfig) -> dict[str, dict[str, float]]:
    cfg = replace(config.normalized(), reward_noise=0.0)
    pool = strategy_pool(cfg.K, cfg.family)
    out = {}
    for typ in all_types(cfg.M, cfg.family):
        env = SandboxEnv(cfg, forced_type=typ)
        env.reset(20_000 + all_types(cfg.M, cfg.family).index(typ))
        out[typ] = {}
        for tau in pool:
            branch = env.clone()
            _, reward, _, _ = branch.step_window(tau)
            out[typ][tau] = float(reward)
    return out


def calibration_rows(
    config: SandboxConfig,
    seeds: list[int] | None = None,
    held_out: str | list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config.normalized()
    excluded = {held_out} if isinstance(held_out, str) else set(held_out or ())
    labels = [label for label in all_types(cfg.M, cfg.family) if label not in excluded]
    rows, y = [], []
    for typ in labels:
        rows.append(expected_feature_for_type(cfg, typ))
        y.append(typ)
    for seed in (seeds or []):
        for typ in labels:
            env = SandboxEnv(cfg, forced_type=typ)
            ctx = env.reset(int(seed))
            rows.append(np.asarray(ctx.features, dtype=float))
            y.append(typ)
    return np.asarray(rows, dtype=float), np.asarray(y, dtype=str)


def expected_feature_for_type(config: SandboxConfig, typ: str) -> np.ndarray:
    cfg = config.normalized()
    if cfg.family == "traits":
        return expected_trait_feature(cfg, typ)
    round_idx = cfg.k if cfg.schedule == "single" else 1
    our_hist = [cfg.default] * max(0, round_idx)
    probs = np.zeros(3, dtype=float)
    for r in range(max(1, round_idx)):
        probs += opponent_dist(typ, r, our_hist[:r], L=cfg.L, sharpness=cfg.sharpness)
    probs /= max(1, round_idx)
    raw = {
        "counts": {"ROCK": probs[0] * round_idx, "PAPER": probs[1] * round_idx, "SCISSORS": probs[2] * round_idx},
        "br_fraction": float(typ == "BEST_RESPONDER"),
        "copy_fraction": float(typ == "GULLIBLE"),
        "common_run": round_idx,
        "round": round_idx,
    }
    return feature_vector(raw, mode=cfg.features)


def expected_trait_feature(cfg: SandboxConfig, typ: str) -> np.ndarray:
    round_idx = cfg.k if cfg.schedule == "single" else 1
    our_hist = [cfg.default] * max(0, round_idx)
    per_round = [
        opponent_dist(typ, r, our_hist[:r], L=cfg.L, sharpness=cfg.sharpness, H=cfg.H) for r in range(max(1, round_idx))
    ]
    probs = np.sum(per_round, axis=0)
    later = per_round[1:]
    br = float(np.mean([d[ACTIONS.index(beats(cfg.default))] for d in later])) if later else 0.0
    cp = float(np.mean([d[ACTIONS.index(cfg.default)] for d in later])) if later else 0.0
    raw = {
        "counts": {a: float(probs[i]) for i, a in enumerate(ACTIONS)},
        "br_fraction": br,
        "copy_fraction": cp,
        "common_run": round_idx,
        "round": round_idx,
    }
    return feature_vector(raw, mode=cfg.features)


def calibration_records(config: SandboxConfig, seeds: list[int], held_out, best_response: dict[str, str]) -> list[dict]:
    cfg = config.normalized()
    encoder = make_sandbox_encoder(cfg)
    excluded = set(held_out or ())
    rows = []
    for typ in [t for t in all_types(cfg.M, cfg.family) if t not in excluded]:
        for seed in seeds:
            ctx = SandboxEnv(cfg, forced_type=typ).reset(int(seed))
            enc = encoder(ctx.extra["raw"], ctx)
            rows.append(
                {
                    "seed": int(seed),
                    "type": typ,
                    "tau": best_response[typ],
                    "text": enc.text,
                    "briefing": ctx.extra["raw"].get("briefing"),
                    "features": np.asarray(enc.features, dtype=float).tolist(),
                }
            )
    return rows


def text_rows(config: SandboxConfig, held_out, best_response: dict[str, str]) -> list[dict]:
    cfg = config.normalized()
    excluded = set(held_out or ())
    rows = []
    for typ in [t for t in all_types(cfg.M, cfg.family) if t not in excluded]:
        for variants, text in all_renderings(typ, offset=0):
            rows.append({"type": typ, "variants": variants, "text": text, "tau": best_response[typ]})
    return rows


def heterogeneity_gate(values: dict[str, dict[str, float]], types: list[str], pool: list[str], best_response: dict[str, str]) -> dict:
    V = np.array([[values[t][tau] for tau in pool] for t in types], dtype=float)
    H_D = float(V.max(axis=1).mean() - V.mean(axis=0).max())
    gaps = {t: float(np.sort(V[i])[-1] - np.sort(V[i])[-2]) for i, t in enumerate(types)}
    distinct = sorted({best_response[t] for t in types})
    return {
        "types": list(types),
        "H_D": H_D,
        "n_distinct_best": len(distinct),
        "distinct_best": distinct,
        "min_gap": float(min(gaps.values())),
        "gaps": gaps,
        "passed": bool(H_D > 1e-9 and len(distinct) >= 3 and min(gaps.values()) > 1e-9),
    }


def gaussian_models(X: np.ndarray, y: np.ndarray) -> dict[str, dict[str, list[float]]]:
    models = {}
    for label in sorted(set(y.tolist())):
        rows = X[y == label]
        models[label] = {
            "mean": np.mean(rows, axis=0).tolist(),
            "var": (np.var(rows, axis=0) + 1e-3).tolist(),
        }
    return models


def model_callables(models: dict[str, dict[str, list[float]]]) -> dict[str, Callable]:
    callables = {}
    for label, params in models.items():
        mean = np.asarray(params["mean"], dtype=float)
        var = np.asarray(params["var"], dtype=float)

        def loglike(ctx, mean=mean, var=var):
            x = np.asarray(ctx.features, dtype=float).ravel()
            return float(-0.5 * np.sum(((x - mean) ** 2) / var + np.log(2.0 * np.pi * var)))

        callables[label] = loglike
    return callables


def build_lock(
    config: SandboxConfig,
    calib_seeds: list[int] | None = None,
    held_out: str | list[str] | None = None,
    path: str | Path | None = "sandbox_lock.json",
) -> dict:
    cfg = config.normalized()
    calib_seeds = calib_seeds or list(range(100, 120))
    X, y = calibration_rows(cfg, calib_seeds, held_out=held_out)
    basic_cfg = replace(cfg, features="basic")
    ctxucb_X, _ = calibration_rows(basic_cfg, calib_seeds, held_out=held_out)
    lock = {
        "config": {
            "M": cfg.M,
            "K": cfg.K,
            "delta": cfg.delta,
            "sharpness": cfg.sharpness,
            "reward_noise": cfg.reward_noise,
            "H": cfg.H,
            "k": cfg.k,
            "schedule": cfg.schedule,
            "features": cfg.features,
            "held_out": held_out,
        },
        "calibration_seeds": [int(seed) for seed in calib_seeds],
        "best_response": canonical_best_response(cfg),
        "fewshot_X": X.tolist(),
        "fewshot_y": y.tolist(),
        "ctxucb_X": ctxucb_X.tolist(),
        "gaussian_models": gaussian_models(X, y),
        "value_table": value_table(cfg),
    }
    if cfg.family == "traits":
        held = [held_out] if isinstance(held_out, str) else list(held_out or [])
        pool = strategy_pool(cfg.K, cfg.family)
        types = all_types(cfg.M, cfg.family)
        best = lock["best_response"]
        lock["config"].update({"family": cfg.family, "wording": cfg.wording, "encoding": cfg.encoding})
        lock["held_out"] = held
        lock["calibration_records"] = calibration_records(cfg, calib_seeds, held, best)
        lock["text_rows"] = text_rows(cfg, held, best)
        lock["gate_seen"] = heterogeneity_gate(lock["value_table"], [t for t in types if t not in held], pool, best)
        lock["gate_all"] = heterogeneity_gate(lock["value_table"], types, pool, best)
        lock["never_best"] = [tau for tau in pool if tau not in set(best.values())]
    if path is not None:
        Path(path).write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    return lock
