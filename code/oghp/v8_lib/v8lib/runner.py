"""Deployment runner and seed split helpers."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Callable

from v8lib.context import Context, redact_hidden_context
from v8lib.probes import EpisodeEnv


def calibration_split(seeds=None) -> tuple[list[int], list[int]]:
    """Return calibration seeds 100-119 and held-out seeds 0-99."""

    if seeds is None:
        return list(range(100, 120)), list(range(100))
    values = [int(s) for s in seeds]
    calibration = [s for s in values if 100 <= s <= 119]
    held_out = [s for s in values if 0 <= s <= 99]
    return calibration, held_out


def assert_disjoint(deploy_seeds, calibration_seeds) -> None:
    """Raise if deployment and calibration seed manifests overlap."""

    deploy = {int(s) for s in deploy_seeds}
    calibration = {int(s) for s in calibration_seeds}
    overlap = sorted(deploy & calibration)
    if overlap:
        raise ValueError(f"deployment seeds overlap calibration seeds: {overlap}")


def run_deployment(
    make_env: Callable[[], EpisodeEnv],
    selector,
    seeds,
    T: int,
    encoder=None,
    log_path=None,
    calibration_seeds=None,
) -> list[dict]:
    """Run a selector for `T` episodes and optionally write JSON lines."""

    seed_list, manifest_calibration = _resolve_seeds(seeds)
    if calibration_seeds is None:
        calibration_seeds = manifest_calibration
    if calibration_seeds is not None and _uses_range_object(seeds):
        raise ValueError("deployment seeds must come from an explicit held-out manifest, not range(T)")
    if calibration_seeds is not None:
        assert_disjoint(seed_list, calibration_seeds)
    if not seed_list:
        raise ValueError("seeds must not be empty")
    records = []
    selector.reset_run(int(seed_list[0]))
    for episode in range(int(T)):
        seed = int(seed_list[episode % len(seed_list)])
        env = make_env()
        ctx = _encode_if_needed(env.reset(seed), encoder)
        true_type = ctx.type_label
        visible = _visible_ctx(ctx, selector)
        selector.reset_episode(visible)
        windows, provenance = [], []
        total_reward = 0.0
        prefix_total = 0.0
        done = False
        last_info = {}
        while not done:
            _observe_if_supported(selector, visible)
            start = perf_counter()
            tau = selector.select(visible)
            prov = getattr(selector, "last_provenance", None)
            if prov is None or prov.get("episode") != visible.episode_index or prov.get("window") != visible.window_index:
                prov = _fallback_provenance(selector, visible, tau, perf_counter() - start)
            if tau not in visible.pool:
                tau = visible.default
                prov = dict(prov)
                prov.update({"tau": tau, "in_pool": False, "fallback": True})
            next_ctx, reward, done, info = env.step_window(tau)
            last_info = info or {}
            window_reward = float(reward)
            prefix_reward = float(last_info.get("prefix_reward", 0.0))
            prefix_total += prefix_reward
            total_reward += window_reward + prefix_reward
            single_decision = done and not windows
            reward_scope = "episode" if single_decision else "window"
            update_reward = window_reward + prefix_reward if single_decision else window_reward
            update_info = dict(last_info)
            update_info["reward_scope"] = reward_scope
            selector.update(visible, tau, update_reward, bool(done), update_info)
            windows.append(
                {
                    "window": visible.window_index,
                    "tau": tau,
                    "reward": window_reward,
                    "update_reward": update_reward,
                    "reward_scope": reward_scope,
                    "prefix_reward": prefix_reward,
                    "fallback": bool(prov.get("fallback", False)),
                }
            )
            provenance.append(prov)
            if next_ctx is not None:
                visible = _visible_ctx(_encode_if_needed(next_ctx, encoder), selector)
        records.append(
            {
                "method": selector.name,
                "episode": episode,
                "seed": seed,
                "type": true_type,
                "windows": windows,
                "total_reward": total_reward,
                "prefix_reward": prefix_total,
                "reward_scope": "episode" if len(windows) == 1 else "window",
                "fallback_count": sum(1 for window in windows if window["fallback"]),
                "success": bool(last_info.get("success", False)),
                "provenance": provenance,
            }
        )
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def _encode_if_needed(ctx: Context, encoder):
    if encoder is None:
        return ctx
    raw = ctx.extra.get("raw") if isinstance(ctx.extra, dict) else None
    return encoder(raw, ctx) if raw is not None else ctx


def _visible_ctx(ctx: Context, selector) -> Context:
    return ctx if getattr(selector, "reads_type_label", False) else redact_hidden_context(ctx)


def _observe_if_supported(selector, ctx: Context) -> None:
    observe = getattr(selector, "observe", None)
    if callable(observe):
        observe(ctx)


def _resolve_seeds(seeds) -> tuple[list[int], list[int] | None]:
    if isinstance(seeds, dict):
        deploy = seeds.get("deploy_seeds", seeds.get("deployment_seeds"))
        if deploy is None:
            raise ValueError("seed manifest must include deploy_seeds")
        calibration = seeds.get("calibration_seeds")
        return [int(s) for s in deploy], None if calibration is None else [int(s) for s in calibration]
    if seeds is None:
        raise ValueError("explicit deployment seeds are required")
    return [int(s) for s in seeds], None


def _uses_range_object(seeds) -> bool:
    if isinstance(seeds, range):
        return True
    if isinstance(seeds, dict):
        deploy = seeds.get("deploy_seeds", seeds.get("deployment_seeds"))
        return isinstance(deploy, range)
    return False


def _fallback_provenance(selector, ctx: Context, tau: str, latency_s: float) -> dict:
    in_pool = tau in ctx.pool
    return {
        "arm": selector.name,
        "episode": ctx.episode_index,
        "window": ctx.window_index,
        "tau": tau if in_pool else ctx.default,
        "in_pool": in_pool,
        "fallback": not in_pool,
        "latency_s": latency_s,
        "extra": {},
    }
