"""Replay-based probes for MAgent regime-switching experiments."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from scripts.rs.magent_env import EpisodeSpec, MAgentRS


REGISTERED_CALIBRATION_SEEDS = tuple(range(1000, 1020))
REGISTERED_CALIBRATION_SEEDS_EXTENDED = tuple(range(1000, 1040))
REGISTERED_DEPLOYMENT_SEEDS = tuple(range(0, 300))  # paper: 0-99 headline, 100-299 extend c(T) for learners; calibration seeds live at 1000+


def fixed_sweep(env_cfg: dict[str, Any], seeds, pool) -> dict[str, Any]:
    """Run every tau as a fixed post-prefix strategy on one locked population."""

    pool = list(pool)
    manifest = coerce_manifest(seeds, env_cfg)
    by_tau: dict[str, dict[str, Any]] = {}
    by_type: dict[str, dict[str, list[float]]] = {}
    rows = []
    per_episode: dict[str, dict[str, float]] = {}
    for tau in pool:
        returns = []
        values_by_key = {}
        for episode_index, spec in enumerate(manifest):
            env = _make_env(env_cfg, spec=spec)
            row = env.replay(spec.seed, lambda ctx, _env, tau=tau: tau)
            value = float(row["total_reward"])
            returns.append(value)
            values_by_key[spec.key] = value
            by_type.setdefault(spec.type1, {}).setdefault(tau, []).append(value)
            rows.append({**row, "episode_index": episode_index, "manifest_key": spec.key, "fixed_tau": tau})
        per_episode[tau] = values_by_key
        by_tau[tau] = {
            "mean": float(np.mean(returns)) if returns else 0.0,
            "per_seed": returns,
        }
    per_type_means = {
        typ: {
            tau: float(np.mean(vals)) if vals else 0.0
            for tau, vals in tau_map.items()
        }
        for typ, tau_map in by_type.items()
        if any(tau_map[tau] for tau in tau_map)
    }
    best_response_table = {
        typ: max(means, key=lambda tau: (means[tau], tau))
        for typ, means in per_type_means.items()
    }
    delta_per_type = {}
    for typ, means in per_type_means.items():
        ordered = sorted(means.values(), reverse=True)
        delta_per_type[typ] = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
    best_fixed = max(by_tau, key=lambda tau: (by_tau[tau]["mean"], tau)) if by_tau else None
    return {
        "by_tau": by_tau,
        "per_type_means": per_type_means,
        "pooled_means": {tau: by_tau[tau]["mean"] for tau in by_tau},
        "best_fixed": best_fixed,
        "best_fixed_value": by_tau[best_fixed]["mean"] if best_fixed else 0.0,
        "best_response_table": best_response_table,
        "Delta_per_type": delta_per_type,
        "rows": rows,
        "manifest": manifest_to_rows(manifest),
        "per_episode": per_episode,
    }


def per_window_oracle_replay(
    env_cfg: dict[str, Any],
    seeds,
    pool,
    default: str,
    placebo: bool = False,
    V_star: float | None = None,
) -> dict[str, Any]:
    """Estimate per-window oracle lift by replaying one locked population."""

    pool = list(pool)
    manifest = coerce_manifest(seeds, env_cfg)
    gaps_by_seed = []
    rows = []
    base_values = []
    base_per_episode = {}
    gap_per_episode = {}
    for episode_index, spec in enumerate(manifest):
        base_env = _make_env(env_cfg, spec=spec)
        base = base_env.replay(spec.seed, lambda ctx, _env: default)
        base_reward = float(base["total_reward"])
        base_values.append(base_reward)
        base_per_episode[spec.key] = base_reward
        seed_gaps = []
        n_windows = max(1, len(base["decisions"]))
        for window_index in range(n_windows):
            branch_pool = [default for _ in pool] if placebo else pool
            branch_values = []
            for tau in branch_pool:
                env = _make_env(env_cfg, spec=spec)

                def choose(ctx, _env, tau=tau, window_index=window_index):
                    return tau if ctx.window_index == window_index else default

                row = env.replay(spec.seed, choose)
                branch_values.append(float(row["total_reward"]))
            gap = float(max(branch_values) - base_reward)
            seed_gaps.append(gap)
            rows.append(
                {
                    "episode_index": episode_index,
                    "seed": spec.seed,
                    "type": spec.type1,
                    "type2": spec.type2,
                    "switch_step": spec.switch_step,
                    "manifest_key": spec.key,
                    "window": window_index,
                    "base": base_reward,
                    "gap": gap,
                    "placebo": bool(placebo),
                }
            )
        gaps_by_seed.append(seed_gaps)
        gap_per_episode[spec.key] = float(sum(seed_gaps))
    delta_def = float(np.mean([sum(xs) for xs in gaps_by_seed])) if gaps_by_seed else 0.0
    v_default = float(np.mean(base_values)) if base_values else 0.0
    if V_star is None:
        sweep = fixed_sweep(env_cfg, seeds, pool)
        V_star = float(sweep["best_fixed_value"])
    placebo_value = delta_def if placebo else 0.0
    h_d = float(delta_def - (float(V_star) - v_default))
    return {
        "Delta_def": delta_def,
        "V_default": v_default,
        "V_star": float(V_star),
        "placebo": placebo_value,
        "H_D": h_d,
        "per_seed": gaps_by_seed,
        "per_window": rows,
        "manifest": manifest_to_rows(manifest),
        "base_per_episode": base_per_episode,
        "gap_per_episode": gap_per_episode,
    }


def oracle_h_d(delta_def: float, placebo: float, v_star: float, v_default: float) -> float:
    return float((float(delta_def) - float(placebo)) - (float(v_star) - float(v_default)))


def build_episode_manifest(
    env_cfg: dict[str, Any],
    seeds: Iterable[int],
    *,
    split: str | None = None,
) -> list[EpisodeSpec]:
    """Materialise the ordered seed/type/switch population once."""

    seed_list = [int(s) for s in seeds]
    if split == "calibration":
        validate_calibration_seeds(seed_list)
    type_names = _type_names(env_cfg)
    out = []
    for seed in seed_list:
        env = MAgentRS(**_constructor_cfg(env_cfg))
        typ, typ2 = env._sample_types(seed)
        switch_step = 50 if env.switch == "mid" else None
        out.append(EpisodeSpec(seed=seed, type1=str(typ), type2=typ2, switch_step=switch_step))
    return out


def coerce_manifest(spec_or_seeds, env_cfg: dict[str, Any]) -> list[EpisodeSpec]:
    vals = list(spec_or_seeds)
    if not vals:
        return []
    first = vals[0]
    if isinstance(first, EpisodeSpec) or isinstance(first, dict):
        return [_coerce_spec(v) for v in vals]
    return build_episode_manifest(env_cfg, [int(v) for v in vals])


def manifest_to_rows(manifest: list[EpisodeSpec]) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(spec.seed),
            "type1": spec.type1,
            "type2": spec.type2,
            "switch_step": spec.switch_step,
            "key": spec.key,
        }
        for spec in manifest
    ]


def validate_calibration_seeds(seeds: list[int]) -> None:
    allowed = {REGISTERED_CALIBRATION_SEEDS, REGISTERED_CALIBRATION_SEEDS_EXTENDED}
    if tuple(seeds) not in allowed:
        raise ValueError("calibration seeds must be exactly 1000-1019 or 1000-1039 (deployment manifest is 0-299)")


def validate_disjoint(calibration_seeds: Iterable[int], deployment_manifest: Iterable[EpisodeSpec | dict[str, Any]]) -> None:
    cal = {int(s) for s in calibration_seeds}
    dep = {int(_coerce_spec(spec).seed) for spec in deployment_manifest}
    overlap = sorted(cal & dep)
    if overlap:
        raise ValueError(f"deployment seeds overlap calibration seeds: {overlap}")


def assert_same_population(*manifests: list[EpisodeSpec] | list[dict[str, Any]]) -> None:
    if not manifests:
        return
    keys = [[_coerce_spec(spec).key for spec in manifest] for manifest in manifests]
    first = keys[0]
    for cur in keys[1:]:
        if cur != first:
            raise AssertionError("episode populations differ")


def _make_env(env_cfg: dict[str, Any], forced_type: str | None = None, spec: EpisodeSpec | dict[str, Any] | None = None) -> MAgentRS:
    cfg = dict(env_cfg)
    cfg = _constructor_cfg(cfg)
    if spec is not None:
        episode = _coerce_spec(spec)
        cfg["forced_type"] = episode.type1
        cfg["forced_type2"] = episode.type2
        cfg["switch_step"] = episode.switch_step
    elif forced_type is not None:
        cfg["forced_type"] = forced_type
    return MAgentRS(**cfg)


def _constructor_cfg(env_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(env_cfg)
    cfg.pop("M", None)
    cfg.pop("encoding", None)
    cfg.pop("type_hints", None)
    cfg.pop("blue", None)
    cfg.pop("deployment_manifest", None)
    cfg.pop("calibration_manifest", None)
    return cfg


def _coerce_spec(spec: EpisodeSpec | dict[str, Any]) -> EpisodeSpec:
    if isinstance(spec, EpisodeSpec):
        return spec
    return EpisodeSpec(
        seed=int(spec["seed"]),
        type1=str(spec.get("type1", spec.get("type"))),
        type2=None if spec.get("type2") is None else str(spec.get("type2")),
        switch_step=None if spec.get("switch_step") is None else int(spec.get("switch_step")),
    )


def _type_names(env_cfg: dict[str, Any]) -> list[str]:
    type_set = env_cfg.get("type_set", "base4")
    if isinstance(type_set, str):
        from scripts.rs.magent_types import TYPE_SETS

        return list(TYPE_SETS[type_set])
    return [str(t) for t in type_set]
