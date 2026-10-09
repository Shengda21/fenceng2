#!/usr/bin/env python3
"""Calibrate the MAgent regime-switching lock file on seeds 100-119."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0] / "v8_lib"))

from scripts.rs.magent_env import FEATURE_NAMES, POOL_SETS, MAgentRS, feature_fn
from scripts.rs.magent_types import TYPE_SETS
from scripts.rs.probes_rs import (
    REGISTERED_DEPLOYMENT_SEEDS,
    assert_same_population,
    build_episode_manifest,
    fixed_sweep,
    manifest_to_rows,
    oracle_h_d,
    per_window_oracle_replay,
    validate_calibration_seeds,
    validate_disjoint,
)


def main() -> None:
    args = parse_args()
    type_names = select_types(args.type_set, args.M, args.blue)
    pool = list(POOL_SETS[args.pool])
    env_cfg = {
        "map_size": args.map_size,
        "max_cycles": args.max_cycles,
        "type_set": type_names,
        "pool": tuple(pool),
        "default": args.default,
        "k": args.k,
        "switch": _schedule_to_switch(args.schedule),
        "delta_scale": args.delta_scale,
        "held_out": args.held_out,
        "force_fake": args.fake,
    }
    seeds = parse_seeds(args)
    validate_calibration_seeds(seeds)
    calibration_manifest = build_episode_manifest(env_cfg, seeds, split="calibration")
    deployment_manifest = build_episode_manifest(env_cfg, REGISTERED_DEPLOYMENT_SEEDS)
    validate_disjoint(seeds, deployment_manifest)
    sweep = fixed_sweep(env_cfg, calibration_manifest, pool)
    oracle = per_window_oracle_replay(
        env_cfg, calibration_manifest, pool, args.default, placebo=False, V_star=sweep["best_fixed_value"]
    )
    placebo = per_window_oracle_replay(
        env_cfg, calibration_manifest, pool, args.default, placebo=True, V_star=sweep["best_fixed_value"]
    )
    assert_same_population(sweep["manifest"], oracle["manifest"], placebo["manifest"])
    default_value = float(sweep["pooled_means"][args.default])
    if not np.isclose(oracle["V_default"], default_value, atol=1e-9, rtol=0.0):
        raise AssertionError(
            f"V_default mismatch: oracle={oracle['V_default']:.12g} value_table={default_value:.12g}"
        )
    v_star = float(sweep["best_fixed_value"])
    h_d = oracle_h_d(oracle["Delta_def"], placebo["Delta_def"], v_star, default_value)
    if h_d < -1e-9:
        raise AssertionError(f"H_D must be nonnegative, got {h_d:.12g}")
    pooled_min = min(float(v) for v in sweep["pooled_means"].values())
    headroom = v_star - pooled_min
    type_report = summarize_type_values(sweep["per_type_means"])
    pool_adequacy = len({row["best"] for row in type_report.values()})
    feature_rows = collect_feature_rows(env_cfg, calibration_manifest, type_names)
    type_models = fit_gaussians(feature_rows)
    diagnostics = classifier_diagnostics(env_cfg, seeds, type_names)
    lock = {
        "lock_version": "fixb-1",
        "pool_name": args.pool,
        "pool": pool,
        "default": args.default,
        "k": args.k,
        "encoder_mode": args.encoding,
        "permutation_seed": args.permutation_seed,
        "type_set": args.type_set,
        "types": type_names,
        "training_types": [t for t in type_names if t != args.held_out],
        "schedule": args.schedule,
        "switch": _schedule_to_switch(args.schedule),
        "map_size": args.map_size,
        "max_cycles": args.max_cycles,
        "delta_scale": args.delta_scale,
        "held_out": args.held_out,
        "calibration_seeds": seeds,
        "calibration_manifest": manifest_to_rows(calibration_manifest),
        "deployment_manifest": manifest_to_rows(deployment_manifest),
        "held_out_rule": "held_out type is excluded from all calibration rows and manifests.",
        "best_response_table": sweep["best_response_table"],
        "Delta_per_type": sweep["Delta_per_type"],
        "value_table": sweep["per_type_means"],
        "per_type_report": type_report,
        "pooled_means": sweep["pooled_means"],
        "V_star": v_star,
        "best_fixed": sweep["best_fixed"],
        "V_default": default_value,
        "Delta_def": oracle["Delta_def"],
        "placebo": placebo["Delta_def"],
        "H_D": h_d,
        "headroom_share": float(h_d / headroom) if headroom > 0.0 else 0.0,
        "pool_adequacy": pool_adequacy,
        "feature_names": list(FEATURE_NAMES),
        "feature_rows": feature_rows,
        "plastic_type_models": type_models,
        "classifier_diagnostics": diagnostics,
        "population_values": {
            "fixed": sweep["per_episode"],
            "default": oracle["base_per_episode"],
            "oracle_gap": oracle["gap_per_episode"],
            "placebo_gap": placebo["gap_per_episode"],
        },
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "file_hashes": file_hashes(),
    }
    require_heterogeneity(lock)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    for typ, row in type_report.items():
        values = ", ".join(f"{tau}={val:.6g}" for tau, val in row["values"].items())
        print(
            f"{typ}: {values}; best={row['best']} second={row['second_best']} gap={row['gap']:.6g}"
        )
    print(
        f"best_fixed={lock['best_fixed']} V_star={lock['V_star']:.6g} "
        f"V_default={lock['V_default']:.6g} Delta_def={lock['Delta_def']:.6g} "
        f"H_D={lock['H_D']:.6g} headroom_share={lock['headroom_share']:.6g} "
        f"pool_adequacy={lock['pool_adequacy']} placebo={lock['placebo']:.6g}"
    )


def collect_feature_rows(env_cfg: dict[str, Any], manifest: list[Any], type_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    held_out = env_cfg.get("held_out")
    for episode_index, spec in enumerate(manifest):
        typ = spec.type1 if hasattr(spec, "type1") else spec["type1"]
        if held_out == typ:
            continue
        env = MAgentRS(**{**env_cfg, "forced_type": typ, "forced_type2": getattr(spec, "type2", None), "switch_step": getattr(spec, "switch_step", None)})
        ctx = env.reset(int(spec.seed if hasattr(spec, "seed") else spec["seed"]), episode_index=episode_index)
        rows.append(
            {
                "episode_index": episode_index,
                "seed": int(spec.seed if hasattr(spec, "seed") else spec["seed"]),
                "type": typ,
                "type2": getattr(spec, "type2", None) if hasattr(spec, "type2") else spec.get("type2"),
                "switch_step": getattr(spec, "switch_step", None) if hasattr(spec, "switch_step") else spec.get("switch_step"),
                "features": [float(x) for x in np.asarray(ctx.features, dtype=float).ravel()],
                "text": ctx.text,
                "raw": env.last_raw,
            }
        )
    return rows


def fit_gaussians(feature_rows: list[dict[str, Any]], shrinkage: float = 1e-3) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[list[float]]] = {}
    for row in feature_rows:
        grouped.setdefault(row["type"], []).append(row["features"])
    models = {}
    for typ, vals in grouped.items():
        x = np.asarray(vals, dtype=float)
        mean = x.mean(axis=0)
        if len(x) <= 1:
            cov = np.eye(x.shape[1], dtype=float)
        else:
            cov = np.cov(x, rowvar=False)
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]])
        scale = float(np.trace(cov) / max(1, cov.shape[0]))
        cov = cov + np.eye(cov.shape[0]) * max(shrinkage, shrinkage * scale)
        inv = np.linalg.pinv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        models[typ] = {
            "mean": mean.tolist(),
            "cov": cov.tolist(),
            "inv_cov": inv.tolist(),
            "logdet": float(logdet if sign > 0 else np.log(max(np.linalg.det(cov), shrinkage))),
        }
    return models


def summarize_type_values(per_type_means: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    out = {}
    for typ, means in per_type_means.items():
        ordered = sorted(means.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        best = ordered[0][0] if ordered else None
        second = ordered[1][0] if len(ordered) > 1 else None
        gap = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else 0.0
        out[typ] = {
            "values": {tau: float(value) for tau, value in means.items()},
            "best": best,
            "second_best": second,
            "gap": gap,
        }
    return out


def classifier_diagnostics(env_cfg: dict[str, Any], seeds: list[int], type_names: list[str]) -> dict[str, Any]:
    out = {}
    for k in (10, 20, 30):
        cfg = {**env_cfg, "k": k}
        manifest = build_episode_manifest(cfg, seeds, split="calibration")
        rows = collect_feature_rows(cfg, manifest, type_names)
        labels = sorted({row["type"] for row in rows})
        if not labels:
            continue
        x = np.asarray([row["features"] for row in rows], dtype=float)
        y = [row["type"] for row in rows]
        centroids = {label: x[[i for i, yy in enumerate(y) if yy == label]].mean(axis=0) for label in labels}
        confusion = {truth: {pred: 0 for pred in labels} for truth in labels}
        for row, feats in zip(rows, x):
            pred = min(labels, key=lambda label: float(np.linalg.norm(feats - centroids[label])))
            confusion[row["type"]][pred] += 1
        for encoding in ("semantic", "numeric", "opaque", "permuted"):
            out.setdefault(encoding, {})[str(k)] = confusion
    return out


def require_heterogeneity(lock: dict[str, Any], epsilon: float = 1e-9) -> None:
    if float(lock.get("H_D", 0.0)) <= epsilon:
        raise AssertionError("heterogeneity gate failed: H_D is not positive")
    reports = lock.get("per_type_report", {})
    winners = {row.get("best") for row in reports.values() if float(row.get("gap", 0.0)) > epsilon}
    if len(winners) < 2:
        raise AssertionError("heterogeneity gate failed: fewer than two separated best responses")


def parse_seeds(args) -> list[int]:
    if not args.seeds:
        return list(range(args.seed_start, args.seed_end + 1))
    spec = str(args.seeds).strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def select_types(type_set: str, M: int, blue: str) -> list[str]:
    if blue == "random":
        return ["random"]
    names = list(TYPE_SETS[type_set])
    if M < 1 or M > len(names):
        raise ValueError(f"M must be between 1 and {len(names)} for {type_set}")
    return names[:M]


def file_hashes() -> dict[str, str]:
    paths = [
        ROOT / "scripts" / "rs" / "magent_types.py",
        ROOT / "scripts" / "rs" / "magent_env.py",
        ROOT / "scripts" / "rs" / "probes_rs.py",
        ROOT / "scripts" / "rs" / "calibrate_rs_magent.py",
        ROOT / "scripts" / "rs" / "run_rs_magent.py",
        ROOT.parents[0] / "v8_lib" / "SPEC.md",
    ]
    out = {}
    for path in paths:
        if path.exists():
            out[str(path.relative_to(ROOT.parents[0]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--type-set", choices=["base4", "traits8"], default="base4")
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--blue", choices=["scripted", "random"], default="scripted")
    p.add_argument("--schedule", choices=["single", "mid", "block20"], default="single")
    p.add_argument("--map-size", type=int, default=20)
    p.add_argument("--max-cycles", type=int, default=100)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--delta-scale", type=float, default=1.0)
    p.add_argument("--held-out")
    p.add_argument("--pool", choices=sorted(POOL_SETS), default="pool8")
    p.add_argument("--default", default="HOLD_POSITION")
    p.add_argument("--encoding", choices=["semantic", "numeric", "opaque", "permuted"], default="semantic")
    p.add_argument("--permutation-seed", type=int, default=0)
    p.add_argument("--seeds", help="Calibration seeds as an inclusive range like 100-139 or a comma list.")
    p.add_argument("--seed-start", type=int, default=100)
    p.add_argument("--seed-end", type=int, default=119)
    p.add_argument("--fake", action="store_true", help="Use the deterministic fake adapter even if magent2 is installed.")
    p.add_argument("--out", type=Path, default=ROOT / "rs_magent_lock.json")
    return p.parse_args()


def _schedule_to_switch(schedule: str) -> str | None:
    return None if schedule == "single" else schedule


if __name__ == "__main__":
    main()
