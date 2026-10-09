"""Calibration split builder for Overcooked regime-switching experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V8_ROOT = ROOT.parent.parent / "v8_lib"
for path in (ROOT, V8_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from v8lib.probes import best_fixed, fixed_strategy_sweep, per_window_oracle

try:
    from .overcooked_roles import ROLES
    from .overcooked_rs import FEATURE_NAMES, OvercookedRS, feature_vector
    from .overcooked_types import ALL_PARTNER_TYPES, PARTNER_TYPES
except ImportError:  # pragma: no cover - direct script fallback
    from overcooked_roles import ROLES  # type: ignore
    from overcooked_rs import FEATURE_NAMES, OvercookedRS, feature_vector  # type: ignore
    from overcooked_types import ALL_PARTNER_TYPES, PARTNER_TYPES  # type: ignore


def gaussian_models(feature_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    labels = sorted({r["ptype"] for r in feature_rows})
    for label in labels:
        X = np.asarray([r["features"] for r in feature_rows if r["ptype"] == label], dtype=float)
        mean = X.mean(axis=0)
        var = X.var(axis=0) + 1e-6
        out[label] = {"mean": mean.tolist(), "var": var.tolist()}
    return out


def collect_feature_rows(layout: str, seeds: list[int], ptypes: tuple[str, ...], horizon: int) -> list[dict[str, Any]]:
    rows = []
    for ptype in ptypes:
        env = OvercookedRS(layout, ptype_set=(ptype,), horizon=horizon)
        try:
            for seed in seeds:
                ctx = env.reset(seed)
                raw = ctx.extra["raw"]
                rows.append(
                    {
                        "layout": layout,
                        "seed": seed,
                        "ptype": ptype,
                        "features": feature_vector(raw).tolist(),
                        "raw": raw,
                    }
                )
        finally:
            env.close()
    return rows


def calibrate_layout(layout: str, seeds: list[int], horizon: int, ptypes: tuple[str, ...]) -> dict[str, Any]:
    sweeps: dict[str, Any] = {}
    best_response: dict[str, str] = {}
    delta_by_type: dict[str, Any] = {}
    for ptype in ptypes:
        make_env = lambda p=ptype: OvercookedRS(layout, ptype_set=(p,), horizon=horizon)
        sweep = fixed_strategy_sweep(make_env, list(ROLES), seeds)
        best_tau, best_mean = best_fixed(sweep)
        oracle = per_window_oracle(make_env, list(ROLES), "ROLE_FULL", seeds, placebo=False)
        sweeps[ptype] = {"sweep": sweep, "best": best_tau, "best_mean": best_mean}
        best_response[ptype] = best_tau
        delta_by_type[ptype] = {
            "Delta_def": oracle["Delta_def"],
            "placebo": oracle["placebo"],
            "V_star": oracle["V_star"],
            "V_default": oracle["V_default"],
            "H_D": oracle["H_D"],
        }
    feature_rows = collect_feature_rows(layout, seeds, ptypes, horizon)
    return {
        "layout": layout,
        "seeds": seeds,
        "horizon": horizon,
        "pool": list(ROLES),
        "feature_names": FEATURE_NAMES,
        "best_response": best_response,
        "sweeps": sweeps,
        "Delta": delta_by_type,
        "V_star": float(np.mean([v["best_mean"] for v in sweeps.values()])),
        "H_D": float(np.mean([v["H_D"] for v in delta_by_type.values()])),
        "feature_rows": feature_rows,
        "gaussian_type_models": gaussian_models(feature_rows),
    }


def calibrate(
    layouts: list[str],
    seeds: list[int] | None = None,
    horizon: int = 400,
    include_held_out: bool = False,
) -> dict[str, Any]:
    seeds = list(range(100, 120)) if seeds is None else list(seeds)
    ptypes = ALL_PARTNER_TYPES if include_held_out else PARTNER_TYPES
    return {
        "meta": {
            "calibration_seeds": seeds,
            "ptype_set": list(ptypes),
            "default": "ROLE_FULL",
        },
        "layouts": {layout: calibrate_layout(layout, seeds, horizon, ptypes) for layout in layouts},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", nargs="+", default=["cramped_room", "asymmetric_advantages"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("rs_oc_lock.json"))
    args = parser.parse_args()

    lock = calibrate(args.layouts, args.seeds, args.horizon, args.include_held_out)
    args.out.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
