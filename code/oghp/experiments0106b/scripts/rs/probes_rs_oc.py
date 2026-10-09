"""Probe utilities for the Overcooked replay-selection experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
V8_ROOT = ROOT.parent.parent / "v8_lib"
for path in (ROOT, V8_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from v8lib.probes import best_fixed, fixed_strategy_sweep, per_window_oracle

try:
    from .overcooked_roles import ROLES
    from .overcooked_rs import OvercookedRS
    from .overcooked_types import ALL_PARTNER_TYPES, PARTNER_TYPES
except ImportError:  # pragma: no cover - direct script fallback
    from overcooked_roles import ROLES  # type: ignore
    from overcooked_rs import OvercookedRS  # type: ignore
    from overcooked_types import ALL_PARTNER_TYPES, PARTNER_TYPES  # type: ignore


def make_typed_env(layout: str, ptype: str, horizon: int = 400, schedule: str = "single"):
    switch = "cadence20" if schedule == "cadence20" else None
    return lambda: OvercookedRS(layout, ptype_set=(ptype,), horizon=horizon, switch=switch)


def fixed_sweep_per_type(
    layout: str,
    seeds: list[int],
    horizon: int = 400,
    ptypes: tuple[str, ...] = PARTNER_TYPES,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ptype in ptypes:
        sweep = fixed_strategy_sweep(make_typed_env(layout, ptype, horizon), list(ROLES), seeds)
        tau, mean = best_fixed(sweep)
        out[ptype] = {"sweep": sweep, "best": tau, "best_mean": mean}
    return out


def replay_oracle_and_placebo(
    layout: str,
    seeds: list[int],
    horizon: int = 400,
    ptypes: tuple[str, ...] = PARTNER_TYPES,
    default: str = "ROLE_FULL",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ptype in ptypes:
        make_env = make_typed_env(layout, ptype, horizon)
        oracle = per_window_oracle(make_env, list(ROLES), default, seeds, placebo=False)
        placebo = per_window_oracle(make_env, list(ROLES), default, seeds, placebo=True)
        out[ptype] = {
            "oracle": oracle,
            "placebo": placebo,
            "placebo_gap": float(placebo["Delta_def"]),
        }
    return out


def run_all_probes(
    layouts: list[str],
    seeds: list[int],
    horizon: int = 400,
    include_held_out: bool = False,
) -> dict[str, Any]:
    ptypes = ALL_PARTNER_TYPES if include_held_out else PARTNER_TYPES
    return {
        layout: {
            "fixed_sweep": fixed_sweep_per_type(layout, seeds, horizon, ptypes),
            "oracle": replay_oracle_and_placebo(layout, seeds, horizon, ptypes),
        }
        for layout in layouts
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", nargs="+", default=["cramped_room", "asymmetric_advantages"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("rs_oc_probes.json"))
    args = parser.parse_args()

    result = run_all_probes(args.layouts, args.seeds, args.horizon, args.include_held_out)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
