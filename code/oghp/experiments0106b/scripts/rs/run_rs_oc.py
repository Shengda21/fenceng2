"""CLI runner for Overcooked replay-selection deployments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V8_ROOT = ROOT.parent.parent / "v8_lib"
for path in (ROOT, V8_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from scripts.diagnostic._diag_common import save_rows  # noqa: E402
from v8lib.arms import (  # noqa: E402
    FewShotClassifierSelector,
    FixedSelector,
    LLMSelector,
    LinTS,
    LinUCB,
    MUCB,
    PLASTICPolicySelector,
    PPOScheduler,
    RandomSelector,
    ScriptedDetector,
    TypeOracleSelector,
)
from v8lib.encoders import make_encoder  # noqa: E402

try:
    from .overcooked_roles import ROLE_FULL, ROLES
    from .overcooked_rs import FEATURE_NAMES, OvercookedRS, feature_vector, run_episode_with_selector, semantic_template
    from .overcooked_types import COUNTER_PASSER, PARTNER_TYPES
except ImportError:  # pragma: no cover - direct script fallback
    from overcooked_roles import ROLE_FULL, ROLES  # type: ignore
    from overcooked_rs import FEATURE_NAMES, OvercookedRS, feature_vector, run_episode_with_selector, semantic_template  # type: ignore
    from overcooked_types import COUNTER_PASSER, PARTNER_TYPES  # type: ignore


def load_lock(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def layout_lock(lock: dict[str, Any], layout: str) -> dict[str, Any]:
    return lock.get("layouts", {}).get(layout, {})


def best_response_from_lock(lock: dict[str, Any], layout: str) -> dict[str, str]:
    br = layout_lock(lock, layout).get("best_response", {})
    return {str(k): str(v) for k, v in br.items()} if br else {ptype: ROLE_FULL for ptype in PARTNER_TYPES}


def value_table_from_lock(lock: dict[str, Any], layout: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for ptype, payload in layout_lock(lock, layout).get("sweeps", {}).items():
        out[ptype] = {tau: float(v["mean"]) for tau, v in payload.get("sweep", {}).items()}
    return out


def gaussian_type_models(lock: dict[str, Any], layout: str) -> dict[str, Callable[[Any], float]]:
    models = layout_lock(lock, layout).get("gaussian_type_models", {})

    def make_model(mean: list[float], var: list[float]) -> Callable[[Any], float]:
        m = np.asarray(mean, dtype=float)
        v = np.asarray(var, dtype=float)

        def logp(ctx: Any) -> float:
            x = np.asarray(ctx.features, dtype=float).ravel()
            n = min(len(x), len(m))
            return float(-0.5 * np.sum(((x[:n] - m[:n]) ** 2) / v[:n] + np.log(v[:n])))

        return logp

    return {label: make_model(payload["mean"], payload["var"]) for label, payload in models.items()}


def feature_training_rows(lock: dict[str, Any], layout: str) -> tuple[list[list[float]], list[str]]:
    rows = layout_lock(lock, layout).get("feature_rows", [])
    return [r["features"] for r in rows], [r["ptype"] for r in rows]


def scripted_rule(ctx: Any) -> str:
    raw = ctx.extra.get("raw", {})
    pickups = raw.get("partner_pickups", {})
    held = raw.get("partner_held", "empty")
    serves = raw.get("partner_serves", 0)
    if serves > 0 or pickups.get("soup", 0) > 0:
        return "ROLE_ONION"
    if pickups.get("onion", 0) >= max(1, pickups.get("dish", 0) + 1):
        return "ROLE_SERVE"
    if pickups.get("dish", 0) > pickups.get("onion", 0) or held == "dish":
        return "ROLE_ONION"
    if raw.get("partner_stay_frac", 0.0) > 0.8:
        return "ROLE_FULL"
    return "ROLE_FULL"


def make_selector(arm: str, layout: str, lock: dict[str, Any], seed: int, horizon: int) -> Any:
    pool = list(ROLES)
    best_response = best_response_from_lock(lock, layout)
    if arm == "random":
        return RandomSelector(seed=seed)
    if arm.startswith("fixed:"):
        return FixedSelector(arm.split(":", 1)[1])
    if arm == "type_oracle":
        return TypeOracleSelector(best_response)
    if arm == "scripted":
        return ScriptedDetector(scripted_rule)
    if arm == "mucb":
        return MUCB(pool=pool)
    if arm == "linucb":
        return LinUCB()
    if arm == "lints":
        return LinTS()
    if arm.startswith("fewshot:"):
        n = int(arm.split(":", 1)[1])
        selector = FewShotClassifierSelector(n, best_response, model="knn", seed=seed)
        X, y = feature_training_rows(lock, layout)
        if X and y:
            selector.fit(X, y)
        return selector
    if arm == "plastic":
        models = gaussian_type_models(lock, layout)
        if not models:
            models = {ptype: (lambda ctx: 0.0) for ptype in PARTNER_TYPES}
        return PLASTICPolicySelector(models, best_response, value_table=value_table_from_lock(lock, layout))
    if arm == "ppo":
        selector = PPOScheduler(pool=pool, feature_dim=len(FEATURE_NAMES))
        train_eps = min(20, max(0, horizon // 20))
        if train_eps:
            selector.train(lambda: OvercookedRS(layout, horizon=horizon), train_eps, seed=seed + 10_000)
        return selector
    if arm.startswith("llm:"):
        parts = arm.split(":")
        model = parts[1]
        mode = parts[2] if len(parts) > 2 else "single"
        return LLMSelector(model=model, mode=mode)
    raise ValueError(f"unknown arm: {arm}")


def make_schedule(schedule: str) -> str | None:
    if schedule == "single":
        return None
    if schedule == "mid":
        return "mid"
    if schedule == "cadence20":
        return "cadence20"
    raise ValueError(f"unknown schedule: {schedule}")


def run(
    layout: str,
    arm: str,
    encoding: str,
    schedule: str,
    T: int,
    out: Path,
    held_out: str | None = None,
    seed_base: int = 0,
    horizon: int = 400,
    lock_path: Path | None = Path("rs_oc_lock.json"),
) -> list[dict[str, Any]]:
    lock = load_lock(lock_path)
    selector = make_selector(arm, layout, lock, seed_base, horizon)
    selector.reset_run(seed_base)
    encoder = make_encoder(
        encoding,
        semantic_template,
        feature_vector,
        list(PARTNER_TYPES) + ([COUNTER_PASSER] if held_out else []),
        permutation_seed=seed_base,
        feature_names=FEATURE_NAMES,
    )
    rows = []
    for ep in range(T):
        seed = seed_base + ep
        env = OvercookedRS(layout, horizon=horizon, switch=make_schedule(schedule), held_out=held_out)
        try:
            row = run_episode_with_selector(env, seed, selector, encoder=encoder)
            row["method"] = arm
            row["layout"] = layout
            row["encoding"] = encoding
            row["schedule"] = schedule
        except Exception as exc:  # noqa: BLE001
            row = {
                "episode": ep,
                "seed": seed,
                "ptype": None,
                "ptype2": None,
                "decisions": [],
                "total_reward": 0.0,
                "success": False,
                "steps": 0,
                "provenance": [{"error": str(exc)}],
                "method": arm,
                "layout": layout,
                "encoding": encoding,
                "schedule": schedule,
            }
        finally:
            env.close()
        row["episode"] = ep
        rows.append(row)
    save_rows(rows, out, meta={"layout": layout, "arm": arm, "encoding": encoding, "schedule": schedule})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--encoding", choices=["semantic", "numeric", "opaque", "permuted"], default="semantic")
    parser.add_argument("--schedule", choices=["single", "mid", "cadence20"], default="single")
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--held-out", choices=[COUNTER_PASSER], default=None)
    parser.add_argument("--lock", type=Path, default=Path("rs_oc_lock.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run(
        layout=args.layout,
        arm=args.arm,
        encoding=args.encoding,
        schedule=args.schedule,
        T=args.T,
        out=args.out,
        held_out=args.held_out,
        seed_base=args.seed_base,
        horizon=args.horizon,
        lock_path=args.lock,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
