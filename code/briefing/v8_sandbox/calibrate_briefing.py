"""Write the calibration locks for the four base cells and a summary of the calibrated values."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from closed_form import delta_min, exact_reference, predicted_T_star, sigma_eff, snr
from sandbox.briefing import resolve_held_out_set
from sandbox.calibrate import build_lock
from sandbox.env import SandboxConfig, SandboxEnv
from sandbox.game import parse_trait

HERE = Path(__file__).resolve().parent
SOURCES = [
    "sandbox/game.py", "sandbox/env.py", "sandbox/briefing.py", "sandbox/calibrate.py", "sandbox/selectors.py",
    "run_sandbox.py", "closed_form.py", "analyze_briefing.py", "../v8_lib/v8lib/arms/text.py", "../v8_lib/v8lib/runner.py",
    "../v8_lib/v8lib/context.py", "sandbox/embed_cache_bge-small-en-v1.5.npz",
]
BASE_CELLS = [(9, 5, 0.8, 0.0), (9, 5, 0.8, 0.5), (18, 10, 0.8, 0.0), (18, 10, 0.8, 0.5)]


def file_hashes() -> dict:
    return {s: hashlib.sha256((HERE / s).read_bytes()).hexdigest() for s in SOURCES}


def manifest(cfg: SandboxConfig, seeds: list[int], held: list[str]) -> dict:
    types = []
    for seed in seeds:
        env = SandboxEnv(cfg)
        env.reset(seed)
        types.append(env.type_label)
    return {
        "types": dict(Counter(types)),
        "seen": sum(t not in held for t in types),
        "held_out": sum(t in held for t in types),
        "reactivity": dict(Counter(parse_trait(t)[1] for t in types)),
        "held_out_by_reactivity": dict(Counter(parse_trait(t)[1] for t in types if t in held)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sigma-samples", type=int, default=20_000)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"sources": file_hashes(), "cells": {}}
    for (M, K, s, noise), kind in [(c, k) for c in BASE_CELLS for k in ("react", "diag")]:
        stem = f"sandbox-traits-M{M}-K{K}-sharp{s:g}-noise{noise:g}-{kind}"
        held = resolve_held_out_set(kind, M)
        cfg = SandboxConfig(M=M, K=K, sharpness=s, reward_noise=noise, family="traits", encoding="briefing", held_out=tuple(held)).normalized()
        lock = build_lock(cfg, calib_seeds=list(range(100, 120)), held_out=held, path=out_dir / f"lock_{stem}.json")
        exact = exact_reference(cfg)
        gap = delta_min(cfg)
        sigma = sigma_eff(cfg, samples=args.sigma_samples, seed_start=50_000_000)
        ratio = snr(gap, sigma)
        summary["cells"][stem] = {
            "held_out": held,
            "gate_seen": {k: lock["gate_seen"][k] for k in ("H_D", "n_distinct_best", "distinct_best", "min_gap", "passed")},
            "gate_all": {k: lock["gate_all"][k] for k in ("H_D", "n_distinct_best", "min_gap", "passed")},
            "never_best": lock["never_best"],
            "best_response": lock["best_response"],
            "exact": {k: v for k, v in exact.items() if k not in ("values", "best_response")},
            "Delta_min": gap,
            "sigma_eff": sigma,
            "snr": ratio,
            "predicted_T_star": predicted_T_star(K, M, ratio),
            "predicted_T_star_known_type": predicted_T_star(K, M, ratio, method="known_type"),
            "text_rows": len(lock["text_rows"]),
            "calibration_records": len(lock["calibration_records"]),
            "deployment_manifest_0_99": manifest(cfg, list(range(100)), held),
            "deployment_manifest_0_99_120_319": manifest(cfg, list(range(100)) + list(range(120, 320)), held),
        }
        print(stem, "gate", lock["gate_seen"]["passed"], "H_D", round(exact["H_D"], 3), "bayes ceiling capture", round(exact["prefix_ceiling_bayes_capture"], 3))
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
