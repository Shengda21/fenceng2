"""Locks and references for held-out-only deployment: calibration uses seen types only; the reference values (V_tau, V*, oracle, H_D, prefix
ceilings, Delta_min, sigma_eff, predicted T*) are computed on the held-out types."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from calibrate_briefing import file_hashes
from closed_form import compute_closed_forms, delta_min, exact_reference, predicted_T_star, sigma_eff, snr
from sandbox.briefing import held_out_set_name, resolve_held_out_set
from sandbox.calibrate import build_lock
from sandbox.env import SandboxConfig, SandboxEnv, deployment_types

CELLS = [(18, 10, kind, noise) for kind in ("react", "react8", "diag") for noise in (0.0, 0.5)] + \
        [(9, 5, kind, noise) for kind in ("react", "diag") for noise in (0.0, 0.5)]
SEEDS = list(range(100)) + list(range(120, 320))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mc-samples", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"sources": file_hashes(), "cells": {}}
    for M, K, kind, noise in CELLS:
        name = f"sandbox-traits-M{M}-K{K}-sharp0.8-noise{noise:g}-{kind}"
        held = resolve_held_out_set(kind, M)
        cfg = SandboxConfig(M=M, K=K, sharpness=0.8, reward_noise=noise, family="traits", encoding="briefing", wording="newword",
                            held_out=tuple(held), deploy_types="heldout").normalized()
        lock = build_lock(cfg, calib_seeds=list(range(100, 120)), held_out=held, path=out / f"lock_{name}.json")
        exact = exact_reference(cfg)
        gap = delta_min(cfg)
        sigma = sigma_eff(cfg, samples=20_000, seed_start=50_000_000)
        ratio = snr(gap, sigma)
        n_dep = len(deployment_types(cfg))
        types = []
        for seed in SEEDS:
            env = SandboxEnv(cfg)
            env.reset(seed)
            types.append(env.type_label)
        row = {
            "held_out_set": held_out_set_name(kind, M),
            "deployment_types": deployment_types(cfg),
            "gate_seen": {k: lock["gate_seen"][k] for k in ("H_D", "n_distinct_best", "distinct_best", "min_gap", "passed")},
            "never_best_all_types": lock["never_best"],
            "deploy_best_response": {t: lock["best_response"][t] for t in deployment_types(cfg)},
            "exact": {k: v for k, v in exact.items() if k not in ("values", "best_response", "prefix_bayes_choice")},
            "Delta_min": gap,
            "sigma_eff": sigma,
            "snr": ratio,
            "n_deploy_types": n_dep,
            "predicted_T_star": predicted_T_star(K, n_dep, ratio),
            "predicted_T_star_known_type": predicted_T_star(K, n_dep, ratio, method="known_type"),
            "deployment_manifest_300": dict(Counter(types)),
            "deployment_all_held_out": all(t in held for t in types),
        }
        if args.mc_samples:
            mc = compute_closed_forms(cfg, samples=args.mc_samples, seed_start=1000)
            row["mc"] = {k: mc[k] for k in ("V_star", "best_tau", "per_decision_oracle", "H_D", "samples")}
        summary["cells"][name] = row
        print(name, "gate", lock["gate_seen"]["passed"], "deploy", n_dep, "V*", round(exact["V_star"], 3), exact["best_tau"], "H_D", round(exact["H_D"], 3),
              "share", round(exact["share"], 3), "bayes cap", round(exact["prefix_ceiling_bayes_capture"], 3), "Dmin", round(gap, 3),
              "sigma", round(sigma, 3), "T*", round(row["predicted_T_star"], 1))
    (out / "calibration_summary_0b.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
