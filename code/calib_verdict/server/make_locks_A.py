"""Derive the fit-half calibration locks of the five MAgent cells from the full calibration locks.

The full locks hold, for each of the 40 calibration seeds 1000-1039, the drawn type, the decision-time features and
text, and the return of every fixed strategy (population_values.fixed). The split takes the even positions of
calibration_seeds as the fit half A and the odd positions as the read half B. Each A lock recomputes, from the A
episodes only and with the calibration functions of code/oghp, every field a selector reads: the counter table
(best_response_table), per-type values, Delta per type, pooled means, best fixed strategy, the PLASTIC Gaussian type
models and the feature rows (which also carry the N-shot examples and the ctxucb bins). Its deployment manifest is the
B episodes, so the runner plays exactly the B calibration episodes (seed and drawn type).

A second lock per cell, *_AB, has the A episodes followed by the B episodes as its deployment manifest and no
calibration seeds, for online learners that warm up on A and are read on B.

Usage: python make_locks_A.py <code_oghp_dir> <locks_dir> <out_dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CELLS = {
    "battle-base4": ("rs_magent_lock_base4.json", "magent"),
    "battle-traits8": ("rs_magent_lock_traits8.json", "magent"),
    "combined-base4": ("rs_combined_lock.json", "combined"),
    "combined-traits8": ("rs_combined_traits8_lock.json", "combined"),
    "combined-base4-delta06": ("rs_combined_lock_delta06.json", "combined"),
}


def posix_hashes(hashes: dict) -> dict:
    return {k.replace("\\", "/"): v for k, v in hashes.items()}


def main() -> None:
    oghp, locks_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    sys.path.insert(0, str(oghp / "experiments0106b"))
    sys.path.insert(0, str(oghp / "v8_lib"))
    import scripts.rs.calibrate_rs_combined as cc
    import scripts.rs.calibrate_rs_magent as cm

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cell, (fname, kind) in CELLS.items():
        mod = cm if kind == "magent" else cc
        src = locks_dir / fname
        lock = json.loads(src.read_text(encoding="utf-8"))
        seeds = [int(s) for s in lock["calibration_seeds"]]
        A, B = seeds[0::2], seeds[1::2]
        man = {int(m["seed"]): m for m in lock["calibration_manifest"]}
        fixed = lock["population_values"]["fixed"]
        pool = list(lock["pool"])
        key = {s: man[s]["key"] for s in seeds}
        typ = {s: man[s]["type1"] for s in seeds}
        # fixed_sweep restricted to A
        by_type: dict = {}
        for s in A:
            for tau in pool:
                by_type.setdefault(typ[s], {}).setdefault(tau, []).append(float(fixed[tau][key[s]]))
        per_type_means = {t: {tau: float(np.mean(v)) for tau, v in m.items()} for t, m in by_type.items()}
        best_response_table = {t: max(m, key=lambda tau: (m[tau], tau)) for t, m in per_type_means.items()}
        delta = {}
        for t, m in per_type_means.items():
            o = sorted(m.values(), reverse=True)
            delta[t] = float(o[0] - o[1]) if len(o) > 1 else 0.0
        pooled = {tau: float(np.mean([fixed[tau][key[s]] for s in A])) for tau in pool}
        best_fixed = max(pooled, key=lambda tau: (pooled[tau], tau))
        rows_A = [r for r in lock["feature_rows"] if int(r["seed"]) in set(A)]
        if len(rows_A) != len(A):
            raise SystemExit(f"{cell}: feature rows do not cover A")
        omax = float(np.mean([max(fixed[tau][key[s]] for tau in pool) for s in A]))
        default = lock["default"]
        lockA = dict(lock)
        lockA.update({
            "calibration_seeds": A,
            "calibration_manifest": [man[s] for s in A],
            "deployment_manifest": [man[s] for s in B],
            "best_response_table": best_response_table,
            "Delta_per_type": delta,
            "value_table": per_type_means,
            "per_type_report": mod.summarize_type_values(per_type_means),
            "pooled_means": pooled,
            "V_star": pooled[best_fixed],
            "best_fixed": best_fixed,
            "V_default": pooled[default],
            "H_D": omax - pooled[best_fixed],
            "feature_rows": rows_A,
            "plastic_type_models": mod.fit_gaussians(rows_A),
            "population_values": {k: {kk: vv for kk, vv in v.items()} if k != "fixed" else
                                  {tau: {key[s]: fixed[tau][key[s]] for s in A} for tau in pool}
                                  for k, v in lock["population_values"].items() if k == "fixed"},
            "file_hashes": posix_hashes(mod.file_hashes()),
            "calib_split": {"source_lock": fname, "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                            "rule": "A = calibration_seeds[0::2], B = calibration_seeds[1::2]", "A": A, "B": B,
                            "A_types": sorted(set(typ[s] for s in A)), "B_types": sorted(set(typ[s] for s in B)),
                            "types_missing_from_A": sorted(set(lock["types"]) - set(typ[s] for s in A))},
        })
        for k in ("classifier_diagnostics", "ppo_checkpoint", "ppo_checkpoint_hash", "ppo_training_seeds", "ppo_train_returns",
                  "Delta_def", "placebo", "headroom_share"):
            lockA.pop(k, None)
        lockAB = dict(lockA)
        lockAB.update({"calibration_seeds": [], "deployment_manifest": [man[s] for s in A] + [man[s] for s in B]})
        (out_dir / f"lockA_{cell}.json").write_text(json.dumps(lockA, indent=1), encoding="utf-8")
        (out_dir / f"lockAB_{cell}.json").write_text(json.dumps(lockAB, indent=1), encoding="utf-8")
        # B replay table: the calibration record of every B episode
        table = [{"seed": s, "key": key[s], "type": typ[s],
                  "fixed": {tau: float(fixed[tau][key[s]]) for tau in pool},
                  "features": next(r["features"] for r in lock["feature_rows"] if int(r["seed"]) == s),
                  "text": next(r["text"] for r in lock["feature_rows"] if int(r["seed"]) == s)} for s in seeds]
        (out_dir / f"calib_table_{cell}.json").write_text(json.dumps({"A": A, "B": B, "pool": pool, "default": default,
                                                                      "episodes": table}, indent=1), encoding="utf-8")
        summary[cell] = {"A_best_response": best_response_table, "full_best_response": lock["best_response_table"],
                         "A_best_fixed": best_fixed, "full_best_fixed": lock["best_fixed"],
                         "types_missing_from_A": lockA["calib_split"]["types_missing_from_A"]}
    (out_dir / "locks_A_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
