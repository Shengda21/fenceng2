"""Calibration-only check of the symbolic composer (reads seen types only).

For every briefing lock (5 calibration settings x sigma 0/0.5 x the deployments that use the setting:
18 locks) the composer, fed the TRUE traits of each SEEN type, must pick the lock's best response for that type. The
check also compares the composer's simulated means with the lock's value-table row of each seen type (sigma = 0: equal
values; sigma = 0.5: equal differences to the chosen strategy, since the reward noise is common to all strategies on a
rollout), reports the gap between the best and the second-best strategy, and flags ties. No held-out type's value row or
best response is read, no deployment seed is used, and no briefing is parsed.

usage: python code/briefing/fcr_sym/calib_check.py [--root <repository root>] [--out calib/sym_calib_check.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # code/briefing/fcr_sym
FCR_DIR = HERE.parent / "fcr"
if str(FCR_DIR) not in sys.path:
    sys.path.insert(0, str(FCR_DIR))

import fcr_common as fc  # noqa: E402

from sym_composer import GAME_RULES, SymbolicComposer  # noqa: E402

LOCKS = OrderedDict([
    ("M9-react2", [("all", "react"), ("heldout", "react")]),
    ("M9-diag3", [("all", "diag"), ("heldout", "diag")]),
    ("M18-react4", [("all", "react"), ("heldout", "react")]),
    ("M18-diag6", [("all", "diag"), ("heldout", "diag")]),
    ("M18-react8", [("heldout", "react8")]),
])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(fc.RELEASE_DEFAULT))
    ap.add_argument("--out", default=str(HERE / "calib" / "sym_calib_check.json"))
    a = ap.parse_args()
    root = Path(a.root)
    composers = {}
    out = OrderedDict()
    n_checks = n_mismatch = 0
    min_gap, max_val_dev, max_diff_dev, n_lock_ties = np.inf, 0.0, 0.0, 0
    for setting, locks in LOCKS.items():
        M = int(setting.split("-")[0][1:])
        K = 5 if M == 9 else 10
        for sigma in (0.0, 0.5):
            comp = composers.setdefault((M, K, sigma), SymbolicComposer(M, K, sigma))
            for deploy, kind in locks:
                name = f"lock_sandbox-traits-M{M}-K{K}-sharp0.8-noise{'0.5' if sigma else '0'}-{kind}.json"
                path = root / "data" / "briefing" / "locks" / deploy / name
                lock = json.loads(path.read_text(encoding="utf-8"))
                cfg = lock["config"]
                rules_ok = all(float(cfg[k]) == float(GAME_RULES[k]) for k in ("delta", "sharpness")) and \
                    all(int(cfg[k]) == int(GAME_RULES[k]) for k in ("H", "k")) and cfg["schedule"] == GAME_RULES["schedule"] and \
                    int(cfg["M"]) == M and int(cfg["K"]) == K and float(cfg["reward_noise"]) == sigma
                held = list(lock["held_out"])
                seen = [t for t in fc.trait_types(M) if t not in set(held)]
                rows = OrderedDict()
                for t in seen:  # seen types only
                    res = comp.simulate(fc.traits_of(t, M))
                    vrow = lock["value_table"][t]
                    lock_best = lock["best_response"][t]
                    pool = comp.pool
                    v = np.array([vrow[tau] for tau in pool])
                    m = np.array([res["means"][tau] for tau in pool])
                    order = np.sort(v)
                    lock_gap = float(order[-1] - order[-2])
                    lock_tie = lock_gap < 1e-9
                    jc = pool.index(res["choice"])
                    val_dev = float(np.max(np.abs(m - v))) if sigma == 0 else None
                    diff_dev = float(np.max(np.abs((m - m[jc]) - (v - v[jc]))))
                    ok = res["choice"] == lock_best
                    n_checks += 1
                    n_mismatch += int(not ok)
                    n_lock_ties += int(lock_tie)
                    min_gap = min(min_gap, res["gap_best_second"])
                    max_diff_dev = max(max_diff_dev, diff_dev)
                    if val_dev is not None:
                        max_val_dev = max(max_val_dev, val_dev)
                    rows[t] = {"composer_choice": res["choice"], "lock_best_response": lock_best, "match": ok,
                               "tied_at_top": res["tied"], "simulated_gap_best_second": res["gap_best_second"],
                               "lock_gap_best_second": lock_gap, "lock_tie_at_top": lock_tie,
                               "max_abs_dev_mean_vs_value_table": val_dev,
                               "max_abs_dev_difference_to_choice_vs_value_table": diff_dev,
                               "max_rollout_sd": float(max(res["rollout_sd"].values())),
                               "max_dev_of_difference_across_rollouts": res["max_dev_of_difference_across_rollouts"]}
                out[f"{deploy}/{name}"] = {"setting": setting, "sigma": sigma, "game_rules_match_lock": bool(rules_ok),
                                           "held_out_not_read": held, "n_seen": len(seen),
                                           "matches": int(sum(r["match"] for r in rows.values())), "types": rows}
                print(f"{deploy}/{name}: {out[f'{deploy}/{name}']['matches']}/{len(seen)} seen types match; rules ok {rules_ok}", flush=True)
    summary = {"locks": len(out), "checks": n_checks, "mismatches": n_mismatch, "lock_ties_at_top_seen": n_lock_ties,
               "min_simulated_gap_best_second": float(min_gap), "max_abs_dev_mean_vs_value_table_sigma0": max_val_dev,
               "max_abs_dev_difference_vs_value_table": max_diff_dev,
               "composer": {f"M{M}-K{K}-noise{s}": c.describe() for (M, K, s), c in composers.items()},
               "tuples_simulated": {f"M{M}-K{K}-noise{s}": sorted("-".join(z) for z in c._cache) for (M, K, s), c in composers.items()}}
    fc.write_json(Path(a.out), {"summary": summary, "locks": out})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("composer", "tuples_simulated")}, indent=1))


if __name__ == "__main__":
    main()
