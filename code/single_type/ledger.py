#!/usr/bin/env python3
"""Rebuild the headroom ledger with each column drawn from one measurement.

Mixing two estimates of the same policy would make the columns inconsistent: on
the four competitive-default cells the domain default and the best constant are
the SAME controller (focus_fire, ATTACK_FORWARD), but the default is measured by
the n=100 router-off arm and the constant by the separate n=50 Probe-1 sweep, so
the table would show dVmax = 0 while the effect and margin columns disagree.

Convention used here, stated in the caption:
  * Probe 1 identifies WHICH skill is the best constant.
  * The VALUE of that skill is taken from the arm that measured it at n=100
    whenever such an arm exists, i.e. from the router-off arm on the four cells
    where the best constant is the default. Probe 1's independent n=50 estimate
    of the same policy is reported separately in the supplementary.
  * On Hanabi the best constant is a different policy from the default, so both
    are reported and the effect and margin genuinely differ.

Run:  python code/ledger.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLOUD = os.path.join(ROOT, "data", "experiments_phase3",
                     "paired_analysis_diagnostic_cloud.json")
OUT = os.path.join(ROOT, "ledger.json")

# Probe 1 identifies the best constant skill per cell, and whether it is the
# same policy as the domain default.
BEST = {
    "hanabi_small_2p": ("PLAY_KNOWN", False),
    "smac_3m": ("focus_fire", True),
    "smac_8m": ("focus_fire", True),
    "smac_MMM": ("focus_fire", True),
    "magent_m20": ("ATTACK_FORWARD", True),
}
NICE = {"hanabi_small_2p": "Hanabi-Small", "smac_3m": "SMAC 3m",
        "smac_8m": "SMAC 8m", "smac_MMM": "SMAC MMM",
        "magent_m20": "MAgent m20"}
# Probe 2, per-window oracle: contextual default gap. None where not measurable.
ORACLE = {"hanabi_small_2p": {"success": 1.00, "reward": None},
          "magent_m20": {"success": 0.02, "reward": 1.85}}

cloud = json.load(open(CLOUD, encoding="utf-8"))
rows = {}
for cell, (tau, same) in BEST.items():
    v = cloud[cell]
    row = {"best_constant": tau, "best_constant_is_default": same}
    for scale, dkey, bkey, sweepkey in (
            ("success", "dsucc_llm_default", "dsucc_llm_bestconst", "succ"),
            ("reward", "drew_llm_default", "drew_llm_bestconst", "mean_rew")):
        d_def = v[dkey]
        d_best = v[bkey]
        sweep = v["best_const"][sweepkey]
        # LLM level implied by the sweep arm, then the default arm's own level
        v_llm = sweep + d_best["mean"]
        v_def = v_llm - d_def["mean"]
        if same:
            # one policy: its n=100 estimate is the default arm's
            v_star = v_def
            dvmax = 0.0
            margin = d_def          # effect and margin are the same comparison
            margin_src = "router-off arm, n=100 (best constant is the default)"
        else:
            v_star = sweep
            dvmax = v_star - v_def
            margin = d_best
            margin_src = "Probe-1 constant arm"
        ddef = ORACLE.get(cell, {}).get(scale)
        row[scale] = {
            "V_default": round(v_def, 4),
            "V_star": round(v_star, 4),
            "V_llm": round(v_llm, 4),
            "dVmax": round(dvmax, 4),
            "Delta_def_oracle": ddef,
            "H_D": (None if ddef is None else round(ddef - dvmax, 4)),
            "effect_vs_default": {"mean": round(d_def["mean"], 4),
                                  "ci": [round(x, 4) for x in d_def["ci95"]]},
            "margin_vs_best_constant": {"mean": round(margin["mean"], 4),
                                        "ci": [round(x, 4) for x in margin["ci95"]],
                                        "source": margin_src},
            "probe1_sweep_estimate_of_best_constant": round(sweep, 4),
        }
    rows[cell] = row

json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"{'cell':14s} {'scale':8s} {'V_def':>8} {'V*':>8} {'dVmax':>7} "
      f"{'Ddef':>6} {'H_D':>6}  {'effect':>26}  {'margin':>26}")
for cell, row in rows.items():
    for scale in ("success", "reward"):
        s = row[scale]
        e, m = s["effect_vs_default"], s["margin_vs_best_constant"]
        print(f"{NICE[cell]:14s} {scale:8s} {s['V_default']:8.2f} {s['V_star']:8.2f} "
              f"{s['dVmax']:7.2f} {str(s['Delta_def_oracle']):>6} {str(s['H_D']):>6}  "
              f"{e['mean']:+8.2f} [{e['ci'][0]:+.2f},{e['ci'][1]:+.2f}]  "
              f"{m['mean']:+8.2f} [{m['ci'][0]:+.2f},{m['ci'][1]:+.2f}]")
    print(f"{'':14s} probe-1 n=50 estimate of {row['best_constant']}: "
          f"succ {row['success']['probe1_sweep_estimate_of_best_constant']}, "
          f"reward {row['reward']['probe1_sweep_estimate_of_best_constant']}")
print(f"\nwrote {OUT}")
