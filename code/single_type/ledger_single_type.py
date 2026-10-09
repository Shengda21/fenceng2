#!/usr/bin/env python3
"""Turn the re-run summary into the manuscript's headroom ledger.

Every quantity here comes from data/rerun2026/, whose per-episode records are in
the release.

  dVmax     best constant minus default, from the exhaustive Probe-1 sweep
  Ddef      per-window oracle (Probe 2)
  placebo   the same oracle with every branch forced to the default; this is the
            floor the estimator reports when deviation buys nothing
  H_D       (Ddef - placebo) - dVmax, clamped at zero

Run:  python code/ledger_single_type.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "rerun2026", "rerun_summary.json")
OUT = os.path.join(ROOT, "rerun_ledger.json")

NICE = {"hanabi/small_2p": "Hanabi-Small", "magent/m20": "MAgent m20",
        "smac/3m": "SMAC 3m", "smac/8m": "SMAC 8m", "smac/MMM": "SMAC MMM"}
ORDER = ["hanabi/small_2p", "smac/3m", "smac/8m", "smac/MMM", "magent/m20"]

R = json.load(open(SRC, encoding="utf-8"))
led = {}

for k in ORDER:
    e = R[k]
    row = {"label": NICE[k],
           "best_constant": e["best_constant"],
           "default_is_best_constant": e["default_is_best_constant"],
           "constant_sweep": e["constant_sweep"],
           "arms": {"fixed_default": e["fixed_default"],
                    "random_router": e["random_router"],
                    "llm": e.get("llm"),
                    "linucb": e.get("linucb")}}
    for scale in ("success", "reward"):
        o = e["oracle"]["delta_def_" + scale]
        p = e["placebo"]["delta_def_" + scale]
        dv = e["dVmax"][scale]
        raw = (o - p) - dv
        row[scale] = {
            "dVmax": dv,
            "Ddef_oracle": o,
            "Ddef_placebo": p,
            "Ddef_corrected": o - p,
            "H_D": max(0.0, raw),
            "H_D_raw": raw,
            "effect_vs_default": (e.get("effect_vs_default") or {}).get(scale),
            "margin_vs_best_constant": (e.get("margin_vs_best_constant") or {}).get(scale),
        }
    led[k] = row

json.dump(led, open(OUT, "w", encoding="utf-8"), indent=1)


def fmt(d, nd=2):
    if not d:
        return "---"
    return f"{d['diff']:+.{nd}f}\\,[{d['lo']:+.{nd}f},{d['hi']:+.{nd}f}]"


print("=" * 112)
print("HEADROOM LEDGER  (all quantities re-measured 2026-08-25/26, n=100 per arm)")
print("=" * 112)
hdr = (f"{'cell':14s} {'scale':7s} {'dVmax':>7} {'Ddef':>7} {'placebo':>8} "
       f"{'corr':>7} {'H_D':>6}  {'effect vs default':>24} {'margin vs best const':>24}")
print(hdr)
print("-" * len(hdr))
for k in ORDER:
    r = led[k]
    for scale in ("success", "reward"):
        s = r[scale]
        nd = 3 if scale == "success" else 2
        print(f"{r['label'] if scale == 'success' else '':14s} {scale:7s} "
              f"{s['dVmax']:7.3f} {s['Ddef_oracle']:7.3f} {s['Ddef_placebo']:8.3f} "
              f"{s['Ddef_corrected']:+7.3f} {s['H_D']:6.3f}  "
              f"{fmt(s['effect_vs_default'], nd):>24} "
              f"{fmt(s['margin_vs_best_constant'], nd):>24}")
    print()

print("=" * 112)
print("LinUCB against the best constant (reward)")
for k in ORDER:
    e = R[k]
    b = e.get("linucb") or {}
    bc = e["best_constant"]["by_reward"]
    bcr = e["constant_sweep"][bc]["reward"]
    print(f"  {NICE[k]:14s} bandit cum={b.get('cum_success'):.2f} last50={b.get('last50_success'):.2f} "
          f"mean_r={b.get('mean_reward'):6.2f}   best constant {bc} = {bcr:6.2f}   "
          f"gap={b.get('mean_reward', 0) - bcr:+6.2f}")
print(f"\nwrote {OUT}")
