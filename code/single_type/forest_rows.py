#!/usr/bin/env python3
"""Assemble the deployment-margin forest's rows from the data files.

Diagnostic cells come from the re-run (rerun_ledger.json): every arm ran at
n=100 on one shared seed list, so the margin is a single paired contrast and the
normalising level is the comparator's own return from the same batch. The two
HLA rows come from hla_paired.json.

Each diagnostic row also carries the second router family's margin against the
same best constant (second_family.json), so the figure can show both routers
against one ceiling rather than implying there is only one. The HLA rows have no
second-family arm and carry None there.

Each row: (label, V_star, margin, lo, hi, H_D, second_family or None), where
second_family is [diff, lo, hi].

Run:  python code/forest_rows.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LED = os.path.join(ROOT, "rerun_ledger.json")
HLA = os.path.join(ROOT, "hla_paired.json")
SF = os.path.join(ROOT, "second_family.json")
OUT = os.path.join(ROOT, "forest_rows.json")

ORDER = ["hanabi/small_2p", "smac/3m", "smac/8m", "smac/MMM", "magent/m20"]

led = json.load(open(LED, encoding="utf-8"))
sf = json.load(open(SF, encoding="utf-8")) if os.path.exists(SF) else {}
rows = []

for k in ORDER:
    e = led[k]
    rw = e["reward"]
    # comparator level: the best constant's own return in this batch
    bc = e["best_constant"]["by_reward"]
    vstar = e["constant_sweep"][bc]["reward"]
    m = rw["margin_vs_best_constant"]

    # second family, measured against the same best constant
    m2 = None
    if k in sf:
        c = sf[k]["contrasts"].get("best_constant", {}).get("reward")
        if c and "diff" in c:
            m2 = [c["diff"], c["lo"], c["hi"]]

    rows.append([e["label"], vstar, m["diff"], m["lo"], m["hi"], rw["H_D"], m2])

hla = json.load(open(HLA, encoding="utf-8"))
for layout, nice in (("ring", "HLA ring"), ("partition", "HLA partition")):
    arms = hla[layout]["arms"]
    ck = next(k for k in arms if "constant" in k)
    fk = next(k for k in arms if k.endswith("_full"))
    pair = hla[layout]["paired"][f"{fk}_minus_{ck}"]["score"]
    rows.append([nice, arms[ck]["mean_score"], pair["diff"], pair["lo"],
                 pair["hi"], None, None])

json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1)
for r in rows:
    hd = "---" if r[5] is None else f"{r[5]:.3f}"
    m2 = "---" if r[6] is None else f"{r[6][0]:+.2f}"
    print(f"{r[0]:16s} V*={r[1]:8.2f}  margin={r[2]:+8.2f} [{r[3]:+.2f},{r[4]:+.2f}]"
          f"  ({100 * r[2] / r[1]:+6.1f}%)  H_D={hd}  2nd={m2}")
print(f"\nwrote {OUT}")
