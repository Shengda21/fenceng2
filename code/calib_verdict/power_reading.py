"""Power reading of the calibration split, from deployment records alone (no read-half record is used).

For every arm: 1,000 resamples of n_B headline-seed episodes with replacement; in each resample the strongest comparator
is re-chosen by its mean (as on B) and the C2 and C3 verdicts are read with 2,000 bootstrap replicates and the
same thresholds. Reports the share of resamples whose C2, C3 and verdict agree with the deployment verdict.
Writes analysis/calib_verdict/power_reading.json.
"""

from __future__ import annotations

import json

import numpy as np

from calib_lib import DATA, OUT, PAYS_MARGIN, deployment_cell, verdict

ARMS = json.load(open(DATA / "tasks" / "arms.json", encoding="utf-8"))
DEP = json.load(open(OUT / "deploy_verdicts.json", encoding="utf-8"))
N_B = {"sandbox": {3: 30, 6: 60}, "briefing": {9: 70, 18: 140}, "magent": 20}
R, B_IN = 1000, 2000


def n_b(domain, cell):
    if domain == "magent":
        return N_B["magent"]
    if domain == "sandbox":
        return N_B["sandbox"][int(cell.split("-M")[1].split("-")[0])]
    return N_B["briefing"][int(cell.split("-M")[1].split("-")[0])]


def main():
    rng = np.random.default_rng(20261006)
    out = {"method": __doc__.strip(), "R": R, "B_inner": B_IN, "cells": {}}
    for domain in ("briefing", "sandbox", "magent"):
        for cell, spec in ARMS[domain].items():
            d = deployment_cell(domain, cell, spec)
            F, comps = d["F"], d["comps"]
            nb = n_b(domain, cell)
            res = {}
            for k, A in d["llm"].items():
                dep = DEP[domain][cell]["arms"][k]
                c2s, c3s, vs = [], [], []
                for _ in range(R):
                    s = rng.integers(0, 100, size=nb)
                    Fs, As = F[:, s], A[s]
                    idx = rng.integers(0, nb, size=(B_IN, nb))
                    vstar_b = Fs[:, idx].mean(axis=2).max(axis=0)
                    mlo = float(np.percentile(As[idx].mean(axis=1) - vstar_b, 5))
                    cs = {c: v[s] for c, v in comps.items()}
                    strongest = max(cs, key=lambda c: cs[c].mean())
                    dd = As - cs[strongest]
                    vlo = float(np.percentile(dd[rng.integers(0, nb, size=(B_IN, nb))].mean(axis=1), 5))
                    c2, c3 = mlo > PAYS_MARGIN, vlo > 0
                    c2s.append(c2 == dep["C2"])
                    c3s.append(c3 == dep["C3"])
                    vs.append(verdict(c2, c3) == dep["verdict"])
                res[k] = {"n_B": nb, "deployment_verdict": dep["verdict"], "deployment_margin_lo95": dep["margin_lo95"],
                          "p_agree_C2": float(np.mean(c2s)), "p_agree_C3": float(np.mean(c3s)), "p_agree_verdict": float(np.mean(vs))}
                print(f"{domain:8s} {cell:52s} {k:45s} n={nb:3d} dep={dep['verdict']:10s} mlo={dep['margin_lo95']:7.2f} "
                      f"P(C2 agree)={res[k]['p_agree_C2']:.2f} P(C3 agree)={res[k]['p_agree_C3']:.2f}")
            out["cells"][f"{domain}/{cell}"] = res
    allp = [v for c in out["cells"].values() for v in c.values()]
    out["expected_C2_agreements"] = float(sum(v["p_agree_C2"] for v in allp))
    out["expected_C3_agreements"] = float(sum(v["p_agree_C3"] for v in allp))
    out["n_arms"] = len(allp)
    (OUT / "power_reading.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print("arms", len(allp), "expected C2 agreements", round(out["expected_C2_agreements"], 1), "expected C3 agreements", round(out["expected_C3_agreements"], 1))


if __name__ == "__main__":
    main()
