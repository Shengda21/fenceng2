"""Deployment verdicts of every arm of the calibration split, read from the analysis outputs.

C2 (deployment) = one-sided 95% lower bound of the paired margin over V* > 0.5 (margin_lo95 in the analysis JSON);
C3 (deployment) = one-sided 95% lower bound of the paired difference against the deployment's strongest non-LLM arm > 0
(vs_strongest.lo95). Writes analysis/calib_verdict/deploy_verdicts.json.
"""

from __future__ import annotations

import json

from calib_lib import DATA, OUT, REL

ARMS = json.load(open(DATA / "tasks" / "arms.json", encoding="utf-8"))


def verdict(c2, c3):
    return "yes" if c2 and c3 else ("fixed only" if c2 else "no")


def entry(st, source):
    vs = st.get("vs_strongest")
    c2 = st["margin_lo95"] > 0.5
    c3 = vs is not None and vs["lo95"] > 0
    return {"source": source, "n": st.get("n"), "mean": st.get("mean"), "capture": st.get("capture"), "margin": st.get("margin"),
            "margin_lo95": st["margin_lo95"], "vs_strongest_arm": st.get("vs_strongest_arm"),
            "vs_diff": vs["diff"] if vs else None, "vs_lo95": vs["lo95"] if vs else None,
            "C2": c2, "C3": c3, "verdict": verdict(c2, c3), "release_pays": st.get("pays")}


def main():
    M = json.load(open(REL / "analysis" / "master.json", encoding="utf-8"))["master"]
    AT = json.load(open(REL / "analysis" / "briefing" / "all_types.json", encoding="utf-8"))["groups"]
    TP = json.load(open(REL / "analysis" / "briefing" / "task_prompt.json", encoding="utf-8"))["s0"]
    out = {"sandbox": {}, "briefing": {}, "magent": {}}
    for cell, spec in ARMS["sandbox"].items():
        e = M[spec["llm_cell"]]
        out["sandbox"][cell] = {"V_star": e["reference"]["V_star"], "H_D": e["reference"]["H_D"],
                                "strongest_nonllm": e.get("strongest_nonllm"), "arms": {}}
        for k in spec["llm_arms"]:
            out["sandbox"][cell]["arms"][k] = entry(e["arms"][k], f"master.json:{spec['llm_cell']}")
    for cell, spec in ARMS["briefing"].items():
        g = cell.replace("-sameword", "")
        out["briefing"][cell] = {"V_star": AT[g]["reference"]["V_star"], "H_D": AT[g]["reference"]["H_D"],
                                 "strongest_nonllm": AT[g]["strongest_nonllm"].get("sameword"), "arms": {}}
        for k in spec["llm_arms"]:
            if k.endswith("__v2"):
                st = TP[g]["arms"][f"briefing-sameword/{k}"]
                out["briefing"][cell]["arms"][k] = entry(st, f"task_prompt.json:s0/{g}/briefing-sameword/{k}")
            else:
                st = AT[g]["cells"]["briefing-sameword"]["arms"][k]
                out["briefing"][cell]["arms"][k] = entry(st, f"all_types.json:{g}/briefing-sameword/{k}")
    for cell, spec in ARMS["magent"].items():
        e = M[spec["master_cell"]]
        out["magent"][cell] = {"V_star": e["reference"]["V_star"], "H_D": e["reference"]["H_D"],
                               "strongest_nonllm": e.get("strongest_nonllm"), "arms": {}}
        for k in spec["llm_arms"]:
            rk = k if k in e["arms"] else k.replace("__nshot", "__nshot-")  # the attenuated cell's files are named nshot-N
            out["magent"][cell]["arms"][k] = entry(e["arms"][rk], f"master.json:{spec['master_cell']}/{rk}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "deploy_verdicts.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    for d, cells in out.items():
        for c, v in cells.items():
            print(f"== {d} {c}  V*={v['V_star']:.2f} H_D={v['H_D']:.2f} strongest={v['strongest_nonllm']}")
            for k, a in v["arms"].items():
                print(f"   {k:45s} cap {a['capture']:6.2f} mlo {a['margin_lo95']:7.2f} vs {a['vs_diff']:7.2f} lo {a['vs_lo95']:7.2f}  C2={a['C2']!s:5} C3={a['C3']!s:5} {a['verdict']:10s} rel={a['release_pays']}")


if __name__ == "__main__":
    main()
