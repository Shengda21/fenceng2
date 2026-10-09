"""Read-half (B) C2/C3 verdicts of the calibration split and their agreement with the deployment verdicts.

Inputs : data/calib_verdict/runs/{sandbox,briefing,magent}/... (the records of the server runs), data/calib_verdict/locks/
         calib_table_*.json, data/calib_verdict/tasks/arms.json, analysis/calib_verdict/{deploy_verdicts,power_reading}.json.
Output : analysis/calib_verdict/calib_verdict.json; a summary table on stdout.
usage  : python code/calib_verdict/calib_verdict.py [--root <repository root>]
C2: one-sided 95% lower bound of the paired margin over V* above 0.5. C3: one-sided 95% lower bound of the paired
difference against the strongest non-LLM arm above 0.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from calib_lib import DATA, OUT, PAYS_MARGIN, TIE_TOL, REL, by_seed, deployment_cell, margin_stats, paired_diff, read_rows, verdict

RUNS = DATA / "runs"
ARMS = json.load(open(DATA / "tasks" / "arms.json", encoding="utf-8"), object_pairs_hook=OrderedDict)
DEP = json.load(open(OUT / "deploy_verdicts.json", encoding="utf-8"))
POWER = json.load(open(OUT / "power_reading.json", encoding="utf-8"))
SB_B = list(range(101, 120, 2))
MG_B = list(range(1001, 1040, 2))


# ------------------------------------------------------------------ record loading

def _tau(r):
    prov = r.get("provenance") or []
    if prov and prov[0].get("tau"):
        return prov[0]["tau"]
    decs = r.get("decisions") or []
    return decs[0].get("tau") if decs else None


def _extra(r):
    prov = r.get("provenance") or [{}]
    return prov[0].get("extra") or {}


def load_jsonl_arm(paths) -> dict:
    """{(seed, type): record} over the B episodes of one arm (shards concatenated, phase A dropped)."""
    out = {}
    for p in paths:
        for r in read_rows(p):
            if r.get("phase", "B") != "B":
                continue
            k = (int(r["seed"]), r["type"])
            if k in out:
                raise SystemExit(f"duplicate episode {k} in {p}")
            out[k] = r
    return out


def load_magent_arm(path: Path) -> dict:
    data = json.load(open(path, encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    out = {}
    for r in rows:
        if int(r["seed"]) in MG_B:
            if r.get("total_reward") is None:
                raise SystemExit(f"episode error in {path}: {r.get('error')}")
            out[(int(r["seed"]), r["type"])] = r
    return out


def arm_paths(domain, cell, key, sub=None):
    d = RUNS / domain / cell / (sub or "")
    single = d / f"{key}.jsonl"
    if single.exists():
        return [single]
    return sorted(d.glob(f"{key}__b*.jsonl"))


def load_cell(domain, cell, spec):
    """Return episodes (sorted keys), fixed matrix F (taus x episodes), comparator arrays, LLM records."""
    if domain == "magent":
        table = json.load(open(DATA / "locks" / f"calib_table_{cell}.json", encoding="utf-8"))
        eps = [(e["seed"], e["type"]) for e in table["episodes"] if e["seed"] in MG_B]
        taus = table["pool"]
        lockF = {(e["seed"], e["type"]): e["fixed"] for e in table["episodes"]}
        F = np.array([[lockF[k][t] for k in eps] for t in taus])
        runF = {}
        for t in taus:
            recs = load_magent_arm(RUNS / "magent" / cell / f"fixed__{t}.json")
            runF[t] = np.array([recs[k]["total_reward"] for k in eps])
        replay_fixed = float(max(np.max(np.abs(runF[t] - F[i])) for i, t in enumerate(taus)))
        comps = {}
        comp_recs = {}
        for c in spec["comparators"]:
            if c == "ppo":
                continue
            recs = load_magent_arm(RUNS / "magent" / cell / f"{c}.json")
            comp_recs[c] = recs
        llm = {k: load_magent_arm(RUNS / "magent" / cell / f"{k}.json") for k in spec["llm_arms"]}
    else:
        if domain == "sandbox":
            fixed_paths = sorted((RUNS / "sandbox" / cell).glob("fixed__*.jsonl"))
            comp_sub = {c: None for c in spec["comparators"]}
            llm_sub = None
        else:
            fixed_paths = sorted((RUNS / "briefing" / cell / "nonllm").glob("fixed__*.jsonl"))
            comp_sub = {c: "nonllm" for c in spec["comparators_prefix"]}
            comp_sub.update({c: "briefing" for c in spec["comparators_text"] + spec["secondary_text"]})
            llm_sub = None
        fx = {p.name[len("fixed__"):-len(".jsonl")]: load_jsonl_arm([p]) for p in fixed_paths}
        taus = sorted(fx)
        eps = sorted(next(iter(fx.values())))
        F = np.array([[fx[t][k]["total_reward"] for k in eps] for t in taus])
        replay_fixed = 0.0
        comp_recs = {c: load_jsonl_arm(arm_paths(domain, cell, c.replace(":", "__"), sub)) for c, sub in comp_sub.items()}
        llm = {k: load_jsonl_arm(arm_paths(domain, cell, k, llm_sub)) for k in spec["llm_arms"]}
    return eps, taus, F, comp_recs, llm, replay_fixed


def replay_check(recs, eps, taus, F):
    bad = 0
    for j, k in enumerate(eps):
        r = recs[k]
        t = _tau(r)
        if t not in taus or abs(float(r["total_reward"]) - F[taus.index(t), j]) > 1e-9:
            bad += 1
    return bad


def per_type(recs, eps, taus, F):
    types = sorted(set(t for _, t in eps))
    out = OrderedDict()
    for ty in types:
        idx = [j for j, k in enumerate(eps) if k[1] == ty]
        m = F[:, idx].mean(axis=1)
        good = [taus[i] for i in range(len(taus)) if m[i] >= m.max() - TIE_TOL]
        ch = [_tau(recs[eps[j]]) for j in idx]
        out[ty] = {"n": len(idx), "best": taus[int(np.argmax(m))], "good_set": good,
                   "p_mis": float(np.mean([c not in good for c in ch])), "choices": dict(Counter(ch))}
    return out


def llm_cost(recs):
    ct, lat, fb = [], [], []
    for r in recs.values():
        ex = _extra(r)
        u = ex.get("usage") or {}
        if u.get("completion_tokens") is not None:
            ct.append(u["completion_tokens"])
        if ex.get("call_latency_s") is not None:
            lat.append(ex["call_latency_s"])
        fb.append(bool(r.get("fallback_count", 0)) or any(bool(d.get("fallback")) for d in (r.get("decisions") or [])))
    fps = sorted({(_extra(r).get("system_fingerprint")) for r in recs.values()} - {None})
    return {"fallback_rate": float(np.mean(fb)) if fb else None, "median_completion_tokens": float(np.median(ct)) if ct else None,
            "median_call_latency_s": float(np.median(lat)) if lat else None, "system_fingerprints": fps}


# ------------------------------------------------------------------ deployment seen-type reading (briefing, secondary)

def seen_type_deployment(cell, spec, rng):
    sys.path.insert(0, str(REL / "code" / "briefing" / "v8_sandbox"))
    sys.path.insert(0, str(REL / "code" / "briefing" / "v8_lib"))
    from sandbox.briefing import resolve_held_out_set
    M = int(cell.split("-M")[1].split("-")[0])
    held = set(resolve_held_out_set("react", M))
    d = deployment_cell("briefing", cell, spec)
    rows = read_rows(REL / "data" / "briefing" / "all" / f"{cell.replace('-sameword', '')}-nonllm" / f"fixed__{d['taus'][0]}.jsonl.gz")
    typ = {int(r["seed"]): r["type"] for r in rows}
    keep = np.array([typ[s] not in held for s in range(100)])
    F = d["F"][:, keep]
    comps = {c: v[keep] for c, v in d["comps"].items()}
    strongest = max(comps, key=lambda c: comps[c].mean())
    out = {"n_seen_seeds": int(keep.sum()), "strongest": strongest, "arms": {}}
    for k, A in d["llm"].items():
        st = margin_stats(F, A[keep], rng)
        pdv = paired_diff(A[keep], comps[strongest], rng)
        c2, c3 = st["margin_lo95"] > PAYS_MARGIN, pdv["lo95"] > 0
        out["arms"][k] = {"margin": st["margin"], "margin_lo95": st["margin_lo95"], "capture": st["capture"],
                          "vs_diff": pdv["diff"], "vs_lo95": pdv["lo95"], "C2": c2, "C3": c3, "verdict": verdict(c2, c3)}
    return out


# ------------------------------------------------------------------ main

def main():
    rng = np.random.default_rng(20261006)
    out = OrderedDict(prereg="PREREG_CALIB.md", B_boot=10_000, rng_seed=20261006, thresholds={"C2_margin_lo95_gt": PAYS_MARGIN, "C3_vs_lo95_gt": 0.0},
                      cells=OrderedDict())
    rows_table = []
    secondary_jobs = []
    for domain in ARMS:
        for cell, spec in ARMS[domain].items():
            eps, taus, F, comp_recs, llm, replay_fixed = load_cell(domain, cell, spec)
            n = len(eps)
            # comparators on B
            comp_vals, comp_info = OrderedDict(), OrderedDict()
            primary_comps = spec["comparators"] if domain != "briefing" else spec["comparators_prefix"] + spec["comparators_text"]
            for c, recs in comp_recs.items():
                missing = [k for k in eps if k not in recs]
                if missing:
                    raise SystemExit(f"{domain}/{cell}/{c}: {len(missing)} B episodes missing")
                vals = np.array([float(recs[k]["total_reward"]) for k in eps])
                comp_vals[c] = vals
                V_star = float(F.mean(axis=1).max())
                H = float(F.max(axis=0).mean() - V_star)
                comp_info[c] = {"mean": float(vals.mean()), "capture": (float(vals.mean()) - V_star) / H if H > 0 else None,
                                "replay_mismatches": replay_check(recs, eps, taus, F), "in_primary_set": c in primary_comps}
            prim = {c: comp_vals[c] for c in primary_comps if c in comp_vals}
            strongest = max(prim, key=lambda c: prim[c].mean())
            V_star = float(F.mean(axis=1).max())
            H = float(F.max(axis=0).mean() - V_star)
            cell_out = OrderedDict(domain=domain, n_B=n, taus=taus, V_star_B=V_star, best_fixed_B=taus[int(np.argmax(F.mean(axis=1)))],
                                   H_B=H, fixed_replay_max_abs_diff=replay_fixed, strongest_nonllm_B=strongest,
                                   strongest_capture_B=comp_info[strongest]["capture"], comparators=comp_info, arms=OrderedDict())
            for k in spec["llm_arms"]:
                recs = llm[k]
                missing = [e for e in eps if e not in recs]
                if missing:
                    raise SystemExit(f"{domain}/{cell}/{k}: {len(missing)} B episodes missing")
                A = np.array([float(recs[e]["total_reward"]) for e in eps])
                st = margin_stats(F, A, rng)
                pdv = paired_diff(A, comp_vals[strongest], rng)
                c2, c3 = st["margin_lo95"] > PAYS_MARGIN, pdv["lo95"] > 0
                dep = DEP[domain][cell]["arms"][k]
                pw = POWER["cells"][f"{domain}/{cell}"][k]
                a = OrderedDict(B=OrderedDict(**st, vs_strongest_arm=strongest, vs_diff=pdv["diff"], vs_ci=pdv["ci"], vs_lo95=pdv["lo95"],
                                              C2=c2, C3=c3, verdict=verdict(c2, c3)),
                                deployment=dep, agree_C2=c2 == dep["C2"], agree_C3=c3 == dep["C3"], agree_verdict=verdict(c2, c3) == dep["verdict"],
                                power=pw, replay_mismatches=replay_check(recs, eps, taus, F), per_type=per_type(recs, eps, taus, F),
                                cost=llm_cost(recs))
                cell_out["arms"][k] = a
                rows_table.append((domain, cell, k, dep["verdict"], a["B"]["verdict"], dep["C2"], c2, dep["C3"], c3, st["margin_lo95"],
                                   pdv["lo95"], st["capture"], pw["p_agree_C2"], a["replay_mismatches"]))
            out["cells"][f"{domain}/{cell}"] = cell_out
            if domain == "briefing":
                secondary_jobs.append((cell, spec, eps, F, comp_vals, llm))
    # secondary readings (after the primary stream)
    sec = OrderedDict()
    for cell, spec, eps, F, comp_vals, llm in secondary_jobs:
        strict = {c: comp_vals[c] for c in spec["comparators_prefix"] + ["scripted_text", "text_linucb"] + spec["secondary_text"] if c in comp_vals}
        sstrong = max(strict, key=lambda c: strict[c].mean())
        s = {"strongest_with_A_only_text_readers": sstrong, "arms": {}}
        for k in spec["llm_arms"]:
            A = np.array([float(llm[k][e]["total_reward"]) for e in eps])
            pdv = paired_diff(A, strict[sstrong], rng)
            s["arms"][k] = {"vs_diff": pdv["diff"], "vs_lo95": pdv["lo95"], "C3": pdv["lo95"] > 0}
        s["deployment_seen_types_only"] = seen_type_deployment(cell, spec, rng)
        sec[cell] = s
    out["secondary"] = sec
    # summary
    n_arms = len(rows_table)
    out["summary"] = {
        "arms": n_arms,
        "agree_C2": sum(r[5] == r[6] for r in rows_table),
        "agree_C3": sum(r[7] == r[8] for r in rows_table),
        "agree_verdict": sum(r[3] == r[4] for r in rows_table),
        "expected_C2_agreements_from_power": POWER["expected_C2_agreements"],
        "replay_mismatches_llm": sum(r[13] for r in rows_table),
        "replay_mismatches_comparators": sum(v["replay_mismatches"] for c in out["cells"].values() for v in c["comparators"].values()),
        "disagreements": [{"cell": f"{r[0]}/{r[1]}", "arm": r[2], "deployment": r[3], "B": r[4], "C2_dep": r[5], "C2_B": r[6], "C3_dep": r[7],
                           "C3_B": r[8], "B_margin_lo95": r[9], "B_vs_lo95": r[10], "p_agree_C2": r[12]}
                          for r in rows_table if r[5] != r[6] or r[7] != r[8]],
    }
    (OUT / "calib_verdict.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8", newline="\n")
    for r in rows_table:
        flag = "" if (r[5] == r[6] and r[7] == r[8]) else "  <-- DISAGREE"
        print(f"{r[0]:8s} {r[1]:52s} {r[2]:45s} dep={r[3]:10s} B={r[4]:10s} mlo={r[9]:7.2f} vlo={r[10]:7.2f} cap={r[11]:6.2f} P2={r[12]:.2f} rp={r[13]}{flag}")
    print(json.dumps({k: v for k, v in out["summary"].items() if k != "disagreements"}, indent=1))


if __name__ == "__main__":
    main()
