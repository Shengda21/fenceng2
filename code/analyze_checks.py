"""Additional checks on the deployment records, none of them part of the pre-registered tests.

  fresh_seeds        V* and H_D of the typed MAgent cells on 500 seeds (5000-5499) that neither calibration nor
                     deployment used, beside the deployment and calibration readings of analysis/extras.json.
  sandbox_examples   the sandbox example arms, rerun with their examples in the prompt on a second GPU, against the
                     recorded zero-shot arms and zero-shot arms rerun on that GPU; paired by seed on seeds 0-99.
  sandbox_extension  the near-threshold sandbox cells (M=6, sharpness 0.6) on 500 seeds (0-99 and 320-719): each
                     gpt-oss-120b arm against the best fixed strategy and the strongest stateless non-LLM arm.
  linux_rerun        seed-by-seed agreement of a Linux rerun of the framework with the records.
  superseded_audit   the prompt-example count applied to the first sandbox example arms, whose runner dropped
                     the examples from the prompt.

Intervals are percentile bootstrap with B = 10^4, seeds resampled, the best fixed strategy re-selected inside each
replicate; every section draws from its own generator, so the sections are independent of one another.
writes analysis/checks.json
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

B = 10000
SEED = 20260930
FRESH_REF = {"battle-base4": "battle-base4-pool8-semantic-single", "battle-traits8": "battle-traits8-pool8-semantic-single",
             "combined-base4": "combined-base4-once", "combined-traits8": "combined-traits8-once",
             "combined-base4-delta06": "combined-base4-delta06-once"}
MODELS = ("gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b")
STATELESS = ("scripted", "plastic", "fewshot__1", "fewshot__5", "fewshot__20")


def read_rows(p):
    p = Path(p)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def values(p):
    return {r["seed"]: r.get("value", r.get("total_reward")) for r in read_rows(p)}


def pct(x, q):
    return np.percentile(x, q).tolist()


def fresh_seeds(data, extras):
    rng = np.random.default_rng(SEED)
    ex = extras["A3_uncertainty"]["cells"]
    out = {}
    for cell, ref in FRESH_REF.items():
        seeds, types, rows, pool = [], [], [], None
        for f in sorted((data / "checks" / "fresh_seeds" / cell).glob("*.json")):
            d = json.load(open(f, encoding="utf-8"))
            pe = d["per_episode"]
            pool = pool or sorted(pe)
            for m in d["manifest"]:
                seeds.append(m["seed"]); types.append(m["type1"])
                rows.append([pe[tau][m["key"]] for tau in pool])
        R = np.array(rows, dtype=float)
        n = R.shape[0]
        v_star = R.mean(0).max()
        h_d = R.max(1).mean() - v_star
        Rb = R[rng.integers(0, n, size=(B, n))]
        vb = Rb.mean(1).max(1)
        hb = Rb.max(2).mean(1) - vb
        T = np.array(types)
        ut = sorted(set(types))
        per = np.array([R[T == t].mean(0) for t in ut])
        v_ps = per.mean(0).max()
        h_ps = np.mean([R[T == t].max(1).mean() for t in ut]) - v_ps
        out[cell] = {"n": n, "seeds": [min(seeds), max(seeds)], "V_star": v_star, "V_star_ci": pct(vb, [2.5, 97.5]),
                     "H_D": h_d, "H_D_ci": pct(hb, [2.5, 97.5]), "V_star_ps": v_ps, "H_D_ps": h_ps,
                     "type_counts": {t: int((T == t).sum()) for t in ut}, "pool": pool,
                     "deployment": {k: ex[ref]["deployment"][k] for k in ("V_star", "V_star_ci", "H_D", "H_D_ci", "H_D_ps", "V_star_ps")},
                     "calibration": {k: ex[ref]["calibration"][k] for k in ("V_star", "H_D", "H_D_ps", "V_star_ps")}}
    return out


def sandbox_examples(data):
    rng = np.random.default_rng(SEED)
    ep = data / "episodes" / "sandbox"
    rerun = data / "checks" / "sandbox_zero_shot_rerun"
    cells = sorted(p.name[:-len("-semantic")] for p in ep.iterdir() if p.name.endswith("-semantic"))
    seeds = list(range(100))
    out = {}
    for c in cells:
        F = np.array([[values(ep / f"{c}-nonllm" / f"fixed__{t}.jsonl.gz")[s] for t in ("ROCK", "PAPER", "SCISSORS")] for s in seeds])
        vstar = F.mean(0).max()
        hd = F.max(1).mean() - vstar
        idx = rng.integers(0, 100, size=(B, 100))
        for m in MODELS:
            arms = {"N0_recorded": ep / f"{c}-semantic" / f"llm__{m}__single__nshot0.jsonl.gz",
                    "N0_rerun": rerun / f"{c}-semantic" / f"llm__{m}__single__nshot0.jsonl.gz",
                    "N1": ep / f"{c}-semantic" / f"llm__{m}__single__nshot1.jsonl.gz",
                    "N5": ep / f"{c}-semantic" / f"llm__{m}__single__nshot5.jsonl.gz"}
            x = {k: np.array([values(p)[s] for s in seeds]) for k, p in arms.items()}
            row = {"capture": {k: float((v.mean() - vstar) / hd) for k, v in x.items()}, "H_D": hd, "diff": {}}
            for a, b in (("N0_rerun", "N0_recorded"), ("N1", "N0_rerun"), ("N5", "N0_rerun")):
                d = (x[a] - x[b]) / hd
                row["diff"][f"{a}-{b}"] = {"mean": float(d.mean()), "ci": pct(d[idx].mean(1), [2.5, 97.5])}
            out[f"{c}|{m}"] = row
    return out


def sandbox_extension(data):
    rng = np.random.default_rng(SEED)
    ep = data / "episodes" / "sandbox"
    ext = data / "checks" / "sandbox_extension"
    out = {}
    for noise in ("0", "0.5"):
        base = f"sandbox-M6-K3-sharp0.6-noise{noise}"

        def pooled(head, tail_glob):
            d = {s: v for s, v in values(head).items() if s < 100}
            for f in sorted(ext.glob(tail_glob)):
                d.update(values(f))
            return d
        fixed = {t: pooled(ep / f"{base}-nonllm" / f"fixed__{t}.jsonl.gz", f"{base}-nonllm/fixed__{t}.jsonl.gz")
                 for t in ("ROCK", "PAPER", "SCISSORS")}
        stateless = {a: pooled(ep / f"{base}-nonllm" / f"{a}.jsonl.gz", f"{base}-nonllm/{a}.jsonl.gz") for a in STATELESS}
        llm = {"semantic, zero-shot": (f"{base}-semantic", "llm__gpt-oss-120b__single__nshot0", "nshot0"),
               "numeric, zero-shot": (f"{base}-numeric", "llm__gpt-oss-120b__single__nshot0", "nshot0"),
               "relabel, zero-shot": (f"{base}-relabel", "llm__gpt-oss-120b__single__nshot-0", "nshot0"),
               "semantic, N=5": (f"{base}-semantic", "llm__gpt-oss-120b__single__nshot5", "nshot5")}
        for label, (cell, arm, tag) in llm.items():
            L = pooled(ep / cell / f"{arm}.jsonl.gz", f"{cell}/llm__gpt-oss-120b__single__{tag}__*.jsonl.gz")
            seeds = sorted(set(L) & set.intersection(*(set(v) for v in fixed.values()))
                           & set.intersection(*(set(v) for v in stateless.values())))
            F = np.array([[fixed[t][s] for t in fixed] for s in seeds])
            l = np.array([L[s] for s in seeds])
            S = {a: np.array([stateless[a][s] for s in seeds]) for a in stateless}
            strongest = max(S, key=lambda a: S[a].mean())
            idx = rng.integers(0, len(seeds), size=(B, len(seeds)))
            vstar = F.mean(0).max()
            hd = F.max(1).mean() - vstar
            margin_b = l[idx].mean(1) - F[idx].mean(1).max(1)
            diff = l - S[strongest]
            diff_b = diff[idx].mean(1)
            dsc = l - S["scripted"]
            dsc_b = dsc[idx].mean(1)
            row = {"n": len(seeds), "seeds": "0-99, 320-719", "H_D": hd, "capture": float((l.mean() - vstar) / hd),
                   "margin": float(l.mean() - vstar), "margin_ci": pct(margin_b, [2.5, 97.5]),
                   "margin_lo95": float(np.percentile(margin_b, 5)), "strongest": strongest,
                   "diff": float(diff.mean()), "diff_ci": pct(diff_b, [2.5, 97.5]),
                   "diff_lo95": float(np.percentile(diff_b, 5)), "diff_hi95": float(np.percentile(diff_b, 95)),
                   "diff_scripted": float(dsc.mean()), "diff_scripted_ci": pct(dsc_b, [2.5, 97.5]),
                   "diff_scripted_hi95": float(np.percentile(dsc_b, 95)),
                   "stateless_captures": {a: float((S[a].mean() - vstar) / hd) for a in S}}
            row["verdict"] = ("pays" if row["margin_lo95"] > 0.5 and row["diff_lo95"] > 0 else
                              "fixed only" if row["margin_lo95"] > 0.5 else "no")
            h = round(row["diff_hi95"], 2)
            row["class"] = "excluded" if h < 0 else ("never above" if h == 0 else "within noise")
            out[f"{base}: {label}"] = row
    return out


def linux_rerun(data):
    lr = data / "checks" / "linux_rerun"
    first = json.load(open(lr / "repro_slim.json", encoding="utf-8"))
    pinned = json.load(open(lr / "repro_np253_slim.json", encoding="utf-8"))

    def compare(rec):
        out = {}
        for key, r in sorted(rec.items()):
            domain, _tag, cell, fname = key.split("/")
            rows = read_rows(data / "episodes" / domain / cell / (fname[:-len(".json")] + ".jsonl.gz"))
            o = {x["seed"]: ((x.get("decisions") or [{}])[0].get("tau", x.get("tau")), x.get("total_reward", x.get("value")))
                 for x in rows}
            n = eq = 0
            for seed, tau, rew in r["rows"]:
                if seed in o:
                    n += 1
                    eq += int(o[seed][0] == tau and abs(o[seed][1] - rew) < 1e-9)
            out[f"{domain}/{cell}/{fname[:-len('.json')]}"] = {"episodes": n, "identical": eq}
        return out
    a, b = compare(first), compare(pinned)
    final = {k: (b.get(k) or v) for k, v in a.items()}
    return {"arms": len(a), "first_run": {"episodes": sum(v["episodes"] for v in a.values()),
                                          "identical": sum(v["identical"] for v in a.values()),
                                          "arms_differing": sorted(k for k, v in a.items() if v["identical"] != v["episodes"])},
            "pinned_numpy_rerun": {"numpy": "2.5.3", "arms": sorted(b), "episodes": sum(v["episodes"] for v in b.values()),
                                   "identical": sum(v["identical"] for v in b.values())},
            "all_arms_identical_under_pinned_versions": all(v["identical"] == v["episodes"] for v in final.values()),
            "per_arm_first_run": a}


def superseded_audit(data):
    """The prompt-example count of analysis/extras.json (A0) applied to the first sandbox example arms."""
    import re
    out = {"files": 0, "prompts": 0, "prompts_with_expected_examples": 0, "found": {}}
    for f in sorted((data / "superseded" / "sandbox_example_arms").glob("*/*.jsonl.gz")):
        m = int(re.search(r"sandbox-M(\d+)-", f.parent.name).group(1))
        n = int(re.search(r"nshot(\d+)", f.name).group(1))
        out["files"] += 1
        for r in read_rows(f):
            for prov in r.get("provenance") or []:
                p = (prov.get("extra") or {}).get("prompt") or ""
                k = len(re.findall(r"(?m)^Strategy: ", p))
                out["prompts"] += 1
                out["prompts_with_expected_examples"] += int(k == n * m)
                out["found"][str(k)] = out["found"].get(str(k), 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    root = Path(ap.parse_args().root).resolve()
    data, out_dir = root / "data", root / "analysis"
    extras = json.load(open(out_dir / "extras.json", encoding="utf-8"))
    res = {"fresh_seeds": fresh_seeds(data, extras), "sandbox_examples": sandbox_examples(data),
           "sandbox_extension": sandbox_extension(data), "linux_rerun": linux_rerun(data),
           "superseded_audit": superseded_audit(data)}
    text = json.dumps(res, indent=1, sort_keys=True, default=float)
    (out_dir / "checks.json").write_text(text + "\n", encoding="utf-8")
    lr = res["linux_rerun"]
    print("wrote", out_dir / "checks.json")
    print(f"linux rerun: {lr['first_run']['identical']}/{lr['first_run']['episodes']} first run; pinned numpy "
          f"{lr['pinned_numpy_rerun']['identical']}/{lr['pinned_numpy_rerun']['episodes']}; all arms identical: "
          f"{lr['all_arms_identical_under_pinned_versions']}")


if __name__ == "__main__":
    main()
