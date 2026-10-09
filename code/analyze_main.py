"""Analysis: master table, per-type regret readings, encoding and n-shot curves, capture curves,
crossover horizons, sandbox closed forms and the scaling table, held-out types, headroom map.

Inputs : a results tree as produced by campaign.py (results/{domain}/{tag}/{cell}/{arm}.json[l]) plus the
         calibration locks, under --data.
Outputs: --out/master.json (everything), --out/tables/*.md, --out/figures/*.{pdf,png}, --out/summary.md.

Conventions
  * headline seeds are the first 100 deployment episodes of every arm (seeds 0-99 in every domain);
  * V* is the best OBSERVED fixed strategy on the headline seeds, re-selected inside every bootstrap replicate;
  * H_D (deployment) is the per-window oracle on the same seeds: at a single decision point behind a deterministic
    engine every fixed-strategy arm IS a branch at the decision point, so max over fixed arms per seed is Probe 2;
  * H_D^type is what the type oracle realises (knowing the type and looking up its counter);
  * capture c = (V_arm - V*) / H_D, paired by seed; learners' c(T) is the cumulative paired capture over episodes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np

B_DEFAULT = 10000
HEADLINE_N = 100
TIE_TOL = 0.5  # reward units within which two counters count as equally good for the misreading reading
LEARNERS = ("mucb", "linucb", "lints", "ctxucb", "plastic", "ppo")
INSTRUMENTS = ("random", "type_oracle")

# --------------------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------------------

def parse_arm(fname: str) -> dict:
    base = re.sub(r"\.jsonl(?:\.gz)?$|\.json$", "", fname)
    if base.startswith("fixed__"):
        return {"arm": "fixed", "tau": base[len("fixed__"):], "key": base}
    if base.startswith("fewshot__"):
        return {"arm": "fewshot", "N": int(base[len("fewshot__"):]), "key": base}
    m = re.match(r"llm__(.+?)__(.+?)__nshot-?(\d+)$", base)
    if m:
        return {"arm": "llm", "model": m.group(1), "mode": m.group(2), "nshot": int(m.group(3)), "key": base}
    return {"arm": base, "key": base}


def parse_cell(domain: str, cell: str) -> dict:
    meta = {"domain": domain, "cell": cell}
    if domain == "battle":
        m = re.match(r"battle-(base4|traits8)-(pool\d)-(\w+)-(\w+)$", cell)
        meta.update(type_set=m.group(1), pool=m.group(2), encoding=m.group(3), schedule=m.group(4),
                    M=4 if m.group(1) == "base4" else 8, ref=cell, group=f"battle-{m.group(1)}")
    elif domain == "combined":
        m = re.match(r"combined-(base4|traits8)-heldout-(\w+)$", cell)
        m2 = re.match(r"combined-(base4|traits8)-delta06-(\w+)$", cell)
        if m:
            meta.update(type_set=m.group(1), encoding="semantic", held_out=m.group(2), M=4,
                        ref=f"combined-{m.group(1)}-once", group=f"combined-{m.group(1)}")
        elif m2:
            enc = "semantic" if m2.group(2) == "once" else m2.group(2)
            meta.update(type_set=m2.group(1), encoding=enc, M=4, delta_scale=0.6,
                        ref=f"combined-{m2.group(1)}-delta06-once", group=f"combined-{m2.group(1)}-delta06")
        else:
            m = re.match(r"combined-(base4|traits8)-(\w+)$", cell)
            enc = "semantic" if m.group(2) == "once" else m.group(2)
            meta.update(type_set=m.group(1), encoding=enc, M=4 if m.group(1) == "base4" else 8,
                        ref=f"combined-{m.group(1)}-once", group=f"combined-{m.group(1)}")
    elif domain == "sandbox":
        m = re.match(r"sandbox-M(\d+)-K(\d+)-sharp([\d.]+)-noise([\d.]+)-(\w+)$", cell)
        enc = "semantic" if m.group(5) == "nonllm" else m.group(5)
        stem = f"sandbox-M{m.group(1)}-K{m.group(2)}-sharp{m.group(3)}-noise{m.group(4)}"
        meta.update(M=int(m.group(1)), K=int(m.group(2)), sharpness=float(m.group(3)), reward_noise=float(m.group(4)),
                    encoding=enc, ref=stem + "-nonllm", group=stem, type_set=f"M{m.group(1)}")
    return meta


def slim_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("total_reward") is None and r.get("value") is None:
            print("  skipping row without reward: seed", r.get("seed"), "error", str(r.get("error"))[:80], file=sys.stderr)
            continue
        if "value" in r and "tau" in r and not (r.get("decisions") or r.get("provenance")):
            out.append({
                "seed": int(r.get("seed")),
                "type": r.get("type"),
                "value": float(r.get("value")),
                "success": bool(r.get("success")),
                "tau": r.get("tau"),
                "fallback": bool(r.get("fallback")),
                "latency": float(r.get("latency") or 0.0),
                "n_dec": int(r.get("n_dec") or 0),
            })
            continue
        decs = r.get("decisions") or []
        prov = r.get("provenance") or []
        tau = None
        if decs:
            tau = decs[0].get("tau")
        elif prov:
            tau = prov[0].get("tau")
        fb = any(bool(d.get("fallback")) for d in decs) or any(bool(p.get("fallback")) for p in prov)
        lat = [float(p.get("latency_s") or 0.0) for p in prov]
        out.append({
            "seed": int(r.get("seed")),
            "type": r.get("type"),
            "value": float(r.get("total_reward", r.get("value"))),
            "success": bool(r.get("success")),
            "tau": tau,
            "fallback": bool(fb),
            "latency": float(np.mean(lat)) if lat else 0.0,
            "n_dec": len(decs) if decs else len(prov),
        })
    return out


def _read_rows(src: Path):
    if src.suffix == ".gz":
        import gzip
        with gzip.open(src, "rt", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    if src.suffix == ".jsonl":
        return [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    raw = json.load(open(src, encoding="utf-8"))
    if isinstance(raw, dict):
        return raw.get("rows") or raw.get("episodes") or []
    return raw


def load_tree(data: Path, cache: Path) -> dict:
    """Return {domain: {cell: {arm_key: {"meta":..., "rows":[...]}}}} using a slim cache."""
    tree: dict = defaultdict(lambda: defaultdict(dict))
    episodes = data / "episodes"
    if episodes.exists():
        for domain in ("battle", "combined", "sandbox"):
            droot = episodes / domain
            if not droot.exists():
                continue
            order_file = episodes / "cell_order.json"
            if order_file.exists():
                cell_dirs = [droot / c for c in json.load(open(order_file, encoding="utf-8"))[domain]]
            else:
                cell_dirs = sorted(p for p in droot.iterdir() if p.is_dir())
            for cell_dir in cell_dirs:
                for src in sorted(cell_dir.glob("*.jsonl.gz")):
                    rows = slim_rows(_read_rows(src))
                    meta = parse_arm(src.name)
                    tree[domain][cell_dir.name][meta["key"]] = {"meta": meta, "rows": rows, "tag": ""}
        return tree
    results = data / "results"
    for domain in ("battle", "combined", "sandbox"):
        droot = results / domain
        if not droot.exists():
            continue
        for tag in sorted(os.listdir(droot)):
            troot = droot / tag
            if not troot.is_dir():
                continue
            for cell in sorted(os.listdir(troot)):
                croot = troot / cell
                if not croot.is_dir():
                    continue
                for fname in sorted(os.listdir(croot)):
                    if not re.search(r"\.jsonl(?:\.gz)?$|\.json$", fname):
                        continue
                    src = croot / fname
                    dst = cache / domain / tag / cell / (re.sub(r"\.jsonl(?:\.gz)?$|\.json$", "", fname) + ".slim.json")
                    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
                        rows = json.load(open(dst, encoding="utf-8"))
                    else:
                        raw = _read_rows(src)
                        rows = slim_rows(raw)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        json.dump(rows, open(dst, "w", encoding="utf-8"))
                    meta = parse_arm(fname)
                    tree[domain][cell][meta["key"]] = {"meta": meta, "rows": rows, "tag": tag}
    return tree


# --------------------------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------------------------

def by_seed(rows: list[dict]) -> dict[int, float]:
    d = {}
    for r in rows:
        d.setdefault(r["seed"], r["value"])  # first occurrence (episode order == seed order for headline)
    return d


def pct(a, q):
    return float(np.percentile(a, q))


class Reference:
    """Fixed-strategy sweep and oracles on the headline seeds of a reference cell."""

    def __init__(self, cell_arms: dict, default: str | None):
        fixed = {k: v for k, v in cell_arms.items() if v["meta"]["arm"] == "fixed"}
        if not fixed:
            raise ValueError("reference cell has no fixed arms")
        seed_sets = [set(by_seed(v["rows"]).keys()) for v in fixed.values()]
        seeds = sorted(set.intersection(*seed_sets))
        seeds = [s for s in seeds if s < 1000][:HEADLINE_N]
        self.seeds = seeds
        self.taus = [v["meta"]["tau"] for v in fixed.values()]
        self.F = np.array([[by_seed(v["rows"])[s] for s in seeds] for v in fixed.values()])  # taus x seeds
        first = next(iter(fixed.values()))["rows"]
        types = {r["seed"]: r["type"] for r in first}
        self.types = [types[s] for s in seeds]
        self.default = default
        self.means = self.F.mean(axis=1)
        self.best_i = int(np.argmax(self.means))
        self.best_tau = self.taus[self.best_i]
        self.V_star = float(self.means[self.best_i])
        self.V_default = float(self.means[self.taus.index(default)]) if default in self.taus else float("nan")
        self.Omax = self.F.max(axis=0)  # per-window oracle per seed
        self.H_D = float(self.Omax.mean() - self.V_star)
        self.V_min = float(self.means.min())
        self.share = self.H_D / (self.V_star - self.V_min) if self.V_star > self.V_min else float("nan")
        to = cell_arms.get("type_oracle")
        if to is not None:
            bs = by_seed(to["rows"])
            self.type_oracle = np.array([bs[s] for s in seeds])
            self.H_type = float(self.type_oracle.mean() - self.V_star)
        else:
            self.type_oracle = None
            self.H_type = float("nan")
        # per-type table
        self.per_type = OrderedDict()
        for th in sorted(set(self.types)):
            idx = [i for i, t in enumerate(self.types) if t == th]
            m = self.F[:, idx].mean(axis=1)
            order = np.argsort(-m)
            best, second = float(m[order[0]]), float(m[order[1]]) if len(order) > 1 else float("nan")
            good = [self.taus[i] for i in range(len(self.taus)) if m[i] >= best - TIE_TOL]
            self.per_type[th] = {
                "share": len(idx) / len(seeds), "n": len(idx),
                "best_tau": self.taus[int(order[0])], "best_value": best, "Delta": best - second,
                "good_set": good, "values": {self.taus[i]: float(m[i]) for i in range(len(self.taus))},
                "seeds": [seeds[i] for i in idx],
            }
        self.Fstar_seed = self.F[self.best_i]  # population-best fixed strategy per seed

    def bootstrap_idx(self, B: int, rng: np.random.Generator):
        return rng.integers(0, len(self.seeds), size=(B, len(self.seeds)))

    def boot_refs(self, idx):
        Fb = self.F[:, idx]  # taus x B x n
        means_b = Fb.mean(axis=2)  # taus x B
        Vstar_b = means_b.max(axis=0)
        H_b = Fb.max(axis=0).mean(axis=1) - Vstar_b
        return Vstar_b, H_b


def arm_stats(ref: Reference, rows: list[dict], B: int, rng: np.random.Generator) -> dict:
    bs = by_seed(rows)
    seeds = [s for s in ref.seeds if s in bs]
    if len(seeds) < len(ref.seeds):
        # align on the seeds present in both
        mask = np.array([s in bs for s in ref.seeds])
    else:
        mask = np.ones(len(ref.seeds), dtype=bool)
    A = np.array([bs[s] for s in np.array(ref.seeds)[mask]])
    F = ref.F[:, mask]
    n = len(A)
    if n == 0:
        print("  no seed overlap with reference:", sorted(bs)[:5], "vs", ref.seeds[:5], file=sys.stderr)
        return {"n": 0, "mean": float("nan"), "success": float("nan"), "margin": float("nan"), "margin_ci": [float("nan")] * 2,
                "margin_lo95": float("nan"), "capture": float("nan"), "capture_ci": [float("nan")] * 2, "capture_lo95": float("nan"),
                "fallback_rate": float("nan"), "latency": float("nan"), "values": {}, "per_type": {}, "expected_regret_pertype": float("nan")}
    idx = rng.integers(0, n, size=(B, n))
    Fb = F[:, idx]
    means_b = Fb.mean(axis=2)
    Vstar_b = means_b.max(axis=0)
    H_b = Fb.max(axis=0).mean(axis=1) - Vstar_b
    Ab = A[idx].mean(axis=1)
    margin_b = Ab - Vstar_b
    with np.errstate(divide="ignore", invalid="ignore"):
        cap_b = np.where(H_b > 0, margin_b / H_b, np.nan)
    V_star = float(F.mean(axis=1).max())
    H = float(F.max(axis=0).mean() - V_star)
    mean = float(A.mean())
    used = set(int(s) for s in np.array(ref.seeds)[mask].tolist())
    succ = [r["success"] for r in rows if r["seed"] in used]
    out = {
        "n": n, "mean": mean, "success": float(np.mean(succ)) if succ else float("nan"),
        "margin": mean - V_star, "margin_ci": [pct(margin_b, 2.5), pct(margin_b, 97.5)], "margin_lo95": pct(margin_b, 5),
        "capture": (mean - V_star) / H if H > 0 else float("nan"),
        "capture_ci": [pct(cap_b[~np.isnan(cap_b)], 2.5), pct(cap_b[~np.isnan(cap_b)], 97.5)] if np.any(~np.isnan(cap_b)) else [float("nan")] * 2,
        "capture_lo95": pct(cap_b[~np.isnan(cap_b)], 5) if np.any(~np.isnan(cap_b)) else float("nan"),
        "fallback_rate": float(np.mean([r["fallback"] for r in rows])),
        "latency": float(np.mean([r["latency"] for r in rows])),
        "values": {int(s): float(bs[s]) for s in np.array(ref.seeds)[mask]},
    }
    # per-type reading: value, misreading frequency and cost
    chosen = {r["seed"]: r["tau"] for r in rows}
    per_type = OrderedDict()
    for th, info in ref.per_type.items():
        sd = [s for s in info["seeds"] if s in bs]
        if not sd:
            continue
        vals = np.array([bs[s] for s in sd])
        mis = [s for s in sd if chosen.get(s) not in info["good_set"]]
        best_i = ref.taus.index(info["best_tau"])
        fstar = {s: ref.F[best_i][ref.seeds.index(s)] for s in sd}
        cost = float(np.mean([fstar[s] - bs[s] for s in mis])) if mis else 0.0
        per_type[th] = {
            "mean": float(vals.mean()), "n": len(sd), "p_mis": len(mis) / len(sd), "cost": cost,
            "regret_vs_best_counter": float(np.mean([fstar[s] - bs[s] for s in sd])),
            "choices": dict(Counter(chosen.get(s) for s in sd)),
        }
    out["per_type"] = per_type
    out["expected_regret_pertype"] = float(sum(info["share"] * per_type[th]["p_mis"] * per_type[th]["cost"]
                                             for th, info in ref.per_type.items() if th in per_type))
    return out


def paired_diff(a_rows, b_rows, B, rng):
    a, b = by_seed(a_rows), by_seed(b_rows)
    seeds = sorted(set(a) & set(b))
    seeds = [s for s in seeds if s < 1000][:HEADLINE_N]
    d = np.array([a[s] - b[s] for s in seeds])
    idx = rng.integers(0, len(d), size=(B, len(d)))
    db = d[idx].mean(axis=1)
    return {"n": len(d), "diff": float(d.mean()), "ci": [pct(db, 2.5), pct(db, 97.5)], "lo95": pct(db, 5), "hi95": pct(db, 95)}


def capture_curve_paired(ref: Reference, rows: list[dict]) -> list[float]:
    """Cumulative paired capture over the learner's episodes in run order, on the reference seeds (<=100)."""
    bs = OrderedDict()
    for r in rows:
        if r["seed"] in bs:
            continue
        bs[r["seed"]] = r["value"]
    num = den = 0.0
    curve = []
    seed_pos = {s: i for i, s in enumerate(ref.seeds)}
    for s, v in bs.items():
        if s not in seed_pos:
            break
        i = seed_pos[s]
        num += v - ref.Fstar_seed[i]
        den += ref.Omax[i] - ref.Fstar_seed[i]
        curve.append(num / den if den > 0 else float("nan"))
    return curve


def capture_curve_population(rows: list[dict], V_star: float, H_D: float) -> list[float]:
    vals = np.array([r["value"] for r in rows], dtype=float)
    cum = np.cumsum(vals) / np.arange(1, len(vals) + 1)
    return ((cum - V_star) / H_D).tolist() if H_D > 0 else [float("nan")] * len(vals)


def crossover(curve: list[float], target: float) -> int | None:
    """First T after which the curve stays >= target to the end of the observed range (persistent crossing)."""
    c = np.array(curve, dtype=float)
    if len(c) == 0 or np.all(np.isnan(c)):
        return None
    ok = c >= target
    # persistent from T onward
    T = None
    for t in range(len(c) - 1, -1, -1):
        if not ok[t]:
            break
        T = t + 1
    return T


# --------------------------------------------------------------------------------------------------------------
# sandbox closed forms
# --------------------------------------------------------------------------------------------------------------

def sandbox_closed(meta: dict, sandbox_root: Path, cache: Path, samples: int, seed_start: int = 0) -> dict:
    key = f"closed-M{meta['M']}-K{meta['K']}-sharp{meta['sharpness']}-noise{meta['reward_noise']}-s{samples}" + (f"-from{seed_start}" if seed_start else "") + ".json"
    f = cache / key
    if f.exists():
        return json.load(open(f))
    for p in (str(sandbox_root), str(sandbox_root.parent / "v8_lib")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from closed_form import compute_closed_forms  # type: ignore
    from sandbox.env import SandboxConfig  # type: ignore
    cfg = SandboxConfig(M=meta["M"], K=meta["K"], delta=1.0, sharpness=meta["sharpness"], reward_noise=meta["reward_noise"],
                        H=30, k=5, schedule="single")
    closed = compute_closed_forms(cfg, samples=samples, seed_start=seed_start)
    json.dump(closed, open(f, "w"), indent=1)
    return closed


# --------------------------------------------------------------------------------------------------------------
# main analysis
# --------------------------------------------------------------------------------------------------------------

def analyse(tree: dict, data: Path, out: Path, sandbox_root: Path, B: int, closed_samples: int) -> dict:
    rng = np.random.default_rng(20260907)
    locks = {}
    for f in (data / "locks").glob("rs_*lock*.json"):
        try:
            locks[f.name] = json.load(open(f, encoding="utf-8"))
        except Exception:
            pass
    defaults = {"battle": "HOLD_POSITION", "combined": "HOLD_LINE", "sandbox": "ROCK"}
    lock_for = {
        "battle-base4": "rs_magent_lock_base4.json", "battle-traits8": "rs_magent_lock_traits8.json",
        "combined-base4": "rs_combined_lock.json", "combined-traits8": "rs_combined_traits8_lock.json",
        "combined-base4-delta06": "rs_combined_lock_delta06.json",
    }
    master = OrderedDict()
    refs: dict[str, Reference] = {}
    closed_cache = out / "closed"
    closed_cache.mkdir(parents=True, exist_ok=True)

    # references first
    for domain, cells in tree.items():
        for cell, arms in cells.items():
            meta = parse_cell(domain, cell)
            if meta["ref"] == cell and any(v["meta"]["arm"] == "fixed" for v in arms.values()):
                refs[cell] = Reference(arms, defaults[domain])

    for domain, cells in tree.items():
        for cell, arms in cells.items():
            meta = parse_cell(domain, cell)
            ref = refs.get(meta["ref"])
            if ref is None:
                print("no reference for", cell, file=sys.stderr)
                continue
            entry = OrderedDict(meta=meta, reference=OrderedDict(
                seeds_n=len(ref.seeds), V_star=ref.V_star, best_fixed=ref.best_tau, V_default=ref.V_default,
                H_D=ref.H_D, H_type=ref.H_type, V_min=ref.V_min, share=ref.share,
                fixed_means={t: float(m) for t, m in zip(ref.taus, ref.means)},
                per_type={th: {k: v for k, v in info.items() if k != "seeds"} for th, info in ref.per_type.items()},
                type_shares=dict(Counter(ref.types)),
            ))
            lk = locks.get(lock_for.get(meta.get("group", ""), ""))
            if meta.get("held_out"):
                lk = locks.get(f"rs_combined_lock_base4_heldout_{meta['held_out']}.json")
            if lk:
                entry["lock"] = {k: lk.get(k) for k in ("V_star", "best_fixed", "V_default", "Delta_def", "placebo", "H_D",
                                                       "headroom_share", "pool_adequacy", "best_response_table", "Delta_per_type", "locked_at")}
            if domain == "sandbox":
                entry["closed"] = sandbox_closed(meta, sandbox_root, closed_cache, closed_samples)
                # the same computation on the deployment seeds 0-99: the probe on those seeds must reproduce it
                entry["closed_headline"] = sandbox_closed(meta, sandbox_root, closed_cache, HEADLINE_N, seed_start=0)
            arm_out = OrderedDict()
            for key, a in arms.items():
                st = arm_stats(ref, a["rows"], B, rng)
                st["meta"] = a["meta"]
                st["episodes"] = len(a["rows"])
                if a["meta"]["arm"] in LEARNERS or a["meta"]["arm"] == "fewshot":
                    st["curve_paired"] = capture_curve_paired(ref, a["rows"])
                    if domain == "sandbox":
                        st["curve_population"] = capture_curve_population(a["rows"], entry["closed"]["V_star"], entry["closed"]["H_D"])
                    else:
                        st["curve_population"] = capture_curve_population(a["rows"], ref.V_star, ref.H_D)
                arm_out[key] = st
            # strongest non-LLM candidate on this cell (learners, scripted, fewshot, ppo)
            cands = {k: v for k, v in arm_out.items() if v["meta"]["arm"] in LEARNERS + ("scripted", "fewshot")}
            strongest = max(cands, key=lambda k: cands[k]["mean"]) if cands else None
            entry["strongest_nonllm"] = strongest
            for key, v in arm_out.items():
                if v["meta"]["arm"] == "llm" and strongest:
                    v["vs_strongest"] = paired_diff(arms[key]["rows"], arms[strongest]["rows"], B, rng)
                    v["vs_strongest_arm"] = strongest
            for key, v in arm_out.items():
                if "curve_paired" in v:
                    v["T_80"] = crossover(v["curve_paired"], 0.8)
                    v["T_50"] = crossover(v["curve_paired"], 0.5)
                    v["capture_at"] = {str(T): (v["curve_paired"][T - 1] if len(v["curve_paired"]) >= T else None) for T in (10, 30, 100)}
                    if "curve_population" in v:
                        v["T_80_population"] = crossover(v["curve_population"], 0.8)
                        v["T_50_population"] = crossover(v["curve_population"], 0.5)
                        v["capture_at_population"] = {str(T): (v["curve_population"][T - 1] if len(v["curve_population"]) >= T else None) for T in (10, 30, 100, 300)}
            entry["arms"] = arm_out
            master[cell] = entry

    # strongest non-LLM comparator for cells that hold none (sandbox LLM cells): take it from the reference cell,
    # whose arms ran the same headline seeds
    for cell, entry in master.items():
        if entry.get("strongest_nonllm") is not None or entry["meta"]["ref"] not in master:
            continue
        ref_entry = master[entry["meta"]["ref"]]
        strongest = ref_entry.get("strongest_nonllm")
        if not strongest:
            continue
        entry["strongest_nonllm"] = strongest
        entry["strongest_nonllm_cell"] = entry["meta"]["ref"]
        dom = entry["meta"]["domain"]
        for key, v in entry["arms"].items():
            if v["meta"]["arm"] == "llm":
                v["vs_strongest"] = paired_diff(tree[dom][cell][key]["rows"], tree[dom][entry["meta"]["ref"]][strongest]["rows"], B, rng)
                v["vs_strongest_arm"] = strongest

    # crossover post-pass: every learner curve in the cell, or in the cell's reference cell when the cell holds none
    # (sandbox LLM cells), against every LLM arm's flat capture
    for cell, entry in master.items():
        learners_here = {k: v for k, v in entry["arms"].items() if "curve_paired" in v}
        if not learners_here and entry["meta"]["ref"] in master:
            learners_here = {k: v for k, v in master[entry["meta"]["ref"]]["arms"].items() if "curve_paired" in v}
        for key, v in entry["arms"].items():
            if v["meta"]["arm"] != "llm" or not learners_here:
                continue
            tstar = OrderedDict()
            c_llm = v["capture"]
            lo, hi = v["capture_ci"][0], v["capture_ci"][1]
            # the population curves are normalised by the learner cell's own (V*, H_D); the model's mean goes on that scale
            pop_cell = cell if any("curve_paired" in a for a in entry["arms"].values()) else entry["meta"]["ref"]
            pop_ref = master[pop_cell].get("closed") if entry["meta"]["domain"] == "sandbox" else None
            if pop_ref and pop_ref["H_D"] > 0:
                c_llm_pop = (v["mean"] - pop_ref["V_star"]) / pop_ref["H_D"]
            else:
                c_llm_pop = c_llm
            v["capture_population"] = c_llm_pop
            for lk_, lv in learners_here.items():
                tstar[lk_] = {
                    "T_star": crossover(lv["curve_paired"], c_llm),
                    "T_star_at_llm_lo": crossover(lv["curve_paired"], lo),
                    "T_star_at_llm_hi": crossover(lv["curve_paired"], hi),
                    "T_max": len(lv["curve_paired"]),
                }
                if "curve_population" in lv:
                    tstar[lk_]["T_star_population"] = crossover(lv["curve_population"], c_llm_pop)
                    tstar[lk_]["T_max_population"] = len(lv["curve_population"])
            v["crossover"] = tstar
            v["crossover_learner_cell"] = cell if any("curve_paired" in a for a in entry["arms"].values()) else entry["meta"]["ref"]

    # held-out reading: per held-out type, compare the arms in the held-out cell with the same arms in the seen cell
    heldout = OrderedDict()
    for cell, entry in master.items():
        ho = entry["meta"].get("held_out")
        if not ho:
            continue
        seen_cell = f"combined-{entry['meta']['type_set']}-semantic"
        seen = master.get(seen_cell)
        ref = refs[entry["meta"]["ref"]]
        info = ref.per_type[ho]
        rows = OrderedDict()
        for key, v in entry["arms"].items():
            pt = v["per_type"].get(ho)
            seen_pt = seen["arms"][key]["per_type"].get(ho) if seen and key in seen["arms"] else None
            rows[key] = {"unseen_mean": pt["mean"] if pt else None, "unseen_p_mis": pt["p_mis"] if pt else None,
                         "seen_mean": seen_pt["mean"] if seen_pt else None, "seen_p_mis": seen_pt["p_mis"] if seen_pt else None,
                         "best_counter_value": info["best_value"], "best_counter": info["best_tau"], "n": pt["n"] if pt else 0,
                         "overall_mean": v["mean"], "overall_capture": v["capture"]}
        heldout[cell] = {"held_out": ho, "share": info["share"], "arms": rows}

    result = OrderedDict(master=master, heldout=heldout, B=B, headline_n=HEADLINE_N)
    return result


# --------------------------------------------------------------------------------------------------------------
# tables and figures
# --------------------------------------------------------------------------------------------------------------

def fmt(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def arm_label(meta: dict) -> str:
    a = meta["arm"]
    if a == "fixed":
        return f"fixed:{meta['tau']}"
    if a == "fewshot":
        return f"fewshot:{meta['N']}"
    if a == "llm":
        return f"{meta['model']}:{meta['mode']}:n{meta['nshot']}"
    return a


def write_tables(res: dict, out: Path):
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    lines = ["# Master table (headline seeds, paired bootstrap)\n"]
    for cell, e in res["master"].items():
        r = e["reference"]
        lines.append(f"\n## {cell}\n")
        lines.append(f"V* = {fmt(r['V_star'])} ({r['best_fixed']}), V_default = {fmt(r['V_default'])}, "
                     f"H_D (per-window oracle) = {fmt(r['H_D'])}, H_D^type = {fmt(r['H_type'])}, share = {fmt(r['share'])}, "
                     f"types = {r['type_shares']}")
        if e.get("lock"):
            lk = e["lock"]
            lines.append(f"lock (calibration): V* = {fmt(lk['V_star'])} ({lk['best_fixed']}), H_D = {fmt(lk['H_D'])}, share = {fmt(lk['headroom_share'])}, "
                         f"best responses = {lk['best_response_table']}")
        if e.get("closed"):
            c = e["closed"]
            lines.append(f"closed form: V* = {fmt(c['V_star'])} ({c['best_tau']}), H_D = {fmt(c['H_D'])}, Delta_min = {fmt(c['Delta_min'])}, "
                         f"sigma = {fmt(c['sigma_eff'])}, snr = {fmt(c['snr'])}, predicted T* = {fmt(c['predicted_T_star'])}")
        lines.append("\nper type: " + "; ".join(f"{th}: share {fmt(i['share'])}, best {i['best_tau']} ({fmt(i['best_value'])}), Delta {fmt(i['Delta'])}"
                                             for th, i in r["per_type"].items()))
        lines.append("\n| arm | n | mean | margin vs V* [95% CI] | capture [CI] | fallback | vs strongest non-LLM | E[regret] per-type | p_mis by type |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for key, v in e["arms"].items():
            vs = v.get("vs_strongest")
            vs_s = f"{fmt(vs['diff'])} [{fmt(vs['ci'][0])}, {fmt(vs['ci'][1])}] vs {v['vs_strongest_arm']}" if vs else ""
            pm = ", ".join(f"{th}:{fmt(pt['p_mis'])}" for th, pt in v["per_type"].items())
            lines.append(f"| {arm_label(v['meta'])} | {v['n']} | {fmt(v['mean'])} | {fmt(v['margin'])} [{fmt(v['margin_ci'][0])}, {fmt(v['margin_ci'][1])}] | "
                         f"{fmt(v['capture'])} [{fmt(v['capture_ci'][0])}, {fmt(v['capture_ci'][1])}] | {fmt(v['fallback_rate'])} | {vs_s} | {fmt(v['expected_regret_pertype'])} | {pm} |")
        curves = [(k, v) for k, v in e["arms"].items() if "curve_paired" in v]
        if curves:
            lines.append("\ncapture curves (paired, T=10/30/100) and T_80: " + "; ".join(
                f"{arm_label(v['meta'])}: {fmt(v['capture_at']['10'])}/{fmt(v['capture_at']['30'])}/{fmt(v['capture_at']['100'])}, T80={v['T_80']}"
                for k, v in curves))
        llms = [(k, v) for k, v in e["arms"].items() if v["meta"]["arm"] == "llm" and v.get("crossover")]
        for k, v in llms:
            lc = res["master"][v.get("crossover_learner_cell", cell)]["arms"]
            lines.append(f"crossover vs {arm_label(v['meta'])} (capture {fmt(v['capture'])}): " + "; ".join(
                f"{arm_label(lc[lk]['meta'])}: T*={t['T_star']} [{t['T_star_at_llm_hi']}, {t['T_star_at_llm_lo']}]"
                + (f", pop300 T*={t.get('T_star_population')}" if 'T_star_population' in t else "")
                for lk, t in v["crossover"].items()))
    (tdir / "master.md").write_text("\n".join(lines), encoding="utf-8")

    # held-out table
    lines = ["# Held-out types\n", "| cell | held-out (share) | arm | unseen mean | seen mean | best counter value | unseen p_mis | seen p_mis |", "|---|---|---|---|---|---|---|---|"]
    for cell, h in res["heldout"].items():
        for key, r in h["arms"].items():
            lines.append(f"| {cell} | {h['held_out']} ({fmt(h['share'])}) | {key} | {fmt(r['unseen_mean'])} | {fmt(r['seen_mean'])} | {fmt(r['best_counter_value'])} ({r['best_counter']}) | {fmt(r['unseen_p_mis'])} | {fmt(r['seen_p_mis'])} |")
    (tdir / "heldout.md").write_text("\n".join(lines), encoding="utf-8")

    # encoding table (LLM single nshot0 and learners) per group
    lines = ["# Encoding (capture, headline seeds)\n", "| group | encoding | arm | mean | capture [CI] | fallback |", "|---|---|---|---|---|---|"]
    for cell, e in res["master"].items():
        if e["meta"].get("held_out"):
            continue
        for key, v in e["arms"].items():
            m = v["meta"]
            if m["arm"] == "llm" and not (m["mode"] == "single" and m["nshot"] == 0):
                continue
            if m["arm"] in ("fixed", "random", "type_oracle"):
                continue
            lines.append(f"| {e['meta']['group']} | {e['meta']['encoding']} | {arm_label(m)} | {fmt(v['mean'])} | {fmt(v['capture'])} [{fmt(v['capture_ci'][0])}, {fmt(v['capture_ci'][1])}] | {fmt(v['fallback_rate'])} |")
    (tdir / "encoding.md").write_text("\n".join(lines), encoding="utf-8")

    # n-shot / mode table for LLMs (semantic cells)
    lines = ["# LLM n-shot and prompt mode (semantic encoding)\n", "| cell | model | mode | nshot | mean | capture [CI] | margin lo95 | vs strongest [CI] | fallback | latency s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cell, e in res["master"].items():
        if e["meta"]["encoding"] != "semantic" or e["meta"].get("held_out"):
            continue
        for key, v in e["arms"].items():
            m = v["meta"]
            if m["arm"] != "llm":
                continue
            vs = v.get("vs_strongest")
            lines.append(f"| {cell} | {m['model']} | {m['mode']} | {m['nshot']} | {fmt(v['mean'])} | {fmt(v['capture'])} [{fmt(v['capture_ci'][0])}, {fmt(v['capture_ci'][1])}] | {fmt(v['margin_lo95'])} | "
                         f"{fmt(vs['diff']) if vs else '--'} [{fmt(vs['ci'][0]) if vs else '--'}, {fmt(vs['ci'][1]) if vs else '--'}] ({v.get('vs_strongest_arm')}) | {fmt(v['fallback_rate'])} | {fmt(v['latency'])} |")
    (tdir / "llm_nshot.md").write_text("\n".join(lines), encoding="utf-8")

    # sandbox scaling table
    lines = ["# Sandbox: closed forms, empirical probes, crossover and prediction\n",
             "| cell | M | sharp | noise | closed V* (20k) | closed V* (seeds 0-99) | emp V* | closed H_D (20k) | closed H_D (0-99) | emp H_D | H_type | Delta_min | snr | pred T* (MK/snr^2) | learner T50 / T80 (paired<=100; pop<=300) | LLM captures (single n0) | T* learners vs LLM (paired / pop300) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell, e in res["master"].items():
        if e["meta"]["domain"] != "sandbox" or e["meta"]["encoding"] != "semantic" or "nonllm" not in cell:
            continue
        c, ch, r = e["closed"], e["closed_headline"], e["reference"]
        t80 = "; ".join(f"{arm_label(v['meta'])}: {v['T_50']}/{v['T_80']}; {v.get('T_50_population')}/{v.get('T_80_population')}" for k, v in e["arms"].items() if "curve_paired" in v)
        sib = res["master"].get(cell.replace("-nonllm", "-semantic"))
        caps = "; ".join(f"{v['meta']['model']}: {fmt(v['capture'])}" for k, v in (sib["arms"].items() if sib else []) if v["meta"]["arm"] == "llm" and v["meta"]["nshot"] == 0)
        ts = ""
        if sib:
            parts = []
            for k, v in sib["arms"].items():
                if v["meta"]["arm"] == "llm" and v["meta"]["nshot"] == 0 and v.get("crossover"):
                    parts.append(v["meta"]["model"] + ": " + ", ".join(f"{arm_label(e['arms'][lk]['meta'])}={t['T_star']}/{t.get('T_star_population')}" for lk, t in v["crossover"].items()))
            ts = " 鈥?".join(parts)
        lines.append(f"| {cell} | {e['meta']['M']} | {e['meta']['sharpness']} | {e['meta']['reward_noise']} | {fmt(c['V_star'])} | {fmt(ch['V_star'])} | {fmt(r['V_star'])} | {fmt(c['H_D'])} | {fmt(ch['H_D'])} | {fmt(r['H_D'])} | {fmt(r['H_type'])} | {fmt(c['Delta_min'])} | {fmt(c['snr'])} | {fmt(c['predicted_T_star'],1)} | {t80} | {caps} | {ts} |")
    (tdir / "sandbox.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sandbox-root", default=None)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--closed-samples", type=int, default=20000)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    data = Path(args.data).resolve() if args.data else root / "data"
    out = Path(args.out).resolve() if args.out else root / "analysis"
    args.sandbox_root = args.sandbox_root or str(root / "code" / "oghp" / "v8_sandbox")
    out.mkdir(parents=True, exist_ok=True)
    tree = load_tree(data, data / "slim")
    print({d: len(c) for d, c in tree.items()})
    res = analyse(tree, data, out, Path(args.sandbox_root), args.B, args.closed_samples)
    # the per-seed values stay in the results (the figures need them)
    slim = json.loads(json.dumps(res, default=float))
    for cell in slim["master"].values():
        for v in cell["arms"].values():
            v.pop("values", None)
    json.dump(slim, open(out / "master.json", "w", encoding="utf-8"), indent=1)
    write_tables(slim, out)
    print("written", out)


if __name__ == "__main__":
    main()
