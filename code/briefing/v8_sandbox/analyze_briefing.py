"""Briefing analysis: capture, paired bootstrap, strata (seen/held-out, reactivity), per-type misreading and loss,
capture curves, persistent crossover, and the paper's "LLM pays" verdict per LLM arm, with the
prediction tests P-A..P-F.

Headline seeds are the first 100 deployment seeds (< 1000) shared by the fixed arms of
the group's nonllm cell; V* is the best observed fixed strategy on those seeds, re-selected inside every bootstrap
replicate; H_D is the per-seed max over fixed arms minus V*; capture = (V_arm - V*) / H_D, paired by seed.
A group is one base cell under one held-out kind (`react` primary, `diag` secondary); cells are named
`sandbox-traits-M{M}-K{K}-sharp{s}-noise{sigma}-{kind}-{encoding}[-{wording}]`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "v8_lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v8lib.analysis import capture, capture_curve, crossover, wilson

from closed_form import delta_min, exact_reference, predicted_T_star, sigma_eff, snr
from sandbox.briefing import held_out_set_name, resolve_held_out_set
from sandbox.env import SandboxConfig, SandboxEnv, deployment_types
from sandbox.game import parse_trait

HEADLINE_N = 100
TIE_TOL = 0.5
PAYS_MARGIN = 0.5
ALPHA = 0.05
PREFIX_READERS = ("scripted", "fewshot", "plastic", "linucb", "lints", "ctxucb", "mucb", "ppo")
TEXT_READERS = ("scripted_text", "text_bow", "text_embed", "text_linucb")
FREE_BANDITS = ("linucb", "lints", "ctxucb", "mucb", "text_linucb")
CURVE_ARMS = ("fewshot", "plastic", "linucb", "lints", "ctxucb", "mucb", "ppo", "text_bow", "text_embed", "text_linucb")
CELL_RE = re.compile(
    r"(sandbox-traits-M(\d+)-K(\d+)-sharp([\d.]+)-noise([\d.]+))-(react8|react|diag)-(nonllm|briefing_prefix|briefing|semantic)(?:-(sameword|newword))?$"
)
PRIMARY = {"M": 18, "reward_noise": 0.0, "kind": "react", "cell": "briefing-newword", "mode": "single", "nshot": 0}
PRIMARY_0B = {**PRIMARY, "model": "gpt-oss-120b"}
DEPLOY = {"types": "all"}


def parse_arm(fname: str) -> dict:
    base = re.sub(r"\.jsonl(?:\.gz)?$", "", fname)
    if base.startswith("fixed__"):
        return {"arm": "fixed", "tau": base[len("fixed__"):], "key": base}
    if base.startswith("fewshot__"):
        return {"arm": "fewshot", "N": int(base[len("fewshot__"):]), "key": base}
    m = re.match(r"llm__(.+?)__(.+?)__nshot-?(\d+)$", base)
    if m:
        return {"arm": "llm", "model": m.group(1), "mode": m.group(2), "nshot": int(m.group(3)), "key": base}
    # a variant suffix after the shot count (e.g. __v2, __v2-shuf, __v2-med, __v2-high, __v2-think)
    m = re.match(r"llm__(.+?)__(.+?)__nshot-?(\d+)__([A-Za-z0-9.\-]+)$", base)
    if m:
        return {"arm": "llm", "model": m.group(1), "mode": m.group(2), "nshot": int(m.group(3)), "variant": m.group(4), "key": base}
    m = re.match(r"(text_bow_matched|text_embed_matched)__nshot(\d+)$", base)
    if m:
        return {"arm": m.group(1), "nshot": int(m.group(2)), "key": base}
    return {"arm": base, "key": base}


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    out = []
    for r in rows:
        prov = r.get("provenance") or []
        out.append({
            "seed": int(r["seed"]),
            "type": r["type"],
            "value": float(r["total_reward"]),
            "tau": prov[0].get("tau") if prov else (r.get("windows") or [{}])[0].get("tau"),
            "fallback": bool(r.get("fallback_count", 0)),
        })
    return out


def load(results: Path, overlay: Path | None = None) -> dict:
    """Read every cell under `results`; with `overlay`, an arm file in overlay/<cell>/ replaces the arm of the same name."""

    groups: dict = OrderedDict()
    for cell_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        m = CELL_RE.match(cell_dir.name)
        if not m:
            continue
        stem, M, K, s, noise, kind, encoding, wording = m.groups()
        g = groups.setdefault(f"{stem}-{kind}", {"stem": stem, "kind": kind, "M": int(M), "K": int(K), "sharpness": float(s),
                                                  "reward_noise": float(noise), "cells": OrderedDict()})
        arms = OrderedDict()
        files = sorted(list(cell_dir.glob("*.jsonl")) + list(cell_dir.glob("*.jsonl.gz")))
        if overlay is not None and (overlay / cell_dir.name).is_dir():
            by_name = {f.name: f for f in files}
            by_name.update({f.name: f for f in list((overlay / cell_dir.name).glob("*.jsonl")) + list((overlay / cell_dir.name).glob("*.jsonl.gz"))})
            files = [by_name[n] for n in sorted(by_name, key=str.lower)]  # one directory's order under Windows path sorting
        for src in files:
            meta = parse_arm(src.name)
            arms[meta["key"]] = {"meta": meta, "rows": read_rows(src)}
        g["cells"][f"{encoding}-{wording}" if wording else encoding] = {"dir": cell_dir.name, "encoding": encoding, "wording": wording, "arms": arms}
    return groups


def by_seed(rows: list[dict], field: str = "value") -> dict:
    d = {}
    for r in rows:
        d.setdefault(r["seed"], r[field])
    return d


def pct(a, q) -> float:
    return float(np.percentile(a, q))


def boot_p(samples: np.ndarray, threshold: float) -> float:
    return float((np.sum(samples <= threshold) + 1) / (len(samples) + 1))


class Reference:
    def __init__(self, arms: dict, held_out: list[str]):
        fixed = {k: v for k, v in arms.items() if v["meta"]["arm"] == "fixed"}
        seeds = sorted(set.intersection(*[set(by_seed(v["rows"])) for v in fixed.values()]))
        self.seeds = [s for s in seeds if s < 1000][:HEADLINE_N]
        self.taus = [v["meta"]["tau"] for v in fixed.values()]
        self.F = np.array([[by_seed(v["rows"])[s] for s in self.seeds] for v in fixed.values()])
        types = by_seed(next(iter(fixed.values()))["rows"], "type")
        self.types = [types[s] for s in self.seeds]
        self.means = self.F.mean(axis=1)
        self.best_i = int(np.argmax(self.means))
        self.V_star = float(self.means[self.best_i])
        self.Omax = self.F.max(axis=0)
        self.Omin = self.F.min(axis=0)
        self.Fstar = self.F[self.best_i]
        self.H_D = float(self.Omax.mean() - self.V_star)
        self.held_out = set(held_out)
        self.per_type = OrderedDict()
        for th in sorted(set(self.types)):
            idx = [i for i, t in enumerate(self.types) if t == th]
            m = self.F[:, idx].mean(axis=1)
            best = float(m.max())
            self.per_type[th] = {
                "n": len(idx), "share": len(idx) / len(self.seeds), "held_out": th in self.held_out,
                "best_tau": self.taus[int(np.argmax(m))], "best_value": best,
                "good_set": [self.taus[i] for i in range(len(self.taus)) if m[i] >= best - TIE_TOL],
                "values": {self.taus[i]: float(m[i]) for i in range(len(self.taus))},
            }

    def strata(self) -> dict:
        react = [parse_trait(t)[1] for t in self.types]
        held = [t in self.held_out for t in self.types]
        out = OrderedDict(all=list(range(len(self.seeds))))
        out["seen"] = [i for i, h in enumerate(held) if not h]
        out["held_out"] = [i for i, h in enumerate(held) if h]
        for r in ("habit", "beat_last", "copy_last"):
            out[r] = [i for i, x in enumerate(react) if x == r]
            out[f"seen:{r}"] = [i for i in out[r] if not held[i]]
            out[f"held_out:{r}"] = [i for i in out[r] if held[i]]
        out["reactive"] = [i for i, x in enumerate(react) if x != "habit"]
        out["seen:reactive"] = [i for i in out["reactive"] if not held[i]]
        out["held_out:reactive"] = [i for i in out["reactive"] if held[i]]
        return out


def pooled_capture(ref: Reference, A: np.ndarray, idx: np.ndarray) -> float:
    room = (ref.Omax[idx] - ref.Fstar[idx]).sum()
    return float((A[idx] - ref.Fstar[idx]).sum() / room) if room > 0 else float("nan")


def boot_capture(ref: Reference, A: np.ndarray, idx: list[int], B: int, rng) -> np.ndarray:
    idx = np.asarray(idx)
    gain = A[idx] - ref.Fstar[idx]
    room = ref.Omax[idx] - ref.Fstar[idx]
    boot = rng.integers(0, len(idx), size=(B, len(idx)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(room[boot].sum(axis=1) > 0, gain[boot].sum(axis=1) / room[boot].sum(axis=1), np.nan)


def contrast(x: np.ndarray, a_idx: list[int], b_idx: list[int], B: int, rng) -> dict:
    if not a_idx or not b_idx:
        return {"n": [len(a_idx), len(b_idx)]}
    a, b = x[np.asarray(a_idx)], x[np.asarray(b_idx)]
    boot = a[rng.integers(0, len(a), size=(B, len(a)))].mean(axis=1) - b[rng.integers(0, len(b), size=(B, len(b)))].mean(axis=1)
    return {"n": [len(a), len(b)], "diff": float(a.mean() - b.mean()), "ci": [pct(boot, 2.5), pct(boot, 97.5)], "lo95": pct(boot, 5), "hi95": pct(boot, 95)}


def capture_contrast(ref: Reference, A: np.ndarray, a_idx: list[int], b_idx: list[int], B: int, rng) -> dict:
    if not a_idx or not b_idx:
        return {"n": [len(a_idx), len(b_idx)]}
    diff = boot_capture(ref, A, a_idx, B, rng) - boot_capture(ref, A, b_idx, B, rng)
    diff = diff[np.isfinite(diff)]
    point = pooled_capture(ref, A, np.asarray(a_idx)) - pooled_capture(ref, A, np.asarray(b_idx))
    if not diff.size:
        return {"n": [len(a_idx), len(b_idx)], "diff": point}
    return {"n": [len(a_idx), len(b_idx)], "diff": point, "ci": [pct(diff, 2.5), pct(diff, 97.5)], "lo95": pct(diff, 5), "hi95": pct(diff, 95)}


def paired_subset(x: np.ndarray, y: np.ndarray, idx: list[int], B: int, rng) -> dict:
    if not idx:
        return {"n": 0}
    d = x[np.asarray(idx)] - y[np.asarray(idx)]
    boot = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return {"n": len(d), "diff": float(d.mean()), "ci": [pct(boot, 2.5), pct(boot, 97.5)], "lo95": pct(boot, 5), "hi95": pct(boot, 95), "p_le0": boot_p(boot, 0.0)}


def prefix_bayes_rows(ref: Reference, cfg: SandboxConfig, choice: dict) -> list[dict]:
    rows = []
    for i, seed in enumerate(ref.seeds):
        env = SandboxEnv(cfg)
        env.reset(seed)
        tau = choice[" ".join(env.opp_history)]
        rows.append({"seed": seed, "type": env.type_label, "value": float(ref.F[ref.taus.index(tau), i]), "tau": tau, "fallback": False})
    return rows


def stratum_stats(ref: Reference, A: np.ndarray, hit: np.ndarray, idx: list[int], B: int, rng) -> dict:
    if not idx:
        return {"n": 0}
    cap_b = boot_capture(ref, A, idx, B, rng)
    idx = np.asarray(idx)
    regret = ref.Omax[idx] - A[idx]
    span = ref.Omax[idx] - ref.Omin[idx]
    k = int(hit[idx].sum())
    finite = cap_b[np.isfinite(cap_b)]
    return {
        "n": int(len(idx)),
        "mean": float(A[idx].mean()),
        "capture": pooled_capture(ref, A, idx),
        "capture_ci": [pct(finite, 2.5), pct(finite, 97.5)] if finite.size else [float("nan")] * 2,
        "hit_rate": k / len(idx),
        "hit_ci": list(wilson(k, len(idx))),
        "regret": float(regret.mean()),
        "normalized_regret": float(regret.sum() / span.sum()) if span.sum() > 0 else float("nan"),
    }


def arm_stats(ref: Reference, rows: list[dict], B: int, rng) -> dict:
    bs, ch = by_seed(rows), by_seed(rows, "tau")
    mask = np.array([s in bs for s in ref.seeds])
    seeds = [s for s in ref.seeds if s in bs]
    if not seeds:
        return {"n": 0}
    A = np.array([bs[s] for s in seeds])
    F = ref.F[:, mask]
    n = len(A)
    idx = rng.integers(0, n, size=(B, n))
    Fb = F[:, idx]
    Vstar_b = Fb.mean(axis=2).max(axis=0)
    H_b = Fb.max(axis=0).mean(axis=1) - Vstar_b
    margin_b = A[idx].mean(axis=1) - Vstar_b
    with np.errstate(divide="ignore", invalid="ignore"):
        cap_b = np.where(H_b > 0, margin_b / H_b, np.nan)
    finite = cap_b[np.isfinite(cap_b)]
    V_star = float(F.mean(axis=1).max())
    H = float(F.max(axis=0).mean() - V_star)
    out = {
        "n": n,
        "mean": float(A.mean()),
        "margin": float(A.mean() - V_star),
        "margin_ci": [pct(margin_b, 2.5), pct(margin_b, 97.5)],
        "margin_lo95": pct(margin_b, 5),
        "p_margin_le_bound": boot_p(margin_b, PAYS_MARGIN),
        "p_margin_le0": boot_p(margin_b, 0.0),
        "capture": float(capture(A.mean(), V_star, H)),
        "capture_ci": [pct(finite, 2.5), pct(finite, 97.5)] if finite.size else [float("nan")] * 2,
        "capture_lo95": pct(finite, 5) if finite.size else float("nan"),
        "fallback_rate": float(np.mean([r["fallback"] for r in rows])),
    }
    if not mask.all():
        return out
    hit = np.array([ch.get(s) in ref.per_type[t]["good_set"] for s, t in zip(ref.seeds, ref.types)], dtype=float)
    strata = ref.strata()
    out["strata"] = OrderedDict((name, stratum_stats(ref, A, hit, sidx, B, rng)) for name, sidx in strata.items())
    out["contrasts"] = {
        "hit_reactive_minus_habit": contrast(hit, strata["reactive"], strata["habit"], B, rng),
        "hit_held_out_minus_seen": contrast(hit, strata["held_out"], strata["seen"], B, rng),
        "capture_held_out_minus_seen": capture_contrast(ref, A, strata["held_out"], strata["seen"], B, rng),
        "capture_held_out_reactive_minus_seen_reactive": capture_contrast(ref, A, strata["held_out:reactive"], strata["seen:reactive"], B, rng),
        "regret_held_out_minus_seen": contrast(ref.Omax - A, strata["held_out"], strata["seen"], B, rng),
    }
    out["_hit"] = hit
    out["_A"] = A
    total_regret = float((ref.Omax - A).sum())
    per_type = OrderedDict()
    for th, info in ref.per_type.items():
        sidx = [i for i, t in enumerate(ref.types) if t == th]
        mis = [i for i in sidx if ch.get(ref.seeds[i]) not in info["good_set"]]
        best_i = ref.taus.index(info["best_tau"])
        reg = float((ref.Omax[sidx] - A[sidx]).sum())
        per_type[th] = {
            "n": len(sidx), "held_out": info["held_out"], "mean": float(A[sidx].mean()),
            "p_mis": len(mis) / len(sidx),
            "cost": float(np.mean([ref.F[best_i][i] - A[i] for i in mis])) if mis else 0.0,
            "regret": reg / len(sidx),
            "loss_share": reg / total_regret if total_regret > 0 else float("nan"),
            "choices": dict(Counter(ch.get(ref.seeds[i]) for i in sidx)),
        }
    out["per_type"] = per_type
    out["expected_regret_pertype"] = float(sum(ref.per_type[t]["share"] * v["p_mis"] * v["cost"] for t, v in per_type.items()))
    return out


def paired_diff(a_rows, b_rows, B: int, rng) -> dict:
    a, b = by_seed(a_rows), by_seed(b_rows)
    seeds = [s for s in sorted(set(a) & set(b)) if s < 1000][:HEADLINE_N]
    d = np.array([a[s] - b[s] for s in seeds])
    db = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return {"n": len(d), "diff": float(d.mean()), "ci": [pct(db, 2.5), pct(db, 97.5)], "lo95": pct(db, 5), "hi95": pct(db, 95), "p_le0": boot_p(db, 0.0)}


def curve_paired(ref: Reference, rows: list[dict]) -> list[float]:
    seen, num, den, curve = set(), 0.0, 0.0, []
    pos = {s: i for i, s in enumerate(ref.seeds)}
    for r in rows:
        if r["seed"] in seen:
            continue
        seen.add(r["seed"])
        if r["seed"] not in pos:
            break
        i = pos[r["seed"]]
        num += r["value"] - ref.Fstar[i]
        den += ref.Omax[i] - ref.Fstar[i]
        curve.append(num / den if den > 0 else float("nan"))
    return curve


def verdict(margin_lo95: float, vs: dict | None, threshold: float) -> str:
    ok1 = margin_lo95 > threshold
    ok2 = vs is not None and vs["lo95"] > 0
    return "yes" if ok1 and ok2 else ("fixed only" if ok1 else "no")


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    out, alive = {}, True
    for rank, k in enumerate(keys):
        alive = alive and pvals[k] <= ALPHA / (m - rank)
        out[k] = alive
    return out


def analyse(groups: dict, B: int, stage0_groups: dict | None = None) -> dict:
    rng = np.random.default_rng(20261004)
    out = OrderedDict()
    for gname, g in groups.items():
        M, K, kind = g["M"], g["K"], g["kind"]
        held = resolve_held_out_set(kind, M)
        if "nonllm" not in g["cells"]:
            continue
        ref = Reference(g["cells"]["nonllm"]["arms"], held)
        cfg = SandboxConfig(M=M, K=K, family="traits", sharpness=g["sharpness"], reward_noise=g["reward_noise"],
                            held_out=tuple(held) or None, deploy_types=DEPLOY["types"]).normalized()
        exact = exact_reference(cfg)
        entry = OrderedDict(
            meta={"M": M, "K": K, "sharpness": g["sharpness"], "reward_noise": g["reward_noise"], "kind": kind,
                  "held_out_set": held_out_set_name(kind, M), "held_out": held},
            reference={"seeds_n": len(ref.seeds), "V_star": ref.V_star, "best_fixed": ref.taus[ref.best_i], "H_D": ref.H_D,
                       "fixed_means": {t: float(m) for t, m in zip(ref.taus, ref.means)}, "type_shares": dict(Counter(ref.types)),
                       "per_type": ref.per_type},
            exact={k: exact[k] for k in ("V_star", "best_tau", "H_D", "share", "per_decision_oracle", "prefix_ceiling_bayes",
                                         "prefix_ceiling_bayes_capture", "prefix_ceiling_class", "prefix_ceiling_class_capture",
                                         "prefix_ceiling_bayes_per_type_regret")},
            cells=OrderedDict(),
        )
        pb = arm_stats(ref, prefix_bayes_rows(ref, replace_cfg(cfg), exact["prefix_bayes_choice"]), B, rng)
        pb["meta"] = {"arm": "prefix_bayes_ceiling", "key": "prefix_bayes_ceiling"}
        entry["prefix_bayes_ceiling"] = strip(pb)
        all_arms = {}
        for cell, c in g["cells"].items():
            arms_out = OrderedDict()
            for key, a in c["arms"].items():
                st = arm_stats(ref, a["rows"], B, rng)
                st["meta"] = a["meta"]
                st["episodes"] = len(a["rows"])
                if a["meta"]["arm"] in CURVE_ARMS:
                    st["curve_paired"] = curve_paired(ref, a["rows"])
                    st["curve_population"] = capture_curve([r["value"] for r in a["rows"]], exact["V_star"], exact["H_D"]).tolist()
                    st["T_80"] = crossover(st["curve_paired"], 0.8)["T_star"]
                    st["T_50"] = crossover(st["curve_paired"], 0.5)["T_star"]
                    st["T_50_population"] = crossover(st["curve_population"], 0.5)["T_star"]
                    st["capture_at"] = {str(T): (st["curve_paired"][T - 1] if len(st["curve_paired"]) >= T else None) for T in (10, 30, 100)}
                    st["capture_at_population"] = {str(T): (st["curve_population"][T - 1] if len(st["curve_population"]) >= T else None) for T in (10, 30, 100, 300)}
                arms_out[key] = st
                all_arms[(cell, key)] = (c, a, st)
            entry["cells"][cell] = {"dir": c["dir"], "encoding": c["encoding"], "wording": c["wording"], "arms": arms_out}
        strata = ref.strata()
        wording_contrasts = OrderedDict()
        same, new = g["cells"].get("briefing-sameword", {}).get("arms", {}), g["cells"].get("briefing-newword", {}).get("arms", {})
        for key in sorted(set(same) & set(new)):
            a, b = all_arms[("briefing-newword", key)][2], all_arms[("briefing-sameword", key)][2]
            if "_hit" in a and "_hit" in b:
                wording_contrasts[key] = {
                    "hit_newword_minus_sameword_held_out": paired_subset(a["_hit"], b["_hit"], strata["held_out"], B, rng),
                    "value_newword_minus_sameword_held_out": paired_subset(a["_A"], b["_A"], strata["held_out"], B, rng),
                }
        entry["wording_contrasts"] = wording_contrasts
        briefing_value = OrderedDict()
        for (cell, key), (c, a, st) in all_arms.items():
            if a["meta"]["arm"] != "llm" or c["encoding"] != "briefing" or a["meta"]["nshot"] != 0:
                continue
            twin = g["cells"].get("semantic", {}).get("arms", {}).get(key)
            if twin:
                briefing_value[f"{cell}/{key}"] = paired_diff(a["rows"], twin["rows"], B, rng)
        entry["briefing_minus_semantic"] = briefing_value

        def candidates(wording, idx=None):
            cands = {}
            for (cell, key), (c, a, st) in all_arms.items():
                arm = a["meta"]["arm"]
                if st.get("n", 0) == 0:
                    continue
                ok = (arm in PREFIX_READERS and c["encoding"] == "nonllm") or (
                    wording and arm in TEXT_READERS and c["encoding"] == "briefing" and c["wording"] == wording)
                if ok:
                    cands[(cell, key)] = st["mean"] if idx is None else float(st["_A"][np.asarray(idx)].mean()) if "_A" in st and idx else None
            return {k: v for k, v in cands.items() if v is not None}

        strongest = {}
        for wording in (None, "sameword", "newword"):
            cands = candidates(wording)
            if cands:
                strongest[wording or "prefix_only"] = max(cands, key=cands.get)
        entry["strongest_nonllm"] = {k: "/".join(v) for k, v in strongest.items()}
        learners = {(cell, key): st for (cell, key), (c, a, st) in all_arms.items() if "curve_paired" in st}
        for (cell, key), (c, a, st) in all_arms.items():
            if a["meta"]["arm"] != "llm" or st.get("n", 0) == 0:
                continue
            which = c["wording"] if c["encoding"] in ("briefing", "briefing_prefix") else "prefix_only"
            target = strongest.get(which)
            if target:
                t_rows = g["cells"][target[0]]["arms"][target[1]]["rows"]
                st["vs_strongest"] = paired_diff(a["rows"], t_rows, B, rng)
                st["vs_strongest_arm"] = "/".join(target)
            st["pays"] = verdict(st["margin_lo95"], st.get("vs_strongest"), PAYS_MARGIN)
            st["pays_margin0"] = verdict(st["margin_lo95"], st.get("vs_strongest"), 0.0)
            p_vs = st["vs_strongest"]["p_le0"] if "vs_strongest" in st else 1.0
            st["p_pays"] = max(st["p_margin_le_bound"], p_vs)
            st["p_pays_margin0"] = max(st["p_margin_le0"], p_vs)
            st["crossover"] = {
                "/".join(lk): {"T_star": crossover(lv["curve_paired"], st["capture"])["T_star"], "T_max": len(lv["curve_paired"])}
                for lk, lv in learners.items()
            }
            if c["wording"] == "newword" and strata["held_out"] and "_A" in st:
                hc = candidates("newword", strata["held_out"])
                best = max(hc, key=hc.get)
                st["vs_strongest_held_out"] = paired_subset(st["_A"], all_arms[best][2]["_A"], strata["held_out"], B, rng)
                st["vs_strongest_held_out_arm"] = "/".join(best)
        if DEPLOY["types"] == "heldout":
            gap, sig = delta_min(cfg), sigma_eff(cfg, samples=20_000, seed_start=50_000_000)
            n_dep = len(deployment_types(cfg))
            entry["reference_0b"] = {"deployment_types": deployment_types(cfg), "Delta_min": gap, "sigma_eff": sig, "snr": snr(gap, sig),
                                     "predicted_T_star": predicted_T_star(K, n_dep, snr(gap, sig)),
                                     "predicted_T_star_known_type": predicted_T_star(K, n_dep, snr(gap, sig), method="known_type")}
            entry["predictions"] = predictions_0b(entry, ref, all_arms, stage0_groups.get(gname) if stage0_groups else None, held, B, rng)
        else:
            entry["predictions"] = predictions(entry, M)
        for cell in entry["cells"].values():
            for st in cell["arms"].values():
                strip(st)
        out[gname] = entry
    out_all = OrderedDict(groups=out)
    out_all["P_E"] = p_e(out, PRIMARY_0B if DEPLOY["types"] == "heldout" else PRIMARY)
    return out_all


def predictions(entry: dict, M: int) -> dict:
    cells = entry["cells"]
    nonllm = cells.get("nonllm", {}).get("arms", {})
    ceiling = entry["exact"]["prefix_ceiling_bayes_capture"]
    pb = OrderedDict()
    for key, st in nonllm.items():
        if st["meta"]["arm"] in PREFIX_READERS and st.get("n"):
            pb[key] = {"capture": st["capture"], "capture_ci": st["capture_ci"], "ceiling": ceiling, "at_or_below": st["capture_ci"][0] <= ceiling,
                       "loss_share_habit_RP": sum(v["loss_share"] for t, v in st.get("per_type", {}).items() if parse_trait(t)[1] == "habit" and parse_trait(t)[0] in ("ROCK", "PAPER"))}
    regret = entry["exact"]["prefix_ceiling_bayes_per_type_regret"]
    top2 = sorted(regret, key=regret.get, reverse=True)[:2]
    scripted = nonllm.get("scripted", {}).get("strata", {}).get("reactive", {})
    p_b = {
        "readers": pb,
        "all_at_or_below_ceiling": all(v["at_or_below"] for v in pb.values()),
        "bayes_top2_loss_types": top2,
        "bayes_top2_is_habit_R_P": set(top2) == {"ROCK-habit", "PAPER-habit"} if M == 9 else None,
        "prefix_bayes_per_type_regret_headline": {t: v["regret"] for t, v in entry["prefix_bayes_ceiling"].get("per_type", {}).items()},
        "scripted_reactive_capture": scripted.get("capture"),
        "scripted_reactive_capture_ci": scripted.get("capture_ci"),
        "scripted_reactive_negative": bool(scripted and scripted["capture_ci"][1] < 0),
    }
    p_c = OrderedDict()
    for wording in ("sameword", "newword"):
        for key, st in cells.get(f"briefing-{wording}", {}).get("arms", {}).items():
            if st["meta"]["arm"] in ("scripted_text", "text_bow", "text_embed") and "strata" in st:
                s = st["strata"]
                p_c[f"{wording}/{key}"] = {
                    "seen_capture": s["seen"].get("capture"), "held_out_capture": s["held_out"].get("capture"),
                    "held_out_capture_ci": s["held_out"].get("capture_ci"),
                    "held_out_minus_seen": st["contrasts"]["capture_held_out_minus_seen"],
                    "held_out_habit": s.get("held_out:habit"), "held_out_reactive": s.get("held_out:reactive"),
                    "within_noise": bool("ci" in st["contrasts"]["capture_held_out_minus_seen"] and st["contrasts"]["capture_held_out_minus_seen"]["ci"][0] <= 0 <= st["contrasts"]["capture_held_out_minus_seen"]["ci"][1]),
                    "held_out_below_0.5": bool(s["held_out"].get("n") and s["held_out"]["capture"] < 0.5),
                }
    p_d = OrderedDict()
    for cell in ("briefing-sameword", "briefing-newword"):
        for key, st in cells.get(cell, {}).get("arms", {}).items():
            if st["meta"]["arm"] == "llm" and st["meta"]["nshot"] == 0 and st.get("n"):
                bm = entry["briefing_minus_semantic"].get(f"{cell}/{key}")
                c = st.get("contrasts", {}).get("hit_held_out_minus_seen", {})
                ok = st["capture"] >= 0.5 and c.get("lo95", -1) >= -0.10 and bm is not None and bm["lo95"] > 0
                p_d[f"{cell}/{key}"] = {"capture": st["capture"], "hit_held_out_minus_seen_lo95": c.get("lo95"),
                                        "briefing_minus_semantic": bm, "holds": bool(ok)}
    p_f = OrderedDict()
    for cell, c in cells.items():
        for key, st in c["arms"].items():
            if st["meta"]["arm"] in FREE_BANDITS and "curve_paired" in st:
                p_f[f"{cell}/{key}"] = {"T_50": st["T_50"], "T_50_population": st["T_50_population"],
                                        "capture_at": st["capture_at"], "capture_at_population": st["capture_at_population"]}
    if M == 18:
        p_f_holds = all(v["T_50_population"] is None for v in p_f.values())
    else:
        p_f_holds = any(v["T_50"] is not None and v["T_50"] <= 100 for k, v in p_f.items() if k.split("/")[-1] in ("lints", "linucb"))
    return {"P_B": p_b, "P_C": p_c, "P_D": {"arms": p_d, "holds": any(v["holds"] for v in p_d.values())},
            "P_F": {"arms": p_f, "holds": bool(p_f_holds)}}


def summary_capture(st: dict | None) -> dict | None:
    if not st or not st.get("n"):
        return None
    return {"capture": st["capture"], "capture_ci": st["capture_ci"], "mean": st["mean"]}


def predictions_0b(entry: dict, ref: Reference, all_arms: dict, s0_group: dict | None, held: list[str], B: int, rng) -> dict:
    new = {k: st for (c, k), (_, _, st) in all_arms.items() if c == "briefing-newword"}
    llm120 = new.get("llm__gpt-oss-120b__single__nshot0")
    p_g = {arm: summary_capture(new.get(arm)) for arm in ("scripted_text", "text_bow", "text_embed")}
    p_g["gpt-oss-120b_single_n0"] = summary_capture(llm120)
    p_g["holds"] = all(p_g[a] is not None and p_g[a]["capture"] < 0.5 for a in ("scripted_text", "text_bow"))
    tstar = entry["reference_0b"]["predicted_T_star"]
    bandits = OrderedDict()
    for (c, k), (_, a, st) in all_arms.items():
        if a["meta"]["arm"] in FREE_BANDITS and "curve_paired" in st:
            bandits[f"{c}/{k}"] = {"T_50": st["T_50"], "T_50_population": st["T_50_population"], "capture": st["capture"],
                                   "capture_at": st["capture_at"], "curve_len": len(st["curve_paired"])}
    crossed = [k for k, v in bandits.items() if v["T_50"] is not None]
    expect_cross = tstar < 300
    p_h = {"predicted_T_star": tstar, "expect_cross": expect_cross, "bandits": bandits, "crossed": crossed,
           "holds": bool(crossed) if expect_cross else not crossed}
    p_i = None
    key = "llm__gpt-oss-120b__single__nshot0"
    if s0_group and llm120 and "_A" in llm120 and key in s0_group["cells"].get("briefing-newword", {}).get("arms", {}):
        ref0 = Reference(s0_group["cells"]["nonllm"]["arms"], held)
        rows0 = s0_group["cells"]["briefing-newword"]["arms"][key]["rows"]
        v0, ch0 = by_seed(rows0), by_seed(rows0, "tau")
        idx0 = [i for i in ref0.strata()["held_out"] if ref0.seeds[i] in v0]
        mb0, mb1 = ref0.taus.index("MIRROR_BEAT"), ref.taus.index("MIRROR_BEAT")
        a0 = np.array([v0[ref0.seeds[i]] for i in idx0])
        f0, o0 = ref0.F[mb0, idx0], ref0.Omax[idx0]
        h0 = np.array([ch0.get(ref0.seeds[i]) in ref0.per_type[ref0.types[i]]["good_set"] for i in idx0], dtype=float)
        a1, f1, o1, h1 = llm120["_A"], ref.F[mb1], ref.Omax, llm120["_hit"]
        cap = lambda a, f, o: float((a - f).sum() / (o - f).sum())
        b0 = rng.integers(0, len(a0), size=(B, len(a0)))
        b1 = rng.integers(0, len(a1), size=(B, len(a1)))
        with np.errstate(divide="ignore", invalid="ignore"):
            c0 = (a0[b0] - f0[b0]).sum(axis=1) / (o0[b0] - f0[b0]).sum(axis=1)
            c1 = (a1[b1] - f1[b1]).sum(axis=1) / (o1[b1] - f1[b1]).sum(axis=1)
        ok = np.isfinite(c0) & np.isfinite(c1)
        c0, c1 = c0[ok], c1[ok]
        d = c1 - c0
        hd = (h1[b1].mean(axis=1) - h0[b0].mean(axis=1))[ok]
        p_i = {"stage0_best_fixed": ref0.taus[ref0.best_i], "normaliser": "MIRROR_BEAT",
               "stage0_held_out_n": len(idx0), "stage0_held_out_capture": cap(a0, f0, o0), "stage0_held_out_capture_ci": [pct(c0, 2.5), pct(c0, 97.5)],
               "stage0b_capture_mb": cap(a1, f1, o1), "stage0b_capture_mb_ci": [pct(c1, 2.5), pct(c1, 97.5)], "stage0b_capture_native": llm120["capture"],
               "diff": cap(a1, f1, o1) - cap(a0, f0, o0), "diff_ci": [pct(d, 2.5), pct(d, 97.5)],
               "hit_stage0": float(h0.mean()), "hit_stage0b": float(h1.mean()), "hit_diff_ci": [pct(hd, 2.5), pct(hd, 97.5)],
               "within_noise": bool(pct(d, 2.5) <= 0 <= pct(d, 97.5))}
    return {"P_G": p_g, "P_H": p_h, "P_I": p_i}


def p_e(groups: dict, primary_spec: dict) -> dict:
    entries = {}
    for gname, e in groups.items():
        for cell, c in e["cells"].items():
            for key, st in c["arms"].items():
                if st["meta"]["arm"] == "llm" and "p_pays" in st:
                    entries[f"{gname}/{cell}/{key}"] = st
    primary = OrderedDict()
    for gname, e in groups.items():
        m = e["meta"]
        if (m["M"], m["reward_noise"], m["kind"]) != (primary_spec["M"], primary_spec["reward_noise"], primary_spec["kind"]):
            continue
        for key, st in e["cells"].get(primary_spec["cell"], {}).get("arms", {}).items():
            if (st["meta"]["arm"] == "llm" and st["meta"]["mode"] == primary_spec["mode"] and st["meta"]["nshot"] == primary_spec["nshot"]
                    and st["meta"]["model"] == primary_spec.get("model", st["meta"]["model"])
                    and st["meta"].get("variant") == primary_spec.get("variant")):
                primary[st["meta"]["model"]] = {k: st.get(k) for k in ("capture", "capture_ci", "margin", "margin_lo95", "vs_strongest", "vs_strongest_arm",
                                                                       "pays", "pays_margin0", "p_pays", "vs_strongest_held_out", "vs_strongest_held_out_arm",
                                                                       "fallback_rate")}
    h5 = holm({k: v["p_pays"] for k, v in entries.items()})
    h0 = holm({k: v["p_pays_margin0"] for k, v in entries.items()})
    newword = {k: v for k, v in entries.items() if "-newword/" in k}
    return {
        "primary_cell": primary_spec,
        "primary": primary,
        "primary_holds": any(v["pays"] == "yes" for v in primary.values()),
        "secondary_counts": {"llm_arm_entries": len(entries), "pays_yes": sum(v["pays"] == "yes" for v in entries.values()),
                             "pays_margin0_yes": sum(v["pays_margin0"] == "yes" for v in entries.values()),
                             "newword_pays_yes": sum(v["pays"] == "yes" for v in newword.values())},
        "holm_surviving": {"pays": [k for k, ok in h5.items() if ok], "pays_margin0": [k for k, ok in h0.items() if ok]},
        "verdicts": {k: {"pays": v["pays"], "pays_margin0": v["pays_margin0"], "p_pays": v["p_pays"], "capture": v["capture"]} for k, v in entries.items()},
    }


def strip(st: dict) -> dict:
    st.pop("_hit", None)
    st.pop("_A", None)
    return st


def replace_cfg(cfg: SandboxConfig) -> SandboxConfig:
    return SandboxConfig(M=cfg.M, K=cfg.K, family=cfg.family, sharpness=cfg.sharpness, reward_noise=cfg.reward_noise, encoding="semantic",
                         held_out=cfg.held_out, deploy_types=cfg.deploy_types)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="repository root: inputs default to data/briefing/..., outputs to analysis/briefing/...")
    parser.add_argument("--results", default=None, help="directory of cells (one sub-directory per cell, one file per arm)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--B", type=int, default=10_000)
    parser.add_argument("--headline-n", type=int, default=None, help="headline seeds per arm (default 100; 300 with --root --deploy-types heldout)")
    parser.add_argument("--deploy-types", default="all", choices=["all", "heldout"])
    parser.add_argument("--stage0-cells", default=None, help="cells of the all-types deployment, for the P-I comparison (deploy-types heldout)")
    parser.add_argument("--overlay", default=None, help="cells whose arm files replace the same-named arms of --results")
    parser.add_argument("--superseded", action="store_true", help="with --root: overlay data/briefing/superseded on data/briefing/all")
    args = parser.parse_args()
    heldout = args.deploy_types == "heldout"
    if args.root:
        data, ana = Path(args.root) / "data" / "briefing", Path(args.root) / "analysis" / "briefing"
        args.results = args.results or str(data / ("heldout" if heldout else "all"))
        if args.superseded:
            args.overlay = args.overlay or str(data / "superseded")
        args.out = args.out or str(ana / ("new_agents.json" if heldout else "bare_prompt_first_format.json" if args.superseded else "all_types.json"))
        if heldout:
            args.stage0_cells = args.stage0_cells or str(data / "all")
            args.headline_n = args.headline_n or 300
    if not args.results:
        parser.error("--results or --root is required")
    global HEADLINE_N
    HEADLINE_N = args.headline_n or 100
    DEPLOY["types"] = args.deploy_types
    res = analyse(load(Path(args.results), Path(args.overlay) if args.overlay else None), args.B,
                  load(Path(args.stage0_cells)) if args.stage0_cells else None)
    out = Path(args.out or "analysis/all_types.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    print("written", out)


if __name__ == "__main__":
    main()
