"""Deployment evaluation of the FCR arms (the script refuses to run unless PREREG_FCR.md records the hash of calib/chosen.json).

For every briefing deployment (all-types and held-out-only, M 9/18, sigma 0/0.5, every held-out set, seen
and new wording) the FCR arms are fitted once per calibration setting with the hyperparameters of calib/chosen.json
and evaluated by exact replay on the recorded episodes: the return of a one-decision selector on a seed is the return of
the fixed strategy it chose on that seed. Comparators are read from the main briefing records; their verdicts and
intervals are read from analysis/briefing/*.json, so every comparator number in fcr.json is the paper's number.

Statistics follow code/briefing/v8_sandbox/analyze_briefing.py: headline seeds (first 100, or 300 with only new agents,
below 1000, shared by the fixed arms), V* re-selected inside every bootstrap replicate, H_D the mean per-seed maximum over
fixed arms minus V*, capture (V - V*)/H_D, paired bootstrap by seed with B = 10,000, one-sided 95% bounds as the 5th
percentile, p = (#{replicate <= 0} + 1)/(B + 1), TIE_TOL 0.5 for the hit rate, the pays criterion (margin lower bound > 0.5
and paired lower bound against the strongest non-LLM arm > 0), Holm at alpha 0.05. The bootstrap draws one index matrix per
deployment and seed set from numpy.random.default_rng([BOOT_SEED, deployment number, crc32 of the seed set]), shared by every
arm of that deployment (common resamples; no dependence on the order in which arms are processed).

usage: python code/briefing/fcr/evaluate.py [--root <repository root>] [--B 10000]  -> analysis/briefing/fcr.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zlib
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

import fcr_common as fc
from fcr_model import FCR, Composer

BOOT_SEED = 20_261_006
FCR_ARMS = OrderedDict([
    ("fcr_full", ("full", "tfidf", "main")),
    ("fcr_full_pairwise", ("full", "tfidf", "pairwise")),
    ("fcr_full_embed", ("full", "embed", "main")),
    ("fcr_n_bare", ("n_bare", "tfidf", "main")),
    ("fcr_n_bare_pairwise", ("n_bare", "tfidf", "pairwise")),
    ("fcr_n_bare_embed", ("n_bare", "embed", "main")),
    ("fcr_n_task", ("n_task", "tfidf", "main")),
    ("fcr_n_task_pairwise", ("n_task", "tfidf", "pairwise")),
    ("fcr_n_task_embed", ("n_task", "embed", "main")),
])
PRIMARY_FCR = ("fcr_full", "fcr_n_bare", "fcr_n_task")
ORACLE_ARMS = OrderedDict([("fcr_oracle", "main"), ("fcr_oracle_pairwise", "pairwise")])
FULL_VARIANTS = ("fcr_full", "fcr_full_pairwise", "fcr_full_embed")
N_VARIANTS = {"n_bare": ("fcr_n_bare", "fcr_n_bare_pairwise", "fcr_n_bare_embed"),
              "n_task": ("fcr_n_task", "fcr_n_task_pairwise", "fcr_n_task_embed")}
PRIMARY_DEPLOYMENTS = OrderedDict([
    ("all/sandbox-traits-M18-K10-sharp0.8-noise0-react/newword", "most agents known"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0-react/newword", "new agents, react4"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0-react8/newword", "new agents, react8"),
])
PRIMARY_LLM = OrderedDict([
    ("120b task 0/type", "task:llm__gpt-oss-120b__single__nshot0__v2"),
    ("Qwen bare 1/type", "briefing:llm__qwen3.8-27b__single__nshot1"),
    ("Qwen task 1/type", "task:llm__qwen3.8-27b__single__nshot1__v2"),
])
SAMPLE_MATCHED = OrderedDict([
    ("Qwen bare 1/type", ("briefing:llm__qwen3.8-27b__single__nshot1", "fcr_n_bare")),
    ("Qwen task 1/type", ("task:llm__qwen3.8-27b__single__nshot1__v2", "fcr_n_task")),
    ("120b task 1/type", ("task:llm__gpt-oss-120b__single__nshot1__v2", "fcr_n_task")),
])
OUTCOME_TEXT = {
    "A": "FCR or an existing cheap reader closes the gate, and the LLM does not beat FCR-N given the same labelled rows",
    "B": "the gate closes, but the LLM beats FCR-N given the same labelled rows: an advantage in sample efficiency only",
    "C": "the LLM still pays against every cheap reader including FCR, and the trait-oracle FCR would close the gap: the shortfall is in reading the words",
    "D": "the LLM still pays against every cheap reader including FCR, and beats the trait-oracle FCR too: the shortfall is in composing the counter",
}


# ----------------------------------------------------------------------------------------------- calibration guard


def freeze_record(prereg: Path, chosen: Path) -> dict:
    if not prereg.exists():
        raise SystemExit("PREREG_FCR.md is missing: the deployment evaluation runs only after the pre-registration is frozen")
    text = prereg.read_text(encoding="utf-8")
    m = re.search(r"^\|\s*(\d{4}-\d{2}-\d{2}T[0-9:.+\-]+)\s*\|\s*\*\*Frozen\*\*", text, flags=re.M)
    if not m:
        raise SystemExit("PREREG_FCR.md has no frozen row in its revision record: refusing to evaluate deployments")
    want = re.search(r"calib/chosen\.json.*?sha256 `([0-9a-f]{64})`", text, flags=re.S)
    have = hashlib.sha256(chosen.read_bytes()).hexdigest()
    if not want or want.group(1) != have:
        raise SystemExit("calib/chosen.json does not match the sha256 recorded in PREREG_FCR.md")
    return {"frozen_utc": m.group(1), "chosen_sha256": have}


# ----------------------------------------------------------------------------------------------- numbers of the main analysis


def released_index(root: Path) -> dict:
    """(deploy, group, cell label, arm key) -> the main analysis's stats for that arm; (deploy, group) -> strongest non-LLM arms."""

    ana = root / "analysis" / "briefing"
    idx, strongest = {}, {}
    for deploy, fn in (("all", "all_types.json"), ("heldout", "new_agents.json")):
        d = json.loads((ana / fn).read_text(encoding="utf-8"))
        for g, e in d["groups"].items():
            strongest[(deploy, g)] = e["strongest_nonllm"]
            for cell, c in e["cells"].items():
                for key, st in c["arms"].items():
                    idx[(deploy, g, cell, key)] = st
    tp = json.loads((ana / "task_prompt.json").read_text(encoding="utf-8"))
    for fam, deploy in (("s0", "all"), ("s0b", "heldout")):
        for g, e in tp[fam].items():
            for k, st in e["arms"].items():
                cell, key = k.split("/", 1)
                if st["meta"].get("arm") == "llm" and st["meta"].get("variant"):
                    idx[(deploy, g, f"task:{cell}", key)] = st
                elif st["meta"].get("arm") in ("text_bow_matched", "text_embed_matched"):
                    idx[(deploy, g, f"task:{cell}", key)] = st
    return {"arms": idx, "strongest": strongest}


# ----------------------------------------------------------------------------------------------- bootstrap


class Boot:
    def __init__(self, ref: fc.Reference, B: int, dep_no: int):
        self.ref, self.B, self.dep_no = ref, int(B), int(dep_no)
        self.cache = {}

    def get(self, mask: np.ndarray) -> dict:
        key = zlib.crc32(np.packbits(mask.astype(np.uint8)).tobytes())
        if key not in self.cache:
            n = int(mask.sum())
            rng = np.random.default_rng([BOOT_SEED, self.dep_no, key])
            idx = rng.integers(0, n, size=(self.B, n))
            F = self.ref.F[:, mask]
            vstar_b = np.max(np.stack([F[k][idx].mean(axis=1) for k in range(F.shape[0])]), axis=0)
            h_b = self.ref.Omax[mask][idx].mean(axis=1) - vstar_b
            V_star = float(F.mean(axis=1).max())
            H = float(F.max(axis=0).mean() - V_star)
            self.cache[key] = {"idx": idx, "vstar_b": vstar_b, "h_b": h_b, "V_star": V_star, "H": H, "n": n}
        return self.cache[key]


def vec(ref: fc.Reference, by: dict) -> tuple[np.ndarray, np.ndarray]:
    mask = np.array([s in by for s in ref.seeds])
    return mask, np.array([by[s] for s in ref.seeds if s in by], dtype=float)


def arm_stats(ref: fc.Reference, boot: Boot, values: dict) -> dict:
    mask, A = vec(ref, values)
    if not mask.any():
        return {"n": 0}
    b = boot.get(mask)
    margin_b = A[b["idx"]].mean(axis=1) - b["vstar_b"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cap_b = np.where(b["h_b"] > 0, margin_b / b["h_b"], np.nan)
    cap_b = cap_b[np.isfinite(cap_b)]
    return {
        "n": int(mask.sum()), "mean": float(A.mean()), "margin": float(A.mean() - b["V_star"]),
        "margin_ci": [fc.pct(margin_b, 2.5), fc.pct(margin_b, 97.5)], "margin_lo95": fc.pct(margin_b, 5),
        "p_margin_le_bound": fc.boot_p(margin_b, fc.PAYS_MARGIN),
        "capture": float((A.mean() - b["V_star"]) / b["H"]), "capture_ci": [fc.pct(cap_b, 2.5), fc.pct(cap_b, 97.5)],
        "capture_lo95": fc.pct(cap_b, 5),
    }


def paired(ref: fc.Reference, boot: Boot, x: dict, y: dict) -> dict:
    mask = np.array([s in x and s in y for s in ref.seeds])
    if not mask.any():
        return {"n": 0}
    seeds = [s for s in ref.seeds if s in x and s in y]
    d = np.array([x[s] - y[s] for s in seeds], dtype=float)
    b = boot.get(mask)
    db = d[b["idx"]].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cb = np.where(b["h_b"] > 0, db / b["h_b"], np.nan)
    cb = cb[np.isfinite(cb)]
    return {"n": len(d), "diff": float(d.mean()), "ci": [fc.pct(db, 2.5), fc.pct(db, 97.5)], "lo95": fc.pct(db, 5),
            "hi95": fc.pct(db, 95), "p_le0": fc.boot_p(db, 0.0), "capture_diff": float(d.mean() / b["H"]),
            "capture_diff_ci": [fc.pct(cb, 2.5), fc.pct(cb, 97.5)]}


def pooled_capture(ref: fc.Reference, A: np.ndarray, idx: list[int]) -> float | None:
    idx = np.asarray(idx, dtype=int)
    if not len(idx):
        return None
    room = (ref.Omax[idx] - ref.Fstar[idx]).sum()
    return float((A[idx] - ref.Fstar[idx]).sum() / room) if room > 0 else None


def verdict(margin_lo95: float, vs_lo95: float | None) -> str:
    ok1 = margin_lo95 > fc.PAYS_MARGIN
    ok2 = vs_lo95 is not None and vs_lo95 > 0
    return "yes" if ok1 and ok2 else ("fixed only" if ok1 else "no")


# ----------------------------------------------------------------------------------------------- deployments


def discover(root: Path) -> list[dict]:
    data = root / "data" / "briefing"
    out = []
    for deploy in ("all", "heldout"):
        groups = OrderedDict()
        for cell in sorted(p for p in (data / deploy).iterdir() if p.is_dir()):
            m = fc.CELL_RE.match(cell.name)
            if m:
                groups.setdefault(f"{m.group(1)}-{m.group(6)}", {"stem": m.group(1), "M": int(m.group(2)), "K": int(m.group(3)),
                                                                 "noise": float(m.group(5)), "kind": m.group(6), "cells": {}})
                label = m.group(7) + (f"-{m.group(8)}" if m.group(8) else "")
                groups[f"{m.group(1)}-{m.group(6)}"]["cells"][label] = cell
        for g, e in groups.items():
            tp = data / "task_prompt" / deploy
            for wording in ("sameword", "newword"):
                if f"briefing-{wording}" not in e["cells"]:
                    continue
                task_dir = tp / f"{g}-briefing-{wording}"
                out.append({"deploy": deploy, "group": g, "wording": wording, **{k: e[k] for k in ("stem", "M", "K", "noise", "kind")},
                            "nonllm": e["cells"]["nonllm"], "briefing": e["cells"][f"briefing-{wording}"],
                            "briefing_prefix": e["cells"].get(f"briefing_prefix-{wording}"),
                            "task": task_dir if task_dir.is_dir() else None})
    return out


def briefing_from_prompt(prompt: str) -> str:
    """The episode briefing in a bare-prompt (v1) record: the line right before the pool listing."""

    head = prompt.split("\nPool:", 1)[0]
    return head.split("\n")[-1].strip()


def fit_models(root: Path, chosen: dict, cache: dict) -> dict:
    models = OrderedDict()
    for name in fc.SETTINGS:
        s = fc.CalibSetting(root, name)
        ch = chosen[name]
        m = OrderedDict()
        for arm, (cond, ext, form) in FCR_ARMS.items():
            rows = s.rows(cond)
            m[arm] = FCR(s.M, ext, ch["classifier"][cond][ext]["C"], form, ch["composer"][form]["alpha"], s.pool, cache).fit(
                [r["text"] for r in rows], [r["type"] for r in rows], s.seen, s.values_seen)
        for arm, form in ORACLE_ARMS.items():
            m[arm] = Composer(s.M, form, ch["composer"][form]["alpha"], s.pool).fit(s.seen, s.values_seen)
        models[name] = {"setting": s, "arms": m}
    return models


def evaluate_deployment(root: Path, dep: dict, dep_no: int, models: dict, rel: dict, B: int) -> dict:
    deploy, g, wording, M = dep["deploy"], dep["group"], dep["wording"], dep["M"]
    key = f"{deploy}/{g}/{wording}"
    sname = fc.setting_name(M, dep["kind"])
    s: fc.CalibSetting = models[sname]["setting"]
    held = s.held_out
    ref = fc.Reference(dep["nonllm"], held, fc.HEADLINE_N[deploy])
    boot = Boot(ref, B, dep_no)
    traits = fc.TRAITS[M]
    texts = [fc.briefing_for(t, sd, held, wording)[0] for sd, t in zip(ref.seeds, ref.types)]
    held_idx = [i for i, t in enumerate(ref.types) if t in set(held)]
    seen_idx = [i for i, t in enumerate(ref.types) if t not in set(held)]
    tau_star = [s.best_response[t] for t in ref.types]
    good = [ref.per_type[t]["good_set"] for t in ref.types]

    # briefing check: every bare zero-shot record of this cell carries the briefing rebuilt here
    chk = dep["briefing"] / "llm__gpt-oss-120b__single__nshot0.jsonl.gz"
    rebuilt = dict(zip(ref.seeds, texts))
    n_chk = n_bad = 0
    for r in fc.read_raw(chk):
        sd = int(r["seed"])
        if sd in rebuilt:
            n_chk += 1
            n_bad += int(briefing_from_prompt(r["provenance"][0]["extra"]["prompt"]) != rebuilt[sd])

    arms = OrderedDict()
    choice = OrderedDict()
    values = OrderedDict()
    zhat = OrderedDict()
    for arm, model in models[sname]["arms"].items():
        if arm in ORACLE_ARMS:
            picks = model.choose([fc.traits_of(t, M) for t in ref.types])
            zs = [fc.traits_of(t, M) for t in ref.types]
        else:
            zs, picks = model.read(texts)
        choice[arm] = dict(zip(ref.seeds, picks))
        values[arm] = dict(zip(ref.seeds, ref.replay(picks).tolist()))
        zhat[arm] = zs

    # comparators from the main briefing records
    comp_files = OrderedDict()
    for f in fc.arm_files(dep["nonllm"]):
        meta = fc.parse_arm(f.name)
        if meta["arm"] != "fixed":
            comp_files[f"nonllm:{meta['key']}"] = (f, meta, "nonllm")
    for f in fc.arm_files(dep["briefing"]):
        meta = fc.parse_arm(f.name)
        comp_files[f"briefing:{meta['key']}"] = (f, meta, f"briefing-{wording}")
    if dep["briefing_prefix"] is not None:
        for f in fc.arm_files(dep["briefing_prefix"]):
            meta = fc.parse_arm(f.name)
            if meta["arm"] == "llm":
                comp_files[f"briefing_prefix:{meta['key']}"] = (f, meta, f"briefing_prefix-{wording}")
    if dep["task"] is not None:
        for f in fc.arm_files(dep["task"]):
            meta = fc.parse_arm(f.name)
            comp_files[f"task:{meta['key']}"] = (f, meta, f"task:briefing-{wording}")
    for ck, (f, meta, cell_label) in comp_files.items():
        rows = fc.read_rows(f)
        values[ck] = fc.by_seed(rows)
        choice[ck] = fc.by_seed(rows, "tau")

    # per-arm statistics
    for arm in list(models[sname]["arms"]) + list(comp_files):
        st = arm_stats(ref, boot, values[arm])
        if st.get("n", 0) == 0:
            continue
        ch = choice[arm]
        present = [i for i, sd in enumerate(ref.seeds) if sd in ch]
        st["counter_accuracy"] = float(np.mean([ch[ref.seeds[i]] == tau_star[i] for i in present]))
        st["hit_rate"] = float(np.mean([ch[ref.seeds[i]] in good[i] for i in present]))
        if arm in models[sname]["arms"]:
            zs = zhat[arm]
            true = [fc.traits_of(t, M) for t in ref.types]
            acc = {t: float(np.mean([z[j] == tz[j] for z, tz in zip(zs, true)])) for j, t in enumerate(traits)}
            acc["tuple"] = float(np.mean([tuple(z) == tz for z, tz in zip(zs, true)]))
            st["trait_accuracy"] = acc
            A = np.array([values[arm][sd] for sd in ref.seeds])
            strata = OrderedDict()
            for sname_, idx in (("seen", seen_idx), ("held_out", held_idx)):
                if not idx:
                    continue
                strata[sname_] = {"n": len(idx), "capture": pooled_capture(ref, A, idx),
                                  "counter_accuracy": float(np.mean([choice[arm][ref.seeds[i]] == tau_star[i] for i in idx])),
                                  **{t: float(np.mean([zs[i][j] == true[i][j] for i in idx])) for j, t in enumerate(traits)},
                                  "tuple": float(np.mean([tuple(zs[i]) == true[i] for i in idx]))}
            st["strata"] = strata
            st["choices_held_out"] = {t: dict(Counter(choice[arm][ref.seeds[i]] for i in held_idx if ref.types[i] == t))
                                      for t in held if any(ref.types[i] == t for i in held_idx)}
        elif len(present) == len(ref.seeds):
            A = np.array([values[arm][sd] for sd in ref.seeds])
            st["strata"] = OrderedDict((nm, {"n": len(idx), "capture": pooled_capture(ref, A, idx),
                                             "counter_accuracy": float(np.mean([ch[ref.seeds[i]] == tau_star[i] for i in idx]))})
                                       for nm, idx in (("seen", seen_idx), ("held_out", held_idx)) if idx)
        meta = comp_files[arm][1] if arm in comp_files else {"arm": "fcr", "variant": arm}
        st["meta"] = meta
        if arm in comp_files:
            st["release"] = released_stats(rel, deploy, g, comp_files[arm][2], meta["key"])
        arms[arm] = st

    # parsing vs composition (trait-oracle decomposition) for every FCR arm that reads text
    decomp = OrderedDict()
    true = [fc.traits_of(t, M) for t in ref.types]
    for arm, (cond, ext, form) in FCR_ARMS.items():
        orc = "fcr_oracle" if form == "main" else "fcr_oracle_pairwise"
        cats = Counter()
        loss = Counter()
        for i, sd in enumerate(ref.seeds):
            tr = tuple(zhat[arm][i]) == true[i]
            ok = choice[arm][sd] == tau_star[i]
            c = ("traits right, counter right" if ok else "traits right, counter wrong (composition)") if tr else \
                ("traits wrong, counter right" if ok else "traits wrong, counter wrong (parsing)")
            cats[c] += 1
            loss[c] += float(ref.Omax[i] - values[arm][sd])
        tot = sum(loss.values())
        decomp[arm] = {"oracle_arm": orc, "capture": arms[arm]["capture"], "oracle_capture": arms[orc]["capture"],
                       "parsing_loss": arms[orc]["capture"] - arms[arm]["capture"], "composition_loss": 1.0 - arms[orc]["capture"],
                       "episode_shares": {k: v / len(ref.seeds) for k, v in sorted(cats.items())},
                       "regret_shares": {k: (v / tot if tot > 0 else None) for k, v in sorted(loss.items())}}

    # strongest non-LLM arms: the candidates and the choice of the main analysis, then with the FCR arms added
    rel_strongest = rel["strongest"].get((deploy, g), {}).get(wording)
    rel_key = None
    if rel_strongest:
        cell, k = rel_strongest.split("/", 1)
        rel_key = f"nonllm:{k}" if cell == "nonllm" else f"briefing:{k}"
    cands = [a for a in arms if (a.startswith("nonllm:") and arms[a]["meta"]["arm"] in fc.PREFIX_READERS)
             or (a.startswith("briefing:") and arms[a]["meta"]["arm"] in fc.TEXT_READERS)]
    cands = [a for a in cands if arms[a]["n"] == len(ref.seeds)]
    own_best = max(cands, key=lambda a: arms[a]["mean"]) if cands else None
    with_fcr = cands + list(FCR_ARMS)
    strongest_fcr = max(with_fcr, key=lambda a: arms[a]["mean"])
    with_primary = cands + list(PRIMARY_FCR)
    strongest_primary = max(with_primary, key=lambda a: arms[a]["mean"])
    bandits = [a for a in arms if arms[a]["meta"]["arm"] in fc.FREE_BANDITS and arms[a]["n"] == len(ref.seeds)]
    best_bandit = max(bandits, key=lambda a: arms[a]["mean"]) if bandits else None
    full_star = max(FULL_VARIANTS, key=lambda a: arms[a]["mean"])
    oracle_star = "fcr_oracle" if FCR_ARMS[full_star][2] == "main" else "fcr_oracle_pairwise"

    # paired comparisons: FCR arms against the non-LLM comparators
    pairs = OrderedDict()
    tfidf = "briefing:text_bow"
    for arm in list(FCR_ARMS) + list(ORACLE_ARMS):
        for other in [tfidf, best_bandit, rel_key, "briefing:text_embed", "briefing:scripted_text"] + \
                     ([f"task:text_bow_matched__nshot1"] if arm.startswith("fcr_n_task") else []):
            if other and other in values and other != arm:
                pairs[f"{arm} - {other}"] = paired(ref, boot, values[arm], values[other])
        if arm != "fcr_full":
            pairs[f"{arm} - fcr_full"] = paired(ref, boot, values[arm], values["fcr_full"])

    # every LLM arm: its paired difference against the FCR arms, and the pays verdict with the FCR arms added
    llm = OrderedDict()
    for ak, st in arms.items():
        if st["meta"].get("arm") != "llm":
            continue
        r = st.get("release") or {}
        m_lo = r.get("margin_lo95", st["margin_lo95"])
        p_m = r.get("p_margin_le_bound")
        p_m_src = "release"
        if p_m is None:
            p_m, p_m_src = st["p_margin_le_bound"], "recomputed"
        entry = OrderedDict(n=st["n"], capture=r.get("capture", st["capture"]), capture_ci=r.get("capture_ci", st["capture_ci"]),
                            margin_lo95=m_lo, p_margin_le_bound=p_m, p_margin_source=p_m_src,
                            counter_accuracy=st["counter_accuracy"], hit_rate=st["hit_rate"],
                            release={"pays": r.get("pays"), "vs_strongest_arm": r.get("vs_strongest_arm"),
                                     "vs_strongest": r.get("vs_strongest"), "p_pays": r.get("p_pays")})
        entry["vs"] = OrderedDict()
        for x in list(FCR_ARMS) + list(ORACLE_ARMS):
            entry["vs"][x] = paired(ref, boot, values[ak], values[x])
        for label, target in (("primary", strongest_fcr), ("primary_fcr_only", strongest_primary)):
            if target in FCR_ARMS:
                vs = entry["vs"][target]
                vs_lo, p_vs, src = vs["lo95"], vs["p_le0"], "fcr"
            else:
                rv = r.get("vs_strongest") or {}
                if rv and r.get("vs_strongest_arm", "").split("/")[-1] == target.split(":", 1)[1]:
                    vs_lo, p_vs, src = rv["lo95"], rv["p_le0"], "release"
                else:
                    vs = paired(ref, boot, values[ak], values[target])
                    vs_lo, p_vs, src = vs["lo95"], vs["p_le0"], "recomputed"
            entry[f"updated_{label}"] = {"strongest": target, "vs_lo95": vs_lo, "p_vs": p_vs, "source": src,
                                         "pays": verdict(m_lo, vs_lo), "p_pays": max(p_m, p_vs)}
        llm[ak] = entry

    out = OrderedDict(
        meta={"deploy": deploy, "group": g, "wording": wording, "M": M, "K": dep["K"], "reward_noise": dep["noise"],
              "kind": dep["kind"], "held_out_set": s.set_name, "held_out": held, "calibration_setting": sname,
              "population": "most agents known" if deploy == "all" else "new agents",
              "primary": key in PRIMARY_DEPLOYMENTS, "label": PRIMARY_DEPLOYMENTS.get(key),
              "briefing_check": {"records": n_chk, "mismatches": n_bad},
              "type_shares": dict(Counter(ref.types)), "n_seen_episodes": len(seen_idx), "n_held_out_episodes": len(held_idx)},
        reference={"seeds_n": len(ref.seeds), "V_star": ref.V_star, "H_D": ref.H_D, "best_fixed": ref.taus[ref.best_i]},
        strongest={"release": rel_key, "release_recomputed": own_best, "with_fcr": strongest_fcr, "with_primary_fcr": strongest_primary,
                   "online_bandit": best_bandit, "fcr_full_star": full_star, "oracle_star": oracle_star},
        arms=arms, decomposition=decomp, paired=pairs, llm=llm)
    return out


def released_stats(rel: dict, deploy: str, g: str, cell_label: str, key: str) -> dict | None:
    st = rel["arms"].get((deploy, g, cell_label, key))
    if st is None:
        return None
    keep = ("n", "capture", "capture_ci", "margin", "margin_lo95", "p_margin_le_bound", "pays", "p_pays", "vs_strongest",
            "vs_strongest_arm", "fallback_rate")
    return {k: st.get(k) for k in keep if k in st}


def outcome(dep: dict, arm_key: str, matched: str | None) -> dict:
    e = dep["llm"].get(arm_key)
    if e is None:
        return {"available": False}
    up = e["updated_primary"]
    closed = up["pays"] != "yes"
    res = {"available": True, "release_pays": e["release"]["pays"], "updated_pays": up["pays"], "strongest": up["strongest"],
           "vs_strongest_lo95": up["vs_lo95"]}
    if closed:
        res["closed_by"] = "release arm" if e["release"]["pays"] != "yes" else ("FCR" if up["strongest"] in FCR_ARMS else "release arm")
        if matched is None:
            res["letter"] = "A/B (zero-shot arm: no sample-matched FCR)"
        else:
            nstar = max(N_VARIANTS[matched], key=lambda a: dep["arms"][a]["mean"])
            vs = e["vs"][nstar]
            res.update({"fcr_n_star": nstar, "llm_minus_fcr_n_star": vs, "letter": "B" if vs["lo95"] > 0 else "A"})
    else:
        orc = dep["strongest"]["oracle_star"]
        vs = e["vs"][orc]
        res.update({"oracle_star": orc, "llm_minus_oracle_star": vs, "letter": "D" if vs["lo95"] > 0 else "C"})
    res["meaning"] = OUTCOME_TEXT.get(res["letter"]) if res.get("letter") in OUTCOME_TEXT else None
    return res


def selftest(root: Path) -> dict:
    """Machinery check on the main analysis's arms only (no FCR arm is fitted or replayed, no briefing text is built): the
    reference, the capture point estimates and the strongest non-LLM arm recomputed here must equal those of the main analysis."""

    rel = released_index(root)
    n_arms = n_bad = n_strong = n_strong_bad = 0
    bad = []
    for dep in discover(root):
        held = fc.resolve_held_out_set(dep["kind"], dep["M"])
        ref = fc.Reference(dep["nonllm"], held, fc.HEADLINE_N[dep["deploy"]])
        means = {}
        for cell_label, d in (("nonllm", dep["nonllm"]), (f"briefing-{dep['wording']}", dep["briefing"]),
                              (f"task:briefing-{dep['wording']}", dep["task"])):
            if d is None:
                continue
            for f in fc.arm_files(d):
                meta = fc.parse_arm(f.name)
                if meta["arm"] == "fixed":
                    continue
                vals = fc.by_seed(fc.read_rows(f))
                mask = np.array([s in vals for s in ref.seeds])
                A = np.array([vals[s] for s in ref.seeds if s in vals])
                F = ref.F[:, mask]
                cap = (A.mean() - F.mean(axis=1).max()) / (F.max(axis=0).mean() - F.mean(axis=1).max())
                r = released_stats(rel, dep["deploy"], dep["group"], cell_label, meta["key"])
                if r is not None and r.get("capture") is not None:
                    n_arms += 1
                    if abs(r["capture"] - cap) > 1e-9:
                        n_bad += 1
                        bad.append(f"{dep['deploy']}/{dep['group']}/{cell_label}/{meta['key']}")
                ok = (cell_label == "nonllm" and meta["arm"] in fc.PREFIX_READERS) or (cell_label.startswith("briefing") and meta["arm"] in fc.TEXT_READERS)
                if ok and mask.all():
                    means[f"{cell_label}/{meta['key']}"] = A.mean()
        want = rel["strongest"].get((dep["deploy"], dep["group"]), {}).get(dep["wording"])
        if want:
            n_strong += 1
            n_strong_bad += int(max(means, key=means.get) != want)
    return {"arms_compared": n_arms, "capture_mismatches": n_bad, "mismatched": bad[:20],
            "strongest_compared": n_strong, "strongest_mismatches": n_strong_bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(fc.RELEASE_DEFAULT))
    ap.add_argument("--B", type=int, default=fc.B_DEFAULT)
    ap.add_argument("--out", default=None, help="default: <root>/analysis/briefing/fcr.json")
    ap.add_argument("--selftest", action="store_true", help="check the machinery on the main analysis's arms only (needs no PREREG_FCR.md)")
    a = ap.parse_args()
    root = Path(a.root)
    if a.selftest:
        res = selftest(root)
        fc.write_json(root / fc.ANALYSIS / "fcr_selftest.json", res)
        print(json.dumps(res, indent=1))
        return
    frozen = freeze_record(root / fc.PREREG, fc.FCR_ROOT / "calib" / "chosen.json")
    chosen = json.loads((fc.FCR_ROOT / "calib" / "chosen.json").read_text(encoding="utf-8"))
    cache = fc.load_embed_cache(root)
    models = fit_models(root, chosen, cache)
    rel = released_index(root)
    deps = discover(root)
    res = OrderedDict()
    for no, dep in enumerate(deps):
        k = f"{dep['deploy']}/{dep['group']}/{dep['wording']}"
        res[k] = evaluate_deployment(root, dep, no, models, rel, a.B)
        print(k, {x: round(res[k]["arms"][x]["capture"], 3) for x in ("fcr_full", "fcr_full_pairwise", "fcr_oracle", "fcr_oracle_pairwise")}, flush=True)

    # primary tests (uncorrected, as in the main analysis)
    primary = OrderedDict()
    outcomes = OrderedDict()
    for k, label in PRIMARY_DEPLOYMENTS.items():
        d = res[k]
        t = OrderedDict()
        for name, ak in PRIMARY_LLM.items():
            e = d["llm"].get(ak)
            t[f"T1 pays with FCR: {name}"] = None if e is None else {
                **e["updated_primary"], "release_pays": e["release"]["pays"], "holds": e["updated_primary"]["pays"] == "yes", "p": e["updated_primary"]["p_pays"]}
        p = d["paired"].get("fcr_full - briefing:text_bow")
        t["T2 FCR-full - TF-IDF whole-type"] = p and {**p, "holds": p["lo95"] > 0, "p": p["p_le0"]}
        for name, (ak, fcrn) in SAMPLE_MATCHED.items():
            e = d["llm"].get(ak)
            v = e and e["vs"][fcrn]
            t[f"T3 {name} - {fcrn}"] = v and {**v, "holds": v["lo95"] > 0, "p": v["p_le0"]}
        primary[label] = t
    hp = fc.holm({f"{lab}|{tn}": tv["p"] for lab, tt in primary.items() for tn, tv in tt.items() if tv})
    for lab, tt in primary.items():
        for tn, tv in tt.items():
            if tv:
                tv["holm_within_primary"] = hp[f"{lab}|{tn}"]
    for k, label in PRIMARY_DEPLOYMENTS.items():
        d = res[k]
        star_name = max((n for n, ak in PRIMARY_LLM.items() if ak in d["llm"]), key=lambda n: d["llm"][PRIMARY_LLM[n]]["capture"])
        per_arm = OrderedDict()
        for name, ak in PRIMARY_LLM.items():
            matched = "n_bare" if ak == "briefing:llm__qwen3.8-27b__single__nshot1" else ("n_task" if ak.endswith("nshot1__v2") else None)
            per_arm[name] = outcome(d, ak, matched)
        outcomes[label] = {"L_star": star_name, "letter": per_arm[star_name].get("letter"), "per_arm": per_arm}

    # secondary families (Holm, alpha 0.05)
    s1, s2 = OrderedDict(), OrderedDict()
    primary_pairs = {(k, ak) for k in PRIMARY_DEPLOYMENTS for ak in PRIMARY_LLM.values()}
    for k, d in res.items():
        for ak, e in d["llm"].items():
            if (k, ak) not in primary_pairs:
                s1[f"{k}/{ak}"] = e["updated_primary"]["p_pays"]
        if k not in PRIMARY_DEPLOYMENTS and "fcr_full - briefing:text_bow" in d["paired"]:
            s2[k] = d["paired"]["fcr_full - briefing:text_bow"]["p_le0"]
    h1, h2 = fc.holm(s1), fc.holm(s2)

    def entry_of(test_key: str) -> dict:
        parts = test_key.split("/")
        return res["/".join(parts[:3])]["llm"]["/".join(parts[3:])]

    rel_holm = fc.holm({k: entry_of(k)["release"]["p_pays"] for k in s1 if entry_of(k)["release"]["p_pays"] is not None})
    secondary = OrderedDict(
        S1_pays_with_fcr={"tests": len(s1),
                          "release_pays_yes": sum(entry_of(k)["release"]["pays"] == "yes" for k in s1),
                          "updated_pays_yes_uncorrected": sum(entry_of(k)["updated_primary"]["pays"] == "yes" for k in s1),
                          "holm_surviving": [k for k, ok in h1.items() if ok],
                          "holm_surviving_release_p_same_family": [k for k, ok in rel_holm.items() if ok],
                          "lost_by_fcr_uncorrected": [k for k in s1 if entry_of(k)["release"]["pays"] == "yes"
                                                      and entry_of(k)["updated_primary"]["pays"] != "yes"]},
        S2_fcr_full_minus_tfidf={"tests": len(s2), "holm_surviving": [k for k, ok in h2.items() if ok],
                                 "uncorrected_lo95_gt0": [k for k in s2 if res[k]["paired"]["fcr_full - briefing:text_bow"]["lo95"] > 0]},
    )
    out = OrderedDict(
        meta={"freeze": frozen, "B": a.B, "boot_seed": BOOT_SEED, "tie_tol": fc.TIE_TOL, "pays_margin": fc.PAYS_MARGIN, "alpha": fc.ALPHA,
              "headline_n": fc.HEADLINE_N, "release_root": "strategy-selection-audit (v9 release)", "fcr_arms": {k: list(v) for k, v in FCR_ARMS.items()},
              "primary_fcr_arms": list(PRIMARY_FCR), "oracle_arms": dict(ORACLE_ARMS), "primary_llm_arms": dict(PRIMARY_LLM),
              "vendor_sources": json.loads((fc.HERE / "SOURCES.json").read_text(encoding="utf-8"))},
        calibration={"chosen": chosen, "loco_cv_summary": loco_summary()},
        primary_tests=primary, outcomes=outcomes, secondary=secondary, deployments=res)
    dest = Path(a.out) if a.out else root / fc.ANALYSIS / "fcr.json"
    fc.write_json(dest, out)
    print("written", dest)


def loco_summary() -> dict:
    cv = json.loads((fc.FCR_ROOT / "calib" / "loco_cv.json").read_text(encoding="utf-8"))
    out = OrderedDict()
    for name in fc.SETTINGS:
        e = cv[name]
        out[name] = {
            "composer": {f: {"alpha": e["composer"][f]["chosen_alpha"],
                             "loco_counter_accuracy": e["composer"][f]["grid"][str(e["composer"][f]["chosen_alpha"])]["counter_accuracy"],
                             "train_counter_accuracy_seen": e["composer"][f]["train_counter_accuracy_seen"]} for f in ("main", "pairwise")},
            "conditions": {c: {"n_rows": e["conditions"][c]["n_rows"],
                               "whole_type_tfidf_loco_counter_accuracy": e["conditions"][c]["whole_type_tfidf"]["counter_accuracy"],
                               **{ext: {"C": e["conditions"][c]["extractors"][ext]["classifier"]["chosen_C"],
                                        "end_to_end": {f: e["conditions"][c]["extractors"][ext]["end_to_end"][f]["mean"] for f in ("main", "pairwise")}}
                                  for ext in ("tfidf", "embed")}}
                           for c in ("full", "n_bare", "n_task")},
        }
    return out


if __name__ == "__main__":
    main()
