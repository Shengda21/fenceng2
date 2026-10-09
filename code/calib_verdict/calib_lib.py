"""Shared pieces of the calibration-split analysis: the verdict statistics and the deployment-record loaders.

Every script of code/calib_verdict takes --root <repository root> (default: the repository this file belongs to); it is read
here, when the module is imported. Records and task lists: data/calib_verdict; outputs: analysis/calib_verdict.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
REL = Path(_ap.parse_known_args()[0].root).resolve()  # the repository root
DATA = REL / "data" / "calib_verdict"
OUT = REL / "analysis" / "calib_verdict"
PAYS_MARGIN = 0.5
TIE_TOL = 0.5
B_BOOT = 10_000


def pct(a, q):
    return float(np.percentile(a, q))


def margin_stats(F: np.ndarray, A: np.ndarray, rng, B: int = B_BOOT) -> dict:
    """Paired margin over the best fixed strategy, V* re-selected in every replicate (analyze_main.arm_stats)."""
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
    mean = float(A.mean())
    return {"n": n, "mean": mean, "V_star": V_star, "H": H, "margin": mean - V_star,
            "margin_ci": [pct(margin_b, 2.5), pct(margin_b, 97.5)], "margin_lo95": pct(margin_b, 5),
            "capture": (mean - V_star) / H if H > 0 else float("nan"),
            "capture_ci": [pct(finite, 2.5), pct(finite, 97.5)] if finite.size else [float("nan")] * 2}


def paired_diff(a: np.ndarray, b: np.ndarray, rng, B: int = B_BOOT) -> dict:
    d = a - b
    db = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return {"n": len(d), "diff": float(d.mean()), "ci": [pct(db, 2.5), pct(db, 97.5)], "lo95": pct(db, 5), "hi95": pct(db, 95)}


def verdict(c2: bool, c3: bool) -> str:
    return "yes" if c2 and c3 else ("fixed only" if c2 else "no")


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def by_seed(rows):
    d = {}
    for r in rows:
        v = r.get("total_reward", r.get("value"))
        if v is None:
            continue
        d.setdefault(int(r["seed"]), float(v))
    return d


# ------------------------------------------------------------------ deployment records (headline seeds 0-99)

def _sandbox_paths(cell):
    ep = REL / "data" / "episodes" / "sandbox"
    return ep / f"{cell}-semantic", ep / f"{cell}-nonllm"


def deployment_cell(domain: str, cell: str, spec: dict) -> dict:
    """Per-seed arrays on the headline seeds: fixed matrix, the cell's comparator set, every LLM arm of the cell."""
    seeds = list(range(100))
    if domain == "sandbox":
        llm_dir, ref_dir = _sandbox_paths(cell)
        fixed_dir, comp_dir = ref_dir, ref_dir
        comps = spec["comparators"]
        llm_paths = {k: llm_dir / f"{k}.jsonl.gz" for k in spec["llm_arms"]}
    elif domain == "magent":
        ep = REL / "data" / "episodes"
        mcell = spec["master_cell"]
        dom = "battle" if mcell.startswith("battle") else "combined"
        comp_dir = ep / dom / mcell
        fixed_dir = comp_dir if dom == "battle" else ep / dom / (mcell.rsplit("-", 1)[0] + "-once")
        comps = spec["comparators"]
        llm_paths = {}
        for k in spec["llm_arms"]:
            p = comp_dir / f"{k}.jsonl.gz"
            if not p.exists():
                p = comp_dir / f"{k.replace('__nshot', '__nshot-')}.jsonl.gz"
            llm_paths[k] = p
    else:
        base = REL / "data" / "briefing" / "all"
        g = cell.replace("-sameword", "")
        fixed_dir = base / f"{g}-nonllm"
        comp_dirs = {c: fixed_dir for c in spec["comparators_prefix"]}
        comp_dirs.update({c: base / f"{g}-briefing-sameword" for c in spec["comparators_text"]})
        llm_paths = {}
        for k in spec["llm_arms"]:
            if k.endswith("__v2"):
                llm_paths[k] = REL / "data" / "briefing" / "task_prompt" / "all" / f"{g}-briefing-sameword" / f"{k}.jsonl.gz"
            else:
                llm_paths[k] = base / f"{g}-briefing-sameword" / f"{k}.jsonl.gz"
        comps = list(comp_dirs)
    fixed = {}
    for p in sorted(fixed_dir.glob("fixed__*.jsonl.gz")):
        bs = by_seed(read_rows(p))
        fixed[p.name[len("fixed__"):-len(".jsonl.gz")]] = np.array([bs[s] for s in seeds])
    comp_vals = {}
    for c in comps:
        p = (comp_dirs[c] if domain == "briefing" else comp_dir) / f"{c}.jsonl.gz"
        if p.exists():
            bs = by_seed(read_rows(p))
            comp_vals[c] = np.array([bs[s] for s in seeds])
    llm_vals = {}
    for k, p in llm_paths.items():
        bs = by_seed(read_rows(p))
        llm_vals[k] = np.array([bs[s] for s in seeds])
    taus = sorted(fixed)
    return {"taus": taus, "F": np.array([fixed[t] for t in taus]), "comps": comp_vals, "llm": llm_vals}
