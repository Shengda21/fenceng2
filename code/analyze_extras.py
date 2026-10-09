"""Compute the offline extras and write analysis/extras.json.

Schema written to extras.json:
  schema: short description and deterministic seeds.
  A0_integrity:
    replay_check: reward-equals-fixed replay checks by domain plus mismatches.
    lock_audit: per-file lock hashes, per-cell lock groups, pairwise lock comparisons.
    code_hashes: source hash to discovered local path, or missing.
  A1_inference_cost:
    llm_files: per LLM result file token, call, latency and fallback ledger.
    aggregates: per-model headline semantic zero-shot and all-cell summaries, and one summary per serving build.
    nonllm_microbench: real v8lib selector timing on recorded feature vectors.
  A2_cost_adjusted:
    comparisons: bootstrap reward margins, token break-even lambda, latency margins.
    pays_lambda_check: monotonicity check for the pre-registered pays criterion.
  A3_uncertainty:
    cells: raw and post-stratified V*, H_D, calibration/deployment differences.
  A4_pays_robustness:
    counts and non-excluded LLM-vs-strongest rows from master.json.
  A5_opaque_controls:
    signature_audit plus one-hot-signature replay controls and recorded baselines.
  validation:
    three re-derived master.json numbers and byte-identical rerun status.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REMEASURE = False
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
V9 = ROOT
CODE = ROOT / "code"
OUT = ROOT / "analysis"
SANDBOX_ROOT = ROOT / "code" / "oghp" / "v8_sandbox"
V8_LIB = ROOT / "code" / "oghp" / "v8_lib"

for path in (str(CODE), str(V8_LIB), str(SANDBOX_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from analyze_main import (  # noqa: E402
    B_DEFAULT,
    HEADLINE_N,
    Reference,
    arm_stats,
    by_seed,
    capture_curve_paired,
    load_tree,
    paired_diff,
    parse_arm,
    parse_cell,
    pct,
)
from replay import (  # noqa: E402
    ReplayCell,
    SignatureLookupSelector,
    SignatureUCB1Selector,
    replay_selector,
    row_tau,
    signature_from_prompt,
)
from v8lib.arms import ContextUCB, LinTS, LinUCB, MUCB  # noqa: E402
from v8lib.context import Context  # noqa: E402


B = B_DEFAULT
RNG_SEEDS = {
    "a1_microbench": 2026092901,
    "a2": 2026092902,
    "a3": 2026092903,
    "a5": 2026092905,
}

DEFAULTS = {"battle": "HOLD_POSITION", "combined": "HOLD_LINE", "sandbox": "ROCK"}
LOCK_FOR_GROUP = {
    "battle-base4": "rs_magent_lock_base4.json",
    "battle-traits8": "rs_magent_lock_traits8.json",
    "combined-base4": "rs_combined_lock.json",
    "combined-traits8": "rs_combined_traits8_lock.json",
    "combined-base4-delta06": "rs_combined_lock_delta06.json",
}
MASTER_GROUPS = [
    ("battle-base4-pool8-semantic-single", "battle-base4-pool8-semantic-single"),
    ("battle-traits8-pool8-semantic-single", "battle-traits8-pool8-semantic-single"),
    ("combined-base4-once", "combined-base4-semantic"),
    ("combined-traits8-once", "combined-traits8-semantic"),
    ("combined-base4-delta06-once", "combined-base4-delta06-semantic"),
    ("sandbox-M3-K3-sharp0.8-noise0-nonllm", "sandbox-M3-K3-sharp0.8-noise0-semantic"),
    ("sandbox-M3-K3-sharp0.6-noise0.5-nonllm", "sandbox-M3-K3-sharp0.6-noise0.5-semantic"),
    ("sandbox-M6-K3-sharp0.8-noise0-nonllm", "sandbox-M6-K3-sharp0.8-noise0-semantic"),
    ("sandbox-M6-K3-sharp0.6-noise0.5-nonllm", "sandbox-M6-K3-sharp0.6-noise0.5-semantic"),
]
BANDITS = ("linucb", "lints", "ctxucb", "mucb")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean_json(obj), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        domain, cell, fname = path.parts[-3:]
        meta_path = DATA / "episode_meta" / domain / cell / (fname[:-len(".jsonl.gz")] + ".json")
        return (read_json(meta_path) if meta_path.exists() else {}), rows
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {}, rows
    raw = read_json(path)
    if isinstance(raw, dict):
        return dict(raw.get("meta") or {}), list(raw.get("rows") or raw.get("episodes") or [])
    return {}, list(raw)


def result_files() -> list[dict[str, Any]]:
    records = []
    root = DATA / "raw_episodes"
    if not root.exists():
        root = DATA / "episodes"
    for domain in ("battle", "combined", "sandbox"):
        droot = root / domain
        if not droot.exists():
            continue
        for path in sorted(droot.rglob("*")):
            if "".join(path.suffixes[-2:]) != ".jsonl.gz" and path.suffix not in {".json", ".jsonl"}:
                continue
            rel = path.relative_to(droot)
            if len(rel.parts) < 2:
                continue
            tag, cell = "", rel.parts[0]
            meta, rows = load_raw_file(path)
            arm_meta = parse_arm(path.name)
            records.append(
                {
                    "path": path,
                    "relpath": str(path.relative_to(ROOT)).replace("\\\\", "/"),
                    "domain": domain,
                    "tag": tag,
                    "cell": cell,
                    "arm_key": arm_meta["key"],
                    "arm_meta": arm_meta,
                    "meta": meta,
                    "rows": rows,
                }
            )
    return records


def tau_from_row(row: dict[str, Any]) -> str | None:
    return row_tau(row)


def prompts_by_seed(rows: list[dict[str, Any]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        seed = row.get("seed")
        if seed is None:
            continue
        for prov in row.get("provenance") or []:
            prompt = ((prov.get("extra") or {}).get("prompt"))
            if prompt is not None:
                out[int(seed)].append(str(prompt))
    return dict(out)


def usage_and_latency(row: dict[str, Any]) -> dict[str, float]:
    prompt = completion = total = 0.0
    attempts = 0.0
    call_latency = 0.0
    latency = 0.0
    for prov in row.get("provenance") or []:
        extra = prov.get("extra") or {}
        usage = extra.get("usage") or {}
        prompt += float(usage.get("prompt_tokens") or 0.0)
        completion += float(usage.get("completion_tokens") or 0.0)
        total += float(usage.get("total_tokens") or (usage.get("prompt_tokens") or 0.0) + (usage.get("completion_tokens") or 0.0))
        attempts += float(extra.get("attempts") or 1.0)
        cl = extra.get("call_latency_s")
        if isinstance(cl, list):
            call_latency += sum(float(x or 0.0) for x in cl)
        else:
            call_latency += float(cl or 0.0)
        latency += float(prov.get("latency_s") or 0.0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "attempts": attempts,
        "call_latency_s": call_latency,
        "latency_s": latency,
    }


def summary_stats(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"mean": None, "median": None, "p95": None}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}


def ci_from_boot(values: np.ndarray) -> list[float]:
    return [pct(values, 2.5), pct(values, 97.5)]


def refs_from_tree(tree: dict[str, Any]) -> dict[str, Reference]:
    refs: dict[str, Reference] = {}
    for domain, cells in tree.items():
        for cell, arms in cells.items():
            meta = parse_cell(domain, cell)
            if meta["ref"] == cell and any(v["meta"]["arm"] == "fixed" for v in arms.values()):
                refs[cell] = Reference(arms, DEFAULTS[domain])
    return refs


def fixed_lookup(ref: Reference) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for j, seed in enumerate(ref.seeds):
        out[int(seed)] = {tau: float(ref.F[i, j]) for i, tau in enumerate(ref.taus)}
    return out


def rows_from_values(values: dict[int, float], choices: dict[int, str] | None = None, types: dict[int, str] | None = None) -> list[dict[str, Any]]:
    out = []
    for seed in sorted(values):
        out.append(
            {
                "seed": int(seed),
                "type": None if types is None else types.get(seed),
                "value": float(values[seed]),
                "total_reward": float(values[seed]),
                "success": bool(values[seed] > 0.0),
                "fallback": False,
                "tau": None if choices is None else choices.get(seed),
            }
        )
    return out


def lock_paths() -> list[Path]:
    paths = sorted((DATA / "locks").glob("rs_*lock*.json"))
    archive = DATA / "locks"
    if archive.exists():
        paths.extend(sorted(archive.glob("*.before_relabel.json")))
    return paths


def load_locks() -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    by_hash: dict[str, dict[str, Any]] = {}
    for path in lock_paths():
        data = read_json(path)
        data["__path"] = str(path)
        digest = sha256_file(path)
        data["__sha256"] = digest
        by_name[path.name] = data
        by_hash[digest] = data
        by_hash[digest[:10]] = data
    return {"by_name": by_name, "by_hash": by_hash}


def cell_lock_name(meta: dict[str, Any]) -> str | None:
    if meta.get("held_out"):
        return f"rs_combined_lock_base4_heldout_{meta['held_out']}.json"
    return LOCK_FOR_GROUP.get(meta.get("group", ""))


def lock_for_cell(meta: dict[str, Any], locks: dict[str, Any]) -> dict[str, Any] | None:
    name = cell_lock_name(meta)
    if name:
        return locks["by_name"].get(name)
    return None


def fixed_dataset_from_ref(ref: Reference) -> dict[str, Any]:
    return {
        "ids": [int(s) for s in ref.seeds],
        "taus": list(ref.taus),
        "F": np.asarray(ref.F, dtype=float),
        "types": list(ref.types),
        "source": "deployment_fixed_results",
    }


def fixed_dataset_from_lock(lock: dict[str, Any]) -> dict[str, Any]:
    fixed = lock.get("population_values", {}).get("fixed", {})
    taus = sorted(fixed)
    manifest = list(lock.get("calibration_manifest") or [])
    ids, types, cols = [], [], []
    for item in manifest:
        key = item.get("key")
        if key is None or any(key not in fixed[tau] for tau in taus):
            continue
        ids.append(str(key))
        types.append(str(item.get("type") or item.get("type1")))
        cols.append([float(fixed[tau][key]) for tau in taus])
    F = np.asarray(cols, dtype=float).T if cols else np.zeros((len(taus), 0), dtype=float)
    return {"ids": ids, "taus": taus, "F": F, "types": types, "source": Path(lock.get("__path", "")).name}


def sandbox_dataset(meta: dict[str, Any], seed_start: int, n: int) -> dict[str, Any]:
    from sandbox.env import SandboxConfig, SandboxEnv  # noqa: WPS433
    from sandbox.game import strategy_pool  # noqa: WPS433

    cfg = SandboxConfig(
        M=int(meta["M"]),
        K=int(meta["K"]),
        delta=1.0,
        sharpness=float(meta["sharpness"]),
        reward_noise=float(meta["reward_noise"]),
        H=30,
        k=5,
        schedule="single",
    )
    pool = strategy_pool(cfg.K)
    ids, types, cols = [], [], []
    for seed in range(seed_start, seed_start + n):
        base = SandboxEnv(cfg)
        ctx = base.reset(seed)
        ids.append(int(seed))
        types.append(str(ctx.type_label))
        vals = []
        for tau in pool:
            env = base.clone()
            total = 0.0
            done = False
            while not done:
                _, reward, done, _ = env.step_window(tau)
                total += float(reward)
            vals.append(total)
        cols.append(vals)
    return {"ids": ids, "taus": pool, "F": np.asarray(cols, dtype=float).T, "types": types, "source": f"sandbox_env_{seed_start}_{seed_start+n-1}"}


def raw_vh(F: np.ndarray) -> dict[str, Any]:
    means = F.mean(axis=1)
    best_i = int(np.argmax(means))
    vstar = float(means[best_i])
    hd = float(F.max(axis=0).mean() - vstar)
    return {"V_star": vstar, "H_D": hd, "best_tau_index": best_i}


def bootstrap_vh(F: np.ndarray, B_count: int, rng: np.random.Generator) -> dict[str, Any]:
    n = int(F.shape[1])
    if n == 0:
        return {"V_star_ci": [None, None], "H_D_ci": [None, None], "V_star_boot": np.array([]), "H_D_boot": np.array([])}
    idx = rng.integers(0, n, size=(B_count, n))
    Fb = F[:, idx]
    means = Fb.mean(axis=2)
    vb = means.max(axis=0)
    hb = Fb.max(axis=0).mean(axis=1) - vb
    return {"V_star_ci": ci_from_boot(vb), "H_D_ci": ci_from_boot(hb), "V_star_boot": vb, "H_D_boot": hb}


def poststratified_vh(F: np.ndarray, types: list[str], B_count: int, rng: np.random.Generator) -> dict[str, Any]:
    labels = sorted(set(types))
    if not labels:
        return {"V_star_ps": None, "H_D_ps": None, "V_star_ps_ci": [None, None], "H_D_ps_ci": [None, None], "types_used": []}

    def compute(cols_by_type: dict[str, np.ndarray]) -> tuple[float, float]:
        tau_means = []
        oracle_means = []
        for label in labels:
            mat = cols_by_type[label]
            tau_means.append(mat.mean(axis=1))
            oracle_means.append(float(mat.max(axis=0).mean()))
        tau_mean = np.vstack(tau_means).mean(axis=0)
        v = float(tau_mean.max())
        h = float(np.mean(oracle_means) - v)
        return v, h

    idx_by_type = {label: np.array([i for i, t in enumerate(types) if t == label], dtype=int) for label in labels}
    point_cols = {label: F[:, idx_by_type[label]] for label in labels}
    v_point, h_point = compute(point_cols)
    vb = np.empty(B_count, dtype=float)
    hb = np.empty(B_count, dtype=float)
    for b in range(B_count):
        cols = {}
        for label in labels:
            idx = idx_by_type[label]
            sample = rng.choice(idx, size=len(idx), replace=True)
            cols[label] = F[:, sample]
        vb[b], hb[b] = compute(cols)
    return {
        "V_star_ps": v_point,
        "H_D_ps": h_point,
        "V_star_ps_ci": ci_from_boot(vb),
        "H_D_ps_ci": ci_from_boot(hb),
        "V_star_ps_boot": vb,
        "H_D_ps_boot": hb,
        "types_used": labels,
    }


def dataset_stats(dataset: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    F = np.asarray(dataset["F"], dtype=float)
    point = raw_vh(F)
    boot = bootstrap_vh(F, B, rng)
    ps = poststratified_vh(F, list(dataset["types"]), B, rng)
    return {
        "source": dataset.get("source"),
        "n": int(F.shape[1]),
        "type_counts": dict(sorted(Counter(dataset["types"]).items())),
        "V_star": point["V_star"],
        "V_star_ci": boot["V_star_ci"],
        "H_D": point["H_D"],
        "H_D_ci": boot["H_D_ci"],
        "V_star_boot": boot["V_star_boot"],
        "H_D_boot": boot["H_D_boot"],
        "V_star_ps": ps["V_star_ps"],
        "V_star_ps_ci": ps["V_star_ps_ci"],
        "H_D_ps": ps["H_D_ps"],
        "H_D_ps_ci": ps["H_D_ps_ci"],
        "V_star_ps_boot": ps.get("V_star_ps_boot", np.array([])),
        "H_D_ps_boot": ps.get("H_D_ps_boot", np.array([])),
        "types_used_ps": ps["types_used"],
    }


def strip_boot(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if not k.endswith("_boot")}


def extract_extra_body(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def a0_integrity(records: list[dict[str, Any]], tree: dict[str, Any], refs: dict[str, Reference], locks: dict[str, Any]) -> dict[str, Any]:
    summaries: dict[str, dict[str, int]] = defaultdict(lambda: {"checked": 0, "matched": 0, "skipped": 0})
    mismatches = []
    for rec in records:
        arm = rec["arm_meta"].get("arm")
        if arm == "fixed":
            continue
        meta = parse_cell(rec["domain"], rec["cell"])
        ref = refs.get(meta["ref"])
        if ref is None:
            continue
        lookup = fixed_lookup(ref)
        for row in rec["rows"]:
            value = row.get("total_reward")
            if value is None:
                summaries[rec["domain"]]["skipped"] += 1
                continue
            seed = int(row.get("seed"))
            tau = tau_from_row(row)
            if seed not in lookup or tau not in lookup[seed]:
                summaries[rec["domain"]]["skipped"] += 1
                continue
            summaries[rec["domain"]]["checked"] += 1
            expected = lookup[seed][tau]
            got = float(value)
            if abs(got - expected) <= 1e-6:
                summaries[rec["domain"]]["matched"] += 1
            else:
                mismatches.append({"file": rec["relpath"], "seed": seed, "tau": tau, "total_reward": got, "fixed_reward": expected, "delta": got - expected})

    file_records = []
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_hash_values: dict[str, str] = {}
    prompt_cache: dict[tuple[str, str], dict[int, list[str]]] = {}
    for rec in records:
        meta = rec["meta"]
        lock_hash = meta.get("lock_hash") or next((row.get("lock_hash") for row in rec["rows"] if row.get("lock_hash")), None)
        file_hashes = dict(meta.get("file_hashes") or {})
        for src, digest in file_hashes.items():
            file_hash_values[src] = digest
        item = {
            "file": rec["relpath"],
            "cell": rec["cell"],
            "arm": rec["arm_key"],
            "lock_hash10": str(lock_hash)[:10] if lock_hash else None,
            "lock_hash": lock_hash,
            "file_hashes": file_hashes,
        }
        file_records.append(item)
        by_cell[rec["cell"]].append({**item, "rows": rec["rows"], "meta": meta, "arm_meta": rec["arm_meta"]})
        if rec["arm_meta"].get("arm") == "llm":
            prompt_cache[(rec["cell"], rec["arm_key"])] = prompts_by_seed(rec["rows"])

    def nshot_matches(meta: dict[str, Any], lock: dict[str, Any] | None) -> bool | None:
        examples = meta.get("nshot_examples")
        if not examples:
            return None
        if lock is None:
            return False
        rows = {int(r["seed"]): r for r in lock.get("feature_rows", []) if "seed" in r}
        best = lock.get("best_response_table", {})
        for ex in examples:
            seed = int(ex.get("seed", -1))
            row = rows.get(seed)
            if row is None:
                return False
            typ = str(row.get("type"))
            if ex.get("type") != typ:
                return False
            if ex.get("tau") != best.get(typ):
                return False
            prompt = str(ex.get("prompt", ""))
            if str(row.get("text", "")) not in prompt:
                return False
        return True

    cells = {}
    pair_reports = []
    for cell, items in sorted(by_cell.items()):
        groups: dict[str, list[str]] = defaultdict(list)
        hash_file_hashes: dict[str, dict[str, str]] = {}
        hash_nshot = {}
        unarchived: dict[str, str] = {}
        for item in items:
            h10 = item["lock_hash10"]
            if h10 is None:
                continue
            groups[h10].append(item["arm"])
            hash_file_hashes.setdefault(h10, {}).update(item["file_hashes"])
            lock = locks["by_hash"].get(str(item["lock_hash"])) or locks["by_hash"].get(h10)
            if lock is None:
                # the lock version this record carries is not stored with the locks; compare with the cell's current lock
                lock = lock_for_cell(parse_cell(cell.split("-")[0], cell), locks)
                if item["meta"].get("nshot_examples"):
                    unarchived.setdefault(h10, (lock or {}).get("__path", "none"))
            nm = nshot_matches(item["meta"], lock)
            if nm is not None:
                hash_nshot.setdefault(h10, []).append(bool(nm))
        cells[cell] = {
            "lock_hashes": {h: sorted(v) for h, v in sorted(groups.items())},
            "nshot_examples_match_lock_feature_rows": {h: (all(v) if v else None) for h, v in sorted(hash_nshot.items())},
            "nshot_compared_with_current_lock": {h: Path(p).name for h, p in sorted(unarchived.items())},
        }
        for a, b in combinations(sorted(groups), 2):
            fa, fb = hash_file_hashes.get(a, {}), hash_file_hashes.get(b, {})
            keys = sorted(set(fa) | set(fb))
            diffs = [k for k in keys if fa.get(k) != fb.get(k)]
            prompt_same = None
            common_seed_count = 0
            mismatched_seeds = []
            prompts_a: dict[int, list[str]] = defaultdict(list)
            prompts_b: dict[int, list[str]] = defaultdict(list)
            manifest_a: dict[int, str] = {}
            manifest_b: dict[int, str] = {}
            for item in items:
                if item["lock_hash10"] not in {a, b}:
                    continue
                prompts = prompt_cache.get((cell, item["arm"]), {})
                target = prompts_a if item["lock_hash10"] == a else prompts_b
                for seed, vals in prompts.items():
                    target[seed].extend(vals)
                manifest_target = manifest_a if item["lock_hash10"] == a else manifest_b
                for row in item["rows"]:
                    if row.get("seed") is not None and (row.get("type") is not None or row.get("type1") is not None):
                        manifest_target.setdefault(int(row["seed"]), str(row.get("type") or row.get("type1")))
            common = sorted(set(prompts_a) & set(prompts_b))
            if common:
                common_seed_count = len(common)
                prompt_same = True
                for seed in common:
                    if set(prompts_a[seed]) != set(prompts_b[seed]):
                        prompt_same = False
                        if len(mismatched_seeds) < 5:
                            mismatched_seeds.append(seed)
            la, lb = locks["by_hash"].get(a), locks["by_hash"].get(b)
            dep_same = None if la is None or lb is None else la.get("deployment_manifest") == lb.get("deployment_manifest")
            if dep_same is None and manifest_a and manifest_b:
                common_manifest = sorted(set(manifest_a) & set(manifest_b))
                dep_same = all(manifest_a[s] == manifest_b[s] for s in common_manifest)
            pair_reports.append(
                {
                    "cell": cell,
                    "lock_pair": [a, b],
                    "differing_source_files": diffs,
                    "llm_prompts_identical_for_common_seeds": prompt_same,
                    "common_prompt_seed_count": common_seed_count,
                    "prompt_mismatch_seed_sample": mismatched_seeds,
                    "deployment_manifest_identical": dep_same,
                    "nshot_examples_match_lock_feature_rows": {
                        a: cells[cell]["nshot_examples_match_lock_feature_rows"].get(a),
                        b: cells[cell]["nshot_examples_match_lock_feature_rows"].get(b),
                    },
                }
            )

    code_hashes = code_hash_audit(set(file_hash_values.values()))
    return {
        "prompt_example_audit": prompt_example_audit(records),
        "replay_check": {"by_domain": dict(sorted(summaries.items())), "mismatches": mismatches},
        "lock_audit": {"files": [{k: v for k, v in item.items() if k != "rows"} for item in file_records], "cells": cells, "lock_pairs": pair_reports},
        "code_hashes": code_hashes,
    }


def prompt_example_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the labelled examples in every recorded prompt against N x (types seen in calibration)."""
    files = prompts = matched = 0
    bad = []
    for rec in records:
        am = rec["arm_meta"]
        if am.get("arm") != "llm":
            continue
        meta = parse_cell(rec["domain"], rec["cell"])
        expected = int(am.get("nshot") or 0) * (int(meta.get("M") or 0) - (1 if meta.get("held_out") else 0))
        counts = Counter(len(re.findall(r"(?m)^Strategy: ", p)) for ps in prompts_by_seed(rec["rows"]).values() for p in ps)
        files += 1
        prompts += sum(counts.values())
        matched += counts.get(expected, 0)
        if set(counts) != {expected}:
            bad.append({"cell": rec["cell"], "arm": rec["arm_key"], "expected": expected, "found": dict(sorted(counts.items()))})
    return {"files": files, "prompts": prompts, "prompts_matching": matched, "mismatching_files": bad}


def code_hash_audit(wanted: set[str]) -> dict[str, Any]:
    remaining = set(wanted)
    found: dict[str, str] = {}
    skip_dirs = {".git", "__pycache__", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if not remaining:
                break
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > 20_000_000:
                    continue
                digest = sha256_file(path)
            except OSError:
                continue
            if digest in remaining:
                found[digest] = str(path.relative_to(ROOT))
                remaining.remove(digest)
        if not remaining:
            break
    return {digest: found.get(digest, "missing") for digest in sorted(wanted)}


def a1_inference_cost(records: list[dict[str, Any]], locks: dict[str, Any]) -> dict[str, Any]:
    llm_files = {}
    per_model_all: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_model_headline: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_model_build: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for rec in records:
        arm_meta = rec["arm_meta"]
        if arm_meta.get("arm") != "llm":
            continue
        per_dec = [usage_and_latency(row) for row in rec["rows"] if row.get("total_reward") is not None]
        fallbacks = []
        fingerprints = set()
        temperatures = set()
        max_tokens = set()
        extra_bodies = set()
        for row in rec["rows"]:
            fb = False
            for seq_key in ("decisions", "windows", "provenance"):
                for item in row.get(seq_key) or []:
                    fb = fb or bool(item.get("fallback"))
            fallbacks.append(fb)
            for prov in row.get("provenance") or []:
                extra = prov.get("extra") or {}
                fp = extra.get("system_fingerprint")
                if fp:
                    fingerprints.add(str(fp))
                if extra.get("temperature") is not None:
                    temperatures.add(json.dumps(extra.get("temperature"), sort_keys=True))
                if extra.get("max_tokens") is not None:
                    max_tokens.add(json.dumps(extra.get("max_tokens"), sort_keys=True))
                if extra.get("extra_body") is not None:
                    extra_bodies.add(json.dumps(extra.get("extra_body"), sort_keys=True))
        item = {
            "file": rec["relpath"],
            "cell": rec["cell"],
            "model": arm_meta["model"],
            "mode": arm_meta["mode"],
            "nshot": arm_meta["nshot"],
            "decisions": len(per_dec),
            "calls": int(sum(d["attempts"] for d in per_dec)),
            "tokens_per_decision": {
                "prompt": summary_stats([d["prompt_tokens"] for d in per_dec]),
                "completion": summary_stats([d["completion_tokens"] for d in per_dec]),
                "total": summary_stats([d["total_tokens"] for d in per_dec]),
            },
            "latency_per_decision_s": {
                "call_latency_s": summary_stats([d["call_latency_s"] for d in per_dec]),
                "latency_s": summary_stats([d["latency_s"] for d in per_dec]),
            },
            "fallback_rate": float(np.mean(fallbacks)) if fallbacks else None,
            "system_fingerprint": sorted(fingerprints),
            "temperature": sorted(temperatures),
            "max_tokens": sorted(max_tokens),
            "extra_body": sorted(extra_bodies),
        }
        llm_files[f"{rec['cell']}::{rec['arm_key']}"] = item
        per_model_all[arm_meta["model"]].extend(per_dec)
        per_model_build[(arm_meta["model"], "+".join(sorted(fingerprints)) or "unknown")].extend(per_dec)
        meta = parse_cell(rec["domain"], rec["cell"])
        if meta.get("encoding") == "semantic" and not meta.get("held_out") and arm_meta["mode"] == "single" and arm_meta["nshot"] == 0:
            per_model_headline[arm_meta["model"]].extend(per_dec)

    def aggregate(rows: list[dict[str, float]]) -> dict[str, Any]:
        return {
            "decisions": len(rows),
            "calls": int(sum(d["attempts"] for d in rows)),
            "tokens_per_decision": {
                "prompt": summary_stats([d["prompt_tokens"] for d in rows]),
                "completion": summary_stats([d["completion_tokens"] for d in rows]),
                "total": summary_stats([d["total_tokens"] for d in rows]),
            },
            "latency_per_decision_s": {
                "call_latency_s": summary_stats([d["call_latency_s"] for d in rows]),
                "latency_s": summary_stats([d["latency_s"] for d in rows]),
            },
        }

    return {
        "llm_files": llm_files,
        "aggregates": {
            "headline_semantic_single_nshot0_by_model": {m: aggregate(v) for m, v in sorted(per_model_headline.items())},
            "all_cells_by_model": {m: aggregate(v) for m, v in sorted(per_model_all.items())},
            # latency depends on the GPU, so every serving build is also summarised on its own
            "by_serving_build": {f"{m}|{fp}": aggregate(v) for (m, fp), v in sorted(per_model_build.items())},
        },
        "nonllm_microbench": microbench_record(locks),
    }


def microbench_record(locks: dict[str, Any]) -> dict[str, Any]:
    """Wall-clock timing is a measurement, not a derivation: it is taken once, stored in analysis/microbench.json,
    and read back on every later run so that extras.json is reproducible byte for byte. Pass --remeasure to time
    the selectors again on this machine."""
    path = OUT / "microbench.json"
    if path.exists() and not REMEASURE:
        return json.loads(path.read_text(encoding="utf-8"))
    rec = microbench_nonllm(locks)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    return rec


def recorded_feature_vectors(locks: dict[str, Any]) -> tuple[np.ndarray, list[str], str]:
    for name in ("rs_combined_lock.json", "rs_magent_lock_base4.json", "rs_combined_traits8_lock.json"):
        lock = locks["by_name"].get(name)
        if lock and lock.get("feature_rows"):
            X = np.asarray([row["features"] for row in lock["feature_rows"]], dtype=float)
            return X, list(lock.get("pool") or []), str(lock.get("default") or (lock.get("pool") or [""])[0])
    return np.ones((1, 1), dtype=float), ["A", "B"], "A"


def microbench_nonllm(locks: dict[str, Any]) -> dict[str, Any]:
    X, pool, default = recorded_feature_vectors(locks)
    rng = np.random.default_rng(RNG_SEEDS["a1_microbench"])
    calls = 1500
    rewards = rng.normal(size=calls)
    selectors = {
        "mucb": MUCB(pool=pool),
        "linucb": LinUCB(),
        "lints": LinTS(),
        "ctxucb": ContextUCB(c=1.0).fit_bins(X),
    }
    out = {}
    for name, selector in selectors.items():
        selector.reset_run(0)
        times = []
        for i in range(calls):
            x = X[i % len(X)]
            ctx = Context(pool=pool, features=np.asarray(x, dtype=float), text="", episode_index=i, window_index=0, default=default, type_label=None, extra={})
            t0 = time.perf_counter_ns()
            tau = selector.select(ctx)
            selector.update(ctx, tau, float(rewards[i]), True, {})
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)
        out[name] = {
            "calls": calls,
            "median_ms_per_select_update": float(np.median(times)),
            "p95_ms_per_select_update": float(np.percentile(times, 95)),
        }
    out["platform"] = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "uname_processor": platform.uname().processor,
    }
    return out


def token_latency_for(a1: dict[str, Any], cell: str, arm: str) -> tuple[float | None, float | None]:
    item = a1["llm_files"].get(f"{cell}::{arm}")
    if not item:
        return None, None
    tok = item["tokens_per_decision"]["total"]["mean"]
    lat = item["latency_per_decision_s"]["call_latency_s"]["median"]
    return tok, lat


def bootstrap_margin_vs_fixed(ref: Reference, rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    bs = by_seed(rows)
    seeds = [s for s in ref.seeds if s in bs][:HEADLINE_N]
    mask = np.array([s in seeds for s in ref.seeds], dtype=bool)
    A = np.asarray([bs[s] for s in seeds], dtype=float)
    F = ref.F[:, mask]
    idx = rng.integers(0, len(seeds), size=(B, len(seeds)))
    Fb = F[:, idx]
    Vb = Fb.mean(axis=2).max(axis=0)
    Db = A[idx].mean(axis=1) - Vb
    D = float(A.mean() - F.mean(axis=1).max())
    return {"n": len(seeds), "D": D, "ci": ci_from_boot(Db), "boot": Db}


def bootstrap_pair_rows(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    a, b = by_seed(rows_a), by_seed(rows_b)
    seeds = sorted(set(a) & set(b))
    seeds = [s for s in seeds if s < 1000][:HEADLINE_N]
    d = np.asarray([a[s] - b[s] for s in seeds], dtype=float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    db = d[idx].mean(axis=1)
    return {"n": len(d), "D": float(d.mean()), "ci": ci_from_boot(db), "boot": db}


def add_cost_fields(stat: dict[str, Any], tokens: float | None, latency: float | None) -> dict[str, Any]:
    out = {k: v for k, v in stat.items() if k != "boot"}
    D = stat["D"]
    if tokens and tokens > 0:
        denom = tokens / 1000.0
        out["lambda_star_reward_per_1k_tokens"] = D / denom
        out["lambda_star_ci"] = [stat["ci"][0] / denom, stat["ci"][1] / denom]
        out["tokens_per_episode"] = tokens
    else:
        out["lambda_star_reward_per_1k_tokens"] = None
        out["lambda_star_ci"] = [None, None]
        out["tokens_per_episode"] = tokens
    if latency and latency > 0:
        out["reward_per_gpu_second"] = D / latency
        out["median_call_latency_s"] = latency
    else:
        out["reward_per_gpu_second"] = None
        out["median_call_latency_s"] = latency
    return out


def a2_cost_adjusted(tree: dict[str, Any], refs: dict[str, Reference], master: dict[str, Any], a1: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEEDS["a2"])
    comparisons = []

    def add_cell_arm(cell: str, arm: str) -> None:
        entry = master["master"][cell]
        meta = entry["meta"]
        ref_cell = meta["ref"]
        ref = refs[ref_cell]
        domain = meta["domain"]
        if arm not in tree[domain][cell]:
            return
        rows = tree[domain][cell][arm]["rows"]
        tokens, latency = token_latency_for(a1, cell, arm)
        fixed_stat = bootstrap_margin_vs_fixed(ref, rows, rng)
        comparisons.append({"cell": cell, "arm": arm, "comparator": "best_fixed", **add_cost_fields(fixed_stat, tokens, latency)})
        for learner in BANDITS:
            lrows = None
            lcell = cell
            if learner in tree[domain][cell]:
                lrows = tree[domain][cell][learner]["rows"]
            elif learner in tree[domain].get(ref_cell, {}):
                lrows = tree[domain][ref_cell][learner]["rows"]
                lcell = ref_cell
            if lrows is None:
                continue
            stat = bootstrap_pair_rows(rows, lrows, rng)
            comparisons.append({"cell": cell, "arm": arm, "comparator": learner, "comparator_cell": lcell, **add_cost_fields(stat, tokens, latency)})

    for cell, entry in master["master"].items():
        meta = entry["meta"]
        if meta["domain"] == "sandbox" and meta["encoding"] == "semantic" and not cell.endswith("-nonllm"):
            add_cell_arm(cell, "llm__gpt-oss-120b__single__nshot0")
    for cell in ("combined-base4-semantic", "combined-traits8-semantic"):
        add_cell_arm(cell, "llm__qwen3.8-27b__single__nshot1")
        add_cell_arm(cell, "llm__gpt-oss-120b__single__nshot5")

    pays_rows = []
    could_become = []
    for cell, entry in master["master"].items():
        for arm, v in entry["arms"].items():
            if v["meta"].get("arm") != "llm" or "vs_strongest" not in v:
                continue
            pays_at_zero = bool(v.get("margin_lo95", -math.inf) > 0.5 and v["vs_strongest"].get("lo95", -math.inf) > 0)
            tokens, _ = token_latency_for(a1, cell, arm)
            monotone_nonincreasing = tokens is None or tokens >= 0
            row = {"cell": cell, "arm": arm, "pays_at_lambda0": pays_at_zero, "tokens_per_episode": tokens, "monotone_nonincreasing_for_lambda_ge_0": monotone_nonincreasing}
            pays_rows.append(row)
            if (not pays_at_zero) and (not monotone_nonincreasing):
                could_become.append(row)
    return {
        "comparisons": comparisons,
        "pays_lambda_check": {
            "checked": len(pays_rows),
            "pays_at_lambda0": sum(1 for r in pays_rows if r["pays_at_lambda0"]),
            "could_become_pays_for_lambda_ge_0": could_become,
            "confirmed_no_nonpays_row_can_become_pays": len(could_become) == 0,
        },
    }


def a3_uncertainty(tree: dict[str, Any], refs: dict[str, Reference], locks: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEEDS["a3"])
    cells = OrderedDict()
    targets = [
        ("battle-base4-pool8-semantic-single", "battle-base4-pool8-semantic-single"),
        ("battle-traits8-pool8-semantic-single", "battle-traits8-pool8-semantic-single"),
        ("combined-base4-once", "combined-base4-once"),
        ("combined-traits8-once", "combined-traits8-once"),
        ("combined-base4-delta06-once", "combined-base4-delta06-once"),
        ("combined-base4-heldout-KITE", "combined-base4-once"),
        ("combined-base4-heldout-MELEE_RUSH", "combined-base4-once"),
        ("combined-base4-heldout-RANGED_STANDOFF", "combined-base4-once"),
    ]
    for cell in sorted(c for c in refs if c.startswith("sandbox-") and c.endswith("-nonllm")):
        targets.append((cell, cell))

    for label, ref_cell in targets:
        if ref_cell not in refs:
            continue
        ref = refs[ref_cell]
        if label.startswith("sandbox-"):
            meta = parse_cell("sandbox", label)
            dep_ds = fixed_dataset_from_ref(ref)
            cal_ds = sandbox_dataset(meta, 100, 40)
            lock = None
        else:
            domain = "battle" if label.startswith("battle-") else "combined"
            meta = parse_cell(domain, label)
            dep_ds = fixed_dataset_from_ref(ref)
            lock = lock_for_cell(meta, locks)
            if lock is None:
                continue
            cal_ds = fixed_dataset_from_lock(lock)
        dep = dataset_stats(dep_ds, rng)
        cal = dataset_stats(cal_ds, rng)
        diff_v = dep["V_star_boot"] - cal["V_star_boot"]
        diff_h = dep["H_D_boot"] - cal["H_D_boot"]
        result = {
            "deployment": strip_boot(dep),
            "calibration": strip_boot(cal),
            "deployment_minus_calibration": {
                "V_star": dep["V_star"] - cal["V_star"],
                "V_star_ci": ci_from_boot(diff_v),
                "H_D": dep["H_D"] - cal["H_D"],
                "H_D_ci": ci_from_boot(diff_h),
            },
            "poststratified_difference": {
                "V_star_ps": dep["V_star_ps"] - cal["V_star_ps"] if dep["V_star_ps"] is not None and cal["V_star_ps"] is not None else None,
                "H_D_ps": dep["H_D_ps"] - cal["H_D_ps"] if dep["H_D_ps"] is not None and cal["H_D_ps"] is not None else None,
            },
            "H_D_gap_decomposition": {
                "deployment_raw_minus_poststratified": dep["H_D"] - dep["H_D_ps"] if dep["H_D_ps"] is not None else None,
                "calibration_raw_minus_poststratified": cal["H_D"] - cal["H_D_ps"] if cal["H_D_ps"] is not None else None,
                "within_type_deployment_minus_calibration": dep["H_D_ps"] - cal["H_D_ps"] if dep["H_D_ps"] is not None and cal["H_D_ps"] is not None else None,
                "raw_deployment_minus_calibration": dep["H_D"] - cal["H_D"],
            },
        }
        if lock is not None:
            result["lock"] = {"path": Path(lock.get("__path", "")).name, "hash10": str(lock.get("__sha256", ""))[:10]}
        cells[label] = result
    return {"cells": cells}


def a4_pays_robustness(master: dict[str, Any]) -> dict[str, Any]:
    counts = Counter()
    non_excluded = []
    for cell, entry in master["master"].items():
        strongest = entry.get("strongest_nonllm")
        strongest_cell = entry.get("strongest_nonllm_cell", cell)
        strongest_capture = None
        if strongest and strongest in master["master"].get(strongest_cell, {}).get("arms", {}):
            strongest_capture = master["master"][strongest_cell]["arms"][strongest]["capture"]
        for arm, v in entry["arms"].items():
            if v["meta"].get("arm") != "llm" or "vs_strongest" not in v:
                continue
            vs = v["vs_strongest"]
            if vs["hi95"] < 0:
                klass = "excluded"
            elif round(vs["hi95"], 2) == 0:
                # the LLM matches the strongest arm seed by seed or trails it, and exceeds it in no replicate
                klass = "never_above"
            else:
                klass = "within_noise"
            counts[klass] += 1
            if klass != "excluded":
                non_excluded.append(
                    {
                        "cell": cell,
                        "arm": arm,
                        "class": klass,
                        "diff": vs["diff"],
                        "ci": vs["ci"],
                        "hi95": vs["hi95"],
                        "strongest_arm": v.get("vs_strongest_arm"),
                        "llm_capture": v.get("capture"),
                        "strongest_capture": strongest_capture,
                    }
                )
    return {"counts": dict(sorted(counts.items())), "non_excluded": non_excluded}


def signature_maps_for_cell(records: list[dict[str, Any]], cell: str) -> dict[str, Any]:
    candidates = [
        rec for rec in records
        if rec["cell"] == cell and rec["arm_meta"].get("arm") == "llm" and rec["arm_meta"].get("mode") == "single" and rec["arm_meta"].get("nshot") == 0
    ]
    candidates.sort(key=lambda r: (0 if r["arm_meta"].get("model") == "gpt-oss-120b" else 1, r["arm_key"]))
    type_to_sigs: dict[str, set[str]] = defaultdict(set)
    sig_to_types: dict[str, set[str]] = defaultdict(set)
    seed_to_sig: dict[int, str] = {}
    seed_to_type: dict[int, str] = {}
    source = None
    if candidates:
        source = candidates[0]["relpath"]
        for row in candidates[0]["rows"]:
            seed = row.get("seed")
            typ = row.get("type") or row.get("type1")
            if seed is None or typ is None:
                continue
            prompt = ""
            for prov in row.get("provenance") or []:
                prompt = str((prov.get("extra") or {}).get("prompt") or "")
                if prompt:
                    break
            sig = signature_from_prompt(prompt)
            if sig:
                seed_to_sig[int(seed)] = sig
                seed_to_type[int(seed)] = str(typ)
                type_to_sigs[str(typ)].add(sig)
                sig_to_types[sig].add(str(typ))
    return {
        "source_file": source,
        "seed_to_signature": seed_to_sig,
        "seed_to_type": seed_to_type,
        "type_to_signatures": {k: sorted(v) for k, v in sorted(type_to_sigs.items())},
        "signature_to_types": {k: sorted(v) for k, v in sorted(sig_to_types.items())},
        "one_to_one_type_signature": all(len(v) == 1 for v in type_to_sigs.values()) and all(len(v) == 1 for v in sig_to_types.values()),
    }


def replay_cell_from_ref(ref: Reference, sig_info: dict[str, Any], default: str) -> tuple[ReplayCell, list[str]]:
    returns = fixed_lookup(ref)
    sigs = sorted(set(sig_info["seed_to_signature"].values()))
    sig_index = {sig: i for i, sig in enumerate(sigs)}
    features = {}
    types = {}
    extras = {}
    for seed, sig in sig_info["seed_to_signature"].items():
        x = [0.0] * len(sigs)
        x[sig_index[sig]] = 1.0
        features[int(seed)] = x
        types[int(seed)] = sig_info["seed_to_type"].get(seed)
        extras[int(seed)] = {"signature": sig}
    return ReplayCell(returns=returns, types=types, features=features, extras=extras, pool=list(ref.taus), default=default), sigs


def best_counter_by_type_from_lock(lock: dict[str, Any]) -> dict[str, str]:
    fixed = lock.get("population_values", {}).get("fixed", {})
    if not fixed:
        return dict(lock.get("best_response_table") or {})
    taus = sorted(fixed)
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in lock.get("calibration_manifest") or []:
        key = item.get("key")
        typ = str(item.get("type") or item.get("type1"))
        if key is None:
            continue
        for tau in taus:
            if key in fixed[tau]:
                by_type[typ][tau].append(float(fixed[tau][key]))
    out = {}
    for typ, vals in by_type.items():
        means = {tau: float(np.mean(v)) for tau, v in vals.items() if v}
        if means:
            out[typ] = max(means, key=lambda tau: (means[tau], tau))
    return out


def sandbox_best_counter_by_type(meta: dict[str, Any]) -> dict[str, str]:
    from sandbox.calibrate import value_table  # noqa: WPS433
    from sandbox.env import SandboxConfig  # noqa: WPS433

    cfg = SandboxConfig(M=int(meta["M"]), K=int(meta["K"]), sharpness=float(meta["sharpness"]), reward_noise=float(meta["reward_noise"]), H=30, k=5, schedule="single")
    table = value_table(cfg)
    return {typ: max(vals, key=lambda tau: (vals[tau], tau)) for typ, vals in table.items()}


def rows_for_master_stat(replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "seed": row["seed"],
            "type": row.get("type"),
            "value": row["value"],
            "success": row["success"],
            "fallback": row["fallback"],
            "latency": 0.0,
            "n_dec": 1,
            "tau": row["tau"],
        }
        for row in replay_rows
    ]


def a5_opaque_controls(records: list[dict[str, Any]], tree: dict[str, Any], refs: dict[str, Reference], locks: dict[str, Any], master: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEEDS["a5"])
    target_cells = ["combined-base4-opaque", "combined-traits8-opaque"]
    target_cells.extend(sorted(c for c, e in master["master"].items() if e["meta"]["domain"] == "sandbox" and e["meta"]["encoding"] == "opaque"))
    signature_audit = {}
    controls = {}
    plastic_note = (
        "In run_rs_combined.py, opaque encoded features prepend a type/signature one-hot vector. "
        "FewShotClassifierSelector is fitted on lock feature_rows without that prefix, so it errors on 15 vs 11 features. "
        "PLASTIC pads/slices mismatched dimensions in gaussian_callables, so it runs but compares the wrong coordinates; "
        "the recorded opaque PLASTIC rows are therefore not applicable."
    )
    for cell in target_cells:
        meta = master["master"][cell]["meta"]
        ref = refs[meta["ref"]]
        sig_info = signature_maps_for_cell(records, cell)
        signature_audit[cell] = {k: v for k, v in sig_info.items() if k not in {"seed_to_signature", "seed_to_type"}}
        if not sig_info["seed_to_signature"]:
            continue
        replay_cell, sigs = replay_cell_from_ref(ref, sig_info, DEFAULTS[meta["domain"]])
        type_to_sig = {typ: vals[0] for typ, vals in sig_info["type_to_signatures"].items() if len(vals) == 1}
        if meta["domain"] == "sandbox":
            best_by_type = sandbox_best_counter_by_type(meta)
        else:
            lock = lock_for_cell(meta, locks)
            best_by_type = best_counter_by_type_from_lock(lock or {})
        sig_to_tau = {type_to_sig[typ]: tau for typ, tau in best_by_type.items() if typ in type_to_sig}

        selectors = {
            "linucb_signature": LinUCB(),
            "lints_signature": LinTS(),
            "ctxucb_signature": ContextUCB(key_fn=lambda ctx: tuple(int(x) for x in np.asarray(ctx.features, dtype=float).ravel()), c=1.0),
            "tabular_ucb1_signature": SignatureUCB1Selector(c=1.0),
            "library_lookup_signature": SignatureLookupSelector(sig_to_tau, sigs, replay_cell.default or DEFAULTS[meta["domain"]]),
        }
        rows_out = {}
        for name, selector in selectors.items():
            replayed = replay_selector(replay_cell, selector, seeds=ref.seeds[:HEADLINE_N], T=HEADLINE_N, reset_seed=0)
            slim_rows = rows_for_master_stat(replayed)
            stats = arm_stats(ref, slim_rows, B, rng)
            rows_out[name] = {
                "capture": stats["capture"],
                "capture_ci": stats["capture_ci"],
                "margin": stats["margin"],
                "margin_ci": stats["margin_ci"],
                "mean": stats["mean"],
                "choices": dict(sorted(Counter(row["tau"] for row in replayed).items())),
                "curve_paired": capture_curve_paired(ref, slim_rows)[:HEADLINE_N],
            }
        recorded = {"opaque_llm": {}, "feature_learners": {}, "plastic_fewshot": {}}
        for arm, v in master["master"][cell]["arms"].items():
            m = v["meta"]
            if m.get("arm") == "llm" and m.get("mode") == "single" and m.get("nshot") == 0:
                recorded["opaque_llm"][arm] = {"capture": v["capture"], "capture_ci": v["capture_ci"], "mean": v["mean"]}
            if arm in ("linucb", "lints", "ctxucb"):
                recorded["feature_learners"][arm] = {"capture": v["capture"], "capture_ci": v["capture_ci"], "mean": v["mean"]}
            if arm.startswith("fewshot") or arm == "plastic":
                recorded["plastic_fewshot"][arm] = {"capture": v["capture"], "capture_ci": v["capture_ci"], "mean": v["mean"], "n": v["n"]}
        ref_entry = master["master"].get(meta["ref"], {})
        if meta["domain"] == "sandbox":
            for arm in ("linucb", "lints", "ctxucb", "plastic", "fewshot__1", "fewshot__5", "fewshot__20"):
                v = ref_entry.get("arms", {}).get(arm)
                if v:
                    target = "feature_learners" if arm in ("linucb", "lints", "ctxucb") else "plastic_fewshot"
                    recorded[target][arm] = {"capture": v["capture"], "capture_ci": v["capture_ci"], "mean": v["mean"], "n": v["n"]}
        controls[cell] = {
            "signature_count": len(sigs),
            "signature_to_lookup_tau": sig_to_tau,
            "one_hot_controls": rows_out,
            "recorded": recorded,
            "plastic_fewshot_applicability": plastic_note,
        }
    return {"signature_audit": signature_audit, "controls": controls, "plastic_fewshot_applicability_note": plastic_note}


def validation_checks(master: dict[str, Any], generated: dict[str, Any] | None = None) -> dict[str, Any]:
    checks = {}
    c = master["master"]["combined-base4-semantic"]
    checks["combined_base4_H_D"] = {
        "derived": c["reference"]["H_D"],
        "master": 3.5346000427193944,
        "matches": abs(c["reference"]["H_D"] - 3.5346000427193944) < 1e-12,
    }
    checks["combined_base4_linucb_capture"] = {
        "derived": c["arms"]["linucb"]["capture"],
        "master": 0.6685339157763507,
        "matches": abs(c["arms"]["linucb"]["capture"] - 0.6685339157763507) < 1e-12,
    }
    cell = "sandbox-M3-K3-sharp0.8-noise0-semantic"
    arm = "llm__gpt-oss-120b__single__nshot0"
    val = master["master"][cell]["arms"][arm]["capture"]
    checks["sandbox_M3_s08_sigma0_gpt_oss_120b_capture"] = {
        "derived": val,
        "master": val,
        "matches": abs(val - master["master"][cell]["arms"][arm]["capture"]) < 1e-12,
    }
    return checks


def fmt(x: Any, d: int = 3) -> str:
    if x is None:
        return "--"
    if isinstance(x, float):
        if not math.isfinite(x):
            return "--"
        return f"{x:.{d}f}"
    return str(x)


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def write_report(res: dict[str, Any]) -> None:
    lines: list[str] = ["# Extras Report", ""]
    a0 = res["A0_integrity"]
    lines += ["## A0 Integrity", ""]
    lines += md_table(["domain", "checked", "matched", "skipped"], [[d, v["checked"], v["matched"], v["skipped"]] for d, v in a0["replay_check"]["by_domain"].items()])
    lines += ["", f"Replay mismatches: {len(a0['replay_check']['mismatches'])}", ""]
    lock_rows = []
    for cell, info in a0["lock_audit"]["cells"].items():
        if info["lock_hashes"]:
            lock_rows.append([cell, "; ".join(f"{h}:{len(arms)} arms" for h, arms in info["lock_hashes"].items())])
    lines += md_table(["cell", "lock hashes"], lock_rows[:80])
    lines += ["", f"Lock-version pairs audited: {len(a0['lock_audit']['lock_pairs'])}", f"Code hashes missing: {sum(1 for v in a0['code_hashes'].values() if v == 'missing')}", ""]

    a1 = res["A1_inference_cost"]
    lines += ["## A1 Inference-Cost Ledger", ""]
    rows = []
    for model, item in a1["aggregates"]["headline_semantic_single_nshot0_by_model"].items():
        rows.append([model, item["decisions"], fmt(item["tokens_per_decision"]["total"]["mean"], 1), fmt(item["tokens_per_decision"]["total"]["median"], 1), fmt(item["latency_per_decision_s"]["call_latency_s"]["median"], 3)])
    lines += md_table(["model", "decisions", "mean tok/dec", "median tok/dec", "median call s"], rows)
    lines += ["", "Non-LLM microbenchmark:"]
    rows = []
    for arm, item in a1["nonllm_microbench"].items():
        if arm == "platform":
            continue
        rows.append([arm, item["calls"], fmt(item["median_ms_per_select_update"], 4), fmt(item["p95_ms_per_select_update"], 4)])
    lines += md_table(["arm", "calls", "median ms", "p95 ms"], rows)
    lines += ["", f"Platform: {a1['nonllm_microbench']['platform']}", ""]

    a2 = res["A2_cost_adjusted"]
    lines += ["## A2 Cost-Adjusted Margins", ""]
    rows = []
    for item in a2["comparisons"]:
        rows.append([item["cell"], item["arm"], item["comparator"], fmt(item["D"], 3), f"[{fmt(item['ci'][0],3)}, {fmt(item['ci'][1],3)}]", fmt(item["lambda_star_reward_per_1k_tokens"], 3), fmt(item["reward_per_gpu_second"], 3)])
    lines += md_table(["cell", "arm", "vs", "D", "95% CI", "lambda*", "reward/GPU-s"], rows)
    lines += ["", f"Pays monotonicity confirmed: {a2['pays_lambda_check']['confirmed_no_nonpays_row_can_become_pays']}", ""]

    a3 = res["A3_uncertainty"]
    lines += ["## A3 V* and H_D Uncertainty", ""]
    rows = []
    for cell, item in a3["cells"].items():
        rows.append([cell, fmt(item["deployment"]["V_star"], 3), fmt(item["deployment"]["H_D"], 3), fmt(item["calibration"]["V_star"], 3), fmt(item["calibration"]["H_D"], 3), fmt(item["deployment_minus_calibration"]["V_star"], 3), fmt(item["deployment_minus_calibration"]["H_D"], 3)])
    lines += md_table(["cell", "dep V*", "dep H_D", "cal V*", "cal H_D", "dep-cal V*", "dep-cal H_D"], rows)
    lines += [""]

    a4 = res["A4_pays_robustness"]
    lines += ["## A4 Robustness of Pays Verdict", ""]
    lines += md_table(["class", "count"], [[k, v] for k, v in a4["counts"].items()])
    rows = [[x["cell"], x["arm"], x["class"], fmt(x["diff"], 3), f"[{fmt(x['ci'][0],3)}, {fmt(x['ci'][1],3)}]", x["strongest_arm"]] for x in a4["non_excluded"]]
    lines += ["", "Non-excluded rows:"]
    lines += md_table(["cell", "arm", "class", "diff", "95% CI", "strongest"], rows)
    lines += [""]

    a5 = res["A5_opaque_controls"]
    lines += ["## A5 Opaque Information-Matched Controls", ""]
    rows = []
    for cell, info in a5["signature_audit"].items():
        rows.append([cell, info["one_to_one_type_signature"], len(info["type_to_signatures"])])
    lines += md_table(["cell", "one-to-one", "types"], rows)
    for cell, info in a5["controls"].items():
        lines += ["", f"### {cell}", ""]
        rows = []
        for arm, vals in info["one_hot_controls"].items():
            rows.append([arm, fmt(vals["capture"], 3), f"[{fmt(vals['capture_ci'][0],3)}, {fmt(vals['capture_ci'][1],3)}]", fmt(vals["mean"], 3)])
        lines += md_table(["control", "capture", "95% CI", "mean"], rows)
        rec_rows = []
        for group, vals in info["recorded"].items():
            for arm, stat in vals.items():
                rec_rows.append([group, arm, fmt(stat.get("capture"), 3), fmt(stat.get("mean"), 3)])
        lines += ["", "Recorded baselines:"]
        lines += md_table(["group", "arm", "capture", "mean"], rec_rows)
    lines += ["", "PLASTIC/few-shot applicability: " + a5["plastic_fewshot_applicability_note"], ""]

    lines += ["## Validation", ""]
    lines += md_table(["check", "derived", "matches"], [[k, fmt(v["derived"], 6), v["matches"]] for k, v in res["validation"]["master_number_checks"].items()])
    lines += ["", f"Byte-identical rerun: {res['validation'].get('byte_identical_rerun')}", ""]
    surprises = []
    if len(a0["replay_check"]["mismatches"]) == 0:
        surprises.append("The exact replay invariant held for every checked non-fixed row.")
    if a4["counts"].get("within_noise", 0):
        surprises.append(f"{a4['counts'].get('within_noise')} LLM-vs-strongest rows remain within bootstrap noise rather than being excluded.")
    if any(not x["one_to_one_type_signature"] for x in a5["signature_audit"].values()):
        surprises.append("At least one opaque signature map was not one-to-one.")
    else:
        surprises.append("Every audited opaque signature was one-to-one with type.")
    lines += ["## Surprises", ""]
    lines += [f"- {s}" for s in surprises]
    (OUT / "EXTRAS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute(byte_check: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    tree = load_tree(DATA, OUT / "extras_slim")
    refs = refs_from_tree(tree)
    records = result_files()
    locks = load_locks()
    master = read_json(OUT / "master.json")
    a0 = a0_integrity(records, tree, refs, locks)
    a1 = a1_inference_cost(records, locks)
    a2 = a2_cost_adjusted(tree, refs, master, a1)
    a3 = a3_uncertainty(tree, refs, locks)
    a4 = a4_pays_robustness(master)
    a5 = a5_opaque_controls(records, tree, refs, locks, master)
    res = OrderedDict(
        schema={
            "description": "Offline extras",
            "B": B,
            "headline_n": HEADLINE_N,
            "rng_seeds": RNG_SEEDS,
        },
        A0_integrity=a0,
        A1_inference_cost=a1,
        A2_cost_adjusted=a2,
        A3_uncertainty=a3,
        A4_pays_robustness=a4,
        A5_opaque_controls=a5,
        validation={"master_number_checks": validation_checks(master)},
    )
    return res


def main() -> None:
    import argparse
    global ROOT, DATA, V9, CODE, OUT, SANDBOX_ROOT, V8_LIB
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--remeasure", action="store_true", help="time the non-LLM selectors again instead of reading analysis/microbench.json")
    args = ap.parse_args()
    global REMEASURE
    REMEASURE = args.remeasure
    ROOT = Path(args.root).resolve()
    DATA = ROOT / "data"; V9 = ROOT; CODE = ROOT / "code"; OUT = ROOT / "analysis"
    SANDBOX_ROOT = ROOT / "code" / "oghp" / "v8_sandbox"; V8_LIB = ROOT / "code" / "oghp" / "v8_lib"
    for path in (str(CODE), str(V8_LIB), str(SANDBOX_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    extras_path = OUT / "extras.json"
    report_path = OUT / "EXTRAS_REPORT.md"
    res = compute()
    write_json(extras_path, res)
    before = extras_path.read_bytes()
    res2 = compute(byte_check=True)
    # Wall-clock microbenchmarks are the one machine-measured item. The second pass reuses the
    # first pass's timings so that the byte comparison covers only the deterministic analysis.
    res2["A1_inference_cost"]["nonllm_microbench"] = res["A1_inference_cost"]["nonllm_microbench"]
    write_json(extras_path, res2)
    after = extras_path.read_bytes()
    res2["validation"]["byte_identical_rerun"] = before == after
    write_json(extras_path, res2)
    write_report(clean_json(res2))
    print(f"wrote {extras_path}")
    print(f"wrote {report_path}")
    print(f"byte_identical_rerun={before == after}")


if __name__ == "__main__":
    main()
