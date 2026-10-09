#!/usr/bin/env python3
"""Aggregate regime-switching campaign outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V8_ROOT = ROOT.parent / "v8_lib"
for path in (ROOT, V8_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from v8lib.analysis import capture, capture_curve, crossover, master_row, paired_bootstrap, scaling_fit

LEARNER_PREFIXES = ("fewshot:",)
LEARNER_ARMS = {"mucb", "linucb", "lints", "ctxucb", "plastic", "ppo"}
DOMAIN_ALIASES = {"magent": "battle"}


@dataclass
class RunData:
    path: Path
    domain: str
    tag: str
    cell: str
    arm: str
    meta: dict[str, Any]
    rows: list[dict[str, Any]]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.results))
    locks = {
        "battle": load_json(Path(args.lock_battle or args.lock_magent)),
        "combined": load_json(Path(args.lock_combined)),
        "sandbox": load_json(Path(args.lock_sandbox)),
        "magent": load_json(Path(args.lock_magent)),
    }
    artifacts = aggregate(runs, locks, out=out, bootstrap_B=int(args.bootstrap_B))
    print(f"wrote {out}")
    print(f"master_rows={len(artifacts['master_table'])} capture_curves={len(artifacts['capture_curves'])}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=ROOT / "results")
    p.add_argument("--lock-battle")
    p.add_argument("--lock-magent", default=ROOT / "rs_magent_lock.json")
    p.add_argument("--lock-combined", default=ROOT / "rs_combined_lock.json")
    p.add_argument("--lock-sandbox", default=ROOT / "sandbox_lock.json")
    p.add_argument("--out", default=ROOT / "analysis")
    p.add_argument("--bootstrap-B", type=int, default=200)
    return p.parse_args(argv)


def aggregate(
    runs: list[RunData],
    locks: dict[str, dict[str, Any]],
    *,
    out: Path,
    bootstrap_B: int = 200,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cells = group_by_cell(runs)
    master_rows = []
    capture_rows = []
    scaling_points = []
    legibility: dict[str, Any] = {}
    heldout: dict[str, Any] = {}
    departure: dict[str, Any] = {}
    for key in sorted(cells):
        cell_runs = cells[key]
        domain, tag, cell_id = key
        params = infer_params(cell_runs[0])
        by_arm = {r.arm: r for r in cell_runs}
        lock_metrics = cell_lock_metrics(domain, params, locks.get(domain, locks.get("magent", {})))
        refs = reference_metrics(params, lock_metrics, by_arm)
        v_star = float(refs["deployment_population"]["V_star"])
        h_d = float(refs["deployment_population"]["H_D"])
        default = str(params.get("default", lock_metrics.get("default", "HOLD_POSITION")))
        means = {arm: mean_reward(run.rows) for arm, run in by_arm.items()}
        non_llm = {arm: val for arm, val in means.items() if not is_llm_arm(arm) and not is_reference_arm(arm)}
        llm_arms = {arm: val for arm, val in means.items() if is_llm_arm(arm)}
        best_arm = max(non_llm, key=lambda arm: (non_llm[arm], arm)) if non_llm else None
        best_mean = non_llm[best_arm] if best_arm is not None else 0.0
        best_run = by_arm[best_arm] if best_arm is not None else None
        best_payload = {
            "name": best_arm,
            "mean_reward": best_mean,
            "margin": best_mean - v_star,
            "capture": capture(best_mean, v_star, h_d),
            "source_file": str(best_run.path) if best_run else None,
        }
        llm_payload = {}
        for arm in sorted(llm_arms):
            run = by_arm[arm]
            family = llm_key(arm)
            rewards = rewards_by_seed(run.rows)
            best_rewards = rewards_by_seed(best_run.rows) if best_run else {}
            paired_seeds = sorted(set(rewards) & set(best_rewards))
            if paired_seeds:
                boot_best = paired_bootstrap(
                    {s: [rewards[s]] for s in paired_seeds},
                    {s: [best_rewards[s]] for s in paired_seeds},
                    B=bootstrap_B,
                    seed=stable_seed(domain, tag, cell_id, arm),
                )
            else:
                boot_best = {"mean": llm_arms[arm] - best_mean, "ci": (None, None)}
            if rewards:
                boot_star = paired_bootstrap(
                    {s: [v] for s, v in rewards.items()},
                    {s: [v_star] for s in rewards},
                    B=bootstrap_B,
                    seed=stable_seed(domain, tag, cell_id, arm, "star"),
                )
            else:
                boot_star = {"mean": llm_arms[arm] - v_star, "ci": (None, None)}
            llm_payload[family] = {
                "arm": arm,
                "mean_reward": llm_arms[arm],
                "margin": llm_arms[arm] - v_star,
                "margin_ci": list(boot_star["ci"]),
                "capture": capture(llm_arms[arm], v_star, h_d),
                "margin_vs_best_non_llm": boot_best["mean"],
                "margin_vs_best_non_llm_ci": list(boot_best["ci"]),
                "source_file": str(run.path),
            }
            legibility.setdefault(params.get("encoding", "unknown"), {}).setdefault(family, []).append(
                llm_payload[family]["capture"]
            )
        row = master_row(
            cell=f"{domain}/{tag}/{cell_id}",
            M=int(params.get("M", lock_metrics.get("M", 0))),
            W=int(params.get("W", max_decisions(cell_runs))),
            encoding=str(params.get("encoding", "unknown")),
            H_D=h_d,
            V_star=v_star,
            best_non_llm=best_payload,
            llm=llm_payload,
            conditions=conditions(h_d, best_payload, llm_payload),
            prediction=prediction(h_d, best_payload, llm_payload),
            verdict=verdict(best_payload, llm_payload),
        )
        row["calibration"] = refs["calibration"]
        row["deployment_population"] = refs["deployment_population"]
        row["reference_sources"] = refs["sources"]
        master_rows.append(row)
        for arm in sorted(by_arm):
            if not is_learner_arm(arm):
                continue
            curve_entry = learner_curve_entry(
                by_arm[arm],
                llm_payload,
                v_star=v_star,
                h_d=h_d,
                bootstrap_B=bootstrap_B,
                seed=stable_seed(domain, tag, cell_id, arm, "curve"),
            )
            capture_rows.append(curve_entry)
            x = scaling_x(params, lock_metrics)
            for family, t_payload in curve_entry["T_star"].items():
                if t_payload["point"] is not None and x > 0:
                    scaling_points.append(
                        {
                            "cell": f"{domain}/{tag}/{cell_id}",
                            "arm": arm,
                            "family": family,
                            "x": x,
                            "T_star": t_payload["point"],
                        }
                    )
        heldout[f"{domain}/{tag}/{cell_id}"] = heldout_summary(cell_runs, v_star, h_d)
        departure[f"{domain}/{tag}/{cell_id}"] = departure_summary(cell_runs, default)
    legibility_out = {
        enc: {family: summarize_values(vals) for family, vals in sorted(families.items())}
        for enc, families in sorted(legibility.items())
    }
    scaling_out = {"points": scaling_points, "fit": None}
    if len(scaling_points) >= 2:
        try:
            scaling_out["fit"] = scaling_fit(
                [p["x"] for p in scaling_points],
                [p["T_star"] for p in scaling_points],
                B=bootstrap_B,
                seed=17,
            )
        except ValueError:
            scaling_out["fit"] = None
    artifacts = {
        "master_table": master_rows,
        "capture_curves": capture_rows,
        "scaling": scaling_out,
        "legibility": legibility_out,
        "heldout": heldout,
        "departure": departure,
    }
    write_json(out / "master_table.json", master_rows)
    write_master_md(out / "master_table.md", master_rows)
    write_json(out / "capture_curves.json", capture_rows)
    write_json(out / "scaling.json", scaling_out)
    write_json(out / "legibility.json", legibility_out)
    write_json(out / "heldout.json", heldout)
    write_json(out / "departure.json", departure)
    write_figures(out, master_rows, capture_rows, scaling_out)
    return artifacts


def load_runs(results: Path) -> list[RunData]:
    out: list[RunData] = []
    for path in sorted([*results.rglob("*.json"), *results.rglob("*.jsonl")]):
        if path.name == "MANIFEST.json" or path.name == "MANIFEST.jsonl":
            continue
        data = load_run_payload(path)
        if not data:
            continue
        rows = data.get("rows", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            continue
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        domain, tag, cell = infer_path_parts(results, path, meta)
        arm = infer_arm(path, meta, rows)
        out.append(RunData(path=path, domain=canonical_domain(domain), tag=tag, cell=cell, arm=arm, meta=meta, rows=rows))
    return out


def infer_path_parts(results: Path, path: Path, meta: dict[str, Any]) -> tuple[str, str, str]:
    try:
        rel = path.relative_to(results)
        parts = rel.parts
        if len(parts) >= 4:
            return str(parts[0]), str(parts[1]), str(parts[2])
    except ValueError:
        pass
    args = meta.get("args", meta)
    domain = str(meta.get("domain") or args.get("domain") or ("oc" if "layout" in args else "battle"))
    tag = str(meta.get("tag") or args.get("tag") or "untagged")
    cell = str(meta.get("cell") or args.get("cell") or path.parent.name)
    return domain, tag, cell


def infer_arm(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    args = meta.get("args", meta)
    if args.get("arm"):
        return str(args["arm"])
    if meta.get("arm"):
        return str(meta["arm"])
    for row in rows:
        for key in ("method", "arm"):
            if row.get(key):
                return str(row[key])
    return path.stem.replace("__", ":")


def group_by_cell(runs: list[RunData]) -> dict[tuple[str, str, str], list[RunData]]:
    grouped: dict[tuple[str, str, str], list[RunData]] = {}
    for run in runs:
        grouped.setdefault((run.domain, run.tag, run.cell), []).append(run)
    return grouped


def infer_params(run: RunData) -> dict[str, Any]:
    args = dict(run.meta.get("args", {}))
    args.update({k: v for k, v in run.meta.items() if k not in {"args", "rows"}})
    if "layout" in args and "type_set" not in args:
        args["type_set"] = args["layout"]
    if run.domain == "sandbox":
        args.update({k: v for k, v in parse_sandbox_cell(run.cell).items() if k not in args})
    if "M" not in args:
        args["M"] = len({type_label(row) for row in run.rows if type_label(row)})
    args["W"] = max((len(row.get("decisions", row.get("windows", []))) for row in run.rows), default=1)
    return args


def cell_lock_metrics(domain: str, params: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    if domain == "oc":
        layout = str(params.get("layout", params.get("type_set", "")))
        payload = lock.get("layouts", {}).get(layout, {})
        return {
            "V_star": payload.get("V_star", lock.get("V_star", 0.0)),
            "H_D": payload.get("H_D", lock.get("H_D", 1.0)),
            "default": lock.get("meta", {}).get("default", "ROLE_FULL"),
            "delta_min": delta_min(payload.get("Delta", {})),
            "K": params.get("horizon", payload.get("horizon", 1)),
            "M": params.get("M", len(payload.get("best_response", {}))),
        }
    if domain == "sandbox":
        value_table = lock.get("value_table", {})
        fixed_vals = pooled_values(value_table)
        best_fixed = max(fixed_vals, key=lambda tau: (fixed_vals[tau], tau)) if fixed_vals else None
        default = str(params.get("default", lock.get("config", {}).get("default", "ROCK")))
        v_star = fixed_vals.get(best_fixed, 0.0) if best_fixed else 0.0
        oracle = type_oracle_value(value_table)
        return {
            "V_star": v_star,
            "H_D": oracle - v_star if oracle is not None else lock.get("H_D", 0.0),
            "V_default": fixed_vals.get(default, 0.0),
            "default": default,
            "delta_min": delta_min(value_table),
            "K": params.get("K", lock.get("config", {}).get("K", 3)),
            "M": params.get("M", lock.get("config", {}).get("M", 0)),
            "headroom_share": lock.get("headroom_share"),
        }
    return {
        "V_star": lock.get("V_star", 0.0),
        "H_D": lock.get("H_D", 1.0),
        "default": lock.get("default", "HOLD_POSITION"),
        "V_default": lock.get("V_default", 0.0),
        "Delta_def": lock.get("Delta_def", 0.0),
        "placebo": lock.get("placebo", 0.0),
        "headroom_share": lock.get("headroom_share"),
        "best_fixed": lock.get("best_fixed"),
        "per_type_report": lock.get("per_type_report", {}),
        "deployment_manifest_n": len(lock.get("deployment_manifest", [])),
        "ppo_checkpoint_hash": lock.get("ppo_checkpoint_hash"),
        "delta_min": delta_min(lock.get("Delta_per_type", lock.get("Delta", {}))),
        "K": params.get("k", lock.get("k", 1)),
        "M": params.get("M", len(lock.get("types", []))),
    }


def reference_metrics(params: dict[str, Any], lock_metrics: dict[str, Any], by_arm: dict[str, RunData]) -> dict[str, Any]:
    default = str(params.get("default", lock_metrics.get("default", "HOLD_POSITION")))
    fixed_means = {
        arm.split(":", 1)[1]: mean_reward(run.rows)
        for arm, run in by_arm.items()
        if arm.startswith("fixed:")
    }
    oracle_run = by_arm.get("type_oracle")
    best_fixed = max(fixed_means, key=lambda tau: (fixed_means[tau], tau)) if fixed_means else None
    dep_v_star = fixed_means[best_fixed] if best_fixed is not None else float(lock_metrics.get("V_star", 0.0))
    dep_v_default = fixed_means.get(default, float(lock_metrics.get("V_default", 0.0)))
    oracle_mean = mean_reward(oracle_run.rows) if oracle_run is not None else None
    dep_h_d = (
        max(0.0, float(oracle_mean) - float(dep_v_star))
        if oracle_mean is not None and fixed_means
        else float(lock_metrics.get("H_D", 0.0))
    )
    calibration = {
        "V_star": float(lock_metrics.get("V_star", 0.0)),
        "H_D": float(lock_metrics.get("H_D", 0.0)),
        "V_default": float(lock_metrics.get("V_default", 0.0)),
        "Delta_def": lock_metrics.get("Delta_def"),
        "placebo": lock_metrics.get("placebo"),
        "headroom_share": lock_metrics.get("headroom_share"),
        "best_fixed": lock_metrics.get("best_fixed"),
        "per_type_report": lock_metrics.get("per_type_report", {}),
        "deployment_manifest_n": lock_metrics.get("deployment_manifest_n"),
        "ppo_checkpoint_hash": lock_metrics.get("ppo_checkpoint_hash"),
    }
    deployment = {
        "V_star": float(dep_v_star),
        "H_D": float(dep_h_d),
        "V_default": float(dep_v_default),
        "V_oracle": None if oracle_mean is None else float(oracle_mean),
        "best_fixed": best_fixed,
        "fixed_means": fixed_means,
    }
    return {
        "calibration": calibration,
        "deployment_population": deployment,
        "sources": {
            "V_star": "deployment_fixed_runs" if fixed_means else "calibration_lock",
            "H_D": "deployment_type_oracle_minus_best_fixed" if oracle_mean is not None and fixed_means else "calibration_lock",
        },
    }


def learner_curve_entry(
    run: RunData,
    llm_payload: dict[str, Any],
    *,
    v_star: float,
    h_d: float,
    bootstrap_B: int,
    seed: int,
) -> dict[str, Any]:
    ordered = sorted(run.rows, key=lambda r: (int(r.get("episode", r.get("seed", 0))), int(r.get("seed", 0))))
    rewards = [float(r.get("total_reward", 0.0)) for r in ordered]
    curve = capture_curve(rewards, v_star, h_d).tolist()
    t_star = {}
    for family, payload in llm_payload.items():
        result = crossover(np.asarray(curve, dtype=float), float(payload["capture"]), B=bootstrap_B, seed=seed)
        t_star[family] = {
            "point": result["T_star"],
            "ci": list(result["ci"]),
            "llm_capture": payload["capture"],
            "censored": result["censored"],
            "status": result["status"],
            "crossing_probability": result["crossing_probability"],
        }
    return {
        "domain": run.domain,
        "tag": run.tag,
        "cell": run.cell,
        "arm": run.arm,
        "T": list(range(1, len(curve) + 1)),
        "capture": curve,
        "T_star": t_star,
        "source_file": str(run.path),
    }


def heldout_summary(cell_runs: list[RunData], v_star: float, h_d: float) -> dict[str, Any]:
    out = {}
    for run in cell_runs:
        held = [float(r.get("total_reward", 0.0)) for r in run.rows if bool(r.get("held_out")) or r.get("split") == "heldout"]
        seen = [float(r.get("total_reward", 0.0)) for r in run.rows if not (bool(r.get("held_out")) or r.get("split") == "heldout")]
        out[run.arm] = {
            "heldout_capture": capture(float(np.mean(held)) if held else 0.0, v_star, h_d) if held else None,
            "seen_capture": capture(float(np.mean(seen)) if seen else 0.0, v_star, h_d) if seen else None,
            "heldout_n": len(held),
            "seen_n": len(seen),
            "source_file": str(run.path),
        }
    return out


def departure_summary(cell_runs: list[RunData], default: str) -> dict[str, Any]:
    default_run = next((r for r in cell_runs if r.arm == f"fixed:{default}"), None)
    default_rewards = rewards_by_seed(default_run.rows) if default_run else {}
    out = {}
    for run in cell_runs:
        departed_effects = []
        reproduced_effects = []
        for row in run.rows:
            seed = int(row.get("seed", row.get("episode", -1)))
            if seed not in default_rewards:
                continue
            effect = float(row.get("total_reward", 0.0)) - default_rewards[seed]
            if departed_from_default(row, default):
                departed_effects.append(effect)
            else:
                reproduced_effects.append(effect)
        n = len(departed_effects) + len(reproduced_effects)
        out[run.arm] = {
            "departed_share": len(departed_effects) / n if n else None,
            "departed_mean_effect": float(np.mean(departed_effects)) if departed_effects else None,
            "reproduced_mean_effect": float(np.mean(reproduced_effects)) if reproduced_effects else None,
            "mean_loss_per_departure": float(np.mean([-x for x in departed_effects])) if departed_effects else None,
            "paired_n": n,
            "source_file": str(run.path),
        }
    return out


def write_figures(out: Path, master_rows: list[dict[str, Any]], capture_rows: list[dict[str, Any]], scaling: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    for domain in sorted({row["cell"].split("/", 1)[0] for row in master_rows} or {"none"}):
        xs = [row["M"] for row in master_rows if row["cell"].startswith(f"{domain}/")]
        ys = [row["H_D"] for row in master_rows if row["cell"].startswith(f"{domain}/")]
        if xs:
            ax.scatter(xs, ys, label=domain)
    ax.set_xlabel("M")
    ax.set_ylabel("H_D")
    ax.legend(loc="best")
    save_fig(fig, out / "fig_headroom_atlas")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for row in capture_rows:
        ax.plot(row["T"], row["capture"], label=f"{row['cell']}:{row['arm']}")
        for payload in row["T_star"].values():
            ax.axhline(payload["llm_capture"], color="0.7", linewidth=0.8)
    ax.set_xlabel("T")
    ax.set_ylabel("capture")
    if capture_rows:
        ax.legend(loc="best", fontsize=7)
    save_fig(fig, out / "fig_capture_T")

    fig, ax = plt.subplots(figsize=(5, 3))
    points = scaling.get("points", [])
    if points:
        x = np.asarray([p["x"] for p in points], dtype=float)
        y = np.asarray([p["T_star"] for p in points], dtype=float)
        ax.scatter(x, y)
        if np.all(x > 0) and np.all(y > 0):
            order = np.argsort(x)
            ax.plot(x[order], y[order][0] * (x[order] / x[order][0]), linestyle="--", color="0.4")
            ax.set_xscale("log")
            ax.set_yscale("log")
    ax.set_xlabel("M*K/Delta_min^2")
    ax.set_ylabel("T*")
    save_fig(fig, out / "fig_scaling")

    fig, ax = plt.subplots(figsize=(5, 3))
    for row in master_rows:
        h_share = row["H_D"] / (abs(row["V_star"]) + abs(row["H_D"]) + 1e-12)
        best_cap = row["best_non_llm"]["capture"]
        for payload in row["llm"].values():
            ax.scatter([h_share], [payload["capture"]], marker="o", color="tab:blue")
            ax.scatter([h_share], [best_cap], marker="x", color="tab:orange")
    ax.set_xlabel("H_D share")
    ax.set_ylabel("capture")
    save_fig(fig, out / "fig_boundary")


def save_fig(fig: Any, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    fig.savefig(stem.with_suffix(".pdf"))
    import matplotlib.pyplot as plt

    plt.close(fig)


def write_master_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["| cell | M | W | encoding | H_D | V_star | best_non_llm | llm | verdict |"]
    lines.append("|---|---:|---:|---|---:|---:|---|---|---|")
    for row in rows:
        llm = ", ".join(f"{k}:{v['capture']:.3g}" for k, v in sorted(row["llm"].items()))
        best = row["best_non_llm"]
        lines.append(
            f"| {row['cell']} | {row['M']} | {row['W']} | {row['encoding']} | "
            f"{row['H_D']:.6g} | {row['V_star']:.6g} | {best['name']}:{best['capture']:.3g} | {llm} | {row['verdict']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def mean_reward(rows: list[dict[str, Any]]) -> float:
    vals = [float(r.get("total_reward")) for r in rows if r.get("total_reward") is not None and not r.get("error")]
    return float(np.mean(vals)) if vals else 0.0


def rewards_by_seed(rows: list[dict[str, Any]]) -> dict[int, float]:
    return {
        int(r.get("seed", r.get("episode", i))): float(r.get("total_reward", 0.0))
        for i, r in enumerate(rows)
        if r.get("total_reward") is not None and not r.get("error")
    }


def max_decisions(runs: list[RunData]) -> int:
    return max((len(row.get("decisions", row.get("windows", []))) for run in runs for row in run.rows), default=1)


def type_label(row: dict[str, Any]) -> str | None:
    return row.get("type") or row.get("ptype")


def departed_from_default(row: dict[str, Any], default: str) -> bool:
    decisions = row.get("decisions", row.get("windows", []))
    return any(str(d.get("tau")) != default for d in decisions)


def is_learner_arm(arm: str) -> bool:
    return arm in LEARNER_ARMS or any(arm.startswith(prefix) for prefix in LEARNER_PREFIXES)


def is_llm_arm(arm: str) -> bool:
    return arm.startswith("llm:")


def is_reference_arm(arm: str) -> bool:
    return arm.startswith("fixed:") or arm == "type_oracle"


def llm_key(arm: str) -> str:
    parts = arm.split(":")
    if len(parts) >= 3:
        return f"{parts[1]}:{parts[2]}"
    if len(parts) >= 2:
        return parts[1]
    return arm


def delta_min(payload: Any) -> float:
    vals: list[float] = []
    if isinstance(payload, dict):
        for val in payload.values():
            if isinstance(val, dict):
                vals.extend(float(x) for x in val.values() if isinstance(x, (int, float)) and x > 0)
            elif isinstance(val, (int, float)) and val > 0:
                vals.append(float(val))
    return min(vals) if vals else 1.0


def scaling_x(params: dict[str, Any], lock_metrics: dict[str, Any]) -> float:
    M = float(params.get("M", lock_metrics.get("M", 1)) or 1)
    K = float(params.get("k", lock_metrics.get("K", 1)) or 1)
    delta = float(params.get("Delta_min", lock_metrics.get("delta_min", 1.0)) or 1.0)
    return M * K / max(delta * delta, 1e-12)


def conditions(h_d: float, best: dict[str, Any], llm: dict[str, Any]) -> dict[str, bool]:
    best_llm = max((v["capture"] for v in llm.values()), default=0.0)
    return {
        "C1": bool(h_d > 0),
        "C2": bool(best_llm > 0),
        "C3": bool(best_llm >= best.get("capture", 0.0)),
    }


def prediction(h_d: float, best: dict[str, Any], llm: dict[str, Any]) -> str:
    cond = conditions(h_d, best, llm)
    return "LLM boundary favorable" if all(cond.values()) else "online learning or oracle likely needed"


def verdict(best: dict[str, Any], llm: dict[str, Any]) -> str:
    best_llm = max((v["capture"] for v in llm.values()), default=-math.inf)
    if best_llm > best.get("capture", 0.0):
        return "llm_leads"
    if math.isclose(best_llm, best.get("capture", 0.0), rel_tol=1e-9, abs_tol=1e-9):
        return "tie"
    return "non_llm_leads"


def summarize_values(vals: list[float]) -> dict[str, Any]:
    return {"n": len(vals), "mean": float(np.mean(vals)) if vals else None}


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % (2**32)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_payload(path: Path) -> Any:
    if path.suffix.lower() != ".jsonl":
        return load_json(path)
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return {"meta": {}, "rows": rows}


def parse_sandbox_cell(cell: str) -> dict[str, Any]:
    out: dict[str, Any] = {"default": "ROCK"}
    patterns = {
        "M": r"(?:^|[-_])M(?P<value>\d+)",
        "K": r"(?:^|[-_])K(?P<value>\d+)",
        "sharpness": r"(?:sharp|sharpness)(?P<value>\d+(?:\.\d+)?)",
        "reward_noise": r"(?:noise|rewardnoise)(?P<value>\d+(?:\.\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re_search(pattern, cell)
        if match:
            raw = match.group("value")
            out[key] = int(raw) if key in {"M", "K"} else float(raw)
    for enc in ("semantic", "numeric", "opaque", "permuted"):
        if enc in cell:
            out["encoding"] = enc
            break
    for sched in ("single", "per_round", "mid"):
        if sched in cell:
            out["schedule"] = sched
            break
    return out


def pooled_values(value_table: dict[str, Any]) -> dict[str, float]:
    by_tau: dict[str, list[float]] = {}
    for tau_map in value_table.values():
        if isinstance(tau_map, dict):
            for tau, val in tau_map.items():
                if isinstance(val, (int, float)):
                    by_tau.setdefault(str(tau), []).append(float(val))
    return {tau: float(np.mean(vals)) for tau, vals in by_tau.items() if vals}


def type_oracle_value(value_table: dict[str, Any]) -> float | None:
    vals = []
    for tau_map in value_table.values():
        if isinstance(tau_map, dict):
            numeric = [float(v) for v in tau_map.values() if isinstance(v, (int, float))]
            if numeric:
                vals.append(max(numeric))
    return float(np.mean(vals)) if vals else None


def canonical_domain(domain: str) -> str:
    return DOMAIN_ALIASES.get(domain, domain)


def re_search(pattern: str, text: str):
    import re

    return re.search(pattern, text)


if __name__ == "__main__":
    main()
