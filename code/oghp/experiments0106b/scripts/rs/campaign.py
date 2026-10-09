#!/usr/bin/env python3
"""Campaign driver for regime-switching experiment grids."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency path
    import yaml
except Exception:  # noqa: BLE001
    yaml = None

ROOT = Path(__file__).resolve().parents[2]
RS_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = "/root/envs/oghp/bin/python"

LEARNER_PREFIXES = ("fewshot:",)
LEARNER_ARMS = {"mucb", "linucb", "lints", "ctxucb", "plastic", "ppo"}

DOMAIN_ALIASES = {"magent": "battle"}


@dataclass(frozen=True)
class RunSpec:
    plan: str
    domain: str
    tag: str
    cell_id: str
    arm: str
    run_id: str
    T: int
    out: Path
    args: list[str]
    env: dict[str, str]
    python: str

    @property
    def is_llm(self) -> bool:
        return is_llm_arm(self.arm)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan_path = resolve_plan_path(args.plan)
    plan = load_plan(plan_path)
    runs = expand_plan(
        plan,
        plan_path=plan_path,
        results_root=Path(args.results),
        python_override=args.python,
    )
    total_episodes = sum(r.T for r in runs)
    complete = [r for r in runs if is_complete(r.out, r.T)]
    pending = [r for r in runs if args.force or not is_complete(r.out, r.T)]

    print(
        f"plan={plan_path.name} runs={len(runs)} episodes={total_episodes} "
        f"complete={len(complete)} pending={len(pending)}"
    )
    if args.dry_run:
        for run in runs[: int(args.preview)]:
            status = "complete" if is_complete(run.out, run.T) else "pending"
            print(f"{status} {run.domain}/{run.tag}/{run.cell_id}/{safe_name(run.run_id)} T={run.T}")
        return

    manifest = Path(args.manifest) if args.manifest else Path(args.results) / "MANIFEST.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    non_llm = [r for r in pending if not r.is_llm]
    llm = [r for r in pending if r.is_llm]
    with futures.ThreadPoolExecutor(max_workers=max(1, int(args.procs))) as pool_a, futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.llm_procs))
    ) as pool_b:
        submitted = [pool_a.submit(run_once_with_retry, r) for r in non_llm]
        submitted += [pool_b.submit(run_once_with_retry, r) for r in llm]
        for fut in futures.as_completed(submitted):
            record = fut.result()
            with manifest.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"finished exit={record['exit_code']} attempts={record['attempts']} "
                f"{record['domain']}/{record['tag']}/{record['cell_id']}/{safe_name(record['run_id'])}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--results", default=ROOT / "results")
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--llm-procs", type=int, default=6)
    p.add_argument("--python", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--manifest")
    p.add_argument("--preview", type=int, default=8)
    return p.parse_args(argv)


def load_plan(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise SystemExit("YAML plans require PyYAML; use JSON or install pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)


def resolve_plan_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    rs_relative = RS_ROOT / path
    if rs_relative.exists():
        return rs_relative
    return path


def expand_plan(
    plan: dict[str, Any],
    plan_path: Path,
    results_root: Path,
    python_override: str | None = None,
) -> list[RunSpec]:
    models = dict(plan.get("models", {}))
    paths = dict(plan.get("paths", {}))
    runs: list[RunSpec] = []
    for idx, cell in enumerate(plan.get("cells", [])):
        fixed = dict(plan.get("fixed_args", {}).get(canonical_domain(str(cell["domain"])), {}))
        merged = {**fixed, **cell}
        cell_id = str(merged.get("cell_id") or make_cell_id(merged, idx))
        domain = canonical_domain(str(merged["domain"]))
        tag = str(cell["tag"])
        for raw_arm in merged["arms"]:
            arm, arm_params, run_id = parse_arm_spec(raw_arm)
            run_cell = {**merged, **arm_params, "domain": domain}
            T = arm_T(arm, run_cell.get("T", {}))
            suffix = ".jsonl" if domain == "sandbox" else ".json"
            out = results_root / domain / tag / cell_id / f"{safe_name(run_id)}{suffix}"
            cli_args = build_runner_args(run_cell, arm, T, out, paths=paths, plan_path=plan_path, models=models)
            env = llm_env(arm, models)
            python_exe = resolve_python(run_cell, paths, plan_path, python_override)
            runs.append(
                RunSpec(
                    plan=plan_path.name,
                    domain=domain,
                    tag=tag,
                    cell_id=cell_id,
                    arm=arm,
                    run_id=run_id,
                    T=T,
                    out=out,
                    args=cli_args,
                    env=env,
                    python=python_exe,
                )
            )
    return runs


def parse_arm_spec(raw: Any) -> tuple[str, dict[str, Any], str]:
    if isinstance(raw, str):
        return raw, {}, raw
    if not isinstance(raw, dict) or "arm" not in raw:
        raise ValueError(f"arm entries must be strings or objects with an arm key: {raw!r}")
    arm = str(raw["arm"])
    params = {k: v for k, v in raw.items() if k not in {"arm", "run_id", "label"}}
    label = str(raw.get("run_id") or raw.get("label") or arm_label(arm, params))
    return arm, params, label


def arm_label(arm: str, params: dict[str, Any]) -> str:
    bits = [arm]
    for key in ("nshot", "temperature", "llm_seed", "held_out"):
        if key in params and params[key] is not None:
            bits.append(f"{key}-{params[key]}")
    return ":".join(bits)


def build_runner_args(
    cell: dict[str, Any],
    arm: str,
    T: int,
    out: Path,
    *,
    paths: dict[str, Any],
    plan_path: Path,
    models: dict[str, Any],
) -> list[str]:
    domain = canonical_domain(str(cell["domain"]))
    if domain == "battle":
        script = resolve_path(paths.get("battle_runner"), plan_path, RS_ROOT / "run_rs_magent.py")
        args = [
            str(script),
            "--type-set",
            str(cell["type_set"]),
            "--M",
            str(cell["M"]),
            "--encoding",
            str(cell["encoding"]),
            "--arm",
            arm,
            "--schedule",
            str(cell["schedule"]),
            "--T",
            str(T),
            "--delta-scale",
            str(cell.get("delta_scale", 1.0)),
            "--lock",
            str(resolve_lock(cell, paths, plan_path, "battle")),
            "--out",
            str(out),
        ]
        if cell.get("type_hints", False):
            args.append("--type-hints")
        else:
            args.append("--no-type-hints")
        if cell.get("held_out"):
            args += ["--held-out", str(cell["held_out"])]
        for key, flag in [
            ("blue", "--blue"),
            ("k", "--k"),
            ("map_size", "--map-size"),
            ("max_cycles", "--max-cycles"),
            ("permutation_seed", "--permutation-seed"),
            ("default", "--default"),
        ]:
            if key in cell:
                args += [flag, str(cell[key])]
        if "pool" in cell:
            args += ["--pool", str(cell["pool"])]
        if cell.get("fake"):
            args.append("--fake")
        if "nshot" in cell:
            args += ["--nshot", str(cell["nshot"])]
        elif arm.startswith("fewshot:"):
            args += ["--nshot", arm.split(":", 1)[1]]
        # LLM generation parameters and the per-family extra_body (reasoning effort / thinking switch) apply to
        # the battle runner exactly as to the combined runner; without them gpt-oss returns empty content.
        for key, flag in [
            ("temperature", "--temperature"),
            ("max_tokens", "--max-tokens"),
            ("llm_seed", "--llm-seed"),
        ]:
            if key in cell and cell[key] is not None and is_llm_arm(arm):
                args += [flag, str(cell[key])]
        extra = llm_extra_body(arm, models, cell)
        if extra is not None:
            args += ["--extra-body-json", json.dumps(extra, sort_keys=True)]
        return args
    if domain == "combined":
        script = resolve_path(paths.get("combined_runner"), plan_path, RS_ROOT / "run_rs_combined.py")
        args = [
            str(script),
            "--type-set",
            str(cell["type_set"]),
            "--M",
            str(cell["M"]),
            "--encoding",
            str(cell["encoding"]),
            "--arm",
            arm,
            "--schedule",
            str(cell["schedule"]),
            "--T",
            str(T),
            "--delta-scale",
            str(cell.get("delta_scale", 1.0)),
            "--lock",
            str(resolve_lock(cell, paths, plan_path, "combined")),
            "--out",
            str(out),
        ]
        if cell.get("type_hints", False):
            args.append("--type-hints")
        else:
            args.append("--no-type-hints")
        if cell.get("held_out"):
            args += ["--held-out", str(cell["held_out"])]
        for key, flag in [
            ("blue", "--blue"),
            ("k", "--k"),
            ("map_size", "--map-size"),
            ("max_cycles", "--max-cycles"),
            ("n_melee", "--n-melee"),
            ("n_ranged", "--n-ranged"),
            ("permutation_seed", "--permutation-seed"),
            ("default", "--default"),
            ("pool", "--pool"),
        ]:
            if key in cell:
                args += [flag, str(cell[key])]
        if cell.get("fake"):
            args.append("--fake")
        if "nshot" in cell:
            args += ["--nshot", str(cell["nshot"])]
        elif arm.startswith("fewshot:"):
            args += ["--nshot", arm.split(":", 1)[1]]
        for key, flag in [
            ("temperature", "--temperature"),
            ("max_tokens", "--max-tokens"),
            ("llm_seed", "--llm-seed"),
        ]:
            if key in cell and cell[key] is not None:
                args += [flag, str(cell[key])]
        extra = llm_extra_body(arm, models, cell)
        if extra is not None:
            args += ["--extra-body-json", json.dumps(extra, sort_keys=True)]
        return args
    if domain == "sandbox":
        script = resolve_path(paths.get("sandbox_runner"), plan_path, ROOT.parent / "v8_sandbox" / "run_sandbox.py")
        args = [
            str(script),
            "--M",
            str(cell["M"]),
            "--sharpness",
            str(cell["sharpness"]),
            "--reward-noise",
            str(cell["reward_noise"]),
            "--encoding",
            str(cell["encoding"]),
            "--arm",
            arm,
            "--T",
            str(T),
            "--schedule",
            str(cell["schedule"]),
            "--out",
            str(out),
        ]
        if cell.get("held_out"):
            args += ["--held-out", str(cell["held_out"])]
        for key, flag in [
            ("K", "--K"),
            ("delta", "--delta"),
            ("H", "--H"),
            ("k", "--k"),
            ("features", "--features"),
            ("linucb_alpha", "--linucb-alpha"),
            ("seeds", "--seeds"),
            ("calib_seeds", "--calib-seeds"),
        ]:
            if key in cell:
                args += [flag, str(cell[key])]
        for key, flag in [
            ("temperature", "--temperature"),
            ("max_tokens", "--max-tokens"),
            ("llm_seed", "--llm-seed"),
        ]:
            if key in cell and cell[key] is not None and is_llm_arm(arm):
                args += [flag, str(cell[key])]
        extra = llm_extra_body(arm, models, cell)
        if extra is not None:
            args += ["--extra-body-json", json.dumps(extra, sort_keys=True)]
        return args
    raise ValueError(f"unknown domain: {domain}")


def run_once_with_retry(run: RunSpec) -> dict[str, Any]:
    attempts = []
    for attempt in range(1, 3):
        started = time.perf_counter()
        run.out.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(run.env)
        proc = subprocess.run(  # noqa: S603
            [run.python, *run.args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        attempts.append(
            {
                "attempt": attempt,
                "exit_code": proc.returncode,
                "wall_clock_s": elapsed,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            }
        )
        if proc.returncode == 0 and is_complete(run.out, run.T):
            break
    last = attempts[-1]
    return {
        "plan": run.plan,
        "domain": run.domain,
        "tag": run.tag,
        "cell_id": run.cell_id,
        "arm": run.arm,
        "run_id": run.run_id,
        "T": run.T,
        "out": str(run.out),
        "args": run.args,
        "python": run.python,
        "env": sorted(run.env),
        "attempts": len(attempts),
        "wall_clock_s": sum(float(a["wall_clock_s"]) for a in attempts),
        "exit_code": int(last["exit_code"]),
        "complete": is_complete(run.out, run.T),
        "attempt_log": attempts,
    }


def is_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            return len(rows) == int(expected_rows)
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows", data) if isinstance(data, dict) else data
        return isinstance(rows, list) and len(rows) == int(expected_rows)
    except Exception:  # noqa: BLE001
        return False


def arm_T(arm: str, T_cfg: dict[str, Any]) -> int:
    key = "learners" if is_learner_arm(arm) else "others"
    if key not in T_cfg:
        raise ValueError(f"T.{key} missing for arm {arm}")
    return int(T_cfg[key])


def is_learner_arm(arm: str) -> bool:
    return arm in LEARNER_ARMS or any(arm.startswith(prefix) for prefix in LEARNER_PREFIXES)


def is_llm_arm(arm: str) -> bool:
    return arm.startswith("llm:")


def llm_env(arm: str, models: dict[str, Any]) -> dict[str, str]:
    if not is_llm_arm(arm):
        return {}
    parts = arm.split(":")
    family = parts[1] if len(parts) > 1 else ""
    cfg = models.get(family, {})
    env: dict[str, str] = {}
    if "base_url" in cfg:
        env["LLM_API_BASE"] = str(cfg["base_url"])
    if "api_base" in cfg:
        env["LLM_API_BASE"] = str(cfg["api_base"])
    if "model" in cfg:
        env["LLM_MODEL"] = str(cfg["model"])
    return env


def llm_extra_body(arm: str, models: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any] | None:
    if not is_llm_arm(arm):
        return None
    if "extra_body" in cell:
        return cell["extra_body"]
    parts = arm.split(":")
    family = parts[1] if len(parts) > 1 else ""
    cfg = models.get(family, {})
    return cfg.get("extra_body")


def resolve_python(
    cell: dict[str, Any],
    paths: dict[str, Any],
    plan_path: Path,
    override: str | None,
) -> str:
    if override:
        return override
    domain = canonical_domain(str(cell["domain"]))
    key = f"{domain}_python"
    value = cell.get("python") or paths.get(key) or paths.get("python") or DEFAULT_PYTHON
    return str(resolve_path(value, plan_path, Path(str(value))) if value else DEFAULT_PYTHON)


def resolve_lock(cell: dict[str, Any], paths: dict[str, Any], plan_path: Path, domain: str) -> Path:
    value = cell.get("lock") or paths.get(f"{domain}_lock")
    if value is None:
        raise ValueError(f"{domain} cells require a lock path via cell.lock or paths.{domain}_lock")
    return resolve_path(value, plan_path)


def resolve_path(value: Any, plan_path: Path, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("path value is required")
        return default.resolve()
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [plan_path.parent / path, RS_ROOT / path, ROOT / path, ROOT.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / path).resolve()


def canonical_domain(domain: str) -> str:
    return DOMAIN_ALIASES.get(domain, domain)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("._") or "arm"


def make_cell_id(cell: dict[str, Any], idx: int) -> str:
    payload = {
        k: v
        for k, v in cell.items()
        if k not in {"arms", "T", "lock", "cell_id"} and not k.startswith("_")
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    label = [
        str(cell.get("domain", "cell")),
        str(cell.get("type_set", "types")),
        f"M{cell.get('M', 'na')}",
        str(cell.get("encoding", "enc")),
        str(cell.get("schedule", "sched")),
    ]
    if cell.get("held_out"):
        label.append(f"heldout-{cell['held_out']}")
    if float(cell.get("delta_scale", 1.0)) != 1.0:
        label.append(f"d{cell['delta_scale']}")
    return safe_name("-".join(label) + f"-{idx:03d}-{digest}")


if __name__ == "__main__":
    main()
