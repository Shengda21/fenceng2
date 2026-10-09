"""Small analyzer for sandbox JSONL deployment records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v8lib.analysis import capture, capture_curve, crossover

from closed_form import compute_closed_forms
from sandbox.env import SandboxConfig


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_capture_json(text: str | None):
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{") or stripped.startswith("[") or stripped[0].isdigit():
        return json.loads(stripped)
    candidate = Path(text)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(text)


def resolve_target_capture(target_capture: float, llm_captures, family: str | None = None) -> float:
    if llm_captures is None:
        return float(target_capture)
    if isinstance(llm_captures, (int, float)):
        return float(llm_captures)
    if not isinstance(llm_captures, dict):
        raise ValueError("--llm-captures must be a number or JSON object")
    if family and family in llm_captures:
        return float(llm_captures[family])
    if "default" in llm_captures:
        return float(llm_captures["default"])
    if len(llm_captures) == 1:
        return float(next(iter(llm_captures.values())))
    return float(target_capture)


def t_star_label(curve: np.ndarray, target: float) -> str | int:
    if curve.size == 0:
        return "never"
    hits = np.flatnonzero(curve >= float(target))
    if hits.size == 0:
        return "never"
    first = int(hits[0] + 1)
    return "<=1" if first <= 1 else first


def crossover_label(result: dict) -> str | int:
    point = result.get("T_star")
    if point is None:
        return "never"
    point = int(point)
    return "<=1" if point <= 1 else point


def summarize(path: str | Path, closed: dict, target_capture: float = 0.8) -> dict:
    rows = read_jsonl(path)
    rewards = np.asarray([row["total_reward"] for row in rows], dtype=float)
    curve = capture_curve(rewards, closed["V_star"], closed["H_D"])
    final_capture = capture(float(np.mean(rewards)), closed["V_star"], closed["H_D"]) if rewards.size else capture(
        float("nan"), closed["V_star"], closed["H_D"]
    )
    cross = crossover(curve, target_capture, B=200, seed=0)
    return {
        "path": str(path),
        "method": rows[0]["method"] if rows else "none",
        "T": int(len(rows)),
        "mean": float(np.mean(rewards)) if rewards.size else float("nan"),
        "capture": float(final_capture),
        "capture_status": getattr(final_capture, "status", "ok"),
        "target_capture": float(target_capture),
        "T_star": crossover_label(cross),
        "T_star_raw": cross["T_star"],
        "T_star_status": cross["status"],
        "T_star_censored": bool(cross["censored"]),
        "T_star_ci": cross["ci"],
        "crossing_probability": float(cross["crossing_probability"]),
        "final_capture": float(final_capture),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--H", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--schedule", default="single")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--sharpness", type=float, default=0.8)
    parser.add_argument("--reward-noise", type=float, default=0.0)
    parser.add_argument("--target-capture", type=float, default=0.8)
    parser.add_argument("--llm-captures", default=None)
    parser.add_argument("--family", default=None)
    args = parser.parse_args()
    cfg = SandboxConfig(
        M=args.M,
        K=args.K,
        delta=args.delta,
        sharpness=args.sharpness,
        reward_noise=args.reward_noise,
        H=args.H,
        k=args.k,
        schedule=args.schedule,
    )
    closed = compute_closed_forms(cfg, samples=args.samples)
    target = resolve_target_capture(
        args.target_capture,
        parse_capture_json(args.llm_captures),
        family=args.family,
    )
    print(json.dumps([summarize(path, closed, target_capture=target) for path in args.paths], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
