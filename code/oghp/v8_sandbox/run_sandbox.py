"""CLI runner for E0 sandbox deployments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v8lib.runner import run_deployment

from sandbox.calibrate import build_lock
from sandbox.game import all_types
from v8lib.runner import _encode_if_needed
from sandbox.env import SandboxConfig, SandboxEnv, make_sandbox_encoder
from sandbox.selectors import make_selector


def parse_range(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=3, choices=[3, 6])
    parser.add_argument("--K", type=int, default=3, choices=[3, 5])
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=0.8)
    parser.add_argument("--reward-noise", type=float, default=0.0)
    parser.add_argument("--H", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--schedule", default="single", choices=["single", "per_round", "mid"])
    parser.add_argument("--encoding", default="semantic", choices=["semantic", "numeric", "opaque", "permuted", "relabel", "swapdesc"])
    parser.add_argument("--features", default="basic", choices=["basic", "rich"])
    parser.add_argument("--arm", default="linucb")
    parser.add_argument("--linucb-alpha", type=float, default=0.6)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--seeds", default="0-99,120-319")
    parser.add_argument("--calib-seeds", default="100-119")
    parser.add_argument("--held-out", default=None)
    parser.add_argument("--out", default="results/sandbox/run.jsonl")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--nshot", type=int, default=0, help="labelled calibration episodes per type in the LLM prompt")
    parser.add_argument("--extra-body-json", default=None, help='e.g. {"reasoning_effort":"low"} or {"chat_template_kwargs":{"enable_thinking":false}}')
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
        encoding=args.encoding,
        features=args.features,
        held_out=args.held_out,
    ).normalized()
    calib_seeds = parse_range(args.calib_seeds)
    lock = build_lock(cfg, calib_seeds=calib_seeds, held_out=args.held_out)
    llm_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.llm_seed,
        "extra_body": json.loads(args.extra_body_json) if args.extra_body_json else None,
    }
    encoder = make_sandbox_encoder(cfg)
    if args.nshot > 0 and args.arm.startswith("llm:"):
        examples = []
        for typ in [t for t in all_types(cfg.M) if t != args.held_out]:
            for seed in calib_seeds[: args.nshot]:
                ctx = _encode_if_needed(SandboxEnv(cfg, forced_type=typ).reset(int(seed)), encoder)
                examples.append((ctx.text, lock["best_response"][typ]))
        llm_kwargs["n_shot_examples"] = examples
        llm_kwargs["prompt_template"] = "{examples}\n{text}\nPool: {pool}\nDefault: {default}"
    selector = make_selector(args.arm, cfg, lock=lock, linucb_alpha=args.linucb_alpha, llm_kwargs=llm_kwargs)
    seeds = parse_range(args.seeds)
    records = run_deployment(
        lambda: SandboxEnv(cfg),
        selector,
        seeds,
        args.T,
        encoder=encoder,
        log_path=args.out,
        calibration_seeds=calib_seeds,
    )
    summary = {
        "out": str(Path(args.out)),
        "method": selector.name,
        "episodes": len(records),
        "mean_total_reward": sum(r["total_reward"] for r in records) / max(1, len(records)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
