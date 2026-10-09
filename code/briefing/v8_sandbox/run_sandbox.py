"""CLI runner for E0 sandbox deployments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v8lib.runner import run_deployment

from sandbox.briefing import resolve_held_out_set
from sandbox.calibrate import build_lock
from sandbox.game import all_types
from v8lib.runner import _encode_if_needed
from sandbox.env import SandboxConfig, SandboxEnv, make_sandbox_encoder
from sandbox.prompt_v2 import PROMPT_VERSIONS, nshot_rows
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="classic", choices=["classic", "traits"])
    parser.add_argument("--M", type=int, default=3, choices=[3, 6, 9, 18])
    parser.add_argument("--K", type=int, default=3, choices=[3, 5, 10])
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=0.8)
    parser.add_argument("--reward-noise", type=float, default=0.0)
    parser.add_argument("--H", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--schedule", default="single", choices=["single", "per_round", "mid"])
    parser.add_argument(
        "--encoding",
        default="semantic",
        choices=["semantic", "numeric", "opaque", "permuted", "relabel", "swapdesc", "briefing", "briefing_prefix"],
    )
    parser.add_argument("--features", default="basic", choices=["basic", "rich"])
    parser.add_argument("--arm", default="linucb")
    parser.add_argument("--linucb-alpha", type=float, default=0.6)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--seeds", default="0-99,120-319")
    parser.add_argument("--calib-seeds", default="100-119")
    parser.add_argument("--held-out", default=None)
    parser.add_argument("--held-out-set", default=None, help="traits family: react (primary; react2 at M=9, react4 at M=18), diag (diag3, diag6), a set name, auto (= react) or none")
    parser.add_argument("--wording", default="sameword", choices=["sameword", "newword"])
    parser.add_argument("--deploy-types", default="all", choices=["all", "heldout"], help="heldout: deployment seeds draw only the held-out types")
    parser.add_argument("--out", default="results/sandbox/run.jsonl")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--nshot", type=int, default=0, help="labelled calibration episodes per type in the LLM prompt")
    parser.add_argument("--extra-body-json", default=None, help='e.g. {"reasoning_effort":"low"} or {"chat_template_kwargs":{"enable_thinking":false}}')
    parser.add_argument("--prompt-version", default="v1", choices=list(PROMPT_VERSIONS),
                        help="v1: the bare template (default). v2: task stated, first-person docs, "
                             "'Final:' parser, every call logged; under v2 --nshot R means R renderings per seen type (1-3)")
    parser.add_argument("--pool-order", default="fixed", choices=["fixed", "shuffled"],
                        help="v2 only: shuffled permutes the pool listing per episode, seeded from the episode seed")
    parser.add_argument("--example-order", default="fixed", choices=["fixed", "shuffled"],
                        help="v2 only: shuffled permutes the N-shot examples per episode, seeded from the episode seed")
    parser.add_argument("--timeout-s", type=float, default=None,
                        help="LLM client timeout per call in seconds (default: the selector's 60 s); long reasoning arms need more")
    return parser


def held_out_list(args) -> list[str]:
    if args.family == "traits":
        if args.held_out:
            raise SystemExit("traits family takes --held-out-set, not --held-out")
        return resolve_held_out_set(args.held_out_set or "auto", args.M)
    if args.held_out_set:
        raise SystemExit("--held-out-set is defined for the traits family only")
    return [args.held_out] if args.held_out else []


def check_family(args) -> None:
    if args.family == "classic":
        if args.M not in (3, 6) or args.K not in (3, 5):
            raise SystemExit("classic family needs M in {3, 6} and K in {3, 5}")
        if args.encoding in ("briefing", "briefing_prefix"):
            raise SystemExit("briefing encodings need the traits family")
    elif args.M not in (9, 18) or args.K not in (5, 10):
        raise SystemExit("traits family needs M in {9, 18} and K in {5, 10}")


def env_factory(cfg, selector):
    """Episode env factory. A selector with `set_episode_seed` (prompt v2) is told each episode's seed before the reset;
    every other selector gets the plain factory."""

    hook = getattr(selector, "set_episode_seed", None)
    if hook is None:
        return lambda: SandboxEnv(cfg)

    def make():
        env = SandboxEnv(cfg)
        reset = env.reset

        def reset_with_seed(seed):
            hook(int(seed))
            return reset(seed)

        env.reset = reset_with_seed
        return env

    return make


def setup(args, client=None):
    check_family(args)
    v2 = getattr(args, "prompt_version", "v1") == "v2"
    if not v2 and (getattr(args, "pool_order", "fixed") != "fixed" or getattr(args, "example_order", "fixed") != "fixed"):
        raise SystemExit("--pool-order and --example-order need --prompt-version v2")
    held = held_out_list(args)
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
        held_out=args.held_out if args.family == "classic" else tuple(held),
        family=args.family,
        wording=args.wording,
        deploy_types=args.deploy_types,
    ).normalized()
    calib_seeds = parse_range(args.calib_seeds)
    if args.family == "classic":
        lock = build_lock(cfg, calib_seeds=calib_seeds, held_out=args.held_out)
    else:
        lock = build_lock(cfg, calib_seeds=calib_seeds, held_out=held, path=None)
    llm_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.llm_seed,
        "extra_body": json.loads(args.extra_body_json) if args.extra_body_json else None,
    }
    if client is not None:
        llm_kwargs["client"] = client
    if getattr(args, "timeout_s", None) is not None:
        llm_kwargs["timeout_s"] = args.timeout_s
    if v2:
        llm_kwargs.update(prompt_version="v2", pool_order=args.pool_order, example_order=args.example_order)
    encoder = make_sandbox_encoder(cfg)
    if v2 and args.nshot > 0 and args.arm.startswith("llm:"):
        if cfg.family != "traits" or cfg.encoding != "briefing":
            raise SystemExit("v2 N-shot examples are defined for the traits family under the briefing encoding")
        llm_kwargs["n_shot_examples"] = [(r["text"], r["tau"]) for r in nshot_rows(cfg, lock["best_response"], held, args.nshot)]
    elif args.nshot > 0 and args.arm.startswith("llm:"):
        examples = []
        for typ in [t for t in all_types(cfg.M, cfg.family) if t not in held]:
            for seed in calib_seeds[: args.nshot]:
                ctx = _encode_if_needed(SandboxEnv(cfg, forced_type=typ).reset(int(seed)), encoder)
                examples.append((ctx.text, lock["best_response"][typ]))
        llm_kwargs["n_shot_examples"] = examples
        llm_kwargs["prompt_template"] = "{examples}\n{text}\nPool: {pool}\nDefault: {default}"
    selector = make_selector(args.arm, cfg, lock=lock, linucb_alpha=args.linucb_alpha, llm_kwargs=llm_kwargs)
    return cfg, calib_seeds, encoder, selector, lock


def main() -> None:
    args = build_parser().parse_args()
    cfg, calib_seeds, encoder, selector, _ = setup(args)
    seeds = parse_range(args.seeds)
    records = run_deployment(
        env_factory(cfg, selector),
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
