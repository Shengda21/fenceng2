"""Calibration-split runner for the sandbox and briefing cells.

Builds the selector exactly as `run_sandbox.py` does (its own `build_parser` and `setup`), with the
calibration seeds restricted to the fit half A, and plays the calibration episodes of the read half B: every seen type
forced on every B seed, as `sandbox.calibrate.calibration_rows` builds calibration episodes. Episodes run through
`v8lib.runner.run_deployment`; only the environment factory differs (the type is forced per episode).

Run with PYTHONPATH pointing at the library and sandbox of the arm: `code/oghp/{v8_lib,v8_sandbox}` for
the first-moves sandbox, `code/briefing/{v8_lib,v8_sandbox}` for the briefing game.

Extra options on top of run_sandbox.py's:
  --a-seeds   comma list, the fit half A (also passed as --calib-seeds to setup)
  --b-seeds   comma list, the read half B (or a subset of it, for sharding)
  --warmup    play the A episodes first (online learners), record them with phase "A"
  --ppo-on-a  build the PPO scheduler trained on the A seeds (the standard training uses a contiguous seed block)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from v8lib.runner import run_deployment

import argparse

import run_sandbox as rs
from sandbox.env import SandboxEnv
from sandbox.game import all_types

CLASSIC = not hasattr(rs, "build_parser")  # code/oghp/v8_sandbox: the first-moves sandbox runner (monolithic main)


def classic_parser() -> argparse.ArgumentParser:
    """The argument list of code/oghp/v8_sandbox/run_sandbox.py main(), verbatim."""
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
    parser.add_argument("--extra-body-json", default=None)
    return parser


def classic_setup(args):
    """The selector construction of code/oghp/v8_sandbox/run_sandbox.py main(), verbatim up to the run."""
    from sandbox.calibrate import build_lock
    from sandbox.env import SandboxConfig, make_sandbox_encoder
    from sandbox.selectors import make_selector
    from v8lib.runner import _encode_if_needed

    cfg = SandboxConfig(
        M=args.M, K=args.K, delta=args.delta, sharpness=args.sharpness, reward_noise=args.reward_noise, H=args.H, k=args.k,
        schedule=args.schedule, encoding=args.encoding, features=args.features, held_out=args.held_out,
    ).normalized()
    calib_seeds = rs.parse_range(args.calib_seeds)
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
    return cfg, calib_seeds, encoder, selector, lock


def seen_types(cfg, held):
    types = all_types(cfg.M) if CLASSIC else all_types(cfg.M, cfg.family)
    return [t for t in types if t not in set(held or ())]


def forced_factory(cfg, selector, episodes):
    """One env per episode, forced to the episode's type; the v2 prompt's per-episode seed hook is kept."""

    state = {"i": 0}
    hook = getattr(selector, "set_episode_seed", None)

    def make():
        seed, typ = episodes[state["i"]]
        state["i"] += 1
        env = SandboxEnv(cfg, forced_type=typ)
        reset = env.reset

        def reset_checked(s):
            if int(s) != int(seed):
                raise RuntimeError(f"episode order broken: runner seed {s}, planned {seed}")
            if hook is not None:
                hook(int(s))
            return reset(s)

        env.reset = reset_checked
        return env

    return make


class _ASeedEnv:
    """Maps PPO's contiguous training seeds start+ep onto the A seeds (types drawn by the seed, as the standard training does)."""

    def __init__(self, cfg, a_seeds, start):
        self.inner = SandboxEnv(cfg)
        self.a_seeds = a_seeds
        self.start = start

    def reset(self, seed):
        return self.inner.reset(self.a_seeds[int(seed) - self.start])

    def __getattr__(self, name):
        return getattr(self.inner, name)


def ppo_on_a(cfg, a_seeds):
    from sandbox.game import strategy_pool
    from v8lib.arms import PPOScheduler

    pool = strategy_pool(cfg.K) if CLASSIC else strategy_pool(cfg.K, cfg.family)
    sel = PPOScheduler(pool=pool, feature_dim=None, hidden=32, lr=3e-3)
    sel.name = "ppo"
    start = int(a_seeds[0])
    sel.train(lambda: _ASeedEnv(cfg, a_seeds, start), len(a_seeds), seed=start)
    return sel


def main() -> None:
    parser = classic_parser() if CLASSIC else rs.build_parser()
    parser.add_argument("--a-seeds", required=True)
    parser.add_argument("--b-seeds", required=True)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--ppo-on-a", action="store_true")
    args = parser.parse_args()
    strict_text = args.arm if args.arm in ("text_bow_A", "text_embed_A") else None
    if strict_text:
        args.arm = "random"  # strict fit-half text readers are built below from the A calibration records
    a_seeds = rs.parse_range(args.a_seeds)
    b_seeds = rs.parse_range(args.b_seeds)
    if set(a_seeds) & set(b_seeds):
        raise SystemExit("A and B overlap")
    args.calib_seeds = ",".join(str(s) for s in a_seeds)
    if args.ppo_on_a:
        if args.arm != "ppo":
            raise SystemExit("--ppo-on-a is for --arm ppo")
        args.arm = "random"  # setup() builds the lock and config; the PPO selector is built below
    cfg, calib_seeds, encoder, selector, lock = classic_setup(args) if CLASSIC else rs.setup(args)
    if args.ppo_on_a:
        selector = ppo_on_a(cfg, a_seeds)
    if strict_text:
        # secondary comparator: the text reader refit on the briefings of the A calibration episodes only
        from v8lib.arms import TextBowSelector, TextEmbedSelector
        from sandbox.selectors import make_embedder
        rows = lock["calibration_records"]
        texts, labels = [r["briefing"] for r in rows], [r["tau"] for r in rows]
        selector = TextBowSelector(texts, labels) if strict_text == "text_bow_A" else TextEmbedSelector(make_embedder(), texts, labels)
        selector.name = strict_text
    held = ([args.held_out] if args.held_out else []) if CLASSIC else rs.held_out_list(args)
    types = seen_types(cfg, held)
    b_eps = [(s, t) for s in b_seeds for t in types]
    a_eps = [(s, t) for s in a_seeds for t in types] if args.warmup else []
    episodes = a_eps + b_eps
    t0 = time.time()
    records = run_deployment(
        forced_factory(cfg, selector, episodes),
        selector,
        [s for s, _ in episodes],
        len(episodes),
        encoder=encoder,
        log_path=None,
        calibration_seeds=None if args.warmup else calib_seeds,
    )
    for rec, (seed, typ), phase in zip(records, episodes, ["A"] * len(a_eps) + ["B"] * len(b_eps)):
        if rec["seed"] != seed or rec["type"] != typ:
            raise RuntimeError("record does not match the planned episode")
        rec["phase"] = phase
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
    meta = {
        "argv": sys.argv[1:],
        "a_seeds": a_seeds,
        "b_seeds": b_seeds,
        "seen_types": types,
        "held_out": held,
        "episodes": len(records),
        "episodes_B": len(b_eps),
        "selector": getattr(selector, "name", None),
        "n_shot_examples": [list(x) for x in getattr(selector, "n_shot_examples", [])],
        "lock_best_response": lock.get("best_response"),
        "lock_calibration_seeds": lock.get("calibration_seeds"),
        "wall_s": time.time() - t0,
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    vals = [r["total_reward"] for r in records if r["phase"] == "B"]
    print(json.dumps({"out": str(out), "B_episodes": len(vals), "B_mean": sum(vals) / max(1, len(vals))}))


if __name__ == "__main__":
    main()
