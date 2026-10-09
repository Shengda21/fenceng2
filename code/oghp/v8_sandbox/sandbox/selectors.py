"""Selector factory and calibrated wrappers."""

from __future__ import annotations

import os

import numpy as np

from v8lib.arms import (
    ContextUCB,
    FixedSelector,
    FewShotClassifierSelector,
    LLMSelector,
    LinTS,
    LinUCB,
    MUCB,
    PLASTICPolicySelector,
    PPOScheduler,
    RandomSelector,
    ScriptedDetector,
    TypeOracleSelector,
)

from sandbox.calibrate import build_lock, model_callables
from sandbox.env import SandboxConfig
from sandbox.game import ACTIONS, beats, strategy_pool


class CalibratedPLASTICSelector(PLASTICPolicySelector):
    """Named PLASTIC selector using the library's observe-before-select phase."""

    name = "plastic"


def basic_tercile_key_fn(bins):
    bins = np.asarray(bins, dtype=float)
    dim = int(bins.shape[0])

    def key(ctx):
        x = np.asarray(ctx.features, dtype=float).ravel()
        if x.size > dim:
            x = x[-dim:]
        if x.size != dim:
            raise ValueError("context feature dimension does not match fitted basic bins")
        return tuple(int(np.searchsorted(bins[i], x[i], side="right")) for i in range(dim))

    return key


def scripted_rule(ctx) -> str:
    raw = ctx.extra.get("raw", {}) if isinstance(ctx.extra, dict) else {}
    counts = raw.get("counts", {})
    if counts:
        common = max(ACTIONS, key=lambda a: (counts.get(a, 0), a))
        return beats(common)
    x = np.asarray(ctx.features, dtype=float).ravel()
    if x.size >= 3:
        return beats(ACTIONS[int(np.argmax(x[:3]))])
    return ctx.default


STRATEGY_DOCS = {
    "ROCK": "always throw rock",
    "PAPER": "always throw paper",
    "SCISSORS": "always throw scissors",
    "MIX_RP": "throw rock or paper with equal probability each round",
    "MIX_PS": "throw paper or scissors with equal probability each round",
}


def make_selector(
    arm: str,
    config: SandboxConfig,
    lock: dict | None = None,
    linucb_alpha: float = 0.6,
    llm_kwargs: dict | None = None,
):
    cfg = config.normalized()
    pool = strategy_pool(cfg.K)
    lock = lock or build_lock(cfg, held_out=cfg.held_out)
    best_response = lock["best_response"]
    if arm == "random":
        return RandomSelector(seed=0)
    if arm.startswith("fixed:"):
        return FixedSelector(arm.split(":", 1)[1])
    if arm == "type_oracle":
        return TypeOracleSelector(best_response)
    if arm == "scripted":
        sel = ScriptedDetector(scripted_rule)
        sel.name = "scripted"
        return sel
    if arm == "mucb":
        return MUCB(pool=pool, window_w=20, threshold_b=2.0, gamma=0.05)
    if arm == "ctxucb":
        ctxucb_X = np.asarray(lock.get("ctxucb_X", lock["fewshot_X"]), dtype=float)
        probe = ContextUCB(c=1.0).fit_bins(ctxucb_X)
        sel = ContextUCB(key_fn=basic_tercile_key_fn(probe.bins), c=1.0)
        sel.name = "ctxucb"
        return sel
    if arm == "linucb":
        sel = LinUCB(alpha=linucb_alpha)
        sel.name = f"linucb:{cfg.features}"
        return sel
    if arm == "lints":
        return LinTS(alpha=0.6, v=0.5)
    if arm.startswith("fewshot:"):
        N = int(arm.split(":", 1)[1])
        sel = FewShotClassifierSelector(N=N, best_response=best_response, model="logistic", seed=0)
        sel.name = f"fewshot:{N}"
        sel.fit(np.asarray(lock["fewshot_X"], dtype=float), np.asarray(lock["fewshot_y"], dtype=str))
        return sel
    if arm == "plastic":
        sel = CalibratedPLASTICSelector(
            model_callables(lock["gaussian_models"]),
            best_response,
            value_table=lock["value_table"],
        )
        return sel
    if arm == "ppo":
        sel = PPOScheduler(pool=pool, feature_dim=None, hidden=32, lr=3e-3)
        sel.name = "ppo"
        train_seeds = [int(seed) for seed in lock.get("calibration_seeds", list(range(100, 120)))]
        train_start = min(train_seeds)
        train_count = max(train_seeds) - train_start + 1
        if train_count != len(set(train_seeds)):
            raise ValueError("PPO calibration seeds must be a contiguous offline training block")
        sel.train(lambda: __import__("sandbox.env", fromlist=["SandboxEnv"]).SandboxEnv(cfg), train_count, seed=train_start)
        return sel
    if arm.startswith("llm:"):
        if not os.environ.get("LLM_API_BASE"):
            raise RuntimeError("LLM_API_BASE must be set for LLMSelector runs")
        parts = arm.split(":")
        model = parts[1]
        mode = parts[2] if len(parts) > 2 else "single"
        kw = dict(llm_kwargs or {})
        encoding = getattr(cfg, "encoding", "semantic")
        if encoding in {"relabel", "swapdesc"}:
            # neutral labels in the prompt; `swapdesc` attaches every description to a different strategy (cyclic shift)
            kw.setdefault("label_map", {tau: f"OPTION_{chr(ord('A') + i)}" for i, tau in enumerate(pool)})
            if encoding == "swapdesc":
                shifted = pool[1:] + pool[:1]
                kw.setdefault("pool_descriptions", {tau: STRATEGY_DOCS[owner] for tau, owner in zip(pool, shifted)})
        kw.setdefault("pool_descriptions", STRATEGY_DOCS)
        return LLMSelector(model=model, mode=mode, **kw)
    raise ValueError(f"unknown arm: {arm}")
