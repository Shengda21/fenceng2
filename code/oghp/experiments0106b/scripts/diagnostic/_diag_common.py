#!/usr/bin/env python3
"""Shared utilities for the default-gap diagnostic experiments.

This module backs the following experiments:
  - Path-2 ablation at n=100 with paired seeds + random-router + best-constant-tau.
  - constant-tau / atomic-pool sweep.
  - best-constant / per-seed-oracle upper reference.
  - prompt/temperature sensitivity.

It depends on nothing in ``src/core`` and adds no model code; it
only provides: a temperature-controllable LLM call, Wilson intervals, paired
bootstrap deltas, and JSON I/O. All other experiment scripts import from here.

Design constraints:
  - CPU/GPU agnostic (pure orchestration; no torch).
  - Linux-friendly (cloud server target).
  - Reproducible: every row records its ``seed`` so methods can be paired.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# ============================================================
# LLM call with controllable temperature
# ------------------------------------------------------------
# A self-contained copy of scripts.hier_v2.call_llm_v2 that exposes
# ``temperature`` as an argument instead of hard-coding 0.5. hier_v2.call_llm_v2
# is left unmodified so the main pipeline is unaffected; this variant is only
# used by the diagnostic sweeps.
# ============================================================
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "30"))


def call_llm_temp(prompt: str, default: str = "", max_tokens: int = 200,
                  temperature: float = 0.5) -> str:
    """Temperature-controllable LLM call (mirrors hier_v2.call_llm_v2 otherwise).

    Args:
        prompt: User prompt text.
        default: Returned when the API key is missing or all retries fail.
        max_tokens: Generation cap (overridable by ``LLM_MAX_TOKENS`` env).
        temperature: Sampling temperature (the knob swept in the temperature-sensitivity experiment).

    Returns:
        Raw model text (stripped); domain-specific parsing is done by callers.
    """
    import requests  # local import keeps the module importable without requests

    if not LLM_API_KEY:
        return default
    body: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", str(max_tokens))),
    }
    eff = os.environ.get("LLM_REASONING_EFFORT")
    if eff or "nvidia" in LLM_API_BASE.lower():
        body["reasoning_effort"] = eff or "low"
    backoff = [0, 5, 15, 30]  # NVIDIA NIM ~2 RPM strict limiter
    last_err: Optional[Exception] = None
    for wait in backoff:
        if wait:
            time.sleep(wait)
        try:
            resp = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}",
                         "Content-Type": "application/json"},
                json=body,
                timeout=LLM_TIMEOUT,
            )
            if resp.status_code == 429:
                last_err = RuntimeError(f"429 {resp.text[:80]}")
                continue
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content")
            if not content:
                content = msg.get("reasoning_content") or msg.get("reasoning") or ""
            return (content or default).strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    print(f"  [LLM-FAIL] {last_err}", flush=True)
    return default


# ============================================================
# Statistics
# ============================================================
def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        wins: Number of successes.
        n: Number of trials.
        z: Normal quantile (1.96 for 95%).

    Returns:
        (low, high) on the [0, 1] scale; (0, 0) when n == 0.
    """
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_delta(
    a_by_seed: dict[int, float],
    b_by_seed: dict[int, float],
    n_boot: int = 10000,
    seed: int = 12345,
) -> dict[str, float]:
    """Paired bootstrap of the mean delta (a - b) over shared seeds.

    Only seeds present in BOTH method dicts are paired, which is the whole point
    of running every method on the same seed base.

    Args:
        a_by_seed: {seed: metric} for method A (e.g. LLM router).
        b_by_seed: {seed: metric} for method B (e.g. fixed default).
        n_boot: Bootstrap resamples.
        seed: RNG seed for the bootstrap itself (reproducibility).

    Returns:
        dict with n_paired, delta_mean, ci_low, ci_high (95%), and frac_pos.
    """
    shared = sorted(set(a_by_seed) & set(b_by_seed))
    if not shared:
        return {"n_paired": 0, "delta_mean": 0.0, "ci_low": 0.0,
                "ci_high": 0.0, "frac_pos": 0.0}
    diffs = [a_by_seed[s] - b_by_seed[s] for s in shared]
    n = len(diffs)
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    frac_pos = sum(1 for d in diffs if d > 0) / n
    return {"n_paired": n, "delta_mean": point, "ci_low": lo,
            "ci_high": hi, "frac_pos": frac_pos}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Win-rate / mean-reward summary over a list of per-episode rows."""
    rows = [r for r in rows if not r.get("error")]
    n = len(rows)
    if n == 0:
        return {"n": 0, "wins": 0, "win_rate": 0.0, "mean_reward": 0.0,
                "ci_low": 0.0, "ci_high": 0.0}
    wins = sum(1 for r in rows if r.get("success"))
    mean_r = sum(float(r.get("total_reward", 0.0)) for r in rows) / n
    lo, hi = wilson_ci(wins, n)
    return {"n": n, "wins": wins, "win_rate": wins / n, "mean_reward": mean_r,
            "ci_low": lo, "ci_high": hi}


# ============================================================
# Router RNG (independent of env seeding so paired seeds stay paired)
# ============================================================
def router_rng(domain: str, cell: str, method: str, episode: int) -> random.Random:
    """Deterministic RNG for random-router choices.

    Keyed by (domain, cell, method, episode) so the random router is
    reproducible yet independent of the environment's own seeding — the env is
    seeded by ``episode`` identically across methods, preserving pairing.
    """
    key = f"{domain}|{cell}|{method}|{episode}"
    return random.Random(abs(hash(key)) % (2 ** 31))


# ============================================================
# JSON I/O
# ============================================================
def save_rows(rows: list[dict[str, Any]], out_path: Path,
              meta: Optional[dict[str, Any]] = None) -> None:
    """Write rows (+ optional run metadata) to JSON, creating parent dirs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Any = rows if meta is None else {"meta": meta, "rows": rows}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read rows from JSON written by save_rows (list or {meta, rows})."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    return data


def by_seed(rows: Sequence[dict[str, Any]], field: str) -> dict[int, float]:
    """Map seed -> metric for pairing; uses 'seed' if present else 'episode'."""
    out: dict[int, float] = {}
    for r in rows:
        if r.get("error"):
            continue
        s = int(r.get("seed", r.get("episode", -1)))
        if s < 0:
            continue
        out[s] = float(r.get(field, 0.0))
    return out
