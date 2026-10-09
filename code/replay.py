"""Exact offline replay helpers for the offline extras.

The replay model uses the single-decision invariant: once a selector
chooses strategy tau on deployment seed s, its episode return is exactly the
fixed-arm return R[s][tau] from the reference cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V8_LIB = ROOT / "code" / "oghp" / "v8_lib"
if str(V8_LIB) not in sys.path:
    sys.path.insert(0, str(V8_LIB))

from v8lib.context import Context  # noqa: E402


@dataclass
class ReplayCell:
    """A fixed-return replay table plus optional per-seed observations."""

    returns: dict[int, dict[str, float]]
    types: dict[int, str | None] = field(default_factory=dict)
    features: dict[int, list[float]] = field(default_factory=dict)
    texts: dict[int, str] = field(default_factory=dict)
    extras: dict[int, dict[str, Any]] = field(default_factory=dict)
    pool: list[str] = field(default_factory=list)
    default: str | None = None

    def seeds(self) -> list[int]:
        return sorted(self.returns)


def load_rows(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a result file, returning (meta, rows), with UTF-8 explicitly."""

    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return {}, [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {}, rows
    raw = json.load(path.open(encoding="utf-8"))
    if isinstance(raw, dict):
        return dict(raw.get("meta") or {}), list(raw.get("rows") or raw.get("episodes") or [])
    return {}, list(raw)


def arm_tau_from_path(path: str | Path) -> str | None:
    """Return the fixed tau encoded in fixed__TAU result filenames."""

    stem = Path(path).stem
    if stem.startswith("fixed__"):
        return stem[len("fixed__") :]
    return None


def row_tau(row: dict[str, Any]) -> str | None:
    """Extract the chosen strategy from any raw row shape used here."""

    for key in ("decisions", "windows", "provenance"):
        seq = row.get(key) or []
        if seq:
            tau = seq[0].get("tau")
            if tau is not None:
                return str(tau)
    tau = row.get("tau")
    return None if tau is None else str(tau)


def row_value(row: dict[str, Any]) -> float | None:
    value = row.get("total_reward", row.get("value"))
    return None if value is None else float(value)


def load_fixed_table(
    fixed_paths: Iterable[str | Path],
    *,
    feature_rows: Iterable[dict[str, Any]] | None = None,
    default: str | None = None,
) -> ReplayCell:
    """Load fixed-arm result files into R[seed][tau] and per-seed metadata."""

    returns: dict[int, dict[str, float]] = {}
    types: dict[int, str | None] = {}
    pool: list[str] = []
    for path in fixed_paths:
        tau = arm_tau_from_path(path)
        if tau is None:
            continue
        pool.append(tau)
        _, rows = load_rows(path)
        for row in rows:
            value = row_value(row)
            if value is None:
                continue
            seed = int(row["seed"])
            returns.setdefault(seed, {})[tau] = value
            types.setdefault(seed, row.get("type") or row.get("type1"))

    features: dict[int, list[float]] = {}
    texts: dict[int, str] = {}
    extras: dict[int, dict[str, Any]] = {}
    for row in feature_rows or []:
        if "seed" not in row:
            continue
        seed = int(row["seed"])
        if row.get("features") is not None:
            features[seed] = [float(x) for x in np.asarray(row["features"], dtype=float).ravel()]
        if row.get("text") is not None:
            texts[seed] = str(row["text"])
        extras[seed] = dict(row.get("raw") or {})
        types.setdefault(seed, row.get("type") or row.get("type1"))

    return ReplayCell(
        returns=returns,
        types=types,
        features=features,
        texts=texts,
        extras=extras,
        pool=pool,
        default=default or (pool[0] if pool else None),
    )


def replay_selector(
    cell: ReplayCell,
    selector: Any,
    *,
    seeds: Iterable[int] | None = None,
    T: int | None = None,
    feature_fn: Callable[[int, ReplayCell], Iterable[float]] | None = None,
    text_fn: Callable[[int, ReplayCell], str] | None = None,
    extra_fn: Callable[[int, ReplayCell], dict[str, Any]] | None = None,
    reset_seed: int = 0,
) -> list[dict[str, Any]]:
    """Run selector.select/update over deployment seeds and fixed returns.

    The selector may implement either the v8lib update signature
    update(ctx, tau, reward, done, info) or the smaller
    update(ctx, tau, reward) protocol.
    """

    run_seeds = list(seeds if seeds is not None else cell.seeds())
    run_seeds = run_seeds[:T] if T is not None else run_seeds
    if hasattr(selector, "reset_run"):
        selector.reset_run(int(reset_seed))

    out: list[dict[str, Any]] = []
    pool = list(cell.pool)
    default = cell.default or (pool[0] if pool else "")
    for episode_index, seed in enumerate(run_seeds):
        if seed not in cell.returns:
            continue
        features = (
            list(feature_fn(seed, cell))
            if feature_fn is not None
            else list(cell.features.get(seed, []))
        )
        if not features:
            features = [1.0]
        text = text_fn(seed, cell) if text_fn is not None else cell.texts.get(seed, "")
        extra = extra_fn(seed, cell) if extra_fn is not None else dict(cell.extras.get(seed, {}))
        ctx = Context(
            pool=pool,
            features=np.asarray(features, dtype=float),
            text=str(text),
            episode_index=episode_index,
            window_index=0,
            default=default,
            type_label=cell.types.get(seed),
            extra=extra,
        )
        if hasattr(selector, "reset_episode"):
            selector.reset_episode(ctx)
        tau = str(selector.select(ctx))
        if tau not in cell.returns[seed]:
            tau = default
        reward = float(cell.returns[seed][tau])
        try:
            selector.update(ctx, tau, reward, True, {"seed": seed, "type": cell.types.get(seed)})
        except TypeError:
            selector.update(ctx, tau, reward)
        prov = dict(getattr(selector, "last_provenance", None) or {})
        out.append(
            {
                "seed": int(seed),
                "type": cell.types.get(seed),
                "tau": tau,
                "value": reward,
                "total_reward": reward,
                "success": bool(reward > 0.0),
                "fallback": bool(prov.get("fallback", False)),
                "provenance": prov,
            }
        )
    return out


class SignatureLookupSelector:
    """Choose a precomputed tau for each one-hot signature."""

    name = "signature_lookup"

    def __init__(self, signature_to_tau: dict[str, str], index_to_signature: list[str], default: str):
        self.signature_to_tau = dict(signature_to_tau)
        self.index_to_signature = list(index_to_signature)
        self.default = default
        self.last_provenance: dict[str, Any] | None = None

    def reset_run(self, seed: int) -> None:
        return None

    def select(self, ctx: Context) -> str:
        x = np.asarray(ctx.features, dtype=float).ravel()
        idx = int(np.argmax(x)) if x.size else -1
        sig = self.index_to_signature[idx] if 0 <= idx < len(self.index_to_signature) else ""
        tau = self.signature_to_tau.get(sig, self.default)
        if tau not in ctx.pool:
            tau = ctx.default
        self.last_provenance = {
            "arm": self.name,
            "episode": ctx.episode_index,
            "window": ctx.window_index,
            "tau": tau,
            "in_pool": tau in ctx.pool,
            "fallback": sig not in self.signature_to_tau,
            "extra": {"signature": sig},
        }
        return tau

    def update(self, ctx: Context, tau: str, reward: float, *args: Any) -> None:
        return None


class SignatureUCB1Selector:
    """A tabular UCB1 learner with an independent bandit per signature."""

    name = "signature_ucb1"

    def __init__(self, c: float = 1.0):
        self.c = float(c)
        self.counts: dict[tuple[float, ...], dict[str, int]] = {}
        self.values: dict[tuple[float, ...], dict[str, float]] = {}
        self.totals: dict[tuple[float, ...], int] = {}
        self.last_key: tuple[float, ...] | None = None
        self.last_provenance: dict[str, Any] | None = None

    def reset_run(self, seed: int) -> None:
        self.counts = {}
        self.values = {}
        self.totals = {}
        self.last_key = None

    def _key(self, ctx: Context) -> tuple[float, ...]:
        return tuple(float(x) for x in np.asarray(ctx.features, dtype=float).ravel())

    def select(self, ctx: Context) -> str:
        key = self._key(ctx)
        self.last_key = key
        self.counts.setdefault(key, {})
        self.values.setdefault(key, {})
        self.totals.setdefault(key, 0)
        for tau in ctx.pool:
            if self.counts[key].get(tau, 0) == 0:
                choice = tau
                break
        else:
            total = max(1, self.totals[key])
            scores = {
                tau: self.values[key][tau]
                + self.c * np.sqrt(2.0 * np.log(total + 1.0) / self.counts[key][tau])
                for tau in ctx.pool
            }
            choice = max(ctx.pool, key=lambda item: (scores[item], item))
        self.last_provenance = {
            "arm": self.name,
            "episode": ctx.episode_index,
            "window": ctx.window_index,
            "tau": choice,
            "in_pool": True,
            "fallback": False,
            "extra": {"signature_key": key},
        }
        return choice

    def update(self, ctx: Context, tau: str, reward: float, *args: Any) -> None:
        key = self.last_key if self.last_key is not None else self._key(ctx)
        self.counts.setdefault(key, {})
        self.values.setdefault(key, {})
        self.totals[key] = self.totals.get(key, 0) + 1
        self.counts[key][tau] = self.counts[key].get(tau, 0) + 1
        n = self.counts[key][tau]
        old = self.values[key].get(tau, 0.0)
        self.values[key][tau] = old + (float(reward) - old) / n


def signature_from_prompt(prompt: str) -> str | None:
    match = re.search(r"opponent signature:\s*([A-Za-z0-9_:-]+)", prompt)
    return match.group(1) if match else None
