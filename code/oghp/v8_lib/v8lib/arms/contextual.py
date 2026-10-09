"""Contextual linear bandit selectors.

`LinUCB` implements disjoint linear UCB from Li et al., WWW 2010. `LinTS`
implements linear Thompson sampling in the style of Agrawal and Goyal, ICML
2013. `LLMInitLinUCB` seeds arm estimates with prior strategy scores.
"""

from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
from math import log, sqrt

import numpy as np

from v8lib.arms.base import Selector
from v8lib.context import Context


class LinUCB(Selector):
    """Disjoint linear UCB; source: Li et al., WWW 2010."""

    name = "linucb"
    role = "online"

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.pool: list[str] = []
        self.dim = 0
        self.A: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}

    def reset_run(self, seed: int) -> None:
        self.pool = []
        self.dim = 0
        self.A = {}
        self.b = {}

    def _ensure(self, ctx: Context) -> None:
        x = np.asarray(ctx.features, dtype=float).ravel()
        if self.pool and self.dim == x.size and set(ctx.pool).issubset(self.A):
            return
        self.pool = list(ctx.pool)
        self.dim = int(x.size)
        self.A = {tau: np.eye(self.dim) for tau in self.pool}
        self.b = {tau: np.zeros(self.dim) for tau in self.pool}
        self._seed_priors(ctx)

    def _seed_priors(self, ctx: Context) -> None:
        return None

    def _score(self, tau: str, x: np.ndarray) -> float:
        inv = np.linalg.inv(self.A[tau])
        theta = inv @ self.b[tau]
        return float(theta @ x + self.alpha * np.sqrt(x @ inv @ x))

    def select(self, ctx: Context) -> str:
        self._ensure(ctx)
        x = np.asarray(ctx.features, dtype=float).ravel()
        tau = max(ctx.pool, key=lambda item: (self._score(item, x), item))
        return self._record(ctx, tau)

    def update(
        self, ctx: Context, tau: str, reward: float, done: bool, info: dict
    ) -> None:
        self._ensure(ctx)
        x = np.asarray(ctx.features, dtype=float).ravel()
        self.A[tau] += np.outer(x, x)
        self.b[tau] += float(reward) * x

    def state_summary(self) -> dict:
        return {"theta": {k: (np.linalg.solve(self.A[k], self.b[k])).tolist() for k in self.A}}


class LinTS(LinUCB):
    """Disjoint linear Thompson sampling; source: Agrawal and Goyal, ICML 2013."""

    name = "lints"

    def __init__(self, alpha: float = 1.0, v: float = 1.0) -> None:
        super().__init__(alpha=alpha)
        self.v = float(v)
        self.rng = np.random.default_rng(0)

    def reset_run(self, seed: int) -> None:
        super().reset_run(seed)
        self.rng = np.random.default_rng(seed)

    def _score(self, tau: str, x: np.ndarray) -> float:
        inv = np.linalg.inv(self.A[tau])
        mean = inv @ self.b[tau]
        cov = (self.v * self.alpha) ** 2 * inv
        theta = self.rng.multivariate_normal(mean, cov)
        return float(theta @ x)


class LLMInitLinUCB(LinUCB):
    """Prior-seeded LinUCB; source: LLM prior plus Li et al., WWW 2010."""

    name = "llm_init_linucb"

    def __init__(
        self,
        llm_prior: dict[str, float] | Callable[[Context], dict[str, float]],
        alpha: float = 1.0,
        prior_strength: float = 1.0,
    ) -> None:
        super().__init__(alpha=alpha)
        self.llm_prior = llm_prior
        self.prior_strength = float(prior_strength)

    def _seed_priors(self, ctx: Context) -> None:
        prior = self.llm_prior(ctx) if callable(self.llm_prior) else dict(self.llm_prior)
        x = np.asarray(ctx.features, dtype=float).ravel()
        denom = float(x @ x) or 1.0
        for tau, score in prior.items():
            if tau in self.b:
                self.b[tau] += self.prior_strength * float(score) * x / denom


class ContextUCB(Selector):
    """Per-discrete-context UCB1 over the strategy pool."""

    name = "context_ucb"
    role = "online"

    def __init__(self, key_fn: Callable[[Context], object] | None = None, c: float = 1.0) -> None:
        super().__init__()
        self.key_fn = key_fn
        self.c = float(c)
        self.pool: list[str] = []
        self.bins: np.ndarray | None = None
        self.counts = defaultdict(lambda: defaultdict(int))
        self.values = defaultdict(lambda: defaultdict(float))
        self.totals = defaultdict(int)

    def reset_run(self, seed: int) -> None:
        self.counts = defaultdict(lambda: defaultdict(int))
        self.values = defaultdict(lambda: defaultdict(float))
        self.totals = defaultdict(int)

    def fit_bins(self, X) -> "ContextUCB":
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("X must contain at least one calibration row")
        if not np.all(np.isfinite(arr)):
            raise ValueError("X must be finite")
        self.bins = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0], axis=0).T
        return self

    def _key(self, ctx: Context):
        if self.key_fn is not None:
            return self.key_fn(ctx)
        if self.bins is None:
            raise ValueError("ContextUCB requires key_fn or fitted tercile bins")
        x = np.asarray(ctx.features, dtype=float).ravel()
        if x.size != self.bins.shape[0]:
            raise ValueError("context feature dimension does not match fitted bins")
        return tuple(int(np.searchsorted(self.bins[i], x[i], side="right")) for i in range(x.size))

    def _ensure_pool(self, ctx: Context) -> list[str]:
        if not self.pool:
            self.pool = list(ctx.pool)
        return [tau for tau in self.pool if tau in ctx.pool] or list(ctx.pool)

    def select(self, ctx: Context) -> str:
        key = self._key(ctx)
        pool = self._ensure_pool(ctx)
        for tau in pool:
            if self.counts[key][tau] == 0:
                return self._record(ctx, tau, extra={"context_key": key})
        total = max(1, self.totals[key])
        scores = {
            tau: self.values[key][tau] + self.c * sqrt(2.0 * log(total + 1.0) / self.counts[key][tau])
            for tau in pool
        }
        tau = max(pool, key=lambda item: (scores[item], item))
        return self._record(ctx, tau, extra={"context_key": key, "scores": scores})

    def update(
        self, ctx: Context, tau: str, reward: float, done: bool, info: dict
    ) -> None:
        key = self._key(ctx)
        self.totals[key] += 1
        self.counts[key][tau] += 1
        n = self.counts[key][tau]
        self.values[key][tau] += (float(reward) - self.values[key][tau]) / n

    def state_summary(self) -> dict:
        return {
            "contexts": {
                repr(key): {"counts": dict(self.counts[key]), "values": dict(self.values[key])}
                for key in self.counts
            }
        }
