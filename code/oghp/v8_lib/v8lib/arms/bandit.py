"""Non-contextual bandit baselines.

`MUCB` follows the UCB plus change-detection recipe of Cao, Wen, Kveton and
Xie, AISTATS 2019, in compact form. `CUSUMUCB` follows the restart intuition of
Liu, Lee and Shroff, AAAI 2018.
"""

from __future__ import annotations

from collections import defaultdict, deque
from math import log, sqrt

import numpy as np

from v8lib.arms.base import Selector
from v8lib.context import Context


class _BanditBase(Selector):
    role = "online"

    def __init__(self, pool: list[str] | None = None, gamma: float = 0.0) -> None:
        super().__init__()
        self.pool = list(pool or [])
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(0)
        self.reset_run(0)

    def reset_run(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
        self.restarts = 0

    def _ensure_pool(self, ctx: Context) -> list[str]:
        if not self.pool:
            self.pool = list(ctx.pool)
        return [tau for tau in self.pool if tau in ctx.pool] or list(ctx.pool)

    def _reset_stats(self) -> None:
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
        self.t = 0
        self.restarts += 1

    def _ucb_choice(self, ctx: Context) -> str:
        pool = self._ensure_pool(ctx)
        if self.rng.random() < self.gamma:
            return str(self.rng.choice(pool))
        for tau in pool:
            if self.counts[tau] == 0:
                return tau
        total = max(1, self.t)
        scores = {
            tau: self.values[tau] + sqrt(2.0 * log(total + 1.0) / self.counts[tau])
            for tau in pool
        }
        return max(pool, key=lambda tau: (scores[tau], tau))

    def select(self, ctx: Context) -> str:
        return self._record(ctx, self._ucb_choice(ctx))

    def state_summary(self) -> dict:
        return {
            "counts": dict(self.counts),
            "values": dict(self.values),
            "restarts": self.restarts,
        }


class MUCB(_BanditBase):
    """Sliding-window change-detection UCB baseline."""

    name = "mucb"

    def __init__(
        self,
        pool: list[str] | None = None,
        window_w: int = 20,
        threshold_b: float = 1.0,
        gamma: float = 0.05,
    ) -> None:
        self.window_w = int(window_w)
        self.threshold_b = float(threshold_b)
        self.histories: dict[str, deque[float]] = {}
        super().__init__(pool, gamma)

    def reset_run(self, seed: int) -> None:
        super().reset_run(seed)
        self.histories = defaultdict(lambda: deque(maxlen=max(2, 2 * self.window_w)))

    def update(
        self, ctx: Context, tau: str, reward: float, done: bool, info: dict
    ) -> None:
        self.t += 1
        self.counts[tau] += 1
        n = self.counts[tau]
        self.values[tau] += (float(reward) - self.values[tau]) / n
        hist = self.histories[tau]
        hist.append(float(reward))
        w = self.window_w
        if len(hist) >= 2 * w:
            old = list(hist)[:w]
            recent = list(hist)[-w:]
            if abs(float(np.mean(recent)) - float(np.mean(old))) > self.threshold_b:
                self._reset_stats()
                self.histories = defaultdict(lambda: deque(maxlen=max(2, 2 * w)))


class CUSUMUCB(_BanditBase):
    """CUSUM-restart UCB baseline."""

    name = "cusum_ucb"

    def __init__(
        self,
        pool: list[str] | None = None,
        threshold_h: float = 5.0,
        drift: float = 0.0,
        gamma: float = 0.05,
    ) -> None:
        self.threshold_h = float(threshold_h)
        self.drift = float(drift)
        self.pos = defaultdict(float)
        self.neg = defaultdict(float)
        super().__init__(pool, gamma)

    def reset_run(self, seed: int) -> None:
        super().reset_run(seed)
        self.pos = defaultdict(float)
        self.neg = defaultdict(float)

    def update(
        self, ctx: Context, tau: str, reward: float, done: bool, info: dict
    ) -> None:
        self.t += 1
        self.counts[tau] += 1
        old_mean = self.values[tau]
        n = self.counts[tau]
        self.values[tau] += (float(reward) - old_mean) / n
        delta = float(reward) - old_mean - self.drift
        self.pos[tau] = max(0.0, self.pos[tau] + delta)
        self.neg[tau] = max(0.0, self.neg[tau] - delta)
        if self.pos[tau] > self.threshold_h or self.neg[tau] > self.threshold_h:
            self._reset_stats()
            self.pos = defaultdict(float)
            self.neg = defaultdict(float)
