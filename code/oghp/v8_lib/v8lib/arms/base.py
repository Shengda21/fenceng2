"""Base selector protocol and provenance helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from v8lib.context import Context


class Selector(ABC):
    """Abstract base class for all high-level strategy selectors."""

    name: str = "selector"
    role: str = "online"
    reads_type_label: bool = False

    def __init__(self) -> None:
        self.last_provenance: dict[str, Any] | None = None

    def reset_run(self, seed: int) -> None:
        """Reset state at the start of a deployment run."""

    def reset_episode(self, ctx0: Context) -> None:
        """Reset state at the start of one episode."""

    @abstractmethod
    def select(self, ctx: Context) -> str:
        """Return a strategy from `ctx.pool`."""

    def update(
        self, ctx: Context, tau: str, reward: float, done: bool, info: dict
    ) -> None:
        """Receive reward feedback for the previously selected strategy."""

    def state_summary(self) -> dict:
        """Return compact JSON-serializable selector state."""

        return {}

    def _record(
        self,
        ctx: Context,
        tau: str,
        *,
        fallback: bool = False,
        latency_s: float = 0.0,
        extra: dict | None = None,
    ) -> str:
        """Store provenance and return an in-pool strategy with fallback."""

        in_pool = tau in ctx.pool
        chosen = tau if in_pool else ctx.default
        used_fallback = fallback or not in_pool
        self.last_provenance = {
            "arm": self.name,
            "episode": ctx.episode_index,
            "window": ctx.window_index,
            "tau": chosen,
            "in_pool": in_pool,
            "fallback": used_fallback,
            "latency_s": latency_s,
            "extra": extra or {},
        }
        return chosen

    def _timed_record(self, ctx: Context, start: float, tau: str, **kwargs: Any) -> str:
        return self._record(ctx, tau, latency_s=perf_counter() - start, **kwargs)
