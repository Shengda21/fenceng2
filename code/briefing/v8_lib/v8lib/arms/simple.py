"""Simple selector arms for floors, instruments, and scripted detectors."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from v8lib.arms.base import Selector
from v8lib.context import Context


class RandomSelector(Selector):
    """Uniform random floor selector; source: standard random policy baseline."""

    name = "random"
    role = "floor"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_run(self, seed: int) -> None:
        self.rng = np.random.default_rng(self.seed if self.seed is not None else seed)

    def select(self, ctx: Context) -> str:
        tau = str(self.rng.choice(ctx.pool))
        return self._record(ctx, tau)


class FixedSelector(Selector):
    """Fixed-strategy instrument; source: Probe 1 fixed-strategy sweep."""

    role = "instrument"

    def __init__(self, tau: str) -> None:
        super().__init__()
        self.tau = tau
        self.name = f"fixed:{tau}"

    def select(self, ctx: Context) -> str:
        return self._record(ctx, self.tau)


class TypeOracleSelector(Selector):
    """Hidden-type oracle instrument; source: paper oracle best-response probe."""

    name = "type_oracle"
    role = "instrument"
    reads_type_label = True

    def __init__(self, best_response: dict[str, str]) -> None:
        super().__init__()
        self.best_response = dict(best_response)

    def select(self, ctx: Context) -> str:
        tau = self.best_response.get(ctx.type_label)
        missing = tau is None
        return self._record(
            ctx,
            ctx.default if missing else tau,
            fallback=missing,
            extra={"type_label": ctx.type_label, "best_response_missing": missing},
        )


class ScriptedDetector(Selector):
    """Rule threshold detector; source: scripted zero-shot baseline."""

    name = "scripted_detector"
    role = "zero_shot"

    def __init__(self, rules: Callable[[Context], str]) -> None:
        super().__init__()
        self.rules = rules

    def select(self, ctx: Context) -> str:
        return self._record(ctx, self.rules(ctx))
