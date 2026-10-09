"""Closed-library Bayesian type inference selector.

This is a compact PLASTIC-Policy/HBA-style baseline inspired by Barrett and
Stone 2017 and Albrecht and Ramamoorthy 2013.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from v8lib.arms.base import Selector
from v8lib.context import Context


class PLASTICPolicySelector(Selector):
    """Bayesian type posterior; source: PLASTIC-Policy/HBA."""

    name = "plastic_policy"
    role = "zero_shot"

    def __init__(
        self,
        type_models: dict[str, Callable[[Context], float]],
        best_response: dict[str, str],
        prior: dict[str, float] | None = None,
        value_table: dict[str, dict[str, float]] | None = None,
    ) -> None:
        super().__init__()
        self.type_models = dict(type_models)
        self.best_response = dict(best_response)
        self.value_table = value_table
        labels = list(self.type_models)
        if prior is None:
            prior = {label: 1.0 / len(labels) for label in labels}
        self.base_logp = {
            label: float(np.log(max(prior.get(label, 0.0), 1e-300))) for label in labels
        }
        self.logp = dict(self.base_logp)
        self._observed: set[tuple[int, int]] = set()

    def reset_episode(self, ctx0: Context) -> None:
        self.logp = dict(self.base_logp)
        self._observed = set()
        self.observe(ctx0)

    def observe(self, ctx: Context) -> None:
        """Score the current observation at most once before selection."""

        key = (int(ctx.episode_index), int(ctx.window_index))
        if key in self._observed:
            return
        for typ, model in self.type_models.items():
            self.logp[typ] += float(model(ctx))
        self._observed.add(key)

    def _posterior(self) -> dict[str, float]:
        labels = list(self.logp)
        vals = np.array([self.logp[label] for label in labels], dtype=float)
        vals -= np.max(vals)
        probs = np.exp(vals)
        probs /= probs.sum()
        return {label: float(prob) for label, prob in zip(labels, probs)}

    def select(self, ctx: Context) -> str:
        self.observe(ctx)
        posterior = self._posterior()
        if self.value_table:
            scores = {
                tau: sum(posterior[t] * self.value_table.get(t, {}).get(tau, 0.0) for t in posterior)
                for tau in ctx.pool
            }
            tau = max(ctx.pool, key=lambda item: (scores[item], item))
            extra = {"posterior": posterior, "policy": "posterior_weighted"}
        else:
            typ = max(posterior, key=posterior.get)
            tau = self.best_response.get(typ)
            fallback = tau is None
            tau = ctx.default if tau is None else tau
            extra = {"posterior": posterior, "map_type": typ, "best_response_missing": fallback}
        return self._record(ctx, tau, fallback=bool(extra.get("best_response_missing", False)), extra=extra)

    def update(
        self, ctx: Context, tau: str, reward: float, done: bool, info: dict
    ) -> None:
        return None

    def state_summary(self) -> dict:
        return {"posterior": self._posterior()}
