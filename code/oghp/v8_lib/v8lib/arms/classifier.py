"""Few-shot classifier selector for calibrated type prediction."""

from __future__ import annotations

import numpy as np

from v8lib.arms.base import Selector
from v8lib.arms.simple import RandomSelector
from v8lib.context import Context


class FewShotClassifierSelector(Selector):
    """Few-shot type classifier; source: calibrated kNN/logistic baseline."""

    role = "zero_shot"

    def __init__(
        self,
        N: int,
        best_response: dict[str, str],
        model: str = "knn",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if model not in {"knn", "logistic"}:
            raise ValueError("model must be 'knn' or 'logistic'")
        self.N = int(N)
        self.best_response = dict(best_response)
        self.model_name = model
        self.name = f"fewshot_{model}_{N}"
        self.random = RandomSelector(seed=seed)
        self.clf = None
        self.is_fit = False

    def reset_run(self, seed: int) -> None:
        self.random.reset_run(seed)

    def fit(self, X, y) -> "FewShotClassifierSelector":
        if self.N <= 0:
            self.is_fit = False
            return self
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        keep = []
        for label in sorted(set(y.tolist())):
            idx = np.flatnonzero(y == label)[: self.N]
            keep.extend(idx.tolist())
        Xf, yf = X[keep], y[keep]
        if self.model_name == "knn":
            from sklearn.neighbors import KNeighborsClassifier

            self.clf = KNeighborsClassifier(n_neighbors=min(3, len(yf)))
        else:
            from sklearn.linear_model import LogisticRegression

            self.clf = LogisticRegression(max_iter=1000, random_state=0)
        self.clf.fit(Xf, yf)
        self.is_fit = True
        return self

    def select(self, ctx: Context) -> str:
        if self.N <= 0 or not self.is_fit:
            tau = self.random.select(ctx)
            self.last_provenance = dict(self.random.last_provenance or {})
            self.last_provenance["arm"] = self.name
            return tau
        label = str(self.clf.predict(np.asarray(ctx.features, dtype=float).reshape(1, -1))[0])
        tau = self.best_response.get(label, ctx.default)
        return self._record(ctx, tau, extra={"predicted_type": label})
