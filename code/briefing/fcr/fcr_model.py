"""Factorized Compositional Reader (FCR).

Layer 1, trait classifiers: one classifier per trait (favourite, reactivity, timing; no timing at M=9) on the same
briefing text the existing readers take.
  - `tfidf` (primary): the text_bow feature extractor (word 1-2-grams + char_wb 2-5-grams, sublinear TF-IDF), and
    one logistic regression per trait.
  - `embed` (sensitivity): the text_embed representation (bge-small-en-v1.5 CLS embedding, read from the
    embedding cache keyed by the SHA-256 of the text), standardised, and one logistic regression per trait.
Layer 2, composer: for each pool strategy tau, a ridge regression of the calibration per-type value V(tau | z) on one-hot
codes of z = (f, r, t): main effects (`main`) or main effects plus all pairwise interactions (`pairwise`). The chosen
strategy is argmax_tau Q_tau(z_hat). The composer never sees a whole-type indicator, so it cannot fall back to a
per-type lookup table except through the interactions.
Trait-oracle FCR: the true z fed to the same composer (a privileged diagnostic).
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np

from fcr_common import TRAIT_VALUES, TRAITS, traits_of

LR_MAX_ITER = 5000


def _logistic(C: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=float(C), max_iter=LR_MAX_ITER, random_state=0)


def _tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union

    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    )


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TraitReader:
    """Per-trait classifiers sharing one feature extractor fitted on the training texts."""

    def __init__(self, M: int, extractor: str, C: dict, embed_cache: dict | None = None):
        self.M = int(M)
        self.traits = TRAITS[self.M]
        self.extractor = extractor
        self.C = {t: float(C[t]) for t in self.traits}
        self.embed_cache = embed_cache
        if extractor == "embed" and embed_cache is None:
            raise ValueError("the embedding extractor needs the release embedding cache")

    def _features(self, texts, fit: bool = False):
        texts = list(texts)
        if self.extractor == "tfidf":
            if fit:
                self.vectorizer = _tfidf()
                return self.vectorizer.fit_transform(texts)
            return self.vectorizer.transform(texts)
        if self.extractor == "embed":
            from sklearn.preprocessing import StandardScaler

            E = np.stack([self.embed_cache[text_key(t)] for t in texts]).astype(float)
            if fit:
                self.scaler = StandardScaler().fit(E)
            return self.scaler.transform(E)
        raise ValueError(f"unknown extractor {self.extractor}")

    def fit(self, texts, types) -> "TraitReader":
        X = self._features(texts, fit=True)
        Z = [traits_of(t, self.M) for t in types]
        self.clf = {}
        for j, trait in enumerate(self.traits):
            y = [z[j] for z in Z]
            self.clf[trait] = _logistic(self.C[trait]).fit(X, y)
        return self

    def predict(self, texts) -> list[tuple]:
        X = self._features(texts)
        cols = [self.clf[t].predict(X) for t in self.traits]
        return [tuple(str(c[i]) for c in cols) for i in range(len(cols[0]))]

    def log_loss(self, texts, types) -> dict:
        """Mean negative log-likelihood of the true value of each trait (a tie-breaker in the CV)."""

        X = self._features(texts)
        Z = [traits_of(t, self.M) for t in types]
        out = {}
        for j, trait in enumerate(self.traits):
            clf = self.clf[trait]
            P = clf.predict_proba(X)
            idx = {c: k for k, c in enumerate(clf.classes_)}
            p = np.array([P[i, idx[z[j]]] if z[j] in idx else 0.0 for i, z in enumerate(Z)])
            out[trait] = float(np.mean(-np.log(np.clip(p, 1e-12, 1.0))))
        return out


class Composer:
    """Ridge regression of V(tau | z) on one-hot trait codes, one output per pool strategy."""

    def __init__(self, M: int, form: str, alpha: float, pool: list[str]):
        if form not in ("main", "pairwise"):
            raise ValueError("form must be main or pairwise")
        self.M = int(M)
        self.traits = TRAITS[self.M]
        self.form = form
        self.alpha = float(alpha)
        self.pool = list(pool)
        self.columns = [(t, v) for t in self.traits for v in TRAIT_VALUES[t]]
        if form == "pairwise":
            for a, b in itertools.combinations(self.traits, 2):
                self.columns += [((a, b), (va, vb)) for va in TRAIT_VALUES[a] for vb in TRAIT_VALUES[b]]

    def design(self, zs) -> np.ndarray:
        pos = {t: j for j, t in enumerate(self.traits)}
        X = np.zeros((len(zs), len(self.columns)))
        for i, z in enumerate(zs):
            for k, (name, val) in enumerate(self.columns):
                if isinstance(name, tuple):
                    X[i, k] = float(z[pos[name[0]]] == val[0] and z[pos[name[1]]] == val[1])
                else:
                    X[i, k] = float(z[pos[name]] == val)
        return X

    def fit(self, types, values: dict) -> "Composer":
        from sklearn.linear_model import Ridge

        zs = [traits_of(t, self.M) for t in types]
        Y = np.array([[values[t][tau] for tau in self.pool] for t in types], dtype=float)
        self.model = Ridge(alpha=self.alpha, fit_intercept=True).fit(self.design(zs), Y)
        return self

    def q(self, zs) -> np.ndarray:
        return np.asarray(self.model.predict(self.design(zs)), dtype=float).reshape(len(zs), len(self.pool))

    def choose(self, zs) -> list[str]:
        """argmax_tau Q(z); exact ties go to the earlier strategy in pool order."""

        Q = self.q(zs)
        return [self.pool[int(np.argmax(row))] for row in Q]


class FCR:
    """Trait reader + composer. `fit` takes the labelled briefings of one data condition (texts and their seen types) and
    the calibration value rows of the seen types."""

    def __init__(self, M: int, extractor: str, C: dict, form: str, alpha: float, pool: list[str], embed_cache: dict | None = None):
        self.reader = TraitReader(M, extractor, C, embed_cache)
        self.composer = Composer(M, form, alpha, pool)

    def fit(self, texts, text_types, value_types, values: dict) -> "FCR":
        self.reader.fit(texts, text_types)
        self.composer.fit(value_types, values)
        return self

    def read(self, texts) -> tuple[list[tuple], list[str]]:
        zs = self.reader.predict(texts)
        return zs, self.composer.choose(zs)

    def oracle(self, types) -> list[str]:
        return self.composer.choose([traits_of(t, self.reader.M) for t in types])
