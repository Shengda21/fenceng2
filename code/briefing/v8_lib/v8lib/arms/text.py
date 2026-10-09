"""Non-LLM selectors that read a natural-language briefing: keyword rule, bag of words, sentence embedding, embedding LinUCB."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from v8lib.arms.base import Selector
from v8lib.arms.contextual import LinUCB
from v8lib.context import Context

BEATS = {"ROCK": "PAPER", "PAPER": "SCISSORS", "SCISSORS": "ROCK"}
LAG = {"ROCK": "LAG_R", "PAPER": "LAG_P", "SCISSORS": "LAG_S"}
FAV_WORDS = {"rock": "ROCK", "paper": "PAPER", "scissors": "SCISSORS"}
BEAT_WORDS = ("beats", "defeats", "wins against")
COPY_WORDS = ("copies", "same move back", "repeats the throw you made")
FLIP_WORDS = ("halfway", "second half")


def briefing_text(ctx: Context) -> str:
    raw = ctx.extra.get("raw", {}) if isinstance(ctx.extra, dict) else {}
    return str(raw.get("briefing") or ctx.text)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scripted_text_rule(text: str, default: str) -> str:
    sentences = [s.lower() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    fav = next((FAV_WORDS[w] for w in re.findall(r"[a-z]+", sentences[0]) if w in FAV_WORDS), None) if sentences else None
    react_sentence = sentences[1] if len(sentences) > 1 else ""
    if any(w in react_sentence for w in BEAT_WORDS):
        react = "beat_last"
    elif any(w in react_sentence for w in COPY_WORDS):
        react = "copy_last"
    else:
        react = "habit"
    flip = len(sentences) > 2 and any(w in sentences[2] for w in FLIP_WORDS)
    if react == "beat_last":
        return "SWITCH_OM" if flip else "OUTPACE"
    if react == "copy_last":
        return "SWITCH_MO" if flip else "MIRROR_BEAT"
    if fav is None:
        return default
    return LAG[fav] if flip else BEATS[fav]


class ScriptedTextSelector(Selector):
    """Hand-written keyword rule over the briefing."""

    name = "scripted_text"
    role = "zero_shot"

    def select(self, ctx: Context) -> str:
        return self._record(ctx, scripted_text_rule(briefing_text(ctx), ctx.default))


class SentenceEmbedder:
    """CLS-pooled, L2-normalised sentence embeddings with an on-disk cache keyed by text hash."""

    def __init__(self, model_dir: str | Path | None = None, cache_path: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, np.ndarray] = {}
        self.tokenizer = None
        self.model = None
        if self.cache_path is not None and self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            self.cache = {str(k): v for k, v in zip(data["keys"], data["vecs"])}

    def _load(self) -> None:
        if self.model is not None:
            return
        if self.model_dir is None or not self.model_dir.exists():
            raise RuntimeError(f"text not in embedding cache and no model at {self.model_dir}")
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModel.from_pretrained(str(self.model_dir)).eval()

    def _compute(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        with torch.no_grad():
            batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            cls = self.model(**batch).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        return cls.numpy().astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = sorted({t for t in texts if text_key(t) not in self.cache})
        for i in range(0, len(missing), 64):
            chunk = missing[i : i + 64]
            for t, v in zip(chunk, self._compute(chunk)):
                self.cache[text_key(t)] = v
        return np.stack([self.cache[text_key(t)] for t in texts]).astype(float)

    def save(self, path: str | Path) -> None:
        keys = sorted(self.cache)
        np.savez(Path(path), keys=np.array(keys), vecs=np.stack([self.cache[k] for k in keys]))


def _logistic(C: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, max_iter=5000, random_state=0)


class TextBowSelector(Selector):
    """TF-IDF word and character n-grams with logistic regression, fit offline on labelled briefings."""

    name = "text_bow"
    role = "zero_shot"

    def __init__(self, texts: list[str], labels: list[str], C: float = 1.0) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_union

        self.vectorizer = make_union(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        )
        X = self.vectorizer.fit_transform(list(texts))
        self.clf = _logistic(C).fit(X, list(labels))
        self.train_accuracy = float(np.mean(self.clf.predict(X) == np.asarray(labels)))

    def predict(self, texts: list[str]) -> list[str]:
        return [str(x) for x in self.clf.predict(self.vectorizer.transform(list(texts)))]

    def select(self, ctx: Context) -> str:
        return self._record(ctx, self.predict([briefing_text(ctx)])[0])


class TextEmbedSelector(Selector):
    """Sentence embedding, standardised, with logistic regression fit offline on labelled briefings."""

    name = "text_embed"
    role = "zero_shot"

    def __init__(self, embedder: SentenceEmbedder, texts: list[str], labels: list[str], C: float = 1.0) -> None:
        super().__init__()
        from sklearn.preprocessing import StandardScaler

        self.embedder = embedder
        E = embedder.encode(list(texts))
        self.scaler = StandardScaler().fit(E)
        self.clf = _logistic(C).fit(self.scaler.transform(E), list(labels))
        self.train_accuracy = float(np.mean(self.clf.predict(self.scaler.transform(E)) == np.asarray(labels)))

    def predict(self, texts: list[str]) -> list[str]:
        return [str(x) for x in self.clf.predict(self.scaler.transform(self.embedder.encode(list(texts))))]

    def select(self, ctx: Context) -> str:
        return self._record(ctx, self.predict([briefing_text(ctx)])[0])


class TextLinUCB(LinUCB):
    """Online disjoint LinUCB whose context is the briefing's sentence embedding."""

    name = "text_linucb"

    def __init__(self, embedder: SentenceEmbedder, alpha: float = 0.6) -> None:
        super().__init__(alpha=alpha)
        self.embedder = embedder

    def _embedded(self, ctx: Context) -> Context:
        return replace(ctx, features=self.embedder.encode([briefing_text(ctx)])[0])

    def select(self, ctx: Context) -> str:
        return super().select(self._embedded(ctx))

    def update(self, ctx: Context, tau: str, reward: float, done: bool, info: dict) -> None:
        super().update(self._embedded(ctx), tau, reward, done, info)
