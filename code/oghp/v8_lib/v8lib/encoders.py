"""Observation encoders for semantic, numeric, opaque, and permuted settings."""

from __future__ import annotations

import hashlib
import re
import string
from dataclasses import replace
from typing import Callable, Literal

import numpy as np

from v8lib.context import Context


class Encoder:
    """Callable observation encoder that fills `Context.features` and `.text`."""

    def __init__(
        self,
        mode: Literal["semantic", "numeric", "opaque", "permuted", "relabel", "swapdesc"],
        semantic_template: Callable[[dict], str],
        feature_fn: Callable[[dict], np.ndarray],
        type_names: list[str],
        permutation_seed: int,
        keep_numeric: bool = False,
        feature_names: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.semantic_template = semantic_template
        self.feature_fn = feature_fn
        self.type_names = list(type_names)
        self.permutation_seed = int(permutation_seed)
        self.keep_numeric = bool(keep_numeric)
        self.feature_names = feature_names
        self.type_mapping = _derangement(self.type_names, self.permutation_seed)
        self.tokens = {
            typ: _token(typ, self.permutation_seed, i) for i, typ in enumerate(self.type_names)
        }

    def __call__(self, raw: dict, ctx_partial: Context) -> Context:
        features = np.asarray(self.feature_fn(raw), dtype=float).ravel()
        if self.mode in {"semantic", "relabel", "swapdesc"}:
            text = self.semantic_template(raw)
            out_features = features
        elif self.mode == "numeric":
            text = _numeric_text(raw, features, self.feature_names)
            out_features = features
        elif self.mode == "opaque":
            typ = raw.get("type_label", raw.get("type"))
            token = self.tokens.get(str(typ), _token(str(typ), self.permutation_seed, 0))
            text = f"opponent signature: {token}"
            one_hot = np.zeros(max(1, len(self.type_names)), dtype=float)
            if str(typ) in self.type_names:
                one_hot[self.type_names.index(str(typ))] = 1.0
            out_features = np.concatenate([one_hot, features]) if self.keep_numeric else one_hot
        elif self.mode == "permuted":
            text = permute_type_names(self.semantic_template(raw), self.type_mapping)
            out_features = features
        else:
            raise ValueError(f"unknown encoder mode: {self.mode}")
        return replace(ctx_partial, features=out_features, text=text)


def make_encoder(
    mode: Literal["semantic", "numeric", "opaque", "permuted", "relabel", "swapdesc"],
    semantic_template: Callable[[dict], str],
    feature_fn: Callable[[dict], np.ndarray],
    type_names: list[str],
    permutation_seed: int,
    keep_numeric: bool = False,
    feature_names: list[str] | None = None,
) -> Encoder:
    return Encoder(
        mode,
        semantic_template,
        feature_fn,
        type_names,
        permutation_seed,
        keep_numeric=keep_numeric,
        feature_names=feature_names,
    )


def permute_type_names(text: str, mapping: dict[str, str]) -> str:
    """Replace type-name mentions using a fixed mapping."""

    if not mapping:
        return text
    pattern = re.compile("|".join(rf"\b{re.escape(k)}\b" for k in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda match: mapping[match.group(0)], text)


def _derangement(items: list[str], seed: int) -> dict[str, str]:
    if len(items) < 2:
        return {item: item for item in items}
    rng = np.random.default_rng(seed)
    perm = list(items)
    for _ in range(1000):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))
    return dict(zip(items, items[1:] + items[:1]))


def _token(typ: str, seed: int, idx: int) -> str:
    digest = hashlib.sha256(f"{seed}:{idx}:{typ}".encode("utf-8")).digest()
    alphabet = string.ascii_uppercase + string.digits
    chars = [alphabet[b % len(alphabet)] for b in digest[:8]]
    return "Z" + "".join(chars)


def _numeric_text(raw: dict, features: np.ndarray, names: list[str] | None) -> str:
    if names is None:
        names = [f"f{i}" for i in range(len(features))]
    pairs = [f"{name}={float(value):.6g}" for name, value in zip(names, features)]
    return "features: " + ", ".join(pairs)
