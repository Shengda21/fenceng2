"""Decision-window context shared by selectors and experiment harnesses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import re
from typing import Any

import numpy as np


HIDDEN_KEYS = {
    "type",
    "type1",
    "type2",
    "type_label",
    "switch_step",
}

OBSERVATION_EXTRA_ALLOWLIST = {
    "raw",
    "observation",
    "observations",
    "obs",
    "public",
    "public_observation",
    "features",
    "feature_names",
    "text",
    "metrics",
    "summary",
    "hint",
    "hints",
    "metadata",
    "schema",
}


@dataclass
class Context:
    """Observation summary passed to a high-level strategy selector."""

    pool: list[str]
    features: np.ndarray
    text: str
    episode_index: int
    window_index: int
    default: str
    type_label: str | None = None
    extra: dict = field(default_factory=dict)

    def masked_type(self) -> "Context":
        """Return a copy with hidden type removed."""

        return redact_hidden_context(self)

    def with_observation(self, features: np.ndarray, text: str) -> "Context":
        """Return a copy with encoder-produced observation fields."""

        return replace(self, features=np.asarray(features, dtype=float), text=text)


def redact_hidden_context(ctx: Context) -> Context:
    """Return a non-oracle context with hidden labels removed deeply.

    Hidden label values are collected before hidden keys are dropped so public
    text or feature-name payloads cannot smuggle the exact labels under a
    harmless-looking key.
    """

    hidden_values = _collect_hidden_values(ctx.extra)
    if ctx.type_label is not None:
        hidden_values.add(str(ctx.type_label))
    redacted_extra = _redact_extra(ctx.extra, hidden_values, top_level=True)
    return replace(
        ctx,
        type_label=None,
        text=_scrub_string(ctx.text, hidden_values),
        extra=redacted_extra if isinstance(redacted_extra, dict) else {},
    )


def _is_hidden_key(key: Any) -> bool:
    text = str(key).lower()
    return text in HIDDEN_KEYS or text.startswith("type") or text.endswith("_type") or "_type_" in text


def _collect_hidden_values(value: Any, *, hidden_parent: bool = False) -> set[str]:
    values: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_hidden = hidden_parent or _is_hidden_key(key)
            values.update(_collect_hidden_values(item, hidden_parent=child_hidden))
    elif isinstance(value, str):
        if hidden_parent:
            values.add(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            values.update(_collect_hidden_values(item, hidden_parent=hidden_parent))
    elif hidden_parent and value is not None:
        values.add(str(value))
    return {v for v in values if v}


def _redact_extra(value: Any, hidden_values: set[str], *, top_level: bool = False) -> Any:
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if _is_hidden_key(key):
                continue
            if top_level and str(key) not in OBSERVATION_EXTRA_ALLOWLIST:
                continue
            redacted = _redact_extra(item, hidden_values)
            if redacted is not _DROP:
                out[key] = redacted
        return out
    if isinstance(value, str):
        return _scrub_string(value, hidden_values)
    if isinstance(value, tuple):
        return tuple(_redact_extra(item, hidden_values) for item in value)
    if isinstance(value, list):
        return [_redact_extra(item, hidden_values) for item in value]
    return value


def _scrub_string(text: str, hidden_values: set[str]) -> str:
    out = str(text)
    for value in sorted(hidden_values, key=len, reverse=True):
        if not value:
            continue
        out = re.sub(re.escape(value), "[REDACTED]", out, flags=re.IGNORECASE)
    return out


class _Drop:
    pass


_DROP = _Drop()
