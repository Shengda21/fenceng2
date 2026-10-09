"""Statistical analysis helpers for experiment tables."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


class CaptureValue(float):
    """Float capture value with an attached validity status."""

    def __new__(cls, value: float, status: str = "ok"):
        obj = float.__new__(cls, value)
        obj.status = status
        return obj


def paired_bootstrap(a, b, B: int = 10000, seed: int = 0, ids=None) -> dict:
    """Bootstrap a paired mean difference `mean(a-b)` by seed clusters."""

    clusters = _paired_clusters(a, b, ids=ids)
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(B), dtype=float)
    n = len(clusters)
    for i in range(int(B)):
        idx = rng.integers(0, n, n)
        diffs[i] = float(np.mean(np.concatenate([clusters[j] for j in idx])))
    flat = np.concatenate(clusters)
    return {
        "mean": float(np.mean(flat)),
        "ci": (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))),
        "n_clusters": n,
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""

    if n < 0 or k < 0 or k > n:
        raise ValueError("wilson requires 0 <= k <= n")
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def capture(v: float, v_star: float, h: float) -> float:
    """Capture fraction `(V(Pi)-V*)/H_D`."""

    if float(h) <= 0.0:
        return CaptureValue(float("nan"), "undefined_h_d")
    return CaptureValue(float((float(v) - float(v_star)) / float(h)), "ok")


def capture_curve(rewards, v_star: float, h: float) -> np.ndarray:
    """Cumulative capture curve for prefixes `T=1..len(rewards)`."""

    rewards = np.asarray(rewards, dtype=float)
    if rewards.size == 0:
        return np.array([], dtype=float)
    means = np.cumsum(rewards) / np.arange(1, rewards.size + 1)
    return np.asarray([float(capture(v, v_star, h)) for v in means], dtype=float)


def crossover(c_learner_curve, c_llm: float, B: int = 10000, seed: int = 0):
    """Return persistent crossing horizon with right-censoring metadata."""

    curves = np.asarray(c_learner_curve, dtype=float)
    if curves.ndim == 1:
        point = _persistent_crossing(curves, c_llm)
        return _crossover_result(point, [], curves.size)
    rng = np.random.default_rng(seed)
    n = curves.shape[0]
    boots = []
    for _ in range(int(B)):
        idx = rng.integers(0, n, n)
        boots.append(_persistent_crossing(np.mean(curves[idx], axis=0), c_llm))
    point = _persistent_crossing(np.mean(curves, axis=0), c_llm)
    return _crossover_result(point, boots, curves.shape[1])


def scaling_fit(x, y, B: int = 1000, seed: int = 0) -> dict:
    """Fit log-log slope with a bootstrap confidence interval."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite")
    if not np.all(x > 0) or not np.all(y > 0):
        raise ValueError("x and y must be positive")
    if np.unique(x).size != x.size:
        raise ValueError("x levels must be distinct")
    xlog, ylog = np.log(x), np.log(y)
    if xlog.size < 2:
        raise ValueError("at least two distinct positive x/y pairs are required")
    slope, intercept = np.polyfit(xlog, ylog, 1)
    rng = np.random.default_rng(seed)
    slopes = []
    degenerate = 0
    n = xlog.size
    for _ in range(int(B)):
        idx = rng.integers(0, n, n)
        if np.unique(xlog[idx]).size < 2:
            degenerate += 1
            continue
        slopes.append(float(np.polyfit(xlog[idx], ylog[idx], 1)[0]))
    if not slopes:
        raise ValueError("all bootstrap resamples were degenerate")
    ci = (float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975)))
    return {"slope": float(slope), "intercept": float(intercept), "ci": ci, "degenerate_resamples": degenerate}


def master_row(
    *,
    cell,
    M,
    W,
    encoding,
    H_D,
    V_star,
    best_non_llm,
    llm,
    conditions,
    prediction,
    verdict,
) -> dict:
    """Build one master-table row with the exact required top-level keys."""

    return {
        "cell": cell,
        "M": M,
        "W": W,
        "encoding": encoding,
        "H_D": H_D,
        "V_star": V_star,
        "best_non_llm": best_non_llm,
        "llm": llm,
        "conditions": conditions,
        "prediction": prediction,
        "verdict": verdict,
    }


def _paired_clusters(a, b, ids=None) -> list[np.ndarray]:
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            raise ValueError("a and b must both be mappings when either is a mapping")
        if set(a) != set(b):
            raise ValueError("paired bootstrap mappings must have identical keys")
        clusters = []
        for key in a:
            left = np.asarray(a[key], dtype=float)
            right = np.asarray(b[key], dtype=float)
            if left.shape != right.shape:
                raise ValueError(f"paired values for key {key!r} have different shapes")
            clusters.append((left - right).ravel())
        return clusters

    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("a and b must have the same shape")
    if ids is None:
        return [(left - right).ravel()[i : i + 1] for i in range(left.size)]

    ids_arr = np.asarray(ids)
    if ids_arr.shape[0] != left.shape[0]:
        raise ValueError("ids length must match the first dimension of a and b")
    clusters = []
    for ident in dict.fromkeys(ids_arr.tolist()):
        mask = ids_arr == ident
        clusters.append((left[mask] - right[mask]).ravel())
    return clusters


def _persistent_crossing(curve: np.ndarray, c_llm: float):
    values = np.asarray(curve, dtype=float).ravel()
    target = float(c_llm)
    for i in range(values.size):
        suffix = values[i:]
        if np.all(np.isfinite(suffix)) and np.all(suffix >= target):
            return int(i + 1)
    return None


def _crossover_result(point, boots: list[int | None], t_max: int) -> dict:
    if not boots:
        prob = 1.0 if point is not None else 0.0
        ci = (point, point) if point is not None else (None, None)
        status = "crossed" if point is not None else "right_censored"
        return {
            "T_star": point,
            "T_max": int(t_max),
            "censored": point is None,
            "crossing_probability": prob,
            "ci": ci,
            "status": status,
        }
    finite = np.asarray([b for b in boots if b is not None], dtype=float)
    prob = float(finite.size / len(boots))
    ci = (
        (int(np.quantile(finite, 0.025)), int(np.quantile(finite, 0.975)))
        if finite.size == len(boots)
        else (None, None)
    )
    status = "crossed" if point is not None and prob == 1.0 else "partially_censored" if prob > 0.0 else "right_censored"
    return {
        "T_star": point,
        "T_max": int(t_max),
        "censored": point is None,
        "crossing_probability": prob,
        "ci": ci,
        "status": status,
        "bootstrap_censored": int(len(boots) - finite.size),
    }
