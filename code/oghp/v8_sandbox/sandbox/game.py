"""Game primitives, strategies, payoffs, and opponent type policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = ["ROCK", "PAPER", "SCISSORS"]
LOVER_TYPES = ["ROCK_LOVER", "PAPER_LOVER", "SCISSORS_LOVER"]
ALL_TYPES = LOVER_TYPES + ["BEST_RESPONDER", "FLIPPER_L", "GULLIBLE"]
MIXED_STRATEGIES = {"MIX_RP": {"ROCK": 0.5, "PAPER": 0.5}, "MIX_PS": {"PAPER": 0.5, "SCISSORS": 0.5}}


@dataclass(frozen=True)
class TypeDescription:
    name: str
    text: str


DESCRIPTIONS = {
    "ROCK_LOVER": "ROCK_LOVER usually throws rock and only rarely deviates.",
    "PAPER_LOVER": "PAPER_LOVER usually throws paper and only rarely deviates.",
    "SCISSORS_LOVER": "SCISSORS_LOVER usually throws scissors and only rarely deviates.",
    "BEST_RESPONDER": "BEST_RESPONDER reacts by throwing the action that beats your previous throw.",
    "FLIPPER_L": "FLIPPER_L rotates its favourite action every L rounds.",
    "GULLIBLE": "GULLIBLE copies your previous throw whenever it can.",
}


def all_types(M: int) -> list[str]:
    if int(M) == 3:
        return list(LOVER_TYPES)
    if int(M) == 6:
        return list(ALL_TYPES)
    raise ValueError("M must be 3 or 6")


def describe_type(type_name: str) -> str:
    return DESCRIPTIONS[str(type_name)]


def strategy_pool(K: int) -> list[str]:
    if int(K) == 3:
        return list(ACTIONS)
    if int(K) == 5:
        return list(ACTIONS) + list(MIXED_STRATEGIES)
    raise ValueError("K must be 3 or 5")


def payoff_matrix(delta: float) -> np.ndarray:
    raw = np.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    return raw * (float(delta) / 0.7)


def beats(action: str) -> str:
    return {"ROCK": "PAPER", "PAPER": "SCISSORS", "SCISSORS": "ROCK"}[action]


def action_index(action: str) -> int:
    return ACTIONS.index(action)


def strategy_dist(strategy: str) -> np.ndarray:
    if strategy in ACTIONS:
        dist = np.zeros(3, dtype=float)
        dist[action_index(strategy)] = 1.0
        return dist
    if strategy in MIXED_STRATEGIES:
        dist = np.zeros(3, dtype=float)
        for action, prob in MIXED_STRATEGIES[strategy].items():
            dist[action_index(action)] = prob
        return dist
    raise ValueError(f"unknown strategy: {strategy}")


def lover_dist(favourite: str, sharpness: float = 0.8) -> np.ndarray:
    p = float(sharpness)
    if not 0.0 <= p <= 1.0:
        raise ValueError("sharpness must be in [0, 1]")
    off = (1.0 - p) / 2.0
    dist = np.full(3, off, dtype=float)
    dist[action_index(favourite)] = p
    return dist


def flipper_favourite(round_index: int, L: int = 5) -> str:
    return ACTIONS[(int(round_index) // int(L)) % len(ACTIONS)]


def opponent_dist(
    type_name: str,
    round_index: int,
    our_history: list[str],
    L: int = 5,
    sharpness: float = 0.8,
) -> np.ndarray:
    typ = str(type_name)
    if typ.endswith("_LOVER"):
        return lover_dist(typ.removesuffix("_LOVER"), sharpness=sharpness)
    if typ == "FLIPPER_L":
        return lover_dist(flipper_favourite(round_index, L=L), sharpness=sharpness)
    if typ == "BEST_RESPONDER":
        if not our_history:
            return np.full(3, 1.0 / 3.0, dtype=float)
        return strategy_dist(beats(our_history[-1]))
    if typ == "GULLIBLE":
        if not our_history:
            return np.full(3, 1.0 / 3.0, dtype=float)
        return strategy_dist(our_history[-1])
    raise ValueError(f"unknown type: {type_name}")


def expected_payoff(our_strategy: str, opp_dist: np.ndarray, A: np.ndarray) -> float:
    return float(strategy_dist(our_strategy) @ A @ np.asarray(opp_dist, dtype=float))


def best_response_to_dist(pool: list[str], opp_dist: np.ndarray, A: np.ndarray) -> tuple[str, float, float]:
    vals = {tau: expected_payoff(tau, opp_dist, A) for tau in pool}
    ordered = sorted(vals.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else top[1]
    return top[0], float(top[1]), float(top[1] - second)
