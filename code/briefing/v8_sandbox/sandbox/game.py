"""Game primitives, strategies, payoffs, and opponent type policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = ["ROCK", "PAPER", "SCISSORS"]
LOVER_TYPES = ["ROCK_LOVER", "PAPER_LOVER", "SCISSORS_LOVER"]
ALL_TYPES = LOVER_TYPES + ["BEST_RESPONDER", "FLIPPER_L", "GULLIBLE"]
MIXED_STRATEGIES = {"MIX_RP": {"ROCK": 0.5, "PAPER": 0.5}, "MIX_PS": {"PAPER": 0.5, "SCISSORS": 0.5}}
FAMILIES = ("classic", "traits")
FAVOURITES = list(ACTIONS)
REACTIVITIES = ["habit", "beat_last", "copy_last"]
TIMINGS = ["stable", "flip"]
SCRIPTED_STRATEGIES = ["OUTPACE", "MIRROR_BEAT", "LAG_R", "LAG_P", "LAG_S", "SWITCH_OM", "SWITCH_MO"]
LAG_TARGET = {"LAG_R": "ROCK", "LAG_P": "PAPER", "LAG_S": "SCISSORS"}


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


def all_types(M: int, family: str = "classic") -> list[str]:
    if family == "traits":
        return trait_types(M)
    if int(M) == 3:
        return list(LOVER_TYPES)
    if int(M) == 6:
        return list(ALL_TYPES)
    raise ValueError("M must be 3 or 6")


def trait_types(M: int) -> list[str]:
    if int(M) == 9:
        return [f"{fav}-{react}" for react in REACTIVITIES for fav in FAVOURITES]
    if int(M) == 18:
        return [f"{fav}-{react}-{timing}" for timing in TIMINGS for react in REACTIVITIES for fav in FAVOURITES]
    raise ValueError("traits family needs M in {9, 18}")


def is_trait_type(type_name: str) -> bool:
    return "-" in str(type_name)


def parse_trait(type_name: str) -> tuple[str, str, str]:
    parts = str(type_name).split("-")
    return parts[0], parts[1], parts[2] if len(parts) > 2 else "stable"


def describe_type(type_name: str) -> str:
    if is_trait_type(type_name):
        fav, react, timing = parse_trait(type_name)
        return f"traits(favourite={fav}, reactivity={react}, timing={timing})"
    return DESCRIPTIONS[str(type_name)]


def strategy_pool(K: int, family: str = "classic") -> list[str]:
    if family == "traits":
        if int(K) == 5:
            return list(ACTIONS) + SCRIPTED_STRATEGIES[:2]
        if int(K) == 10:
            return list(ACTIONS) + list(SCRIPTED_STRATEGIES)
        raise ValueError("traits family needs K in {5, 10}")
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


def script_action(strategy: str, our_history: list[str] | None, round_index: int, H: int) -> str:
    last = our_history[-1] if our_history else "ROCK"
    first_half = int(round_index) < int(H) // 2
    if strategy == "OUTPACE":
        return beats(beats(last))
    if strategy == "MIRROR_BEAT":
        return beats(last)
    if strategy in LAG_TARGET:
        target = LAG_TARGET[strategy]
        return beats(target) if first_half else beats(beats(target))
    if strategy == "SWITCH_OM":
        return script_action("OUTPACE" if first_half else "MIRROR_BEAT", our_history, round_index, H)
    if strategy == "SWITCH_MO":
        return script_action("MIRROR_BEAT" if first_half else "OUTPACE", our_history, round_index, H)
    raise ValueError(f"unknown strategy: {strategy}")


def strategy_dist(strategy: str, our_history: list[str] | None = None, round_index: int = 0, H: int = 30) -> np.ndarray:
    if strategy in SCRIPTED_STRATEGIES:
        strategy = script_action(strategy, our_history, round_index, H)
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


def trait_dist(type_name: str, round_index: int, our_history: list[str], sharpness: float, H: int) -> np.ndarray:
    fav, react, timing = parse_trait(type_name)
    if timing == "flip" and int(round_index) >= int(H) // 2:
        if react == "habit":
            fav = beats(fav)
        else:
            react = "copy_last" if react == "beat_last" else "beat_last"
    if react == "habit":
        return lover_dist(fav, sharpness=sharpness)
    if not our_history:
        return strategy_dist(fav)
    target = beats(our_history[-1]) if react == "beat_last" else our_history[-1]
    dist = (1.0 - float(sharpness)) * strategy_dist(fav)
    dist[action_index(target)] += float(sharpness)
    return dist


def opponent_dist(
    type_name: str,
    round_index: int,
    our_history: list[str],
    L: int = 5,
    sharpness: float = 0.8,
    H: int = 30,
) -> np.ndarray:
    typ = str(type_name)
    if is_trait_type(typ):
        return trait_dist(typ, round_index, our_history, sharpness, H)
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


def expected_payoff(
    our_strategy: str,
    opp_dist: np.ndarray,
    A: np.ndarray,
    our_history: list[str] | None = None,
    round_index: int = 0,
    H: int = 30,
) -> float:
    return float(strategy_dist(our_strategy, our_history, round_index, H) @ A @ np.asarray(opp_dist, dtype=float))


def best_response_to_dist(pool: list[str], opp_dist: np.ndarray, A: np.ndarray) -> tuple[str, float, float]:
    vals = {tau: expected_payoff(tau, opp_dist, A) for tau in pool}
    ordered = sorted(vals.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else top[1]
    return top[0], float(top[1]), float(top[1] - second)
