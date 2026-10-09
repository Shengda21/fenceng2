"""Natural-language scouting briefings and held-out sets for the traits family."""

from __future__ import annotations

import itertools

import numpy as np

from sandbox.game import FAVOURITES, REACTIVITIES, TIMINGS, parse_trait, trait_types

WORDINGS = ("sameword", "newword")
BRIEFING_ENCODINGS = ("briefing", "briefing_prefix")

PHRASES = {
    ("favourite", "ROCK"): [
        "This opponent's favourite throw is rock.",
        "The opponent likes to throw rock.",
        "Your opponent's preferred move is rock.",
        "Rock is the move this player keeps coming back to.",
        "Left alone, this player tends to make a fist.",
        "This player's default hand is stone.",
    ],
    ("favourite", "PAPER"): [
        "This opponent's favourite throw is paper.",
        "The opponent likes to throw paper.",
        "Your opponent's preferred move is paper.",
        "Paper is the move this player keeps coming back to.",
        "Left alone, this player tends to show an open hand.",
        "This player's default hand is the flat palm.",
    ],
    ("favourite", "SCISSORS"): [
        "This opponent's favourite throw is scissors.",
        "The opponent likes to throw scissors.",
        "Your opponent's preferred move is scissors.",
        "Scissors is the move this player keeps coming back to.",
        "Left alone, this player tends to hold out two fingers.",
        "This player's default hand is the two-finger V.",
    ],
    ("reactivity", "habit"): [
        "It plays that favourite most of the time and pays no attention to what you throw.",
        "It sticks to its habit and does not react to your moves.",
        "Your previous throws make no difference to it; it mostly repeats its favourite.",
        "It is a creature of routine, unmoved by anything you do.",
        "Whatever happened last round, it simply goes with what it likes.",
        "It ignores your play entirely and keeps to its preference nearly every round.",
    ],
    ("reactivity", "beat_last"): [
        "From the second round on, it usually throws whatever beats your previous throw, and otherwise falls back on its favourite.",
        "It mostly answers your last move with the throw that defeats it.",
        "It watches your previous throw and usually plays the move that wins against it.",
        "It likes to punish what you just did by picking the hand that tops your last one.",
        "Most rounds it reacts to your last hand by choosing the one that would have won against it.",
        "Expect it to answer each of your moves, a round late, with whatever would have defeated it.",
    ],
    ("reactivity", "copy_last"): [
        "From the second round on, it usually repeats the throw you made in the previous round, and otherwise falls back on its favourite.",
        "It mostly copies your last move.",
        "It watches your previous throw and usually plays that same move back.",
        "It likes to imitate you, echoing whatever you just played.",
        "Most rounds it borrows your last hand and plays it right back at you.",
        "Expect it to follow your lead, reusing your previous move a round later.",
    ],
    ("timing", "stable"): [
        "It keeps this style for the whole match.",
        "Its behaviour does not change as the match goes on.",
        "It plays the same way from the first round to the last.",
        "There is no mid-game shift; what you see early is what you get late.",
        "It holds one approach throughout all thirty rounds.",
        "Its tendencies stay fixed until the final round.",
    ],
    ("timing", "flip", "habit"): [
        "Halfway through the match, its favourite moves on to the throw that beats its old favourite.",
        "After the halfway point it adopts a new favourite: the throw that defeats the old one.",
        "In the second half it changes its favourite to whatever beats its first-half favourite.",
        "At the midpoint it upgrades its pet move to the one that would have won against it.",
        "Once the match is half over, it starts preferring the hand that wins against its earlier preference.",
        "Its preference rotates once, mid-match, to the move that would have defeated its original choice.",
    ],
    ("timing", "flip", "beat_last"): [
        "Halfway through the match, it stops beating your last throw and starts copying it instead.",
        "After the halfway point it no longer plays what beats your previous move; it repeats your previous move instead.",
        "In the second half it changes from beating your last throw to copying it.",
        "At the midpoint it trades punishing your last hand for imitating it.",
        "Once the match is half over, it starts echoing your previous move rather than defeating it.",
        "Mid-match its reaction turns around: instead of topping your last move, it plays it back to you.",
    ],
    ("timing", "flip", "copy_last"): [
        "Halfway through the match, it stops copying your last throw and starts throwing whatever beats it instead.",
        "After the halfway point it no longer repeats your previous move; it plays what beats your previous move instead.",
        "In the second half it changes from copying your last throw to beating it.",
        "At the midpoint it trades imitating your last hand for punishing it.",
        "Once the match is half over, it starts defeating your previous move rather than echoing it.",
        "Mid-match its reaction turns around: instead of playing your last move back, it picks the hand that tops it.",
    ],
}


def axis_keys(type_name: str) -> list[tuple]:
    fav, react, timing = parse_trait(type_name)
    keys = [("favourite", fav), ("reactivity", react)]
    if str(type_name).count("-") == 2:
        keys.append(("timing", "flip", react) if timing == "flip" else ("timing", "stable"))
    return keys


def render(type_name: str, variants) -> str:
    return " ".join(PHRASES[key][int(v)] for key, v in zip(axis_keys(type_name), variants))


def variants_for_seed(type_name: str, seed: int, new_words: bool) -> list[int]:
    n = len(axis_keys(type_name))
    draw = np.random.default_rng(int(seed) + 7_777_777).integers(0, 3, size=n)
    return [int(v) + (3 if new_words else 0) for v in draw]


def briefing_for(type_name: str, seed: int, held_out, wording: str) -> tuple[str, list[int]]:
    new_words = wording == "newword" and str(type_name) in set(held_out or ())
    variants = variants_for_seed(type_name, seed, new_words)
    return render(type_name, variants), variants


def all_renderings(type_name: str, offset: int = 0) -> list[tuple[list[int], str]]:
    n = len(axis_keys(type_name))
    out = []
    for combo in itertools.product(range(3), repeat=n):
        variants = [int(v) + int(offset) for v in combo]
        out.append((variants, render(type_name, variants)))
    return out


def held_out_rule(M: int) -> list[str]:
    if int(M) == 9:
        return [f"{FAVOURITES[i]}-{REACTIVITIES[i]}" for i in range(3)]
    if int(M) == 18:
        return [f"{FAVOURITES[i % 3]}-{REACTIVITIES[(i + i // 3) % 3]}-{TIMINGS[i // 3]}" for i in range(6)]
    raise ValueError("held-out rule defined for M in {9, 18}")


HELD_OUT_SETS = {
    "none": {9: [], 18: []},
    "diag3": {9: held_out_rule(9)},
    "diag6": {18: held_out_rule(18)},
    "react2": {9: [t for t in held_out_rule(9) if "habit" not in t]},
    "react4": {18: [t for t in held_out_rule(18) if "habit" not in t]},
    "react8": {18: [f"{f}-{r}-{t}" for t in TIMINGS for r in REACTIVITIES[1:] for f in FAVOURITES
                    if f"{f}-{r}-{t}" not in {"ROCK-beat_last-stable", "ROCK-copy_last-stable", "SCISSORS-beat_last-flip", "SCISSORS-copy_last-flip"}]},
}
HELD_OUT_KINDS = {"react": {9: "react2", 18: "react4"}, "diag": {9: "diag3", 18: "diag6"}, "react8": {18: "react8"}}


def held_out_set_name(name: str | None, M: int) -> str:
    if not name or name == "none":
        return "none"
    if name == "auto":
        name = "react"
    return HELD_OUT_KINDS[name][int(M)] if name in HELD_OUT_KINDS else name


def resolve_held_out_set(name: str | None, M: int) -> list[str]:
    name = held_out_set_name(name, M)
    if name == "none":
        return []
    names = HELD_OUT_SETS[name][int(M)]
    assert set(names) <= set(trait_types(M))
    return list(names)
