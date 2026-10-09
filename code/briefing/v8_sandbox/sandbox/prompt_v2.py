"""Prompt version 2: the task is stated, the pool docs are first person, and the N-shot examples are a fixed,
seen-types-only design with varied phrasings that text_bow_matched refits on.

Nothing here says which strategy counters which opponent, and no type name appears."""

from __future__ import annotations

import numpy as np

from sandbox.briefing import axis_keys, render
from sandbox.game import all_types

PROMPT_VERSIONS = ("v1", "v2")
# First-person versions of selectors.STRATEGY_DOCS, same semantics.
STRATEGY_DOCS_V2 = {
    "ROCK": "you throw rock every round.",
    "PAPER": "you throw paper every round.",
    "SCISSORS": "you throw scissors every round.",
    "MIX_RP": "each round you throw rock or paper, each with probability one half.",
    "MIX_PS": "each round you throw paper or scissors, each with probability one half.",
    "OUTPACE": "each round you throw whatever beats the throw that beats your own previous throw.",
    "MIRROR_BEAT": "each round you throw whatever beats your own previous throw.",
    "LAG_R": "you throw paper every round until the halfway point of the match, then scissors every round for the rest.",
    "LAG_P": "you throw scissors every round until the halfway point of the match, then rock every round for the rest.",
    "LAG_S": "you throw rock every round until the halfway point of the match, then paper every round for the rest.",
    "SWITCH_OM": "until the halfway point of the match, each round you throw whatever beats the throw that beats your own previous throw; "
                 "after that, each round you throw whatever beats your own previous throw.",
    "SWITCH_MO": "until the halfway point of the match, each round you throw whatever beats your own previous throw; "
                 "after that, each round you throw whatever beats the throw that beats your own previous throw.",
}
INSTRUCTIONS_V2 = {
    "single": "Reply with 'Final: ' followed by exactly one strategy name from the pool.",
    "hypothesis_first": "First write one sentence describing how this opponent plays, in your own words and without using any "
                        "strategy name from the pool. Then write 'Final: ' followed by exactly one strategy name from the pool: "
                        "the strategy you will play.",
}
EXAMPLE_DESIGN_SEED = 20_261_005


def frame_parts(cfg) -> dict:
    """Prompt pieces for FramedLLMSelector. Single schedule only: the prefix of k rounds is played with the default."""

    if cfg.schedule != "single":
        raise ValueError("prompt v2 is written for the single schedule (one decision after the default prefix)")
    H, k = int(cfg.H), int(cfg.k)
    frame = (
        f"You are playing repeated rock-paper-scissors against one opponent for {H} rounds. "
        f"Rounds 1-{k} are played with {cfg.default}, the default strategy; you now choose one strategy from your pool, "
        f"and it plays rounds {k + 1}-{H} for you.\n"
        "Each round, rock beats scissors, scissors beats paper and paper beats rock; "
        "you score +1 for a win, -1 for a loss and 0 for a draw.\n"
        "Your objective is to maximise your total payoff against this opponent."
    )
    return {
        "frame": frame,
        "pool_header": "Your pool (the strategies you can play; each line describes your own throws):",
        "default_line": f"If you return no valid strategy name, {{default}} is played in rounds {k + 1}-{H} too.",
        "example_header": "Labelled examples from calibration: other opponents, each with the strategy from your pool that "
                          "scored best against it.",
        "test_header": "The opponent you face now:",
        "item_label": "Opponent",
        "answer_label": "Best strategy for you",
        "instructions": dict(INSTRUCTIONS_V2),
    }


def nshot_rows(cfg, best_response: dict, held_out, renderings: int) -> list[dict]:
    """Labelled briefings for the v2 N-shot prompt and for text_bow_matched: `renderings` per seen type (held-out types never
    appear). Each seen type gets one random permutation of the three seen phrasings per axis (fixed design seed); rendering j
    uses position j of every permutation, so renderings=3 covers every seen phrase of every axis exactly once per type, and
    renderings=1 is the first of those three. Order: canonical type order (the prompt may shuffle it per episode)."""

    if cfg.family != "traits":
        raise ValueError("v2 N-shot rows are defined for the traits family")
    if not 1 <= int(renderings) <= 3:
        raise ValueError("renderings per seen type must be 1, 2 or 3 (three seen phrasings per axis)")
    held = set(held_out or ())
    rng = np.random.default_rng(EXAMPLE_DESIGN_SEED)
    rows = []
    for typ in all_types(cfg.M, cfg.family):
        perms = [rng.permutation(3) for _ in axis_keys(typ)]
        if typ in held:
            continue
        for j in range(int(renderings)):
            variants = [int(p[j]) for p in perms]
            rows.append({"type": typ, "variants": variants, "text": render(typ, variants), "tau": best_response[typ]})
    return rows
