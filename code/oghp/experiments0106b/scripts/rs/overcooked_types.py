"""Scripted partner types for the Overcooked regime-switching experiments."""

from __future__ import annotations

from typing import Any

try:
    from .oc_common import (
        held_item,
        i1_held_task,
        nearest_shared_counter_to_pot,
        object_positions,
        pot_state,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from oc_common import (  # type: ignore
        held_item,
        i1_held_task,
        nearest_shared_counter_to_pot,
        object_positions,
        pot_state,
    )

ONION_RUNNER = "ONION_RUNNER"
PLATER = "PLATER"
SERVER = "SERVER"
IDLE = "IDLE"
COUNTER_PASSER = "COUNTER_PASSER"

PARTNER_TYPES = (ONION_RUNNER, PLATER, SERVER, IDLE)
ALL_PARTNER_TYPES = PARTNER_TYPES + (COUNTER_PASSER,)


def describe_type(name: str) -> str:
    descriptions = {
        ONION_RUNNER: "ONION_RUNNER repeatedly brings onions to the pot and never handles dishes.",
        PLATER: "PLATER handles dishes, waits for ready soup, plates it, and leaves soup on a shared counter.",
        SERVER: "SERVER prioritizes plated soup delivery and otherwise plates ready soup without fetching onions.",
        IDLE: "IDLE stays still for the whole episode.",
        COUNTER_PASSER: "COUNTER_PASSER fetches onions and passes them on the shared counter nearest the pot.",
    }
    return descriptions.get(name, f"{name} is an unknown partner type.")


def partner_task(env_wrap: Any, mapper: Any, ptype: str, step: int, rng: Any) -> str:
    del step, rng
    if ptype == IDLE:
        return "stay"

    held = held_item(mapper, 1)
    ps = pot_state(env_wrap, mapper)

    if ptype == PLATER and held == "soup":
        return "drop_held_item" if nearest_shared_counter_to_pot(mapper, require_empty=True) else "stay"
    if ptype == COUNTER_PASSER and held == "onion":
        return "drop_held_item" if nearest_shared_counter_to_pot(mapper, require_empty=True) else "stay"

    if ptype == ONION_RUNNER and held == "onion" and (ps["onions"] >= 3 or ps["cooking"] or ps["ready"]):
        return "stay"

    forced = i1_held_task(mapper, 1)
    if forced:
        if ptype == ONION_RUNNER and held != "onion":
            return "drop_held_item"
        if ptype == SERVER and held == "onion":
            return "drop_held_item"
        if ptype == PLATER and held == "onion":
            return "drop_held_item"
        return forced

    if ptype == ONION_RUNNER:
        if ps["onions"] >= 3 or ps["cooking"] or ps["ready"]:
            return "get_onion"
        return "get_onion"

    if ptype == PLATER:
        if ps["ready"] or ps["cooking"]:
            return "get_dish"
        return "stay"

    if ptype == SERVER:
        if object_positions(env_wrap, "soup"):
            return "serve_soup"
        if ps["ready"] or ps["cooking"]:
            return "get_dish"
        return "stay"

    if ptype == COUNTER_PASSER:
        return "get_onion"

    return "stay"
