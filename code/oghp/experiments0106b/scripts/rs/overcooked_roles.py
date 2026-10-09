"""Chef-0 role macros for the Overcooked regime-switching experiments."""

from __future__ import annotations

from typing import Any

from scripts.run_overcooked_sweep import scripted_pick_task

try:
    from .oc_common import i1_held_task, pot_state, urgent_i2_task
except ImportError:  # pragma: no cover - direct script fallback
    from oc_common import i1_held_task, pot_state, urgent_i2_task  # type: ignore

ROLE_ONION = "ROLE_ONION"
ROLE_PLATE = "ROLE_PLATE"
ROLE_SERVE = "ROLE_SERVE"
ROLE_FULL = "ROLE_FULL"

ROLES = (ROLE_ONION, ROLE_PLATE, ROLE_SERVE, ROLE_FULL)


def describe_role(name: str) -> str:
    descriptions = {
        ROLE_ONION: "Fetch onions, put them in the pot, and start cooking when the pot is full.",
        ROLE_PLATE: "Fetch dishes, plate ready soup, then drop plated soup on a shared counter.",
        ROLE_SERVE: "Fetch dishes, plate ready soup, and serve plated soup.",
        ROLE_FULL: "Use the full scripted soup-production chain.",
    }
    return descriptions.get(name, f"{name} is an unknown role.")


def role_task(
    env_wrap: Any,
    mapper: Any,
    role: str,
    step: int,
    partner_serves: bool = True,
) -> str:
    del step

    if role == ROLE_PLATE:
        forced = i1_held_task(mapper, 0, soup_task="drop_held_item")
    else:
        forced = i1_held_task(mapper, 0)
    if forced:
        return forced

    urgent = urgent_i2_task(env_wrap, mapper)
    if urgent and role != ROLE_ONION:
        return urgent
    if urgent == "cook" and role == ROLE_ONION:
        return "cook"

    ps = pot_state(env_wrap, mapper)

    if role == ROLE_ONION:
        if ps["onions"] >= 3 and not ps["cooking"] and not ps["ready"]:
            return "cook"
        return "get_onion"

    if role == ROLE_PLATE:
        if ps["ready"] or ps["cooking"]:
            return "get_dish"
        return "stay"

    if role == ROLE_SERVE:
        if ps["ready"] or ps["cooking"]:
            return "get_dish"
        return "stay"

    if role == ROLE_FULL:
        return scripted_pick_task(env_wrap, mapper, 0)

    return scripted_pick_task(env_wrap, mapper, 0) if partner_serves else "stay"


def role_uses_partner_server(role: str) -> bool:
    return role in {ROLE_ONION, ROLE_PLATE}
