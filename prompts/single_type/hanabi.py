# Extracted from scripts/hier_v2.py (call_llm_v2, build_hanabi_prompt_v2, parse_hanabi_meta_v2), scripts/run_hanabi_sweep.py (observation parsers) and scripts/run_hanabi_sweep_v2.py (run_episode_v2, the call site).
"""Hanabi-Small (2 players, 2 colours, 5 ranks, one life): the meta-task prompt.

The selector reads the text observation and names one of four meta-tasks. A fixed
executor carries the meta-task out until the next call. Prompt strings, parsing and
call parameters are copied from the code that ran; comments are shortened.
"""
import os
import re
import time
from typing import Any

import requests

# ---------------------------------------------------------------- call settings
# The runs set LLM_MODEL to gpt-oss-120b and LLM_API_BASE to an OpenAI-compatible
# server through the environment; the address and the key are not part of this file.
LLM_API_BASE = os.environ.get("LLM_API_BASE", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "30"))


def call_llm_v2(prompt: str, default: str = "", max_tokens: int = 200) -> str:
    """One user message, no system message. Returns the raw text; parsing is done by the caller."""
    if not LLM_API_KEY:
        return default
    body: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", str(max_tokens))),
    }
    eff = os.environ.get("LLM_REASONING_EFFORT")
    if eff or "nvidia" in LLM_API_BASE.lower():
        body["reasoning_effort"] = eff or "low"
    backoff = [0, 5, 15, 30]  # waits in seconds between attempts
    last_err: Exception | None = None
    for wait in backoff:
        if wait:
            time.sleep(wait)
        try:
            resp = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}",
                         "Content-Type": "application/json"},
                json=body,
                timeout=LLM_TIMEOUT,
            )
            if resp.status_code == 429:
                last_err = RuntimeError(f"429 {resp.text[:80]}")
                continue
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content")
            if not content:
                content = msg.get("reasoning_content") or msg.get("reasoning") or ""
            return (content or default).strip()
        except Exception as e:
            last_err = e
            continue
    print(f"  [LLM-FAIL] {last_err}", flush=True)
    return default


# ------------------------------------------------------- observation parsing
COLOR_LETTERS = ["R", "Y", "G", "W", "B"]
RE_FIREWORKS = re.compile(r"Fireworks:\s*([^\n]+)")


def parse_fireworks(obs: str) -> dict[str, int]:
    m = RE_FIREWORKS.search(obs)
    if not m:
        return {}
    out: dict[str, int] = {}
    for tok in m.group(1).split():
        if len(tok) >= 2 and tok[1:].isdigit():
            out[tok[0]] = int(tok[1:])
    return out


def parse_partner_hand(obs: str) -> list[dict[str, Any] | None]:
    """Visible cards of the partner (the second block of the Hands section).

    "XX" marks an unknown slot and gives None.
    """
    parts = obs.split("Hands:")
    if len(parts) < 2:
        return []
    body = parts[1].split("Deck size:")[0]
    sections = [s.strip() for s in body.split("-----") if s.strip()]
    if len(sections) < 2:
        return []
    # the second block is the partner (two-player game)
    partner_block = sections[1]
    cards: list[dict[str, Any] | None] = []
    for line in partner_block.splitlines():
        line = line.strip()
        if not line or "player" in line.lower():
            continue
        first_tok = line.split("|")[0].strip()
        if first_tok in ("XX", "") or len(first_tok) < 2:
            cards.append(None)
            continue
        color = first_tok[0]
        rank_part = first_tok[1:]
        if rank_part.isdigit():
            cards.append({"color": color, "rank": int(rank_part)})
        else:
            cards.append(None)
    return cards


def parse_self_hand_knowledge(obs: str) -> list[dict[str, Any]]:
    """What the current player knows about their own cards.

    A card line looks like "XX || XX|RYGWB12345"; the part after the last "|" lists the
    colours and ranks that hints have not yet excluded.
    """
    parts = obs.split("Hands:")
    if len(parts) < 2:
        return []
    body = parts[1].split("Deck size:")[0]
    sections = [s.strip() for s in body.split("-----") if s.strip()]
    if not sections:
        return []
    cur_block = sections[0]
    out: list[dict[str, Any]] = []
    started = False
    for line in cur_block.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("cur player"):
            started = True
            continue
        if not started:
            continue
        # a card line such as "XX || XX|RYGWB12345"
        if "||" not in s:
            continue
        right = s.split("||", 1)[1].strip()
        if "|" in right:
            belief = right.split("|", 1)[1]
        else:
            belief = right
        colors = set(ch for ch in belief if ch.isalpha())
        ranks = set(int(ch) for ch in belief if ch.isdigit())
        out.append({"colors": colors, "ranks": ranks})
    return out


def find_playable_partner_card(partner_hand, fireworks):
    for slot, card in enumerate(partner_hand):
        if card is None:
            continue
        if fireworks.get(card["color"], 0) + 1 == card["rank"]:
            return slot, card
    return None, None


class HanabiEnvWrapper:
    """Only the static helper that the prompt builder calls."""

    @staticmethod
    def _extract_int(text: str, prefix: str) -> int:
        for line in text.splitlines():
            if prefix in line:
                tail = line.split(prefix, 1)[1].strip()
                num = ""
                for ch in tail:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num:
                    return int(num)
        return 0


# ------------------------------------------------------------------- the prompt
HANABI_META_TASKS = ["INFORM_PLAYABLE", "PLAY_KNOWN", "DISCARD_USELESS", "DISCARD_OLDEST"]
HANABI_DEFAULT_META = "DISCARD_OLDEST"


def build_hanabi_prompt_v2(env, parse_fireworks_fn, parse_self_hand_knowledge_fn,
                            parse_partner_hand_fn, find_playable_partner_card_fn,
                            HanabiEnvWrapper) -> str:
    """Observation fields, then the strategy guide, then the output format."""
    obs = env.observation_string()
    fw = parse_fireworks_fn(obs)
    info_tokens = HanabiEnvWrapper._extract_int(obs, "Info tokens:")
    life_tokens = HanabiEnvWrapper._extract_int(obs, "Life tokens:")
    deck = HanabiEnvWrapper._extract_int(obs, "Deck size:")
    sk = parse_self_hand_knowledge_fn(env.observation_string())
    ph = parse_partner_hand_fn(env.observation_string())
    ph_str = ", ".join(
        f"slot{i}={c['color']}{c['rank']}" if c else f"slot{i}=?"
        for i, c in enumerate(ph)
    )
    sk_str = ", ".join(
        f"slot{i}=col∈{{{','.join(sorted(k['colors']))}}}/rank∈{{{','.join(map(str, sorted(k['ranks'])))}}}"
        for i, k in enumerate(sk)
    )
    playable_partner_slot, playable_partner = find_playable_partner_card_fn(ph, fw)
    playable_str = (f"YES (slot{playable_partner_slot}={playable_partner['color']}{playable_partner['rank']})"
                    if playable_partner else "NO")

    own_known_playable = any(
        len(k["colors"]) == 1 and len(k["ranks"]) == 1
        and fw.get(next(iter(k["colors"])), 0) + 1 == next(iter(k["ranks"]))
        for k in sk
    )
    own_known_useless = any(
        len(k["colors"]) == 1 and len(k["ranks"]) == 1
        and fw.get(next(iter(k["colors"])), 0) >= next(iter(k["ranks"]))
        for k in sk
    )

    rules = (
        "STRATEGY GUIDE v2 (life=1 means a single misplay loses the entire game):\n"
        "  PRIORITY ORDER (apply first that matches):\n"
        f"  1. If your_card_known_playable={own_known_playable}: pick PLAY_KNOWN.\n"
        f"  2. If info_tokens={info_tokens}>0 AND (early_game[deck>=30] OR partner_has_playable={playable_str.startswith('YES')}):\n"
        "       pick INFORM_PLAYABLE.  (Hints are CHEAP and SAFE; never lose life from hinting.)\n"
        f"  3. If any_card_known_useless={own_known_useless}: pick DISCARD_USELESS.\n"
        f"  4. If info_tokens={info_tokens}=0 AND no other option: pick DISCARD_OLDEST as last resort.\n\n"
        f"  DISCARD_OLDEST IS ALMOST NEVER CORRECT in life=1 except as rule-4 last resort.\n"
        f"  Current deck={deck}, info_tokens={info_tokens}.\n"
    )
    return (
        "You are a Hanabi meta-planner for a 2-color/5-rank/life=1 cooperative game. Pick exactly ONE meta-task.\n"
        f"Fireworks: {fw}; info_tokens={info_tokens}; life_tokens={life_tokens}; deck={deck}.\n"
        f"Partner hand (you can see): {ph_str}.\n"
        f"Your hand (your knowledge from past hints): {sk_str}.\n"
        f"Partner has a directly playable card right now? {playable_str}.\n"
        + rules +
        "Available meta-tasks: INFORM_PLAYABLE | PLAY_KNOWN | DISCARD_USELESS | DISCARD_OLDEST.\n"
        "Output ONLY one line: META: <token>\n"
        "Example: META: INFORM_PLAYABLE\n"
        "Answer:"
    )


# ---------------------------------------------------------------------- parsing
def parse_hanabi_meta_v2(text: str) -> str:
    """Case-insensitive meta-task extraction; falls back to the default.

    The last non-empty line is read first, so a reasoning model that mentions
    DISCARD_OLDEST earlier in its text is not misread. Tokens are tried in the
    order of HANABI_META_TASKS.
    """
    if not text:
        return HANABI_DEFAULT_META
    t = text.upper()
    # take the last non-empty line (the answer is usually at the end)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    last = lines[-1] if lines else t
    # drop a "META: " or "META_TASK: " prefix
    if "META" in last and ":" in last:
        last = last.split(":", 1)[1].strip()
    # priority match
    for mt in HANABI_META_TASKS:
        if mt in last:
            return mt
    # no match on the last line: search the whole text
    for mt in HANABI_META_TASKS:
        if mt in t:
            return mt
    return HANABI_DEFAULT_META


# --------------------------------------------------------------------- call site
# Selector calls per episode: --replan-every 99 in the reported runs, so one call at
# the start and a second only if an episode lasts 99 turns (the small variant does not).
# The strict per-turn variant used --replan-every 1.
# The temperature-controlled router runs call a copy of call_llm_v2 that takes the
# temperature as an argument (0.5 here); every other line of the call is the same.
def select_meta(env) -> str:
    prompt = build_hanabi_prompt_v2(
        env, parse_fireworks, parse_self_hand_knowledge,
        parse_partner_hand, find_playable_partner_card, HanabiEnvWrapper,
    )
    text = call_llm_v2(prompt, default="META: INFORM_PLAYABLE", max_tokens=80)
    return parse_hanabi_meta_v2(text)
