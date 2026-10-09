# Prompts shown to the language-model selectors

This directory collects every prompt that a language-model selector saw in the experiments of the paper. I wrote it so that a reader can check what the models were told, word for word, without running the code. The catalogue below is built from the code that assembles the prompts and from the per-call records in the data archive. `examples/` holds one real prompt, with the model's reply, for each family and encoding. `single_type/` holds the verbatim code of the single-type experiments (Hanabi, SMAC, MAgent, Overcooked and the HLA transfer), with one rendered prompt per domain.

## How to read this file

- Paths that start with `code/` point into this repository. Paths that start with `data/` point into the data archive that accompanies the release; that archive is not part of this repository tree.
- Code blocks are quoted from the code files. I removed docstrings and comment-only lines from the quotations and changed nothing else, so the comments in the files themselves may differ from what is shown here.
- `{text}`, `{pool}`, `{default}` and `{examples}` are placeholders that the builder fills in. Everything else in a template is sent as written.
- Every file in `examples/` is a verbatim copy of a stored call record (user prompt and raw completion). The system instruction of the bare-prompt arms is not stored in the records; for those files the header says that the instruction is derived from the code.
- The files in `single_type/examples/` are different: the records of those experiments do not store the prompt text, so each file is rendered by the original builder, and its first line says that it is not a recorded call. The quotations in Sections 6 and 7 are copied from the builders without the lines that compute the fields; the files in `single_type/` hold the full code.
- Section 6 covers the single-type slice (Hanabi, SMAC, MAgent and Overcooked) and Section 7 the transfer to the HLA stack. The native rules and situation texts of the HLA stack are not part of this release, so Section 7 shows the part of that prompt that I wrote and marks the rest as missing.

## 1. Common machinery

All selectors except the task-stating briefing prompt are built from one class, `LLMSelector`, in `code/oghp/v8_lib/v8lib/arms/llm.py` (the briefing line has its own copy in `code/briefing/v8_lib/v8lib/arms/llm.py`). The copy under `code/briefing` adds one override, `hypothesis_instruction`, and the framed selector of Section 5; `_parse`, `_shown` and `select` are identical in the two copies. The selector sends two chat messages: a system message that holds the instruction, and a user message that holds the prompt. This is the builder as used for the battle, combined-arms and sandbox runs:

```python
def _messages(self, ctx: Context) -> list[dict[str, str]]:
    examples = "\n".join(
        f"Observation: {text}\nStrategy: {self._shown(tau)}" for text, tau in self.n_shot_examples
    )
    pool_text = "\n".join(
        f"- {self._shown(tau)}: {self.pool_descriptions[tau]}"
        if self.pool_descriptions.get(tau)
        else f"- {self._shown(tau)}"
        for tau in ctx.pool
    )
    instruction = "Return exactly one strategy name from the pool."
    if self.label_map:
        instruction = "Return exactly one option label from the pool."
    if self.mode == "hypothesis_first":
        final_token = "OPTION_LABEL" if self.label_map else "STRATEGY"
        instruction = f"First write one sentence naming the inferred type. Then write Final: {final_token}."
    prompt = self.prompt_template.format(
        text=ctx.text,
        pool=pool_text,
        default=self._shown(ctx.default),
        examples=examples,
    )
    self._last_prompt = prompt
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]
```

The one difference in the briefing copy is the instruction line of the `hypothesis_first` mode:

```python
final_token = "OPTION_LABEL" if self.label_map else "STRATEGY"
instruction = self.hypothesis_instruction or f"First write one sentence naming the inferred type. Then write Final: {final_token}."
```

The system instruction therefore depends only on the mode and on whether the pool is shown with option labels:

| Mode | Pool shown by name | Pool shown by option label (`relabel`, `swapdesc`) |
| --- | --- | --- |
| `single` | `Return exactly one strategy name from the pool.` | `Return exactly one option label from the pool.` |
| `hypothesis_first` | `First write one sentence naming the inferred type. Then write Final: STRATEGY.` | `First write one sentence naming the inferred type. Then write Final: OPTION_LABEL.` |
| `deliberate` (`react` is an alias) | same as `single` on the first turn | same as `single` on the first turn |

In the sandbox briefing runs of the traits family, `hypothesis_first` uses `HYPOTHESIS_INSTRUCTION` from `code/briefing/v8_sandbox/sandbox/selectors.py` instead of the default:

> First write one sentence naming the inferred type. Then write 'Final: ' followed by exactly one strategy name from the pool.

In `deliberate` mode, if a reply names no unique pool entry, the code appends the assistant reply and the user message `Now output the final strategy name only.` and asks again, for at most `max_react_turns` = 3 turns. The recorded calls of this mode hold the user prompt of the first turn.

### N-shot examples

The `{examples}` block is the n-shot examples, one pair of lines each:

```
Observation: <text of a calibration observation>
Strategy: <best response to that opponent type>
```

The pairs are joined by a single newline. The battle and combined-arms templates start with `{examples}` followed by a newline, so a zero-shot prompt of these two families begins with an empty line. The sandbox template puts `{examples}` directly in front of the observation.

### Answer parsing and fallback

`_parse` accepts a reply only if exactly one pool entry occurs in it as a whole word. `select` asks up to `retries` = 3 times; if no attempt gives a unique entry, the selector returns the default strategy with `fallback=True` and source `give_up`.

```python
def _parse(self, text: str, pool: list[str]) -> tuple[str | None, list[str]]:
    hits = []
    for tau in pool:
        shown = self._shown(tau)
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(shown)}(?![A-Za-z0-9_])", text):
            hits.append((tau, shown))
    return (hits[0][0] if len(hits) == 1 else None), [shown for _, shown in hits]
```

```python
def _shown(self, tau: str) -> str:
    return self.label_map.get(tau, tau)
```

```python
def select(self, ctx: Context) -> str:
    start = perf_counter()
    messages = self._messages(ctx)
    attempts = 0
    raw = ""
    usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    call_latencies = []
    response_ids = []
    fingerprints = []
    err = None
    for attempts in range(1, self.retries + 1):
        try:
            turns = max(1, self.max_react_turns) if self.mode == "deliberate" else 1
            for turn in range(turns):
                call_start = perf_counter()
                resp = self._completion(messages)
                call_latencies.append(perf_counter() - call_start)
                raw = self._content(resp)
                self._add_usage(usage, self._usage(resp))
                response_ids.append(self._response_id(resp))
                fingerprints.append(self._fingerprint(resp))
                tau, hits = self._parse(raw, ctx.pool)
                if tau is not None:
                    return self._timed_record(
                        ctx,
                        start,
                        tau,
                        extra=self._audit_extra(
                            attempts,
                            "model",
                            raw,
                            hits,
                            usage,
                            call_latencies,
                            response_ids,
                            fingerprints,
                        ),
                    )
                if self.mode != "deliberate" or turn == turns - 1:
                    break
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Now output the final strategy name only."},
                ]
        except Exception as exc:  # pragma: no cover - exercised by fake failures if needed
            err = str(exc)
    self._episode_fallbacks += 1
    return self._timed_record(
        ctx,
        start,
        ctx.default,
        fallback=True,
        extra={
            **self._audit_extra(
                attempts,
                "give_up",
                raw,
                [],
                usage,
                call_latencies,
                response_ids,
                fingerprints,
            ),
            "error": err,
        },
    )
```

For `relabel` and `swapdesc` the model's reply is an option label (`OPTION_A`, ...); `_shown` maps pool names to labels for display and `_parse` returns the pool name, so the parsed strategy in a record is the mapped-back name while the raw completion is the label.

### Sampling settings as recorded

The values below are read from the stored call records (`temperature`, `max_tokens`, `extra_body`). All calls use temperature 0.5. The records of the briefing line carry `seed: null`.

| Family | `max_tokens` | gpt-oss-120b and gpt-oss-20b (`extra_body`) | qwen3.8-27b (`extra_body`) |
| --- | --- | --- | --- |
| MAgent battle, combined arms, sandbox | 256 | `{"reasoning_effort": "low"}` | `{"chat_template_kwargs": {"enable_thinking": false}}` |
| Briefing line, bare and task-stating prompt | 1024 | `{"reasoning_effort": "low"}` | `{"chat_template_kwargs": {"enable_thinking": false}}` |

The task-stating briefing prompt also has arms with another generation setting; the arm suffix in the file name names the setting. They use the same prompt builder. In the cells I compared (`...-react-briefing-sameword` for `v2-med` and `v2-think`, `...-react-briefing-newword` for `v2-high`, `v2-high-long` and `v2-think-long`), the stored prompt and system message are identical to those of the plain arm.

| Arm suffix | Model | `max_tokens` | `extra_body` |
| --- | --- | --- | --- |
| `v2-med` | gpt-oss | 4096 | `{"reasoning_effort": "medium"}` |
| `v2-high` | gpt-oss-120b | 4096 | `{"reasoning_effort": "high"}` |
| `v2-high-long` | gpt-oss-120b | 16384 | `{"reasoning_effort": "high"}` |
| `v2-think` | qwen3.8-27b | 4096 | `{"chat_template_kwargs": {"enable_thinking": true}}` |
| `v2-think-long` | qwen3.8-27b | 16384 | `{"chat_template_kwargs": {"enable_thinking": true}}` |

## 2. MAgent battle

Files: `code/oghp/experiments0106b/scripts/rs/run_rs_magent.py` (`prompt_template`, `nshot_examples`, `MAgentEncoder`, `TACTIC_DOCS`), `magent_env.py` (observation text), `magent_types.py` (type descriptors). Records: `data/llm_calls/battle/`. Examples: `examples/battle_*.txt`.

I play the RED team against a hidden BLUE type, and the selector picks one of eight RED strategies at a decision window.

### User prompt

```python
def prompt_template(type_names: list[str], type_hints: bool, encoding: str, mapping: dict[str, str]) -> str:
    hint_text = ""
    if type_hints:
        lines = []
        aliases = _aliases(type_names)
        for typ in type_names:
            mapped = mapping.get(typ, typ) if encoding == "permuted" else typ
            lines.append(f"- {aliases[typ]}: {semantic_descriptor(mapped)}")
        hint_text = "Possible co-player behavior descriptions:\n" + "\n".join(lines) + "\n"
    return (
        "{examples}\n"
        "You control the RED team at a decision window in MAgent battle.\n"
        "{text}\n"
        f"{hint_text}"
        "Available RED strategies:\n{pool}\n"
        "Default strategy: {default}\n"
        "Return exactly one available RED strategy name."
    )
```

The recorded prompts contain the hint block (the run script's `--type-hints` flag). It lists one line per BLUE type in the type set, under the neutral names `Pattern A`, `Pattern B`, ... in the order of the type set, so the labels do not reveal which type is the hidden one. The type sets are:

```python
BASE_TYPES = ["RUSH", "TURTLE", "KITE", "FLANK"]
TRAIT_TYPES = [
    f"{advance}-{formation}-{fire}"
    for advance, formation, fire in product(
        ["push", "hold"], ["tight", "spread"], ["focus", "spray"]
    )
]
TYPE_SETS = {"base4": BASE_TYPES, "traits8": TRAIT_TYPES}
```

The descriptor shown for each type is `TYPE_SEMANTICS[type]`:

```python
TYPE_SEMANTICS = {
    "RUSH": "direct advance with immediate adjacent attacks",
    "TURTLE": "defensive holding with line repair and adjacent attacks only",
    "KITE": "hit-and-step-back pressure that avoids staying adjacent",
    "FLANK": "two-wing lateral movement toward offset enemy positions",
}
```

and, for the eight trait types (`traits8`), the string built from the three trait words (the `TYPE_SEMANTICS[_name]` assignment below):

```python
for _name in TRAIT_TYPES:
    _advance, _formation, _fire = _name.split("-")
    TYPE_DOCS[_name] = (
        f"{_advance} movement with {_formation} formation control and {_fire} adjacent-target selection."
    )
    TYPE_SEMANTICS[_name] = (
        f"{_advance} movement, {_formation} formation control, and {_fire} adjacent-target selection"
    )
```

### Pool and default

The pool is `POOL8` (`POOL4` plus four more strategies); the descriptions are `TACTIC_DOCS`. The default is `HOLD_POSITION` in the recorded runs.

```python
POOL4 = ("ATTACK_FORWARD", "HOLD_POSITION", "SPREAD_OUT", "RETREAT")
POOL8 = POOL4 + ("FOCUS_FIRE", "FLANK", "KITE", "RETREAT_REAL")
POOL_SETS = {"pool4": POOL4, "pool8": POOL8}
```

```python
TACTIC_DOCS = {
    "ATTACK_FORWARD": "each unit moves toward the nearest enemy and attacks when adjacent",
    "HOLD_POSITION": "units stay where they are and attack only adjacent enemies",
    "SPREAD_OUT": "units move away from their nearest ally to disperse, attacking adjacent enemies",
    "RETREAT": "units stay in place and do not attack",
    "FOCUS_FIRE": "units tighten toward their own centre, advance as a block toward the enemy centre, and concentrate attacks on the same adjacent target",
    "FLANK": "units split into two wings that approach the enemy from the sides, attacking when adjacent",
    "KITE": "units attack when adjacent and then step away, otherwise approaching only to two cells",
    "RETREAT_REAL": "units move away from the nearest enemy every step and never attack",
}
```

Each pool line is `- NAME: description`, built by `LLMSelector._messages`.

### Observation text (semantic encoding)

The text is computed from nine measured features (`FEATURE_NAMES`). The code never writes the hidden type's descriptor into it.

```python
FEATURE_NAMES = [
    "blue_centroid_velocity_toward_red",
    "blue_dispersion",
    "blue_dispersion_change",
    "engagement_distance",
    "blue_fraction_advanced",
    "blue_fraction_attacked",
    "blue_alive",
    "red_alive",
    "step",
]
```

```python
def semantic_template(raw: dict, concept_mapping: dict[str, str] | None = None) -> str:

    vals = dict(zip(FEATURE_NAMES, feature_fn(raw)))
    typ = str(raw.get("type_label", raw.get("type", "")))
    concept_type = (concept_mapping or {}).get(typ, typ)
    try:
        concept = semantic_descriptor(concept_type)
    except ValueError:
        concept = "behavior that must be inferred from the current movement window"
    speed = vals["blue_centroid_velocity_toward_red"]
    dispersion = vals["blue_dispersion"]
    spread_change = vals["blue_dispersion_change"]
    gap = vals["engagement_distance"]
    advanced = vals["blue_fraction_advanced"]
    fired = vals["blue_fraction_attacked"]
    formation = "tight" if dispersion < 2.0 else "wide"
    spread_phrase = "spreading out" if spread_change > 0.15 else "compressing" if spread_change < -0.15 else "holding shape"
    pace = "advancing quickly" if speed > 0.35 else "edging forward" if speed > 0.05 else "not advancing"
    move_quant = "almost every" if advanced > 0.75 else "many" if advanced > 0.4 else "few"
    fire_quant = "many" if fired > 0.4 else "some" if fired > 0.1 else "few"
    del concept
    return (
        f"The enemy force is {pace} in a {formation} formation and is now {gap:.1f} cells away. "
        f"{move_quant.capitalize()} enemy units have been moving toward us while {fire_quant} have fired. "
        f"Their spacing is {spread_phrase}, with {int(vals['blue_alive'])} enemies still active."
    )
```

### Encodings

`MAgentEncoder` has four modes. Only the observation text and the hint block change between them; the pool, the default and the final instruction line stay the same.

- `semantic`: the text above.
- `numeric`: `features: ` followed by `name=value` pairs for the nine features, with values formatted by `:.6g`.
- `opaque`: `opponent signature: <token>`, where the token is `Z` plus eight letters taken from a SHA-256 digest of `seed:index:type`.
- `permuted`: the observation text is unchanged. The hint block is permuted: the descriptors are reassigned among the pattern labels by a derangement (a permutation with no fixed point) drawn from the permutation seed.

The battle encoder has no `relabel` or `swapdesc` mode.

```python
class MAgentEncoder:

    def __init__(
        self,
        mode: str,
        type_names: list[str],
        permutation_seed: int,
        keep_numeric: bool = False,
        feature_names: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.type_names = list(type_names)
        self.permutation_seed = int(permutation_seed)
        self.keep_numeric = bool(keep_numeric)
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.type_mapping = _derangement(self.type_names, self.permutation_seed)
        self.tokens = {
            typ: _opaque_token(typ, self.permutation_seed, i) for i, typ in enumerate(self.type_names)
        }

    def __call__(self, raw: dict, ctx_partial):
        from dataclasses import replace

        features = self._features(raw)
        if self.mode == "semantic":
            text = self.render_text(raw)
            out_features = features
        elif self.mode == "numeric":
            text = "features: " + ", ".join(
                f"{name}={float(value):.6g}" for name, value in zip(self.feature_names, features)
            )
            out_features = features
        elif self.mode == "opaque":
            text = self.render_text(raw)
            typ = str(raw.get("type_label", raw.get("type")))
            one_hot = np.zeros(max(1, len(self.type_names)), dtype=float)
            if typ in self.type_names:
                one_hot[self.type_names.index(typ)] = 1.0
            out_features = np.concatenate([one_hot, features]) if self.keep_numeric else one_hot
        elif self.mode == "permuted":
            text = self.render_text(raw)
            out_features = features
        else:
            raise ValueError(f"unknown encoder mode: {self.mode}")
        return replace(ctx_partial, features=out_features, text=text)

    def render_text(self, raw: dict) -> str:
        if self.mode == "numeric":
            features = self._features(raw)
            return "features: " + ", ".join(
                f"{name}={float(value):.6g}" for name, value in zip(self.feature_names, features)
            )
        if self.mode == "opaque":
            typ = str(raw.get("type_label", raw.get("type")))
            return f"opponent signature: {self.tokens.get(typ, _opaque_token(typ, self.permutation_seed, 0))}"
        mapping = self.type_mapping if self.mode == "permuted" else None
        return semantic_template(raw, mapping)

    def _features(self, raw: dict) -> np.ndarray:
        if "_locked_features" in raw:
            return np.asarray(raw["_locked_features"], dtype=float).ravel()
        return np.asarray(feature_fn(raw), dtype=float).ravel()
```

```python
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
```

```python
def _opaque_token(typ: str, seed: int, idx: int) -> str:
    digest = hashlib.sha256(f"{seed}:{idx}:{typ}".encode("utf-8")).digest()
    alphabet = string.ascii_uppercase
    return "Z" + "".join(alphabet[b % len(alphabet)] for b in digest[:8])
```

```python
def _aliases(type_names: list[str]) -> dict[str, str]:
    return {typ: f"Pattern {chr(ord('A') + i)}" for i, typ in enumerate(type_names)}
```

### N-shot examples

`nshot_examples` takes, for each type in sorted order, the first `n` calibration rows (sorted by seed), renders each with the same encoder, and pairs it with the locked best response of that type. The recorded battle calls are zero-shot (`nshot0`).

```python
def nshot_examples(lock: dict[str, Any] | None, n: int, encoder: "MAgentEncoder") -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    if not lock or n <= 0:
        return [], []
    out: list[tuple[str, str]] = []
    meta: list[dict[str, Any]] = []
    best = lock.get("best_response_table", {})
    rows = sorted(lock.get("feature_rows", []), key=lambda row: (str(row.get("type")), int(row.get("seed", 0))))
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("type")), []).append(row)
    for typ in sorted(by_type):
        picked = by_type[typ][:n]
        if len(picked) != n:
            raise ValueError(f"not enough calibration examples for type {typ}: need {n}")
        for row in picked:
            tau = best.get(typ)
            if not tau:
                continue
            raw = dict(row.get("raw") or {"type": typ, "type_label": typ})
            raw.setdefault("type", typ)
            raw.setdefault("type_label", typ)
            if "features" in row:
                raw["_locked_features"] = row["features"]
            text = encoder.render_text(raw)
            out.append((text, tau))
            rendered = f"Observation: {text}\nStrategy: {tau}"
            meta.append({"seed": row.get("seed"), "type": typ, "tau": tau, "prompt": rendered})
    return out, meta
```

### Settings and parsing

Section 1 gives the settings (256 tokens, temperature 0.5) and the parser. The selector is built with `pool_descriptions=TACTIC_DOCS`:

```python
if arm.startswith("llm:"):
    parts = arm.split(":")
    model = parts[1] if len(parts) > 1 and parts[1] else None
    mode = parts[2] if len(parts) > 2 else "single"
    extra_body = json.loads(args.extra_body_json) if getattr(args, "extra_body_json", None) else None
    return LLMSelector(
        model=model,
        mode=mode,
        temperature=getattr(args, "temperature", 0.5),
        max_tokens=getattr(args, "max_tokens", 256),
        seed=getattr(args, "llm_seed", None),
        extra_body=extra_body,
        pool_descriptions=TACTIC_DOCS,
    )
```

## 3. MAgent combined arms

Files: `code/oghp/experiments0106b/scripts/rs/run_rs_combined.py`, `combined_env.py`, `combined_types.py`. Records: `data/llm_calls/combined/`. Examples: `examples/combined_*.txt`.

The setting is the same as in Section 2, with melee and ranged units on both sides, six RED strategies and a different observation text.

### User prompt

```python
def prompt_template(type_names: list[str], type_hints: bool, encoding: str, mapping: dict[str, str]) -> str:
    hint_text = ""
    if type_hints:
        lines = []
        aliases = _aliases(type_names)
        for typ in type_names:
            mapped = mapping.get(typ, typ) if encoding == "permuted" else typ
            lines.append(f"- {aliases[typ]}: {semantic_descriptor(mapped)}")
        hint_text = "Possible enemy behavior descriptions:\n" + "\n".join(lines) + "\n"
    return (
        "{examples}\n"
        "You control the RED team in MAgent combined_arms with melee and ranged units.\n"
        "{text}\n"
        f"{hint_text}"
        "Available RED strategies:\n{pool}\n"
        "Default strategy: {default}\n"
        "Return exactly one available RED strategy name."
    )
```

The hint block is present in every recorded prompt. The type sets and the descriptors shown in it:

```python
BASE_TYPES = ["MELEE_RUSH", "RANGED_STANDOFF", "MIXED_ADVANCE", "KITE"]
TRAIT_TYPES = [
    f"{advance}-{ranged_stance}-{focus}"
    for advance, ranged_stance, focus in product(
        ["push", "hold"], ["standoff", "screen"], ["melee_first", "ranged_first"]
    )
]
TYPE_SETS = {"base4": BASE_TYPES, "traits8": TRAIT_TYPES}
```

```python
TYPE_SEMANTICS = {
    "MELEE_RUSH": "isolated melee-first pressure with ranged units anchored behind the charge",
    "RANGED_STANDOFF": "ranged standoff fire behind a melee screen",
    "MIXED_ADVANCE": "coordinated whole-force advance with ranged units firing at range two",
    "KITE": "ranged fire-and-step-back pressure behind a melee screen",
}
```

```python
for _name in TRAIT_TYPES:
    _advance, _stance, _focus = _name.split("-")
    TYPE_DOCS[_name] = (
        f"{_advance} movement, ranged {_stance} positioning, and {_focus.replace('_', ' ')} target priority."
    )
    TYPE_SEMANTICS[_name] = (
        f"{_advance} movement with ranged {_stance} behavior and {_focus.replace('_', ' ')} focus"
    )
```

### Pool and default

The pool is `COMBINED_POOL6`; the default in the recorded runs is `HOLD_LINE`. The descriptions are `TACTIC_DOCS` of `run_rs_combined.py`.

```python
COMBINED_POOL6 = (
    "ALL_ATTACK",
    "HOLD_LINE",
    "RANGED_FIRST",
    "MELEE_FIRST",
    "SCREEN_KITE",
    "FLANK_RANGED",
)
```

```python
TACTIC_DOCS = {
    "ALL_ATTACK": "every unit closes on the enemy formation and attacks whatever is adjacent or in range",
    "HOLD_LINE": "melee units hold a line in front, ranged units stay behind them and fire at anything within range; nobody advances",
    "RANGED_FIRST": "all units advance and prioritise the enemy's ranged units as targets",
    "MELEE_FIRST": "all units advance and prioritise the enemy's melee units as targets",
    "SCREEN_KITE": "melee units form a screen while ranged units fire and step back to keep their distance",
    "FLANK_RANGED": "ranged units swing to the sides to shoot past the enemy's melee line while melee units screen",
}
```

### Observation text (semantic encoding)

```python
FEATURE_NAMES = [
    "blue_centroid_velocity_toward_red",
    "blue_melee_advance_fraction",
    "blue_ranged_fire_fraction",
    "blue_ranged_retreat_fraction",
    "blue_melee_screen_fraction",
    "engagement_distance",
    "blue_melee_mean_distance",
    "blue_ranged_mean_distance",
    "blue_alive",
    "red_alive",
    "step",
]
```

```python
def semantic_template(raw: dict, concept_mapping: dict[str, str] | None = None) -> str:
    vals = dict(zip(FEATURE_NAMES, feature_fn(raw)))
    pressure = "closing on our line" if vals["blue_centroid_velocity_toward_red"] > 0.1 else "holding its distance"
    melee_adv = vals["blue_melee_advance_fraction"]
    melee = "most of its melee units have been advancing" if melee_adv > 0.5 else (
        "some of its melee units have been advancing" if melee_adv > 0.15 else "its melee units have barely moved forward")
    fire = "heavy ranged fire" if vals["blue_ranged_fire_fraction"] > 0.35 else "little ranged fire"
    kite = "and its ranged units step back often after firing" if vals["blue_ranged_retreat_fraction"] > 0.25 else "and its ranged units rarely step back"
    return (
        f"Over the first {int(vals.get('step', 0)) or 'observed'} steps the enemy has been {pressure}; {melee} "
        f"(advance fraction {melee_adv:.2f}); we have seen {fire} {kite}. "
        f"Centroid distance is {vals['engagement_distance']:.1f}; "
        f"enemy melee/ranged mean distances are {vals['blue_melee_mean_distance']:.1f}/{vals['blue_ranged_mean_distance']:.1f}."
    )
```

### Encodings

`CombinedEncoder` has the same four modes as the battle encoder (`semantic`, `numeric` with its eleven features, `opaque`, `permuted`), and the recorded cells add two label encodings. The opaque token is `C` plus eight letters from a SHA-256 digest of `combined:seed:index:type`. In `permuted` the observation text is unchanged and only the hint block is permuted.

- `relabel`: the pool names are replaced by `OPTION_A`, `OPTION_B`, ... in pool order, in the pool lines, the default line and the n-shot answers. The descriptions stay with their strategies. The system instruction is the label version (Section 1).
- `swapdesc`: the same labels, and the descriptions are reassigned among the labels by a derangement drawn from the permutation seed, so every label carries another strategy's description.

```python
def _label_map(pool: list[str]) -> dict[str, str]:
    return {tau: f"OPTION_{chr(ord('A') + i)}" for i, tau in enumerate(pool)}
```

```python
def _pool_descriptions(encoding: str, pool: list[str], seed: int, docs: dict[str, str]) -> dict[str, str]:
    if encoding == "swapdesc":
        description_owner = _derangement(pool, seed)
        return {tau: docs.get(description_owner[tau], "") for tau in pool}
    return {tau: docs[tau] for tau in pool if tau in docs}
```

```python
def _opaque_token(typ: str, seed: int, idx: int) -> str:
    digest = hashlib.sha256(f"combined:{seed}:{idx}:{typ}".encode("utf-8")).digest()
    alphabet = string.ascii_uppercase
    return "C" + "".join(alphabet[b % len(alphabet)] for b in digest[:8])
```

The selector is built as follows; the labels and the descriptions are passed in here.

```python
if arm.startswith("llm:"):
    parts = arm.split(":")
    model = parts[1] if len(parts) > 1 and parts[1] else None
    mode = parts[2] if len(parts) > 2 else "single"
    return LLMSelector(
        model=model,
        mode=mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.llm_seed,
        extra_body=parse_extra_body(args.extra_body_json),
        pool_descriptions=_pool_descriptions(args.encoding, pool, getattr(args, "permutation_seed", 0), TACTIC_DOCS),
        label_map=_label_map(pool) if args.encoding in {"relabel", "swapdesc"} else None,
    )
```

### N-shot examples

`nshot_examples` is the same function as in Section 2 and produces the `Observation:` / `Strategy:` pairs of Section 1. The records contain `nshot0` and `nshot1` arms, and the modes `single`, `hypothesis_first` and `deliberate`.

## 4. Rock-paper-scissors sandbox (first-moves format)

Files: `code/oghp/v8_sandbox/` (`sandbox/env.py`, `sandbox/selectors.py`, `run_sandbox.py`) with the encoders in `code/oghp/v8_lib/v8lib/encoders.py`. Records: `data/llm_calls/sandbox/`. Examples: `examples/sandbox_*.txt`.

The selector watches the opponent for the first five rounds of a 30-round match, then picks one strategy for the remaining rounds. The default (ROCK) plays the first five rounds.

### User prompt

The template is the `LLMSelector` default; the n-shot version puts the examples in front:

```python
"{text}\nPool: {pool}\nDefault: {default}"
"{examples}\n{text}\nPool: {pool}\nDefault: {default}"
```

`{pool}` is the list of pool lines, so the rendered prompt reads `Pool: - ROCK: always throw rock` on one line and the remaining pool lines under it (see `examples/sandbox_semantic.txt`).

### Observation text (semantic encoding)

```python
def semantic_template(raw: dict) -> str:
    counts = raw.get("counts", {})
    pieces = [f"{name.lower()} {counts.get(name, 0)} times" for name in ACTIONS]
    copy = raw.get("copy_fraction", 0.0)
    br = raw.get("br_fraction", 0.0)
    return (
        f"So far the opponent has thrown {', '.join(pieces)}. "
        f"It copied your previous move {copy:.0%} of the time and best-responded {br:.0%} of the time."
    )
```

### Pool

The pool of the recorded cell (three strategies, `K3`) is ROCK, PAPER, SCISSORS, with the descriptions

```python
STRATEGY_DOCS = {
    "ROCK": "always throw rock",
    "PAPER": "always throw paper",
    "SCISSORS": "always throw scissors",
    "MIX_RP": "throw rock or paper with equal probability each round",
    "MIX_PS": "throw paper or scissors with equal probability each round",
}
```

### Encodings

The encoder is the generic one below; the sandbox passes `semantic_template` and `BASIC_FEATURE_NAMES` (`opp_rock`, `opp_paper`, `opp_scissors`, `best_response_fraction`, `copy_fraction`, `common_run_fraction`, `round_over_30`).

```python
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
```

- `semantic`: the text above.
- `numeric`: `features: ` plus `name=value` pairs (`:.6g`) over the feature names.
- `opaque`: `opponent signature: Z` plus eight characters from `A-Z0-9`, taken from a SHA-256 digest of `seed:index:type`.
- `permuted`: type names in the semantic text are replaced by a derangement of the type names. The sandbox observation names no opponent type, so the text is unchanged; I checked that the stored `permuted` prompts equal the `semantic` prompts in all 2400 cases I compared.
- `relabel`, `swapdesc`: the observation is the semantic text and the pool is shown by `OPTION_A`, `OPTION_B`, ... With `swapdesc` each label carries the description of the next strategy in a cyclic shift of the pool (`pool[1:] + pool[:1]`). The system instruction is the label version.

The selector construction that implements the label encodings:

```python
if arm.startswith("llm:"):
    if not os.environ.get("LLM_API_BASE"):
        raise RuntimeError("LLM_API_BASE must be set for LLMSelector runs")
    parts = arm.split(":")
    model = parts[1]
    mode = parts[2] if len(parts) > 2 else "single"
    kw = dict(llm_kwargs or {})
    encoding = getattr(cfg, "encoding", "semantic")
    if encoding in {"relabel", "swapdesc"}:
        kw.setdefault("label_map", {tau: f"OPTION_{chr(ord('A') + i)}" for i, tau in enumerate(pool)})
        if encoding == "swapdesc":
            shifted = pool[1:] + pool[:1]
            kw.setdefault("pool_descriptions", {tau: STRATEGY_DOCS[owner] for tau, owner in zip(pool, shifted)})
    kw.setdefault("pool_descriptions", STRATEGY_DOCS)
    return LLMSelector(model=model, mode=mode, **kw)
```

### N-shot examples

For every seen type and each of the first `n` calibration seeds, the example is the encoded observation of that episode paired with the locked best response for the type; held-out types are excluded.

```python
if args.nshot > 0 and args.arm.startswith("llm:"):
    examples = []
    for typ in [t for t in all_types(cfg.M) if t != args.held_out]:
        for seed in calib_seeds[: args.nshot]:
            ctx = _encode_if_needed(SandboxEnv(cfg, forced_type=typ).reset(int(seed)), encoder)
            examples.append((ctx.text, lock["best_response"][typ]))
    llm_kwargs["n_shot_examples"] = examples
    llm_kwargs["prompt_template"] = "{examples}\n{text}\nPool: {pool}\nDefault: {default}"
```

## 5. Sandbox briefing line

Files: `code/briefing/v8_sandbox/` (`sandbox/briefing.py`, `sandbox/env.py`, `sandbox/selectors.py`, `sandbox/prompt_v2.py`, `run_sandbox.py`) and `code/briefing/v8_lib/v8lib/arms/llm.py`. Records: `data/briefing/`. Examples: `examples/briefing_*.txt` and `examples/taskstating_*.txt`.

In this setting the opponent types are built from three traits (a favourite throw, how the opponent reacts to my last throw, and whether that changes at the halfway point), and the prompt carries a short natural-language scouting briefing that states the traits. The pool has ten strategies: ROCK, PAPER, SCISSORS and the seven scripted strategies. The default is ROCK, the match has 30 rounds, and the first five are played with the default.

### The briefing text

The briefing is the concatenation of one phrase per trait. There are three wordings of each phrase that the n-shot examples and the seen types use (indices 0 to 2) and three more wordings (indices 3 to 5) that only the held-out types receive under the `newword` condition. The wording index of each trait is drawn from the episode seed.

```python
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
```

```python
def axis_keys(type_name: str) -> list[tuple]:
    fav, react, timing = parse_trait(type_name)
    keys = [("favourite", fav), ("reactivity", react)]
    if str(type_name).count("-") == 2:
        keys.append(("timing", "flip", react) if timing == "flip" else ("timing", "stable"))
    return keys
```

```python
def render(type_name: str, variants) -> str:
    return " ".join(PHRASES[key][int(v)] for key, v in zip(axis_keys(type_name), variants))
```

```python
def variants_for_seed(type_name: str, seed: int, new_words: bool) -> list[int]:
    n = len(axis_keys(type_name))
    draw = np.random.default_rng(int(seed) + 7_777_777).integers(0, 3, size=n)
    return [int(v) + (3 if new_words else 0) for v in draw]
```

```python
def briefing_for(type_name: str, seed: int, held_out, wording: str) -> tuple[str, list[int]]:
    new_words = wording == "newword" and str(type_name) in set(held_out or ())
    variants = variants_for_seed(type_name, seed, new_words)
    return render(type_name, variants), variants
```

The `briefing` encoding shows the briefing alone; `briefing_prefix` shows the briefing followed by one space and the play-count text of Section 4.

```python
def briefing_template(raw: dict) -> str:
    return raw["briefing"]
```

```python
def briefing_prefix_template(raw: dict) -> str:
    return raw["briefing"] + " " + semantic_template(raw)
```

In the cells whose name ends in `semantic` the prompt shows the play-count text only.

### Bare prompt

The bare prompt uses the template and the parser of Section 1. Its text is the briefing, the pool descriptions are `STRATEGY_DOCS` below, and the system message is the `single` or `hypothesis_first` instruction of Section 1 (for `hypothesis_first` in this family, `HYPOTHESIS_INSTRUCTION`).

```python
STRATEGY_DOCS = {
    "ROCK": "always throw rock",
    "PAPER": "always throw paper",
    "SCISSORS": "always throw scissors",
    "MIX_RP": "throw rock or paper with equal probability each round",
    "MIX_PS": "throw paper or scissors with equal probability each round",
    "OUTPACE": "each round, throw the move that beats the move that would beat your own previous throw",
    "MIRROR_BEAT": "each round, throw the move that beats your own previous throw",
    "LAG_R": "throw paper until the halfway point of the match, then scissors for the rest",
    "LAG_P": "throw scissors until the halfway point of the match, then rock for the rest",
    "LAG_S": "throw rock until the halfway point of the match, then paper for the rest",
    "SWITCH_OM": "until the halfway point, each round throw the move that beats the move that would beat your own previous throw; after that, throw the move that beats your own previous throw",
    "SWITCH_MO": "until the halfway point, each round throw the move that beats your own previous throw; after that, throw the move that beats the move that would beat your own previous throw",
}
```

The selector is built as follows; the first branch is the task-stating prompt of the next subsection and the second is the bare prompt.

```python
if arm.startswith("llm:") and (llm_kwargs or {}).get("prompt_version", "v1") == "v2":
    if not os.environ.get("LLM_API_BASE"):
        raise RuntimeError("LLM_API_BASE must be set for LLMSelector runs")
    parts = arm.split(":")
    kw = {k: v for k, v in (llm_kwargs or {}).items() if k != "prompt_version"}
    if getattr(cfg, "encoding", "semantic") not in {"semantic", "briefing", "briefing_prefix"}:
        raise ValueError("prompt v2 is defined for the semantic, briefing and briefing_prefix encodings")
    kw.setdefault("pool_descriptions", STRATEGY_DOCS_V2)
    return FramedLLMSelector(model=parts[1], mode=parts[2] if len(parts) > 2 else "single", **frame_parts(cfg), **kw)
if arm.startswith("llm:"):
    if not os.environ.get("LLM_API_BASE"):
        raise RuntimeError("LLM_API_BASE must be set for LLMSelector runs")
    parts = arm.split(":")
    model = parts[1]
    mode = parts[2] if len(parts) > 2 else "single"
    kw = dict(llm_kwargs or {})
    encoding = getattr(cfg, "encoding", "semantic")
    if encoding in {"relabel", "swapdesc"}:
        kw.setdefault("label_map", {tau: f"OPTION_{chr(ord('A') + i)}" for i, tau in enumerate(pool)})
        if encoding == "swapdesc":
            shifted = pool[1:] + pool[:1]
            kw.setdefault("pool_descriptions", {tau: STRATEGY_DOCS[owner] for tau, owner in zip(pool, shifted)})
    kw.setdefault("pool_descriptions", STRATEGY_DOCS)
    if cfg.family == "traits":
        kw.setdefault("hypothesis_instruction", HYPOTHESIS_INSTRUCTION)
    return LLMSelector(model=model, mode=mode, **kw)
```

The n-shot examples of the bare prompt are `Observation: <briefing>` / `Strategy: <best response>` pairs, one per seen type for each of the first `n` calibration seeds. The task-stating prompt uses the branch `if v2 ...` below, the bare prompt the branch after it.

```python
if v2 and args.nshot > 0 and args.arm.startswith("llm:"):
    if cfg.family != "traits" or cfg.encoding != "briefing":
        raise SystemExit("v2 N-shot examples are defined for the traits family under the briefing encoding")
    llm_kwargs["n_shot_examples"] = [(r["text"], r["tau"]) for r in nshot_rows(cfg, lock["best_response"], held, args.nshot)]
elif args.nshot > 0 and args.arm.startswith("llm:"):
    examples = []
    for typ in [t for t in all_types(cfg.M, cfg.family) if t not in held]:
        for seed in calib_seeds[: args.nshot]:
            ctx = _encode_if_needed(SandboxEnv(cfg, forced_type=typ).reset(int(seed)), encoder)
            examples.append((ctx.text, lock["best_response"][typ]))
    llm_kwargs["n_shot_examples"] = examples
    llm_kwargs["prompt_template"] = "{examples}\n{text}\nPool: {pool}\nDefault: {default}"
```

### Task-stating prompt

This prompt tells the model what game it is playing, the payoff rule and its objective, states that the pool lists its own strategies, describes them in the first person, explains the default line, and repeats the instruction at the end. It is built by `FramedLLMSelector` from the pieces in `code/briefing/v8_sandbox/sandbox/prompt_v2.py`. The arm files carry the suffix `__v2`.

Pieces:

```python
def frame_parts(cfg) -> dict:

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
```

```python
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
```

```python
INSTRUCTIONS_V2 = {
    "single": "Reply with 'Final: ' followed by exactly one strategy name from the pool.",
    "hypothesis_first": "First write one sentence describing how this opponent plays, in your own words and without using any "
                        "strategy name from the pool. Then write 'Final: ' followed by exactly one strategy name from the pool: "
                        "the strategy you will play.",
}
```

Assembly. The blocks are joined by one blank line, in this order: the frame; the pool header, the pool lines and the default line; the example header and the examples (only when examples are used); the test header with the item to classify; and the instruction, which is also the system message. In the example and test blocks the labels are `Opponent:` and `Best strategy for you:`.

```python
def _messages(self, ctx: Context) -> list[dict[str, str]]:
    import numpy as np

    pool = list(ctx.pool)
    if self.pool_order == "shuffled":
        perm = np.random.default_rng(self._seed_for(POOL_ORDER_OFFSET)).permutation(len(pool))
        pool = [pool[int(i)] for i in perm]
    self._pool_listing = [self._shown(tau) for tau in pool]
    pool_text = "\n".join(
        f"- {self._shown(tau)}: {self.pool_descriptions[tau]}" if self.pool_descriptions.get(tau) else f"- {self._shown(tau)}"
        for tau in pool
    )
    order = list(range(len(self.n_shot_examples)))
    if order and self.example_order == "shuffled":
        order = [int(i) for i in np.random.default_rng(self._seed_for(EXAMPLE_ORDER_OFFSET)).permutation(len(order))]
    self._example_permutation = order if self.n_shot_examples else None
    blocks = [self.frame, f"{self.pool_header}\n{pool_text}\n{self.default_line.format(default=self._shown(ctx.default))}"]
    if self.n_shot_examples:
        items = "\n\n".join(
            f"{self.item_label}: {self.n_shot_examples[i][0]}\n{self.answer_label}: {self._shown(self.n_shot_examples[i][1])}"
            for i in order
        )
        blocks.append(f"{self.example_header}\n\n{items}")
    instruction = self.instructions[self.mode]
    blocks.append(f"{self.test_header}\n{self.item_label}: {ctx.text}\n{self.answer_label}: ?")
    blocks.append(instruction)
    prompt = "\n\n".join(blocks)
    self._last_prompt = prompt
    self._last_system = instruction
    return [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}]
```

Examples. `nshot_rows` fixes the labelled examples: the arm `nshot1` has one briefing per seen type (14 examples) and `nshot42` has three per seen type (42 examples). Held-out types never appear, and the rows are in canonical type order. In the `v2-shuf` arms the pool listing is shuffled with a generator seeded by the episode seed plus `POOL_ORDER_OFFSET`; the records store the listing as `pool_listing` and the setting as `pool_order`.

```python
EXAMPLE_DESIGN_SEED = 20_261_005
```

```python
def nshot_rows(cfg, best_response: dict, held_out, renderings: int) -> list[dict]:

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
```

```python
POOL_ORDER_OFFSET = 31_337_000
```

Parsing. The parser takes the name after the last `Final:` (case-insensitive) in the reply, and falls back to the unique-name rule of Section 1. Think blocks are split off before parsing.

```python
_FINAL_RE = re.compile(r"final(?:\s+(?:answer|strategy|choice))?\s*[:\uff1a]", re.IGNORECASE)
_NAME_AFTER_FINAL_RE = re.compile(r"[\s*`\"'\[\(]*([A-Za-z][A-Za-z0-9_]*)")
```

```python
def split_think(text: str) -> tuple[str, str | None]:

    text = text or ""
    if "</think>" in text:
        head, _, tail = text.rpartition("</think>")
        return tail.strip(), head.replace("<think>", "", 1).strip()
    if "<think>" in text:
        return "", text.replace("<think>", "", 1).strip()
    return text, None
```

```python
def _parse(self, text: str, pool: list[str]) -> tuple[str | None, list[str]]:
    finals = list(_FINAL_RE.finditer(text or ""))
    if finals:
        m = _NAME_AFTER_FINAL_RE.match(text[finals[-1].end():])
        if m:
            token = m.group(1).upper()
            for tau in pool:
                if self._shown(tau).upper() == token:
                    self._parse_rule = "final"
                    return tau, [self._shown(tau)]
    tau, hits = super()._parse(text, pool)
    self._parse_rule = "unique_name" if tau is not None else None
    return tau, hits
```

The record stores the system message (`system_prompt`), the prompt, the raw reply, the parse rule (`final` or `unique_name`) and the number of examples.

## 6. Single-type slice

In the single-type slice a language model chooses one strategy for a fixed executor, once per planning window. The executor then runs that strategy until the next call. The four domains are Hanabi-Small (2 players), SMAC (maps 3m, 8m and MMM), MAgent battle (map size 20) and Overcooked (four layouts). The verbatim extracts are in `single_type/`: `hanabi.py`, `smac.py`, `magent.py` and `overcooked.py`. Each file starts with a comment that names the original script and function. In each file, only comments and docstrings are shortened and translated, and the prompt strings, the parsers and the request bodies are copied from the code that ran. The endpoint address and the keys are left out. `single_type/examples/` holds one rendered prompt per domain. None of these prompts is a recorded call, because the run records of this slice keep the outcome of each episode and the number and length of the calls but not the prompt text. Each example file says so in its first line and states how the state was built. The parsing examples in those files use replies that I wrote by hand.

If a supplementary listing of one of these prompts and the text in this section differ, the text here is the one that was sent.

### Call and settings

The calls go to the chat completions interface of an OpenAI-compatible server with gpt-oss-120b. The request has one user message and no system message, and the temperature is 0.5. The reply is read from `content`; if that is empty, from `reasoning_content` or `reasoning`. The field `reasoning_effort` is added only when the environment variable `LLM_REASONING_EFFORT` is set or the server address contains `nvidia`, and then it is `low` unless the variable says otherwise. The environment variable `LLM_MAX_TOKENS` overrides the token cap in every domain.

| Domain and cell | Choices | Planning window | Calls per episode | Cap in the code | On a failed call or an unreadable reply |
| --- | --- | --- | --- | --- | --- |
| Hanabi-Small, 2 players | `INFORM_PLAYABLE`, `PLAY_KNOWN`, `DISCARD_USELESS`, `DISCARD_OLDEST` | 99 turns | 1 (the episodes end before turn 99) | 80 tokens | failed call: `META: INFORM_PLAYABLE`; reply with no token: `DISCARD_OLDEST` |
| SMAC 3m, 8m, MMM | `focus_fire`, `spread_fire`, `kite`, `retreat`, `advance` | 20 steps | 1 at the start, then 1 per 20 steps | 400 tokens | `focus_fire` |
| MAgent battle, map size 20, 100 steps | `ATTACK_FORWARD`, `HOLD_POSITION`, `SPREAD_OUT`, `RETREAT` | 99 steps | 2 (steps 0 and 99) | 60 tokens in the sweep, 80 in the temperature-controlled router | `ATTACK_FORWARD` |
| Overcooked, 4 layouts, 400 steps | 8 tasks, one pair per window | 20 steps | 20 | 120 tokens | `get_onion` for both chefs |

The Hanabi and MAgent calls make up to four attempts, after waits of 0, 5, 15 and 30 seconds, when the server answers 429 or the call fails. The timeout is 30 seconds. The SMAC call is made once with a timeout of 30 seconds. The Overcooked call is made once with a timeout of 30 seconds, or, in the runs that spread the load over several keys, up to three times with a timeout of 45 seconds. In the runs with several keys, a call whose attempts all fail is counted as an API failure and is kept apart from a reply that could not be parsed.

The runs that I report call these builders through two paths: the sweep scripts (`call_llm`, `call_llm_v2`) and a diagnostic router script that passes the temperature as an argument (0.5 here) to a copy of `call_llm_v2`. The prompt text and the request body are the same in both. The value of `LLM_MAX_TOKENS` in the runs is not archived. The logged completions of the hosted runs are longer than the caps above (Hanabi up to 1,618 tokens), so those runs used a larger cap than the code defaults.

### Hanabi (`single_type/hanabi.py`)

The builder is `build_hanabi_prompt_v2`. It reads the text observation and computes the fireworks, the number of info and life tokens, the deck size, the partner's visible cards, what the current player knows about its own cards, whether a partner card is playable, and whether an own card is known to be playable or useless. Then it assembles the text below, which is quoted without the lines that compute the fields:

```python
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
```

The parser `parse_hanabi_meta_v2` upper-cases the reply and reads the last non-empty line, after removing a `META:` prefix. It returns the first token of the choice list that occurs in that line, in the order `INFORM_PLAYABLE`, `PLAY_KNOWN`, `DISCARD_USELESS`, `DISCARD_OLDEST`. If the last line holds no token, it searches the whole reply in the same order. If there is still none, it returns `DISCARD_OLDEST`. The extract also holds the observation parsers that the builder uses. The example file shows two prompts of 1,147 and 1,158 characters; the call logs of the hosted runs record 438 to 444 prompt tokens for the first call of each episode.

### SMAC (`single_type/smac.py`)

The prompt is written in Chinese and was sent in Chinese. The builder is `get_tactical_plan`. The health values are the hit points of each unit divided by 100 (0 for a dead unit), printed as percentages.

```python
prompt = (
    f"你是星际争霸 II 战术指挥官。地图: {map_name}。"
    f"我方 {n_agents} 单位 vs 敌方 {n_enemies} 单位。\n"
    f"我方血量: {[f'{h:.0%}' for h in ally_health]}\n"
    f"敌方血量: {[f'{h:.0%}' for h in enemy_health]}\n\n"
    "可用战术: focus_fire / spread_fire / kite / retreat / advance\n"
    "DSL 组合: 'A ; B' 顺序; 'A | B' 并行\n"
    "只输出 DSL 表达式（不超过两个战术）："
)
```

The parser `parse_smac_tactics_v2` removes brackets and keeps only the text before the first `;`. It splits that text at `|` and takes, for each part, the first tactic name that occurs in it. The executor gives the unit with index i the tactic `tactics[i % len(tactics)]`. A reply with no tactic name gives `focus_fire`. The call logs of the hosted runs record 184 (3m), 214 (8m) and 224 (MMM) prompt tokens per call.

### MAgent (`single_type/magent.py`)

The builder is `build_magent_prompt`. It counts the living red and blue units, averages their positions into two centroids and prints the distance between the centroids. The positions come from the underlying environment through `get_positions`.

```python
return (
    "You are a meta-planner for the RED team in MAgent battle (a multi-agent skirmish).\n"
    f"Red alive: {n_red}, Blue alive: {n_blue}, step: {step}/{max_cycles}.\n"
    f"Red centroid: {red_centroid}, Blue centroid: {blue_centroid}, centroid_dist: {dist:.1f}.\n"
    "Pick exactly ONE meta-task for the team this phase:\n"
    "  ATTACK_FORWARD - move toward nearest enemy and attack adjacent ones (default offensive)\n"
    "  HOLD_POSITION  - stay and only attack adjacent enemies (when outnumbering blue locally)\n"
    "  SPREAD_OUT     - spread away from allies; attack adjacent enemies (avoid clustering)\n"
    "  RETREAT        - hold (a pacifist last resort; rarely useful)\n"
    "Heuristic: if red >= blue and dist < 5 → ATTACK_FORWARD; if red < blue → SPREAD_OUT (avoid being surrounded).\n"
    "Output ONLY the meta-task token. Example: ATTACK_FORWARD\n"
    "Answer:"
)
```

The parser `parse_magent_meta_v2` upper-cases the reply and returns the first choice that occurs in it; a reply with none gives `ATTACK_FORWARD`. The choices are held in a set, so a reply that names two choices is resolved in an order that is not fixed. The call logs of the hosted runs record 273 to 290 prompt tokens per call. The example file shows the prompt at step 0 with the real initial positions (759 characters).

### Overcooked (`single_type/overcooked.py`)

The arm that calls the model is `llm_task`. The prompt is built by `build_plan_prompt`. The pot status comes from the soup object on the first pot, and the reach strings come from a breadth-first search of the chef's mapper over the four facility types.

```python
return (
    f"You are an Overcooked task planner. Layout: {layout}.\n"
    f"Pot status: {pot_summary}.\n"
    f"Chef 0 holds: {held[0] if len(held) > 0 else 'empty'}. {reach0}\n"
    f"Chef 1 holds: {held[1] if len(held) > 1 else 'empty'}. {reach1}\n"
    "Note: if a chef has reach=False for a facility, they CANNOT do "
    "tasks needing it directly; they must drop items on a shared counter "
    "for the other chef to pick up (handover).\n"
    "Pick the best next high-level task for each chef.\n"
    "Available tasks: get_onion, put_onion, cook, get_dish, plate_soup, "
    "serve_soup, drop_held_item, stay.\n"
    "Output ONLY two task names separated by ' | ' (chef_0_task | chef_1_task).\n"
    "Example: get_onion | get_dish\n"
    "Answer:"
)
```

The parser `parse_plan` removes brackets, reads the last non-empty line, keeps the text before the first `;`, splits at `|`, drops a `chef_k:` prefix, and keeps the first word of each part if it is one of the eight task names. One valid name is repeated for the second chef. If there is none, the pair is `get_onion | get_onion`, and the call is counted as a parse fallback. The task pair is applied to the chefs for the next 20 steps, and a fixed executor resolves it with the safety rules of `task_resolve.py` and carries it out. The run records keep, per episode, the task log of the first 40 changes, the number of calls (20), the number of parse fallbacks and the summed latency; the runs with several keys also keep the number of attempts and API failures. The cap and the server of the main run are not recorded in the archive; the code default for the cap is 120 tokens. The example file shows the prompt of the first step in two layouts.

### Not included

Two other builders exist in the scripts and are not in `single_type/`: `build_hanabi_prompt` in `run_hanabi_sweep.py` and the role-based `build_overcooked_prompt_v2`. The reported runs do not call them. The Hanabi sweep, the temperature-controlled router and the substitution check call `build_hanabi_prompt_v2`, and the `llm_task` arm of the Overcooked deployment script calls `build_plan_prompt`. The prompt lengths in the call logs (Hanabi 438 to 444 tokens for the 1,147 and 1,158 characters of the builder above) agree with the builder that I include.

## 7. HLA transfer

The transfer experiment uses gpt-oss-120b as a sampled prior over macro actions, not as a strategy selector. The verbatim extract is `single_type/hla.py`, taken from `sample_prior` and its helpers in `data/hla_transfer/audit/llm_client.py`; the address and the keys are left out. The rendering in `single_type/examples/hla.txt` uses the real code with placeholders for the native texts and with the available actions and the action history of one recorded decision. It also lists the recorded counts of that decision.

Each decision draws M = 10 independent completions at temperature 1.0 with `max_tokens` 700 and a timeout of 90 seconds. A failed call is retried up to five times with growing waits and keys that rotate. The reply is read from `content` only. One episode has about 16 decisions (1,642 decisions in 100 episodes on the partition map and 1,581 on the ring map), so about 160 calls. The records show about 743 to 753 prompt tokens and 135 to 197 completion tokens per call, and 18 retries and 4 parse failures in 32,230 calls.

The system message is the game rules of the native stack (`native_prompts[0][0]`). The user message is the native situation text followed by three fixed parts:

```python
acted = ("Actions you have already taken, in order: " + chosen_so_far + ".\n") \
    if chosen_so_far else ""
user = (situation + "\n" + acted +
        "Available actions right now: " + ", ".join(available) + ".\n" +
        "Reply with exactly one action, copied verbatim from the available list. "
        "No explanation, no punctuation, nothing else.")
```

A reply is parsed by an exact, case-insensitive match against the available actions, then by a unique substring match, then by the earliest mention; a reply that matches nothing is a parse failure and leaves the counts unchanged. The counts are smoothed with an additive constant of 0.5, and the log of the smoothed frequency is the prior that enters the native combination `prior - prob_base`; the largest value selects the macro action.

The native rules text and situation text come from the HLA stack, which is not part of this release, and I could not recover them, so I cannot show a complete prompt.

## 8. Examples index

All files are in `examples/`. Each starts with a header (domain, cell, arm file, the record, generation settings, system instruction) followed by `=== USER PROMPT (verbatim) ===`, `=== RAW COMPLETION (verbatim) ===` and `=== PARSED STRATEGY ===`. All come from gpt-oss-120b, episode 0, first stored call. The user prompt and the raw completion are copied from the record. For the bare-prompt arms the system instruction is derived from the code (Section 1) because the record does not store it; for the task-stating arms it is the system message stored in the record.

| File | Family and encoding |
| --- | --- |
| `battle_semantic.txt`, `battle_traits8_semantic.txt` | MAgent battle, base four types and eight trait types |
| `combined_semantic.txt`, `combined_numeric.txt`, `combined_opaque.txt`, `combined_permuted.txt`, `combined_relabel.txt`, `combined_swapdesc.txt` | MAgent combined arms, one per encoding |
| `combined_nshot1_single.txt`, `combined_nshot1_hypothesis_first.txt`, `combined_deliberate.txt` | combined arms, one-shot and the two other modes |
| `combined_traits8_semantic.txt` | combined arms, eight trait types |
| `sandbox_semantic.txt`, `sandbox_numeric.txt`, `sandbox_opaque.txt`, `sandbox_permuted.txt`, `sandbox_relabel.txt`, `sandbox_swapdesc.txt` | rock-paper-scissors sandbox, one per encoding |
| `sandbox_nshot1.txt` | sandbox, one example per seen type |
| `briefing_bare_single_nshot0.txt`, `briefing_bare_single_nshot1.txt`, `briefing_bare_hypothesis_first_nshot0.txt`, `briefing_bare_hypothesis_first_nshot1.txt` | briefing line, bare prompt |
| `briefing_bare_newword_heldout_single_nshot0.txt` | bare prompt, held-out type with the new wordings |
| `briefing_prefix_single_nshot0.txt`, `briefing_diag_semantic_single_nshot0.txt` | briefing followed by the play counts; play counts only |
| `taskstating_single_nshot0.txt`, `taskstating_hypothesis_first_nshot0.txt`, `taskstating_single_nshot1.txt`, `taskstating_single_nshot42.txt` | briefing line, task-stating prompt |
| `taskstating_single_shuffled_pool.txt`, `taskstating_newword_heldout_single_nshot0.txt`, `taskstating_classic_semantic_single_nshot0.txt` | task-stating prompt with shuffled pool, new wordings on a held-out type, and the three-strategy sandbox |

The files of the single-type slice are in `single_type/examples/`. Each starts with a line that says how it was rendered, and each ends with parsing examples whose replies I wrote by hand.

| File | Content |
| --- | --- |
| `hanabi.txt` | Hanabi-Small, two prompts (no playable partner card, and one playable partner card) |
| `smac.txt` | SMAC, first-step prompts for 3m and MMM |
| `magent.txt` | MAgent battle, prompt at step 0 with the real initial positions and a prompt at step 99 |
| `overcooked.txt` | Overcooked, first-step prompts for cramped_room and forced_coordination |
| `hla.txt` | HLA transfer, system and user message with placeholders for the native texts, and the recorded counts of one decision |
