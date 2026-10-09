"""LLM-backed selectors using OpenAI-compatible chat completions."""

from __future__ import annotations

import os
import re
from time import perf_counter
from typing import Any

from v8lib.arms.contextual import LinUCB
from v8lib.arms.base import Selector
from v8lib.context import Context


class LLMSelector(Selector):
    """OpenAI-compatible chat selector; source: zero-shot/n-shot LLM prompting baseline."""

    role = "llm"

    def __init__(
        self,
        client=None,
        model: str | None = None,
        prompt_template: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 120,
        n_shot_examples: list[tuple[str, str]] | None = None,
        mode: str = "single",
        max_react_turns: int = 3,
        timeout_s: float = 60,
        retries: int = 3,
        extra_body: dict | None = None,
        seed: int | None = None,
        pool_descriptions: dict[str, str] | None = None,
        label_map: dict[str, str] | None = None,
        hypothesis_instruction: str | None = None,
    ) -> None:
        super().__init__()
        self.hypothesis_instruction = hypothesis_instruction
        self.pool_descriptions = dict(pool_descriptions or {})
        self.label_map = dict(label_map or {})
        labels = list(self.label_map.values())
        if len(set(labels)) != len(labels):
            raise ValueError("label_map values must be unique")
        for i, label in enumerate(labels):
            if any(i != j and label in other for j, other in enumerate(labels)):
                raise ValueError("label_map labels must not collide as substrings")
        self._last_prompt: str | None = None
        if mode == "react":
            mode = "deliberate"
        if mode not in {"single", "hypothesis_first", "deliberate"}:
            raise ValueError("mode must be single, hypothesis_first, or deliberate")
        self.client = client
        self.model = model or os.environ.get("LLM_MODEL", "model")
        self.prompt_template = prompt_template or "{text}\nPool: {pool}\nDefault: {default}"
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.n_shot_examples = list(n_shot_examples or [])
        self.mode = mode
        self.max_react_turns = int(max_react_turns)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.extra_body = extra_body or {}
        self.seed = seed
        self._episode_fallbacks = 0
        self.name = f"llm:{self.model}:{self.mode}"

    def reset_episode(self, ctx0: Context) -> None:
        self._episode_fallbacks = 0

    def _client(self):
        if self.client is not None:
            return self.client
        from openai import OpenAI

        self.client = OpenAI(
            base_url=os.environ.get("LLM_API_BASE"),
            api_key=os.environ.get("LLM_API_KEY"),
            timeout=self.timeout_s,
        )
        return self.client

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
            instruction = self.hypothesis_instruction or f"First write one sentence naming the inferred type. Then write Final: {final_token}."
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

    def _completion(self, messages: list[dict[str, str]]):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        return self._client().chat.completions.create(**kwargs)

    @staticmethod
    def _content(resp) -> str:
        return resp.choices[0].message.content or ""

    @staticmethod
    def _usage(resp) -> dict[str, int | None]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    @staticmethod
    def _response_id(resp):
        return getattr(resp, "id", None)

    @staticmethod
    def _fingerprint(resp):
        return getattr(resp, "system_fingerprint", None)

    @staticmethod
    def _add_usage(total: dict[str, int | None], usage: dict[str, int | None]) -> dict[str, int | None]:
        for key, value in usage.items():
            if value is None:
                total[key] = total.get(key)
            else:
                total[key] = int(total.get(key) or 0) + int(value)
        return total

    def _parse(self, text: str, pool: list[str]) -> tuple[str | None, list[str]]:
        hits = []
        for tau in pool:
            shown = self._shown(tau)
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(shown)}(?![A-Za-z0-9_])", text):
                hits.append((tau, shown))
        return (hits[0][0] if len(hits) == 1 else None), [shown for _, shown in hits]

    def _shown(self, tau: str) -> str:
        return self.label_map.get(tau, tau)

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

    def _audit_extra(
        self,
        attempts,
        source,
        raw,
        hits,
        usage,
        call_latencies,
        response_ids,
        fingerprints,
    ) -> dict[str, Any]:
        clean_ids = [x for x in response_ids if x is not None]
        clean_fingerprints = [x for x in fingerprints if x is not None]
        return {
            "attempts": attempts,
            "source": source,
            "raw": raw,
            "matches": hits,
            "prompt": self._last_prompt,
            "usage": usage,
            "call_latency_s": float(sum(call_latencies)),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "extra_body": self.extra_body,
            "label_map": self.label_map,
            "response_id": clean_ids[-1] if clean_ids else None,
            "response_ids": clean_ids,
            "system_fingerprint": clean_fingerprints[-1] if clean_fingerprints else None,
            "system_fingerprints": clean_fingerprints,
            "fallback_count_episode": self._episode_fallbacks + (1 if source == "give_up" else 0),
            "hypothesis": raw.splitlines()[0] if self.mode == "hypothesis_first" and raw else None,
        }


POOL_ORDER_OFFSET = 31_337_000
EXAMPLE_ORDER_OFFSET = 41_414_000
_FINAL_RE = re.compile(r"final(?:\s+(?:answer|strategy|choice))?\s*[:\uff1a]", re.IGNORECASE)
_NAME_AFTER_FINAL_RE = re.compile(r"[\s*`\"'\[\(]*([A-Za-z][A-Za-z0-9_]*)")


def split_think(text: str) -> tuple[str, str | None]:
    """Split a completion into (answer, thinking) when the server left `<think>...</think>` in the content.

    A completion that opened a think block and never closed it (truncated) has no answer."""

    text = text or ""
    if "</think>" in text:
        head, _, tail = text.rpartition("</think>")
        return tail.strip(), head.replace("<think>", "", 1).strip()
    if "<think>" in text:
        return "", text.replace("<think>", "", 1).strip()
    return text, None


class FramedLLMSelector(LLMSelector):
    """Framed prompt: the user message states the task (game, payoff rule, objective, that the pool lists the
    model's own strategies), the pool docs are in the first person, the default line is explained, N-shot examples sit after
    the pool under a header in the same format as the test item, and the reply is parsed on the name after the last
    "Final:" (the unique-name parser of `LLMSelector` is the fallback). Every completion call is logged with its tokens, latency,
    finish reason and reasoning text. Optional per-episode shuffles of the pool listing and the example order are seeded from
    the episode seed, which the runner passes in through `set_episode_seed`.

    `LLMSelector` itself is unchanged; its plain prompts never pass through this class."""

    def __init__(
        self,
        *,
        frame: str,
        pool_header: str,
        default_line: str,
        example_header: str,
        test_header: str,
        item_label: str,
        answer_label: str,
        instructions: dict[str, str],
        pool_order: str = "fixed",
        example_order: str = "fixed",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.mode not in instructions:
            raise ValueError(f"prompt v2 defines no instruction for mode {self.mode}")
        if pool_order not in {"fixed", "shuffled"} or example_order not in {"fixed", "shuffled"}:
            raise ValueError("pool_order and example_order must be fixed or shuffled")
        self.frame = frame
        self.pool_header = pool_header
        self.default_line = default_line
        self.example_header = example_header
        self.test_header = test_header
        self.item_label = item_label
        self.answer_label = answer_label
        self.instructions = dict(instructions)
        self.pool_order = pool_order
        self.example_order = example_order
        self.prompt_version = "v2"
        self._episode_seed: int | None = None
        self._calls: list[dict[str, Any]] = []
        self._parse_rule: str | None = None
        self._last_system: str | None = None
        self._pool_listing: list[str] | None = None
        self._example_permutation: list[int] | None = None

    def set_episode_seed(self, seed: int) -> None:
        self._episode_seed = int(seed)

    def _seed_for(self, offset: int) -> int:
        if self._episode_seed is None:
            raise RuntimeError("shuffled order needs the episode seed; build the env with run_sandbox.env_factory")
        return self._episode_seed + offset

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

    def _completion(self, messages: list[dict[str, str]]):
        start = perf_counter()
        try:
            resp = super()._completion(messages)
        except Exception as exc:
            self._calls.append({"latency_s": perf_counter() - start, "error": str(exc)})
            raise
        self._calls.append(self._call_record(resp, perf_counter() - start))
        return resp

    def _call_record(self, resp, latency_s: float) -> dict[str, Any]:
        choice = resp.choices[0]
        message = choice.message
        answer, think = split_think(getattr(message, "content", None) or "")
        reasoning, field = None, None
        for name in ("reasoning_content", "reasoning"):
            value = getattr(message, name, None)
            if value is None:
                value = (getattr(message, "model_extra", None) or {}).get(name)
            if value:
                reasoning, field = str(value), name
                break
        if reasoning is None and think is not None:
            reasoning, field = think, "think_tags"
        usage = self._usage(resp)
        details = getattr(getattr(resp, "usage", None), "completion_tokens_details", None)
        return {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details is not None else None,
            "latency_s": latency_s,
            "finish_reason": getattr(choice, "finish_reason", None),
            "content": answer,
            "reasoning": reasoning,
            "reasoning_field": field,
        }

    def _content(self, resp) -> str:
        return split_think(resp.choices[0].message.content or "")[0]

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

    def select(self, ctx: Context) -> str:
        self._calls = []
        self._parse_rule = None
        return super().select(ctx)

    def _audit_extra(self, attempts, source, raw, hits, usage, call_latencies, response_ids, fingerprints) -> dict[str, Any]:
        out = super()._audit_extra(attempts, source, raw, hits, usage, call_latencies, response_ids, fingerprints)
        out.update({
            "prompt_version": self.prompt_version,
            "system_prompt": self._last_system,
            "parse_rule": self._parse_rule if source == "model" else None,
            "calls": list(self._calls),
            "episode_seed": self._episode_seed,
            "pool_order": self.pool_order,
            "pool_order_seed": self._episode_seed + POOL_ORDER_OFFSET if self.pool_order == "shuffled" and self._episode_seed is not None else None,
            "pool_listing": self._pool_listing,
            "example_order": self.example_order if self.n_shot_examples else None,
            "example_order_seed": self._episode_seed + EXAMPLE_ORDER_OFFSET
            if self.n_shot_examples and self.example_order == "shuffled" and self._episode_seed is not None else None,
            "example_permutation": self._example_permutation,
            "n_examples": len(self.n_shot_examples),
            "timeout_s": self.timeout_s,
        })
        return out


class LLMInitLinUCB(LinUCB):
    """Compatibility import; actual implementation lives in contextual.py."""

    def __init__(self, *args, **kwargs):
        from v8lib.arms.contextual import LLMInitLinUCB as Impl

        self.__class__ = Impl
        Impl.__init__(self, *args, **kwargs)
