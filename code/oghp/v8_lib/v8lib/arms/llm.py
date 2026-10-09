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
    ) -> None:
        super().__init__()
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


class LLMInitLinUCB(LinUCB):
    """Compatibility import; actual implementation lives in contextual.py."""

    def __init__(self, *args, **kwargs):
        from v8lib.arms.contextual import LLMInitLinUCB as Impl

        self.__class__ = Impl
        Impl.__init__(self, *args, **kwargs)
