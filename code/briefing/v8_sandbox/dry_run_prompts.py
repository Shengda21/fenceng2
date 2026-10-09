"""Render LLM prompts through the real runner with a stub client (no server) and count their tokens."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

from v8lib.runner import run_deployment

from run_sandbox import build_parser, env_factory, parse_range, setup


class StubClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="Stub reply.\nFinal: ROCK")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None, id=None, system_fingerprint=None)


def count_tokens(text: str) -> dict:
    try:
        import tiktoken

        tiktoken_count = len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        tiktoken_count = None
    return {"tiktoken_o200k": tiktoken_count, "chars_div4": (len(text) + 3) // 4}


def render(argv: list[str], T: int) -> list[list[dict]]:
    os.environ.setdefault("LLM_API_BASE", "http://stub.invalid/v1")
    args = build_parser().parse_args(argv)
    stub = StubClient()
    cfg, calib_seeds, encoder, selector, _ = setup(args, client=stub)
    run_deployment(env_factory(cfg, selector), selector, parse_range(args.seeds), T, encoder=encoder, log_path=None, calibration_seeds=calib_seeds)
    return [call["messages"] for call in stub.calls]


def prompt_tokens(messages: list[dict]) -> int:
    counts = [count_tokens(m["content"]) for m in messages]
    return max(sum(c["tiktoken_o200k"] or 0 for c in counts), sum(c["chars_div4"] for c in counts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--out", default=None)
    a, rest = parser.parse_known_args()
    messages = render(rest, a.T)
    report = []
    for msgs in messages:
        full = "\n".join(m["content"] for m in msgs)
        report.append({"system": msgs[0]["content"], "user": msgs[1]["content"], "tokens": count_tokens(full)})
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
