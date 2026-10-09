"""Embed every briefing the traits family can render (all types, seen and new wordings) into the selector cache."""

from __future__ import annotations

import time

from sandbox.briefing import all_renderings
from sandbox.game import trait_types
from sandbox.selectors import EMBED_CACHE, make_embedder


def all_briefings() -> list[str]:
    texts = []
    for M in (9, 18):
        for typ in trait_types(M):
            for offset in (0, 3):
                texts.extend(text for _, text in all_renderings(typ, offset=offset))
    return sorted(set(texts))


def main() -> None:
    texts = all_briefings()
    embedder = make_embedder()
    start = time.perf_counter()
    vecs = embedder.encode(texts)
    embedder.save(EMBED_CACHE)
    print({"texts": len(texts), "dim": int(vecs.shape[1]), "seconds": round(time.perf_counter() - start, 1), "cache": EMBED_CACHE})


if __name__ == "__main__":
    main()
