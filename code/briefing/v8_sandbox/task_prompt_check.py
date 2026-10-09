"""Local checks (no server): v2 prompt renders and token counts, the v2 parser and call log on stub replies,
per-episode shuffle reproducibility, the human answer key against the calibration locks, and text_bow_matched on 10 seeds.

usage (from code/briefing/v8_sandbox, PYTHONPATH=../v8_lib:.):  python task_prompt_check.py --out <dir>"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from v8lib.runner import run_deployment

from dry_run_prompts import StubClient, count_tokens
from run_sandbox import build_parser, env_factory, parse_range, setup
from sandbox.game import ACTIONS, beats, trait_types
from sandbox.prompt_v2 import nshot_rows

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("LLM_API_BASE", "http://stub.invalid/v1")
PRIMARY = ["--family", "traits", "--M", "18", "--K", "10", "--sharpness", "0.8", "--reward-noise", "0", "--held-out-set", "react",
           "--seeds", "0-99,120-319", "--calib-seeds", "100-119", "--encoding", "briefing", "--wording", "newword"]
RENDERS = {
    "briefing_single_N0": PRIMARY + ["--arm", "llm:stub:single", "--prompt-version", "v2", "--nshot", "0"],
    "briefing_hypothesis_first_N0": PRIMARY + ["--arm", "llm:stub:hypothesis_first", "--prompt-version", "v2", "--nshot", "0"],
    "briefing_single_N1": PRIMARY + ["--arm", "llm:stub:single", "--prompt-version", "v2", "--nshot", "1", "--example-order", "shuffled"],
    "briefing_single_N42": PRIMARY + ["--arm", "llm:stub:single", "--prompt-version", "v2", "--nshot", "3", "--example-order", "shuffled"],
    "briefing_single_N0_shuffled_pool": PRIMARY + ["--arm", "llm:stub:single", "--prompt-version", "v2", "--nshot", "0", "--pool-order", "shuffled"],
    "classic_M6_semantic_single_N0": ["--M", "6", "--K", "3", "--sharpness", "0.8", "--reward-noise", "0", "--encoding", "semantic",
                                      "--seeds", "0-99,120-319", "--calib-seeds", "100-119", "--arm", "llm:stub:single", "--prompt-version", "v2"],
    "react8_heldout_single_N30": ["--family", "traits", "--M", "18", "--K", "10", "--held-out-set", "react8", "--deploy-types", "heldout",
                                  "--seeds", "0-99,120-319", "--calib-seeds", "100-119", "--encoding", "briefing", "--wording", "newword",
                                  "--arm", "llm:stub:single", "--prompt-version", "v2", "--nshot", "3", "--example-order", "shuffled"],
}


def run_with(argv: list[str], T: int, client):
    args = build_parser().parse_args(argv)
    cfg, calib, encoder, sel, lock = setup(args, client=client)
    recs = run_deployment(env_factory(cfg, sel), sel, parse_range(args.seeds), T, encoder=encoder, log_path=None, calibration_seeds=calib)
    return recs, sel, lock


def renders(out: Path) -> dict:
    report, text = {}, []
    for name, argv in RENDERS.items():
        stub = StubClient()
        recs, _, _ = run_with(argv, 100 if "N42" in name or "N30" in name else 5, stub)
        toks = [count_tokens(call["messages"][1]["content"]) for call in stub.calls]
        sys_c = count_tokens(stub.calls[0]["messages"][0]["content"])
        sys_toks = sys_c["tiktoken_o200k"]
        user = [t["tiktoken_o200k"] for t in toks]
        conservative = max(max(t["tiktoken_o200k"], t["chars_div4"]) for t in toks) + max(sys_toks, sys_c["chars_div4"])
        report[name] = {"prompts": len(user), "user_tokens_o200k_max": max(user), "user_tokens_o200k_mean": float(np.mean(user)),
                        "system_tokens_o200k": sys_toks, "max_prompt_conservative": conservative,
                        "budget_max_with_1024_completion": conservative + 200 + 1024,
                        "budget_max_with_4096_completion": conservative + 200 + 4096}
        text.append(f"===== {name}  (user {user[0]} o200k tokens; system {sys_toks}; max over {len(user)} prompts {max(user)})\n"
                    f"[system]\n{stub.calls[0]['messages'][0]['content']}\n[user]\n{stub.calls[0]['messages'][1]['content']}\n")
    (out / "prompts_v2_examples.txt").write_text("\n".join(text), encoding="utf-8")
    return report


class ScriptedClient:
    """Returns the given replies in order (each a dict of message fields), with usage and finish_reason."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        r = dict(self.replies.pop(0))
        finish = r.pop("finish_reason", "stop")
        msg = SimpleNamespace(**r)
        usage = SimpleNamespace(prompt_tokens=500, completion_tokens=42, total_tokens=542, completion_tokens_details=SimpleNamespace(reasoning_tokens=30))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], usage=usage, id="x", system_fingerprint="fp")


def parser_tests() -> dict:
    cases = [
        ("bare name", [{"content": "OUTPACE"}], "OUTPACE", "unique_name", None),
        ("final", [{"content": "Final: SWITCH_MO"}], "SWITCH_MO", "final", None),
        ("final markdown", [{"content": "The opponent copies me.\n**Final:** `MIRROR_BEAT`."}], "MIRROR_BEAT", "final", None),
        ("last final wins", [{"content": "Final: ROCK\nOn reflection, Final: PAPER"}], "PAPER", "final", None),
        ("names before final", [{"content": "ROCK and MIRROR_BEAT would lose. Final: OUTPACE"}], "OUTPACE", "final", None),
        ("final then invalid -> unique fallback", [{"content": "Final: Default: ROCK"}], "ROCK", "unique_name", None),
        ("lowercase final name", [{"content": "final: paper"}], "PAPER", "final", None),
        ("gpt-oss reasoning_content", [{"content": "Final: LAG_R", "reasoning_content": "They throw rock then paper."}], "LAG_R", "final", "reasoning_content"),
        ("vllm reasoning field", [{"content": "Final: LAG_S", "reasoning": "rock then paper"}], "LAG_S", "final", "reasoning"),
        ("qwen think tags", [{"content": "<think>\nThe opponent copies; Final: ROCK? no.\n</think>\n\nFinal: MIRROR_BEAT"}], "MIRROR_BEAT", "final", "think_tags"),
        ("think tags without opener", [{"content": "copies me, so beat my last.\n</think>\n\nFinal: MIRROR_BEAT"}], "MIRROR_BEAT", "final", "think_tags"),
        ("truncated think then retry", [{"content": "<think>\nlong...", "finish_reason": "length"}, {"content": "Final: OUTPACE"}], "OUTPACE", "final", "think_tags"),
        ("two names no final -> retry", [{"content": "ROCK or PAPER"}, {"content": "PAPER"}], "PAPER", "unique_name", None),
        ("three failures -> default", [{"content": ""}, {"content": "ROCK PAPER"}, {"content": "Final: none"}], "ROCK", None, None),
    ]
    out = {}
    for name, replies, want, rule, field in cases:
        client = ScriptedClient(replies)
        recs, _, _ = run_with(PRIMARY + ["--arm", "llm:stub:single", "--prompt-version", "v2", "--seeds", "0", "--T", "1"], 1, client)
        p = recs[0]["provenance"][0]
        e = p["extra"]
        got_field = next((c.get("reasoning_field") for c in e["calls"] if c.get("reasoning_field")), None)
        ok = p["tau"] == want and e["parse_rule"] == rule and got_field == field and len(e["calls"]) == len(replies)
        ok = ok and all(c["prompt_tokens"] == 500 and c["completion_tokens"] == 42 and c["reasoning_tokens"] == 30 and c["latency_s"] >= 0 for c in e["calls"])
        out[name] = {"ok": bool(ok), "tau": p["tau"], "parse_rule": e["parse_rule"], "reasoning_field": got_field, "calls": len(e["calls"]),
                     "finish_reasons": [c["finish_reason"] for c in e["calls"]], "fallback": p["fallback"]}
    return out


def shuffle_tests() -> dict:
    a, _, _ = run_with(RENDERS["briefing_single_N0_shuffled_pool"], 6, StubClient())
    b, _, _ = run_with(RENDERS["briefing_single_N0_shuffled_pool"], 6, StubClient())
    c, _, _ = run_with(RENDERS["briefing_single_N1"], 6, StubClient())
    listings = [r["provenance"][0]["extra"]["pool_listing"] for r in a]
    perms = [r["provenance"][0]["extra"]["example_permutation"] for r in c]
    return {
        "pool_listing_reproducible": listings == [r["provenance"][0]["extra"]["pool_listing"] for r in b],
        "pool_listings_distinct_over_6_episodes": len({tuple(x) for x in listings}),
        "pool_order_seeds": [r["provenance"][0]["extra"]["pool_order_seed"] for r in a],
        "episode_seeds": [r["provenance"][0]["extra"]["episode_seed"] for r in a],
        "record_seeds": [r["seed"] for r in a],
        "example_permutations_distinct_over_6_episodes": len({tuple(x) for x in perms}),
        "n_examples_N1": c[0]["provenance"][0]["extra"]["n_examples"],
    }


# ---------------------------------------------------------------------------------------------------------------
# Human answer key. HAND_KEY is what a careful reader of the briefing and the v2 docs picks.
# DOC_READING re-implements each v2 doc from its English text only (not from game.py); `reader_choice` scores every doc
# against the briefing's opponent read literally (s = 1, my throw in round 5 is ROCK, flip after round 15).
HAND_KEY = {
    "habit-stable": lambda f: beats(f),
    "habit-flip": lambda f: {"ROCK": "LAG_R", "PAPER": "LAG_P", "SCISSORS": "LAG_S"}[f],
    "beat_last-stable": lambda f: "OUTPACE",
    "copy_last-stable": lambda f: "MIRROR_BEAT",
    "beat_last-flip": lambda f: "SWITCH_OM",
    "copy_last-flip": lambda f: "SWITCH_MO",
}


def _wb(x):  # "whatever beats x"
    return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[x]


DOC_READING = {
    "ROCK": lambda prev, r: "rock",
    "PAPER": lambda prev, r: "paper",
    "SCISSORS": lambda prev, r: "scissors",
    "OUTPACE": lambda prev, r: _wb(_wb(prev)),               # whatever beats the throw that beats your own previous throw
    "MIRROR_BEAT": lambda prev, r: _wb(prev),                # whatever beats your own previous throw
    "LAG_R": lambda prev, r: "paper" if r <= 15 else "scissors",
    "LAG_P": lambda prev, r: "scissors" if r <= 15 else "rock",
    "LAG_S": lambda prev, r: "rock" if r <= 15 else "paper",
    "SWITCH_OM": lambda prev, r: _wb(_wb(prev)) if r <= 15 else _wb(prev),
    "SWITCH_MO": lambda prev, r: _wb(prev) if r <= 15 else _wb(_wb(prev)),
}


def reader_choice(typ: str, pool: list[str]) -> tuple[str, dict]:
    parts = typ.split("-")
    fav, react, timing = parts[0].lower(), parts[1], parts[2] if len(parts) > 2 else "stable"
    scores = {}
    for tau in pool:
        prev, total = "rock", 0
        for r in range(6, 31):  # 1-based rounds 6..30; "after the halfway point" = rounds 16..30
            second = timing == "flip" and r > 15
            f, mode = fav, react
            if second:
                f = _wb(fav) if react == "habit" else fav
                mode = {"habit": "habit", "beat_last": "copy_last", "copy_last": "beat_last"}[react]
            opp = f if mode == "habit" else (_wb(prev) if mode == "beat_last" else prev)
            mine = DOC_READING[tau](prev, r)
            total += 1 if _wb(opp) == mine else (-1 if _wb(mine) == opp else 0)
            prev = mine
        scores[tau] = total
    best = max(scores.values())
    winners = [t for t, v in scores.items() if v == best]
    return (winners[0] if len(winners) == 1 else "TIE:" + "/".join(winners)), scores


def answer_key() -> dict:
    out = {}
    for M, K in [(18, 10), (9, 5)]:
        pool = list(DOC_READING)[:K]
        locks = {}
        kinds = ["react", "diag", "react8"] if M == 18 else ["react", "diag"]
        for kind in kinds:
            for noise in ("0", "0.5"):
                path = ROOT / ("calib_0b" if kind == "react8" else "calib") / f"lock_sandbox-traits-M{M}-K{K}-sharp0.8-noise{noise}-{kind}.json"
                if path.exists():
                    locks[f"{kind}/noise{noise}"] = json.loads(path.read_text(encoding="utf-8"))["best_response"]
                else:  # server tree: rebuild the lock exactly as run_sandbox does
                    argv = ["--family", "traits", "--M", str(M), "--K", str(K), "--reward-noise", noise, "--held-out-set", kind,
                            "--calib-seeds", "100-119", "--arm", "fixed:ROCK"]
                    locks[f"{kind}/noise{noise}(rebuilt)"] = setup(build_parser().parse_args(argv))[4]["best_response"]
        rows = []
        for typ in trait_types(M):
            fav, react = typ.split("-")[:2]
            timing = typ.split("-")[2] if M == 18 else "stable"
            hand = HAND_KEY[f"{react}-{timing}"](fav)
            reader, scores = reader_choice(typ, pool)
            lock_vals = {k: v[typ] for k, v in locks.items()}
            rows.append({"type": typ, "hand_key": hand, "doc_reader": reader, "doc_reader_margin": sorted(scores.values())[-1] - sorted(scores.values())[-2],
                         "lock": lock_vals, "agree": hand == reader and all(v == hand for v in lock_vals.values())})
        out[f"M{M}"] = {"rows": rows, "all_agree": all(r["agree"] for r in rows)}
    return out


def matched_sanity(out_dir: Path) -> dict:
    """text_bow_matched N1 / N42 (and text_bow on the full table) on the primary cell, seeds 0-9; capture against the
    fixed arms of the all-types deployment on the same 10 seeds."""

    cells = ROOT / "results" / "final" / "cells"
    nonllm = cells / "sandbox-traits-M18-K10-sharp0.8-noise0-react-nonllm"
    fixed = {}
    for f in sorted(nonllm.glob("fixed__*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
        fixed[f.stem.split("__", 1)[1]] = {r["seed"]: r["total_reward"] for r in rows}
    seeds = list(range(10))
    F = np.array([[fixed[t][s] for s in seeds] for t in fixed])
    v_star, omax = F.mean(axis=1).max(), F.max(axis=0)
    h_d = omax.mean() - v_star
    res = {"seeds": seeds, "V_star_10": float(v_star), "H_D_10": float(h_d), "best_fixed_10": list(fixed)[int(F.mean(axis=1).argmax())]}
    for arm in ["text_bow_matched:1", "text_bow_matched:3", "text_embed_matched:1", "text_embed_matched:3", "text_bow"]:
        recs, sel, lock = run_with(PRIMARY + ["--arm", arm, "--seeds", "0-9", "--T", "10"], 10, None)
        A = np.array([r["total_reward"] for r in recs])
        if arm == "text_bow":
            n_rows = len(lock["text_rows"])
        else:
            cfg = setup(build_parser().parse_args(PRIMARY + ["--arm", "fixed:ROCK"]))[0]
            n_rows = len(nshot_rows(cfg, lock["best_response"], cfg.held_out, int(arm.split(":")[1])))
        res[arm] = {"capture_10": float((A.mean() - v_star) / h_d), "mean_10": float(A.mean()), "train_rows": n_rows,
                    "train_accuracy": getattr(sel, "train_accuracy", None), "choices": [r["provenance"][0]["tau"] for r in recs],
                    "types": [r["type"] for r in recs]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tasks_0c" / "validation"))
    ap.add_argument("--skip-matched", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"renders": renders(out), "parser": parser_tests(), "shuffle": shuffle_tests(), "answer_key": answer_key()}
    if not a.skip_matched:
        report["matched_sanity"] = matched_sanity(out)
    (out / "task_prompt_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"renders": report["renders"], "parser_all_ok": all(v["ok"] for v in report["parser"].values()),
                      "parser_failures": {k: v for k, v in report["parser"].items() if not v["ok"]}, "shuffle": report["shuffle"],
                      "answer_key_all_agree": {k: v["all_agree"] for k, v in report["answer_key"].items()},
                      "matched": {k: (v["capture_10"] if isinstance(v, dict) and "capture_10" in v else v) for k, v in report.get("matched_sanity", {}).items() if k not in ("seeds",)}},
                     indent=1))


if __name__ == "__main__":
    main()
