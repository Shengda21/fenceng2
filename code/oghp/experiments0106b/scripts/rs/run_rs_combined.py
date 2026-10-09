#!/usr/bin/env python3
"""Run MAgent combined_arms regime-switching deployment experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import string
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0] / "v8_lib"))

from scripts.diagnostic._diag_common import save_rows
from scripts.rs.calibrate_rs_combined import file_hashes, select_types
from scripts.rs.combined_env import FEATURE_NAMES, CombinedArmsRS, feature_fn, redact_hidden_context, semantic_template

# One-line mechanics descriptions of the RED pool, shown to the LLM selector (the pool definition is part of the
# environment; every arm shares it).
TACTIC_DOCS = {
    "ALL_ATTACK": "every unit closes on the enemy formation and attacks whatever is adjacent or in range",
    "HOLD_LINE": "melee units hold a line in front, ranged units stay behind them and fire at anything within range; nobody advances",
    "RANGED_FIRST": "all units advance and prioritise the enemy's ranged units as targets",
    "MELEE_FIRST": "all units advance and prioritise the enemy's melee units as targets",
    "SCREEN_KITE": "melee units form a screen while ranged units fire and step back to keep their distance",
    "FLANK_RANGED": "ranged units swing to the sides to shoot past the enemy's melee line while melee units screen",
}
from scripts.rs.combined_types import POOL_SETS, semantic_descriptor
from scripts.rs.probes_rs_common import coerce_manifest, validate_disjoint

from v8lib.arms import (
    ContextUCB,
    FewShotClassifierSelector,
    FixedSelector,
    LLMSelector,
    LinTS,
    LinUCB,
    MUCB,
    PLASTICPolicySelector,
    PPOScheduler,
    RandomSelector,
    ScriptedDetector,
    TypeOracleSelector,
)


def main() -> None:
    args = parse_args()
    type_names = select_types(args.type_set, args.M, args.blue)
    pool = list(POOL_SETS[args.pool])
    lock = load_lock(args.lock)
    if lock is None:
        raise SystemExit("deployment requires --lock produced by calibrate_rs_combined.py")
    lock_hash = hash_file(args.lock)
    validate_lock(args, lock, type_names, pool)
    encoder = CombinedEncoder(args.encoding, type_names, args.permutation_seed, keep_numeric=True, feature_names=FEATURE_NAMES)
    selector = build_selector(args, lock, type_names, pool)
    env_cfg = {
        "map_size": args.map_size,
        "max_cycles": args.max_cycles,
        "n_melee": args.n_melee,
        "n_ranged": args.n_ranged,
        "type_set": type_names,
        "pool": tuple(pool),
        "default": args.default,
        "k": args.k,
        "switch": schedule_to_switch(args.schedule),
        "delta_scale": args.delta_scale,
        "held_out": args.held_out,
        "force_fake": args.fake,
        "encoder": encoder,
    }
    prompt = prompt_template(type_names, args.type_hints, args.encoding, getattr(encoder, "type_mapping", {}))
    if isinstance(selector, LLMSelector):
        selector.prompt_template = prompt
        examples, example_meta = nshot_examples(lock, args.nshot, encoder)
        selector.n_shot_examples = examples
    else:
        example_meta = []
    selector.reset_run(0)
    rows = []
    manifest = coerce_manifest(CombinedArmsRS, lock.get("deployment_manifest", []), env_cfg)
    if args.T > len(manifest):
        raise SystemExit(f"--T={args.T} exceeds locked deployment manifest length {len(manifest)}")
    manifest = manifest[: args.T]
    for episode_index, spec in enumerate(manifest):
        env = CombinedArmsRS(
            **{**env_cfg, "forced_type": spec.type1, "forced_type2": spec.type2, "switch_step": spec.switch_step}
        )
        try:
            ctx = env.reset(spec.seed, episode_index=episode_index)
            selector_ctx = ctx if selector.reads_type_label else redact_hidden_context(ctx)
            selector.reset_episode(selector_ctx)
            decisions = []
            provenance = []
            total = float(env._prefix_reward)
            done = env._done
            while not done:
                selector_ctx = ctx if selector.reads_type_label else redact_hidden_context(ctx)
                tau = selector.select(selector_ctx)
                prov = dict(selector.last_provenance or {})
                prov["episode_index"] = episode_index
                prov["seed"] = spec.seed
                decisions.append(
                    {
                        "episode_index": episode_index,
                        "seed": spec.seed,
                        "step": env.step,
                        "tau": tau,
                        "fallback": bool(prov.get("fallback", tau not in pool)),
                    }
                )
                provenance.append(prov)
                next_ctx, reward, done, info = env.step_window(tau)
                total += float(reward)
                reward_scope = "episode" if args.schedule == "single" else "window"
                update_reward = float(total) if reward_scope == "episode" else float(reward)
                info = {
                    **info,
                    "episode_index": episode_index,
                    "seed": spec.seed,
                    "reward_scope": reward_scope,
                    "prefix_reward": float(env._prefix_reward),
                    "window_reward": float(reward),
                    "episode_reward": float(total),
                }
                selector.update(selector_ctx, tau, update_reward, bool(done), info)
                if decisions:
                    decisions[-1]["reward_scope"] = reward_scope
                    decisions[-1]["update_reward"] = update_reward
                    decisions[-1]["window_reward"] = float(reward)
                if next_ctx is not None:
                    ctx = next_ctx
            summary = env.state_summary()
            rows.append(
                {
                    "episode": episode_index,
                    "episode_index": episode_index,
                    "seed": spec.seed,
                    "manifest_key": spec.key,
                    "type": env.blue_type,
                    "type2": env.blue_type2,
                    "switch_step": spec.switch_step,
                    "decisions": decisions,
                    "total_reward": float(total),
                    "success": bool(total > 0),
                    "steps": int(env.step),
                    "provenance": provenance,
                    "selector_state": selector.state_summary(),
                    "red_alive_final": summary["red_alive"],
                    "blue_alive_final": summary["blue_alive"],
                    "red_melee_alive_final": summary["red_melee_alive"],
                    "red_ranged_alive_final": summary["red_ranged_alive"],
                    "blue_melee_alive_final": summary["blue_melee_alive"],
                    "blue_ranged_alive_final": summary["blue_ranged_alive"],
                    "prefix_action_trace": [x for x in env.action_trace if x["step"] < args.k],
                    "rng_trace": env.rng_trace,
                    "lock_hash": lock_hash,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "episode": episode_index,
                    "episode_index": episode_index,
                    "seed": spec.seed,
                    "manifest_key": spec.key,
                    "type": spec.type1,
                    "type2": spec.type2,
                    "switch_step": spec.switch_step,
                    "total_reward": None,
                    "success": False,
                    "decisions": [],
                    "provenance": [],
                    "selector_state": selector.state_summary(),
                    "lock_hash": lock_hash,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
    meta = {
        "args": json_safe(vars(args)),
        "types": type_names,
        "feature_names": FEATURE_NAMES,
        "lock_hash": lock_hash,
        "nshot_examples": example_meta,
        "file_hashes": file_hashes(),
    }
    save_rows(rows, args.out, meta=meta)
    print(f"wrote {args.out}")


def build_selector(args, lock: dict[str, Any] | None, type_names: list[str], pool: list[str] | None = None):
    arm = args.arm
    pool = list(pool or POOL_SETS[args.pool])
    if arm == "random":
        return RandomSelector(seed=0)
    if arm.startswith("fixed:"):
        return FixedSelector(arm.split(":", 1)[1])
    if arm == "type_oracle":
        require_lock(lock, arm)
        return TypeOracleSelector(lock["best_response_table"])
    if arm == "scripted":
        return ScriptedDetector(lambda ctx: scripted_rule(ctx, pool, args.default))
    if arm == "mucb":
        return MUCB(pool=pool)
    if arm == "linucb":
        return LinUCB()
    if arm == "lints":
        return LinTS()
    if arm == "ctxucb":
        require_lock(lock, arm)
        sel = ContextUCB()
        sel.pool = pool
        sel.fit_bins(encoded_feature_matrix(lock, type_names, args))
        return sel
    if arm.startswith("fewshot:"):
        require_lock(lock, arm)
        n = int(arm.split(":", 1)[1])
        sel = FewShotClassifierSelector(n, lock["best_response_table"], model="knn", seed=0)
        x = [row["features"] for row in lock.get("feature_rows", [])]
        y = [row["type"] for row in lock.get("feature_rows", [])]
        if x and y:
            sel.fit(x, y)
        return sel
    if arm == "plastic":
        require_lock(lock, arm)
        return PLASTICPolicySelector(gaussian_callables(lock["plastic_type_models"]), lock["best_response_table"], value_table=lock.get("value_table"))
    if arm == "ppo":
        require_lock(lock, arm)
        checkpoint = lock.get("ppo_checkpoint")
        expected_hash = lock.get("ppo_checkpoint_hash")
        if not checkpoint or not expected_hash:
            raise SystemExit("ppo requires a calibration lock with ppo_checkpoint and ppo_checkpoint_hash")
        lock_path = Path(lock.get("__lock_path", ".")).resolve()
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (lock_path.parent / checkpoint_path).resolve()
        if hash_file(checkpoint_path) != expected_hash:
            raise SystemExit("ppo checkpoint hash mismatch")
        return PPOScheduler.load(checkpoint_path)
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
    raise ValueError(f"unknown arm: {arm}")


def encoded_feature_matrix(lock: dict[str, Any], type_names: list[str], args) -> list[list[float]]:
    rows = lock.get("feature_rows", [])
    out = []
    for row in rows:
        features = np.asarray(row["features"], dtype=float).ravel()
        if args.encoding == "opaque":
            typ = str(row.get("type"))
            one_hot = np.zeros(max(1, len(type_names)), dtype=float)
            if typ in type_names:
                one_hot[type_names.index(typ)] = 1.0
            features = np.concatenate([one_hot, features])
        out.append([float(x) for x in features])
    if not out:
        raise ValueError("ctxucb requires calibration feature_rows in the lock")
    return out


def scripted_rule(ctx, pool: list[str], default: str) -> str:
    vals = dict(zip(FEATURE_NAMES, np.asarray(ctx.features, dtype=float).ravel()[: len(FEATURE_NAMES)]))
    if vals.get("blue_ranged_fire_fraction", 0.0) > 0.35 and "RANGED_FIRST" in pool:
        return "RANGED_FIRST"
    if vals.get("blue_ranged_retreat_fraction", 0.0) > 0.25 and "FLANK_RANGED" in pool:
        return "FLANK_RANGED"
    if vals.get("blue_melee_advance_fraction", 0.0) > 0.45 and "MELEE_FIRST" in pool:
        return "MELEE_FIRST"
    return default if default in pool else pool[0]


def gaussian_callables(models: dict[str, dict[str, Any]]):
    out = {}
    for typ, params in models.items():
        mean = np.asarray(params["mean"], dtype=float)
        inv = np.asarray(params.get("inv_cov"), dtype=float)
        logdet = float(params.get("logdet", 0.0))

        def loglike(ctx, mean=mean, inv=inv, logdet=logdet):
            x = np.asarray(ctx.features, dtype=float).ravel()
            if x.size != mean.size:
                x = x[: mean.size] if x.size > mean.size else np.pad(x, (0, mean.size - x.size))
            d = x - mean
            return -0.5 * float(d @ inv @ d) - 0.5 * logdet

        out[typ] = loglike
    return out


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


def nshot_examples(lock: dict[str, Any] | None, n: int, encoder: "CombinedEncoder") -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
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
            meta.append({"seed": row.get("seed"), "type": typ, "tau": tau, "prompt": f"Observation: {text}\nStrategy: {tau}"})
    return out, meta


class CombinedEncoder:
    def __init__(self, mode: str, type_names: list[str], permutation_seed: int, keep_numeric: bool = False, feature_names: list[str] | None = None) -> None:
        self.mode = mode
        self.type_names = list(type_names)
        self.permutation_seed = int(permutation_seed)
        self.keep_numeric = bool(keep_numeric)
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.type_mapping = _derangement(self.type_names, self.permutation_seed)
        self.tokens = {typ: _opaque_token(typ, self.permutation_seed, i) for i, typ in enumerate(self.type_names)}

    def __call__(self, raw: dict, ctx_partial):
        features = np.asarray(raw.get("_locked_features", feature_fn(raw)), dtype=float).ravel()
        if self.mode in {"semantic", "relabel", "swapdesc"}:
            text = self.render_text(raw)
            out_features = features
        elif self.mode == "numeric":
            text = self.render_text(raw)
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
            features = np.asarray(raw.get("_locked_features", feature_fn(raw)), dtype=float).ravel()
            return "features: " + ", ".join(f"{name}={float(value):.6g}" for name, value in zip(self.feature_names, features))
        if self.mode == "opaque":
            typ = str(raw.get("type_label", raw.get("type")))
            return f"opponent signature: {self.tokens.get(typ, _opaque_token(typ, self.permutation_seed, 0))}"
        mapping = self.type_mapping if self.mode == "permuted" else None
        return semantic_template(raw, mapping)


def validate_lock(args, lock: dict[str, Any], type_names: list[str], pool: list[str]) -> None:
    expected = {
        "map_size": args.map_size,
        "max_cycles": args.max_cycles,
        "n_melee": args.n_melee,
        "n_ranged": args.n_ranged,
        "pool": pool,
        "default": args.default,
        "k": args.k,
        "schedule": args.schedule,
        "types": type_names,
        "delta_scale": args.delta_scale,
        "held_out": args.held_out,
        "feature_names": FEATURE_NAMES,
        # The observation encoding is an intervention on the selector's input (E3), not an environment field:
        # every encoding runs against the same lock; the run records its own encoding in meta and rows.
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise SystemExit(f"lock validation failed for {key}: lock={lock.get(key)!r} run={value!r}")
    validate_disjoint(lock.get("calibration_seeds", []), coerce_manifest(CombinedArmsRS, lock.get("deployment_manifest", []), {}))
    current_hashes = file_hashes()
    for path, digest in lock.get("file_hashes", {}).items():
        if current_hashes.get(path) != digest:
            raise SystemExit(f"lock validation failed for source hash {path}")


def load_lock(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lock = json.loads(path.read_text(encoding="utf-8"))
    lock["__lock_path"] = str(path)
    return lock


def require_lock(lock, arm: str) -> None:
    if lock is None:
        raise SystemExit(f"{arm} requires --lock produced by calibrate_rs_combined.py")


def hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_extra_body(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--extra-body-json must decode to an object")
    return value


def _label_map(pool: list[str]) -> dict[str, str]:
    return {tau: f"OPTION_{chr(ord('A') + i)}" for i, tau in enumerate(pool)}


def _pool_descriptions(encoding: str, pool: list[str], seed: int, docs: dict[str, str]) -> dict[str, str]:
    if encoding == "swapdesc":
        description_owner = _derangement(pool, seed)
        return {tau: docs.get(description_owner[tau], "") for tau in pool}
    return {tau: docs[tau] for tau in pool if tau in docs}

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


def _opaque_token(typ: str, seed: int, idx: int) -> str:
    digest = hashlib.sha256(f"combined:{seed}:{idx}:{typ}".encode("utf-8")).digest()
    alphabet = string.ascii_uppercase
    return "C" + "".join(alphabet[b % len(alphabet)] for b in digest[:8])


def _aliases(type_names: list[str]) -> dict[str, str]:
    return {typ: f"Pattern {chr(ord('A') + i)}" for i, typ in enumerate(type_names)}


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--type-set", choices=["base4", "traits8"], default="base4")
    p.add_argument("--M", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--blue", choices=["scripted", "random"], default="scripted")
    p.add_argument("--encoding", choices=["semantic", "numeric", "opaque", "permuted", "relabel", "swapdesc"], default="semantic")
    p.add_argument("--type-hints", dest="type_hints", action="store_true")
    p.add_argument("--no-type-hints", dest="type_hints", action="store_false")
    p.set_defaults(type_hints=False)
    p.add_argument("--arm", required=True)
    p.add_argument("--schedule", choices=["single", "mid", "block20"], default="single")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--delta-scale", type=float, default=1.0)
    p.add_argument("--held-out")
    p.add_argument("--lock", type=Path, default=ROOT / "rs_combined_lock.json")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "rs_combined" / "run.json")
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--nshot", type=int, default=0)
    p.add_argument("--permutation-seed", type=int, default=0)
    p.add_argument("--map-size", type=int, default=16)
    p.add_argument("--max-cycles", type=int, default=120)
    p.add_argument("--n-melee", type=int, default=6)
    p.add_argument("--n-ranged", type=int, default=6)
    p.add_argument("--pool", choices=sorted(POOL_SETS), default="combined_pool6")
    p.add_argument("--default", default="HOLD_LINE")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--llm-seed", type=int)
    p.add_argument("--extra-body-json")
    return p.parse_args()


def schedule_to_switch(schedule: str) -> str | None:
    return None if schedule == "single" else schedule


if __name__ == "__main__":
    main()
