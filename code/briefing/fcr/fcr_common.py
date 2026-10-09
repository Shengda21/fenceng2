"""Shared pieces of the FCR experiment: repository paths, record loading, the reference conventions of the main analysis
(V*, H_D, capture, TIE_TOL), calibration settings read from the locks, and the labelled rows of the LLM N-shot prompts.

The sandbox modules the FCR needs (briefing.py, game.py, prompt_v2.py) are imported from
code/briefing/v8_sandbox, with code/briefing/v8_lib, which that package imports; SOURCES.json holds their hashes, which
run_all.py checks. Data are read from the repository root given by --root; outputs go to its analysis/briefing.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent  # code/briefing/fcr
for _p in (HERE.parent / "v8_sandbox", HERE.parent / "v8_lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sandbox.briefing import (  # noqa: E402
    HELD_OUT_SETS,
    PHRASES,
    all_renderings,
    axis_keys,
    briefing_for,
    held_out_set_name,
    render,
    resolve_held_out_set,
    variants_for_seed,
)
from sandbox.game import FAVOURITES, REACTIVITIES, TIMINGS, parse_trait, strategy_pool, trait_types  # noqa: E402
from sandbox.prompt_v2 import nshot_rows  # noqa: E402

RELEASE_DEFAULT = HERE.parents[2]  # the repository root
FCR_ROOT = HERE  # holds calib/, the calibration settings
ANALYSIS = Path("analysis") / "briefing"  # outputs, relative to the repository root
PREREG = Path("preregistration") / "briefing" / "PREREG_FCR.md"

# Conventions of the main analysis (code/briefing/v8_sandbox/analyze_briefing.py)
HEADLINE_N = {"all": 100, "heldout": 300}
TIE_TOL = 0.5
PAYS_MARGIN = 0.5
ALPHA = 0.05
B_DEFAULT = 10_000
PREFIX_READERS = ("scripted", "fewshot", "plastic", "linucb", "lints", "ctxucb", "mucb", "ppo")
TEXT_READERS = ("scripted_text", "text_bow", "text_embed", "text_linucb")
FREE_BANDITS = ("linucb", "lints", "ctxucb", "mucb", "text_linucb")
CALIB_SEEDS = list(range(100, 120))  # every briefing lock: calibration_seeds 100-119
CELL_RE = re.compile(
    r"(sandbox-traits-M(\d+)-K(\d+)-sharp([\d.]+)-noise([\d.]+))-(react8|react|diag)-(nonllm|briefing_prefix|briefing|semantic)(?:-(sameword|newword))?$"
)

TRAITS = {9: ("favourite", "reactivity"), 18: ("favourite", "reactivity", "timing")}
TRAIT_VALUES = {"favourite": tuple(FAVOURITES), "reactivity": tuple(REACTIVITIES), "timing": tuple(TIMINGS)}
SETTINGS = ("M9-react2", "M9-diag3", "M18-react4", "M18-diag6", "M18-react8")


def traits_of(type_name: str, M: int) -> tuple:
    f, r, t = parse_trait(type_name)
    return (f, r, t)[: len(TRAITS[int(M)])]


def type_of(z, M: int) -> str:
    return "-".join(z[: len(TRAITS[int(M)])])


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------------------------- records


def parse_arm(fname: str) -> dict:
    """Arm metadata from a record file name."""

    base = re.sub(r"\.jsonl(?:\.gz)?$", "", fname)
    if base.startswith("fixed__"):
        return {"arm": "fixed", "tau": base[len("fixed__"):], "key": base}
    if base.startswith("fewshot__"):
        return {"arm": "fewshot", "N": int(base[len("fewshot__"):]), "key": base}
    m = re.match(r"llm__(.+?)__(.+?)__nshot-?(\d+)$", base)
    if m:
        return {"arm": "llm", "model": m.group(1), "mode": m.group(2), "nshot": int(m.group(3)), "key": base}
    m = re.match(r"llm__(.+?)__(.+?)__nshot-?(\d+)__([A-Za-z0-9.\-]+)$", base)
    if m:
        return {"arm": "llm", "model": m.group(1), "mode": m.group(2), "nshot": int(m.group(3)), "variant": m.group(4), "key": base}
    m = re.match(r"(text_bow_matched|text_embed_matched)__nshot(\d+)$", base)
    if m:
        return {"arm": m.group(1), "nshot": int(m.group(2)), "key": base}
    return {"arm": base, "key": base}


def read_raw(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_rows(path: Path) -> list[dict]:
    """One dict per episode: seed, type, value, chosen strategy, fallback."""

    out = []
    for r in read_raw(path):
        prov = r.get("provenance") or []
        out.append({
            "seed": int(r["seed"]),
            "type": r["type"],
            "value": float(r["total_reward"]),
            "tau": prov[0].get("tau") if prov else (r.get("windows") or [{}])[0].get("tau"),
            "fallback": bool(r.get("fallback_count", 0)),
        })
    return out


def by_seed(rows: list[dict], field: str = "value") -> dict:
    d = {}
    for r in rows:
        d.setdefault(r["seed"], r[field])
    return d


def arm_files(cell_dir: Path) -> list[Path]:
    return sorted(list(cell_dir.glob("*.jsonl")) + list(cell_dir.glob("*.jsonl.gz")))


class Reference:
    """Reference as in analyze_briefing.Reference: the headline seeds are the first HEADLINE_N seeds below 1000 shared by
    every fixed arm; V* is the best fixed mean; H_D is the mean per-seed maximum over fixed arms minus V*."""

    def __init__(self, nonllm_dir: Path, held_out: list[str], headline_n: int):
        fixed = OrderedDict()
        for f in arm_files(nonllm_dir):
            meta = parse_arm(f.name)
            if meta["arm"] == "fixed":
                fixed[meta["tau"]] = read_rows(f)
        seeds = sorted(set.intersection(*[set(by_seed(v)) for v in fixed.values()]))
        self.seeds = [s for s in seeds if s < 1000][:headline_n]
        self.taus = list(fixed)
        self.F = np.array([[by_seed(v)[s] for s in self.seeds] for v in fixed.values()])
        types = by_seed(next(iter(fixed.values())), "type")
        self.types = [types[s] for s in self.seeds]
        self.means = self.F.mean(axis=1)
        self.best_i = int(np.argmax(self.means))
        self.V_star = float(self.means[self.best_i])
        self.Omax = self.F.max(axis=0)
        self.Omin = self.F.min(axis=0)
        self.Fstar = self.F[self.best_i]
        self.H_D = float(self.Omax.mean() - self.V_star)
        self.held_out = set(held_out)
        self.pos = {s: i for i, s in enumerate(self.seeds)}
        self.tau_index = {t: i for i, t in enumerate(self.taus)}
        self.per_type = OrderedDict()
        for th in sorted(set(self.types)):
            idx = [i for i, t in enumerate(self.types) if t == th]
            m = self.F[:, idx].mean(axis=1)
            best = float(m.max())
            self.per_type[th] = {
                "n": len(idx), "held_out": th in self.held_out, "best_tau": self.taus[int(np.argmax(m))],
                "good_set": [self.taus[i] for i in range(len(self.taus)) if m[i] >= best - TIE_TOL],
            }

    def replay(self, choices: list[str]) -> np.ndarray:
        """Exact replay: the return of a one-decision selector on a seed is the return of the fixed strategy it chose."""

        return np.array([self.F[self.tau_index[c], i] for i, c in enumerate(choices)])


# ----------------------------------------------------------------------------------------------- calibration


class CalibSetting:
    """What calibration fixes for one (M, held-out set): seen types, text rows (seen types x phrasings 0-2), the lock's
    per-type fixed-strategy values and best responses. Locks for sigma 0 and 0.5 and for both deployments carry identical
    text rows, value tables and best responses for the same (M, held-out set) (checked in survey_checks.py)."""

    def __init__(self, root: Path, name: str):
        self.name = name
        M_txt, set_name = name.split("-", 1)
        self.M = int(M_txt[1:])
        self.K = 5 if self.M == 9 else 10
        self.set_name = set_name
        kind = {"react2": "react", "react4": "react", "diag3": "diag", "diag6": "diag", "react8": "react8"}[set_name]
        self.kind = kind
        deploy = "heldout" if kind == "react8" else "all"
        path = root / "data" / "briefing" / "locks" / deploy / f"lock_sandbox-traits-M{self.M}-K{self.K}-sharp0.8-noise0-{kind}.json"
        lock = json.loads(path.read_text(encoding="utf-8"))
        self.lock_path = str(path.relative_to(root)).replace("\\", "/")
        self.held_out = list(lock["held_out"])
        assert self.held_out == resolve_held_out_set(kind, self.M)
        self.types = trait_types(self.M)
        self.seen = [t for t in self.types if t not in set(self.held_out)]
        self.pool = strategy_pool(self.K, "traits")
        self.best_response = dict(lock["best_response"])  # held-out entries are never read before calibration is fixed
        self.values_seen = {t: dict(lock["value_table"][t]) for t in self.seen}
        self.text_rows = [dict(r) for r in lock["text_rows"]]
        assert {r["type"] for r in self.text_rows} == set(self.seen)
        assert all(0 <= v <= 2 for r in self.text_rows for v in r["variants"])

    def rows_full(self) -> list[dict]:
        """Every calibration briefing the existing text readers use: each seen type in every combination of phrasings 0-2."""

        return [{"type": r["type"], "text": r["text"], "variants": list(r["variants"])} for r in self.text_rows]

    def rows_n_bare(self) -> list[dict]:
        """The labelled rows of the bare-prompt N=1 arms (run_sandbox.setup, prompt v1): for each seen type, in canonical type
        order, the briefing of calibration seed calib_seeds[0] = 100; seen types are always briefed in phrasings 0-2."""

        out = []
        for t in self.seen:
            v = variants_for_seed(t, CALIB_SEEDS[0], False)
            out.append({"type": t, "text": render(t, v), "variants": v})
        return out

    def rows_n_task(self) -> list[dict]:
        """The labelled rows of the task-prompt N=1 arms (prompt_v2.nshot_rows with one rendering per seen type)."""

        cfg = SimpleNamespace(family="traits", M=self.M)
        return [{"type": r["type"], "text": r["text"], "variants": list(r["variants"])}
                for r in nshot_rows(cfg, self.best_response, self.held_out, 1)]

    def rows(self, condition: str) -> list[dict]:
        return {"full": self.rows_full, "n_bare": self.rows_n_bare, "n_task": self.rows_n_task}[condition]()


def setting_name(M: int, kind: str) -> str:
    return f"M{int(M)}-{held_out_set_name(kind, int(M))}"


def load_embed_cache(root: Path) -> dict:
    path = root / "code" / "briefing" / "v8_sandbox" / "sandbox" / "embed_cache_bge-small-en-v1.5.npz"
    data = np.load(path, allow_pickle=False)
    return {str(k): np.asarray(v, dtype=float) for k, v in zip(data["keys"], data["vecs"])}


def pct(a, q) -> float:
    return float(np.percentile(a, q))


def boot_p(samples: np.ndarray, threshold: float) -> float:
    """One-sided bootstrap p-value of 'statistic <= threshold'."""

    return float((np.sum(samples <= threshold) + 1) / (len(samples) + 1))


def holm(pvals: dict, alpha: float = ALPHA) -> dict:
    keys = sorted(pvals, key=lambda k: (pvals[k], k))
    m = len(keys)
    out, alive = {}, True
    for rank, k in enumerate(keys):
        alive = alive and pvals[k] <= alpha / (m - rank)
        out[k] = alive
    return out
