"""Leave-one-combination-out cross-validation inside calibration (no deployment seed, held-out combination, phrasing 3-5,
deployment label or return is read).

For each calibration setting (M, held-out set) and each data condition:
  full    every calibration briefing the existing text readers use (seen types x phrasings 0-2);
  n_bare  the labelled rows of the bare-prompt N=1 arms (one per seen type);
  n_task  the labelled rows of the task-prompt N=1 arms (one per seen type);
one seen trait combination u is held out at a time; the trait classifiers are fitted on the rows of the other seen types and
scored on u's rows of the same condition (27 or 9 renderings under `full`, u's one row under `n_*`); the composer is fitted
on the other seen types' calibration values and scored on u with u's true traits.

Selection rules:
  classifier C, per setting x condition x extractor x trait: highest mean LOCO accuracy over held-out combinations; ties by
    lower mean LOCO log-loss of the true trait value; then by |log10 C| (closest to the default C = 1).
  ridge alpha, per setting x composer form: highest LOCO counter accuracy (argmax Q equals the lock's best response of u);
    ties by lower mean LOCO regret V_u(best) - V_u(chosen); then lower mean squared error of Q on u; then larger alpha.
The composer form is not selected here: `main` is the primary form and `pairwise` the sensitivity form; both are scored.

usage: python code/calib_cv.py [--root <repository root>]  -> calib/loco_cv.json, calib/chosen.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

import fcr_common as fc
from fcr_model import FCR, Composer, TraitReader, _logistic, _tfidf

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
ALPHA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
EXTRACTORS = ("tfidf", "embed")
CONDITIONS = ("full", "n_bare", "n_task")
FORMS = ("main", "pairwise")


def loco_classifier(s: fc.CalibSetting, rows: list[dict], extractor: str, cache: dict | None) -> dict:
    traits = fc.TRAITS[s.M]
    out = OrderedDict()
    for C in C_GRID:
        per = OrderedDict()
        for u in s.seen:
            tr = [r for r in rows if r["type"] != u]
            te = [r for r in rows if r["type"] == u]
            rd = TraitReader(s.M, extractor, {t: C for t in traits}, cache).fit([r["text"] for r in tr], [r["type"] for r in tr])
            zs = rd.predict([r["text"] for r in te])
            ll = rd.log_loss([r["text"] for r in te], [r["type"] for r in te])
            true = fc.traits_of(u, s.M)
            per[u] = {t: float(np.mean([z[j] == true[j] for z in zs])) for j, t in enumerate(traits)}
            per[u]["tuple"] = float(np.mean([tuple(z) == true for z in zs]))
            per[u]["log_loss"] = ll
        out[str(C)] = {
            "per_combination": per,
            "mean_accuracy": {t: float(np.mean([per[u][t] for u in s.seen])) for t in traits},
            "mean_tuple_accuracy": float(np.mean([per[u]["tuple"] for u in s.seen])),
            "mean_log_loss": {t: float(np.mean([per[u]["log_loss"][t] for u in s.seen])) for t in traits},
        }
    chosen = {}
    for t in traits:
        chosen[t] = min(C_GRID, key=lambda C: (-round(out[str(C)]["mean_accuracy"][t], 12), round(out[str(C)]["mean_log_loss"][t], 12),
                                               abs(math.log10(C)), C))
    return {"grid": out, "chosen_C": chosen}


def loco_composer(s: fc.CalibSetting, form: str) -> dict:
    out = OrderedDict()
    for alpha in ALPHA_GRID:
        per = OrderedDict()
        for u in s.seen:
            comp = Composer(s.M, form, alpha, s.pool).fit([t for t in s.seen if t != u], s.values_seen)
            q = comp.q([fc.traits_of(u, s.M)])[0]
            v = np.array([s.values_seen[u][tau] for tau in s.pool])
            pick = s.pool[int(np.argmax(q))]
            per[u] = {"chosen": pick, "best": s.best_response[u], "correct": pick == s.best_response[u],
                      "regret": float(v.max() - s.values_seen[u][pick]), "mse": float(np.mean((q - v) ** 2))}
        out[str(alpha)] = {"per_combination": per,
                           "counter_accuracy": float(np.mean([per[u]["correct"] for u in s.seen])),
                           "mean_regret": float(np.mean([per[u]["regret"] for u in s.seen])),
                           "mean_mse": float(np.mean([per[u]["mse"] for u in s.seen]))}
    alpha = min(ALPHA_GRID, key=lambda a: (-round(out[str(a)]["counter_accuracy"], 12), round(out[str(a)]["mean_regret"], 9),
                                           round(out[str(a)]["mean_mse"], 9), -a))
    fit_all = Composer(s.M, form, alpha, s.pool).fit(s.seen, s.values_seen)
    train_picks = fit_all.choose([fc.traits_of(t, s.M) for t in s.seen])
    return {"grid": out, "chosen_alpha": alpha,
            "train_counter_accuracy_seen": float(np.mean([p == s.best_response[t] for p, t in zip(train_picks, s.seen)])),
            "train_misses_seen": {t: p for p, t in zip(train_picks, s.seen) if p != s.best_response[t]}}


def loco_end_to_end(s: fc.CalibSetting, rows: list[dict], extractor: str, C: dict, form: str, alpha: float, cache) -> dict:
    traits = fc.TRAITS[s.M]
    per = OrderedDict()
    for u in s.seen:
        tr = [r for r in rows if r["type"] != u]
        te = [r for r in rows if r["type"] == u]
        model = FCR(s.M, extractor, C, form, alpha, s.pool, cache).fit(
            [r["text"] for r in tr], [r["type"] for r in tr], [t for t in s.seen if t != u], s.values_seen)
        zs, picks = model.read([r["text"] for r in te])
        true = fc.traits_of(u, s.M)
        v = s.values_seen[u]
        per[u] = {t: float(np.mean([z[j] == true[j] for z in zs])) for j, t in enumerate(traits)}
        per[u]["tuple"] = float(np.mean([tuple(z) == true for z in zs]))
        per[u]["counter"] = float(np.mean([p == s.best_response[u] for p in picks]))
        per[u]["regret"] = float(np.mean([max(v.values()) - v[p] for p in picks]))
    keys = list(traits) + ["tuple", "counter", "regret"]
    return {"per_combination": per, "mean": {k: float(np.mean([per[u][k] for u in s.seen])) for k in keys}}


def loco_whole_type(s: fc.CalibSetting, rows: list[dict]) -> dict:
    """Reference: the whole-type TF-IDF reader (text_bow: labels are best responses, C = 1) under the same LOCO."""

    per = OrderedDict()
    for u in s.seen:
        tr = [r for r in rows if r["type"] != u]
        te = [r for r in rows if r["type"] == u]
        vec = _tfidf()
        X = vec.fit_transform([r["text"] for r in tr])
        clf = _logistic(1.0).fit(X, [s.best_response[r["type"]] for r in tr])
        picks = [str(x) for x in clf.predict(vec.transform([r["text"] for r in te]))]
        per[u] = float(np.mean([p == s.best_response[u] for p in picks]))
    return {"per_combination": per, "counter_accuracy": float(np.mean(list(per.values())))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(fc.RELEASE_DEFAULT))
    ap.add_argument("--out-dir", default=str(fc.FCR_ROOT / "calib"))
    a = ap.parse_args()
    root, out_dir = Path(a.root), Path(a.out_dir)
    cache = fc.load_embed_cache(root)
    cv, chosen = OrderedDict(), OrderedDict()
    cv["grids"] = {"C": list(C_GRID), "alpha": list(ALPHA_GRID)}
    for name in fc.SETTINGS:
        s = fc.CalibSetting(root, name)
        entry = OrderedDict(meta={"M": s.M, "held_out_set": s.set_name, "lock": s.lock_path, "n_seen": len(s.seen),
                                  "seen": s.seen, "pool": s.pool})
        ch = OrderedDict(M=s.M, held_out_set=s.set_name, seen=s.seen, composer={}, classifier={})
        entry["composer"] = OrderedDict()
        for form in FORMS:
            entry["composer"][form] = loco_composer(s, form)
            ch["composer"][form] = {"alpha": entry["composer"][form]["chosen_alpha"]}
        entry["conditions"] = OrderedDict()
        for cond in CONDITIONS:
            rows = s.rows(cond)
            ce = OrderedDict(n_rows=len(rows), rows=[{"type": r["type"], "variants": r["variants"]} for r in rows],
                             whole_type_tfidf=loco_whole_type(s, rows), extractors=OrderedDict())
            ch["classifier"][cond] = {}
            for ext in EXTRACTORS:
                clf = loco_classifier(s, rows, ext, cache)
                e2e = OrderedDict()
                for form in FORMS:
                    e2e[form] = loco_end_to_end(s, rows, ext, clf["chosen_C"], form, ch["composer"][form]["alpha"], cache)
                ce["extractors"][ext] = {"classifier": clf, "end_to_end": e2e}
                ch["classifier"][cond][ext] = {"C": clf["chosen_C"]}
            entry["conditions"][cond] = ce
            print(name, cond, "whole-type", round(ce["whole_type_tfidf"]["counter_accuracy"], 3),
                  {ext: {f: {k: round(v, 3) for k, v in ce["extractors"][ext]["end_to_end"][f]["mean"].items()} for f in FORMS} for ext in EXTRACTORS},
                  {ext: ce["extractors"][ext]["classifier"]["chosen_C"] for ext in EXTRACTORS}, flush=True)
        print(name, "composer", {f: (entry["composer"][f]["chosen_alpha"], entry["composer"][f]["grid"][str(entry["composer"][f]["chosen_alpha"])]["counter_accuracy"],
                                     entry["composer"][f]["train_counter_accuracy_seen"]) for f in FORMS}, flush=True)
        cv[name] = entry
        chosen[name] = ch
    fc.write_json(out_dir / "loco_cv.json", cv)
    fc.write_json(out_dir / "chosen.json", chosen)
    print("written", out_dir / "loco_cv.json", out_dir / "chosen.json")


if __name__ == "__main__":
    main()
