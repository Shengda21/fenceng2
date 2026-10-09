"""Deployment evaluation of the FCR-sym arms (the script refuses to run unless PREREG_SYM.md carries its frozen row).

FCR-sym = the FCR trait readers (code/briefing/fcr/fcr_model.py, settings code/briefing/fcr/calib/chosen.json) + the
symbolic composer of code/briefing/fcr_sym/sym_composer.py, which simulates every pool strategy against the scripted
opponent of the predicted trait tuple and picks the best. Arms:

  sym_full          full calibration rows, TF-IDF reader       primary FCR-sym-full
  sym_full_embed    full calibration rows, embedding reader    extractor sensitivity (FCR-sym-embed)
  sym_n_bare        bare-prompt 1/type rows, TF-IDF            primary FCR-sym-N (bare rows)
  sym_n_task        task-prompt 1/type rows, TF-IDF            primary FCR-sym-N (task rows)
  sym_n_bare_embed, sym_n_task_embed                           extractor sensitivities
  sym_oracle        TRUE traits into the symbolic composer     privileged diagnostic (trait-oracle-sym), never a baseline

Every deployment is first evaluated exactly as the FCR experiment did (code/briefing/fcr/evaluate.py), and the result is
compared with the published analysis/briefing/fcr.json. The FCR-sym arms are then replayed on the same headline seeds
and compared on the same bootstrap index matrices. Statistics, the pays criterion, the strongest non-LLM arm rule, the
primary/secondary split and Holm are those of the FCR evaluation; see PREREG_SYM.md.

usage: python code/briefing/fcr_sym/evaluate_sym.py [--root <repository root>] [--B 10000]  -> analysis/briefing/fcr_sym.json
       python code/briefing/fcr_sym/evaluate_sym.py --placebo   (machinery test: every FCR-sym arm is
                                               replaced by the FCR arm with the same reader and the main-effects ridge
                                               composer, and trait-oracle-sym by the FCR main-effects trait oracle; no
                                               FCR-sym choice is computed)  -> analysis/briefing/placebo/placebo_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # code/briefing/fcr_sym
FCR_DIR = HERE.parent / "fcr"
if str(FCR_DIR) not in sys.path:
    sys.path.insert(0, str(FCR_DIR))

import fcr_common as fc  # noqa: E402
import evaluate as fe  # noqa: E402  (code/briefing/fcr/evaluate.py)

from sym_composer import SIM_ROLLOUTS, SIM_SEED_BASE, TIE_EPS, SymbolicComposer, all_tuples  # noqa: E402

FCR_CHOSEN_SHA = "93328267e05e99cfd90a252c02e0fb11424c852cdd1ab517575190d327597332"  # hash of code/briefing/fcr/calib/chosen.json

SYM_ARMS = OrderedDict([  # arm -> (condition, extractor, FCR arm whose fitted trait reader it uses)
    ("sym_full", ("full", "tfidf", "fcr_full")),
    ("sym_full_embed", ("full", "embed", "fcr_full_embed")),
    ("sym_n_bare", ("n_bare", "tfidf", "fcr_n_bare")),
    ("sym_n_bare_embed", ("n_bare", "embed", "fcr_n_bare_embed")),
    ("sym_n_task", ("n_task", "tfidf", "fcr_n_task")),
    ("sym_n_task_embed", ("n_task", "embed", "fcr_n_task_embed")),
])
PRIMARY_SYM = ("sym_full", "sym_n_bare", "sym_n_task")
SYM_ORACLE = "sym_oracle"
SYM_FULL_VARIANTS = ("sym_full", "sym_full_embed")
SYM_N_VARIANTS = {"n_bare": ("sym_n_bare", "sym_n_bare_embed"), "n_task": ("sym_n_task", "sym_n_task_embed")}
FCR_TWINS = {  # FCR arms with the same trait reader (for the paired comparisons sym arm - FCR arm)
    "sym_full": ("fcr_full", "fcr_full_pairwise"), "sym_full_embed": ("fcr_full_embed",),
    "sym_n_bare": ("fcr_n_bare", "fcr_n_bare_pairwise"), "sym_n_bare_embed": ("fcr_n_bare_embed",),
    "sym_n_task": ("fcr_n_task", "fcr_n_task_pairwise"), "sym_n_task_embed": ("fcr_n_task_embed",),
    "sym_oracle": ("fcr_oracle", "fcr_oracle_pairwise"),
}
SAMPLE_MATCHED_SYM = OrderedDict([
    ("Qwen bare 1/type", ("briefing:llm__qwen3.8-27b__single__nshot1", "sym_n_bare")),
    ("Qwen task 1/type", ("task:llm__qwen3.8-27b__single__nshot1__v2", "sym_n_task")),
    ("120b task 1/type", ("task:llm__gpt-oss-120b__single__nshot1__v2", "sym_n_task")),
])
OUTCOME_TEXT = {
    "A": "the gate closes, and the LLM does not beat the best sample-matched cheap reader given the same labelled rows",
    "B": "the gate closes, but the LLM beats the best sample-matched cheap reader given the same labelled rows: an advantage in sample efficiency only",
    "C": "the LLM still pays against every cheap reader including FCR-sym, and trait-oracle-sym would close the gap: the shortfall is in reading the words",
    "D": "the LLM still pays against every cheap reader including FCR-sym, and beats trait-oracle-sym too: the shortfall is in composing the counter",
}
MOST_KNOWN = "all/sandbox-traits-M18-K10-sharp0.8-noise0-react/newword"
REACT4 = "heldout/sandbox-traits-M18-K10-sharp0.8-noise0-react/newword"
REACT8 = "heldout/sandbox-traits-M18-K10-sharp0.8-noise0-react8/newword"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jnorm(obj):
    """The object as it reads back from the JSON file (for exact comparisons with the published fcr.json)."""

    return json.loads(json.dumps(obj, default=float))


# ----------------------------------------------------------------------------------------------- freeze guard


def freeze_record(prereg: Path, chosen: Path) -> dict:
    """The freeze row and the hash of the FCR reader settings (pure data), as the FCR's own evaluate.py checks its
    calib/chosen.json. PREREG_SYM.md quotes the hash of code/sym_composer.py as it stood in the original layout; the
    composer's identity here is guarded by check_sources.py against SOURCES.json."""

    if not prereg.exists():
        raise SystemExit("PREREG_SYM.md is missing: the deployment evaluation runs only after the pre-registration is frozen")
    text = prereg.read_text(encoding="utf-8")
    m = re.search(r"^\|\s*(\d{4}-\d{2}-\d{2}T[0-9:.+\-]+)\s*\|\s*\*\*Frozen\*\*", text, flags=re.M)
    if not m:
        raise SystemExit("PREREG_SYM.md has no frozen row in its revision record: refusing to evaluate deployments")
    want = re.search(r"calib/fcr_chosen\.json`\s*sha256 `([0-9a-f]{64})`", text)
    have = sha(chosen)
    if not want or want.group(1) != have:
        raise SystemExit("calib/fcr_chosen.json does not match the sha256 recorded in PREREG_SYM.md")
    return {"frozen_utc": m.group(1), "chosen_sha256": have}


# ----------------------------------------------------------------------------------------------- one deployment


def comparator_files(dep: dict) -> OrderedDict:
    """The recorded arm results the FCR evaluation reads (same keys as code/briefing/fcr/evaluate.py)."""

    wording = dep["wording"]
    out = OrderedDict()
    for f in fc.arm_files(dep["nonllm"]):
        meta = fc.parse_arm(f.name)
        if meta["arm"] != "fixed":
            out[f"nonllm:{meta['key']}"] = f
    for f in fc.arm_files(dep["briefing"]):
        out[f"briefing:{fc.parse_arm(f.name)['key']}"] = f
    if dep["briefing_prefix"] is not None:
        for f in fc.arm_files(dep["briefing_prefix"]):
            meta = fc.parse_arm(f.name)
            if meta["arm"] == "llm":
                out[f"briefing_prefix:{meta['key']}"] = f
    if dep["task"] is not None:
        for f in fc.arm_files(dep["task"]):
            out[f"task:{fc.parse_arm(f.name)['key']}"] = f
    assert wording in ("sameword", "newword")
    return out


def evaluate_sym_deployment(root: Path, dep: dict, dep_no: int, models: dict, comp: SymbolicComposer, fdep: dict, B: int,
                            placebo: bool) -> OrderedDict:
    deploy, g, wording, M = dep["deploy"], dep["group"], dep["wording"], dep["M"]
    sname = fc.setting_name(M, dep["kind"])
    s: fc.CalibSetting = models[sname]["setting"]
    held = s.held_out
    ref = fc.Reference(dep["nonllm"], held, fc.HEADLINE_N[deploy])
    boot = fe.Boot(ref, B, dep_no)  # same seed and deployment number as the FCR evaluation: same index matrices
    traits = fc.TRAITS[M]
    texts = [fc.briefing_for(t, sd, held, wording)[0] for sd, t in zip(ref.seeds, ref.types)]
    held_idx = [i for i, t in enumerate(ref.types) if t in set(held)]
    seen_idx = [i for i, t in enumerate(ref.types) if t not in set(held)]
    tau_star = [s.best_response[t] for t in ref.types]
    good = [ref.per_type[t]["good_set"] for t in ref.types]
    true = [fc.traits_of(t, M) for t in ref.types]
    n = len(ref.seeds)

    values, choice, zhat = OrderedDict(), OrderedDict(), OrderedDict()
    for arm, model in models[sname]["arms"].items():  # FCR arms, exactly as the FCR evaluation computes them
        if arm in fe.ORACLE_ARMS:
            zs, picks = true, model.choose(true)
        else:
            zs, picks = model.read(texts)
        values[arm] = dict(zip(ref.seeds, ref.replay(picks).tolist()))
        choice[arm] = dict(zip(ref.seeds, picks))
        zhat[arm] = zs
    for arm, (_cond, _ext, reader_arm) in SYM_ARMS.items():
        zs = zhat[reader_arm]  # the FCR arm's fitted trait reader (identical condition, extractor and C)
        picks = models[sname]["arms"][reader_arm].composer.choose(zs) if placebo else comp.choose(zs)
        values[arm] = dict(zip(ref.seeds, ref.replay(picks).tolist()))
        choice[arm] = dict(zip(ref.seeds, picks))
        zhat[arm] = zs
    picks = models[sname]["arms"]["fcr_oracle"].choose(true) if placebo else comp.choose(true)
    values[SYM_ORACLE] = dict(zip(ref.seeds, ref.replay(picks).tolist()))
    choice[SYM_ORACLE] = dict(zip(ref.seeds, picks))
    zhat[SYM_ORACLE] = true
    files = comparator_files(dep)
    for ck, f in files.items():
        rows = fc.read_rows(f)
        values[ck] = fc.by_seed(rows)
        choice[ck] = fc.by_seed(rows, "tau")

    # the FCR arms' per-seed values must reproduce the FCR evaluation
    for arm in list(fe.FCR_ARMS) + list(fe.ORACLE_ARMS):
        assert abs(float(np.mean(list(values[arm].values()))) - fdep["arms"][arm]["mean"]) < 1e-9, arm

    # statistics of the FCR-sym arms (the FCR's metric set)
    sym_list = list(SYM_ARMS) + [SYM_ORACLE]
    arms = OrderedDict()
    for arm in sym_list:
        st = fe.arm_stats(ref, boot, values[arm])
        ch = choice[arm]
        zs = zhat[arm]
        right = [ch[ref.seeds[i]] == tau_star[i] for i in range(n)]
        tup = [tuple(zs[i]) == true[i] for i in range(n)]
        st["counter_accuracy"] = float(np.mean(right))
        st["hit_rate"] = float(np.mean([ch[ref.seeds[i]] in good[i] for i in range(n)]))
        acc = {t: float(np.mean([zs[i][j] == true[i][j] for i in range(n)])) for j, t in enumerate(traits)}
        acc["tuple"] = float(np.mean(tup))
        st["trait_accuracy"] = acc
        st["counter_accuracy_given_tuple_right"] = float(np.mean([r for r, t in zip(right, tup) if t])) if any(tup) else None
        st["counter_accuracy_given_tuple_wrong"] = float(np.mean([r for r, t in zip(right, tup) if not t])) if not all(tup) else None
        A = np.array([values[arm][sd] for sd in ref.seeds])
        strata = OrderedDict()
        for nm, idx in (("seen", seen_idx), ("held_out", held_idx)):
            if not idx:
                continue
            e = OrderedDict(n=len(idx), capture=fe.pooled_capture(ref, A, idx),
                            counter_accuracy=float(np.mean([right[i] for i in idx])),
                            hit_rate=float(np.mean([ch[ref.seeds[i]] in good[i] for i in idx])))
            for j, t in enumerate(traits):
                e[t] = float(np.mean([zs[i][j] == true[i][j] for i in idx]))
            e["tuple"] = float(np.mean([tup[i] for i in idx]))
            react_idx = [i for i in idx if true[i][1] != "habit"]
            if react_idx:  # reactive types: the counter depends on (reactivity, timing) only
                rt = [all(zs[i][j] == true[i][j] for j in range(1, len(traits))) for i in react_idx]
                e["reactive_types"] = OrderedDict(
                    n=len(react_idx), counter_accuracy=float(np.mean([right[i] for i in react_idx])),
                    reactivity_timing_joint=float(np.mean(rt)),
                    **{t: float(np.mean([zs[i][j] == true[i][j] for i in react_idx])) for j, t in enumerate(traits)})
            strata[nm] = e
        st["strata"] = strata
        st["choices_held_out"] = OrderedDict((t, dict(Counter(ch[ref.seeds[i]] for i in held_idx if ref.types[i] == t)))
                                             for t in held if any(ref.types[i] == t for i in held_idx))
        st["meta"] = {"arm": "fcr_sym", "variant": arm, **({"reader": SYM_ARMS[arm][2]} if arm in SYM_ARMS else {"reader": "true traits"})}
        arms[arm] = st

    # parsing vs composition, against trait-oracle-sym
    decomp = OrderedDict()
    for arm in SYM_ARMS:
        cats, loss = Counter(), Counter()
        for i, sd in enumerate(ref.seeds):
            tr = tuple(zhat[arm][i]) == true[i]
            ok = choice[arm][sd] == tau_star[i]
            c = ("traits right, counter right" if ok else "traits right, counter wrong (composition)") if tr else \
                ("traits wrong, counter right" if ok else "traits wrong, counter wrong (parsing)")
            cats[c] += 1
            loss[c] += float(ref.Omax[i] - values[arm][sd])
        tot = sum(loss.values())
        decomp[arm] = {"oracle_arm": SYM_ORACLE, "capture": arms[arm]["capture"], "oracle_capture": arms[SYM_ORACLE]["capture"],
                       "parsing_loss": arms[SYM_ORACLE]["capture"] - arms[arm]["capture"],
                       "composition_loss": 1.0 - arms[SYM_ORACLE]["capture"],
                       "episode_shares": {k: v / n for k, v in sorted(cats.items())},
                       "regret_shares": {k: (v / tot if tot > 0 else None) for k, v in sorted(loss.items())}}

    # strongest non-LLM arm: baseline candidates, + FCR (the FCR rule; must equal the FCR's choice), + FCR-sym
    farms = fdep["arms"]
    mean_of = lambda a: arms[a]["mean"] if a in arms else farms[a]["mean"]  # noqa: E731
    cands = [a for a in farms if (a.startswith("nonllm:") and farms[a]["meta"]["arm"] in fc.PREFIX_READERS)
             or (a.startswith("briefing:") and farms[a]["meta"]["arm"] in fc.TEXT_READERS)]
    cands = [a for a in cands if farms[a]["n"] == n]
    before = max(cands + list(fe.FCR_ARMS), key=mean_of)
    assert before == fdep["strongest"]["with_fcr"], (before, fdep["strongest"]["with_fcr"])
    after = max(cands + list(fe.FCR_ARMS) + list(SYM_ARMS), key=mean_of)  # ties: baselines, then FCR, then FCR-sym
    after_primary = max(cands + list(fe.FCR_ARMS) + list(PRIMARY_SYM), key=mean_of)
    sym_best = max(SYM_ARMS, key=mean_of)
    sym_full_star = max(SYM_FULL_VARIANTS, key=mean_of)
    rel_key = fdep["strongest"]["release"]
    bandit = fdep["strongest"]["online_bandit"]

    # paired comparisons: FCR-sym arms against the non-LLM comparators and their FCR twins
    pairs = OrderedDict()
    for arm in sym_list:
        others = ["briefing:text_bow", bandit, rel_key, before, "briefing:text_embed", "briefing:scripted_text"]
        others += list(FCR_TWINS[arm])
        if arm.startswith("sym_n_task"):
            others.append("task:text_bow_matched__nshot1")
        for other in OrderedDict.fromkeys(o for o in others if o):
            if other in values and other != arm:
                pairs[f"{arm} - {other}"] = fe.paired(ref, boot, values[arm], values[other])
        if arm != "sym_full":
            pairs[f"{arm} - sym_full"] = fe.paired(ref, boot, values[arm], values["sym_full"])

    # every LLM arm: verdict before (baselines + FCR) and after (baselines + FCR + FCR-sym)
    llm = OrderedDict()
    for ak, fl in fdep["llm"].items():
        bef = fl["updated_primary"]
        vs_sym = OrderedDict((x, fe.paired(ref, boot, values[ak], values[x])) for x in sym_list)
        m_lo, p_m = fl["margin_lo95"], fl["p_margin_le_bound"]

        def judged(target: str) -> dict:
            if target in SYM_ARMS:
                lo, p, src = vs_sym[target]["lo95"], vs_sym[target]["p_le0"], "fcr_sym"
            elif target == bef["strongest"]:
                lo, p, src = bef["vs_lo95"], bef["p_vs"], bef["source"]
            else:  # cannot occur: adding FCR-sym arms changes the strongest arm only to an FCR-sym arm
                v = fe.paired(ref, boot, values[ak], values[target])
                lo, p, src = v["lo95"], v["p_le0"], "recomputed"
            return {"strongest": target, "vs_lo95": lo, "p_vs": p, "source": src, "pays": fe.verdict(m_lo, lo), "p_pays": max(p_m, p)}

        aft = judged(after)
        aftp = judged(after_primary)
        llm[ak] = OrderedDict(
            n=fl["n"], capture=fl["capture"], capture_ci=fl["capture_ci"], margin_lo95=m_lo, p_margin_le_bound=p_m,
            p_margin_source=fl["p_margin_source"], counter_accuracy=fl["counter_accuracy"], hit_rate=fl["hit_rate"],
            release=fl["release"], before_with_fcr=bef, after_with_fcr_sym=aft, after_with_fcr_primary_sym=aftp,
            changed=aft["pays"] != bef["pays"], changed_vs_release=aft["pays"] != fl["release"].get("pays"),
            vs_sym=vs_sym,
            vs_fcr={x: {k: fl["vs"][x][k] for k in ("diff", "lo95", "p_le0", "capture_diff")} for x in fl["vs"]})

    comparators = OrderedDict()
    for a, st in farms.items():
        keep = {k: st[k] for k in ("n", "mean", "capture", "capture_ci", "capture_lo95", "margin_lo95", "counter_accuracy", "hit_rate")
                if k in st}
        keep["meta"] = st["meta"]
        if "trait_accuracy" in st:
            keep["trait_accuracy"] = st["trait_accuracy"]
        if "strata" in st:
            keep["strata"] = {nm: {k: v for k, v in e.items() if k in ("n", "capture", "counter_accuracy", "tuple")}
                              for nm, e in st["strata"].items()}
        if st.get("release"):
            keep["release"] = {k: st["release"].get(k) for k in ("capture", "capture_ci", "pays") if k in st["release"]}
        comparators[a] = keep

    return OrderedDict(
        meta=OrderedDict(**{k: fdep["meta"][k] for k in ("deploy", "group", "wording", "M", "K", "reward_noise", "kind", "held_out_set",
                                                          "held_out", "calibration_setting", "population", "primary", "label",
                                                          "briefing_check", "type_shares", "n_seen_episodes", "n_held_out_episodes")}),
        reference=fdep["reference"],
        strongest=OrderedDict(release=rel_key, before_with_fcr=before, after_with_fcr_sym=after,
                              after_with_fcr_primary_sym=after_primary, changed=after != before, online_bandit=bandit,
                              strongest_fcr_sym=sym_best, sym_full_star=sym_full_star, oracle_star=SYM_ORACLE,
                              fcr_full_star=fdep["strongest"]["fcr_full_star"], fcr_oracle_star=fdep["strongest"]["oracle_star"]),
        arms=arms, comparators=comparators, decomposition=decomp, paired=pairs, llm=llm)


# ----------------------------------------------------------------------------------------------- outcomes


def outcome_sym(fdep: dict, sdep: dict, ak: str, matched: str | None) -> dict:
    e = sdep["llm"].get(ak)
    if e is None:
        return {"available": False}
    aft = e["after_with_fcr_sym"]
    mean_of = lambda a: sdep["arms"][a]["mean"] if a in sdep["arms"] else fdep["arms"][a]["mean"]  # noqa: E731
    res = OrderedDict(available=True, release_pays=e["release"].get("pays"), before_pays=e["before_with_fcr"]["pays"],
                      after_pays=aft["pays"], strongest=aft["strongest"], vs_strongest_lo95=aft["vs_lo95"])
    if aft["pays"] != "yes":
        if e["release"].get("pays") != "yes":
            res["closed_by"] = "release arm"
        elif e["before_with_fcr"]["pays"] != "yes":
            res["closed_by"] = "FCR arm"
        else:
            res["closed_by"] = "FCR-sym arm" if aft["strongest"] in SYM_ARMS else "release arm"
        if matched is None:
            res["letter"] = "closed (zero-shot arm: no sample-matched reader)"
        else:
            variants = list(fe.N_VARIANTS[matched]) + list(SYM_N_VARIANTS[matched])
            nstar = max(variants, key=mean_of)  # ties: FCR variants first
            vs = fdep["llm"][ak]["vs"][nstar] if nstar in fe.FCR_ARMS else e["vs_sym"][nstar]
            nsym = max(SYM_N_VARIANTS[matched], key=mean_of)
            vsym = e["vs_sym"][nsym]
            res.update(n_star=nstar, llm_minus_n_star={k: vs[k] for k in ("diff", "lo95", "p_le0", "capture_diff")},
                       letter="B" if vs["lo95"] > 0 else "A", sym_n_star=nsym,
                       llm_minus_sym_n_star={k: vsym[k] for k in ("diff", "lo95", "p_le0", "capture_diff")},
                       letter_sym_n_only="B" if vsym["lo95"] > 0 else "A")
    else:
        vs = e["vs_sym"][SYM_ORACLE]
        res.update(oracle_star=SYM_ORACLE, llm_minus_oracle_star={k: vs[k] for k in ("diff", "lo95", "p_le0", "capture_diff")},
                   letter="D" if vs["lo95"] > 0 else "C")
    res["meaning"] = OUTCOME_TEXT.get(res["letter"])
    before = fe.outcome(fdep, ak, matched)
    res["letter_fcr_experiment"] = before.get("letter")
    return res


# ----------------------------------------------------------------------------------------------- checks


def composer_tables(composers: dict, models: dict) -> OrderedDict:
    """The composer's choice for every trait tuple of every game cell, compared with every lock's best responses (all types,
    seen and held out; held-out labels are read here only to report agreement)."""

    out = OrderedDict()
    for (M, K, sigma), comp in sorted(composers.items()):
        rows = OrderedDict()
        for z in all_tuples(M):
            r = comp.simulate(z)
            rows[r["type"]] = {"choice": r["choice"], "tied": r["tied"], "gap_best_second": r["gap_best_second"],
                               "means": r["means"], "max_rollout_sd": float(max(r["rollout_sd"].values())),
                               "max_dev_of_difference_across_rollouts": r["max_dev_of_difference_across_rollouts"]}
        agree = OrderedDict()
        for name, mm in models.items():
            s = mm["setting"]
            if s.M != M:
                continue
            bad = [t for t in rows if rows[t]["choice"] != s.best_response[t]]
            agree[name] = {"types": len(rows), "match_lock_best_response": len(rows) - len(bad), "mismatched": bad,
                           "held_out_types_match": sum(rows[t]["choice"] == s.best_response[t] for t in s.held_out)}
        out[f"M{M}-K{K}-noise{sigma:g}"] = {"settings": comp.describe(), "types": rows, "agreement_with_locks": agree}
    return out


def simulator_fidelity(deps: list[dict]) -> OrderedDict:
    """Replays every fixed arm on every headline seed of every distinct non-LLM cell with the composer's simulator (forced
    type = the recorded type, the recorded seed) and compares with the recorded return. A check of the simulator only:
    deployment seeds never enter the composer."""

    from sandbox.env import SandboxEnv

    out = OrderedDict()
    done = set()
    for dep in deps:
        key = (dep["deploy"], dep["group"])
        if key in done:
            continue
        done.add(key)
        held = fc.resolve_held_out_set(dep["kind"], dep["M"])
        ref = fc.Reference(dep["nonllm"], held, fc.HEADLINE_N[dep["deploy"]])
        cfg = SymbolicComposer(dep["M"], dep["K"], dep["noise"]).cfg
        dev, cnt = 0.0, 0
        for j, tau in enumerate(ref.taus):
            for i, (sd, typ) in enumerate(zip(ref.seeds, ref.types)):
                v = SandboxEnv(cfg, forced_type=typ).replay(int(sd), lambda _c, tau=tau: tau)
                dev = max(dev, abs(v - float(ref.F[j, i])))
                cnt += 1
        out[f"{dep['deploy']}/{dep['group']}"] = {"episodes": cnt, "max_abs_dev": dev}
        print(f"fidelity {dep['deploy']}/{dep['group']}: {cnt} episodes, max |sim - recorded| = {dev:.3g}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------- predictions


PREDICTED_VERDICTS = OrderedDict([  # every LLM arm that pays with the FCR, with its expected verdict after FCR-sym
    (f"{MOST_KNOWN}/task:llm__qwen3.8-27b__single__nshot1__v2", "fixed only"),
    (f"{MOST_KNOWN}/task:llm__qwen3.8-27b__single__nshot1__v2-think-long", "yes"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0-diag/newword/briefing:llm__qwen3.8-27b__single__nshot1", "yes"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0.5-diag/newword/briefing:llm__qwen3.8-27b__single__nshot1", "yes"),
    (f"{REACT4}/briefing:llm__qwen3.8-27b__single__nshot1", "yes"),
    (f"{REACT4}/task:llm__gpt-oss-120b__single__nshot0__v2", "yes"),
    (f"{REACT4}/task:llm__qwen3.8-27b__single__nshot1__v2", "yes"),
    (f"{REACT4}/task:llm__gpt-oss-120b__single__nshot0__v2-high-long", "yes"),
    (f"{REACT4}/task:llm__gpt-oss-120b__single__nshot0__v2-med", "yes"),
    (f"{REACT4}/task:llm__gpt-oss-120b__single__nshot0__v2-shuf", "fixed only"),
    (f"{REACT4}/task:llm__gpt-oss-120b__single__nshot1__v2", "yes"),
    (f"{REACT4}/task:llm__gpt-oss-120b__single__nshot42__v2", "yes"),
    (f"{REACT4}/task:llm__qwen3.8-27b__single__nshot0__v2-think-long", "yes"),
    (f"{REACT4}/task:llm__qwen3.8-27b__single__nshot1__v2-think-long", "yes"),
    (f"{REACT4}/task:llm__qwen3.8-27b__single__nshot42__v2", "fixed only"),
    (f"{REACT8}/briefing:llm__qwen3.8-27b__single__nshot1", "yes"),
    (f"{REACT8}/task:llm__gpt-oss-120b__single__nshot0__v2-med", "yes"),
    (f"{REACT8}/task:llm__gpt-oss-120b__single__nshot0__v2-shuf", "yes"),
    (f"{REACT8}/task:llm__gpt-oss-120b__single__nshot1__v2", "yes"),
    (f"{REACT8}/task:llm__gpt-oss-120b__single__nshot30__v2", "yes"),
    (f"{REACT8}/task:llm__qwen3.8-27b__single__nshot1__v2", "yes"),
    (f"{REACT8}/task:llm__qwen3.8-27b__single__nshot30__v2", "yes"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0.5-react/newword/briefing:llm__qwen3.8-27b__single__nshot1", "yes"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0.5-react8/newword/briefing:llm__qwen3.8-27b__single__nshot1", "yes"),
])


def ledger(res: dict, primary: dict, outcomes: dict) -> OrderedDict:
    L = OrderedDict()

    # trait-oracle-sym capture 1.00 and counter accuracy 1.00 in all 36 deployments
    bad = [k for k, d in res.items() if not (d["arms"][SYM_ORACLE]["counter_accuracy"] == 1.0 and abs(d["arms"][SYM_ORACLE]["capture"] - 1.0) < 1e-9)]
    L["P-SYM-2"] = {"claim": "trait-oracle-sym: counter accuracy 1.00 and capture 1.00 in all 36 deployments",
                    "holds": not bad, "value": f"{len(res) - len(bad)}/{len(res)} deployments", "exceptions": bad}

    # right tuple -> right counter; counter accuracy >= tuple accuracy (every non-privileged arm, every deployment)
    bad = []
    for k, d in res.items():
        for a in SYM_ARMS:
            st = d["arms"][a]
            g = st["counter_accuracy_given_tuple_right"]
            if (g is not None and g != 1.0) or st["counter_accuracy"] < st["trait_accuracy"]["tuple"] - 1e-12:
                bad.append(f"{k}/{a}")
    L["P-SYM-3a"] = {"claim": "every non-privileged FCR-sym arm in every deployment: counter accuracy 1.00 on episodes whose tuple is read right, so counter accuracy >= tuple accuracy",
                     "holds": not bad, "value": f"{36 * len(SYM_ARMS) - len(bad)}/{36 * len(SYM_ARMS)} arm-deployments", "exceptions": bad}

    # on held-out reactive-type episodes, counter accuracy <= min(reactivity accuracy, timing accuracy)
    bad, n_chk = [], 0
    for k, d in res.items():
        for a in SYM_ARMS:
            e = d["arms"][a]["strata"].get("held_out", {}).get("reactive_types")
            if not e:
                continue
            n_chk += 1
            bound = min(e["reactivity"], e.get("timing", 1.0))
            if e["counter_accuracy"] > bound + 1e-12:
                bad.append(f"{k}/{a}")
    L["P-SYM-3b"] = {"claim": "every non-privileged FCR-sym arm: on held-out reactive-type episodes, counter accuracy <= min(reactivity accuracy, timing accuracy)",
                     "holds": not bad, "value": f"{n_chk - len(bad)}/{n_chk} arm-deployments", "exceptions": bad}

    # same wording: FCR-sym-full counter accuracy 1.00 and capture 1.00 in all 16 deployments
    same = [k for k in res if k.endswith("/sameword")]
    bad = [k for k in same if not (res[k]["arms"]["sym_full"]["counter_accuracy"] == 1.0 and abs(res[k]["arms"]["sym_full"]["capture"] - 1.0) < 1e-9)]
    L["P-SYM-4"] = {"claim": "same wording: FCR-sym-full counter accuracy 1.00 and capture 1.00 in all 16 deployments",
                    "holds": not bad, "value": f"{len(same) - len(bad)}/{len(same)}",
                    "exceptions": {k: [res[k]["arms"]["sym_full"]["counter_accuracy"], res[k]["arms"]["sym_full"]["capture"]] for k in bad}}

    # most agents known primary deployment
    d = res[MOST_KNOWN]
    ca, cap = d["arms"]["sym_full"]["counter_accuracy"], d["arms"]["sym_full"]["capture"]
    strong = d["strongest"]["after_with_fcr_sym"]
    L["P-SYM-5"] = {"claim": "most agents known (primary): FCR-sym-full counter accuracy in [0.90, 0.97], capture in [0.85, 0.97], and an FCR-sym arm becomes the strongest non-LLM arm",
                    "holds": 0.90 <= ca <= 0.97 and 0.85 <= cap <= 0.97 and strong in SYM_ARMS,
                    "value": f"counter accuracy {ca:.3f}, capture {cap:.3f}, strongest {strong}"}

    # most agents known verdicts and letter
    want = {"120b task 0/type": "fixed only", "Qwen bare 1/type": "fixed only", "Qwen task 1/type": "fixed only"}
    lab = fe.PRIMARY_DEPLOYMENTS[MOST_KNOWN]
    got = {nm: primary[lab][f"T1 pays with FCR-sym: {nm}"]["pays"] for nm in want}
    think = res[MOST_KNOWN]["llm"]["task:llm__qwen3.8-27b__single__nshot1__v2-think-long"]["after_with_fcr_sym"]["pays"]
    letter = outcomes[lab]["letter"]
    L["P-SYM-6"] = {"claim": "most agents known: T1 Qwen task 1/type changes from yes to fixed only, 120b task 0/type and Qwen bare 1/type stay fixed only, Qwen task 1/type long reasoning keeps yes; headline letter B",
                    "holds": got == want and think == "yes" and letter == "B",
                    "value": f"T1 {got}; Qwen task 1/type long reasoning {think}; letter {letter}"}

    # new agents, new words, M = 18
    c4, c8 = res[REACT4]["arms"]["sym_full"]["capture"], res[REACT8]["arms"]["sym_full"]["capture"]
    new18 = [k for k in res if k.startswith("heldout/") and "-M18-" in k and k.endswith("/newword")]
    hi = {k: max(res[k]["arms"][a]["capture"] for a in SYM_ARMS) for k in new18}
    L["P-SYM-7"] = {"claim": "new agents, new words, M = 18: FCR-sym-full capture in [-0.10, 0.45] at react4 and in [-0.25, 0.35] at react8; no non-privileged FCR-sym arm reaches capture 0.5 in any of the six such deployments",
                    "holds": -0.10 <= c4 <= 0.45 and -0.25 <= c8 <= 0.35 and max(hi.values()) < 0.5,
                    "value": f"react4 {c4:.3f}, react8 {c8:.3f}; best FCR-sym capture per deployment " + ", ".join(f"{k.split('noise')[1]} {v:.3f}" for k, v in hi.items())}

    # the verdict of every LLM arm that pays with the FCR, after FCR-sym; every other verdict unchanged; letters C, C
    miss = OrderedDict()
    for key, want_v in PREDICTED_VERDICTS.items():
        parts = key.split("/")
        dk, ak = "/".join(parts[:3]), "/".join(parts[3:])
        got_v = res[dk]["llm"][ak]["after_with_fcr_sym"]["pays"]
        if got_v != want_v:
            miss[key] = {"predicted": want_v, "observed": got_v}
    others = [f"{k}/{ak}" for k, d in res.items() for ak, e in d["llm"].items()
              if f"{k}/{ak}" not in PREDICTED_VERDICTS and e["changed"]]
    let4, let8 = outcomes[fe.PRIMARY_DEPLOYMENTS[REACT4]]["letter"], outcomes[fe.PRIMARY_DEPLOYMENTS[REACT8]]["letter"]
    L["P-SYM-8"] = {"claim": "verdicts after FCR-sym as listed for the 24 LLM arms that pay with the FCR (3 change, 21 keep yes, including all six new-agent T1 verdicts), every other LLM verdict unchanged; headline letters C at react4 and react8",
                    "holds": not miss and not others and let4 == "C" and let8 == "C",
                    "value": f"{len(PREDICTED_VERDICTS) - len(miss)}/{len(PREDICTED_VERDICTS)} listed verdicts as predicted; {len(others)} unlisted verdicts changed; letters react4 {let4}, react8 {let8}",
                    "mismatches": miss, "unlisted_changes": others}

    # T2' (FCR-sym-full - TF-IDF) holds at most agents known and react4, fails at react8
    t2 = "T2' FCR-sym-full - TF-IDF whole-type"
    want = {fe.PRIMARY_DEPLOYMENTS[MOST_KNOWN]: True, fe.PRIMARY_DEPLOYMENTS[REACT4]: True, fe.PRIMARY_DEPLOYMENTS[REACT8]: False}
    got = {lab: primary[lab][t2]["holds"] for lab in want}
    L["P-SYM-9"] = {"claim": "T2' (FCR-sym-full - TF-IDF, lower bound > 0) holds with most agents known and at react4, and fails at react8",
                    "holds": got == want,
                    "value": {lab: f"diff {primary[lab][t2]['diff']:+.2f}, lower bound {primary[lab][t2]['lo95']:+.2f}" for lab in want}}

    # T3' holds in all nine primary pairs
    t3 = [(lab, tn, tv) for lab, tt in primary.items() for tn, tv in tt.items() if tn.startswith("T3'")]
    L["P-SYM-10"] = {"claim": "T3' (one-example LLM arm - FCR-sym-N on the same rows, lower bound > 0) holds in all nine primary pairs",
                     "holds": all(tv and tv["holds"] for _, _, tv in t3),
                     "value": f"{sum(bool(tv and tv['holds']) for _, _, tv in t3)}/{len(t3)}"}
    return L


# ----------------------------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(fc.RELEASE_DEFAULT))
    ap.add_argument("--B", type=int, default=fc.B_DEFAULT)
    ap.add_argument("--placebo", action="store_true", help="machinery test before the freeze (no FCR-sym choice is computed)")
    ap.add_argument("--out", default=None, help="default: <root>/analysis/briefing/{fcr_sym.json|placebo/placebo_check.json}")
    a = ap.parse_args()
    root = Path(a.root)
    chosen_path = root / "code" / "briefing" / "fcr" / "calib" / "chosen.json"
    if sha(chosen_path) != FCR_CHOSEN_SHA:
        raise SystemExit("code/briefing/fcr/calib/chosen.json is not the FCR's frozen settings file")
    prereg_path = root / "preregistration" / "briefing" / "PREREG_SYM.md"
    frozen = None if a.placebo else freeze_record(prereg_path, chosen_path)
    out_default = root / "analysis" / "briefing" / ("placebo/placebo_check.json" if a.placebo else "fcr_sym.json")
    out_path = Path(a.out) if a.out else out_default
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    cache = fc.load_embed_cache(root)
    models = fe.fit_models(root, chosen, cache)
    rel = fe.released_index(root)
    deps = fe.discover(root)
    stored_path = root / "analysis" / "briefing" / "fcr.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8")) if stored_path.exists() else None

    composers, fres, res, repro = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    for no, dep in enumerate(deps):
        k = f"{dep['deploy']}/{dep['group']}/{dep['wording']}"
        fdep = fe.evaluate_deployment(root, dep, no, models, rel, a.B)
        if stored is not None and a.B == stored["meta"]["B"]:
            repro[k] = jnorm(fdep) == stored["deployments"][k]
        fres[k] = fdep
        comp = composers.setdefault((dep["M"], dep["K"], dep["noise"]), SymbolicComposer(dep["M"], dep["K"], dep["noise"]))
        res[k] = evaluate_sym_deployment(root, dep, no, models, comp, fdep, a.B, a.placebo)
        print(k, {x: round(res[k]["arms"][x]["capture"], 3) for x in ("sym_full", "sym_full_embed", "sym_n_bare", "sym_n_task", SYM_ORACLE)},
              "strongest", res[k]["strongest"]["before_with_fcr"], "->", res[k]["strongest"]["after_with_fcr_sym"], flush=True)

    # primary tests (uncorrected, as in the FCR evaluation)
    primary = OrderedDict()
    for k, label in fe.PRIMARY_DEPLOYMENTS.items():
        d = res[k]
        t = OrderedDict()
        for name, ak in fe.PRIMARY_LLM.items():
            e = d["llm"].get(ak)
            t[f"T1 pays with FCR-sym: {name}"] = None if e is None else OrderedDict(
                **e["after_with_fcr_sym"], release_pays=e["release"].get("pays"), before_pays=e["before_with_fcr"]["pays"],
                changed=e["changed"], holds=e["after_with_fcr_sym"]["pays"] == "yes", p=e["after_with_fcr_sym"]["p_pays"])
        p = d["paired"].get("sym_full - briefing:text_bow")
        t["T2' FCR-sym-full - TF-IDF whole-type"] = p and {**p, "holds": p["lo95"] > 0, "p": p["p_le0"]}
        for name, (ak, symn) in SAMPLE_MATCHED_SYM.items():
            e = d["llm"].get(ak)
            v = e and e["vs_sym"][symn]
            t[f"T3' {name} - {symn}"] = v and {**v, "holds": v["lo95"] > 0, "p": v["p_le0"]}
        primary[label] = t
    hp = fc.holm({f"{lab}|{tn}": tv["p"] for lab, tt in primary.items() for tn, tv in tt.items() if tv})
    for lab, tt in primary.items():
        for tn, tv in tt.items():
            if tv:
                tv["holm_within_primary"] = hp[f"{lab}|{tn}"]

    # outcome letters (oracle* = trait-oracle-sym)
    lstar = {"most agents known": "Qwen task 1/type", "new agents, react4": "Qwen bare 1/type", "new agents, react8": "Qwen bare 1/type"}
    outcomes = OrderedDict()
    for k, label in fe.PRIMARY_DEPLOYMENTS.items():
        per_arm = OrderedDict()
        for name, ak in fe.PRIMARY_LLM.items():
            matched = "n_bare" if ak == "briefing:llm__qwen3.8-27b__single__nshot1" else ("n_task" if ak.endswith("nshot1__v2") else None)
            per_arm[name] = outcome_sym(fres[k], res[k], ak, matched)
        outcomes[label] = {"L_star": lstar[label], "letter": per_arm[lstar[label]].get("letter"),
                           "letter_fcr_experiment": per_arm[lstar[label]].get("letter_fcr_experiment"), "per_arm": per_arm}

    # secondary families (Holm, alpha 0.05)
    s1, s1_before, s2 = OrderedDict(), OrderedDict(), OrderedDict()
    primary_pairs = {(k, ak) for k in fe.PRIMARY_DEPLOYMENTS for ak in fe.PRIMARY_LLM.values()}
    for k, d in res.items():
        for ak, e in d["llm"].items():
            if (k, ak) not in primary_pairs:
                s1[f"{k}/{ak}"] = e["after_with_fcr_sym"]["p_pays"]
                s1_before[f"{k}/{ak}"] = e["before_with_fcr"]["p_pays"]
        if k not in fe.PRIMARY_DEPLOYMENTS and "sym_full - briefing:text_bow" in d["paired"]:
            s2[k] = d["paired"]["sym_full - briefing:text_bow"]["p_le0"]
    h1, h1b, h2 = fc.holm(s1), fc.holm(s1_before), fc.holm(s2)

    def entry_of(key: str) -> dict:
        parts = key.split("/")
        return res["/".join(parts[:3])]["llm"]["/".join(parts[3:])]

    s1_rows = OrderedDict()
    for key in s1:
        e = entry_of(key)
        if e["before_with_fcr"]["pays"] == "yes" or e["after_with_fcr_sym"]["pays"] == "yes" or e["changed"]:
            s1_rows[key] = OrderedDict(capture=e["capture"], release=e["release"].get("pays"), before=e["before_with_fcr"]["pays"],
                                       after=e["after_with_fcr_sym"]["pays"], strongest_before=e["before_with_fcr"]["strongest"],
                                       strongest_after=e["after_with_fcr_sym"]["strongest"], lo95_before=e["before_with_fcr"]["vs_lo95"],
                                       lo95_after=e["after_with_fcr_sym"]["vs_lo95"], p_before=s1_before[key], p_after=s1[key],
                                       holm_before=h1b[key], holm_after=h1[key])
    secondary = OrderedDict(
        S1_pays_with_fcr_sym=OrderedDict(
            tests=len(s1),
            before_pays_yes_uncorrected=sum(entry_of(k)["before_with_fcr"]["pays"] == "yes" for k in s1),
            after_pays_yes_uncorrected=sum(entry_of(k)["after_with_fcr_sym"]["pays"] == "yes" for k in s1),
            holm_surviving_before=[k for k, ok in h1b.items() if ok],
            holm_surviving_after=[k for k, ok in h1.items() if ok],
            changed_uncorrected=[k for k in s1 if entry_of(k)["changed"]],
            lost_holm=[k for k in s1 if h1b[k] and not h1[k]], gained_holm=[k for k in s1 if h1[k] and not h1b[k]],
            rows=s1_rows),
        S2_sym_full_minus_tfidf=OrderedDict(
            tests=len(s2), holm_surviving=[k for k, ok in h2.items() if ok],
            uncorrected_lo95_gt0=[k for k in s2 if res[k]["paired"]["sym_full - briefing:text_bow"]["lo95"] > 0],
            rows=OrderedDict((k, {**{x: res[k]["paired"]["sym_full - briefing:text_bow"][x] for x in ("diff", "lo95", "p_le0", "capture_diff")},
                                  "holm": h2[k]}) for k in s2)),
    )

    changed = [OrderedDict(deployment=k, arm=ak, primary=(k, ak) in primary_pairs, capture=e["capture"],
                           release=e["release"].get("pays"), before=e["before_with_fcr"]["pays"], after=e["after_with_fcr_sym"]["pays"],
                           strongest_before=e["before_with_fcr"]["strongest"], strongest_after=e["after_with_fcr_sym"]["strongest"],
                           lo95_before=e["before_with_fcr"]["vs_lo95"], lo95_after=e["after_with_fcr_sym"]["vs_lo95"],
                           margin_lo95=e["margin_lo95"])
               for k, d in res.items() for ak, e in d["llm"].items() if e["changed"]]
    strongest_changes = OrderedDict((k, {"before": d["strongest"]["before_with_fcr"], "after": d["strongest"]["after_with_fcr_sym"],
                                         "before_capture": d["comparators"][d["strongest"]["before_with_fcr"]]["capture"],
                                         "after_capture": (d["arms"].get(d["strongest"]["after_with_fcr_sym"]) or d["comparators"][d["strongest"]["after_with_fcr_sym"]])["capture"]})
                                    for k, d in res.items() if d["strongest"]["changed"])

    checks = OrderedDict(fcr_reproduction={"compared": len(repro), "identical": sum(repro.values()),
                                           "differing": [k for k, v in repro.items() if not v],
                                           "source": "analysis/briefing/fcr.json" if stored is not None else None})
    if a.placebo:
        checks["placebo_after_equals_before"] = {
            "llm_verdicts": sum(e["after_with_fcr_sym"]["pays"] == e["before_with_fcr"]["pays"] for d in res.values() for e in d["llm"].values()),
            "llm_arms": sum(len(d["llm"]) for d in res.values()),
            "strongest_unchanged": sum(not d["strongest"]["changed"] for d in res.values()), "deployments": len(res)}
    else:
        checks["composer_tables"] = composer_tables(composers, models)
        checks["simulator_fidelity"] = simulator_fidelity(deps)

    led = ledger(res, primary, outcomes)
    out = OrderedDict(
        meta=OrderedDict(
            freeze=frozen, placebo=a.placebo, B=a.B, boot_seed=fe.BOOT_SEED, tie_tol=fc.TIE_TOL, pays_margin=fc.PAYS_MARGIN,
            alpha=fc.ALPHA, headline_n=fc.HEADLINE_N, release_root="strategy-selection-audit (v9 release)",
            fcr_experiment="code/briefing/fcr (PREREG_FCR.md frozen 2026-10-06T10:24:04.984618+00:00)",
            composer={"rollouts": SIM_ROLLOUTS, "seed_base": SIM_SEED_BASE, "tie_eps": TIE_EPS, "common_random_numbers": True,
                      "tie_rule": "earliest strategy in pool order among those within tie_eps of the maximum"},
            sym_arms={k: list(v) for k, v in SYM_ARMS.items()}, primary_sym_arms=list(PRIMARY_SYM), oracle_arm=SYM_ORACLE,
            primary_llm_arms=dict(fe.PRIMARY_LLM), sample_matched=dict(SAMPLE_MATCHED_SYM),
            vendor_sources=json.loads((HERE / "SOURCES.json").read_text(encoding="utf-8"))),
        checks=checks, primary_tests=primary, outcomes=outcomes, secondary=secondary,
        changed_verdicts=changed, strongest_changes=strongest_changes, predictions_ledger=led, deployments=res)
    fc.write_json(out_path, out)
    print("written", out_path)
    print(json.dumps({k: v["holds"] for k, v in led.items()}, indent=1))
    print("changed verdicts:", len(changed), "; FCR reproduction:", checks["fcr_reproduction"]["identical"], "/", checks["fcr_reproduction"]["compared"])


if __name__ == "__main__":
    main()
