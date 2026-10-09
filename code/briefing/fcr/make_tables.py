"""Tables and the prediction ledger, read from analysis/briefing/fcr.json (formatting only).

usage: python code/briefing/fcr/make_tables.py [--root <repository root>]  -> analysis/briefing/fcr_tables.md, fcr_ledger.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import fcr_common as fc

PRIMARY = OrderedDict([
    ("all/sandbox-traits-M18-K10-sharp0.8-noise0-react/newword", "most agents known"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0-react/newword", "new agents, react4"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0-react8/newword", "new agents, react8"),
])
NAMES = OrderedDict([
    ("briefing:text_bow", "TF-IDF whole-type"),
    ("fcr_full", "FCR-full (TF-IDF, main) [primary]"),
    ("fcr_full_pairwise", "FCR-full, pairwise composer"),
    ("fcr_full_embed", "FCR-full, embedding"),
    ("fcr_n_bare", "FCR-N bare rows (main) [primary]"),
    ("fcr_n_task", "FCR-N task rows (main) [primary]"),
    ("fcr_n_bare_pairwise", "FCR-N bare, pairwise"),
    ("fcr_n_bare_embed", "FCR-N bare, embedding"),
    ("fcr_n_task_pairwise", "FCR-N task, pairwise"),
    ("fcr_n_task_embed", "FCR-N task, embedding"),
    ("fcr_oracle", "trait-oracle FCR, main (privileged)"),
    ("fcr_oracle_pairwise", "trait-oracle FCR, pairwise (privileged)"),
    ("task:llm__gpt-oss-120b__single__nshot0__v2", "gpt-oss-120b task 0/type"),
    ("briefing:llm__qwen3.8-27b__single__nshot1", "Qwen3.8-27B bare 1/type"),
    ("task:llm__qwen3.8-27b__single__nshot1__v2", "Qwen3.8-27B task 1/type"),
    ("task:llm__gpt-oss-120b__single__nshot1__v2", "gpt-oss-120b task 1/type"),
])


def f2(x, nd=2, sign=True):
    if x is None:
        return "--"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(c):
    return f"[{c[0]:+.2f}, {c[1]:+.2f}]" if c else ""


def short(a):
    return a.split(":", 1)[-1]


def primary_table(d) -> list[str]:
    out = []
    for k, label in PRIMARY.items():
        dep = d["deployments"][k]
        ref = dep["reference"]
        st = dep["strongest"]
        out.append(f"\n**{label}** (`{k}`; {ref['seeds_n']} seeds; V* = {ref['V_star']:.2f}, H_D = {ref['H_D']:.2f}; "
                   f"strongest non-LLM in the release: `{short(st['release'])}`; with the FCR arms added: `{short(st['with_fcr'])}`)\n")
        out.append("| arm | capture [95%] | counter acc. | hit rate | tuple acc. | Δ vs strongest non-LLM (lo95) | pays: release → with FCR |")
        out.append("|---|---|---|---|---|---|---|")
        rows = list(NAMES)
        bandit = st["online_bandit"]
        rows.insert(1, bandit)
        if st["release"] not in rows:
            rows.insert(2, st["release"])
        for a in rows:
            if a not in dep["arms"]:
                continue
            s = dep["arms"][a]
            name = NAMES.get(a, f"strongest online bandit ({short(a)})" if a == bandit else short(a))
            if a == bandit and a == st["release"]:
                name = f"strongest online bandit = strongest non-LLM ({short(a)})"
            tup = f2(s["trait_accuracy"]["tuple"], 2, False) if "trait_accuracy" in s else "--"
            pays, vs = "", ""
            if a in dep["llm"]:
                e = dep["llm"][a]
                rel = e["release"]
                up = e["updated_primary"]
                pays = f"{rel['pays']} → {up['pays']}"
                vs = f"{rel['vs_strongest']['diff']:+.2f} ({up['vs_lo95']:+.2f})" if rel.get("vs_strongest") and up["source"] == "release" else f"({up['vs_lo95']:+.2f})"
                cap = f"{e['capture']:.2f} {ci(e['capture_ci'])}"
            elif s.get("release") and s["release"].get("capture_ci"):
                cap = f"{s['release']['capture']:.2f} {ci(s['release']['capture_ci'])}"
            else:
                cap = f"{s['capture']:.2f} {ci(s['capture_ci'])}"
            out.append(f"| {name} | {cap} | {s['counter_accuracy']:.2f} | {s['hit_rate']:.2f} | {tup} | {vs} | {pays} |")
    return out


def tests_table(d) -> list[str]:
    out = ["| deployment | test | estimate | lo95 | p | holds | Holm within primary |", "|---|---|---|---|---|---|---|"]
    for label, tt in d["primary_tests"].items():
        for tn, tv in tt.items():
            if not tv:
                out.append(f"| {label} | {tn} | n/a | | | | |")
                continue
            if tn.startswith("T1"):
                est = f"{tv['release_pays']} → {tv['pays']} (vs `{short(tv['strongest'])}`)"
                lo = tv["vs_lo95"]
            else:
                est = f"{tv['diff']:+.2f} reward ({tv['capture_diff']:+.2f} capture)"
                lo = tv["lo95"]
            out.append(f"| {label} | {tn} | {est} | {lo:+.2f} | {tv['p']:.4f} | {'yes' if tv['holds'] else 'no'} | {'yes' if tv['holm_within_primary'] else 'no'} |")
    return out


def outcome_table(d) -> list[str]:
    out = ["| deployment | LLM arm | pays: release → with FCR | gate | letter (pre-registered) | deciding contrast (lo95) | letter with the other oracle form |",
           "|---|---|---|---|---|---|---|"]
    for k, label in PRIMARY.items():
        dep = d["deployments"][k]
        o = d["outcomes"][label]
        for name, r in o["per_arm"].items():
            if not r.get("available"):
                continue
            star = " (L*)" if name == o["L_star"] else ""
            gate = "open" if r["updated_pays"] == "yes" else f"closed by {r.get('closed_by')}"
            if "llm_minus_oracle_star" in r:
                orc = r["oracle_star"]
                other = "fcr_oracle_pairwise" if orc == "fcr_oracle" else "fcr_oracle"
                alt = dep["llm"][d["meta"]["primary_llm_arms"][name]]["vs"][other]["lo95"]
                dec = f"L − {orc}: {r['llm_minus_oracle_star']['diff']:+.2f} ({r['llm_minus_oracle_star']['lo95']:+.2f})"
                alt_letter = f"{'D' if alt > 0 else 'C'} (L − {other} lo95 {alt:+.2f})"
            elif "llm_minus_fcr_n_star" in r:
                dec = f"L − {r['fcr_n_star']}: {r['llm_minus_fcr_n_star']['diff']:+.2f} ({r['llm_minus_fcr_n_star']['lo95']:+.2f})"
                alt_letter = "--"
            else:
                dec, alt_letter = "--", "--"
            out.append(f"| {label} | {name}{star} | {r['release_pays']} → {r['updated_pays']} | {gate} | {r['letter']} | {dec} | {alt_letter} |")
    return out


def decomposition_table(d) -> list[str]:
    out = ["| deployment | FCR arm | capture | oracle capture | parsing loss | composition loss | traits right, counter wrong | traits wrong, counter wrong | regret share: composition | regret share: parsing |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k, label in PRIMARY.items():
        dep = d["deployments"][k]
        for a in ("fcr_full", "fcr_full_pairwise", "fcr_full_embed"):
            x = dep["decomposition"][a]
            es, rs = x["episode_shares"], x["regret_shares"]
            out.append(f"| {label} | {a} | {x['capture']:+.2f} | {x['oracle_capture']:+.2f} | {x['parsing_loss']:+.2f} | {x['composition_loss']:+.2f} | "
                       f"{es.get('traits right, counter wrong (composition)', 0):.2f} | {es.get('traits wrong, counter wrong (parsing)', 0):.2f} | "
                       f"{(rs.get('traits right, counter wrong (composition)') or 0):.2f} | {(rs.get('traits wrong, counter wrong (parsing)') or 0):.2f} |")
    return out


def strata_table(d) -> list[str]:
    out = ["| deployment | arm | stratum | n | favourite | reactivity | timing | tuple | counter acc. | capture |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, label in PRIMARY.items():
        dep = d["deployments"][k]
        for a in ("fcr_full", "fcr_full_pairwise", "fcr_full_embed", "fcr_n_bare", "fcr_n_task", "fcr_oracle", "fcr_oracle_pairwise"):
            for sname, s in dep["arms"][a]["strata"].items():
                out.append(f"| {label} | {a} | {sname} | {s['n']} | {s['favourite']:.2f} | {s['reactivity']:.2f} | {s.get('timing', float('nan')):.2f} | "
                           f"{s['tuple']:.2f} | {s['counter_accuracy']:.2f} | {f2(s['capture'])} |")
    return out


def secondary_table(d) -> list[str]:
    s2 = set(d["secondary"]["S2_fcr_full_minus_tfidf"]["holm_surviving"])
    out = ["| deployment | wording | σ | TF-IDF | FCR-full | FCR-full pairwise | FCR-full emb. | oracle main | oracle pairwise | strongest non-LLM (release) | strongest with FCR | FCR-full − TF-IDF [lo95] | S2 Holm |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, dep in d["deployments"].items():
        m = dep["meta"]
        a = dep["arms"]
        p = dep["paired"].get("fcr_full - briefing:text_bow")
        rel = dep["strongest"]["release"]
        lab = f"{'most agents known' if m['deploy'] == 'all' else 'new agents'} M{m['M']} {m['held_out_set']}" + (" **(primary)**" if m["primary"] else "")
        out.append(f"| {lab} | {m['wording']} | {m['reward_noise']:g} | {a['briefing:text_bow']['capture']:+.2f} | {a['fcr_full']['capture']:+.2f} | "
                   f"{a['fcr_full_pairwise']['capture']:+.2f} | {a['fcr_full_embed']['capture']:+.2f} | {a['fcr_oracle']['capture']:+.2f} | "
                   f"{a['fcr_oracle_pairwise']['capture']:+.2f} | {short(rel)} {a[rel]['capture']:+.2f} | {short(dep['strongest']['with_fcr'])} | "
                   f"{p['diff']:+.2f} [{p['lo95']:+.2f}] | {'yes' if k in s2 else ('n/a' if m['primary'] else 'no')} |")
    return out


def s1_lines(d) -> list[str]:
    s1 = d["secondary"]["S1_pays_with_fcr"]
    out = [f"- S1 tests: {s1['tests']}; release verdict yes: {s1['release_pays_yes']}; yes with the FCR arms (uncorrected): "
           f"{s1['updated_pays_yes_uncorrected']}; Holm survivors with the FCR: {len(s1['holm_surviving'])}; Holm survivors of the same family "
           f"with the release's p-values: {len(s1['holm_surviving_release_p_same_family'])}; verdicts lost to the FCR: {len(s1['lost_by_fcr_uncorrected'])}."]
    changed = [(k, dep["strongest"]["with_fcr"]) for k, dep in d["deployments"].items() if dep["strongest"]["with_fcr"] != dep["strongest"]["release"]]
    out.append(f"- Deployments where an FCR arm becomes the strongest non-LLM arm: {len(changed)}" + (": " + "; ".join(f"`{k}` → `{v}`" for k, v in changed) if changed else "."))
    out.append("- Holm survivors (S1): " + "; ".join(f"`{x}`" for x in s1["holm_surviving"]))
    return out


def cv_table(d) -> list[str]:
    out = ["| setting | condition | rows | TF-IDF C (f/r/t) | LOCO tuple acc. | LOCO counter acc. main / pairwise | embedding LOCO tuple | whole-type TF-IDF LOCO counter |",
           "|---|---|---|---|---|---|---|---|"]
    cv = d["calibration"]["loco_cv_summary"]
    for s, e in cv.items():
        for c, ce in e["conditions"].items():
            t = ce["tfidf"]
            C = "/".join(f"{v:g}" for v in t["C"].values())
            out.append(f"| {s} | {c} | {ce['n_rows']} | {C} | {t['end_to_end']['main']['tuple']:.3f} | {t['end_to_end']['main']['counter']:.3f} / "
                       f"{t['end_to_end']['pairwise']['counter']:.3f} | {ce['embed']['end_to_end']['main']['tuple']:.3f} | {ce['whole_type_tfidf_loco_counter_accuracy']:.3f} |")
    out.append("")
    out.append("| setting | composer | α | LOCO counter acc. (true traits) | fit on seen types |")
    out.append("|---|---|---|---|---|")
    for s, e in cv.items():
        for f, x in e["composer"].items():
            out.append(f"| {s} | {f} | {x['alpha']:g} | {x['loco_counter_accuracy']:.3f} | {x['train_counter_accuracy_seen']:.3f} |")
    return out


def ledger(d) -> OrderedDict:
    deps = d["deployments"]
    mak, r4, r8 = (deps[k] for k in PRIMARY)
    L = OrderedDict()
    s = mak["arms"]["fcr_full"]["strata"]["seen"]
    L["P-FCR-1a"] = {"text": "most agents known: FCR-full trait accuracy on seen types >= 0.95 each",
                     "values": {t: s[t] for t in ("favourite", "reactivity", "timing")}, "holds": all(s[t] >= 0.95 for t in ("favourite", "reactivity", "timing"))}
    sw = {k: v["arms"]["fcr_full"]["strata"]["held_out"]["tuple"] for k, v in deps.items()
          if v["meta"]["M"] == 18 and v["meta"]["wording"] == "sameword" and "held_out" in v["arms"]["fcr_full"]["strata"]}
    L["P-FCR-1b"] = {"text": "M=18 seen words: FCR-full held-out tuple accuracy >= 0.90", "values": sw, "holds": all(v >= 0.90 for v in sw.values())}
    nw = {lab: deps[k]["arms"]["fcr_full"]["strata"]["held_out"]["tuple"] for k, lab in PRIMARY.items()}
    L["P-FCR-2"] = {"text": "new words: FCR-full held-out tuple accuracy < 0.5 in each primary deployment", "values": nw, "holds": all(v < 0.5 for v in nw.values())}
    om = mak["arms"]["fcr_oracle"]["strata"]["seen"]["counter_accuracy"]
    t2 = {lab: d["primary_tests"][lab]["T2 FCR-full - TF-IDF whole-type"]["lo95"] for lab in PRIMARY.values()}
    L["P-FCR-3"] = {"text": "trait-oracle main counter acc. on seen types (most agents known) <= 0.60, and T2 fails in all three",
                    "values": {"oracle_main_seen_counter": om, "T2_lo95": t2}, "holds": om <= 0.60 and all(v <= 0 for v in t2.values())}
    op = {lab: {"seen": deps[k]["arms"]["fcr_oracle_pairwise"]["strata"].get("seen", {}).get("counter_accuracy"),
                "held_out": deps[k]["arms"]["fcr_oracle_pairwise"]["strata"]["held_out"]["counter_accuracy"]} for k, lab in PRIMARY.items()}
    L["P-FCR-4"] = {"text": "trait-oracle pairwise: all seen-type episodes right, at most half of held-out-type episodes right, in each primary deployment",
                    "values": op, "holds": all((v["seen"] is None or v["seen"] == 1.0) and v["held_out"] <= 0.5 for v in op.values()),
                    "holds_per_deployment": {lab: (v["seen"] is None or v["seen"] == 1.0) and v["held_out"] <= 0.5 for lab, v in op.items()}}
    want5 = {("new agents, react4", "120b task 0/type"): "yes", ("new agents, react8", "120b task 0/type"): "fixed only",
             ("new agents, react4", "Qwen bare 1/type"): "yes", ("new agents, react8", "Qwen bare 1/type"): "yes",
             ("new agents, react4", "Qwen task 1/type"): "yes", ("new agents, react8", "Qwen task 1/type"): "yes"}
    got5 = {f"{lab} / {n}": d["primary_tests"][lab][f"T1 pays with FCR: {n}"]["pays"] for (lab, n) in want5}
    L["P-FCR-5"] = {"text": "new agents: T1 verdicts unchanged", "values": got5, "holds": all(got5[f"{lab} / {n}"] == v for (lab, n), v in want5.items())}
    tf = mak["arms"]["briefing:text_bow"]["mean"]
    best = max(mak["arms"][a]["mean"] for a in mak["arms"] if a.startswith("fcr_") and "oracle" not in a)
    want6 = {"120b task 0/type": "fixed only", "Qwen bare 1/type": "fixed only", "Qwen task 1/type": "yes"}
    got6 = {n: d["primary_tests"]["most agents known"][f"T1 pays with FCR: {n}"]["pays"] for n in want6}
    L["P-FCR-6"] = {"text": "most agents known: no FCR arm above TF-IDF by more than 1.0 reward; T1 verdicts unchanged",
                    "values": {"best_fcr_minus_tfidf_mean": best - tf, "T1": got6}, "holds": best - tf <= 1.0 and got6 == want6}
    t3 = {f"{lab} / {tn}": tv["holds"] for lab, tt in d["primary_tests"].items() for tn, tv in tt.items() if tn.startswith("T3")}
    L["P-FCR-7"] = {"text": "T3 holds for all three 1/type arms in all three primary deployments", "values": t3, "holds": all(t3.values())}
    letters = {lab: d["outcomes"][lab]["letter"] for lab in PRIMARY.values()}
    want8 = {"most agents known": "C", "new agents, react4": "D", "new agents, react8": "D"}
    L["P-FCR-8"] = {"text": "headline letters C (most agents known), D (react4), D (react8)", "values": letters, "holds": letters == want8}
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(fc.RELEASE_DEFAULT))
    ana = Path(ap.parse_args().root) / fc.ANALYSIS
    d = json.loads((ana / "fcr.json").read_text(encoding="utf-8"))
    lines = [f"# FCR tables (generated from analysis/briefing/fcr.json; freeze {d['meta']['freeze']['frozen_utc']})", "", "## Primary deployments"]
    lines += primary_table(d)
    lines += ["", "## Primary tests", ""] + tests_table(d)
    lines += ["", "## Outcomes A–D", ""] + outcome_table(d)
    lines += ["", "## Parsing vs composition (trait-oracle decomposition)", ""] + decomposition_table(d)
    lines += ["", "## Accuracies by stratum", ""] + strata_table(d)
    lines += ["", "## Every briefing deployment (S2: FCR-full − TF-IDF, Holm)", ""] + secondary_table(d)
    lines += ["", "## S1: the C3 verdict of every other LLM arm with the FCR arms added", ""] + s1_lines(d)
    lines += ["", "## Calibration LOCO-CV", ""] + cv_table(d)
    L = ledger(d)
    lines += ["", "## Prediction ledger", "", "| prediction | holds | values |", "|---|---|---|"]
    for k, v in L.items():
        lines.append(f"| {k}: {v['text']} | {'yes' if v['holds'] else 'no'} | `{json.dumps(v['values'], default=float)}` |")
    out = ana / "fcr_tables.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    fc.write_json(ana / "fcr_ledger.json", L)
    print("written", out)


if __name__ == "__main__":
    main()
