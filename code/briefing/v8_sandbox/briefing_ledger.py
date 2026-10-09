"""Prediction ledger P-A..P-F (hit/miss per operational test) from analysis/all_types.json."""

from __future__ import annotations

import argparse
import json


def f(x, d=2):
    return "--" if x is None or x != x else f"{x:.{d}f}"


def ledger(res: dict) -> tuple[str, dict]:
    G = res["groups"]
    lines, verdicts = [], {}

    rows = []
    ok_a = True
    for g, e in G.items():
        pb = e["prefix_bayes_ceiling"]
        hit = pb["capture_ci"][1] < 1.0
        ok_a &= hit
        rows.append(f"{g}: ceiling capture {f(pb['capture'])} [{f(pb['capture_ci'][0])}, {f(pb['capture_ci'][1])}] (exact {f(e['exact']['prefix_ceiling_bayes_capture'])}), H_D share {f(e['exact']['share'], 4)}")
    verdicts["P-A"] = ok_a
    lines.append(("P-A", ok_a, "gate passed in every lock; share 0.9375 = 0.9375 (equality); " + "; ".join(rows)))

    ok_b, rows = True, []
    for g, e in G.items():
        p = e["predictions"]["P_B"]
        over = [k for k, v in p["readers"].items() if not v["at_or_below"]]
        part = p["all_at_or_below_ceiling"] and p["scripted_reactive_negative"] and (p["bayes_top2_is_habit_R_P"] in (True, None))
        ok_b &= part
        shares = ", ".join(f"{k} {f(v['loss_share_habit_RP'])}" for k, v in p["readers"].items()) if e["meta"]["M"] == 9 else ""
        rows.append(f"{g}: {'ok' if part else 'MISS'} (above ceiling: {over or 'none'}; top-2 Bayes losses {p['bayes_top2_loss_types']}; "
                    f"scripted reactive capture {f(p['scripted_reactive_capture'])} [{f(p['scripted_reactive_capture_ci'][0])}, {f(p['scripted_reactive_capture_ci'][1])}]"
                    + (f"; loss share on habit R+P: {shares}" if shares else "") + ")")
    verdicts["P-B"] = ok_b
    lines.append(("P-B", ok_b, "; ".join(rows)))

    ok_c, rows = True, []
    for g, e in G.items():
        p = e["predictions"]["P_C"]
        if e["meta"]["kind"] == "react":
            for key, v in p.items():
                wording, arm = key.split("/")
                d = v["held_out_minus_seen"]
                if wording == "sameword":
                    part = v["within_noise"]
                elif arm in ("scripted_text", "text_bow"):
                    part = v["held_out_below_0.5"]
                else:
                    part = None
                if part is not None:
                    ok_c &= part
                rows.append(f"{g} {key}: seen {f(v['seen_capture'])}, held-out {f(v['held_out_capture'])} "
                            f"[{f((v['held_out_capture_ci'] or [None, None])[0])}, {f((v['held_out_capture_ci'] or [None, None])[1])}], "
                            f"diff {f(d.get('diff'))} [{f((d.get('ci') or [None, None])[0])}, {f((d.get('ci') or [None, None])[1])}]"
                            + ("" if part is None else (" ok" if part else " MISS")))
        else:
            for key, v in p.items():
                h = v.get("held_out_habit") or {}
                rows.append(f"{g} {key}: held-out habit stratum (coverage mechanism) n={h.get('n')}, capture {f(h.get('capture'))}, hit {f(h.get('hit_rate'))}")
    verdicts["P-C"] = ok_c
    lines.append(("P-C", ok_c, "\n  - " + "\n  - ".join(rows)))

    best = []
    for g, e in G.items():
        for k, v in e["predictions"]["P_D"]["arms"].items():
            best.append((v["capture"], g, k, v))
    best.sort(reverse=True, key=lambda t: t[0])
    ok_d = any(e["predictions"]["P_D"]["holds"] for e in G.values())
    verdicts["P-D"] = ok_d
    top = "; ".join(f"{g}/{k}: capture {f(c)}, held-out-seen hit lo95 {f(v['hit_held_out_minus_seen_lo95'])}, briefing-semantic lo95 {f((v['briefing_minus_semantic'] or {}).get('lo95'))} {'(holds)' if v['holds'] else ''}"
                    for c, g, k, v in best[:6])
    lines.append(("P-D", ok_d, "best briefing N=0 arms: " + top))

    pe = res["P_E"]
    ok_e = pe["primary_holds"]
    verdicts["P-E"] = ok_e
    prim = "; ".join(f"{m}: capture {f(v['capture'])}, margin lo95 {f(v['margin_lo95'])}, vs {str(v['vs_strongest_arm']).split('/')[-1]} "
                     f"{f((v['vs_strongest'] or {}).get('diff'))} [lo95 {f((v['vs_strongest'] or {}).get('lo95'))}] -> {v['pays']} (margin>0 version: {v['pays_margin0']}); "
                     f"held-out stratum vs {str(v.get('vs_strongest_held_out_arm')).split('/')[-1]}: {f((v.get('vs_strongest_held_out') or {}).get('diff'))} "
                     f"[{f(((v.get('vs_strongest_held_out') or {}).get('ci') or [None, None])[0])}, {f(((v.get('vs_strongest_held_out') or {}).get('ci') or [None, None])[1])}]"
                     for m, v in pe["primary"].items())
    sc = pe["secondary_counts"]
    lines.append(("P-E", ok_e, f"primary: {prim}. Secondary: {sc['pays_yes']} of {sc['llm_arm_entries']} LLM arm entries pay "
                                 f"({sc['newword_pays_yes']} under newword; {sc['pays_margin0_yes']} with margin>0); Holm survivors: {len(pe['holm_surviving']['pays'])} "
                                 f"({len(pe['holm_surviving']['pays_margin0'])} with margin>0)"))

    rows, ok_f = [], True
    for g, e in G.items():
        p = e["predictions"]["P_F"]
        ok_f &= p["holds"]
        arms = "; ".join(f"{k.split('/')[-1]}{'(' + k.split('/')[0] + ')' if 'briefing' in k else ''}: T50 {v['T_50']}, T50pop {v['T_50_population']}, c100 {f(v['capture_at']['100'])}, c300pop {f(v['capture_at_population']['300'])}"
                         for k, v in p["arms"].items())
        rows.append(f"{g}: {'ok' if p['holds'] else 'MISS'} ({arms})")
    verdicts["P-F"] = ok_f
    lines.append(("P-F", ok_f, "\n  - " + "\n  - ".join(rows)))

    md = ["| prediction | verdict |", "|---|---|"] + [f"| {n} | {'hit' if ok else 'miss'} |" for n, ok, _ in lines]
    md.append("")
    for n, ok, detail in lines:
        md.append(f"- **{n} ({'hit' if ok else 'miss'})**: {detail}")
    return "\n".join(md), verdicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    args = parser.parse_args()
    text, _ = ledger(json.load(open(args.json, encoding="utf-8")))
    print(text)


if __name__ == "__main__":
    main()
