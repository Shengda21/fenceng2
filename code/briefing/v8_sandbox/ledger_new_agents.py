"""Ledger (P-E', P-G, P-H, P-I) and per-cell tables from analysis/new_agents.json, plain ASCII markdown."""

from __future__ import annotations

import argparse
import json


def f(x, d=2):
    return "--" if x is None or x != x else f"{x:.{d}f}"


def ci(c):
    return f"[{f(c[0])}, {f(c[1])}]" if c and c[0] is not None and c[0] == c[0] else "[--]"


def short(g):
    return g.replace("sandbox-traits-", "").replace("-sharp0.8", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    args = parser.parse_args()
    r = json.load(open(args.json, encoding="utf-8"))
    G = r["groups"]
    pe = r["P_E"]
    print("## P-E' primary")
    for m, v in pe["primary"].items():
        vs, ho = v.get("vs_strongest") or {}, v.get("vs_strongest_held_out") or {}
        print(f"- {m}: capture {f(v['capture'])} {ci(v['capture_ci'])}, margin {f(v['margin'])} lo95 {f(v['margin_lo95'])}, "
              f"vs {v['vs_strongest_arm']} diff {f(vs.get('diff'))} {ci(vs.get('ci'))} lo95 {f(vs.get('lo95'))} -> pays={v['pays']} (margin>0: {v['pays_margin0']}), "
              f"p={f(v['p_pays'], 4)}, fallback {f(v['fallback_rate'])}")
    sc = pe["secondary_counts"]
    print(f"- secondary: {sc}; Holm survivors pays={pe['holm_surviving']['pays']} margin0={pe['holm_surviving']['pays_margin0']}")
    print("\n## P-G / P-H / P-I per cell")
    for g, e in G.items():
        p = e["predictions"]
        pg, ph, pi = p["P_G"], p["P_H"], p["P_I"]
        caps = ", ".join(f"{k} {f((pg[k] or {}).get('capture'))} {ci((pg[k] or {}).get('capture_ci'))}" for k in ("scripted_text", "text_bow", "text_embed", "gpt-oss-120b_single_n0"))
        print(f"- {short(g)}: P-G {'hit' if pg['holds'] else 'miss'} ({caps})")
        cross = ", ".join(f"{k.split('/')[-1]}{'(' + k.split('/')[0] + ')' if 'briefing' in k else ''} T50={v['T_50']} c={f(v['capture'])}" for k, v in ph["bandits"].items())
        print(f"    P-H predicted T* {f(ph['predicted_T_star'], 1)} expect_cross={ph['expect_cross']} -> {'hit' if ph['holds'] else 'miss'} ({cross})")
        if pi:
            print(f"    P-I stage0 held-out {f(pi['stage0_held_out_capture'])} {ci(pi['stage0_held_out_capture_ci'])} n={pi['stage0_held_out_n']} vs 0b(MB-normalised) "
                  f"{f(pi['stage0b_capture_mb'])} {ci(pi['stage0b_capture_mb_ci'])}; diff {f(pi['diff'])} {ci(pi['diff_ci'])} -> {'within noise' if pi['within_noise'] else 'NOT within noise'}; "
                  f"hit {f(pi['hit_stage0'])} -> {f(pi['hit_stage0b'])} {ci(pi['hit_diff_ci'])}; native 0b capture {f(pi['stage0b_capture_native'])}")
    print("\n## per-cell arm table")
    for g, e in G.items():
        ref, ex, r0 = e["reference"], e["exact"], e["reference_0b"]
        print(f"\n### {short(g)}: V* {f(ref['V_star'])} ({ref['best_fixed']}), H_D {f(ref['H_D'])} on 300 seeds (exact {f(ex['V_star'])}/{f(ex['H_D'])}); "
              f"prefix ceiling exact {f(ex['prefix_ceiling_bayes_capture'])}, headline {f(e['prefix_bayes_ceiling']['capture'])}; predicted T* {f(r0['predicted_T_star'], 1)}; "
              f"strongest non-LLM {e['strongest_nonllm']}")
        print("| cell/arm | capture [CI] | margin lo95 | hit | T50 | vs strongest [lo95] | pays | fallback |")
        print("|---|---|---|---|---|---|---|---|")
        for cell, c in e["cells"].items():
            for k, a in c["arms"].items():
                if a["meta"]["arm"] == "fixed" or not a.get("n"):
                    continue
                vs = a.get("vs_strongest") or {}
                hit = (a.get("strata", {}).get("all") or {}).get("hit_rate")
                print(f"| {cell}/{k.replace('llm__', '')} | {f(a['capture'])} {ci(a['capture_ci'])} | {f(a['margin_lo95'])} | {f(hit)} | {a.get('T_50', '')} | "
                      f"{f(vs.get('diff'))} [{f(vs.get('lo95'))}] | {a.get('pays', '')} | {f(a['fallback_rate'])} |")


if __name__ == "__main__":
    main()
