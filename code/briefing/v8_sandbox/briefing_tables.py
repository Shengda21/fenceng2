"""Markdown tables from analysis/all_types.json: per group, every arm's capture overall, on seen/held-out, by reactivity."""

from __future__ import annotations

import argparse
import json


def f(x, d=2):
    return "--" if x is None or x != x else f"{x:.{d}f}"


def ci(c):
    return f"[{f(c[0])}, {f(c[1])}]" if c and c[0] == c[0] else ""


def label(cell, key):
    return f"{cell}/{key.replace('llm__', '').replace('__nshot', ' n')}"


def tables(res: dict, groups=None) -> str:
    out = []
    for g, e in res["groups"].items():
        if groups and g not in groups:
            continue
        r, x, m = e["reference"], e["exact"], e["meta"]
        out.append(f"\n### {g}\n")
        out.append(f"V* {f(r['V_star'])} ({r['best_fixed']}), H_D {f(r['H_D'])} on seeds 0-99 (exact V* {f(x['V_star'])}, H_D {f(x['H_D'])}); "
                   f"held-out set {m['held_out_set']}; prefix Bayes ceiling capture {f(x['prefix_ceiling_bayes_capture'])} "
                   f"(headline pseudo-arm {f(e['prefix_bayes_ceiling']['capture'])}); strongest non-LLM: "
                   + ", ".join(f"{k}: {v.split('/')[-1]} ({v.split('/')[0]})" for k, v in e["strongest_nonllm"].items()))
        out.append("")
        out.append("| cell / arm | capture [95% CI] | seen | held-out | habit | beat | copy | hit seen / held | fallback | T50 / T50pop |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for cell, c in e["cells"].items():
            for key, a in c["arms"].items():
                if a["meta"]["arm"] == "fixed" or not a.get("n"):
                    continue
                s = a.get("strata", {})
                g_ = lambda k, q: s.get(k, {}).get(q)
                t = f"{a.get('T_50')} / {a.get('T_50_population')}" if "T_50" in a else ""
                out.append(f"| {label(cell, key)} | {f(a['capture'])} {ci(a['capture_ci'])} | {f(g_('seen', 'capture'))} | {f(g_('held_out', 'capture'))} | "
                           f"{f(g_('habit', 'capture'))} | {f(g_('beat_last', 'capture'))} | {f(g_('copy_last', 'capture'))} | "
                           f"{f(g_('seen', 'hit_rate'))} / {f(g_('held_out', 'hit_rate'))} | {f(a.get('fallback_rate'))} | {t} |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    parser.add_argument("--groups", nargs="*")
    args = parser.parse_args()
    print(tables(json.load(open(args.json, encoding="utf-8")), args.groups))


if __name__ == "__main__":
    main()
