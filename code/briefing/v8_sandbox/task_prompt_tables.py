"""Markdown tables from analysis/task_prompt.json.

usage: python task_prompt_tables.py ../../analysis/task_prompt.json > ../../analysis/task_prompt_tables.md"""

from __future__ import annotations

import json
import sys

MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b"]
S0 = "sandbox-traits-M18-K10-sharp0.8-noise0-react"
S0B = {"react4": "sandbox-traits-M18-K10-sharp0.8-noise0-react", "react8": "sandbox-traits-M18-K10-sharp0.8-noise0-react8"}


def f(x, d=3, sign=True):
    if x is None:
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(c, d=2):
    return "—" if not c else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def arm_label(key: str) -> str:
    k = key.split("/", 1)[1]
    if k.startswith("llm__"):
        _, model, mode, ns, *rest = k.split("__")
        return f"{model} {'hf' if mode == 'hypothesis_first' else 'single'} {ns.replace('nshot', 'N')} {rest[0] if rest else 'v1'}"
    return k.replace("__nshot", " N")


def arm_table(group: dict, cell: str, title: str) -> str:
    rows = [f"### {title}", "", "| arm | capture [95% CI] | margin lo95 | vs strongest (diff [lo95]) | verdict | hit seen / held-out | fallback | describe rate | ROCK ratio |",
            "|---|---|---|---|---|---|---|---|---|"]
    for key, a in group["arms"].items():
        if not key.startswith(cell + "/"):
            continue
        vs = a.get("vs_strongest") or {}
        st = a.get("strata") or {}
        mis = a.get("misreading") or {}
        rows.append(f"| {arm_label(key)} | {f(a['capture'], 3, False)} {ci(a['capture_ci'])} | {f(a.get('margin_lo95'), 2)} | "
                    f"{f(vs.get('diff'), 2)} [{f(vs.get('lo95'), 2)}] | {a.get('pays') or '—'} | "
                    f"{f((st.get('seen') or {}).get('hit_rate'), 2, False)} / {f((st.get('held_out') or {}).get('hit_rate'), 2, False)} | "
                    f"{f(a.get('fallback_rate'), 3, False)} | {f(mis.get('describe_rate'), 2, False)} | {f(mis.get('rock_ratio'), 2, False)} |")
    rows.append(f"\nstrongest non-LLM: {group['strongest_nonllm']}\n")
    return "\n".join(rows)


def contrast_table(group: dict, cell: str, title: str) -> str:
    names = ["v2_minus_v1_single_n0", "v2_minus_v1_hf_n0", "v2_minus_v1_single_n1", "n1_minus_n0_v2", "nmany_minus_n0_v2", "hf_minus_single_v2",
             "shuf_minus_fixed_v2", "v2-med_minus_v2_n0", "v2-high_minus_v2_n0", "v2-think_minus_v2_n0", "v2-think_minus_v2_n1",
             "n1_v2_minus_text_bow_matched_n1", "nmany_v2_minus_text_bow_matched_nmany", "n1_v2_minus_text_bow_full", "nmany_v2_minus_text_bow_full"]
    rows = [f"### {title}", "", "| contrast | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    c = group["contrasts"].get(cell, {})
    for n in names:
        cells = []
        for m in MODELS:
            v = c.get(m, {}).get(n)
            cells.append("—" if not v else f"Δcap {f(v['capture_diff'])} {ci(v['capture_ci'])}; Δv {f(v['value_diff'], 2)} [lo95 {f(v['value_lo95'], 2)}]")
        if any(x != "—" for x in cells):
            rows.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def cost_table(res: dict) -> str:
    rows = ["### cost per episode (all calls incl. retries)", "", "| family / cell | arm | mean prompt tok | mean completion tok | p90 completion | mean latency s | p90 latency s | reasoning stored | length-truncated calls | retried | fallback |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in ("s0", "s0b"):
        for gname, g in res[fam].items():
            for key, a in g["arms"].items():
                if "cost" not in a or not a["meta"].get("variant"):
                    continue
                if a["meta"]["nshot"] > 1 or a["meta"]["mode"] != "single":
                    continue
                c = a["cost"]
                kind = "react8" if gname.endswith("react8") else "react4"
                rows.append(f"| {fam} {kind} {key.split('/')[0]} | {arm_label(key)} | {f(c['mean_prompt_tokens'], 0, False)} | {f(c['mean_completion_tokens'], 0, False)} | "
                            f"{f(c['p90_completion_tokens'], 0, False)} | {f(c['mean_latency_s'], 1, False)} | {f(c['p90_latency_s'], 1, False)} | "
                            f"{f(c['reasoning_stored_rate'], 2, False)} | {f(c['length_finish_rate'], 3, False)} | {f(c['retried_rate'], 3, False)} | {f(c['fallback_rate'], 3, False)} |")
    return "\n".join(rows) + "\n"


def classic_table(res: dict) -> str:
    cl = res.get("classic") or {}
    rows = ["### classic sandbox: zero-shot single, released v1 vs v2", "", "| cell | V* (best fixed) | H_D | " + " | ".join(f"{m} v1 / v2 / Δcap [95% CI]" for m in MODELS) + " |",
            "|---|---|---|" + "---|" * len(MODELS)]
    for stem, c in (cl.get("cells") or {}).items():
        cells = []
        for m in MODELS:
            e = c["models"].get(m, {})
            d = e.get("v2_minus_v1")
            cells.append(f"{f((e.get('v1') or {}).get('capture'), 2, False)} / {f((e.get('v2') or {}).get('capture'), 2, False)} / "
                         + (f"{f(d['capture_diff'], 2)} {ci(d['capture_ci'])}" if d else "—"))
        rows.append(f"| {stem.replace('sandbox-', '')} | {c['V_star']:.2f} ({c['best_fixed']}) | {c['H_D']:.2f} | " + " | ".join(cells) + " |")
    pn = cl.get("P_N") or {}
    rows.append("\n| model | mean Δcap over 8 cells [95% CI] | Stage 0 width | material | cells beyond width | CI excludes 0 |\n|---|---|---|---|---|---|")
    for m, v in pn.items():
        rows.append(f"| {m} | {f(v['mean_capture_diff'])} {ci(v['mean_ci'])} | {v['stage0_width']:.3f} | {v['material']} | {v['cells_beyond_width']} | {v['ci_excludes_0']} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    res = json.load(open(sys.argv[1], encoding="utf-8"))
    out = ["# Task-prompt tables (generated by task_prompt_tables.py)", ""]
    d = res["decision"]
    out.append(f"**Decision: {d['verdict']}** — primary yes: {d['primary_yes']}; Holm survivors: {d['holm_surviving']} of {d['secondary_entries']}; "
               f"uncorrected yes: {d['secondary_uncorrected_yes']}\n")
    out.append("| primary test | capture [CI] | margin lo95 | vs strongest diff [lo95] (arm) | verdict | p |\n|---|---|---|---|---|---|")
    for k, v in d["primary_0c"].items():
        if v:
            vs = v.get("vs_strongest") or {}
            out.append(f"| {k} | {f(v['capture'], 3, False)} {ci(v['capture_ci'])} | {f(v['margin_lo95'], 2)} | {f(vs.get('diff'), 2)} [{f(vs.get('lo95'), 2)}] ({v.get('vs_strongest_arm')}) | {v['pays']} | {f(v['p_pays'], 4, False)} |")
    out.append("")
    if S0 in res["s0"]:
        g = res["s0"][S0]
        for cell, t in (("briefing-newword", "s0 primary cell, newword (100 seeds)"), ("briefing-sameword", "s0, sameword (100 seeds)")):
            out.append(arm_table(g, cell, t))
            out.append(contrast_table(g, cell, t + " — paired contrasts"))
    for kind, gname in S0B.items():
        if gname in res["s0b"]:
            g = res["s0b"][gname]
            out.append(arm_table(g, "briefing-newword", f"s0b {kind} held-out-only, newword (300 seeds)"))
            out.append(contrast_table(g, "briefing-newword", f"s0b {kind} — paired contrasts"))
    n1 = res["predictions"].get("s0b_n1") or {}
    if n1:
        out.append("### s0b held-out-only cells: N=1 under the old prompt (Stage 0b) and under v2\n")
        out.append("| cell | model | arm | n | capture [CI] | vs strongest diff [lo95] | verdict | p |\n|---|---|---|---|---|---|---|---|")
        for kind, e in n1.items():
            for m, row in e.items():
                for t in ("v1", "v2", "v2-think-long"):
                    a = row.get(t)
                    if a:
                        vs = a.get("vs_strongest") or {}
                        out.append(f"| {kind} | {m} | {t} | {a['n']} | {f(a['capture'], 3, False)} {ci(a['capture_ci'])} | {f(vs.get('diff'), 2)} [{f(vs.get('lo95'), 2)}] | {a['pays']} | {f(a['p_pays'], 4, False)} |")
                d = row.get("v2_minus_v1")
                if d:
                    out.append(f"| {kind} | {m} | v2 - v1 (paired) | {d['n']} | dcap {f(d['capture_diff'])} {ci(d['capture_ci'])} | dv {f(d['value_diff'], 2)} [lo95 {f(d['value_lo95'], 2)}] | | |")
        out.append("")
    out.append(cost_table(res))
    out.append(classic_table(res))
    print("\n".join(out))


if __name__ == "__main__":
    main()
