"""Tables of the FCR-sym evaluation, read from analysis/briefing/fcr_sym.json (reporting only; no number is recomputed
except the regret per wrong pick, H_D * (1 - capture) / (1 - counter accuracy), which is exact because a right counter
has zero regret on every seed: the reward noise is common to all strategies on a seed).

usage: python code/briefing/fcr_sym/make_tables_sym.py [--root <repository root>]  -> analysis/briefing/fcr_sym_tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY = [
    ("all/sandbox-traits-M18-K10-sharp0.8-noise0-react/newword", "most agents known", "对手大多已知(100 个种子)"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0-react/newword", "new agents, react4", "全新对手,四个新类型 react4(300 个种子)"),
    ("heldout/sandbox-traits-M18-K10-sharp0.8-noise0-react8/newword", "new agents, react8", "全新对手,八个新类型 react8(300 个种子)"),
]
NAMES = {
    "sym_full": "FCR-sym-full(TF-IDF,主)", "sym_full_embed": "FCR-sym-full,嵌入", "sym_n_bare": "FCR-sym-N,裸提示行(主)",
    "sym_n_task": "FCR-sym-N,任务提示行(主)", "sym_n_bare_embed": "FCR-sym-N 裸提示行,嵌入", "sym_n_task_embed": "FCR-sym-N 任务提示行,嵌入",
    "sym_oracle": "性状 oracle-sym(特权)", "briefing:text_bow": "整类型 TF-IDF", "briefing:text_embed": "整类型嵌入",
    "briefing:scripted_text": "关键词规则", "nonllm:linucb": "LinUCB", "nonllm:lints": "LinTS", "nonllm:ctxucb": "ctxUCB",
    "nonllm:mucb": "mUCB", "briefing:text_linucb": "文本 LinUCB", "nonllm:fewshot__1": "few-shot 1",
    "fcr_full": "FCR-full(主效应)", "fcr_full_pairwise": "FCR-full 两两交互", "fcr_full_embed": "FCR-full 嵌入",
    "fcr_oracle": "FCR 性状 oracle 主效应", "fcr_oracle_pairwise": "FCR 性状 oracle 两两交互",
    "task:llm__gpt-oss-120b__single__nshot0__v2": "gpt-oss-120b 任务提示 0/type",
    "briefing:llm__qwen3.8-27b__single__nshot1": "Qwen3.8-27B 裸提示 1/type",
    "task:llm__qwen3.8-27b__single__nshot1__v2": "Qwen3.8-27B 任务提示 1/type",
    "task:llm__gpt-oss-120b__single__nshot1__v2": "gpt-oss-120b 任务提示 1/type",
}
VERD = {"yes": "有效", "fixed only": "仅胜过固定", "no": "否", None: "--"}
LLM4 = ["task:llm__gpt-oss-120b__single__nshot0__v2", "briefing:llm__qwen3.8-27b__single__nshot1",
        "task:llm__qwen3.8-27b__single__nshot1__v2", "task:llm__gpt-oss-120b__single__nshot1__v2"]


def nm(a: str) -> str:
    if a in NAMES:
        return NAMES[a]
    if a.startswith(("task:", "briefing:", "briefing_prefix:", "nonllm:")):
        cell, key = a.split(":", 1)
        key = key.replace("llm__", "").replace("__single__", " ").replace("__hypothesis_first__", " 先假设 ").replace("__", " ")
        return {"task": "任务提示 ", "briefing": "", "briefing_prefix": "简报+前缀 ", "nonllm": ""}[cell] + key
    return a


def dname(k: str) -> str:
    deploy, g, w = k.split("/")
    g = g.replace("sandbox-traits-", "").replace("-sharp0.8", "")
    M, K, noise, kind = g.split("-")
    pop = "大多已知" if deploy == "all" else "全新"
    return f"{pop} {M} {noise.replace('noise', 'σ')} {kind} {'已知措辞' if w == 'sameword' else '新措辞'}"


def f2(x, s=True):
    return "--" if x is None else (f"{x:+.2f}" if s else f"{x:.2f}")


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def cap(st):
    return f"{st['capture']:+.2f} [{st['capture_ci'][0]:+.2f}, {st['capture_ci'][1]:+.2f}]"


def regret_wrong(st, H):
    ca = st.get("counter_accuracy")
    if ca is None or ca >= 1.0:
        return "--"
    return f"{H * (1 - st['capture']) / (1 - ca):.1f}"


def delta(e: dict, judged: dict) -> float:
    """The LLM arm's paired difference against the strongest arm of a verdict: FCR-sym (this experiment), FCR (the FCR
    evaluation) or the baseline's own number."""

    t = judged["strongest"]
    if t in e["vs_sym"]:
        return e["vs_sym"][t]["diff"]
    if t in e["vs_fcr"]:
        return e["vs_fcr"][t]["diff"]
    return e["release"]["vs_strongest"]["diff"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args()
    ana = Path(a.root) / "analysis" / "briefing"
    d = json.loads((ana / "fcr_sym.json").read_text(encoding="utf-8"))
    deps = d["deployments"]
    L = ["# FCR-sym 表格(由 `code/briefing/fcr_sym/make_tables_sym.py` 从 `analysis/briefing/fcr_sym.json` 生成)", ""]
    fr = d["meta"]["freeze"]
    L += [f"冻结:{fr['frozen_utc']} UTC;组合器:{d['meta']['composer']['rollouts']} 次模拟,种子 {d['meta']['composer']['seed_base']:,} + r,"
          f"共同随机数,平局阈值 {d['meta']['composer']['tie_eps']},平局归池中靠前的策略。B = {d['meta']['B']:,},自助法种子 {d['meta']['boot_seed']}。", ""]

    # ---- primary table
    L += ["## 1. 主部署", "", "收取率为 (V − V\\*)/H_D,方括号为 95% 区间;反制准确率为所选策略等于该类型最优反制的比例;"
          "每次错选的遗憾 = H_D·(1 − 收取率)/(1 − 反制准确率),单位为每局回报。Δ 为 LLM 臂相对加入 FCR-sym 后最强非 LLM 臂的配对回报差,括号内为单侧 95% 下界。", ""]
    for k, lab, title in PRIMARY:
        dep = deps[k]
        H = dep["reference"]["H_D"]
        st_ = dep["strongest"]
        L += [f"**{title}**:V\\* = {dep['reference']['V_star']:.2f},H_D = {H:.2f};最强非 LLM 臂:发布包 {nm(st_['release'])},"
              f"加入 FCR 后 {nm(st_['before_with_fcr'])},加入 FCR-sym 后 **{nm(st_['after_with_fcr_sym'])}**。", ""]
        L += ["| 臂 | 收取率 [95%] | 反制准确率 | 元组准确率 | 每次错选的遗憾 | Δ(下界) | 判据:发布包 → FCR → FCR-sym |", "|---|---|---|---|---|---|---|"]
        for a in ["sym_full", "sym_full_embed", "sym_n_bare", "sym_n_task", "sym_n_bare_embed", "sym_n_task_embed", "sym_oracle"]:
            s = dep["arms"][a]
            L.append(f"| {nm(a)} | {cap(s)} | {f2(s['counter_accuracy'], False)} | {f2(s['trait_accuracy']['tuple'], False)} | {regret_wrong(s, H)} | | |")
        comps = ["briefing:text_bow", st_["online_bandit"]]
        if st_["release"] not in comps:
            comps.append(st_["release"])
        comps += [st_["fcr_full_star"], st_["fcr_oracle_star"]]
        for a in comps:
            s = dep["comparators"][a]
            tup = s.get("trait_accuracy", {}).get("tuple")
            L.append(f"| {nm(a)} | {cap(s)} | {f2(s['counter_accuracy'], False)} | {f2(tup, False) if tup is not None else '--'} | {regret_wrong(s, H)} | | |")
        for a in LLM4:
            e = dep["llm"].get(a)
            if e is None:
                continue
            aft = e["after_with_fcr_sym"]
            L.append(f"| {nm(a)} | {e['capture']:+.2f} [{e['capture_ci'][0]:+.2f}, {e['capture_ci'][1]:+.2f}] | {f2(e['counter_accuracy'], False)} | -- | "
                     f"{regret_wrong({'capture': e['capture'], 'counter_accuracy': e['counter_accuracy']}, H)} | "
                     f"{delta(e, aft):+.2f} ({aft['vs_lo95']:+.2f}) | {VERD[e['release'].get('pays')]} → {VERD[e['before_with_fcr']['pays']]} → **{VERD[aft['pays']]}** |")
        L.append("")
        L += ["按层(FCR-sym 臂):", "", "| 臂 | 层 | 局数 | 偏好 | 反应 | 时机 | 元组 | 反制准确率 | 反应型类型:(反应, 时机) 同时读对 | 收取率 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for a in ["sym_full", "sym_full_embed", "sym_n_bare", "sym_n_task"]:
            for sn, e in dep["arms"][a]["strata"].items():
                rt = e.get("reactive_types") or {}
                L.append(f"| {nm(a)} | {'已见' if sn == 'seen' else '留出'} | {e['n']} | {f2(e['favourite'], False)} | {f2(e['reactivity'], False)} | "
                         f"{f2(e.get('timing'), False)} | {f2(e['tuple'], False)} | {f2(e['counter_accuracy'], False)} | {f2(rt.get('reactivity_timing_joint'), False)} | {f2(e['capture'])} |")
        L.append("")

    # ---- primary tests
    L += ["## 2. 主检验(未校正;另报 21 项内的 Holm)", "", "| 部署 | 检验 | 估计 | 单侧下界 | p | 成立 | 主检验内 Holm |", "|---|---|---|---|---|---|---|"]
    for lab, tt in d["primary_tests"].items():
        for tn, tv in tt.items():
            if tv is None:
                continue
            if tn.startswith("T1"):
                est = f"{VERD[tv['release_pays']]} → {VERD[tv['before_pays']]} → {VERD[tv['pays']]}(对 {nm(tv['strongest'])})"
                lo = tv["vs_lo95"]
            else:
                est = f"{tv['diff']:+.2f} 回报({tv['capture_diff']:+.3f} 收取率)"
                lo = tv["lo95"]
            L.append(f"| {lab} | {tn} | {est} | {lo:+.2f} | {tv['p']:.4f} | {'是' if tv['holds'] else '否'} | {'是' if tv['holm_within_primary'] else '否'} |")
    L.append("")

    # ---- outcomes
    L += ["## 3. 结果字母(oracle\\* = 性状 oracle-sym)", "", "| 部署 | LLM 臂 | 判据:发布包 → FCR → FCR-sym | 门 | 字母 | FCR 实验的字母 | 决定性对比:差(下界) |", "|---|---|---|---|---|---|---|"]
    for lab, o in d["outcomes"].items():
        for name, r in o["per_arm"].items():
            if not r.get("available"):
                continue
            star = "**" if name == o["L_star"] else ""
            if "llm_minus_oracle_star" in r:
                v = r["llm_minus_oracle_star"]
                dec = f"L − oracle-sym:{v['diff']:+.2f}({v['lo95']:+.2f})"
                gate = "开"
            elif "llm_minus_n_star" in r:
                v = r["llm_minus_n_star"]
                dec = f"L − {nm(r['n_star'])}:{v['diff']:+.2f}({v['lo95']:+.2f})"
                gate = f"关({r['closed_by']})"
            else:
                dec, gate = "--", f"关({r['closed_by']})"
            letter = r["letter"] if len(r["letter"]) == 1 else "关(零样本臂,无样本匹配读者)"
            L.append(f"| {lab} | {star}{name}{star} | {VERD[r['release_pays']]} → {VERD[r['before_pays']]} → {VERD[r['after_pays']]} | {gate} | {star}{letter}{star} | "
                     f"{r['letter_fcr_experiment'] if len(str(r['letter_fcr_experiment'])) == 1 else '关'} | {dec} |")
    L.append("")

    # ---- changed verdicts
    L += ["## 4. 判据变化(全部 36 个部署、467 个 LLM 臂)", "", "未校正判据的变化:", "",
          "| 部署 | 臂 | 主检验 | 收取率 | 判据:发布包 → FCR → FCR-sym | 最强非 LLM 臂:前 → 后 | 下界:前 → 后 |", "|---|---|---|---|---|---|---|"]
    for c in d["changed_verdicts"]:
        L.append(f"| {dname(c['deployment'])} | {nm(c['arm'])} | {'是' if c['primary'] else '否'} | {c['capture']:.3f} | {VERD[c['release']]} → {VERD[c['before']]} → **{VERD[c['after']]}** | "
                 f"{nm(c['strongest_before'])} → {nm(c['strongest_after'])} | {c['lo95_before']:+.2f} → {c['lo95_after']:+.2f} |")
    s1 = d["secondary"]["S1_pays_with_fcr_sym"]
    L += ["", f"次要家族 S1′ 的 Holm 读数:{s1['tests']} 项;未校正有效 {s1['before_pays_yes_uncorrected']} → {s1['after_pays_yes_uncorrected']};"
          f"Holm 存活 {len(s1['holm_surviving_before'])} → {len(s1['holm_surviving_after'])};失去 Holm 存活:"
          + ("、".join(f"{dname('/'.join(x.split('/')[:3]))} {nm('/'.join(x.split('/')[3:]))}" for x in s1["lost_holm"]) or "无") + ";新获 Holm 存活:"
          + ("、".join(s1["gained_holm"]) or "无") + "。", ""]

    # ---- strongest
    L += ["## 5. 最强非 LLM 臂(36 个部署)", "", "| 部署 | 发布包 | 加入 FCR | 加入 FCR-sym | 收取率:前 → 后 |", "|---|---|---|---|---|"]
    for k, dep in deps.items():
        s = dep["strongest"]
        b, a = s["before_with_fcr"], s["after_with_fcr_sym"]
        ca = (dep["arms"].get(a) or dep["comparators"][a])["capture"]
        mark = "**" if s["changed"] else ""
        L.append(f"| {dname(k)} | {nm(s['release'])} | {nm(b)} | {mark}{nm(a)}{mark} | {dep['comparators'][b]['capture']:+.3f} → {ca:+.3f} |")
    L.append("")

    # ---- S1 rows
    L += ["## 6. 次要家族 S1′:加入 FCR 后有效的 18 个臂(Holm,α = 0.05)", "",
          "| 部署 | 臂 | 收取率 | 判据:FCR → FCR-sym | 最强非 LLM 臂 | 下界:前 → 后 | p:前 → 后 | Holm:前 → 后 |", "|---|---|---|---|---|---|---|---|"]
    for key, r in s1["rows"].items():
        parts = key.split("/")
        L.append(f"| {dname('/'.join(parts[:3]))} | {nm('/'.join(parts[3:]))} | {r['capture']:.3f} | {VERD[r['before']]} → {VERD[r['after']]} | {nm(r['strongest_after'])} | "
                 f"{r['lo95_before']:+.2f} → {r['lo95_after']:+.2f} | {r['p_before']:.4f} → {r['p_after']:.4f} | {'是' if r['holm_before'] else '否'} → {'是' if r['holm_after'] else '否'} |")
    L.append("")

    # ---- S2 and all deployments
    s2 = d["secondary"]["S2_sym_full_minus_tfidf"]
    L += ["## 7. 全部部署:FCR-sym 臂的收取率与次要家族 S2′(FCR-sym-full − 整类型 TF-IDF)", "",
          "| 部署 | TF-IDF | FCR-sym-full | 反制准确率 | FCR-sym 嵌入 | FCR-sym-N 裸 | FCR-sym-N 任务 | oracle-sym | 最强非 LLM 臂(加入 FCR-sym 后) | FCR-sym-full − TF-IDF [下界] | Holm |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, dep in deps.items():
        A = dep["arms"]
        tb = dep["comparators"].get("briefing:text_bow")
        p = dep["paired"].get("sym_full - briefing:text_bow")
        s = dep["strongest"]["after_with_fcr_sym"]
        sc = (A.get(s) or dep["comparators"][s])["capture"]
        h = "主检验" if k in [x[0] for x in PRIMARY] else ("是" if s2["rows"].get(k, {}).get("holm") else "否")
        L.append(f"| {dname(k)} | {f2(tb['capture'] if tb else None)} | {f2(A['sym_full']['capture'])} | {f2(A['sym_full']['counter_accuracy'], False)} | "
                 f"{f2(A['sym_full_embed']['capture'])} | {f2(A['sym_n_bare']['capture'])} | {f2(A['sym_n_task']['capture'])} | {f2(A['sym_oracle']['capture'])} | "
                 f"{nm(s)} {sc:+.2f} | {p['diff']:+.2f} [{p['lo95']:+.2f}] | {h} |")
    L.append("")

    # ---- decomposition
    L += ["## 8. 解析与组合的分解(相对性状 oracle-sym)", "", "| 部署 | 臂 | 收取率 | oracle 收取率 | 解析损失 | 组合损失 | 性状对、反制对 | 性状错、反制对 | 性状错、反制错 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, lab, _ in PRIMARY:
        for a, v in deps[k]["decomposition"].items():
            es = v["episode_shares"]
            L.append(f"| {lab} | {nm(a)} | {f2(v['capture'])} | {f2(v['oracle_capture'])} | {f2(v['parsing_loss'], False)} | {f2(v['composition_loss'], False)} | "
                     f"{f2(es.get('traits right, counter right', 0.0), False)} | {f2(es.get('traits wrong, counter right', 0.0), False)} | {f2(es.get('traits wrong, counter wrong (parsing)', 0.0), False)} |")
    L.append("")

    # ---- integrity
    ct = d["checks"]["composer_tables"]
    L += ["## 9. 完整性核对", "", f"FCR 实验的部署评估在此重跑,与已发布的 `fcr.json` 逐项相同:{d['checks']['fcr_reproduction']['identical']}/{d['checks']['fcr_reproduction']['compared']} 个部署。", ""]
    L += ["组合器对每个性状元组的选择与各锁定文件最优反制的一致性(冻结后核对,含留出类型):", "", "| 博弈单元 | 校准设置 | 一致 | 其中留出类型 |", "|---|---|---|---|"]
    for c, v in ct.items():
        for s, a in v["agreement_with_locks"].items():
            L.append(f"| {c} | {s} | {a['match_lock_best_response']}/{a['types']} | {a['held_out_types_match']} |")
    L += ["", "组合器表(M = 18,σ = 0;最优与次优之差):", "", "| 类型 | 选择 | 差 |", "|---|---|---|"]
    for t, r in ct["M18-K10-noise0"]["types"].items():
        L.append(f"| {t} | {r['choice']} | {r['gap_best_second']:.2f} |")
    fid = d["checks"]["simulator_fidelity"]
    L += ["", f"模拟器保真度:{sum(v['episodes'] for v in fid.values()):,} 局(全部 {len(fid)} 个非 LLM 单元的每个固定臂 × 每个主种子),"
          f"模拟回报与记录回报的最大绝对差 {max(v['max_abs_dev'] for v in fid.values()):.1e}。", ""]

    # ---- ledger
    L += ["## 10. 预测账本", "", "| 预测 | 成立 | 数值 |", "|---|---|---|"]
    for k, v in d["predictions_ledger"].items():
        val = v["value"] if isinstance(v["value"], str) else "; ".join(f"{a}: {b}" for a, b in v["value"].items())
        L.append(f"| {k}:{v['claim']} | {'是' if v['holds'] else '否'} | {val} |")
    for k in ("P-SYM-8",):
        mm = d["predictions_ledger"][k].get("mismatches") or {}
        for key, r in mm.items():
            parts = key.split("/")
            L.append(f"| {k} 未中:{dname('/'.join(parts[:3]))} {nm('/'.join(parts[3:]))} | 预测 {VERD[r['predicted']]} | 实际 {VERD[r['observed']]} |")
    L.append("")
    out = ana / "fcr_sym_tables.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    led = {k: {"claim": v["claim"], "holds": v["holds"], "value": v["value"]} for k, v in d["predictions_ledger"].items()}
    (ana / "fcr_sym_ledger.json").write_text(json.dumps(led, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("written", out)


if __name__ == "__main__":
    main()
