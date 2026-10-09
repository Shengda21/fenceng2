"""Markdown tables (Chinese labels), generated from analysis/calib_verdict/calib_verdict.json.

usage: python code/calib_verdict/report_tables.py [--root <repository root>]  -> build/calib_verdict/report_table_*.md
"""

import json

from calib_lib import OUT, REL

d = json.load(open(OUT / "calib_verdict.json", encoding="utf-8"))
TAB = REL / "build" / "calib_verdict"
TAB.mkdir(parents=True, exist_ok=True)

CELL = {
    "sandbox/sandbox-M3-K3-sharp0.8-noise0": "沙盒 M3 s0.8 σ0", "sandbox/sandbox-M3-K3-sharp0.8-noise0.5": "沙盒 M3 s0.8 σ0.5",
    "sandbox/sandbox-M3-K3-sharp0.6-noise0": "沙盒 M3 s0.6 σ0", "sandbox/sandbox-M3-K3-sharp0.6-noise0.5": "沙盒 M3 s0.6 σ0.5",
    "sandbox/sandbox-M6-K3-sharp0.8-noise0": "沙盒 M6 s0.8 σ0", "sandbox/sandbox-M6-K3-sharp0.8-noise0.5": "沙盒 M6 s0.8 σ0.5",
    "sandbox/sandbox-M6-K3-sharp0.6-noise0": "沙盒 M6 s0.6 σ0", "sandbox/sandbox-M6-K3-sharp0.6-noise0.5": "沙盒 M6 s0.6 σ0.5",
    "briefing/sandbox-traits-M18-K10-sharp0.8-noise0-react-sameword": "简报 M18 已知措辞",
    "briefing/sandbox-traits-M9-K5-sharp0.8-noise0-react-sameword": "简报 M9 已知措辞",
    "magent/battle-base4": "battle base4", "magent/battle-traits8": "battle traits8", "magent/combined-base4": "combined base4",
    "magent/combined-traits8": "combined traits8", "magent/combined-base4-delta06": "combined base4 衰减",
}
MODEL = {"gpt-oss-120b": "120b", "gpt-oss-20b": "20b", "qwen3.8-27b": "Qwen"}


def arm_name(k):
    p = k.split("__")
    model, mode, n = MODEL[p[1]], p[2], p[3].replace("nshot", "")
    prompt = "任务提示" if len(p) > 4 and p[4] == "v2" else ""
    m = {"single": "", "hypothesis_first": " 先假设", "deliberate": " 三轮"}[mode]
    return f"{model}{m} N={n}" + (f" {prompt}" if prompt else "")


def yn(x):
    return "成立" if x else "不成立"


lines = ["| 单元格 | 臂 | n_B | C2 部署 | C2 B | C3 部署 | C3 B | B 边际下界 | B 对最强差下界 | P2 | 一致 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for c, v in d["cells"].items():
    for k, a in v["arms"].items():
        B, D = a["B"], a["deployment"]
        ok = "是" if a["agree_C2"] and a["agree_C3"] else "**否**"
        lines.append(f"| {CELL[c]} | {arm_name(k)} | {v['n_B']} | {yn(D['C2'])} | {yn(B['C2'])} | {yn(D['C3'])} | {yn(B['C3'])} | "
                     f"{B['margin_lo95']:+.2f} | {B['vs_lo95']:+.2f} | {a['power']['p_agree_C2']:.2f} | {ok} |")
(TAB / "report_table_agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

lines = ["| 单元格 | n_B | V\\*_B（最佳固定策略） | H_B | B 上最强非 LLM 臂 | 其捕获 | 部署最强非 LLM 臂 |", "| --- | --- | --- | --- | --- | --- | --- |"]
dep = json.load(open(OUT / "deploy_verdicts.json", encoding="utf-8"))
for c, v in d["cells"].items():
    dom, cell = c.split("/", 1)
    lines.append(f"| {CELL[c]} | {v['n_B']} | {v['V_star_B']:.2f}（{v['best_fixed_B']}） | {v['H_B']:.2f} | {v['strongest_nonllm_B']} | "
                 f"{v['strongest_capture_B']:.2f} | {dep[dom][cell]['strongest_nonllm']} |")
(TAB / "report_table_cells.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

lines = ["| 单元格 | 臂 | 部署捕获 | 部署边际下界 | B 捕获 [95%] | B 边际 | B 边际下界 | P2 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for x in d["summary"]["disagreements"]:
    a = d["cells"][x["cell"]]["arms"][x["arm"]]
    B, D = a["B"], a["deployment"]
    lines.append(f"| {CELL[x['cell']]} | {arm_name(x['arm'])} | {D['capture']:.2f} | {D['margin_lo95']:+.2f} | {B['capture']:.2f} "
                 f"[{B['capture_ci'][0]:.2f}, {B['capture_ci'][1]:.2f}] | {B['margin']:+.2f} | {B['margin_lo95']:+.2f} | {a['power']['p_agree_C2']:.2f} |")
(TAB / "report_table_disagreements.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(open(TAB / "report_table_cells.md", encoding="utf-8").read())
print(open(TAB / "report_table_disagreements.md", encoding="utf-8").read())
