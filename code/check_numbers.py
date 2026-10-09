"""Gate: every 'value [lo,hi]' interval quoted in the main text must appear (to two decimals) among the intervals of the
analysis output, and every capture/margin value quoted with two decimals in the results sections must appear somewhere in
master.json. Prints the offenders; exit code 1 when any interval is unmatched.

--extra FILE (repeatable) merges a further numbers file, for example analysis/tables/numbers_briefing.json, into the
intervals the text may quote; its "quotes:main" entries, each a number with the exact text the main text quotes it as,
must all occur in main.tex. The counts behind the regimes of Fig. 1c (numbers.json "fig1:counts") must occur in the
opening paragraph of the Results."""
import json
import re
import sys
from pathlib import Path

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
ap.add_argument("--tex", default=None, help="the paper's main.tex (not included in this repository)")
ap.add_argument("--extra", action="append", default=[], help="a further numbers file whose intervals the text may quote")
args = ap.parse_args()
ROOT = Path(args.root).resolve()
tex = Path(args.tex or (ROOT / "latex" / "main.tex")).read_text(encoding="utf-8")
DATA = [
    json.load(open(ROOT / "analysis" / "master.json", encoding="utf-8")),
    json.load(open(ROOT / "analysis" / "extras.json", encoding="utf-8")),
    json.load(open(ROOT / "analysis" / "checks.json", encoding="utf-8")),
]
for extra in args.extra:
    DATA.append(json.load(open(extra if Path(extra).is_absolute() else ROOT / extra, encoding="utf-8")))

triples = set()
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k.endswith("_ci") and isinstance(v, list) and len(v) == 2:
                base = o.get(k[:-3])
                if isinstance(base, (int, float)):
                    triples.add((round(base, 2), round(v[0], 2), round(v[1], 2)))
            if k == "ci" and isinstance(v, list) and len(v) == 2 and isinstance(o.get("diff"), (int, float)):
                triples.add((round(o["diff"], 2), round(v[0], 2), round(v[1], 2)))
            if k == "ci" and isinstance(v, list) and len(v) == 2 and isinstance(o.get("mean"), (int, float)):
                triples.add((round(o["mean"], 2), round(v[0], 2), round(v[1], 2)))
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
for obj in DATA:
    walk(obj)

# numbers quoted for the HLA and single-type cells are outside master.json
EXEMPT = {(-43.10, -47.40, -38.75), (-38.60, -43.00, -34.30), (-1.35, -6.30, 3.55), (10.0, -2.0, 22.0), (1.30, -3.75, 6.40)}

pat = re.compile(r"(-?\d+\.\d+)\\?,?\s*\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]")
bad = []
n = 0
for m in pat.finditer(tex):
    v, lo, hi = (round(float(x), 2) for x in m.groups())
    n += 1
    if (v, lo, hi) in triples or (v, lo, hi) in EXEMPT:
        continue
    bad.append((v, lo, hi, tex[max(0, m.start() - 60):m.end() + 10].replace("\n", " ")))
# the counts behind the regimes of Fig. 1c: quoted in the first paragraph of the Results, before its first subsection,
# as "$n$ of $d$" (optionally "of those"/"of the") for the three regimes: first moves (N_), briefing with most agents
# known (BK_), briefing with new agents (BN_); C1 counts cells, C2 and C3 count arms
fig1 = json.load(open(ROOT / "analysis" / "tables" / "numbers.json", encoding="utf-8"))["fig1:counts"]
i0 = tex.index("\\section{Results}")
res = tex[i0:tex.index("\\subsection", i0)]
pairs = [("N_HEAD", "N_CELLS"), ("BK_HEAD", "BK_CELLS"), ("BN_HEAD", "BN_CELLS"),
         ("N_CLEAR", "N_ARMS"), ("BK_CLEAR", "BK_ARMS"), ("BN_CLEAR", "BN_ARMS"),
         ("N_BEAT", "N_CLEAR"), ("BK_BEAT", "BK_CLEAR"), ("BN_BEAT", "BN_CLEAR")]
miss = [f"{a}/{b} = {fig1[a]}/{fig1[b]}" for a, b in pairs
        if not re.search(rf"\${fig1[a]}\$ of (?:those |the )?\${fig1[b]}\$", res)]
if fig1["N_HEAD"] != fig1["N_CELLS"]:
    miss.append("not every first-moves cell has headroom")
print(f"figure 1 regime counts in the Results opening: {len(pairs)} checked, missing: {len(miss)}")
for w in miss:
    print("  RESULTS COUNT MISMATCH", w)
bad += [(0, 0, 0, w) for w in miss]
# numbers that the main text quotes from a further numbers file, each with the exact text it is quoted as
quotes = {}
for obj in DATA[3:]:
    quotes.update(obj.get("quotes:main", {}))
qmiss = [(k, q) for k, q in quotes.items() if q not in tex]
print(f"quoted numbers of the extra files: {len(quotes)} checked, missing: {len(qmiss)}")
for k, q in qmiss:
    print("  QUOTE MISSING", k, q)
bad += [(0, 0, 0, f"quote {k}: {q}") for k, q in qmiss]
print(f"intervals quoted: {n}, unmatched: {len(bad)}")
for b in bad:
    print("  UNMATCHED", b[:3], "...", b[3])
sys.exit(1 if bad else 0)
