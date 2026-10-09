"""Comparative-advantage (C3) verdicts of the briefing game's language-model arms with the factorized readers among the
comparators.

The strongest tested non-LLM arm of a briefing deployment is chosen by headline mean among the original candidates, the
nine arms of the factorized compositional reader with a fitted composer (fcr.json) and the six with the simulator
composer (fcr_sym.json) that do not see the true traits; fcr_sym.json records every language-model arm's verdict against
that extended set ("after_with_fcr_sym"); the original candidate set is stored under the key "release". This module turns those verdicts into the paper's convention, one rule for the
counts behind Fig. 1c and the Results opening (make_tables.py), Table 2 and the supplement (make_tables_briefing.py) and
Fig. 5 (make_fig_briefing.py):

* a primary test of a briefing family is judged by the pre-registered criterion uncorrected;
* a secondary test is judged after Holm correction within its family, on the p-values against the strongest arm of the
  extended set. The families are those of the briefing pre-registration: the bare-prompt arms of all_types.json (every
  type deployed, most agents known), those of new_agents.json (held-out types only, every agent new), each without its
  primary tests, and the new-wording variant arms of task_prompt.json without its primary tests.

With the original p-values the recomputed Holm sets equal the original lists exactly; this is asserted, so the
family definitions here are the pre-registered ones. Arms that do not read the briefing (the prefix-only "semantic"
cell) have no factorized-reader comparator and keep their original verdict.

Status letters: P primary, H secondary surviving Holm, S secondary not surviving it (or outside any family)."""
from __future__ import annotations

import json
from pathlib import Path

ALPHA = 0.05
NEW = "briefing-newword"


def first_file(*cands):
    for c in cands:
        if Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(cands[0])


def load_fcr_sym(root: Path):
    """fcr_sym.json: analysis/briefing/ in the repository, analysis/ in the paper tree."""
    return json.load(open(first_file(root / "analysis" / "briefing" / "fcr_sym.json", root / "analysis" / "fcr_sym.json"),
                          encoding="utf-8"))


def variant(key):
    return key.split("__")[4] if key.startswith("llm__") and key.count("__") >= 4 else None


def holm(pvals: dict, alpha=ALPHA) -> set:
    """Holm's step-down procedure: the names whose null is rejected."""
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    out = set()
    for i, k in enumerate(order):
        if pvals[k] > alpha / (len(order) - i):
            break
        out.add(k)
    return out


def _primary_bare(d, group, cell, key):
    pc = d["P_E"]["primary_cell"]
    m = d["groups"][group]["meta"]
    return (m["M"] == pc["M"] and m["reward_noise"] == pc["reward_noise"] and m["kind"] == pc["kind"]
            and cell == pc["cell"] and f"__{pc['mode']}__nshot{pc['nshot']}" in key and variant(key) is None
            and (pc.get("model") is None or f"__{pc['model']}__" in key))


class C3:
    def __init__(self, s0, s0b, s0c, sym):
        self.sym = sym
        self.arms = {}
        dec = s0c["decision"]
        holm_c = set(dec["holm_surviving"])
        self.release_holm = {}
        for part, d in (("s0", s0), ("s0b", s0b)):
            rel = set(d["P_E"]["holm_surviving"]["pays"])
            self.release_holm[part] = {(part,) + tuple(x.split("/", 2)) for x in rel}
            for fk, v in d["P_E"]["verdicts"].items():
                group, cell, key = fk.split("/", 2)
                prim = _primary_bare(d, group, cell, key)
                self._add(part, group, cell, key, v["pays"], v["p_pays"], prim, None if prim else part)
        self.release_holm["variants"] = set()
        fam_n = 0
        for part in ("s0", "s0b"):
            for group, gv in s0c[part].items():
                for ck, v in gv["arms"].items():
                    cell, key = ck.split("/", 1)
                    if not key.startswith("llm__") or variant(key) is None:
                        continue
                    model = key.split("__")[1]
                    pkey = "s0" if part == "s0" else ("s0b-react8" if group.endswith("-react8") else "s0b-react4")
                    prim = (f"{pkey}/{model}" in dec["primary_0c"] and cell == NEW
                            and key == f"llm__{model}__single__nshot0__v2")
                    fam = "variants" if (cell == NEW and not prim) else None
                    fam_n += fam is not None
                    self._add(part, group, cell, key, v["pays"], v["p_pays"], prim, fam)
                    if f"{part}/{group}/{ck}" in holm_c:
                        self.release_holm["variants"].add((part, group, cell, key))
        assert fam_n == dec["secondary_entries"], (fam_n, dec["secondary_entries"])
        # Holm within each family: with the original p-values it must give the original lists
        for fam in ("s0", "s0b", "variants"):
            members = {k: a for k, a in self.arms.items() if a["family"] == fam}
            rej = holm({k: a["p_release"] for k, a in members.items()})
            got = {k for k in rej if members[k]["pays_release"] == "yes"}
            assert got == self.release_holm[fam], (fam, sorted(got ^ self.release_holm[fam])[:4])
            rej2 = holm({k: a["p"] for k, a in members.items()})
            for k, a in members.items():
                a["holm_release"] = k in got
                a["holm"] = k in rej2 and a["pays"] == "yes"
        for a in self.arms.values():
            for suf in ("_release", ""):
                h = a.get("holm" + suf, False)
                a["status" + suf] = "P" if a["prim"] else ("H" if h else "S")
                a["c3" + suf] = a["pays" + suf] == "yes" and a["status" + suf] in ("P", "H")

    @staticmethod
    def dep_key(part, group, cell):
        wording = cell.split("-", 1)[1]
        return f"{'all' if part == 's0' else 'heldout'}/{group}/{wording}"

    def _add(self, part, group, cell, key, pays, p, prim, fam):
        rec = dict(part=part, group=group, cell=cell, key=key, prim=prim, family=fam, pays_release=pays, p_release=p,
                   pays=pays, p=p, strongest=None, strongest_release=None, vs=None, margin_lo95=None, sym=None)
        if cell.startswith(("briefing-", "briefing_prefix-")):
            dep = self.sym["deployments"].get(self.dep_key(part, group, cell))
            prefix = "task" if variant(key) else cell.split("-", 1)[0]
            e = dep["llm"].get(f"{prefix}:{key}") if dep else None
            if e is not None:
                assert e["release"]["pays"] == pays, (part, group, cell, key)
                aft = e["after_with_fcr_sym"]
                rec.update(pays=aft["pays"], p=aft["p_pays"], strongest=aft["strongest"], sym=e,
                           strongest_release=e["release"]["vs_strongest_arm"], margin_lo95=e["margin_lo95"])
                if aft["source"] == "fcr_sym":
                    v = e["vs_sym"][aft["strongest"]]
                else:
                    assert aft["source"] == "release", aft
                    v = e["release"]["vs_strongest"]
                assert abs(v["lo95"] - aft["vs_lo95"]) < 1e-9, (key, v["lo95"], aft["vs_lo95"])
                rec["vs"] = dict(diff=v["diff"], ci=v["ci"], lo95=v["lo95"], hi95=v["hi95"])
        self.arms[(part, group, cell, key)] = rec

    def get(self, part, group, cell, key):
        return self.arms.get((part, group, cell, key))

    def strongest(self, part, group, wording="newword"):
        """The strongest tested non-LLM arm of a deployment with the factorized readers among the candidates, as an
        fcr_sym.json arm key ('sym_full', 'briefing:text_bow', 'nonllm:linucb', ...)."""
        return self.sym["deployments"][f"{'all' if part == 's0' else 'heldout'}/{group}/{wording}"]["strongest"][
            "after_with_fcr_sym"]

    def family_counts(self):
        """Per family: entries, uncorrected yes and Holm survivors, with the original comparators and the extended set."""
        out = {}
        for fam in ("s0", "s0b", "variants"):
            m = [a for a in self.arms.values() if a["family"] == fam]
            out[fam] = dict(entries=len(m), yes_release=sum(a["pays_release"] == "yes" for a in m),
                            holm_release=sum(a["holm_release"] for a in m), yes=sum(a["pays"] == "yes" for a in m),
                            holm=sum(a["holm"] for a in m))
        return out
