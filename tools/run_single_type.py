"""Regenerate the single-type slice and HLA transfer records and check them against the released outputs.

The scripts in code/single_type and code/hla_transfer were written for a layout in which they sit in <root>/code,
read <root>/data/{experiments_phase3,oc_audit,rerun2026,hla_audit}, and write their JSON into <root>. This tool builds
that layout in a temporary folder from the released files, runs the scripts in dependency order, and compares every
output byte for byte with analysis/single_type_outputs/.

usage: python tools/run_single_type.py [--keep]
"""
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["analyze.py", "ledger.py", "derive_headroom.py", "hla_paired.py", "ledger_single_type.py", "rerun_departure.py",
         "regret_decomposition.py", "second_family.py", "second_family_oc.py", "forest_rows.py"]
DATA_MAP = {"experiments_phase3": "single_type/experiments_phase3", "oc_audit": "single_type/oc_audit",
            "rerun2026": "single_type/rerun2026", "hla_audit": "hla_transfer"}


def main() -> int:
    keep = "--keep" in sys.argv
    stage = Path(tempfile.mkdtemp(prefix="oghp_single_type_"))
    (stage / "code").mkdir()
    for sub in ("single_type", "hla_transfer"):
        for f in (ROOT / "code" / sub).glob("*.py"):
            shutil.copy2(f, stage / "code" / f.name)
    (stage / "data").mkdir()
    for dst, src in DATA_MAP.items():
        shutil.copytree(ROOT / "data" / src, stage / "data" / dst)
    for script in ORDER:
        r = subprocess.run([sys.executable, str(stage / "code" / script)], cwd=stage, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {script}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
            return 1
        print(f"ran {script}")
    bad = 0
    for ref in sorted((ROOT / "analysis" / "single_type_outputs").glob("*.json")):
        out = stage / ref.name
        same = out.exists() and filecmp.cmp(out, ref, shallow=False)
        bad += not same
        print(f"{'identical' if same else 'DIFFERENT'}  {ref.name}")
    if keep:
        print("staging folder kept at", stage)
    else:
        shutil.rmtree(stage, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
