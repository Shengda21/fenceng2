"""Regenerate every FCR-sym output from the recorded data and the FCR calibration under code/briefing/fcr/,
deterministically (CPU only, no model call).

  1. check_sources.py           checks that the FCR modules this reads (code/briefing/fcr/{fcr_common,fcr_model,
                                 evaluate}.py, calib/chosen.json) and the sandbox and v8lib modules the
                                 symbolic composer needs carry the hashes in SOURCES.json
  2. calib_check.py             -> a temporary file, compared byte for byte with calib/sym_calib_check.json
                                 (the composer with the true traits of every seen type against the best responses and
                                 value rows of all 18 locks); the run stops if it is not reproduced
  3. evaluate_sym.py --placebo  -> analysis/briefing/placebo/placebo_check.json (the machinery test)
  4. evaluate_sym.py            -> analysis/briefing/fcr_sym.json (uses the FCR's calib/chosen.json; refuses
                                 without the freeze row of preregistration/briefing/PREREG_SYM.md)
  5. make_tables_sym.py         -> analysis/briefing/fcr_sym_tables.md, analysis/briefing/fcr_sym_ledger.json

usage: python code/briefing/fcr_sym/run_all.py [--root <repository root>] [--skip-checks]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent  # code/briefing/fcr_sym


def run(args: list[str]) -> None:
    cmd = [sys.executable, "-B", "-W", "ignore", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[2]), help="release root")
    ap.add_argument("--skip-checks", action="store_true", help="run steps 4 and 5 only")
    a = ap.parse_args()
    root_dir = Path(a.root).resolve()
    root = ["--root", str(root_dir)]
    out = root_dir / "analysis" / "briefing"
    if not a.skip_checks:
        run([str(HERE / "check_sources.py"), *root])
        with tempfile.TemporaryDirectory(prefix="fcr_sym_calib_rerun_") as tmp:
            rerun = Path(tmp) / "sym_calib_check.json"
            run([str(HERE / "calib_check.py"), *root, "--out", str(rerun)])
            same = sha(rerun) == sha(HERE / "calib" / "sym_calib_check.json")
            print(f"calibration check rerun: {'identical to the frozen file' if same else 'DIFFERS from the frozen file'}")
            if not same:
                raise SystemExit("the frozen calibration check is not reproduced; stopping before the evaluation")
        run([str(HERE / "evaluate_sym.py"), "--placebo", *root, "--out", str(out / "placebo" / "placebo_check.json")])
    run([str(HERE / "evaluate_sym.py"), *root, "--out", str(out / "fcr_sym.json")])
    run([str(HERE / "make_tables_sym.py"), *root])
    for f in ("fcr_sym.json", "fcr_sym_tables.md", "fcr_sym_ledger.json", "placebo/placebo_check.json"):
        p = out / f
        if p.exists():
            print(f"analysis/briefing/{f}", "sha256", sha(p))


if __name__ == "__main__":
    main()
