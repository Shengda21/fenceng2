"""Regenerate every FCR output from the repository data, deterministically (CPU only, no model call).

  0. the sandbox modules the FCR imports (code/briefing/v8_sandbox/sandbox/{briefing,game,prompt_v2}.py) must carry the
     hashes in SOURCES.json
  1. survey_checks.py         -> analysis/briefing/fcr_survey_checks.json (lock identity, N-shot rows vs prompts, exact replay)
  2. evaluate.py --selftest   -> analysis/briefing/fcr_selftest.json (machinery reproduces the captures and strongest arms of the main analysis)
  3. calib_cv.py              -> {loco_cv,chosen}.json in a temporary folder, compared byte for byte with the stored calib/
                                 files; the run stops if the stored settings are not reproduced
  4. evaluate.py              -> analysis/briefing/fcr.json (uses calib/chosen.json; refuses to run unless
                                 preregistration/briefing/PREREG_FCR.md records its hash)
  5. make_tables.py           -> analysis/briefing/fcr_tables.md, analysis/briefing/fcr_ledger.json

usage: python code/briefing/fcr/run_all.py [--root <repository root>] [--skip-checks]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent  # code/briefing/fcr


def run(args: list[str]) -> None:
    cmd = [sys.executable, "-B", "-W", "ignore", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[2]), help="repository root")
    ap.add_argument("--skip-checks", action="store_true", help="skip steps 1-3")
    a = ap.parse_args()
    root_dir = Path(a.root).resolve()
    sources = json.loads((HERE / "SOURCES.json").read_text(encoding="utf-8"))
    for name, e in sources.items():
        if sha(root_dir / e["release_path"]) != e["sha256"]:
            raise SystemExit(f"{e['release_path']} does not carry the hash recorded in SOURCES.json")
    root = ["--root", str(root_dir)]
    out = root_dir / "analysis" / "briefing"
    if not a.skip_checks:
        run([str(HERE / "survey_checks.py"), *root, "--out", str(out / "fcr_survey_checks.json")])
        run([str(HERE / "evaluate.py"), "--selftest", *root])
        with tempfile.TemporaryDirectory(prefix="fcr_calib_rerun_") as tmp:
            rerun = Path(tmp)
            run([str(HERE / "calib_cv.py"), *root, "--out-dir", str(rerun)])
            for name in ("chosen.json", "loco_cv.json"):
                same = sha(rerun / name) == sha(HERE / "calib" / name)
                print(f"calibration rerun {name}: {'identical to the frozen file' if same else 'DIFFERS from the frozen file'}")
                if name == "chosen.json" and not same:
                    raise SystemExit("the frozen hyperparameters are not reproduced; stopping before the evaluation")
    run([str(HERE / "evaluate.py"), *root, "--out", str(out / "fcr.json")])
    run([str(HERE / "make_tables.py"), *root])
    print("analysis/briefing/fcr.json sha256", sha(out / "fcr.json"))


if __name__ == "__main__":
    main()
