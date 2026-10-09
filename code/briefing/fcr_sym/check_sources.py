"""Hashes of the modules FCR-sym reads from elsewhere in the repository: the FCR's trait-reader code and settings
(`code/briefing/fcr/{fcr_common,fcr_model,evaluate}.py`, `calib/chosen.json`) and the sandbox and v8lib modules the
symbolic composer's simulator needs (`sandbox/{briefing,game,prompt_v2,env}.py`; `v8lib/{__init__,context,encoders}.py`).
FCR-sym imports these modules directly (no copies); this script only checks that they still carry the hashes recorded
in `SOURCES.json`. `--write` regenerates that file from the current files.

usage: python code/briefing/fcr_sym/check_sources.py [--write] [--root <repository root>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent  # code/briefing/fcr_sym
MANIFEST = HERE / "SOURCES.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--root", default=str(HERE.parents[2]))
    a = ap.parse_args()
    root = Path(a.root)
    if a.write:
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for name, e in man.items():
            e["sha256"] = sha(root / e["release_path"])
        MANIFEST.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {MANIFEST} ({len(man)} files)")
        return
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = [f"{name}: {e['release_path']} does not carry the hash recorded in SOURCES.json"
           for name, e in man.items() if sha(root / e["release_path"]) != e["sha256"]]
    if bad:
        raise SystemExit("source check failed:\n  " + "\n  ".join(bad))
    print(f"source check: {len(man)} release files match SOURCES.json")


if __name__ == "__main__":
    main()
