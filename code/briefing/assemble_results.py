"""Copy every run into each analysis cell that uses it (shared base runs go into both held-out kinds' cells)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--cells", required=True)
    args = parser.parse_args()
    runs, cells = Path(args.runs), Path(args.cells)
    missing, copied = [], 0
    for src, dsts in json.loads(Path(args.map).read_text(encoding="utf-8")).items():
        if not (runs / src).exists():
            missing.append(src)
            continue
        for dst in dsts:
            (cells / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runs / src, cells / dst)
            copied += 1
    print(json.dumps({"copied": copied, "missing": len(missing), "missing_files": missing[:20]}, indent=1))


if __name__ == "__main__":
    main()
