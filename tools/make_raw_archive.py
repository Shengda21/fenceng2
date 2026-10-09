#!/usr/bin/env python3
"""Pack the unreduced raw result tree for deposition.

This helper is not part of the release build. Run it from the original
workspace:

    python tools/make_raw_archive.py --source <LOCAL_PATH>\v8_data\results --out raw-results.tar.zst
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    with tarfile.open(out, "w:xz") as tf:
        tf.add(source, arcname="results")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
