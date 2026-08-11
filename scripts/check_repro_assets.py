#!/usr/bin/env python3
"""Validate recovered GSplat files without importing Torch or Nerfstudio."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "repro_assets" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quad Pilots asset preflight")
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST
    )
    parser.add_argument(
        "--asset-root", type=Path, default=REPOSITORY_ROOT / "outputs"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Explicit run directory for one selected track; relative paths are "
            "resolved from the current working directory"
        ),
    )
    parser.add_argument(
        "--track", choices=("all", "circle", "uturn", "lemniscate"), default="all"
    )
    parser.add_argument(
        "--hash", action="store_true", help="Calculate SHA-256 for recovered files"
    )
    args = parser.parse_args(argv)
    if args.run_dir is not None and args.track == "all":
        parser.error("--run-dir requires one explicit --track, not --track all")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = (
        manifest["tracks"]
        if args.track == "all"
        else {args.track: manifest["tracks"][args.track]}
    )
    all_ok = True

    for track, track_info in selected.items():
        run_root = (
            args.run_dir.expanduser().resolve()
            if args.run_dir is not None
            else args.asset_root / track / "splatfacto" / track_info["run"]
        )
        track_ok = True
        print(f"{track}: {run_root}")
        for relative, expected in track_info["files"].items():
            path = run_root / relative
            if not path.is_file():
                print(f"  MISSING {relative}")
                track_ok = False
                continue
            actual_size = path.stat().st_size
            expected_size = expected.get("size_bytes")
            if expected_size is not None and actual_size != expected_size:
                print(
                    f"  BAD_SIZE {relative}: {actual_size} != {expected_size}"
                )
                track_ok = False
                continue
            if args.hash and expected.get("sha256"):
                actual_hash = sha256(path)
                if actual_hash.lower() != expected["sha256"].lower():
                    print(f"  BAD_HASH {relative}: {actual_hash}")
                    track_ok = False
                    continue
            suffix = " hash=verified" if args.hash and expected.get("sha256") else ""
            print(f"  OK {relative} ({actual_size} bytes){suffix}")
        print(f"  STATUS {'READY' if track_ok else 'INCOMPLETE'}")
        all_ok &= track_ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
