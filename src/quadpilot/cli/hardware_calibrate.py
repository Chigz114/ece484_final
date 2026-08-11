#!/usr/bin/env python3
"""Estimate and validate the Vicon <- NeRF similarity transform offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

from quadpilot.hardware.readiness import (  # noqa: E402
    estimate_similarity_transform,
    sha256_file,
    similarity_payload,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Vicon <- NeRF calibration")
    parser.add_argument("correspondences", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-rmse-m", type=float, default=0.03)
    parser.add_argument("--fixed-scale", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _atomic_write(path: Path, payload: dict, *, overwrite: bool) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite calibration: {resolved}")
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_path = args.correspondences.expanduser().resolve()
    data = json.loads(source_path.read_text(encoding="utf-8"))
    transform, report = estimate_similarity_transform(
        data["nerf_points_m"],
        data["vicon_points_m"],
        source_frame="nerf_world",
        target_frame="vicon_world",
        estimate_scale=not args.fixed_scale,
    )
    payload = similarity_payload(
        transform,
        report,
        input_sha256=sha256_file(source_path),
        accepted_rmse_m=args.maximum_rmse_m,
    )
    _atomic_write(args.output, payload, overwrite=args.overwrite)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
