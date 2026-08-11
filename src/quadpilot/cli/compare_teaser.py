#!/usr/bin/env python3
"""Compare a locked Lemniscate visual-loop run with the teaser metric slate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

from quadpilot.verification.teaser import compare_teaser_metrics  # noqa: E402

VIDEO_URL = "https://www.youtube.com/watch?v=8l80orgLiXs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute teaser-compatible NPE/EKF/DYN/GT metrics"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--track", default="lemniscate", choices=("lemniscate",))
    parser.add_argument("--dyn-seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_report(run_dir: Path, *, track: str, dyn_seed: int) -> dict[str, Any]:
    resolved_run = run_dir.expanduser().resolve()
    json_path = resolved_run / f"{track}_ekf.json"
    npz_path = resolved_run / f"{track}_ekf.npz"
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError("locked EKF JSON and NPZ artifacts are both required")

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = saved.get("metrics", {})
    if metrics.get("track") != track or metrics.get("estimator") != "ekf":
        raise ValueError("saved JSON is not the requested track's EKF run")
    if metrics.get("succeeded") is not True:
        raise ValueError("saved EKF run did not succeed")
    passes = metrics.get("controller_passes", [])
    pass_steps = [entry["step"] for entry in passes]

    with np.load(npz_path, allow_pickle=False) as arrays:
        report = compare_teaser_metrics(
            states=arrays["states"],
            observations=arrays["observations"],
            estimated_states=arrays["estimated_states"],
            controls=arrays["controls"],
            controller_pass_steps=pass_steps,
            dt=float(metrics["dt"]),
            dyn_seed=dyn_seed,
        )
    return {
        "schema_version": 1,
        "video_url": VIDEO_URL,
        "track": track,
        "source_artifacts": {
            "json": str(json_path),
            "json_sha256": _sha256(json_path),
            "npz": str(npz_path),
            "npz_sha256": _sha256(npz_path),
            "npe_checkpoint_sha256": saved["metadata"]["npe_checkpoint_sha256"],
        },
        **report,
    }


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {resolved}")
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
    report = build_report(args.run_dir, track=args.track, dyn_seed=args.dyn_seed)
    if args.output is not None:
        _write_json(args.output, report, overwrite=args.overwrite)
        print(f"Report written to {args.output.expanduser().resolve()}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
