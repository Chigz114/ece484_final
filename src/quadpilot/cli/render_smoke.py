#!/usr/bin/env python3
"""Render a fixed Circle Gate A view and record a deterministic smoke report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from quadpilot.datasets.generation import pose_to_camera_matrix
from quadpilot.perception.renderer import RenderOnlySplatRenderer
from quadpilot.perception.runtime import configure_wsl_cuda_toolchain

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSET_MANIFEST = REPOSITORY_ROOT / "configs" / "assets" / "manifest.json"
CIRCLE_RUN = Path(
    json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))["tracks"]["circle"][
        "runtime"
    ]["renderer_run_dir"]
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circle GSplat renderer smoke test")
    parser.add_argument("--run-dir", type=Path, default=CIRCLE_RUN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "repro_outputs" / "renderer" / "circle",
    )
    parser.add_argument("--cuda-home", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cuda_home is not None:
        configure_wsl_cuda_toolchain(args.cuda_home)
    else:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    checkpoint = args.run_dir / "nerfstudio_models" / "step-000029999.ckpt"
    transform = args.run_dir / "dataparser_transforms.json"
    missing = [path for path in (checkpoint, transform) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing renderer assets: " + ", ".join(map(str, missing))
        )

    renderer = RenderOnlySplatRenderer(
        checkpoint,
        transform,
        expected_step=29999,
        expected_gaussians=308832,
    )
    pose = np.array([-0.3, -2.8, -0.4, 0.0, 0.0, -np.pi / 2.0])
    camera_to_world = pose_to_camera_matrix(pose)
    expected_c2w = np.array(
        [
            [-0.584951222, -0.028173886, 0.810578942, 0.391613956],
            [0.810578942, -0.055022836, 0.583038807, -0.237141754],
            [0.028173886, 0.998087525, 0.055022836, -0.048376037],
        ],
        dtype=np.float32,
    )
    transformed = renderer.pose_transform.to_nerfstudio_c2w(camera_to_world)
    if not np.allclose(transformed, expected_c2w, atol=1e-6):
        raise RuntimeError("Circle coordinate transform smoke assertion failed")

    started = time.perf_counter()
    image = renderer.render_rgb_u8(camera_to_world)
    render_seconds = time.perf_counter() - started
    if image.std() <= 5.0:
        raise RuntimeError(f"rendered image is degenerate; std={image.std()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "gate_a_front_rgb.png"
    Image.fromarray(image, mode="RGB").save(image_path)
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "pose_project_frame": pose.tolist(),
        "nerfstudio_c2w": transformed.tolist(),
        "checkpoint_step": renderer.checkpoint_step,
        "gaussian_count": renderer.gaussian_count,
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "image_min": int(image.min()),
        "image_max": int(image.max()),
        "image_mean": float(image.mean()),
        "image_std": float(image.std()),
        "render_seconds": render_seconds,
        "image_sha256_nonportable": image_hash,
        "image": image_path.name,
    }
    report_path = args.output_dir / "smoke_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
