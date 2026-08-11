#!/usr/bin/env python3
"""Generate deterministic pre-gate views for NPE fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from quadpilot_repro.data_generation import (  # noqa: E402
    BASE_DATASET_BOUNDS,
    CameraIntrinsics,
    ReproDatasetGenerator,
)
from quadpilot_repro.environment import configure_wsl_cuda_toolchain  # noqa: E402
from quadpilot_repro.gate_sampling import (  # noqa: E402
    GateFocusConfig,
    GateFocusedPoseSampler,
)
from quadpilot_repro.renderer import RenderOnlySplatRenderer  # noqa: E402


DEFAULT_RUNS = {
    "circle": REPOSITORY_ROOT / "outputs/circle/splatfacto/2025-05-09_144210",
    "uturn": REPOSITORY_ROOT / "outputs/uturn/splatfacto/2025-05-09_151825",
    "lemniscate": REPOSITORY_ROOT
    / "outputs/lemniscate/splatfacto/2025-05-09_153156",
}
EXPECTED_ASSETS = {
    "circle": {
        "step": 29999,
        "gaussians": 308832,
        "checkpoint_sha256": "af37b9e28b033d0b21a47d26e56b0479649ba0fc092a97979c031b8217767069",
        "transform_sha256": "c43166261f14fa78e3c9c8134dd16e716b2a1977adfeba08d0dc6b942740b874",
    },
    "uturn": {
        "step": 29999,
        "gaussians": 437285,
        "checkpoint_sha256": "c3a884a5765ed86789facca4648416c01141afc93a684dcab724a5df8613b5b7",
        "transform_sha256": "abddf07924fa64e3ba57376ea27dfc67e8ff483a730d82005122f962b9f4324f",
    },
    "lemniscate": {
        "step": 29999,
        "gaussians": 394366,
        "checkpoint_sha256": "a8a1064a1d95a9bdc642c1ad540c8dcd2b00b28680c23978f6c47e258b611a32",
        "transform_sha256": "d5be6872b9a89c07547bff962ce32f5513b66fec175eef4ac7585cacbeb46333",
    },
}

GENERATION_CODE_PATHS = (
    Path(__file__).resolve(),
    REPOSITORY_ROOT / "quadpilot_repro" / "data_generation.py",
    REPOSITORY_ROOT / "quadpilot_repro" / "environment.py",
    REPOSITORY_ROOT / "quadpilot_repro" / "gate_sampling.py",
    REPOSITORY_ROOT / "quadpilot_repro" / "renderer.py",
    REPOSITORY_ROOT / "quadpilot_repro" / "tracks.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def generation_code_sha256() -> dict[str, str]:
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): sha256_file(path)
        for path in GENERATION_CODE_PATHS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=sorted(DEFAULT_RUNS), default="circle")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--min-distance", type=float, default=0.35)
    parser.add_argument("--max-distance", type=float, default=2.0)
    parser.add_argument("--lateral", type=float, default=0.55)
    parser.add_argument("--vertical", type=float, default=0.32)
    parser.add_argument("--yaw-jitter-deg", type=float, default=25.0)
    parser.add_argument("--image-margin-px", type=float, default=32.0)
    parser.add_argument("--maximum-sampling-rejections", type=int, default=100)
    parser.add_argument("--maximum-failures", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a strictly verified deterministic prefix; may also be used "
            "for the first invocation"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    focus_config = GateFocusConfig(
        min_approach_distance_m=args.min_distance,
        max_approach_distance_m=args.max_distance,
        max_lateral_offset_m=args.lateral,
        max_vertical_offset_m=args.vertical,
        max_yaw_jitter_deg=args.yaw_jitter_deg,
        image_margin_px=args.image_margin_px,
        maximum_rejections=args.maximum_sampling_rejections,
    )
    intrinsics = CameraIntrinsics()
    # Construct the CPU-only sampler before loading renderer assets so invalid
    # intrinsics/margins fail before any CUDA work.
    sampler = GateFocusedPoseSampler(
        args.track,
        BASE_DATASET_BOUNDS[args.track],
        focus_config,
        intrinsics,
    )
    configure_wsl_cuda_toolchain(args.cuda_home)
    run_dir = (args.run_dir or DEFAULT_RUNS[args.track]).resolve()
    checkpoints = sorted((run_dir / "nerfstudio_models").glob("step-*.ckpt"))
    transform = run_dir / "dataparser_transforms.json"
    if len(checkpoints) != 1 or not transform.is_file():
        raise FileNotFoundError(
            f"expected exactly one checkpoint and dataparser transform in {run_dir}"
        )

    expectation = EXPECTED_ASSETS[args.track]
    checkpoint_sha256 = sha256_file(checkpoints[0])
    transform_sha256 = sha256_file(transform)
    code_sha256 = generation_code_sha256()
    if checkpoint_sha256 != expectation["checkpoint_sha256"]:
        raise RuntimeError("renderer checkpoint SHA-256 does not match the locked asset")
    if transform_sha256 != expectation["transform_sha256"]:
        raise RuntimeError("dataparser transform SHA-256 does not match the locked asset")
    renderer = RenderOnlySplatRenderer(
        checkpoints[0],
        transform,
        expected_step=expectation["step"],
        expected_gaussians=expectation["gaussians"],
    )
    generator = ReproDatasetGenerator(
        renderer,
        args.output_dir,
        track=args.track,
        bounds=BASE_DATASET_BOUNDS[args.track],
        intrinsics=intrinsics,
        seed=args.seed,
        pose_sampler=sampler,
        provenance={
            "generator": "scripts/generate_repro_gate_dataset.py",
            "checkpoint": str(checkpoints[0]),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": renderer.checkpoint_step,
            "gaussian_count": renderer.gaussian_count,
            "dataparser_transform": str(transform),
            "dataparser_transform_sha256": transform_sha256,
            "generation_code_sha256": code_sha256,
            "sampling": (
                "uniform gate, incoming-side axial/lateral/vertical offsets, "
                "gate center inside margin-safe camera FOV"
            ),
            "gate_focus_config": focus_config.to_dict(),
        },
    )
    metadata = generator.generate(
        args.samples,
        maximum_failures=args.maximum_failures,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "dataset": str(args.output_dir.resolve()),
                "track": metadata["track"],
                "samples": metadata["n_frames"],
                "attempts": metadata["attempts"],
                "render_failures": metadata["render_failures"],
                "gate_focus_config": focus_config.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
