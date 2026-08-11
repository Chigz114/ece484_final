#!/usr/bin/env python3
"""Generate label-safe NPE images from a recovered Quad Pilots GSplat run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

from quadpilot.datasets.generation import (  # noqa: E402
    ReproDatasetGenerator,
    resolve_dataset_bounds,
)
from quadpilot.perception.renderer import RenderOnlySplatRenderer  # noqa: E402
from quadpilot.perception.runtime import configure_wsl_cuda_toolchain  # noqa: E402

DEFAULT_RUNS = {
    "circle": REPOSITORY_ROOT
    / "outputs"
    / "circle"
    / "splatfacto"
    / "2025-05-09_144210",
    "uturn": REPOSITORY_ROOT / "outputs" / "uturn" / "splatfacto" / "2025-05-09_151825",
    "lemniscate": REPOSITORY_ROOT
    / "outputs"
    / "lemniscate"
    / "splatfacto"
    / "2025-05-09_153156",
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
    REPOSITORY_ROOT / "src" / "quadpilot" / "datasets" / "generation.py",
    REPOSITORY_ROOT / "src" / "quadpilot" / "perception" / "runtime.py",
    REPOSITORY_ROOT / "src" / "quadpilot" / "perception" / "renderer.py",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=sorted(DEFAULT_RUNS), default="circle")
    parser.add_argument(
        "--region",
        choices=("base", "launch-corridor"),
        default="base",
        help="Named pose region; launch-corridor is defined only for Lemniscate",
    )
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--maximum-failures", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a strictly verified deterministic prefix; may also be used "
            "for the first invocation"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bounds = resolve_dataset_bounds(args.track, args.region)
    configure_wsl_cuda_toolchain(args.cuda_home)

    run_dir = (args.run_dir or DEFAULT_RUNS[args.track]).resolve()
    checkpoint_dir = run_dir / "nerfstudio_models"
    checkpoints = sorted(checkpoint_dir.glob("step-*.ckpt"))
    transform = run_dir / "dataparser_transforms.json"
    if len(checkpoints) != 1 or not transform.is_file():
        raise FileNotFoundError(
            f"expected exactly one checkpoint and one dataparser transform in {run_dir}"
        )

    expectation = EXPECTED_ASSETS[args.track]
    checkpoint_sha256 = sha256_file(checkpoints[0])
    transform_sha256 = sha256_file(transform)
    code_sha256 = generation_code_sha256()
    if checkpoint_sha256 != expectation["checkpoint_sha256"]:
        raise RuntimeError(
            "renderer checkpoint SHA-256 does not match the locked asset"
        )
    if transform_sha256 != expectation["transform_sha256"]:
        raise RuntimeError(
            "dataparser transform SHA-256 does not match the locked asset"
        )
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
        bounds=bounds,
        seed=args.seed,
        provenance={
            "generator": "scripts/generate_repro_npe_dataset.py",
            "checkpoint": str(checkpoints[0]),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": renderer.checkpoint_step,
            "gaussian_count": renderer.gaussian_count,
            "dataparser_transform": str(transform),
            "dataparser_transform_sha256": transform_sha256,
            "generation_code_sha256": code_sha256,
            "sampling": (
                "legacy axis-aligned bounds and uniform full yaw"
                if args.region == "base"
                else "Lemniscate launch-corridor bounds with uniform xyz and bounded yaw"
            ),
            "sampling_region": args.region,
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
                "region": args.region,
                "samples": metadata["n_frames"],
                "attempts": metadata["attempts"],
                "render_failures": metadata["render_failures"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
