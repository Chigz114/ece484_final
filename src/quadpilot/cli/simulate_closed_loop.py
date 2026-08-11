#!/usr/bin/env python3
"""Run the recovered GSplat -> NPE -> controller closed loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

from quadpilot.perception.npe import (  # noqa: E402
    build_image_transform,
    load_repro_npe_checkpoint,
    seed_everything,
)
from quadpilot.perception.renderer import RenderOnlySplatRenderer  # noqa: E402
from quadpilot.perception.runtime import configure_wsl_cuda_toolchain  # noqa: E402
from quadpilot.simulation.tracks import TRACKS  # noqa: E402
from quadpilot.simulation.visual_loop import (  # noqa: E402
    TorchNPEPredictor,
    run_visual_closed_loop,
    save_visual_loop_result,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the real Quad Pilots visual closed loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--track", choices=sorted(TRACKS), default="circle")
    parser.add_argument("--estimator", choices=("raw", "ekf", "both"), default="both")
    parser.add_argument("--npe-checkpoint", type=Path)
    parser.add_argument("--renderer-checkpoint", type=Path)
    parser.add_argument("--dataparser-transform", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=None,
        help="Optional CUDA toolkit root for gsplat first-use JIT loading",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--gate-radius", type=float, default=0.38)
    parser.add_argument("--crossing-hysteresis-m", type=float, default=0.05)
    parser.add_argument("--laps", type=int, default=2)
    parser.add_argument("--ekf-outlier-threshold", type=float, default=4.0)
    parser.add_argument("--expected-renderer-step", type=int, default=29999)
    parser.add_argument("--expected-gaussians", type=int)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Write one RGB PNG every N observations; zero writes no frames",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "repro_outputs" / "npe_closed_loop",
    )
    return parser.parse_args(argv)


def _manifest_track(track: str) -> dict[str, object]:
    manifest_path = REPOSITORY_ROOT / "configs" / "assets" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(manifest["tracks"][track])


def _resolve_assets(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    track_manifest = _manifest_track(args.track)
    runtime = dict(track_manifest.get("runtime", {}))
    run_dir = Path(
        runtime.get(
            "renderer_run_dir",
            REPOSITORY_ROOT
            / "outputs"
            / args.track
            / "splatfacto"
            / str(track_manifest["run"]),
        )
    )
    renderer_checkpoint = (
        args.renderer_checkpoint
        or run_dir / "nerfstudio_models" / "step-000029999.ckpt"
    )
    dataparser_transform = (
        args.dataparser_transform or run_dir / "dataparser_transforms.json"
    )
    npe_checkpoint = args.npe_checkpoint or Path(
        runtime.get("npe_checkpoint", REPOSITORY_ROOT / TRACKS[args.track].model_path)
    )
    resolved = tuple(
        path.expanduser().resolve()
        for path in (renderer_checkpoint, dataparser_transform, npe_checkpoint)
    )
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing closed-loop assets: " + ", ".join(missing))
    return resolved  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cuda_toolchain = (
        configure_wsl_cuda_toolchain(args.cuda_home)
        if args.cuda_home is not None
        else {}
    )
    # Splatfacto's recovered configuration uses a random background.  Seed it
    # explicitly, and reset the same stream for each independent estimator.
    seed_everything(args.seed, deterministic=False)
    renderer_checkpoint, dataparser_transform, npe_checkpoint = _resolve_assets(args)
    loaded = load_repro_npe_checkpoint(npe_checkpoint, device="cpu")
    loaded.model.to(args.device)
    image_transform = build_image_transform(loaded.preprocess, training=False)
    predictor = TorchNPEPredictor(
        loaded.model,
        loaded.normalizer,
        image_transform,
        device=args.device,
        amp_enabled=args.amp,
    )
    renderer = RenderOnlySplatRenderer(
        renderer_checkpoint,
        dataparser_transform,
        device=args.device,
        expected_step=args.expected_renderer_step,
        expected_gaussians=args.expected_gaussians,
    )
    estimators = ("raw", "ekf") if args.estimator == "both" else (args.estimator,)
    all_succeeded = True
    for estimator in estimators:
        seed_everything(args.seed, deterministic=False)
        snapshot_dir = (
            args.output_dir / f"{args.track}_{estimator}_frames"
            if args.snapshot_every
            else None
        )
        # Each estimator gets a fresh controller, plant, and EKF state while
        # sharing the same track truth initial condition and model assets.
        result = run_visual_closed_loop(
            args.track,
            renderer=renderer,
            predictor=predictor,
            estimator=estimator,
            max_steps=args.max_steps,
            dt=args.dt,
            gate_radius=args.gate_radius,
            crossing_hysteresis_m=args.crossing_hysteresis_m,
            laps=args.laps,
            ekf_outlier_threshold=args.ekf_outlier_threshold,
            snapshot_every=args.snapshot_every,
            snapshot_dir=snapshot_dir,
        )
        metadata = {
            "npe_checkpoint": str(loaded.path),
            "npe_checkpoint_sha256": loaded.sha256,
            "renderer_checkpoint": str(renderer_checkpoint),
            "renderer_checkpoint_step": renderer.checkpoint_step,
            "renderer_gaussian_count": renderer.gaussian_count,
            "dataparser_transform": str(dataparser_transform),
            "device": args.device,
            "cuda_toolchain": cuda_toolchain,
            "amp_enabled": bool(args.amp),
            "seed": int(args.seed),
        }
        json_path, npz_path = save_visual_loop_result(
            result, args.output_dir, metadata=metadata
        )
        evaluation = result.evaluation
        success_rate = 0.0 if evaluation is None else evaluation.success_rate
        mean_error = None if evaluation is None else evaluation.mean_gate_error_m
        print(
            f"{args.track}/{estimator}: success={result.succeeded} "
            f"steps={result.steps} SR={100.0 * success_rate:.2f}% "
            f"MGE={(100.0 * mean_error if mean_error is not None else float('nan')):.2f}cm"
        )
        print(f"  JSON: {json_path}")
        print(f"  NPZ:  {npz_path}")
        if result.failure_reason:
            print(f"  stopped safely: {result.failure_reason}")
        all_succeeded &= result.succeeded
    return 0 if all_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
