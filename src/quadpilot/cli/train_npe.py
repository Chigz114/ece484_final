#!/usr/bin/env python3
"""Train or fine-tune the Quad Pilots NPE with frozen data provenance.

Examples (all paths stay local)::

    python scripts/train_repro_npe.py \
        --data-dir npe_datasets/lemniscate \
        --output-dir npe_models/lemniscate_repro

    python scripts/train_repro_npe.py \
        --data-dir npe_datasets/lemniscate \
        --data-dir npe_datasets/lemniscate_gate_focused \
        --output-dir npe_models/lemniscate_finetuned_repro \
        --init-checkpoint npe_models/lemniscate_repro/best_npe.pth \
        --lr 1e-5 --epochs 30

ImageNet initialization is never downloaded by default.  It is only requested
when ``--weights imagenet1k_v1`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

from quadpilot.perception.npe import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    NPEImageDataset,
    NPEModel,
    PoseLossConfig,
    PoseNormalizer,
    PreprocessConfig,
    atomic_json_save,
    atomic_torch_save,
    build_dataset_index,
    build_image_transform,
    capture_rng_state,
    convert_legacy_state_to_normalized_outputs,
    create_or_load_split_manifest,
    evaluate_model,
    load_torch_checkpoint,
    make_dataloader,
    model_from_checkpoint,
    records_for_split,
    restore_rng_state,
    seed_everything,
    sha256_file,
    software_versions,
    torchvision_weight_provenance,
    train_one_epoch,
    validate_repro_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible ResNet NPE training (RGB -> xyz + sin/cos yaw)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        action="append",
        required=True,
        help="Dataset directory; repeat to combine base and gate-focused data",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Frozen split JSON (default: OUTPUT_DIR/split_manifest.json)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fingerprint-mode", choices=("full", "stat"), default="full")

    parser.add_argument(
        "--backbone", choices=("resnet18", "resnet34", "resnet50"), default="resnet50"
    )
    parser.add_argument(
        "--weights",
        choices=("none", "imagenet1k_v1"),
        default="none",
        help="Initialization; imagenet1k_v1 is the only option that may access the network",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--brightness", type=float, default=0.05)
    parser.add_argument("--contrast", type=float, default=0.05)
    parser.add_argument("--saturation", type=float, default=0.02)

    parser.add_argument(
        "--epochs", type=int, default=100, help="Total epochs, including resumed epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Physical device batch size"
    )
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Opt out of strict deterministic PyTorch algorithms",
    )
    parser.add_argument("--save-every", type=int, default=10)

    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--resume",
        help="Resume an interrupted repro checkpoint, including optimizer/scheduler/RNG",
    )
    initialization.add_argument(
        "--init-checkpoint",
        help="Start a new fine-tune run from a repro checkpoint with a fresh optimizer",
    )
    initialization.add_argument(
        "--init-legacy-checkpoint",
        help="Explicitly import the historical raw-XYZ five-output checkpoint",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested, but torch.cuda.is_available() is false"
        )
    return torch.device(requested)


def resolve_amp(requested: str, device: torch.device) -> bool:
    if requested == "on" and device.type != "cuda":
        raise RuntimeError("--amp on requires a CUDA device")
    return device.type == "cuda" and requested in {"auto", "on"}


def _git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def repository_provenance() -> dict[str, Any]:
    status = _git_value("status", "--short")
    return {
        "root": str(REPOSITORY_ROOT),
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "tracked_code_sha256": {
            "src/quadpilot/perception/npe.py": sha256_file(
                REPOSITORY_ROOT / "src" / "quadpilot" / "perception" / "npe.py"
            ),
            "src/quadpilot/cli/train_npe.py": sha256_file(Path(__file__).resolve()),
        },
    }


def training_config(args: argparse.Namespace, amp_enabled: bool) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "deterministic": not args.allow_nondeterministic,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "num_workers": args.num_workers,
        "amp": amp_enabled,
        "scheduler": {"name": "CosineAnnealingLR", "T_max": args.epochs},
    }


def validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        raise ValueError("--batch-size and --accumulation-steps must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("--lr must be positive and --weight-decay non-negative")
    if args.image_size <= 0 or args.num_workers < 0:
        raise ValueError("--image-size must be positive and --num-workers non-negative")
    if args.save_every < 0:
        raise ValueError("--save-every cannot be negative")
    if min(args.brightness, args.contrast, args.saturation) < 0:
        raise ValueError("color jitter values cannot be negative")


def _same_normalizer(first: PoseNormalizer, second: PoseNormalizer) -> bool:
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(first.mean + first.std, second.mean + second.std)
    )


def _maybe_save_best_checkpoint(
    *,
    candidate_val_position_cm: float,
    best_val_position_cm: float,
    payload: dict[str, Any],
    destination: Path,
) -> tuple[float, bool]:
    """Atomically retain a candidate only when its validation mean improves."""

    candidate = float(candidate_val_position_cm)
    if not math.isfinite(candidate):
        raise ValueError("validation position error must be finite")
    if candidate >= best_val_position_cm:
        return best_val_position_cm, False
    atomic_torch_save(destination, payload)
    return candidate, True


def _checkpoint_payload(
    *,
    model: NPEModel,
    model_config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_val_position_cm: float,
    normalizer: PoseNormalizer,
    preprocess: PreprocessConfig,
    loss_config: PoseLossConfig,
    dataset: Any,
    split_manifest: dict[str, Any],
    run_config: dict[str, Any],
    history: list[dict[str, Any]],
    epoch_metrics: dict[str, Any],
    provenance: dict[str, Any],
    checkpoint_kind: str = "trained_epoch",
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "epoch": epoch,
        "completed_epochs": max(0, epoch + 1),
        "global_step": global_step,
        "best_val_position_cm": best_val_position_cm,
        "model_config": copy.deepcopy(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "normalizer": normalizer.to_dict(),
        "preprocess": preprocess.to_dict(),
        "loss_config": vars(loss_config),
        "dataset": {
            "fingerprint": dataset.fingerprint,
            "fingerprint_mode": dataset.fingerprint_mode,
            "sources": list(dataset.sources),
        },
        "split_manifest": copy.deepcopy(split_manifest),
        "training_config": copy.deepcopy(run_config),
        "history": copy.deepcopy(history),
        "epoch_metrics": copy.deepcopy(epoch_metrics),
        "rng_state": capture_rng_state(),
        "software_versions": software_versions(),
        "provenance": copy.deepcopy(provenance),
        "saved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_arguments(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = (
        Path(args.split_manifest).expanduser().resolve()
        if args.split_manifest
        else output_dir / "split_manifest.json"
    )
    if not args.resume and any(
        (output_dir / name).exists() for name in ("last_npe.pth", "best_npe.pth")
    ):
        raise FileExistsError(
            f"refusing to overwrite an existing run in {output_dir}; use --resume or a new output directory"
        )

    device = resolve_device(args.device)
    amp_enabled = resolve_amp(args.amp, device)
    seed_everything(args.seed, deterministic=not args.allow_nondeterministic)
    print(
        f"Indexing {len(args.data_dir)} dataset source(s) with {args.fingerprint_mode!r} fingerprints..."
    )
    dataset = build_dataset_index(args.data_dir, fingerprint_mode=args.fingerprint_mode)
    split_manifest = create_or_load_split_manifest(
        split_path,
        dataset,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    train_records = records_for_split(dataset, split_manifest, "train")
    val_records = records_for_split(dataset, split_manifest, "val")
    if not train_records or not val_records:
        raise ValueError("training and validation splits must both contain samples")

    preprocess = PreprocessConfig(
        width=args.image_size,
        height=args.image_size,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
    )
    loss_config = PoseLossConfig()
    run_config = training_config(args, amp_enabled)
    resume_checkpoint: dict[str, Any] | None = None
    initial_checkpoint_hash: str | None = None
    weight_initialization: dict[str, Any] | None = None
    checkpoint_lineage: list[dict[str, Any]] = []
    normalizer_provenance: dict[str, Any] | None = None
    start_epoch = 0
    global_step = 0
    best_val_position_cm = math.inf
    history: list[dict[str, Any]] = []
    initialization_baseline: dict[str, Any] | None = None

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        resume_checkpoint = load_torch_checkpoint(resume_path, map_location=device)
        validate_repro_checkpoint(resume_checkpoint)
        if resume_checkpoint["dataset"]["fingerprint"] != dataset.fingerprint:
            raise ValueError("resume dataset fingerprint differs from the checkpoint")
        if resume_checkpoint["dataset"]["fingerprint_mode"] != dataset.fingerprint_mode:
            raise ValueError("resume fingerprint mode differs from the checkpoint")
        if resume_checkpoint["split_manifest"].get(
            "manifest_sha256"
        ) != split_manifest.get("manifest_sha256"):
            raise ValueError("resume split manifest differs from the checkpoint")
        stored_config = resume_checkpoint["training_config"]
        if stored_config != run_config:
            differing = sorted(
                key
                for key in set(stored_config) | set(run_config)
                if stored_config.get(key) != run_config.get(key)
            )
            raise ValueError(f"resume training configuration changed for: {differing}")
        model, normalizer, stored_preprocess, stored_loss = model_from_checkpoint(
            resume_checkpoint, device=device
        )
        if resume_checkpoint["model_config"]["backbone"] != args.backbone:
            raise ValueError("--backbone must match the resume checkpoint")
        if args.weights != "none":
            raise ValueError("--weights must be none when --resume is used")
        if stored_preprocess != preprocess or stored_loss != loss_config:
            raise ValueError("resume preprocessing or loss configuration changed")
        model_config = copy.deepcopy(resume_checkpoint["model_config"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        global_step = int(resume_checkpoint.get("global_step", 0))
        best_val_position_cm = float(
            resume_checkpoint.get("best_val_position_cm", math.inf)
        )
        history = list(resume_checkpoint.get("history", []))
        initial_checkpoint_hash = sha256_file(resume_path)
        weight_initialization = copy.deepcopy(
            resume_checkpoint.get("provenance", {}).get(
                "weight_initialization",
                {
                    "identifier": resume_checkpoint["model_config"].get(
                        "initial_weights", "unknown"
                    )
                },
            )
        )
        checkpoint_lineage = list(
            resume_checkpoint.get("provenance", {}).get("checkpoint_lineage", [])
        ) + [
            {
                "operation": "resume",
                "path": str(resume_path),
                "sha256": initial_checkpoint_hash,
            }
        ]
        initialization_baseline = copy.deepcopy(
            resume_checkpoint.get("provenance", {}).get("initialization_baseline")
        )
        normalizer_provenance = copy.deepcopy(
            resume_checkpoint.get("provenance", {}).get("normalizer")
        )
    elif args.init_checkpoint:
        init_path = Path(args.init_checkpoint).expanduser().resolve()
        init_checkpoint = load_torch_checkpoint(init_path, map_location=device)
        validate_repro_checkpoint(init_checkpoint)
        model, normalizer, stored_preprocess, stored_loss = model_from_checkpoint(
            init_checkpoint, device=device
        )
        if init_checkpoint["model_config"]["backbone"] != args.backbone:
            raise ValueError("--backbone must match the fine-tune checkpoint")
        if args.weights != "none":
            raise ValueError("--weights must be none when --init-checkpoint is used")
        if stored_preprocess != preprocess or stored_loss != loss_config:
            raise ValueError(
                "fine-tune preprocessing/loss must match the initialization checkpoint"
            )
        model_config = copy.deepcopy(init_checkpoint["model_config"])
        model_config["initial_weights"] = "checkpoint"
        initial_checkpoint_hash = sha256_file(init_path)
        weight_initialization = {
            "identifier": "repro_checkpoint",
            "path": str(init_path),
            "sha256": initial_checkpoint_hash,
        }
        checkpoint_lineage = [{"operation": "fine_tune_init", **weight_initialization}]
        normalizer_provenance = {
            "strategy": "inherited_initial_checkpoint",
            "source_checkpoint_sha256": initial_checkpoint_hash,
            "source_dataset_fingerprint": init_checkpoint["dataset"]["fingerprint"],
            "normalizer": normalizer.to_dict(),
        }
    elif args.init_legacy_checkpoint:
        legacy_path = Path(args.init_legacy_checkpoint).expanduser().resolve()
        legacy_checkpoint = load_torch_checkpoint(legacy_path, map_location=device)
        legacy_state = legacy_checkpoint.get("model_state_dict", legacy_checkpoint)
        if not isinstance(legacy_state, dict):
            raise ValueError(
                "legacy checkpoint does not contain a model state dictionary"
            )
        if args.weights != "none":
            raise ValueError(
                "--weights must be none when --init-legacy-checkpoint is used"
            )
        normalizer = PoseNormalizer.fit(train_records)
        model = NPEModel(backbone=args.backbone, weights="none").to(device)
        model.load_state_dict(
            convert_legacy_state_to_normalized_outputs(legacy_state, normalizer),
            strict=True,
        )
        model_config = copy.deepcopy(model.config)
        model_config["initial_weights"] = "legacy_checkpoint_affine_converted"
        initial_checkpoint_hash = sha256_file(legacy_path)
        weight_initialization = {
            "identifier": "legacy_checkpoint_affine_converted",
            "path": str(legacy_path),
            "sha256": initial_checkpoint_hash,
        }
        checkpoint_lineage = [{"operation": "legacy_import", **weight_initialization}]
        normalizer_provenance = {
            "strategy": "fit_frozen_training_split",
            "dataset_fingerprint": dataset.fingerprint,
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "sample_count": len(train_records),
            "normalizer": normalizer.to_dict(),
        }
    else:
        normalizer = PoseNormalizer.fit(train_records)
        model = NPEModel(backbone=args.backbone, weights=args.weights).to(device)
        model_config = copy.deepcopy(model.config)
        weight_initialization = torchvision_weight_provenance(
            args.backbone, args.weights
        )
        normalizer_provenance = {
            "strategy": "fit_frozen_training_split",
            "dataset_fingerprint": dataset.fingerprint,
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "sample_count": len(train_records),
            "normalizer": normalizer.to_dict(),
        }

    if args.resume:
        # Older checkpoints did not record how the normalizer was obtained.  A
        # base run can be identified by reproducing the frozen-train fit.  A
        # fine-tune intentionally inherits the initialization checkpoint's
        # output space, so it must not be compared to a new combined-data fit.
        if normalizer_provenance is None:
            fitted = PoseNormalizer.fit(train_records)
            if _same_normalizer(normalizer, fitted):
                normalizer_provenance = {
                    "strategy": "fit_frozen_training_split",
                    "dataset_fingerprint": dataset.fingerprint,
                    "split_manifest_sha256": split_manifest["manifest_sha256"],
                    "sample_count": len(train_records),
                    "normalizer": normalizer.to_dict(),
                    "inferred_from_legacy_checkpoint": True,
                }
            elif model_config.get("initial_weights") == "checkpoint":
                normalizer_provenance = {
                    "strategy": "inherited_initial_checkpoint",
                    "source_checkpoint_sha256": (weight_initialization or {}).get(
                        "sha256"
                    ),
                    "normalizer": normalizer.to_dict(),
                    "inferred_from_legacy_checkpoint": True,
                }
            else:
                raise ValueError(
                    "resume normalizer is neither the frozen-training fit nor a marked "
                    "fine-tune inheritance"
                )

        recorded_normalizer = normalizer_provenance.get("normalizer")
        if recorded_normalizer is not None and not _same_normalizer(
            normalizer, PoseNormalizer.from_dict(recorded_normalizer)
        ):
            raise ValueError("resume normalizer differs from its recorded provenance")
        strategy = normalizer_provenance.get("strategy")
        if strategy == "fit_frozen_training_split":
            fitted = PoseNormalizer.fit(train_records)
            if not _same_normalizer(normalizer, fitted):
                raise ValueError(
                    "resume training-position normalizer differs despite matching dataset"
                )
        elif strategy != "inherited_initial_checkpoint":
            raise ValueError(
                f"unsupported resume normalizer provenance strategy: {strategy!r}"
            )

    train_dataset = NPEImageDataset(
        train_records,
        normalizer,
        build_image_transform(preprocess, training=True),
    )
    val_dataset = NPEImageDataset(
        val_records,
        normalizer,
        build_image_transform(preprocess, training=False),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(resume_checkpoint.get("scaler_state_dict", {}))
        restore_rng_state(resume_checkpoint["rng_state"])

    provenance = {
        "command": [sys.executable, *sys.argv],
        "repository": repository_provenance(),
        "initial_checkpoint": (
            {
                "path": str(
                    Path(
                        args.resume
                        or args.init_checkpoint
                        or args.init_legacy_checkpoint
                    )
                    .expanduser()
                    .resolve()
                ),
                "sha256": initial_checkpoint_hash,
                "legacy_raw_xyz_affine_conversion": bool(args.init_legacy_checkpoint),
            }
            if initial_checkpoint_hash
            else None
        ),
        "position_coordinates": "NeRF world coordinates in meters",
        "pose_label_format": ["x", "y", "z", "roll=0", "pitch=0", "yaw radians"],
        "model_output_semantics": "normalized xyz plus sin(yaw), cos(yaw)",
        "weight_initialization": weight_initialization,
        "normalizer": normalizer_provenance,
        "checkpoint_lineage": checkpoint_lineage,
    }

    if args.init_checkpoint or args.init_legacy_checkpoint:
        baseline_loader = make_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed - 1,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        baseline_metrics = evaluate_model(
            model,
            baseline_loader,
            device=device,
            normalizer=normalizer,
            loss_config=loss_config,
            amp_enabled=amp_enabled,
        )
        baseline_val = float(baseline_metrics["position_error_cm"]["mean"])
        initialization_baseline = {
            "split": "val",
            "sample_count": int(baseline_metrics["sample_count"]),
            "metrics": baseline_metrics,
        }
        provenance["initialization_baseline"] = copy.deepcopy(initialization_baseline)
        baseline_epoch_metrics = {
            "epoch": -1,
            "checkpoint_kind": "initialization_baseline",
            "train": None,
            "val": baseline_metrics,
        }
        baseline_payload = _checkpoint_payload(
            model=model,
            model_config=model_config,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=-1,
            global_step=0,
            best_val_position_cm=baseline_val,
            normalizer=normalizer,
            preprocess=preprocess,
            loss_config=loss_config,
            dataset=dataset,
            split_manifest=split_manifest,
            run_config=run_config,
            history=[],
            epoch_metrics=baseline_epoch_metrics,
            provenance=provenance,
            checkpoint_kind="initialization_baseline",
        )
        best_val_position_cm, saved = _maybe_save_best_checkpoint(
            candidate_val_position_cm=baseline_val,
            best_val_position_cm=math.inf,
            payload=baseline_payload,
            destination=output_dir / "best_npe.pth",
        )
        if not saved:
            raise AssertionError(
                "initialization baseline was not retained as the first candidate"
            )
        print(
            "Initialization baseline on combined frozen val: "
            f"position={baseline_val:.3f} cm, "
            f"yaw={baseline_metrics['yaw_error_deg']['mean']:.3f} deg"
        )
    elif initialization_baseline is not None:
        provenance["initialization_baseline"] = copy.deepcopy(initialization_baseline)

    print(
        f"Device={device}, AMP={amp_enabled}, backbone={model_config['backbone']}, "
        f"train/val/test={split_manifest['counts']['train']}/"
        f"{split_manifest['counts']['val']}/{split_manifest['counts']['test']}"
    )
    print(f"Dataset SHA-256: {dataset.fingerprint}")
    print(f"Split SHA-256:   {split_manifest['manifest_sha256']}")
    print(f"Position mean:   {normalizer.mean}")
    print(f"Position std:    {normalizer.std}")
    if start_epoch >= args.epochs:
        raise ValueError(
            f"checkpoint already completed {start_epoch} epochs (target={args.epochs})"
        )

    last_payload: dict[str, Any] | None = None
    for epoch in range(start_epoch, args.epochs):
        # Epoch-derived loader seeds make sample order and worker augmentation
        # repeatable after a resume at an epoch boundary.
        train_loader = make_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + 2 * epoch,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = make_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 2 * epoch + 1,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        train_metrics, steps = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            normalizer=normalizer,
            loss_config=loss_config,
            accumulation_steps=args.accumulation_steps,
            amp_enabled=amp_enabled,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
        )
        global_step += steps
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            normalizer=normalizer,
            loss_config=loss_config,
            amp_enabled=amp_enabled,
        )
        scheduler.step()
        current_val = float(val_metrics["position_error_cm"]["mean"])
        if not math.isfinite(current_val):
            raise ValueError("validation position error must be finite")
        previous_best_val_position_cm = best_val_position_cm
        next_best_val_position_cm = min(previous_best_val_position_cm, current_val)
        epoch_metrics = {
            "epoch": epoch + 1,
            "learning_rate_used": learning_rate_used,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimizer_steps": steps,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_metrics)
        last_payload = _checkpoint_payload(
            model=model,
            model_config=model_config,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_val_position_cm=next_best_val_position_cm,
            normalizer=normalizer,
            preprocess=preprocess,
            loss_config=loss_config,
            dataset=dataset,
            split_manifest=split_manifest,
            run_config=run_config,
            history=history,
            epoch_metrics=epoch_metrics,
            provenance=provenance,
        )
        atomic_torch_save(output_dir / "last_npe.pth", last_payload)
        best_val_position_cm, improved = _maybe_save_best_checkpoint(
            candidate_val_position_cm=current_val,
            best_val_position_cm=previous_best_val_position_cm,
            payload=last_payload,
            destination=output_dir / "best_npe.pth",
        )
        if best_val_position_cm != next_best_val_position_cm:
            raise AssertionError(
                "best-checkpoint decision disagrees with checkpoint payload"
            )
        if args.save_every and (epoch + 1) % args.save_every == 0:
            atomic_torch_save(
                output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth", last_payload
            )
        atomic_json_save(
            output_dir / "training_history.json",
            {
                "dataset_fingerprint": dataset.fingerprint,
                "split_manifest_sha256": split_manifest["manifest_sha256"],
                "best_val_position_cm": best_val_position_cm,
                "initialization_baseline": initialization_baseline,
                "history": history,
            },
        )
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}: "
            f"loss {train_metrics['loss']['total']:.5f}/{val_metrics['loss']['total']:.5f}, "
            f"position {train_metrics['position_error_cm']['mean']:.2f}/"
            f"{current_val:.2f} cm, yaw {train_metrics['yaw_error_deg']['mean']:.2f}/"
            f"{val_metrics['yaw_error_deg']['mean']:.2f} deg"
            + (" [best]" if improved else "")
        )

    if last_payload is None:
        raise AssertionError("training loop produced no checkpoint")
    atomic_torch_save(output_dir / "final_npe.pth", last_payload)
    print(
        f"Training complete. Best validation position error: {best_val_position_cm:.3f} cm"
    )
    print(f"Local outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
