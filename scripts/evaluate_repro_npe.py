#!/usr/bin/env python3
"""Evaluate a reproducible Quad Pilots NPE checkpoint sample-by-sample."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from quadpilot_repro.npe import (  # noqa: E402
    MetricAccumulator,
    NPEImageDataset,
    atomic_json_save,
    build_dataset_index,
    build_image_transform,
    create_or_load_split_manifest,
    decode_predictions,
    filter_records_by_source_ids,
    load_repro_npe_checkpoint,
    make_dataloader,
    pose_error_vectors,
    predict_poses,
    records_for_split,
    seed_everything,
    software_versions,
    validate_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a repro NPE checkpoint with sample-weighted metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-dir",
        action="append",
        help="Dataset directory; defaults to the paths recorded in the checkpoint",
    )
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument(
        "--source-id",
        action="append",
        help=(
            "Exact checkpoint dataset source_id to retain after selecting the frozen split; "
            "repeat to retain multiple sources"
        ),
    )
    parser.add_argument(
        "--split-manifest",
        help="Split JSON; defaults to CHECKPOINT_DIR/split_manifest.json, then embedded manifest",
    )
    parser.add_argument("--output", help="Metrics JSON output path")
    parser.add_argument("--predictions-jsonl", help="Optional per-sample decoded predictions")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Evaluate a different dataset; requires --split all to avoid reusing the training split",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    return torch.device(requested)


def resolve_amp(requested: str, device: torch.device) -> bool:
    if requested == "on" and device.type != "cuda":
        raise RuntimeError("--amp on requires CUDA")
    return device.type == "cuda" and requested in {"auto", "on"}


def atomic_jsonl_save(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    *,
    device: torch.device,
    normalizer: Any,
    loss_config: Any,
    amp_enabled: bool,
    collect_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    accumulator = MetricAccumulator()
    prediction_rows: list[dict[str, Any]] = []
    for images, targets, keys in loader:
        targets = targets.to(device, non_blocking=True)
        prediction_batch = predict_poses(
            model,
            images,
            normalizer,
            device=device,
            amp_enabled=amp_enabled,
        )
        outputs = prediction_batch.normalized_output
        accumulator.update(outputs, targets, normalizer, loss_config)
        if collect_predictions:
            decoded_outputs = prediction_batch.xyz_yaw.detach().cpu()
            decoded_targets = decode_predictions(targets, normalizer).detach().cpu()
            position_errors, yaw_errors = pose_error_vectors(outputs, targets, normalizer)
            for index, key in enumerate(keys):
                prediction_rows.append(
                    {
                        "key": key,
                        "prediction": {
                            "x": float(decoded_outputs[index, 0]),
                            "y": float(decoded_outputs[index, 1]),
                            "z": float(decoded_outputs[index, 2]),
                            "yaw_rad": float(decoded_outputs[index, 3]),
                        },
                        "target": {
                            "x": float(decoded_targets[index, 0]),
                            "y": float(decoded_targets[index, 1]),
                            "z": float(decoded_targets[index, 2]),
                            "yaw_rad": float(decoded_targets[index, 3]),
                        },
                        "position_error_cm": float(position_errors[index]),
                        "yaw_error_deg": float(yaw_errors[index]),
                    }
                )
    return accumulator.compute(), prediction_rows


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative")
    if args.source_id and args.split == "all":
        raise ValueError(
            "--source-id requires a frozen train, val, or test split; "
            "--split all could mix training and validation records"
        )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    loaded = load_repro_npe_checkpoint(checkpoint_path, device="cpu")
    checkpoint = loaded.checkpoint
    data_dirs = args.data_dir or [source["path"] for source in checkpoint["dataset"]["sources"]]
    fingerprint_mode = checkpoint["dataset"]["fingerprint_mode"]
    print(f"Indexing evaluation data with {fingerprint_mode!r} fingerprints...")
    dataset = build_dataset_index(data_dirs, fingerprint_mode=fingerprint_mode)
    dataset_matches = dataset.fingerprint == checkpoint["dataset"]["fingerprint"]
    if not dataset_matches and not args.allow_dataset_mismatch:
        raise ValueError(
            "evaluation dataset fingerprint differs from the checkpoint; use the original data or "
            "explicit --allow-dataset-mismatch"
        )
    if not dataset_matches and args.split != "all":
        raise ValueError("a mismatched evaluation dataset must use --split all")
    if not dataset_matches and args.source_id:
        raise ValueError(
            "--source-id requires the checkpoint-matching dataset and its frozen split"
        )

    if dataset_matches:
        if args.split_manifest:
            manifest = create_or_load_split_manifest(
                Path(args.split_manifest).expanduser().resolve(), dataset, create=False
            )
        else:
            sibling_manifest = checkpoint_path.parent / "split_manifest.json"
            if sibling_manifest.is_file():
                manifest = create_or_load_split_manifest(sibling_manifest, dataset, create=False)
            else:
                manifest = checkpoint["split_manifest"]
                validate_split_manifest(manifest, dataset)
        if manifest.get("manifest_sha256") != checkpoint["split_manifest"].get("manifest_sha256"):
            raise ValueError("evaluation split manifest differs from the checkpoint")
        records = records_for_split(dataset, manifest, args.split)
        split_sha256 = manifest.get("manifest_sha256")
    else:
        records = dataset.records
        split_sha256 = None
    if not records:
        raise ValueError(f"evaluation split {args.split!r} is empty")
    pre_filter_sample_count = len(records)
    requested_source_ids = list(args.source_id or [])
    records = filter_records_by_source_ids(records, dataset, requested_source_ids)
    evaluated_source_ids = sorted({record.source_id for record in records})
    evaluated_sources = [
        source for source in dataset.sources if source.get("source_id") in evaluated_source_ids
    ]
    if requested_source_ids:
        print(
            "Frozen-split source filter: "
            f"requested={requested_source_ids}, samples={pre_filter_sample_count}->{len(records)}, "
            f"evaluated_sources={evaluated_source_ids}"
        )

    device = resolve_device(args.device)
    amp_enabled = resolve_amp(args.amp, device)
    seed_everything(args.seed, deterministic=True)
    model = loaded.model.to(device)
    normalizer = loaded.normalizer
    preprocess = loaded.preprocess
    loss_config = loaded.loss_config
    image_dataset = NPEImageDataset(
        records,
        normalizer,
        build_image_transform(preprocess, training=False),
    )
    loader = make_dataloader(
        image_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics, prediction_rows = evaluate(
        model,
        loader,
        device=device,
        normalizer=normalizer,
        loss_config=loss_config,
        amp_enabled=amp_enabled,
        collect_predictions=bool(args.predictions_jsonl),
    )
    result = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": loaded.sha256,
            "epoch": checkpoint.get("epoch"),
            "model_config": checkpoint["model_config"],
        },
        "evaluation_dataset": {
            "fingerprint": dataset.fingerprint,
            "fingerprint_mode": dataset.fingerprint_mode,
            "matches_training_dataset": dataset_matches,
            "sources": list(dataset.sources),
            "split": args.split,
            "split_manifest_sha256": split_sha256,
            "source_filter": {
                "applied": bool(requested_source_ids),
                "requested_source_ids": requested_source_ids,
                "pre_filter_sample_count": pre_filter_sample_count,
                "post_filter_sample_count": len(records),
            },
            "evaluated_sources": evaluated_sources,
        },
        "metrics": metrics,
        "reference_targets": {
            "fine_tuned_offline_position_mean_cm_lte": 8.9,
            "meets_position_target": metrics["position_error_cm"]["mean"] <= 8.9,
        },
        "runtime": {
            "device": str(device),
            "amp": amp_enabled,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "software_versions": software_versions(),
        },
        "evaluated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint_path.parent / f"evaluation_{args.split}.json"
    )
    targets = [output_path]
    predictions_path = Path(args.predictions_jsonl).expanduser().resolve() if args.predictions_jsonl else None
    if predictions_path:
        targets.append(predictions_path)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing evaluation outputs: {existing}")
    atomic_json_save(output_path, result)
    if predictions_path:
        atomic_jsonl_save(predictions_path, prediction_rows)

    position = metrics["position_error_cm"]
    yaw = metrics["yaw_error_deg"]
    print(
        f"{args.split}: n={metrics['sample_count']}, position mean/std/p95/max="
        f"{position['mean']:.3f}/{position['std']:.3f}/{position['p95']:.3f}/{position['max']:.3f} cm"
    )
    print(
        f"yaw mean/std/p95/max={yaw['mean']:.3f}/{yaw['std']:.3f}/"
        f"{yaw['p95']:.3f}/{yaw['max']:.3f} deg"
    )
    print(f"Metrics written locally to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
