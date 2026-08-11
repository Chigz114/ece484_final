"""CPU regression tests for the reproducible NPE training/evaluation chain."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from quadpilot.cli.train_npe import _maybe_save_best_checkpoint
from quadpilot.perception.npe import (
    DatasetRecord,
    MetricAccumulator,
    NPEModel,
    PoseLossConfig,
    PoseNormalizer,
    build_dataset_index,
    capture_rng_state,
    convert_legacy_state_to_normalized_outputs,
    convert_state_dict_to_legacy_raw_xyz,
    create_or_load_split_manifest,
    decode_predictions,
    filter_records_by_source_ids,
    make_dataloader,
    pose_error_vectors,
    pose_loss_components,
    predict_poses,
    records_for_split,
    restore_rng_state,
    train_one_epoch,
    validate_repro_checkpoint,
)


def _write_dataset(root: Path, count: int, *, legacy: bool = False) -> None:
    images = root / "images"
    images.mkdir(parents=True)
    poses = []
    rows = []
    for index in range(count):
        pose = [
            float(index),
            -float(index) / 2.0,
            0.1 * index,
            0.0,
            0.0,
            -math.pi + index * 0.2,
        ]
        poses.append(pose)
        relative = f"images/frame_{index:05d}.png"
        pixels = np.zeros((6, 8, 3), dtype=np.uint8)
        pixels[:, :, 0] = index
        pixels[:, :, 1] = np.arange(8, dtype=np.uint8)
        pixels[:, :, 2] = 200
        Image.fromarray(pixels, mode="RGB").save(root / relative)
        rows.append(
            {"sample_id": index, "image": relative, "pose": pose, "attempt": index + 1}
        )
    metadata = {
        "schema_version": 2,
        "n_frames": count,
        "track": root.name,
        "poses": poses,
        "pose_format": ["x", "y", "z", "roll", "pitch", "yaw"],
        "image_size": [8, 6],
    }
    if not legacy:
        metadata["samples_manifest"] = "samples.jsonl"
        (root / "samples.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _record(key: str, xyz: tuple[float, float, float]) -> DatasetRecord:
    return DatasetRecord(
        key=key,
        source_id="0:test",
        source_root=Path("."),
        relative_image="unused.png",
        pose=(xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0),
        image_sha256="0" * 64,
        width=8,
        height=6,
    )


class DatasetIndexTests(unittest.TestCase):
    def test_label_safe_manifest_is_indexed_and_content_hashed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "circle"
            _write_dataset(root, 4)
            first = build_dataset_index([root])
            second = build_dataset_index([root])
            self.assertEqual(first.fingerprint_mode, "full")
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertEqual(len(first.records), 4)
            self.assertEqual(first.records[3].pose[0], 3.0)
            self.assertEqual(first.records[3].relative_image, "images/frame_00003.png")

            pixels = np.asarray(
                Image.open(first.records[0].image_path).convert("RGB")
            ).copy()
            pixels[0, 0, 0] = 99
            Image.fromarray(pixels).save(first.records[0].image_path)
            changed = build_dataset_index([root])
            self.assertNotEqual(first.fingerprint, changed.fingerprint)

    def test_legacy_index_gap_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "legacy"
            _write_dataset(root, 4, legacy=True)
            (root / "images" / "frame_00002.png").unlink()
            with self.assertRaisesRegex(ValueError, "not one-to-one"):
                build_dataset_index([root])

    def test_manifest_pose_disagreement_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "circle"
            _write_dataset(root, 3)
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            metadata["poses"][1][0] += 1.0
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                build_dataset_index([root])


class SplitManifestTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_complete_and_source_balanced(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_source = root / "base"
            second_source = root / "focused"
            _write_dataset(first_source, 20)
            _write_dataset(second_source, 10)
            dataset = build_dataset_index([first_source, second_source])
            first_path = root / "split_a.json"
            second_path = root / "split_b.json"
            first = create_or_load_split_manifest(first_path, dataset, seed=17)
            second = create_or_load_split_manifest(second_path, dataset, seed=17)
            self.assertEqual(first, second)
            sets = [set(first["splits"][name]) for name in ("train", "val", "test")]
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])
            self.assertEqual(
                set.union(*sets), {record.key for record in dataset.records}
            )
            self.assertEqual(first["counts"], {"train": 24, "val": 3, "test": 3})
            self.assertEqual(len(records_for_split(dataset, first, "test")), 3)

            test_records = records_for_split(dataset, first, "test")
            base_source = str(dataset.sources[0]["source_id"])
            gate_source = str(dataset.sources[1]["source_id"])
            base_records = filter_records_by_source_ids(
                test_records, dataset, [base_source]
            )
            gate_records = filter_records_by_source_ids(
                test_records, dataset, [gate_source]
            )
            self.assertEqual(len(base_records), 2)
            self.assertEqual(len(gate_records), 1)
            self.assertTrue(
                all(record.source_id == base_source for record in base_records)
            )
            self.assertTrue(
                all(record.source_id == gate_source for record in gate_records)
            )

            with self.assertRaisesRegex(ValueError, "unknown source_id"):
                filter_records_by_source_ids(test_records, dataset, ["9:not-a-source"])

            with self.assertRaisesRegex(ValueError, "duplicate source_id"):
                filter_records_by_source_ids(
                    test_records, dataset, [base_source, base_source]
                )

            with self.assertRaisesRegex(ValueError, "selected zero records"):
                filter_records_by_source_ids(base_records, dataset, [gate_source])

    def test_best_checkpoint_retains_initialization_until_strict_improvement(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "best_npe.pth"
            best, saved = _maybe_save_best_checkpoint(
                candidate_val_position_cm=10.0,
                best_val_position_cm=math.inf,
                payload={"checkpoint_kind": "initialization_baseline", "epoch": -1},
                destination=destination,
            )
            self.assertTrue(saved)
            self.assertEqual(best, 10.0)

            best, saved = _maybe_save_best_checkpoint(
                candidate_val_position_cm=12.0,
                best_val_position_cm=best,
                payload={"checkpoint_kind": "trained_epoch", "epoch": 0},
                destination=destination,
            )
            self.assertFalse(saved)
            self.assertEqual(best, 10.0)
            retained = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertEqual(retained["checkpoint_kind"], "initialization_baseline")
            self.assertEqual(retained["epoch"], -1)

            best, saved = _maybe_save_best_checkpoint(
                candidate_val_position_cm=8.0,
                best_val_position_cm=best,
                payload={"checkpoint_kind": "trained_epoch", "epoch": 1},
                destination=destination,
            )
            self.assertTrue(saved)
            self.assertEqual(best, 8.0)
            improved = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertEqual(improved["checkpoint_kind"], "trained_epoch")
            self.assertEqual(improved["epoch"], 1)

    def test_existing_split_rejects_modified_dataset(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "base"
            _write_dataset(root, 10)
            dataset = build_dataset_index([root])
            split_path = Path(temporary) / "split.json"
            create_or_load_split_manifest(split_path, dataset, seed=1)
            pixels = np.asarray(
                Image.open(root / "images" / "frame_00000.png").convert("RGB")
            ).copy()
            pixels[0, 0, :] = 123
            Image.fromarray(pixels).save(root / "images" / "frame_00000.png")
            modified = build_dataset_index([root])
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                create_or_load_split_manifest(split_path, modified)

    def test_adding_gate_source_preserves_frozen_base_membership(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_source = root / "base"
            gate_source = root / "gate"
            _write_dataset(base_source, 20)
            _write_dataset(gate_source, 10)

            base_dataset = build_dataset_index([base_source])
            combined_dataset = build_dataset_index([base_source, gate_source])
            base_manifest = create_or_load_split_manifest(
                root / "base_split.json", base_dataset, seed=42
            )
            combined_manifest = create_or_load_split_manifest(
                root / "combined_split.json", combined_dataset, seed=42
            )

            for split in ("train", "val", "test"):
                expected = set(base_manifest["splits"][split])
                combined_base = {
                    key
                    for key in combined_manifest["splits"][split]
                    if key.startswith("0:base/")
                }
                self.assertEqual(combined_base, expected)


class PoseSemanticsTests(unittest.TestCase):
    def test_normalized_position_roundtrip_and_wrapped_yaw_error(self) -> None:
        records = (
            _record("a", (-2.0, -6.0, -1.0)),
            _record("b", (0.0, -2.0, 1.0)),
        )
        normalizer = PoseNormalizer.fit(records)
        pose = (0.0, -2.0, 1.0, 0.0, 0.0, math.pi - math.radians(1.0))
        target = normalizer.encode_pose(pose).unsqueeze(0)
        prediction = target.clone()
        prediction[:, 3] = math.sin(-math.pi + math.radians(1.0))
        prediction[:, 4] = math.cos(-math.pi + math.radians(1.0))
        decoded = normalizer.decode_outputs(target)
        torch.testing.assert_close(decoded[0, :3], torch.tensor(pose[:3]))
        position, yaw = pose_error_vectors(prediction, target, normalizer)
        self.assertAlmostEqual(float(position[0]), 0.0, places=6)
        self.assertAlmostEqual(float(yaw[0]), 2.0, places=4)

    def test_loss_preserves_position_and_sincos_components(self) -> None:
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        prediction = torch.tensor([[1.0, 2.0, 2.0, 1.0, 0.0]])
        components = pose_loss_components(prediction, target, PoseLossConfig())
        self.assertAlmostEqual(float(components["position"][0]), 3.0)
        self.assertAlmostEqual(float(components["orientation"][0]), 1.0)
        self.assertAlmostEqual(float(components["orientation_norm"][0]), 0.0)
        self.assertAlmostEqual(float(components["total"][0]), 3.5)

    def test_legacy_output_layer_affine_conversion_preserves_physical_xyz(self) -> None:
        normalizer = PoseNormalizer((2.0, -4.0, 1.0), (2.0, 4.0, 0.5))
        legacy_weight = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10.0
        legacy_bias = torch.tensor([3.0, -2.0, 1.5, 0.2, 0.8])
        converted = convert_legacy_state_to_normalized_outputs(
            {
                "regressor.7.weight": legacy_weight,
                "regressor.7.bias": legacy_bias,
            },
            normalizer,
        )
        features = torch.tensor([0.1, -0.2, 0.3, 0.4])
        legacy_output = legacy_weight @ features + legacy_bias
        normalized_output = (
            converted["regressor.7.weight"] @ features + converted["regressor.7.bias"]
        )
        decoded_position = normalizer.decode_positions(
            normalized_output[:3].unsqueeze(0)
        )[0]
        torch.testing.assert_close(decoded_position, legacy_output[:3])
        torch.testing.assert_close(normalized_output[3:], legacy_output[3:])
        roundtrip = convert_state_dict_to_legacy_raw_xyz(converted, normalizer)
        self.assertTrue(torch.equal(roundtrip["regressor.7.weight"], legacy_weight))
        self.assertTrue(torch.equal(roundtrip["regressor.7.bias"], legacy_bias))

    def test_decode_and_predict_helpers_fail_closed_on_undefined_yaw(self) -> None:
        normalizer = PoseNormalizer((1.0, 2.0, 3.0), (2.0, 2.0, 2.0))

        class FixedModel(torch.nn.Module):
            def __init__(self, output: torch.Tensor) -> None:
                super().__init__()
                self.register_buffer("output", output)

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.output.expand(images.shape[0], -1)

        model = FixedModel(torch.tensor([[0.5, -0.5, 1.0, 1.0, 0.0]]))
        prediction = predict_poses(
            model,
            torch.zeros(2, 3, 4, 4),
            normalizer,
            device="cpu",
        )
        torch.testing.assert_close(
            prediction.xyz_yaw[0],
            torch.tensor([2.0, 1.0, 5.0, math.pi / 2.0]),
        )

        class GradTrackingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                value = images.mean(dim=(1, 2, 3)) * self.scale
                ones = torch.ones_like(value)
                zeros = torch.zeros_like(value)
                return torch.stack([value, value, value, ones, zeros], dim=1)

        inference = predict_poses(
            GradTrackingModel(),
            torch.ones(2, 3, 4, 4, requires_grad=True),
            normalizer,
            device="cpu",
        )
        self.assertFalse(inference.normalized_output.requires_grad)
        self.assertFalse(inference.xyz_yaw.requires_grad)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            decode_predictions(torch.zeros(1, 5), normalizer)

    def test_metrics_are_sample_weighted_not_batch_weighted(self) -> None:
        normalizer = PoseNormalizer((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        targets = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 1.0]] * 3,
            dtype=torch.float32,
        )
        predictions = targets.clone()
        predictions[2, 0] = 3.0
        metrics = MetricAccumulator()
        metrics.update(predictions[:2], targets[:2], normalizer)
        metrics.update(predictions[2:], targets[2:], normalizer)
        result = metrics.compute()
        self.assertEqual(result["sample_count"], 3)
        self.assertAlmostEqual(result["position_error_cm"]["mean"], 100.0)
        self.assertAlmostEqual(result["loss"]["position"], 1.0)


class ArchitectureAndTrainingTests(unittest.TestCase):
    def test_rng_restore_normalizes_map_location_tensors_to_cpu(self) -> None:
        state = capture_rng_state()
        cpu_serialized = mock.MagicMock(spec=torch.Tensor)
        cpu_serialized.detach.return_value.cpu.return_value = torch.arange(
            16, dtype=torch.uint8
        )
        cuda_serialized = mock.MagicMock(spec=torch.Tensor)
        cuda_serialized.detach.return_value.cpu.return_value = torch.arange(
            24, dtype=torch.uint8
        )
        state["torch_cpu"] = cpu_serialized
        state["torch_cuda"] = [cuda_serialized]

        with (
            mock.patch("torch.set_rng_state") as set_cpu_state,
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.set_rng_state_all") as set_cuda_states,
        ):
            restore_rng_state(state)

        cpu_serialized.detach.return_value.cpu.assert_called_once_with()
        cuda_serialized.detach.return_value.cpu.assert_called_once_with()
        restored_cpu = set_cpu_state.call_args.args[0]
        restored_cuda = set_cuda_states.call_args.args[0]
        self.assertEqual(restored_cpu.device.type, "cpu")
        self.assertEqual(restored_cpu.dtype, torch.uint8)
        self.assertEqual(len(restored_cuda), 1)
        self.assertEqual(restored_cuda[0].device.type, "cpu")
        self.assertEqual(restored_cuda[0].dtype, torch.uint8)

    def test_rng_restore_rejects_malformed_tensor_state(self) -> None:
        state = capture_rng_state()
        state["torch_cpu"] = torch.zeros((2, 2), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            restore_rng_state(state)

    def test_none_weights_never_invokes_download_and_output_shape_is_legacy_five(
        self,
    ) -> None:
        with mock.patch(
            "torch.hub.download_url_to_file",
            side_effect=AssertionError("network download"),
        ):
            model = NPEModel(backbone="resnet18", weights="none").eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 5))

    def test_gradient_accumulation_steps_on_partial_final_group(self) -> None:
        class TinyDataset(Dataset):
            def __len__(self) -> int:
                return 5

            def __getitem__(self, index: int):
                image = torch.tensor([[[float(index)]]])
                target = torch.tensor([0.1 * index, 0.0, 0.0, 0.0, 1.0])
                return image, target, str(index)

        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 5))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loader = make_dataloader(
            TinyDataset(),
            batch_size=2,
            shuffle=False,
            seed=3,
            num_workers=0,
            pin_memory=False,
        )
        metrics, optimizer_steps = train_one_epoch(
            model,
            loader,
            optimizer,
            device=torch.device("cpu"),
            normalizer=PoseNormalizer((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            loss_config=PoseLossConfig(),
            accumulation_steps=2,
            amp_enabled=False,
        )
        self.assertEqual(optimizer_steps, 2)
        self.assertEqual(metrics["sample_count"], 5)

    def test_checkpoint_validation_requires_complete_provenance(self) -> None:
        complete = {
            "schema_version": 1,
            "model_config": {
                "backbone": "resnet50",
                "output_format": [
                    "x_normalized",
                    "y_normalized",
                    "z_normalized",
                    "sin_yaw",
                    "cos_yaw",
                ],
                "output_space": "normalized_xyz_sincos",
            },
            "model_state_dict": {},
            "normalizer": {},
            "preprocess": {},
            "loss_config": {},
            "dataset": {},
            "split_manifest": {},
            "provenance": {},
        }
        validate_repro_checkpoint(complete)
        del complete["dataset"]
        with self.assertRaisesRegex(ValueError, "dataset"):
            validate_repro_checkpoint(complete)


if __name__ == "__main__":
    unittest.main()
