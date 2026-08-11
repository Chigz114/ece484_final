"""Independent CPU-only tests for the strict NPE dataset verifier."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from quadpilot_repro.data_generation import BASE_DATASET_BOUNDS, CameraIntrinsics
from quadpilot_repro.gate_sampling import GateFocusConfig, GateFocusedPoseSampler
from scripts import verify_repro_npe_dataset as verifier


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticNpeDataset:
    def __init__(self, root: Path, frames: int = 3) -> None:
        self.root = root
        self.frames = frames
        self.dataset = root / "dataset"
        self.images = self.dataset / "images"
        self.images.mkdir(parents=True)
        self.renderer = root / "renderer"
        self.renderer.mkdir()
        self.checkpoint = self.renderer / "step-000000123.ckpt"
        self.transform = self.renderer / "dataparser_transforms.json"
        self.checkpoint.write_bytes(b"synthetic-checkpoint")
        self.transform.write_text('{"synthetic":true}\n', encoding="utf-8")
        self.checkpoint_sha = _sha256(self.checkpoint)
        self.transform_sha = _sha256(self.transform)

        self.poses = [
            [float(index), -float(index), 0.1 * index, 0.0, 0.0, 0.2 * index]
            for index in range(frames)
        ]
        for index in range(frames):
            Image.new("RGB", (640, 480), color=(index, 10, 20)).save(
                self.images / f"frame_{index:05d}.png", format="PNG"
            )
        self.samples = self.dataset / "samples.jsonl"
        self._write_samples()
        self.metadata = self.dataset / "metadata.json"
        self.metadata_payload = {
            "schema_version": 2,
            "n_frames": frames,
            "track": "lemniscate",
            "seed": 42,
            "attempts": frames,
            "render_failures": 0,
            "bounds": {"x": [-1, 1], "y": [-1, 1], "z": [-1, 1], "yaw": [-3.14, 3.14]},
            "pose_format": ["x", "y", "z", "roll", "pitch", "yaw"],
            "pose_units": ["m", "m", "m", "rad", "rad", "rad"],
            "pose_coordinate_frame": "original_nerf_world",
            "body_axis_convention": "+X forward, +Y left, +Z up",
            "camera_axis_convention": "OpenCV",
            "yaw_convention": "radians",
            "image_format": "RGB uint8 PNG",
            "image_size": [640, 480],
            "intrinsics": {"width": 640, "height": 480},
            "poses": self.poses,
            "samples_manifest": "samples.jsonl",
            "provenance": {
                "generator": "scripts/generate_repro_npe_dataset.py",
                "checkpoint": str(self.checkpoint.resolve()),
                "checkpoint_step": 123,
                "gaussian_count": 17,
                "dataparser_transform": str(self.transform.resolve()),
                "sampling": "synthetic",
            },
        }
        self._write_metadata()
        self.progress = self.dataset / "progress.json"
        self.progress.write_text(
            json.dumps(
                {
                    "attempts": frames,
                    "successes": frames,
                    "failures": 0,
                    "rng_state": {"synthetic": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.builder_calls: list[tuple[list[Path], str]] = []

    def _write_samples(self, rows: list[dict[str, object]] | None = None) -> None:
        if rows is None:
            rows = [
                {
                    "sample_id": index,
                    "image": f"images/frame_{index:05d}.png",
                    "pose": self.poses[index],
                    "attempt": index + 1,
                }
                for index in range(self.frames)
            ]
        self.samples.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_metadata(self) -> None:
        self.metadata.write_text(
            json.dumps(self.metadata_payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def enable_resume_schema(
        self,
        *,
        generator: str = "scripts/generate_repro_npe_dataset.py",
        maximum_failures: int = 10,
    ) -> None:
        rows = [
            json.loads(line)
            for line in self.samples.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            image = self.dataset / Path(*str(row["image"]).split("/"))
            row["image_sha256"] = _sha256(image)
        self._write_samples(rows)
        self.metadata_payload["target_samples"] = self.frames
        self.metadata_payload["maximum_failures"] = maximum_failures
        self.metadata_payload["provenance"]["generator"] = generator
        self._write_metadata()
        sampler = verifier.GENERATOR_SAMPLERS[generator]
        progress = {
            "schema_version": 1,
            "target_samples": self.frames,
            "maximum_failures": maximum_failures,
            "generation_contract": {
                "track": self.metadata_payload["track"],
                "seed": self.metadata_payload["seed"],
                "bounds": self.metadata_payload["bounds"],
                "intrinsics": self.metadata_payload["intrinsics"],
                "provenance": self.metadata_payload["provenance"],
                "pose_sampler": sampler,
            },
            "attempts": self.frames,
            "successes": self.frames,
            "failures": 0,
            "rng_state": {"synthetic": True},
        }
        self.progress.write_text(
            json.dumps(progress, indent=2) + "\n", encoding="utf-8"
        )

    def enable_gate_focused_schema(self) -> None:
        intrinsics = CameraIntrinsics()
        focus_config = GateFocusConfig()
        sampler = GateFocusedPoseSampler(
            "lemniscate",
            BASE_DATASET_BOUNDS["lemniscate"],
            focus_config,
            intrinsics,
        )
        rng = np.random.default_rng(42)
        samples = [sampler.sample(rng) for _ in range(self.frames)]
        self.poses = [[float(value) for value in sample.pose] for sample in samples]
        rows = [
            {
                "sample_id": index,
                "image": f"images/frame_{index:05d}.png",
                "pose": self.poses[index],
                "attempt": index + 1,
                "annotations": dict(sample.annotations or {}),
            }
            for index, sample in enumerate(samples)
        ]
        self._write_samples(rows)
        self.metadata_payload["poses"] = self.poses
        self.metadata_payload["bounds"] = {
            key: list(getattr(BASE_DATASET_BOUNDS["lemniscate"], key))
            for key in ("x", "y", "z", "yaw")
        }
        self.metadata_payload["intrinsics"] = {
            key: getattr(intrinsics, key)
            for key in ("width", "height", "fx", "fy", "cx", "cy")
        }
        self.metadata_payload["provenance"]["sampling"] = (
            verifier.GATE_SAMPLING_DESCRIPTION
        )
        self.metadata_payload["provenance"]["gate_focus_config"] = (
            focus_config.to_dict()
        )
        self.enable_resume_schema(generator="scripts/generate_repro_gate_dataset.py")

    def sync_resume_contract_from_metadata(self) -> None:
        progress = json.loads(self.progress.read_text(encoding="utf-8"))
        progress["generation_contract"]["bounds"] = self.metadata_payload["bounds"]
        progress["generation_contract"]["intrinsics"] = self.metadata_payload[
            "intrinsics"
        ]
        progress["generation_contract"]["provenance"] = self.metadata_payload[
            "provenance"
        ]
        self.progress.write_text(
            json.dumps(progress, indent=2) + "\n", encoding="utf-8"
        )

    def fake_builder(self, data_dirs: list[Path], *, fingerprint_mode: str) -> object:
        self.builder_calls.append((data_dirs, fingerprint_mode))
        root = Path(data_dirs[0]).resolve()
        source_id = f"0:{root.name}"
        records = []
        for index, pose in enumerate(self.poses):
            relative = f"images/frame_{index:05d}.png"
            path = root / relative
            records.append(
                types.SimpleNamespace(
                    key=f"{source_id}/{relative}",
                    source_id=source_id,
                    source_root=root,
                    relative_image=relative,
                    pose=tuple(pose),
                    image_sha256=_sha256(path),
                    width=640,
                    height=480,
                    image_path=path,
                )
            )
        payload = "".join(record.image_sha256 for record in records).encode("ascii")
        return types.SimpleNamespace(
            records=tuple(records),
            fingerprint=hashlib.sha256(payload).hexdigest(),
            fingerprint_mode=fingerprint_mode,
            sources=(
                {
                    "source_id": source_id,
                    "path": str(root),
                    "track": "lemniscate",
                    "schema_version": 2,
                    "metadata_sha256": _sha256(self.metadata),
                    "sample_count": self.frames,
                },
            ),
        )

    def verify(self, **changes: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "track": "lemniscate",
            "seed": 42,
            "expected_frames": self.frames,
            "index_builder": self.fake_builder,
        }
        arguments.update(changes)
        return verifier.verify_dataset(self.dataset, **arguments)


class NpeDatasetVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticNpeDataset(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_dataset_uses_full_single_source_index_and_is_read_only(self) -> None:
        before = verifier._tree_snapshot(self.fixture.dataset)
        report = self.fixture.verify()
        after = verifier._tree_snapshot(self.fixture.dataset)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["cpu_only"])
        self.assertFalse(report["dataset_modified"])
        self.assertEqual(report["frames"], self.fixture.frames)
        self.assertEqual(report["images"]["fully_decoded"], self.fixture.frames)
        self.assertEqual(report["dataset_index"]["fingerprint_mode"], "full")
        self.assertTrue(report["dataset_index"]["single_source"])
        self.assertEqual(
            report["samples_manifest"]["per_record_image_integrity"],
            "legacy_unhashed",
        )
        self.assertEqual(report["progress_schema"]["mode"], "legacy")
        self.assertNotIn("gate_focused", report)
        self.assertEqual(
            self.fixture.builder_calls, [([self.fixture.dataset.resolve()], "full")]
        )
        self.assertEqual(before, after)

    def test_resume_schema_requires_and_verifies_every_record_hash(self) -> None:
        self.fixture.enable_resume_schema()
        report = self.fixture.verify()
        self.assertEqual(
            report["samples_manifest"]["per_record_image_integrity"],
            "verified_per_record",
        )
        self.assertEqual(report["progress_schema"]["mode"], "resume_v1")
        self.assertTrue(report["progress_schema"]["generation_contract_verified"])
        self.assertEqual(
            report["progress_schema"]["target_samples"], self.fixture.frames
        )
        self.assertEqual(report["progress_schema"]["maximum_failures"], 10)

        rows = [
            json.loads(line)
            for line in self.fixture.samples.read_text(encoding="utf-8").splitlines()
        ]
        rows[1]["image_sha256"] = "0" * 64
        self.fixture._write_samples(rows)
        with self.assertRaisesRegex(
            verifier.VerificationError, "image SHA-256 differs"
        ):
            self.fixture.verify()

    def test_mixed_record_progress_or_metadata_schema_is_rejected(self) -> None:
        self.fixture.enable_resume_schema()
        rows = [
            json.loads(line)
            for line in self.fixture.samples.read_text(encoding="utf-8").splitlines()
        ]
        del rows[0]["image_sha256"]
        self.fixture._write_samples(rows)
        with self.assertRaisesRegex(verifier.VerificationError, "mixed samples schema"):
            self.fixture.verify()

        self.tearDown()
        self.setUp()
        progress = json.loads(self.fixture.progress.read_text(encoding="utf-8"))
        progress.update(
            {
                "schema_version": 1,
                "target_samples": self.fixture.frames,
                "maximum_failures": 10,
                "generation_contract": {},
            }
        )
        self.fixture.progress.write_text(json.dumps(progress) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(
            verifier.VerificationError, "resume progress requires hashed"
        ):
            self.fixture.verify()

    def test_resume_target_maximum_contract_and_counts_are_frozen(self) -> None:
        mutations = (
            ("target_samples", self.fixture.frames + 1, "target_samples"),
            ("maximum_failures", 11, "maximum_failures"),
            ("generation_contract", {"tampered": True}, "generation_contract"),
            ("successes", self.fixture.frames - 1, "successes"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                self.tearDown()
                self.setUp()
                self.fixture.enable_resume_schema()
                progress = json.loads(self.fixture.progress.read_text(encoding="utf-8"))
                progress[field] = value
                self.fixture.progress.write_text(
                    json.dumps(progress) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(verifier.VerificationError, message):
                    self.fixture.verify()

    def test_gate_resume_contract_uses_gate_sampler_identity(self) -> None:
        self.fixture.enable_resume_schema(
            generator="scripts/generate_repro_gate_dataset.py"
        )
        report = self.fixture.verify()
        self.assertEqual(
            report["renderer_provenance"]["generator"],
            "scripts/generate_repro_gate_dataset.py",
        )
        self.assertTrue(report["progress_schema"]["generation_contract_verified"])

    def test_gate_focused_expectation_verifies_geometry_projection_and_replay(
        self,
    ) -> None:
        self.fixture.enable_gate_focused_schema()
        report = self.fixture.verify(expect_gate_focused=True)
        gate_report = report["gate_focused"]
        self.assertTrue(gate_report["enabled"])
        self.assertEqual(gate_report["geometry_verified"], self.fixture.frames)
        self.assertEqual(gate_report["projection_verified"], self.fixture.frames)
        self.assertEqual(gate_report["seed_replay_verified"], self.fixture.frames)
        self.assertEqual(set(gate_report["counts"]), {"A", "B", "C", "D"})
        self.assertEqual(sum(gate_report["counts"].values()), self.fixture.frames)
        self.assertEqual(
            gate_report["rejection_stats"]["total"],
            sum(
                row["annotations"]["rejections_before_acceptance"]
                for row in map(
                    json.loads,
                    self.fixture.samples.read_text(encoding="utf-8").splitlines(),
                )
            ),
        )
        self.assertFalse(gate_report["count_gate"]["applied"])

    def test_gate_focused_expectation_is_opt_in_and_requires_gate_generator(
        self,
    ) -> None:
        report = self.fixture.verify()
        self.assertNotIn("gate_focused", report)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "requires the gate dataset generator",
        ):
            self.fixture.verify(expect_gate_focused=True)

    def test_gate_focused_metadata_defaults_are_exact(self) -> None:
        mutations = (
            (
                lambda metadata: metadata["provenance"][
                    "gate_focus_config"
                ].__setitem__("maximum_rejections", 100.0),
                "maximum_rejections differs",
            ),
            (
                lambda metadata: metadata["provenance"][
                    "gate_focus_config"
                ].__setitem__("image_margin_px", 31.0),
                "image_margin_px differs",
            ),
            (
                lambda metadata: metadata["provenance"].__setitem__(
                    "sampling", "different"
                ),
                "sampling description changed",
            ),
            (
                lambda metadata: metadata["intrinsics"].__setitem__("fx", 500.0),
                "intrinsics fx differs",
            ),
            (
                lambda metadata: metadata["bounds"].__setitem__("x", [-4.4, 0.0]),
                "bounds x differs",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                self.tearDown()
                self.setUp()
                self.fixture.enable_gate_focused_schema()
                mutate(self.fixture.metadata_payload)
                self.fixture._write_metadata()
                self.fixture.sync_resume_contract_from_metadata()
                with self.assertRaisesRegex(verifier.VerificationError, message):
                    self.fixture.verify(expect_gate_focused=True)

    def test_gate_focused_annotation_schema_types_and_ranges_are_strict(self) -> None:
        cases = (
            (
                lambda annotations: annotations.pop("gate_center_u_px"),
                "exact required fields",
            ),
            (
                lambda annotations: annotations.__setitem__("unexpected", 1.0),
                "exact required fields",
            ),
            (
                lambda annotations: annotations.__setitem__("approach_distance_m", 1),
                "must be a JSON float",
            ),
            (
                lambda annotations: annotations.__setitem__(
                    "approach_distance_m", float("inf")
                ),
                "non-standard non-finite",
            ),
            (
                lambda annotations: annotations.__setitem__(
                    "approach_distance_m", 2.01
                ),
                "outside the configured range",
            ),
            (
                lambda annotations: annotations.__setitem__("focus_gate", "gate a"),
                "focus_gate is not",
            ),
            (
                lambda annotations: annotations.__setitem__(
                    "rejections_before_acceptance", True
                ),
                "must be an integer",
            ),
            (
                lambda annotations: annotations.__setitem__(
                    "rejections_before_acceptance", 100
                ),
                "outside the configured range",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                self.tearDown()
                self.setUp()
                self.fixture.enable_gate_focused_schema()
                rows = [
                    json.loads(line)
                    for line in self.fixture.samples.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                mutate(rows[0]["annotations"])
                self.fixture._write_samples(rows)
                with self.assertRaisesRegex(verifier.VerificationError, message):
                    self.fixture.verify(expect_gate_focused=True)

    def test_gate_focused_tampering_is_caught_by_geometry_projection_or_replay(
        self,
    ) -> None:
        cases = (
            (
                lambda rows: rows[0]["annotations"].__setitem__(
                    "lateral_offset_m",
                    rows[0]["annotations"]["lateral_offset_m"] + 0.01,
                ),
                "differs from reconstructed gate geometry",
            ),
            (
                lambda rows: rows[0]["annotations"].__setitem__(
                    "gate_center_u_px",
                    rows[0]["annotations"]["gate_center_u_px"] + 0.01,
                ),
                "gate_center_u_px differs",
            ),
            (
                lambda rows: rows[0]["annotations"].__setitem__(
                    "rejections_before_acceptance",
                    (rows[0]["annotations"]["rejections_before_acceptance"] + 1) % 100,
                ),
                "annotations differ from deterministic gate sampler replay",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                self.tearDown()
                self.setUp()
                self.fixture.enable_gate_focused_schema()
                rows = [
                    json.loads(line)
                    for line in self.fixture.samples.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                mutate(rows)
                self.fixture._write_samples(rows)
                with self.assertRaisesRegex(verifier.VerificationError, message):
                    self.fixture.verify(expect_gate_focused=True)

        self.tearDown()
        self.setUp()
        self.fixture.enable_gate_focused_schema()
        rows = [
            json.loads(line)
            for line in self.fixture.samples.read_text(encoding="utf-8").splitlines()
        ]
        rows[0]["pose"][0] += 0.01
        self.fixture.poses[0][0] += 0.01
        self.fixture.metadata_payload["poses"] = self.fixture.poses
        self.fixture._write_samples(rows)
        self.fixture._write_metadata()
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "differs from reconstructed gate geometry",
        ):
            self.fixture.verify(expect_gate_focused=True)

    def test_4000_frame_gate_count_hard_gate_is_exact(self) -> None:
        valid = {
            "Gate A": 850,
            "Gate B": 1150,
            "Gate C": 1000,
            "Gate D": 1000,
        }
        report = verifier._verify_gate_counts(valid, expected_frames=4000)
        self.assertTrue(report["applied"])
        self.assertTrue(report["passed"])

        for counts, gate_name in (
            (
                {"Gate A": 849, "Gate B": 1150, "Gate C": 1000, "Gate D": 1001},
                "Gate A",
            ),
            (
                {"Gate A": 1151, "Gate B": 949, "Gate C": 950, "Gate D": 950},
                "Gate A",
            ),
        ):
            with self.subTest(counts=counts):
                with self.assertRaisesRegex(verifier.VerificationError, gate_name):
                    verifier._verify_gate_counts(counts, expected_frames=4000)

    def test_optional_renderer_provenance_pins_step_count_and_hashes(self) -> None:
        report = self.fixture.verify(
            expected_checkpoint_step=123,
            expected_gaussians=17,
            expected_checkpoint_sha256=self.fixture.checkpoint_sha,
            expected_transform_sha256=self.fixture.transform_sha,
        )
        provenance = report["renderer_provenance"]
        self.assertEqual(provenance["checkpoint_step"], 123)
        self.assertEqual(provenance["gaussian_count"], 17)
        self.assertEqual(provenance["checkpoint_sha256"], self.fixture.checkpoint_sha)
        self.assertEqual(
            provenance["dataparser_transform_sha256"], self.fixture.transform_sha
        )

        with self.assertRaisesRegex(verifier.VerificationError, "checkpoint SHA-256"):
            self.fixture.verify(expected_checkpoint_sha256="0" * 64)

    def test_track_seed_frame_and_failure_contracts_fail_closed(self) -> None:
        mutations = (
            ("track", "circle", "metadata track"),
            ("seed", 41, "metadata seed"),
            ("expected_frames", 4, "n_frames"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key):
                with self.assertRaisesRegex(verifier.VerificationError, message):
                    self.fixture.verify(**{key: value})

        self.fixture.metadata_payload["render_failures"] = 1
        self.fixture.metadata_payload["attempts"] = self.fixture.frames + 1
        self.fixture._write_metadata()
        with self.assertRaisesRegex(verifier.VerificationError, "render failures"):
            self.fixture.verify()

    def test_sample_ids_pose_images_and_attempts_must_be_exactly_aligned(self) -> None:
        cases = (
            (
                [
                    {
                        "sample_id": 1,
                        "image": "images/frame_00000.png",
                        "pose": self.fixture.poses[0],
                        "attempt": 1,
                    },
                    {
                        "sample_id": 0,
                        "image": "images/frame_00001.png",
                        "pose": self.fixture.poses[1],
                        "attempt": 2,
                    },
                    {
                        "sample_id": 2,
                        "image": "images/frame_00002.png",
                        "pose": self.fixture.poses[2],
                        "attempt": 3,
                    },
                ],
                "continuous and line-ordered",
            ),
            (
                [
                    {
                        "sample_id": index,
                        "image": f"images/frame_{index:05d}.png",
                        "pose": (
                            [99.0] * 6 if index == 1 else self.fixture.poses[index]
                        ),
                        "attempt": index + 1,
                    }
                    for index in range(self.fixture.frames)
                ],
                "metadata/sample pose",
            ),
            (
                [
                    {
                        "sample_id": index,
                        "image": (
                            "images/wrong.png"
                            if index == 1
                            else f"images/frame_{index:05d}.png"
                        ),
                        "pose": self.fixture.poses[index],
                        "attempt": index + 1,
                    }
                    for index in range(self.fixture.frames)
                ],
                "image path is not canonical",
            ),
        )
        for rows, message in cases:
            with self.subTest(message=message):
                self.fixture._write_samples(rows)
                with self.assertRaisesRegex(verifier.VerificationError, message):
                    self.fixture.verify()
                self.fixture._write_samples()

    def test_png_must_be_fully_decodable_rgb_and_exactly_640x480(self) -> None:
        path = self.fixture.images / "frame_00001.png"
        Image.new("RGBA", (640, 480), color=(1, 2, 3, 4)).save(path)
        with self.assertRaisesRegex(verifier.VerificationError, "not RGB"):
            self.fixture.verify()

        Image.new("RGB", (320, 240)).save(path)
        with self.assertRaisesRegex(
            verifier.VerificationError, "dimensions are not 640x480"
        ):
            self.fixture.verify()

        path.write_bytes(b"not-a-png")
        with self.assertRaisesRegex(
            verifier.VerificationError, "cannot be fully decoded"
        ):
            self.fixture.verify()

    def test_extra_image_temporary_partial_and_symlink_are_rejected(self) -> None:
        Image.new("RGB", (640, 480)).save(self.fixture.images / "frame_99999.png")
        with self.assertRaisesRegex(
            verifier.VerificationError, "does not exactly match"
        ):
            self.fixture.verify()

        (self.fixture.images / "frame_99999.png").unlink()
        (self.fixture.dataset / "metadata.json.tmp").write_text(
            "partial", encoding="utf-8"
        )
        with self.assertRaisesRegex(verifier.VerificationError, "temporary/partial"):
            self.fixture.verify()

        (self.fixture.dataset / "metadata.json.tmp").unlink()
        try:
            (self.fixture.dataset / "image-link.png").symlink_to(
                self.fixture.images / "frame_00000.png"
            )
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is unavailable")
        with self.assertRaisesRegex(verifier.VerificationError, "symlink"):
            self.fixture.verify()

    def test_progress_failure_and_duplicate_json_key_are_rejected(self) -> None:
        self.fixture.progress.write_text(
            json.dumps({"attempts": 4, "successes": 3, "failures": 1}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "progress attempts|progress reports|neither the frozen legacy schema",
        ):
            self.fixture.verify()

        self.fixture.progress.write_text(
            '{"attempts":3,"attempts":3,"successes":3,"failures":0}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate key"):
            self.fixture.verify()

    def test_cli_emits_json_to_stdout_and_does_not_create_files(self) -> None:
        before = verifier._tree_snapshot(self.fixture.dataset)
        output = io.StringIO()
        with (
            mock.patch.object(
                verifier, "_default_index_builder", self.fixture.fake_builder
            ),
            contextlib.redirect_stdout(output),
        ):
            result = verifier.main(
                [
                    str(self.fixture.dataset),
                    "--track",
                    "lemniscate",
                    "--seed",
                    "42",
                    "--expected-frames",
                    str(self.fixture.frames),
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "PASS")
        self.assertEqual(verifier._tree_snapshot(self.fixture.dataset), before)

    def test_cli_gate_focused_flag_enables_strict_gate_report(self) -> None:
        self.fixture.enable_gate_focused_schema()
        before = verifier._tree_snapshot(self.fixture.dataset)
        output = io.StringIO()
        with (
            mock.patch.object(
                verifier, "_default_index_builder", self.fixture.fake_builder
            ),
            contextlib.redirect_stdout(output),
        ):
            result = verifier.main(
                [
                    str(self.fixture.dataset),
                    "--track",
                    "lemniscate",
                    "--seed",
                    "42",
                    "--expected-frames",
                    str(self.fixture.frames),
                    "--expect-gate-focused",
                ]
            )
        report = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(report["gate_focused"]["enabled"])
        self.assertEqual(verifier._tree_snapshot(self.fixture.dataset), before)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None, "requires CPU PyTorch"
    )
    def test_real_build_dataset_index_is_used_when_torch_is_available(self) -> None:
        report = verifier.verify_dataset(
            self.fixture.dataset,
            track="lemniscate",
            seed=42,
            expected_frames=self.fixture.frames,
        )
        self.assertEqual(report["dataset_index"]["fingerprint_mode"], "full")
        self.assertRegex(report["dataset_index"]["fingerprint"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
