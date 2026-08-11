"""Strict CPU-only resume tests for reproducible renderer datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from quadpilot.cli import data_generate_gate, data_generate_uniform
from quadpilot.datasets.generation import (
    CameraIntrinsics,
    PoseBounds,
    PoseSample,
    ReproDatasetGenerator,
)
from quadpilot.perception.npe import build_dataset_index

TARGET_SAMPLES = 6
MAXIMUM_FAILURES = 4
SEED = 9182
INTRINSICS = CameraIntrinsics(
    width=12,
    height=8,
    fx=10.0,
    fy=10.0,
    cx=6.0,
    cy=4.0,
)
BOUNDS = PoseBounds(
    x=(-1.0, 1.0),
    y=(-2.0, 2.0),
    z=(-0.5, 0.5),
)


class SimulatedInterruption(BaseException):
    """Model a process interruption that the renderer failure path cannot catch."""


class AnnotatedPoseSampler:
    """Small deterministic sampler exercising annotation replay as well as poses."""

    def sample(self, rng: np.random.Generator) -> PoseSample:
        pose = np.array(
            [
                rng.uniform(*BOUNDS.x),
                rng.uniform(*BOUNDS.y),
                rng.uniform(*BOUNDS.z),
                0.0,
                0.0,
                rng.uniform(*BOUNDS.yaw),
            ],
            dtype=np.float64,
        )
        annotations = {
            "bucket": int(rng.integers(0, 17)),
            "quality": float(rng.uniform()),
            "nested": {"accepted": True},
        }
        return PoseSample(pose=pose, annotations=annotations)


class DeterministicRenderer:
    """Render from pose only, with one deterministic rejected-pose region."""

    def __init__(self, *, interrupt_on_call: int | None = None) -> None:
        self.calls = 0
        self.interrupt_on_call = interrupt_on_call

    def render_rgb(self, camera_to_world: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.calls == self.interrupt_on_call:
            raise SimulatedInterruption("intentional clean-boundary interruption")
        if float(camera_to_world[0, 3]) < -0.35:
            raise RuntimeError("deterministic rejected renderer pose")

        height, width = INTRINSICS.height, INTRINSICS.width
        rows = np.arange(height, dtype=np.int64)[:, None]
        columns = np.arange(width, dtype=np.int64)[None, :]
        anchor = int(
            np.rint(
                31.0 * camera_to_world[0, 3]
                + 47.0 * camera_to_world[1, 3]
                + 59.0 * camera_to_world[2, 3]
                + 61.0 * camera_to_world[0, 0]
            )
        )
        red = np.broadcast_to((columns * 13 + anchor) % 256, (height, width))
        green = np.broadcast_to((rows * 17 + anchor + 53) % 256, (height, width))
        blue = (columns * 7 + rows * 11 + anchor + 127) % 256
        return np.stack((red, green, blue), axis=-1).astype(np.uint8)


class NeverRenderer:
    def render_rgb(self, camera_to_world: np.ndarray) -> np.ndarray:
        del camera_to_world
        raise AssertionError("completed resume must not invoke the renderer")


def make_generator(
    output: Path,
    renderer: object,
) -> ReproDatasetGenerator:
    return ReproDatasetGenerator(
        renderer,  # type: ignore[arg-type]
        output,
        track="resume-test",
        bounds=BOUNDS,
        intrinsics=INTRINSICS,
        seed=SEED,
        pose_sampler=AnnotatedPoseSampler(),
        provenance={
            "generator": "tests/test_repro_dataset_resume.py",
            "renderer": "deterministic-cpu-v1",
            "sampler_config": {"revision": 1},
        },
    )


def directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def load_rows(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def write_rows(root: Path, rows: list[dict[str, object]]) -> None:
    payload = "".join(
        json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    )
    (root / "samples.jsonl").write_text(
        payload,
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReproDatasetResumeTests(unittest.TestCase):
    def create_partial(self, output: Path) -> None:
        with self.assertRaises(SimulatedInterruption):
            make_generator(
                output,
                DeterministicRenderer(interrupt_on_call=4),
            ).generate(
                TARGET_SAMPLES,
                maximum_failures=MAXIMUM_FAILURES,
            )
        rows = load_rows(output)
        self.assertEqual(len(rows), 2)
        progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(
            (progress["attempts"], progress["successes"], progress["failures"]),
            (3, 2, 1),
        )

    def assert_resume_rejected(self, output: Path, pattern: str) -> None:
        with self.assertRaisesRegex((ValueError, FileExistsError), pattern):
            make_generator(output, DeterministicRenderer()).generate(
                TARGET_SAMPLES,
                maximum_failures=MAXIMUM_FAILURES,
                resume=True,
            )

    def test_interrupted_resume_is_byte_exact_and_npe_index_equivalent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted = root / "uninterrupted" / "dataset"
            resumed = root / "resumed" / "dataset"

            make_generator(uninterrupted, DeterministicRenderer()).generate(
                TARGET_SAMPLES,
                maximum_failures=MAXIMUM_FAILURES,
            )
            self.create_partial(resumed)
            make_generator(resumed, DeterministicRenderer()).generate(
                TARGET_SAMPLES,
                maximum_failures=MAXIMUM_FAILURES,
                resume=True,
            )

            self.assertEqual(directory_bytes(uninterrupted), directory_bytes(resumed))
            first_index = build_dataset_index([uninterrupted], fingerprint_mode="full")
            resumed_index = build_dataset_index([resumed], fingerprint_mode="full")
            self.assertEqual(first_index.fingerprint, resumed_index.fingerprint)
            self.assertEqual(
                [record.fingerprint_payload() for record in first_index.records],
                [record.fingerprint_payload() for record in resumed_index.records],
            )
            rows = load_rows(resumed)
            self.assertTrue(all("image_sha256" in row for row in rows))
            self.assertEqual(
                [row["image_sha256"] for row in rows],
                [record.image_sha256 for record in resumed_index.records],
            )

    def test_first_resume_invocation_and_fresh_nonresume_both_work(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            resume_first = root / "resume-first"
            ordinary_first = root / "ordinary-first"
            resumed_metadata = make_generator(
                resume_first, DeterministicRenderer()
            ).generate(
                2,
                maximum_failures=MAXIMUM_FAILURES,
                resume=True,
            )
            ordinary_metadata = make_generator(
                ordinary_first, DeterministicRenderer()
            ).generate(
                2,
                maximum_failures=MAXIMUM_FAILURES,
            )
            self.assertEqual(resumed_metadata, ordinary_metadata)
            self.assertEqual(
                directory_bytes(resume_first), directory_bytes(ordinary_first)
            )

    def test_completed_resume_is_idempotent_and_rebuilds_missing_metadata(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            expected_metadata = make_generator(
                output, DeterministicRenderer()
            ).generate(
                TARGET_SAMPLES,
                maximum_failures=MAXIMUM_FAILURES,
            )
            expected_files = directory_bytes(output)

            actual_metadata = make_generator(output, NeverRenderer()).generate(
                TARGET_SAMPLES,
                maximum_failures=MAXIMUM_FAILURES,
                resume=True,
            )
            self.assertEqual(actual_metadata, expected_metadata)
            self.assertEqual(directory_bytes(output), expected_files)

            (output / "metadata.json").unlink()
            rebuilt_metadata = make_generator(output, NeverRenderer()).generate(
                TARGET_SAMPLES,
                maximum_failures=MAXIMUM_FAILURES,
                resume=True,
            )
            self.assertEqual(rebuilt_metadata, expected_metadata)
            self.assertEqual(directory_bytes(output), expected_files)

    def test_pose_annotation_attempt_and_progress_tampering_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "partial"
            self.create_partial(source)

            pose_tamper = root / "pose-tamper"
            shutil.copytree(source, pose_tamper)
            rows = load_rows(pose_tamper)
            rows[0]["pose"][0] += 0.125  # type: ignore[index,operator]
            write_rows(pose_tamper, rows)
            self.assert_resume_rejected(pose_tamper, "pose disagrees with seed replay")

            annotation_tamper = root / "annotation-tamper"
            shutil.copytree(source, annotation_tamper)
            rows = load_rows(annotation_tamper)
            rows[1]["annotations"]["bucket"] += 1  # type: ignore[index,operator]
            write_rows(annotation_tamper, rows)
            self.assert_resume_rejected(
                annotation_tamper, "annotations disagree with seed replay"
            )

            attempt_tamper = root / "attempt-tamper"
            shutil.copytree(source, attempt_tamper)
            rows = load_rows(attempt_tamper)
            rows[1]["attempt"] = rows[0]["attempt"]
            write_rows(attempt_tamper, rows)
            self.assert_resume_rejected(attempt_tamper, "strictly increasing")

            progress_count_tamper = root / "progress-count-tamper"
            shutil.copytree(source, progress_count_tamper)
            progress_path = progress_count_tamper / "progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress["successes"] -= 1
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            self.assert_resume_rejected(progress_count_tamper, "successes do not match")

            progress_rng_tamper = root / "progress-rng-tamper"
            shutil.copytree(source, progress_rng_tamper)
            progress_path = progress_rng_tamper / "progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress["rng_state"]["state"]["state"] += 1
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            self.assert_resume_rejected(
                progress_rng_tamper, "rng_state disagrees with seed replay"
            )

            duplicate_record_key = root / "duplicate-record-key"
            shutil.copytree(source, duplicate_record_key)
            records_path = duplicate_record_key / "samples.jsonl"
            lines = records_path.read_text(encoding="utf-8").splitlines()
            first_attempt = load_rows(duplicate_record_key)[0]["attempt"]
            lines[0] = lines[0][:-1] + f',"attempt":{first_attempt}' + "}"
            records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assert_resume_rejected(
                duplicate_record_key, "invalid samples.jsonl record"
            )

            duplicate_progress_key = root / "duplicate-progress-key"
            shutil.copytree(source, duplicate_progress_key)
            progress_path = duplicate_progress_key / "progress.json"
            progress_text = progress_path.read_text(encoding="utf-8")
            progress_text = progress_text.replace(
                '"attempts": 3,',
                '"attempts": 3,\n  "attempts": 3,',
                1,
            )
            progress_path.write_text(progress_text, encoding="utf-8")
            self.assert_resume_rejected(
                duplicate_progress_key, "invalid progress receipt"
            )

    def test_image_tamper_orphans_and_unexpected_files_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "partial"
            self.create_partial(source)

            content_tamper = root / "content-tamper"
            shutil.copytree(source, content_tamper)
            image_path = content_tamper / "images" / "frame_00000.png"
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("RGB")).copy()
            pixels[0, 0, 0] ^= np.uint8(1)
            Image.fromarray(pixels, mode="RGB").save(image_path)
            self.assert_resume_rejected(content_tamper, "image SHA-256 mismatch")

            mode_tamper = root / "mode-tamper"
            shutil.copytree(source, mode_tamper)
            image_path = mode_tamper / "images" / "frame_00000.png"
            with Image.open(image_path) as image:
                grayscale = image.convert("L")
            grayscale.save(image_path)
            rows = load_rows(mode_tamper)
            rows[0]["image_sha256"] = sha256(image_path)
            write_rows(mode_tamper, rows)
            self.assert_resume_rejected(mode_tamper, "PNG/RGB/size validation")

            orphan = root / "orphan"
            shutil.copytree(source, orphan)
            shutil.copy2(
                orphan / "images" / "frame_00000.png",
                orphan / "images" / "frame_99999.png",
            )
            self.assert_resume_rejected(orphan, "image set mismatch")

            unexpected = root / "unexpected"
            shutil.copytree(source, unexpected)
            (unexpected / "stale.part").write_bytes(b"incomplete")
            self.assert_resume_rejected(unexpected, "unexpected files")

    def test_changed_intent_nonresume_and_partial_metadata_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "partial"
            self.create_partial(output)

            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                make_generator(output, DeterministicRenderer()).generate(
                    TARGET_SAMPLES,
                    maximum_failures=MAXIMUM_FAILURES,
                )
            for changed_target in (TARGET_SAMPLES - 1, TARGET_SAMPLES + 1):
                with self.subTest(target=changed_target):
                    with self.assertRaisesRegex(ValueError, "target cannot change"):
                        make_generator(output, DeterministicRenderer()).generate(
                            changed_target,
                            maximum_failures=MAXIMUM_FAILURES,
                            resume=True,
                        )
            with self.assertRaisesRegex(ValueError, "maximum_failures cannot change"):
                make_generator(output, DeterministicRenderer()).generate(
                    TARGET_SAMPLES,
                    maximum_failures=MAXIMUM_FAILURES + 1,
                    resume=True,
                )

            (output / "metadata.json").write_text("{}\n", encoding="utf-8")
            self.assert_resume_rejected(output, "partial resume.*metadata")


class ResumeCliTests(unittest.TestCase):
    def test_uniform_and_gate_clis_accept_explicit_resume(self) -> None:
        required = [
            "program",
            "--samples",
            "1",
            "--output-dir",
            "dataset",
            "--cuda-home",
            "/cuda",
            "--resume",
        ]
        for module in (data_generate_uniform, data_generate_gate):
            with (
                self.subTest(module=module.__name__),
                patch.object(sys, "argv", required),
            ):
                self.assertTrue(module.parse_args().resume)


if __name__ == "__main__":
    unittest.main()
