"""Offline tests for the strict public-Drive GSplat source downloader."""

from __future__ import annotations

import hashlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import download_repro_gsplat_source as downloader


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, *, status: int, headers: dict[str, str]):
        super().__init__(data)
        self.status = status
        self.headers = headers

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class FakeDriveClient:
    def __init__(
        self,
        contents: dict[str, bytes],
        listing: list[downloader.RemoteFile],
    ) -> None:
        self.contents = contents
        self.listing = listing
        self.events: list[tuple[object, ...]] = []

    def list_folder(self, folder_id: str) -> list[downloader.RemoteFile]:
        self.events.append(("list", folder_id))
        return list(self.listing)

    def stat_size(self, file_id: str) -> int:
        self.events.append(("stat", file_id))
        return len(self.contents[file_id])

    def read_bytes(self, file_id: str) -> bytes:
        self.events.append(("read", file_id))
        return self.contents[file_id]

    def open_content(self, file_id: str, start: int = 0) -> FakeResponse:
        self.events.append(("open", file_id, start))
        data = self.contents[file_id]
        if start:
            return FakeResponse(
                data[start:],
                status=206,
                headers={"Content-Range": f"bytes {start}-{len(data)-1}/{len(data)}"},
            )
        return FakeResponse(data, status=200, headers={})


class FailingDriveClient(FakeDriveClient):
    def __init__(
        self,
        contents: dict[str, bytes],
        listing: list[downloader.RemoteFile],
        *,
        failing_id: str,
    ) -> None:
        super().__init__(contents, listing)
        self.failing_id = failing_id

    def open_content(self, file_id: str, start: int = 0) -> FakeResponse:
        if file_id == self.failing_id:
            self.events.append(("open-failed", file_id, start))
            raise OSError("injected transfer failure")
        return super().open_content(file_id, start)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(
    path: Path,
    *,
    transforms: bytes,
    sparse: bytes,
    selected: dict[str, bytes],
) -> None:
    manifest = {
        "schema_version": 1,
        "external_sources": {
            "gsplat_tracks": {
                "lemniscate": {
                    "folder_id": "track-folder",
                    "images_folder_id": "images-folder",
                    "transforms_json": {
                        "file_id": "transforms-id",
                        "size_bytes": len(transforms),
                        "sha256": digest(transforms),
                    },
                    "sparse_pc_ply": {
                        "file_id": "sparse-id",
                        "size_bytes": len(sparse),
                        "sha256": digest(sparse),
                    },
                    "transforms_referenced_images": len(selected),
                    "transforms_referenced_bytes": sum(
                        len(value) for value in selected.values()
                    ),
                }
            }
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


class StrictSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image_data = {
            "image-a": b"first-image",
            "image-b": b"second-image-data",
            "unused": b"not-referenced",
        }
        self.transforms = json.dumps(
            {
                "frames": [
                    {"file_path": "images/frame_00002.png"},
                    {"file_path": "images/frame_00001.png"},
                ]
            },
            separators=(",", ":"),
        ).encode()
        self.sparse = b"ply\nsmall point cloud"

    def make_client(self) -> FakeDriveClient:
        contents = {
            "transforms-id": self.transforms,
            "sparse-id": self.sparse,
            **self.image_data,
        }
        listing = [
            downloader.RemoteFile("image-a", "frame_00001.png"),
            downloader.RemoteFile("image-b", "frame_00002.png"),
            downloader.RemoteFile("unused", "frame_09999.png"),
        ]
        return FakeDriveClient(contents, listing)

    def test_dry_run_selects_only_transform_basenames_and_writes_nothing(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            output = root / "assets"
            write_manifest(
                manifest,
                transforms=self.transforms,
                sparse=self.sparse,
                selected={
                    "frame_00001.png": self.image_data["image-a"],
                    "frame_00002.png": self.image_data["image-b"],
                },
            )
            client = self.make_client()
            plan = downloader.download_track(
                track="lemniscate",
                manifest_path=manifest,
                output_dir=output,
                dry_run=True,
                workers=1,
                client=client,
            )

            self.assertFalse(output.exists())
            self.assertEqual(plan["selected_images"], 2)
            self.assertEqual(plan["excluded_original_images"], 1)
            self.assertEqual(
                [item["file_id"] for item in plan["files"]],
                ["image-b", "image-a"],
            )
            self.assertFalse(any(event[0] == "open" for event in client.events))

    def test_real_run_verifies_roots_before_listing_and_records_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            output = root / "assets"
            write_manifest(
                manifest,
                transforms=self.transforms,
                sparse=self.sparse,
                selected={
                    "frame_00001.png": self.image_data["image-a"],
                    "frame_00002.png": self.image_data["image-b"],
                },
            )
            client = self.make_client()
            plan = downloader.download_track(
                track="lemniscate",
                manifest_path=manifest,
                output_dir=output,
                dry_run=False,
                workers=1,
                client=client,
            )

            track_dir = output / "lemniscate"
            self.assertEqual((track_dir / "transforms.json").read_bytes(), self.transforms)
            self.assertEqual((track_dir / "sparse_pc.ply").read_bytes(), self.sparse)
            self.assertEqual(
                (track_dir / "images" / "frame_00001.png").read_bytes(),
                self.image_data["image-a"],
            )
            self.assertFalse((track_dir / "images" / "frame_09999.png").exists())
            first_list = next(i for i, event in enumerate(client.events) if event[0] == "list")
            root_opens = [
                i
                for i, event in enumerate(client.events)
                if event[:2] in {("open", "transforms-id"), ("open", "sparse-id")}
            ]
            self.assertTrue(root_opens)
            self.assertLess(max(root_opens), first_list)
            receipt = json.loads((track_dir / downloader.RECEIPT_NAME).read_text())
            self.assertEqual(len(receipt["files"]), 4)
            self.assertEqual(
                receipt["files"]["images/frame_00001.png"]["sha256"],
                digest(self.image_data["image-a"]),
            )
            failures = json.loads((track_dir / downloader.FAILURES_NAME).read_text())
            self.assertEqual(failures["failures"], [])
            self.assertEqual(plan["selected_images"], 2)

    def test_missing_referenced_image_fails_before_image_download(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            write_manifest(
                manifest,
                transforms=self.transforms,
                sparse=self.sparse,
                selected={
                    "frame_00001.png": self.image_data["image-a"],
                    "frame_00002.png": self.image_data["image-b"],
                },
            )
            client = self.make_client()
            client.listing = client.listing[:1]
            with self.assertRaises(downloader.SelectionError):
                downloader.download_track(
                    track="lemniscate",
                    manifest_path=manifest,
                    output_dir=root / "assets",
                    dry_run=True,
                    workers=1,
                    client=client,
                )
            self.assertFalse(any(event[0] == "open" for event in client.events))

    def test_drive_path_traversal_is_rejected(self) -> None:
        with self.assertRaises(downloader.SelectionError):
            downloader.normalize_drive_path("../frame_00001.png")

    def test_bad_root_hash_stops_before_listing_and_writes_failure_list(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            output = root / "assets"
            write_manifest(
                manifest,
                transforms=self.transforms,
                sparse=self.sparse,
                selected={
                    "frame_00001.png": self.image_data["image-a"],
                    "frame_00002.png": self.image_data["image-b"],
                },
            )
            document = json.loads(manifest.read_text())
            document["external_sources"]["gsplat_tracks"]["lemniscate"][
                "transforms_json"
            ]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(document), encoding="utf-8")
            client = self.make_client()

            with self.assertRaises(downloader.AssetDownloadError):
                downloader.download_track(
                    track="lemniscate",
                    manifest_path=manifest,
                    output_dir=output,
                    dry_run=False,
                    workers=1,
                    client=client,
                )

            self.assertFalse(any(event[0] == "list" for event in client.events))
            failure_path = output / "lemniscate" / downloader.FAILURES_NAME
            failures = json.loads(failure_path.read_text())
            self.assertEqual(failures["failures"][0]["phase"], "root")

    def test_existing_image_without_trusted_receipt_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            output = root / "assets"
            write_manifest(
                manifest,
                transforms=self.transforms,
                sparse=self.sparse,
                selected={
                    "frame_00001.png": self.image_data["image-a"],
                    "frame_00002.png": self.image_data["image-b"],
                },
            )
            existing = output / "lemniscate" / "images" / "frame_00001.png"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(self.image_data["image-a"])

            with self.assertRaises(downloader.AssetDownloadError):
                downloader.download_track(
                    track="lemniscate",
                    manifest_path=manifest,
                    output_dir=output,
                    dry_run=False,
                    workers=1,
                    client=self.make_client(),
                )

            failures = json.loads(
                (output / "lemniscate" / downloader.FAILURES_NAME).read_text()
            )
            matching = [
                item
                for item in failures["failures"]
                if item["path"] == "images/frame_00001.png"
            ]
            self.assertEqual(len(matching), 1)
            self.assertIn("no matching trusted receipt", matching[0]["error"])

    def test_partial_failure_persists_success_receipt_for_verified_rerun(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            output = root / "assets"
            write_manifest(
                manifest,
                transforms=self.transforms,
                sparse=self.sparse,
                selected={
                    "frame_00001.png": self.image_data["image-a"],
                    "frame_00002.png": self.image_data["image-b"],
                },
            )
            healthy = self.make_client()
            failing = FailingDriveClient(
                healthy.contents,
                healthy.listing,
                failing_id="image-a",
            )
            with self.assertRaises(downloader.AssetDownloadError):
                downloader.download_track(
                    track="lemniscate",
                    manifest_path=manifest,
                    output_dir=output,
                    dry_run=False,
                    workers=1,
                    client=failing,
                )

            track_dir = output / "lemniscate"
            receipt = json.loads(
                (track_dir / downloader.RECEIPT_NAME).read_text()
            )
            completed_key = "images/frame_00002.png"
            self.assertEqual(
                receipt["files"][completed_key]["sha256"],
                digest(self.image_data["image-b"]),
            )

            rerun = self.make_client()
            downloader.download_track(
                track="lemniscate",
                manifest_path=manifest,
                output_dir=output,
                dry_run=False,
                workers=1,
                client=rerun,
            )
            completed_opens = [
                event for event in rerun.events if event[:2] == ("open", "image-b")
            ]
            self.assertEqual(completed_opens, [])


class AtomicDownloadTests(unittest.TestCase):
    def test_part_file_is_resumed_with_range_then_atomically_renamed(self) -> None:
        payload = b"0123456789abcdef"
        client = FakeDriveClient({"file": payload}, [])
        remote = downloader.RemoteFile("file", "frame.png", len(payload))
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "images" / "frame.png"
            destination.parent.mkdir()
            part = destination.with_name(destination.name + ".part")
            part.write_bytes(payload[:6])

            actual_sha, status = downloader.download_atomic(
                client,
                remote,
                destination,
                expected_sha256=digest(payload),
            )

            self.assertEqual(client.events, [("open", "file", 6)])
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(part.exists())
            self.assertEqual(actual_sha, digest(payload))
            self.assertEqual(status, "resumed")

    def test_bad_existing_file_is_not_overwritten(self) -> None:
        expected = b"good-data"
        client = FakeDriveClient({"file": expected}, [])
        remote = downloader.RemoteFile("file", "frame.png", len(expected))
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "frame.png"
            destination.write_bytes(b"bad--data")
            with self.assertRaises(downloader.VerificationError):
                downloader.download_atomic(
                    client,
                    remote,
                    destination,
                    expected_sha256=digest(expected),
                )
            self.assertEqual(destination.read_bytes(), b"bad--data")
            self.assertEqual(client.events, [])


if __name__ == "__main__":
    unittest.main()
