"""Download only the GSplat source files referenced by a track transform.

The public FalconGym Drive contains original images, downscaled copies, COLMAP
artifacts, and calibration frames.  This downloader deliberately starts from
``transforms.json`` and selects original images by exact basename, so invoking
it can never silently expand into a full-folder download.

Google Drive folder enumeration requires ``gdown==6.1.0``.  File transfer,
hashing, resume support, and atomic replacement use only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Protocol, Sequence

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "configs" / "assets" / "manifest.json"
)
DOWNLOAD_URL = "https://drive.usercontent.google.com/download"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ROOT_FILES = {
    "transforms_json": "transforms.json",
    "sparse_pc_ply": "sparse_pc.ply",
}
RECEIPT_NAME = ".quadpilot_source_receipt.json"
PLAN_NAME = "download_plan.json"
FAILURES_NAME = "download_failures.json"


class AssetDownloadError(RuntimeError):
    """Base class for fail-closed asset download errors."""


class ManifestError(AssetDownloadError):
    """The checked-in asset manifest is missing or internally inconsistent."""


class SelectionError(AssetDownloadError):
    """Drive contents do not exactly cover the transform-referenced images."""


class VerificationError(AssetDownloadError):
    """A local or transferred file failed size or SHA-256 verification."""


class DependencyError(AssetDownloadError):
    """An optional dependency needed for public Drive enumeration is absent."""


@dataclass(frozen=True)
class ExpectedFile:
    file_id: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class TrackSource:
    name: str
    folder_id: str
    images_folder_id: str
    root_files: Mapping[str, ExpectedFile]
    referenced_images: int
    referenced_bytes: int


@dataclass(frozen=True)
class RemoteFile:
    file_id: str
    path: str
    size_bytes: int | None = None

    @property
    def basename(self) -> str:
        return PurePosixPath(self.path).name


class DriveClient(Protocol):
    """Small interface used by the orchestration and its offline tests."""

    def list_folder(self, folder_id: str) -> Sequence[RemoteFile]: ...

    def stat_size(self, file_id: str) -> int: ...

    def read_bytes(self, file_id: str) -> bytes: ...

    def open_content(self, file_id: str, start: int = 0) -> BinaryIO: ...


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ManifestError(f"{label} is not a valid public Drive file/folder id")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManifestError(f"{label} must be a non-negative integer")
    return value


def _expected_file(value: Any, label: str) -> ExpectedFile:
    item = _require_mapping(value, label)
    file_id = _require_id(item.get("file_id"), f"{label}.file_id")
    size = _require_nonnegative_int(item.get("size_bytes"), f"{label}.size_bytes")
    sha = item.get("sha256")
    if not isinstance(sha, str) or not SHA256_PATTERN.fullmatch(sha.lower()):
        raise ManifestError(f"{label}.sha256 must contain 64 hexadecimal digits")
    return ExpectedFile(file_id=file_id, size_bytes=size, sha256=sha.lower())


def load_track_source(manifest_path: Path, track: str) -> TrackSource:
    """Load one track exclusively from the checked-in manifest."""

    if not re.fullmatch(r"[a-z0-9_-]+", track):
        raise ManifestError(f"unsafe track name: {track!r}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    root = _require_mapping(manifest, "manifest")
    if root.get("schema_version") != 1:
        raise ManifestError("unsupported manifest schema_version; expected 1")
    external = _require_mapping(root.get("external_sources"), "external_sources")
    tracks = _require_mapping(external.get("gsplat_tracks"), "gsplat_tracks")
    if track not in tracks:
        raise ManifestError(
            f"track {track!r} is absent; available tracks: {', '.join(sorted(tracks))}"
        )
    item = _require_mapping(tracks[track], f"gsplat_tracks.{track}")
    root_files = {
        key: _expected_file(item.get(key), f"gsplat_tracks.{track}.{key}")
        for key in ROOT_FILES
    }
    return TrackSource(
        name=track,
        folder_id=_require_id(item.get("folder_id"), f"{track}.folder_id"),
        images_folder_id=_require_id(
            item.get("images_folder_id"), f"{track}.images_folder_id"
        ),
        root_files=root_files,
        referenced_images=_require_nonnegative_int(
            item.get("transforms_referenced_images"),
            f"{track}.transforms_referenced_images",
        ),
        referenced_bytes=_require_nonnegative_int(
            item.get("transforms_referenced_bytes"),
            f"{track}.transforms_referenced_bytes",
        ),
    )


def normalize_drive_path(path: str) -> str:
    """Normalize a gdown path and reject traversal/absolute path spellings."""

    if not isinstance(path, str) or not path:
        raise SelectionError("Drive returned an empty or non-string path")
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise SelectionError(f"unsafe Drive path: {path!r}")
    return pure.as_posix()


class PublicGoogleDriveClient:
    """Anonymous public Google Drive client with bounded retries."""

    def __init__(self, timeout: float = 60.0, retries: int = 3) -> None:
        if timeout <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries non-negative")
        self.timeout = timeout
        self.retries = retries
        self.user_agent = "QuadPilot-Reproduction/1.0"

    @staticmethod
    def _url(file_id: str) -> str:
        query = urllib.parse.urlencode(
            {"id": file_id, "export": "download", "confirm": "t"}
        )
        return f"{DOWNLOAD_URL}?{query}"

    def _open(self, request: urllib.request.Request) -> BinaryIO:
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise AssetDownloadError(f"HTTP request failed: {last_error}") from last_error

    def list_folder(self, folder_id: str) -> Sequence[RemoteFile]:
        try:
            import gdown
        except ImportError as exc:
            raise DependencyError(
                "public Drive enumeration requires gdown==6.1.0; "
                "install it in the reproduction environment"
            ) from exc
        try:
            listed = gdown.download_folder(
                id=folder_id,
                quiet=True,
                use_cookies=False,
                skip_download=True,
            )
        except Exception as exc:  # gdown exposes several parser/network errors
            raise AssetDownloadError(
                f"cannot enumerate public Drive folder {folder_id}: {exc}"
            ) from exc
        if listed is None:
            raise AssetDownloadError(f"Drive folder {folder_id} returned no listing")
        files: list[RemoteFile] = []
        for item in listed:
            file_id = _require_id(getattr(item, "id", None), "Drive listing id")
            path = normalize_drive_path(str(getattr(item, "path", "")))
            files.append(RemoteFile(file_id=file_id, path=path))
        return files

    def stat_size(self, file_id: str) -> int:
        request = urllib.request.Request(
            self._url(file_id),
            method="HEAD",
            headers={"User-Agent": self.user_agent},
        )
        with self._open(request) as response:
            length = response.headers.get("Content-Length")
            if length is None or not str(length).isdigit():
                raise AssetDownloadError(
                    f"Drive file {file_id} did not provide Content-Length"
                )
            return int(length)

    def read_bytes(self, file_id: str) -> bytes:
        with self.open_content(file_id) as response:
            return response.read()

    def open_content(self, file_id: str, start: int = 0) -> BinaryIO:
        headers = {"User-Agent": self.user_agent}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(self._url(file_id), headers=headers)
        return self._open(request)


def _response_status(response: BinaryIO) -> int:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return int(status or 200)


def _header(response: BinaryIO, name: str) -> str | None:
    headers = getattr(response, "headers", {})
    value = headers.get(name) if hasattr(headers, "get") else None
    return None if value is None else str(value)


def verify_existing(path: Path, expected_size: int, expected_sha256: str | None) -> str:
    """Verify size and always compute SHA-256 before accepting an existing file."""

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise VerificationError(
            f"existing {path} has {actual_size} bytes, expected {expected_size}"
        )
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise VerificationError(
            f"existing {path} SHA-256 {actual_sha} != {expected_sha256}"
        )
    return actual_sha


def download_atomic(
    client: DriveClient,
    remote: RemoteFile,
    destination: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[str, str]:
    """Resume into ``.part``, verify, then atomically publish the final file."""

    if remote.size_bytes is None:
        raise ValueError("download_atomic requires a known remote size")
    if destination.exists():
        return verify_existing(
            destination, remote.size_bytes, expected_sha256
        ), "existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    if part.exists() and part.stat().st_size > remote.size_bytes:
        part.unlink()
    if part.exists() and part.stat().st_size == remote.size_bytes:
        part_sha = sha256_file(part)
        if expected_sha256 is None or part_sha == expected_sha256:
            os.replace(part, destination)
            return part_sha, "resumed"
        part.unlink()

    offset = part.stat().st_size if part.exists() else 0
    resumed = False
    with client.open_content(remote.file_id, start=offset) as response:
        status = _response_status(response)
        append = offset > 0 and status == 206
        if append:
            content_range = _header(response, "Content-Range")
            if content_range is None or not content_range.startswith(
                f"bytes {offset}-"
            ):
                raise AssetDownloadError(
                    f"Drive returned an invalid Content-Range for {remote.path}: "
                    f"{content_range!r}"
                )
        elif status not in {200, 206}:
            raise AssetDownloadError(f"Drive returned HTTP {status} for {remote.path}")
        mode = "ab" if append else "wb"
        resumed = append
        with part.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())

    actual_size = part.stat().st_size
    if actual_size != remote.size_bytes:
        raise VerificationError(
            f"partial {part} has {actual_size} bytes, expected {remote.size_bytes}"
        )
    actual_sha = sha256_file(part)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise VerificationError(
            f"downloaded {remote.path} SHA-256 {actual_sha} != {expected_sha256}"
        )
    os.replace(part, destination)
    return actual_sha, "resumed" if resumed else "downloaded"


def read_verified_bytes(
    client: DriveClient, expected: ExpectedFile, local_path: Path | None = None
) -> bytes:
    if local_path is not None and local_path.is_file():
        verify_existing(local_path, expected.size_bytes, expected.sha256)
        return local_path.read_bytes()
    data = client.read_bytes(expected.file_id)
    if len(data) != expected.size_bytes:
        raise VerificationError(
            f"Drive file {expected.file_id} has {len(data)} bytes, "
            f"expected {expected.size_bytes}"
        )
    actual_sha = sha256_bytes(data)
    if actual_sha != expected.sha256:
        raise VerificationError(
            f"Drive file {expected.file_id} SHA-256 {actual_sha} != {expected.sha256}"
        )
    return data


def referenced_basenames(transforms_data: bytes) -> list[str]:
    try:
        document = json.loads(transforms_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"invalid transforms.json: {exc}") from exc
    frames = document.get("frames") if isinstance(document, Mapping) else None
    if not isinstance(frames, list) or not frames:
        raise SelectionError("transforms.json has no non-empty frames list")
    names: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or not isinstance(
            frame.get("file_path"), str
        ):
            raise SelectionError(f"frame {index} has no string file_path")
        normalized = normalize_drive_path(frame["file_path"])
        name = PurePosixPath(normalized).name
        if name in {"", ".", ".."}:
            raise SelectionError(f"frame {index} has an unsafe basename")
        names.append(name)
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise SelectionError(
            "transforms.json repeats image basenames: " + ", ".join(duplicates[:5])
        )
    return names


def enumerate_with_sizes(
    client: DriveClient,
    folder_id: str,
    *,
    workers: int,
) -> list[RemoteFile]:
    listed = list(client.list_folder(folder_id))
    if not listed:
        raise SelectionError(f"images folder {folder_id} is empty")
    normalized: list[RemoteFile] = []
    ids: set[str] = set()
    for item in listed:
        file_id = _require_id(item.file_id, "Drive listing id")
        if file_id in ids:
            raise SelectionError(f"Drive listing repeated file id {file_id}")
        ids.add(file_id)
        normalized.append(
            RemoteFile(
                file_id=file_id,
                path=normalize_drive_path(item.path),
                size_bytes=item.size_bytes,
            )
        )

    missing_sizes = [item for item in normalized if item.size_bytes is None]
    discovered: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(client.stat_size, item.file_id): item
            for item in missing_sizes
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                size = future.result()
            except Exception as exc:
                raise AssetDownloadError(
                    f"cannot read size for {item.path} ({item.file_id}): {exc}"
                ) from exc
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise AssetDownloadError(f"invalid size for {item.path}: {size!r}")
            discovered[item.file_id] = size
    return sorted(
        [
            RemoteFile(
                item.file_id,
                item.path,
                item.size_bytes
                if item.size_bytes is not None
                else discovered[item.file_id],
            )
            for item in normalized
        ],
        key=lambda item: item.path,
    )


def select_referenced_images(
    references: Sequence[str],
    remote_files: Sequence[RemoteFile],
    source: TrackSource,
) -> list[RemoteFile]:
    by_name: dict[str, RemoteFile] = {}
    duplicate_names: set[str] = set()
    for item in remote_files:
        name = item.basename
        if name in by_name:
            duplicate_names.add(name)
        by_name[name] = item
    ambiguous = sorted(set(references).intersection(duplicate_names))
    if ambiguous:
        raise SelectionError(
            "images folder has ambiguous basenames: " + ", ".join(ambiguous[:5])
        )
    missing = [name for name in references if name not in by_name]
    if missing:
        raise SelectionError(
            f"{len(missing)} transform-referenced images are absent from Drive: "
            + ", ".join(missing[:5])
        )
    selected = [by_name[name] for name in references]
    if len(selected) != source.referenced_images:
        raise SelectionError(
            f"selected {len(selected)} images, manifest expects "
            f"{source.referenced_images}"
        )
    selected_bytes = sum(int(item.size_bytes or 0) for item in selected)
    if selected_bytes != source.referenced_bytes:
        raise SelectionError(
            f"selected images total {selected_bytes} bytes, manifest expects "
            f"{source.referenced_bytes}"
        )
    return selected


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)


def _receipt_sha(
    receipt: Mapping[str, Any], relative_path: str, remote: RemoteFile
) -> str | None:
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        return None
    item = files.get(relative_path)
    if not isinstance(item, Mapping):
        return None
    sha = item.get("sha256")
    if (
        item.get("file_id") != remote.file_id
        or item.get("size_bytes") != remote.size_bytes
        or not isinstance(sha, str)
        or not SHA256_PATTERN.fullmatch(sha.lower())
    ):
        return None
    return sha.lower()


def _failure(phase: str, path: str, file_id: str, exc: BaseException) -> dict[str, str]:
    return {
        "phase": phase,
        "path": path,
        "file_id": file_id,
        "error": f"{type(exc).__name__}: {exc}",
    }


def download_track(
    *,
    track: str,
    manifest_path: Path,
    output_dir: Path,
    dry_run: bool,
    workers: int = 8,
    client: DriveClient | None = None,
) -> dict[str, Any]:
    """Plan or execute one strictly transform-referenced track download."""

    if workers < 1:
        raise ValueError("workers must be at least one")
    source = load_track_source(manifest_path, track)
    drive: DriveClient = client or PublicGoogleDriveClient()
    track_dir = output_dir.expanduser().resolve() / source.name
    failures: list[dict[str, str]] = []
    root_records: dict[str, dict[str, Any]] = {}

    if dry_run:
        transforms_path = track_dir / ROOT_FILES["transforms_json"]
        transforms_data = read_verified_bytes(
            drive,
            source.root_files["transforms_json"],
            transforms_path if transforms_path.is_file() else None,
        )
        sparse = source.root_files["sparse_pc_ply"]
        sparse_path = track_dir / ROOT_FILES["sparse_pc_ply"]
        if sparse_path.is_file():
            verify_existing(sparse_path, sparse.size_bytes, sparse.sha256)
        else:
            remote_size = drive.stat_size(sparse.file_id)
            if remote_size != sparse.size_bytes:
                raise VerificationError(
                    f"remote sparse_pc.ply has {remote_size} bytes, "
                    f"manifest expects {sparse.size_bytes}"
                )
    else:
        track_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in ROOT_FILES.items():
            expected = source.root_files[key]
            remote = RemoteFile(expected.file_id, filename, expected.size_bytes)
            try:
                sha, status = download_atomic(
                    drive,
                    remote,
                    track_dir / filename,
                    expected_sha256=expected.sha256,
                )
            except Exception as exc:
                failures.append(_failure("root", filename, expected.file_id, exc))
                _write_json_atomic(
                    track_dir / FAILURES_NAME,
                    {"schema_version": 1, "track": track, "failures": failures},
                )
                raise AssetDownloadError(
                    f"root asset verification failed; see {track_dir / FAILURES_NAME}"
                ) from exc
            root_records[filename] = {
                "file_id": expected.file_id,
                "size_bytes": expected.size_bytes,
                "sha256": sha,
                "status": status,
            }
        transforms_data = (track_dir / "transforms.json").read_bytes()

    try:
        references = referenced_basenames(transforms_data)
        remote_files = enumerate_with_sizes(
            drive, source.images_folder_id, workers=workers
        )
        selected = select_referenced_images(references, remote_files, source)
    except Exception as exc:
        if not dry_run:
            failures.append(
                _failure("inventory", "images/", source.images_folder_id, exc)
            )
            _write_json_atomic(
                track_dir / FAILURES_NAME,
                {"schema_version": 1, "track": track, "failures": failures},
            )
        raise
    plan = {
        "schema_version": 1,
        "track": track,
        "track_folder_id": source.folder_id,
        "images_folder_id": source.images_folder_id,
        "enumerated_original_images": len(remote_files),
        "selected_images": len(selected),
        "selected_bytes": sum(int(item.size_bytes or 0) for item in selected),
        "excluded_original_images": len(remote_files) - len(selected),
        "files": [
            {
                "file_id": item.file_id,
                "drive_path": item.path,
                "output_path": f"images/{item.basename}",
                "size_bytes": item.size_bytes,
            }
            for item in selected
        ],
    }
    if dry_run:
        plan["dry_run"] = True
        return plan

    _write_json_atomic(track_dir / PLAN_NAME, plan)
    receipt_path = track_dir / RECEIPT_NAME
    try:
        receipt = _read_json_object(receipt_path)
    except Exception as exc:
        failures.append(_failure("receipt", RECEIPT_NAME, "local", exc))
        _write_json_atomic(
            track_dir / FAILURES_NAME,
            {"schema_version": 1, "track": track, "failures": failures},
        )
        raise
    receipt_files: dict[str, dict[str, Any]] = {}
    for filename, record in root_records.items():
        receipt_files[filename] = {
            key: value for key, value in record.items() if key != "status"
        }

    # Preserve only receipt entries that still describe this exact plan.  This
    # makes each completed image independently resumable even if the process is
    # interrupted before the remaining futures finish.
    for item in selected:
        relative = f"images/{item.basename}"
        receipt_sha = _receipt_sha(receipt, relative, item)
        if receipt_sha is not None:
            receipt_files[relative] = {
                "file_id": item.file_id,
                "drive_path": item.path,
                "size_bytes": item.size_bytes,
                "sha256": receipt_sha,
            }

    def write_receipt() -> None:
        _write_json_atomic(
            receipt_path,
            {
                "schema_version": 1,
                "track": track,
                "images_folder_id": source.images_folder_id,
                "files": receipt_files,
            },
        )

    write_receipt()

    def transfer(item: RemoteFile) -> tuple[str, dict[str, Any]]:
        relative = f"images/{item.basename}"
        expected_sha = _receipt_sha(receipt, relative, item)
        destination = track_dir / "images" / item.basename
        if destination.exists() and expected_sha is None:
            raise VerificationError(
                f"existing {destination} has no matching trusted receipt; "
                "refusing to accept it by size alone"
            )
        sha, status = download_atomic(
            drive,
            item,
            destination,
            expected_sha256=expected_sha,
        )
        return relative, {
            "file_id": item.file_id,
            "drive_path": item.path,
            "size_bytes": item.size_bytes,
            "sha256": sha,
            "status": status,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(transfer, item): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            try:
                relative, record = future.result()
                receipt_files[relative] = {
                    key: value for key, value in record.items() if key != "status"
                }
                write_receipt()
            except Exception as exc:
                failures.append(
                    _failure("image", f"images/{item.basename}", item.file_id, exc)
                )

    write_receipt()
    _write_json_atomic(
        track_dir / FAILURES_NAME,
        {"schema_version": 1, "track": track, "failures": failures},
    )
    if failures:
        raise AssetDownloadError(
            f"{len(failures)} image downloads failed; rerun to resume after reviewing "
            f"{track_dir / FAILURES_NAME}"
        )
    plan["dry_run"] = False
    plan["receipt"] = str(receipt_path)
    plan["failure_list"] = str(track_dir / FAILURES_NAME)
    return plan


def _summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "files"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download only original GSplat images explicitly referenced by "
            "transforms.json"
        )
    )
    parser.add_argument("--track", required=True, help="circle, uturn, or lemniscate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="parent directory; a track-named data directory is created beneath it",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="asset manifest path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify metadata and print a plan without writing or downloading images",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="parallel HEAD/download workers"
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout")
    parser.add_argument("--retries", type=int, default=3, help="HTTP open retries")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = download_track(
            track=args.track,
            manifest_path=args.manifest.resolve(),
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            workers=args.workers,
            client=PublicGoogleDriveClient(timeout=args.timeout, retries=args.retries),
        )
    except (AssetDownloadError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_summary(plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
