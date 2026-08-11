"""CLI regression tests for explicit GSplat run-directory checks."""

from __future__ import annotations

import hashlib
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import check_repro_assets


class CheckReproAssetsTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(root: Path, payload: bytes) -> Path:
        manifest = {
            "tracks": {
                "lemniscate": {
                    "run": "historical-run",
                    "files": {
                        "config.yml": {
                            "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    },
                }
            }
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_absolute_and_relative_run_dir_override_historical_path(self) -> None:
        payload = b"candidate config\n"
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, payload)
            run_dir = root / "candidate-run"
            run_dir.mkdir()
            (run_dir / "config.yml").write_bytes(payload)

            for supplied in (run_dir.resolve(), Path(os.path.relpath(run_dir))):
                with self.subTest(run_dir=supplied):
                    output = StringIO()
                    with redirect_stdout(output):
                        result = check_repro_assets.main(
                            [
                                "--manifest",
                                str(manifest),
                                "--asset-root",
                                str(root / "missing-history"),
                                "--track",
                                "lemniscate",
                                "--run-dir",
                                str(supplied),
                                "--hash",
                            ]
                        )
                    self.assertEqual(result, 0)
                    self.assertIn(str(run_dir.resolve()), output.getvalue())
                    self.assertIn("STATUS READY", output.getvalue())
                    self.assertIn("hash=verified", output.getvalue())

    def test_run_dir_with_all_fails_closed(self) -> None:
        errors = StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
            check_repro_assets.main(["--track", "all", "--run-dir", "candidate"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires one explicit --track", errors.getvalue())

    def test_default_layout_still_uses_manifest_run_name(self) -> None:
        payload = b"historical config\n"
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, payload)
            run_dir = root / "lemniscate" / "splatfacto" / "historical-run"
            run_dir.mkdir(parents=True)
            (run_dir / "config.yml").write_bytes(payload)

            output = StringIO()
            with redirect_stdout(output):
                result = check_repro_assets.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--asset-root",
                        str(root),
                        "--track",
                        "lemniscate",
                        "--hash",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertIn(str(run_dir), output.getvalue())
            self.assertIn("STATUS READY", output.getvalue())


if __name__ == "__main__":
    unittest.main()
