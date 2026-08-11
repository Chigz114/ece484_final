"""CPU-only contract tests for the pinned GSplat Docker wrapper.

The behavioral tests replace both Docker and nvidia-smi with tiny local fakes;
they never contact a daemon, pull an image, start a container, or touch a GPU.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "reproduce_gsplat_docker.sh"
IMAGE_REF = (
    "dromni/nerfstudio@sha256:"
    "ff0107a7db96bb8ee29c638729328b832b268b890c50f2a2ff25988bb84d4f75"
)
LPIPS_ALEXNET_RELATIVE_PATH = "torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
LPIPS_ALEXNET_SIZE_BYTES = 244408911
LPIPS_ALEXNET_SHA256 = (
    "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
)


def embedded_heredoc(source: str, variable: str) -> str:
    match = re.search(
        rf"{re.escape(variable)}=\$\(cat <<'PY'\n(?P<python>.*?)\nPY\n\)",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"embedded Python variable not found: {variable}")
    return match.group("python")


class StaticDockerWrapperContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_image_is_pinned_to_the_complete_requested_digest(self) -> None:
        self.assertIn(f"readonly IMAGE_REF='{IMAGE_REF}'", self.source)
        self.assertNotIn("dromni/nerfstudio:latest", self.source)
        self.assertNotIn("dromni/nerfstudio:1.1.4", self.source)

    def test_container_has_no_implicit_pull_or_network_and_data_is_read_only(self) -> None:
        self.assertIn("--pull=never", self.source)
        self.assertIn("--network none", self.source)
        self.assertIn("dst=/data,readonly", self.source)
        self.assertNotRegex(self.source, r"(?m)^\s*docker\s+pull\b")

    def test_source_profile_is_explicit_and_both_profiles_are_frozen(self) -> None:
        self.assertIn("TRACK=''", self.source)
        self.assertIn("--track is required; choose exactly lemniscate or uturn", self.source)
        self.assertIn("data directory basename", self.source)
        for name, value in (
            ("LEMNISCATE_RECEIPT_SHA256", "6614c5be765ab7456eac95403af4b2c6fb34e757afc263ba3aa7b9f075cd356a"),
            ("LEMNISCATE_RECEIPT_FILES", "1555"),
            ("LEMNISCATE_RECEIPT_IMAGES", "1553"),
            ("LEMNISCATE_RECEIPT_IMAGE_BYTES", "3362065056"),
            ("LEMNISCATE_RECEIPT_TOTAL_BYTES", "3370611629"),
            ("LEMNISCATE_SPARSE_POINTS", "183994"),
            ("UTURN_RECEIPT_SHA256", "a42c422dc084375e7f2bf5ef530ac7a5409e9abc0d6c5b3fa90ccd840beb6023"),
            ("UTURN_RECEIPT_FILES", "1442"),
            ("UTURN_RECEIPT_IMAGES", "1440"),
            ("UTURN_RECEIPT_IMAGE_BYTES", "3062618402"),
            ("UTURN_RECEIPT_TOTAL_BYTES", "3070717413"),
            ("UTURN_SPARSE_POINTS", "175292"),
        ):
            self.assertIn(f"readonly {name}='{value}'", self.source)
        for key in (
            "source_profile",
            "expected_source_receipt_files",
            "expected_source_receipt_images",
            "expected_source_receipt_image_bytes",
            "expected_source_receipt_total_bytes",
            "expected_source_sparse_points",
        ):
            self.assertIn(f"record_value {key}", self.source)

    def test_lpips_alexnet_cache_is_pinned_and_overmounted_read_only(self) -> None:
        self.assertIn(
            f"readonly LPIPS_ALEXNET_RELATIVE_PATH='{LPIPS_ALEXNET_RELATIVE_PATH}'",
            self.source,
        )
        self.assertIn(
            f"readonly LPIPS_ALEXNET_SIZE_BYTES='{LPIPS_ALEXNET_SIZE_BYTES}'",
            self.source,
        )
        self.assertIn(
            f"readonly LPIPS_ALEXNET_SHA256='{LPIPS_ALEXNET_SHA256}'",
            self.source,
        )
        self.assertIn("LPIPS AlexNet size mismatch", self.source)
        self.assertIn("LPIPS AlexNet SHA-256 mismatch", self.source)
        self.assertIn("LPIPS AlexNet cache path must not contain symlinks", self.source)
        self.assertIn(
            'src=$LPIPS_ALEXNET_PATH,dst=/cache/$LPIPS_ALEXNET_RELATIVE_PATH,readonly',
            self.source,
        )
        self.assertIn("lpips_alexnet_cache_verified", self.source)

    def test_training_output_isolated_and_periodic_eval_is_disabled(self) -> None:
        self.assertIn('TRAINING_OUTPUT_DIR="$RUN_DIR/training-output"', self.source)
        self.assertIn('src=$TRAINING_OUTPUT_DIR,dst=/outputs', self.source)
        self.assertNotIn('src=$RUN_DIR,dst=/outputs', self.source)
        for option in (
            "--steps-per-eval-batch 0",
            "--steps-per-eval-image 0",
            "--steps-per-eval-all-images 0",
        ):
            self.assertIn(option, self.source)
        self.assertIn("readonly TRAIN_MAX_JOBS='4'", self.source)
        self.assertIn('--env "MAX_JOBS=$TRAIN_MAX_JOBS"', self.source)
        self.assertIn('record_value training_max_jobs "$TRAIN_MAX_JOBS"', self.source)

    def test_modes_and_runtime_half_resolution_are_explicit(self) -> None:
        self.assertRegex(self.source, r"smoke-1\)\s+MAX_ITERATIONS='1'")
        self.assertRegex(self.source, r"smoke-101\)\s+MAX_ITERATIONS='101'")
        self.assertRegex(self.source, r"train-30k\)\s+MAX_ITERATIONS='30000'")
        self.assertIn(
            '--pipeline.datamanager.camera-res-scale-factor "$HALF_RES_SCALE"',
            self.source,
        )
        self.assertRegex(
            self.source,
            r'--pipeline\.datamanager\.camera-res-scale-factor "\$HALF_RES_SCALE"\s+'
            r'--pipeline\.model\.num-downscales "\$NUM_DOWNSCALES"\s+'
            r'--pipeline\.model\.resolution-schedule 3000\s+'
            r'nerfstudio-data\s+--downscale-factor 1',
        )
        self.assertIn("original full-resolution experiment", self.source)
        self.assertIn("1/8 for steps 0-2999", self.source)
        self.assertIn("1/4 for 3000-5999", self.source)

    def test_audit_and_busy_gpu_gates_are_present(self) -> None:
        for artifact in (
            "nvidia-smi.before.txt",
            "nvidia-smi.after.txt",
            "gpu-query.csv",
            "compute-apps.csv",
            "docker-image-inspect.json",
            "provenance.env",
            "command.sh",
        ):
            self.assertIn(artifact, self.source)
        self.assertIn("--allow-busy-gpu", self.source)
        self.assertIn("refusing by default", self.source)

    def test_cpu_preflight_pins_versions_and_only_two_known_pip_deviations(self) -> None:
        for name, expected in (
            ("nerfstudio", "1.1.4"),
            ("gsplat", "1.0.0"),
            ("torch", "2.1.2+cu118"),
            ("torchvision", "0.16.2+cu118"),
            ("viser", "0.2.3"),
        ):
            self.assertIn(f'"{name}": "{expected}"', self.source)
        self.assertIn("rawpy 0.22.0 has requirement numpy>=2.0", self.source)
        self.assertNotIn("rawpy 0.22.0 has requirement numpy>=2,", self.source)
        self.assertIn("ninja 1.11.1.1 is not supported on this platform", self.source)
        self.assertIn("actual_deviations != expected_deviations", self.source)
        executable_source = "\n".join(
            line for line in self.source.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotRegex(executable_source, r"ns-train\s+splatfacto\s+--help")
        self.assertIn('"images": expected_image_count', self.source)
        self.assertIn('"sparse_points": expected_sparse_points', self.source)
        self.assertIn('os.environ["QUADPILOT_SOURCE_PROFILE"]', self.source)
        self.assertIn("--env NVIDIA_VISIBLE_DEVICES=void", self.source)
        self.assertIn("--env CUDA_VISIBLE_DEVICES=", self.source)

    def test_builtin_only_launcher_is_exactly_audited_and_compiles(self) -> None:
        guard = embedded_heredoc(self.source, "METHOD_PLUGIN_GUARD_PY")
        training_body = embedded_heredoc(self.source, "TRAIN_LAUNCHER_BODY_PY")
        compile(guard, "<method-plugin-guard>", "exec")
        compile(guard + "\n" + training_body, "<training-launcher>", "exec")
        for evidence in (
            '"bionerf.bionerf_config:bionerf_method"',
            '"8dd1975af4901f2d5c8e0f1ec9401e4bae1f016d36c84b049c80f46de0f204f6"',
            '"nerfstudio.method_configs"',
            '"2c955c48ff6b42e7823c90fc36dca9344fc6010c511238337b1528aa36c6930f"',
            '"b004bcf9e7ba5de52d94138a86aae260c58dbd751eb901ae41f0ab3f75a22718"',
            '"2a3b31c832427ca6c56b068a9b18039ea616834da5046852e0231c9df1b6d3c9"',
            "_qp_actual != _QP_EXPECTED_METHOD_ENTRY_POINTS",
            "_qp_metadata.EntryPoints(())",
            "METHOD_PLUGIN_AUDIT",
        ):
            self.assertIn(evidence, guard)
        self.assertIn('sys.argv[0] = "ns-train"', training_body)
        self.assertIn("from nerfstudio.scripts.train import entrypoint", training_body)
        self.assertIn('record_value method_plugin_policy \'built-in-only\'', self.source)
        self.assertIn('record_value method_plugin_guard_sha256', self.source)
        self.assertIn('record_value training_launcher_sha256', self.source)
        self.assertNotRegex(guard, r"\b(pip|subprocess|unlink|remove|rename)\b")

    def test_embedded_preflight_python_compiles_and_receipt_hashes_every_file(self) -> None:
        guard = embedded_heredoc(self.source, "METHOD_PLUGIN_GUARD_PY")
        embedded = embedded_heredoc(self.source, "PREFLIGHT_BODY_PY")
        compile(guard + "\n" + embedded, "<embedded-preflight>", "exec")
        self.assertIn('"method_count": 43', embedded)
        self.assertIn('"splatfacto_present": "splatfacto" in all_methods', embedded)

        function_source = embedded[: embedded.index("\nexpected_versions =")]
        namespace: dict[str, object] = {}
        exec(compile(function_source, "<receipt-verifier>", "exec"), namespace)
        verifier = namespace["verify_source_receipt"]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = {
                "transforms.json": b"{}\n",
                "sparse_pc.ply": b"ply\n",
                "images/frame.png": b"png-bytes",
            }
            files: dict[str, dict[str, object]] = {}
            for relative, payload in contents.items():
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                files[relative] = {
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            receipt_bytes = json.dumps(
                {
                    "schema_version": 1,
                    "track": "lemniscate",
                    "files": files,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            (root / ".quadpilot_source_receipt.json").write_bytes(receipt_bytes)
            arguments = {
                "expected_track": "lemniscate",
                "expected_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "expected_file_count": 3,
                "expected_image_count": 1,
                "expected_image_bytes": len(contents["images/frame.png"]),
                "expected_total_bytes": sum(map(len, contents.values())),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                summary = verifier(root, **arguments)
            self.assertEqual(summary["verified_files"], 3)
            self.assertEqual(summary["source_profile"], "lemniscate")
            self.assertEqual(
                summary["verified_image_bytes"], arguments["expected_image_bytes"]
            )
            self.assertEqual(summary["verified_bytes"], arguments["expected_total_bytes"])

            wrong_profile = dict(arguments, expected_track="uturn")
            with self.assertRaisesRegex(RuntimeError, "schema/track mismatch"):
                with contextlib.redirect_stdout(io.StringIO()):
                    verifier(root, **wrong_profile)

            (root / "images" / "frame.png").write_bytes(b"corrupted")
            with self.assertRaisesRegex(RuntimeError, "size mismatch|SHA-256 mismatch"):
                with contextlib.redirect_stdout(io.StringIO()):
                    verifier(root, **arguments)

    def test_cache_home_preserves_the_image_user_site_and_executable_path(self) -> None:
        self.assertIn("readonly IMAGE_EDITABLE_SOURCE='/home/user/nerfstudio'", self.source)
        self.assertIn(
            "readonly IMAGE_USER_SITE='/home/user/.local/lib/python3.10/site-packages'",
            self.source,
        )
        self.assertIn(
            'readonly IMAGE_PYTHONPATH="$IMAGE_EDITABLE_SOURCE:$IMAGE_USER_SITE"',
            self.source,
        )
        self.assertIn("/home/user/.local/bin'", self.source)
        self.assertEqual(self.source.count('--env "PYTHONPATH=$IMAGE_PYTHONPATH"'), 2)
        self.assertEqual(self.source.count('--env "PATH=$IMAGE_PATH"'), 2)


@unittest.skipUnless(
    os.environ.get("QUADPILOT_RUN_REAL_DOCKER_DIAGNOSTIC") == "1",
    "set QUADPILOT_RUN_REAL_DOCKER_DIAGNOSTIC=1 for the pinned CPU container check",
)
class RealPinnedImageMethodGuardTests(unittest.TestCase):
    """Explicit opt-in test; it never pulls, mounts host data, or requests a GPU."""

    def test_pinned_image_imports_splatfacto_with_audited_external_plugins_disabled(
        self,
    ) -> None:
        if os.name == "nt" or shutil.which("docker") is None:
            self.skipTest("requires Docker from a POSIX/WSL environment")
        source = SCRIPT.read_text(encoding="utf-8")
        guard = embedded_heredoc(source, "METHOD_PLUGIN_GUARD_PY")
        diagnostic = guard + textwrap.dedent(
            """

            from nerfstudio.configs.method_configs import all_methods
            from nerfstudio.scripts.train import entrypoint

            assert len(all_methods) == 43, len(all_methods)
            assert "splatfacto" in all_methods
            assert callable(entrypoint)
            print("REAL_BUILTIN_METHOD_DIAGNOSTIC_OK", flush=True)
            """
        )
        command = [
            "docker",
            "run",
            "-i",
            "--rm",
            "--pull=never",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/tmp/quadpilot-home",
            "--env",
            "PYTHONPATH=/home/user/nerfstudio:/home/user/.local/lib/python3.10/site-packages",
            "--env",
            "PATH=/home/user/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=void",
            "--env",
            "CUDA_VISIBLE_DEVICES=",
            "--workdir",
            "/tmp",
            IMAGE_REF,
            "python3.10",
            "-",
        ]
        self.assertNotIn("--gpus", command)
        result = subprocess.run(
            command,
            input=diagnostic,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("METHOD_PLUGIN_AUDIT ", result.stdout)
        self.assertIn("REAL_BUILTIN_METHOD_DIAGNOSTIC_OK", result.stdout)


@unittest.skipIf(os.name == "nt" or shutil.which("bash") is None, "requires a POSIX bash")
class MockedDockerWrapperBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bin_dir = self.root / "bin"
        self.data = self.root / "lemniscate"
        self.uturn_data = self.root / "uturn"
        self.output = self.root / "output"
        self.cache = self.root / "cache"
        self.bin_dir.mkdir()
        for data_dir, track in (
            (self.data, "lemniscate"),
            (self.uturn_data, "uturn"),
        ):
            (data_dir / "images").mkdir(parents=True)
            (data_dir / "transforms.json").write_text(
                '{"frames": []}\n', encoding="utf-8"
            )
            (data_dir / "sparse_pc.ply").write_bytes(b"ply\nmock\n")
            (data_dir / ".quadpilot_source_receipt.json").write_text(
                json.dumps(
                    {"schema_version": 1, "track": track, "files": {}}
                )
                + "\n",
                encoding="utf-8",
            )
        self.lpips_checkpoint = (
            self.cache
            / "nerfstudio-ff0107a7db96"
            / LPIPS_ALEXNET_RELATIVE_PATH
        )
        self.lpips_checkpoint.parent.mkdir(parents=True)
        with self.lpips_checkpoint.open("wb") as handle:
            # Sparse fixture: stat sees the pinned size while no 244 MB payload
            # is allocated. The mock sha256sum below controls only this path.
            handle.truncate(LPIPS_ALEXNET_SIZE_BYTES)
        self.docker_log = self.root / "docker-calls.txt"

        self._write_executable(
            self.bin_dir / "docker",
            """
            #!/usr/bin/env bash
            set -Eeuo pipefail
            printf '%s\n' "$*" >>"$MOCK_DOCKER_LOG"
            if [[ "${1-}" == version ]]; then
              printf 'mock-docker-1.0\n'
              exit 0
            fi
            if [[ "${1-}" == image && "${2-}" == inspect ]]; then
              if [[ "${MOCK_IMAGE_MISSING:-0}" == 1 ]]; then
                printf 'mock: image missing\n' >&2
                exit 1
              fi
              printf '[{"RepoDigests":["%s"]}]\n' "${3-}"
              exit 0
            fi
            if [[ "${1-}" == run ]]; then
              if [[ "$*" == *"python3.10 -c"* && "$*" != *"--gpus"* ]]; then
                if [[ "$*" != *"--env PYTHONPATH=/home/user/nerfstudio:/home/user/.local/lib/python3.10/site-packages"* ||
                      "$*" != *"--env PATH="*"/home/user/.local/bin"* ]]; then
                  printf 'importlib.metadata.PackageNotFoundError: nerfstudio\n' >&2
                  exit 8
                fi
                if [[ "${MOCK_PREFLIGHT_FAIL:-0}" == 1 ]]; then
                  printf 'mock unknown preflight deviation\n' >&2
                  exit 9
                fi
                printf 'METHOD_PLUGIN_AUDIT {"policy":"built-in-only","disabled_entry_points":9}\n'
                printf 'BUILTIN_METHOD_CONFIGS_OK {"method_count":43,"splatfacto_present":true}\n'
                if [[ "$*" == *"QUADPILOT_SOURCE_PROFILE=uturn"* ]]; then
                  printf 'RECEIPT_OK {"source_profile":"uturn","verified_files":1442,"verified_images":1440,"verified_image_bytes":3062618402,"verified_bytes":3070717413}\n'
                else
                  printf 'RECEIPT_OK {"source_profile":"lemniscate","verified_files":1555,"verified_images":1553,"verified_image_bytes":3362065056,"verified_bytes":3370611629}\n'
                fi
                exit 0
              fi
              output_source=''
              experiment=''
              timestamp=''
              iterations=''
              shift
              while (($#)); do
                case "$1" in
                  --mount)
                    if [[ "${2-}" == type=bind,src=*,dst=/outputs ]]; then
                      output_source=${2#type=bind,src=}
                      output_source=${output_source%,dst=/outputs}
                    fi
                    shift 2
                    ;;
                  --experiment-name)
                    experiment=${2-}; shift 2 ;;
                  --timestamp)
                    timestamp=${2-}; shift 2 ;;
                  --max-num-iterations)
                    iterations=${2-}; shift 2 ;;
                  *) shift ;;
                esac
              done
              if [[ "${MOCK_NO_ARTIFACTS:-0}" != 1 ]]; then
                [[ -n "$output_source" && -n "$experiment" && -n "$timestamp" && -n "$iterations" ]]
                final_step=$((iterations - 1))
                printf -v checkpoint 'step-%09d.ckpt' "$final_step"
                artifact_root="$output_source/$experiment/splatfacto/$timestamp"
                mkdir -p "$artifact_root/nerfstudio_models"
                printf 'max_num_iterations: %s\n' "$iterations" >"$artifact_root/config.yml"
                printf 'mock checkpoint step %s\n' "$final_step" \
                  >"$artifact_root/nerfstudio_models/$checkpoint"
              fi
              printf 'METHOD_PLUGIN_AUDIT {"policy":"built-in-only","disabled_entry_points":9}\n'
              printf 'mock container should only run in explicit non-dry tests\n'
              exit 0
            fi
            exit 64
            """,
        )
        self._write_executable(
            self.bin_dir / "nvidia-smi",
            """
            #!/usr/bin/env bash
            set -Eeuo pipefail
            joined="$*"
            if [[ "$joined" == *"--query-gpu="* ]]; then
              printf '0, GPU-mock-0000, Mock GPU, 8192, %s, %s\n' \
                "${MOCK_GPU_MEMORY_MIB:-100}" "${MOCK_GPU_UTIL:-0}"
              exit 0
            fi
            if [[ "$joined" == *"--query-compute-apps="* ]]; then
              if [[ "${MOCK_GPU_PROCESS:-0}" == 1 ]]; then
                printf 'GPU-mock-0000, 4242, mock-worker, 2048\n'
              fi
              exit 0
            fi
            if [[ "${1-}" == -q ]]; then
              printf 'MOCK NVIDIA-SMI GPU %s\n' "${3-unknown}"
              exit 0
            fi
            exit 65
            """,
        )
        self._write_executable(
            self.bin_dir / "sha256sum",
            f"""
            #!/usr/bin/env bash
            set -Eeuo pipefail
            last=''
            if (($#)); then
              last=${{!#}}
            fi
            if [[ "$last" == "${{MOCK_LPIPS_PATH:-}}" ]]; then
              if [[ "${{MOCK_BAD_LPIPS_HASH:-0}}" == 1 ]]; then
                printf '%064d  %s\n' 0 "$last"
              else
                printf '{LPIPS_ALEXNET_SHA256}  %s\n' "$last"
              fi
              exit 0
            fi
            exec /usr/bin/sha256sum "$@"
            """,
        )

        self.environment = os.environ.copy()
        self.environment["PATH"] = f"{self.bin_dir}{os.pathsep}{self.environment['PATH']}"
        self.environment["MOCK_DOCKER_LOG"] = str(self.docker_log)
        self.environment["MOCK_LPIPS_PATH"] = str(self.lpips_checkpoint.resolve())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_executable(path: Path, body: str) -> None:
        path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
        path.chmod(0o755)

    def _invoke(
        self,
        mode: str,
        run_id: str,
        *extra: str,
        environment: dict[str, str] | None = None,
        track: str = "lemniscate",
        data: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "bash",
            str(SCRIPT),
            mode,
            "--data",
            str(data or self.data),
            "--output-root",
            str(self.output),
            "--cache",
            str(self.cache),
            "--run-id",
            run_id,
            "--track",
            track,
            *extra,
        ]
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment or self.environment,
            timeout=10,
        )

    def _run_dir(self, mode: str, run_id: str, *, track: str = "lemniscate") -> Path:
        return self.output / track / mode / run_id

    def test_preflight_runs_only_the_cpu_container_and_records_evidence(self) -> None:
        result = self._invoke("preflight", "preflight-ok")
        self.assertEqual(result.returncode, 0, result.stderr)
        run_dir = self._run_dir("preflight", "preflight-ok")
        self.assertFalse((run_dir / "command.sh").exists())
        self.assertTrue((run_dir / "preflight-command.sh").is_file())
        self.assertTrue((run_dir / "preflight-container.log").is_file())
        receipt_record = run_dir / "receipt-verification.json"
        self.assertTrue(receipt_record.is_file())
        self.assertEqual(json.loads(receipt_record.read_text())["verified_files"], 1555)
        self.assertEqual(
            json.loads(receipt_record.read_text())["verified_image_bytes"],
            3362065056,
        )
        method_audit = json.loads(
            (run_dir / "method-plugin-audit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(method_audit["policy"], "built-in-only")
        builtin_methods = json.loads(
            (run_dir / "builtin-method-configs.json").read_text(encoding="utf-8")
        )
        self.assertEqual(builtin_methods["method_count"], 43)
        self.assertTrue(builtin_methods["splatfacto_present"])
        self.assertFalse((run_dir / "nvidia-smi.before.txt").exists())
        self.assertFalse((run_dir / "nvidia-smi.after.txt").exists())
        self.assertTrue((run_dir / "docker-image-inspect.json").is_file())
        provenance = (run_dir / "provenance.env").read_text(encoding="utf-8")
        self.assertIn(f"image_ref={IMAGE_REF}", provenance)
        self.assertIn("data_mount_mode=readonly", provenance)
        self.assertIn("full_resolution_reproduction=false", provenance)
        self.assertIn("method_plugin_policy=built-in-only", provenance)
        self.assertIn("method_plugin_guard_sha256=", provenance)
        self.assertIn("source_profile=lemniscate", provenance)
        self.assertIn("expected_source_receipt_images=1553", provenance)
        calls = self.docker_log.read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(line.startswith("version ") for line in calls))
        self.assertTrue(any(line.startswith("image inspect ") for line in calls))
        runs = [line for line in calls if line == "run" or line.startswith("run ")]
        self.assertEqual(len(runs), 1)
        self.assertNotIn("--gpus", runs[0])
        self.assertIn("--network none", runs[0])
        self.assertIn("dst=/data,readonly", runs[0])
        self.assertIn(
            "--env PYTHONPATH=/home/user/nerfstudio:/home/user/.local/lib/python3.10/site-packages",
            runs[0],
        )
        self.assertIn("/home/user/.local/bin", runs[0])

    def test_uturn_profile_injects_its_exact_cpu_preflight_contract(self) -> None:
        result = self._invoke(
            "preflight",
            "uturn-profile",
            track="uturn",
            data=self.uturn_data,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        run_dir = self._run_dir("preflight", "uturn-profile", track="uturn")
        receipt = json.loads(
            (run_dir / "receipt-verification.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["source_profile"], "uturn")
        self.assertEqual(receipt["verified_files"], 1442)
        self.assertEqual(receipt["verified_images"], 1440)
        self.assertEqual(receipt["verified_image_bytes"], 3062618402)
        self.assertEqual(receipt["verified_bytes"], 3070717413)
        provenance = (run_dir / "provenance.env").read_text(encoding="utf-8")
        for expected in (
            "source_profile=uturn",
            "expected_source_receipt_files=1442",
            "expected_source_receipt_images=1440",
            "expected_source_receipt_image_bytes=3062618402",
            "expected_source_receipt_total_bytes=3070717413",
            "expected_source_sparse_points=175292",
        ):
            self.assertIn(expected, provenance)
        run_call = next(
            line
            for line in self.docker_log.read_text(encoding="utf-8").splitlines()
            if line == "run" or line.startswith("run ")
        )
        for expected in (
            "QUADPILOT_SOURCE_PROFILE=uturn",
            "QUADPILOT_SOURCE_RECEIPT_FILES=1442",
            "QUADPILOT_SOURCE_RECEIPT_IMAGES=1440",
            "QUADPILOT_SOURCE_RECEIPT_IMAGE_BYTES=3062618402",
            "QUADPILOT_SOURCE_RECEIPT_TOTAL_BYTES=3070717413",
            "QUADPILOT_SOURCE_SPARSE_POINTS=175292",
        ):
            self.assertIn(expected, run_call)

    def test_track_is_required_and_data_basename_must_match_profile(self) -> None:
        missing_track_command = [
            "bash",
            str(SCRIPT),
            "preflight",
            "--data",
            str(self.data),
            "--output-root",
            str(self.output),
            "--cache",
            str(self.cache),
            "--run-id",
            "missing-track",
        ]
        missing = subprocess.run(
            missing_track_command,
            check=False,
            capture_output=True,
            text=True,
            env=self.environment,
            timeout=10,
        )
        self.assertEqual(missing.returncode, 2)
        self.assertIn("--track is required", missing.stderr)
        self.assertFalse(self.docker_log.exists())

        mismatch = self._invoke(
            "preflight",
            "profile-path-mismatch",
            track="uturn",
            data=self.data,
        )
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn("does not match --track profile 'uturn'", mismatch.stderr)
        self.assertFalse(self.docker_log.exists())

    def test_all_training_modes_build_only_the_expected_dry_run(self) -> None:
        cases = (("smoke-1", "1"), ("smoke-101", "101"), ("train-30k", "30000"))
        for index, (mode, iterations) in enumerate(cases):
            with self.subTest(mode=mode):
                run_id = f"dry-{index}"
                result = self._invoke(mode, run_id, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                command = (self._run_dir(mode, run_id) / "command.sh").read_text(
                    encoding="utf-8"
                )
                self.assertIn("--pull=never", command)
                self.assertIn("--network none", command)
                self.assertIn("dst=/data,readonly", command.replace(r"\,", ","))
                self.assertIn(f"--max-num-iterations {iterations}", command)
                self.assertIn("--pipeline.datamanager.camera-res-scale-factor 0.5", command)
                expected_downscales = "2" if mode == "train-30k" else "0"
                self.assertIn(
                    f"--pipeline.model.num-downscales {expected_downscales}", command
                )
                self.assertIn("--pipeline.model.resolution-schedule 3000", command)
                self.assertIn(
                    "PYTHONPATH=/home/user/nerfstudio:/home/user/.local/lib/python3.10/site-packages",
                    command,
                )
                self.assertIn("python3.10 -c", command)
                self.assertIn("METHOD_PLUGIN_AUDIT", command)
                self.assertIn("from nerfstudio.scripts.train import entrypoint", command)
                self.assertIn("--env MAX_JOBS=4", command)
                for option in (
                    "--steps-per-eval-batch 0",
                    "--steps-per-eval-image 0",
                    "--steps-per-eval-all-images 0",
                ):
                    self.assertIn(option, command)
                self.assertIn("training-output", command)
                self.assertRegex(command, r"nerfstudio-data --downscale-factor 1\s*$")
                self.assertIn(IMAGE_REF, command)
        calls = self.docker_log.read_text(encoding="utf-8").splitlines()
        self.assertFalse(any(line == "run" or line.startswith("run ") for line in calls))

    def test_missing_or_wrong_lpips_cache_fails_before_any_docker_call(self) -> None:
        self.lpips_checkpoint.unlink()
        missing = self._invoke("smoke-1", "lpips-missing")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("missing readable LPIPS AlexNet", missing.stderr)
        self.assertFalse(self.docker_log.exists())
        missing_status = (
            self._run_dir("smoke-1", "lpips-missing") / "status.env"
        ).read_text(encoding="utf-8")
        self.assertIn("result=failed", missing_status)

        with self.lpips_checkpoint.open("wb") as handle:
            handle.truncate(LPIPS_ALEXNET_SIZE_BYTES)
        bad_environment = self.environment.copy()
        bad_environment["MOCK_BAD_LPIPS_HASH"] = "1"
        wrong_hash = self._invoke(
            "smoke-1",
            "lpips-wrong-hash",
            environment=bad_environment,
        )
        self.assertEqual(wrong_hash.returncode, 2)
        self.assertIn("LPIPS AlexNet SHA-256 mismatch", wrong_hash.stderr)
        self.assertFalse(self.docker_log.exists())

    def test_busy_gpu_fails_closed_and_override_is_audited(self) -> None:
        busy_environment = self.environment.copy()
        busy_environment["MOCK_GPU_UTIL"] = "90"
        denied = self._invoke(
            "smoke-1",
            "busy-denied",
            environment=busy_environment,
        )
        self.assertEqual(denied.returncode, 2)
        self.assertIn("refusing by default", denied.stderr)
        denied_provenance = (
            self._run_dir("smoke-1", "busy-denied") / "provenance.env"
        ).read_text(encoding="utf-8")
        self.assertIn("gpu_busy_detected=true", denied_provenance)

        allowed = self._invoke(
            "smoke-1",
            "busy-allowed",
            "--allow-busy-gpu",
            environment=busy_environment,
        )
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        allowed_provenance = (
            self._run_dir("smoke-1", "busy-allowed") / "provenance.env"
        ).read_text(encoding="utf-8")
        self.assertIn("allow_busy_gpu=1", allowed_provenance)
        self.assertIn("gpu_busy_detected=true", allowed_provenance)
        self.assertIn("lpips_alexnet_cache_verified=true", allowed_provenance)
        self.assertIn("training_max_jobs=4", allowed_provenance)
        self.assertIn(
            f"lpips_alexnet_actual_sha256={LPIPS_ALEXNET_SHA256}",
            allowed_provenance,
        )
        allowed_run = self._run_dir("smoke-1", "busy-allowed")
        self.assertTrue((allowed_run / "training-artifacts.sha256").is_file())
        self.assertEqual(
            (allowed_run / "method-plugin-audit.json").read_bytes(),
            (allowed_run / "training-method-plugin-audit.json").read_bytes(),
        )
        self.assertIn("training_method_plugin_audit_path=", allowed_provenance)
        training_calls = [
            line
            for line in self.docker_log.read_text(encoding="utf-8").splitlines()
            if (line == "run" or line.startswith("run ")) and "--gpus" in line
        ]
        self.assertEqual(len(training_calls), 1)
        self.assertIn(
            f"src={self.lpips_checkpoint.resolve()},dst=/cache/{LPIPS_ALEXNET_RELATIVE_PATH},readonly",
            training_calls[0],
        )
        self.assertTrue(
            (
                allowed_run
                / "training-output"
                / "lemniscate"
                / "splatfacto"
                / "busy-allowed"
                / "nerfstudio_models"
                / "step-000000000.ckpt"
            ).is_file()
        )

    def test_zero_exit_without_final_checkpoint_fails_closed(self) -> None:
        environment = self.environment.copy()
        environment["MOCK_NO_ARTIFACTS"] = "1"
        result = self._invoke("smoke-1", "missing-artifacts", environment=environment)
        self.assertEqual(result.returncode, 2)
        self.assertIn("without a non-empty config.yml", result.stderr)
        status = (
            self._run_dir("smoke-1", "missing-artifacts") / "status.env"
        ).read_text(encoding="utf-8")
        self.assertIn("result=failed", status)

    def test_missing_pinned_image_fails_without_attempting_a_pull_or_run(self) -> None:
        environment = self.environment.copy()
        environment["MOCK_IMAGE_MISSING"] = "1"
        result = self._invoke("preflight", "image-missing", environment=environment)
        self.assertEqual(result.returncode, 2)
        self.assertIn("will not pull", result.stderr)
        calls = self.docker_log.read_text(encoding="utf-8").splitlines()
        self.assertFalse(any(line.startswith("pull ") for line in calls))
        self.assertFalse(any(line == "run" or line.startswith("run ") for line in calls))

    def test_cpu_preflight_failure_blocks_gpu_checks_and_training(self) -> None:
        environment = self.environment.copy()
        environment["MOCK_PREFLIGHT_FAIL"] = "1"
        result = self._invoke("smoke-1", "preflight-failed", environment=environment)
        self.assertEqual(result.returncode, 2)
        self.assertIn("CPU preflight failed", result.stderr)
        self.assertFalse(
            (self._run_dir("smoke-1", "preflight-failed") / "nvidia-smi.before.txt").exists()
        )
        runs = [
            line
            for line in self.docker_log.read_text(encoding="utf-8").splitlines()
            if line == "run" or line.startswith("run ")
        ]
        self.assertEqual(len(runs), 1)
        self.assertIn("python3.10 -c", runs[0])
        self.assertNotIn("--gpus", runs[0])

    def test_overlapping_mount_roots_are_rejected_before_docker(self) -> None:
        nested_output = self.data / "unsafe-output"
        command = [
            "bash",
            str(SCRIPT),
            "preflight",
            "--data",
            str(self.data),
            "--output-root",
            str(nested_output),
            "--cache",
            str(self.cache),
            "--run-id",
            "overlap",
            "--track",
            "lemniscate",
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=self.environment,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must not overlap", result.stderr)
        self.assertFalse(self.docker_log.exists())
        self.assertFalse(nested_output.exists())


if __name__ == "__main__":
    unittest.main()
