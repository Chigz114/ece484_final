"""Small, explicit helpers for the isolated WSL rendering toolchain."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_environment_path(name: str, entries: list[Path]) -> None:
    current = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    additions = [str(path) for path in entries if path.exists()]
    os.environ[name] = os.pathsep.join(
        additions + [part for part in current if part not in additions]
    )


def configure_wsl_cuda_toolchain(
    cuda_home: Path,
    *,
    architecture: str = "8.9",
    maximum_jobs: int = 1,
) -> dict[str, str]:
    """Configure gsplat's first-use JIT build without mutating a base env.

    ``gsplat==1.0.0`` discovers ``nvcc`` through ``PATH`` rather than only
    through ``CUDA_HOME``.  A single compile worker is intentional: parallel
    nvcc jobs exceeded the memory available to this WSL instance during the
    verified Circle build.
    """

    cuda_home = Path(cuda_home).expanduser().resolve()
    nvcc = cuda_home / "bin" / "nvcc"
    runtime_header = cuda_home / "include" / "cuda_runtime.h"
    if not nvcc.is_file() or not runtime_header.is_file():
        raise FileNotFoundError(
            f"incomplete CUDA toolkit at {cuda_home}; expected {nvcc} and "
            f"{runtime_header}"
        )
    if maximum_jobs <= 0:
        raise ValueError("maximum_jobs must be positive")

    # Keep the venv/overlay path itself.  Resolving the Python symlink would
    # jump into the inherited base environment and hide overlay-local ninja.
    python_bin = Path(sys.executable).parent
    _prepend_environment_path("PATH", [python_bin, cuda_home / "bin"])
    _prepend_environment_path(
        "LD_LIBRARY_PATH", [cuda_home / "lib", cuda_home / "lib64"]
    )
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["TORCH_CUDA_ARCH_LIST"] = architecture
    os.environ["MAX_JOBS"] = str(maximum_jobs)

    compiler_candidates = {
        "CC": cuda_home / "bin" / "x86_64-conda-linux-gnu-gcc",
        "CXX": cuda_home / "bin" / "x86_64-conda-linux-gnu-g++",
    }
    for variable, compiler in compiler_candidates.items():
        if compiler.is_file():
            os.environ[variable] = str(compiler)

    return {
        name: os.environ[name]
        for name in (
            "CUDA_HOME",
            "TORCH_CUDA_ARCH_LIST",
            "MAX_JOBS",
            "CC",
            "CXX",
        )
        if name in os.environ
    }
