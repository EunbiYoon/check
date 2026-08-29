"""GPU / bitsandbytes env — must run before bitsandbytes is first imported."""

from __future__ import annotations

import os
from pathlib import Path

# UMass NVHPC fallback when `module load cuda/12.6` did not export CUDA_HOME
_NVHPC_ROOT = Path("/modules/opt/linux-ubuntu24.04-x86_64/nvhpc/Linux_x86_64/24.9")
_NVHPC_MATH_LIB64 = _NVHPC_ROOT / "math_libs" / "lib64"
_NVHPC_CUDA_LIB64 = _NVHPC_ROOT / "cuda" / "12.6" / "lib64"


def _nvhpc_math_lib64(cuda_home: Path) -> Path | None:
    # .../nvhpc/.../24.9/cuda/12.6 -> .../24.9/math_libs/lib64 (libcublas.so.12)
    math = cuda_home.parent.parent / "math_libs" / "lib64"
    return math if math.is_dir() else None


def _collect_ld_paths() -> list[str]:
    paths: list[str] = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        ch = Path(cuda_home)
        math = _nvhpc_math_lib64(ch)
        if math is not None:
            paths.append(str(math))
        cuda_lib = ch / "lib64"
        if cuda_lib.is_dir():
            paths.append(str(cuda_lib))
    elif _NVHPC_MATH_LIB64.is_dir():
        paths.append(str(_NVHPC_MATH_LIB64))
        if _NVHPC_CUDA_LIB64.is_dir():
            paths.append(str(_NVHPC_CUDA_LIB64))
    return paths


def configure_bitsandbytes_cuda(version: str = "126") -> None:
    """Match cluster cuda/12.6 when torch reports cu13+ (no bnb cuda130 wheel)."""
    os.environ["BNB_CUDA_VERSION"] = os.environ.get("BNB_CUDA_VERSION") or version

    paths = _collect_ld_paths()
    if not paths:
        return

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(paths)
    if not any(p in ld.split(":") for p in paths):
        os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{ld}" if ld else prefix


if os.environ.get("SAL_DISABLE_BNB_CONFIG") != "1":
    configure_bitsandbytes_cuda()
