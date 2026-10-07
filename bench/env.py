"""Hardware / software provenance recorded with every run."""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
from importlib import metadata
from typing import Any

from bench.config import REPO_ROOT


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=True, cwd=REPO_ROOT
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpus": []}
    q = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if q:
        for line in q.splitlines():
            name, mem, driver, cap = (x.strip() for x in line.split(","))
            info["gpus"].append({"name": name, "memory_total_mib": int(mem), "compute_cap": cap})
            info["driver_version"] = driver
    header = _run(["nvidia-smi"])
    if header and (m := re.search(r"CUDA Version:\s*([\d.]+)", header)):
        info["cuda_driver_api_version"] = m.group(1)
    try:
        import torch  # optional: GPU host only

        info["torch_cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    return info


def git_info() -> dict[str, Any]:
    """Commit SHA + dirty flag. A cloud container has no .git, so the launcher passes them in."""
    if "BENCH_GIT_SHA" in os.environ:
        dirty = os.environ.get("BENCH_GIT_DIRTY")
        return {
            "sha": os.environ["BENCH_GIT_SHA"] or None,
            "dirty": None if dirty is None else dirty == "1",
            "source": "env",
        }
    sha = _run(["git", "log", "-1", "--format=%H"])
    status = _run(["git", "status", "--porcelain"])
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def collect_env() -> dict[str, Any]:
    return {
        **gpu_info(),
        "versions": {
            pkg: _version(pkg)
            for pkg in ("vllm", "torch", "transformers", "flashinfer-python", "httpx")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "git": git_info(),
    }
