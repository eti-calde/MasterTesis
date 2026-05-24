"""Capture the runtime environment to an ``env.json`` manifest (S1, S11)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class EnvManifest:
    """Snapshot of versions, hardware, and git state at run time."""

    python: str
    torch: str
    torch_cuda: str | None
    cudnn: int | None
    numpy: str
    platform: str
    machine: str
    gpu_name: str | None
    gpu_vram_mb: int | None
    gpu_count: int
    nvidia_driver: str | None
    git_commit: str | None
    git_dirty: bool

    def write(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, timeout=5)
        return out.decode().strip()
    except Exception:
        return None


def _git_state(cwd: Path) -> tuple[str | None, bool]:
    sha = _run_cmd(["git", "rev-parse", "HEAD"], cwd=cwd)
    if sha is None:
        return None, False
    diff = _run_cmd(["git", "status", "--porcelain"], cwd=cwd)
    return sha, bool(diff)


def _nvidia_driver() -> str | None:
    out = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.splitlines()[0] if out else None


def capture(repo_root: Path | str | None = None) -> EnvManifest:
    """Build an :class:`EnvManifest` from the current runtime."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    sha, dirty = _git_state(root)

    gpu_name: str | None = None
    gpu_vram_mb: int | None = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

    return EnvManifest(
        python=sys.version.split()[0],
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        numpy=np.__version__,
        platform=platform.platform(),
        machine=platform.node(),
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        gpu_count=torch.cuda.device_count(),
        nvidia_driver=_nvidia_driver(),
        git_commit=sha,
        git_dirty=dirty,
    )
