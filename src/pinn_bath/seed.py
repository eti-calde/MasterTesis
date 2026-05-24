"""Seed control and deterministic mode (S1).

Used both at the start of a run and for snapshotting/restoring RNG state
across checkpoints (S9).
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Fix random state for Python, NumPy, and PyTorch (CPU + CUDA).

    When ``deterministic`` is True, also enables PyTorch's deterministic
    algorithms and configures cuBLAS for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict[str, Any]:
    """Snapshot RNG state for checkpointing (S9)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(legacy=True),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state from a previous snapshot."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
