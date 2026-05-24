"""Atomic checkpointing with auto-resume and signal handling (S9).

The :class:`CheckpointManager` packs the full training state (model weights,
optimizer state, RNG state from :mod:`pinn_bath.seed`, epoch, phase) into
``.pt`` files written atomically under ``run_dir/checkpoints/``.

Files written:

- ``last.pt`` --- most recent snapshot. Used by :meth:`load_resume` to pick up
  an interrupted run automatically.
- ``best.pt`` --- the snapshot with the lowest tracked metric so far.
- ``epoch_{N:08d}.pt`` --- periodic snapshots; rotated so only the most
  recent ``keep_last_k`` are kept on disk.

All writes use the ``tmp -> rename`` idiom: a partial crash leaves the
previous checkpoint intact.

The trainer registers signal handlers (SIGINT, SIGTERM) so a Ctrl-C or
external kill flushes the latest checkpoint before exit.
"""

from __future__ import annotations

import secrets
import signal
from pathlib import Path
from types import FrameType
from typing import Any

import torch

from pinn_bath.seed import get_rng_state, set_rng_state


def build_state(
    *,
    epoch: int,
    phase: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Snapshot model + optimizer + RNG state into a single picklable dict."""
    state: dict[str, Any] = {
        "epoch": int(epoch),
        "phase": str(phase),
        "model": model.state_dict(),
        "rng": get_rng_state(),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
        state["optimizer_class"] = optimizer.__class__.__name__
    return state


def restore_state(
    state: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> None:
    """Load model weights, optimizer state, and RNG state from a snapshot."""
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng and "rng" in state:
        set_rng_state(state["rng"])


class CheckpointManager:
    """Periodic, atomic checkpoint snapshots with rotation and resume."""

    LAST_NAME = "last.pt"
    BEST_NAME = "best.pt"

    def __init__(
        self,
        run_dir: Path | str,
        *,
        keep_last_k: int = 3,
        keep_best: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = max(int(keep_last_k), 0)
        self.keep_best = bool(keep_best)
        self._best_metric: float | None = None

    # --- Save / load ----------------------------------------------------

    def save(
        self,
        state: dict[str, Any],
        *,
        metric: float | None = None,
    ) -> Path:
        """Atomically write ``state`` as ``last.pt`` plus an epoch snapshot.

        If ``metric`` improves over the previously saved best, also overwrites
        ``best.pt``. Returns the path to the epoch snapshot.
        """
        epoch = int(state["epoch"])
        # Periodic snapshot
        epoch_path = self.ckpt_dir / f"epoch_{epoch:08d}.pt"
        _atomic_torch_save(epoch_path, state)
        # Last (always)
        _atomic_torch_save(self.ckpt_dir / self.LAST_NAME, state)
        # Best (if applicable)
        if (
            self.keep_best
            and metric is not None
            and (self._best_metric is None or metric < self._best_metric)
        ):
            self._best_metric = float(metric)
            _atomic_torch_save(self.ckpt_dir / self.BEST_NAME, state)
        self._rotate()
        return epoch_path

    def load(self, name: str = LAST_NAME) -> dict[str, Any] | None:
        path = self.ckpt_dir / name
        if not path.exists():
            return None
        return torch.load(path, weights_only=False, map_location="cpu")

    def load_resume(self) -> dict[str, Any] | None:
        """Return the latest checkpoint state, or ``None`` if no resume exists."""
        return self.load(self.LAST_NAME)

    def has_resume_point(self) -> bool:
        return (self.ckpt_dir / self.LAST_NAME).exists()

    # --- Rotation -------------------------------------------------------

    def _rotate(self) -> None:
        epoch_ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"), reverse=True)
        for old in epoch_ckpts[self.keep_last_k :]:
            try:
                old.unlink()
            except FileNotFoundError:
                pass


def _atomic_torch_save(path: Path, state: dict[str, Any]) -> None:
    """``torch.save(state, path)`` via a temp file in the same directory."""
    tmp = path.parent / f".{path.stem}.{secrets.token_hex(4)}{path.suffix}"
    try:
        torch.save(state, tmp)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


class SignalCheckpoint:
    """Context manager that installs SIGINT/SIGTERM handlers (S9).

    On signal, the handler sets ``interrupted = True``. The trainer polls
    that flag each step and persists a final checkpoint before exiting
    cleanly.
    """

    def __init__(self, signals: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)) -> None:
        self.signals = signals
        self.interrupted: bool = False
        self._previous: dict[int, Any] = {}

    def _handler(self, signum: int, frame: FrameType | None) -> None:
        self.interrupted = True

    def __enter__(self) -> SignalCheckpoint:
        for sig in self.signals:
            try:
                self._previous[sig] = signal.signal(sig, self._handler)
            except (ValueError, OSError):
                # signal() must be called from the main thread of the main
                # interpreter; skip silently when used from a worker.
                pass
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for sig, prev in self._previous.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._previous.clear()
