"""Numerical safety: NaN/Inf guards, gradient norms, crash dumps (S10).

A PINN can diverge silently --- a single NaN in the loss propagates through
the optimizer and corrupts the weights without raising. The helpers in this
module catch that early.

The trainer wraps each step with :func:`check_finite_loss` and a sanity-bounds
check; on divergence it raises :class:`TrainingDiverged`, which the outer
training loop catches to write ``summary.json`` with ``status="diverged"`` and
a ``crash_dump.pt`` file containing the last good state.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


class TrainingDiverged(RuntimeError):
    """Raised when the trainer detects a non-finite loss or output."""


def check_finite_loss(loss: torch.Tensor, *, context: str = "loss") -> None:
    """Raise :class:`TrainingDiverged` if ``loss`` is NaN or Inf."""
    if not torch.isfinite(loss).all():
        raise TrainingDiverged(f"{context} is not finite: {loss.item()!r}")


def check_sanity_bounds(outputs: dict[str, torch.Tensor]) -> None:
    """Ensure ``h > 0`` and all output fields are finite.

    The :class:`~pinn_bath.models.base.BaseModel` enforces ``h > 0`` via
    softplus, but this guards against silent breakage if a subclass overrides
    that behavior.
    """
    h = outputs.get("h")
    if h is not None and not (h > 0).all():
        raise TrainingDiverged(f"depth h has non-positive values: min={float(h.min()):.4e}")
    for name, v in outputs.items():
        if not torch.isfinite(v).all():
            raise TrainingDiverged(f"field {name!r} has non-finite values")


def gradient_norm(model: torch.nn.Module) -> float:
    """L2 norm of all gradients of trainable parameters."""
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_sq += float(p.grad.detach().pow(2).sum())
    return math.sqrt(total_sq)


def gradient_norm_per_term(
    losses: dict[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, float]:
    """Per-term gradient norm (S12 diagnostic).

    For each ``(name, scalar)`` in ``losses``, compute ``||grad(scalar)||`` by
    a fresh backward (``retain_graph=True``). The model's gradients are zeroed
    between calls so each result is the contribution of that loss alone.
    """
    out: dict[str, float] = {}
    params = [p for p in model.parameters() if p.requires_grad]
    for name, loss in losses.items():
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward(retain_graph=True)
        out[name] = gradient_norm(model)
    return out


def dump_crash_state(
    path: Path | str,
    *,
    model: torch.nn.Module,
    epoch: int,
    phase: str,
    last_loss: float | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist the last-known training state when a divergence is detected."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "phase": str(phase),
        "last_loss": float(last_loss) if last_loss is not None else None,
        "model": model.state_dict(),
        "extra": dict(extra or {}),
    }
    torch.save(payload, p)
