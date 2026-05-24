"""Individual loss components.

These are tensor-in, scalar-out functions. The composite loss (in
:mod:`pinn_bath.losses.composite`) sums them with weights from
:class:`pinn_bath.config.LossWeights`.

The design is intentionally case-agnostic: each component takes only the
tensors it needs (predictions, observations, masks, coordinates) and knows
nothing about the specific experiment. Case-specific BC/IC terms live in
the trainer where they have access to the :class:`pinn_bath.data.Case`.
"""

from __future__ import annotations

import torch


def data_mse(
    pred: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """MSE of ``pred - obs``, optionally restricted to the indices where ``mask`` is True.

    Both ``pred`` and ``obs`` must be broadcastable to the same shape. ``mask``
    is a boolean tensor matching that shape; if ``None``, every element counts.
    """
    diff = pred - obs
    if mask is None:
        return torch.mean(diff**2)
    selected = diff[mask]
    if selected.numel() == 0:
        return torch.zeros((), dtype=diff.dtype, device=diff.device)
    return torch.mean(selected**2)


def pde_mse(residuals: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean of squared residuals, summed across components (cont, mom_x, [mom_y])."""
    total = torch.zeros(
        (), dtype=next(iter(residuals.values())).dtype, device=next(iter(residuals.values())).device
    )
    for r in residuals.values():
        total = total + torch.mean(r**2)
    return total


def tv_1d(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Total variation regularization for a 1D field sampled at ``x``.

    Both ``values`` and ``x`` are shape ``(N,)`` or ``(N, 1)`` and sorted by
    ``x``. Implements :math:`\\sum_i |v_{i+1} - v_i| / |x_{i+1} - x_i|` divided
    by ``N-1``.
    """
    v = values.reshape(-1)
    xs = x.reshape(-1)
    if v.numel() < 2:
        return torch.zeros((), dtype=v.dtype, device=v.device)
    dv = v[1:] - v[:-1]
    dx = xs[1:] - xs[:-1]
    return torch.mean(torch.abs(dv) / (torch.abs(dx) + 1.0e-12))


def tv_2d(values: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation for a 2D field on a regular grid.

    ``values`` has shape ``(Ny, Nx)``, ``x`` is the unique x-coordinates
    (shape ``(Nx,)``) and ``y`` the unique y-coordinates (shape ``(Ny,)``).
    """
    v = values
    xs = x.reshape(-1)
    ys = y.reshape(-1)
    if v.dim() != 2 or v.shape != (ys.numel(), xs.numel()):
        raise ValueError(
            f"tv_2d expects values (Ny, Nx) matching y ({ys.numel()}), x ({xs.numel()}); got {tuple(v.shape)}"
        )
    dx = (xs[1:] - xs[:-1]).abs().mean() + 1.0e-12
    dy = (ys[1:] - ys[:-1]).abs().mean() + 1.0e-12
    gx = (v[:, 1:] - v[:, :-1]) / dx
    gy = (v[1:, :] - v[:-1, :]) / dy
    # Mean magnitude on overlapping interior cells.
    return torch.mean(torch.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2 + 1.0e-12))


def tikhonov(values: torch.Tensor) -> torch.Tensor:
    """Mean of squared values — encourages zb close to zero."""
    return torch.mean(values**2)


def positivity(h: torch.Tensor) -> torch.Tensor:
    """Penalty for negative depths: mean of :math:`\\max(0, -h)^2`."""
    return torch.mean(torch.relu(-h) ** 2)


def discharge(h: torch.Tensor, u: torch.Tensor, q_target: float) -> torch.Tensor:
    """Discharge constraint :math:`(h u - q)^2`, averaged over points (Exp 1)."""
    return torch.mean((h * u - q_target) ** 2)


def boundary_dirichlet(
    value_at_boundary: torch.Tensor, target: float | torch.Tensor
) -> torch.Tensor:
    """Squared error at boundary nodes against a Dirichlet target."""
    target_tensor = (
        target
        if isinstance(target, torch.Tensor)
        else torch.full_like(value_at_boundary, float(target))
    )
    return torch.mean((value_at_boundary - target_tensor) ** 2)
