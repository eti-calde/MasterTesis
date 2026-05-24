"""Evaluation metrics for inverse bathymetry (S4).

The canonical trio for the §5.1 reporting is

- :func:`rmse`: root mean squared error.
- :func:`nrmse`: RMSE normalized by the range of the true field.
- :func:`r_squared`: coefficient of determination.

:func:`evaluate_zb` runs a model on the full case eval grid and returns the
trio for the bathymetry field. :func:`baseline_rmse_zb` reports the trivial
``z_b \\equiv 0`` baseline so the §5 tables can show absolute improvements.
"""

from __future__ import annotations

import torch

from pinn_bath.data import Case
from pinn_bath.models.base import BaseModel


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Root mean squared error."""
    return float(((pred - true) ** 2).mean().sqrt())


def nrmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """RMSE normalized by the range (max - min) of ``true``."""
    rng = float(true.max() - true.min())
    if rng == 0.0:
        return float("nan")
    return rmse(pred, true) / rng


def r_squared(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Coefficient of determination R² = 1 - SS_res / SS_tot."""
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def evaluate_zb(
    model: BaseModel,
    case: Case,
    *,
    chunk_size: int = 200_000,
) -> dict[str, float]:
    """Compute the RMSE / NRMSE / R² of ``zb`` on the full eval grid.

    For large grids (e.g. Exp 4/5 2D transient where ``Nt * Ny * Nx`` can
    exceed 10⁶), the full forward is split into chunks of ``chunk_size``
    rows to bound peak VRAM. With ``chunk_size = 200000`` the largest 2D
    eval grid stays under ~1 GB activation footprint on an A1/large
    model. Set ``chunk_size = None`` to disable chunking (legacy single
    forward).
    """
    coords_eval, fields_eval = case.eval_grid()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    coords_on_device = {
        axis: t.to(device=device, dtype=dtype).detach() for axis, t in coords_eval.items()
    }
    n_rows = next(iter(coords_on_device.values())).shape[0]
    with torch.no_grad():
        if chunk_size is None or n_rows <= chunk_size:
            out = model(coords_on_device)
            zb_pred = out["zb"]
        else:
            chunks: list[torch.Tensor] = []
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                slice_coords = {axis: t[start:stop] for axis, t in coords_on_device.items()}
                chunks.append(model(slice_coords)["zb"])
            zb_pred = torch.cat(chunks, dim=0)
    zb_pred = zb_pred.cpu()
    zb_true = fields_eval["zb"].cpu()
    if zb_pred.shape != zb_true.shape:
        zb_pred = zb_pred.reshape(zb_true.shape)
    return {
        "rmse_zb": rmse(zb_pred, zb_true),
        "nrmse_zb": nrmse(zb_pred, zb_true),
        "r2_zb": r_squared(zb_pred, zb_true),
    }


def baseline_rmse_zb(case: Case) -> dict[str, float]:
    """Baseline metrics for the trivial predictor ``zb_pred = 0``.

    If A1 small cannot beat this, the method is not learning anything.
    """
    zb_true = torch.as_tensor(case.fields["zb"], dtype=torch.float64)
    zb_zero = torch.zeros_like(zb_true)
    return {
        "rmse_zb_baseline": rmse(zb_zero, zb_true),
        "nrmse_zb_baseline": nrmse(zb_zero, zb_true),
        "r2_zb_baseline": r_squared(zb_zero, zb_true),
    }
