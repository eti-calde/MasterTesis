"""Initial-condition loss for transient cases.

For Exps 2, 3, 5 the case ships ground-truth fields at ``t=0`` (the
initial water depth, velocity, etc.). This module evaluates the model
at the spatial grid with ``t = t_min`` and penalises mismatch against
those fields. Used for `bc_type` values that imply known IC (closed
basin Thacker, IC-anchored 2D).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from pinn_bath.data import Case
from pinn_bath.models.base import BaseModel

DEFAULT_IC_FIELDS: tuple[str, ...] = ("h", "u", "v")


def initial_condition_loss(
    model: BaseModel,
    case: Case,
    *,
    fields: Iterable[str] = DEFAULT_IC_FIELDS,
    n_pts: int | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mean squared error at ``t = t_min`` between model and ground truth.

    Evaluates the model on the spatial collocation grid (full grid if
    ``n_pts`` is None, otherwise a random subset of size ``n_pts``) at
    ``t = case.coords['t'][0]`` and compares to the ``t=0`` slice of
    ``case.fields[field]`` for each ``field`` in ``fields``. Fields not
    present in the case (e.g. ``v`` for 1D) or not produced by the model
    are silently skipped.

    Raises ``ValueError`` if the case is steady (``has_t = False``) — the
    caller should not invoke IC loss on a steady-state case.
    """
    if not case.metadata.has_t:
        raise ValueError(
            f"initial_condition_loss requires a transient case; "
            f"{case.metadata.case_id!r} has has_t=False"
        )
    device_t = torch.device(device)
    spatial_dim = case.metadata.spatial_dim
    fields_check = list(fields)

    # Build spatial grid at t=t_min.
    x = np.asarray(case.coords["x"])
    if spatial_dim == 1:
        X = x.reshape(-1)
        Y_flat: np.ndarray | None = None
    elif spatial_dim == 2:
        y = np.asarray(case.coords["y"])
        Xg, Yg = np.meshgrid(x, y)
        X = Xg.reshape(-1)
        Y_flat = Yg.reshape(-1)
    else:
        raise NotImplementedError(f"spatial_dim={spatial_dim} not supported")

    t0 = float(case.coords["t"][0])
    N = X.size
    if n_pts is not None and n_pts < N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_pts, replace=False)
        X = X[idx]
        if Y_flat is not None:
            Y_flat = Y_flat[idx]
    else:
        idx = np.arange(N)

    coords: dict[str, torch.Tensor] = {
        "x": torch.as_tensor(X, dtype=dtype, device=device_t).reshape(-1, 1),
        "t": torch.full((X.size, 1), t0, dtype=dtype, device=device_t),
    }
    if Y_flat is not None:
        coords["y"] = torch.as_tensor(Y_flat, dtype=dtype, device=device_t).reshape(-1, 1)

    out = model(coords)

    total = torch.zeros((), dtype=dtype, device=device_t)
    for f in fields_check:
        if f not in case.fields or f not in out:
            continue
        gt = case.fields[f]
        if f == "zb" and case.metadata.has_t:
            # zb is time-independent in the schema; broadcast.
            gt_slice = gt
        else:
            gt_slice = gt[0]  # t=0 slice
        gt_flat = np.asarray(gt_slice).reshape(-1)[idx]
        gt_t = torch.as_tensor(gt_flat, dtype=dtype, device=device_t).reshape(-1, 1)
        total = total + torch.mean((out[f] - gt_t) ** 2)
    return total
