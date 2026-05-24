"""Helpers to build explicit (obs_coords, obs_values) tensor dicts.

Used by sweep studies that need to restrict observations to a subset
of the case's t-grid (Exp 2 / Exp 5 N_t sweeps) instead of letting
``case.sample_observations`` pick uniformly random points.
"""

from __future__ import annotations

import numpy as np
import torch

from pinn_bath.data import Case


def subsample_t_observations(
    case: Case,
    t_indices: list[int] | np.ndarray,
    *,
    fields: tuple[str, ...] = ("eta",),
    dtype: torch.dtype = torch.float64,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return (obs_coords, obs_values) for the selected t-indices.

    Every spatial grid point at each ``t_indices[k]`` snapshot becomes a
    row. Total rows = ``len(t_indices) * Nx * (Ny if 2D)``. Matches the
    legacy snapshot-sweep semantics: full spatial coverage at a subset
    of the time stamps.

    Shapes:
      - 1D transient: (n_rows = n_t * Nx, 1) for each coord/field.
      - 2D transient: (n_rows = n_t * Ny * Nx, 1).
    """
    if not case.metadata.has_t:
        raise ValueError("subsample_t_observations requires a transient case")
    t = np.asarray(case.coords["t"])
    t_idx = np.asarray(t_indices, dtype=int)
    if (t_idx < 0).any() or (t_idx >= t.size).any():
        raise ValueError(f"t_indices out of range [0, {t.size}): got {t_idx}")

    x = np.asarray(case.coords["x"])
    if case.metadata.spatial_dim == 1:
        T_sel = t[t_idx]
        Xg, Tg = np.meshgrid(x, T_sel, indexing="xy")  # (n_t, Nx)
        coords_np = {"x": Xg.reshape(-1), "t": Tg.reshape(-1)}
        n_rows = Xg.size
    elif case.metadata.spatial_dim == 2:
        y = np.asarray(case.coords["y"])
        T_sel = t[t_idx]
        Tg, Yg, Xg = np.meshgrid(T_sel, y, x, indexing="ij")
        coords_np = {"x": Xg.reshape(-1), "y": Yg.reshape(-1), "t": Tg.reshape(-1)}
        n_rows = Xg.size
    else:
        raise NotImplementedError(f"spatial_dim={case.metadata.spatial_dim}")

    obs_coords = {
        axis: torch.as_tensor(arr, dtype=dtype).reshape(-1, 1) for axis, arr in coords_np.items()
    }

    obs_values: dict[str, torch.Tensor] = {}
    for f in fields:
        if f == "eta" and f not in case.fields:
            gt = case.fields["h"] + (
                case.fields["zb"][None, ...] if case.metadata.has_t else case.fields["zb"]
            )
        else:
            gt = case.fields[f]
        gt_sel = gt[t_idx]
        obs_values[f] = torch.as_tensor(gt_sel.reshape(-1), dtype=dtype).reshape(-1, 1)
        if obs_values[f].numel() != n_rows:
            raise ValueError(
                f"field {f!r}: built {obs_values[f].numel()} obs but expected {n_rows}"
            )
    return obs_coords, obs_values


def evenly_spaced_indices(n_total: int, k: int) -> np.ndarray:
    """``k`` evenly-spaced integer indices spanning ``[0, n_total-1]``.

    Matches the legacy ``snapshot_vs_timeseries.py`` / ``n_t_sweep.py``
    convention so the new sweeps select the same t-slices as the
    historical runs (when the underlying ``Nt`` matches).
    """
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")
    if k == 1:
        return np.array([n_total // 4], dtype=int)
    return np.linspace(0, n_total - 1, k).astype(int)
