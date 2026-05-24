"""Boundary-condition loss terms for inverse SWE PINN.

Currently provides :func:`periodic_bc_loss` for cases like Tian dT10
(``bc_type="periodic"``). Other BC kinds (``closed``, ``open_dirichlet``,
``real_sensor``) are not enforced explicitly here yet — for those cases the
trainer relies on the PDE residual and data fit at observation points.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from pinn_bath.data import Case
from pinn_bath.models.base import BaseModel

FLOW_FIELDS: tuple[str, ...] = ("h", "u", "v")


def periodic_bc_loss(
    model: BaseModel,
    case: Case,
    *,
    n_bc: int = 200,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    include_zb: bool = True,
    fields: Iterable[str] = FLOW_FIELDS,
) -> torch.Tensor:
    r"""Penalize :math:`f(\text{lo}, *, t) - f(\text{hi}, *, t)` per spatial axis.

    For each spatial axis of ``case`` (``x``, plus ``y`` when 2D), samples
    ``n_bc`` random pairs that share their non-periodic coordinates (the other
    spatial axis and ``t`` if the case is transient) but have the periodic
    axis fixed at its lower and upper domain bounds. The result is the sum of
    per-axis MSEs over the requested fields.

    ``include_zb=True`` adds the bathymetry to the checked fields: A1's
    BathymetryNet has no t-dependence but can still drift in space, so
    enforcing periodicity in ``zb`` matches the underlying Tian dT10 setup
    where ``z(\pm L_x, y) = z(\mp L_x, y)``.

    Parameters
    ----------
    model
        Any :class:`~pinn_bath.models.base.BaseModel`; called with a coords
        dict the same way as during regular training.
    case
        Source of domain bounds and dimensionality.
    n_bc
        Number of boundary pairs per axis (so ``2 * spatial_dim * n_bc``
        forward evaluations in total).
    seed
        RNG seed for the boundary point sampling; same seed gives the same
        points every call (the trainer typically calls this once per step).
    device, dtype
        Match the model's device + dtype.
    include_zb
        Whether to add ``zb`` to the checked fields.
    fields
        Flow fields to check; missing fields in ``model(coords)`` are skipped.
    """
    device = torch.device(device)
    rng = np.random.default_rng(seed)

    spatial: list[str] = ["x"]
    if case.metadata.spatial_dim == 2:
        spatial.append("y")
    has_t = case.metadata.has_t

    fields_check = list(fields)
    if include_zb:
        fields_check.append("zb")

    total = torch.zeros((), dtype=dtype, device=device)

    for axis in spatial:
        lo, hi = case.metadata.domain[axis]
        coords_lo: dict[str, torch.Tensor] = {}
        coords_hi: dict[str, torch.Tensor] = {}
        for other in spatial:
            if other == axis:
                coords_lo[other] = torch.full((n_bc, 1), float(lo), dtype=dtype, device=device)
                coords_hi[other] = torch.full((n_bc, 1), float(hi), dtype=dtype, device=device)
            else:
                o_lo, o_hi = case.metadata.domain[other]
                samp = torch.as_tensor(
                    rng.uniform(o_lo, o_hi, size=n_bc), dtype=dtype, device=device
                ).reshape(-1, 1)
                coords_lo[other] = samp
                coords_hi[other] = samp
        if has_t:
            t_lo, t_hi = case.metadata.domain["t"]
            t_samp = torch.as_tensor(
                rng.uniform(t_lo, t_hi, size=n_bc), dtype=dtype, device=device
            ).reshape(-1, 1)
            coords_lo["t"] = t_samp
            coords_hi["t"] = t_samp

        out_lo = model(coords_lo)
        out_hi = model(coords_hi)
        for f in fields_check:
            if f in out_lo and f in out_hi:
                total = total + ((out_lo[f] - out_hi[f]) ** 2).mean()

    return total


def flat_bed_loss(
    model: BaseModel,
    case: Case,
    *,
    n_pts: int = 200,
    seed: int = 0,
    x_center_key: str = "x_0",
    half_width_key: str = "w",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    r"""Penalize :math:`z_b^2` where the bed is known flat (outside a bump's support).

    For 1D cases with a bump of half-width ``w`` centered at ``x_0`` (both
    read from ``case.metadata.constants``), samples ``n_pts`` points
    uniformly in the two flat sub-regions ``[x_min, x_0 - w]`` and
    ``[x_0 + w, x_max]`` (proportional to their lengths), evaluates the model there and
    returns the mean of :math:`z_b^2`. This enforces the structural prior
    that the bathymetry is identically zero outside the bump support — a
    piece of information that the data + PDE residual loss alone does NOT
    constrain (the original ``loss_bc`` only pinned the two endpoints).

    Requires ``case.metadata.constants`` to contain the keys ``x_center_key``
    and ``half_width_key`` (defaults ``"x_0"`` and ``"w"``). Raises
    ``KeyError`` if absent. Currently only the 1D case is implemented; for
    2D, the analogous flat-bed prior is case-specific and not yet supported.
    """
    if case.metadata.spatial_dim != 1:
        raise NotImplementedError(
            f"flat_bed_loss is only implemented for 1D cases; got spatial_dim={case.metadata.spatial_dim}"
        )

    consts = case.metadata.constants
    if x_center_key not in consts or half_width_key not in consts:
        raise KeyError(
            f"flat_bed_loss requires {x_center_key!r} and {half_width_key!r} "
            f"in case.metadata.constants; available: {sorted(consts)}"
        )
    x_center = float(consts[x_center_key])
    half_width = float(consts[half_width_key])

    x_min, x_max = case.metadata.domain["x"]
    x_min, x_max = float(x_min), float(x_max)
    lo_left, hi_left = x_min, x_center - half_width
    lo_right, hi_right = x_center + half_width, x_max
    len_left = max(hi_left - lo_left, 0.0)
    len_right = max(hi_right - lo_right, 0.0)
    total_len = len_left + len_right
    if total_len <= 0.0:
        # Bump covers the whole domain -> no flat region to supervise.
        return torch.zeros((), dtype=dtype, device=torch.device(device))

    rng = np.random.default_rng(seed)
    # Allocate samples proportionally to sub-region lengths.
    n_left = round(n_pts * (len_left / total_len))
    n_right = n_pts - n_left
    samples: list[np.ndarray] = []
    if n_left > 0 and len_left > 0:
        samples.append(rng.uniform(lo_left, hi_left, size=n_left))
    if n_right > 0 and len_right > 0:
        samples.append(rng.uniform(lo_right, hi_right, size=n_right))
    x_flat_np = np.concatenate(samples) if samples else np.empty(0, dtype=np.float64)
    if x_flat_np.size == 0:
        return torch.zeros((), dtype=dtype, device=torch.device(device))

    x_flat = torch.as_tensor(x_flat_np, dtype=dtype, device=torch.device(device)).reshape(-1, 1)
    out = model({"x": x_flat})
    zb = out["zb"]
    return (zb**2).mean()
