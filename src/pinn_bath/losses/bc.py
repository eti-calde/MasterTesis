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


def wall_bc_loss(
    model: BaseModel,
    case: Case,
    *,
    n_bc: int = 200,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    axes: Iterable[str] | None = None,
) -> torch.Tensor:
    r"""Penalize :math:`u = 0` (and :math:`v = 0` in 2D) at closed walls.

    For each requested spatial axis, samples ``n_bc`` random points along
    each of its two boundaries (low and high) — the other spatial axis
    and ``t`` (if transient) are sampled uniformly inside their domain.
    The flow velocity components are evaluated and penalised toward zero.

    By default ``axes = (\"x\", )`` for 1D and ``(\"x\", \"y\")`` for 2D,
    i.e. all spatial axes have closed walls (Thacker basin). Pass a
    subset to model partially-closed domains.
    """
    device_t = torch.device(device)
    spatial_dim = case.metadata.spatial_dim
    if axes is None:
        axes = ("x",) if spatial_dim == 1 else ("x", "y")
    spatial: list[str] = ["x"] if spatial_dim == 1 else ["x", "y"]
    has_t = case.metadata.has_t

    rng = np.random.default_rng(seed)
    total = torch.zeros((), dtype=dtype, device=device_t)
    fields_check = ("u", "v")

    for axis in axes:
        if axis not in case.metadata.domain:
            raise KeyError(
                f"wall_bc_loss: axis {axis!r} not in case domain {sorted(case.metadata.domain)}"
            )
        lo, hi = case.metadata.domain[axis]
        for boundary_val in (float(lo), float(hi)):
            coords: dict[str, torch.Tensor] = {}
            coords[axis] = torch.full((n_bc, 1), boundary_val, dtype=dtype, device=device_t)
            for other in spatial:
                if other == axis:
                    continue
                o_lo, o_hi = case.metadata.domain[other]
                samp = torch.as_tensor(
                    rng.uniform(o_lo, o_hi, size=n_bc), dtype=dtype, device=device_t
                ).reshape(-1, 1)
                coords[other] = samp
            if has_t:
                t_lo, t_hi = case.metadata.domain["t"]
                t_samp = torch.as_tensor(
                    rng.uniform(t_lo, t_hi, size=n_bc), dtype=dtype, device=device_t
                ).reshape(-1, 1)
                coords["t"] = t_samp
            out = model(coords)
            for f in fields_check:
                if f in out:
                    total = total + (out[f] ** 2).mean()
    return total


def inflow_outflow_1d_loss(
    model: BaseModel,
    case: Case,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    h_downstream_key: str = "h_down",
    q_key: str = "q",
) -> torch.Tensor:
    r"""Combined Exp 1-style BCs for an open 1D channel.

    For a steady 1D case with inflow at ``x_min`` and outflow at
    ``x_max``, enforces three boundary penalties:

    1. ``z_b(x_min) = z_b(x_max) = 0`` (flat bed at the endpoints).
    2. ``h(x_max) = h_downstream`` (known outlet depth, Dirichlet).
    3. ``h*u = q`` at both endpoints (discharge consistency).

    Constants ``h_downstream`` and ``q`` are read from
    ``case.metadata.constants`` under ``h_downstream_key`` (default
    ``"h_down"``) and ``q_key`` (default ``"q"``).
    """
    if case.metadata.spatial_dim != 1:
        raise NotImplementedError(
            f"inflow_outflow_1d_loss only supports 1D cases; "
            f"got spatial_dim={case.metadata.spatial_dim}"
        )
    consts = case.metadata.constants
    if h_downstream_key not in consts or q_key not in consts:
        raise KeyError(
            f"inflow_outflow_1d_loss requires {h_downstream_key!r} and "
            f"{q_key!r} in case.metadata.constants; available: {sorted(consts)}"
        )
    h_down = float(consts[h_downstream_key])
    q_known = float(consts[q_key])

    device_t = torch.device(device)
    x_min, x_max = case.metadata.domain["x"]
    coords_lo: dict[str, torch.Tensor] = {
        "x": torch.full((1, 1), float(x_min), dtype=dtype, device=device_t),
    }
    coords_hi: dict[str, torch.Tensor] = {
        "x": torch.full((1, 1), float(x_max), dtype=dtype, device=device_t),
    }
    out_lo = model(coords_lo)
    out_hi = model(coords_hi)

    zb_lo = out_lo["zb"]
    zb_hi = out_hi["zb"]
    h_hi = out_hi["h"]
    h_lo = out_lo["h"]
    u_hi = out_hi["u"]
    u_lo = out_lo["u"]

    # 1. zb = 0 at both endpoints
    bc_zb = (zb_lo**2).mean() + (zb_hi**2).mean()
    # 2. h(x_max) = h_downstream
    bc_h = ((h_hi - h_down) ** 2).mean()
    # 3. q = h*u at both endpoints
    bc_q = ((h_lo * u_lo - q_known) ** 2).mean() + ((h_hi * u_hi - q_known) ** 2).mean()
    return bc_zb + bc_h + bc_q
