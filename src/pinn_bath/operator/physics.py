r"""Finite-difference SWE residual on gridded fields (operator physics term).

The autodiff residual in :mod:`pinn_bath.losses.residual` is coordinate-MLP
only (it differentiates through ``torch.autograd.grad`` on coordinate inputs).
The amortized operator works on gridded ``(t, x)`` fields, so this physics
regularizer uses central finite differences instead. Only ``zb`` carries
gradients (``eta``, ``u`` are observed data); the residual couples the
predicted ``zb`` to the observed dynamics:

.. math::
    r_c &= \partial_t h + \partial_x(h u),
        \qquad h = \eta - z_b \ (\text{$z_b$ time-independent}) \\
    r_m &= \partial_t(h u) + \partial_x(h u^2 + \tfrac{1}{2} g h^2)
        + g\, h\, \partial_x z_b \ (+\ \text{friction})

Continuity is exactly zero at lake-at-rest (``u = 0``); the conservative
momentum term carries an :math:`O(\Delta x^2)` discrete imbalance there
(documented, acceptable for a soft regularizer). ``g`` matches the solver and
``losses/residual`` convention (9.81).
"""

from __future__ import annotations

import torch

DEFAULT_G = 9.81


def _ddx(f: torch.Tensor, dx: float) -> torch.Tensor:
    """Central difference along the last (x) axis → interior ``(..., Nx-2)``."""
    return (f[..., 2:] - f[..., :-2]) / (2.0 * dx)


def _ddt(f: torch.Tensor, dt: float) -> torch.Tensor:
    """Central difference along the time axis ``(B, Nt, Nx)`` → ``(B, Nt-2, Nx)``."""
    return (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dt)


def swe_residual_grid(
    eta: torch.Tensor,
    u: torch.Tensor,
    zb: torch.Tensor,
    dx: float,
    dt: float,
    *,
    g: float = DEFAULT_G,
    dry_tol: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gridded SWE continuity/momentum residuals on the interior.

    Parameters
    ----------
    eta, u : (B, Nt, Nx) observed free-surface and velocity fields.
    zb : (B, Nx) predicted bathymetry (the only tensor that needs grad).
    dx, dt : grid spacings.

    Returns
    -------
    r_c, r_m, wet : each (B, Nt-2, Nx-2). ``wet`` is the boolean mask
        ``h > dry_tol`` over the same interior region.
    """
    h = eta - zb.unsqueeze(1)  # (B, Nt, Nx)
    hu = h * u
    flux_mom = hu * u + 0.5 * g * h * h

    # Continuity.
    h_t = _ddt(h, dt)[..., 1:-1]  # (B, Nt-2, Nx-2)
    hu_x = _ddx(hu, dx)[:, 1:-1, :]  # (B, Nt-2, Nx-2)
    r_c = h_t + hu_x

    # Momentum (conservative) with topography source.
    hu_t = _ddt(hu, dt)[..., 1:-1]
    fmom_x = _ddx(flux_mom, dx)[:, 1:-1, :]
    zb_x = _ddx(zb, dx)  # (B, Nx-2)
    h_int = h[:, 1:-1, 1:-1]
    r_m = hu_t + fmom_x + g * h_int * zb_x.unsqueeze(1)

    wet = h_int > dry_tol
    return r_c, r_m, wet


def physics_loss(
    eta: torch.Tensor,
    u: torch.Tensor,
    zb: torch.Tensor,
    dx: float,
    dt: float,
    *,
    g: float = DEFAULT_G,
    dry_tol: float = 1e-3,
    w_cont: float = 1.0,
    w_mom: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Wet-masked mean-squared SWE residual (continuity + momentum).

    Returns the scalar loss and a detached dict ``{"cont", "mom"}`` for logging.

    Default ``w_mom=0``: the conservative momentum residual is dominated by the
    noisy ``∂x(½ g h²)`` flux term and barely depends on ``zb`` (a flat-vs-true
    ``zb`` changes it only ~2%), so as a soft regularizer it injects mostly
    finite-difference noise and drowns the continuity term's informative signal
    (~14% flat-vs-true). Continuity ``∂t h + ∂x(h u)`` depends on ``zb`` directly
    via ``h = η - zb`` and is the term that carries learnable gradient — so the
    physics regularizer uses it alone by default. Set ``w_mom>0`` to re-include
    momentum.
    """
    r_c, r_m, wet = swe_residual_grid(eta, u, zb, dx, dt, g=g, dry_tol=dry_tol)
    wetf = wet.to(r_c.dtype)
    denom = wetf.sum().clamp_min(1.0)
    lc = (r_c**2 * wetf).sum() / denom
    lm = (r_m**2 * wetf).sum() / denom
    loss = w_cont * lc + w_mom * lm
    return loss, {"cont": float(lc.detach()), "mom": float(lm.detach())}


# --------------------------------------------------------------------------- #
# 2D extension (additive; the 1D functions above are in production).
# --------------------------------------------------------------------------- #
def _ddy(f: torch.Tensor, dy: float) -> torch.Tensor:
    """Central difference along the y axis ``(..., Ny, Nx)`` -> ``(..., Ny-2, Nx)``."""
    return (f[..., 2:, :] - f[..., :-2, :]) / (2.0 * dy)


def swe_residual_grid_2d(
    eta: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    zb: torch.Tensor,
    dx: float,
    dy: float,
    dt: float,
    *,
    g: float = DEFAULT_G,
    dry_tol: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Gridded 2D SWE residuals on the interior, by central differences.

    .. math::
        r_c &= \partial_t h + \partial_x(hu) + \partial_y(hv) \\
        r_{mx} &= \partial_t(hu) + \partial_x(hu^2 + \tfrac{1}{2}gh^2)
            + \partial_y(huv) + g\,h\,\partial_x z_b \\
        r_{my} &= \partial_t(hv) + \partial_x(huv)
            + \partial_y(hv^2 + \tfrac{1}{2}gh^2) + g\,h\,\partial_y z_b

    with ``h = eta - zb`` (``zb`` time-independent, the only tensor that
    needs grad). Continuity is exactly zero at lake-at-rest; the momentum
    fluxes carry the usual :math:`O(\Delta^2)` discrete imbalance there.

    Parameters
    ----------
    eta, u, v : (B, Nt, Ny, Nx) observed fields.
    zb : (B, Ny, Nx) predicted bathymetry.
    dx, dy, dt : grid spacings.

    Returns
    -------
    r_c, r_mx, r_my, wet : each (B, Nt-2, Ny-2, Nx-2); ``wet`` is the
        boolean ``h > dry_tol`` mask over the same interior region.
    """
    h = eta - zb.unsqueeze(1)  # (B, Nt, Ny, Nx)
    hu = h * u
    hv = h * v
    half_gh2 = 0.5 * g * h * h

    def interior_t(f: torch.Tensor) -> torch.Tensor:  # crop y, x after d/dt
        return f[..., 1:-1, 1:-1]

    def interior_x(f: torch.Tensor) -> torch.Tensor:  # crop t, y after d/dx
        return f[:, 1:-1, 1:-1, :]

    def interior_y(f: torch.Tensor) -> torch.Tensor:  # crop t, x after d/dy
        return f[:, 1:-1, :, 1:-1]

    # Continuity.
    r_c = interior_t(_ddt(h, dt)) + interior_x(_ddx(hu, dx)) + interior_y(_ddy(hv, dy))

    # Momentum with topography source (interior of all axes).
    h_int = h[:, 1:-1, 1:-1, 1:-1]
    zb_x = _ddx(zb, dx)[:, 1:-1, :]  # (B, Ny-2, Nx-2)
    zb_y = _ddy(zb, dy)[:, :, 1:-1]
    r_mx = (
        interior_t(_ddt(hu, dt))
        + interior_x(_ddx(hu * u + half_gh2, dx))
        + interior_y(_ddy(hu * v, dy))
        + g * h_int * zb_x.unsqueeze(1)
    )
    r_my = (
        interior_t(_ddt(hv, dt))
        + interior_x(_ddx(hu * v, dx))
        + interior_y(_ddy(hv * v + half_gh2, dy))
        + g * h_int * zb_y.unsqueeze(1)
    )

    wet = h_int > dry_tol
    return r_c, r_mx, r_my, wet


def physics_loss_2d(
    eta: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    zb: torch.Tensor,
    dx: float,
    dy: float,
    dt: float,
    *,
    g: float = DEFAULT_G,
    dry_tol: float = 1e-3,
    w_cont: float = 1.0,
    w_mom: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Wet-masked mean-squared 2D SWE residual.

    Same default as the 1D loss (``w_mom=0``): the momentum residuals are
    dominated by the noisy pressure-flux terms and barely depend on ``zb``,
    so continuity alone carries the learnable signal. ``w_mom`` weights the
    *sum* of both momentum components when enabled.
    """
    r_c, r_mx, r_my, wet = swe_residual_grid_2d(eta, u, v, zb, dx, dy, dt, g=g, dry_tol=dry_tol)
    wetf = wet.to(r_c.dtype)
    denom = wetf.sum().clamp_min(1.0)
    lc = (r_c**2 * wetf).sum() / denom
    lm = ((r_mx**2 + r_my**2) * wetf).sum() / denom
    loss = w_cont * lc + w_mom * lm
    return loss, {"cont": float(lc.detach()), "mom": float(lm.detach())}
