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
    w_mom: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Wet-masked mean-squared SWE residual (continuity + momentum).

    Returns the scalar loss and a detached dict ``{"cont", "mom"}`` for logging.
    """
    r_c, r_m, wet = swe_residual_grid(eta, u, zb, dx, dt, g=g, dry_tol=dry_tol)
    wetf = wet.to(r_c.dtype)
    denom = wetf.sum().clamp_min(1.0)
    lc = (r_c**2 * wetf).sum() / denom
    lm = (r_m**2 * wetf).sum() / denom
    loss = w_cont * lc + w_mom * lm
    return loss, {"cont": float(lc.detach()), "mom": float(lm.detach())}
