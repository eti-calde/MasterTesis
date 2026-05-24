"""SWE residuals in three forms (primitive, primitive-conservative, conservative).

Dispatched by ``(spatial_dim, has_t)`` to one of the case-specific kernels:

- 1D steady (``spatial_dim=1``, ``has_t=False``): used for Exp 1.
- 1D transient (``spatial_dim=1``, ``has_t=True``): Exp 2, Exp 6.
- 2D transient (``spatial_dim=2``, ``has_t=True``): Exp 3, Exp 4, Exp 5.

For the (h, u) parameterization used by the architectures here, the
primitive-conservative form (Tian 2025) and the conservative form yield
analytically identical residuals; both implementations are exposed so the
ablation of §5.4 distinguishes them by their AD trees, not by their math.
"""

from __future__ import annotations

from typing import Literal

import torch

Form = Literal["primitive", "prim_cons", "conservative"]
Friction = Literal["none", "manning", "linear_kappa"]


def _friction_term_1d(
    u: torch.Tensor, h: torch.Tensor, *, model: Friction, g: float, params: dict[str, float]
) -> torch.Tensor:
    """Source term Sf to add to the primitive momentum residual.

    - ``"none"``: returns 0.
    - ``"manning"``: ``Sf = g·n²·u·|u|/h^(4/3)``, with ``n = params["n_manning"]``.
    - ``"linear_kappa"``: ``Sf = κ·u/(h + eps)``, with ``κ = params["kappa"]``
      and ``eps = params.get("eps_dry", 1e-4)`` (Angel et al. 2024 setup).

    The conservative-form momentum source becomes ``g·h·Sf`` automatically
    when applied through the A-matrix weighting.
    """
    if model == "none":
        return torch.zeros_like(u)
    if model == "manning":
        n = float(params["n_manning"])
        return g * (n**2) * u * torch.abs(u) / h ** (4.0 / 3.0)
    if model == "linear_kappa":
        kappa = float(params["kappa"])
        eps = float(params.get("eps_dry", 1.0e-4))
        return kappa * u / (h + eps)
    raise ValueError(f"unknown friction model: {model!r}")


def _grad(out: torch.Tensor, wrt: torch.Tensor) -> torch.Tensor:
    """Reverse-mode derivative ``d out / d wrt``.

    Returns ``zeros_like(wrt)`` when ``out`` is independent of ``wrt`` in the
    autograd graph (or has no grad_fn at all). The analytic derivative of a
    constant is zero, so this is the mathematically correct fallback.
    """
    if not out.requires_grad:
        return torch.zeros_like(wrt)
    (g,) = torch.autograd.grad(
        out,
        wrt,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        allow_unused=True,
    )
    if g is None:
        return torch.zeros_like(wrt)
    return g


def swe_residual(
    form: Form,
    coords: dict[str, torch.Tensor],
    fields: dict[str, torch.Tensor],
    *,
    g: float = 9.81,
    spatial_dim: int,
    has_t: bool,
    friction: Friction = "none",
    friction_params: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the SWE residual in the requested form.

    ``coords`` must contain ``"x"`` (and ``"y"`` if 2D, ``"t"`` if transient),
    each shape ``(N, 1)`` with ``requires_grad=True``. ``fields`` must contain
    ``"h"``, ``"u"``, ``"zb"`` (and ``"v"`` if 2D), each shape ``(N, 1)`` and
    differentiable wrt ``coords``.

    ``friction``: ``"none"`` (default, backward-compat), ``"manning"``, or
    ``"linear_kappa"``. ``friction_params`` is a dict consumed by the friction
    model: ``{"n_manning": 0.025}`` or ``{"kappa": 0.2, "eps_dry": 1e-4}``.

    Returns a dict with key ``"cont"`` (continuity residual) and ``"mom_x"``
    (x-momentum); for 2D the result also contains ``"mom_y"``.
    """
    params = friction_params or {}
    if spatial_dim == 1 and not has_t:
        return _residual_1d_steady(form, coords, fields, g, friction, params)
    if spatial_dim == 1 and has_t:
        return _residual_1d_transient(form, coords, fields, g, friction, params)
    if spatial_dim == 2 and has_t:
        return _residual_2d_transient(form, coords, fields, g, friction, params)
    raise NotImplementedError(
        f"residual not implemented for spatial_dim={spatial_dim}, has_t={has_t}"
    )


# --- 1D steady ----------------------------------------------------------------


def _residual_1d_steady(
    form: Form,
    coords: dict[str, torch.Tensor],
    fields: dict[str, torch.Tensor],
    g: float,
    friction: Friction,
    friction_params: dict[str, float],
) -> dict[str, torch.Tensor]:
    x = coords["x"]
    h, u, zb = fields["h"], fields["u"], fields["zb"]
    Sf_prim = _friction_term_1d(u, h, model=friction, g=g, params=friction_params)

    if form == "primitive":
        h_x = _grad(h, x)
        u_x = _grad(u, x)
        zb_x = _grad(zb, x)
        return {
            "cont": u * h_x + h * u_x,
            "mom_x": u * u_x + g * (h_x + zb_x) + Sf_prim,
        }
    if form == "prim_cons":
        h_x = _grad(h, x)
        u_x = _grad(u, x)
        zb_x = _grad(zb, x)
        r_cont_prim = u * h_x + h * u_x
        r_mom_prim = u * u_x + g * (h_x + zb_x) + Sf_prim
        return {
            "cont": r_cont_prim,
            "mom_x": u * r_cont_prim + h * r_mom_prim,
        }
    if form == "conservative":
        hu = h * u
        flux = h * u * u + 0.5 * g * h * h
        zb_x = _grad(zb, x)
        return {
            "cont": _grad(hu, x),
            "mom_x": _grad(flux, x) + g * h * zb_x + h * Sf_prim,
        }
    raise ValueError(f"unknown residual form: {form!r}")


# --- 1D transient -------------------------------------------------------------


def _residual_1d_transient(
    form: Form,
    coords: dict[str, torch.Tensor],
    fields: dict[str, torch.Tensor],
    g: float,
    friction: Friction,
    friction_params: dict[str, float],
) -> dict[str, torch.Tensor]:
    x, t = coords["x"], coords["t"]
    h, u, zb = fields["h"], fields["u"], fields["zb"]
    Sf_prim = _friction_term_1d(u, h, model=friction, g=g, params=friction_params)

    if form == "primitive":
        h_x = _grad(h, x)
        h_t = _grad(h, t)
        u_x = _grad(u, x)
        u_t = _grad(u, t)
        zb_x = _grad(zb, x)
        return {
            "cont": h_t + u * h_x + h * u_x,
            "mom_x": u_t + u * u_x + g * (h_x + zb_x) + Sf_prim,
        }
    if form == "prim_cons":
        h_x = _grad(h, x)
        h_t = _grad(h, t)
        u_x = _grad(u, x)
        u_t = _grad(u, t)
        zb_x = _grad(zb, x)
        r_cont_prim = h_t + u * h_x + h * u_x
        r_mom_prim = u_t + u * u_x + g * (h_x + zb_x) + Sf_prim
        return {
            "cont": r_cont_prim,
            "mom_x": u * r_cont_prim + h * r_mom_prim,
        }
    if form == "conservative":
        hu = h * u
        flux = h * u * u + 0.5 * g * h * h
        zb_x = _grad(zb, x)
        return {
            "cont": _grad(h, t) + _grad(hu, x),
            "mom_x": _grad(hu, t) + _grad(flux, x) + g * h * zb_x + h * Sf_prim,
        }
    raise ValueError(f"unknown residual form: {form!r}")


# --- 2D transient -------------------------------------------------------------


def _residual_2d_transient(
    form: Form,
    coords: dict[str, torch.Tensor],
    fields: dict[str, torch.Tensor],
    g: float,
    friction: Friction,
    friction_params: dict[str, float],
) -> dict[str, torch.Tensor]:
    x, y, t = coords["x"], coords["y"], coords["t"]
    h, u, v, zb = fields["h"], fields["u"], fields["v"], fields["zb"]
    # Friction per-component (Manning uses speed magnitude in 2D; for
    # simplicity we apply the 1D form per-component, which matches each
    # legacy experiment's convention. Refine if a 2D drag test demands it.)
    Sf_u = _friction_term_1d(u, h, model=friction, g=g, params=friction_params)
    Sf_v = _friction_term_1d(v, h, model=friction, g=g, params=friction_params)

    if form == "primitive":
        h_x, h_y, h_t = _grad(h, x), _grad(h, y), _grad(h, t)
        u_x, u_y, u_t = _grad(u, x), _grad(u, y), _grad(u, t)
        v_x, v_y, v_t = _grad(v, x), _grad(v, y), _grad(v, t)
        zb_x, zb_y = _grad(zb, x), _grad(zb, y)
        return {
            "cont": h_t + u * h_x + h * u_x + v * h_y + h * v_y,
            "mom_x": u_t + u * u_x + v * u_y + g * (h_x + zb_x) + Sf_u,
            "mom_y": v_t + u * v_x + v * v_y + g * (h_y + zb_y) + Sf_v,
        }
    if form == "prim_cons":
        h_x, h_y, h_t = _grad(h, x), _grad(h, y), _grad(h, t)
        u_x, u_y, u_t = _grad(u, x), _grad(u, y), _grad(u, t)
        v_x, v_y, v_t = _grad(v, x), _grad(v, y), _grad(v, t)
        zb_x, zb_y = _grad(zb, x), _grad(zb, y)
        r_cont_prim = h_t + u * h_x + h * u_x + v * h_y + h * v_y
        r_mom_x_prim = u_t + u * u_x + v * u_y + g * (h_x + zb_x) + Sf_u
        r_mom_y_prim = v_t + u * v_x + v * v_y + g * (h_y + zb_y) + Sf_v
        # A_2D = [[1, 0, 0], [u, h, 0], [v, 0, h]] -> momentum weighted by h, mass coupled.
        return {
            "cont": r_cont_prim,
            "mom_x": u * r_cont_prim + h * r_mom_x_prim,
            "mom_y": v * r_cont_prim + h * r_mom_y_prim,
        }
    if form == "conservative":
        hu = h * u
        hv = h * v
        flux_xx = h * u * u + 0.5 * g * h * h
        flux_xy = h * u * v
        flux_yy = h * v * v + 0.5 * g * h * h
        zb_x = _grad(zb, x)
        zb_y = _grad(zb, y)
        return {
            "cont": _grad(h, t) + _grad(hu, x) + _grad(hv, y),
            "mom_x": _grad(hu, t) + _grad(flux_xx, x) + _grad(flux_xy, y) + g * h * zb_x + h * Sf_u,
            "mom_y": _grad(hv, t) + _grad(flux_xy, x) + _grad(flux_yy, y) + g * h * zb_y + h * Sf_v,
        }
    raise ValueError(f"unknown residual form: {form!r}")
