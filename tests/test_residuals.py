"""Tests for the SWE residual implementations (3 forms across 3 case kinds).

Test strategy (S7-2):

1. **AD correctness on synthetic functions.** For each form, evaluate on a
   smooth analytic ``(h, u, zb)`` and compare the AD-computed residual to a
   hand-derived expected expression. Validates both the residual formulas and
   the autograd wiring.

2. **PDE satisfaction on the Bernoulli analytical solution.** For the steady
   1D case (Exp 1) with a smooth Gaussian bump, the three forms should produce
   residuals of magnitude near zero. This validates that "what we compute"
   matches "what the SWE requires".
"""

from __future__ import annotations

import pytest
import torch

from pinn_bath.losses import swe_residual

# --- AD correctness on synthetic functions ----------------------------------


@pytest.mark.fast
@pytest.mark.parametrize("form", ["primitive", "prim_cons", "conservative"])
def test_residual_1d_steady_ad_matches_analytic(form: str) -> None:
    """Synthetic case: h = 2 + sin(x), u = 1, zb = 0.5*x. Compute residuals by AD
    and compare to expressions derived by hand."""
    g = 9.81
    x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    h = 2.0 + torch.sin(x)
    u = torch.ones_like(x)
    zb = 0.5 * x

    out = swe_residual(
        form,
        coords={"x": x},
        fields={"h": h, "u": u, "zb": zb},
        g=g,
        spatial_dim=1,
        has_t=False,
    )

    cos_x = torch.cos(x)
    # Primitive:   r_cont = u*h_x + h*u_x = cos(x);  r_mom = u*u_x + g*(h_x + zb_x) = g*(cos(x) + 0.5)
    if form == "primitive":
        expected_cont = cos_x
        expected_mom = g * (cos_x + 0.5)
    else:
        # prim_cons and conservative are analytically identical for steady 1D
        r_cont = cos_x
        r_mom = g * (cos_x + 0.5)
        expected_cont = r_cont  # row 1 of A is [1, 0]
        expected_mom = u * r_cont + h * r_mom

    torch.testing.assert_close(out["cont"], expected_cont, atol=1e-10, rtol=1e-6)
    torch.testing.assert_close(out["mom_x"], expected_mom, atol=1e-9, rtol=1e-6)


@pytest.mark.fast
@pytest.mark.parametrize("form", ["primitive", "prim_cons", "conservative"])
def test_residual_1d_transient_ad_matches_analytic(form: str) -> None:
    """Synthetic case: h = 1 + 0.1*sin(t), u = x, zb = 0."""
    g = 9.81
    n = 30
    x = torch.linspace(-0.5, 0.5, n, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    h = 1.0 + 0.1 * torch.sin(t)
    u = x.clone()  # u(x) = x; needs to be a tensor of x
    zb = torch.zeros_like(x)

    out = swe_residual(
        form,
        coords={"x": x, "t": t},
        fields={"h": h, "u": u, "zb": zb},
        g=g,
        spatial_dim=1,
        has_t=True,
    )

    # Derivatives:
    # h_t = 0.1*cos(t), h_x = 0, u_x = 1, u_t = 0, zb_x = 0.
    # Primitive: r_cont = h_t + u*h_x + h*u_x = 0.1*cos(t) + h;
    #            r_mom  = u_t + u*u_x + g*(h_x + zb_x) = x + 0 = x.
    expected_cont_prim = 0.1 * torch.cos(t) + h
    expected_mom_prim = u  # = x

    if form == "primitive":
        torch.testing.assert_close(out["cont"], expected_cont_prim, atol=1e-9, rtol=1e-6)
        torch.testing.assert_close(out["mom_x"], expected_mom_prim, atol=1e-9, rtol=1e-6)
    else:
        # prim_cons: cont same; mom = u*cont + h*mom_prim
        expected_mom_pc = u * expected_cont_prim + h * expected_mom_prim
        torch.testing.assert_close(out["cont"], expected_cont_prim, atol=1e-9, rtol=1e-6)
        torch.testing.assert_close(out["mom_x"], expected_mom_pc, atol=1e-9, rtol=1e-6)


@pytest.mark.fast
@pytest.mark.parametrize("form", ["primitive", "prim_cons", "conservative"])
def test_residual_2d_transient_runs(form: str) -> None:
    """Smoke test the 2D transient residual: correct keys and finite values."""
    n = 12
    x = torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    y = torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    h = 2.0 + 0.1 * torch.sin(x + y + t)
    u = 0.3 * torch.cos(x)
    v = 0.2 * torch.sin(y)
    zb = 0.05 * (x + y)

    out = swe_residual(
        form,
        coords={"x": x, "y": y, "t": t},
        fields={"h": h, "u": u, "v": v, "zb": zb},
        g=9.81,
        spatial_dim=2,
        has_t=True,
    )
    assert set(out.keys()) == {"cont", "mom_x", "mom_y"}
    for v_t in out.values():
        assert torch.isfinite(v_t).all()
        assert v_t.shape == (n, 1)


# --- PDE satisfaction on the Bernoulli solution -----------------------------


def _bernoulli_implicit(
    x: torch.Tensor, zb_fn, q: float, h_down: float, g: float = 9.81
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the Bernoulli cubic at each ``x`` using a torch-friendly Newton iteration.

    Returns ``(h, u, zb)``, each shape ``(N, 1)``, with autograd connecting them
    back to ``x`` via the implicit relation
        :math:`q^2 / (2 g h^2) + h + zb = C` .
    """
    zb = zb_fn(x)
    # Bernoulli constant from downstream BC
    C = q * q / (2.0 * g * h_down * h_down) + h_down + zb_fn(x[-1:])
    # Newton iteration on the cubic h^3 + (zb - C) h^2 + q^2/(2g) = 0
    a = zb - C
    b = q * q / (2.0 * g)
    h = torch.full_like(x, h_down)
    for _ in range(60):
        f = h * h * h + a * h * h + b
        fp = 3.0 * h * h + 2.0 * a * h
        h = h - f / fp
    u = q / h
    return h, u, zb


@pytest.mark.fast
@pytest.mark.parametrize("form", ["primitive", "prim_cons", "conservative"])
def test_residual_vanishes_on_bernoulli_solution(form: str) -> None:
    """The three forms should give near-zero residuals on the steady analytical solution."""
    g = 9.81
    q = 4.42
    h_down = 2.0
    # Smooth Gaussian bump so all derivatives are well-defined everywhere.
    A_bump, sigma = 0.2, 1.0

    def zb_fn(xx: torch.Tensor) -> torch.Tensor:
        return A_bump * torch.exp(-(xx**2) / (2.0 * sigma**2))

    x = torch.linspace(-8.0, 8.0, 200, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    h, u, zb = _bernoulli_implicit(x, zb_fn, q=q, h_down=h_down, g=g)

    out = swe_residual(
        form,
        coords={"x": x},
        fields={"h": h, "u": u, "zb": zb},
        g=g,
        spatial_dim=1,
        has_t=False,
    )

    # Use sup-norm; subcritical Froude is moderate so values are O(1) and the
    # residual should be at most Newton tolerance times the local derivatives.
    eps = 1e-6
    assert out["cont"].abs().max().item() < eps, (
        f"{form}: |r_cont| max = {out['cont'].abs().max().item():.2e}"
    )
    assert out["mom_x"].abs().max().item() < eps, (
        f"{form}: |r_mom| max = {out['mom_x'].abs().max().item():.2e}"
    )


@pytest.mark.fast
def test_residual_unknown_form_raises() -> None:
    x = torch.zeros(3, 1, requires_grad=True)
    with pytest.raises(ValueError):
        swe_residual(
            "wrong",
            coords={"x": x},
            fields={"h": torch.ones_like(x), "u": torch.ones_like(x), "zb": torch.zeros_like(x)},
            g=9.81,
            spatial_dim=1,
            has_t=False,
        )


@pytest.mark.fast
def test_residual_unknown_dim_raises() -> None:
    with pytest.raises(NotImplementedError):
        swe_residual(
            "primitive",
            coords={},
            fields={},
            g=9.81,
            spatial_dim=3,
            has_t=False,
        )
