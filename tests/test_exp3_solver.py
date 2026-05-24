"""Tests for the Exp 3 FV-HLL solver with Audusse hydrostatic reconstruction.

The key property we verify here is **well-balancedness**: a lake at rest
($u = v = 0$, $\\eta = h + z_b$ uniform) is preserved to machine precision
even across the sharp cylinder walls of the Ruppenthal §7.2 setup. The pre-fix
solver used central differences on a discontinuous $z_b$, generating spurious
momentum sources of order $5\\,\\mathrm{m/s}^2$ at the cylinder edges; with
Audusse HR those should now vanish.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Experiments/ is sibling to tests/ — make the solver importable.
_EXP3_DIR = Path(__file__).resolve().parents[1] / "Experiments" / "03-two-cylinders-2d"
sys.path.insert(0, str(_EXP3_DIR))

import ground_truth as gt  # noqa: E402, I001


# --- Helpers ---------------------------------------------------------------


def _ruppenthal_grid(Nx: int = 50, Ny: int = 50, Lx: float = 25.0, Ly: float = 25.0):
    """Cell-centered grid + sharp two-cylinder bathymetry (Ruppenthal §7.2)."""
    x = np.linspace(0.5 * Lx / Nx, Lx - 0.5 * Lx / Nx, Nx)
    y = np.linspace(0.5 * Ly / Ny, Ly - 0.5 * Ly / Ny, Ny)
    X, Y = np.meshgrid(x, y)
    dx = Lx / Nx
    dy = Ly / Ny
    zb = gt.bathymetry_two_cylinders(X, Y, smooth=0.0)  # sharp indicator
    return x, y, dx, dy, zb


def _one_step(
    h,
    hu,
    hv,
    zb,
    dx,
    dy,
    g: float = 9.81,
    cfl: float = 0.3,
    *,
    eta_inflow=None,
    u_inflow=None,
    v_inflow=None,
):
    """Run a single FV step with the same logic as run_fv_simulation.

    BC kwargs default to None (Neumann everywhere); pass them to enable
    Dirichlet inflow at small-x and small-y faces.
    """
    eps = 1e-8
    u = hu / (h + eps)
    v = hv / (h + eps)
    c = np.sqrt(g * np.maximum(h, 0.0))
    max_speed = max(np.max(np.abs(u) + c), np.max(np.abs(v) + c), 1e-6)
    dt = cfl * min(dx, dy) / max_speed

    h_padx, hu_padx, hv_padx, zb_padx = gt._pad_x(
        h,
        hu,
        hv,
        zb,
        eta_inflow=eta_inflow,
        u_inflow=u_inflow,
        v_inflow=v_inflow,
    )
    h_pady, hu_pady, hv_pady, zb_pady = gt._pad_y(
        h,
        hu,
        hv,
        zb,
        eta_inflow=eta_inflow,
        u_inflow=u_inflow,
        v_inflow=v_inflow,
    )
    F_h, F_L_hu, F_R_hu, F_hv = gt._audusse_x(h_padx, hu_padx, hv_padx, zb_padx, g)
    G_h, G_hu, G_B_hv, G_T_hv = gt._audusse_y(h_pady, hu_pady, hv_pady, zb_pady, g)

    h_new = h - dt / dx * (F_h[:, 1:] - F_h[:, :-1]) - dt / dy * (G_h[1:, :] - G_h[:-1, :])
    hu_new = (
        hu - dt / dx * (F_L_hu[:, 1:] - F_R_hu[:, :-1]) - dt / dy * (G_hu[1:, :] - G_hu[:-1, :])
    )
    hv_new = (
        hv - dt / dx * (F_hv[:, 1:] - F_hv[:, :-1]) - dt / dy * (G_B_hv[1:, :] - G_T_hv[:-1, :])
    )
    h_new = np.maximum(h_new, 0.0)
    return h_new, hu_new, hv_new, dt


# --- Tests -----------------------------------------------------------------


@pytest.mark.fast
def test_lake_at_rest_one_step() -> None:
    """Single step over Ruppenthal's sharp cylinders preserves rest state."""
    _, _, dx, dy, zb = _ruppenthal_grid()
    eta0 = 2.0  # uniform free surface; h = eta0 - zb varies cell to cell
    h = eta0 - zb
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)

    h_new, hu_new, hv_new, _ = _one_step(h, hu, hv, zb, dx, dy)

    np.testing.assert_allclose(h_new, h, atol=1e-12)
    assert np.abs(hu_new).max() < 1e-12, f"max |hu| spurious = {np.abs(hu_new).max():.3e}"
    assert np.abs(hv_new).max() < 1e-12, f"max |hv| spurious = {np.abs(hv_new).max():.3e}"


@pytest.mark.fast
def test_lake_at_rest_many_steps() -> None:
    """100 steps over sharp cylinders stay at rest within accumulated FP noise."""
    _, _, dx, dy, zb = _ruppenthal_grid()
    eta0 = 2.0
    h = eta0 - zb
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)
    for _ in range(100):
        h, hu, hv, _ = _one_step(h, hu, hv, zb, dx, dy)

    np.testing.assert_allclose(h, eta0 - zb, atol=1e-10)
    assert np.abs(hu).max() < 1e-10, f"max |hu| after 100 steps = {np.abs(hu).max():.3e}"
    assert np.abs(hv).max() < 1e-10, f"max |hv| after 100 steps = {np.abs(hv).max():.3e}"


@pytest.mark.fast
def test_uniform_flow_no_bathymetry() -> None:
    """Flat bed + uniform IC must stay (nearly) uniform under HLL."""
    Nx = Ny = 30
    dx = dy = 0.5
    zb = np.zeros((Ny, Nx))
    h = np.full((Ny, Nx), 2.0)
    u0, v0 = 2.21, 2.21
    hu = h * u0
    hv = h * v0

    for _ in range(10):
        h, hu, hv, _ = _one_step(h, hu, hv, zb, dx, dy)

    # Edge cells leak via Neumann; central interior should track the IC.
    interior = slice(2, -2)
    np.testing.assert_allclose(h[interior, interior], 2.0, atol=1e-6)
    u = hu / h
    v = hv / h
    np.testing.assert_allclose(u[interior, interior], u0, atol=1e-3)
    np.testing.assert_allclose(v[interior, interior], v0, atol=1e-3)


@pytest.mark.fast
def test_lake_at_rest_with_dirichlet_zero() -> None:
    """Dirichlet inflow prescribing rest state (u=v=0, eta=2) does not
    disturb the interior — Audusse + zero-inflow stays lake-at-rest."""
    _, _, dx, dy, zb = _ruppenthal_grid()
    eta0 = 2.0
    h = eta0 - zb
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)

    for _ in range(50):
        h, hu, hv, _ = _one_step(
            h,
            hu,
            hv,
            zb,
            dx,
            dy,
            eta_inflow=eta0,
            u_inflow=0.0,
            v_inflow=0.0,
        )

    np.testing.assert_allclose(h, eta0 - zb, atol=1e-10)
    assert np.abs(hu).max() < 1e-10, f"max |hu| after Dirichlet-rest = {np.abs(hu).max():.3e}"
    assert np.abs(hv).max() < 1e-10, f"max |hv| after Dirichlet-rest = {np.abs(hv).max():.3e}"


@pytest.mark.fast
def test_dirichlet_inflow_pins_boundary_cells() -> None:
    """With Dirichlet inflow at small-x and small-y, a uniform crossflow
    over a flat bed stays uniform across the ENTIRE domain (including
    boundary cells) — the buggy old default of all-Neumann let those drift.
    """
    Nx = Ny = 30
    dx = dy = 0.5
    zb = np.zeros((Ny, Nx))
    eta0 = 2.0
    u0, v0 = 2.21, 2.21
    h = np.full((Ny, Nx), eta0)
    hu = h * u0
    hv = h * v0

    for _ in range(50):
        h, hu, hv, _ = _one_step(
            h,
            hu,
            hv,
            zb,
            dx,
            dy,
            eta_inflow=eta0,
            u_inflow=u0,
            v_inflow=v0,
        )

    np.testing.assert_allclose(h, eta0, atol=1e-3)
    u = hu / h
    v = hv / h
    np.testing.assert_allclose(u, u0, atol=1e-3)
    np.testing.assert_allclose(v, v0, atol=1e-3)


@pytest.mark.fast
def test_ruppenthal_run_stays_bounded() -> None:
    """Smoke: 2 s of the canonical Ruppenthal setup runs cleanly."""
    data = gt.generate_dataset(
        Lx=25.0,
        Ly=25.0,
        Nx=50,
        Ny=50,
        t_end=2.0,
        n_save=4,
        eta_init=2.0,
        u_init=2.21,
        v_init=2.21,
        smooth=0.0,
        verbose=False,
    )
    assert np.all(np.isfinite(data["h"]))
    assert np.all(np.isfinite(data["u"]))
    assert np.all(np.isfinite(data["v"]))
    assert data["h"].min() > 0.0
    assert np.abs(data["u"]).max() < 5.0
    assert np.abs(data["v"]).max() < 5.0
