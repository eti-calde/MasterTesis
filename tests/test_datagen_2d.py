"""Solver-free tests for the 2D datagen additions (Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest

from pinn_bath.datagen import (
    BathymetrySampler2D,
    Grid2D,
    SimResult,
)
from pinn_bath.datagen.solvers.pyclaw2d import _make_characteristic_inflow_2d


def _result_1d_like(v=None):
    nt, nx = 4, 6
    return SimResult(
        t=np.linspace(0, 1, nt),
        x=np.linspace(0, 1, nx),
        zb=np.zeros(nx),
        eta=np.ones((nt, nx)),
        u=np.zeros((nt, nx)),
        h=np.ones((nt, nx)),
        v=v,
    )


@pytest.mark.fast
def test_simresult_ok_checks_v_when_present() -> None:
    """NaN confined to the transverse momentum must fail the finiteness gate."""
    assert _result_1d_like(v=None).ok  # 1D path: v absent, unaffected
    v = np.zeros((4, 6))
    assert _result_1d_like(v=v).ok
    v[0, 0] = np.nan
    assert not _result_1d_like(v=v).ok


class _FakeDim:
    name = "x"


@pytest.mark.fast
def test_inflow_2d_upwinds_tangential_velocity() -> None:
    """Ghost v must be 0 while the normal flow enters, extrapolated otherwise."""
    g, ng, ny = 9.81, 2, 5
    h_rest, wl = 1.0, 1.0
    qbc = np.zeros((3, 4 + 2 * ng, ny))
    qbc[0] = h_rest
    qbc[2, ng, :] = 0.3 * h_rest  # interior tangential momentum (v_i = 0.3)
    auxbc = np.zeros((1, 4 + 2 * ng, ny))  # flat bed

    # Inflow phase: positive surface anomaly at x_lower -> u_g > 0 -> v_g = 0.
    bc_in = _make_characteristic_inflow_2d(
        lambda t: 0.05, h_rest=h_rest, g=g, edge="x_lower", dry_tolerance=1e-3, water_level=wl
    )
    bc_in(None, _FakeDim(), 0.0, qbc.copy(), auxbc, ng)  # warm call (no assert)
    q = qbc.copy()
    bc_in(None, _FakeDim(), 0.0, q, auxbc, ng)
    assert q[1, 0, 0] > 0.0  # normal flow enters
    assert np.allclose(q[2, :ng, :], 0.0)  # tangential prescribed to 0

    # Outflow phase: negative anomaly -> u_g < 0 -> v extrapolated.
    bc_out = _make_characteristic_inflow_2d(
        lambda t: -0.05, h_rest=h_rest, g=g, edge="x_lower", dry_tolerance=1e-3, water_level=wl
    )
    q = qbc.copy()
    bc_out(None, _FakeDim(), 0.0, q, auxbc, ng)
    assert q[1, 0, 0] < 0.0  # normal flow leaves
    h_g = q[0, 0, 0]
    assert np.allclose(q[2, :ng, :], h_g * 0.3)  # tangential extrapolated


@pytest.mark.fast
def test_inflow_2d_survives_garbage_ghost_corners() -> None:
    """Uninitialised corner garbage must not overflow the ghost products."""
    g, ng, ny = 9.81, 2, 5
    qbc = np.full((3, 4 + 2 * ng, ny), 1e300)  # np.empty-style garbage
    qbc[:, ng:-ng, 1:-1] = 0.0
    qbc[0, ng:-ng, 1:-1] = 1.0
    auxbc = np.zeros((1, 4 + 2 * ng, ny))
    bc = _make_characteristic_inflow_2d(
        lambda t: 0.05, h_rest=1.0, g=g, edge="x_lower", dry_tolerance=1e-3, water_level=1.0
    )
    with np.errstate(over="raise"):
        bc(None, _FakeDim(), 0.0, qbc, auxbc, ng)
    assert np.isfinite(qbc[:, :ng, :]).all()


@pytest.mark.fast
def test_sampler2d_cap_holds_pointwise() -> None:
    """trend + features never exceeds the deep-water cap, any tier/slope."""
    grid = Grid2D(nx=64, ny=32)
    X, Y = grid.meshgrid()
    cap = 0.55
    for slope in (-0.0125, 0.0, 0.0125):
        sampler = BathymetrySampler2D(slope_range=(slope, slope))
        rng = np.random.default_rng(0)
        for d in ("easy", "medium", "hard"):
            for _ in range(25):
                f = sampler.sample(d, rng, X, Y, sea_level=1.0, max_bed_elevation=cap)
                assert f.profile(X, Y).max() <= cap + 1e-9
