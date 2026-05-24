"""Tests for the loss-component primitives in pinn_bath.losses.components."""

from __future__ import annotations

import pytest
import torch

from pinn_bath.losses import (
    boundary_dirichlet,
    data_mse,
    discharge,
    pde_mse,
    positivity,
    tikhonov,
    tv_1d,
    tv_2d,
)


@pytest.mark.fast
def test_data_mse_no_mask() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    obs = torch.tensor([1.0, 2.0, 4.0])
    assert data_mse(pred, obs).item() == pytest.approx(1.0 / 3.0)


@pytest.mark.fast
def test_data_mse_with_mask() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    obs = torch.tensor([1.0, 2.0, 4.0])
    mask = torch.tensor([False, True, False])
    assert data_mse(pred, obs, mask).item() == pytest.approx(0.0)


@pytest.mark.fast
def test_data_mse_empty_mask() -> None:
    pred = torch.zeros(3)
    obs = torch.ones(3)
    mask = torch.zeros(3, dtype=torch.bool)
    out = data_mse(pred, obs, mask)
    assert out.item() == 0.0


@pytest.mark.fast
def test_pde_mse_sums_components() -> None:
    r = {"cont": torch.tensor([1.0, -1.0]), "mom_x": torch.tensor([2.0, 0.0])}
    # mean(cont^2) + mean(mom_x^2) = 1.0 + 2.0 = 3.0
    assert pde_mse(r).item() == pytest.approx(3.0)


@pytest.mark.fast
def test_tv_1d_constant_field_is_zero() -> None:
    x = torch.linspace(0.0, 1.0, 11)
    v = torch.ones(11)
    assert tv_1d(v, x).item() == pytest.approx(0.0)


@pytest.mark.fast
def test_tv_1d_linear_field_matches_slope() -> None:
    x = torch.linspace(0.0, 1.0, 11)
    v = 2.0 * x  # slope = 2
    out = tv_1d(v, x).item()
    assert out == pytest.approx(2.0, abs=1e-6)


@pytest.mark.fast
def test_tv_2d_constant_field_near_zero() -> None:
    x = torch.linspace(0.0, 1.0, 5)
    y = torch.linspace(0.0, 1.0, 5)
    v = torch.ones(5, 5)
    out = tv_2d(v, x, y).item()
    assert out < 1.0e-3  # only the epsilon


@pytest.mark.fast
def test_tikhonov_is_mean_squared() -> None:
    v = torch.tensor([1.0, -1.0, 2.0])
    # mean(1, 1, 4) = 2
    assert tikhonov(v).item() == pytest.approx(2.0)


@pytest.mark.fast
def test_positivity_zero_for_positive_h() -> None:
    h = torch.tensor([1.0, 2.0, 0.1])
    assert positivity(h).item() == 0.0


@pytest.mark.fast
def test_positivity_penalizes_negative_h() -> None:
    h = torch.tensor([1.0, -0.5, 0.0])
    # mean(max(0,-h)^2) = mean(0, 0.25, 0) = 0.25/3
    assert positivity(h).item() == pytest.approx(0.25 / 3.0)


@pytest.mark.fast
def test_discharge_zero_at_target() -> None:
    h = torch.tensor([2.0, 2.0])
    u = torch.tensor([2.21, 2.21])
    q = 4.42
    assert discharge(h, u, q).item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.fast
def test_boundary_dirichlet_zero_when_matching() -> None:
    v = torch.tensor([0.0, 0.0])
    assert boundary_dirichlet(v, 0.0).item() == 0.0


@pytest.mark.fast
def test_boundary_dirichlet_squared_error() -> None:
    v = torch.tensor([1.0, -1.0])
    # mean((1-0)^2, (-1-0)^2) = 1
    assert boundary_dirichlet(v, 0.0).item() == pytest.approx(1.0)
