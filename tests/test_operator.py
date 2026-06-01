"""Tests for the inverse-operator module (F3)."""

from __future__ import annotations

import pytest
import torch

from pinn_bath.operator.architectures import build_operator
from pinn_bath.operator.data import Normalizer
from pinn_bath.operator.physics import physics_loss, swe_residual_grid


@pytest.mark.fast
def test_physics_lake_at_rest_continuity_exact() -> None:
    """u=0, eta=const → continuity residual exactly 0 over arbitrary bed."""
    B, Nt, Nx = 2, 40, 80
    dx, dt = 10.0 / Nx, 8.0 / (Nt - 1)
    x = torch.linspace(dx / 2, 10 - dx / 2, Nx)
    zb = (0.4 * torch.exp(-(((x - 3) / 0.8) ** 2))).repeat(B, 1)
    eta = torch.ones(B, Nt, Nx)
    u = torch.zeros(B, Nt, Nx)
    r_c, r_m, _ = swe_residual_grid(eta, u, zb, dx, dt)
    assert r_c.abs().max() < 1e-6  # continuity exact at rest
    assert r_m.abs().max() < 5e-2  # conservative momentum: O(dx^2) imbalance


@pytest.mark.fast
def test_physics_grad_flows_to_zb_only() -> None:
    B, Nt, Nx = 1, 20, 40
    dx, dt = 0.04, 0.07
    eta = torch.randn(B, Nt, Nx) * 0.05 + 1.0
    u = torch.randn(B, Nt, Nx) * 0.1
    zb = torch.zeros(B, Nx, requires_grad=True)
    loss, parts = physics_loss(eta, u, zb, dx, dt)
    loss.backward()
    assert zb.grad is not None and torch.isfinite(zb.grad).all()
    assert set(parts) == {"cont", "mom"}


@pytest.mark.fast
def test_cnn_operator_shapes() -> None:
    B, Nt, Nx = 4, 30, 64
    model = build_operator("cnn", width=16)
    x_in = torch.randn(B, 2, Nt, Nx)
    out = model(x_in)
    assert out.shape == (B, Nx)
    assert torch.isfinite(out).all()


@pytest.mark.fast
def test_normalizer_roundtrip() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    eta = rng.normal(1.0, 0.05, (5, 10, 12)).astype("float32")
    u = rng.normal(0.0, 0.1, (5, 10, 12)).astype("float32")
    zb = rng.normal(0.0, 0.2, (5, 12)).astype("float32")
    norm = Normalizer.fit(eta, u, zb)
    zbt = torch.from_numpy(zb)
    back = norm.denorm_zb(norm.norm_zb(zbt))
    assert torch.allclose(back, zbt, atol=1e-5)


@pytest.mark.fast
def test_overfit_single_batch() -> None:
    """The operator can drive train MSE down on one batch (capacity sanity).

    Inputs are smooth (not white noise) so the time-pooling operator has a
    learnable target; we require a clear monotone decrease, not memorization.
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    b, nt, nx = 4, 20, 48
    model = build_operator("cnn", width=24)
    xs = torch.linspace(0, 1, nx)
    # smooth structured input + target (correlated across the batch)
    x_in = torch.stack(
        [torch.sin((i + 1) * 3.14 * xs).expand(b, nt, nx) for i in range(2)], dim=1
    )
    target = torch.cos(2 * 3.14 * xs).expand(b, nx).contiguous()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = float(F.mse_loss(model(x_in), target))
    for _ in range(120):
        loss = F.mse_loss(model(x_in), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.25 * first  # clearly learning the structured target
