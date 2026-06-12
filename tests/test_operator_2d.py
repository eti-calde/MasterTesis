"""Tests for the 2D operator additions: physics residual, architecture, loaders."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinn_bath.operator.architectures import build_operator
from pinn_bath.operator.data import Normalizer, make_loaders
from pinn_bath.operator.physics import physics_loss_2d, swe_residual_grid_2d


@pytest.mark.fast
def test_physics2d_lake_at_rest_continuity_exact() -> None:
    """u=v=0, eta=const -> 2D continuity residual exactly 0 over any bed."""
    B, Nt, Ny, Nx = 2, 12, 16, 20
    dx, dy, dt = 0.2, 0.2, 0.1
    yy, xx = torch.meshgrid(torch.arange(Ny) * dy, torch.arange(Nx) * dx, indexing="ij")
    zb = (0.4 * torch.exp(-(((xx - 2) ** 2 + (yy - 1.5) ** 2) / 0.5))).repeat(B, 1, 1)
    eta = torch.ones(B, Nt, Ny, Nx)
    u = torch.zeros(B, Nt, Ny, Nx)
    v = torch.zeros(B, Nt, Ny, Nx)
    r_c, _, _, wet = swe_residual_grid_2d(eta, u, v, zb, dx, dy, dt)
    assert wet.all()
    assert torch.allclose(r_c, torch.zeros_like(r_c))


@pytest.mark.fast
def test_physics2d_zb_carries_gradient() -> None:
    """The 2D physics loss must backprop into zb (the only trained tensor)."""
    B, Nt, Ny, Nx = 1, 8, 10, 12
    g = torch.Generator().manual_seed(0)
    eta = 1.0 + 0.01 * torch.randn(B, Nt, Ny, Nx, generator=g)
    u = 0.01 * torch.randn(B, Nt, Ny, Nx, generator=g)
    v = 0.01 * torch.randn(B, Nt, Ny, Nx, generator=g)
    zb = (0.1 * torch.randn(B, Ny, Nx, generator=g)).requires_grad_(True)
    loss, parts = physics_loss_2d(eta, u, v, zb, 0.2, 0.2, 0.1)
    loss.backward()
    assert zb.grad is not None and torch.isfinite(zb.grad).all()
    assert parts["cont"] > 0.0


@pytest.mark.fast
def test_cnn2d_output_matches_input_grid() -> None:
    """Fully shape-robust: output (Ny, Nx) for even and odd input sizes."""
    m = build_operator("cnn2d", size="tiny")
    for nt, ny, nx in ((21, 30, 50), (16, 32, 48), (9, 17, 23)):
        out = m(torch.randn(2, 3, nt, ny, nx))
        assert out.shape == (2, ny, nx), (nt, ny, nx, tuple(out.shape))


@pytest.mark.fast
def test_normalizer_three_channels() -> None:
    rng = np.random.default_rng(0)
    eta = rng.normal(1.0, 0.1, (4, 6, 8, 10)).astype(np.float32)
    u = rng.normal(0.0, 0.1, (4, 6, 8, 10)).astype(np.float32)
    v = rng.normal(0.0, 0.1, (4, 6, 8, 10)).astype(np.float32)
    zb = rng.normal(0.0, 0.2, (4, 8, 10)).astype(np.float32)
    norm = Normalizer.fit(eta, u, zb, v=v)
    inp = norm.input_tensor(torch.from_numpy(eta), torch.from_numpy(u), torch.from_numpy(v))
    assert inp.shape == (4, 3, 6, 8, 10)
    # 1D normalizer rejects a stray v.
    norm1d = Normalizer.fit(eta, u, zb)
    with pytest.raises(ValueError):
        norm1d.input_tensor(torch.from_numpy(eta), torch.from_numpy(u), torch.from_numpy(v))


def _write_split_2d(path, n, nt=5, ny=6, nx=8, seed=0):
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        zb=rng.normal(size=(n, ny, nx)).astype(np.float32),
        eta=rng.normal(size=(n, nt, ny, nx)).astype(np.float32),
        u=rng.normal(size=(n, nt, ny, nx)).astype(np.float32),
        v=rng.normal(size=(n, nt, ny, nx)).astype(np.float32),
        score=rng.uniform(size=n).astype(np.float32),
        difficulty=rng.integers(0, 3, size=n).astype(np.int8),
        seed=np.arange(n, dtype=np.int64),
        x=np.linspace(0.5, 7.5, nx, dtype=np.float32),
        y=np.linspace(0.5, 5.5, ny, dtype=np.float32),
        t=np.linspace(0.0, 4.0, nt, dtype=np.float32),
    )


@pytest.mark.fast
def test_loaders_2d_carry_v_and_dy(tmp_path) -> None:
    for name, n in (("train", 6), ("val", 3), ("test", 4)):
        _write_split_2d(tmp_path / f"{name}.npz", n, seed=hash(name) % 100)
    for cache in (None, "cpu"):
        L = make_loaders(tmp_path, batch_size=2, cache_device=cache)
        assert L["dim"] == 2 and L["dy"] is not None and L["ny"] == 6
        b = next(iter(L["train"]))
        assert b["v"].shape == (2, 5, 6, 8)
        assert L["normalizer"].v_std is not None
