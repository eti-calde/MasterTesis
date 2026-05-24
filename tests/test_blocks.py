"""Smoke tests for pinn_bath.models.blocks."""

import pytest
import torch

from pinn_bath.models.blocks import MLP, FourierFeatures, count_parameters


@pytest.mark.fast
def test_fourier_features_output_shape() -> None:
    ff = FourierFeatures(in_dim=2, n_features=8, sigma=1.0, seed=42)
    x = torch.randn(5, 2)
    out = ff(x)
    assert out.shape == (5, 16)
    assert ff.out_dim == 16


@pytest.mark.fast
def test_fourier_features_seed_is_deterministic() -> None:
    ff1 = FourierFeatures(in_dim=1, n_features=4, sigma=2.0, seed=7)
    ff2 = FourierFeatures(in_dim=1, n_features=4, sigma=2.0, seed=7)
    x = torch.randn(3, 1)
    assert torch.equal(ff1(x), ff2(x))


@pytest.mark.fast
def test_fourier_features_different_seeds_differ() -> None:
    ff1 = FourierFeatures(in_dim=1, n_features=4, sigma=2.0, seed=7)
    ff2 = FourierFeatures(in_dim=1, n_features=4, sigma=2.0, seed=8)
    assert not torch.equal(ff1.B, ff2.B)


@pytest.mark.fast
def test_mlp_forward_shape() -> None:
    mlp = MLP(in_dim=3, out_dim=2, depth=4, width=32)
    x = torch.randn(7, 3)
    out = mlp(x)
    assert out.shape == (7, 2)


@pytest.mark.fast
def test_mlp_param_count_matches_formula() -> None:
    # depth=4, width=64, in_dim=1, out_dim=1:
    # Linear(1,64)   -> 1*64+64  = 128
    # Linear(64,64) x3 -> (64*64+64)*3 = 12480
    # Linear(64,1)   -> 64*1+1   = 65
    mlp = MLP(in_dim=1, out_dim=1, depth=4, width=64)
    expected = (1 * 64 + 64) + (64 * 64 + 64) * 3 + (64 * 1 + 1)
    assert count_parameters(mlp) == expected


@pytest.mark.fast
def test_mlp_rejects_zero_depth() -> None:
    with pytest.raises(ValueError):
        MLP(in_dim=1, out_dim=1, depth=0, width=16)


@pytest.mark.fast
def test_mlp_gradient_flows() -> None:
    mlp = MLP(in_dim=2, out_dim=1, depth=3, width=16)
    x = torch.randn(8, 2, requires_grad=True)
    loss = mlp(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mlp.parameters())
