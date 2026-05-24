"""Tests for pinn_bath.metrics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_bath.data import Case, CaseMetadata
from pinn_bath.metrics import baseline_rmse_zb, evaluate_zb, nrmse, r_squared, rmse


@pytest.mark.fast
def test_rmse_zero_for_identical() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0


@pytest.mark.fast
def test_rmse_known_value() -> None:
    pred = torch.tensor([0.0, 0.0])
    true = torch.tensor([1.0, 3.0])
    # mean(1, 9) = 5, sqrt = sqrt(5)
    assert rmse(pred, true) == pytest.approx(math.sqrt(5.0))


@pytest.mark.fast
def test_nrmse_normalizes_by_range() -> None:
    pred = torch.tensor([0.0, 0.0])
    true = torch.tensor([0.0, 1.0])
    # rmse = sqrt(mean(0, 1)) = sqrt(0.5); range = 1.
    assert nrmse(pred, true) == pytest.approx(math.sqrt(0.5))


@pytest.mark.fast
def test_nrmse_nan_for_constant_true() -> None:
    assert math.isnan(nrmse(torch.zeros(3), torch.ones(3)))


@pytest.mark.fast
def test_r_squared_one_for_perfect() -> None:
    true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert r_squared(true, true) == pytest.approx(1.0)


@pytest.mark.fast
def test_r_squared_zero_for_mean_predictor() -> None:
    true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = torch.full_like(true, true.mean().item())
    assert r_squared(pred, true) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.fast
def test_baseline_rmse_zb_is_norm_of_true(tmp_path: Path) -> None:
    x = np.linspace(-2.0, 2.0, 11)
    zb = np.exp(-(x**2))
    case = Case(
        metadata=CaseMetadata(
            case_id="x",
            spatial_dim=1,
            has_t=False,
            bc_type="open_dirichlet",
            constants={},
            domain={"x": [-2.0, 2.0]},
            gt_source="analytical_bernoulli",
        ),
        coords={"x": x},
        fields={"h": np.full_like(x, 1.0), "u": np.zeros_like(x), "zb": zb, "eta": 1.0 + zb},
    )
    out = baseline_rmse_zb(case)
    expected_rmse = math.sqrt(float((zb**2).mean()))
    assert out["rmse_zb_baseline"] == pytest.approx(expected_rmse)
    assert "nrmse_zb_baseline" in out
    assert "r2_zb_baseline" in out


@pytest.mark.fast
def test_evaluate_zb_runs(tmp_path: Path) -> None:
    """evaluate_zb consumes a model + case and returns finite metrics."""
    from pinn_bath.models import build

    x = np.linspace(-1.0, 1.0, 21)
    zb = 0.1 * np.exp(-(x**2))
    case = Case(
        metadata=CaseMetadata(
            case_id="eval",
            spatial_dim=1,
            has_t=False,
            bc_type="open_dirichlet",
            constants={},
            domain={"x": [-1.0, 1.0]},
            gt_source="analytical_bernoulli",
        ),
        coords={"x": x},
        fields={"h": np.full_like(x, 2.0), "u": np.full_like(x, 1.0), "zb": zb, "eta": 2.0 + zb},
    )
    model = build(
        "A1",
        "small",
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=0,
    ).double()
    metrics = evaluate_zb(model, case)
    for k in ("rmse_zb", "nrmse_zb", "r2_zb"):
        assert k in metrics
        assert math.isfinite(metrics[k])
