"""Smoke tests for the M1 additions to pinn_bath: IC loss, wall_bc_loss,
inflow_outflow_1d_loss, and friction in swe_residual.

Each test exercises the new function on a hand-rolled toy case + a
trivial model so the math is verifiable by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinn_bath.data import Case, CaseMetadata
from pinn_bath.losses import (
    inflow_outflow_1d_loss,
    initial_condition_loss,
    swe_residual,
    wall_bc_loss,
)

# --- toy models ----------------------------------------------------------


class _ConstModel(nn.Module):
    """Outputs ``{h: h_const, u: u_const, zb: zb_const}`` everywhere."""

    def __init__(self, h: float, u: float, zb: float, v: float | None = None) -> None:
        super().__init__()
        self.h = nn.Parameter(torch.tensor(h, dtype=torch.float64))
        self.u = nn.Parameter(torch.tensor(u, dtype=torch.float64))
        self.zb = nn.Parameter(torch.tensor(zb, dtype=torch.float64))
        if v is not None:
            self.v = nn.Parameter(torch.tensor(v, dtype=torch.float64))

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        n = next(iter(coords.values())).shape[0]
        out = {
            "h": self.h.expand(n, 1),
            "u": self.u.expand(n, 1),
            "zb": self.zb.expand(n, 1),
        }
        if hasattr(self, "v"):
            out["v"] = self.v.expand(n, 1)
        return out


# --- cases ---------------------------------------------------------------


def _case_1d_steady_exp1_like() -> Case:
    x = np.linspace(-10.0, 10.0, 41)
    return Case(
        metadata=CaseMetadata(
            case_id="exp1_like",
            spatial_dim=1,
            has_t=False,
            bc_type="open_dirichlet",
            constants={"g": 9.81, "h_down": 2.0, "q": 4.42, "x_0": 0.0, "w": 2.0},
            domain={"x": [-10.0, 10.0]},
            gt_source="analytical",
        ),
        coords={"x": x},
        fields={
            "h": np.full_like(x, 2.0),
            "u": np.full_like(x, 2.21),
            "zb": np.zeros_like(x),
            "eta": np.full_like(x, 2.0),
        },
    )


def _case_1d_transient_thacker_like() -> Case:
    x = np.linspace(-2.0, 2.0, 21)
    t = np.linspace(0.0, 1.0, 6)
    Nt, Nx = t.size, x.size
    zb = 0.5 * (x**2 - 1.0)
    h0 = np.maximum(1.0 - zb, 0.0)
    h = np.tile(h0, (Nt, 1))
    u = np.zeros((Nt, Nx))
    eta = h + zb[None, :]
    return Case(
        metadata=CaseMetadata(
            case_id="thacker_like",
            spatial_dim=1,
            has_t=True,
            bc_type="closed_walls",
            constants={"g": 9.81},
            domain={"x": [-2.0, 2.0], "t": [0.0, 1.0]},
            gt_source="analytical",
        ),
        coords={"x": x, "t": t},
        fields={"h": h, "u": u, "zb": zb, "eta": eta},
    )


# --- IC loss -------------------------------------------------------------


@pytest.mark.fast
def test_initial_condition_loss_zero_when_match() -> None:
    case = _case_1d_transient_thacker_like()
    # Model outputs h = 1 - zb (matches IC), u = 0, zb = whatever (we ignore).
    # For simplicity, IC checks h, u (default fields); use a model that
    # outputs constant h = mean(h_init), constant u = 0.
    h_target = float(case.fields["h"][0].mean())
    model = _ConstModel(h=h_target, u=0.0, zb=0.0)
    loss = initial_condition_loss(model, case, fields=("u",))  # only check u (which is exactly 0)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.fast
def test_initial_condition_loss_positive_when_mismatch() -> None:
    case = _case_1d_transient_thacker_like()
    model = _ConstModel(h=1.0, u=0.5, zb=0.0)  # u != 0
    loss = initial_condition_loss(model, case, fields=("u",))
    assert float(loss.item()) > 0.0


@pytest.mark.fast
def test_initial_condition_loss_raises_on_steady_case() -> None:
    case = _case_1d_steady_exp1_like()
    model = _ConstModel(h=2.0, u=2.21, zb=0.0)
    with pytest.raises(ValueError, match="transient"):
        initial_condition_loss(model, case)


# --- wall_bc_loss --------------------------------------------------------


@pytest.mark.fast
def test_wall_bc_loss_zero_for_zero_u() -> None:
    case = _case_1d_transient_thacker_like()
    model = _ConstModel(h=1.0, u=0.0, zb=0.0)
    loss = wall_bc_loss(model, case, n_bc=10, seed=0)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.fast
def test_wall_bc_loss_positive_for_nonzero_u() -> None:
    case = _case_1d_transient_thacker_like()
    model = _ConstModel(h=1.0, u=0.7, zb=0.0)
    loss = wall_bc_loss(model, case, n_bc=10, seed=0)
    # Two boundaries x 1 axis x u^2 = 0.49 each -> total = 2*0.49 = 0.98
    assert float(loss.item()) == pytest.approx(0.98, rel=1e-9)


# --- inflow_outflow_1d_loss ---------------------------------------------


@pytest.mark.fast
def test_inflow_outflow_loss_zero_for_satisfying_state() -> None:
    case = _case_1d_steady_exp1_like()  # h_down=2.0, q=4.42
    # zb=0 at both ends, h=2.0 at outlet (constant), u = q/h = 2.21
    model = _ConstModel(h=2.0, u=2.21, zb=0.0)
    loss = inflow_outflow_1d_loss(model, case)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.fast
def test_inflow_outflow_loss_positive_for_mismatch() -> None:
    case = _case_1d_steady_exp1_like()
    model = _ConstModel(h=1.0, u=1.0, zb=0.1)  # nothing matches
    loss = inflow_outflow_1d_loss(model, case)
    assert float(loss.item()) > 0.0


# --- friction in swe_residual --------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize(
    "friction,params,expected_kind",
    [
        ("none", {}, "zero"),
        ("manning", {"n_manning": 0.025}, "positive"),
        ("linear_kappa", {"kappa": 0.2}, "positive"),
    ],
)
def test_friction_adds_to_momentum(friction, params, expected_kind) -> None:
    """Manning + linear_kappa friction contribute > 0 to the momentum
    residual; ``none`` leaves it unchanged."""
    torch.manual_seed(0)
    x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    # Smooth fields satisfying the steady SWE only approximately.
    h = 2.0 + 0.1 * torch.sin(x)
    u = 0.5 * torch.ones_like(x) + 0.05 * torch.cos(x)
    zb = 0.05 * torch.exp(-(x**2))
    base = swe_residual(
        "primitive",
        {"x": x},
        {"h": h, "u": u, "zb": zb},
        spatial_dim=1,
        has_t=False,
        friction="none",
    )
    with_fr = swe_residual(
        "primitive",
        {"x": x},
        {"h": h, "u": u, "zb": zb},
        spatial_dim=1,
        has_t=False,
        friction=friction,
        friction_params=params,
    )
    delta = (with_fr["mom_x"] - base["mom_x"]).detach()
    if expected_kind == "zero":
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0, rtol=0)
    else:
        assert float((delta**2).mean().item()) > 0.0
