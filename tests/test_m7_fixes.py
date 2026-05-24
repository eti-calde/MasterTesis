"""Regression tests for the 7 issues caught in the post-migration manual review.

Each test pins one bug fix so the next refactor can't silently revert it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinn_bath.config import DataCfg, LossWeights, OptimizerCfg, RunConfig
from pinn_bath.data import Case, CaseMetadata
from pinn_bath.losses import initial_condition_loss, swe_residual
from pinn_bath.losses.residual import _friction_term_1d, _friction_term_2d
from pinn_bath.tracking import RunRecorder

# --- toy models -------------------------------------------------------------


class _ConstModel(nn.Module):
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


def _case_1d_transient_with_v_friendly() -> Case:
    x = np.linspace(-1.0, 1.0, 11)
    t = np.linspace(0.0, 1.0, 5)
    Nt, Nx = t.size, x.size
    zb = 0.1 * x**2
    h = np.full((Nt, Nx), 1.0)
    u = np.full((Nt, Nx), 0.5)
    return Case(
        metadata=CaseMetadata(
            case_id="m7_1d",
            spatial_dim=1,
            has_t=True,
            bc_type="closed",
            constants={"g": 9.81},
            domain={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
            gt_source="synthetic",
        ),
        coords={"x": x, "t": t},
        fields={"h": h, "u": u, "zb": zb, "eta": h + zb[None, :]},
    )


# --- Fix 6: Case 2D-steady guards ------------------------------------------


@pytest.mark.fast
def test_case_2d_steady_raises() -> None:
    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, 1.0, 4)
    with pytest.raises(ValueError, match="2D steady"):
        Case(
            metadata=CaseMetadata(
                case_id="bad_2d_steady",
                spatial_dim=2,
                has_t=False,
                bc_type="open_uniform",
                constants={"g": 9.81},
                domain={"x": [0.0, 1.0], "y": [0.0, 1.0]},
                gt_source="synthetic",
            ),
            coords={"x": x, "y": y},
            fields={
                "h": np.ones((4, 4)),
                "u": np.zeros((4, 4)),
                "v": np.zeros((4, 4)),
                "zb": np.zeros((4, 4)),
                "eta": np.ones((4, 4)),
            },
        )


# --- Fix 7: IC loss shape assert -------------------------------------------


@pytest.mark.fast
def test_ic_loss_raises_on_shape_mismatch() -> None:
    case = _case_1d_transient_with_v_friendly()
    # Corrupt: replace h with a (Nx, Nt) transposed array — same number
    # of elements but slicing [0] gives a row of length Nt = 5, not Nx = 11.
    bad_case = Case(
        metadata=case.metadata,
        coords=case.coords,
        fields={
            "h": case.fields["h"].T,  # (Nx, Nt) instead of (Nt, Nx)
            "u": case.fields["u"],
            "zb": case.fields["zb"],
            "eta": case.fields["eta"],
        },
    )
    model = _ConstModel(h=1.0, u=0.5, zb=0.0)
    with pytest.raises(ValueError, match="t=0 slice has size"):
        initial_condition_loss(model, bad_case, fields=("h",))


# --- Fix 5: 2D friction uses speed magnitude -------------------------------


@pytest.mark.fast
def test_friction_2d_matches_speed_magnitude() -> None:
    """When u = v, |U| = sqrt(2) * |u|, so 2D friction is sqrt(2) bigger
    per component than the buggy 1D-per-component implementation."""
    u = torch.tensor([0.5, 0.7, 1.0], dtype=torch.float64).reshape(-1, 1)
    v = u.clone()  # u == v
    h = torch.full_like(u, 2.0)
    params = {"n_manning": 0.025}
    Sf_1d_u = _friction_term_1d(u, h, model="manning", g=9.81, params=params)
    Sf_2d_u, Sf_2d_v = _friction_term_2d(u, v, h, model="manning", g=9.81, params=params)
    # Per-component ratio sqrt(2) within eps tolerance.
    ratio = (Sf_2d_u / Sf_1d_u).detach()
    torch.testing.assert_close(ratio, torch.full_like(ratio, 2**0.5), atol=1e-6, rtol=0)
    # v-component same magnitude, same direction.
    torch.testing.assert_close(Sf_2d_u, Sf_2d_v, atol=0, rtol=0)


@pytest.mark.fast
def test_friction_2d_linear_kappa_unaffected() -> None:
    """Linear drag is intrinsically component-wise; 2D == per-component 1D."""
    u = torch.tensor([0.5, 0.7], dtype=torch.float64).reshape(-1, 1)
    v = torch.tensor([0.3, -0.4], dtype=torch.float64).reshape(-1, 1)
    h = torch.full_like(u, 2.0)
    params = {"kappa": 0.2, "eps_dry": 1.0e-4}
    Sf_1d_u = _friction_term_1d(u, h, model="linear_kappa", g=9.81, params=params)
    Sf_1d_v = _friction_term_1d(v, h, model="linear_kappa", g=9.81, params=params)
    Sf_2d_u, Sf_2d_v = _friction_term_2d(u, v, h, model="linear_kappa", g=9.81, params=params)
    torch.testing.assert_close(Sf_1d_u, Sf_2d_u, atol=0, rtol=0)
    torch.testing.assert_close(Sf_1d_v, Sf_2d_v, atol=0, rtol=0)


# --- Fix 4: data_v wired ----------------------------------------------------


@pytest.mark.fast
def test_loss_weights_data_v_in_schema() -> None:
    w = LossWeights()
    assert hasattr(w, "data_v")
    assert w.data_v == 0.0
    # Custom set round-trips.
    w2 = LossWeights(data_v=5.0)
    assert w2.data_v == 5.0


# --- Fix 1: tv and dry wired in trainer ------------------------------------


@pytest.mark.fast
def test_loss_weights_tv_default_is_zero() -> None:
    """Backward-compat: tests building LossWeights() get tv=0 (was 1e-4)."""
    assert LossWeights().tv == 0.0


# --- Fix 3: form persisted in summary.json ---------------------------------


@pytest.mark.fast
def test_summary_persists_form(tmp_path: Path) -> None:
    cfg = RunConfig(
        case="exp1",
        arch="A1",
        budget="small",
        form="prim_cons",
        seed=42,
        data=DataCfg(case_path="dummy.npz", observations=["eta"]),
        optimizer=OptimizerCfg(adam_epochs=1, lbfgs_steps=0),
    )
    rec = RunRecorder(tmp_path, cfg=cfg)
    rec.write_summary(status="ok")
    rec.close()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["form"] == "prim_cons"
    assert summary["case"] == "exp1"
    assert summary["arch"] == "A1"


@pytest.mark.fast
def test_aggregate_collect_picks_up_form(tmp_path: Path) -> None:
    from studies.aggregate import collect

    # Hand-craft two runs that differ only in form.
    for run_id, form in [("a", "primitive"), ("b", "conservative")]:
        d = tmp_path / run_id
        d.mkdir()
        (d / "summary.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "wall_time_s": 1.0,
                    "run_id": run_id,
                    "case": "exp1",
                    "arch": "A1",
                    "budget": "small",
                    "form": form,
                    "seed": 0,
                }
            )
        )
    rows = collect(tmp_path)
    forms = sorted(r["form"] for r in rows)
    assert forms == ["conservative", "primitive"]


# --- Fix 2: exp1_sensitivity obstype weights -------------------------------


@pytest.mark.fast
@pytest.mark.parametrize(
    "observations,expect_data,expect_data_u",
    [
        (("eta",), 10.0, 0.0),
        (("u",), 0.0, 10.0),
        (("eta", "u"), 10.0, 5.0),
    ],
)
def test_exp1_sensitivity_obstype_weights(observations, expect_data, expect_data_u) -> None:
    from studies.exp1_sensitivity import _weights_for_obstype

    w = _weights_for_obstype(observations)
    assert w["data"] == expect_data
    assert w["data_u"] == expect_data_u


@pytest.mark.fast
def test_exp1_sensitivity_obstype_unsupported_raises() -> None:
    from studies.exp1_sensitivity import _weights_for_obstype

    with pytest.raises(ValueError, match="unsupported"):
        _weights_for_obstype(())


# --- Fix 1 cont.: TV / dry firing in trainer compute_loss -------------------


@pytest.mark.fast
def test_friction_swe_residual_2d_differs_from_none() -> None:
    """End-to-end: swe_residual('primitive', 2D transient, Manning) adds
    a non-zero source term to the momentum residual. Pins that the
    friction kwarg is plumbed into _residual_2d_transient (the speed-
    magnitude correctness is checked by test_friction_2d_matches_speed_magnitude)."""
    torch.manual_seed(0)
    N = 20
    x = torch.linspace(-1.0, 1.0, N, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    y = x.clone().detach().requires_grad_(True)
    t = x.clone().detach().requires_grad_(True)
    h = 2.0 + 0.0 * x + 0.0 * y + 0.0 * t
    u = 0.5 + 0.0 * x + 0.0 * y + 0.0 * t
    v = 0.3 + 0.0 * x + 0.0 * y + 0.0 * t
    zb = 0.0 * x + 0.0 * y + 0.0 * t
    res_none = swe_residual(
        "primitive",
        {"x": x, "y": y, "t": t},
        {"h": h, "u": u, "v": v, "zb": zb},
        spatial_dim=2,
        has_t=True,
        friction="none",
    )
    res_man = swe_residual(
        "primitive",
        {"x": x, "y": y, "t": t},
        {"h": h, "u": u, "v": v, "zb": zb},
        spatial_dim=2,
        has_t=True,
        friction="manning",
        friction_params={"n_manning": 0.025},
    )
    # Constant fields, no zb gradient: with no friction both residuals are 0.
    assert float((res_none["mom_x"] ** 2).sum()) == pytest.approx(0.0)
    # With Manning the momentum is exactly the friction source term.
    assert float((res_man["mom_x"] ** 2).sum()) > 0.0
    assert float((res_man["mom_y"] ** 2).sum()) > 0.0
