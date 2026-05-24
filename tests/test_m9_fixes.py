"""Regression tests for the M9 batch (tech-debt quick wins)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinn_bath.config import DataCfg, LossWeights, OptimizerCfg, RunConfig
from pinn_bath.data import Case, CaseMetadata
from pinn_bath.diagnostics import gradient_norm_per_term
from pinn_bath.metrics import evaluate_zb
from pinn_bath.tracking import RunRecorder
from pinn_bath.trainers import _SEED_OFFSETS

# --- P2: _SEED_OFFSETS exposed + populated --------------------------------


@pytest.mark.fast
def test_seed_offsets_dict_well_formed() -> None:
    """The three streams (collocation, bc, ic) all live in _SEED_OFFSETS
    with distinct prime offsets to decorrelate the RNGs."""
    assert set(_SEED_OFFSETS.keys()) == {"collocation", "bc", "ic"}
    vals = list(_SEED_OFFSETS.values())
    assert len(set(vals)) == 3  # distinct
    for v in vals:
        assert isinstance(v, int)
        assert v > 0


# --- PR2: evaluate_zb chunked path --------------------------------------


class _LinearZbModel(nn.Module):
    """Toy: zb = scale * sum(coords) so the output depends on coords
    and the model has at least one parameter (needed by ``evaluate_zb``
    which queries ``next(model.parameters())`` for device/dtype)."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        n = next(iter(coords.values())).shape[0]
        s = sum(coords[a] for a in coords) * self.scale
        zero = torch.zeros(n, 1, dtype=s.dtype, device=s.device)
        return {"h": torch.ones(n, 1, dtype=s.dtype, device=s.device), "u": zero, "zb": s}


def _small_case_1d_steady() -> Case:
    x = np.linspace(-1.0, 1.0, 11)
    return Case(
        metadata=CaseMetadata(
            case_id="m9_1d",
            spatial_dim=1,
            has_t=False,
            bc_type="open_dirichlet",
            constants={"g": 9.81, "x_0": 0.0, "w": 0.5},
            domain={"x": [-1.0, 1.0]},
            gt_source="synthetic",
        ),
        coords={"x": x},
        fields={
            "h": np.ones_like(x),
            "u": np.zeros_like(x),
            "zb": x.copy(),
            "eta": np.ones_like(x),
        },
    )


@pytest.mark.fast
def test_evaluate_zb_chunked_matches_single_pass() -> None:
    """Chunked and single-pass evaluate_zb produce byte-identical metrics."""
    case = _small_case_1d_steady()
    model = _LinearZbModel()
    full = evaluate_zb(model, case, chunk_size=None)
    chunked = evaluate_zb(model, case, chunk_size=3)
    assert full == chunked


# --- Regression: _compute_tv_loss must pass `t` for transient cases -------


@pytest.mark.fast
def test_compute_tv_loss_works_on_transient_1d() -> None:
    """Caught in the local overnight sweep (M9 first launch): the M9-PR2
    TV-loss helper only populated ``x`` (and ``y`` in 2D) in the model
    coords dict. For transient cases the unified ``BaseModel.forward``
    iterates over all axes including ``t`` and raised ``KeyError: 't'``.

    Fix: ``_compute_tv_loss`` now adds ``t = t_min`` to the coords for
    transient cases (zb is time-independent, so the value doesn't matter).
    """
    from pinn_bath.models import build
    from pinn_bath.trainers import AdamLBFGSTrainer

    x = np.linspace(0.0, 1.0, 7)
    t = np.linspace(0.0, 1.0, 4)
    Nt, Nx = t.size, x.size
    case = Case(
        metadata=CaseMetadata(
            case_id="m9_regression_tv_transient",
            spatial_dim=1,
            has_t=True,
            bc_type="closed",
            constants={"g": 9.81},
            domain={"x": [0.0, 1.0], "t": [0.0, 1.0]},
            gt_source="synthetic",
        ),
        coords={"x": x, "t": t},
        fields={
            "h": np.ones((Nt, Nx)),
            "u": np.zeros((Nt, Nx)),
            "zb": np.zeros(Nx),
            "eta": np.ones((Nt, Nx)),
        },
    )
    cfg = RunConfig(
        case="exp_test",
        arch="A1",
        budget="small",
        form="primitive",
        seed=0,
        # tv > 0 triggers the previously-buggy code path.
        loss=LossWeights(data=10.0, pde=1.0, pos=10.0, tv=1.0e-4),
        optimizer=OptimizerCfg(adam_epochs=1, lbfgs_steps=0),
        data=DataCfg(case_path="/dummy.npz", observations=["eta"]),
    )
    model = build(
        "A1",
        "small",
        spatial_dim=1,
        has_t=True,
        output_fields=("h", "u", "zb"),
        ff_seed=0,
    ).double()
    # Inject dummy obs to bypass random sampler.
    obs_coords = {
        "x": torch.zeros(2, 1, dtype=torch.float64),
        "t": torch.zeros(2, 1, dtype=torch.float64),
    }
    obs_values = {"eta": torch.zeros(2, 1, dtype=torch.float64)}
    trainer = AdamLBFGSTrainer(
        model,
        case,
        cfg,
        n_collocation=20,
        n_bc=5,
        obs_coords=obs_coords,
        obs_values=obs_values,
    )
    # The pre-fix KeyError happened inside compute_loss → _compute_tv_loss.
    _, losses = trainer.compute_loss()
    assert losses["tv"] >= 0.0
    assert np.isfinite(losses["tv"])


# --- PR3: RunRecorder fallback summary ------------------------------------


@pytest.mark.fast
def test_recorder_close_writes_incomplete_summary_when_forgotten(tmp_path: Path) -> None:
    cfg = RunConfig(
        case="exp1",
        arch="A1",
        budget="small",
        form="primitive",
        seed=0,
        data=DataCfg(case_path="/dummy.npz", observations=["eta"]),
        optimizer=OptimizerCfg(adam_epochs=1, lbfgs_steps=0),
    )
    with RunRecorder(tmp_path, cfg=cfg):
        pass  # never called write_summary
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "incomplete"


@pytest.mark.fast
def test_recorder_close_keeps_explicit_summary(tmp_path: Path) -> None:
    cfg = RunConfig(
        case="exp1",
        arch="A1",
        budget="small",
        form="primitive",
        seed=0,
        data=DataCfg(case_path="/dummy.npz", observations=["eta"]),
        optimizer=OptimizerCfg(adam_epochs=1, lbfgs_steps=0),
    )
    with RunRecorder(tmp_path, cfg=cfg) as rec:
        rec.write_summary(status="ok", final_losses={"total": 1.0})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["final_losses"]["total"] == 1.0


# --- S2: Case shape check at _validate ------------------------------------


@pytest.mark.fast
def test_case_validate_catches_wrong_field_shape() -> None:
    """A field with wrong shape (e.g., zb with time axis or h missing t)
    must raise at Case construction."""
    x = np.linspace(-1.0, 1.0, 5)
    t = np.linspace(0.0, 1.0, 3)
    with pytest.raises(ValueError, match="expected"):
        Case(
            metadata=CaseMetadata(
                case_id="bad_shape",
                spatial_dim=1,
                has_t=True,
                bc_type="closed",
                constants={"g": 9.81},
                domain={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
                gt_source="synthetic",
            ),
            coords={"x": x, "t": t},
            fields={
                "h": np.ones((3, 5)),
                "u": np.ones((5, 3)),  # WRONG: should be (Nt, Nx)
                "zb": np.zeros(5),
                "eta": np.ones((3, 5)),
            },
        )


# --- S3: gradient_norm_per_term smoke -------------------------------------


@pytest.mark.fast
def test_gradient_norm_per_term_smoke() -> None:
    """diagnostics.gradient_norm_per_term is exported but had no test.
    Pin its basic contract: takes {name: scalar_loss}, returns
    {name: float >= 0}."""
    model = nn.Linear(3, 1)
    x = torch.randn(8, 3)
    out = model(x)
    losses = {
        "a": (out**2).sum(),
        "b": out.mean(),
    }
    norms = gradient_norm_per_term(losses, model)
    assert set(norms.keys()) == {"a", "b"}
    for _name, n in norms.items():
        assert isinstance(n, float)
        assert n >= 0.0
