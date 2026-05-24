"""Tests for pinn_bath.diagnostics (NaN guards, grad norms, crash dump, S10)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_bath.checkpoint import CheckpointManager
from pinn_bath.config import CheckpointCfg, DataCfg, LossWeights, OptimizerCfg, RunConfig
from pinn_bath.data import Case, CaseMetadata
from pinn_bath.diagnostics import (
    TrainingDiverged,
    check_finite_loss,
    check_sanity_bounds,
    dump_crash_state,
    gradient_norm,
)
from pinn_bath.models import build
from pinn_bath.tracking import RunRecorder
from pinn_bath.trainers import AdamLBFGSTrainer

# --- Unit tests for individual helpers --------------------------------------


@pytest.mark.fast
def test_check_finite_passes_on_finite_loss() -> None:
    check_finite_loss(torch.tensor(0.5))


@pytest.mark.fast
def test_check_finite_raises_on_nan() -> None:
    with pytest.raises(TrainingDiverged, match="not finite"):
        check_finite_loss(torch.tensor(float("nan")))


@pytest.mark.fast
def test_check_finite_raises_on_inf() -> None:
    with pytest.raises(TrainingDiverged):
        check_finite_loss(torch.tensor(float("inf")))


@pytest.mark.fast
def test_check_sanity_passes_on_positive_h() -> None:
    out = {"h": torch.tensor([0.1, 0.2, 0.3]), "u": torch.zeros(3), "zb": torch.zeros(3)}
    check_sanity_bounds(out)


@pytest.mark.fast
def test_check_sanity_raises_on_nonpositive_h() -> None:
    out = {"h": torch.tensor([0.1, -0.5, 0.2]), "u": torch.zeros(3), "zb": torch.zeros(3)}
    with pytest.raises(TrainingDiverged, match="non-positive"):
        check_sanity_bounds(out)


@pytest.mark.fast
def test_check_sanity_raises_on_nonfinite_field() -> None:
    out = {
        "h": torch.tensor([1.0]),
        "u": torch.tensor([float("nan")]),
        "zb": torch.zeros(1),
    }
    with pytest.raises(TrainingDiverged, match="non-finite"):
        check_sanity_bounds(out)


@pytest.mark.fast
def test_gradient_norm_matches_manual() -> None:
    model = torch.nn.Linear(3, 2)
    x = torch.randn(5, 3)
    loss = model(x).pow(2).mean()
    loss.backward()
    manual = math.sqrt(
        sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)
    )
    assert gradient_norm(model) == pytest.approx(manual, rel=1.0e-6)


@pytest.mark.fast
def test_gradient_norm_zero_before_backward() -> None:
    model = torch.nn.Linear(3, 2)
    assert gradient_norm(model) == 0.0


@pytest.mark.fast
def test_dump_crash_state_writes_atomic(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    out = tmp_path / "dump.pt"
    dump_crash_state(
        out,
        model=model,
        epoch=42,
        phase="adam",
        last_loss=0.3,
        extra={"diagnosis": "test"},
    )
    assert out.exists()
    payload = torch.load(out, weights_only=False)
    assert payload["epoch"] == 42
    assert payload["phase"] == "adam"
    assert payload["last_loss"] == pytest.approx(0.3)
    assert payload["extra"]["diagnosis"] == "test"
    assert "model" in payload


# --- Integration: trainer aborts cleanly on divergence ---------------------


def _build_synthetic_case(tmp_path: Path) -> Case:
    g = 9.81
    q = 4.42
    h_down = 2.0
    x = np.linspace(-8.0, 8.0, 81)
    zb = 0.2 * np.exp(-(x**2) / 2.0)
    C = q * q / (2.0 * g * h_down**2) + h_down + zb[-1]
    a = zb - C
    b = q * q / (2.0 * g)
    h = np.full_like(x, h_down)
    for _ in range(80):
        f = h**3 + a * h**2 + b
        fp = 3.0 * h**2 + 2.0 * a * h
        h = h - f / fp
    u = q / h
    eta = h + zb
    case = Case(
        metadata=CaseMetadata(
            case_id="diverge_test",
            spatial_dim=1,
            has_t=False,
            bc_type="open_dirichlet",
            constants={"g": g, "q": q, "h_down": h_down},
            domain={"x": [float(x.min()), float(x.max())]},
            gt_source="analytical_bernoulli",
        ),
        coords={"x": x},
        fields={"h": h, "u": u, "zb": zb, "eta": eta},
    )
    out = tmp_path / "synth.npz"
    case.save(out)
    return Case.load(out)


@pytest.mark.fast
def test_trainer_handles_divergence_cleanly(tmp_path: Path) -> None:
    """Inject a NaN in the model weights; the trainer must abort with status='diverged'."""
    case = _build_synthetic_case(tmp_path)
    cfg = RunConfig(
        case="diverge_test",
        arch="A1",
        budget="small",
        seed=0,
        loss=LossWeights(data=10.0, pde=1.0, pos=10.0, tv=0.0, tikh=0.0),
        optimizer=OptimizerCfg(adam_epochs=5, adam_lr=1.0e-3, lbfgs_steps=0),
        checkpoint=CheckpointCfg(every_epochs=999),
        data=DataCfg(case_path=case.source_path.as_posix(), observations=["eta"], n_obs_points=40),
    )
    model = build(
        "A1",
        "small",
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=0,
    ).double()
    # Poison one weight with NaN so the first forward produces non-finite outputs.
    with torch.no_grad():
        first = next(iter(model.parameters()))
        first.copy_(first * float("nan"))

    run_dir = tmp_path / "run"
    cm = CheckpointManager(run_dir)
    with RunRecorder(run_dir, cfg=cfg) as rec:
        trainer = AdamLBFGSTrainer(
            model,
            case,
            cfg,
            recorder=rec,
            checkpoint=cm,
            n_collocation=100,
            n_observations=40,
        )
        result = trainer.train()
    assert result["status"] == "diverged"
    assert (run_dir / "summary.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "diverged"
    assert "error" in summary
    # Crash dump should also exist.
    assert (run_dir / "crash_dump.pt").exists()
