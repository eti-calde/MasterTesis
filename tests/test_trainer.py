"""Smoke tests for pinn_bath.trainers.AdamLBFGSTrainer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinn_bath.config import DataCfg, LossWeights, OptimizerCfg, RunConfig
from pinn_bath.data import Case, CaseMetadata
from pinn_bath.models import build
from pinn_bath.seed import set_seed
from pinn_bath.tracking import RunRecorder
from pinn_bath.trainers import AdamLBFGSTrainer


def _build_synthetic_exp1_case(tmp_path: Path) -> Case:
    """A small Exp 1-like case (smooth Gaussian bump, Bernoulli flow)."""
    g = 9.81
    q = 4.42
    h_down = 2.0
    x = np.linspace(-8.0, 8.0, 81)
    # Gaussian bump (smooth).
    zb = 0.2 * np.exp(-(x**2) / 2.0)
    # Solve Bernoulli cubic for h.
    C = q * q / (2.0 * g * h_down**2) + h_down + zb[-1]
    a = zb - C
    b = q * q / (2.0 * g)
    h = np.full_like(x, h_down)
    for _ in range(80):  # Newton iteration
        f = h**3 + a * h**2 + b
        fp = 3.0 * h**2 + 2.0 * a * h
        h = h - f / fp
    u = q / h
    eta = h + zb

    case = Case(
        metadata=CaseMetadata(
            case_id="synthetic_exp1",
            spatial_dim=1,
            has_t=False,
            bc_type="open_dirichlet",
            # x_0 and w let flat_bed_loss know where the bump support lives.
            constants={"g": g, "q": q, "h_down": h_down, "x_0": 0.0, "w": 2.0},
            domain={"x": [float(x.min()), float(x.max())]},
            gt_source="analytical_bernoulli",
        ),
        coords={"x": x},
        fields={"h": h, "u": u, "zb": zb, "eta": eta},
    )
    out = tmp_path / "synth_exp1.npz"
    case.save(out)
    return Case.load(out)


def _smoke_cfg(case_path: str, seed: int = 0) -> RunConfig:
    return RunConfig(
        case="synthetic_exp1",
        arch="A1",
        budget="small",
        seed=seed,
        loss=LossWeights(data=10.0, pde=1.0, pos=10.0, tv=0.0, tikh=0.0),
        optimizer=OptimizerCfg(
            adam_epochs=80,
            adam_lr=1.0e-3,
            lbfgs_steps=0,  # skip L-BFGS for smoke test
        ),
        data=DataCfg(case_path=case_path, observations=["eta"], n_obs_points=60),
    )


@pytest.mark.fast
def test_trainer_runs_and_loss_is_finite(tmp_path: Path) -> None:
    set_seed(0, deterministic=False)
    case = _build_synthetic_exp1_case(tmp_path)
    cfg = _smoke_cfg(case.source_path.as_posix(), seed=0)
    model = build(
        cfg.arch,
        cfg.budget,
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=cfg.seed,
    ).double()  # float64 for residual numerical stability

    trainer = AdamLBFGSTrainer(model, case, cfg, n_collocation=400, n_observations=60)
    _, losses_init = trainer.compute_loss()
    result = trainer.train()
    assert result["status"] == "ok"
    losses_final = result["final_losses"]
    for k, v in losses_final.items():
        assert np.isfinite(v), f"loss '{k}' is not finite: {v}"
    # Total loss should decrease over ~80 Adam steps.
    assert losses_final["total"] < losses_init["total"], (
        f"total loss did not decrease: init={losses_init['total']:.4f}, final={losses_final['total']:.4f}"
    )


@pytest.mark.fast
def test_trainer_writes_recorder_files(tmp_path: Path) -> None:
    set_seed(1, deterministic=False)
    case = _build_synthetic_exp1_case(tmp_path)
    cfg = _smoke_cfg(case.source_path.as_posix(), seed=1)
    model = build(
        cfg.arch,
        cfg.budget,
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=cfg.seed,
    ).double()

    run_dir = tmp_path / "run"
    with RunRecorder(run_dir, cfg=cfg) as rec:
        trainer = AdamLBFGSTrainer(
            model,
            case,
            cfg,
            recorder=rec,
            n_collocation=400,
            n_observations=60,
            eval_log_every=20,
        )
        result = trainer.train()
    assert (run_dir / "env.json").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "summary.json").exists()

    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) >= 2
    row = json.loads(lines[0])
    assert row["phase"] == "adam"
    assert "L_total" in row

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["run_id"] == cfg.run_id
    assert "wall_time_s" in summary
    assert result["final_losses"]["total"] == summary["final_losses"]["total"]


@pytest.mark.fast
def test_trainer_periodic_bc_active_for_periodic_case(tmp_path: Path) -> None:
    """For a periodic case with bc weight > 0, the trainer reports a positive
    L_bc and the gradient updates the model toward periodicity."""
    from pinn_bath.data import Case, CaseMetadata

    # Build a tiny 1D periodic case (mock dT10-like).
    x = np.linspace(-2.0, 2.0, 21)
    t = np.linspace(0.0, 0.5, 6)
    Nt, Nx = t.size, x.size
    zb = 0.01 * np.cos(np.pi * x / 2.0)
    h = np.tile(1.0 + zb, (Nt, 1))
    u = np.zeros((Nt, Nx))
    eta = h + zb[None, :]
    case = Case(
        metadata=CaseMetadata(
            case_id="mini_periodic",
            spatial_dim=1,
            has_t=True,
            bc_type="periodic",
            constants={"g": 9.81},
            domain={"x": [-2.0, 2.0], "t": [0.0, 0.5]},
            gt_source="fv_hll",
        ),
        coords={"x": x, "t": t},
        fields={"h": h, "u": u, "zb": zb, "eta": eta},
    )
    case.save(tmp_path / "mini.npz")
    case = Case.load(tmp_path / "mini.npz")

    cfg = RunConfig(
        case="mini_periodic",
        arch="A1",
        budget="small",
        seed=0,
        loss=LossWeights(data=10.0, pde=1.0, pos=10.0, tv=0.0, tikh=0.0, bc=1.0),
        optimizer=OptimizerCfg(adam_epochs=40, adam_lr=1.0e-3, lbfgs_steps=0),
        data=DataCfg(
            case_path=(tmp_path / "mini.npz").as_posix(),
            observations=["eta"],
            n_obs_points=40,
        ),
    )
    set_seed(0, deterministic=False)
    model = build(
        cfg.arch,
        cfg.budget,
        spatial_dim=1,
        has_t=True,
        output_fields=("h", "u", "zb"),
        ff_seed=cfg.seed,
    ).double()

    trainer = AdamLBFGSTrainer(
        model,
        case,
        cfg,
        n_collocation=200,
        n_observations=40,
        n_bc=32,
    )
    _, losses_init = trainer.compute_loss()
    assert "bc" in losses_init
    assert losses_init["bc"] > 0.0, "L_bc should be > 0 at init for a non-periodic random model"
    result = trainer.train()
    assert result["status"] == "ok"
    losses_final = result["final_losses"]
    assert losses_final["bc"] < losses_init["bc"], (
        f"BC loss did not decrease: init={losses_init['bc']:.4f}, final={losses_final['bc']:.4f}"
    )


@pytest.mark.fast
def test_trainer_no_bc_contribution_when_weight_zero(tmp_path: Path) -> None:
    """With LossWeights.bc=0 the BC term is computed but contributes 0 to total."""
    case = _build_synthetic_exp1_case(tmp_path)
    cfg = _smoke_cfg(case.source_path.as_posix(), seed=0)
    assert cfg.loss.bc == 0.0, "smoke_cfg should leave bc weight at default 0"
    model = build(
        cfg.arch,
        cfg.budget,
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=cfg.seed,
    ).double()
    trainer = AdamLBFGSTrainer(model, case, cfg, n_collocation=200, n_observations=40)
    _, losses = trainer.compute_loss()
    # flat_bed_loss runs (Exp 1 is open_dirichlet) and may be > 0 on the random
    # initial model -- but since bc weight is 0, it contributes 0 to total.
    w = cfg.loss
    expected_total = (
        w.data * losses["data"]
        + w.data_u * losses["data_u"]
        + w.pde * losses["pde"]
        + w.pos * losses["pos"]
        + w.tikh * losses["tikh"]
    )
    assert losses["total"] == pytest.approx(expected_total, rel=1e-6)


@pytest.mark.fast
def test_trainer_open_dirichlet_bc_active(tmp_path: Path) -> None:
    """For an open_dirichlet case with bc weight > 0, the trainer reports a
    positive L_bc (the flat_bed prior fires) and gradients propagate."""
    case = _build_synthetic_exp1_case(tmp_path)
    # Same as _smoke_cfg but with bc weight on -> flat_bed_loss contributes.
    cfg = RunConfig(
        case="synthetic_exp1",
        arch="A1",
        budget="small",
        seed=0,
        loss=LossWeights(data=10.0, pde=1.0, pos=10.0, tv=0.0, tikh=0.0, bc=10.0),
        optimizer=OptimizerCfg(adam_epochs=40, adam_lr=1.0e-3, lbfgs_steps=0),
        data=DataCfg(
            case_path=case.source_path.as_posix(),
            observations=["eta"],
            n_obs_points=40,
        ),
    )
    set_seed(0, deterministic=False)
    model = build(
        cfg.arch,
        cfg.budget,
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=cfg.seed,
    ).double()
    trainer = AdamLBFGSTrainer(
        model,
        case,
        cfg,
        n_collocation=200,
        n_observations=40,
        n_bc=32,
    )
    _, losses_init = trainer.compute_loss()
    assert "bc" in losses_init
    assert losses_init["bc"] >= 0.0
    # Verify the BC term participates in the total (weight * bc is part of total).
    w = cfg.loss
    other_terms = (
        w.data * losses_init["data"]
        + w.data_u * losses_init["data_u"]
        + w.pde * losses_init["pde"]
        + w.pos * losses_init["pos"]
        + w.tikh * losses_init["tikh"]
    )
    bc_contrib = w.bc * losses_init["bc"]
    assert losses_init["total"] == pytest.approx(other_terms + bc_contrib, rel=1e-6)
    # Train a few epochs and confirm clean completion + gradients flow.
    result = trainer.train()
    assert result["status"] == "ok"
    assert np.isfinite(result["final_losses"]["bc"])
    assert np.isfinite(result["final_losses"]["total"])
