"""Tests for pinn_bath.checkpoint.CheckpointManager + resume integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_bath.checkpoint import (
    CheckpointManager,
    SignalCheckpoint,
    build_state,
)
from pinn_bath.config import CheckpointCfg, DataCfg, LossWeights, OptimizerCfg, RunConfig
from pinn_bath.data import Case, CaseMetadata
from pinn_bath.models import build
from pinn_bath.seed import set_seed
from pinn_bath.trainers import AdamLBFGSTrainer

# --- Synthetic Exp 1-like case ----------------------------------------------


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
            case_id="synthetic_resume",
            spatial_dim=1,
            has_t=False,
            bc_type="open_dirichlet",
            constants={"g": g, "q": q, "h_down": h_down, "x_0": 0.0, "w": 2.0},
            domain={"x": [float(x.min()), float(x.max())]},
            gt_source="analytical_bernoulli",
        ),
        coords={"x": x},
        fields={"h": h, "u": u, "zb": zb, "eta": eta},
    )
    out = tmp_path / "synth.npz"
    case.save(out)
    return Case.load(out)


def _resume_cfg(case_path: str, adam_epochs: int, every: int, seed: int = 0) -> RunConfig:
    return RunConfig(
        case="synthetic_resume",
        arch="A1",
        budget="small",
        seed=seed,
        loss=LossWeights(data=10.0, pde=1.0, pos=10.0, tv=0.0, tikh=0.0),
        optimizer=OptimizerCfg(adam_epochs=adam_epochs, adam_lr=1.0e-3, lbfgs_steps=0),
        checkpoint=CheckpointCfg(every_epochs=every, keep_last_k=3),
        data=DataCfg(case_path=case_path, observations=["eta"], n_obs_points=60),
    )


# --- CheckpointManager unit tests -------------------------------------------


@pytest.mark.fast
def test_build_state_includes_model_and_rng() -> None:
    set_seed(0, deterministic=False)
    model = build("A1", "small", spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"))
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    state = build_state(epoch=10, phase="adam", model=model, optimizer=opt)
    assert state["epoch"] == 10
    assert state["phase"] == "adam"
    assert "model" in state
    assert "optimizer" in state
    assert "rng" in state and "torch_cpu" in state["rng"]


@pytest.mark.fast
def test_save_load_roundtrip_atomic(tmp_path: Path) -> None:
    set_seed(0, deterministic=False)
    model = build("A1", "small", spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"))
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    state = build_state(epoch=5, phase="adam", model=model, optimizer=opt)
    cm = CheckpointManager(tmp_path / "run")
    cm.save(state, metric=0.5)
    assert (tmp_path / "run" / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "epoch_00000005.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "best.pt").exists()
    # No stray tmp files
    leftover = list((tmp_path / "run" / "checkpoints").glob(".*"))
    assert leftover == []
    loaded = cm.load_resume()
    assert loaded is not None and loaded["epoch"] == 5


@pytest.mark.fast
def test_best_tracks_minimum_metric(tmp_path: Path) -> None:
    set_seed(0, deterministic=False)
    model = build("A1", "small", spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"))
    cm = CheckpointManager(tmp_path / "run")
    for i, metric in enumerate([0.5, 0.3, 0.7, 0.2, 0.6]):
        state = build_state(epoch=i, phase="adam", model=model, optimizer=None)
        cm.save(state, metric=metric)
    best = cm.load("best.pt")
    assert best is not None and best["epoch"] == 3  # the 0.2 winner


@pytest.mark.fast
def test_rotate_keeps_only_last_k(tmp_path: Path) -> None:
    set_seed(0, deterministic=False)
    model = build("A1", "small", spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"))
    cm = CheckpointManager(tmp_path / "run", keep_last_k=2)
    for i in range(6):
        cm.save(build_state(epoch=i, phase="adam", model=model, optimizer=None))
    epoch_files = sorted((tmp_path / "run" / "checkpoints").glob("epoch_*.pt"))
    assert [p.name for p in epoch_files] == ["epoch_00000004.pt", "epoch_00000005.pt"]


@pytest.mark.fast
def test_signal_checkpoint_handles_signals(tmp_path: Path) -> None:
    import os

    with SignalCheckpoint() as sc:
        assert sc.interrupted is False
        # Send ourselves a SIGINT (raises KeyboardInterrupt by default but our
        # handler intercepts it).
        os.kill(os.getpid(), 2)  # SIGINT
        # Give the handler a moment
        import time as _t

        _t.sleep(0.05)
        assert sc.interrupted is True


# --- Resume integration: interrupted + resumed == uninterrupted (S7-4) ------


@pytest.mark.fast
def test_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """S7-4: a run interrupted halfway and resumed must match the uninterrupted run."""
    case = _build_synthetic_case(tmp_path)
    cfg_full = _resume_cfg(case.source_path.as_posix(), adam_epochs=60, every=999, seed=0)
    cfg_half = _resume_cfg(case.source_path.as_posix(), adam_epochs=30, every=999, seed=0)
    cfg_resume = _resume_cfg(case.source_path.as_posix(), adam_epochs=60, every=999, seed=0)

    # Run 1: uninterrupted 60 epochs
    set_seed(0, deterministic=True)
    model_full = build(
        "A1",
        "small",
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=0,
    ).double()
    trainer_full = AdamLBFGSTrainer(
        model_full,
        case,
        cfg_full,
        n_collocation=200,
        n_observations=40,
    )
    res_full = trainer_full.train()
    weights_full = {k: v.detach().cpu().clone() for k, v in model_full.state_dict().items()}

    # Run 2a: half-run with checkpoint at the end
    run_dir = tmp_path / "resume_run"
    set_seed(0, deterministic=True)
    model_half = build(
        "A1",
        "small",
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=0,
    ).double()
    cm_half = CheckpointManager(run_dir)
    trainer_half = AdamLBFGSTrainer(
        model_half,
        case,
        cfg_half,
        checkpoint=cm_half,
        n_collocation=200,
        n_observations=40,
    )
    res_half = trainer_half.train()
    assert res_half["status"] == "ok"
    assert (run_dir / "checkpoints" / "last.pt").exists()

    # Run 2b: resume from the checkpoint and run another 30 epochs
    set_seed(0, deterministic=True)  # set_seed again; resume will restore RNG anyway
    model_resume = build(
        "A1",
        "small",
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=0,
    ).double()
    cm_resume = CheckpointManager(run_dir)
    trainer_resume = AdamLBFGSTrainer(
        model_resume,
        case,
        cfg_resume,
        checkpoint=cm_resume,
        n_collocation=200,
        n_observations=40,
    )
    res_resume = trainer_resume.train()
    assert res_resume["status"] == "ok"

    # Compare model parameters
    weights_resume = {k: v.detach().cpu().clone() for k, v in model_resume.state_dict().items()}
    for k in weights_full:
        torch.testing.assert_close(
            weights_full[k],
            weights_resume[k],
            atol=1.0e-10,
            rtol=1.0e-8,
            msg=f"parameter {k} diverged after resume",
        )
    # And final losses
    assert res_full["final_losses"]["total"] == pytest.approx(
        res_resume["final_losses"]["total"], abs=1.0e-9
    )


@pytest.mark.fast
def test_periodic_save_during_training(tmp_path: Path) -> None:
    case = _build_synthetic_case(tmp_path)
    cfg = _resume_cfg(case.source_path.as_posix(), adam_epochs=15, every=5, seed=0)
    set_seed(0, deterministic=False)
    model = build(
        "A1",
        "small",
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=0,
    ).double()
    cm = CheckpointManager(tmp_path / "run", keep_last_k=10)
    trainer = AdamLBFGSTrainer(
        model,
        case,
        cfg,
        checkpoint=cm,
        n_collocation=200,
        n_observations=40,
    )
    trainer.train()
    # Saved at epochs 4, 9, 14 (i.e., (epoch+1) % 5 == 0), plus final last.pt
    epoch_files = sorted((tmp_path / "run" / "checkpoints").glob("epoch_*.pt"))
    epochs = [int(p.stem.split("_")[1]) for p in epoch_files]
    assert 4 in epochs
    assert 9 in epochs
    assert 14 in epochs
