"""Smoke tests for the study harnesses (grid generation + tiny end-to-end run)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinn_bath.config import CheckpointCfg, DataCfg, LossWeights, OptimizerCfg, RunConfig
from pinn_bath.data import Case, CaseMetadata
from studies import ablation_forms, arch_scaling
from studies._runner import run_one

# --- Grid generation -------------------------------------------------------


@pytest.mark.fast
def test_arch_scaling_grid_size_full() -> None:
    grid = arch_scaling.build_grid()
    # Default: 4 cases (exp1, exp2, exp3, exp5) x 3 archs x 3 budgets x 3 seeds.
    # Exp 5 was added to the default cases tuple in batch M8.4.
    assert len(grid) == 4 * 3 * 3 * 3


@pytest.mark.fast
def test_arch_scaling_grid_run_ids_are_unique() -> None:
    grid = arch_scaling.build_grid()
    assert len({c.run_id for c in grid}) == len(grid)


@pytest.mark.fast
def test_ablation_forms_grid_size() -> None:
    grid = ablation_forms.build_grid()
    assert len(grid) == 3 * 3  # 3 forms x 3 seeds


@pytest.mark.fast
def test_ablation_forms_uses_only_exp1() -> None:
    for cfg in ablation_forms.build_grid():
        assert cfg.case == "exp1"


@pytest.mark.fast
def test_arch_scaling_dry_run_is_a_no_op(tmp_path: Path) -> None:
    """Dry-run reports decisions but does not create any run dirs."""
    grid = arch_scaling.build_grid(seeds=(0,))  # smaller grid for speed
    counts = arch_scaling.run_study(tmp_path / "study", grid=grid, dry_run=True)
    assert counts["run"] == len(grid)
    # No run dirs were created.
    assert list((tmp_path / "study").glob("*/summary.json")) == []


# --- Tiny end-to-end -------------------------------------------------------


def _build_micro_case(tmp_path: Path) -> Case:
    """Tiny Exp 1-like synthetic case for fast smoke tests."""
    g, q, h_down = 9.81, 4.42, 2.0
    x = np.linspace(-8.0, 8.0, 41)
    zb = 0.2 * np.exp(-(x**2) / 2.0)
    C = q * q / (2.0 * g * h_down**2) + h_down + zb[-1]
    a = zb - C
    b = q * q / (2.0 * g)
    h = np.full_like(x, h_down)
    for _ in range(60):
        f = h**3 + a * h**2 + b
        fp = 3.0 * h**2 + 2.0 * a * h
        h = h - f / fp
    u = q / h
    eta = h + zb
    case = Case(
        metadata=CaseMetadata(
            case_id="micro_exp1",
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
    out = tmp_path / "micro.npz"
    case.save(out)
    return case


def _micro_cfg(case_path: str, seed: int = 0) -> RunConfig:
    return RunConfig(
        case="micro_exp1",
        arch="A1",
        budget="small",
        seed=seed,
        loss=LossWeights(data=10.0, pde=1.0, pos=10.0, tv=0.0, tikh=0.0),
        optimizer=OptimizerCfg(adam_epochs=20, adam_lr=1.0e-3, lbfgs_steps=0),
        checkpoint=CheckpointCfg(every_epochs=999),
        data=DataCfg(case_path=case_path, observations=["eta"], n_obs_points=30),
    )


@pytest.mark.fast
def test_run_one_executes_end_to_end(tmp_path: Path) -> None:
    _build_micro_case(tmp_path)
    cfg = _micro_cfg((tmp_path / "micro.npz").as_posix())
    result = run_one(cfg, tmp_path / "study")
    assert result["status"] == "ok"
    summary = json.loads(((tmp_path / "study" / cfg.run_id) / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert "final_losses" in summary


@pytest.mark.fast
def test_registry_skip_after_completion(tmp_path: Path) -> None:
    """A second invocation should mark the run as 'skip' (idempotency)."""
    from pinn_bath.registry import Registry

    _build_micro_case(tmp_path)
    cfg = _micro_cfg((tmp_path / "micro.npz").as_posix())
    run_one(cfg, tmp_path / "study")
    reg = Registry(tmp_path / "study")
    decision = reg.decide(cfg)
    assert decision.action == "skip"
    assert "ok" in decision.reason
