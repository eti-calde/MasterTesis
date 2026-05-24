"""Regression tests for the M8 batch of important fixes.

Covers T1+T2 (BC early-return + warn), S2 (--cases CLI), S5 (budget
order), S6 (run_id subset + schema_version), S7 (Registry poison).
S1 (sweep continue) needs a full sweep + intentional failure to
verify, kept as a manual sanity until a cheap integration harness
is built.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pinn_bath.config import SCHEMA_VERSION, DataCfg, LossWeights, OptimizerCfg, RunConfig
from pinn_bath.data import Case, CaseMetadata
from pinn_bath.registry import Registry

# --- Shared fixtures --------------------------------------------------------


def _exp1_like_case() -> Case:
    x = np.linspace(-1.0, 1.0, 9)
    return Case(
        metadata=CaseMetadata(
            case_id="m8_test",
            spatial_dim=1,
            has_t=False,
            bc_type="real_sensor",  # unenforced -> should warn at w.bc > 0
            constants={"g": 9.81},
            domain={"x": [-1.0, 1.0]},
            gt_source="synthetic",
        ),
        coords={"x": x},
        fields={
            "h": np.ones_like(x),
            "u": np.zeros_like(x),
            "zb": np.zeros_like(x),
            "eta": np.ones_like(x),
        },
    )


# --- T1: _compute_bc_loss early-return -------------------------------------


@pytest.mark.fast
def test_bc_loss_skipped_when_weight_zero(tmp_path: Path) -> None:
    """When w.bc = 0 the trainer never invokes the BC dispatch — even on
    a case whose bc_type would normally trigger a forward pass."""
    from pinn_bath.models import build
    from pinn_bath.trainers import AdamLBFGSTrainer

    case = _exp1_like_case()
    case.metadata = CaseMetadata(  # rewrite to a known-enforced bc_type
        case_id=case.metadata.case_id,
        spatial_dim=1,
        has_t=False,
        bc_type="open_dirichlet",
        constants={"g": 9.81, "x_0": 0.0, "w": 0.5},
        domain={"x": [-1.0, 1.0]},
        gt_source="synthetic",
    )
    case.save(tmp_path / "c.npz")
    case = Case.load(tmp_path / "c.npz")
    cfg = RunConfig(
        case="exp1",
        arch="A1",
        budget="small",
        seed=0,
        loss=LossWeights(data=10.0, pde=1.0, pos=10.0, bc=0.0),  # bc OFF
        optimizer=OptimizerCfg(adam_epochs=1, lbfgs_steps=0),
        data=DataCfg(case_path=str(tmp_path / "c.npz"), observations=["eta"]),
    )
    model = build(
        "A1", "small", spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"), ff_seed=0
    ).double()
    trainer = AdamLBFGSTrainer(model, case, cfg, n_collocation=20, n_observations=5, n_bc=10)
    _, losses = trainer.compute_loss()
    # BC reported as a hard zero (skipped path returns torch.zeros).
    assert losses["bc"] == 0.0


# --- T2: warning on unenforced bc_type with positive w.bc -----------------


@pytest.mark.fast
def test_warn_on_unenforced_bc_type(tmp_path: Path) -> None:
    """Building a trainer with bc_type=real_sensor and w.bc > 0 must emit
    a UserWarning so we don't silently lose the constraint."""
    from pinn_bath.models import build
    from pinn_bath.trainers import AdamLBFGSTrainer

    case = _exp1_like_case()  # bc_type="real_sensor"
    case.save(tmp_path / "c.npz")
    case = Case.load(tmp_path / "c.npz")
    cfg = RunConfig(
        case="exp1",
        arch="A1",
        budget="small",
        seed=0,
        loss=LossWeights(data=10.0, pde=1.0, pos=10.0, bc=100.0),  # bc ON
        optimizer=OptimizerCfg(adam_epochs=1, lbfgs_steps=0),
        data=DataCfg(case_path=str(tmp_path / "c.npz"), observations=["eta"]),
    )
    model = build(
        "A1", "small", spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"), ff_seed=0
    ).double()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        AdamLBFGSTrainer(model, case, cfg, n_collocation=20, n_observations=5, n_bc=10)
    msgs = [str(w.message) for w in ws]
    assert any("real_sensor" in m and "NOT enforced" in m for m in msgs), msgs


# --- S2: arch_scaling --cases CLI parsing ----------------------------------


@pytest.mark.fast
def test_arch_scaling_build_grid_cases_arg() -> None:
    from studies.arch_scaling import build_grid

    grid_full = build_grid()  # default 4 cases
    grid_2 = build_grid(cases=("exp1", "exp2"))
    assert len(grid_full) == 4 * 3 * 3 * 3
    assert len(grid_2) == 2 * 3 * 3 * 3
    assert {c.case for c in grid_2} == {"exp1", "exp2"}


# --- S5: aggregate budget order --------------------------------------------


@pytest.mark.fast
def test_aggregate_table_orders_budget() -> None:
    from studies.aggregate import _ordered_sort_key, format_text_table

    # _ordered_sort_key alone first.
    axes = ("budget",)
    assert _ordered_sort_key(axes, ("small",)) < _ordered_sort_key(axes, ("medium",))
    assert _ordered_sort_key(axes, ("medium",)) < _ordered_sort_key(axes, ("large",))
    # End-to-end through format_text_table.
    rows = [
        {"budget": b, "metric_rmse_zb": 1.0 + i} for i, b in enumerate(["large", "small", "medium"])
    ]
    text = format_text_table(rows, axes=("budget",), metric_key="metric_rmse_zb")
    # Locate the data rows after the separator line.
    data_lines = [
        line
        for line in text.splitlines()
        if line and not line.startswith("budget") and "-+-" not in line
    ]
    seen_order = [line.split("|")[0].strip() for line in data_lines]
    assert seen_order == ["small", "medium", "large"]


# --- S6: schema_version + run_id subset ------------------------------------


@pytest.mark.fast
def test_schema_version_constant_is_int() -> None:
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 2  # bumped in M7


@pytest.mark.fast
def test_run_id_excludes_checkpoint_settings() -> None:
    """Changing the checkpoint cadence (a non-scientific setting) MUST NOT
    change run_id. The whole point of the subset hashing is that movable
    bookkeeping doesn't invalidate cached runs."""
    from pinn_bath.config import CheckpointCfg

    base_kwargs = dict(
        case="exp1",
        arch="A1",
        budget="small",
        form="primitive",
        seed=0,
        data=DataCfg(case_path="/path/A.npz", observations=["eta"]),
    )
    cfg_a = RunConfig(**base_kwargs, checkpoint=CheckpointCfg(every_epochs=500, keep_last_k=2))
    cfg_b = RunConfig(**base_kwargs, checkpoint=CheckpointCfg(every_epochs=1000, keep_last_k=5))
    assert cfg_a.run_id == cfg_b.run_id


@pytest.mark.fast
def test_run_id_excludes_case_path() -> None:
    """Moving the .npz on disk doesn't invalidate cached runs."""
    base_kwargs = dict(
        case="exp1",
        arch="A1",
        budget="small",
        form="primitive",
        seed=0,
    )
    cfg_a = RunConfig(**base_kwargs, data=DataCfg(case_path="/A.npz", observations=["eta"]))
    cfg_b = RunConfig(**base_kwargs, data=DataCfg(case_path="/B.npz", observations=["eta"]))
    assert cfg_a.run_id == cfg_b.run_id


@pytest.mark.fast
def test_run_id_includes_loss_weights() -> None:
    """The scientific subset DOES include loss weights — changing data_u
    must produce a different run_id."""
    base_kwargs = dict(
        case="exp1",
        arch="A1",
        budget="small",
        form="primitive",
        seed=0,
        data=DataCfg(case_path="/A.npz", observations=["eta"]),
    )
    cfg_a = RunConfig(**base_kwargs, loss=LossWeights(data=10.0, data_u=0.0))
    cfg_b = RunConfig(**base_kwargs, loss=LossWeights(data=10.0, data_u=5.0))
    assert cfg_a.run_id != cfg_b.run_id


# --- S7: Registry poison check ---------------------------------------------


@pytest.mark.fast
def test_registry_skips_poisoned_run(tmp_path: Path) -> None:
    """A run with the latest manifest status="error" is skipped, not retried,
    unless ``retry_errors=True``."""
    cfg = RunConfig(
        case="exp1",
        arch="A1",
        budget="small",
        form="primitive",
        seed=0,
        data=DataCfg(case_path="/dummy.npz", observations=["eta"]),
    )

    # 1) Fresh registry: action="run".
    reg = Registry(tmp_path / "s1")
    assert reg.decide(cfg).action == "run"

    # 2) Mark error; new registry instance reads the manifest and skips.
    reg.mark_started(cfg)
    reg.mark_finished(cfg, status="error", error="boom")
    reg2 = Registry(tmp_path / "s1")
    decision = reg2.decide(cfg)
    assert decision.action == "skip"
    assert "errored" in decision.reason

    # 3) retry_errors=True overrides.
    reg3 = Registry(tmp_path / "s1", retry_errors=True)
    assert reg3.decide(cfg).action == "run"


# --- Schema version round-trips through summary ----------------------------


@pytest.mark.fast
def test_summary_persists_schema_version(tmp_path: Path) -> None:
    from pinn_bath.tracking import RunRecorder

    cfg = RunConfig(
        case="exp1",
        arch="A1",
        budget="small",
        form="primitive",
        seed=0,
        data=DataCfg(case_path="/dummy.npz", observations=["eta"]),
    )
    rec = RunRecorder(tmp_path, cfg=cfg)
    rec.write_summary(status="ok")
    rec.close()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["schema_version"] == SCHEMA_VERSION
