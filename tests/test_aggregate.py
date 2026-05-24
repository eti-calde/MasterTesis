"""Tests for studies.aggregate (collection, aggregation, paired tests)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studies.aggregate import (
    aggregate_cell,
    collect,
    emit_latex_table,
    format_text_table,
    group_by,
    wilcoxon_pairs,
)


def _write_summary(
    run_dir: Path,
    *,
    case: str,
    arch: str,
    budget: str,
    seed: int,
    rmse_zb: float,
    total_loss: float = 0.5,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": "ok",
                "case": case,
                "arch": arch,
                "budget": budget,
                "seed": seed,
                "wall_time_s": 10.0,
                "final_losses": {"total": total_loss, "data": 0.1, "pde": 0.2},
                "final_metrics": {"rmse_zb": rmse_zb, "nrmse_zb": rmse_zb / 0.2, "r2_zb": 0.9},
                "baseline_metrics": {
                    "rmse_zb_baseline": 0.1,
                    "nrmse_zb_baseline": 0.5,
                    "r2_zb_baseline": 0.0,
                },
            }
        )
    )


def _build_study(tmp_path: Path) -> Path:
    """A small study with 2 archs x 3 seeds x 1 case x 1 budget = 6 runs."""
    study = tmp_path / "study"
    seeds = [0, 1, 2]
    # A1 is slightly worse than A2 for this synthetic dataset.
    for s in seeds:
        _write_summary(
            study / f"A1_s{s}",
            case="exp1",
            arch="A1",
            budget="small",
            seed=s,
            rmse_zb=0.020 + 0.001 * s,
        )
        _write_summary(
            study / f"A2_s{s}",
            case="exp1",
            arch="A2",
            budget="small",
            seed=s,
            rmse_zb=0.015 + 0.001 * s,
        )
    return study


@pytest.mark.fast
def test_collect_returns_flat_rows(tmp_path: Path) -> None:
    study = _build_study(tmp_path)
    rows = collect(study)
    assert len(rows) == 6
    assert {r["arch"] for r in rows} == {"A1", "A2"}
    assert "metric_rmse_zb" in rows[0]
    assert "loss_total" in rows[0]


@pytest.mark.fast
def test_group_by_axes(tmp_path: Path) -> None:
    rows = collect(_build_study(tmp_path))
    groups = group_by(rows, ("case", "arch", "budget"))
    assert len(groups) == 2  # A1 and A2 cells
    for cell in groups.values():
        assert len(cell) == 3


@pytest.mark.fast
def test_aggregate_cell_mean_std(tmp_path: Path) -> None:
    rows = collect(_build_study(tmp_path))
    a1_rows = [r for r in rows if r["arch"] == "A1"]
    agg = aggregate_cell(a1_rows, "metric_rmse_zb")
    assert agg.n == 3
    assert agg.mean == pytest.approx((0.020 + 0.021 + 0.022) / 3.0)
    assert agg.std > 0.0
    assert agg.ci_low <= agg.mean <= agg.ci_high


@pytest.mark.fast
def test_aggregate_cell_handles_single() -> None:
    agg = aggregate_cell([{"metric_rmse_zb": 0.05}], "metric_rmse_zb")
    assert agg.n == 1
    assert agg.mean == 0.05
    assert agg.std == 0.0


@pytest.mark.fast
def test_aggregate_cell_handles_empty() -> None:
    agg = aggregate_cell([], "metric_rmse_zb")
    assert agg.n == 0


@pytest.mark.fast
def test_wilcoxon_pairs_detects_a1_vs_a2(tmp_path: Path) -> None:
    rows = collect(_build_study(tmp_path))
    pairs = wilcoxon_pairs(rows, axis="arch", value_key="metric_rmse_zb", pair_axis="seed")
    assert len(pairs) == 1
    p = pairs[0]
    assert {p.a, p.b} == {"A1", "A2"}
    assert p.n == 3
    # A1 is systematically larger by 0.005 -> nonzero effect size.
    assert abs(p.effect_size) > 0.0


@pytest.mark.fast
def test_format_text_table_has_header_and_rows(tmp_path: Path) -> None:
    rows = collect(_build_study(tmp_path))
    table = format_text_table(rows, metric_key="metric_rmse_zb")
    lines = table.splitlines()
    assert "case" in lines[0] and "arch" in lines[0]
    # 1 header + 1 separator + 2 group rows
    assert len(lines) == 4


@pytest.mark.fast
def test_emit_latex_table_returns_tabular(tmp_path: Path) -> None:
    rows = collect(_build_study(tmp_path))
    tex = emit_latex_table(
        rows,
        metric_key="metric_rmse_zb",
        caption="Test caption",
        label="tab:test",
    )
    assert r"\begin{tabular}" in tex
    assert r"\caption{Test caption}" in tex
    assert r"\label{tab:test}" in tex
    assert r"A1" in tex and r"A2" in tex
