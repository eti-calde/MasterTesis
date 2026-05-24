"""Tests for pinn_bath.tracking.RunRecorder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinn_bath.config import DataCfg, RunConfig
from pinn_bath.tracking import RunRecorder


def _make_cfg() -> RunConfig:
    return RunConfig(
        case="exp1",
        arch="A1",
        budget="small",
        data=DataCfg(case_path="dummy.npz"),
    )


@pytest.mark.fast
def test_recorder_creates_env_and_config(tmp_path: Path) -> None:
    cfg = _make_cfg()
    with RunRecorder(tmp_path / "run", cfg=cfg):
        pass
    run_dir = tmp_path / "run"
    assert (run_dir / "env.json").exists()
    assert (run_dir / "config.yaml").exists()
    env = json.loads((run_dir / "env.json").read_text())
    assert "torch" in env
    assert "python" in env


@pytest.mark.fast
def test_log_epoch_appends_jsonl(tmp_path: Path) -> None:
    with RunRecorder(tmp_path / "run") as rec:
        rec.log_epoch(epoch=0, phase="adam", L_total=1.0)
        rec.log_epoch(epoch=1, phase="adam", L_total=0.5)
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    a = json.loads(lines[0])
    b = json.loads(lines[1])
    assert a["epoch"] == 0 and a["L_total"] == 1.0 and a["phase"] == "adam"
    assert b["epoch"] == 1 and b["L_total"] == 0.5
    assert "t_elapsed_s" in a


@pytest.mark.fast
def test_heartbeat_is_atomic_and_overwrites(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path / "run")
    rec.heartbeat(epoch=10, phase="adam", L_total=0.7)
    rec.heartbeat(epoch=20, phase="adam", L_total=0.4)
    rec.close()
    hb = json.loads((tmp_path / "run" / "heartbeat.json").read_text())
    assert hb["epoch"] == 20
    assert hb["L_total"] == 0.4
    # No stray tmp file
    leftover = list((tmp_path / "run").glob(".heartbeat.*.json"))
    assert leftover == []


@pytest.mark.fast
def test_write_summary_includes_status_and_runtime(tmp_path: Path) -> None:
    cfg = _make_cfg()
    rec = RunRecorder(tmp_path / "run", cfg=cfg)
    rec.write_summary(status="ok", final_RMSE=0.05)
    rec.close()
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert "wall_time_s" in summary
    assert summary["final_RMSE"] == 0.05
    assert summary["run_id"] == cfg.run_id
    assert summary["arch"] == "A1"


@pytest.mark.fast
def test_context_manager_closes(tmp_path: Path) -> None:
    with RunRecorder(tmp_path / "run") as rec:
        rec.log_epoch(epoch=0, phase="adam", L_total=1.0)
    # After context: file handle closed; no errors.
    assert (tmp_path / "run" / "metrics.jsonl").exists()
