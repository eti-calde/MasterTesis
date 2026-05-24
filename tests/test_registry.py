"""Tests for pinn_bath.registry (run_id, idempotency, manifest)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinn_bath.config import DataCfg, RunConfig
from pinn_bath.registry import (
    Manifest,
    ManifestEntry,
    Registry,
    run_dir_for,
    run_id_for,
)


def _cfg(seed: int = 0, arch: str = "A1", budget: str = "small") -> RunConfig:
    return RunConfig(
        case="exp1",
        arch=arch,
        budget=budget,
        seed=seed,
        data=DataCfg(case_path="dummy.npz"),
    )


@pytest.mark.fast
def test_run_id_is_deterministic_and_12_chars() -> None:
    cfg = _cfg(seed=0)
    rid = run_id_for(cfg)
    assert len(rid) == 12
    assert rid == run_id_for(_cfg(seed=0))


@pytest.mark.fast
def test_run_id_changes_with_any_axis() -> None:
    base = run_id_for(_cfg())
    assert base != run_id_for(_cfg(seed=1))
    assert base != run_id_for(_cfg(arch="A2"))
    assert base != run_id_for(_cfg(budget="medium"))


@pytest.mark.fast
def test_run_dir_path(tmp_path: Path) -> None:
    cfg = _cfg()
    assert run_dir_for(tmp_path / "study", cfg) == tmp_path / "study" / run_id_for(cfg)


# --- Manifest ---------------------------------------------------------------


@pytest.mark.fast
def test_manifest_appends_jsonl(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.jsonl")
    m.append(
        ManifestEntry(
            run_id="abc",
            status="started",
            ts=1.0,
            case="exp1",
            arch="A1",
            budget="small",
            seed=0,
            form="primitive",
        )
    )
    m.append(
        ManifestEntry(
            run_id="abc",
            status="ok",
            ts=2.0,
            case="exp1",
            arch="A1",
            budget="small",
            seed=0,
            form="primitive",
            wall_time_s=10.5,
        )
    )
    lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["status"] == "started"
    assert json.loads(lines[1])["status"] == "ok"


@pytest.mark.fast
def test_manifest_latest_status(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.jsonl")
    m.append(
        ManifestEntry(
            run_id="a",
            status="started",
            ts=1.0,
            case="exp1",
            arch="A1",
            budget="small",
            seed=0,
            form="primitive",
        )
    )
    m.append(
        ManifestEntry(
            run_id="a",
            status="ok",
            ts=2.0,
            case="exp1",
            arch="A1",
            budget="small",
            seed=0,
            form="primitive",
        )
    )
    m.append(
        ManifestEntry(
            run_id="b",
            status="started",
            ts=1.5,
            case="exp1",
            arch="A1",
            budget="small",
            seed=0,
            form="primitive",
        )
    )
    assert m.latest_status("a") == "ok"
    assert m.latest_status("b") == "started"
    assert m.latest_status("never_seen") is None


# --- Registry decisions ----------------------------------------------------


@pytest.mark.fast
def test_decide_run_when_empty(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "study")
    cfg = _cfg()
    d = reg.decide(cfg)
    assert d.action == "run"
    assert d.run_id == run_id_for(cfg)


@pytest.mark.fast
def test_decide_skip_when_summary_ok(tmp_path: Path) -> None:
    cfg = _cfg()
    rdir = run_dir_for(tmp_path / "study", cfg)
    rdir.mkdir(parents=True)
    (rdir / "summary.json").write_text(json.dumps({"status": "ok"}))
    reg = Registry(tmp_path / "study")
    assert reg.decide(cfg).action == "skip"


@pytest.mark.fast
def test_decide_skip_when_diverged(tmp_path: Path) -> None:
    cfg = _cfg()
    rdir = run_dir_for(tmp_path / "study", cfg)
    rdir.mkdir(parents=True)
    (rdir / "summary.json").write_text(json.dumps({"status": "diverged"}))
    reg = Registry(tmp_path / "study")
    d = reg.decide(cfg)
    assert d.action == "skip"
    assert "diverged" in d.reason


@pytest.mark.fast
def test_decide_resume_when_checkpoint_present(tmp_path: Path) -> None:
    cfg = _cfg()
    rdir = run_dir_for(tmp_path / "study", cfg)
    (rdir / "checkpoints").mkdir(parents=True)
    (rdir / "checkpoints" / "last.pt").write_bytes(b"x")
    # status was "interrupted" (or absent).
    (rdir / "summary.json").write_text(json.dumps({"status": "interrupted"}))
    reg = Registry(tmp_path / "study")
    assert reg.decide(cfg).action == "resume"


@pytest.mark.fast
def test_plan_for_grid(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "study")
    grid = [_cfg(seed=i) for i in range(4)]
    decisions = reg.plan(grid)
    assert {d.action for d in decisions} == {"run"}
    assert {d.run_id for d in decisions} == {run_id_for(c) for c in grid}


@pytest.mark.fast
def test_mark_started_and_finished(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "study")
    cfg = _cfg()
    reg.mark_started(cfg, machine="laptop")
    reg.mark_finished(cfg, status="ok", wall_time_s=12.3, final_loss=0.05)
    rows = reg.manifest.rows()
    assert len(rows) == 2
    assert rows[0]["status"] == "started"
    assert rows[1]["status"] == "ok"
    assert rows[1]["wall_time_s"] == 12.3
    assert reg.manifest.latest_status(run_id_for(cfg)) == "ok"
