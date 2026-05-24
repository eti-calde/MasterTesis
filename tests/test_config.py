"""Tests for pinn_bath.config."""

import pytest
from pydantic import ValidationError

from pinn_bath.config import DataCfg, RunConfig


def _make(**overrides) -> RunConfig:
    base = dict(
        case="exp1",
        arch="A1",
        budget="small",
        data=DataCfg(case_path="x.npz"),
    )
    base.update(overrides)
    return RunConfig(**base)


@pytest.mark.fast
def test_run_id_is_12_chars() -> None:
    cfg = _make()
    assert len(cfg.run_id) == 12


@pytest.mark.fast
def test_run_id_is_deterministic() -> None:
    assert _make(seed=0).run_id == _make(seed=0).run_id


@pytest.mark.fast
def test_run_id_changes_with_seed() -> None:
    assert _make(seed=0).run_id != _make(seed=1).run_id


@pytest.mark.fast
def test_run_id_changes_with_arch() -> None:
    assert _make(arch="A1").run_id != _make(arch="A2").run_id


@pytest.mark.fast
def test_config_rejects_unknown_arch() -> None:
    with pytest.raises(ValidationError):
        _make(arch="A4")


@pytest.mark.fast
def test_config_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        RunConfig(
            case="exp1",
            arch="A1",
            budget="small",
            data=DataCfg(case_path="x.npz"),
            unknown_field=1,  # type: ignore[call-arg]
        )


@pytest.mark.fast
def test_yaml_roundtrip(tmp_path) -> None:
    cfg = _make()
    p = tmp_path / "config.yaml"
    cfg.to_yaml(p)
    cfg2 = RunConfig.from_yaml(p)
    assert cfg == cfg2
