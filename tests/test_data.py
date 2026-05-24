"""Tests for pinn_bath.data.Case (load/save/sample)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_bath.data import Case, CaseMetadata

# --- Fixtures ---------------------------------------------------------------


def _meta_1d_steady() -> CaseMetadata:
    return CaseMetadata(
        case_id="test_1d_steady",
        spatial_dim=1,
        has_t=False,
        bc_type="open_dirichlet",
        constants={"g": 9.81, "q": 4.42, "h_down": 2.0},
        domain={"x": [-10.0, 10.0]},
        gt_source="analytical_bernoulli",
    )


def _meta_1d_transient() -> CaseMetadata:
    return CaseMetadata(
        case_id="test_1d_trans",
        spatial_dim=1,
        has_t=True,
        bc_type="closed",
        constants={"g": 9.81, "h0": 0.5, "a": 1.0},
        domain={"x": [-2.0, 2.0], "t": [0.0, 2.0]},
        gt_source="analytical_thacker",
    )


def _meta_2d_transient() -> CaseMetadata:
    return CaseMetadata(
        case_id="test_2d_trans",
        spatial_dim=2,
        has_t=True,
        bc_type="periodic",
        constants={"g": 9.81},
        domain={"x": [-2.0, 2.0], "y": [-2.0, 2.0], "t": [0.0, 0.5]},
        gt_source="fv_hll",
    )


def _build_case_1d_steady() -> Case:
    x = np.linspace(-10.0, 10.0, 41)
    zb = np.maximum(0.0, 0.2 - (0.2 / 4.0) * x**2)
    h = 2.0 - zb
    u = 4.42 / h
    eta = h + zb
    return Case(
        metadata=_meta_1d_steady(),
        coords={"x": x},
        fields={"h": h, "u": u, "zb": zb, "eta": eta},
    )


def _build_case_1d_trans() -> Case:
    x = np.linspace(-2.0, 2.0, 21)
    t = np.linspace(0.0, 2.0, 11)
    Nt, Nx = t.size, x.size
    zb = 0.5 * (x**2 - 1.0)
    h = np.tile(np.maximum(0.0, 0.5 - zb), (Nt, 1))
    u = np.zeros((Nt, Nx))
    eta = h + zb[None, :]
    return Case(
        metadata=_meta_1d_transient(),
        coords={"x": x, "t": t},
        fields={"h": h, "u": u, "zb": zb, "eta": eta},
    )


def _build_case_2d_trans() -> Case:
    x = np.linspace(-2.0, 2.0, 11)
    y = np.linspace(-2.0, 2.0, 11)
    t = np.linspace(0.0, 0.5, 6)
    Nt, Ny, Nx = t.size, y.size, x.size
    X, Y = np.meshgrid(x, y)
    zb = 1.0 + 0.01 * np.cos(np.pi * X / 2.0) * np.cos(np.pi * Y / 2.0)
    h = np.broadcast_to(zb[None, :, :], (Nt, Ny, Nx)).copy()
    u = np.zeros((Nt, Ny, Nx))
    v = np.zeros((Nt, Ny, Nx))
    eta = h + zb[None, :, :]
    return Case(
        metadata=_meta_2d_transient(),
        coords={"x": x, "y": y, "t": t},
        fields={"h": h, "u": u, "v": v, "zb": zb, "eta": eta},
    )


# --- Validation -------------------------------------------------------------


@pytest.mark.fast
def test_case_validates_ok_for_supported_kinds() -> None:
    for case in (_build_case_1d_steady(), _build_case_1d_trans(), _build_case_2d_trans()):
        case._validate()  # type: ignore[attr-defined]


@pytest.mark.fast
def test_case_rejects_missing_coord() -> None:
    case = _build_case_1d_trans()
    case.coords.pop("t")
    with pytest.raises(ValueError, match="missing coord 't'"):
        case._validate()  # type: ignore[attr-defined]


@pytest.mark.fast
def test_case_rejects_missing_field() -> None:
    case = _build_case_2d_trans()
    case.fields.pop("v")
    with pytest.raises(ValueError, match="missing field 'v'"):
        case._validate()  # type: ignore[attr-defined]


@pytest.mark.fast
def test_case_rejects_domain_with_unknown_axis() -> None:
    case = _build_case_1d_steady()
    case.metadata.domain["t"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="domain has axis 't'"):
        case._validate()  # type: ignore[attr-defined]


# --- Save / load roundtrip --------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize(
    "builder, name",
    [
        (_build_case_1d_steady, "1d_steady"),
        (_build_case_1d_trans, "1d_trans"),
        (_build_case_2d_trans, "2d_trans"),
    ],
)
def test_case_save_load_roundtrip(builder, name, tmp_path: Path) -> None:
    case = builder()
    out = tmp_path / f"{name}.npz"
    case.save(out)
    assert out.exists()
    loaded = Case.load(out)
    assert loaded.metadata.case_id == case.metadata.case_id
    assert loaded.metadata.spatial_dim == case.metadata.spatial_dim
    assert loaded.metadata.has_t == case.metadata.has_t
    assert loaded.metadata.bc_type == case.metadata.bc_type
    assert loaded.metadata.constants == case.metadata.constants
    for axis in case.coords:
        np.testing.assert_array_equal(loaded.coords[axis], case.coords[axis])
    for field in case.fields:
        np.testing.assert_array_equal(loaded.fields[field], case.fields[field])
    assert loaded.file_hash is not None
    assert loaded.source_path == out


@pytest.mark.fast
def test_load_fails_without_metadata(tmp_path: Path) -> None:
    p = tmp_path / "bare.npz"
    np.savez(p, x=np.linspace(0.0, 1.0, 5))
    with pytest.raises(ValueError, match="missing 'metadata_json'"):
        Case.load(p)


@pytest.mark.fast
def test_load_fails_when_path_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Case.load(tmp_path / "does_not_exist.npz")


@pytest.mark.fast
def test_save_is_atomic(tmp_path: Path) -> None:
    """After save, no stray .tmp file remains."""
    case = _build_case_1d_steady()
    out = tmp_path / "atomic.npz"
    case.save(out)
    assert not (tmp_path / "atomic.npz.tmp").exists()


# --- Eval grid --------------------------------------------------------------


@pytest.mark.fast
def test_eval_grid_1d_steady_shapes() -> None:
    case = _build_case_1d_steady()
    coords, fields = case.eval_grid()
    n = case.coords["x"].size
    assert coords["x"].shape == (n, 1)
    assert fields["h"].shape == (n, 1)


@pytest.mark.fast
def test_eval_grid_2d_transient_shapes() -> None:
    case = _build_case_2d_trans()
    coords, fields = case.eval_grid()
    n_total = case.coords["x"].size * case.coords["y"].size * case.coords["t"].size
    for axis in ("x", "y", "t"):
        assert coords[axis].shape == (n_total, 1)
    for field in ("h", "u", "v", "zb"):
        assert fields[field].shape == (n_total, 1)


# --- Observation sampler ----------------------------------------------------


@pytest.mark.fast
def test_sample_observations_deterministic() -> None:
    case = _build_case_2d_trans()
    a = case.sample_observations(seed=42, n_obs=20)
    b = case.sample_observations(seed=42, n_obs=20)
    for k in a:
        assert torch.equal(a[k], b[k])


@pytest.mark.fast
def test_sample_observations_different_seeds_differ() -> None:
    case = _build_case_1d_trans()
    a = case.sample_observations(seed=1, n_obs=15)
    b = case.sample_observations(seed=2, n_obs=15)
    assert not torch.equal(a["index"], b["index"])


@pytest.mark.fast
def test_sample_observations_returns_requested_fields() -> None:
    case = _build_case_2d_trans()
    out = case.sample_observations(seed=0, n_obs=8, fields=("eta", "u", "v"))
    for k in ("x", "y", "t", "eta", "u", "v", "index"):
        assert k in out, f"missing {k}"
        assert out[k].shape == (8, 1)


@pytest.mark.fast
def test_sample_observations_noise_injection() -> None:
    case = _build_case_1d_steady()
    clean = case.sample_observations(seed=0, n_obs=20, fields=("h",), noise_std=0.0)
    noisy = case.sample_observations(seed=0, n_obs=20, fields=("h",), noise_std=0.1)
    # Coordinates identical at the same seed, but values differ in the noisy case
    assert torch.equal(clean["x"], noisy["x"])
    assert not torch.equal(clean["h"], noisy["h"])


@pytest.mark.fast
def test_sample_observations_eta_derivable_when_missing() -> None:
    case = _build_case_1d_steady()
    case.fields.pop("eta")
    out = case.sample_observations(seed=0, n_obs=5, fields=("eta",))
    assert "eta" in out


@pytest.mark.fast
def test_sample_observations_raises_when_too_many() -> None:
    case = _build_case_1d_steady()
    with pytest.raises(ValueError, match="> grid size"):
        case.sample_observations(seed=0, n_obs=10_000)


# --- Collocation sampler ----------------------------------------------------


@pytest.mark.fast
def test_sample_collocation_in_domain() -> None:
    case = _build_case_2d_trans()
    coll = case.sample_collocation(seed=0, n_coll=100)
    for axis, lo_hi in case.metadata.domain.items():
        lo, hi = lo_hi
        v = coll[axis].detach()
        assert (v >= lo).all() and (v <= hi).all(), f"{axis} out of [{lo}, {hi}]"


@pytest.mark.fast
def test_sample_collocation_requires_grad_by_default() -> None:
    case = _build_case_1d_trans()
    coll = case.sample_collocation(seed=0, n_coll=10)
    for v in coll.values():
        assert v.requires_grad


@pytest.mark.fast
def test_sample_collocation_no_grad_when_off() -> None:
    case = _build_case_1d_trans()
    coll = case.sample_collocation(seed=0, n_coll=10, requires_grad=False)
    for v in coll.values():
        assert not v.requires_grad


@pytest.mark.fast
def test_sample_collocation_deterministic() -> None:
    case = _build_case_2d_trans()
    a = case.sample_collocation(seed=7, n_coll=20, requires_grad=False)
    b = case.sample_collocation(seed=7, n_coll=20, requires_grad=False)
    for k in a:
        assert torch.equal(a[k], b[k])
