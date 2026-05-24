"""Tests for pinn_bath.losses.bc.periodic_bc_loss."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinn_bath.data import Case, CaseMetadata
from pinn_bath.losses import flat_bed_loss, periodic_bc_loss

# --- Mock models -----------------------------------------------------------


class _ConstZbModel(nn.Module):
    """Mock model whose every field is a constant times a parameter.

    Has ``h, u, [v,] zb`` outputs that do not depend on coords -> trivially
    periodic. Used to check the loss vanishes on a periodic model.
    """

    def __init__(self, spatial_dim: int, has_v: bool = False) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.has_v = has_v
        self.const = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        n = next(iter(coords.values())).shape[0]
        c = self.const.expand(n, 1)
        out = {"h": c, "u": c, "zb": c}
        if self.has_v:
            out["v"] = c
        return out


class _LinearXModel(nn.Module):
    """Mock model whose h, u, zb are equal to x (NOT periodic in x)."""

    def __init__(self, spatial_dim: int, has_v: bool = False) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.has_v = has_v
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = coords["x"]
        out = {"h": self.scale * x, "u": self.scale * x, "zb": self.scale * x}
        if self.has_v:
            out["v"] = self.scale * x
        return out


class _PeriodicCosModel(nn.Module):
    """Mock model with output cos(pi x / L_half) -- exactly periodic on [-L, L]."""

    def __init__(self, L_half: float = 2.0, spatial_dim: int = 2, has_v: bool = True) -> None:
        super().__init__()
        self.L_half = L_half
        self.spatial_dim = spatial_dim
        self.has_v = has_v
        self.amp = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = coords["x"]
        f = self.amp * torch.cos(math.pi * x / self.L_half)
        out = {"h": 1.0 + f, "u": f, "zb": f}
        if self.has_v:
            out["v"] = f
        if self.spatial_dim == 2 and "y" in coords:
            g = self.amp * torch.cos(math.pi * coords["y"] / self.L_half)
            out["h"] = 1.0 + f + g
            out["u"] = f + g
            if self.has_v:
                out["v"] = f + g
            out["zb"] = f + g
        return out


# --- Cases ----------------------------------------------------------------


def _case_1d_periodic() -> Case:
    x = np.linspace(-2.0, 2.0, 11)
    zb = np.zeros_like(x)
    h = np.ones_like(x)
    u = np.zeros_like(x)
    return Case(
        metadata=CaseMetadata(
            case_id="t1d",
            spatial_dim=1,
            has_t=True,
            bc_type="periodic",
            constants={},
            domain={"x": [-2.0, 2.0], "t": [0.0, 0.5]},
            gt_source="fv_hll",
        ),
        coords={"x": x, "t": np.linspace(0.0, 0.5, 6)},
        fields={
            "h": np.tile(h, (6, 1)),
            "u": np.tile(u, (6, 1)),
            "zb": zb,
            "eta": np.tile(h, (6, 1)),
        },
    )


def _case_2d_periodic() -> Case:
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
        metadata=CaseMetadata(
            case_id="t2d",
            spatial_dim=2,
            has_t=True,
            bc_type="periodic",
            constants={},
            domain={"x": [-2.0, 2.0], "y": [-2.0, 2.0], "t": [0.0, 0.5]},
            gt_source="fv_hll",
        ),
        coords={"x": x, "y": y, "t": t},
        fields={"h": h, "u": u, "v": v, "zb": zb, "eta": eta},
    )


# --- Behavior --------------------------------------------------------------


@pytest.mark.fast
def test_periodic_loss_zero_for_constant_model() -> None:
    case = _case_1d_periodic()
    model = _ConstZbModel(spatial_dim=1)
    loss = periodic_bc_loss(model, case, n_bc=64, seed=0)
    assert float(loss.item()) == pytest.approx(0.0)


@pytest.mark.fast
def test_periodic_loss_positive_for_linear_model() -> None:
    case = _case_1d_periodic()
    model = _LinearXModel(spatial_dim=1)
    loss = periodic_bc_loss(model, case, n_bc=64, seed=0)
    # f(lo) - f(hi) = (-2) - 2 = -4 for each of h, u, zb -> mean(16) summed
    # over 3 fields = 48.
    assert float(loss.item()) == pytest.approx(48.0, rel=1.0e-9)


@pytest.mark.fast
def test_periodic_loss_zero_for_cosine_model_2d() -> None:
    case = _case_2d_periodic()
    model = _PeriodicCosModel(L_half=2.0, spatial_dim=2, has_v=True)
    loss = periodic_bc_loss(model, case, n_bc=64, seed=0)
    # cos(pi * (-2) / 2) = cos(-pi) = -1; cos(pi * 2 / 2) = cos(pi) = -1.
    # Differences vanish exactly.
    assert float(loss.item()) == pytest.approx(0.0, abs=1.0e-14)


@pytest.mark.fast
def test_periodic_loss_includes_zb_when_requested() -> None:
    case = _case_1d_periodic()
    model = _LinearXModel(spatial_dim=1)
    loss_with_zb = periodic_bc_loss(model, case, n_bc=32, seed=0, include_zb=True)
    loss_no_zb = periodic_bc_loss(model, case, n_bc=32, seed=0, include_zb=False)
    # With zb included, h + u + zb each contribute 16 -> 48 total.
    # Without zb, only h + u -> 32 total.
    assert float(loss_with_zb.item()) == pytest.approx(48.0, rel=1.0e-9)
    assert float(loss_no_zb.item()) == pytest.approx(32.0, rel=1.0e-9)


@pytest.mark.fast
def test_periodic_loss_deterministic_by_seed() -> None:
    case = _case_2d_periodic()
    model = _LinearXModel(spatial_dim=2, has_v=True)
    a = periodic_bc_loss(model, case, n_bc=32, seed=42)
    b = periodic_bc_loss(model, case, n_bc=32, seed=42)
    assert float(a.item()) == float(b.item())


@pytest.mark.fast
def test_periodic_loss_changes_with_seed() -> None:
    """If the model's boundary difference depends on the random (y, t)
    samples, two seeds must produce different loss values."""

    class _ProductModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        def forward(self, coords):
            x = coords["x"]
            y = coords.get("y", torch.zeros_like(x))
            t = coords.get("t", torch.zeros_like(x))
            f = self.scale * x * y * t
            return {"h": 1.0 + f, "u": f, "v": f, "zb": f}

    case = _case_2d_periodic()
    model = _ProductModel()
    a = float(periodic_bc_loss(model, case, n_bc=32, seed=0).item())
    b = float(periodic_bc_loss(model, case, n_bc=32, seed=1).item())
    assert a != b
    assert a > 0.0 and b > 0.0


@pytest.mark.fast
def test_periodic_loss_2d_uses_both_axes() -> None:
    """If only x were periodic-tested, swapping y bounds would not affect loss."""
    case = _case_2d_periodic()

    # Model is anti-symmetric in y: u depends on y as well.
    class _AntiYModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        def forward(self, coords):
            y = coords["y"]
            x_zero = torch.zeros_like(y)
            return {"h": 1.0 + x_zero, "u": self.scale * y, "zb": x_zero, "v": x_zero}

    model = _AntiYModel()
    loss = float(periodic_bc_loss(model, case, n_bc=64, seed=0).item())
    # x-axis pair: u is same on both x_lo and x_hi (same y) -> 0.
    # y-axis pair: u(y=-2) - u(y=2) = -4 -> mean(16) for u, plus 0 for h, v, zb.
    # Total = 16.
    assert loss == pytest.approx(16.0, rel=1.0e-9)


@pytest.mark.fast
def test_periodic_loss_gradient_flows_to_model() -> None:
    case = _case_1d_periodic()
    model = _LinearXModel(spatial_dim=1)
    loss = periodic_bc_loss(model, case, n_bc=16, seed=0)
    loss.backward()
    assert model.scale.grad is not None
    assert float(model.scale.grad.abs().item()) > 0.0


# --- flat_bed_loss --------------------------------------------------------


def _case_1d_open_dirichlet(x_0: float = 0.0, w: float = 2.0) -> Case:
    """Mock 1D open_dirichlet case with bump support (x_0, w)."""
    x = np.linspace(-10.0, 10.0, 21)
    zb = np.where(np.abs(x - x_0) < w, 0.05, 0.0)
    h = 2.0 - zb
    u = 4.42 / h
    return Case(
        metadata=CaseMetadata(
            case_id="t1d_open",
            spatial_dim=1,
            has_t=False,
            bc_type="open_dirichlet",
            constants={"g": 9.81, "q": 4.42, "h_down": 2.0, "x_0": x_0, "w": w},
            domain={"x": [-10.0, 10.0]},
            gt_source="analytical_bernoulli",
        ),
        coords={"x": x},
        fields={"h": h, "u": u, "zb": zb, "eta": h + zb},
    )


class _ConstantZbModel(nn.Module):
    """Model whose z_b is a constant trainable parameter."""

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.const = nn.Parameter(torch.tensor(value, dtype=torch.float64))

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        n = next(iter(coords.values())).shape[0]
        c = self.const.expand(n, 1)
        return {"h": torch.ones(n, 1, dtype=c.dtype), "u": c, "zb": c}


class _BumpOnlyModel(nn.Module):
    """Model whose z_b is nonzero only inside |x - x0| < w (analytic)."""

    def __init__(self, x0: float = 0.0, w: float = 2.0, peak: float = 1.0) -> None:
        super().__init__()
        self.x0 = x0
        self.w = w
        self.peak = nn.Parameter(torch.tensor(peak, dtype=torch.float64))

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = coords["x"]
        inside = (x - self.x0).abs() < self.w
        zb = torch.where(inside, self.peak.expand_as(x), torch.zeros_like(x))
        return {"h": torch.ones_like(x), "u": torch.zeros_like(x), "zb": zb}


@pytest.mark.fast
def test_flat_bed_loss_zero_for_zb_zero() -> None:
    case = _case_1d_open_dirichlet()
    model = _ConstantZbModel(value=0.0)
    loss = flat_bed_loss(model, case, n_pts=64, seed=0)
    assert float(loss.item()) == pytest.approx(0.0)


@pytest.mark.fast
def test_flat_bed_loss_squares_constant_offset() -> None:
    case = _case_1d_open_dirichlet()
    model = _ConstantZbModel(value=0.1)
    loss = flat_bed_loss(model, case, n_pts=64, seed=0)
    # zb=0.1 everywhere -> mean(0.01) = 0.01.
    assert float(loss.item()) == pytest.approx(0.01, rel=1.0e-9)


@pytest.mark.fast
def test_flat_bed_loss_ignores_bump_interior() -> None:
    """A model that only ever puts mass inside the bump support should incur no
    flat-bed penalty (loss samples only the flat region)."""
    case = _case_1d_open_dirichlet(x_0=0.0, w=2.0)
    model = _BumpOnlyModel(x0=0.0, w=2.0, peak=1.0)
    loss = flat_bed_loss(model, case, n_pts=128, seed=0)
    assert float(loss.item()) == pytest.approx(0.0)


@pytest.mark.fast
def test_flat_bed_loss_deterministic_by_seed() -> None:
    case = _case_1d_open_dirichlet()
    model = _ConstantZbModel(value=0.05)
    a = flat_bed_loss(model, case, n_pts=32, seed=7)
    b = flat_bed_loss(model, case, n_pts=32, seed=7)
    assert float(a.item()) == float(b.item())


@pytest.mark.fast
def test_flat_bed_loss_reads_constants_from_case() -> None:
    """Changing x_0 / w in metadata should change which samples count as 'flat'."""
    case_centered = _case_1d_open_dirichlet(x_0=0.0, w=2.0)
    case_offset = _case_1d_open_dirichlet(x_0=5.0, w=2.0)
    model = _BumpOnlyModel(x0=0.0, w=2.0, peak=1.0)
    loss_centered = float(flat_bed_loss(model, case_centered, n_pts=128, seed=0).item())
    loss_offset = float(flat_bed_loss(model, case_offset, n_pts=128, seed=0).item())
    # When bump support matches model (centered), all model mass is "inside" -> 0.
    assert loss_centered == pytest.approx(0.0)
    # When support is shifted to x_0=5, the model's bump mass at |x|<2 lies in
    # the "flat" region of the case -> positive loss.
    assert loss_offset > 0.0


@pytest.mark.fast
def test_flat_bed_loss_raises_on_missing_keys() -> None:
    case = _case_1d_open_dirichlet()
    del case.metadata.constants["x_0"]
    with pytest.raises(KeyError, match="x_0"):
        flat_bed_loss(_ConstantZbModel(0.0), case, n_pts=16, seed=0)


@pytest.mark.fast
def test_flat_bed_loss_rejects_non_1d() -> None:
    from pinn_bath.data import CaseMetadata

    case = Case(
        metadata=CaseMetadata(
            case_id="t2d",
            spatial_dim=2,
            has_t=False,
            bc_type="open_dirichlet",
            constants={"x_0": 0.0, "w": 1.0},
            domain={"x": [-1, 1], "y": [-1, 1]},
            gt_source="fv_hll",
        ),
        coords={"x": np.linspace(-1, 1, 5), "y": np.linspace(-1, 1, 5)},
        fields={
            "h": np.ones((5, 5)),
            "u": np.zeros((5, 5)),
            "v": np.zeros((5, 5)),
            "zb": np.zeros((5, 5)),
        },
    )
    with pytest.raises(NotImplementedError, match="1D"):
        flat_bed_loss(_ConstantZbModel(0.0), case, n_pts=16, seed=0)
