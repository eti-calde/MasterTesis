"""Frozen legacy network blocks for the Experiments 01/02/03/05
``pinn_inverse.py`` orchestration files.

Each class is moved **verbatim** from its source experiment to preserve
the exact ``torch.randn`` consumption order: given a fixed
``torch.manual_seed``, instantiating one of these from this module
produces a ``state_dict`` byte-identical to instantiating the
pre-refactor local definition. This is verified by
``tests/test_legacy_dedup.py``.

**Do NOT use these for new work.** The canonical pipeline is
``pinn_bath.models`` (``A1TwoNets`` etc.), which seeds its Fourier
matrices and uses different default sigmas. These legacy classes exist
solely so the four legacy ``pinn_inverse.py`` files share one source of
truth — without disturbing the historical RMSE baselines that the
thesis cites.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "BathymetryNet2D_Exp03",
    "BathymetryNet_Exp01",
    "BathymetryNet_Exp02",
    "BathymetryNet_Exp05",
    "FourierFeatures",
    "SolutionNet2D_Exp03",
    "SolutionNet_Exp01",
    "SolutionNet_Exp02",
    "SolutionNet_Exp05",
    "plot_results_1d_steady",
    "plot_results_2d",
]


# ============================================================
# FourierFeatures — literally identical across all 4 legacy files
# ============================================================


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding to mitigate spectral bias.

    Maps ``x -> [sin(2*pi*B*x), cos(2*pi*B*x)]`` where ``B`` is a random
    matrix sampled at construction time. Reference: Tancik et al. 2020.

    NOTE: ``B`` is sampled with an unseeded ``torch.randn`` call —
    determinism is controlled by ``torch.manual_seed(...)`` from the
    caller. This matches the legacy behavior exactly.
    """

    def __init__(self, n_features=16, sigma=2.0, in_dim=1):
        super().__init__()
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ============================================================
# Exp 01 — 1D steady (subcritical bump)
# ============================================================


class SolutionNet_Exp01(nn.Module):
    """Maps x -> (h, u). Learns the flow field."""

    def __init__(
        self,
        n_hidden=4,
        n_neurons=64,
        activation=nn.Tanh,
        use_fourier=True,
        fourier_features=16,
        fourier_sigma=1.0,
    ):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeatures(n_features=fourier_features, sigma=fourier_sigma)
            in_dim = 2 * fourier_features
        else:
            in_dim = 1
        layers = [nn.Linear(in_dim, n_neurons), activation()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), activation()]
        layers.append(nn.Linear(n_neurons, 2))  # output: (h, u)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.fourier(x) if self.use_fourier else x
        out = self.net(feat)
        h = nn.functional.softplus(out[:, 0:1])  # h > 0 always
        u = out[:, 1:2]
        return h, u


class BathymetryNet_Exp01(nn.Module):
    """Maps x -> z_b. Separate small network for the unknown bathymetry."""

    def __init__(
        self,
        n_hidden=3,
        n_neurons=32,
        activation=nn.Tanh,
        use_fourier=True,
        fourier_features=16,
        fourier_sigma=1.0,
    ):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeatures(n_features=fourier_features, sigma=fourier_sigma)
            in_dim = 2 * fourier_features
        else:
            in_dim = 1
        layers = [nn.Linear(in_dim, n_neurons), activation()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), activation()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.fourier(x) if self.use_fourier else x
        return self.net(feat)


# ============================================================
# Exp 02 — 1D transient (Thacker basin)
# ============================================================


class SolutionNet_Exp02(nn.Module):
    """(x, t) -> (h, u). Uses separate Fourier features for x and t."""

    def __init__(
        self,
        n_hidden=5,
        n_neurons=96,
        fourier_features=24,
        fourier_sigma_x=2.0,
        fourier_sigma_t=2.0,
    ):
        super().__init__()
        self.fx = FourierFeatures(fourier_features, fourier_sigma_x, in_dim=1)
        self.ft = FourierFeatures(fourier_features, fourier_sigma_t, in_dim=1)
        in_dim = 2 * fourier_features * 2  # x feats + t feats, each 2*n

        layers = [nn.Linear(in_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 2))  # (h_raw, u)
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        fx = self.fx(x)
        ft = self.ft(t)
        feat = torch.cat([fx, ft], dim=-1)
        out = self.net(feat)
        h = nn.functional.softplus(out[:, 0:1])  # h >= 0
        u = out[:, 1:2]
        return h, u


class BathymetryNet_Exp02(nn.Module):
    """x -> z_b. Output unconstrained (basin can be negative)."""

    def __init__(self, n_hidden=3, n_neurons=48, fourier_features=16, fourier_sigma=2.0):
        super().__init__()
        self.f = FourierFeatures(fourier_features, fourier_sigma, in_dim=1)
        in_dim = 2 * fourier_features

        layers = [nn.Linear(in_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.f(x)
        return self.net(feat)


# ============================================================
# Exp 03 — 2D transient (two cylinders)
# ============================================================


class SolutionNet2D_Exp03(nn.Module):
    """(x, y, t) -> (h, u, v)."""

    def __init__(
        self, n_hidden=5, n_neurons=128, fourier_features=24, sigma_space=2.0, sigma_time=2.0
    ):
        super().__init__()
        self.fxy = FourierFeatures(fourier_features, sigma_space, in_dim=2)
        self.ft = FourierFeatures(fourier_features, sigma_time, in_dim=1)
        in_dim = 2 * fourier_features * 2  # (sin + cos) * (xy feats + t feats)

        layers = [nn.Linear(in_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 3))  # h_raw, u, v
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        xy = torch.cat([x, y], dim=-1)
        fxy = self.fxy(xy)
        ft = self.ft(t)
        feat = torch.cat([fxy, ft], dim=-1)
        out = self.net(feat)
        h = nn.functional.softplus(out[:, 0:1])
        u = out[:, 1:2]
        v = out[:, 2:3]
        return h, u, v


class BathymetryNet2D_Exp03(nn.Module):
    """(x, y) -> z_b. Soft positivity via softplus + offset (cylinders above datum)."""

    def __init__(self, n_hidden=4, n_neurons=64, fourier_features=32, sigma=3.0):
        super().__init__()
        self.fxy = FourierFeatures(fourier_features, sigma, in_dim=2)
        in_dim = 2 * fourier_features

        layers = [nn.Linear(in_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        feat = self.fxy(xy)
        zb_raw = self.net(feat)
        # softplus → [0, ∞). The previous "softplus - 0.1" shift allowed
        # zb ∈ [-0.1, ∞), which is non-physical for Ruppenthal's cylinders
        # (zb ∈ {0, 0.2, 0.3} m). Removed 2026-05-24.
        return nn.functional.softplus(zb_raw)


# ============================================================
# Exp 05 — 2D transient (Thacker paraboloid)
# ============================================================


class SolutionNet_Exp05(nn.Module):
    """(x, y, t) -> (h, u, v). Same structure as Exp 03 with sigma_space=3.0."""

    def __init__(
        self, n_hidden=5, n_neurons=128, fourier_features=24, sigma_space=3.0, sigma_time=2.0
    ):
        super().__init__()
        self.fxy = FourierFeatures(fourier_features, sigma_space, in_dim=2)
        self.ft = FourierFeatures(fourier_features, sigma_time, in_dim=1)
        in_dim = 2 * fourier_features * 2
        layers = [nn.Linear(in_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        xy = torch.cat([x, y], dim=-1)
        feat = torch.cat([self.fxy(xy), self.ft(t)], dim=-1)
        out = self.net(feat)
        h = nn.functional.softplus(out[:, 0:1])
        return h, out[:, 1:2], out[:, 2:3]


class BathymetryNet_Exp05(nn.Module):
    """(x, y) -> z_b. No positivity (paraboloid is below datum at center)."""

    def __init__(self, n_hidden=4, n_neurons=64, fourier_features=32, sigma=4.0):
        super().__init__()
        self.fxy = FourierFeatures(fourier_features, sigma, in_dim=2)
        in_dim = 2 * fourier_features
        layers = [nn.Linear(in_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(self.fxy(torch.cat([x, y], dim=-1)))


# ============================================================
# Plot helpers — deferred dedup
# ============================================================
# The four legacy `plot_results` differ enough (1D-steady vs 1D-transient
# vs 2D-snapshot vs 2D-snapshot-with-title-suffix) that a single
# parametric helper would obscure rather than clarify. We leave them
# inline in each pinn_inverse.py for now. If a future pass shows two
# of them are byte-identical after small tweaks, dedup here.


def plot_results_1d_steady(pinn, zb_true, save_path=None):  # pragma: no cover
    """Placeholder. See Experiments/01-subcritical-bump-1d/pinn_inverse.py
    for the actual implementation; not yet promoted to a shared helper."""
    raise NotImplementedError(
        "plot_results_1d_steady is reserved for a future dedup pass; "
        "use the local plot_results in Experiments/01-subcritical-bump-1d/pinn_inverse.py."
    )


def plot_results_2d(pinn, data, save_path=None, title_suffix=""):  # pragma: no cover
    """Placeholder. See Experiments/03-* and 05-* pinn_inverse.py."""
    raise NotImplementedError(
        "plot_results_2d is reserved for a future dedup pass; "
        "use the local plot_results in each 2D experiment's pinn_inverse.py."
    )
