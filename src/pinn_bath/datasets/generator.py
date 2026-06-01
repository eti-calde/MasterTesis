r"""Parametric 1D bathymetry case generator with a difficulty axis.

Core of the operator-learning pivot (see memory ``operator-pivot``): instead of
one PINN per case, we learn an *amortized inverse operator* across a
distribution of bathymetries. This module samples that distribution.

A *case* is ``(bathymetry zb(x), initial/forcing spec)``; pushed through
:func:`pinn_bath.solver.forward_solve` it yields the observation field
``eta(x, t)`` (and ``u(x, t)``) that the operator must invert back to ``zb``.

Difficulty axis
---------------
Cases are stratified into ``easy / medium / hard`` tiers by sampling ranges on:
number of features ``K``, amplitude (proximity to emergence / wet-dry),
feature width (high-frequency features are harder to recover), and sign mixing
(pure bumps vs bumps+holes). A continuous :func:`difficulty_score` lets us plot
``RMSE vs difficulty`` and split train (easy+medium) from test (hard) for the
out-of-distribution generalization study.

Forcing regime
--------------
Default ``regime="free_transient"``: a free-surface perturbation over a
still pool in a closed (wall) domain — rich η(x,t), reuses the validated
solver, no inflow/outflow BCs. ``regime="through_flow"`` (inflow discharge /
outflow depth) is left as a future option (needs custom BCs).

All sampling is seeded → the dataset is exactly regenerable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

Difficulty = Literal["easy", "medium", "hard"]
Regime = Literal["free_transient", "through_flow"]
Kind = Literal["gaussian", "parabolic"]


# --------------------------------------------------------------------------- #
# Fixed space-time grid (shared by every case so the operator sees field→field
# on a consistent discretisation).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    xlower: float = 0.0
    xupper: float = 10.0
    nx: int = 256
    t_end: float = 8.0
    n_t: int = 120  # snapshots after t=0 (total frames = n_t + 1)
    sea_level: float = 1.0  # still-water free surface η_rest

    @property
    def dx(self) -> float:
        return (self.xupper - self.xlower) / self.nx

    @property
    def centers(self) -> np.ndarray:
        return np.linspace(self.xlower + self.dx / 2, self.xupper - self.dx / 2, self.nx)


# --------------------------------------------------------------------------- #
# Bathymetry as a sum of parametric features (signed: bump > 0, hole < 0).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Feature:
    kind: Kind
    amplitude: float  # signed; >0 bump (toward emergence), <0 hole (deeper)
    center: float
    width: float  # gaussian sigma, or parabolic half-width


def _feature_profile(f: Feature, x: np.ndarray) -> np.ndarray:
    if f.kind == "gaussian":
        return f.amplitude * np.exp(-(((x - f.center) / f.width) ** 2))
    if f.kind == "parabolic":
        z = f.amplitude * (1.0 - ((x - f.center) / f.width) ** 2)
        # clip to the feature's support (parabola only where it has the
        # feature's sign), like the SWASHES/Dazzi bump.
        return np.where(np.sign(z) == np.sign(f.amplitude), z, 0.0)
    raise ValueError(f"unknown feature kind: {f.kind!r}")


def bathymetry(features: tuple[Feature, ...], grid: Grid) -> np.ndarray:
    """Bed elevation zb(x) = sum of feature profiles on the grid."""
    x = grid.centers
    zb = np.zeros_like(x)
    for f in features:
        zb = zb + _feature_profile(f, x)
    return zb


# --------------------------------------------------------------------------- #
# Difficulty tiers: sampling ranges. Amplitudes are fractions of sea_level.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tier:
    k_choices: tuple[int, ...]
    amp_frac: tuple[float, float]  # |A| / sea_level
    width: tuple[float, float]  # metres
    allow_holes: bool
    allow_drying: bool  # if False, zb is capped so the pool stays wet


TIERS: dict[Difficulty, Tier] = {
    "easy": Tier(
        k_choices=(1,),
        amp_frac=(0.15, 0.35),
        width=(1.2, 2.0),
        allow_holes=True,
        allow_drying=False,
    ),
    "medium": Tier(
        k_choices=(2, 3),
        amp_frac=(0.25, 0.55),
        width=(0.7, 1.4),
        allow_holes=True,
        allow_drying=False,
    ),
    "hard": Tier(
        k_choices=(4, 5, 6),
        amp_frac=(0.40, 0.85),
        width=(0.4, 0.9),
        allow_holes=True,
        allow_drying=True,
    ),
}


@dataclass(frozen=True)
class CaseSpec:
    features: tuple[Feature, ...]
    difficulty: Difficulty
    seed: int
    regime: Regime = "free_transient"
    # free-surface perturbation (free_transient regime)
    pert_amp: float = 0.15
    pert_center: float = 5.0
    pert_width: float = 0.6
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_case(
    difficulty: Difficulty,
    rng: np.random.Generator,
    grid: Grid,
    *,
    regime: Regime = "free_transient",
) -> CaseSpec:
    """Draw one reproducible CaseSpec for the given difficulty tier."""
    tier = TIERS[difficulty]
    k = int(rng.choice(tier.k_choices))
    margin = 1.5  # keep feature centres away from the walls
    feats: list[Feature] = []
    for _ in range(k):
        kind: Kind = "gaussian" if rng.random() < 0.7 else "parabolic"
        mag = rng.uniform(*tier.amp_frac) * grid.sea_level
        sign = 1.0 if (not tier.allow_holes or rng.random() < 0.5) else -1.0
        feats.append(
            Feature(
                kind=kind,
                amplitude=sign * mag,
                center=rng.uniform(grid.xlower + margin, grid.xupper - margin),
                width=rng.uniform(*tier.width),
            )
        )
    features = tuple(feats)

    # Cap bumps below the surface for wet tiers (no drying).
    if not tier.allow_drying:
        zb = bathymetry(features, grid)
        zmax = zb.max()
        cap = 0.85 * grid.sea_level
        if zmax > cap and zmax > 0:
            scale = cap / zmax
            features = tuple(replace(f, amplitude=f.amplitude * scale) for f in features)

    pert_amp = rng.uniform(0.10, 0.25) * grid.sea_level
    pert_center = rng.uniform(grid.xlower + margin, grid.xupper - margin)
    return CaseSpec(
        features=features,
        difficulty=difficulty,
        seed=int(rng.integers(0, 2**31 - 1)),
        regime=regime,
        pert_amp=pert_amp,
        pert_center=pert_center,
        pert_width=0.6,
    )


# --------------------------------------------------------------------------- #
# Initial condition for the forcing regime
# --------------------------------------------------------------------------- #
def initial_condition(spec: CaseSpec, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return (h0, hu0) for the case's forcing regime.

    ``free_transient``: still pool at ``sea_level`` plus a localized
    free-surface Gaussian hump (u=0); it collapses and sloshes, interrogating
    the bathymetry across the closed domain.
    """
    zb = bathymetry(spec.features, grid)
    x = grid.centers
    if spec.regime == "free_transient":
        eta0 = grid.sea_level + spec.pert_amp * np.exp(
            -(((x - spec.pert_center) / spec.pert_width) ** 2)
        )
        h0 = np.maximum(eta0 - zb, 0.0)
        hu0 = np.zeros_like(h0)
        return h0, hu0
    raise NotImplementedError(f"regime {spec.regime!r} not implemented yet")


# --------------------------------------------------------------------------- #
# Difficulty scoring
# --------------------------------------------------------------------------- #
def difficulty_components(zb: np.ndarray, grid: Grid) -> dict[str, float]:
    """Interpretable scalars characterising how hard a case is to invert."""
    H0 = grid.sea_level
    amp_ratio = float(np.max(np.abs(zb)) / H0)  # proximity to emergence
    emergent_frac = float(np.mean(zb > H0))  # fraction dry at rest
    # Spectral "wiggliness": energy-weighted mean wavenumber, normalised.
    zc = zb - zb.mean()
    spec = np.abs(np.fft.rfft(zc)) ** 2
    k = np.arange(spec.size)
    bandwidth = float((spec @ k) / (spec.sum() + 1e-12) / max(spec.size - 1, 1))
    # Count distinct sign-coherent features (zero-crossings of zb proxy).
    sign_changes = int(np.count_nonzero(np.diff(np.sign(zc[np.abs(zc) > 0.02 * H0]))))
    return {
        "amp_ratio": amp_ratio,
        "emergent_frac": emergent_frac,
        "bandwidth": bandwidth,
        "sign_changes": float(sign_changes),
    }


def difficulty_score(components: dict[str, float]) -> float:
    """Scalar in ~[0, 1+] combining the components (higher = harder)."""
    return float(
        0.35 * min(components["amp_ratio"], 1.2)
        + 0.25 * min(components["bandwidth"] * 4.0, 1.0)
        + 0.20 * min(components["sign_changes"] / 6.0, 1.0)
        + 0.20 * min(components["emergent_frac"] * 5.0, 1.0)
    )


# --------------------------------------------------------------------------- #
# Full record (runs the solver)
# --------------------------------------------------------------------------- #
def generate_record(spec: CaseSpec, grid: Grid, **solver_kw: Any) -> dict[str, Any]:
    """Sample → solve → package one training record.

    Returns ``zb`` (Nx,), ``eta``/``u``/``h`` (Nt, Nx), ``t``/``x``, the
    difficulty score + components, and the flattened spec metadata.
    """
    from pinn_bath.solver import forward_solve

    zb = bathymetry(spec.features, grid)
    h0, hu0 = initial_condition(spec, grid)
    bc = "wall" if spec.regime == "free_transient" else "extrap"
    sol = forward_solve(
        zb,
        h0,
        hu0,
        xlower=grid.xlower,
        xupper=grid.xupper,
        t_end=grid.t_end,
        num_output_times=grid.n_t,
        bc=bc,
        **solver_kw,
    )
    comps = difficulty_components(zb, grid)
    return {
        "zb": zb,
        "x": sol["x"],
        "t": sol["t"],
        "eta": sol["eta"],
        "u": sol["u"],
        "h": sol["h"],
        "difficulty": spec.difficulty,
        "score": difficulty_score(comps),
        "components": comps,
        "seed": spec.seed,
        "regime": spec.regime,
    }
