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
Default ``regime="incident_wave"``: still deep pool forced by a continuous
*incident wave train* entering one boundary (sampled per case: left or right)
as a simple wave, with a transmissive (``extrap``) outflow on the other end.
This is the open fjord-mouth geometry — sustained η(x,t) over the whole window
(no draining to a quiet box), where the only backscatter comes from the
bathymetry itself. The inflow uses :func:`pinn_bath.solver.make_characteristic_inflow`
(Riemann-invariant ghost fill, near non-reflecting). ``regime="free_transient"``
(initial free-surface hump in a closed ``wall`` domain) is kept for backward
compatibility / solver validation.

The bed never emerges (deep-water target application: southern-Chile fjords);
``allow_drying=False`` on every tier caps bumps below the surface, so the
wet/dry discontinuity — the operator's pathological regime — is excluded by
design.

Two excitation axes vary per case, both sampled identically across difficulty
tiers (so the OOD-by-difficulty split stays *purely bathymetric*): the incident
wave-train amplitude (gentle *neap* to strong *spring* tides) and the **tidal
stage** — the still water level (mean depth), which sets the wave celerity
``c=sqrt(g H)`` (low water → slower). The bed cap is tied to the lowest tide so
nothing emerges even at low water. The difficulty score is normalised by the
*fixed reference* level (``Grid.sea_level``), so the tidal stage does not leak
into the difficulty label.

All sampling is seeded → the dataset is exactly regenerable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

Difficulty = Literal["easy", "medium", "hard"]
Regime = Literal["incident_wave", "free_transient"]
Side = Literal["left", "right"]
Kind = Literal["gaussian", "parabolic"]

# Tidal-stage axis: the per-case still water level (mean depth) is drawn from
# this range, in units of the reference ``Grid.sea_level``. Low water → shallower
# → slower waves (c=sqrt(g H)); high water → faster. Sampled identically across
# difficulty tiers (orthogonal to the bathymetric difficulty axis). The bed cap
# is tied to the *lowest* tide so no cell ever emerges, even at low water.
WATER_LEVEL_RANGE: tuple[float, float] = (0.85, 1.15)
# Minimum still-water column (fraction of sea_level) left over the *tallest*
# feature at the *lowest* tide: bed peaks are capped at
# ``min(WATER_LEVEL_RANGE)*sea_level - MIN_REST_COLUMN_FRAC*sea_level``. Keeping
# this comfortably large (deep water over the crest) avoids the thin-film /
# near-dry regime where a shoaling wave produces a fast jet over a shallow crest
# (sharp gradients + noisy desingularised velocity) — the pathology the operator
# struggles with.
MIN_REST_COLUMN_FRAC: float = 0.30
# Incident wave train: per-component amplitude range (fraction of sea_level,
# gentle neap → moderate spring) and a cap on the *total* summed amplitude.
# Periods are kept long (comparable to / above the crossing time L/sqrt(gH)
# ≈ 3.2 s) so the waves are gentle, tide-like swells: short periods produce
# short wavelengths that steepen into shock-like bores — the discontinuities
# the operator struggles with. Together the amplitude cap + long periods keep
# the strongest spring tides smooth while preserving multi-frequency variety.
WAVE_AMP_FRAC: tuple[float, float] = (0.02, 0.08)
MAX_WAVE_AMP_FRAC: float = 0.12
WAVE_PERIOD_RANGE: tuple[float, float] = (2.5, 4.5)


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
        allow_drying=False,  # no emergence: target application is deep water
    ),
}


@dataclass(frozen=True)
class CaseSpec:
    features: tuple[Feature, ...]
    difficulty: Difficulty
    seed: int
    regime: Regime = "incident_wave"
    # Tidal stage: still water level (mean free surface) for this case. Varies
    # the depth → wave celerity sqrt(g H). Reference rest level is Grid.sea_level.
    water_level: float = 1.0
    # incident_wave regime: a tren of 1-3 sinusoidal components entering one
    # boundary, ramped in smoothly. Amplitudes/periods/phases are per-component.
    inflow_side: Side = "left"
    wave_amps: tuple[float, ...] = (0.08,)  # metres (δη per component)
    wave_periods: tuple[float, ...] = (2.2,)  # seconds
    wave_phases: tuple[float, ...] = (0.0,)  # radians
    ramp_tau: float = 0.8  # seconds (tanh ramp time)
    # free_transient regime: initial free-surface hump (kept for back-compat).
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
    regime: Regime = "incident_wave",
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

    # Cap bumps so a comfortable water column remains over the tallest crest even
    # at the lowest tide — keeps the bank in the deep-water regime (no thin-film
    # near-dry jets), well clear of emergence.
    if not tier.allow_drying:
        min_level = WATER_LEVEL_RANGE[0] * grid.sea_level
        cap = min_level - MIN_REST_COLUMN_FRAC * grid.sea_level
        zb = bathymetry(features, grid)
        zmax = zb.max()
        if zmax > cap and zmax > 0:
            scale = cap / zmax
            features = tuple(replace(f, amplitude=f.amplitude * scale) for f in features)

    # Tidal stage (mean water level) — orthogonal to the difficulty tier.
    water_level = float(rng.uniform(*WATER_LEVEL_RANGE) * grid.sea_level)

    # Incident-wave forcing — sampled identically across tiers (orthogonal to
    # the bathymetric difficulty axis, so OOD stays purely bathymetric). A train
    # of 1-3 sinusoidal components entering one (randomly chosen) boundary; the
    # amplitude spans gentle neap to moderate spring, with the total summed
    # amplitude capped so the strongest tides stay below the shock-forming regime.
    n_comp = int(rng.integers(1, 4))  # 1..3
    wave_amps = tuple(float(a) for a in rng.uniform(*WAVE_AMP_FRAC, size=n_comp) * grid.sea_level)
    total_amp = sum(wave_amps)
    amp_cap = MAX_WAVE_AMP_FRAC * grid.sea_level
    if total_amp > amp_cap:
        scale = amp_cap / total_amp
        wave_amps = tuple(a * scale for a in wave_amps)
    wave_periods = tuple(float(p) for p in rng.uniform(*WAVE_PERIOD_RANGE, size=n_comp))
    wave_phases = tuple(float(ph) for ph in rng.uniform(0.0, 2.0 * np.pi, size=n_comp))
    inflow_side: Side = "left" if rng.random() < 0.5 else "right"
    ramp_tau = float(rng.uniform(0.5, 1.0))

    # free_transient back-compat forcing (unused by incident_wave).
    pert_amp = rng.uniform(0.10, 0.25) * grid.sea_level
    pert_center = rng.uniform(grid.xlower + margin, grid.xupper - margin)
    return CaseSpec(
        features=features,
        difficulty=difficulty,
        seed=int(rng.integers(0, 2**31 - 1)),
        regime=regime,
        water_level=water_level,
        inflow_side=inflow_side,
        wave_amps=wave_amps,
        wave_periods=wave_periods,
        wave_phases=wave_phases,
        ramp_tau=ramp_tau,
        pert_amp=pert_amp,
        pert_center=pert_center,
        pert_width=0.6,
    )


# --------------------------------------------------------------------------- #
# Initial condition for the forcing regime
# --------------------------------------------------------------------------- #
def initial_condition(spec: CaseSpec, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return (h0, hu0) for the case's forcing regime.

    ``incident_wave``: still pool at rest at the case's tidal stage
    (``h0 = water_level - zb``, u=0); all energy enters later through the
    time-dependent inflow boundary.

    ``free_transient``: still pool at ``water_level`` plus a localized
    free-surface Gaussian hump (u=0); it collapses and sloshes, interrogating
    the bathymetry across the closed domain.
    """
    zb = bathymetry(spec.features, grid)
    x = grid.centers
    if spec.regime == "incident_wave":
        h0 = np.maximum(spec.water_level - zb, 0.0)
        hu0 = np.zeros_like(h0)
        return h0, hu0
    if spec.regime == "free_transient":
        eta0 = spec.water_level + spec.pert_amp * np.exp(
            -(((x - spec.pert_center) / spec.pert_width) ** 2)
        )
        h0 = np.maximum(eta0 - zb, 0.0)
        hu0 = np.zeros_like(h0)
        return h0, hu0
    raise NotImplementedError(f"regime {spec.regime!r} not implemented yet")


def wave_signal(spec: CaseSpec) -> Any:
    """Build the boundary surface-perturbation function ``δη(t)`` for the
    incident-wave forcing: a smoothly ramped sum of sinusoidal components.
    """
    amps = np.asarray(spec.wave_amps, dtype=float)
    periods = np.asarray(spec.wave_periods, dtype=float)
    phases = np.asarray(spec.wave_phases, dtype=float)
    tau = float(spec.ramp_tau)

    def eta_signal(t: float) -> float:
        ramp = np.tanh(t / tau)
        return float(ramp * np.sum(amps * np.sin(2.0 * np.pi * t / periods + phases)))

    return eta_signal


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
    """Scalar in ~[0, 1] combining the components (higher = harder).

    Three bathymetric descriptors — relative amplitude, spectral bandwidth and
    number of features. The wet/dry ``emergent_frac`` term was dropped (the
    deep-water bank never emerges, so it is identically 0); the remaining
    weights are rescaled from the original 0.35/0.25/0.20 to sum to 1.
    """
    return float(
        0.44 * min(components["amp_ratio"], 1.2)
        + 0.31 * min(components["bandwidth"] * 4.0, 1.0)
        + 0.25 * min(components["sign_changes"] / 6.0, 1.0)
    )


# --------------------------------------------------------------------------- #
# Full record (runs the solver)
# --------------------------------------------------------------------------- #
def generate_record(spec: CaseSpec, grid: Grid, **solver_kw: Any) -> dict[str, Any]:
    """Sample → solve → package one training record.

    Returns ``zb`` (Nx,), ``eta``/``u``/``h`` (Nt, Nx), ``t``/``x``, the
    difficulty score + components, and the flattened spec metadata.
    """
    from pinn_bath.solver import DEFAULT_G, forward_solve, make_characteristic_inflow

    zb = bathymetry(spec.features, grid)
    h0, hu0 = initial_condition(spec, grid)

    bc_kw: dict[str, Any] = {}
    if spec.regime == "incident_wave":
        # Time-dependent simple-wave inflow on the sampled side, transmissive
        # (outflow) on the other. h_rest is the still depth at the inflow edge.
        side = spec.inflow_side
        zb_b = float(zb[0] if side == "left" else zb[-1])
        h_rest_b = spec.water_level - zb_b
        cb = make_characteristic_inflow(
            wave_signal(spec),
            h_rest=h_rest_b,
            g=DEFAULT_G,
            side="lower" if side == "left" else "upper",
        )
        if side == "left":
            bc_kw = dict(bc_lower="custom", bc_upper="extrap", user_bc_lower=cb)
        else:
            bc_kw = dict(bc_lower="extrap", bc_upper="custom", user_bc_upper=cb)
    else:  # free_transient: closed reflective box
        bc_kw = dict(bc="wall")

    sol = forward_solve(
        zb,
        h0,
        hu0,
        xlower=grid.xlower,
        xupper=grid.xupper,
        t_end=grid.t_end,
        num_output_times=grid.n_t,
        **bc_kw,
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
        "water_level": spec.water_level,
        "inflow_side": spec.inflow_side,
    }
