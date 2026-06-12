"""1D open fjord-mouth environment: incident wave train over a sloped bed.

The scenario of the operator paper, plus the new background-slope axis: a
still deep pool at the case's tidal stage, forced by a continuous wave train
entering one boundary as a simple wave (Riemann-invariant inflow) and leaving
through a transmissive outflow on the other. The only backscatter comes from
the bathymetry itself, so the persistent signal *is* interrogation of the bed.

Orthogonal excitation axes (sampled identically across difficulty tiers):
a spring-neap factor ``f`` coherently driving tidal stage and wave-train
amplitude (see :mod:`pinn_bath.datagen.forcing`), wave periods/phases/side,
and the background slope. The deep-water cap ties the highest allowed bed
point to the *lowest* tide (spring low water) minus a minimum rest column,
so nothing emerges even at low water with a shoaling trend.
"""

from __future__ import annotations

import numpy as np

from pinn_bath.datagen.bathymetry import (
    BathymetrySampler,
    Difficulty,
    difficulty_components,
    difficulty_score,
)
from pinn_bath.datagen.cases import BoundarySpec, CaseSpec, SimulationProblem
from pinn_bath.datagen.environments.base import Environment
from pinn_bath.datagen.forcing import ForcingSampler
from pinn_bath.datagen.grids import Grid1D

# Minimum still-water column (fraction of sea_level) left over the highest bed
# point at the lowest tide. Keeping this comfortably large avoids the
# thin-film / near-dry regime (fast jets over shallow crests: sharp gradients
# + noisy desingularised velocity), the pathology excluded by design.
MIN_REST_COLUMN_FRAC: float = 0.30


class IncidentWaveFjord1D(Environment):
    """Open 1D fjord transect with incident-wave forcing and sloped beds."""

    def __init__(
        self,
        grid: Grid1D | None = None,
        bathymetry: BathymetrySampler | None = None,
        forcing: ForcingSampler | None = None,
        min_rest_column_frac: float = MIN_REST_COLUMN_FRAC,
    ) -> None:
        self.grid = grid if grid is not None else Grid1D()
        self.bathymetry = bathymetry if bathymetry is not None else BathymetrySampler()
        self.forcing = forcing if forcing is not None else ForcingSampler()
        sea = self.grid.sea_level
        # Deep-water cap: highest bed elevation allowed anywhere, tied to the
        # lowest tidal stage so the bed never emerges even at low water.
        self.bed_cap = self.forcing.min_water_level(sea) - min_rest_column_frac * sea
        if self.bed_cap <= 0:
            raise ValueError(
                "deep-water cap is non-positive; check water_level_range / "
                "min_rest_column_frac against grid.sea_level"
            )

    # ------------------------------------------------------------------ #
    def sample_case(self, difficulty: Difficulty, rng: np.random.Generator) -> CaseSpec:
        grid = self.grid
        x = grid.centers
        bed = self.bathymetry.sample(
            difficulty,
            rng,
            x,
            sea_level=grid.sea_level,
            max_bed_elevation=self.bed_cap,
        )
        # Spring-neap factor f drives stage and wave amplitude coherently.
        tide = self.forcing.sample_tide(rng, grid.sea_level)
        wave = self.forcing.sample_wave(rng, grid.sea_level, tide.spring_neap)
        # Score on the detrended bed: the slope axis stays out of the label.
        comps = difficulty_components(bed.detrended(x), grid.sea_level)
        return CaseSpec(
            bathymetry=bed,
            forcing=wave,
            water_level=tide.water_level,
            spring_neap=tide.spring_neap,
            difficulty=difficulty,
            seed=int(rng.integers(0, 2**31 - 1)),
            score=difficulty_score(comps),
            components=comps,
        )

    def make_problem(self, spec: CaseSpec) -> SimulationProblem:
        grid = self.grid
        x = grid.centers
        zb = spec.bathymetry.profile(x)
        # Still pool at rest at the case's tidal stage; all energy enters
        # later through the time-dependent inflow boundary.
        h0 = np.maximum(spec.water_level - zb, 0.0)
        hu0 = np.zeros_like(h0)

        side = spec.forcing.side
        zb_edge = float(zb[0] if side == "left" else zb[-1])
        inflow = BoundarySpec(
            kind="incident_wave",
            eta_signal=spec.forcing.signal(),
            h_rest=spec.water_level - zb_edge,
        )
        outflow = BoundarySpec(kind="outflow")
        bc_lower, bc_upper = (inflow, outflow) if side == "left" else (outflow, inflow)
        return SimulationProblem(
            grid=grid, zb=zb, h0=h0, hu0=hu0, bc_lower=bc_lower, bc_upper=bc_upper
        )
