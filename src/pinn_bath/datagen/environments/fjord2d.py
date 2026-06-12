"""2D open fjord-reach environment: plane incident wave over sloped 2D beds.

The 2D extension of :mod:`pinn_bath.datagen.environments.fjord1d`, sharing
its physical regime and excitation axes verbatim: the spring-neap factor
``f`` drives tidal stage and incident amplitude through the *same*
:class:`~pinn_bath.datagen.forcing.ForcingSampler`, the slope axis and tier
tables come from the shared constants, and the deep-water cap is anchored to
spring low water. New in 2D: the bed carries seamounts and oriented ridges
(:class:`~pinn_bath.datagen.bathymetry.BathymetrySampler2D`), and the wave
field refracts laterally around them, which is the observability the 2D
operator will exploit.

Boundary layout: the incident train enters one x edge as a plane wave at
normal incidence (the forcing's ``side``: left -> x_lower, right ->
x_upper); the opposite x edge and both y edges are transmissive outflow
(open coastal-patch geometry). Set ``y_boundary="wall"`` for a walled
channel instead (closer to a confined fjord cross-section, but introduces
cross-channel resonances).

Type note: this environment returns the 2D variants (``CaseSpec2D``,
``SimulationProblem2D``) of the base-class contracts; the runtime flow
(``sample_case`` -> ``make_problem`` -> ``backend.solve``) is identical and
:meth:`Environment.simulate` works unchanged.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pinn_bath.datagen.bathymetry import (
    BathymetrySampler2D,
    Difficulty,
    difficulty_components_2d,
    difficulty_score,
)
from pinn_bath.datagen.cases import BoundarySpec, CaseSpec2D, SimulationProblem2D
from pinn_bath.datagen.environments.base import Environment
from pinn_bath.datagen.environments.fjord1d import MIN_REST_COLUMN_FRAC
from pinn_bath.datagen.forcing import ForcingSampler
from pinn_bath.datagen.grids import Grid2D


class IncidentWaveFjord2D(Environment):
    """Open 2D fjord reach with plane incident-wave forcing."""

    def __init__(
        self,
        grid: Grid2D | None = None,
        bathymetry: BathymetrySampler2D | None = None,
        forcing: ForcingSampler | None = None,
        min_rest_column_frac: float = MIN_REST_COLUMN_FRAC,
        y_boundary: Literal["outflow", "wall"] = "outflow",
    ) -> None:
        self.grid = grid if grid is not None else Grid2D()
        self.bathymetry = bathymetry if bathymetry is not None else BathymetrySampler2D()
        self.forcing = forcing if forcing is not None else ForcingSampler()
        self.y_boundary = y_boundary
        sea = self.grid.sea_level
        self.bed_cap = self.forcing.min_water_level(sea) - min_rest_column_frac * sea
        if self.bed_cap <= 0:
            raise ValueError(
                "deep-water cap is non-positive; check water_level_range / "
                "min_rest_column_frac against grid.sea_level"
            )

    # ------------------------------------------------------------------ #
    def sample_case(self, difficulty: Difficulty, rng: np.random.Generator) -> CaseSpec2D:
        grid = self.grid
        X, Y = grid.meshgrid()
        bed = self.bathymetry.sample(
            difficulty,
            rng,
            X,
            Y,
            sea_level=grid.sea_level,
            max_bed_elevation=self.bed_cap,
        )
        tide = self.forcing.sample_tide(rng, grid.sea_level)
        wave = self.forcing.sample_wave(rng, grid.sea_level, tide.spring_neap)
        comps = difficulty_components_2d(bed.detrended(X, Y), grid.sea_level)
        return CaseSpec2D(
            bathymetry=bed,
            forcing=wave,
            water_level=tide.water_level,
            spring_neap=tide.spring_neap,
            difficulty=difficulty,
            seed=int(rng.integers(0, 2**31 - 1)),
            score=difficulty_score(comps),
            components=comps,
        )

    def make_problem(self, spec: CaseSpec2D) -> SimulationProblem2D:
        grid = self.grid
        X, Y = grid.meshgrid()
        zb = spec.bathymetry.profile(X, Y)  # (Ny, Nx)
        h0 = np.maximum(spec.water_level - zb, 0.0)
        hu0 = np.zeros_like(h0)
        hv0 = np.zeros_like(h0)

        # Plane-wave inflow on the sampled x edge. h_rest (edge-mean still
        # depth) is informational; water_level lets the backend reconstruct
        # the rest depth per y-column from the edge bathymetry, exact even
        # when feature tails reach the inflow column.
        side = spec.forcing.side
        edge_col = zb[:, 0] if side == "left" else zb[:, -1]
        inflow = BoundarySpec(
            kind="incident_wave",
            eta_signal=spec.forcing.signal(),
            h_rest=float(np.mean(spec.water_level - edge_col)),
            water_level=spec.water_level,
        )
        outflow = BoundarySpec(kind="outflow")
        lateral = BoundarySpec(kind=self.y_boundary)
        bc_x_lower, bc_x_upper = (inflow, outflow) if side == "left" else (outflow, inflow)
        return SimulationProblem2D(
            grid=grid,
            zb=zb,
            h0=h0,
            hu0=hu0,
            hv0=hv0,
            bc_x_lower=bc_x_lower,
            bc_x_upper=bc_x_upper,
            bc_y_lower=lateral,
            bc_y_upper=lateral,
        )
