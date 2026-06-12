"""Modular dataset-generation pipeline (environments x solver backends).

Successor to :mod:`pinn_bath.datasets.generator` (which stays untouched for
exact reproducibility of the published sweeps). New here: a per-case
background-slope axis on the bathymetry, and an Environment / SolverBackend
split so 1D -> 2D and PyClaw -> other solvers are isolated swaps.

Typical use::

    from pinn_bath.datagen import IncidentWaveFjord1D, PyClawSWE1D
    import numpy as np

    env = IncidentWaveFjord1D()
    backend = PyClawSWE1D()
    rng = np.random.default_rng(0)
    spec = env.sample_case("medium", rng)
    result = env.simulate(spec, backend)   # SimResult: eta/u (Nt, Nx), zb (Nx,)
"""

from pinn_bath.datagen.bathymetry import (
    SLOPE_RANGE,
    TIERS,
    BathymetryField,
    BathymetryField2D,
    BathymetrySampler,
    BathymetrySampler2D,
    Difficulty,
    Feature,
    Feature2D,
    Tier,
    difficulty_components,
    difficulty_components_2d,
    difficulty_score,
)
from pinn_bath.datagen.cases import (
    BoundarySpec,
    CaseSpec,
    CaseSpec2D,
    SimResult,
    SimulationProblem,
    SimulationProblem2D,
)
from pinn_bath.datagen.environments.base import Environment
from pinn_bath.datagen.environments.fjord1d import IncidentWaveFjord1D
from pinn_bath.datagen.environments.fjord2d import IncidentWaveFjord2D
from pinn_bath.datagen.forcing import ForcingSampler, TidalState, WaveForcing
from pinn_bath.datagen.grids import Grid1D, Grid2D
from pinn_bath.datagen.solvers.base import SolverBackend
from pinn_bath.datagen.solvers.pyclaw1d import PyClawSWE1D
from pinn_bath.datagen.solvers.pyclaw2d import PyClawSWE2D

__all__ = [
    "SLOPE_RANGE",
    "TIERS",
    "BathymetryField",
    "BathymetryField2D",
    "BathymetrySampler",
    "BathymetrySampler2D",
    "BoundarySpec",
    "CaseSpec",
    "CaseSpec2D",
    "Difficulty",
    "Environment",
    "Feature",
    "Feature2D",
    "ForcingSampler",
    "Grid1D",
    "Grid2D",
    "IncidentWaveFjord1D",
    "IncidentWaveFjord2D",
    "PyClawSWE1D",
    "PyClawSWE2D",
    "SimResult",
    "SimulationProblem",
    "SimulationProblem2D",
    "SolverBackend",
    "TidalState",
    "Tier",
    "WaveForcing",
    "difficulty_components",
    "difficulty_components_2d",
    "difficulty_score",
]
