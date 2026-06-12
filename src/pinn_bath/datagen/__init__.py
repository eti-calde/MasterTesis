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
    BathymetrySampler,
    Difficulty,
    Feature,
    Tier,
    difficulty_components,
    difficulty_score,
)
from pinn_bath.datagen.cases import (
    BoundarySpec,
    CaseSpec,
    SimResult,
    SimulationProblem,
)
from pinn_bath.datagen.environments.base import Environment
from pinn_bath.datagen.environments.fjord1d import IncidentWaveFjord1D
from pinn_bath.datagen.forcing import ForcingSampler, TidalState, WaveForcing
from pinn_bath.datagen.grids import Grid1D
from pinn_bath.datagen.solvers.base import SolverBackend
from pinn_bath.datagen.solvers.pyclaw1d import PyClawSWE1D

__all__ = [
    "SLOPE_RANGE",
    "TIERS",
    "BathymetryField",
    "BathymetrySampler",
    "BoundarySpec",
    "CaseSpec",
    "Difficulty",
    "Environment",
    "Feature",
    "ForcingSampler",
    "Grid1D",
    "IncidentWaveFjord1D",
    "PyClawSWE1D",
    "SimResult",
    "SimulationProblem",
    "SolverBackend",
    "TidalState",
    "Tier",
    "WaveForcing",
    "difficulty_components",
    "difficulty_score",
]
