"""Environment interface: the physical scenario being sampled.

An environment owns *what* is simulated: the grid, the bathymetry and
forcing distributions, and the constraint logic (deep-water cap, orthogonal
excitation axes). It is backend-free: ``make_problem`` emits a declarative
:class:`~pinn_bath.datagen.cases.SimulationProblem` that any
:class:`~pinn_bath.datagen.solvers.base.SolverBackend` can integrate.

Swapping 1D -> 2D (or fjord -> basin geometry) means writing a new
environment; swapping PyClaw -> anything else means writing a new backend.
Neither touches the other, nor the dataset builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pinn_bath.datagen.bathymetry import Difficulty
from pinn_bath.datagen.cases import CaseSpec, SimResult, SimulationProblem
from pinn_bath.datagen.solvers.base import SolverBackend


class Environment(ABC):
    """Abstract scenario: samples cases and states them as solver problems."""

    @abstractmethod
    def sample_case(self, difficulty: Difficulty, rng: np.random.Generator) -> CaseSpec:
        """Draw one reproducible case for the given difficulty tier."""
        raise NotImplementedError

    @abstractmethod
    def make_problem(self, spec: CaseSpec) -> SimulationProblem:
        """Discretise a case: initial state + boundary specs on the grid."""
        raise NotImplementedError

    def simulate(self, spec: CaseSpec, backend: SolverBackend) -> SimResult:
        """Convenience: ``make_problem`` then ``backend.solve``, with the
        case metadata merged into ``SimResult.meta``."""
        result = backend.solve(self.make_problem(spec))
        result.meta.update(spec.to_metadata())
        return result
