"""Solver backend interface: integrate one :class:`SimulationProblem`.

A backend owns *how* the SWE are integrated (scheme, kernels, BC machinery);
it knows nothing about case sampling or difficulty. Implementations map the
declarative :class:`~pinn_bath.datagen.cases.BoundarySpec` onto their own
boundary mechanisms and return the standard tensor package
:class:`~pinn_bath.datagen.cases.SimResult`.

Current backends: :class:`~pinn_bath.datagen.solvers.pyclaw1d.PyClawSWE1D`.
Planned: ``PyClawSWE2D`` (same Clawpack API, 2D Riemann kernel); a SWASH
subprocess backend would also fit this interface if cross-validation against
a coastal community model is ever needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pinn_bath.datagen.cases import SimResult, SimulationProblem


class SolverBackend(ABC):
    """Abstract SWE integrator: ``SimulationProblem -> SimResult``."""

    #: short identifier recorded in ``SimResult.meta`` and dataset manifests
    name: str = "abstract"

    @abstractmethod
    def solve(self, problem: SimulationProblem) -> SimResult:
        """Integrate the problem and return fields on the problem's grid."""
        raise NotImplementedError
