from pinn_bath.datagen.solvers.base import SolverBackend
from pinn_bath.datagen.solvers.pyclaw1d import PyClawSWE1D
from pinn_bath.datagen.solvers.pyclaw2d import PyClawSWE2D

__all__ = ["PyClawSWE1D", "PyClawSWE2D", "SolverBackend"]
