"""PyClaw (Clawpack) 1D SWE backend.

Thin adapter over the validated wrapper in :mod:`pinn_bath.solver.swe1d`
(well-balanced f-wave / augmented GeoClaw-family kernels, in-memory frames,
Riemann-invariant characteristic inflow). This class only translates the
declarative :class:`~pinn_bath.datagen.cases.BoundarySpec` pairs into
``forward_solve`` keyword arguments; all numerics stay in one place.
"""

from __future__ import annotations

from typing import Any, Literal

from pinn_bath.datagen.cases import BoundarySpec, SimResult, SimulationProblem
from pinn_bath.datagen.solvers.base import SolverBackend
from pinn_bath.solver import DEFAULT_G, forward_solve, make_characteristic_inflow


class PyClawSWE1D(SolverBackend):
    """1D shallow-water backend on Clawpack's compiled Riemann kernels.

    Parameters mirror :func:`pinn_bath.solver.forward_solve`; the default
    ``aug`` kernel is the augmented GeoClaw-style solver (robust at wet/dry
    fronts, validated against the Thacker analytic solution to <1%).
    """

    name = "pyclaw_swe1d"

    def __init__(
        self,
        *,
        kernel: str = "aug",
        limiter: str = "vanleer",
        cfl_desired: float = 0.8,
        cfl_max: float = 0.9,
        dry_tolerance: float = 1e-3,
        g: float = DEFAULT_G,
        quiet: bool = True,
    ) -> None:
        self.kernel = kernel
        self.limiter = limiter
        self.cfl_desired = cfl_desired
        self.cfl_max = cfl_max
        self.dry_tolerance = dry_tolerance
        self.g = g
        self.quiet = quiet

    # ------------------------------------------------------------------ #
    def _edge_kwargs(self, spec: BoundarySpec, which: Literal["lower", "upper"]) -> dict[str, Any]:
        """Map one BoundarySpec to forward_solve kwargs for that edge."""
        if spec.kind == "incident_wave":
            cb = make_characteristic_inflow(
                spec.eta_signal,
                h_rest=float(spec.h_rest),
                g=self.g,
                side=which,
                dry_tolerance=self.dry_tolerance,
            )
            return {f"bc_{which}": "custom", f"user_bc_{which}": cb}
        kind_map = {"outflow": "extrap", "wall": "wall", "periodic": "periodic"}
        return {f"bc_{which}": kind_map[spec.kind]}

    def solve(self, problem: SimulationProblem) -> SimResult:
        lo, hi = problem.bc_lower, problem.bc_upper
        if (lo.kind == "periodic") != (hi.kind == "periodic"):
            raise ValueError("periodic boundaries must be used on both edges")
        bc_kw = {**self._edge_kwargs(lo, "lower"), **self._edge_kwargs(hi, "upper")}

        grid = problem.grid
        sol = forward_solve(
            problem.zb,
            problem.h0,
            problem.hu0,
            xlower=grid.xlower,
            xupper=grid.xupper,
            t_end=grid.t_end,
            num_output_times=grid.n_t,
            g=self.g,
            kernel=self.kernel,
            dry_tolerance=self.dry_tolerance,
            cfl_desired=self.cfl_desired,
            cfl_max=self.cfl_max,
            limiter=self.limiter,
            quiet=self.quiet,
            **bc_kw,
        )
        return SimResult(
            t=sol["t"],
            x=sol["x"],
            zb=sol["zb"],
            eta=sol["eta"],
            u=sol["u"],
            h=sol["h"],
            meta={"solver": self.name, "kernel": self.kernel, "limiter": self.limiter},
        )
