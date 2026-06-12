r"""PyClaw (Clawpack) 2D SWE backend.

Solves the 2D shallow water equations with variable bathymetry,

.. math::
    h_t + (hu)_x + (hv)_y &= 0 \\
    (hu)_t + (hu^2 + \tfrac{1}{2}gh^2)_x + (huv)_y &= -g\,h\,b_x \\
    (hv)_t + (huv)_x + (hv^2 + \tfrac{1}{2}gh^2)_y &= -g\,h\,b_y,

with state ``q = [h, hu, hv]`` and ``aux[0] = zb``. Default kernel is
``sw_aug_2D`` (augmented GeoClaw-family solver, same lineage as the 1D
default), run with f-waves and dimensional (Godunov) splitting: this
configuration preserves lake-at-rest to machine precision (verified ~2e-16
over a Gaussian seamount), matching the well-balanced guarantee of the 1D
backend. ``shallow_bathymetry_fwave_2D`` (the documented pyclaw
``shallow_2d/sill.py`` kernel) is available as ``kernel="fwave"``; it
requires the split (its transverse solver degrades well-balancing to ~1e-7).

The incident wave enters an x edge as a **plane wave at normal incidence**:
the 1D Riemann-invariant ghost fill is applied column-wise along y (incoming
invariant prescribed from the signal, outgoing extrapolated from the first
interior column). The tangential velocity is **upwinded on the shear
characteristic**: prescribed as 0 while the reconstructed normal flow enters
the domain (the u-family is then incoming), extrapolated while it leaves.
Unlike the 1D solver split (``pinn_bath.solver.swe1d`` wrapped by
``pyclaw1d``), the 2D PyClaw machinery lives entirely in this module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from pinn_bath.datagen.cases import BoundarySpec, SimResult, SimulationProblem2D
from pinn_bath.datagen.solvers.base import SolverBackend
from pinn_bath.solver import DEFAULT_G
from pinn_bath.solver.swe1d import _suppress_fortran_stdout

# 2D Riemann kernels with topography (riemann module attribute names).
_KERNELS_2D = {
    "aug": "sw_aug_2D",  # augmented GeoClaw-family (default; has rpt2)
    "fwave": "shallow_bathymetry_fwave_2D",  # pyclaw sill.py example kernel
}

XEdge = Literal["x_lower", "x_upper"]


def _make_characteristic_inflow_2d(
    eta_signal: Callable[[float], float],
    *,
    h_rest: float,
    g: float,
    edge: XEdge,
    dry_tolerance: float,
    water_level: float | None = None,
) -> Callable[..., None]:
    """Plane-wave Riemann-invariant ghost filler for one x edge (vectorised
    over y). Mirrors :func:`pinn_bath.solver.make_characteristic_inflow`:
    incoming invariant prescribed, outgoing extrapolated; the tangential
    velocity is upwinded on the shear characteristic (0 while the normal flow
    enters the domain, extrapolated while it leaves).

    When ``water_level`` is given, the rest depth is computed *per y-column*
    from the edge bathymetry (``water_level - zb`` via ``auxbc``): feature
    tails reaching the inflow column then produce an exactly consistent
    at-rest ghost state (lake-at-rest stays machine-precision). Otherwise the
    scalar ``h_rest`` is used uniformly.
    """
    h_rest = max(float(h_rest), dry_tolerance)

    def bc_fn(state: Any, dim: Any, t: float, qbc: np.ndarray, auxbc: Any, num_ghost: int) -> None:
        if dim.name != "x":  # invoked only for custom edges; guard anyway
            return
        col = num_ghost if edge == "x_lower" else -num_ghost - 1
        # The y-ghost corner entries of the interior column are uninitialised
        # when the x boundary pass runs before the y pass; the corners we
        # write are overwritten by the subsequent y fill, but unclipped
        # garbage there overflows float64 in the products below. The bounds
        # are far outside the physical regime (|u| < 1 m/s, h ~ 1 m).
        h_i = np.clip(qbc[0, col, :], dry_tolerance, 1e3)
        u_i = np.clip(np.nan_to_num(qbc[1, col, :] / h_i), -1e2, 1e2)
        v_i = np.clip(np.nan_to_num(qbc[2, col, :] / h_i), -1e2, 1e2)
        c_i = np.sqrt(g * h_i)

        if water_level is not None and auxbc is not None:
            h_r = np.maximum(water_level - auxbc[0, col, :], dry_tolerance)
        else:
            h_r = np.full_like(h_i, h_rest)
        c_r = np.sqrt(g * h_r)

        deta = float(eta_signal(t))
        h_in = np.maximum(h_r + deta, dry_tolerance)
        c_in = np.sqrt(g * h_in)
        if edge == "x_lower":
            u_in = c_r * deta / h_r  # simple wave -> +x
            r_plus = u_in + 2.0 * c_in  # prescribed incoming (per y)
            r_minus = u_i - 2.0 * c_i  # outgoing, extrapolated (per y)
        else:
            u_in = -c_r * deta / h_r  # simple wave -> -x
            r_minus = u_in - 2.0 * c_in
            r_plus = u_i + 2.0 * c_i
        u_g = 0.5 * (r_plus + r_minus)
        c_g = 0.25 * (r_plus - r_minus)
        h_g = np.maximum(c_g, 0.0) ** 2 / g

        # Shear-characteristic upwinding: in subcritical inflow phases the
        # u-family enters the domain too, so the tangential velocity must be
        # *prescribed* (0: plane wave at normal incidence); extrapolating it
        # there imposes dv/dx = 0, a non-transparent condition that re-advects
        # refraction-generated transverse momentum back inside. On outflow
        # phases extrapolation is the transparent choice.
        inflowing = (u_g > 0.0) if edge == "x_lower" else (u_g < 0.0)
        v_g = np.where(inflowing, 0.0, v_i)

        sl = slice(None, num_ghost) if edge == "x_lower" else slice(-num_ghost, None)
        qbc[0, sl, :] = h_g[None, :]
        qbc[1, sl, :] = (h_g * u_g)[None, :]
        qbc[2, sl, :] = (h_g * v_g)[None, :]

    return bc_fn


class PyClawSWE2D(SolverBackend):
    """2D shallow-water backend on Clawpack's compiled Riemann kernels."""

    name = "pyclaw_swe2d"

    def __init__(
        self,
        *,
        kernel: str = "aug",
        limiter: str = "vanleer",
        cfl_desired: float = 0.45,
        cfl_max: float = 0.5,
        dry_tolerance: float = 1e-3,
        g: float = DEFAULT_G,
        quiet: bool = True,
    ) -> None:
        if kernel not in _KERNELS_2D:
            raise ValueError(f"kernel must be one of {sorted(_KERNELS_2D)}; got {kernel!r}")
        self.kernel = kernel
        self.limiter = limiter
        # Dimensional splitting halves the stable CFL vs unsplit (<=0.5).
        self.cfl_desired = cfl_desired
        self.cfl_max = cfl_max
        self.dry_tolerance = dry_tolerance
        self.g = g
        self.quiet = quiet

    # ------------------------------------------------------------------ #
    def _edge(
        self,
        pyclaw: Any,
        solver: Any,
        spec: BoundarySpec,
        dim_idx: int,
        which: Literal["lower", "upper"],
    ) -> None:
        """Apply one BoundarySpec to (dim_idx, which) on the solver."""
        bcmap = {
            "outflow": pyclaw.BC.extrap,
            "wall": pyclaw.BC.wall,
            "periodic": pyclaw.BC.periodic,
        }
        bc_arr = solver.bc_lower if which == "lower" else solver.bc_upper
        aux_arr = solver.aux_bc_lower if which == "lower" else solver.aux_bc_upper
        if spec.kind == "incident_wave":
            if dim_idx != 0:
                raise ValueError("incident_wave is only supported on the x edges")
            bc_arr[0] = pyclaw.BC.custom
            cb = _make_characteristic_inflow_2d(
                spec.eta_signal,
                h_rest=float(spec.h_rest),
                g=self.g,
                edge="x_lower" if which == "lower" else "x_upper",
                dry_tolerance=self.dry_tolerance,
                water_level=spec.water_level,
            )
            if which == "lower":
                solver.user_bc_lower = cb
            else:
                solver.user_bc_upper = cb
            aux_arr[0] = pyclaw.BC.extrap
        else:
            bc_arr[dim_idx] = bcmap[spec.kind]
            # Static bathymetry ghost: extrapolate at walls, mirror otherwise.
            aux_arr[dim_idx] = pyclaw.BC.extrap if spec.kind == "wall" else bcmap[spec.kind]

    def solve(self, problem: SimulationProblem2D) -> SimResult:
        from clawpack import pyclaw, riemann

        grid = problem.grid
        rp = getattr(riemann, _KERNELS_2D[self.kernel])
        solver = pyclaw.ClawSolver2D(rp)
        solver.num_eqn = 3
        solver.num_waves = 3
        solver.fwave = True
        # Godunov dimensional splitting: machine-precision lake-at-rest for
        # both kernels (the fwave kernel ships no usable transverse solver).
        solver.dimensional_split = True
        solver.limiters = getattr(
            pyclaw.limiters.tvd,
            {"vanleer": "vanleer", "minmod": "minmod", "mc": "MC"}.get(self.limiter, "vanleer"),
        )
        solver.cfl_desired = self.cfl_desired
        solver.cfl_max = self.cfl_max

        edges = [
            (problem.bc_x_lower, 0, "lower"),
            (problem.bc_x_upper, 0, "upper"),
            (problem.bc_y_lower, 1, "lower"),
            (problem.bc_y_upper, 1, "upper"),
        ]
        n_periodic = sum(1 for s, _, _ in edges if s.kind == "periodic")
        if n_periodic not in (0, 2, 4):
            raise ValueError("periodic boundaries must come in matched pairs")
        for spec, dim_idx, which in edges:
            self._edge(pyclaw, solver, spec, dim_idx, which)

        x = pyclaw.Dimension(grid.xlower, grid.xupper, grid.nx, name="x")
        y = pyclaw.Dimension(grid.ylower, grid.yupper, grid.ny, name="y")
        domain = pyclaw.Domain([x, y])
        state = pyclaw.State(domain, 3, 1)
        state.problem_data["grav"] = float(self.g)
        state.problem_data["dry_tolerance"] = float(self.dry_tolerance)
        state.problem_data["sea_level"] = 0.0
        # Problem arrays are (Ny, Nx); PyClaw state is (eqn, Nx, Ny).
        state.aux[0] = problem.zb.T
        state.q[0] = problem.h0.T
        state.q[1] = problem.hu0.T
        state.q[2] = problem.hv0.T

        claw = pyclaw.Controller()
        claw.keep_copy = True
        claw.output_format = None
        claw.verbosity = 0
        claw.solution = pyclaw.Solution(state, domain)
        claw.solver = solver
        claw.tfinal = float(grid.t_end)
        claw.num_output_times = int(grid.n_t)
        with _suppress_fortran_stdout(self.quiet):
            claw.run()

        nt = len(claw.frames)
        ny, nx = grid.ny, grid.nx
        h = np.empty((nt, ny, nx))
        hu = np.empty((nt, ny, nx))
        hv = np.empty((nt, ny, nx))
        for k, fr in enumerate(claw.frames):
            h[k] = fr.q[0].T  # (Nx, Ny) -> (Ny, Nx)
            hu[k] = fr.q[1].T
            hv[k] = fr.q[2].T
        t = np.array([fr.t for fr in claw.frames], dtype=float)

        eps = 1e-12
        u = hu / np.maximum(h, eps)
        v = hv / np.maximum(h, eps)
        dry = h < self.dry_tolerance
        u[dry] = 0.0
        v[dry] = 0.0
        eta = h + problem.zb[None, :, :]
        return SimResult(
            t=t,
            x=grid.x_centers,
            zb=problem.zb,
            eta=eta,
            u=u,
            h=h,
            y=grid.y_centers,
            v=v,
            meta={"solver": self.name, "kernel": self.kernel, "limiter": self.limiter},
        )
