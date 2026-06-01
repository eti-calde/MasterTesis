r"""1D shallow-water forward solver via PyClaw (Clawpack).

Generates ground-truth ``(zb -> eta(x,t), u(x,t))`` for arbitrary 1D
bathymetry, used to build the inverse-operator training set (no analytic
solution for arbitrary bumps + holes). Wraps Clawpack's well-balanced f-wave
Riemann solver ``shallow_bathymetry_fwave_1D`` (2nd-order TVD) behind a
solver-agnostic :func:`forward_solve`.

Physics matches ``pinn_bath.losses.residual`` (``g = 9.81``). The SWE solved:

.. math::
    h_t + (h u)_x &= 0 \\
    (h u)_t + (h u^2 + \tfrac{1}{2} g h^2)_x &= -g h\, (z_b)_x

clawpack is an *optional* dependency (data-generation only) — it is imported
lazily inside :func:`forward_solve`, so this module imports fine without it.
Install with ``uv pip install clawpack`` (needs gfortran) or via conda-forge.

Conventions
-----------
- ``zb`` is bed elevation (Clawpack's ``aux[0]`` = bathymetry ``b``); the free
  surface is ``eta = h + zb``.
- State ``q = [h, h u]``; ``u`` is desingularised and zeroed below
  ``dry_tolerance``.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any

import numpy as np

DEFAULT_G = 9.81


@contextmanager
def _suppress_fortran_stdout(enabled: bool = True):
    """Silence OS-level stdout (fd 1) around a block.

    The augmented GeoClaw kernel prints ``Negative input: hl,hr`` from Fortran
    when it clips O(1e-4) negative depths at dry fronts (harmless — it recovers
    and stays <1% accurate). Those prints bypass Python's ``sys.stdout``, so we
    redirect the underlying file descriptor; over thousands of cases this spam
    would otherwise drown the pipeline.
    """
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


# Clawpack Riemann kernels usable for 1D SWE with topography.
_KERNELS = {
    # Well-balanced f-wave (documented in pyclaw examples/shallow_1d/sill.py).
    "fwave": "shallow_bathymetry_fwave_1D",
    # Augmented (GeoClaw-style) solver — more robust to wet/dry fronts.
    "aug": "sw_aug_1D",
}


def forward_solve(
    zb: np.ndarray,
    h0: np.ndarray,
    hu0: np.ndarray,
    *,
    xlower: float,
    xupper: float,
    t_end: float,
    num_output_times: int = 100,
    g: float = DEFAULT_G,
    bc: str = "extrap",
    kernel: str = "aug",
    dry_tolerance: float = 1e-3,
    sea_level: float = 0.0,
    cfl_desired: float = 0.8,
    cfl_max: float = 0.9,
    limiter: str = "vanleer",
    quiet: bool = True,
) -> dict[str, Any]:
    """Run a 1D SWE forward simulation over bathymetry ``zb``.

    Parameters
    ----------
    zb, h0, hu0 : (Nx,) arrays
        Bed elevation, initial depth, initial discharge at cell centres.
    xlower, xupper : float
        Domain bounds (cell-centred grid of ``len(zb)`` cells).
    t_end : float
        Final simulation time.
    num_output_times : int
        Number of saved snapshots after ``t=0`` (total frames = this + 1).
    g : float
        Gravity (match the SWE residual; default 9.81).
    bc : {"extrap", "wall", "periodic"}
        Boundary condition on both ends. ``extrap`` = transmissive/outflow;
        ``wall`` = reflective; ``periodic`` = wrap.
    kernel : {"aug", "fwave"}
        Riemann kernel. ``aug`` (default) is the augmented GeoClaw-style
        solver, robust to wet/dry fronts (required for drying basins; validated
        against the Thacker analytic solution to <1%). ``fwave`` is the
        documented well-balanced f-wave — fine for always-wet flow but
        blows up at dry fronts.
    dry_tolerance : float
        Depth below which a cell is treated as dry (velocity zeroed).

    Returns
    -------
    dict
        ``t`` (Nt,), ``x`` (Nx,), ``h``/``hu``/``u``/``eta`` (Nt, Nx),
        ``zb`` (Nx,).
    """
    from clawpack import pyclaw, riemann

    zb = np.asarray(zb, dtype=float)
    h0 = np.asarray(h0, dtype=float)
    hu0 = np.asarray(hu0, dtype=float)
    nx = zb.shape[0]
    if h0.shape != (nx,) or hu0.shape != (nx,):
        raise ValueError(f"zb/h0/hu0 must all be 1D of length {nx}")

    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {sorted(_KERNELS)}; got {kernel!r}")
    rp = getattr(riemann, _KERNELS[kernel])
    solver = pyclaw.ClawSolver1D(rp)
    solver.num_waves = 2
    solver.num_eqn = 2
    # Both kernels are f-wave based (the augmented solver also requires it).
    solver.fwave = True
    solver.limiters = getattr(
        pyclaw.limiters.tvd,
        {"vanleer": "vanleer", "minmod": "minmod", "mc": "MC"}.get(limiter, "vanleer"),
    )
    solver.cfl_desired = cfl_desired
    solver.cfl_max = cfl_max

    bcmap = {"extrap": pyclaw.BC.extrap, "wall": pyclaw.BC.wall, "periodic": pyclaw.BC.periodic}
    if bc not in bcmap:
        raise ValueError(f"bc must be one of {sorted(bcmap)}; got {bc!r}")
    solver.bc_lower[0] = bcmap[bc]
    solver.bc_upper[0] = bcmap[bc]
    # Bathymetry (aux) cannot be reflected like a state; extrapolate at walls.
    aux_bc = bcmap[bc] if bc != "wall" else pyclaw.BC.extrap
    solver.aux_bc_lower[0] = aux_bc
    solver.aux_bc_upper[0] = aux_bc

    x = pyclaw.Dimension(xlower, xupper, nx, name="x")
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 2, 1)
    state.problem_data["grav"] = float(g)
    state.problem_data["dry_tolerance"] = float(dry_tolerance)
    state.problem_data["sea_level"] = float(sea_level)
    state.aux[0, :] = zb
    state.q[0, :] = h0
    state.q[1, :] = hu0

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.output_format = None
    claw.verbosity = 0
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.tfinal = float(t_end)
    claw.num_output_times = int(num_output_times)
    with _suppress_fortran_stdout(quiet):
        claw.run()

    xc = np.asarray(domain.grid.x.centers, dtype=float)
    frames = claw.frames
    nt = len(frames)
    h = np.empty((nt, nx))
    hu = np.empty((nt, nx))
    for k, fr in enumerate(frames):
        h[k] = fr.q[0]
        hu[k] = fr.q[1]
    t = np.array([fr.t for fr in frames], dtype=float)

    eps = 1e-12
    u = hu / np.maximum(h, eps)
    u[h < dry_tolerance] = 0.0
    eta = h + zb[None, :]
    return {"t": t, "x": xc, "h": h, "hu": hu, "u": u, "eta": eta, "zb": zb}
