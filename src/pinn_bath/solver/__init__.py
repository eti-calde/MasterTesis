"""Forward 1D shallow-water solver(s) for generating inverse-operator data.

The operator-learning pivot (2026-05) needs ground-truth ``(zb -> eta(x,t),
u(x,t))`` pairs for *arbitrary* bathymetry, where no analytic solution exists.
:func:`forward_solve` wraps Clawpack's well-balanced solver behind a
solver-agnostic interface so the case generator / dataset pipeline never import
clawpack directly.
"""

from pinn_bath.solver.swe1d import DEFAULT_G, forward_solve, make_characteristic_inflow

__all__ = ["DEFAULT_G", "forward_solve", "make_characteristic_inflow"]
