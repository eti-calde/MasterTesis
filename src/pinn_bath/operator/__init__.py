"""Amortized inverse operator for 1D bathymetry (operator-learning pivot).

Maps a gridded shallow-water observation field ``(eta(x,t), u(x,t))`` to the
bathymetry ``zb(x)``, trained supervised across a distribution of cases, with
an optional finite-difference SWE residual as a physics regularizer (the
experimental "does physics help OOD?" variable).

See memory ``operator-pivot`` for the methodology. Datasets come from
:mod:`pinn_bath.datasets.operator_dataset`.
"""

from pinn_bath.operator.physics import physics_loss, swe_residual_grid

__all__ = ["physics_loss", "swe_residual_grid"]
