"""Loss functions for inverse SWE PINN: SWE residuals + composite terms."""

from pinn_bath.losses.bc import flat_bed_loss, periodic_bc_loss
from pinn_bath.losses.components import (
    boundary_dirichlet,
    data_mse,
    discharge,
    pde_mse,
    positivity,
    tikhonov,
    tv_1d,
    tv_2d,
)
from pinn_bath.losses.residual import Form, swe_residual

__all__ = [
    "Form",
    "boundary_dirichlet",
    "data_mse",
    "discharge",
    "flat_bed_loss",
    "pde_mse",
    "periodic_bc_loss",
    "positivity",
    "swe_residual",
    "tikhonov",
    "tv_1d",
    "tv_2d",
]
