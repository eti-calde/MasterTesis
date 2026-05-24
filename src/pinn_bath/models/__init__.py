"""PINN architectures (A1/A2/A3) and shared building blocks."""

from pinn_bath.models.a1 import A1TwoNets
from pinn_bath.models.a2 import A2Monolithic
from pinn_bath.models.a3 import A3PerField
from pinn_bath.models.base import BaseModel, PerAxisFourier, softplus_positive
from pinn_bath.models.blocks import MLP, FourierFeatures, count_parameters
from pinn_bath.models.factory import BUDGET_TARGETS, BUDGET_TOL, build, shape_for

__all__ = [
    "BUDGET_TARGETS",
    "BUDGET_TOL",
    "MLP",
    "A1TwoNets",
    "A2Monolithic",
    "A3PerField",
    "BaseModel",
    "FourierFeatures",
    "PerAxisFourier",
    "build",
    "count_parameters",
    "shape_for",
    "softplus_positive",
]
