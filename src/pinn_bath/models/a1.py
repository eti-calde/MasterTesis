"""A1: dos redes (propuesta de la tesis).

SolutionNet con embedding de Fourier por eje toma todas las coordenadas
(espaciales + temporal si hay) y produce :math:`(h, u, [v])`.
BathymetryNet con embedding de Fourier toma sólo coordenadas espaciales y
produce :math:`z_b`; la ausencia de :math:`t` en su entrada impone
:math:`\\partial z_b / \\partial t = 0` de forma estructural.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pinn_bath.models.base import BaseModel, Field, PerAxisFourier, softplus_positive
from pinn_bath.models.blocks import MLP


class A1TwoNets(BaseModel):
    def __init__(
        self,
        *,
        spatial_dim: int,
        has_t: bool,
        output_fields: tuple[Field, ...],
        sol_depth: int,
        sol_width: int,
        bath_depth: int,
        bath_width: int,
        ff_n: int = 16,
        ff_sigma: float = 2.0,
        ff_seed: int | None = None,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.has_t = has_t
        self.output_fields = tuple(output_fields)

        sol_axes = self.all_coords(spatial_dim, has_t)
        self.sol_ff = PerAxisFourier(sol_axes, ff_n, ff_sigma, seed_base=ff_seed)
        flow_fields: tuple[Field, ...] = tuple(
            f for f in ("h", "u", "v") if f in self.output_fields
        )
        if "h" not in flow_fields:
            raise ValueError("A1 requires 'h' in output_fields.")
        self._flow_field_order = flow_fields
        self.sol_net = MLP(self.sol_ff.out_dim, len(flow_fields), sol_depth, sol_width, activation)

        bath_axes = self.spatial_coords(spatial_dim)
        bath_seed = None if ff_seed is None else ff_seed + 100
        self.bath_ff = PerAxisFourier(bath_axes, ff_n, ff_sigma, seed_base=bath_seed)
        self.bath_net = MLP(self.bath_ff.out_dim, 1, bath_depth, bath_width, activation)

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sol_feat = self.sol_ff(coords)
        sol_out = self.sol_net(sol_feat)
        out: dict[str, torch.Tensor] = {}
        for i, field in enumerate(self._flow_field_order):
            col = sol_out[:, i : i + 1]
            out[field] = softplus_positive(col) if field == "h" else col
        if "zb" in self.output_fields:
            bath_feat = self.bath_ff(coords)
            out["zb"] = self.bath_net(bath_feat)
        return out
