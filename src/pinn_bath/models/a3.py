"""A3: una MLP por cada campo (estilo Ohara).

Para cada campo de salida hay una MLP totalmente conectada independiente,
sobre las coordenadas crudas. La red de :math:`z_b` recibe sólo las
coordenadas espaciales, lo que impone
:math:`\\partial z_b / \\partial t = 0` estructuralmente.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pinn_bath.models.base import BaseModel, Coord, Field, softplus_positive, stack_coords
from pinn_bath.models.blocks import MLP


class A3PerField(BaseModel):
    def __init__(
        self,
        *,
        spatial_dim: int,
        has_t: bool,
        output_fields: tuple[Field, ...],
        depth: int,
        width: int,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.has_t = has_t
        self.output_fields = tuple(output_fields)
        if "h" not in self.output_fields:
            raise ValueError("A3 requires 'h' in output_fields.")

        flow_axes = self.all_coords(spatial_dim, has_t)
        spatial_axes = self.spatial_coords(spatial_dim)
        axes_per_field: dict[Field, tuple[Coord, ...]] = {
            field: (spatial_axes if field == "zb" else flow_axes) for field in self.output_fields
        }
        for field, axes in axes_per_field.items():
            net = MLP(len(axes), 1, depth, width, activation)
            self.add_module(f"net_{field}", net)
        self._axes_per_field = axes_per_field

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for field, axes in self._axes_per_field.items():
            net: MLP = getattr(self, f"net_{field}")
            raw = net(stack_coords(coords, axes))
            out[field] = softplus_positive(raw) if field == "h" else raw
        return out
