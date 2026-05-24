"""A2: arquitectura monolítica (estilo Dazzi).

Una sola MLP toma las coordenadas crudas (sin Fourier features) y produce
conjuntamente todos los campos de salida, incluyendo :math:`z_b`. La condición
:math:`\\partial z_b / \\partial t = 0` se debe imponer como término de pérdida
(no estructuralmente).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pinn_bath.models.base import BaseModel, Field, softplus_positive, stack_coords
from pinn_bath.models.blocks import MLP


class A2Monolithic(BaseModel):
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

        self._input_axes = self.all_coords(spatial_dim, has_t)
        flow_fields: tuple[Field, ...] = tuple(
            f for f in ("h", "u", "v") if f in self.output_fields
        )
        if "h" not in flow_fields:
            raise ValueError("A2 requires 'h' in output_fields.")
        self._flow_field_order = flow_fields
        out_dim = len(flow_fields) + (1 if "zb" in self.output_fields else 0)
        self.net = MLP(len(self._input_axes), out_dim, depth, width, activation)

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = stack_coords(coords, self._input_axes)
        raw = self.net(x)
        out: dict[str, torch.Tensor] = {}
        i = 0
        for field in self._flow_field_order:
            col = raw[:, i : i + 1]
            out[field] = softplus_positive(col) if field == "h" else col
            i += 1
        if "zb" in self.output_fields:
            out["zb"] = raw[:, i : i + 1]
        return out
