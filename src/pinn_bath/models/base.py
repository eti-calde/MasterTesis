"""Common interface for PINN architectures."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from pinn_bath.models.blocks import FourierFeatures

Field = Literal["h", "u", "v", "zb"]
Coord = Literal["x", "y", "t"]


class BaseModel(nn.Module):
    """Shared interface for A1/A2/A3.

    A model maps a dict of input coordinates (each shape ``(N, 1)``) to a dict
    of output fields. Subclasses set ``spatial_dim``, ``has_t``,
    ``output_fields`` and implement :meth:`forward`.
    """

    spatial_dim: int
    has_t: bool
    output_fields: tuple[Field, ...]

    def forward(self, coords: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def spatial_coords(spatial_dim: int) -> tuple[Coord, ...]:
        return ("x", "y") if spatial_dim == 2 else ("x",)

    @staticmethod
    def all_coords(spatial_dim: int, has_t: bool) -> tuple[Coord, ...]:
        axes: tuple[Coord, ...] = BaseModel.spatial_coords(spatial_dim)
        if has_t:
            axes = (*axes, "t")
        return axes


class PerAxisFourier(nn.Module):
    """Apply :class:`FourierFeatures` separately to each named axis, then concat.

    Each axis goes through its own ``FourierFeatures(in_dim=1, n, sigma)`` and
    the outputs are concatenated along the last dimension. This matches the
    convention used in the thesis (Exp 2 / Exp 3).
    """

    def __init__(
        self,
        axes: tuple[Coord, ...],
        n_features: int,
        sigma: float,
        *,
        seed_base: int | None = None,
    ) -> None:
        super().__init__()
        self.axes: tuple[Coord, ...] = axes
        self.n_features = n_features
        self.sigma = sigma
        for i, axis in enumerate(axes):
            ff = FourierFeatures(
                in_dim=1,
                n_features=n_features,
                sigma=sigma,
                seed=(seed_base + i) if seed_base is not None else None,
            )
            self.add_module(f"ff_{axis}", ff)
        self.out_dim = len(axes) * 2 * n_features

    def forward(self, coords: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [getattr(self, f"ff_{a}")(coords[a]) for a in self.axes]
        return torch.cat(parts, dim=-1)


def stack_coords(coords: dict[str, torch.Tensor], axes: tuple[Coord, ...]) -> torch.Tensor:
    """Concatenate the named coord tensors along the last dim (for non-Fourier nets)."""
    return torch.cat([coords[a] for a in axes], dim=-1)


__all__ = [
    "BaseModel",
    "Coord",
    "Field",
    "PerAxisFourier",
    "softplus_positive",
    "stack_coords",
]


def softplus_positive(raw: torch.Tensor) -> torch.Tensor:
    """Strictly positive activation for water depth ``h``."""
    return nn.functional.softplus(raw)
