"""Shared neural network building blocks: Fourier features and MLPs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    r"""Random Fourier feature embedding.

    Maps :math:`x \in \mathbb{R}^{d}` to
    :math:`[\sin(2\pi B x), \cos(2\pi B x)] \in \mathbb{R}^{2m}` with
    :math:`B \in \mathbb{R}^{m \times d}`, :math:`B_{ij} \sim \mathcal{N}(0, \sigma^2)`.

    The matrix :math:`B` is a fixed (non-trainable) buffer, set at
    construction time. Reference: Tancik et al. (2020).
    """

    def __init__(
        self,
        in_dim: int,
        n_features: int,
        sigma: float,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is None:
            B = torch.randn(in_dim, n_features) * sigma
        else:
            gen = torch.Generator()
            gen.manual_seed(seed)
            B = torch.randn(in_dim, n_features, generator=gen) * sigma
        self.register_buffer("B", B)
        self.in_dim = in_dim
        self.n_features = n_features
        self.sigma = sigma

    @property
    def out_dim(self) -> int:
        return 2 * self.n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    """Plain feed-forward MLP with constant hidden width.

    Architecture::

        Linear(in_dim, width) -> act -> [Linear(width, width) -> act] x (depth-1)
            -> Linear(width, out_dim)

    so ``depth`` counts the hidden layers (input projection included) of
    width ``width``. The final projection to ``out_dim`` is added on top.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        width: int,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        layers: list[nn.Module] = [nn.Linear(in_dim, width), activation()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(activation())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
