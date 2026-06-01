"""Field→field operator architectures for inverse bathymetry (F3).

Input: observation field ``(eta, u)`` as channels over the (t, x) grid,
shape ``(B, 2, Nt, Nx)``. Output: ``zb(x)`` (normalized), shape ``(B, Nx)``.

``CNN1DOperator``: a 2D-conv **residual** encoder over (t, x) that preserves the
x-resolution, collapses the time axis by adaptive pooling, then a 1D-conv
**residual** decoder over x → ``zb(x)``. Residual blocks + GroupNorm let the
network scale in depth/width without the optimization degradation that plain
stacked convs show; this is what makes the ``medium``/``large`` presets train
stably. Translation-equivariant in x.

Size presets (see :data:`PRESETS`) are the knob for the scaling study; an FNO
variant is planned as a second architecture for the operator comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _groups(width: int) -> int:
    """Pick a GroupNorm group count that divides ``width`` (<= 8)."""
    for g in (8, 4, 2, 1):
        if width % g == 0:
            return g
    return 1


class _ResBlock2d(nn.Module):
    """Conv2d → GN → GELU → Conv2d → GN, with identity skip (constant width)."""

    def __init__(self, width: int, kernel: tuple[int, int] = (3, 5)) -> None:
        super().__init__()
        pad = (kernel[0] // 2, kernel[1] // 2)
        g = _groups(width)
        self.body = nn.Sequential(
            nn.Conv2d(width, width, kernel, padding=pad),
            nn.GroupNorm(g, width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel, padding=pad),
            nn.GroupNorm(g, width),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class _ResBlock1d(nn.Module):
    """Conv1d → GN → GELU → Conv1d → GN, with identity skip (constant width)."""

    def __init__(self, width: int, kernel: int = 5) -> None:
        super().__init__()
        g = _groups(width)
        self.body = nn.Sequential(
            nn.Conv1d(width, width, kernel, padding=kernel // 2),
            nn.GroupNorm(g, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel, padding=kernel // 2),
            nn.GroupNorm(g, width),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class CNN1DOperator(nn.Module):
    """Residual 2D-conv encoder over (t, x) → pool t → residual 1D-conv decoder.

    Parameters
    ----------
    in_fields : input channels (2 for η,u).
    width : channel width throughout.
    t_blocks, x_blocks : number of residual blocks in encoder / decoder.
    """

    def __init__(
        self,
        in_fields: int = 2,
        width: int = 48,
        t_blocks: int = 3,
        x_blocks: int = 3,
    ) -> None:
        super().__init__()
        g = _groups(width)
        # Stem lifts input channels → width (no residual across the channel jump).
        self.stem = nn.Sequential(
            nn.Conv2d(in_fields, width, kernel_size=(3, 5), padding=(1, 2)),
            nn.GroupNorm(g, width),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(*[_ResBlock2d(width) for _ in range(t_blocks)])
        # Collapse the time axis only (keep Nx via None).
        self.collapse_t = nn.AdaptiveAvgPool2d((1, None))
        self.decoder = nn.Sequential(*[_ResBlock1d(width) for _ in range(x_blocks)])
        self.head = nn.Conv1d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_fields, Nt, Nx)
        h = self.stem(x)
        h = self.encoder(h)  # (B, width, Nt, Nx)
        h = self.collapse_t(h).squeeze(-2)  # (B, width, Nx)
        h = self.decoder(h)  # (B, width, Nx)
        return self.head(h).squeeze(-2)  # (B, Nx) normalized zb


# Size presets — the scaling knob. Widths divisible by 8 for clean GroupNorm.
PRESETS: dict[str, dict[str, int]] = {
    "small": {"width": 48, "t_blocks": 3, "x_blocks": 3},
    "medium": {"width": 96, "t_blocks": 4, "x_blocks": 4},
    "large": {"width": 160, "t_blocks": 6, "x_blocks": 6},
}


def build_operator(
    arch: str = "cnn",
    *,
    size: str | None = None,
    **kwargs: object,
) -> nn.Module:
    """Factory. ``size`` in {small, medium, large} applies a preset; explicit
    kwargs (width/t_blocks/x_blocks/in_fields) override preset entries.
    """
    if arch != "cnn":
        raise ValueError(f"unknown operator arch: {arch!r}")
    params: dict[str, object] = {}
    if size is not None:
        if size not in PRESETS:
            raise ValueError(f"unknown size {size!r}; choose from {sorted(PRESETS)}")
        params.update(PRESETS[size])
    params.update(kwargs)  # explicit kwargs win
    return CNN1DOperator(**params)  # type: ignore[arg-type]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
