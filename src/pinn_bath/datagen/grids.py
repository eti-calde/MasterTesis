"""Fixed space-time grids shared by every case in a dataset.

Every case in a dataset is solved on the *same* discretisation so the
operator sees field->field on a consistent grid. ``Grid1D`` is the v2 1D
bank; ``Grid2D`` extends the same physical setup (reference depth, window,
incident band) to a 2D channel, so the kh / tidal / slope justifications
carry over unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid1D:
    """v2 bank defaults (physically justified; see paper, 'Justificación
    física de los parámetros'): L = 40 H0 holds 1.4-2.5 wavelengths of the
    hydrostatic incident band (T = 5-9 s -> kh = 0.22-0.40 at H0); dx = 7.8 cm
    gives >=10 cells across the narrowest feature and ~190 per shortest
    wavelength; t_end = one establishment crossing at the lowest tide
    (L/sqrt(g h_min) ~= 14 s) plus >=3 periods of the longest component;
    dt_out = 0.25 s samples the shortest period >=20 times. Froude similarity
    1:25 reads this as a 1 km reach of a 25 m deep fjord margin.
    """

    xlower: float = 0.0
    xupper: float = 40.0
    nx: int = 512
    t_end: float = 40.0
    n_t: int = 160  # snapshots after t=0 (total frames = n_t + 1)
    sea_level: float = 1.0  # reference still-water free surface (eta_rest)

    @property
    def dx(self) -> float:
        return (self.xupper - self.xlower) / self.nx

    @property
    def centers(self) -> np.ndarray:
        return np.linspace(self.xlower + self.dx / 2, self.xupper - self.dx / 2, self.nx)


@dataclass(frozen=True)
class Grid2D:
    """2D channel grid: same physical regime as :class:`Grid1D` (reference
    depth, 40 s window, hydrostatic incident band), with a transverse y axis.
    Ly = 20 m holds ~1 wavelength across the channel; dx = dy ~= 0.156 m keeps
    the narrowest tier feature at >= 5 cells across its minor axis (preview
    resolution; revisit for the production 2D bank). Field arrays follow the
    ``(Nt, Ny, Nx)`` layout; PyClaw's internal (x, y) order is handled by the
    backend.
    """

    xlower: float = 0.0
    xupper: float = 40.0
    nx: int = 256
    ylower: float = 0.0
    yupper: float = 20.0
    ny: int = 128
    t_end: float = 40.0
    n_t: int = 160  # snapshots after t=0 (total frames = n_t + 1)
    sea_level: float = 1.0  # reference still-water free surface (eta_rest)

    @property
    def dx(self) -> float:
        return (self.xupper - self.xlower) / self.nx

    @property
    def dy(self) -> float:
        return (self.yupper - self.ylower) / self.ny

    @property
    def x_centers(self) -> np.ndarray:
        return np.linspace(self.xlower + self.dx / 2, self.xupper - self.dx / 2, self.nx)

    @property
    def y_centers(self) -> np.ndarray:
        return np.linspace(self.ylower + self.dy / 2, self.yupper - self.dy / 2, self.ny)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre meshgrid ``(X, Y)``, each of shape ``(Ny, Nx)``."""
        return np.meshgrid(self.x_centers, self.y_centers)
