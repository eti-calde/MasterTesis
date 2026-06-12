"""Dimension-agnostic contracts between environments and solver backends.

The flow is::

    Environment.sample_case(difficulty, rng)  ->  CaseSpec      (what to run)
    Environment.make_problem(CaseSpec)        ->  SimulationProblem
    SolverBackend.solve(SimulationProblem)    ->  SimResult     (tensors out)

``CaseSpec`` is pure parameters (frozen, regenerable from seeds);
``SimulationProblem`` is the discretised, solver-agnostic statement of one
run (arrays + declarative boundary specs); ``SimResult`` is the standard
``(t, x)`` tensor package the dataset builder and the operator consume.
Boundary conditions are *declarative* (:class:`BoundarySpec`) so each backend
maps them to its own machinery (PyClaw: ghost-cell callbacks; a hypothetical
SWASH backend: wavemaker config), keeping environments backend-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pinn_bath.datagen.bathymetry import BathymetryField, Difficulty
from pinn_bath.datagen.forcing import WaveForcing
from pinn_bath.datagen.grids import Grid1D

BoundaryKind = Literal["incident_wave", "outflow", "wall", "periodic"]


@dataclass(frozen=True)
class CaseSpec:
    """Full parametric description of one case (bed + excitation + labels)."""

    bathymetry: BathymetryField
    forcing: WaveForcing
    water_level: float  # tidal stage: still water level for this case [m]
    spring_neap: float  # f in [0, 1]: neap -> spring (drives stage + amplitude)
    difficulty: Difficulty
    seed: int
    score: float  # difficulty score on the *detrended* bed (slope-orthogonal)
    components: dict[str, float] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Flat, JSON-serialisable summary (manifests, logs, GIF titles)."""
        return {
            "difficulty": self.difficulty,
            "seed": self.seed,
            "score": self.score,
            "slope": self.bathymetry.slope,
            "n_features": len(self.bathymetry.features),
            "water_level": self.water_level,
            "spring_neap": self.spring_neap,
            "inflow_side": self.forcing.side,
            "wave_amps": list(self.forcing.amps),
            "wave_periods": list(self.forcing.periods),
            **{f"score_{k}": v for k, v in self.components.items()},
        }


@dataclass(frozen=True)
class BoundarySpec:
    """Declarative boundary condition for one domain edge.

    ``incident_wave`` requires ``eta_signal`` (surface perturbation ``t ->
    delta_eta`` at the edge) and ``h_rest`` (still depth at the edge); the
    other kinds carry no payload. ``periodic`` must be used on both edges.
    """

    kind: BoundaryKind
    eta_signal: Callable[[float], float] | None = None
    h_rest: float | None = None

    def __post_init__(self) -> None:
        if self.kind == "incident_wave" and (self.eta_signal is None or self.h_rest is None):
            raise ValueError("incident_wave boundary requires eta_signal and h_rest")


@dataclass(frozen=True)
class SimulationProblem:
    """One discretised 1D run: arrays at cell centres + edge conditions."""

    grid: Grid1D
    zb: np.ndarray  # (Nx,) bed elevation
    h0: np.ndarray  # (Nx,) initial depth
    hu0: np.ndarray  # (Nx,) initial discharge
    bc_lower: BoundarySpec
    bc_upper: BoundarySpec


@dataclass
class SimResult:
    """Standard tensor package: ``(Nt, Nx)`` fields on the shared grid.

    2D extension: ``y``/``v`` become arrays and the fields gain an axis
    (``(Nt, Ny, Nx)``); consumers should treat ``y is None`` as "1D".
    """

    t: np.ndarray  # (Nt,)
    x: np.ndarray  # (Nx,)
    zb: np.ndarray  # (Nx,)
    eta: np.ndarray  # (Nt, Nx)
    u: np.ndarray  # (Nt, Nx)
    h: np.ndarray  # (Nt, Nx)
    y: np.ndarray | None = None
    v: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Finite everywhere (the builder drops the rare unstable case)."""
        return bool(np.isfinite(self.eta).all() and np.isfinite(self.u).all())
