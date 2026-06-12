r"""Tidal excitation axes: spring-neap factor, stage, and incident wave train.

One latent **spring-neap factor** ``f ~ U(0, 1)`` per case (the position in
the fortnightly modulation cycle, neap -> spring) coherently drives both
tidal axes, and is sampled identically across difficulty tiers so the
OOD-by-difficulty split stays purely bathymetric:

- **Tidal stage** (still water level): ``wl = H0 * (1 + 0.15 f sin(theta))``
  with phase ``theta ~ U(0, 2pi)``. At springs the stage swings +/-15% of
  H0; at neaps it barely moves (the physically correct coupling: extreme
  low/high water only happens on spring tides). Under Froude similarity 1:25
  the +/-15% half-range maps to +/-3.75 m on a 25 m depth, matching the
  spring range of the Reloncavi / Puerto Montt area (~7 m, among Chile's
  largest). Uniform-in-time sampling of a sinusoid yields the arcsine
  density: more mass near high and low water, as a real tide spends its time.
- **Incident-train total amplitude**: ``A_tot = H0 * (0.02 + 0.10 f)``, from
  2 cm (marea muerta) to 12 cm (agua viva), split among 1-3 sinusoidal
  components by Dirichlet weights. ``a/h <= 0.12`` keeps nonlinearity
  moderate (no bore formation within the window); induced currents
  ``u ~ A_tot sqrt(g/H0)`` span 0.06-0.38 m/s (field: 0.3-1.9 m/s, quiet
  interior to energetic constrictions at springs).

**Periods** ``T in [5, 9] s`` put the train in the hydrostatic band:
``kh = 2 pi sqrt(H0/g) / T in [0.22, 0.40]``, i.e. a celerity error of the
non-dispersive approximation of 0.8-2.5% (an order of magnitude below the
bathymetric celerity signal being inverted). Field reading (x5): T = 25-45 s,
the infragravity/seiche band, genuinely hydrostatic at 25 m depth; ocean
swell (10-20 s, kh >~ 0.8 at 25 m) is dispersive and correctly excluded from
an SWE benchmark. kh is Froude-invariant, so this holds at any scale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

Side = Literal["left", "right"]

# Stage half-range at full springs (fraction of sea_level). 1:25 field
# reading: +/-3.75 m on 25 m, ~ the Reloncavi spring tide half-range.
SPRING_STAGE_FRAC: float = 0.15
# Incident-train total amplitude (fraction of sea_level): neap -> spring.
WAVE_AMP_FRAC_NEAP: float = 0.02
WAVE_AMP_FRAC_SPRING: float = 0.12
# Hydrostatic band: kh = 2 pi sqrt(H0/g) / T in [0.22, 0.40] for H0 = 1 m.
WAVE_PERIOD_RANGE: tuple[float, float] = (5.0, 9.0)
# tanh ramp time ~ half the shortest period (smooth turn-on).
RAMP_TAU_RANGE: tuple[float, float] = (2.0, 4.0)


@dataclass(frozen=True)
class TidalState:
    """Per-case tide: spring-neap factor and the resulting still water level."""

    spring_neap: float  # f in [0, 1]: neap -> spring
    water_level: float  # still water level (stage) [m]


@dataclass(frozen=True)
class WaveForcing:
    """One case's incident wave train (enters at ``side``, exits opposite)."""

    side: Side
    amps: tuple[float, ...]  # metres (delta-eta per component)
    periods: tuple[float, ...]  # seconds
    phases: tuple[float, ...]  # radians
    ramp_tau: float  # seconds (tanh ramp time)

    def signal(self) -> Callable[[float], float]:
        """Boundary surface perturbation ``delta_eta(t)``: ramped sinusoid sum."""
        amps = np.asarray(self.amps, dtype=float)
        periods = np.asarray(self.periods, dtype=float)
        phases = np.asarray(self.phases, dtype=float)
        tau = float(self.ramp_tau)

        def eta_signal(t: float) -> float:
            ramp = np.tanh(t / tau)
            return float(ramp * np.sum(amps * np.sin(2.0 * np.pi * t / periods + phases)))

        return eta_signal


@dataclass(frozen=True)
class ForcingSampler:
    """Draws the excitation axes (spring-neap tide + wave train) per case."""

    stage_frac: float = SPRING_STAGE_FRAC
    amp_frac_neap: float = WAVE_AMP_FRAC_NEAP
    amp_frac_spring: float = WAVE_AMP_FRAC_SPRING
    period_range: tuple[float, float] = WAVE_PERIOD_RANGE
    ramp_tau_range: tuple[float, float] = RAMP_TAU_RANGE
    max_components: int = 3

    def min_water_level(self, sea_level: float) -> float:
        """Lowest possible stage (spring low water); anchors the deep-water cap."""
        return (1.0 - self.stage_frac) * sea_level

    def sample_tide(self, rng: np.random.Generator, sea_level: float) -> TidalState:
        """Draw the spring-neap factor and the coupled tidal stage."""
        f = float(rng.uniform(0.0, 1.0))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        wl = sea_level * (1.0 + self.stage_frac * f * np.sin(theta))
        return TidalState(spring_neap=f, water_level=float(wl))

    def sample_wave(
        self, rng: np.random.Generator, sea_level: float, spring_neap: float
    ) -> WaveForcing:
        """Draw the incident train; total amplitude is set by the f factor."""
        n_comp = int(rng.integers(1, self.max_components + 1))
        a_tot = (
            self.amp_frac_neap + (self.amp_frac_spring - self.amp_frac_neap) * spring_neap
        ) * sea_level
        weights = rng.dirichlet(np.ones(n_comp))
        amps = tuple(float(a_tot * w) for w in weights)
        periods = tuple(float(p) for p in rng.uniform(*self.period_range, size=n_comp))
        phases = tuple(float(ph) for ph in rng.uniform(0.0, 2.0 * np.pi, size=n_comp))
        side: Side = "left" if rng.random() < 0.5 else "right"
        ramp_tau = float(rng.uniform(*self.ramp_tau_range))
        return WaveForcing(side=side, amps=amps, periods=periods, phases=phases, ramp_tau=ramp_tau)
