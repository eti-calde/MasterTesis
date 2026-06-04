"""Incident-wave regime: characteristic inflow BC + deep-water dataset.

Covers the F-pivot change to the dataset forcing (continuous incident wave
train entering one boundary, transmissive outflow on the other):

1. The custom inflow with **zero forcing** is (near) non-reflecting — an
   interior pulse leaves through it just like a plain ``extrap`` outflow.
2. With forcing on, the prescribed wave train actually enters the domain
   (interior surface oscillates at ~the prescribed period) — and stays
   bounded (no resonant growth, since energy leaves through the outflow).
3. Well-balancing survives the custom BC: a bumpy (non-emergent) bed at rest
   with forcing off stays at rest (no spurious currents).
4. Deep-water bank: no tier emerges — ``max(zb) < sea_level`` always.

The solver runs use a small grid so they stay quick enough for ``-m fast``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinn_bath.datasets.generator import (
    Grid,
    bathymetry,
    sample_case,
    wave_signal,
)
from pinn_bath.solver import forward_solve, make_characteristic_inflow

G = 9.81
SL = 1.0  # still free surface
_SOLVE_KW = dict(kernel="aug", cfl_desired=0.45, dry_tolerance=1e-3)


def _flat(nx: int = 120):
    x = np.linspace(0.0, 10.0, nx)
    zb = np.zeros(nx)
    return zb, x


def _energy(eta: np.ndarray) -> np.ndarray:
    """Surface-perturbation energy per frame."""
    return ((eta - SL) ** 2).mean(axis=1)


@pytest.mark.fast
def test_zero_forcing_inflow_is_nonreflecting() -> None:
    # An interior Gaussian hump splits into two pulses. With a transmissive
    # outflow on the right and a *zero-forcing* custom inflow on the left, both
    # pulses must leave the domain — leaving ~as little residual energy as a
    # plain extrap/extrap reference. A reflecting inflow would trap the
    # left-going pulse and keep the residual high.
    nx = 160
    zb, x = _flat(nx)
    eta0 = SL + 0.1 * np.exp(-(((x - 5.0) / 0.5) ** 2))
    h0 = eta0 - zb
    hu0 = np.zeros_like(h0)
    quiet_signal = lambda t: 0.0  # noqa: E731 — forcing off
    cb = make_characteristic_inflow(quiet_signal, h_rest=SL, side="lower")

    common = dict(
        xlower=0.0, xupper=10.0, t_end=6.0, num_output_times=60, **_SOLVE_KW
    )
    sol_custom = forward_solve(
        zb, h0, hu0, bc_lower="custom", bc_upper="extrap", user_bc_lower=cb, **common
    )
    sol_ref = forward_solve(zb, h0, hu0, bc="extrap", **common)

    e_custom = _energy(sol_custom["eta"])
    e_ref = _energy(sol_ref["eta"])
    e0 = e_custom[0]
    # Both drain to a small fraction of the initial energy ...
    assert e_custom[-1] < 0.10 * e0, f"residual too high: {e_custom[-1] / e0:.3f}"
    # ... and the custom inflow drains essentially like a plain outflow.
    assert abs(e_custom[-1] - e_ref[-1]) < 0.05 * e0


@pytest.mark.fast
def test_forcing_enters_and_stays_bounded() -> None:
    # With a single-component train of period T entering the left edge, the
    # interior surface must (a) oscillate (energy well above rest) and (b) stay
    # bounded — peak displacement only a few times the forcing amplitude, since
    # the outflow lets energy escape (no resonance).
    nx = 160
    zb, _ = _flat(nx)
    h0 = np.full(nx, SL)
    hu0 = np.zeros(nx)
    amp, period = 0.08, 2.2
    signal = lambda t: np.tanh(t / 0.8) * amp * np.sin(2 * np.pi * t / period)  # noqa: E731
    cb = make_characteristic_inflow(signal, h_rest=SL, side="lower")
    sol = forward_solve(
        zb, h0, hu0, xlower=0.0, xupper=10.0, t_end=8.0, num_output_times=120,
        bc_lower="custom", bc_upper="extrap", user_bc_lower=cb, **_SOLVE_KW,
    )
    eta = sol["eta"]
    assert np.isfinite(eta).all()
    # Forcing entered: interior column (away from the inflow edge) oscillates.
    interior = eta[:, nx // 2]
    assert interior.std() > 0.2 * amp
    # Bounded: no resonant blow-up.
    assert np.abs(eta - SL).max() < 6.0 * amp
    # Dominant period of the interior signal is near the forcing period.
    t = sol["t"]
    sig = interior - interior.mean()
    freqs = np.fft.rfftfreq(sig.size, d=float(t[1] - t[0]))
    f_peak = freqs[1:][np.argmax(np.abs(np.fft.rfft(sig))[1:])]
    assert abs(1.0 / f_peak - period) < 0.6 * period


@pytest.mark.fast
def test_wellbalanced_with_custom_bc_at_rest() -> None:
    # Bumpy (non-emergent) bed, at rest, forcing off: the custom inflow must
    # not manufacture currents — lake-at-rest is preserved.
    nx = 160
    x = np.linspace(0.0, 10.0, nx)
    zb = 0.4 * np.exp(-(((x - 4.0) / 0.8) ** 2)) - 0.3 * np.exp(-(((x - 7.0) / 0.6) ** 2))
    assert zb.max() < SL  # non-emergent
    h0 = np.maximum(SL - zb, 0.0)
    hu0 = np.zeros_like(h0)
    cb = make_characteristic_inflow(lambda t: 0.0, h_rest=SL - float(zb[0]), side="lower")
    sol = forward_solve(
        zb, h0, hu0, xlower=0.0, xupper=10.0, t_end=5.0, num_output_times=25,
        bc_lower="custom", bc_upper="extrap", user_bc_lower=cb, **_SOLVE_KW,
    )
    wet = sol["h"] > 1e-3
    assert float(np.abs(sol["u"][wet]).max()) < 1e-3


@pytest.mark.fast
def test_right_side_inflow_runs() -> None:
    # Symmetric path: inflow from the right edge (wave -> -x), outflow left.
    nx = 120
    zb, _ = _flat(nx)
    h0 = np.full(nx, SL)
    hu0 = np.zeros(nx)
    signal = lambda t: np.tanh(t / 0.8) * 0.07 * np.sin(2 * np.pi * t / 2.0)  # noqa: E731
    cb = make_characteristic_inflow(signal, h_rest=SL, side="upper")
    sol = forward_solve(
        zb, h0, hu0, xlower=0.0, xupper=10.0, t_end=6.0, num_output_times=60,
        bc_lower="extrap", bc_upper="custom", user_bc_upper=cb, **_SOLVE_KW,
    )
    assert np.isfinite(sol["eta"]).all()
    # Wave entered from the right: right-half column oscillates.
    assert sol["eta"][:, 3 * nx // 4].std() > 0.01


@pytest.mark.fast
def test_custom_bc_requires_callback() -> None:
    zb, _ = _flat(40)
    h0 = np.full(40, SL)
    hu0 = np.zeros(40)
    with pytest.raises(ValueError, match="user_bc_lower"):
        forward_solve(
            zb, h0, hu0, xlower=0.0, xupper=10.0, t_end=0.5, num_output_times=2,
            bc_lower="custom", bc_upper="extrap", **_SOLVE_KW,
        )


@pytest.mark.fast
def test_hard_tier_never_emerges() -> None:
    # Deep-water bank: every sampled hard case stays fully wet at rest — even at
    # the case's own (possibly low) tidal stage.
    grid = Grid()
    rng = np.random.default_rng(3)
    for _ in range(40):
        spec = sample_case("hard", rng, grid)
        zb = bathymetry(spec.features, grid)
        assert zb.max() < spec.water_level, (
            f"emergent at low water: max zb = {zb.max():.3f} >= level {spec.water_level:.3f}"
        )


@pytest.mark.fast
def test_tidal_stage_varies_and_stays_wet() -> None:
    # The per-case water level (tidal stage) spans the configured range, is
    # orthogonal to the difficulty tier, and never lets the bed emerge.
    from pinn_bath.datasets.generator import WATER_LEVEL_RANGE

    grid = Grid()
    lo, hi = WATER_LEVEL_RANGE
    levels = []
    rng = np.random.default_rng(7)
    for tier in ("easy", "medium", "hard"):
        for _ in range(20):
            spec = sample_case(tier, rng, grid)
            zb = bathymetry(spec.features, grid)
            assert lo * grid.sea_level - 1e-9 <= spec.water_level <= hi * grid.sea_level + 1e-9
            assert zb.max() < spec.water_level  # >=15% wet column guaranteed
            levels.append(spec.water_level)
    # Actually varies (not all pinned to the reference level).
    assert np.std(levels) > 0.02


@pytest.mark.fast
def test_water_level_sets_celerity() -> None:
    # Physics check: a pulse on a flat bed at low water is slower than at high
    # water (c = sqrt(g H)). Track the crest position at a fixed time.
    nx = 300
    x = np.linspace(0.05, 9.95, nx)
    zb = np.zeros(nx)

    def crest_at(level: float, k: int = 30) -> float:
        eta0 = level + 0.05 * np.exp(-(((x - 1.5) / 0.4) ** 2))
        sol = forward_solve(
            zb, eta0 - zb, np.zeros(nx), xlower=0.0, xupper=10.0, t_end=3.0,
            num_output_times=60, bc="extrap", **_SOLVE_KW,
        )
        return float(x[np.argmax(sol["eta"][k])])

    assert crest_at(1.15) > crest_at(0.6) + 0.3  # high tide front is ahead


@pytest.mark.fast
def test_wave_signal_ramps_from_zero() -> None:
    grid = Grid()
    spec = sample_case("medium", np.random.default_rng(0), grid)
    sig = wave_signal(spec)
    assert abs(sig(0.0)) < 1e-9  # tanh ramp → 0 at t=0
    # Later in the window the signal is active.
    assert max(abs(sig(t)) for t in np.linspace(2.0, 8.0, 50)) > 1e-3
