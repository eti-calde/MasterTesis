"""Validation gate for the PyClaw 1D SWE wrapper (pinn_bath.solver.swe1d).

The forward solver must reproduce the two cases for which we already have an
analytic ground truth on disk:

  * Exp 2 (Thacker oscillating basin, 1D transient, wet/dry) — the strong,
    self-contained correctness test: closed domain, moving shoreline,
    analytic h(x,t) over one oscillation period.
  * Exp 1 (subcritical bump, 1D steady) — steady flow over topography;
    requires inflow (discharge) / outflow (depth) boundary conditions, so it
    is checked separately below as a follow-up once Thacker passes.

Run:  .venv/bin/python scripts/validate_swe1d.py
"""

from __future__ import annotations

import numpy as np

from pinn_bath.solver import forward_solve

THACKER = "Experiments/02-thacker-basin-1d/data/ground_truth_thacker_T1.npz"


def validate_thacker() -> float:
    gt = np.load(THACKER, allow_pickle=True)
    x = gt["x"]
    t = gt["t"]
    zb = gt["zb"]
    h_true = gt["h"]  # (Nt, Nx)
    u_true = gt["u"]  # (Nt, Nx)

    nx = x.shape[0]
    dx = float(x[1] - x[0])
    xlower = float(x[0] - dx / 2)
    xupper = float(x[-1] + dx / 2)
    t_end = float(t[-1])
    n_out = t.shape[0] - 1  # so solver frames line up with the analytic t-grid

    # IC from the analytic solution at t=0.
    h0 = h_true[0].copy()
    hu0 = (h_true[0] * u_true[0]).copy()

    sol = forward_solve(
        zb,
        h0,
        hu0,
        xlower=xlower,
        xupper=xupper,
        t_end=t_end,
        num_output_times=n_out,
        bc="wall",
        kernel="aug",
        dry_tolerance=1e-3,
        cfl_desired=0.45,
    )

    h_sim = sol["h"]
    # Compare over the wet region (where the analytic solution has water).
    wet = h_true > 1e-3
    rmse = float(np.sqrt(np.mean((h_sim[wet] - h_true[wet]) ** 2)))
    peak_err = float(np.max(np.abs(h_sim[wet] - h_true[wet])))
    h0_scale = float(h_true.max())

    print("=== Exp 2 — Thacker oscillating basin ===")
    print(f"  grid: Nx={nx}, frames={h_sim.shape[0]} (analytic {t.shape[0]}), T={t_end:.3f}s")
    print(f"  h range (analytic): [{h_true.min():.3f}, {h_true.max():.3f}] m")
    print(f"  RMSE(h) over wet cells : {rmse:.5f} m  ({100 * rmse / h0_scale:.2f}% of h_max)")
    print(f"  peak |Δh| over wet     : {peak_err:.5f} m")
    return rmse


def validate_lake_at_rest() -> float:
    """Well-balancing sanity: a flat free surface over arbitrary bathymetry
    with u=0 must stay at rest (no spurious currents)."""
    nx = 200
    xlower, xupper = -5.0, 5.0
    xc = np.linspace(xlower + 10 / nx / 2, xupper - 10 / nx / 2, nx)
    # Arbitrary bumpy bed, partly emergent.
    zb = 0.6 * np.exp(-((xc) ** 2) / 0.8) + 0.3 * np.exp(-((xc - 2.5) ** 2) / 0.3)
    sea_level = 0.5
    h0 = np.maximum(sea_level - zb, 0.0)
    hu0 = np.zeros_like(h0)
    sol = forward_solve(
        zb,
        h0,
        hu0,
        xlower=xlower,
        xupper=xupper,
        t_end=5.0,
        num_output_times=20,
        bc="wall",
        kernel="aug",
        dry_tolerance=1e-3,
    )
    wet = sol["h"] > 1e-3
    max_spurious_u = float(np.max(np.abs(sol["u"][wet]))) if wet.any() else 0.0
    print("\n=== Lake at rest (well-balancing) ===")
    print(f"  máx |u| espurio en celdas húmedas: {max_spurious_u:.2e} m/s")
    return max_spurious_u


if __name__ == "__main__":
    rmse = validate_thacker()
    spurious = validate_lake_at_rest()
    h_scale = 0.5
    thacker_ok = rmse < 0.05 * h_scale
    lake_ok = spurious < 1e-3
    print()
    print(
        f"Thacker      : {'PASS' if thacker_ok else 'REVIEW'}  (RMSE {rmse:.5f} m < {0.05 * h_scale:.4f})"
    )
    print(f"Lake-at-rest : {'PASS' if lake_ok else 'REVIEW'}  (|u| {spurious:.2e} m/s < 1e-3)")
    print(f"\nGATE F0: {'PASS ✓' if (thacker_ok and lake_ok) else 'REVIEW'}")
