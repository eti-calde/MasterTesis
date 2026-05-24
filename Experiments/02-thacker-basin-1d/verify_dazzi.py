"""
Verify our Thacker ground truth against Dazzi's reference implementation.

Compares:
- Bathymetry z_b(x)
- Water depth h(x, t) at multiple time instants
- Velocity u(x, t)
- Period T, angular frequency ω

Dazzi's code is at reference_dazzi2024/swe_utils/thacker_problems.py (symlinked from Exp 1).
"""

import sys
import numpy as np
from pathlib import Path

# Import our implementation
sys.path.insert(0, str(Path(__file__).parent))
from ground_truth import (
    solution_thacker, bathymetry_thacker, omega_thacker, period_thacker
)

# Import Dazzi's implementation (from Experiment 1's cloned repo)
dazzi_utils = Path(__file__).parent / ".." / "01-subcritical-bump-1d" / "reference_dazzi2024" / "swe_utils"
sys.path.insert(0, str(dazzi_utils))
from thacker_problems import (
    Thacker,
    analytical_solution_thacker_planar,
    temporal_profile_funct_thacker_planar,
)


def compare():
    # Set up Dazzi T1 problem
    tp = Thacker(code="T1")
    print("Dazzi T1 configuration:")
    print(f"  a = {tp.values[0]}, h_0 = {tp.values[1]}, ω = {tp.values[2]:.6f}")
    print(f"  domain: [{tp.x_start}, {tp.x_end}], L = {tp.L}, xdisc = {tp.xdisc}")
    print()

    a, h_0 = tp.values[0], tp.values[1]
    omega_dazzi = tp.values[2]

    # Our ω
    omega_ours = omega_thacker(a=a, h_0=h_0)
    print(f"Angular frequency:")
    print(f"  Dazzi: {omega_dazzi:.10f}")
    print(f"  Ours:  {omega_ours:.10f}")
    print(f"  diff:  {abs(omega_dazzi - omega_ours):.2e}")
    print()

    # Compare at multiple time instants
    T = period_thacker(a=a, h_0=h_0)
    times_to_test = [0.0, T / 8, T / 4, 3 * T / 8, T / 2]
    n_pts = 200

    print("Comparing solutions at various times:")
    print(f"  {'t / T':<10} {'max |h_diff|':<15} {'max |u_diff|':<15} {'max |z_b_diff|':<15}")

    worst_h = worst_u = worst_zb = 0.0

    for t_val in times_to_test:
        # Dazzi's analytical
        xa_dazzi, ha_dazzi, ua_dazzi, za_dazzi = analytical_solution_thacker_planar(
            tp, t_val, n_pts=n_pts
        )

        # Ours at same x points
        h_ours, u_ours, eta_ours, zb_ours = solution_thacker(
            xa_dazzi, t_val, a=a, h_0=h_0
        )

        diff_h = np.max(np.abs(h_ours - ha_dazzi))
        diff_u = np.max(np.abs(u_ours - ua_dazzi))
        diff_zb = np.max(np.abs(zb_ours - za_dazzi))

        worst_h = max(worst_h, diff_h)
        worst_u = max(worst_u, diff_u)
        worst_zb = max(worst_zb, diff_zb)

        print(f"  {t_val / T:<10.4f} {diff_h:<15.2e} {diff_u:<15.2e} {diff_zb:<15.2e}")

    print()
    print(f"Overall worst: h = {worst_h:.2e}, u = {worst_u:.2e}, z_b = {worst_zb:.2e}")

    if worst_h < 1e-10 and worst_u < 1e-10 and worst_zb < 1e-10:
        print("\nPASSED: our solver matches Dazzi to machine precision")
    elif worst_h < 1e-6:
        print("\nPASSED: our solver matches Dazzi to < 1 micron")
    else:
        print(f"\nWARNING: max discrepancy > 1 micron — investigate")


if __name__ == "__main__":
    compare()
