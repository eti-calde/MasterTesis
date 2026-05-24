"""
Verify our Bernoulli solver against the SWASHES library.

SWASHES case: type=1, domain=1, choice=1 (subcritical flow over bump)
Parameters: L=25m, q=4.42 m^2/s, h_downstream=2.0m, no friction
Bump: parabolic, centered at x=10, height=0.2m, half-width=2m
"""

import numpy as np
import subprocess
import sys
from pathlib import Path

# Add parent to path for ground_truth import
sys.path.insert(0, str(Path(__file__).parent))
from ground_truth import generate_dataset

# --- Run SWASHES via pyswashes ---

try:
    import pyswashes

    # Find the swashes binary
    import swashes
    swashes_dir = Path(swashes.__file__).parent
    swashes_bin = swashes_dir / "data" / "bin" / "swashes"

    if not swashes_bin.exists():
        # Try alternative locations
        for candidate in swashes_dir.rglob("swashes"):
            if candidate.is_file():
                swashes_bin = candidate
                break

    print(f"SWASHES binary: {swashes_bin}")
    print(f"  exists: {swashes_bin.exists()}")

    # Run SWASHES: type=1 (dam/flow), domain=1, choice=1 (subcritical bump)
    n_cells = 499  # to get 500 points
    s = pyswashes.OneDimensional(
        stype=1, domain=1, choice=1,
        num_cell_x=n_cells,
        swashes_bin=str(swashes_bin),
    )
    df = s.dataframe()
    print(f"\nSWASHES output columns: {list(df.columns)}")
    print(f"SWASHES output shape: {df.shape}")
    print(df.head())

    x_sw = df.index.values  # x coordinates
    h_sw = df["depth"].values
    u_sw = df["u"].values
    zb_sw = df["gd_elev"].values if "gd_elev" in df.columns else None

    swashes_ok = True

except Exception as e:
    print(f"pyswashes failed: {e}")
    print("Falling back to direct SWASHES binary call...")
    swashes_ok = False

# --- Run our solver with the same parameters ---

data = generate_dataset(
    L=25.0, n_points=500,
    bump_type="parabolic",
    bump_params={"x0": 10.0, "height": 0.2, "half_width": 2.0},
    q=4.42, h_downstream=2.0, n_manning=0.0,
)

# --- Compare ---

if swashes_ok and len(x_sw) > 0:
    # Interpolate our solution to SWASHES grid for fair comparison
    h_ours = np.interp(x_sw, data["x"], data["h"])
    u_ours = np.interp(x_sw, data["x"], data["u"])

    diff_h = np.abs(h_ours - h_sw)
    diff_u = np.abs(u_ours - u_sw)

    print("\n" + "=" * 60)
    print("VERIFICATION: Our solver vs SWASHES")
    print("=" * 60)
    print(f"  Number of comparison points: {len(x_sw)}")
    print(f"  h max abs error:  {diff_h.max():.2e} m")
    print(f"  h mean abs error: {diff_h.mean():.2e} m")
    print(f"  u max abs error:  {diff_u.max():.2e} m/s")
    print(f"  u mean abs error: {diff_u.mean():.2e} m/s")

    if diff_h.max() < 1e-4:
        print("\n  PASSED: Solver matches SWASHES to < 0.1 mm")
    elif diff_h.max() < 1e-3:
        print("\n  PASSED: Solver matches SWASHES to < 1 mm")
    else:
        print(f"\n  WARNING: Max error {diff_h.max():.4f} m — investigate")
else:
    print("\nCould not compare with SWASHES. Performing self-consistency checks instead.")

    # Self-consistency: Bernoulli constant should be uniform
    g = 9.81
    E = data["u"]**2 / (2 * g) + data["h"] + data["zb"]
    E_var = np.std(E)
    print(f"\n  Bernoulli energy E = u^2/(2g) + h + zb")
    print(f"  E mean:   {E.mean():.6f} m")
    print(f"  E std:    {E_var:.2e} m")
    print(f"  E range:  [{E.min():.6f}, {E.max():.6f}] m")
    if E_var < 1e-10:
        print("  PASSED: Bernoulli constant is uniform to machine precision")
    else:
        print(f"  WARNING: Bernoulli constant varies by {E_var:.2e}")
