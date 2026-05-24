"""
Generate the ground truth dataset for Experiment 1 and produce a visualization.

Three configurations matching the literature exactly:
1. Dazzi B1 — exact reproduction of Dazzi (2024) case
2. SWASHES standard — verification against SWASHES library
3. Low-flow with friction — custom case for harder inversion (weak surface signal)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ground_truth import generate_dataset

OUT_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# --- Config 1: Dazzi B1 (exact reproduction) ---
# Domain [-10, 10], bump at x=0, q=4.42, h_down=2.0, no friction
# Source: bump_problems.py line 28-34
data_dazzi = generate_dataset(
    L=20.0, n_points=500,
    x_start=-10.0,  # Dazzi: x_start = xdisc - L/2 = 0 - 10 = -10
    bump_type="parabolic",
    bump_params={"x0": 0.0, "height": 0.2, "half_width": 2.0},
    q=4.42, h_downstream=2.0, n_manning=0.0,
    save_path=OUT_DIR / "ground_truth_dazzi_B1.npz",
)

# --- Config 2: SWASHES standard (same physics, different domain) ---
# Domain [0, 25], bump at x=10, q=4.42, h_down=2.0, no friction
data_swashes = generate_dataset(
    L=25.0, n_points=500,
    bump_type="parabolic",
    bump_params={"x0": 10.0, "height": 0.2, "half_width": 2.0},
    q=4.42, h_downstream=2.0, n_manning=0.0,
    save_path=OUT_DIR / "ground_truth_swashes.npz",
)

# --- Config 3: Low-flow with Manning friction (custom, harder inversion) ---
# Weaker surface signal (~15mm depression vs ~93mm), tests friction effects
data_lowflow = generate_dataset(
    L=20.0, n_points=500,
    x_start=-10.0,
    bump_type="parabolic",
    bump_params={"x0": 0.0, "height": 0.2, "half_width": 2.0},
    q=0.18, h_downstream=0.5, n_manning=0.02,
    save_path=OUT_DIR / "ground_truth_lowflow_friction.npz",
)


# --- Visualization ---

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex="col")

configs = [
    (data_dazzi, "Dazzi B1 (exact)\n$q=4.42$, $n=0$, $x\\in[-10,10]$"),
    (data_swashes, "SWASHES standard\n$q=4.42$, $n=0$, $x\\in[0,25]$"),
    (data_lowflow, "Low-flow + friction\n$q=0.18$, $n=0.02$, $x\\in[-10,10]$"),
]

for col, (data, title) in enumerate(configs):
    x = data["x"]
    zb = data["zb"]
    h = data["h"]
    u = data["u"]
    eta = data["eta"]
    Fr = data["Fr"]

    # Row 0: Bathymetry + water surface
    ax = axes[0, col]
    ax.fill_between(x, 0, zb, color="saddlebrown", alpha=0.6, label="Bed $z_b$")
    ax.fill_between(x, zb, eta, color="cornflowerblue", alpha=0.4, label="Water")
    ax.plot(x, eta, "b-", linewidth=1.5, label="$\\eta = h + z_b$")
    ax.plot(x, zb, "k-", linewidth=1.0)
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Row 1: Velocity
    ax = axes[1, col]
    ax.plot(x, u, "r-", linewidth=1.5)
    ax.set_ylabel("Velocity $u$ (m/s)")
    ax.grid(True, alpha=0.3)

    # Row 2: Froude number
    ax = axes[2, col]
    ax.plot(x, Fr, "g-", linewidth=1.5)
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="$Fr = 1$")
    ax.set_ylabel("Froude number")
    ax.set_xlabel("x (m)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle("Experiment 1 — Ground Truth: Subcritical Flow Over Bump", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ground_truth_overview.png", dpi=150, bbox_inches="tight")
print(f"Figure saved to {FIG_DIR / 'ground_truth_overview.png'}")

# --- Summary stats ---
print("\n" + "=" * 60)
for data, name in [(data_dazzi, "Dazzi B1"), (data_swashes, "SWASHES"), (data_lowflow, "Low-flow")]:
    print(f"\n{name}:")
    print(f"  domain: [{data['x'].min():.1f}, {data['x'].max():.1f}] m")
    print(f"  h:   [{data['h'].min():.4f}, {data['h'].max():.4f}] m")
    print(f"  u:   [{data['u'].min():.4f}, {data['u'].max():.4f}] m/s")
    print(f"  eta: [{data['eta'].min():.4f}, {data['eta'].max():.4f}] m")
    print(f"  Fr:  [{data['Fr'].min():.4f}, {data['Fr'].max():.4f}]")
    print(f"  Surface depression: {data['eta'].max() - data['eta'].min():.4f} m")

    # Bernoulli consistency check
    g = 9.81
    E = data["u"]**2 / (2*g) + data["h"] + data["zb"]
    if data.get("params", {}).get("n_manning", 0) == 0:
        print(f"  Bernoulli energy std: {np.std(E):.2e} m (should be ~1e-15)")

print("\n" + "=" * 60)
print("All datasets saved to:", OUT_DIR)
print("Figures saved to:", FIG_DIR)
