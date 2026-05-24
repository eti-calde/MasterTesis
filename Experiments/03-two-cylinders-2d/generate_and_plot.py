"""
Generate full 2D ground truth dataset and visualize.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ground_truth import generate_dataset

OUT_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


data = generate_dataset(
    Lx=25.0, Ly=25.0, Nx=50, Ny=50,
    t_end=60.0, n_save=60,                 # Ruppenthal §7.2 production config
    eta_init=2.0, u_init=2.21, v_init=2.21,
    smooth=0.0, verbose=True,
    save_path=OUT_DIR / "ground_truth_cylinders.npz",
)

# --- Figure 1: bathymetry ---
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.pcolormesh(data["x"], data["y"], data["zb"], cmap="terrain", shading="auto")
plt.colorbar(im, ax=ax, label="z_b (m)")
# Annotate cylinders
for (xc, yc, r, H) in data["params"]["cylinders"]:
    ax.add_patch(plt.Circle((xc, yc), r, fill=False, edgecolor="red", linewidth=2, linestyle="--"))
    ax.text(xc, yc, f"H={H}m", color="white", ha="center", va="center", fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5))
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Two-cylinder bathymetry (truth)")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(FIG_DIR / "bathymetry.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'bathymetry.png'}")

# --- Figure 2: snapshots of eta ---
n_show = 4
fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 7), sharex=True, sharey=True)
ts_idx = np.linspace(0, len(data["t"]) - 1, n_show).astype(int)

for col, ti in enumerate(ts_idx):
    # eta
    ax = axes[0, col]
    im = ax.pcolormesh(data["x"], data["y"], data["eta"][ti], cmap="coolwarm", shading="auto",
                       vmin=data["eta"].min(), vmax=data["eta"].max())
    ax.set_title(f"η at t = {data['t'][ti]:.2f} s")
    ax.set_aspect("equal")
    if col == 0: ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax)

    # velocity magnitude
    ax = axes[1, col]
    vmag = np.sqrt(data["u"][ti]**2 + data["v"][ti]**2)
    im = ax.pcolormesh(data["x"], data["y"], vmag, cmap="viridis", shading="auto")
    ax.set_title(f"|v| at t = {data['t'][ti]:.2f} s")
    ax.set_xlabel("x (m)"); ax.set_aspect("equal")
    if col == 0: ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax)

fig.suptitle("Two-cylinder SWE simulation (FV ground truth)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ground_truth_snapshots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'ground_truth_snapshots.png'}")

# --- Stats ---
print()
print("=" * 60)
print(f"t: {data['t'][0]:.3f} -> {data['t'][-1]:.3f} s, {len(data['t'])} snapshots")
print(f"zb range: [{data['zb'].min():.4f}, {data['zb'].max():.4f}] m")
print(f"eta range: [{data['eta'].min():.4f}, {data['eta'].max():.4f}] m")
print(f"u range:  [{data['u'].min():.4f}, {data['u'].max():.4f}] m/s")
print(f"v range:  [{data['v'].min():.4f}, {data['v'].max():.4f}] m/s")
print(f"eta std across time: {data['eta'].std(axis=0).mean():.4f} m (higher = more dynamic)")
