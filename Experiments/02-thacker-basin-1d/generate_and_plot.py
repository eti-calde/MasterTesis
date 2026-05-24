"""
Generate Thacker ground truth dataset and visualize.
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
    a=1.0, h_0=0.5,
    L=4.0, n_points_x=200,
    n_periods=1.0, n_points_t=100,
    save_path=OUT_DIR / "ground_truth_thacker_T1.npz",
)

x = data["x"]
t = data["t"]
h = data["h"]
u = data["u"]
eta = data["eta"]
zb = data["zb"]

# --- Figure 1: snapshots at key time instants ---
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)

key_fractions = [0.0, 0.125, 0.25, 0.5]
for col, frac in enumerate(key_fractions):
    it = int(frac * (len(t) - 1))
    t_val = t[it]

    # Row 0: bathymetry + water column
    ax = axes[0, col]
    ax.fill_between(x, -0.6, zb, color="saddlebrown", alpha=0.6, label="Bed" if col == 0 else None)
    wet = h[it] > 1e-6
    ax.fill_between(x, zb, eta[it], where=wet, color="cornflowerblue", alpha=0.5, label="Water" if col == 0 else None)
    ax.plot(x, eta[it], "b-", linewidth=1.5, label="η" if col == 0 else None)
    ax.plot(x, zb, "k-", linewidth=1.0)
    ax.set_ylim(-0.6, 1.6)
    ax.set_ylabel("Elevation (m)" if col == 0 else "")
    ax.set_title(f"t = {t_val:.4f} s  (t/T = {frac:.3f})")
    if col == 0:
        ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1: depth + velocity
    ax = axes[1, col]
    ax2 = ax.twinx()
    ax.plot(x, h[it], "b-", linewidth=2, label="h")
    ax2.plot(x, u[it], "r-", linewidth=2, label="u")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth h (m)" if col == 0 else "", color="b")
    ax2.set_ylabel("velocity u (m/s)" if col == 3 else "", color="r")
    ax.set_ylim(-0.05, 0.55)
    ax2.set_ylim(-1.7, 1.7)
    ax.grid(True, alpha=0.3)

fig.suptitle("Thacker T1 — Oscillating Parabolic Basin (analytical)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ground_truth_snapshots.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIG_DIR / 'ground_truth_snapshots.png'}")
plt.close()

# --- Figure 2: space-time plots ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

titles = ["h(x, t)", "u(x, t)", "η(x, t) = h + z_b"]
fields = [h, u, eta]
cmaps = ["Blues", "RdBu_r", "coolwarm"]

for ax, title, field, cmap in zip(axes, titles, fields, cmaps):
    vmax = np.max(np.abs(field))
    vmin = -vmax if title.startswith("u") else 0
    if title.startswith("η"):
        vmin = field.min()
    im = ax.pcolormesh(x, t, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (s)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

fig.suptitle("Thacker T1 — Space-Time Fields", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ground_truth_spacetime.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIG_DIR / 'ground_truth_spacetime.png'}")
plt.close()

# --- Figure 3: bathymetry only with wet extent envelope ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(x, -0.6, zb, color="saddlebrown", alpha=0.6, label="Bed $z_b(x)$")
ax.plot(x, zb, "k-", linewidth=2)

# Shade the ever-wet region
wet_ever = (h > 1e-6).any(axis=0)
ax.fill_between(x, -0.55, 1.55, where=wet_ever, color="cornflowerblue", alpha=0.15,
                label="Ever-wet region")

# Annotate
omega = data["params"]["omega"]
T = data["params"]["period"]
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
ax.text(-1.8, 0.8, f"$a = 1$ m, $h_0 = 0.5$ m\n$\\omega = {omega:.3f}$ rad/s\n$T = {T:.3f}$ s",
        bbox=dict(facecolor="white", alpha=0.8))

ax.set_xlabel("x (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Thacker Basin — Bathymetry and Ever-Wet Region")
ax.set_ylim(-0.6, 1.6)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "bathymetry_overview.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIG_DIR / 'bathymetry_overview.png'}")
plt.close()

# --- Summary stats ---
print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"  Bump height (z_b range): {zb.max() - zb.min():.4f} m")
print(f"  h max:                    {h.max():.4f} m")
print(f"  u max:                    {np.abs(u).max():.4f} m/s")
print(f"  Wet fraction min:         {(h > 1e-6).mean(axis=1).min():.3f}")
print(f"  Wet fraction max:         {(h > 1e-6).mean(axis=1).max():.3f}")
print(f"  Ever-wet fraction:        {wet_ever.mean():.3f}")
