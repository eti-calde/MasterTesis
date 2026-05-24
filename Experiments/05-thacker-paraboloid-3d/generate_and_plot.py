"""Generate 3D Thacker ground truth dataset + rich visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ground_truth import generate_dataset, analytical_thacker3d

OUT_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

data = generate_dataset(
    L=4.0, Nx=50, Ny=50,
    x_c=2.0, y_c=2.0, h_0=0.1, a=1.0, r_0=0.8,
    n_periods=3, n_save=60,
    save_path=OUT_DIR / "ground_truth_thacker3d.npz",
)

x, y, t = data["x"], data["y"], data["t"]
X, Y = data["X"], data["Y"]
zb = data["zb"]; h = data["h"]; u = data["u"]; v = data["v"]; eta = data["eta"]
p = data["params"]

# --- Figure 1: bathymetry 2D + radial profile ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im = axes[0].pcolormesh(x, y, zb, cmap="terrain", shading="auto",
                         vmin=-0.15, vmax=0.7)
axes[0].set_title(r"Paraboloid $z_b(x,y) = -h_0(1 - r^2/a^2)$")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
axes[0].set_aspect("equal")
# Circle at r=a (rest shoreline)
axes[0].add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                              edgecolor="red", linewidth=2, linestyle="--",
                              label=f"r = a = {p['a']} m"))
axes[0].legend()
plt.colorbar(im, ax=axes[0], label=r"$z_b$ (m)")

# Radial profile
r_plot = np.linspace(0, p["L"] / 2 * np.sqrt(2), 200)
zb_r = -p["h_0"] * (1 - (r_plot / p["a"]) ** 2)
axes[1].plot(r_plot, zb_r, "k-", linewidth=2, label=r"$z_b(r)$")
axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[1].axvline(p["a"], color="red", linestyle="--", linewidth=1, label=f"r = a = {p['a']} m")
axes[1].fill_between([0, p["a"]], [-0.11] * 2, [0.0] * 2, color="cornflowerblue",
                      alpha=0.2, label="basin at rest")
axes[1].set_xlabel("r (m)"); axes[1].set_ylabel(r"$z_b$ (m)")
axes[1].set_xlim(0, 2); axes[1].set_ylim(-0.12, 0.4)
axes[1].grid(True, alpha=0.3); axes[1].legend()
axes[1].set_title("Radial bathymetry profile")

plt.tight_layout()
plt.savefig(FIG_DIR / "bathymetry.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'bathymetry.png'}")

# --- Figure 2: snapshots at key phases ---
T_p = p["period"]
key_phases = [0.0, 0.25, 0.5, 0.75]  # within first period
snap_idxs = [int(np.argmin(np.abs(t - ph * T_p))) for ph in key_phases]

fig, axes = plt.subplots(3, 4, figsize=(16, 11))
for col, (ti, ph) in enumerate(zip(snap_idxs, key_phases)):
    # Row 0: h
    ax = axes[0, col]
    wet = h[ti] > 1e-4
    im = ax.pcolormesh(x, y, h[ti], cmap="Blues", shading="auto",
                        vmin=0, vmax=h.max())
    ax.add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                             edgecolor="red", linewidth=1, linestyle="--"))
    ax.set_title(f"h at t/T = {ph:.2f} (t = {t[ti]:.3f} s)")
    ax.set_aspect("equal")
    if col == 0: ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax)

    # Row 1: eta
    ax = axes[1, col]
    im = ax.pcolormesh(x, y, eta[ti], cmap="coolwarm", shading="auto",
                        vmin=eta.min(), vmax=eta.max())
    ax.add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                             edgecolor="red", linewidth=1, linestyle="--"))
    ax.set_title(f"η at t/T = {ph:.2f}")
    ax.set_aspect("equal")
    if col == 0: ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax)

    # Row 2: velocity magnitude + quiver
    ax = axes[2, col]
    vmag = np.sqrt(u[ti] ** 2 + v[ti] ** 2)
    im = ax.pcolormesh(x, y, vmag, cmap="viridis", shading="auto",
                        vmin=0, vmax=np.sqrt(u ** 2 + v ** 2).max())
    # sparse quiver
    step = 4
    ax.quiver(X[::step, ::step], Y[::step, ::step],
               u[ti, ::step, ::step], v[ti, ::step, ::step],
               color="white", scale=2)
    ax.add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                             edgecolor="red", linewidth=1, linestyle="--"))
    ax.set_title(f"|v| at t/T = {ph:.2f}")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    if col == 0: ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax)

fig.suptitle("3D Thacker paraboloid — analytical ground truth",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ground_truth_snapshots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'ground_truth_snapshots.png'}")

# --- Figure 3: time series at 3 radial points ---
ic_center = np.argmin(np.abs(x - p["x_c"]))
jc_center = np.argmin(np.abs(y - p["y_c"]))
ic_mid = np.argmin(np.abs(x - (p["x_c"] + 0.5 * p["a"])))  # r = a/2
ic_edge = np.argmin(np.abs(x - (p["x_c"] + 0.9 * p["a"])))  # r ≈ a

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(t, eta[:, jc_center, ic_center], "r-", label="r = 0 (center)")
axes[0].plot(t, eta[:, jc_center, ic_mid], "g-", label=f"r = {0.5 * p['a']} m")
axes[0].plot(t, eta[:, jc_center, ic_edge], "b-", label=f"r = {0.9 * p['a']} m")
axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[0].set_ylabel(r"$\eta$ (m)"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
axes[0].set_title("Free surface oscillation at 3 radii")

axes[1].plot(t, h[:, jc_center, ic_center], "r-", label="r = 0 (center)")
axes[1].plot(t, h[:, jc_center, ic_mid], "g-", label=f"r = {0.5 * p['a']} m")
axes[1].plot(t, h[:, jc_center, ic_edge], "b-", label=f"r = {0.9 * p['a']} m")
axes[1].set_xlabel("t (s)"); axes[1].set_ylabel(r"$h$ (m)"); axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_title("Water depth at 3 radii (zero = dry)")

plt.tight_layout()
plt.savefig(FIG_DIR / "radial_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'radial_timeseries.png'}")

# --- Stats ---
print()
print(f"  zb range: [{zb.min():.4f}, {zb.max():.4f}] m")
print(f"  h max:    {h.max():.4f} m")
print(f"  |v| max:  {np.sqrt(u**2+v**2).max():.4f} m/s")
print(f"  eta range: [{eta.min():.4f}, {eta.max():.4f}] m")
wet_ever = (h > 1e-4).any(axis=0)
print(f"  ever-wet fraction: {wet_ever.mean():.2%}")
