"""Generate Tian dT10 ground truth dataset + visualizations.

Reproduces the snapshots of Tian (2025) Figure 5 (water depth and velocity at
t = 0, 0.25, 0.5 s) using our FV-HLL solver with periodic BCs.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ground_truth import generate_dataset

OUT_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

data = generate_dataset(
    Lx=4.0, Ly=4.0, Nx=100, Ny=100,
    t_end=0.5, n_save=51,
    verbose=False,
    save_path=OUT_DIR / "ground_truth_dT10.npz",
)

# --- Bathymetry plot ---
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.pcolormesh(
    data["x"], data["y"], data["zb"], cmap="terrain",
    vmin=0.99, vmax=1.01, shading="auto",
)
plt.colorbar(im, ax=ax, label=r"$z_b$ (m)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(r"Tian dT10 topography: $z = 1 + 0.01 \cos(\pi x/2) \cos(\pi y/2)$")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(FIG_DIR / "bathymetry.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'bathymetry.png'}")

# --- Snapshots at t = 0, 0.25, 0.5 s (matching Tian Figure 5) ---
phase_times = [0.0, 0.25, 0.5]
phase_indices = [int(np.argmin(np.abs(data["t"] - pt))) for pt in phase_times]

h_min = float(data["h"].min())
h_max = float(data["h"].max())
u_max = float(max(np.abs(data["u"]).max(), np.abs(data["v"]).max()))

fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
for col, (ti, pt) in enumerate(zip(phase_indices, phase_times)):
    # h
    ax = axes[0, col]
    im = ax.pcolormesh(
        data["x"], data["y"], data["h"][ti],
        cmap="coolwarm", vmin=h_min, vmax=h_max, shading="auto",
    )
    ax.set_title(f"$h$ @ t = {data['t'][ti]:.3f} s")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    # u
    ax = axes[1, col]
    im = ax.pcolormesh(
        data["x"], data["y"], data["u"][ti],
        cmap="RdBu_r", vmin=-u_max, vmax=u_max, shading="auto",
    )
    ax.set_title(f"$u$ @ t = {data['t'][ti]:.3f} s")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    # v
    ax = axes[2, col]
    im = ax.pcolormesh(
        data["x"], data["y"], data["v"][ti],
        cmap="RdBu_r", vmin=-u_max, vmax=u_max, shading="auto",
    )
    ax.set_title(f"$v$ @ t = {data['t'][ti]:.3f} s")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    plt.colorbar(im, ax=ax)

axes[0, 0].set_ylabel("y (m)")
axes[1, 0].set_ylabel("y (m)")
axes[2, 0].set_ylabel("y (m)")
fig.suptitle("Tian dT10 — FV-HLL ground truth (matches paper Figure 5)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ground_truth_snapshots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'ground_truth_snapshots.png'}")

# --- Time series at center vs corner vs mid-edge ---
ic = len(data["x"]) // 2
jc = len(data["y"]) // 2
eta_center = data["eta"][:, jc, ic]                       # (0, 0): zb_max
eta_corner = data["eta"][:, 0, 0]                         # (-2, -2): zb_max
eta_mid_edge = data["eta"][:, jc, 0]                      # (-2, 0): zb_min

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data["t"], eta_center, "r-", label=r"$\eta$ at center (0, 0)")
ax.plot(data["t"], eta_corner, "g--", label=r"$\eta$ at corner (-2, -2)")
ax.plot(data["t"], eta_mid_edge, "b-", label=r"$\eta$ at mid-edge (-2, 0)")
ax.set_xlabel("t (s)")
ax.set_ylabel(r"$\eta$ (m)")
ax.set_title("Free-surface relaxation: high (center / corner) drains to low (mid-edge)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "relaxation_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'relaxation_timeseries.png'}")

# --- Stats ---
print()
print(f"  domain: [{data['x'].min():.2f}, {data['x'].max():.2f}] x [{data['y'].min():.2f}, {data['y'].max():.2f}] m")
print(f"  grid: {len(data['x'])} x {len(data['y'])}")
print(f"  t: {data['t'][0]:.3f} -> {data['t'][-1]:.3f} s, {len(data['t'])} snapshots")
print(f"  zb range:  [{data['zb'].min():.4f}, {data['zb'].max():.4f}] m  (expected 0.99-1.01)")
print(f"  h(0) range: [{data['h'][0].min():.4f}, {data['h'][0].max():.4f}] m  (expected 0.99-1.01, h = z)")
print(f"  h final range: [{data['h'][-1].min():.4f}, {data['h'][-1].max():.4f}] m")
print(f"  eta range over time: [{data['eta'].min():.4f}, {data['eta'].max():.4f}] m")
