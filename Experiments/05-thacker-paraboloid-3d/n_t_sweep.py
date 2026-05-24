"""
Key experiment: how many time snapshots (N_t) are needed to recover the
2D Thacker paraboloid?

Tests the Exp-2/Exp-4 finding in the axisymmetric (closed-basin) case:
does the "2-to-4 snapshots suffice" result still hold for a true bowl?

NOTE: the Thacker basin is **closed** -- no tidal forcing. The name
``n_t_sweep`` (and the outputs ``n_t_sweep.json`` / ``n_t_panel.png`` /
``n_t_sweep_curves.png``) reflects the actual swept variable: number of
temporal observation snapshots. Renamed 2026-05-24 from the previous
misleading ``tidal_phases_sweep.py``.
"""

import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from ground_truth import generate_dataset
from pinn_inverse import Thacker3DInversePINN


RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)


def evenly_spaced_indices(N, k):
    if k == 1:
        return np.array([N // 4], dtype=int)
    return np.linspace(0, N - 1, k).astype(int)


def run_one(data, N_t, n_epochs_adam=4000, n_epochs_lbfgs=80, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    t_obs_indices = evenly_spaced_indices(len(data["t"]), N_t)

    pinn = Thacker3DInversePINN(
        x_grid=data["x"], y_grid=data["y"], t_grid=data["t"],
        eta_obs=data["eta"],
        u_obs=None, v_obs=None,
        h_ic=data["h"][0], u_ic=data["u"][0], v_ic=data["v"][0],
        t_obs_indices=t_obs_indices,
        n_colloc=10000, n_ic=1500,
        sol_hidden=5, sol_neurons=128,
        bath_hidden=4, bath_neurons=64,
        fourier_features=24, sigma_space=3.0, sigma_time=2.0,
        fourier_features_bath=32, sigma_bath=4.0,
        lambda_data=10.0, lambda_pde=1.0, lambda_ic=100.0,
        lambda_dry=10.0, lambda_tv=1e-5,
    )
    pinn.zb_true = data["zb"]
    pinn.wet_mask = (data["h"] > 1e-4).any(axis=0)

    t0 = time.time()
    pinn.train_adam(n_epochs=n_epochs_adam, print_every=n_epochs_adam + 1)
    pinn.train_lbfgs(n_steps=n_epochs_lbfgs, print_every=n_epochs_lbfgs + 1)
    wall = time.time() - t0

    zb_pred = pinn.get_zb_2d()
    err = zb_pred - data["zb"]
    wet = pinn.wet_mask
    rmse_all = float(np.sqrt(np.mean(err ** 2)))
    rmse_wet = float(np.sqrt(np.mean(err[wet] ** 2)))
    return {
        "N_t": int(N_t),
        "t_obs_indices": t_obs_indices.tolist(),
        "zb_rmse_all": rmse_all,
        "zb_rmse_wet": rmse_wet,
        "zb_max_err_wet": float(np.max(np.abs(err[wet]))),
        "wall_time_s": float(wall),
        "zb_pred": zb_pred.tolist(),
    }


def main():
    print("Generating 3D Thacker ground truth...", flush=True)
    data = generate_dataset(
        L=4.0, Nx=40, Ny=40,
        x_c=2.0, y_c=2.0, h_0=0.1, a=1.0, r_0=0.8,
        n_periods=3, n_save=30,
    )

    N_t_values = [1, 2, 4, 8, 30]
    results = []
    for i, N_t in enumerate(N_t_values):
        print(f"\n[{i+1}/{len(N_t_values)}] N_t = {N_t}", flush=True)
        r = run_one(data, N_t)
        results.append(r)
        print(f"  -> RMSE_wet = {r['zb_rmse_wet']*1000:.2f} mm  |  time = {r['wall_time_s']:.0f}s", flush=True)

    with open(RESULTS_DIR / "n_t_sweep.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "zb_pred"} for r in results], f, indent=2)

    # Recovery panel
    p = data["params"]
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(3.2 * (len(results) + 1), 4))
    im = axes[0].pcolormesh(data["x"], data["y"], data["zb"], cmap="terrain",
                             vmin=-0.12, vmax=0.7, shading="auto")
    axes[0].add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                                  edgecolor="red", linewidth=1, linestyle="--"))
    axes[0].set_title("True z_b"); axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    plt.colorbar(im, ax=axes[0])
    for ax, r in zip(axes[1:], results):
        im = ax.pcolormesh(data["x"], data["y"], np.asarray(r["zb_pred"]),
                           cmap="terrain", vmin=-0.12, vmax=0.7, shading="auto")
        ax.add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                                 edgecolor="red", linewidth=1, linestyle="--"))
        ax.set_title(f"N_t={r['N_t']}\nRMSE_wet = {r['zb_rmse_wet']*1000:.2f} mm")
        ax.set_aspect("equal"); ax.set_xlabel("x (m)")
        plt.colorbar(im, ax=ax)
    fig.suptitle("3D Thacker — recovered z_b across N_t (eta only)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "n_t_panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'n_t_panel.png'}", flush=True)

    # RMSE curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot([r["N_t"] for r in results], [r["zb_rmse_wet"] * 1000 for r in results],
                  "o-", linewidth=2, markersize=8, label="ever-wet region")
    axes[0].plot([r["N_t"] for r in results], [r["zb_rmse_all"] * 1000 for r in results],
                  "s--", linewidth=1.5, markersize=6, color="gray", alpha=0.7, label="full domain")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of observation time instants $N_t$")
    axes[0].set_ylabel("$z_b$ RMSE (mm)")
    axes[0].set_title("3D Thacker inversion vs temporal sampling (eta only)")
    axes[0].grid(True, alpha=0.3, which="both"); axes[0].legend()

    best = min(results, key=lambda r: r["zb_rmse_wet"])
    err = np.asarray(best["zb_pred"]) - data["zb"]
    im = axes[1].pcolormesh(data["x"], data["y"], err, cmap="RdBu_r",
                             vmin=-0.1, vmax=0.1, shading="auto")
    axes[1].add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                                   edgecolor="black", linewidth=1, linestyle="--"))
    axes[1].set_title(f"Best (N_t={best['N_t']}): error map\nRMSE_wet = {best['zb_rmse_wet']*1000:.2f} mm")
    axes[1].set_aspect("equal"); axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)")
    plt.colorbar(im, ax=axes[1])

    fig.suptitle("Experiment 5 — 2D Thacker $N_t$ sweep (temporal snapshots, closed basin)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "n_t_sweep_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'n_t_sweep_curves.png'}", flush=True)

    print("\nSUMMARY")
    print(f"  {'N_t':<6} {'RMSE_wet (mm)':<18} {'RMSE_all (mm)':<18} {'time (s)':<10}")
    for r in results:
        print(f"  {r['N_t']:<6} {r['zb_rmse_wet']*1000:<18.2f} "
              f"{r['zb_rmse_all']*1000:<18.2f} {r['wall_time_s']:<10.0f}")


if __name__ == "__main__":
    main()
