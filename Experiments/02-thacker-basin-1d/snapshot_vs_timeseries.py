"""
Key experiment for Experiment 2: does temporal information improve bathymetry inversion?

Compares inversion quality with:
    N_t = 1  (single snapshot at t=T/4)
    N_t = 2  (at t=0, t=T/2)
    N_t = 4  (evenly spaced)
    N_t = 8, 16, 40 (nearly continuous)

Each config trained from scratch with same hyperparameters.
Reports z_b RMSE on the ever-wet region vs N_t.
"""

import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from ground_truth import generate_dataset
from pinn_inverse import ThackerInversePINN


RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def evenly_spaced_indices(N, k):
    """Pick k evenly-spaced indices from [0, N-1]."""
    if k == 1:
        return np.array([N // 4], dtype=int)  # t = T/4 (max velocity)
    return np.linspace(0, N - 1, k).astype(int)


def run_one_config(data, N_t, n_epochs_adam=5000, n_epochs_lbfgs=200, seed=42):
    """Train the Thacker inverse PINN with N_t observation time instants."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Nt_full = len(data["t"])
    t_obs_indices = evenly_spaced_indices(Nt_full, N_t)

    pinn = ThackerInversePINN(
        x_grid=data["x"], t_grid=data["t"],
        eta_obs=data["eta"],
        t_obs_indices=t_obs_indices,
        h_ic=data["h"][0], u_ic=data["u"][0],
        g=data["params"]["g"],
        sol_hidden=5, sol_neurons=96,
        bath_hidden=3, bath_neurons=48,
        fourier_features=16,
        fourier_sigma_x=2.0,
        fourier_sigma_t=2.0,
        lambda_data=10.0,
        lambda_pde=1.0,
        lambda_ic=100.0,
        lambda_bc=10.0,
        lambda_dry=10.0,
        lambda_pos=10.0,
        lambda_tv=1e-4,
    )
    pinn.zb_true = data["zb"]

    t0 = time.time()
    pinn.train_adam(n_epochs=n_epochs_adam, print_every=n_epochs_adam + 1)
    pinn.train_lbfgs(n_steps=n_epochs_lbfgs, print_every=n_epochs_lbfgs + 1)
    wall_time = time.time() - t0

    res = pinn.get_results()
    err = res["zb_1d"] - data["zb"]
    wet_ever = (data["h"] > 1e-4).any(axis=0)

    return {
        "N_t": int(N_t),
        "t_obs_indices": t_obs_indices.tolist(),
        "zb_rmse_all": float(np.sqrt(np.mean(err**2))),
        "zb_rmse_wet": float(np.sqrt(np.mean(err[wet_ever]**2))),
        "zb_max_err_wet": float(np.max(np.abs(err[wet_ever]))),
        "wall_time_s": float(wall_time),
        "zb_pred": res["zb_1d"].tolist(),
    }


def main():
    print("Generating Thacker ground truth...")
    data = generate_dataset(
        a=1.0, h_0=0.5,
        L=4.0, n_points_x=80,
        n_periods=1.0, n_points_t=40,
    )

    # Reduced set for initial run: extremes plus a few middle points
    N_t_values = [1, 4, 10, 40]
    results = []

    for i, N_t in enumerate(N_t_values):
        print(f"\n[{i+1}/{len(N_t_values)}] N_t = {N_t}")
        r = run_one_config(data, N_t)
        results.append(r)
        print(f"  -> z_b RMSE (wet) = {r['zb_rmse_wet']*1000:.2f} mm  |  time = {r['wall_time_s']:.0f}s")

    # Save
    with open(RESULTS_DIR / "snapshot_vs_timeseries.json", "w") as f:
        serializable = [{k: v for k, v in r.items() if k != "zb_pred"} for r in results]
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'snapshot_vs_timeseries.json'}")

    # --- Plot: RMSE vs N_t ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    Nts = [r["N_t"] for r in results]
    rmse_wet = [r["zb_rmse_wet"] * 1000 for r in results]

    ax = axes[0]
    ax.plot(Nts, rmse_wet, "o-", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Number of observation time instants $N_t$")
    ax.set_ylabel("$z_b$ RMSE (ever-wet region, mm)")
    ax.set_title("Info gain from temporal sampling")
    ax.grid(True, alpha=0.3, which="both")

    # --- Plot: recovered bathymetries ---
    ax = axes[1]
    ax.plot(data["x"], data["zb"], "k-", linewidth=3, label="True $z_b$")
    wet_ever = (data["h"] > 1e-4).any(axis=0)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(results)))
    for r, c in zip(results, colors):
        ax.plot(data["x"], r["zb_pred"], "--", color=c, linewidth=1.5,
                label=f"$N_t = {r['N_t']}$")
    # Shade ever-wet region
    ax.fill_between(data["x"], -1, 2, where=wet_ever, color="lightblue", alpha=0.15)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Bed elevation (m)")
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("Bathymetry recovery vs $N_t$")
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Thacker — Snapshot vs Time Series (Key Experiment)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "snapshot_vs_timeseries.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'snapshot_vs_timeseries.png'}")
    plt.close()

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: z_b inversion quality vs number of observation times")
    print("=" * 60)
    print(f"  {'N_t':<6} {'RMSE_wet (mm)':<15} {'max_err_wet (mm)':<18} {'time (s)':<10}")
    for r in results:
        print(f"  {r['N_t']:<6} {r['zb_rmse_wet']*1000:<15.2f} {r['zb_max_err_wet']*1000:<18.2f} {r['wall_time_s']:<10.0f}")


if __name__ == "__main__":
    main()
