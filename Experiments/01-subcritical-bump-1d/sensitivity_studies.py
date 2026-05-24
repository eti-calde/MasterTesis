"""
Sensitivity studies for Experiment 1 — Subcritical bump bathymetry inversion.

Sweeps:
1. Observation density: 100%, 50%, 20%, 10%, 5% of domain points
2. Noise robustness: 0%, 1%, 2%, 5% Gaussian noise on eta
3. Observation type: eta only | u only | eta + u combined

Each run trains a PINN and records final metrics. Results saved to CSV
and summary figures generated at the end.
"""

import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from ground_truth import generate_dataset
from pinn_inverse import InverseBathymetryPINN


RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ============================================================
# Single experiment runner
# ============================================================

def run_one_experiment(
    x_domain, zb_true, eta_true, u_true,
    obs_density=1.0,
    noise_eta=0.0,
    noise_u=0.0,
    use_eta=True,
    use_velocity=False,
    q_known=4.42,
    h_downstream=2.0,
    n_manning=0.0,
    n_epochs_adam=12000,
    n_epochs_lbfgs=600,
    fourier_sigma=2.0,
    n_seeds=2,
    verbose=False,
):
    """Run a single inverse PINN experiment with specified conditions.

    Parameters
    ----------
    x_domain, zb_true, eta_true, u_true : np.ndarray
        Ground truth (from generate_dataset).
    obs_density : float
        Fraction of points with observations (0 < x <= 1).
    noise_eta, noise_u : float
        Relative Gaussian noise std (e.g., 0.01 = 1% of signal range).
    use_eta, use_velocity : bool
        Which observation types to provide to the PINN.
    q_known, h_downstream, n_manning : physics params.
    n_epochs_adam, n_epochs_lbfgs : training budget.
    seed : random seed for reproducibility.

    Returns
    -------
    metrics : dict with zb_rmse, zb_max_err, zb_r2, eta_rmse, wall_time, history
    """
    n = len(x_domain)
    all_runs = []

    for seed in range(42, 42 + n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Sample observation locations
        if obs_density >= 1.0:
            obs_indices = None  # all points observed
        else:
            n_obs = max(2, int(obs_density * n))
            interior_indices = np.random.choice(
                np.arange(1, n - 1), size=min(n_obs - 2, n - 2), replace=False
            )
            obs_indices = np.sort(np.concatenate([[0], interior_indices, [n - 1]]))

        # Add noise to observations
        eta_range = eta_true.max() - eta_true.min()
        u_range = u_true.max() - u_true.min()
        eta_obs = eta_true + noise_eta * eta_range * np.random.randn(n)
        u_obs = u_true + noise_u * u_range * np.random.randn(n)

        # Build PINN
        pinn = InverseBathymetryPINN(
            x_domain=x_domain,
            eta_obs=eta_obs if use_eta else np.zeros_like(eta_obs),
            u_obs=u_obs if use_velocity else None,
            obs_indices=obs_indices,
            q_known=q_known,
            h_downstream=h_downstream,
            n_manning=n_manning,
            sol_hidden=4, sol_neurons=64,
            bath_hidden=3, bath_neurons=32,
            use_fourier=True, fourier_features=16, fourier_sigma=fourier_sigma,
            lambda_data_eta=10.0 if use_eta else 0.0,
            lambda_data_u=10.0 if use_velocity else 0.0,
            lambda_pde=1.0,
            lambda_q=10.0,
            lambda_bc=100.0,
            lambda_tv=1e-4,
            lambda_tikh=1e-5,
            lambda_pos=10.0,
        )
        pinn.zb_true = zb_true

        # Train
        t0 = time.time()
        pinn.train_adam(n_epochs=n_epochs_adam, print_every=n_epochs_adam + 1)
        pinn.train_lbfgs(n_steps=n_epochs_lbfgs, print_every=n_epochs_lbfgs + 1)
        wall_time = time.time() - t0

        results = pinn.get_results()
        zb_pred = results["zb"]
        eta_pred = results["eta"]
        zb_err = zb_pred - zb_true
        ss_res = np.sum(zb_err ** 2)
        ss_tot = np.sum((zb_true - zb_true.mean()) ** 2)
        zb_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        all_runs.append({
            "seed": seed,
            "zb_rmse": float(np.sqrt(np.mean(zb_err ** 2))),
            "zb_max_err": float(np.max(np.abs(zb_err))),
            "zb_r2": float(zb_r2),
            "eta_rmse": float(np.sqrt(np.mean((eta_pred - eta_true) ** 2))),
            "wall_time_s": float(wall_time),
            "zb_pred": zb_pred.tolist(),
            "eta_pred": eta_pred.tolist(),
        })

    # Return best-of-seeds (lowest RMSE) + stats across seeds
    best = min(all_runs, key=lambda r: r["zb_rmse"])
    rmses = [r["zb_rmse"] for r in all_runs]
    r2s = [r["zb_r2"] for r in all_runs]

    return {
        "zb_rmse": best["zb_rmse"],
        "zb_max_err": best["zb_max_err"],
        "zb_r2": best["zb_r2"],
        "eta_rmse": best["eta_rmse"],
        "wall_time_s": sum(r["wall_time_s"] for r in all_runs),
        "zb_pred": best["zb_pred"],
        "eta_pred": best["eta_pred"],
        # Stats across seeds
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "rmse_min": float(np.min(rmses)),
        "rmse_max": float(np.max(rmses)),
        "r2_mean": float(np.mean(r2s)),
        "n_seeds": n_seeds,
    }


# ============================================================
# Sweep functions
# ============================================================

def sweep_observation_density(ground_truth, densities, **common_kwargs):
    """Sweep observation density."""
    results = []
    print("\n" + "=" * 70)
    print("SWEEP: Observation Density")
    print("=" * 70)
    for i, density in enumerate(densities):
        print(f"\n[{i+1}/{len(densities)}] density = {density * 100:.0f}%")
        r = run_one_experiment(
            x_domain=ground_truth["x"], zb_true=ground_truth["zb"],
            eta_true=ground_truth["eta"], u_true=ground_truth["u"],
            obs_density=density, **common_kwargs,
        )
        r["density"] = density
        results.append(r)
        print(f"  -> z_b RMSE best = {r['zb_rmse']*1000:.2f} mm  |  mean = {r['rmse_mean']*1000:.2f} ± {r['rmse_std']*1000:.2f} mm  |  R² = {r['zb_r2']:.4f}  |  time = {r['wall_time_s']:.0f}s")
    return results


def sweep_noise(ground_truth, noise_levels, **common_kwargs):
    """Sweep noise levels on eta observations."""
    results = []
    print("\n" + "=" * 70)
    print("SWEEP: Noise Robustness (on eta)")
    print("=" * 70)
    for i, noise in enumerate(noise_levels):
        print(f"\n[{i+1}/{len(noise_levels)}] noise = {noise * 100:.1f}%")
        r = run_one_experiment(
            x_domain=ground_truth["x"], zb_true=ground_truth["zb"],
            eta_true=ground_truth["eta"], u_true=ground_truth["u"],
            noise_eta=noise, **common_kwargs,
        )
        r["noise_eta"] = noise
        results.append(r)
        print(f"  -> z_b RMSE best = {r['zb_rmse']*1000:.2f} mm  |  mean = {r['rmse_mean']*1000:.2f} ± {r['rmse_std']*1000:.2f} mm  |  R² = {r['zb_r2']:.4f}  |  time = {r['wall_time_s']:.0f}s")
    return results


def sweep_observation_type(ground_truth, obs_types, **common_kwargs):
    """Sweep observation types: eta only | u only | eta + u."""
    results = []
    print("\n" + "=" * 70)
    print("SWEEP: Observation Type")
    print("=" * 70)
    for i, (name, use_eta, use_u) in enumerate(obs_types):
        print(f"\n[{i+1}/{len(obs_types)}] {name}")
        r = run_one_experiment(
            x_domain=ground_truth["x"], zb_true=ground_truth["zb"],
            eta_true=ground_truth["eta"], u_true=ground_truth["u"],
            use_eta=use_eta, use_velocity=use_u, **common_kwargs,
        )
        r["obs_type"] = name
        r["use_eta"] = use_eta
        r["use_velocity"] = use_u
        results.append(r)
        print(f"  -> z_b RMSE best = {r['zb_rmse']*1000:.2f} mm  |  mean = {r['rmse_mean']*1000:.2f} ± {r['rmse_std']*1000:.2f} mm  |  R² = {r['zb_r2']:.4f}  |  time = {r['wall_time_s']:.0f}s")
    return results


# ============================================================
# Plotting
# ============================================================

def plot_sweep_curves(results, param_key, x_label, title, save_path, log_x=False):
    """Plot RMSE and R² vs a swept parameter, with seed spread."""
    xs = [r[param_key] for r in results]
    rmse_best = [r["zb_rmse"] * 1000 for r in results]
    rmse_mean = [r.get("rmse_mean", r["zb_rmse"]) * 1000 for r in results]
    rmse_min = [r.get("rmse_min", r["zb_rmse"]) * 1000 for r in results]
    rmse_max = [r.get("rmse_max", r["zb_rmse"]) * 1000 for r in results]
    r2 = [r["zb_r2"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    # Shaded range across seeds
    ax.fill_between(xs, rmse_min, rmse_max, alpha=0.25, color="tab:blue", label="seed range")
    ax.plot(xs, rmse_mean, "o--", color="gray", linewidth=1.5, markersize=6, label="mean")
    ax.plot(xs, rmse_best, "o-", color="tab:blue", linewidth=2, markersize=8, label="best")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("$z_b$ RMSE (mm)")
    ax.set_title("Bathymetry recovery error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, r2, "o-", color="green", linewidth=2, markersize=8)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("$R^2$")
    ax.set_title("Bathymetry fit quality (best of seeds)")
    ax.axhline(y=0.95, color="k", linestyle="--", alpha=0.5, label="$R^2 = 0.95$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_bathymetry_comparison(results, ground_truth, param_key, label_fmt, title, save_path):
    """Plot recovered bathymetries for all sweep values on one figure."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ground_truth["x"], ground_truth["zb"], "k-", linewidth=3, label="True $z_b$")

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for r, c in zip(results, colors):
        label = label_fmt.format(r[param_key])
        ax.plot(ground_truth["x"], r["zb_pred"], "--", color=c, linewidth=1.5, label=label)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("Bed elevation (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_obs_type_bars(results, save_path):
    """Bar chart for observation type results."""
    names = [r["obs_type"] for r in results]
    rmse = [r["zb_rmse"] * 1000 for r in results]
    r2 = [r["zb_r2"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    ax = axes[0]
    ax.bar(names, rmse, color=colors)
    ax.set_ylabel("$z_b$ RMSE (mm)")
    ax.set_title("Bathymetry recovery error")
    for i, v in enumerate(rmse):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(names, r2, color=colors)
    ax.set_ylabel("$R^2$")
    ax.set_title("Bathymetry fit quality")
    ax.axhline(y=0.95, color="k", linestyle="--", alpha=0.5)
    for i, v in enumerate(r2):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Observation Type — Information Content", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Ground truth (Dazzi B1)
    print("Generating ground truth (Dazzi B1)...")
    gt = generate_dataset(
        L=20.0, n_points=500, x_start=-10.0,
        bump_type="parabolic",
        bump_params={"x0": 0.0, "height": 0.2, "half_width": 2.0},
        q=4.42, h_downstream=2.0, n_manning=0.0,
    )
    print(f"  bump height: {gt['zb'].max() * 1000:.0f} mm")
    print(f"  eta depression: {(gt['eta'].max() - gt['eta'].min()) * 1000:.0f} mm")

    all_results = {}

    # ---------- Sweep 1: observation density ----------
    densities = [1.0, 0.5, 0.2, 0.1, 0.05]
    t0 = time.time()
    density_results = sweep_observation_density(gt, densities)
    all_results["density"] = density_results
    print(f"\n  Density sweep total time: {(time.time() - t0):.1f}s")

    plot_sweep_curves(
        density_results, "density",
        "Observation density (fraction of domain)",
        "Sensitivity: Observation Density",
        FIG_DIR / "sensitivity_density.png",
    )
    plot_bathymetry_comparison(
        density_results, gt, "density",
        "density = {:.0%}",
        "Bathymetry recovery vs observation density",
        FIG_DIR / "sensitivity_density_profiles.png",
    )

    # ---------- Sweep 2: noise ----------
    noise_levels = [0.0, 0.01, 0.02, 0.05]
    t0 = time.time()
    noise_results = sweep_noise(gt, noise_levels)
    all_results["noise"] = noise_results
    print(f"\n  Noise sweep total time: {(time.time() - t0):.1f}s")

    plot_sweep_curves(
        noise_results, "noise_eta",
        "Noise level (fraction of signal range)",
        "Sensitivity: Noise Robustness",
        FIG_DIR / "sensitivity_noise.png",
    )
    plot_bathymetry_comparison(
        noise_results, gt, "noise_eta",
        "noise = {:.0%}",
        "Bathymetry recovery vs noise level",
        FIG_DIR / "sensitivity_noise_profiles.png",
    )

    # ---------- Sweep 3: observation type ----------
    obs_types = [
        ("eta only", True, False),
        ("u only", False, True),
        ("eta + u", True, True),
    ]
    t0 = time.time()
    obstype_results = sweep_observation_type(gt, obs_types)
    all_results["obs_type"] = obstype_results
    print(f"\n  Obs-type sweep total time: {(time.time() - t0):.1f}s")

    plot_obs_type_bars(obstype_results, FIG_DIR / "sensitivity_obstype.png")
    plot_bathymetry_comparison(
        obstype_results, gt, "obs_type",
        "{}",
        "Bathymetry recovery vs observation type",
        FIG_DIR / "sensitivity_obstype_profiles.png",
    )

    # ---------- Save all results ----------
    with open(RESULTS_DIR / "sensitivity_results.json", "w") as f:
        # Strip large arrays for JSON
        summary = {}
        for sweep_name, sweep_results in all_results.items():
            summary[sweep_name] = [
                {k: v for k, v in r.items() if k not in ("zb_pred", "eta_pred")}
                for r in sweep_results
            ]
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'sensitivity_results.json'}")

    # ---------- Summary table ----------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\nObservation Density:")
    print(f"  {'density':<12} {'z_b RMSE (mm)':<15} {'R²':<8} {'time (s)':<10}")
    for r in density_results:
        print(f"  {r['density'] * 100:>8.0f}%    {r['zb_rmse'] * 1000:>10.2f}    {r['zb_r2']:>6.4f}    {r['wall_time_s']:>6.1f}")

    print("\nNoise (on eta):")
    print(f"  {'noise':<12} {'z_b RMSE (mm)':<15} {'R²':<8} {'time (s)':<10}")
    for r in noise_results:
        print(f"  {r['noise_eta'] * 100:>8.1f}%    {r['zb_rmse'] * 1000:>10.2f}    {r['zb_r2']:>6.4f}    {r['wall_time_s']:>6.1f}")

    print("\nObservation Type:")
    print(f"  {'type':<12} {'z_b RMSE (mm)':<15} {'R²':<8} {'time (s)':<10}")
    for r in obstype_results:
        print(f"  {r['obs_type']:<12} {r['zb_rmse'] * 1000:>10.2f}    {r['zb_r2']:>6.4f}    {r['wall_time_s']:>6.1f}")
