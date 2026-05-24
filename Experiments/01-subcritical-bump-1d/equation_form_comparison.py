"""
Equation form ablation: compare primitive, primitive-conservative, and conservative
SWE residual forms for inverse bathymetry recovery.

Validates the thesis choice of primitive-conservative form (Tian 2025) on the
inverse problem — an original contribution since no prior work compares forms
for PINN-based inversion.

Same ground truth, architecture, and training budget for all forms.
Only the physics loss residual computation changes.
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

FORMS = [
    ("primitive", "Primitive"),
    ("primitive_conservative", "Primitive-Conservative"),
    ("conservative", "Conservative"),
]

SEEDS = [42, 123]
N_EPOCHS_ADAM = 12000
N_EPOCHS_LBFGS = 600


def run_single(x_domain, zb_true, eta_true, u_true, swe_form, seed,
               q_known=4.42, h_downstream=2.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    pinn = InverseBathymetryPINN(
        x_domain=x_domain,
        eta_obs=eta_true,
        u_obs=None,
        obs_indices=None,
        q_known=q_known,
        h_downstream=h_downstream,
        n_manning=0.0,
        sol_hidden=4, sol_neurons=64,
        bath_hidden=3, bath_neurons=32,
        use_fourier=True, fourier_features=16, fourier_sigma=2.0,
        lambda_data_eta=10.0,
        lambda_data_u=0.0,
        lambda_pde=1.0,
        lambda_q=10.0,
        lambda_bc=100.0,
        lambda_tv=1e-4,
        lambda_tikh=1e-5,
        lambda_pos=10.0,
        swe_form=swe_form,
    )
    pinn.zb_true = zb_true

    t0 = time.time()
    pinn.train_adam(n_epochs=N_EPOCHS_ADAM, print_every=N_EPOCHS_ADAM + 1)
    pinn.train_lbfgs(n_steps=N_EPOCHS_LBFGS, print_every=N_EPOCHS_LBFGS + 1)
    wall_time = time.time() - t0

    results = pinn.get_results()
    zb_err = results["zb"] - zb_true
    ss_res = np.sum(zb_err**2)
    ss_tot = np.sum((zb_true - zb_true.mean())**2)

    return {
        "seed": seed,
        "zb_rmse": float(np.sqrt(np.mean(zb_err**2))),
        "zb_max_err": float(np.max(np.abs(zb_err))),
        "zb_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
        "eta_rmse": float(np.sqrt(np.mean((results["eta"] - (zb_true + (eta_true - zb_true)))**2))),
        "wall_time_s": float(wall_time),
        "zb_pred": results["zb"].tolist(),
    }


def run_form_sweep(gt):
    all_results = []

    for form_key, form_label in FORMS:
        print(f"\n{'='*70}")
        print(f"SWE FORM: {form_label} ({form_key})")
        print(f"{'='*70}")

        runs = []
        for seed in SEEDS:
            print(f"  seed={seed} ...", end=" ", flush=True)
            r = run_single(
                gt["x"], gt["zb"], gt["eta"], gt["u"],
                swe_form=form_key, seed=seed,
            )
            runs.append(r)
            print(f"RMSE={r['zb_rmse']*1000:.2f} mm, R²={r['zb_r2']:.4f}, time={r['wall_time_s']:.0f}s")

        best = min(runs, key=lambda r: r["zb_rmse"])
        rmses = [r["zb_rmse"] for r in runs]

        entry = {
            "form": form_key,
            "label": form_label,
            "zb_rmse": best["zb_rmse"],
            "zb_max_err": best["zb_max_err"],
            "zb_r2": best["zb_r2"],
            "eta_rmse": best["eta_rmse"],
            "wall_time_s": sum(r["wall_time_s"] for r in runs),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "rmse_min": float(np.min(rmses)),
            "rmse_max": float(np.max(rmses)),
            "n_seeds": len(SEEDS),
            "zb_pred_best": best["zb_pred"],
            "runs": [{k: v for k, v in r.items() if k != "zb_pred"} for r in runs],
        }
        all_results.append(entry)

        print(f"  -> best RMSE = {best['zb_rmse']*1000:.2f} mm | "
              f"mean = {np.mean(rmses)*1000:.2f} ± {np.std(rmses)*1000:.2f} mm | "
              f"R² = {best['zb_r2']:.4f}")

    return all_results


def plot_bar_chart(results, save_path):
    labels = [r["label"] for r in results]
    rmse_best = [r["zb_rmse"] * 1000 for r in results]
    rmse_mean = [r["rmse_mean"] * 1000 for r in results]
    rmse_min = [r["rmse_min"] * 1000 for r in results]
    rmse_max = [r["rmse_max"] * 1000 for r in results]
    r2 = [r["zb_r2"] for r in results]

    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, rmse_best, color=colors, width=0.5, alpha=0.85)
    ax.errorbar(x_pos, rmse_mean,
                yerr=[[m - lo for m, lo in zip(rmse_mean, rmse_min)],
                      [hi - m for m, hi in zip(rmse_mean, rmse_max)]],
                fmt="none", ecolor="black", capsize=5, capthick=1.5)
    ax.set_ylabel("$z_b$ RMSE (mm)")
    ax.set_title("Bathymetry Recovery Error")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    for bar, val in zip(bars, rmse_best):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(x_pos, r2, color=colors, width=0.5, alpha=0.85)
    ax.set_ylabel("$R^2$")
    ax.set_title("Bathymetry Fit Quality")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(y=0.95, color="k", linestyle="--", alpha=0.5, label="$R^2 = 0.95$")
    for bar, val in zip(bars, r2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim([min(r2) * 0.95, 1.005])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("SWE Equation Form Ablation — Inverse Bathymetry (Dazzi B1)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_profiles(results, gt, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(gt["x"], 0, gt["zb"], alpha=0.15, color="saddlebrown")
    ax.plot(gt["x"], gt["zb"], "k-", linewidth=2.5, label="True $z_b$")

    styles = [("--", "#3498db"), ("-.", "#2ecc71"), (":", "#e74c3c")]
    for r, (ls, color) in zip(results, styles):
        ax.plot(gt["x"], r["zb_pred_best"], ls, color=color, linewidth=2,
                label=f"{r['label']} (RMSE={r['zb_rmse']*1000:.1f} mm)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("$z_b$ (m)")
    ax.set_title("Recovered Bathymetry by SWE Form (best of 2 seeds)")
    ax.set_xlim(-5, 5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Generating ground truth (Dazzi B1)...")
    gt = generate_dataset(
        L=20.0, n_points=500, x_start=-10.0,
        bump_type="parabolic",
        bump_params={"x0": 0.0, "height": 0.2, "half_width": 2.0},
        q=4.42, h_downstream=2.0, n_manning=0.0,
    )
    print(f"  bump height: {gt['zb'].max()*1000:.0f} mm")

    results = run_form_sweep(gt)

    # Save results (strip large arrays for JSON)
    summary = []
    for r in results:
        summary.append({k: v for k, v in r.items() if k != "zb_pred_best"})
    with open(RESULTS_DIR / "equation_form_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'equation_form_results.json'}")

    # Plots
    plot_bar_chart(results, FIG_DIR / "equation_form_comparison.png")
    plot_profiles(results, gt, FIG_DIR / "equation_form_profiles.png")

    # Summary table
    print(f"\n{'='*80}")
    print("EQUATION FORM COMPARISON — SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Form':<25} {'best RMSE (mm)':<16} {'mean ± std (mm)':<20} {'R²':<10} {'time (s)':<10}")
    print(f"  {'-'*75}")
    for r in results:
        print(f"  {r['label']:<25} {r['zb_rmse']*1000:<16.2f} "
              f"{r['rmse_mean']*1000:.2f} ± {r['rmse_std']*1000:.2f}     "
              f"{r['zb_r2']:<10.4f} {r['wall_time_s']:<10.0f}")
