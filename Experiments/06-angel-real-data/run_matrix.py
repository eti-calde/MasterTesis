"""
Phase 2 — run matrix (Exp. 6).

Compares two sensor configurations, each over multiple seeds:
  A  Angel-minimal      : inlet S1  + obs S2
  B  thesis-advantage   : inlet S1  + obs S2, S3, S4

Reports mean +/- std NRMSE per config (informative span + full domain),
side-by-side with Angel's adjoint benchmark (10-14 %), and a bar figure.

Optional --ensemble repeats config B over the 20 individual flume runs
(angel2024_run01..20.npz) to get an observational-uncertainty band on the
recovered bathymetry. Off by default (adds ~6 h at full budget).

Usage:
  python run_matrix.py --adam 15000 --lbfgs 200 --seeds 0 1 2
  python run_matrix.py --ensemble --adam 15000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data_angel import load_angel_windowed, eval_bathymetry, nrmse
from pinn_angel import AngelInversePINN

HERE = Path(__file__).parent
RES_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
INFORMATIVE_SPAN = (1.5, 9.0)

CONFIGS = {
    "A_minimal":   (1,),          # S2 only
    "B_advantage": (1, 2, 3),     # S2 + S3 + S4
}


def run_one(seed, obs_sensors, n_adam, n_lbfgs, nx=136, run=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = load_angel_windowed(nx=nx, target_hz=10.0, t_window=(40.0, 60.0),
                            obs_sensors=obs_sensors, run=run)
    pinn = AngelInversePINN(
        x_grid=d["x_grid"], t_grid=d["t_grid"],
        eta_obs=d["eta_obs"],
        t_obs_indices=d["t_obs_indices"],
        x_obs_indices=d["x_obs_indices"],
        kappa=d["kappa"],
        sol_hidden=5, sol_neurons=96,
        bath_hidden=3, bath_neurons=48,
        fourier_features=16, fourier_sigma_x=2.0, fourier_sigma_t=2.0,
        lambda_data=10.0, lambda_pde=1.0, lambda_pos=10.0, lambda_tv=1e-4,
    )
    pinn.zb_true = np.interp(d["x_grid"], d["x_bathymetry"], d["zb_true"])
    t0 = time.time()
    pinn.train_adam(n_epochs=n_adam, print_every=max(n_adam, 1))
    pinn.train_lbfgs(n_steps=n_lbfgs, print_every=max(n_lbfgs, 1))
    elapsed = time.time() - t0

    zb_pred = eval_bathymetry(pinn, d["x_bathymetry"])
    nr_full, rmse_full = nrmse(zb_pred, d["zb_true"])
    nr_span, rmse_span = nrmse(zb_pred, d["zb_true"],
                               x=d["x_bathymetry"], span=INFORMATIVE_SPAN)
    return {
        "seed": seed, "obs_sensors": list(obs_sensors), "run": run,
        "nrmse_full": nr_full, "nrmse_span": nr_span,
        "rmse_full_mm": rmse_full * 1000, "rmse_span_mm": rmse_span * 1000,
        "train_min": elapsed / 60,
        "zb_pred": zb_pred.tolist(), "x_b": d["x_bathymetry"].tolist(),
        "zb_true": d["zb_true"].tolist(),
    }


def make_bar_figure(agg, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(agg.keys())
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    for off, key, lab, col in [(-0.18, "nrmse_span", "informative span", "#1f77b4"),
                               (0.18, "nrmse_full", "full domain", "#ff7f0e")]:
        means = [agg[n][key]["mean"] * 100 for n in names]
        stds = [agg[n][key]["std"] * 100 for n in names]
        ax.bar(x + off, means, 0.36, yerr=stds, capsize=5, label=lab, color=col)
    ax.axhspan(10, 14, color="green", alpha=0.15,
               label="Angel adjoint benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(S1+{'+'.join('S'+str(s+1) for s in CONFIGS[n])})"
                        for n in names])
    ax.set_ylabel("NRMSE (%)")
    ax.set_title("Exp. 6 — bathymetry-inversion NRMSE by sensor configuration")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam", type=int, default=15000)
    ap.add_argument("--lbfgs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ensemble", action="store_true",
                    help="config B over the 20 individual flume runs")
    args = ap.parse_args()
    RES_DIR.mkdir(exist_ok=True)

    if args.ensemble:
        runs = []
        for r in range(1, 21):
            print(f"\n=== ensemble run {r:02d}/20 (config B, seed 0) ===")
            runs.append(run_one(0, CONFIGS["B_advantage"], args.adam,
                                 args.lbfgs, run=r))
        (RES_DIR / "ensemble_results.json").write_text(json.dumps(runs, indent=2))
        nr = np.array([x["nrmse_full"] for x in runs])
        print(f"\nEnsemble NRMSE full: {nr.mean()*100:.2f} +/- "
              f"{nr.std()*100:.2f} %  (n=20)")
        return

    all_runs, agg = [], {}
    for name, sensors in CONFIGS.items():
        rs = []
        for s in args.seeds:
            print(f"\n=== {name}  seed={s} ===")
            r = run_one(s, sensors, args.adam, args.lbfgs)
            r["config"] = name
            rs.append(r)
            all_runs.append(r)
            print(f"  NRMSE span={r['nrmse_span']*100:.2f}%  "
                  f"full={r['nrmse_full']*100:.2f}%  ({r['train_min']:.1f} min)")
        agg[name] = {
            k: {"mean": float(np.mean([x[k] for x in rs])),
                "std": float(np.std([x[k] for x in rs]))}
            for k in ("nrmse_span", "nrmse_full")
        }

    (RES_DIR / "matrix_results.json").write_text(
        json.dumps({"agg": agg, "runs": all_runs,
                    "angel_benchmark": [0.10, 0.14]}, indent=2))
    print("\n" + "=" * 60)
    print("MATRIX SUMMARY (vs Angel adjoint NRMSE 10-14%)")
    print("=" * 60)
    for n in agg:
        print(f"  {n:14s}  span={agg[n]['nrmse_span']['mean']*100:5.2f}"
              f"+/-{agg[n]['nrmse_span']['std']*100:.2f}%   "
              f"full={agg[n]['nrmse_full']['mean']*100:5.2f}"
              f"+/-{agg[n]['nrmse_full']['std']*100:.2f}%")
    make_bar_figure(agg, FIG_DIR / "matrix_nrmse.png")


if __name__ == "__main__":
    main()
