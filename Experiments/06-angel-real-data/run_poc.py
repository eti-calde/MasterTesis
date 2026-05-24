"""
Phase 1 — minimal proof-of-concept inversion (Exp. 6).

Angel-minimal setup: inlet Dirichlet eta(t) from S1 (x=1.5 m) + a single
interior observation sensor S2 (x=3.5 m, nearest the bump). Single seed.
Recovers z_b(x) from the real Hamburg-flume data and compares to ground
truth, reporting NRMSE the way Angel et al. do (RMSE / range), on both the
informative span x in [1.5, 9] m and the full domain.

Usage:
  python run_poc.py --adam 15000 --lbfgs 400 --seed 42
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
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"
INFORMATIVE_SPAN = (1.5, 9.0)   # waves carry bed signal here (matches Angel)


def make_figure(pinn, d, res, metrics, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_b = d["x_bathymetry"]
    zb_true = d["zb_true"]
    zb_pred = eval_bathymetry(pinn, x_b)

    Nt, Nx = d["Nt"], d["Nx"]
    eta_pred_grid = (res["h"] + res["zb_on_col"]).reshape(Nt, Nx)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (a) bathymetry recovery
    a = ax[0, 0]
    a.fill_between(x_b, 0, zb_true, color="0.85", label="True bed")
    a.plot(x_b, zb_true, "k-", lw=2, label="True $z_b$")
    a.plot(x_b, zb_pred, "r--", lw=2, label="PINN $z_b$")
    a.plot(d["snap"][0]["x_node"], 0, "b^", ms=10, label="S1 inlet (BC)")
    for s in d["obs_sensors"]:
        a.plot(d["snap"][s]["x_node"], 0, "gv", ms=10,
               label=f"S{s+1} obs" if s == d["obs_sensors"][0] else None)
    a.set(xlabel="x (m)", ylabel="bed elevation (m)",
          title="Bathymetry recovery — Angel 2024 real flume data")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # (b) error + NRMSE
    b = ax[0, 1]
    b.plot(x_b, (zb_pred - zb_true) * 1000, "r-")
    b.axhline(0, color="k", ls="--", alpha=0.5)
    b.axvspan(*INFORMATIVE_SPAN, color="green", alpha=0.07,
              label="informative span")
    b.set(xlabel="x (m)", ylabel="$z_b$ error (mm)",
          title=f"Error  |  NRMSE span={metrics['nrmse_span']*100:.1f}%  "
                f"full={metrics['nrmse_full']*100:.1f}%  "
                f"(Angel: 10–14%)")
    b.legend(fontsize=8)
    b.grid(alpha=0.3)

    # (c) S2 surface-elevation fit (time series at the obs node)
    c = ax[1, 0]
    s2 = d["obs_sensors"][0]
    node = d["snap"][s2]["node_idx"]
    c.plot(d["t_grid"], d["eta_obs"][:, node], "k-", lw=1.5,
           label=f"measured η @ S{s2+1}")
    c.plot(d["t_grid"], eta_pred_grid[:, node], "r--", lw=1.2,
           label="PINN η")
    c.set(xlabel="t (s, re-zeroed)", ylabel="η (m)",
          title=f"Surface-elevation fit at S{s2+1} (x={d['snap'][s2]['x_node']} m)")
    c.legend(fontsize=8)
    c.grid(alpha=0.3)

    # (d) loss curves + live zb RMSE
    dd = ax[1, 1]
    dd.semilogy(pinn.history["total"], label="total", alpha=0.8)
    dd.semilogy(pinn.history["data"], label="data", alpha=0.6)
    dd.semilogy(pinn.history["pde"], label="pde", alpha=0.6)
    dd.set(xlabel="iteration", ylabel="loss", title="Training history")
    dd.legend(fontsize=8, loc="upper right")
    dd.grid(alpha=0.3)
    if pinn.history["zb_rmse"]:
        tw = dd.twinx()
        tw.plot(np.array(pinn.history["zb_rmse"]) * 1000, "g-", alpha=0.5)
        tw.set_ylabel("$z_b$ RMSE (mm)", color="g")

    fig.suptitle("Exp. 6 POC — PINN bathymetry inversion from real data",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam", type=int, default=15000)
    ap.add_argument("--lbfgs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nx", type=int, default=136)   # sensors land on nodes
    ap.add_argument("--sigma-x", type=float, default=2.0,
                    help="BathNet Fourier sigma (raise for sharp bumps)")
    ap.add_argument("--lambda-pde", type=float, default=1.0)
    ap.add_argument("--obs", type=int, nargs="+", default=[1],
                    help="interior obs sensor indices (1=S2,2=S3,3=S4)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_angel_windowed(nx=args.nx, target_hz=10.0, t_window=(40.0, 60.0),
                            obs_sensors=tuple(args.obs))
    obs_lbl = "+".join(f"S{s+1}" for s in args.obs)
    print(f"POC: Nx={d['Nx']} Nt={d['Nt']} n_coll={d['n_coll']}  "
          f"inlet=S1  obs={obs_lbl}  seed={args.seed}  "
          f"sigma_x={args.sigma_x}  lambda_pde={args.lambda_pde}")

    pinn = AngelInversePINN(
        x_grid=d["x_grid"], t_grid=d["t_grid"],
        eta_obs=d["eta_obs"],
        t_obs_indices=d["t_obs_indices"],
        x_obs_indices=d["x_obs_indices"],
        kappa=d["kappa"],
        sol_hidden=5, sol_neurons=96,
        bath_hidden=3, bath_neurons=48,
        fourier_features=16, fourier_sigma_x=args.sigma_x, fourier_sigma_t=2.0,
        lambda_data=10.0, lambda_pde=args.lambda_pde,
        lambda_pos=10.0, lambda_tv=1e-4,
    )
    # live RMSE during training, on the collocation x-grid
    pinn.zb_true = np.interp(d["x_grid"], d["x_bathymetry"], d["zb_true"])

    t0 = time.time()
    pinn.train_adam(n_epochs=args.adam, print_every=max(args.adam // 10, 1))
    pinn.train_lbfgs(n_steps=args.lbfgs, print_every=max(args.lbfgs // 5, 1))
    elapsed = time.time() - t0
    print(f"\nTotal train time: {elapsed/60:.1f} min")

    res = pinn.get_results()
    zb_pred = eval_bathymetry(pinn, d["x_bathymetry"])
    nr_full, rmse_full = nrmse(zb_pred, d["zb_true"])
    nr_span, rmse_span = nrmse(zb_pred, d["zb_true"],
                               x=d["x_bathymetry"], span=INFORMATIVE_SPAN)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024**2
                    if pinn.device.type == "cuda" else None)

    metrics = {
        "seed": args.seed, "n_adam": args.adam, "n_lbfgs": args.lbfgs,
        "Nx": d["Nx"], "Nt": d["Nt"], "n_coll": d["n_coll"],
        "inlet": "S1", "obs": "S2",
        "nrmse_full": nr_full, "rmse_full_mm": rmse_full * 1000,
        "nrmse_span": nr_span, "rmse_span_mm": rmse_span * 1000,
        "span": INFORMATIVE_SPAN,
        "angel_benchmark_nrmse": [0.10, 0.14],
        "train_min": elapsed / 60,
        "peak_vram_mb": peak_vram_mb,
        "ms_per_adam_epoch": (elapsed * 1000 / (args.adam + args.lbfgs)),
    }

    print("\n" + "=" * 60)
    print("RESULTS (vs Angel adjoint benchmark NRMSE 10–14%)")
    print("=" * 60)
    print(f"  NRMSE informative span {INFORMATIVE_SPAN}: "
          f"{nr_span*100:.2f}%  (RMSE {rmse_span*1000:.2f} mm)")
    print(f"  NRMSE full domain:                {nr_full*100:.2f}%  "
          f"(RMSE {rmse_full*1000:.2f} mm)")

    RES_DIR.mkdir(exist_ok=True)
    out = RES_DIR / "poc.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {out}")

    make_figure(pinn, d, res, metrics, FIG_DIR / "poc_inversion.png")


if __name__ == "__main__":
    main()
