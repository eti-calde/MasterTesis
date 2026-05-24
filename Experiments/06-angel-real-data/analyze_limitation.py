"""
Exp. 6 — honest limitation analysis.

The soft-penalty PINN does NOT recover the Angel bump from real sparse-sensor
data. This script demonstrates that the failure is *robust* (not seed luck)
and quantifies it with the physically meaningful metric (recovered bump
peak height / location), not the flat-dominated domain NRMSE.

3 seeds, minimal Angel-replication config (S1 inlet + S2 obs), moderate
budget (the result plateaus by ~3k epochs — see results/poc_run.log).
Produces one honest figure + a JSON.
"""

import json
from pathlib import Path

import numpy as np
import torch

from data_angel import load_angel_windowed, eval_bathymetry, nrmse
from pinn_angel import AngelInversePINN

HERE = Path(__file__).parent
RES_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
SEEDS = [42, 0, 1]
N_ADAM = 4000
SPAN = (1.5, 9.0)


def one(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = load_angel_windowed(nx=136, target_hz=10.0, t_window=(40.0, 60.0),
                            obs_sensors=(1,))  # S1 inlet + S2 obs (Angel-minimal)
    p = AngelInversePINN(
        x_grid=d["x_grid"], t_grid=d["t_grid"], eta_obs=d["eta_obs"],
        t_obs_indices=d["t_obs_indices"], x_obs_indices=d["x_obs_indices"],
        kappa=d["kappa"], sol_hidden=5, sol_neurons=96,
        bath_hidden=3, bath_neurons=48, fourier_features=16,
        fourier_sigma_x=2.0, fourier_sigma_t=2.0,
        lambda_data=10.0, lambda_pde=1.0, lambda_pos=10.0, lambda_tv=1e-4)
    p.zb_true = np.interp(d["x_grid"], d["x_bathymetry"], d["zb_true"])
    p.train_adam(n_epochs=N_ADAM, print_every=N_ADAM)

    xb, zt = d["x_bathymetry"], d["zb_true"]
    zb = eval_bathymetry(p, xb)
    ip = int(np.argmax(zt))
    nr_full, rmse_full = nrmse(zb, zt)
    nr_span, rmse_span = nrmse(zb, zt, x=xb, span=SPAN)
    return {
        "seed": seed,
        "true_peak_mm": float(zt.max() * 1000),
        "true_peak_x": float(xb[ip]),
        "pinn_at_peak_mm": float(zb[ip] * 1000),
        "pinn_peak_recovery_pct": float(zb[ip] / zt.max() * 100),
        "pinn_max_mm": float(zb.max() * 1000),
        "pinn_max_x": float(xb[int(np.argmax(zb))]),
        "nrmse_full": nr_full, "nrmse_span": nr_span,
        "rmse_full_mm": rmse_full * 1000, "rmse_span_mm": rmse_span * 1000,
        "zb_pred": zb.tolist(),
    }


def main():
    flume = load_angel_windowed(nx=136, obs_sensors=(1,))
    xb, zt = flume["x_bathymetry"], flume["zb_true"]

    runs = []
    for s in SEEDS:
        print(f"=== seed {s} ===")
        r = one(s)
        runs.append(r)
        print(f"  peak recovery: {r['pinn_at_peak_mm']:.1f} mm / "
              f"{r['true_peak_mm']:.1f} mm = {r['pinn_peak_recovery_pct']:.1f}%  "
              f"| NRMSE span={r['nrmse_span']*100:.1f}% full={r['nrmse_full']*100:.1f}%")

    rec = np.array([r["pinn_peak_recovery_pct"] for r in runs])
    summary = {
        "config": "S1 inlet (Dirichlet) + S2 obs  [Angel-minimal]",
        "n_adam": N_ADAM, "seeds": SEEDS,
        "peak_recovery_pct_mean": float(rec.mean()),
        "peak_recovery_pct_std": float(rec.std()),
        "nrmse_span_mean": float(np.mean([r["nrmse_span"] for r in runs])),
        "nrmse_full_mean": float(np.mean([r["nrmse_full"] for r in runs])),
        "angel_adjoint_nrmse": [0.10, 0.14],
        "verdict": (
            f"soft-penalty PINN fails to localize the bump: peak recovery "
            f"{rec.mean():.1f} +/- {rec.std():.1f}% across seeds "
            f"(~{100-rec.mean():.0f}% of the bump height missed). The "
            f"flat-dominated domain NRMSE "
            f"({np.mean([r['nrmse_full'] for r in runs])*100:.0f}%) is not a "
            f"meaningful success metric here; the physically relevant peak "
            f"recovery is."),
        "runs": runs,
    }
    RES_DIR.mkdir(exist_ok=True)
    (RES_DIR / "limitation_analysis.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPeak recovery across seeds: "
          f"{rec.mean():.1f} +/- {rec.std():.1f} %  (target: ~100%)")

    # ---- honest figure -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    a = ax[0]
    a.fill_between(xb, 0, zt * 1000, color="0.8", label="True bed (Angel)")
    a.plot(xb, zt * 1000, "k-", lw=2)
    for r in runs:
        a.plot(xb, np.array(r["zb_pred"]) * 1000, "--", lw=1.3,
                label=f"PINN seed {r['seed']}")
    a.plot(1.5, 0, "b^", ms=11, label="S1 inlet (BC)")
    a.plot(3.5, 0, "gv", ms=11, label="S2 obs")
    a.annotate("true bump\n199.7 mm", (3.99, 200), (6, 165),
               arrowprops=dict(arrowstyle="->"), fontsize=9)
    a.set(xlabel="x (m)", ylabel="bed elevation (mm)",
          title="Bathymetry: soft-PINN misses the bump (real Angel data)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    b = ax[1]
    b.bar(range(len(runs)), [r["pinn_peak_recovery_pct"] for r in runs],
          color="#d62728", width=0.5)
    b.axhline(100, color="k", ls="--", label="full recovery (target)")
    b.axhspan(70, 90, color="green", alpha=0.12,
              label="Angel adjoint (NRMSE 10–14% ⇒ good peak recovery)")
    b.set_xticks(range(len(runs)))
    b.set_xticklabels([f"seed {r['seed']}" for r in runs])
    b.set(ylabel="bump-peak recovery (%)", ylim=(0, 110),
          title=f"Peak recovery {rec.mean():.1f}±{rec.std():.1f}% — "
                f"robust failure across seeds")
    b.legend(fontsize=8)
    b.grid(axis="y", alpha=0.3)

    fig.suptitle("Exp. 6 — soft-penalty PINN limitation on real sparse-sensor "
                 "bathymetry inversion", fontweight="bold")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "limitation_analysis.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_DIR/'limitation_analysis.png'}")


if __name__ == "__main__":
    main()
