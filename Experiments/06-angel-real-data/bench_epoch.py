"""
Phase 0 — per-epoch timing benchmark (Exp. 6).

Builds the REAL Angel PINN (real flume data, real kappa drag) at a chosen
config, warms up, then measures:
  - Adam: mean / p90 ms per epoch
  - L-BFGS: mean s per step
  - peak GPU memory (allocated / reserved)
and prints a projected wall-clock table for a full run (15k Adam + 400 L-BFGS)
across the {mid, poc, high} configs so the compute decision is data-driven.

Usage:
  python bench_epoch.py --config poc --epochs 100 --lbfgs 20
"""

import argparse
import time

import numpy as np
import torch

from data_angel import load_angel_windowed
from pinn_angel import AngelInversePINN

# (Nx, target_hz, t_window) per config — see plan's cost table.
# nx values 136 / 271 / 541 give dx = 100 / 50 / 25 mm and place every
# Angel sensor (1.5, 3.5, 5.5, 7.5 m) exactly on a grid node. Previous
# values (80/120/160) caused 38–70 mm snap errors — see data_angel.py.
CONFIGS = {
    "mid":  dict(nx=136, target_hz=8.0,  t_window=(40.0, 55.0)),
    "poc":  dict(nx=136, target_hz=10.0, t_window=(40.0, 60.0)),
    "high": dict(nx=271, target_hz=13.0, t_window=(38.0, 62.0)),
}
FULL_ADAM, FULL_LBFGS = 15_000, 400  # full-run budget for extrapolation


def build_pinn(cfg):
    d = load_angel_windowed(
        nx=cfg["nx"], target_hz=cfg["target_hz"], t_window=cfg["t_window"],
        obs_sensors=(1,),  # S2 only (POC sensor set) — count is what drives cost
    )
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
    pinn.zb_true = None  # skip per-epoch RMSE eval during pure timing
    return pinn, d


def time_adam(pinn, n_epochs, warmup=10):
    cuda = pinn.device.type == "cuda"
    for _ in range(warmup):
        pinn.optimizer_adam.zero_grad()
        loss, _ = pinn.compute_loss()
        loss.backward()
        pinn.optimizer_adam.step()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    per = np.empty(n_epochs)
    for i in range(n_epochs):
        t0 = time.perf_counter()
        pinn.optimizer_adam.zero_grad()
        loss, _ = pinn.compute_loss()
        loss.backward()
        pinn.optimizer_adam.step()
        if cuda:
            torch.cuda.synchronize()
        per[i] = (time.perf_counter() - t0) * 1e3  # ms
    return per


def time_lbfgs(pinn, n_steps):
    cuda = pinn.device.type == "cuda"
    params = list(pinn.sol_net.parameters()) + list(pinn.bath_net.parameters())
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=20, history_size=50,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss, _ = pinn.compute_loss()
        loss.backward()
        return loss

    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt.step(closure)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps  # s/step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="poc")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lbfgs", type=int, default=20)
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    pinn, d = build_pinn(cfg)
    dev = pinn.device
    print(f"Device: {dev} "
          f"({torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'})")
    print(f"Config '{args.config}': Nx={d['Nx']} Nt={d['Nt']} "
          f"n_coll={d['n_coll']}  window={d['t_window']}s @ {d['target_hz']}Hz")
    print()

    per = time_adam(pinn, args.epochs)
    ms_mean, ms_p90 = float(per.mean()), float(np.percentile(per, 90))
    print(f"Adam:   mean={ms_mean:.1f} ms/epoch  p90={ms_p90:.1f} ms/epoch  "
          f"(n={args.epochs})")

    s_lbfgs = time_lbfgs(pinn, args.lbfgs)
    print(f"L-BFGS: mean={s_lbfgs:.2f} s/step  (n={args.lbfgs})")

    if dev.type == "cuda":
        mb = 1024 ** 2
        print(f"Peak VRAM: allocated={torch.cuda.max_memory_allocated()/mb:.0f} MB "
              f"reserved={torch.cuda.max_memory_reserved()/mb:.0f} MB "
              f"(GTX 1650 = 4096 MB)")

    full = (FULL_ADAM * ms_mean / 1e3 + FULL_LBFGS * s_lbfgs)
    print(f"\nProjected full run ({FULL_ADAM} Adam + {FULL_LBFGS} L-BFGS) "
          f"at THIS config: {full/60:.0f} min ({full/3600:.2f} h)")

    # extrapolate to other configs by collocation-point scaling
    print("\nExtrapolation (scaling per-epoch by n_coll, ~48 ms fixed overhead):")
    OVERHEAD = 48.0
    slope = max(ms_mean - OVERHEAD, 0.0) / d["n_coll"]
    print(f"  {'config':6s} {'n_coll':>8s} {'ms/epoch':>9s} "
          f"{'full(h)':>8s}")
    for name, c in CONFIGS.items():
        dd = load_angel_windowed(nx=c["nx"], target_hz=c["target_hz"],
                                 t_window=c["t_window"], obs_sensors=(1,))
        est_ms = OVERHEAD + slope * dd["n_coll"]
        est_full = (FULL_ADAM * est_ms / 1e3 + FULL_LBFGS * s_lbfgs) / 3600
        mark = "  <- measured" if name == args.config else ""
        print(f"  {name:6s} {dd['n_coll']:8d} {est_ms:9.1f} "
              f"{est_full:8.2f}{mark}")


if __name__ == "__main__":
    main()
