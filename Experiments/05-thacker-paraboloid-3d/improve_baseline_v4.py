"""
Round 4: extend the rim z_b supervision into the wet disk via the IC.

Key insight from Rounds 1-3 (all plateaued at ~50 mm zb_rmse_wet):
the predicted z_b inside the wet disk is offset uniformly +0.07 m above
truth — the optimizer is locked into a "shallow basin" branch of the
(h, z_b) equifinality and neither velocity supervision (Round 2) nor a
20× heavier IC weight (Round 3) breaks it.

But h_ic is part of the IC supervision (analytical/measured initial state),
and η_obs(t=0) is observed. So z_b = η_obs(t=0) - h_ic gives a *direct,
dense supervision* of the bath net at every IC point — including the wet
disk. This subsumes the dry-rim supervision and adds explicit anchoring
inside the basin.

We expose this as a new loss term L_zb_ic in pinn_inverse.py and switch it
on in this driver.
"""

import time
from pathlib import Path

import numpy as np

from ground_truth import generate_dataset
from pinn_inverse import Thacker3DInversePINN, plot_results


def main():
    fig_dir = Path(__file__).parent / "figures"
    res_dir = Path(__file__).parent / "results"
    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    print("Generating 3D Thacker ground truth...", flush=True)
    data = generate_dataset(
        L=4.0, Nx=40, Ny=40,
        x_c=2.0, y_c=2.0, h_0=0.1, a=1.0, r_0=0.8,
        n_periods=3, n_save=30,
    )

    pinn = Thacker3DInversePINN(
        x_grid=data["x"], y_grid=data["y"], t_grid=data["t"],
        eta_obs=data["eta"],
        u_obs=data["u"], v_obs=data["v"],
        h_ic=data["h"][0], u_ic=data["u"][0], v_ic=data["v"][0],
        t_obs_indices=np.arange(len(data["t"])),
        n_colloc=12000, n_ic=1500,
        sol_hidden=5, sol_neurons=128,
        bath_hidden=4, bath_neurons=64,
        fourier_features=24, sigma_space=3.0, sigma_time=2.0,
        fourier_features_bath=32, sigma_bath=4.0,
        lambda_data=10.0, lambda_data_uv=5.0,
        lambda_pde=1.0,
        lambda_ic=100.0,           # back to canonical (Round 3's 2000 didn't help)
        lambda_dry=10.0,
        lambda_dry_zb=1000.0,      # rim anchor
        lambda_zb_ic=1000.0,       # NEW: bath_net anchored at IC (wet disk too)
        lambda_tv=1e-5,
    )
    pinn.zb_true = data["zb"]
    pinn.wet_mask = (data["h"] > 1e-4).any(axis=0)

    print("\n=== Joint training (η+u+v, IC z_b anchor active) ===", flush=True)
    t0 = time.time()
    pinn.train_adam(n_epochs=6000, print_every=1000)
    pinn.train_lbfgs(n_steps=120, print_every=30)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)

    rmse_wet = plot_results(
        pinn, data,
        save_path=fig_dir / "improved_v4_inversion.png",
        title_suffix="— Round 4 (z_b supervised at IC via h_ic)",
    )
    print(f"Final z_b RMSE (ever-wet): {rmse_wet * 1000:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
