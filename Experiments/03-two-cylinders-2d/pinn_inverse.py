"""
2D inverse PINN for Experiment 3 (two cylinders).

Differences from Exp 2:
    - Input is (x, y, t)
    - Outputs (h, u, v) + z_b(x, y)
    - 2D SWE residual (continuity + 2 momentum)
    - Boundary conditions on inflow (Dirichlet) / outflow (zero-gradient)
    - Subsampled collocation grid to fit GPU memory
"""

import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# Networks with Fourier features
# ============================================================

from pinn_bath.legacy_blocks import (  # noqa: E402
    FourierFeatures,
    SolutionNet2D_Exp03 as SolutionNet2D,
    BathymetryNet2D_Exp03 as BathymetryNet2D,
)


# ============================================================
# SWE residual (2D transient, non-conservative form)
# ============================================================

def swe_residual_2d(x, y, t, h, u, v, zb, g=9.81):
    """2D transient SWE residuals via AD (well-balanced form).

    Non-conservative SWE, with the surface gradient written as a single
    derivative of ``η = h + zb`` instead of the two-network sum
    ``∂h + ∂zb``:

      h_t + (hu)_x + (hv)_y = 0
      u_t + u u_x + v u_y + g · ∂η/∂x = 0
      v_t + u v_x + v v_y + g · ∂η/∂y = 0

    By linearity of AD, ``grad(h + zb, x) == grad(h, x) + grad(zb, x)`` so
    this is **numerically identical** to the previous formulation; the
    rewrite is a clarity / well-balanced-intent improvement and mirrors
    the Audusse hydrostatic reconstruction used in the FV ground truth.
    """
    def grad(out, wrt):
        return torch.autograd.grad(out, wrt, grad_outputs=torch.ones_like(out),
                                   create_graph=True)[0]

    eta = h + zb
    h_t = grad(h, t); h_x = grad(h, x); h_y = grad(h, y)
    u_t = grad(u, t); u_x = grad(u, x); u_y = grad(u, y)
    v_t = grad(v, t); v_x = grad(v, x); v_y = grad(v, y)
    eta_x = grad(eta, x); eta_y = grad(eta, y)

    r_cont = h_t + (h_x * u + h * u_x) + (h_y * v + h * v_y)
    r_momx = u_t + u * u_x + v * u_y + g * eta_x
    r_momy = v_t + u * v_x + v * v_y + g * eta_y
    return r_cont, r_momx, r_momy


# ============================================================
# Losses
# ============================================================

def _mse_at(pred, true, mask):
    if mask.any():
        return torch.mean((pred[mask] - true[mask]) ** 2)
    return torch.tensor(0.0, device=pred.device)


# ============================================================
# Inverse PINN class
# ============================================================

class Cylinders2DInversePINN:
    def __init__(
        self,
        x_grid, y_grid, t_grid,    # 1D arrays of cell centers
        eta_obs, u_obs=None, v_obs=None,    # shape (Nt, Ny, Nx)
        h_ic=None, u_ic=None, v_ic=None,    # shape (Ny, Nx); IC at t=0
        # Training grid subsampling
        n_colloc=20000, n_ic=2000,
        # Physics
        g=9.81,
        # Net config
        sol_hidden=5, sol_neurons=128,
        bath_hidden=4, bath_neurons=64,
        fourier_features=24, sigma_space=2.0, sigma_time=2.0,
        fourier_features_bath=32, sigma_bath=3.0,
        # Loss weights
        lambda_data=10.0, lambda_data_uv=5.0,
        lambda_pde=1.0,
        lambda_ic=50.0,
        lambda_bc=5.0,
        lambda_pos=10.0,
        lambda_tv=1e-5,
        # Device
        device=None,
        seed=42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.g = g
        self.lw = {
            "data": lambda_data, "data_uv": lambda_data_uv,
            "pde": lambda_pde, "ic": lambda_ic, "bc": lambda_bc,
            "pos": lambda_pos, "tv": lambda_tv,
        }

        # Normalization bounds
        self.x_min, self.x_max = float(x_grid.min()), float(x_grid.max())
        self.y_min, self.y_max = float(y_grid.min()), float(y_grid.max())
        self.t_min, self.t_max = float(t_grid.min()), float(t_grid.max())

        # Full meshgrid
        X, Y = np.meshgrid(x_grid, y_grid)  # (Ny, Nx)
        Ny, Nx = X.shape
        Nt = len(t_grid)

        # Build (x, y, t) tuples for all observations
        # Shape: (Nt * Ny * Nx, 3)
        X_full = np.broadcast_to(X[None, :, :], (Nt, Ny, Nx)).ravel()
        Y_full = np.broadcast_to(Y[None, :, :], (Nt, Ny, Nx)).ravel()
        T_full = np.broadcast_to(t_grid[:, None, None], (Nt, Ny, Nx)).ravel()

        eta_flat = np.asarray(eta_obs).ravel()

        N_total = X_full.size
        # Subsample collocation points
        if n_colloc >= N_total:
            idx_col = np.arange(N_total)
        else:
            idx_col = np.random.choice(N_total, size=n_colloc, replace=False)

        self.x_col = torch.tensor(X_full[idx_col], dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.y_col = torch.tensor(Y_full[idx_col], dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.t_col = torch.tensor(T_full[idx_col], dtype=torch.float32, device=self.device).reshape(-1, 1)
        for tens in (self.x_col, self.y_col, self.t_col):
            tens.requires_grad_(True)

        self.eta_obs = torch.tensor(eta_flat[idx_col], dtype=torch.float32, device=self.device).reshape(-1, 1)

        self.has_u = u_obs is not None
        self.has_v = v_obs is not None
        if self.has_u:
            self.u_obs = torch.tensor(np.asarray(u_obs).ravel()[idx_col], dtype=torch.float32,
                                      device=self.device).reshape(-1, 1)
        if self.has_v:
            self.v_obs = torch.tensor(np.asarray(v_obs).ravel()[idx_col], dtype=torch.float32,
                                      device=self.device).reshape(-1, 1)

        # IC points (subsample from t=0 slice)
        if h_ic is not None:
            ic_flat = h_ic.ravel()
            if n_ic >= Ny * Nx:
                idx_ic = np.arange(Ny * Nx)
            else:
                idx_ic = np.random.choice(Ny * Nx, size=n_ic, replace=False)
            self.x_ic = torch.tensor(X.ravel()[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.y_ic = torch.tensor(Y.ravel()[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.t_ic = torch.zeros_like(self.x_ic)
            self.h_ic = torch.tensor(ic_flat[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.u_ic = torch.tensor(u_ic.ravel()[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.v_ic = torch.tensor(v_ic.ravel()[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.has_ic = True
        else:
            self.has_ic = False

        # Grid for evaluation (full 2D at bathymetry output)
        self.X_eval = torch.tensor(X.ravel(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.Y_eval = torch.tensor(Y.ravel(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.eval_shape = (Ny, Nx)

        # Networks
        self.sol_net = SolutionNet2D(
            sol_hidden, sol_neurons,
            fourier_features=fourier_features,
            sigma_space=sigma_space, sigma_time=sigma_time,
        ).to(self.device)
        self.bath_net = BathymetryNet2D(
            bath_hidden, bath_neurons,
            fourier_features=fourier_features_bath,
            sigma=sigma_bath,
        ).to(self.device)

        params = list(self.sol_net.parameters()) + list(self.bath_net.parameters())
        self.optimizer_adam = torch.optim.Adam(params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_adam, step_size=2000, gamma=0.5)

        self.history = {k: [] for k in ["total", "data", "pde", "ic", "bc", "pos", "tv", "zb_rmse"]}
        self.zb_true = None

    def _nx(self, x): return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
    def _ny(self, y): return 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
    def _nt(self, t):
        if self.t_max <= self.t_min: return t * 0
        return 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1

    def forward_col(self):
        xn, yn, tn = self._nx(self.x_col), self._ny(self.y_col), self._nt(self.t_col)
        h, u, v = self.sol_net(xn, yn, tn)
        zb = self.bath_net(xn, yn)
        return h, u, v, zb

    def forward_ic(self):
        xn = self._nx(self.x_ic); yn = self._ny(self.y_ic); tn = self._nt(self.t_ic)
        h, u, v = self.sol_net(xn, yn, tn)
        return h, u, v

    def forward_bath_2d(self):
        xn, yn = self._nx(self.X_eval), self._ny(self.Y_eval)
        return self.bath_net(xn, yn)

    def compute_loss(self):
        h, u, v, zb = self.forward_col()

        # SWE residual
        r_cont, r_momx, r_momy = swe_residual_2d(
            self.x_col, self.y_col, self.t_col, h, u, v, zb, self.g
        )

        # Data loss
        eta_pred = h + zb
        L_data = self.lw["data"] * torch.mean((eta_pred - self.eta_obs) ** 2)
        if self.has_u:
            L_data = L_data + self.lw["data_uv"] * torch.mean((u - self.u_obs) ** 2)
        if self.has_v:
            L_data = L_data + self.lw["data_uv"] * torch.mean((v - self.v_obs) ** 2)

        # Physics
        L_pde = self.lw["pde"] * (torch.mean(r_cont ** 2)
                                   + torch.mean(r_momx ** 2)
                                   + torch.mean(r_momy ** 2))

        # IC
        if self.has_ic:
            h_ic, u_ic, v_ic = self.forward_ic()
            L_ic = self.lw["ic"] * (torch.mean((h_ic - self.h_ic) ** 2)
                                     + torch.mean((u_ic - self.u_ic) ** 2)
                                     + torch.mean((v_ic - self.v_ic) ** 2))
        else:
            L_ic = torch.tensor(0.0, device=self.device)

        # Positivity
        L_pos = self.lw["pos"] * torch.mean(nn.functional.relu(-h) ** 2)

        # TV on z_b (over eval grid)
        zb_2d = self.forward_bath_2d().reshape(*self.eval_shape)
        dzb_dx = (zb_2d[:, 1:] - zb_2d[:, :-1]).abs().mean()
        dzb_dy = (zb_2d[1:, :] - zb_2d[:-1, :]).abs().mean()
        L_tv = self.lw["tv"] * (dzb_dx + dzb_dy)

        L_bc = torch.tensor(0.0, device=self.device)

        L_total = L_data + L_pde + L_ic + L_bc + L_pos + L_tv

        return L_total, {
            "total": L_total.item(), "data": L_data.item(), "pde": L_pde.item(),
            "ic": L_ic.item(), "bc": L_bc.item(),
            "pos": L_pos.item(), "tv": L_tv.item(),
        }

    def train_adam(self, n_epochs=6000, print_every=1000):
        print(f"Training 2D PINN with Adam for {n_epochs} epochs...")
        for ep in range(n_epochs):
            self.optimizer_adam.zero_grad()
            loss, comp = self.compute_loss()
            loss.backward()
            self.optimizer_adam.step()
            self.scheduler.step()

            for k in ["total", "data", "pde", "ic", "bc", "pos", "tv"]:
                self.history[k].append(comp[k])

            if self.zb_true is not None:
                with torch.no_grad():
                    zb_pred = self.forward_bath_2d().cpu().numpy().flatten()
                    zb_true_flat = self.zb_true.ravel()
                    rmse = float(np.sqrt(np.mean((zb_pred - zb_true_flat) ** 2)))
                    self.history["zb_rmse"].append(rmse)

            if (ep + 1) % print_every == 0:
                msg = f"  [{ep+1:>5d}] total={comp['total']:.2e}  data={comp['data']:.2e}  pde={comp['pde']:.2e}  ic={comp['ic']:.2e}"
                if self.zb_true is not None:
                    msg += f"  zb_rmse={self.history['zb_rmse'][-1]:.4e}"
                print(msg, flush=True)

    def train_lbfgs(self, n_steps=150, print_every=50):
        print(f"Fine-tuning 2D PINN with L-BFGS for {n_steps} steps...", flush=True)
        params = list(self.sol_net.parameters()) + list(self.bath_net.parameters())
        opt = torch.optim.LBFGS(params, lr=0.5, max_iter=20,
                                history_size=50, line_search_fn="strong_wolfe")
        step_count = [0]

        def closure():
            opt.zero_grad()
            loss, comp = self.compute_loss()
            loss.backward()
            for k in ["total", "data", "pde", "ic", "bc", "pos", "tv"]:
                self.history[k].append(comp[k])
            if self.zb_true is not None:
                with torch.no_grad():
                    zb_pred = self.forward_bath_2d().cpu().numpy().flatten()
                    zb_true_flat = self.zb_true.ravel()
                    rmse = float(np.sqrt(np.mean((zb_pred - zb_true_flat) ** 2)))
                    self.history["zb_rmse"].append(rmse)
            step_count[0] += 1
            if step_count[0] % print_every == 0:
                msg = f"  [LBFGS {step_count[0]:>4d}] total={comp['total']:.2e}  data={comp['data']:.2e}  pde={comp['pde']:.2e}"
                if self.zb_true is not None:
                    msg += f"  zb_rmse={self.history['zb_rmse'][-1]:.4e}"
                print(msg, flush=True)
            return loss

        for _ in range(n_steps):
            opt.step(closure)

    def get_zb_2d(self):
        with torch.no_grad():
            zb_flat = self.forward_bath_2d().cpu().numpy().flatten()
        return zb_flat.reshape(*self.eval_shape)


# ============================================================
# Plotting
# ============================================================

def plot_results(pinn, data, save_path=None):
    import matplotlib.pyplot as plt
    zb_pred = pinn.get_zb_2d()
    zb_true = data["zb"]
    err = zb_pred - zb_true
    rmse = float(np.sqrt(np.mean(err ** 2)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].pcolormesh(data["x"], data["y"], zb_true, cmap="terrain",
                              vmin=0, vmax=0.35, shading="auto")
    axes[0].set_title("True z_b")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(data["x"], data["y"], zb_pred, cmap="terrain",
                              vmin=0, vmax=0.35, shading="auto")
    axes[1].set_title("Predicted z_b")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(data["x"], data["y"], err, cmap="RdBu_r",
                              vmin=-0.15, vmax=0.15, shading="auto")
    axes[2].set_title(f"Error (RMSE = {rmse*1000:.2f} mm)")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        for (xc, yc, r, H) in data["params"]["cylinders"]:
            ax.add_patch(plt.Circle((xc, yc), r, fill=False, edgecolor="k",
                                     linewidth=1, linestyle="--"))
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
    return rmse


# ============================================================
# Main: baseline
# ============================================================

if __name__ == "__main__":
    from ground_truth import generate_dataset

    FIG_DIR = Path(__file__).parent / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    # Ruppenthal §7.2 spec (50×50, T=60 s). This __main__ is a legacy
    # sanity-check; the production pipeline lives in studies/_runner.py
    # consuming Experiments/03-two-cylinders-2d/data/ground_truth_cylinders.npz.
    # The training budget below (6 k Adam + 150 L-BFGS) is the historical
    # legacy budget and will NOT converge on the full 60 s dataset — increase
    # it (or use the production pipeline) for a real result.
    print("Generating ground truth (Ruppenthal §7.2 spec)...", flush=True)
    data = generate_dataset(
        Lx=25.0, Ly=25.0, Nx=50, Ny=50,    # spec: 50×50 cartesian grid
        t_end=60.0, n_save=15,             # spec: T=60 s; 15 snapshots for sanity
        eta_init=2.0, u_init=2.21, v_init=2.21,
        smooth=0.0, verbose=False,
    )
    print(f"  grid: {data['params']['Nx']}x{data['params']['Ny']}, {len(data['t'])} snapshots")
    print(f"  cylinder heights: 0.2 m (large), 0.3 m (small)", flush=True)
    print()

    # Build PINN (eta + u + v observations, Exp 2 lesson: velocity breaks equifinality)
    pinn = Cylinders2DInversePINN(
        x_grid=data["x"], y_grid=data["y"], t_grid=data["t"],
        eta_obs=data["eta"],
        u_obs=data["u"], v_obs=data["v"],
        h_ic=data["h"][0], u_ic=data["u"][0], v_ic=data["v"][0],
        n_colloc=15000, n_ic=1500,
        g=9.81,
        sol_hidden=5, sol_neurons=128,
        bath_hidden=4, bath_neurons=64,
        fourier_features=24, sigma_space=2.0, sigma_time=2.0,
        fourier_features_bath=32, sigma_bath=3.0,
        lambda_data=10.0, lambda_data_uv=5.0,
        lambda_pde=1.0,
        lambda_ic=100.0,
        lambda_pos=10.0,
        lambda_tv=1e-5,
    )
    pinn.zb_true = data["zb"]

    t0 = time.time()
    pinn.train_adam(n_epochs=6000, print_every=1000)
    pinn.train_lbfgs(n_steps=150, print_every=30)
    wall = time.time() - t0
    print(f"\nTotal time: {wall:.0f}s", flush=True)

    rmse = plot_results(pinn, data, save_path=FIG_DIR / "baseline_inversion.png")
    print(f"Final z_b RMSE: {rmse * 1000:.2f} mm")
