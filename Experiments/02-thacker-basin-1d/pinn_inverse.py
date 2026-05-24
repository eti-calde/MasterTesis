"""
Transient inverse PINN for Experiment 2 (Thacker oscillating basin).

Differences from Experiment 1:
    - Input is (x, t), not just x
    - SWE residual includes time derivatives
    - Closed basin BCs: u = 0 at x = ±L/2 (no flow through walls)
    - Wetting/drying: h can be zero; add dry-cell velocity loss
    - Bathymetry z_b can be negative (basin below datum)
    - Initial condition loss (known h, u at t=0)

Architecture:
    - Solution network: (x, t) -> (h, u)
    - Bathymetry network: x -> z_b (time-independent)
    - Separate Fourier features for each

Loss:
    L = lambda_data   * L_eta           (observations)
      + lambda_pde    * L_SWE           (physics residual)
      + lambda_ic     * L_IC            (known initial state)
      + lambda_bc     * L_BC            (u = 0 at walls)
      + lambda_dry    * L_dry           (u = 0 where h = 0)
      + lambda_tv     * L_TV(z_b)       (regularization)
      + lambda_pos    * L_pos           (h >= 0)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# Networks (with Fourier features)
# ============================================================

from pinn_bath.legacy_blocks import (  # noqa: E402
    FourierFeatures,
    SolutionNet_Exp02 as SolutionNet,
    BathymetryNet_Exp02 as BathymetryNet,
)


# ============================================================
# SWE residual (1D transient, non-conservative form)
# ============================================================

def swe_residual_transient(x, t, h, u, zb, g=9.81, n_manning=0.0,
                            eps_dry=1e-4, wet_scale=None):
    """Compute 1D transient SWE residuals via AD.

    Non-conservative form (simpler for PINNs near dry bed):
      continuity:  dh/dt + u * dh/dx + h * du/dx = 0
      momentum:    du/dt + u * du/dx + g * (dh/dx + dz_b/dx) + friction = 0

    Returns ``(r_cont, r_mom, wet)`` where ``wet`` is a smooth indicator
    ``≈1`` in wet cells (``h >> eps_dry``) and ``≈0`` in dry cells. The
    caller is expected to weight the squared residuals by ``wet`` before
    averaging — see ``ThackerInversePINN.compute_loss``. ``wet_scale``
    controls the transition width (defaults to ``eps_dry``).

    Why mask: ``h`` is parameterised through ``softplus`` (always > 0)
    so the dry region never has exactly zero h or zero gradients, and
    ``g · ∂h/∂x``, ``u · ∂u/∂x`` etc. inject a fictitious force into
    the PDE loss. Masking by a smooth wet indicator keeps shoreline
    motion informative (the transition cells still contribute since
    ``wet`` is smooth) while suppressing the fictitious dry-cell force.
    """
    dh_dt = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h),
                                create_graph=True)[0]
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    dh_dx = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h),
                                create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    dzb_dx = torch.autograd.grad(zb, x, grad_outputs=torch.ones_like(zb),
                                 create_graph=True)[0]

    r_cont = dh_dt + u * dh_dx + h * du_dx
    r_mom = du_dt + u * du_dx + g * (dh_dx + dzb_dx)

    if n_manning > 0:
        Sf = g * n_manning**2 * u * torch.abs(u) / (h + eps_dry)**(4.0 / 3.0)
        r_mom = r_mom + Sf

    scale = wet_scale if wet_scale is not None else eps_dry
    wet = torch.sigmoid((h - eps_dry) / scale)
    return r_cont, r_mom, wet


# ============================================================
# Loss components
# ============================================================

def loss_data_eta(h, zb, eta_obs, obs_mask=None):
    eta_pred = h + zb
    if obs_mask is not None:
        return torch.mean((eta_pred[obs_mask] - eta_obs[obs_mask])**2)
    return torch.mean((eta_pred - eta_obs)**2)


def loss_data_u(u, u_obs, obs_mask=None):
    if obs_mask is not None:
        return torch.mean((u[obs_mask] - u_obs[obs_mask])**2)
    return torch.mean((u - u_obs)**2)


def loss_ic(h, u, h_ic, u_ic, ic_mask):
    """Initial condition: predicted (h, u) at t=0 must match known."""
    return (torch.mean((h[ic_mask] - h_ic[ic_mask])**2)
            + torch.mean((u[ic_mask] - u_ic[ic_mask])**2))


def loss_bc_walls(u, bc_mask):
    """u = 0 at basin walls (closed BC)."""
    if bc_mask.any():
        return torch.mean(u[bc_mask]**2)
    return torch.tensor(0.0, device=u.device)


def loss_dry_cells(h, u, eps_dry=1e-4):
    """Where h is small, u should be zero (no velocity on dry bed)."""
    dry = torch.sigmoid(1.0 - h / eps_dry)  # ~1 where dry, ~0 where wet
    return torch.mean((dry * u)**2)


def loss_positivity(h):
    """Penalize negative h."""
    return torch.mean(nn.functional.relu(-h)**2)


def loss_tv(zb_1d, x_1d):
    """Total variation regularization on bathymetry."""
    dzb = zb_1d[1:] - zb_1d[:-1]
    dx = x_1d[1:] - x_1d[:-1]
    return torch.mean(torch.abs(dzb / dx))


# ============================================================
# Inverse PINN
# ============================================================

class ThackerInversePINN:
    """Transient inverse PINN for Thacker case."""

    def __init__(
        self,
        # Spatial / temporal samples
        x_grid, t_grid,
        # Observations (possibly sparse in space-time)
        eta_obs,                # shape (Nt_obs, Nx_obs)
        t_obs_indices,          # indices into t_grid that have observations
        x_obs_indices=None,     # spatial indices (None = all x)
        u_obs=None,             # optional velocity observations, same shape
        # Initial condition (known at t=0)
        h_ic=None, u_ic=None,
        # Physical params
        g=9.81,
        n_manning=0.0,
        eps_dry=1e-4,
        # Network config
        sol_hidden=5, sol_neurons=96,
        bath_hidden=3, bath_neurons=48,
        fourier_features=16, fourier_sigma_x=2.0, fourier_sigma_t=2.0,
        # Loss weights
        lambda_data=10.0,
        lambda_data_u=10.0,
        lambda_pde=1.0,
        lambda_ic=100.0,
        lambda_bc=10.0,
        lambda_dry=10.0,
        lambda_pos=10.0,
        lambda_tv=1e-4,
        # Device
        device=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.g = g
        self.n_manning = n_manning
        self.eps_dry = eps_dry

        # Loss weights
        self.lw = {
            "data": lambda_data, "data_u": lambda_data_u,
            "pde": lambda_pde, "ic": lambda_ic, "bc": lambda_bc,
            "dry": lambda_dry, "pos": lambda_pos, "tv": lambda_tv,
        }

        # Build full (x, t) collocation grid
        Nx, Nt = len(x_grid), len(t_grid)
        X_coll, T_coll = np.meshgrid(x_grid, t_grid)  # shape (Nt, Nx)
        self.x_col = torch.tensor(X_coll.ravel(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.t_col = torch.tensor(T_coll.ravel(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.x_col.requires_grad_(True)
        self.t_col.requires_grad_(True)

        # Normalization (x, t -> [-1, 1])
        self.x_min, self.x_max = float(x_grid.min()), float(x_grid.max())
        self.t_min, self.t_max = float(t_grid.min()), float(t_grid.max())

        # 1D x-only grid for bathymetry network + TV
        self.x_1d = torch.tensor(x_grid, dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.x_1d.requires_grad_(True)

        # Observation data (full grid shape Nt x Nx)
        self.eta_obs_full = torch.tensor(eta_obs, dtype=torch.float32, device=self.device)
        self.has_u = u_obs is not None
        if self.has_u:
            self.u_obs_full = torch.tensor(u_obs, dtype=torch.float32, device=self.device)

        # Build observation mask (which collocation points have data)
        obs_mask = np.zeros((Nt, Nx), dtype=bool)
        if x_obs_indices is None:
            x_obs_indices = np.arange(Nx)
        for ti in t_obs_indices:
            for xi in x_obs_indices:
                obs_mask[ti, xi] = True
        self.obs_mask = torch.tensor(obs_mask.ravel(), device=self.device)
        self.t_obs_indices = t_obs_indices
        self.x_obs_indices = x_obs_indices

        # IC mask (t == 0)
        self.ic_mask = (self.t_col.flatten() == self.t_min)
        if h_ic is not None and u_ic is not None:
            # Repeat IC across all t rows in the grid (for broadcasting at ic_mask)
            h_ic_full = np.tile(h_ic, (Nt, 1))
            u_ic_full = np.tile(u_ic, (Nt, 1))
            self.h_ic = torch.tensor(h_ic_full.ravel(), dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.u_ic = torch.tensor(u_ic_full.ravel(), dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.has_ic = True
        else:
            self.has_ic = False

        # BC mask (x at boundaries, all t)
        tol_x = 0.5 * (x_grid[1] - x_grid[0])
        at_left = (self.x_col.flatten() - self.x_min).abs() < tol_x
        at_right = (self.x_max - self.x_col.flatten()).abs() < tol_x
        self.bc_mask = (at_left | at_right)

        # Networks
        self.sol_net = SolutionNet(
            sol_hidden, sol_neurons,
            fourier_features=fourier_features,
            fourier_sigma_x=fourier_sigma_x,
            fourier_sigma_t=fourier_sigma_t,
        ).to(self.device)
        self.bath_net = BathymetryNet(
            bath_hidden, bath_neurons,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma_x,
        ).to(self.device)

        # Optimizers
        params = list(self.sol_net.parameters()) + list(self.bath_net.parameters())
        self.optimizer_adam = torch.optim.Adam(params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_adam, step_size=2000, gamma=0.5)

        # History
        self.history = {k: [] for k in ["total", "data", "pde", "ic", "bc", "dry", "pos", "tv", "zb_rmse"]}
        self.zb_true = None

    def _normalize_x(self, x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1

    def _normalize_t(self, t):
        return 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1 if self.t_max > self.t_min else t * 0

    def forward(self):
        x_n = self._normalize_x(self.x_col)
        t_n = self._normalize_t(self.t_col)
        h, u = self.sol_net(x_n, t_n)
        # z_b at each collocation x
        zb = self.bath_net(self._normalize_x(self.x_col))
        return h, u, zb

    def forward_1d(self):
        """Bathymetry network only, on 1D x grid (for TV and evaluation)."""
        return self.bath_net(self._normalize_x(self.x_1d))

    def compute_loss(self):
        h, u, zb = self.forward()

        # SWE residuals (with smooth wet indicator — see swe_residual_transient
        # docstring; weights the squared residuals to suppress fictitious
        # force in dry cells where softplus never reaches zero).
        r_cont, r_mom, wet = swe_residual_transient(
            self.x_col, self.t_col, h, u, zb, self.g, self.n_manning, self.eps_dry
        )

        # Data loss
        eta_pred = h + zb
        eta_obs_flat = self.eta_obs_full.reshape(-1, 1)
        L_data = self.lw["data"] * torch.mean((eta_pred[self.obs_mask] - eta_obs_flat[self.obs_mask])**2)
        if self.has_u:
            u_obs_flat = self.u_obs_full.reshape(-1, 1)
            L_data = L_data + self.lw["data_u"] * torch.mean((u[self.obs_mask] - u_obs_flat[self.obs_mask])**2)

        # Physics — masked by wet indicator
        L_pde = self.lw["pde"] * (torch.mean(wet * r_cont**2) + torch.mean(wet * r_mom**2))

        # IC
        if self.has_ic:
            L_ic = self.lw["ic"] * (torch.mean((h[self.ic_mask] - self.h_ic[self.ic_mask])**2)
                                     + torch.mean((u[self.ic_mask] - self.u_ic[self.ic_mask])**2))
        else:
            L_ic = torch.tensor(0.0, device=self.device)

        # BCs (walls)
        if self.bc_mask.any():
            L_bc = self.lw["bc"] * torch.mean(u[self.bc_mask]**2)
        else:
            L_bc = torch.tensor(0.0, device=self.device)

        # Dry cells
        L_dry = self.lw["dry"] * loss_dry_cells(h, u, self.eps_dry)

        # Positivity
        L_pos = self.lw["pos"] * loss_positivity(h)

        # TV on z_b (evaluated on 1D x grid)
        zb_1d = self.forward_1d().flatten()
        L_tv = self.lw["tv"] * loss_tv(zb_1d, self.x_1d.flatten())

        L_total = L_data + L_pde + L_ic + L_bc + L_dry + L_pos + L_tv

        return L_total, {
            "total": L_total.item(), "data": L_data.item(), "pde": L_pde.item(),
            "ic": L_ic.item(), "bc": L_bc.item(), "dry": L_dry.item(),
            "pos": L_pos.item(), "tv": L_tv.item(),
        }

    def train_adam(self, n_epochs=10000, print_every=1000):
        print(f"Training with Adam for {n_epochs} epochs...")
        for ep in range(n_epochs):
            self.optimizer_adam.zero_grad()
            loss, comp = self.compute_loss()
            loss.backward()
            self.optimizer_adam.step()
            self.scheduler.step()

            for k in ["total", "data", "pde", "ic", "bc", "dry", "pos", "tv"]:
                self.history[k].append(comp[k])

            if self.zb_true is not None:
                with torch.no_grad():
                    zb_1d = self.forward_1d().cpu().numpy().flatten()
                    rmse = np.sqrt(np.mean((zb_1d - self.zb_true)**2))
                    self.history["zb_rmse"].append(rmse)

            if (ep + 1) % print_every == 0:
                msg = f"  [{ep+1:>6d}] total={comp['total']:.2e}  data={comp['data']:.2e}  pde={comp['pde']:.2e}  ic={comp['ic']:.2e}  bc={comp['bc']:.2e}"
                if self.zb_true is not None:
                    msg += f"  zb_rmse={self.history['zb_rmse'][-1]:.4e}"
                print(msg)

    def train_lbfgs(self, n_steps=500, print_every=100):
        print(f"Fine-tuning with L-BFGS for {n_steps} steps...")
        params = list(self.sol_net.parameters()) + list(self.bath_net.parameters())
        optimizer = torch.optim.LBFGS(params, lr=0.5, max_iter=20,
                                       history_size=50, line_search_fn="strong_wolfe")
        step_count = [0]

        def closure():
            optimizer.zero_grad()
            loss, comp = self.compute_loss()
            loss.backward()
            for k in ["total", "data", "pde", "ic", "bc", "dry", "pos", "tv"]:
                self.history[k].append(comp[k])
            if self.zb_true is not None:
                with torch.no_grad():
                    zb_1d = self.forward_1d().cpu().numpy().flatten()
                    rmse = np.sqrt(np.mean((zb_1d - self.zb_true)**2))
                    self.history["zb_rmse"].append(rmse)
            step_count[0] += 1
            if step_count[0] % print_every == 0:
                msg = f"  [LBFGS {step_count[0]:>4d}] total={comp['total']:.2e}  data={comp['data']:.2e}  pde={comp['pde']:.2e}"
                if self.zb_true is not None:
                    msg += f"  zb_rmse={self.history['zb_rmse'][-1]:.4e}"
                print(msg)
            return loss

        for _ in range(n_steps):
            optimizer.step(closure)

    def get_results(self):
        """Extract predictions on the full collocation grid."""
        with torch.no_grad():
            h, u, zb = self.forward()
            zb_1d = self.forward_1d()
        return {
            "h": h.cpu().numpy().flatten(),
            "u": u.cpu().numpy().flatten(),
            "zb_on_col": zb.cpu().numpy().flatten(),
            "zb_1d": zb_1d.cpu().numpy().flatten(),
        }


# ============================================================
# Plotting
# ============================================================

def plot_results(pinn, data, save_path=None):
    import matplotlib.pyplot as plt

    res = pinn.get_results()
    x = data["x"]
    t = data["t"]
    zb_true = data["zb"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Bathymetry
    ax = axes[0, 0]
    ax.plot(x, zb_true, "k-", linewidth=2, label="True $z_b$")
    ax.plot(x, res["zb_1d"], "r--", linewidth=2, label="Predicted $z_b$")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Bed elevation (m)")
    ax.set_title("Bathymetry Recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error
    ax = axes[0, 1]
    err = res["zb_1d"] - zb_true
    ax.plot(x, err, "r-")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("$z_b$ error (m)")
    rmse = float(np.sqrt(np.mean(err**2)))
    ax.set_title(f"Error (RMSE = {rmse*1000:.2f} mm)")
    ax.grid(True, alpha=0.3)

    # Solution at a snapshot
    ax = axes[1, 0]
    it = len(t) // 4
    Nx = len(x)
    h_pred_snap = res["h"].reshape(len(t), Nx)[it]
    h_true_snap = data["h"][it]
    ax.plot(x, h_true_snap, "k-", linewidth=2, label="True h")
    ax.plot(x, h_pred_snap, "b--", linewidth=2, label="Predicted h")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("h (m)")
    ax.set_title(f"Depth at t = {t[it]:.3f} s (t/T ≈ 0.25)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss curves
    ax = axes[1, 1]
    ax.semilogy(pinn.history["total"], label="total", alpha=0.8)
    ax.semilogy(pinn.history["data"], label="data", alpha=0.6)
    ax.semilogy(pinn.history["pde"], label="pde", alpha=0.6)
    ax.semilogy(pinn.history["ic"], label="ic", alpha=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Training History")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Thacker Inverse PINN — Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Main: baseline inversion
# ============================================================

if __name__ == "__main__":
    from ground_truth import generate_dataset

    FIG_DIR = Path(__file__).parent / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    # Ground truth — lower resolution for training speed
    data = generate_dataset(
        a=1.0, h_0=0.5,
        L=4.0, n_points_x=80,
        n_periods=1.0, n_points_t=40,
    )
    print(f"Domain: Nx={len(data['x'])}, Nt={len(data['t'])}")
    print(f"Total collocation points: {len(data['x']) * len(data['t'])}")
    print()

    # PINN — baseline uses eta + u observations (Exp 1 showed this breaks equifinality)
    pinn = ThackerInversePINN(
        x_grid=data["x"], t_grid=data["t"],
        eta_obs=data["eta"],
        u_obs=data["u"],      # include velocity to break equifinality
        t_obs_indices=np.arange(len(data["t"])),  # all times observed
        h_ic=data["h"][0], u_ic=data["u"][0],
        g=data["params"]["g"],
        # Network config
        sol_hidden=5, sol_neurons=96,
        bath_hidden=3, bath_neurons=48,
        fourier_features=16,
        fourier_sigma_x=2.0,
        fourier_sigma_t=2.0,
        # Loss weights
        lambda_data=10.0,
        lambda_data_u=10.0,
        lambda_pde=1.0,
        lambda_ic=100.0,
        lambda_bc=10.0,
        lambda_dry=10.0,
        lambda_pos=10.0,
        lambda_tv=1e-4,
    )
    pinn.zb_true = data["zb"]

    t0 = time.time()
    pinn.train_adam(n_epochs=10000, print_every=2000)
    pinn.train_lbfgs(n_steps=400, print_every=100)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Final evaluation
    res = pinn.get_results()
    err = res["zb_1d"] - data["zb"]
    # Only evaluate on the "ever-wet" region (where we have information)
    wet_ever = (data["h"] > 1e-4).any(axis=0)
    err_wet = err[wet_ever]

    rmse_all = float(np.sqrt(np.mean(err**2)))
    rmse_wet = float(np.sqrt(np.mean(err_wet**2)))

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  z_b RMSE (all domain):  {rmse_all*1000:.2f} mm")
    print(f"  z_b RMSE (ever-wet):    {rmse_wet*1000:.2f} mm  ({wet_ever.mean():.0%} of domain)")
    print(f"  z_b max error (wet):    {np.max(np.abs(err_wet))*1000:.2f} mm")

    plot_results(pinn, data, save_path=FIG_DIR / "baseline_inversion.png")
