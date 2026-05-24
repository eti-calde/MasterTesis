"""
Inverse PINN for Experiment 5: 3D Thacker paraboloid (axisymmetric basin).

Adapts the Experiment 4 architecture for a closed basin with no external tidal
forcing (dynamics are driven purely by the initial tilt / initial state):

    - Solution net: (x, y, t) -> (h, u, v)
    - Bathymetry net: (x, y) -> z_b, unconstrained sign
    - 2D SWE residuals (non-conservative form)
    - Strong IC loss (only external forcing is the initial condition)
    - Wet/dry handling: softplus(h), dry-cell velocity loss
    - No outer BC loss: the paraboloidal rim rises above mean water level, so
      the outer domain is always dry. PINN should learn this from data/physics.
    - Error evaluated only on the "ever-wet" disk (where observations carry info)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


from pinn_bath.legacy_blocks import (  # noqa: E402
    FourierFeatures,
    SolutionNet_Exp05 as SolutionNet,
    BathymetryNet_Exp05 as BathymetryNet,
)


def swe_residual_2d(x, y, t, h, u, v, zb, g=9.81, eps_wet=0.0):
    """2D transient SWE residuals via AD (primitive-conservative form).

    Tian 2025 weighting: ``R = A r`` with
    ``A = [[1,0,0], [u,h,0], [v,0,h]]``. In dry cells (``h ≈ 0``,
    ``u ≈ v ≈ 0``) the momentum residuals ``R_mx, R_my`` carry an
    explicit ``h`` factor and so collapse to ≈0 automatically.
    ``R_cont`` is NOT weighted by ``h`` — it equals the raw primitive
    continuity residual.

    Optional ``eps_wet > 0``: hard-mask every residual by ``(h > eps_wet)``
    to zero out dry cells explicitly. Use to empirically verify that the
    A-matrix weighting alone is sufficient (run with eps_wet=0 vs >0 and
    compare ``mean(R*²)`` contribution from dry cells). With softplus on
    ``h`` the network's "dry" h is never exactly 0, so a tiny residual
    from dry cells survives the A weighting; whether it matters depends
    on the dry-cell collocation density (~20 % under the wet-biased
    sampling already done in ``__init__``).
    """
    def grad(out, wrt):
        return torch.autograd.grad(out, wrt, grad_outputs=torch.ones_like(out),
                                   create_graph=True)[0]
    h_t, h_x, h_y = grad(h, t), grad(h, x), grad(h, y)
    u_t, u_x, u_y = grad(u, t), grad(u, x), grad(u, y)
    v_t, v_x, v_y = grad(v, t), grad(v, x), grad(v, y)
    zb_x, zb_y = grad(zb, x), grad(zb, y)
    r_cont = h_t + h_x * u + h * u_x + h_y * v + h * v_y
    r_mx = u_t + u * u_x + v * u_y + g * (h_x + zb_x)
    r_my = v_t + u * v_x + v * v_y + g * (h_y + zb_y)
    R_cont = r_cont
    R_mx = u * r_cont + h * r_mx
    R_my = v * r_cont + h * r_my
    if eps_wet > 0.0:
        wet = (h > eps_wet).to(h.dtype)
        R_cont = R_cont * wet
        R_mx = R_mx * wet
        R_my = R_my * wet
    return R_cont, R_mx, R_my


class Thacker3DInversePINN:
    def __init__(
        self,
        x_grid, y_grid, t_grid,
        eta_obs, u_obs=None, v_obs=None,
        h_ic=None, u_ic=None, v_ic=None,
        t_obs_indices=None,
        # training grid
        n_colloc=12000, n_ic=1500,
        # Physics
        g=9.81,
        # Network
        sol_hidden=5, sol_neurons=128,
        bath_hidden=4, bath_neurons=64,
        fourier_features=24, sigma_space=3.0, sigma_time=2.0,
        fourier_features_bath=24, sigma_bath=2.0,
        # Losses
        lambda_data=10.0, lambda_data_uv=5.0,
        lambda_pde=1.0, lambda_ic=500.0,
        lambda_dry=10.0,
        lambda_dry_zb=1000.0,  # direct z_b supervision on always-dry cells (rim anchor)
        lambda_zb_ic=0.0,      # direct z_b supervision at IC: zb = eta_obs(t=0) - h_ic
        lambda_tv=1e-5,
        dry_std_threshold=1e-5,  # eta std threshold to classify a cell as "always dry"
        device=None, seed=42,
    ):
        torch.manual_seed(seed); np.random.seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.g = g
        self.lw = {
            "data": lambda_data, "data_uv": lambda_data_uv,
            "pde": lambda_pde, "ic": lambda_ic,
            "dry": lambda_dry, "dry_zb": lambda_dry_zb,
            "zb_ic": lambda_zb_ic,
            "tv": lambda_tv,
        }
        self.dry_std_threshold = dry_std_threshold

        self.x_min, self.x_max = float(x_grid.min()), float(x_grid.max())
        self.y_min, self.y_max = float(y_grid.min()), float(y_grid.max())
        self.t_min, self.t_max = float(t_grid.min()), float(t_grid.max())

        X, Y = np.meshgrid(x_grid, y_grid)
        Ny, Nx = X.shape
        Nt = len(t_grid)

        X_full = np.broadcast_to(X[None], (Nt, Ny, Nx)).ravel()
        Y_full = np.broadcast_to(Y[None], (Nt, Ny, Nx)).ravel()
        T_full = np.broadcast_to(t_grid[:, None, None], (Nt, Ny, Nx)).ravel()

        # Wet-biased collocation: 80% in ever-wet ∪ shoreline (cells where η
        # varies in time → h>0 at some t), 20% in always-dry rim (where the
        # PDE residual is uninformative anyway since h=u=v=0). Without this
        # bias, ~75% of random collocation points land in always-dry cells
        # whose residuals carry no signal about the basin's bathymetry.
        N_total = Nt * Ny * Nx
        t_obs_for_mask = (np.asarray(t_obs_indices) if t_obs_indices is not None
                          else np.arange(Nt))
        eta_for_mask = np.asarray(eta_obs)[t_obs_for_mask]
        ever_wet_2d = eta_for_mask.std(axis=0) >= dry_std_threshold  # (Ny, Nx)
        ever_wet_full = np.broadcast_to(ever_wet_2d[None], (Nt, Ny, Nx)).ravel()
        wet_idx = np.where(ever_wet_full)[0]
        dry_idx = np.where(~ever_wet_full)[0]
        n_wet_col = min(int(0.8 * n_colloc), len(wet_idx))
        n_dry_col = min(n_colloc - n_wet_col, len(dry_idx))
        idx_col = np.concatenate([
            np.random.choice(wet_idx, size=n_wet_col, replace=False),
            np.random.choice(dry_idx, size=n_dry_col, replace=False),
        ])
        print(f"  [PINN] collocation: {n_wet_col} wet + {n_dry_col} dry "
              f"(target {n_colloc}, available wet={len(wet_idx)} dry={len(dry_idx)})")
        self.x_col = torch.tensor(X_full[idx_col], dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.y_col = torch.tensor(Y_full[idx_col], dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.t_col = torch.tensor(T_full[idx_col], dtype=torch.float32, device=self.device).reshape(-1, 1)
        for tens in (self.x_col, self.y_col, self.t_col):
            tens.requires_grad_(True)

        # Observations restricted to selected time indices
        if t_obs_indices is None:
            t_obs_indices = np.arange(Nt)
        obs_mask_t = np.zeros(Nt, dtype=bool); obs_mask_t[np.asarray(t_obs_indices)] = True
        obs_mask_full = np.broadcast_to(obs_mask_t[:, None, None], (Nt, Ny, Nx)).ravel()

        self.x_obs = torch.tensor(X_full[obs_mask_full], dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.y_obs = torch.tensor(Y_full[obs_mask_full], dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.t_obs = torch.tensor(T_full[obs_mask_full], dtype=torch.float32, device=self.device).reshape(-1, 1)
        eta_flat = np.asarray(eta_obs).ravel()
        self.eta_obs = torch.tensor(eta_flat[obs_mask_full], dtype=torch.float32,
                                     device=self.device).reshape(-1, 1)

        self.has_u = u_obs is not None
        self.has_v = v_obs is not None
        if self.has_u:
            self.u_obs = torch.tensor(np.asarray(u_obs).ravel()[obs_mask_full],
                                       dtype=torch.float32, device=self.device).reshape(-1, 1)
        if self.has_v:
            self.v_obs = torch.tensor(np.asarray(v_obs).ravel()[obs_mask_full],
                                       dtype=torch.float32, device=self.device).reshape(-1, 1)

        # IC at t=0
        if h_ic is not None:
            idx_ic = np.random.choice(Ny * Nx, size=min(n_ic, Ny * Nx), replace=False)
            self.x_ic = torch.tensor(X.ravel()[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.y_ic = torch.tensor(Y.ravel()[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.t_ic = torch.zeros_like(self.x_ic)
            self.h_ic = torch.tensor(h_ic.ravel()[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.u_ic = torch.tensor(u_ic.ravel()[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.v_ic = torch.tensor(v_ic.ravel()[idx_ic], dtype=torch.float32, device=self.device).reshape(-1, 1)
            # z_b target at IC: from η_obs(t=0) and known h_ic, since eta = h + z_b.
            # Provides direct dense supervision for the bath net inside the wet
            # disk (where always-dry rim supervision can't reach). Only meaningful
            # when h_ic is known at every IC point (it is, by construction).
            eta_t0 = np.asarray(eta_obs)[0].ravel()[idx_ic]
            zb_target_ic = eta_t0 - h_ic.ravel()[idx_ic]
            self.zb_target_ic = torch.tensor(zb_target_ic, dtype=torch.float32,
                                              device=self.device).reshape(-1, 1)
            self.has_ic = True
        else:
            self.has_ic = False

        # Eval grid (for extracting z_b after training)
        self.X_eval = torch.tensor(X.ravel(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.Y_eval = torch.tensor(Y.ravel(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.eval_shape = (Ny, Nx)

        # --- Always-dry cell detection: where eta is constant in time, h = 0 and eta = z_b ---
        # Only use time instants that we observe (so we don't peek at hidden test data)
        eta_obs_subset = np.asarray(eta_obs)[np.asarray(t_obs_indices)]  # (N_obs_t, Ny, Nx)
        eta_std = eta_obs_subset.std(axis=0)  # (Ny, Nx)
        eta_mean = eta_obs_subset.mean(axis=0)
        always_dry = eta_std < dry_std_threshold  # (Ny, Nx)
        print(f"  [PINN] always-dry cells: {always_dry.sum()}/{always_dry.size} "
              f"({100 * always_dry.mean():.1f}% of domain)")
        self.n_always_dry = int(always_dry.sum())

        if self.n_always_dry > 0:
            xy_dry = np.stack([X[always_dry], Y[always_dry]], axis=-1)
            zb_dry = eta_mean[always_dry]  # where h=0, eta = z_b
            self.x_dry = torch.tensor(xy_dry[:, 0], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.y_dry = torch.tensor(xy_dry[:, 1], dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.zb_dry = torch.tensor(zb_dry, dtype=torch.float32, device=self.device).reshape(-1, 1)
            self.has_dry_supervision = True
        else:
            self.has_dry_supervision = False

        # Networks
        self.sol_net = SolutionNet(
            sol_hidden, sol_neurons,
            fourier_features=fourier_features,
            sigma_space=sigma_space, sigma_time=sigma_time,
        ).to(self.device)
        self.bath_net = BathymetryNet(
            bath_hidden, bath_neurons,
            fourier_features=fourier_features_bath,
            sigma=sigma_bath,
        ).to(self.device)

        params = list(self.sol_net.parameters()) + list(self.bath_net.parameters())
        self.optimizer_adam = torch.optim.Adam(params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_adam, step_size=2000, gamma=0.5)

        self.history = {k: [] for k in ["total", "data", "pde", "ic", "dry", "dry_zb", "zb_ic", "tv", "zb_rmse", "zb_rmse_wet"]}
        self.zb_true = None
        self.wet_mask = None  # (Ny, Nx) boolean of ever-wet

    def _nx(self, x): return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
    def _ny(self, y): return 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
    def _nt(self, t):
        if self.t_max <= self.t_min: return t * 0
        return 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1

    def forward_col(self):
        h, u, v = self.sol_net(self._nx(self.x_col), self._ny(self.y_col), self._nt(self.t_col))
        zb = self.bath_net(self._nx(self.x_col), self._ny(self.y_col))
        return h, u, v, zb

    def forward_obs(self):
        h, u, v = self.sol_net(self._nx(self.x_obs), self._ny(self.y_obs), self._nt(self.t_obs))
        zb = self.bath_net(self._nx(self.x_obs), self._ny(self.y_obs))
        return h, u, v, zb

    def forward_ic(self):
        return self.sol_net(self._nx(self.x_ic), self._ny(self.y_ic), self._nt(self.t_ic))

    def forward_bath_2d(self):
        return self.bath_net(self._nx(self.X_eval), self._ny(self.Y_eval))

    def compute_loss(self):
        h_c, u_c, v_c, zb_c = self.forward_col()
        r_c, r_mx, r_my = swe_residual_2d(self.x_col, self.y_col, self.t_col,
                                           h_c, u_c, v_c, zb_c, self.g)
        L_pde = self.lw["pde"] * (torch.mean(r_c ** 2) + torch.mean(r_mx ** 2) + torch.mean(r_my ** 2))

        h_o, u_o, v_o, zb_o = self.forward_obs()
        eta_pred = h_o + zb_o
        L_data = self.lw["data"] * torch.mean((eta_pred - self.eta_obs) ** 2)
        if self.has_u:
            L_data = L_data + self.lw["data_uv"] * torch.mean((u_o - self.u_obs) ** 2)
        if self.has_v:
            L_data = L_data + self.lw["data_uv"] * torch.mean((v_o - self.v_obs) ** 2)

        if self.has_ic:
            h_i, u_i, v_i = self.forward_ic()
            L_ic = self.lw["ic"] * (torch.mean((h_i - self.h_ic) ** 2)
                                     + torch.mean((u_i - self.u_ic) ** 2)
                                     + torch.mean((v_i - self.v_ic) ** 2))
        else:
            L_ic = torch.tensor(0.0, device=self.device)

        # dry-cell velocity loss
        eps_dry = 1e-3
        dry_weight = torch.sigmoid(1.0 - h_c / eps_dry)
        L_dry = self.lw["dry"] * torch.mean((dry_weight * (u_c ** 2 + v_c ** 2)))

        # Direct z_b supervision on always-dry cells (where eta = z_b is directly observable)
        if self.has_dry_supervision:
            zb_pred_dry = self.bath_net(self._nx(self.x_dry), self._ny(self.y_dry))
            L_dry_zb = self.lw["dry_zb"] * torch.mean((zb_pred_dry - self.zb_dry) ** 2)
        else:
            L_dry_zb = torch.tensor(0.0, device=self.device)

        # Direct z_b supervision at IC points (covers the wet disk too):
        # at t=0, η_obs is observed and h_ic is known, so z_b = η_obs(0) - h_ic.
        if self.has_ic and self.lw["zb_ic"] > 0:
            zb_pred_ic = self.bath_net(self._nx(self.x_ic), self._ny(self.y_ic))
            L_zb_ic = self.lw["zb_ic"] * torch.mean((zb_pred_ic - self.zb_target_ic) ** 2)
        else:
            L_zb_ic = torch.tensor(0.0, device=self.device)

        # TV on z_b
        zb_2d = self.forward_bath_2d().reshape(*self.eval_shape)
        L_tv = self.lw["tv"] * ((zb_2d[:, 1:] - zb_2d[:, :-1]).abs().mean()
                                  + (zb_2d[1:] - zb_2d[:-1]).abs().mean())

        L_total = L_data + L_pde + L_ic + L_dry + L_dry_zb + L_zb_ic + L_tv
        return L_total, {
            "total": L_total.item(), "data": L_data.item(), "pde": L_pde.item(),
            "ic": L_ic.item(), "dry": L_dry.item(), "dry_zb": L_dry_zb.item(),
            "zb_ic": L_zb_ic.item(), "tv": L_tv.item(),
        }

    def train_adam(self, n_epochs=6000, print_every=1000):
        print(f"Training 3D Thacker PINN with Adam for {n_epochs} epochs...", flush=True)
        for ep in range(n_epochs):
            self.optimizer_adam.zero_grad()
            loss, comp = self.compute_loss()
            loss.backward()
            self.optimizer_adam.step()
            self.scheduler.step()
            for k in ["total", "data", "pde", "ic", "dry", "dry_zb", "zb_ic", "tv"]:
                self.history[k].append(comp[k])
            if self.zb_true is not None:
                with torch.no_grad():
                    zb_flat = self.forward_bath_2d().cpu().numpy().flatten()
                    true_flat = self.zb_true.ravel()
                    rmse = float(np.sqrt(np.mean((zb_flat - true_flat) ** 2)))
                    self.history["zb_rmse"].append(rmse)
                    if self.wet_mask is not None:
                        wet_flat = self.wet_mask.ravel()
                        rmse_w = float(np.sqrt(np.mean((zb_flat[wet_flat] - true_flat[wet_flat]) ** 2)))
                        self.history["zb_rmse_wet"].append(rmse_w)
            if (ep + 1) % print_every == 0:
                msg = (f"  [{ep+1:>5d}] total={comp['total']:.2e}  data={comp['data']:.2e}"
                       f"  pde={comp['pde']:.2e}  ic={comp['ic']:.2e}"
                       f"  dry_zb={comp['dry_zb']:.2e}  zb_ic={comp['zb_ic']:.2e}")
                if self.zb_true is not None:
                    msg += f"  zb_rmse_wet={self.history['zb_rmse_wet'][-1]:.4e}"
                print(msg, flush=True)

    def train_lbfgs(self, n_steps=100, print_every=25):
        print(f"L-BFGS fine-tune for {n_steps} steps...", flush=True)
        params = list(self.sol_net.parameters()) + list(self.bath_net.parameters())
        opt = torch.optim.LBFGS(params, lr=0.5, max_iter=20,
                                history_size=50, line_search_fn="strong_wolfe")
        cnt = [0]

        def closure():
            opt.zero_grad()
            loss, comp = self.compute_loss()
            loss.backward()
            for k in ["total", "data", "pde", "ic", "dry", "dry_zb", "zb_ic", "tv"]:
                self.history[k].append(comp[k])
            if self.zb_true is not None:
                with torch.no_grad():
                    zb_flat = self.forward_bath_2d().cpu().numpy().flatten()
                    true_flat = self.zb_true.ravel()
                    rmse = float(np.sqrt(np.mean((zb_flat - true_flat) ** 2)))
                    self.history["zb_rmse"].append(rmse)
                    if self.wet_mask is not None:
                        wet_flat = self.wet_mask.ravel()
                        rmse_w = float(np.sqrt(np.mean((zb_flat[wet_flat] - true_flat[wet_flat]) ** 2)))
                        self.history["zb_rmse_wet"].append(rmse_w)
            cnt[0] += 1
            if cnt[0] % print_every == 0:
                msg = (f"  [LBFGS {cnt[0]:>4d}] total={comp['total']:.2e}  data={comp['data']:.2e}"
                       f"  pde={comp['pde']:.2e}  dry_zb={comp['dry_zb']:.2e}"
                       f"  zb_ic={comp['zb_ic']:.2e}")
                if self.zb_true is not None:
                    msg += f"  zb_rmse_wet={self.history['zb_rmse_wet'][-1]:.4e}"
                print(msg, flush=True)
            return loss

        for _ in range(n_steps):
            opt.step(closure)

    def get_zb_2d(self):
        with torch.no_grad():
            return self.forward_bath_2d().cpu().numpy().flatten().reshape(*self.eval_shape)


def plot_results(pinn, data, save_path=None, title_suffix=""):
    import matplotlib.pyplot as plt
    zb_pred = pinn.get_zb_2d()
    zb_true = data["zb"]
    err = zb_pred - zb_true
    wet = (data["h"] > 1e-4).any(axis=0)
    rmse_all = float(np.sqrt(np.mean(err ** 2)))
    rmse_wet = float(np.sqrt(np.mean(err[wet] ** 2)))

    p = data["params"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].pcolormesh(data["x"], data["y"], zb_true, cmap="terrain",
                              vmin=-0.12, vmax=0.7, shading="auto")
    axes[0].set_title("True $z_b$"); axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    axes[0].add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                                  edgecolor="red", linewidth=1, linestyle="--"))
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(data["x"], data["y"], zb_pred, cmap="terrain",
                              vmin=-0.12, vmax=0.7, shading="auto")
    axes[1].set_title("Predicted $z_b$"); axes[1].set_aspect("equal")
    axes[1].set_xlabel("x (m)")
    axes[1].add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                                  edgecolor="red", linewidth=1, linestyle="--"))
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(data["x"], data["y"], err, cmap="RdBu_r",
                              vmin=-0.1, vmax=0.1, shading="auto")
    axes[2].set_title(f"Error (wet RMSE={rmse_wet*1000:.2f} mm, all={rmse_all*1000:.2f} mm)")
    axes[2].set_aspect("equal"); axes[2].set_xlabel("x (m)")
    axes[2].add_patch(plt.Circle((p["x_c"], p["y_c"]), p["a"], fill=False,
                                  edgecolor="black", linewidth=1, linestyle="--"))
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"3D Thacker inverse PINN {title_suffix}".strip(),
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    plt.close()
    return rmse_wet


# ============================================================
# Main: baseline (eta only, full time series)
# ============================================================

if __name__ == "__main__":
    from ground_truth import generate_dataset

    FIG_DIR = Path(__file__).parent / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    print("Generating 3D Thacker ground truth...", flush=True)
    data = generate_dataset(
        L=4.0, Nx=40, Ny=40,
        x_c=2.0, y_c=2.0, h_0=0.1, a=1.0, r_0=0.8,
        n_periods=3, n_save=30,
    )

    pinn = Thacker3DInversePINN(
        x_grid=data["x"], y_grid=data["y"], t_grid=data["t"],
        eta_obs=data["eta"],
        u_obs=None, v_obs=None,  # eta only (testing temporal-richness breakthrough)
        h_ic=data["h"][0], u_ic=data["u"][0], v_ic=data["v"][0],
        t_obs_indices=np.arange(len(data["t"])),
        n_colloc=12000, n_ic=1500,
        sol_hidden=5, sol_neurons=128,
        bath_hidden=4, bath_neurons=64,
        fourier_features=24, sigma_space=3.0, sigma_time=2.0,
        fourier_features_bath=32, sigma_bath=4.0,
        lambda_data=10.0, lambda_pde=1.0, lambda_ic=100.0,
        lambda_dry=10.0, lambda_dry_zb=100.0, lambda_tv=1e-5,
    )
    pinn.zb_true = data["zb"]
    pinn.wet_mask = (data["h"] > 1e-4).any(axis=0)

    t0 = time.time()
    pinn.train_adam(n_epochs=6000, print_every=1000)
    pinn.train_lbfgs(n_steps=120, print_every=30)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)

    rmse_wet = plot_results(pinn, data, save_path=FIG_DIR / "baseline_inversion.png",
                             title_suffix="— eta only (full time series)")
    print(f"Final z_b RMSE (ever-wet): {rmse_wet * 1000:.2f} mm", flush=True)
