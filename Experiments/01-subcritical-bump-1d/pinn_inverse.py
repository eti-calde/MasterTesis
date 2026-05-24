"""
PINN for inverse bathymetry recovery from surface observations.

Given observations of water surface elevation eta (and optionally velocity u),
recover the unknown bed elevation z_b(x) by enforcing the 1D steady SWE.

Architecture:
    - Solution network: x -> (h, u)  [water depth and velocity]
    - Bathymetry network: x -> z_b   [separate small network]
    - SWE residual via automatic differentiation

Loss:
    L = λ_data * L_data + λ_pde * L_pde + λ_tv * L_tv + λ_tikh * L_tikh + λ_pos * L_pos

References:
    - Dazzi et al. (2024), WRR — augmented SWE formulation
    - Tian et al. (2025), WRR — primitive-conservative form
    - Ruppenthal & Kuzmin (2026) — TV + Tikhonov regularization
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


# ============================================================
# Network architectures
# ============================================================

from pinn_bath.legacy_blocks import (  # noqa: E402
    FourierFeatures,
    SolutionNet_Exp01 as SolutionNet,
    BathymetryNet_Exp01 as BathymetryNet,
)


# ============================================================
# SWE residual (1D steady, primitive-conservative form)
# ============================================================

def swe_residual_steady(x, h, u, zb, q_known, n_manning=0.0, g=9.81):
    """Compute 1D steady SWE residuals via AD.

    Continuity:  d(hu)/dx = 0  =>  hu = q (constant)
    Momentum:    u * du/dx + g * d(h + zb)/dx + friction = 0

    Using primitive-conservative weighting (Tian 2025):
        R = A * r, where A = [[1, 0], [u, h]]

    Parameters
    ----------
    x : torch.Tensor (N, 1), requires_grad=True
    h, u, zb : torch.Tensor (N, 1), from networks
    q_known : float, known discharge (for continuity check)
    n_manning : float, Manning coefficient (0 = frictionless)
    g : float, gravity

    Returns
    -------
    r_cont : continuity residual (primitive)
    r_mom : momentum residual (primitive)
    R_cont : continuity residual (primitive-conservative)
    R_mom : momentum residual (primitive-conservative)
    """
    # Derivatives via AD
    dh_dx = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h),
                                 create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]
    dzb_dx = torch.autograd.grad(zb, x, grad_outputs=torch.ones_like(zb),
                                  create_graph=True)[0]

    # Primitive residuals
    # Continuity: dh/dx * u + h * du/dx = 0
    r_cont = dh_dx * u + h * du_dx

    # Momentum: u * du/dx + g * (dh/dx + dzb/dx) + friction = 0
    r_mom = u * du_dx + g * (dh_dx + dzb_dx)

    # Add Manning friction if present.
    # Primitive momentum eq:  u_t + u·u_x + g·(h_x + zb_x) + Sf = 0
    # with  Sf = tau_b/(rho·h) = g·n²·u|u|/h^(4/3)  (Manning–Strickler).
    if n_manning > 0:
        Sf = g * n_manning**2 * u * torch.abs(u) / h**(4.0 / 3.0)
        r_mom = r_mom + Sf

    # Primitive-conservative weighting: R = A * r
    # A = [[1, 0], [u, h]]
    # R_cont = 1 * r_cont + 0 * r_mom = r_cont
    # R_mom  = u * r_cont + h * r_mom
    R_cont = r_cont
    R_mom = u * r_cont + h * r_mom

    return r_cont, r_mom, R_cont, R_mom


def swe_residual_conservative_steady(x, h, u, zb, q_known, n_manning=0.0, g=9.81):
    """Compute 1D steady SWE residuals in conservative form via AD.

    Conservative variables: (h, hu)
    Mass:      d(hu)/dx = 0
    Momentum:  d(hu² + ½gh²)/dx + g·h·dz_b/dx = 0

    Network still outputs (h, u) — conservative fluxes are composed from these.
    """
    hu = h * u
    flux_mom = h * u**2 + 0.5 * g * h**2

    dhu_dx = torch.autograd.grad(hu, x, grad_outputs=torch.ones_like(hu),
                                  create_graph=True)[0]
    dflux_dx = torch.autograd.grad(flux_mom, x, grad_outputs=torch.ones_like(flux_mom),
                                    create_graph=True)[0]
    dzb_dx = torch.autograd.grad(zb, x, grad_outputs=torch.ones_like(zb),
                                  create_graph=True)[0]

    R_mass = dhu_dx
    R_mom = dflux_dx + g * h * dzb_dx

    # Conservative momentum eq:  (hu)_t + (hu² + ½g h²)_x = -g·h·zb_x - g·h·Sf
    # where Sf is the primitive friction (g·n²·u|u|/h^(4/3)). Multiplying
    # through by h gives the source term used here:
    #   g·h·Sf = g·n²·u|u|·h / h^(4/3) = g·n²·u|u| / h^(1/3).
    # NOTE the exponent is 1/3 (not 7/3): the 7/3 form would arise if Sf
    # were expressed in (hu) units, since u|u| = (hu)|hu|/h².
    # tests/test_exp1_manning_equivalence.py verifies this is consistent
    # with the primitive-conservative weighting R = A·r computed in
    # swe_residual_steady above.
    if n_manning > 0:
        Sf = g * n_manning**2 * u * torch.abs(u) / h**(1.0 / 3.0)
        R_mom = R_mom + Sf

    return R_mass, R_mom


# ============================================================
# Loss functions
# ============================================================

def loss_data_eta(h, zb, eta_obs, obs_mask=None):
    """Data loss on water surface elevation eta = h + z_b."""
    eta_pred = h + zb
    if obs_mask is not None:
        return torch.mean((eta_pred[obs_mask] - eta_obs[obs_mask])**2)
    return torch.mean((eta_pred - eta_obs)**2)


def loss_data_velocity(u, u_obs, obs_mask=None):
    """Data loss on velocity."""
    if obs_mask is not None:
        return torch.mean((u[obs_mask] - u_obs[obs_mask])**2)
    return torch.mean((u - u_obs)**2)


def loss_pde(R_cont, R_mom):
    """Physics loss: mean squared SWE residuals (primitive-conservative)."""
    return torch.mean(R_cont**2) + torch.mean(R_mom**2)


def loss_tv(zb, x):
    """Total variation regularization on z_b (encourages piecewise smoothness)."""
    # Finite differences along sorted x
    dzb = zb[1:] - zb[:-1]
    dx = x[1:] - x[:-1]
    return torch.mean(torch.abs(dzb / dx))


def loss_tikhonov(zb):
    """Tikhonov regularization: penalize large z_b values (shrink toward zero)."""
    return torch.mean(zb**2)


def loss_positivity(h):
    """Penalize negative water depth (soft constraint backup)."""
    return torch.mean(nn.functional.relu(-h)**2)


def loss_discharge(h, u, q_known):
    """Enforce known discharge q = h*u everywhere."""
    return torch.mean((h * u - q_known)**2)


def loss_bc(h, u, zb, x, h_downstream, q_known, x_start, x_end):
    """Boundary condition losses.

    - z_b = 0 at both boundaries (flat bed at edges)
    - h = h_downstream at outlet
    - q = h*u at boundaries
    """
    tol = (x_end - x_start) / 500  # ~1 grid cell
    left = (x - x_start).abs() < tol
    right = (x_end - x).abs() < tol

    loss = torch.tensor(0.0, device=x.device)
    if left.any():
        loss = loss + torch.mean(zb[left]**2)  # z_b(x_start) = 0
        loss = loss + torch.mean((h[left] * u[left] - q_known)**2)
    if right.any():
        loss = loss + torch.mean(zb[right]**2)  # z_b(x_end) = 0
        loss = loss + torch.mean((h[right] - h_downstream)**2)  # h(x_end) = h_down
    return loss


def loss_flat_bed(zb, x, x_center=0.0, half_width=2.0):
    """Penalize z_b^2 across the entire known-flat region |x - x_center| > half_width.

    The bump has compact support |x - x_center| <= half_width by definition
    of the case; outside that region the bed is known to be flat (z_b = 0).
    Without this term, ``loss_bc`` only pins z_b at the two domain endpoints,
    so the network is free to drift to a constant offset across the flat
    regions [x_start, x_center - half_width] and [x_center + half_width, x_end].
    """
    flat = (x - x_center).abs() > half_width
    if not flat.any():
        return torch.tensor(0.0, device=x.device)
    return torch.mean(zb[flat] ** 2)


# ============================================================
# Full inverse PINN
# ============================================================

class InverseBathymetryPINN:
    """Orchestrates the inverse bathymetry recovery."""

    def __init__(
        self,
        x_domain,
        eta_obs,
        u_obs=None,
        obs_indices=None,
        q_known=4.42,
        h_downstream=2.0,
        n_manning=0.0,
        g=9.81,
        # Network config
        sol_hidden=4, sol_neurons=64,
        bath_hidden=3, bath_neurons=32,
        use_fourier=True,
        fourier_features=16,
        fourier_sigma=1.0,
        # Loss weights
        lambda_data_eta=1.0,
        lambda_data_u=1.0,
        lambda_pde=1.0,
        lambda_q=10.0,
        lambda_bc=100.0,
        lambda_flat=100.0,
        lambda_tv=1e-4,
        lambda_tikh=1e-5,
        lambda_pos=10.0,
        # Bump support — flat_bed prior penalizes z_b for |x - bump_x0| > bump_half_width.
        bump_x0=0.0,
        bump_half_width=2.0,
        # SWE residual form. Default "primitive": ablation (May 2026,
        # see REPORT-ABLATION.md) shows it converges 2/2 seeds with
        # mean RMSE 4.05±0.04 mm, while "primitive_conservative" only
        # converges 1/2 (17.99±13.97 mm). This reverses Tian 2025's
        # forward-problem ranking and is the production default.
        swe_form="primitive",
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.g = g
        self.q_known = q_known
        self.n_manning = n_manning
        self.swe_form = swe_form

        # Domain info for BCs
        self.x_start = float(x_domain[0])
        self.x_end = float(x_domain[-1])
        self.h_downstream = h_downstream

        # Loss weights
        self.lambda_data_eta = lambda_data_eta
        self.lambda_data_u = lambda_data_u
        self.lambda_pde = lambda_pde
        self.lambda_q = lambda_q
        self.lambda_bc = lambda_bc
        self.lambda_flat = lambda_flat
        self.bump_x0 = float(bump_x0)
        self.bump_half_width = float(bump_half_width)
        self.lambda_tv = lambda_tv
        self.lambda_tikh = lambda_tikh
        self.lambda_pos = lambda_pos

        # Normalize x to [-1, 1] for Fourier features
        self.x_min = float(x_domain.min())
        self.x_max = float(x_domain.max())
        self.x_scale = 2.0 / (self.x_max - self.x_min)
        self.x_offset = -1.0 - self.x_min * self.x_scale

        # Networks
        self.sol_net = SolutionNet(
            sol_hidden, sol_neurons,
            use_fourier=use_fourier,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
        ).to(self.device)
        self.bath_net = BathymetryNet(
            bath_hidden, bath_neurons,
            use_fourier=use_fourier,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
        ).to(self.device)

        # Data
        self.x = torch.tensor(x_domain, dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.x.requires_grad_(True)

        self.eta_obs = torch.tensor(eta_obs, dtype=torch.float32, device=self.device).reshape(-1, 1)

        self.has_velocity_obs = u_obs is not None
        if self.has_velocity_obs:
            self.u_obs = torch.tensor(u_obs, dtype=torch.float32, device=self.device).reshape(-1, 1)

        # Observation mask (which points have observations)
        if obs_indices is not None:
            self.obs_mask = torch.zeros(len(x_domain), dtype=torch.bool, device=self.device)
            self.obs_mask[obs_indices] = True
        else:
            self.obs_mask = None  # all points observed

        # Optimizers
        all_params = list(self.sol_net.parameters()) + list(self.bath_net.parameters())
        self.optimizer_adam = torch.optim.Adam(all_params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_adam, step_size=2000, gamma=0.5)

        # History
        self.history = {
            "loss_total": [], "loss_data": [], "loss_pde": [],
            "loss_q": [], "loss_bc": [], "loss_flat": [],
            "loss_tv": [], "loss_tikh": [], "loss_pos": [],
            "zb_rmse": [],
        }
        self.zb_true = None  # set externally for tracking

    def forward(self):
        """Run both networks. x is normalized to [-1, 1] before network input."""
        x_norm = self.x * self.x_scale + self.x_offset
        h, u = self.sol_net(x_norm)
        zb = self.bath_net(x_norm)
        return h, u, zb

    def compute_loss(self):
        """Compute all loss components."""
        h, u, zb = self.forward()

        # SWE residual — branch on equation form
        if self.swe_form == "conservative":
            R_cont, R_mom = swe_residual_conservative_steady(
                self.x, h, u, zb, self.q_known, self.n_manning, self.g
            )
        else:
            r_cont, r_mom, Rc_cont, Rc_mom = swe_residual_steady(
                self.x, h, u, zb, self.q_known, self.n_manning, self.g
            )
            if self.swe_form == "primitive":
                R_cont, R_mom = r_cont, r_mom
            else:  # primitive_conservative (default)
                R_cont, R_mom = Rc_cont, Rc_mom

        # Losses
        L_data = self.lambda_data_eta * loss_data_eta(h, zb, self.eta_obs, self.obs_mask)
        if self.has_velocity_obs:
            L_data += self.lambda_data_u * loss_data_velocity(u, self.u_obs, self.obs_mask)

        L_pde = self.lambda_pde * loss_pde(R_cont, R_mom)
        L_q = self.lambda_q * loss_discharge(h, u, self.q_known)
        L_bc = self.lambda_bc * loss_bc(
            h, u, zb, self.x, self.h_downstream, self.q_known,
            self.x_start, self.x_end
        )
        L_flat = self.lambda_flat * loss_flat_bed(
            zb, self.x, self.bump_x0, self.bump_half_width
        )
        L_tv = self.lambda_tv * loss_tv(zb, self.x)
        L_tikh = self.lambda_tikh * loss_tikhonov(zb)
        L_pos = self.lambda_pos * loss_positivity(h)

        L_total = L_data + L_pde + L_q + L_bc + L_flat + L_tv + L_tikh + L_pos

        return L_total, {
            "data": L_data.item(), "pde": L_pde.item(),
            "q": L_q.item(), "bc": L_bc.item(), "flat": L_flat.item(),
            "tv": L_tv.item(), "tikh": L_tikh.item(), "pos": L_pos.item(),
            "total": L_total.item(),
        }

    def train_adam(self, n_epochs=10000, print_every=1000):
        """Train with Adam optimizer."""
        print(f"Training with Adam for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            self.optimizer_adam.zero_grad()
            loss, components = self.compute_loss()
            loss.backward()
            self.optimizer_adam.step()
            self.scheduler.step()

            # Record history
            for key in ["total", "data", "pde", "q", "bc", "flat", "tv", "tikh", "pos"]:
                self.history[f"loss_{key}"].append(components[key])

            if self.zb_true is not None:
                with torch.no_grad():
                    x_norm = self.x * self.x_scale + self.x_offset
                    zb_pred = self.bath_net(x_norm).cpu().numpy().flatten()
                    rmse = np.sqrt(np.mean((zb_pred - self.zb_true)**2))
                    self.history["zb_rmse"].append(rmse)

            if (epoch + 1) % print_every == 0:
                lr = self.optimizer_adam.param_groups[0]["lr"]
                msg = f"  [{epoch+1:>6d}] total={components['total']:.2e}  data={components['data']:.2e}  pde={components['pde']:.2e}  bc={components['bc']:.2e}"
                if self.zb_true is not None:
                    msg += f"  zb_rmse={self.history['zb_rmse'][-1]:.4e}"
                msg += f"  lr={lr:.1e}"
                print(msg)

    def train_lbfgs(self, n_steps=500, print_every=100):
        """Fine-tune with L-BFGS optimizer."""
        print(f"Fine-tuning with L-BFGS for {n_steps} steps...")
        all_params = list(self.sol_net.parameters()) + list(self.bath_net.parameters())
        optimizer = torch.optim.LBFGS(all_params, lr=0.5, max_iter=20,
                                       history_size=50, line_search_fn="strong_wolfe")
        step_count = [0]

        def closure():
            optimizer.zero_grad()
            loss, components = self.compute_loss()
            loss.backward()

            for key in ["total", "data", "pde", "q", "bc", "flat", "tv", "tikh", "pos"]:
                self.history[f"loss_{key}"].append(components[key])
            if self.zb_true is not None:
                with torch.no_grad():
                    x_norm = self.x * self.x_scale + self.x_offset
                    zb_pred = self.bath_net(x_norm).cpu().numpy().flatten()
                    rmse = np.sqrt(np.mean((zb_pred - self.zb_true)**2))
                    self.history["zb_rmse"].append(rmse)

            step_count[0] += 1
            if step_count[0] % print_every == 0:
                msg = f"  [LBFGS {step_count[0]:>4d}] total={components['total']:.2e}  data={components['data']:.2e}  pde={components['pde']:.2e}"
                if self.zb_true is not None:
                    msg += f"  zb_rmse={self.history['zb_rmse'][-1]:.4e}"
                print(msg)
            return loss

        for _ in range(n_steps):
            optimizer.step(closure)

    def get_results(self):
        """Extract final predictions as numpy arrays."""
        with torch.no_grad():
            h, u, zb = self.forward()
            x_np = self.x.detach().cpu().numpy().flatten()
            h_np = h.cpu().numpy().flatten()
            u_np = u.cpu().numpy().flatten()
            zb_np = zb.cpu().numpy().flatten()
            eta_np = (h + zb).cpu().numpy().flatten()
        return {"x": x_np, "h": h_np, "u": u_np, "zb": zb_np, "eta": eta_np}


# ============================================================
# Visualization
# ============================================================

def plot_results(pinn, zb_true, save_path=None):
    """Plot inversion results vs ground truth."""
    import matplotlib.pyplot as plt

    results = pinn.get_results()
    x = results["x"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Bathymetry recovery
    ax = axes[0, 0]
    ax.plot(x, zb_true, "k-", linewidth=2, label="True $z_b$")
    ax.plot(x, results["zb"], "r--", linewidth=2, label="Predicted $z_b$")
    ax.set_ylabel("Bed elevation (m)")
    ax.set_title("Bathymetry Recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Water surface
    ax = axes[0, 1]
    eta_true = pinn.eta_obs.cpu().numpy().flatten()
    ax.plot(x, eta_true, "k-", linewidth=2, label="True $\\eta$")
    ax.plot(x, results["eta"], "b--", linewidth=2, label="Predicted $\\eta$")
    ax.set_ylabel("Surface elevation (m)")
    ax.set_title("Water Surface Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bathymetry error
    ax = axes[1, 0]
    error = results["zb"] - zb_true
    ax.plot(x, error, "r-", linewidth=1.5)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("$z_b$ error (m)")
    ax.set_xlabel("x (m)")
    ax.set_title(f"Bathymetry Error (RMSE={np.sqrt(np.mean(error**2)):.4e} m)")
    ax.grid(True, alpha=0.3)

    # Loss history
    ax = axes[1, 1]
    ax.semilogy(pinn.history["loss_total"], label="Total", alpha=0.7)
    ax.semilogy(pinn.history["loss_data"], label="Data", alpha=0.7)
    ax.semilogy(pinn.history["loss_pde"], label="PDE", alpha=0.7)
    if pinn.history["zb_rmse"]:
        ax2 = ax.twinx()
        ax2.semilogy(pinn.history["zb_rmse"], "r-", alpha=0.5, label="$z_b$ RMSE")
        ax2.set_ylabel("$z_b$ RMSE (m)", color="r")
        ax2.legend(loc="center right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Inverse Bathymetry PINN — Results", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.close()


# ============================================================
# Main: baseline inversion on Dazzi B1
# ============================================================

if __name__ == "__main__":
    from ground_truth import generate_dataset

    FIG_DIR = Path(__file__).parent / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    # Generate ground truth (Dazzi B1)
    data = generate_dataset(
        L=20.0, n_points=500, x_start=-10.0,
        bump_type="parabolic",
        bump_params={"x0": 0.0, "height": 0.2, "half_width": 2.0},
        q=4.42, h_downstream=2.0, n_manning=0.0,
    )

    print("Ground truth loaded:")
    print(f"  x: [{data['x'].min():.1f}, {data['x'].max():.1f}], {len(data['x'])} points")
    print(f"  zb max: {data['zb'].max():.4f} m")
    print(f"  eta depression: {data['eta'].max() - data['eta'].min():.4f} m")
    print()

    # Create PINN
    pinn = InverseBathymetryPINN(
        x_domain=data["x"],
        eta_obs=data["eta"],
        u_obs=None,           # eta only (harder case)
        obs_indices=None,     # all points observed (baseline)
        q_known=4.42,
        h_downstream=2.0,
        n_manning=0.0,
        # Network
        sol_hidden=4, sol_neurons=64,
        bath_hidden=3, bath_neurons=32,
        # Loss weights
        lambda_data_eta=10.0,
        lambda_pde=1.0,
        lambda_q=10.0,        # enforce hu = q everywhere
        lambda_bc=100.0,      # enforce z_b=0 at boundaries, h=h_down at outlet
        lambda_flat=100.0,    # enforce z_b=0 across the entire flat region |x-x_0|>w
        lambda_tv=1e-4,
        lambda_tikh=1e-5,
        lambda_pos=10.0,
    )
    pinn.zb_true = data["zb"]  # for RMSE tracking

    # Train
    pinn.train_adam(n_epochs=10000, print_every=2000)
    pinn.train_lbfgs(n_steps=500, print_every=100)

    # Results
    results = pinn.get_results()
    zb_error = results["zb"] - data["zb"]
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  zb RMSE:     {np.sqrt(np.mean(zb_error**2)):.4e} m")
    print(f"  zb max err:  {np.max(np.abs(zb_error)):.4e} m")
    print(f"  zb R²:       {1 - np.sum(zb_error**2) / np.sum((data['zb'] - data['zb'].mean())**2):.6f}")
    print(f"  eta RMSE:    {np.sqrt(np.mean((results['eta'] - data['eta'])**2)):.4e} m")

    # Plot
    plot_results(pinn, data["zb"], save_path=FIG_DIR / "baseline_inversion.png")
