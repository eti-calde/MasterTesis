"""
AngelInversePINN — Exp. 6.

Subclass of `ThackerInversePINN` (02-thacker-basin-1d). Reuses the two-network
architecture, Fourier features, sparse-observation mask, training loops
(`train_adam`/`train_lbfgs`) and `get_results` verbatim.

Overrides only `compute_loss`:
  - physics uses `swe_residual_angel` (linear drag kappa*u/h instead of Manning)
  - NO initial-condition loss (window starts mid-experiment, no known IC)
  - NO wall-BC loss (the inlet has flow; outlet is open — not a closed basin)
  - NO dry-cell loss (H_rest 0.3 m > bump 0.2 m -> never dry)
  - keeps data (incl. inlet S1 + sparse S2/S3/S4), PDE, positivity, TV

The inlet boundary at x=1.5 m is implemented as a **soft Dirichlet**:
S1 is included in `x_obs_indices` (see data_angel.py), so the data MSE
loss penalises mismatch with the measured eta(t) at that node with the
same weight as the interior S2 observation. This is NOT a hard
constraint -- the network can deviate from S1 if other loss terms pull
it away. The adjoint method of Angel et al. enforces the inlet as a
hard BC through their forward solver, so the comparison is not
apples-to-apples; see REPORT.md for how this asymmetry feeds into the
diagnosis of the negative result.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "02-thacker-basin-1d"))
from pinn_inverse import ThackerInversePINN, loss_positivity, loss_tv  # noqa: E402

from physics_angel import swe_residual_angel  # noqa: E402


class AngelInversePINN(ThackerInversePINN):
    def __init__(self, *args, kappa=0.2,
                 lambda_zbbc=100.0, lambda_zbpos=100.0, **kwargs):
        # No closed-wall BC, no dry-cell, no IC for the flume window.
        kwargs.setdefault("lambda_bc", 0.0)
        kwargs.setdefault("lambda_dry", 0.0)
        kwargs.setdefault("lambda_ic", 0.0)
        kwargs.setdefault("n_manning", 0.0)
        super().__init__(*args, **kwargs)
        self.kappa = kappa
        # Physical priors that break the eta = h + z_b equifinality:
        #  - z_b = 0 at both flume ends (flat approach/downstream — exact for
        #    this flume; same idiom as Exp 1 loss_bc :240-254)
        #  - z_b >= 0 everywhere (solid flume floor; the bed analog of the
        #    h>=0 softplus already used for the solution net)
        self.lambda_zbbc = lambda_zbbc
        self.lambda_zbpos = lambda_zbpos

    def compute_loss(self):
        h, u, zb = self.forward()

        r_cont, r_mom = swe_residual_angel(
            self.x_col, self.t_col, h, u, zb,
            g=self.g, kappa=self.kappa, eps_dry=self.eps_dry,
        )

        # Data loss: eta = h + zb vs measured surface at the snapped sensor
        # nodes (S1 inlet Dirichlet + interior S2/S3/S4), masked by obs_mask.
        eta_pred = h + zb
        eta_obs_flat = self.eta_obs_full.reshape(-1, 1)
        L_data = self.lw["data"] * torch.mean(
            (eta_pred[self.obs_mask] - eta_obs_flat[self.obs_mask]) ** 2)
        if self.has_u:
            u_obs_flat = self.u_obs_full.reshape(-1, 1)
            L_data = L_data + self.lw["data_u"] * torch.mean(
                (u[self.obs_mask] - u_obs_flat[self.obs_mask]) ** 2)

        L_pde = self.lw["pde"] * (torch.mean(r_cont ** 2) + torch.mean(r_mom ** 2))
        L_pos = self.lw["pos"] * loss_positivity(h)

        zb_1d = self.forward_1d().flatten()
        L_tv = self.lw["tv"] * loss_tv(zb_1d, self.x_1d.flatten())

        # z_b physical priors (break eta = h + z_b equifinality)
        L_zbbc = self.lambda_zbbc * (zb_1d[0] ** 2 + zb_1d[-1] ** 2)
        L_zbpos = self.lambda_zbpos * torch.mean(
            torch.relu(-zb_1d) ** 2)

        L_total = L_data + L_pde + L_pos + L_tv + L_zbbc + L_zbpos

        # fold z_b priors into existing history keys so the inherited
        # train_adam/train_lbfgs logging keeps working unchanged
        return L_total, {
            "total": L_total.item(), "data": L_data.item(),
            "pde": L_pde.item(), "ic": 0.0,
            "bc": L_zbbc.item(), "dry": 0.0,
            "pos": L_pos.item() + L_zbpos.item(), "tv": L_tv.item(),
        }
