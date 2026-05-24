"""
SWE residual for Exp. 6 — adapted from
`02-thacker-basin-1d/pinn_inverse.py:swe_residual_transient` (:99-130).

Only change vs. the Thacker residual: the friction term. Angel's flume has a
LINEAR bottom drag  kappa * u / h  (kappa = 0.2), not a Manning law. Everything
else (continuity, non-conservative momentum, AD derivatives) is identical.

  continuity:  dh/dt + u*dh/dx + h*du/dx = 0
  momentum:    du/dt + u*du/dx + g*(dh/dx + dz_b/dx) + kappa*u/(h+eps) = 0
"""

import torch


def swe_residual_angel(x, t, h, u, zb, g=9.81, kappa=0.2, eps_dry=1e-4):
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

    if kappa > 0:
        r_mom = r_mom + kappa * u / (h + eps_dry)

    return r_cont, r_mom
