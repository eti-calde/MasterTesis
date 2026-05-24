"""Verify that the primitive-conservative weighting of the Exp 1 SWE
residual is mathematically equivalent to the explicit conservative form,
including Manning friction.

For any smooth ``(h, u, zb)`` the A-matrix weighting

    R_mom = u · r_cont + h · r_mom

(where ``r_cont, r_mom`` are the primitive residuals) must equal the
pure conservative momentum residual

    R_mom_conservative = d(hu² + ½gh²)/dx + g·h·∂zb + g·n²·u|u|/h^{1/3}.

This pins the friction-term exponent (``1/3``, not ``7/3``) and prevents
future "corrections" based on misreading the conservative form.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_EXP1_DIR = Path(__file__).resolve().parents[1] / "Experiments" / "01-subcritical-bump-1d"
sys.path.insert(0, str(_EXP1_DIR))

import pinn_inverse as pi  # noqa: E402


@pytest.mark.fast
@pytest.mark.parametrize("n_manning", [0.0, 0.025, 0.05])
def test_primitive_conservative_equivalence(n_manning: float) -> None:
    """A·r weighting matches the pure conservative momentum residual."""
    torch.manual_seed(0)
    N = 200
    # Smooth (h, u, zb) fields with positive h
    x = torch.linspace(-5.0, 5.0, N, dtype=torch.float64).reshape(-1, 1)
    x.requires_grad_(True)
    h = 1.0 + 0.3 * torch.sin(0.7 * x) + 0.2 * torch.cos(1.3 * x) + 1.0
    u = 0.5 + 0.1 * torch.cos(0.4 * x)
    zb = 0.2 * torch.exp(-(x**2) / 4.0)

    # Primitive + A-weighted (primitive-conservative)
    _, _, R_cont_pc, R_mom_pc = pi.swe_residual_steady(
        x, h, u, zb, q_known=1.0, n_manning=n_manning, g=9.81
    )
    # Pure conservative
    R_mass_c, R_mom_c = pi.swe_residual_conservative_steady(
        x, h, u, zb, q_known=1.0, n_manning=n_manning, g=9.81
    )

    # Continuity: A-weighted R_cont equals primitive r_cont; pure
    # conservative R_mass = d(hu)/dx. By product rule these are equal
    # (chain: d(hu)/dx = h·du/dx + u·dh/dx = r_cont).
    torch.testing.assert_close(R_cont_pc, R_mass_c, atol=1e-10, rtol=1e-10)
    # Momentum: the entire point of the test.
    torch.testing.assert_close(R_mom_pc, R_mom_c, atol=1e-9, rtol=1e-9)
