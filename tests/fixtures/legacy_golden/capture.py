"""Capture golden state_dicts for the legacy pinn_inverse SolutionNet /
BathymetryNet classes.

Used by ``tests/test_legacy_dedup.py`` to verify that the dedup refactor
(extracting these classes into ``pinn_bath.legacy_blocks``) preserves
byte-exact initialization given a fixed seed.

Workflow
--------
1. Run this script ONCE before the refactor:

       .venv/bin/python tests/fixtures/legacy_golden/capture.py

   It writes ``exp{01,02,03,05}.pt`` next to itself.

2. Do the refactor.

3. Run ``pytest tests/test_legacy_dedup.py -v`` — it imports
   ``get_state_dicts`` from this module and compares the
   post-refactor state_dicts against the goldens.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
FIX = Path(__file__).resolve().parent

# Each entry: (exp_dir_relative_to_repo, sol_kwargs, bath_kwargs,
#              (sol_class_name, bath_class_name))
# Hyperparameters mirror the legacy defaults inside each pinn_inverse.py.
EXPS: dict[str, tuple[Path, dict, dict, tuple[str, str]]] = {
    "exp01": (
        REPO / "Experiments" / "01-subcritical-bump-1d",
        dict(n_hidden=4, n_neurons=64, use_fourier=True, fourier_features=16, fourier_sigma=1.0),
        dict(n_hidden=3, n_neurons=32, use_fourier=True, fourier_features=16, fourier_sigma=1.0),
        ("SolutionNet", "BathymetryNet"),
    ),
    "exp02": (
        REPO / "Experiments" / "02-thacker-basin-1d",
        dict(
            n_hidden=5, n_neurons=96, fourier_features=24, fourier_sigma_x=2.0, fourier_sigma_t=2.0
        ),
        dict(n_hidden=3, n_neurons=48, fourier_features=16, fourier_sigma=2.0),
        ("SolutionNet", "BathymetryNet"),
    ),
    "exp03": (
        REPO / "Experiments" / "03-two-cylinders-2d",
        dict(n_hidden=5, n_neurons=128, fourier_features=24, sigma_space=2.0, sigma_time=2.0),
        dict(n_hidden=4, n_neurons=64, fourier_features=32, sigma=3.0),
        ("SolutionNet2D", "BathymetryNet2D"),
    ),
    "exp05": (
        REPO / "Experiments" / "05-thacker-paraboloid-3d",
        dict(n_hidden=5, n_neurons=128, fourier_features=24, sigma_space=3.0, sigma_time=2.0),
        dict(n_hidden=4, n_neurons=64, fourier_features=32, sigma=4.0),
        ("SolutionNet", "BathymetryNet"),
    ),
}


def get_state_dicts(
    exp_dir: Path,
    sol_kwargs: dict,
    bath_kwargs: dict,
    cls_names: tuple[str, str],
    seed: int = 0,
) -> tuple[dict, dict]:
    """Instantiate the exp's SolutionNet + BathymetryNet under a fixed seed
    and return cloned state_dicts. Isolates the import (each exp ships its
    own ``pinn_inverse`` module, so we clear sys.modules between calls)."""
    sys.path.insert(0, str(exp_dir))
    sys.modules.pop("pinn_inverse", None)
    sys.modules.pop("ground_truth", None)
    try:
        pi = importlib.import_module("pinn_inverse")
        SolCls = getattr(pi, cls_names[0])
        BatCls = getattr(pi, cls_names[1])
        torch.manual_seed(seed)
        sol = SolCls(**sol_kwargs)
        bath = BatCls(**bath_kwargs)
        sol_sd = {k: v.detach().clone() for k, v in sol.state_dict().items()}
        bath_sd = {k: v.detach().clone() for k, v in bath.state_dict().items()}
    finally:
        sys.path.pop(0)
        sys.modules.pop("pinn_inverse", None)
        sys.modules.pop("ground_truth", None)
    return sol_sd, bath_sd


if __name__ == "__main__":
    for label, (exp_dir, sol_k, bath_k, cls_names) in EXPS.items():
        sol_sd, bath_sd = get_state_dicts(exp_dir, sol_k, bath_k, cls_names)
        out = FIX / f"{label}.pt"
        torch.save({"sol": sol_sd, "bath": bath_sd}, out)
        print(f"{label}: saved {len(sol_sd)} sol + {len(bath_sd)} bath tensors → {out.name}")
