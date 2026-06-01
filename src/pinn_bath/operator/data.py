"""Torch dataset + normalization for the inverse operator (F3).

Wraps the ``.npz`` splits from :mod:`pinn_bath.datasets.operator_dataset`.
The dataset yields *raw* physical fields (``eta``, ``u``, ``zb``); the
:class:`Normalizer` (fit on the train split only) maps them to/from the
network's working scale. Raw fields are kept around because the physics
residual must be evaluated in physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pinn_bath.datasets.operator_dataset import load_split


@dataclass
class Normalizer:
    """Standardizes (eta, u) inputs and zb targets to ~unit scale."""

    eta_mean: float
    eta_std: float
    u_std: float
    zb_mean: float
    zb_std: float

    @classmethod
    def fit(cls, eta: np.ndarray, u: np.ndarray, zb: np.ndarray) -> Normalizer:
        return cls(
            eta_mean=float(eta.mean()),
            eta_std=float(eta.std() + 1e-8),
            u_std=float(u.std() + 1e-8),  # u is ~zero-mean (sloshing)
            zb_mean=float(zb.mean()),
            zb_std=float(zb.std() + 1e-8),
        )

    def input_tensor(self, eta: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Stack normalized (eta, u) into a (..., 2, Nt, Nx) channel tensor."""
        en = (eta - self.eta_mean) / self.eta_std
        un = u / self.u_std
        return torch.stack([en, un], dim=-3)

    def norm_zb(self, zb: torch.Tensor) -> torch.Tensor:
        return (zb - self.zb_mean) / self.zb_std

    def denorm_zb(self, zb_n: torch.Tensor) -> torch.Tensor:
        return zb_n * self.zb_std + self.zb_mean

    def as_dict(self) -> dict[str, float]:
        return {
            "eta_mean": self.eta_mean,
            "eta_std": self.eta_std,
            "u_std": self.u_std,
            "zb_mean": self.zb_mean,
            "zb_std": self.zb_std,
        }


class OperatorDataset(Dataset):
    """Yields raw (eta, u, zb) fields + difficulty for one split."""

    def __init__(self, split_path: str | Path):
        d = load_split(split_path)
        self.eta = torch.from_numpy(d["eta"]).float()  # (N, Nt, Nx)
        self.u = torch.from_numpy(d["u"]).float()
        self.zb = torch.from_numpy(d["zb"]).float()  # (N, Nx)
        self.score = torch.from_numpy(d["score"]).float()
        self.difficulty = torch.from_numpy(d["difficulty"]).long()
        self.x = torch.from_numpy(d["x"]).float()
        self.t = torch.from_numpy(d["t"]).float()

    def __len__(self) -> int:
        return self.eta.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "eta": self.eta[i],
            "u": self.u[i],
            "zb": self.zb[i],
            "score": self.score[i],
            "difficulty": self.difficulty[i],
        }

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def make_loaders(
    dataset_dir: str | Path,
    *,
    batch_size: int = 16,
) -> dict[str, object]:
    """Build train/val/test loaders + a train-fit Normalizer + grid spacings."""
    dataset_dir = Path(dataset_dir)
    train = OperatorDataset(dataset_dir / "train.npz")
    val = OperatorDataset(dataset_dir / "val.npz")
    test = OperatorDataset(dataset_dir / "test.npz")
    norm = Normalizer.fit(train.eta.numpy(), train.u.numpy(), train.zb.numpy())
    return {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test, batch_size=batch_size, shuffle=False),
        "normalizer": norm,
        "dx": train.dx,
        "dt": train.dt,
        "nx": train.eta.shape[-1],
        "nt": train.eta.shape[-2],
    }
