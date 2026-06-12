"""Torch dataset + normalization for the inverse operator (F3).

Wraps the ``.npz`` splits from :mod:`pinn_bath.datasets.operator_dataset` /
:mod:`pinn_bath.datagen.builder` (same format). The dataset yields *raw*
physical fields (``eta``, ``u``, ``zb``); the :class:`Normalizer` (fit on the
train split only) maps them to/from the network's working scale. Raw fields
are kept around because the physics residual must be evaluated in physical
units.

Two loading modes (same batch dicts, interchangeable in the training loop):

- default: a torch ``DataLoader`` over the CPU-resident dataset, copying each
  batch to the compute device per step;
- ``cache_device="cuda"``: every split is moved to the device **once** and
  batches are sliced directly from device tensors. This removes the per-step
  host-to-device copy entirely (the v2 10k-case bank is ~7 GB float32, well
  inside a modern training GPU), which is a large win at these model sizes.
  Shuffling uses torch's global RNG in both modes, so ``set_seed`` keeps the
  batch order reproducible either way.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pinn_bath.datasets.operator_dataset import load_split


@dataclass
class Normalizer:
    """Standardizes (eta, u[, v]) inputs and zb targets to ~unit scale.

    ``v_std`` is fit only for 2D datasets (transverse velocity present);
    it stays None on 1D datasets and the input tensor then has 2 channels.
    """

    eta_mean: float
    eta_std: float
    u_std: float
    zb_mean: float
    zb_std: float
    v_std: float | None = None

    @classmethod
    def fit(
        cls, eta: np.ndarray, u: np.ndarray, zb: np.ndarray, v: np.ndarray | None = None
    ) -> Normalizer:
        return cls(
            eta_mean=float(eta.mean()),
            eta_std=float(eta.std() + 1e-8),
            # u (and v in 2D) is ~zero-mean across the dataset: the inflow side
            # is randomized per case, so transport directions cancel overall.
            u_std=float(u.std() + 1e-8),
            zb_mean=float(zb.mean()),
            zb_std=float(zb.std() + 1e-8),
            v_std=None if v is None else float(v.std() + 1e-8),
        )

    def input_tensor(
        self, eta: torch.Tensor, u: torch.Tensor, v: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Stack normalized fields into a channel tensor:
        ``(..., 2, Nt, Nx)`` in 1D, ``(..., 3, Nt, Ny, Nx)`` with ``v``."""
        chans = [(eta - self.eta_mean) / self.eta_std, u / self.u_std]
        if v is not None:
            if self.v_std is None:
                raise ValueError("v given but Normalizer was fit without v")
            chans.append(v / self.v_std)
        # Channel axis sits before the field block: (Nt, Nx) -> -3 in 1D,
        # (Nt, Ny, Nx) -> -4 in 2D (with or without leading batch dims).
        return torch.stack(chans, dim=-4 if v is not None else -3)

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
    """Yields raw (eta, u[, v], zb) fields + difficulty for one split.

    1D splits: ``eta``/``u`` are (N, Nt, Nx), ``zb`` (N, Nx). 2D splits
    additionally carry ``v`` and ``y``; fields are (N, Nt, Ny, Nx) and
    ``zb`` (N, Ny, Nx). ``v is None`` marks a 1D dataset.
    """

    def __init__(self, split_path: str | Path):
        d = load_split(split_path)
        self.eta = torch.from_numpy(d["eta"]).float()
        self.u = torch.from_numpy(d["u"]).float()
        self.v = torch.from_numpy(d["v"]).float() if "v" in d else None
        self.zb = torch.from_numpy(d["zb"]).float()
        self.score = torch.from_numpy(d["score"]).float()
        self.difficulty = torch.from_numpy(d["difficulty"]).long()
        self.x = torch.from_numpy(d["x"]).float()
        self.y = torch.from_numpy(d["y"]).float() if "y" in d else None
        self.t = torch.from_numpy(d["t"]).float()

    def __len__(self) -> int:
        return self.eta.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = {
            "eta": self.eta[i],
            "u": self.u[i],
            "zb": self.zb[i],
            "score": self.score[i],
            "difficulty": self.difficulty[i],
        }
        if self.v is not None:
            item["v"] = self.v[i]
        return item

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float | None:
        return None if self.y is None else float(self.y[1] - self.y[0])

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


class TensorBatchLoader:
    """Slices dict batches directly from (possibly device-resident) tensors.

    Drop-in replacement for the ``DataLoader`` path: same batch keys, same
    epoch semantics. Shuffling draws from torch's global (CPU) RNG, so seeded
    runs are reproducible; the permutation indexes device tensors directly
    (CPU index tensors are valid for CUDA advanced indexing).
    """

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = next(iter(tensors.values())).shape[0]

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        idx = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for s in range(0, self.n, self.batch_size):
            b = idx[s : s + self.batch_size]
            yield {k: v[b] for k, v in self.tensors.items()}


def _cache_split(ds: OperatorDataset, device: str) -> dict[str, torch.Tensor]:
    out = {
        "eta": ds.eta.to(device),
        "u": ds.u.to(device),
        "zb": ds.zb.to(device),
        "score": ds.score.to(device),
        "difficulty": ds.difficulty.to(device),
    }
    if ds.v is not None:
        out["v"] = ds.v.to(device)
    return out


def make_loaders(
    dataset_dir: str | Path,
    *,
    batch_size: int = 16,
    cache_device: str | None = None,
) -> dict[str, object]:
    """Build train/val/test loaders + a train-fit Normalizer + grid spacings.

    Parameters
    ----------
    cache_device : optional device string ("cuda", "cuda:0", "cpu")
        When set, every split is moved there once and batches are sliced from
        resident tensors (no per-step H2D copies). Requires the whole dataset
        to fit in device memory (~7 GB for the 10k-case v2 bank).
    """
    dataset_dir = Path(dataset_dir)
    train = OperatorDataset(dataset_dir / "train.npz")
    val = OperatorDataset(dataset_dir / "val.npz")
    test = OperatorDataset(dataset_dir / "test.npz")
    # Fit on CPU tensors before any device caching.
    norm = Normalizer.fit(
        train.eta.numpy(),
        train.u.numpy(),
        train.zb.numpy(),
        v=None if train.v is None else train.v.numpy(),
    )
    if cache_device is None:
        train_loader: object = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader: object = DataLoader(val, batch_size=batch_size, shuffle=False)
        test_loader: object = DataLoader(test, batch_size=batch_size, shuffle=False)
    else:
        train_loader = TensorBatchLoader(
            _cache_split(train, cache_device), batch_size=batch_size, shuffle=True
        )
        val_loader = TensorBatchLoader(
            _cache_split(val, cache_device), batch_size=batch_size, shuffle=False
        )
        test_loader = TensorBatchLoader(
            _cache_split(test, cache_device), batch_size=batch_size, shuffle=False
        )
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "normalizer": norm,
        "dx": train.dx,
        "dy": train.dy,  # None on 1D datasets
        "dt": train.dt,
        "nx": train.eta.shape[-1],
        "ny": train.eta.shape[-2] if train.v is not None else None,
        "nt": train.eta.shape[1],
        "dim": 2 if train.v is not None else 1,
        "cache_device": cache_device,
    }
