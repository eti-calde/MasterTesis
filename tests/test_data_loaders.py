"""Tests for the operator data loaders (DataLoader vs device-cached path)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinn_bath.operator.data import make_loaders


def _write_split(path, n, nt=6, nx=8, seed=0):
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        zb=rng.normal(size=(n, nx)).astype(np.float32),
        eta=rng.normal(size=(n, nt, nx)).astype(np.float32),
        u=rng.normal(size=(n, nt, nx)).astype(np.float32),
        score=rng.uniform(size=n).astype(np.float32),
        difficulty=rng.integers(0, 3, size=n).astype(np.int8),
        seed=np.arange(n, dtype=np.int64),
        x=np.linspace(0.5, 7.5, nx, dtype=np.float32),
        t=np.linspace(0.0, 5.0, nt, dtype=np.float32),
    )


@pytest.fixture
def dataset_dir(tmp_path):
    _write_split(tmp_path / "train.npz", 10, seed=1)
    _write_split(tmp_path / "val.npz", 5, seed=2)
    _write_split(tmp_path / "test.npz", 7, seed=3)
    return tmp_path


@pytest.mark.fast
def test_cached_loader_matches_dataloader(dataset_dir) -> None:
    """cpu-cached TensorBatchLoader yields the same data as the DataLoader."""
    plain = make_loaders(dataset_dir, batch_size=4)
    cached = make_loaders(dataset_dir, batch_size=4, cache_device="cpu")
    assert plain["normalizer"].as_dict() == cached["normalizer"].as_dict()
    assert (plain["dx"], plain["dt"]) == (cached["dx"], cached["dt"])
    for split in ("val", "test"):  # shuffle=False -> order comparable
        for bp, bc in zip(plain[split], cached[split], strict=True):
            for k in ("eta", "u", "zb", "score", "difficulty"):
                assert torch.equal(bp[k], bc[k]), (split, k)


@pytest.mark.fast
def test_cached_loader_epoch_covers_all_cases(dataset_dir) -> None:
    """Shuffled cached loader yields every case exactly once per epoch."""
    cached = make_loaders(dataset_dir, batch_size=3, cache_device="cpu")
    train = cached["train"]
    assert len(train) == 4  # ceil(10 / 3)
    seen = torch.cat([b["zb"] for b in train])
    assert seen.shape[0] == 10
    # Same multiset of rows as the raw split (order-independent check).
    raw = torch.cat([b["zb"] for b in make_loaders(dataset_dir, batch_size=10)["train"]])
    assert torch.equal(
        seen.sort(dim=0).values,
        raw.sort(dim=0).values,
    )
