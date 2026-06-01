"""Build / load the amortized-inverse-operator dataset (F2).

Loops the case generator (:mod:`pinn_bath.datasets.generator`) over many seeds
per difficulty tier, runs each through the forward solver, and stacks the
results into train / val / test splits stored as ``.npz``.

The out-of-distribution split is *by difficulty*: train = easy + medium,
test = hard. A held-out fraction of easy+medium is the in-distribution
validation set. Tiers are drawn from independent RNG streams and every case
carries a unique seed, so the splits are disjoint (no leakage).

Each split stores, as float32 unless noted:
  ``zb`` (N, Nx)            bathymetry — the inverse target
  ``eta`` (N, Nt, Nx)       free-surface field — the operator input
  ``u`` (N, Nt, Nx)         velocity field — needed by the physics residual
  ``score`` (N,)            continuous difficulty score
  ``difficulty`` (N,) int8  tier code (see :data:`DIFFICULTY_CODE`)
  ``seed`` (N,) int64       per-case seed (disjointness check)
  ``x`` (Nx,), ``t`` (Nt,)  shared grid axes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pinn_bath.datasets.generator import (
    Difficulty,
    Grid,
    generate_record,
    sample_case,
)

DIFFICULTY_CODE: dict[Difficulty, int] = {"easy": 0, "medium": 1, "hard": 2}


def build_records(
    difficulty: Difficulty,
    n: int,
    grid: Grid,
    rng: np.random.Generator,
    *,
    progress_every: int = 25,
    **solver_kw: Any,
) -> dict[str, np.ndarray]:
    """Generate ``n`` solved cases for one difficulty tier, stacked."""
    zb, eta, u, score, seed = [], [], [], [], []
    for i in range(n):
        spec = sample_case(difficulty, rng, grid)
        rec = generate_record(spec, grid, **solver_kw)
        if not np.isfinite(rec["eta"]).all():
            # Skip the rare unstable case rather than poison the dataset.
            continue
        zb.append(rec["zb"].astype(np.float32))
        eta.append(rec["eta"].astype(np.float32))
        u.append(rec["u"].astype(np.float32))
        score.append(np.float32(rec["score"]))
        seed.append(np.int64(rec["seed"]))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {difficulty}: {i + 1}/{n}", flush=True)
    code = DIFFICULTY_CODE[difficulty]
    n_ok = len(zb)
    return {
        "zb": np.stack(zb),
        "eta": np.stack(eta),
        "u": np.stack(u),
        "score": np.asarray(score, dtype=np.float32),
        "difficulty": np.full(n_ok, code, dtype=np.int8),
        "seed": np.asarray(seed, dtype=np.int64),
        "x": grid.centers.astype(np.float32),
        "t": np.linspace(0.0, grid.t_end, grid.n_t + 1, dtype=np.float32),
    }


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-tier record dicts along the case axis."""
    stack_keys = ("zb", "eta", "u", "score", "difficulty", "seed")
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in stack_keys}
    out["x"] = parts[0]["x"]
    out["t"] = parts[0]["t"]
    return out


def split_train_val(
    records: dict[str, np.ndarray],
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Shuffle and partition in-distribution records into (train, val)."""
    n = records["zb"].shape[0]
    perm = rng.permutation(n)
    n_val = round(val_frac * n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    case_keys = ("zb", "eta", "u", "score", "difficulty", "seed")

    def take(idx: np.ndarray) -> dict[str, np.ndarray]:
        d = {k: records[k][idx] for k in case_keys}
        d["x"] = records["x"]
        d["t"] = records["t"]
        return d

    return take(train_idx), take(val_idx)


def save_split(path: str | Path, data: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def grid_to_meta(grid: Grid) -> dict[str, Any]:
    return {
        "xlower": grid.xlower,
        "xupper": grid.xupper,
        "nx": grid.nx,
        "t_end": grid.t_end,
        "n_t": grid.n_t,
        "sea_level": grid.sea_level,
        "difficulty_code": DIFFICULTY_CODE,
    }


def write_meta(path: str | Path, meta: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(meta, indent=2))
