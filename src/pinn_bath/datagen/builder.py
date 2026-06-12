"""Dataset builder: environment x backend -> train/val/test ``.npz`` splits.

Successor to :mod:`pinn_bath.datasets.operator_dataset` for the v2 bank.
Keeps the exact split format that :mod:`pinn_bath.operator.data` loads
(``zb``, ``eta``, ``u``, ``score``, ``difficulty``, ``seed``, ``x``, ``t``)
and adds per-case excitation labels for posterior analysis: ``slope``,
``spring_neap`` (the f factor) and ``water_level``.

Design for the 10k-case run on a remote machine:

- **Parallel solves**: a ``ProcessPoolExecutor`` integrates cases across
  workers; sampling stays in the main process on one RNG stream per tier, so
  the dataset content is *independent of the worker count* (same seeds ->
  same cases, in the same slots).
- **Preallocated splits**: arrays are allocated once at target size and
  filled in place (~7 GB for 10k cases), avoiding the list-then-stack double
  peak.
- **Skip-and-replace**: the rare non-finite solve is dropped and a fresh case
  is sampled from the same tier stream until every slot is filled (counted in
  the manifest as ``n_skipped``).
- **No leakage**: per-tier RNG streams are spawned from independent seed
  sequences; every case carries a unique seed and disjointness is asserted
  across splits before saving.

Split layout (OOD by difficulty, as in the paper): train and val are drawn
half easy / half medium from the same tier streams (first block -> train,
remainder -> val; cases are i.i.d., so no shuffle is needed), test is
entirely hard.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pinn_bath.datagen.bathymetry import Difficulty
from pinn_bath.datagen.cases import CaseSpec
from pinn_bath.datagen.environments.base import Environment
from pinn_bath.datagen.solvers.base import SolverBackend

log = logging.getLogger(__name__)

DIFFICULTY_CODE: dict[Difficulty, int] = {"easy": 0, "medium": 1, "hard": 2}

# Worker-process globals, set once by the pool initializer (cheaper than
# pickling env/backend with every task).
_ENV: Environment | None = None
_BACKEND: SolverBackend | None = None


def _init_worker(env: Environment, backend: SolverBackend) -> None:
    global _ENV, _BACKEND
    _ENV = env
    _BACKEND = backend


def _solve_case(spec: CaseSpec) -> dict[str, np.ndarray] | None:
    """Integrate one case; return float32 fields, or None if non-finite."""
    res = _ENV.simulate(spec, _BACKEND)
    if not res.ok:
        return None
    out = {
        "zb": res.zb.astype(np.float32),
        "eta": res.eta.astype(np.float32),
        "u": res.u.astype(np.float32),
    }
    if res.v is not None:  # 2D backends fill the transverse velocity
        out["v"] = res.v.astype(np.float32)
    return out


class DatasetBuilder:
    """Builds the operator dataset splits from an environment + backend."""

    def __init__(
        self,
        env: Environment,
        backend: SolverBackend,
        *,
        seed: int = 0,
        workers: int = 1,
        chunk: int = 64,
    ) -> None:
        self.env = env
        self.backend = backend
        self.seed = seed
        self.workers = max(1, workers)
        self.chunk = max(1, chunk)

    # ------------------------------------------------------------------ #
    @property
    def _is2d(self) -> bool:
        return hasattr(self.env.grid, "ny")

    def _alloc_split(self, n: int) -> dict[str, np.ndarray]:
        grid = self.env.grid
        nt = grid.n_t + 1
        if self._is2d:
            fshape: tuple[int, ...] = (n, nt, grid.ny, grid.nx)
            zshape: tuple[int, ...] = (n, grid.ny, grid.nx)
        else:
            fshape = (n, nt, grid.nx)
            zshape = (n, grid.nx)
        out = {
            "zb": np.empty(zshape, dtype=np.float32),
            "eta": np.empty(fshape, dtype=np.float32),
            "u": np.empty(fshape, dtype=np.float32),
            "score": np.empty(n, dtype=np.float32),
            "difficulty": np.empty(n, dtype=np.int8),
            "seed": np.empty(n, dtype=np.int64),
            "slope": np.empty(n, dtype=np.float32),
            "spring_neap": np.empty(n, dtype=np.float32),
            "water_level": np.empty(n, dtype=np.float32),
        }
        if self._is2d:
            out["v"] = np.empty(fshape, dtype=np.float32)
        return out

    def _fill_tier(
        self,
        tier: Difficulty,
        slots: list[tuple[dict[str, np.ndarray], int]],
        rng: np.random.Generator,
        pool: ProcessPoolExecutor,
    ) -> int:
        """Sample+solve until every (split_arrays, index) slot is filled."""
        code = DIFFICULTY_CODE[tier]
        filled, skipped = 0, 0
        t0 = time.perf_counter()
        while filled < len(slots):
            n_batch = min(self.chunk, len(slots) - filled)
            specs = [self.env.sample_case(tier, rng) for _ in range(n_batch)]
            for spec, rec in zip(specs, pool.map(_solve_case, specs), strict=True):
                if rec is None:
                    skipped += 1
                    continue
                arrs, i = slots[filled]
                arrs["zb"][i] = rec["zb"]
                arrs["eta"][i] = rec["eta"]
                arrs["u"][i] = rec["u"]
                if "v" in arrs:
                    arrs["v"][i] = rec["v"]
                arrs["score"][i] = spec.score
                arrs["difficulty"][i] = code
                arrs["seed"][i] = spec.seed
                arrs["slope"][i] = spec.bathymetry.slope
                arrs["spring_neap"][i] = spec.spring_neap
                arrs["water_level"][i] = spec.water_level
                filled += 1
            rate = filled / max(time.perf_counter() - t0, 1e-9)
            eta_s = (len(slots) - filled) / max(rate, 1e-9)
            log.info(
                "  %s: %d/%d (%.2f case/s, ETA %.1f min, skipped %d)",
                tier,
                filled,
                len(slots),
                rate,
                eta_s / 60.0,
                skipped,
            )
        return skipped

    # ------------------------------------------------------------------ #
    def build(
        self,
        out_dir: str | Path,
        *,
        n_train: int = 6000,
        n_val: int = 1000,
        n_test: int = 3000,
    ) -> dict[str, Any]:
        """Generate, solve and save the three splits + ``meta.json``.

        Returns the manifest dict (also written to ``out_dir/meta.json``).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        grid = self.env.grid
        t_axis = np.linspace(0.0, grid.t_end, grid.n_t + 1, dtype=np.float32)

        splits = {
            "train": self._alloc_split(n_train),
            "val": self._alloc_split(n_val),
            "test": self._alloc_split(n_test),
        }
        # In-distribution tiers feed train then val from one stream each
        # (half easy / half medium per split); hard is entirely the OOD test.
        n_e_train, n_e_val = n_train // 2, n_val // 2
        n_m_train, n_m_val = n_train - n_e_train, n_val - n_e_val
        tier_slots: dict[Difficulty, list[tuple[dict[str, np.ndarray], int]]] = {
            "easy": [(splits["train"], i) for i in range(n_e_train)]
            + [(splits["val"], i) for i in range(n_e_val)],
            "medium": [(splits["train"], n_e_train + i) for i in range(n_m_train)]
            + [(splits["val"], n_e_val + i) for i in range(n_m_val)],
            "hard": [(splits["test"], i) for i in range(n_test)],
        }
        # Independent, reproducible stream per tier -> disjoint cases.
        seeds = np.random.SeedSequence(self.seed).spawn(3)
        rngs = {
            t: np.random.default_rng(s)
            for t, s in zip(("easy", "medium", "hard"), seeds, strict=True)
        }

        log.info(
            "building %d cases (train %d / val %d / test %d) on grid "
            "Nx=%d, Nt=%d, L=%.0f m, t_end=%.0f s with %d workers",
            n_train + n_val + n_test,
            n_train,
            n_val,
            n_test,
            grid.nx,
            grid.n_t + 1,
            grid.xupper - grid.xlower,
            grid.t_end,
            self.workers,
        )
        t0 = time.perf_counter()
        skipped: dict[str, int] = {}
        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_worker,
            initargs=(self.env, self.backend),
        ) as pool:
            for tier in ("easy", "medium", "hard"):
                skipped[tier] = self._fill_tier(tier, tier_slots[tier], rngs[tier], pool)
        elapsed = time.perf_counter() - t0

        # No-leakage check: per-case seeds disjoint across splits.
        s_tr = set(splits["train"]["seed"].tolist())
        s_va = set(splits["val"]["seed"].tolist())
        s_te = set(splits["test"]["seed"].tolist())
        if (s_tr & s_va) or (s_tr & s_te) or (s_va & s_te):
            raise RuntimeError("seed leakage across splits")

        axes: dict[str, np.ndarray] = {"t": t_axis}
        if self._is2d:
            axes["x"] = grid.x_centers.astype(np.float32)
            axes["y"] = grid.y_centers.astype(np.float32)
        else:
            axes["x"] = grid.centers.astype(np.float32)
        for name, arrs in splits.items():
            path = out_dir / f"{name}.npz"
            log.info("saving %s ...", path)
            np.savez_compressed(path, **axes, **arrs)

        manifest = self._manifest(n_train, n_val, n_test, skipped, elapsed)
        (out_dir / "meta.json").write_text(json.dumps(manifest, indent=2))
        log.info(
            "done: %d cases in %.1f min (%.2f s/case), %d skipped, saved to %s",
            n_train + n_val + n_test,
            elapsed / 60.0,
            elapsed / max(n_train + n_val + n_test, 1),
            sum(skipped.values()),
            out_dir,
        )
        return manifest

    # ------------------------------------------------------------------ #
    def _manifest(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        skipped: dict[str, int],
        elapsed: float,
    ) -> dict[str, Any]:
        env = self.env
        m: dict[str, Any] = {
            "datagen_version": "v2",
            "grid": asdict(env.grid),
            "difficulty_code": DIFFICULTY_CODE,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "split_design": "train/val = easy+medium (50/50), test = hard (OOD)",
            "seed": self.seed,
            "workers": self.workers,
            "n_skipped": skipped,
            "elapsed_s": round(elapsed, 1),
            "backend": {
                "name": self.backend.name,
                **{
                    k: getattr(self.backend, k)
                    for k in ("kernel", "limiter", "cfl_desired", "cfl_max", "dry_tolerance", "g")
                    if hasattr(self.backend, k)
                },
            },
        }
        # Sampler configuration (full reproducibility record). asdict recurses
        # into the tier table and works for both the 1D and 2D samplers.
        if hasattr(env, "bathymetry"):
            m["bathymetry_sampler"] = {
                "type": type(env.bathymetry).__name__,
                **asdict(env.bathymetry),
            }
        if hasattr(env, "forcing"):
            m["forcing_sampler"] = asdict(env.forcing)
        if hasattr(env, "bed_cap"):
            m["bed_cap"] = env.bed_cap
        return m
