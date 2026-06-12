#!/usr/bin/env python
"""Build the v2 operator dataset (40 m fjord bank: slopes + spring-neap tides).

Splits (OOD by difficulty): train/val = easy+medium (50/50), test = hard.
Output under --out: ``train.npz``, ``val.npz``, ``test.npz``, ``meta.json``.
``--env fjord2d`` builds the 2D bank instead (fields gain the y axis and the
transverse velocity ``v``). 2D cases are ~25x heavier on disk than 1D at full
resolution (~60 MB/case at 161x128x256): size the counts accordingly, e.g.
a first 2D bank of 1500/250/750 at --nx 128 --ny 64 --n-t 80 is ~20 GB.

2D smoke (a few minutes)::

    .venv/bin/python scripts/build_dataset.py --env fjord2d \
        --out runs/op2d_smoke --n-train 4 --n-val 2 --n-test 2 \
        --nx 128 --ny 64 --n-t 80 --workers 4

Smoke run (a few minutes, validates the pipeline end to end)::

    .venv/bin/python scripts/build_dataset.py --out runs/op_dataset_v2_smoke \
        --n-train 8 --n-val 4 --n-test 4 --workers 4

Full build (~10k cases, ~7 GB; designed for the remote machine)::

    nohup .venv/bin/python scripts/build_dataset.py \
        --out runs/op_dataset_v2 --workers 14 \
        --log-file runs/op_dataset_v2.log &

The dataset content is independent of --workers (sampling is single-stream
per tier in the main process); only wall-clock changes. The legacy v1 bank
remains reproducible via ``pinn_bath.datasets`` (see git history for the old
CLI).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

from pinn_bath.datagen import (
    Grid1D,
    Grid2D,
    IncidentWaveFjord1D,
    IncidentWaveFjord2D,
    PyClawSWE1D,
    PyClawSWE2D,
)
from pinn_bath.datagen.builder import DatasetBuilder

log = logging.getLogger("build_dataset")


def _summary(name: str, path: Path) -> str:
    with np.load(path) as d:
        s = d["score"]
        by_tier = {int(c): int((d["difficulty"] == c).sum()) for c in np.unique(d["difficulty"])}
        return (
            f"  {name:5s}: N={d['zb'].shape[0]:5d}  eta{tuple(d['eta'].shape)}  "
            f"score[{s.min():.2f},{s.max():.2f}]  slope[{d['slope'].min():+.4f},"
            f"{d['slope'].max():+.4f}]  f[{d['spring_neap'].min():.2f},"
            f"{d['spring_neap'].max():.2f}]  tiers={by_tier}"
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="parallel solver processes (default: cpu_count - 2)",
    )
    p.add_argument("--chunk", type=int, default=64, help="cases per scheduling batch")
    p.add_argument("--cfl", type=float, default=0.45, help="solver CFL (conservative default)")
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument(
        "--env",
        choices=["fjord1d", "fjord2d"],
        default="fjord1d",
        help="environment: 1D transect (default) or 2D channel",
    )
    # Grid overrides (defaults are the justified v2 bank; see paper).
    p.add_argument("--length", type=float, default=40.0)
    p.add_argument("--nx", type=int, default=None, help="cells in x (default: 512 1D / 256 2D)")
    p.add_argument("--t-end", type=float, default=40.0)
    p.add_argument("--n-t", type=int, default=160)
    # 2D-only grid overrides.
    p.add_argument("--ly", type=float, default=20.0, help="channel width [m] (2D)")
    p.add_argument("--ny", type=int, default=128, help="cells in y (2D)")
    args = p.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    if args.env == "fjord1d":
        grid = Grid1D(xupper=args.length, nx=args.nx or 512, t_end=args.t_end, n_t=args.n_t)
        env = IncidentWaveFjord1D(grid=grid)
        backend = PyClawSWE1D(cfl_desired=args.cfl)
    else:
        grid2 = Grid2D(
            xupper=args.length,
            nx=args.nx or 256,
            yupper=args.ly,
            ny=args.ny,
            t_end=args.t_end,
            n_t=args.n_t,
        )
        env = IncidentWaveFjord2D(grid=grid2)
        backend = PyClawSWE2D(cfl_desired=args.cfl)
    builder = DatasetBuilder(env, backend, seed=args.seed, workers=args.workers, chunk=args.chunk)

    try:
        builder.build(args.out, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
    except Exception:
        log.exception("dataset build failed")
        return 1

    for name in ("train", "val", "test"):
        log.info("%s", _summary(name, args.out / f"{name}.npz"))
    disk = sum(f.stat().st_size for f in args.out.glob("*.npz")) / 1e9
    log.info("disk: %.2f GB under %s", disk, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
