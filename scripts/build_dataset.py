"""Build the operator dataset: easy+medium = train/val, hard = OOD test (F2).

Examples
--------
Smoke (tiny, ~1 min) — validates the pipeline end to end::

    .venv/bin/python scripts/build_dataset.py --out runs/op_dataset_smoke \
        --n-easy 6 --n-medium 6 --n-hard 6

Full build (configurable; ~1 s/case)::

    .venv/bin/python scripts/build_dataset.py --out runs/op_dataset \
        --n-easy 800 --n-medium 800 --n-hard 500 --val-frac 0.15

Output: ``train.npz``, ``val.npz``, ``test.npz``, ``meta.json`` under --out.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pinn_bath.datasets.generator import Grid
from pinn_bath.datasets.operator_dataset import (
    _concat,
    build_records,
    grid_to_meta,
    load_split,
    save_split,
    split_train_val,
    write_meta,
)


def _summary(name: str, d: dict[str, np.ndarray]) -> str:
    s = d["score"]
    by_tier = {int(c): int((d["difficulty"] == c).sum()) for c in np.unique(d["difficulty"])}
    return (
        f"  {name:5s}: N={d['zb'].shape[0]:4d}  eta{tuple(d['eta'].shape)}  "
        f"score[{s.min():.2f},{s.max():.2f}]  tiers(code:count)={by_tier}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-easy", type=int, default=800)
    p.add_argument("--n-medium", type=int, default=800)
    p.add_argument("--n-hard", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfl", type=float, default=0.45)
    p.add_argument(
        "--regime",
        choices=["incident_wave", "free_transient"],
        default="incident_wave",
        help="forcing regime (default: incident_wave — continuous inflow + outflow)",
    )
    # Grid overrides (defaults match generator.Grid).
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--t-end", type=float, default=8.0)
    p.add_argument("--n-t", type=int, default=120)
    args = p.parse_args()

    grid = Grid(nx=args.nx, t_end=args.t_end, n_t=args.n_t)
    print(f"Grid: Nx={grid.nx}, Nt={grid.n_t + 1}, t_end={grid.t_end}, sea_level={grid.sea_level}")
    print(f"Generating easy={args.n_easy}, medium={args.n_medium}, hard={args.n_hard} ...")

    t0 = time.time()
    # Independent RNG stream per tier → disjoint cases.
    rng_e = np.random.default_rng(args.seed + 1)
    rng_m = np.random.default_rng(args.seed + 2)
    rng_h = np.random.default_rng(args.seed + 3)
    rec_e = build_records("easy", args.n_easy, grid, rng_e, regime=args.regime, cfl_desired=args.cfl)
    rec_m = build_records("medium", args.n_medium, grid, rng_m, regime=args.regime, cfl_desired=args.cfl)
    rec_h = build_records("hard", args.n_hard, grid, rng_h, regime=args.regime, cfl_desired=args.cfl)

    # Splits: train+val from easy+medium (in-distribution), test = hard (OOD).
    in_dist = _concat([rec_e, rec_m])
    rng_split = np.random.default_rng(args.seed + 99)
    train, val = split_train_val(in_dist, args.val_frac, rng_split)
    test = rec_h

    # No-leakage assertion: seed sets disjoint across splits.
    s_tr, s_va, s_te = set(train["seed"]), set(val["seed"]), set(test["seed"])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "seed leakage!"

    save_split(args.out / "train.npz", train)
    save_split(args.out / "val.npz", val)
    save_split(args.out / "test.npz", test)
    meta = grid_to_meta(grid)
    meta.update(
        {
            "n_easy": args.n_easy,
            "n_medium": args.n_medium,
            "n_hard": args.n_hard,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "cfl": args.cfl,
            "regime": args.regime,
        }
    )
    write_meta(args.out / "meta.json", meta)

    dt = time.time() - t0
    n_total = train["zb"].shape[0] + val["zb"].shape[0] + test["zb"].shape[0]
    print(f"\nDone in {dt / 60:.1f} min ({dt / max(n_total, 1):.2f} s/case, {n_total} cases)")
    print(_summary("train", train))
    print(_summary("val", val))
    print(_summary("test", test))
    # Round-trip check + disk size.
    rt = load_split(args.out / "train.npz")
    assert rt["eta"].shape == train["eta"].shape
    disk = sum(f.stat().st_size for f in args.out.glob("*.npz")) / 1e6
    print(f"\nSaved to {args.out}/ (train+val+test = {disk:.1f} MB)")
    print("Split: train+val = easy+medium (in-dist), test = hard (OOD by difficulty)")


if __name__ == "__main__":
    main()
