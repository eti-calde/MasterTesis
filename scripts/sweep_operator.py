"""Operator scaling + physics sweep (F4-full).

Grid over (size, lambda_phys, seed); one ``train_operator`` run per cell, each
saved to its own dir. Idempotent: a cell with a finished ``summary.json`` is
skipped, so the sweep resumes after interruption. Aggregates all cells into a
text table + ``sweep_results.json`` at the end.

Examples
--------
Smoke (tiny, validates plumbing on the smoke dataset)::

    .venv/bin/python scripts/sweep_operator.py --data runs/op_dataset_smoke \
        --sizes medium --lambdas 0 --seeds 0 --epochs 4 --out runs/op_sweep_smoke

Full (on demerzel, big dataset)::

    .venv/bin/python scripts/sweep_operator.py --data runs/op_dataset \
        --sizes small medium large --lambdas 0 1e-3 1e-2 1e-1 --seeds 0 1 2 \
        --epochs 300 --out runs/op_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pinn_bath.operator.train import train_operator


def _cell_dir(out: Path, size: str, lam: float, seed: int) -> Path:
    return out / f"{size}_lam{lam:g}_s{seed}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 1e-3, 1e-2, 1e-1])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cells = [(sz, lam, sd) for sz in args.sizes for lam in args.lambdas for sd in args.seeds]
    print(
        f"Sweep: {len(cells)} cells = {len(args.sizes)} sizes x "
        f"{len(args.lambdas)} lambdas x {len(args.seeds)} seeds"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    for i, (size, lam, seed) in enumerate(cells):
        cdir = _cell_dir(args.out, size, lam, seed)
        done = cdir / "summary.json"
        tag = f"[{i + 1}/{len(cells)}] size={size} lambda={lam:g} seed={seed}"
        if done.exists():
            print(f"{tag} -> SKIP (done)", flush=True)
            results.append(json.loads(done.read_text()))
            continue
        print(f"{tag} -> running ...", flush=True)
        res = train_operator(
            args.data,
            arch="cnn",
            size=size,
            lambda_phys=lam,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            device=args.device,
            out_dir=cdir,
            log_every=max(args.epochs // 10, 1),
        )
        results.append(res)

    # Aggregate.
    (args.out / "sweep_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Table:\n")
    hdr = f"{'size':8s} {'lambda':>8s} {'seed':>4s} {'params':>10s} {'val_rmse':>10s} {'OOD_rmse':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['size']:8s} {r['lambda_phys']:>8g} {r['seed']:>4d} "
            f"{r['params']:>10,d} {r['val_rmse']:>10.4f} {r['test_rmse_ood']:>10.4f}"
        )


if __name__ == "__main__":
    main()
