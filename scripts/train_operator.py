"""Train one inverse-operator config (F3) on a built dataset (v2 bank).

Headline experiment = two runs (compare OOD test RMSE)::

    .venv/bin/python scripts/train_operator.py --data runs/op_dataset_v2 \
        --lambda-phys 0    --cache-data --out runs/operator_v2/cnn_nophys
    .venv/bin/python scripts/train_operator.py --data runs/op_dataset_v2 \
        --lambda-phys 1e-2 --cache-data --out runs/operator_v2/cnn_phys

``--cache-data`` keeps the whole dataset resident on the compute device
(recommended on the training GPU; the v2 bank is ~7 GB). Training runs at
most ``--epochs`` but stops early after ``--patience`` epochs without val
improvement; validation runs every epoch, so the saved checkpoint is the
exact val optimum.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pinn_bath.operator.train import train_operator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--arch", default="cnn")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--lambda-phys", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=300, help="maximum epoch budget")
    p.add_argument(
        "--patience", type=int, default=50, help="early stop after N epochs without val improvement"
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--grad-clip", type=float, default=None, help="max grad norm (default: measure only)"
    )
    p.add_argument(
        "--cache-data",
        action="store_true",
        help="cache the whole dataset on the compute device (no per-step H2D copies)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    res = train_operator(
        args.data,
        arch=args.arch,
        size=args.size,
        lambda_phys=args.lambda_phys,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        cache_data=args.cache_data,
        seed=args.seed,
        device=args.device,
        out_dir=args.out,
    )
    print("\n=== resultado ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
