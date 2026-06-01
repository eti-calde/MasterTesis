"""Train one inverse-operator config (F3).

Headline experiment = two runs (compare OOD test RMSE):

    .venv/bin/python scripts/train_operator.py --data runs/op_dataset_dev \
        --lambda-phys 0   --out runs/operator/cnn_nophys
    .venv/bin/python scripts/train_operator.py --data runs/op_dataset_dev \
        --lambda-phys 1e-2 --out runs/operator/cnn_phys
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pinn_bath.operator.train import train_operator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/op_dataset_dev"))
    p.add_argument("--arch", default="cnn")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--lambda-phys", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
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
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        out_dir=args.out,
    )
    print("\n=== resultado ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
