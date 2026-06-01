"""Plot operator training progress from metrics.jsonl (works live, mid-run).

One run::

    .venv/bin/python scripts/plot_training.py runs/operator/cnn_nophys

Compare several runs on one axis::

    .venv/bin/python scripts/plot_training.py runs/op_sweep/medium_lam0_s0 \
        runs/op_sweep/medium_lam0.01_s0 --out analysis/compare.png

Reads only ``metrics.jsonl`` (no checkpoints), so it can be run while training
is still in progress to watch the curves fill in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(run_dir: Path) -> list[dict]:
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def _series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    """(epochs, values) for rows that contain ``key`` (eval keys are sparse)."""
    ep, val = [], []
    for r in rows:
        if key in r and r[key] is not None:
            ep.append(r["epoch"])
            val.append(r[key])
    return ep, val


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = plt.cm.tab10.colors

    for i, rd in enumerate(args.run_dirs):
        rows = load_metrics(rd)
        if not rows:
            print(f"(sin metrics) {rd}")
            continue
        c = colors[i % len(colors)]
        label = rd.name

        # Panel 0: training losses (total / mse / physics) — per epoch.
        ep, loss = _series(rows, "train_loss")
        ax[0].plot(ep, loss, color=c, lw=1.6, label=f"{label}")
        ep_m, mse = _series(rows, "train_mse")
        if any(m != tot for m, tot in zip(mse, loss, strict=False)):  # physics → mse≠loss
            ax[0].plot(ep_m, mse, color=c, lw=1.0, ls=":", alpha=0.7)

        # Panel 1: val (in-dist) solid vs OOD test dashed.
        ev, v = _series(rows, "val_rmse")
        eo, o = _series(rows, "test_rmse_ood")
        ax[1].plot(ev, v, color=c, lw=1.8, label=f"{label} val")
        ax[1].plot(eo, o, color=c, lw=1.8, ls="--", alpha=0.7)

        # Panel 2: physics components (if logged).
        ep_c, cont = _series(rows, "train_phys_cont")
        ep_mo, mom = _series(rows, "train_phys_mom")
        if any(cont):
            ax[2].plot(ep_c, cont, color=c, lw=1.4, label=f"{label} cont")
            ax[2].plot(ep_mo, mom, color=c, lw=1.4, ls="--", alpha=0.7)

    ax[0].set_title("Loss de entrenamiento (sólido=total, punteado=MSE)")
    ax[0].set_xlabel("época")
    ax[0].set_ylabel("loss")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].set_title("RMSE $z_b$ (sólido=val in-dist, guión=test OOD)")
    ax[1].set_xlabel("época")
    ax[1].set_ylabel("RMSE [m]")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    ax[2].set_title("Residuo físico (sólido=continuidad, guión=momento)")
    ax[2].set_xlabel("época")
    ax[2].set_ylabel("residuo")
    ax[2].set_yscale("log")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = args.out or (args.run_dirs[0] / "training_curves.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
