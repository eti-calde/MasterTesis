"""§5.1 architecture scaling study.

Sweep ``A1 x A2 x A3`` at ``small x medium x large`` budgets on the three
canonical cases (Exp 1, Exp 2, Exp 3) with three seeds: 81 runs.

Usage::

    python -m studies.arch_scaling --study-dir runs/arch_scaling
    python -m studies.arch_scaling --dry-run

Runs are idempotent (see :mod:`pinn_bath.registry`): re-launching the script
after a crash skips completed runs and resumes interrupted ones from their
last checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pinn_bath.config import CheckpointCfg, DataCfg, LossWeights, OptimizerCfg, RunConfig
from pinn_bath.registry import Registry
from studies._runner import run_one

# Map case_id -> ground-truth .npz path (relative to repo root).
CASE_PATHS: dict[str, str] = {
    "exp1": "Experiments/01-subcritical-bump-1d/data/ground_truth_dazzi_B1.npz",
    "exp2": "Experiments/02-thacker-basin-1d/data/ground_truth_thacker_T1.npz",
    "exp3": "Experiments/03-two-cylinders-2d/data/ground_truth_cylinders.npz",
    "exp5": "Experiments/05-thacker-paraboloid-3d/data/ground_truth_thacker3d.npz",
    # Exp 4 uses pinn_bath directly with a periodic case (separate study).
    # Exp 6 (Angel) is built programmatically via pinn_bath.datasets — not a
    # standalone .npz; the study script passes a constructed Case to
    # run_one(...).
}

ARCHS: tuple[str, ...] = ("A1", "A2", "A3")
BUDGETS: tuple[str, ...] = ("small", "medium", "large")
SEEDS: tuple[int, ...] = (0, 1, 2)


def _case_loss_weights(case: str) -> LossWeights:
    """Per-case ``LossWeights`` that activate the relevant M1 BC/IC terms.

    - Exp 1: bc=100 fires both `flat_bed_loss` (z_b=0 outside bump) and
      `inflow_outflow_1d_loss` (h=h_down outlet + q=h·u boundaries).
    - Exp 2: ic=100 (basin starts at known state); bc=10 (closed-walls
      u=0); wet/dry mask if the case carries `eps_wet` constant.
    - Exp 3: ic=100 (uniform IC). BC is handled implicitly by the
      uniform inflow observations + FV Dirichlet ground truth.
    """
    if case == "exp1":
        return LossWeights(data=10.0, pde=1.0, pos=10.0, bc=100.0, tikh=1.0e-5)
    if case == "exp2":
        return LossWeights(data=10.0, pde=1.0, pos=10.0, ic=100.0, bc=10.0, tv=1.0e-4)
    if case == "exp3":
        return LossWeights(data=10.0, data_u=5.0, pde=1.0, pos=10.0, ic=100.0, tv=1.0e-5)
    if case == "exp5":
        return LossWeights(data=10.0, data_u=5.0, pde=1.0, pos=10.0, ic=500.0, bc=10.0, tv=1.0e-5)
    if case == "exp6":
        # Angel real-data: no IC (window starts mid-experiment), no BC
        # explicitly (inlet S1 is a soft observation, outlet is open).
        return LossWeights(data=10.0, pde=1.0, pos=10.0, tv=1.0e-4)
    # Default: just data + pde + pos.
    return LossWeights(data=10.0, pde=1.0, pos=10.0)


def build_grid(
    *,
    cases: tuple[str, ...] = ("exp1", "exp2", "exp3"),
    archs: tuple[str, ...] = ARCHS,
    budgets: tuple[str, ...] = BUDGETS,
    seeds: tuple[int, ...] = SEEDS,
    adam_epochs: int = 12_000,
    lbfgs_steps: int = 600,
) -> list[RunConfig]:
    """Enumerate the full grid of configs for the scaling study."""
    grid: list[RunConfig] = []
    for case in cases:
        if case not in CASE_PATHS:
            raise ValueError(f"unknown case {case!r}")
        for arch in archs:
            for budget in budgets:
                for seed in seeds:
                    grid.append(
                        RunConfig(
                            case=case,
                            arch=arch,  # type: ignore[arg-type]
                            budget=budget,  # type: ignore[arg-type]
                            seed=seed,
                            loss=_case_loss_weights(case),
                            optimizer=OptimizerCfg(
                                adam_epochs=adam_epochs,
                                lbfgs_steps=lbfgs_steps,
                            ),
                            checkpoint=CheckpointCfg(every_epochs=1000, keep_last_k=2),
                            data=DataCfg(
                                case_path=CASE_PATHS[case],
                                observations=["eta", "u", "v"] if case == "exp3" else ["eta"],
                            ),
                        )
                    )
    return grid


def run_study(
    study_dir: Path | str,
    *,
    grid: list[RunConfig] | None = None,
    device: str = "cpu",
    dry_run: bool = False,
) -> dict[str, int]:
    """Execute the study; returns a histogram of outcomes."""
    grid = grid if grid is not None else build_grid()
    reg = Registry(study_dir)
    plan = reg.plan(grid)

    counts = {
        "run": 0,
        "resume": 0,
        "skip": 0,
        "ok": 0,
        "diverged": 0,
        "interrupted": 0,
        "error": 0,
    }
    for d in plan:
        counts[d.action] += 1
    print(
        f"Study {study_dir}: {len(grid)} configs -> "
        f"run={counts['run']}, resume={counts['resume']}, skip={counts['skip']}"
    )
    if dry_run:
        return counts

    for d in plan:
        if d.action == "skip":
            continue
        reg.mark_started(d.cfg)
        try:
            result = run_one(d.cfg, study_dir, device=device)
            reg.mark_finished(
                d.cfg,
                status=result.get("status", "ok"),
                wall_time_s=result.get("wall_time_s"),
                final_loss=(result.get("final_losses") or {}).get("total"),
            )
            counts[result.get("status", "ok")] = counts.get(result.get("status", "ok"), 0) + 1
        except Exception as e:
            reg.mark_finished(d.cfg, status="error", error=repr(e))
            counts["error"] += 1
            raise
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("runs/arch_scaling"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_study(args.study_dir, device=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
