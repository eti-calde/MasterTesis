"""Exp 6 (Angel et al. 2024) run matrix — pinn_bath port.

Replaces the legacy `Experiments/06-angel-real-data/run_matrix.py`. The
canonical config swept here corresponds to the legacy "S2 only,
sigma_x=2, lambda_pde=1, 3 seeds" cell (see Exp 6 REPORT.md). The negative
result (~34 mm z_b RMSE, bump peak missed) reproduces under the new
pipeline.

Usage::

    python -m studies.exp6_run_matrix --study-dir runs/exp6
    python -m studies.exp6_run_matrix --dry-run

The Angel data is loaded by ``case_from_angel_flume`` and the sparse
sensor observations are passed explicitly to the trainer (bypassing the
default random sampler). Linear bottom drag (κ = 0.2) is auto-detected
from ``case.constants["kappa"]`` and applied in ``swe_residual``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pinn_bath.config import CheckpointCfg, DataCfg, OptimizerCfg, RunConfig
from pinn_bath.datasets import case_from_angel_flume
from pinn_bath.registry import Registry
from studies._runner import run_one
from studies.arch_scaling import _case_loss_weights

# Default flume path (mean of 20 runs).
FLUME_NPZ = Path("Experiments/datasets/angel2024/processed/angel2024_flume.npz")

SEEDS: tuple[int, ...] = (0, 1, 2)


def _exp6_cfg(seed: int, adam_epochs: int, lbfgs_steps: int) -> RunConfig:
    return RunConfig(
        case="exp6",
        arch="A1",
        budget="small",
        form="primitive",
        seed=seed,
        loss=_case_loss_weights("exp6"),
        optimizer=OptimizerCfg(adam_epochs=adam_epochs, lbfgs_steps=lbfgs_steps),
        checkpoint=CheckpointCfg(every_epochs=1000, keep_last_k=2),
        data=DataCfg(
            case_path=str(FLUME_NPZ),
            observations=["eta"],
            n_obs_points=0,  # overridden by explicit obs_coords/obs_values
        ),
    )


def build_grid(
    *,
    seeds: tuple[int, ...] = SEEDS,
    adam_epochs: int = 15_000,
    lbfgs_steps: int = 200,
) -> list[RunConfig]:
    """Canonical: 3 seeds, A1/small, S2-only obs, sigma_x=2 (default A1)."""
    return [_exp6_cfg(s, adam_epochs, lbfgs_steps) for s in seeds]


def run_study(
    study_dir: Path | str,
    *,
    grid: list[RunConfig] | None = None,
    device: str = "cpu",
    dry_run: bool = False,
) -> dict[str, int]:
    """Execute the Exp 6 run matrix."""
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
        f"Exp 6 study {study_dir}: {len(grid)} configs -> "
        f"run={counts['run']}, resume={counts['resume']}, skip={counts['skip']}"
    )
    if dry_run:
        return counts

    # Build the Case once (cheap) — same flume window for every seed.
    case, obs = case_from_angel_flume(FLUME_NPZ)

    for d in plan:
        if d.action == "skip":
            continue
        reg.mark_started(d.cfg)
        try:
            result: dict[str, Any] = run_one(
                d.cfg,
                study_dir,
                device=device,
                case=case,
                obs_coords=obs.obs_coords,
                obs_values=obs.obs_values,
                # Exp 6 has Nt*Nx ~= 27k cells; use a wet-biased fraction.
                n_collocation=2_000,
                n_bc=50,
            )
            reg.mark_finished(
                d.cfg,
                status=result.get("status", "ok"),
                wall_time_s=result.get("wall_time_s"),
                final_loss=(result.get("final_losses") or {}).get("total"),
            )
            counts[result.get("status", "ok")] = counts.get(result.get("status", "ok"), 0) + 1
        except Exception as e:
            # Log + continue so one failing config doesn't kill the whole
            # multi-hour sweep. The error is recorded in the registry; the
            # final counts dict carries an "error" tally for the CLI to
            # report. Re-run with --retry-errors to retry these.
            import traceback

            reg.mark_finished(d.cfg, status="error", error=repr(e))
            counts["error"] += 1
            print(f"  ERROR run_id={d.cfg.run_id}: {e!r}", flush=True)
            traceback.print_exc()
    if counts["error"]:
        print(
            f"[sweep finished with {counts['error']} errors; see registry]",
            flush=True,
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("runs/exp6"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_study(args.study_dir, device=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
