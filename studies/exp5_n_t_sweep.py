"""Exp 5 (Thacker 2D paraboloid) N_t sweep — pinn_bath port.

Replaces the legacy `Experiments/05-thacker-paraboloid-3d/n_t_sweep.py`.
Sweeps the number of temporal observation snapshots N_t ∈ {1, 2, 4, 8,
30} on the 2D axisymmetric paraboloid (eta-only observations); 3 seeds
per cell.

Tests whether the Exp-2/Exp-4 "2-to-4 snapshots suffice" finding
extends to the fully 2D closed-basin case.

Usage::

    python -m studies.exp5_n_t_sweep --study-dir runs/exp5_n_t
    python -m studies.exp5_n_t_sweep --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pinn_bath.config import CheckpointCfg, DataCfg, OptimizerCfg, RunConfig
from pinn_bath.data import Case
from pinn_bath.datasets import evenly_spaced_indices, subsample_t_observations
from pinn_bath.registry import Registry
from studies._runner import run_one
from studies.arch_scaling import CASE_PATHS, _case_loss_weights

N_T_VALUES: tuple[int, ...] = (1, 2, 4, 8, 30)
SEEDS: tuple[int, ...] = (0, 1, 2)


def _cfg(n_t: int, seed: int, adam_epochs: int, lbfgs_steps: int) -> RunConfig:
    return RunConfig(
        case="exp5",
        arch="A1",
        budget="small",
        form="primitive",
        seed=seed,
        loss=_case_loss_weights("exp5"),
        optimizer=OptimizerCfg(adam_epochs=adam_epochs, lbfgs_steps=lbfgs_steps),
        checkpoint=CheckpointCfg(every_epochs=1000, keep_last_k=2),
        data=DataCfg(
            case_path=CASE_PATHS["exp5"],
            observations=["eta"],
            n_obs_points=0,
        ),
    )


def build_grid(
    *,
    seeds: tuple[int, ...] = SEEDS,
    n_t_values: tuple[int, ...] = N_T_VALUES,
    adam_epochs: int = 8_000,
    lbfgs_steps: int = 200,
) -> list[tuple[RunConfig, int]]:
    return [(_cfg(n_t, s, adam_epochs, lbfgs_steps), n_t) for n_t in n_t_values for s in seeds]


def run_study(
    study_dir: Path | str,
    *,
    grid: list[tuple[RunConfig, int]] | None = None,
    device: str = "cpu",
    dry_run: bool = False,
) -> dict[str, int]:
    grid = grid if grid is not None else build_grid()
    cfgs = [g[0] for g in grid]
    reg = Registry(study_dir)
    plan = reg.plan(cfgs)
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
        f"Exp 5 N_t sweep {study_dir}: {len(grid)} configs -> "
        f"run={counts['run']}, resume={counts['resume']}, skip={counts['skip']}"
    )
    if dry_run:
        return counts

    case = Case.load(CASE_PATHS["exp5"])
    cfg_to_nt = {cfg.run_id: n_t for cfg, n_t in grid}

    for d in plan:
        if d.action == "skip":
            continue
        n_t = cfg_to_nt[d.cfg.run_id]
        t_idx = evenly_spaced_indices(case.coords["t"].size, n_t)
        obs_coords, obs_values = subsample_t_observations(case, t_idx, fields=("eta",))
        reg.mark_started(d.cfg)
        try:
            result: dict[str, Any] = run_one(
                d.cfg,
                study_dir,
                device=device,
                case=case,
                obs_coords=obs_coords,
                obs_values=obs_values,
                n_collocation=4_000,
                n_bc=30,
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
    parser.add_argument("--study-dir", type=Path, default=Path("runs/exp5_n_t"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_study(args.study_dir, device=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
