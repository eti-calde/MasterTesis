"""Exp 2 (Thacker 1D) snapshot-vs-time-series sweep — pinn_bath port.

Replaces the legacy `Experiments/02-thacker-basin-1d/snapshot_vs_timeseries.py`.
Sweeps the number of temporal observation snapshots N_t ∈ {1, 2, 4, 10, 40}
on the closed parabolic basin (eta-only observations); 3 seeds per cell.

Key historical finding to replicate: N_t = 4 already breaks the
eta-only equifinality (~5 mm RMSE), N_t = 1 fails completely
(~180 mm RMSE).

Usage::

    python -m studies.exp2_n_t_sweep --study-dir runs/exp2_n_t
    python -m studies.exp2_n_t_sweep --dry-run
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

N_T_VALUES: tuple[int, ...] = (1, 2, 4, 10, 40)
SEEDS: tuple[int, ...] = (0, 1, 2)


def _cfg(n_t: int, seed: int, adam_epochs: int, lbfgs_steps: int) -> RunConfig:
    return RunConfig(
        case="exp2",
        arch="A1",
        budget="small",
        form="primitive",
        seed=seed,
        loss=_case_loss_weights("exp2"),
        optimizer=OptimizerCfg(adam_epochs=adam_epochs, lbfgs_steps=lbfgs_steps),
        checkpoint=CheckpointCfg(every_epochs=1000, keep_last_k=2),
        data=DataCfg(
            case_path=CASE_PATHS["exp2"],
            observations=["eta"],
            n_obs_points=0,  # overridden via explicit obs_coords/obs_values
        ),
    )


def build_grid(
    *,
    seeds: tuple[int, ...] = SEEDS,
    n_t_values: tuple[int, ...] = N_T_VALUES,
    adam_epochs: int = 12_000,
    lbfgs_steps: int = 600,
) -> list[tuple[RunConfig, int]]:
    """Return (cfg, n_t) pairs; n_t is stored alongside to drive the obs."""
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
        f"Exp 2 N_t sweep {study_dir}: {len(grid)} configs -> "
        f"run={counts['run']}, resume={counts['resume']}, skip={counts['skip']}"
    )
    if dry_run:
        return counts

    case = Case.load(CASE_PATHS["exp2"])
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
                n_collocation=2_000,
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
            reg.mark_finished(d.cfg, status="error", error=repr(e))
            counts["error"] += 1
            raise
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("runs/exp2_n_t"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_study(args.study_dir, device=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
