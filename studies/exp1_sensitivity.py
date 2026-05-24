"""§5.2 Exp 1 sensitivity study (port of legacy sensitivity_studies.py).

Three independent 1D sweeps on the Dazzi B1 subcritical bump:

- **density**: fraction of grid points observed ∈ {100, 50, 20, 10, 5} %.
- **noise**: Gaussian noise std on η ∈ {0, 1, 2, 5} % of the signal range.
- **obstype**: which fields are observed ∈ {η, u, η+u}.

Each cell of each sweep runs with multiple seeds (default 3) to report
mean ± std. Architecture is held fixed at A1/small (matching the legacy
sensitivity baseline so cross-sweep cifras stay comparable).

Usage::

    python -m studies.exp1_sensitivity --study-dir runs/exp1_sensitivity
    python -m studies.exp1_sensitivity --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pinn_bath.config import CheckpointCfg, DataCfg, LossWeights, OptimizerCfg, RunConfig
from pinn_bath.registry import Registry
from studies._runner import run_one
from studies.arch_scaling import CASE_PATHS

# A1 small: same architecture used to generate the legacy 5.80 mm baseline.
ARCH = "A1"
BUDGET = "small"
FORM = "primitive"  # post-ablation default (REPORT-ABLATION.md)
SEEDS: tuple[int, ...] = (0, 1, 2)

# Sweep axes
DENSITIES: tuple[int, ...] = (500, 250, 100, 50, 25)  # 100/50/20/10/5 % of 500 obs pts
NOISES: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
OBSTYPES: tuple[tuple[str, ...], ...] = (
    ("eta",),
    ("u",),
    ("eta", "u"),
)


def _base_cfg(
    seed: int,
    n_obs: int,
    noise: float,
    observations: tuple[str, ...],
    adam_epochs: int,
    lbfgs_steps: int,
) -> RunConfig:
    return RunConfig(
        case="exp1",
        arch=ARCH,
        budget=BUDGET,
        form=FORM,
        seed=seed,
        loss=LossWeights(
            data=10.0,
            pde=1.0,
            pos=10.0,
            bc=100.0,
            tikh=1.0e-5,
        ),
        optimizer=OptimizerCfg(
            adam_epochs=adam_epochs,
            lbfgs_steps=lbfgs_steps,
        ),
        checkpoint=CheckpointCfg(every_epochs=1000, keep_last_k=2),
        data=DataCfg(
            case_path=CASE_PATHS["exp1"],
            observations=list(observations),
            n_obs_points=n_obs,
            obs_noise_std=noise,
        ),
    )


def build_grid(
    *,
    seeds: tuple[int, ...] = SEEDS,
    adam_epochs: int = 12_000,
    lbfgs_steps: int = 600,
) -> list[RunConfig]:
    """Enumerate all sensitivity-sweep configs.

    Three independent axes; the 100 %/0 %/η baseline cell is shared so we
    avoid duplicating runs.
    """
    grid: list[RunConfig] = []
    seen: set[tuple] = set()

    def add(n_obs: int, noise: float, obs: tuple[str, ...]) -> None:
        for s in seeds:
            cfg = _base_cfg(s, n_obs, noise, obs, adam_epochs, lbfgs_steps)
            key = (n_obs, noise, obs, s)
            if key in seen:
                continue
            seen.add(key)
            grid.append(cfg)

    # density axis (noise=0, eta-only)
    for n_obs in DENSITIES:
        add(n_obs, 0.0, ("eta",))
    # noise axis (full density, eta-only)
    for noise in NOISES:
        add(500, noise, ("eta",))
    # obstype axis (full density, no noise)
    for obs in OBSTYPES:
        add(500, 0.0, obs)
    return grid


def run_study(
    study_dir: Path | str,
    *,
    grid: list[RunConfig] | None = None,
    device: str = "cpu",
    dry_run: bool = False,
) -> dict[str, int]:
    """Execute the sensitivity study (same shape as arch_scaling.run_study)."""
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
    parser.add_argument("--study-dir", type=Path, default=Path("runs/exp1_sensitivity"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_study(args.study_dir, device=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
