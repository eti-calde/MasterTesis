"""§5.4 ablation: three forms of the SWE residual on Exp 1.

Sweep ``{primitive, prim_cons, conservative} x {3 seeds} x {h_floor=0, 0.05}``
= 18 runs, all on Exp 1 with the A1 small configuration (so the comparison
isolates the residual form, holding architecture and budget constant).

The ``h_floor`` axis activates the structural fix introduced in M10: with
``h_floor=0.0`` the model output is ``h = softplus(raw)`` (can collapse
to ~0); with ``h_floor>0`` it is ``h = softplus(raw) + h_floor`` so the
conservative SWE residual can't trivially satisfy ``h -> 0``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pinn_bath.config import CheckpointCfg, DataCfg, OptimizerCfg, RunConfig
from pinn_bath.registry import Registry
from studies._runner import run_one
from studies.arch_scaling import CASE_PATHS

FORMS: tuple[str, ...] = ("primitive", "prim_cons", "conservative")
SEEDS: tuple[int, ...] = (0, 1, 2)
H_FLOORS: tuple[float, ...] = (0.0, 0.05)  # "without/with fix" comparison


def build_grid(
    *,
    seeds: tuple[int, ...] = SEEDS,
    h_floors: tuple[float, ...] = H_FLOORS,
    adam_epochs: int = 12_000,
    lbfgs_steps: int = 600,
    arch: str = "A1",
    budget: str = "small",
) -> list[RunConfig]:
    grid: list[RunConfig] = []
    for form in FORMS:
        for seed in seeds:
            for h_floor in h_floors:
                grid.append(
                    RunConfig(
                        case="exp1",
                        arch=arch,  # type: ignore[arg-type]
                        budget=budget,  # type: ignore[arg-type]
                        form=form,  # type: ignore[arg-type]
                        seed=seed,
                        h_floor=h_floor,
                        optimizer=OptimizerCfg(
                            adam_epochs=adam_epochs,
                            lbfgs_steps=lbfgs_steps,
                        ),
                        checkpoint=CheckpointCfg(every_epochs=1000, keep_last_k=2),
                        data=DataCfg(case_path=CASE_PATHS["exp1"], observations=["eta"]),
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
        f"Ablation {study_dir}: {len(grid)} configs -> "
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
    parser.add_argument("--study-dir", type=Path, default=Path("runs/ablation_forms"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_study(args.study_dir, device=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
