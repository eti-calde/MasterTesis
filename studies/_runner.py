"""Shared run executor for the study harnesses.

Given a :class:`~pinn_bath.config.RunConfig`, ``run_one`` wires up the
case, model, recorder, checkpoint manager, and trainer and returns the
trainer's result dict. The caller (an `arch_scaling.py` or
`ablation_forms.py` orchestrator) decides which configs to launch and
records their outcomes in the :class:`~pinn_bath.registry.Registry`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pinn_bath.checkpoint import CheckpointManager
from pinn_bath.config import RunConfig
from pinn_bath.data import Case
from pinn_bath.models import build
from pinn_bath.seed import set_seed
from pinn_bath.tracking import RunRecorder
from pinn_bath.trainers import AdamLBFGSTrainer


def _output_fields(case: Case) -> tuple[str, ...]:
    if case.metadata.spatial_dim == 2:
        return ("h", "u", "v", "zb")
    return ("h", "u", "zb")


def run_one(
    cfg: RunConfig,
    study_dir: Path | str,
    *,
    device: str = "cpu",
    n_collocation: int = 1000,
    n_observations: int | None = None,
    n_bc: int = 200,
    eval_log_every: int = 50,
    double_precision: bool = True,
    case: Case | None = None,
    obs_coords: dict | None = None,
    obs_values: dict | None = None,
) -> dict[str, Any]:
    """Execute one run end-to-end and return the trainer's result dict.

    Pass ``case`` to bypass ``Case.load(cfg.data.case_path)`` (e.g.,
    when the case is built programmatically — Exp 6 with the Angel
    adapter). Pass ``obs_coords`` + ``obs_values`` to override the
    trainer's random observation sampler with explicit sparse obs
    (Exp 6 sensor positions).
    """
    if case is None:
        case = Case.load(cfg.data.case_path)
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    model = build(
        cfg.arch,
        cfg.budget,
        spatial_dim=case.metadata.spatial_dim,
        has_t=case.metadata.has_t,
        output_fields=_output_fields(case),
        ff_seed=cfg.seed,
    )
    if double_precision:
        model = model.double()

    run_dir = Path(study_dir) / cfg.run_id
    cm = CheckpointManager(
        run_dir,
        keep_last_k=cfg.checkpoint.keep_last_k,
        keep_best=cfg.checkpoint.keep_best,
    )
    with RunRecorder(run_dir, cfg=cfg) as rec:
        trainer = AdamLBFGSTrainer(
            model,
            case,
            cfg,
            recorder=rec,
            checkpoint=cm,
            device=device,
            n_collocation=n_collocation,
            n_observations=n_observations,
            n_bc=n_bc,
            eval_log_every=eval_log_every,
            obs_coords=obs_coords,
            obs_values=obs_values,
        )
        return trainer.train()
