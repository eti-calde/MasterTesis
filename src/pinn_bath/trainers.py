"""Training loop for inverse SWE PINN (Adam + L-BFGS per protocolo §3.10).

The :class:`AdamLBFGSTrainer` orchestrates one full training run:

1. Build static collocation and observation point sets from the
   :class:`~pinn_bath.data.Case` (deterministic via ``cfg.seed``, ensuring
   A1/A2/A3 see the same data).
2. Run Adam for ``cfg.optimizer.adam_epochs`` steps.
3. Switch to L-BFGS for ``cfg.optimizer.lbfgs_steps`` steps.
4. Stream per-epoch losses to the :class:`~pinn_bath.tracking.RunRecorder`.

If a :class:`~pinn_bath.checkpoint.CheckpointManager` is provided, the trainer
(i) checks for a resume point at startup and continues from there, (ii) saves
periodic snapshots every ``cfg.checkpoint.every_epochs`` steps, and (iii)
installs SIGINT/SIGTERM handlers so an interrupt flushes a final checkpoint
before exit (S9).

A :class:`~pinn_bath.diagnostics.TrainingDiverged` raised mid-training is
caught: ``summary.json`` records ``status="diverged"`` and ``crash_dump.pt``
holds the last model state for offline inspection (S10).
"""

from __future__ import annotations

import time
from typing import Any

import torch

from pinn_bath.checkpoint import (
    CheckpointManager,
    SignalCheckpoint,
    build_state,
    restore_state,
)
from pinn_bath.config import RunConfig
from pinn_bath.data import Case
from pinn_bath.diagnostics import (
    TrainingDiverged,
    check_finite_loss,
    check_sanity_bounds,
    dump_crash_state,
    gradient_norm,
)
from pinn_bath.losses import (
    data_mse,
    flat_bed_loss,
    inflow_outflow_1d_loss,
    initial_condition_loss,
    pde_mse,
    periodic_bc_loss,
    positivity,
    swe_residual,
    tikhonov,
    wall_bc_loss,
)
from pinn_bath.metrics import baseline_rmse_zb, evaluate_zb
from pinn_bath.models.base import BaseModel
from pinn_bath.seed import set_rng_state
from pinn_bath.tracking import RunRecorder


class AdamLBFGSTrainer:
    """Two-phase Adam + L-BFGS trainer for inverse SWE PINN."""

    def __init__(
        self,
        model: BaseModel,
        case: Case,
        cfg: RunConfig,
        *,
        recorder: RunRecorder | None = None,
        checkpoint: CheckpointManager | None = None,
        device: str | torch.device = "cpu",
        n_collocation: int = 1000,
        n_observations: int | None = None,
        n_bc: int = 200,
        eval_log_every: int = 50,
        heartbeat_every_s: float = 60.0,
    ) -> None:
        self.cfg = cfg
        self.case = case
        self.recorder = recorder
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.eval_log_every = max(eval_log_every, 1)
        self.heartbeat_every_s = heartbeat_every_s
        self.n_bc = int(n_bc)

        n_obs = n_observations if n_observations is not None else (cfg.data.n_obs_points or 200)
        obs = case.sample_observations(
            seed=cfg.seed,
            n_obs=n_obs,
            fields=tuple(cfg.data.observations),
            noise_std=cfg.data.obs_noise_std,
        )
        self.obs_coords: dict[str, torch.Tensor] = {
            axis: obs[axis].to(self.device) for axis in ("x", "y", "t") if axis in obs
        }
        self.obs_values: dict[str, torch.Tensor] = {
            f: obs[f].to(self.device) for f in cfg.data.observations
        }

        coll = case.sample_collocation(
            seed=cfg.seed + 7919,
            n_coll=n_collocation,
            requires_grad=True,
        )
        self.coll_coords: dict[str, torch.Tensor] = {
            axis: v.to(self.device).detach().requires_grad_(True) for axis, v in coll.items()
        }

        # Auto-resume: load model weights now; optimizer + RNG restored once
        # the optimizers exist inside train().
        self._resumed_state: dict[str, Any] | None = None
        if checkpoint is not None and checkpoint.has_resume_point():
            self._resumed_state = checkpoint.load_resume()
            assert self._resumed_state is not None
            restore_state(self._resumed_state, model=self.model, optimizer=None, restore_rng=False)

    # --- Loss -----------------------------------------------------------

    def compute_loss(self) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + residual + composite loss.

        Raises :class:`~pinn_bath.diagnostics.TrainingDiverged` if any output
        field has ``h <= 0`` or non-finite values (S10).
        """
        out_coll = self.model(self.coll_coords)
        check_sanity_bounds(out_coll)
        friction, friction_params = self._friction_from_case()
        residuals = swe_residual(
            self.cfg.form,
            self.coll_coords,
            out_coll,
            spatial_dim=self.case.metadata.spatial_dim,
            has_t=self.case.metadata.has_t,
            friction=friction,
            friction_params=friction_params,
        )

        out_obs = self.model(self.obs_coords)
        eta_pred = out_obs["h"] + out_obs["zb"]

        w = self.cfg.loss
        dtype = eta_pred.dtype
        zero = torch.zeros((), dtype=dtype, device=self.device)
        L_data = data_mse(eta_pred, self.obs_values["eta"]) if "eta" in self.obs_values else zero
        L_data_u = data_mse(out_obs["u"], self.obs_values["u"]) if "u" in self.obs_values else zero
        # Optional wet-mask: when ``eps_wet > 0`` in the case constants,
        # weight each squared residual by a smooth wet indicator
        # ``sigmoid((h - eps_wet)/scale)``, so dry cells (where ``h``
        # cannot reach 0 because of the softplus output) don't inject
        # fictitious PDE residual into the loss. Disabled by default
        # (eps_wet=0 → wet=1 everywhere → identical to old behavior).
        eps_wet = float(self.case.metadata.constants.get("eps_wet", 0.0))
        if eps_wet > 0.0:
            wet_scale = float(self.case.metadata.constants.get("wet_scale", eps_wet))
            wet = torch.sigmoid((out_coll["h"].detach() - eps_wet) / wet_scale)
            residuals = {k: v * wet for k, v in residuals.items()}
        L_pde = pde_mse(residuals)
        L_pos = positivity(out_coll["h"])
        L_tikh = tikhonov(out_coll["zb"])
        L_bc = self._compute_bc_loss(dtype)
        L_ic = self._compute_ic_loss(dtype)

        total = (
            w.data * L_data
            + w.data_u * L_data_u
            + w.pde * L_pde
            + w.pos * L_pos
            + w.tikh * L_tikh
            + w.bc * L_bc
            + w.ic * L_ic
        )

        losses = {
            "data": float(L_data.item()),
            "data_u": float(L_data_u.item()),
            "pde": float(L_pde.item()),
            "pos": float(L_pos.item()),
            "tikh": float(L_tikh.item()),
            "bc": float(L_bc.item()),
            "ic": float(L_ic.item()),
            "total": float(total.item()),
        }
        return total, losses

    def _friction_from_case(self) -> tuple[str, dict[str, float]]:
        """Pick friction model + params from ``case.metadata.constants``.

        - ``"n_manning"`` present and > 0 → Manning-Strickler.
        - ``"kappa"`` present → linear drag (Angel et al. 2024).
        - else → no friction.
        """
        c = self.case.metadata.constants
        n = float(c.get("n_manning", 0.0))
        if n > 0.0:
            return "manning", {"n_manning": n}
        if "kappa" in c:
            return "linear_kappa", {
                "kappa": float(c["kappa"]),
                "eps_dry": float(c.get("eps_dry", 1.0e-4)),
            }
        return "none", {}

    def _compute_ic_loss(self, dtype: torch.dtype) -> torch.Tensor:
        """Mean squared error against the t=0 slice of the case fields.

        Only fired for transient cases when ``w.ic > 0``. Otherwise returns
        a zero scalar (and skips the model evaluation).
        """
        if not self.case.metadata.has_t or self.cfg.loss.ic == 0.0:
            return torch.zeros((), dtype=dtype, device=self.device)
        return initial_condition_loss(
            self.model,
            self.case,
            n_pts=None,  # use full spatial grid; cheap for typical N
            seed=self.cfg.seed + 31_337,
            device=self.device,
            dtype=dtype,
        )

    def _compute_bc_loss(self, dtype: torch.dtype) -> torch.Tensor:
        """Dispatch a BC penalty for the case's ``bc_type``.

        Implemented:

        - ``periodic`` (Tian dT10) → :func:`pinn_bath.losses.periodic_bc_loss`.
        - ``open_dirichlet`` (Exp 1 bump): always includes
          :func:`pinn_bath.losses.flat_bed_loss` (``z_b = 0`` outside the
          bump support). If ``h_down`` and ``q`` are in ``constants``, also
          adds :func:`pinn_bath.losses.inflow_outflow_1d_loss`.
        - ``closed_walls`` (Thacker basin Exps 2 / 5):
          :func:`pinn_bath.losses.wall_bc_loss`.

        Other types return 0 (the data and PDE terms handle them implicitly).
        """
        bc_type = self.case.metadata.bc_type
        seed = self.cfg.seed + 13_337
        if bc_type == "periodic":
            return periodic_bc_loss(
                self.model,
                self.case,
                n_bc=self.n_bc,
                seed=seed,
                device=self.device,
                dtype=dtype,
            )
        if bc_type == "open_dirichlet":
            loss = flat_bed_loss(
                self.model,
                self.case,
                n_pts=self.n_bc,
                seed=seed,
                device=self.device,
                dtype=dtype,
            )
            c = self.case.metadata.constants
            if "h_down" in c and "q" in c:
                loss = loss + inflow_outflow_1d_loss(
                    self.model, self.case, device=self.device, dtype=dtype
                )
            return loss
        if bc_type == "closed_walls":
            return wall_bc_loss(
                self.model,
                self.case,
                n_bc=self.n_bc,
                seed=seed,
                device=self.device,
                dtype=dtype,
            )
        return torch.zeros((), dtype=dtype, device=self.device)

    # --- Training loop --------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the full Adam + L-BFGS schedule, return final metrics."""
        opt = self.cfg.optimizer

        # Resume bookkeeping
        adam_start = 0
        lbfgs_start = 0
        resumed_phase: str | None = None
        if self._resumed_state is not None:
            resumed_phase = str(self._resumed_state.get("phase", "adam"))
            resumed_epoch = int(self._resumed_state["epoch"])
            if resumed_phase == "adam":
                adam_start = resumed_epoch + 1
            elif resumed_phase == "lbfgs":
                adam_start = opt.adam_epochs
                lbfgs_start = resumed_epoch - opt.adam_epochs + 1

        t_start = time.perf_counter()
        signal_ctx = SignalCheckpoint()

        try:
            with signal_ctx:
                adam_result = self._run_adam(
                    opt=opt,
                    adam_start=adam_start,
                    resumed_phase=resumed_phase,
                    signal_ctx=signal_ctx,
                    t_start=t_start,
                )
                if adam_result["status"] == "interrupted":
                    return adam_result
                adam = adam_result["optimizer"]
                t_adam = adam_result["t_phase"]

                lbfgs_result = self._run_lbfgs(
                    opt=opt,
                    lbfgs_start=lbfgs_start,
                    resumed_phase=resumed_phase,
                    signal_ctx=signal_ctx,
                    t_start=t_start,
                )
                if lbfgs_result["status"] == "interrupted":
                    return lbfgs_result
                t_lbfgs = lbfgs_result["t_phase"]

            # Final eval + summary (still inside try; outside the with block).
            with torch.no_grad():
                _, final_losses = self.compute_loss()
            final_metrics = evaluate_zb(self.model, self.case)
            baseline_metrics = baseline_rmse_zb(self.case)
            wall = time.perf_counter() - t_start
            self._persist_final(opt=opt, adam=adam, final_losses=final_losses)
            result: dict[str, Any] = {
                "status": "ok",
                "wall_time_s": round(wall, 4),
                "adam_time_s": round(t_adam, 4),
                "lbfgs_time_s": round(t_lbfgs, 4),
                "final_losses": final_losses,
                "final_metrics": final_metrics,
                "baseline_metrics": baseline_metrics,
            }
            if self.recorder is not None:
                self.recorder.write_summary(
                    status="ok",
                    adam_time_s=round(t_adam, 4),
                    lbfgs_time_s=round(t_lbfgs, 4),
                    final_losses=final_losses,
                    final_metrics=final_metrics,
                    baseline_metrics=baseline_metrics,
                )
            return result
        except TrainingDiverged as err:
            return self._finalize_diverged(err=err, t_start=t_start)

    # --- Adam phase -----------------------------------------------------

    def _run_adam(
        self,
        *,
        opt: Any,
        adam_start: int,
        resumed_phase: str | None,
        signal_ctx: SignalCheckpoint,
        t_start: float,
    ) -> dict[str, Any]:
        t_phase_start = time.perf_counter()
        adam = torch.optim.Adam(self.model.parameters(), lr=opt.adam_lr, betas=opt.adam_betas)
        if resumed_phase == "adam" and self._resumed_state is not None:
            if "optimizer" in self._resumed_state:
                adam.load_state_dict(self._resumed_state["optimizer"])
            if "rng" in self._resumed_state:
                set_rng_state(self._resumed_state["rng"])

        last_loss = float("nan")
        last_grad_norm = float("nan")
        last_heartbeat = t_start
        self._last_phase = "adam"
        self._last_epoch = adam_start

        for epoch in range(adam_start, opt.adam_epochs):
            self._last_epoch = epoch
            adam.zero_grad()
            total, losses = self.compute_loss()
            check_finite_loss(total, context=f"adam epoch {epoch}")
            total.backward()
            last_grad_norm = gradient_norm(self.model)
            adam.step()
            last_loss = losses["total"]

            now = time.perf_counter()
            if self.recorder is not None and (
                epoch % self.eval_log_every == 0 or epoch == opt.adam_epochs - 1
            ):
                self.recorder.log_epoch(
                    epoch=epoch,
                    phase="adam",
                    lr=opt.adam_lr,
                    grad_norm=last_grad_norm,
                    **{f"L_{k}": v for k, v in losses.items()},
                )
            if self.recorder is not None and (now - last_heartbeat) >= self.heartbeat_every_s:
                self.recorder.heartbeat(epoch=epoch, phase="adam", L_total=last_loss)
                last_heartbeat = now
            if self.checkpoint is not None and (
                (epoch + 1) % self.cfg.checkpoint.every_epochs == 0
            ):
                self.checkpoint.save(
                    build_state(epoch=epoch, phase="adam", model=self.model, optimizer=adam),
                    metric=last_loss,
                )
            if signal_ctx.interrupted:
                return self._finalize_interrupted(
                    epoch=epoch,
                    phase="adam",
                    model_opt=adam,
                    last_loss=last_loss,
                    t_start=t_start,
                )

        return {
            "status": "ok",
            "optimizer": adam,
            "t_phase": time.perf_counter() - t_phase_start,
            "last_loss": last_loss,
        }

    # --- L-BFGS phase ---------------------------------------------------

    def _run_lbfgs(
        self,
        *,
        opt: Any,
        lbfgs_start: int,
        resumed_phase: str | None,
        signal_ctx: SignalCheckpoint,
        t_start: float,
    ) -> dict[str, Any]:
        t_phase_start = time.perf_counter()
        if opt.lbfgs_steps == 0:
            return {"status": "ok", "t_phase": 0.0, "last_loss": float("nan")}

        lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=opt.lbfgs_lr,
            max_iter=opt.lbfgs_history,
            history_size=opt.lbfgs_history,
            line_search_fn="strong_wolfe",
        )
        if (
            resumed_phase == "lbfgs"
            and self._resumed_state is not None
            and "rng" in self._resumed_state
        ):
            set_rng_state(self._resumed_state["rng"])

        last_losses: dict[str, float] = {}
        last_heartbeat = t_start
        self._last_phase = "lbfgs"

        for step in range(lbfgs_start, opt.lbfgs_steps):
            self._last_epoch = opt.adam_epochs + step

            def closure() -> torch.Tensor:
                nonlocal last_losses
                lbfgs.zero_grad()
                total, losses = self.compute_loss()
                last_losses = losses
                check_finite_loss(total, context=f"lbfgs step {step}")  # noqa: B023
                total.backward()
                return total

            lbfgs.step(closure)
            last_loss = last_losses.get("total", float("nan"))

            now = time.perf_counter()
            if self.recorder is not None and (
                step % self.eval_log_every == 0 or step == opt.lbfgs_steps - 1
            ):
                self.recorder.log_epoch(
                    epoch=opt.adam_epochs + step,
                    phase="lbfgs",
                    lr=opt.lbfgs_lr,
                    **{f"L_{k}": v for k, v in last_losses.items()},
                )
            if self.recorder is not None and (now - last_heartbeat) >= self.heartbeat_every_s:
                self.recorder.heartbeat(
                    epoch=opt.adam_epochs + step,
                    phase="lbfgs",
                    L_total=last_loss,
                )
                last_heartbeat = now
            if self.checkpoint is not None and ((step + 1) % self.cfg.checkpoint.every_epochs == 0):
                self.checkpoint.save(
                    build_state(
                        epoch=opt.adam_epochs + step,
                        phase="lbfgs",
                        model=self.model,
                        optimizer=None,
                    ),
                    metric=last_loss,
                )
            if signal_ctx.interrupted:
                return self._finalize_interrupted(
                    epoch=opt.adam_epochs + step,
                    phase="lbfgs",
                    model_opt=None,
                    last_loss=last_loss,
                    t_start=t_start,
                )

        return {
            "status": "ok",
            "t_phase": time.perf_counter() - t_phase_start,
            "last_loss": last_losses.get("total", float("nan")),
        }

    # --- Finalize helpers -----------------------------------------------

    def _persist_final(
        self, *, opt: Any, adam: torch.optim.Optimizer, final_losses: dict[str, float]
    ) -> None:
        if self.checkpoint is None:
            return
        last_phase = "lbfgs" if opt.lbfgs_steps > 0 else "adam"
        last_epoch = (
            opt.adam_epochs + opt.lbfgs_steps - 1
            if opt.lbfgs_steps > 0
            else max(opt.adam_epochs - 1, 0)
        )
        last_opt = adam if opt.lbfgs_steps == 0 else None
        self.checkpoint.save(
            build_state(epoch=last_epoch, phase=last_phase, model=self.model, optimizer=last_opt),
            metric=final_losses["total"],
        )

    def _finalize_interrupted(
        self,
        *,
        epoch: int,
        phase: str,
        model_opt: torch.optim.Optimizer | None,
        last_loss: float,
        t_start: float,
    ) -> dict[str, Any]:
        """Save checkpoint + summary and return an interrupted result."""
        wall = time.perf_counter() - t_start
        if self.checkpoint is not None:
            self.checkpoint.save(
                build_state(epoch=epoch, phase=phase, model=self.model, optimizer=model_opt),
                metric=last_loss,
            )
        if self.recorder is not None:
            self.recorder.write_summary(
                status="interrupted",
                stopped_at_epoch=epoch,
                stopped_in_phase=phase,
                last_loss=last_loss,
            )
        return {
            "status": "interrupted",
            "wall_time_s": round(wall, 4),
            "stopped_at_epoch": epoch,
            "stopped_in_phase": phase,
            "last_loss": last_loss,
        }

    def _finalize_diverged(self, *, err: TrainingDiverged, t_start: float) -> dict[str, Any]:
        """Catch a TrainingDiverged: dump state + record status='diverged'."""
        wall = time.perf_counter() - t_start
        epoch = getattr(self, "_last_epoch", -1)
        phase = getattr(self, "_last_phase", "unknown")
        if self.checkpoint is not None:
            dump_crash_state(
                self.checkpoint.run_dir / "crash_dump.pt",
                model=self.model,
                epoch=epoch,
                phase=phase,
                last_loss=None,
                extra={"error": str(err)},
            )
        if self.recorder is not None:
            self.recorder.write_summary(
                status="diverged",
                error=str(err),
                stopped_at_epoch=epoch,
                stopped_in_phase=phase,
            )
        return {
            "status": "diverged",
            "wall_time_s": round(wall, 4),
            "stopped_at_epoch": epoch,
            "stopped_in_phase": phase,
            "error": str(err),
        }
