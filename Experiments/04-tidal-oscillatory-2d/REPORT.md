# Experiment 4 — Tian dT10 Report

**Date**: 2026-05-23 (refactor; PINN results pending).
**Case**: Tian et al. (2025) dT10 — variable-topography "tidal" problem
(see naming caveat in `experiment4-detail.md`).
**Status**: **Pending** — case spec finalized, ground truth regenerated,
periodic BC integrated in `pinn_bath`; full PINN inversion results will
populate this file once a dedicated barrido is executed.

## Summary

Once the run completes, this report will collect inverse PINN results on
the dT10 case under periodic BCs. The smoke run already validates the
end-to-end pipeline: 200 Adam epochs of A1 small drive `loss_bc` from
$1.7\times 10^{-2}$ to $4.5\times 10^{-5}$ (the model learns to be
periodic) and `loss_total` from 64 to $1.2\times 10^{-3}$.

## Setup

| Parameter | Value |
|---|---|
| Domain | $(x, y) \in [-2, 2]^2$ m |
| Grid | $100 \times 100$ ($\Delta x = \Delta y = 0.04$ m) |
| Topography | $z = 1 + 0.01\,\cos(\pi x / 2)\,\cos(\pi y / 2)$ m, range $[0.99, 1.01]$ |
| IC | $h(0) = z$, $u = v = 0$ → $\eta(0) = 2 z$ |
| BC | periodic in $x$ and $y$ |
| Simulation time | $T = 0.5$ s, $n_\text{save} = 51$ snapshots |

These parameters are pinned to `Report/sections/07-apendice-casos-sinteticos.tex` (A.4).

## Ground truth

Reference solution from a FV-HLL solver with periodic ghost cells
(`ground_truth.py`). Tian uses an entropy-stable scheme ES1 (Fjordholm
2011) on a much finer grid; both are convergent SWE discretizations.
Regenerate with:

```bash
cd Experiments/04-tidal-oscillatory-2d
python generate_and_plot.py
```

This writes `data/ground_truth_dT10.npz` (loadable with
`pinn_bath.data.Case.load`) and the three figures listed below.

## PINN pipeline

Driven by `pinn_bath.trainers.AdamLBFGSTrainer` with periodic BC support:

- `case.metadata.bc_type = "periodic"` → the trainer's `_compute_bc_loss`
  dispatches to `pinn_bath.losses.periodic_bc_loss`, which penalizes
  $f(\text{lo}, *, t) - f(\text{hi}, *, t)$ for each spatial axis on
  random sampled pairs.
- `cfg.loss.bc` controls the weight (default 0; use ~10 for dT10).
- A smoke run is available under `runs/smoke_bc/` for sanity checks.

Exp 4 is **not part of the §5.1 architecture-scaling grid** (canonical
grid is Exp 1, 2, 3). It is launched from an ad-hoc `RunConfig` with the
periodic case path and `bc=10.0`:

```python
from pinn_bath.config import CheckpointCfg, DataCfg, LossWeights, OptimizerCfg, RunConfig
from studies._runner import run_one

cfg = RunConfig(
    case="exp4", arch="A1", budget="small", seed=0,
    loss=LossWeights(data=10, pde=1, pos=10, tikh=0, bc=10),
    optimizer=OptimizerCfg(adam_epochs=12_000, lbfgs_steps=600),
    checkpoint=CheckpointCfg(every_epochs=1000, keep_last_k=2),
    data=DataCfg(
        case_path="Experiments/04-tidal-oscillatory-2d/data/ground_truth_dT10.npz",
        observations=["eta"], n_obs_points=400,
    ),
)
run_one(cfg, "runs/exp4_full", device="cuda")
```

## Results

> **Pendiente.** Las cifras de RMSE/NRMSE/$R^2$ de $z_b$, comparación
> A1/A2/A3 a presupuestos pequeño/medio/grande, y la curva de
> aprendizaje de `L_bc` se llenarán desde `runs/exp4_*/summary.json` una
> vez ejecutado el barrido en azirafel.

## Files

- `experiment4-detail.md` — case specification.
- `ground_truth.py` — 2D FV-HLL solver with periodic ghost cells.
- `generate_and_plot.py` — dataset + figures (reproduces Tian Figure 5).
- `data/ground_truth_dT10.npz` — unified-schema dataset.
- `figures/bathymetry.png` — true $z$ map.
- `figures/ground_truth_snapshots.png` — $h$, $u$, $v$ at $t = 0, 0.25, 0.5$ s.
- `figures/relaxation_timeseries.png` — $\eta$ time series at center / corner / mid-edge.
