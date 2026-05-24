# Experiment 3 — Two cylinders 2D Report

**Date**: 2026-05-23 (refactor; PINN results pending).
**Case**: Ruppenthal & Kuzmin (2026) §7.2 — two vertical-walled cylinders.
**Status**: **Pending** — case spec finalized, ground truth regenerated; PINN
inversion results will populate this file once the §5.1 architecture
scaling barrido is executed on `azirafel`.

---

> **Solver fix note (2026-05-24).** El solver FV-HLL previo computaba el
> término fuente $-g\,h\,\partial_x z_b$ por diferencias centradas sobre la
> batimetría indicador-discontinua de Ruppenthal §7.2. En los bordes
> verticales del cilindro mayor ($H = 0.3$\,m, $\Delta x = 0.5$\,m) esto
> generaba un gradiente espurio $\sim H/(2\Delta x) = 0.3$ m/m, equivalente
> a una fuerza local $g\,h\,\partial_x z_b \approx 5$ m/s² (comparable a la
> gravedad), que contaminaba la "verdad de referencia" con oscilaciones
> numéricas y propagaba a través del PINN como sesgo en la inversión.
>
> Reemplazado por reconstrucción hidrostática de Audusse
> ([2004](https://doi.org/10.1137/S1064827503431090)), well-balanced para
> topografía discontinua: el "lake-at-rest" ($u = v = 0$,
> $\eta = h + z_b = \mathrm{const}$) se preserva a precisión de máquina
> incluso sobre el indicador agudo (test `test_lake_at_rest_one_step` →
> `max |hu|` $< 10^{-12}$). El dataset
> `data/ground_truth_cylinders.npz` fue regenerado y debe usarse para los
> resultados §5.1; la versión pre-fix queda obsoleta.

> **BC fix (2026-05-24).** El solver previo usaba `np.pad(mode='edge')`
> en las cuatro caras → Neumann global. Con la IC uniforme
> $(u, v) = (2.21, 2.21)$ no había mecanismo de frontera para mantener
> la corriente entrante: el flujo decae con el tiempo (el `u range`
> bajaba a $[1.79, 2.59]$ a $t = 60$ s) y los wakes detrás de los
> cilindros quedaban en régimen transitorio. Reemplazado por Dirichlet
> de inflow en $x = 0$ y $y = 0$ (prescribiendo el estado IC vía celdas
> fantasma) + outflow zero-gradient en las opuestas. Con el fix, `u
> range` se mantiene en $[1.85, 2.59]$ y la `eta std` temporal cae a la
> mitad ($0.031 \to 0.015$), señal de que el flujo alcanza el patrón
> cuasi-estacionario esperado.

> **BathymetryNet fix (2026-05-24).** `BathymetryNet2D` antes hacía
> `softplus(zb_raw) - 0.1`, permitiendo $z_b \in [-0.1, \infty)$ —
> valores negativos físicamente imposibles para los cilindros de
> Ruppenthal ($z_b \in \{0, 0.2, 0.3\}$ m). Cambiado a
> `softplus(zb_raw)` → rango $[0, \infty)$. El residuo SWE también
> se reescribió en forma well-balanced ($g \, \partial_x \eta$ con
> $\eta = h + z_b$ calculado una sola vez) — matemáticamente idéntico
> al anterior por linealidad de AD, pero más claro y alineado con
> el solver FV de referencia (Audusse).

---

## Summary

Once the barrido runs, this report will collect the inverse PINN results
for the §5.1 architecture scaling study on this case: three architectures
(A1, A2, A3) at three parameter budgets (small ≈ 20K, medium ≈ 100K,
large ≈ 500K) with three seeds. For each cell of the grid we will report
$z_b$ RMSE, NRMSE, $R^2$, wall-time and peak VRAM, with mean ± std and
bootstrap CIs aggregated by `studies/aggregate.py`.

## Setup

| Parameter | Value |
|---|---|
| Domain | $[0, 25]^2$ m² |
| Grid | $50 \times 50$ cell centers ($\Delta x = \Delta y = 0.5$ m) |
| Cylinder 1 | center $(8, 8)$ m, radius $4$ m, height $0.2$ m |
| Cylinder 2 | center $(15, 15)$ m, radius $2$ m, height $0.3$ m |
| Bathymetry shape | vertical-walled indicators (no smoothing) |
| Initial free surface | $\eta(x, y, 0) = h + z_b = 2$ m uniform |
| Initial velocity | $\mathbf{v}(x, y, 0) = (2.21, 2.21)$ m/s uniform |
| Simulation time | $T = 60$ s, $\Delta t = 10^{-2}$ s |

These parameters are pinned to `Report/sections/07-apendice-casos-sinteticos.tex` (A.3).

## Ground truth

Reference solution from a FV-HLL solver on the $50 \times 50$ cartesian
grid (LeVeque 2002 formulation, implemented in `ground_truth.py`).
Regenerate with:

```bash
cd Experiments/03-two-cylinders-2d
python generate_and_plot.py
```

This writes `data/ground_truth_cylinders.npz` in the unified
`pinn_bath.data.Case` schema, along with `figures/bathymetry.png` and
`figures/ground_truth_snapshots.png`.

## PINN pipeline

Inversion is performed by the canonical harness, not by a per-experiment
script. Each (arch, budget, seed) configuration is materialized as a
`RunConfig` via `studies/arch_scaling.py:build_grid()` and trained by
`pinn_bath.trainers.AdamLBFGSTrainer` (12 000 Adam epochs + 600 L-BFGS
steps; protocol §3.10).

Launch (on azirafel):

```bash
python -m studies.arch_scaling --study-dir runs/arch_scaling --device cuda
```

The runner writes one summary per config under
`runs/arch_scaling/<run_id>/summary.json` and an append-only manifest at
`runs/arch_scaling/manifest.jsonl`. Re-launching skips completed configs
(idempotency) and resumes interrupted ones from the last checkpoint.

## Results

> **Pendiente.** Las cifras de RMSE/NRMSE/$R^2$ y la tabla de comparación
> A1 vs A2 vs A3 por presupuesto se llenarán desde
> `runs/arch_scaling/<run_id>/summary.json` con `studies/aggregate.py` una
> vez ejecutado el barrido en azirafel.

## Files

- `experiment3-detail.md` — case specification.
- `ground_truth.py` — 2D FV-HLL solver.
- `generate_and_plot.py` — dataset + figure regeneration.
- `data/ground_truth_cylinders.npz` — unified-schema dataset.
- `figures/bathymetry.png` — true $z_b$ map.
- `figures/ground_truth_snapshots.png` — $\eta$ and $|\mathbf{v}|$ snapshots.
