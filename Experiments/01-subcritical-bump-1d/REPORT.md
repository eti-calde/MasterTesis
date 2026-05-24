# Experiment 1 — Subcritical Bump (1D, Steady) REPORT

**Case**: Dazzi B1. Smooth parabolic bump on a flat bed, subcritical
flow ($Fr \in [0.50, 0.63]$), $q = 4.42$ m²/s, $h_\text{down} = 2$ m.
**Status**: **Pending — re-run on `azirafel` via `pinn_bath`**.

---

> **Re-baseline note (post-migration).** The legacy `pinn_inverse.py`
> orchestration was deleted in M6 of the legacy → pinn_bath migration.
> All historical cifras (the 12.9 mm initial baseline, the 5.80 mm
> sensitivity baseline, the density/noise/obstype sweep tables) were
> produced under the legacy code, **before** the bug fixes catalogued
> in the pre-azirafel batches:
> - Flat-bed prior $z_b = 0$ on $|x| > 2$ (batch #4, was missing).
> - Hard SWE-form default switched from `primitive_conservative` to
>   `primitive` (batch #7, ablation evidence).
> - Loss-weight defaults case-aware (batch M3).
>
> Those numbers are no longer reproducible from this repo and are
> superseded. Final cifras land in this file after the azirafel sweep.

---

## Setup

- Domain: $x \in [-10, 10]$ m, 500 grid points.
- Bathymetry: $z_b(x) = 0.2 - 0.05 x^2$ for $|x| \le 2$, else 0
  (200 mm parabolic bump centred at the origin).
- Steady-state Bernoulli ground truth; same configuration as
  `data/ground_truth_dazzi_B1.npz`.
- Inverse PINN: `pinn_bath` A1 architecture (small/medium/large
  budgets), `swe_form="primitive"` (post-May-17 ablation default).
- Composite loss:
  $\lambda_\text{data} L_\text{data} + \lambda_\text{PDE} L_\text{PDE}
  + \lambda_\text{pos} L_\text{pos} + \lambda_\text{BC} L_\text{BC}
  + \lambda_\text{Tikh} L_\text{Tikh}$,
  with $L_\text{BC}$ combining `flat_bed_loss` ($z_b = 0$ outside the
  bump support) and `inflow_outflow_1d_loss` ($h = h_\text{down}$ at
  outlet, $q = h \cdot u$ at endpoints, $z_b = 0$ at endpoints).

## Reproducible studies

| Study | Command | Configs |
|---|---|---|
| §5.1 baseline | `python -m studies.arch_scaling --study-dir runs/arch_scaling` | A1×A2×A3 × small×medium×large × 3 seeds × {Exp 1, 2, 3, 5} |
| §5.2 sensitivity | `python -m studies.exp1_sensitivity --study-dir runs/exp1_sensitivity` | density × noise × obstype × 3 seeds (30 configs) |
| §5.4 SWE-form ablation | `python -m studies.ablation_forms --study-dir runs/ablation_forms` | primitive × prim_cons × conservative × 3 seeds (9 configs) |

## Results

**TODO**. Populated from the azirafel sweep manifests
(`runs/<study>/manifest.jsonl`) and per-run `summary.json`. Aggregation
via `python -m studies.aggregate runs/<study>`. Expected output for
this REPORT:

- Baseline best/mean/std RMSE_zb per arch×budget cell (§5.1).
- Density / noise / obstype sweep tables (§5.2): RMSE_zb mean ± std at
  each sweep point.
- Comparison vs Ruppenthal 2026 (optimal-control + FEM) on the same
  case.

## Original contribution

The SWE-form ablation (see [`REPORT-ABLATION.md`](REPORT-ABLATION.md))
is claimed as an original contribution in
`Report/sections/05-comparacion-literatura.tex`: primitive form is the
most robust for **inverse** bathymetry recovery, **reversing**
Tian et al. 2025's forward-problem ranking. Re-running the ablation on
azirafel confirms (or refines) the cifras concretas.

## Files

- `ground_truth.py` — Bernoulli analytical solver.
- `data/ground_truth_*.npz` — Cases consumed by `pinn_bath`
  (`Case.load(...)`).
- `figures/` — historical figures (rebuildable from the new sweeps via
  `studies.aggregate` + matplotlib).
- `results/*.log` — historical legacy training logs (read-only;
  retained as audit trail of the pre-migration runs).
