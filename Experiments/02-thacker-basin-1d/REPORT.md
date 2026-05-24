# Experiment 2 — Thacker Parabolic Basin (1D, Transient) REPORT

**Case**: Dazzi T1 / Thacker planar-surface oscillation in a parabolic
basin. Closed-walls BC, wetting-drying at the moving shoreline, no
flow-based BC.
**Status**: **Pending — re-run on `azirafel` via `pinn_bath`**.

---

> **Re-baseline note (post-migration).** The historical cifras (eta+u
> baseline 24 mm; eta-only failure 180 mm; the snapshot-vs-time-series
> finding "N_t=4 achieves ~5 mm") came from the legacy
> `pinn_inverse.py` + `snapshot_vs_timeseries.py` orchestration
> (deleted in M6). They predate:
> - Wet/dry mask on the SWE residual (batch #11 — softplus($h$) never
>   reaches 0, leaking fictitious force in dry cells; partial cause of
>   the "basin too shallow" failure mode at $N_t = 10/40$).
> - Case-aware loss weights (M3: bc=10 for closed-walls,
>   ic=100 for the known initial state).
>
> Both fixes are now in the pinn_bath path. Numbers regenerate on
> azirafel and land here.

---

## Setup

- Domain: $x \in [-2, 2]$ m, $t \in [0, T]$ with one full period
  $T \approx 2.006$ s.
- Bathymetry (concave parabola): $z_b(x) = 0.5(x^2 - 1)$,
  $z_b(0) = -0.5$ m (lowest point), shoreline at rest at $x = \pm 1$.
- Closed basin ($u = 0$ at walls), known IC.
- Inverse PINN: `pinn_bath` A1/small, 1D transient SWE residual,
  closed-walls BC dispatch, IC loss, wet-cell mask via
  `case.constants["eps_wet"]` (opt-in).

## Reproducible studies

| Study | Command |
|---|---|
| §5.1 baseline | `python -m studies.arch_scaling --study-dir runs/arch_scaling` |
| §5.3 N_t snapshot-vs-series | `python -m studies.exp2_n_t_sweep --study-dir runs/exp2_n_t` |

`exp2_n_t_sweep` covers N_t ∈ {1, 2, 4, 10, 40} × 3 seeds = 15 configs.

## Results

**TODO**. Filled from the azirafel sweeps. Expected:

- Baseline RMSE_zb on the ever-wet region (eta-only, eta+u variants).
- N_t sweep table: RMSE_zb mean ± std at each N_t.
- Discussion of whether the bug fixes change the "N_t=4 suffices"
  claim — historically attributed to temporal richness alone breaking
  the equifinality.

## Files

- `ground_truth.py` — Thacker analytical solution.
- `data/ground_truth_thacker_T1.npz` — Case (`bc_type="closed"`).
- `figures/` — historical figures.
- `results/*.log` — legacy training logs (kept as audit trail).
