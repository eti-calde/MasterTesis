# Experiment 5 — Thacker Paraboloid (2D, Transient) REPORT

**Case**: 2D axisymmetric Thacker paraboloid (SWASHES §4.2.2). Closed
basin, true bowl-shaped bathymetry, wetting-drying.
**Status**: **Pending — re-run on `azirafel` via `pinn_bath`**.

---

> **Re-baseline note (post-migration).** Historical cifras (~50 mm
> zb_rmse_wet plateau across rounds; v4 improvements via rim z_b
> supervision) came from the legacy `pinn_inverse.py` + the
> `improve_baseline_v[1-4]` diagnostic loop (deleted in M6). The
> production path now uses `pinn_bath` with case-aware weights
> (`ic=500, bc=10`) plus the trainer's optional wet-mask via
> `case.constants["eps_wet"]`. Final cifras land here after azirafel.

---

## Setup

- Domain: $[0, 4]^2$ m, $t \in [0, T]$ with $T \approx 2.243$ s
  ($\sim 3$ periods).
- Bathymetry (true bowl, below datum): $z_b(x, y) = h_0 (r/a)^2 - h_0$
  with $r = \sqrt{(x - 2)^2 + (y - 2)^2}$, $a = 1$, $h_0 = 0.1$.
- Closed basin ($u = v = 0$ at walls), known IC.
- Inverse PINN: `pinn_bath` A1/small (or larger), 2D transient SWE
  residual, IC loss + `wall_bc_loss`, eta+u+v observations.

## Reproducible studies

| Study | Command |
|---|---|
| §5.1 baseline | `python -m studies.arch_scaling --study-dir runs/arch_scaling` |
| §5.3 N_t sweep | `python -m studies.exp5_n_t_sweep --study-dir runs/exp5_n_t` |

`exp5_n_t_sweep` covers N_t ∈ {1, 2, 4, 8, 30} × 3 seeds = 15 configs.

## Results

**TODO**. Filled from azirafel sweeps. Expected:

- Bowl recovery on the ever-wet disk vs. full domain (RMSE_zb_wet,
  RMSE_zb_all).
- N_t sweep: does the "2-to-4 snapshots suffice" finding from Exps 2/4
  extend to the 2D axisymmetric case?
- Per-arch×budget cell results from §5.1.

## Files

- `ground_truth.py` — 2D Thacker analytical (axisymmetric).
- `data/ground_truth_thacker3d.npz` — Case (`bc_type="closed"`).
- `figures/` — historical visualisations (rebuildable from the new
  sweeps).
- `results/*.log` — legacy training logs (audit trail).
