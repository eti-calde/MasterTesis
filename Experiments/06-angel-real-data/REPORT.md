# Experiment 6 — Real Sensor Data (Angel Hamburg flume) REPORT

**Case**: Angel et al. (2024) Hamburg WBL flume. Real wave-gauge
$\eta(t)$ at 4 sensors over a Gaussian-like bump ($H = 200$ mm). First
real-data stress test of the soft-PDE PINN bathymetry inversion.
**Status**: **Pending — re-run on `azirafel` via `pinn_bath`** (the
**negative result** is expected to reproduce).

---

> **Re-baseline note (post-migration).** Historical cifras
> (~12% peak recovery, ~34 mm $z_b$ RMSE on the "stronger-lever
> configs", NRMSE ~22% on the informative span) came from the legacy
> `pinn_angel.py` + `data_angel.py` + `run_matrix.py` (all deleted in
> M6). They predate:
> - Sensor-snap guard tightened to 1 mm (batch #8 — the default
>   `nx=120` silently misplaced sensor S2 by 42 mm; the new
>   `case_from_angel_flume(...)` default `nx=136` snaps exactly).
> - Soft-vs-hard BC honesty in the docs (batch #14).
> - Reproducibility cleanup in this very REPORT (batch #15 — removed
>   the unreproducible "stronger-lever" cifras claim).
>
> The new path goes through `pinn_bath.datasets.case_from_angel_flume`
> + `studies/exp6_run_matrix.py`. Final cifras land here after
> azirafel.

---

## Setup

- **Data**: `datasets/angel2024/processed/angel2024_flume.npz` (mean
  of 20 Hamburg-flume runs + measured ground-truth bathymetry).
- **Window**: established-wave $t \in [40, 60]$ s, decimated 100 Hz →
  10 Hz, re-zeroed.
- **Spatial grid**: $\text{Nx} = 136$ over $x \in [1.5, 15]$ m
  ($dx = 100$ mm) → all four sensors snap exactly to grid nodes.
- **Sensors**: S1 ($x = 1.5$ m) = **soft** inlet Dirichlet $\eta(t)$
  (MSE penalty via the same observation mask as S2; **not** a hard BC,
  contrast Angel's adjoint which imposes the inlet hard); S2
  ($x = 3.5$ m) = canonical interior observation. The bump peak is at
  $x \approx 3.99$ m — *no sensor sits on the bump*.
- **Physics**: 1D transient SWE with linear bottom drag
  $\kappa = 0.2$ (auto-detected from `case.constants["kappa"]` and fed
  into `swe_residual` via the `linear_kappa` friction model).
- **Priors**: $z_b \ge 0$ everywhere (soft positivity).

## Reproducible study

```bash
python -m studies.exp6_run_matrix --study-dir runs/exp6
```

Canonical config: 3 seeds × A1/small, S2-only obs, `sigma_x=2`
(default A1), `lambda_pde=1`.

## Result (expected: the negative finding)

**TODO**. Surface $\eta$ should still fit excellently at the sensors,
but the **bathymetry is not recovered** — the soft PDE penalty admits
the $\eta = h + z_b$ equifinality when sensors are off the bump.
Angel's adjoint method (hard SWE forward solver) succeeds on the same
2-sensor data; the gap is a clean motivation for future
hard-constrained or Lagrangian-augmented PINN formulations.

Earlier exploratory runs (NOT in the current harness — added as TODO
for the defence) suggested the failure mode is robust across
$\sigma_x \in \{2, 6\}$, $\lambda_\text{PDE} \in \{1, 5\}$, and
sensor subsets {S2, S2+S3+S4}; azirafel re-run with the full grid
will confirm.

## Diagnosis (still applies)

With sensors *off* the bump and a soft PDE penalty, the network fits
the observed wave by adjusting $h$ while leaving $z_b \approx$ flat —
a solution that satisfies data **and** approximately satisfies the
soft residual. Angel's adjoint succeeds on the same 2-sensor data
because the SWE is a **hard** forward constraint, so $z_b$ is the
only free field that can explain the wave dynamics. $kh \approx 1.3$
(intermediate depth) adds ~10% inherent SWE model error on top.

## Files

- Adapter: [`src/pinn_bath/datasets/angel.py`](../../src/pinn_bath/datasets/angel.py)
- Study: [`studies/exp6_run_matrix.py`](../../studies/exp6_run_matrix.py)
- Detail: [`experiment6-detail.md`](experiment6-detail.md)
- `figures/` — historical visualisations (rebuildable from the new
  sweep).
- `results/*.log` — legacy training logs (audit trail).
