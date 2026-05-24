# Experiment 6 — Real Sensor Data (Angel et al. 2024 Hamburg flume)

**Reference**: Angel et al. (2024) "Bathymetry reconstruction from wave-gauge
measurements via adjoint optimisation", *Coast. Eng.*; dataset
[`datasets/angel2024/processed/angel2024_flume.npz`](../datasets/angel2024/processed/angel2024_flume.npz)
(mean of 20 runs + measured ground-truth bathymetry).

## What's new vs Experiments 1–5

| Aspect | Exps 1–5 (synthetic) | Experiment 6 (Angel) |
|---|---|---|
| Ground truth | Analytical or FV simulation | **Real measurement** (probe-scanned bed) |
| Observations | $\eta(x, t)$ from numerical solver | Real wave-gauge η(t) at 4 fixed sensors |
| Noise | Optional Gaussian | Real instrument noise (signal-to-noise ≈ 30:1) |
| Bathymetry | Smooth (Gaussian / paraboloid / cylinders) | Smooth Gaussian-like bump, $H = 200$ mm |
| BCs | Analytical / synthesised | Real wavemaker forcing (windowed t ∈ [40, 60] s) |
| Drag | Manning $n$ or none | Linear $\kappa \cdot u$ ($\kappa = 0.2$, fit by Angel) |
| Outcome | Bathymetry recovered | **Negative result** — bathymetry NOT recovered |
| Purpose | Method development | First real-data stress test → motivates future work |

## Physical setup (Angel §2)

**Flume geometry** (Hamburg WBL):
- Length: $L_x = 13.5$ m, width 0.4 m, still-water depth $H_\text{rest} = 0.3$ m.
- Bathymetry: a single Gaussian-like bump centred at **$x \approx 3.99$ m**,
  height $H = 200$ mm. Bed elevation $z_b \ge 0$ everywhere; the
  measured profile is published in `flume.npz["zb_true"]` on 500 points.
- The wavemaker at $x = 0$ generates regular waves; sensors at
  $x = \{1.5, 3.5, 5.5, 7.5\}$ m (S1, S2, S3, S4) record $\eta(t)$ at
  100 Hz. Wavelength $\lambda \approx 1.9$ m → $kh \approx 1.3$
  (intermediate depth; SWE has $\approx 10\,\%$ inherent model error).
- Linear bottom drag $\kappa = 0.2$ (Angel-fitted).

**Window**:
- The wavemaker is silent until $t \approx 34$ s; the established-wave
  window is $t \in [40, 60]$ s (20 s). Data is decimated 100 Hz → 10 Hz,
  re-zeroed to window start (SWE + linear drag are time-translation
  invariant).

**Inverse problem statement** (see
[`src/pinn_bath/datasets/angel.py`](../../src/pinn_bath/datasets/angel.py)):
Given $\eta(x_{S_k}, t)$ for selected sensors $S_k$ on the windowed
grid, recover $z_b(x)$ subject to:
- SWE (transient 1D, linear drag $\kappa$) as a **soft** PDE penalty,
- $z_b(x) \ge 0$ (soft positivity via `pos` loss weight),
- the inlet sensor S1 included in the observation mask (also soft).

## Sensor placement

- **S1 (x = 1.5 m)**: **soft** inlet Dirichlet η(t) — included as one
  of the sensor columns in `AngelObservations.obs_coords/obs_values`,
  fit by the same MSE data loss as interior sensors. *Not* a hard BC.
  Contrast Angel's adjoint method, which imposes the inlet as a hard
  BC through the forward solver.
- **S2 (x = 3.5 m)**: canonical interior observation. The bump peak
  is at $x \approx 3.99$ m — **no sensor sits on the bump itself**.
- **S3 / S4 (x = 5.5, 7.5 m)**: optional additional sensors via the
  `obs_sensors` kwarg of `case_from_angel_flume(...)`.
- **Grid alignment**: $\text{Nx} = 136$ → $\Delta x = 100$ mm → all
  four sensors land exactly on grid nodes (snap error 0). Other valid
  choices: 271 ($\Delta x = 50$ mm), 541 ($\Delta x = 25$ mm).
  `case_from_angel_flume` raises if `snap_tol_m = 1` mm is exceeded.

## Method

The canonical pipeline is `pinn_bath` end-to-end. The adapter
[`case_from_angel_flume`](../../src/pinn_bath/datasets/angel.py)
returns `(Case, AngelObservations)`; the `AngelObservations` tensors
are passed to
[`AdamLBFGSTrainer`](../../src/pinn_bath/trainers.py) via the
`obs_coords` / `obs_values` override, bypassing the trainer's default
random sampler so the sensors stay at their exact positions.

- Physics: `swe_residual(... friction="linear_kappa", friction_params={"kappa": 0.2, "eps_dry": 1e-4})`
  in place of Manning).
- **No** initial-condition loss (the window starts mid-experiment, no
  known IC).
- **No** explicit BC loss (the flume has soft inlet at S1 via the
  observations + an open outlet — `bc_type="soft_inlet_outlet"`).
- **No** dry-cell mask ($H_\text{rest} = 0.3$ m $\gg$ bump 0.2 m →
  never dry).
- Soft positivity prior $z_b \ge 0$ via `pos` loss weight.

## Configurations swept ([`studies/exp6_run_matrix.py`](../../studies/exp6_run_matrix.py))

| Config | seeds | $\sigma_x$ | $\lambda_\text{PDE}$ | obs sensors | result (TODO post-azirafel) |
|---|---|---|---|---|---|
| canonical | 3 (0, 1, 2) | 2 (A1 default) | 1 | S2 only | expected: bump missed (negative finding reproduced) |

Exploratory runs not in the harness (see REPORT.md): stronger Fourier
bandwidth ($\sigma_x = 6$), stronger PDE weight ($\lambda_\text{PDE} =
5$), and the 3-sensor variant (S2 + S3 + S4). Adding these to
`exp6_run_matrix.py` is a defence TODO.

## Outcome

**Documented negative result**. With sensors off the bump and a soft PDE
penalty, the PINN fits the observed $\eta$ at the sensors but cannot
recover the bump — the $\eta = h + z_b$ equifinality is admitted by the
loss landscape. Angel's adjoint method, which uses the SWE as a hard
forward solver, succeeds on the same 2-sensor data (NRMSE 10–14 %). The
PINN-vs-adjoint gap is a clean motivation for future hard-constrained
or Lagrangian-augmented PINN formulations.

See [`REPORT.md`](REPORT.md) for the full result tables, the diagnosis,
and the comparison with Angel's adjoint method.

## Files (post-migration)

All Exp 6 code now lives under `pinn_bath` and `studies/`:

- [`src/pinn_bath/datasets/angel.py`](../../src/pinn_bath/datasets/angel.py)
  — `case_from_angel_flume(...)`: loads the flume `.npz`, windows +
  decimates + snaps sensors, returns `(Case, AngelObservations)`.
  Replaces the legacy `data_angel.py`.
- [`src/pinn_bath/losses/residual.py`](../../src/pinn_bath/losses/residual.py)
  — `friction="linear_kappa"` dispatch (`κ·u/(h+ε)`) replaces the
  legacy `physics_angel.swe_residual_angel`. Auto-detected from
  `case.constants["kappa"]`.
- [`studies/exp6_run_matrix.py`](../../studies/exp6_run_matrix.py) —
  the canonical sweep (3 seeds × A1/small × S2-only). Replaces the
  legacy `run_matrix.py` + `run_poc.py`.
- `REPORT.md` — findings, comparison with Angel adjoint, defence notes.

The legacy files (`pinn_angel.py`, `data_angel.py`,
`physics_angel.py`, `run_matrix.py`, `run_poc.py`, `bench_epoch.py`,
`analyze_limitation.py`) were removed in the M6 nuke.

## References

- Angel et al. (2024) *Coast. Eng.* — adjoint reconstruction on the
  same flume, defines the dataset.
- Pre-flight bug log entries #8 (sensor snap), #14 (soft-BC honesty),
  #15 (reproducibility), #20 (this file).
