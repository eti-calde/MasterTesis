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

**Inverse problem statement** ([`data_angel.py`](data_angel.py)):
Given $\eta(x_{S_k}, t)$ for selected sensors $S_k$ on the windowed grid,
recover $z_b(x)$ subject to:
- SWE (transient 1D, linear drag) as a **soft** PDE penalty,
- $z_b(x) \ge 0$ (soft positivity),
- $z_b(x = x_\text{ends}) = 0$ (the flume floor is flat at the
  approach/downstream ends; same idiom as Exp 1 `loss_bc`).

## Sensor placement (`data_angel.py:32-36`)

- **S1 (x = 1.5 m)**: **soft** inlet Dirichlet η(t) — included in
  `x_obs_indices`, fit by the same MSE data loss as interior sensors.
  *Not* a hard BC. See discussion in `REPORT.md` and
  [`pinn_angel.py`](pinn_angel.py).
- **S2 (x = 3.5 m)**: canonical interior observation. The bump peak is
  at $x \approx 3.99$ m — **no sensor sits on the bump itself**.
- **S3 / S4 (x = 5.5, 7.5 m)**: optional additional sensors for the
  multi-sensor variant (currently not in `run_matrix.py`; see #15 in
  the pre-flight bug log).
- **Grid alignment**: $\text{Nx} = 136$ → $\Delta x = 100$ mm → all
  sensors land exactly on grid nodes (snap error 0). Other valid choices
  are 271 (Δx = 50 mm) and 541 (Δx = 25 mm). `load_angel_windowed`
  raises if `snap_tol_m = 1$ mm` is exceeded.

## Method (`pinn_angel.py`)

`AngelInversePINN` subclasses [`ThackerInversePINN`](../02-thacker-basin-1d/pinn_inverse.py)
from Exp 2 (two-network architecture, Fourier features, sparse
observation mask, Adam + L-BFGS). Only `compute_loss` is overridden:

- Physics: `swe_residual_angel` (linear drag $\kappa \cdot u/(h+\epsilon)$
  in place of Manning); see [`physics_angel.py`](physics_angel.py).
- **No** initial-condition loss (the window starts mid-experiment).
- **No** wall-BC loss (the flume has inflow at S1 + open outlet).
- **No** dry-cell loss ($H_\text{rest} = 0.3$ m $\gg$ bump 0.2 m → never dry).
- Adds two physical priors to break the $\eta = h + z_b$ equifinality:
  $z_b \ge 0$ (soft) and $z_b(\text{ends}) = 0$.

## Configurations swept (`run_matrix.py`)

| Config | seeds | $\sigma_x$ | $\lambda_\text{PDE}$ | obs sensors | result |
|---|---|---|---|---|---|
| canonical | 3 (0, 1, 2) | 2 | 1 | S2 only | ~34 mm $z_b$ RMSE (bump missed) |

Exploratory runs not in the harness (see REPORT.md): stronger Fourier
bandwidth ($\sigma_x = 6$), stronger PDE weight ($\lambda_\text{PDE} = 5$),
and 3-sensor variant (S2 + S3 + S4). Adding these to `run_matrix.py`
is a thesis-defence TODO.

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

## Files

- `data_angel.py` — load, window, decimate flume `.npz`; snap sensors to
  grid; NRMSE helper. (Suffix `_angel` because these are
  Angel-dataset-specific; the project's convention is `pinn_inverse.py`
  + `ground_truth.py` + `generate_and_plot.py` for synthetic exps.)
- `physics_angel.py` — `swe_residual_angel` with linear $\kappa$ drag.
- `pinn_angel.py` — `AngelInversePINN` subclass of `ThackerInversePINN`.
- `run_matrix.py` — canonical sweep harness.
- `run_poc.py` — single-config CLI runner (uses `argparse`).
- `bench_epoch.py` — per-epoch wall-time benchmark across grid sizes.
- `analyze_limitation.py` — diagnostics for the negative result.
- `REPORT.md` — findings, comparison with Angel adjoint, defence notes.

## References

- Angel et al. (2024) *Coast. Eng.* — adjoint reconstruction on the same
  flume, defines the dataset.
- Exp 2 [`pinn_inverse.py`](../02-thacker-basin-1d/pinn_inverse.py) —
  parent class `ThackerInversePINN`.
- Pre-flight bug log entries #8 (sensor snap), #14 (soft-BC honesty),
  #15 (reproducibility), #20 (this file).
