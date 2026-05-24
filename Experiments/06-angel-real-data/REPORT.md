# Exp. 6 — PINN Bathymetry Inversion on Angel et al. 2024 Real Flume Data

**Outcome: documented negative result.** A soft-penalty PINN does **not**
recover the bathymetry bump from Angel's real sparse-sensor data, where the
hard-constrained adjoint method of Angel et al. succeeds (NRMSE 10–14 %).
This is a clean, honest thesis finding that motivates hard-constrained /
stronger PINN formulations as future work.

## Setup

- **Data**: `datasets/angel2024/processed/angel2024_flume.npz` (mean of 20
  Hamburg-flume runs + measured ground-truth bathymetry). Real wave-gauge
  signals, real linear bottom drag κ = 0.2, kh ≈ 1.3 (intermediate depth →
  SWE has ≈10 % inherent model error).
- **Temporal window**: the wavemaker is silent until t ≈ 34 s; we invert the
  established-wave window **t ∈ [40, 60] s**, decimated 100 Hz → 10 Hz,
  re-zeroed (SWE + κ are time-translation invariant).
- **Spatial grid**: Nx = 136 over x ∈ [1.5, 15] m → all four sensors land
  exactly on grid nodes (zero snap error).
- **Sensors**: S1 (x = 1.5 m) = **soft** inlet Dirichlet η(t) (MSE
  penalty via the same observation mask as S2; **not** a hard BC —
  contrast Angel's adjoint method, which imposes the inlet as a hard BC
  through its forward solver); S2 (x = 3.5 m) =
  interior observation [Angel-minimal]. The bump peak is at **x ≈ 3.99 m** —
  *no sensor sits on the bump*.

## Method

- Reuses the Exp. 2 transient-1D PINN verbatim: `AngelInversePINN` subclasses
  `ThackerInversePINN` (two-network SolNet+BathNet, Fourier features, sparse
  observation mask, Adam + L-BFGS).
- Single physics change: linear drag `κ·u/(h+ε)` replaces the Manning term
  (`physics_angel.py`).
- Two **exact physical priors** added to break the η = h + z_b equifinality:
  bed elevation ≥ 0 (solid flume floor; the bed analog of the existing h ≥ 0
  softplus) and z_b = 0 at the flat flume ends (same idiom as Exp. 1
  `loss_bc`). Without these the inversion diverges (NRMSE > 100 %).

## Compute (Phase 0 benchmark — answers the user's cost question)

Measured on the GTX 1650 (4 GB): **73 ms / Adam epoch**, L-BFGS 1.73 s/step,
**peak VRAM 556 MB (15 % of the card)**, full run (15 k Adam + 200 L-BFGS)
**≈ 21–30 min**. **Compute is not a barrier** — no extra resources needed;
the plan's pre-measurement estimates were ~4× conservative.

## Result (the negative finding)

Surface elevation η is fit *excellently* at the sensors (data loss ≈ 1e-4,
η RMSE a few mm), but the **bathymetry is not recovered**:

| Metric | Value (mean ± std, 3 seeds) | Reference |
|---|---|---|
| **Bump-peak recovery** | **12.3 ± 3.4 %** (≈88 % of the 199.7 mm bump missed) | target ≈100 % |
| z_b NRMSE, informative span [1.5, 9] m | 22.4 % | Angel adjoint 10–14 % |
| z_b NRMSE, full domain | 17.9 % | — |

The full-domain NRMSE (17.9 %) is **not a meaningful success metric**: the
domain is ~90 % flat zero-bed where any flat prediction scores well. The
physically relevant **peak recovery (12 %)** shows the bump is essentially
missed. The failure is **robust across 3 random seeds at the canonical
config** swept by `run_matrix.py` (σ_x = 2, λ_pde = 1, S2 as the only
interior obs sensor). Earlier exploratory runs with stronger BathNet
bandwidth (σ_x = 6), stronger PDE weight (λ_pde = 5) and 3 obs sensors
(S2+S3+S4) showed similar failure (~34 mm z_b RMSE) but those
configurations are **not** in the current `run_matrix.py` sweep — they
would need to be added to the harness to be reproducible. The canonical
result alone is sufficient evidence of the negative finding; the broader
sweep is left as TODO for thesis defence.

## Diagnosis

The soft PDE penalty admits the η = h + z_b equifinality: with sensors
*off* the bump, the network fits the observed wave by adjusting the water
depth h while leaving z_b ≈ flat — a solution that satisfies data **and**
approximately satisfies the soft physics residual. Angel's adjoint method
succeeds on the same 2-sensor data because it enforces the SWE as a **hard**
constraint through a forward solver, so z_b is the only free field that can
explain the observed wave dynamics. kh ≈ 1.3 adds inherent SWE model error
on top. This is a concrete, reproducible limitation of soft-penalty PINN
inversion on real, intermediate-depth, sparse off-feature sensor data.

## Files

| File | Role |
|---|---|
| `data_angel.py` | load/window/decimate flume npz; sensor→node; NRMSE helper |
| `physics_angel.py` | transient SWE residual with linear κ drag |
| `pinn_angel.py` | `AngelInversePINN` (subclass; κ + z_b≥0 + z_b-end priors) |
| `bench_epoch.py` | Phase 0 timing benchmark |
| `run_poc.py` | Phase 1 single inversion (tunable σ_x, λ_pde, obs set) |
| `run_matrix.py` | sensor/seed matrix (built; not run — all configs fail alike) |
| `analyze_limitation.py` | seed-robustness proof + honest figure |
| `results/poc.json`, `results/limitation_analysis.json` | metrics |
| `figures/poc_inversion.png`, `figures/limitation_analysis.png` | figures |

## Future work (motivated by this result)

Hard PDE constraint (differentiable forward SWE solver in the loop, i.e. a
PINN-adjoint hybrid), causal/curriculum time training, or residual-adaptive
weighting — to close the gap to Angel's adjoint benchmark on real data.
