# Experiment Plan — Bathymetry Inversion with PINNs

## Overview

Six experiments ordered by increasing difficulty (Exp 1–4 are the
synthetic-benchmark core; Exp 5 is the 2D Thacker stress-test;
Exp 6 is the first contact with real sensor data). Each adds a specific
new challenge on top of the previous. A separate **SWE-form ablation
study** sits cross-cutting under Exp 1 (see [`01-subcritical-bump-1d/
REPORT-ABLATION.md`](01-subcritical-bump-1d/REPORT-ABLATION.md)).

---

## Experiment 1 — Subcritical Bump (1D, Steady) [Dazzi B1]

**Canonical baseline**. Dazzi 2024, Liu & Song 2025, and Ruppenthal 2026 all use this case. Smooth parabolic bump, steady subcritical flow, known analytical solution. Isolates the core question before adding complexity.

### Phase 1.0 — Infrastructure (DONE)
- [x] Analytical Bernoulli solver for ground truth (exact to machine precision)
- [x] Verified against SWASHES library (max 3.2mm, only at grid-misaligned edge)
- [x] Cloned Dazzi 2024 reference code (github.com/sdazzi00/PINN_augmSWEs_public)
- [x] Dataset generator for three configs: Dazzi B1 exact, SWASHES standard, Low-flow+friction

### Phase 1.1 — Baseline inverse PINN (DONE)
- [x] PyTorch PINN architecture: solution net (x -> h, u) + bathymetry net (x -> z_b)
- [x] SWE residual via automatic differentiation (initially primitive-conservative; production default became **primitive** after the May-17 ablation — see [01-subcritical-bump-1d/REPORT-ABLATION.md](01-subcritical-bump-1d/REPORT-ABLATION.md))
- [x] Composite loss: data(eta) + PDE + discharge(hu=q) + BC(z_b=0 at edges, h=h_down at outlet) + TV + Tikhonov + positivity
- [x] Baseline result on Dazzi B1: **z_b RMSE = 12.9 mm** (6.5% of 200mm bump), R^2 = 0.953
- [x] Identified spectral bias issue at bump peak (underestimated by ~41mm)

### Phase 1.2 — Baseline improvement (IN PROGRESS)
- [ ] Add Fourier feature embedding to reduce spectral bias
- [ ] Longer training budget + better learning rate schedule
- [ ] Target: z_b RMSE < 5mm on Dazzi B1

### Phase 1.3 — Sensitivity studies (NEXT)
- [ ] **Observation density**: 100%, 50%, 20%, 10%, 5% of points observed
- [ ] **Noise robustness**: 0%, 1%, 2%, 5% Gaussian noise on eta
- [ ] **Observation type**: eta only | u only | eta + u combined
- [ ] **Unknown friction**: co-invert z_b and Manning n

### Phase 1.4 — Sensitivity report
- [ ] Summary tables with RMSE, R^2, convergence iters across all sweeps
- [ ] Comparison figures per sensitivity axis
- [ ] Written findings: minimum observation requirements, noise tolerance, info value of each data type
- [ ] Comparison with Ruppenthal 2026 (optimal control + FEM baseline)

---

## Experiment 2 — Thacker Parabolic Basin (1D, Transient)

Adds transient dynamics and wetting-drying. Exact analytical solution (Thacker 1981). Tests whether temporal data improves identifiability, which is the core hypothesis from `Observations-for-Bathymetry-Inversion.md`.

### Phase 2.0 — Infrastructure (DONE)
- [x] Analytical Thacker solution (planar-surface T1 case)
- [x] Verified against Dazzi `thacker_problems.py` — exact to machine precision
- [x] Ground truth dataset generator + visualizations (snapshots, space-time maps)

### Phase 2.1 — Transient PINN (DONE)
- [x] Network: (x, t) -> (h, u), separate Fourier features for x and t
- [x] Bathymetry net: x -> z_b, unconstrained (basin below datum)
- [x] SWE residual with time derivatives (continuity + momentum)
- [x] Wetting/drying handling: softplus on h, dry-cell velocity loss, wet-indicator weighting on PDE residual
- [x] Initial condition loss (known h, u at t=0)
- [x] Closed basin BC: u=0 at walls

### Phase 2.2 — Baseline inversion (DONE)
- [x] eta only: failed (180 mm RMSE, equifinality)
- [x] eta + u: 24 mm RMSE on ever-wet region

### Phase 2.3 — Key experiment: snapshot vs time series (DONE)
- [x] Trained N_t = 1, 4, 10, 40 with eta only
- [x] **N_t=4 achieves 5 mm RMSE** — temporal richness alone breaks equifinality
- [x] N_t ≥ 10 degrades due to loss-weighting imbalance (documented, not identifiability issue)

### Phase 2.4 — Report (DONE)
- [x] `REPORT.md` with all findings, limitations, comparison with Experiment 1

---

## Experiment 3 — Two Cylinders (2D, Transient)

First 2D case. Localized features test spatial resolution and spectral bias. Ruppenthal 2026 provides non-ML baseline.

### Phase 3.0 — Infrastructure (DONE)
- [x] 2D HLL finite-volume SWE solver with topography source
- [x] Ground truth with Ruppenthal's parameters (cyls at (8,12,r=4,H=0.2) and (17,13,r=2,H=0.3))
- [x] Visualization: bathymetry, eta/velocity snapshots

### Phase 3.1 — 2D PINN (DONE)
- [x] Solution net (x,y,t) -> (h,u,v), bathymetry net (x,y) -> z_b
- [x] 2D Fourier features, AD-based SWE residual, IC loss
- [x] Softplus positivity, TV regularization

### Phase 3.2 — Baseline (DONE)
- [x] eta+u+v observations on 40×40 grid × 15 snapshots
- [x] **z_b RMSE = 40 mm, both cylinders located correctly**, heights underestimated ~40%
- [x] Pipeline scales from 1D to 2D with zero architectural changes

### Phase 3.3 — Report (DONE)
- [x] `REPORT.md` with findings, comparison with Exp 1 and Exp 2, next steps

---

## Experiment 4 — Oscillatory Topography + Tide (2D, Periodic)

Bridge to Chilean coastal application. Periodic tidal forcing, multi-scale bathymetry. Tests multi-phase tidal observations for identifiability.

### Phase 4.0 — Infrastructure (DONE)
- [x] FV solver with tidal water-level BC on all 4 boundaries
- [x] Ground truth: 30×30 grid, 2 periods, 24 snapshots
- [x] Visualizations: bathymetry, tidal snapshots, center-vs-BC time series

### Phase 4.1 — 2D tidal PINN (DONE)
- [x] Sign-unconstrained z_b, tidal BC loss, IC loss
- [x] All three SWE residuals (continuity + 2 momentum) via AD

### Phase 4.2 — Baseline (DONE)
- [x] **10.4 mm RMSE** on full time series, eta-only — 2.6% of 400 mm bathymetry range
- [x] Visually indistinguishable from truth

### Phase 4.3 — Multi-phase tidal sweep (DONE)
- [x] N_t = 1 fails (99 mm, degenerate)
- [x] **N_t = 2 succeeds (6.4 mm)** — two tidal phases enough to break equifinality in 2D
- [x] Non-monotonic trend at N_t ≥ 8 = loss-weighting artifact (same as Exp 2)

### Phase 4.4 — Report (DONE)
- [x] `REPORT.md` with findings, cross-experiment synthesis, Chilean tide-gauge viability claim

---

## Experiment 5 — Thacker Parabolic Basin (2D, Transient)

Direct 2D extension of Exp 2: a paraboloid basin with an exact analytical
Thacker solution. Tests whether the 1D wetting/drying findings carry over
to 2D and at what cost.

### Phase 5.0 — Infrastructure (DONE)
- [x] 2D Thacker analytical solution (axially symmetric, oscillating)
- [x] Ground truth: 60×60 grid, multiple periods, snapshots saved
- [x] Visualizations: bathymetry, snapshots, time-series at probes

### Phase 5.1 — Baseline (DONE — v4 sealed)
- [x] (x, y, t) → (h, u, v) network + (x, y) → z_b net with Fourier features
- [x] Wet-indicator masking, soft IC/BC losses
- [x] Final baseline migrated to `pinn_bath` via `studies/arch_scaling.py` (Exp 5 cell) + `studies/exp5_n_t_sweep.py`; legacy `pinn_inverse.py` removed in the M6 nuke.

### Phase 5.2 — Report (DONE)
- [x] `Experiments/05-thacker-2d/REPORT.md` (improve-baseline v4 finalized)

---

## Experiment 6 — Real Sensor Data (Angel river)

First experiment using real measurements, not synthetic ground truth.
Documented as a **negative result**: a soft-constrained PINN cannot recover
the bathymetry from sparse real sensor data without further regularization
or hard constraints. Used in the thesis as motivation for future work.

### Phase 6.0 — Data ingest (DONE)
- [x] Angel sensors → `pinn_bath.data.Case` adapter
- [x] Sensor placement audit (S2 default `nx=120` misplaces by 42 mm; warning added)

### Phase 6.1 — PINN inversion (DONE, negative)
- [x] Soft SWE constraint fails on real noisy sparse data
- [x] `REPORT.md` documents the negative finding with discussion

---

## Cross-cutting study — SWE residual form ablation (Exp 1)

Compares the three SWE residual formulations (primitive, primitive-
conservative, conservative) on the inverse problem. Original contribution:
primitive is most robust (2/2 seeds converge, mean RMSE 4.05 ± 0.04 mm),
reversing Tian et al. (2025)'s forward-problem ranking. Used to set the
production default of `swe_form` in the `pinn_bath` canonical pipeline.

- Script: [`studies/ablation_forms.py`](../studies/ablation_forms.py)
- Report: [`01-subcritical-bump-1d/REPORT-ABLATION.md`](01-subcritical-bump-1d/REPORT-ABLATION.md)
- Discussed in `Report/sections/05-comparacion-literatura.tex`.

---

## Progress Summary

| Phase | Status | Notes |
|---|---|---|
| 1.0 Infrastructure | DONE | Ground truth verified against Dazzi code |
| 1.1 Baseline PINN | DONE | z_b RMSE = 12.9mm |
| 1.2 Baseline improvement | IN PROGRESS | Target < 5mm |
| 1.3 Sensitivity studies | NEXT | 15-20 training runs total |
| 1.4 Report | DONE | See Experiments/01-subcritical-bump-1d/REPORT.md |
| 2. Thacker transient | DONE | Key finding: N_t=4 snapshots recover basin to 5 mm (eta only) |
| 3. 2D cylinders | DONE | Both cylinders located, 40 mm RMSE — pipeline scales to 2D |
| 4. 2D tidal | DONE | 10.4 mm full / 6.4 mm at N_t=2 — Chilean tide-gauge use case validated |
| 5. 2D Thacker | DONE (v4) | Analytical baseline; v1–v3 diagnostic runs archived |
| 6. Angel real data | DONE (negative) | Soft SWE fails on real sparse sensors; motivates future hard-constrained work |
| Cross-cut: SWE-form ablation | DONE | Primitive form is most robust (see REPORT-ABLATION.md) — production default for swe_form |

### Final thesis claim

**In 2D with tidal forcing, two water-level snapshots at different tidal phases are enough to recover a multi-scale bathymetry to ~6 mm RMSE with eta-only observations.** This directly supports the Chilean tide-gauge application — no velocity/ADCP data needed. Equifinality, which plagued Exps 1–4, is breakable via three complementary mechanisms: inflow/outflow BCs (Exp 1), velocity observations (Exp 1/3), or multi-phase tidal sampling (Exp 2/4). The pipeline scales from 1D to 2D without architectural changes.

---

## File Structure

```
Experiments/
|-- experiment-plan.md                    # this file
|-- 01-subcritical-bump-1d/               # canonical 1D steady baseline (+ experiment1-detail.md)
|-- 02-thacker-basin-1d/                  # 1D transient wetting/drying
|-- 03-two-cylinders-2d/                  # 2D transient (Ruppenthal §7.2)
|-- 04-tidal-oscillatory-2d/              # 2D periodic tidal (uses pinn_bath)
|-- 05-thacker-paraboloid-3d/             # 2D Thacker paraboloid baseline
|-- 06-angel-real-data/                   # real-data Angel flume (negative result)
`-- datasets/                             # shared input datasets (Angel, SWASHES)

src/pinn_bath/                             # canonical pipeline (Case, models,
                                           # trainers, losses, residuals)
studies/                                   # cross-experiment harnesses
                                           # (arch_scaling, sensitivity sweeps)
```

Each `0X-*` folder typically contains: `ground_truth.py` (data
generator or loader), `data/` (`.npz` consumed by `pinn_bath.data.Case`),
`figures/`, `results/` (legacy training `*.log` audit trail), and a
`REPORT.md` with reproducible study commands + pending-azirafel
results. The per-experiment `pinn_inverse.py` orchestration was
removed in the legacy → `pinn_bath` migration; the canonical pipeline
lives in `studies/` (one script per study).

---

## Prior freeform description (superseded)

The original plan proposed four experiments; the scope expanded to six
plus a cross-cutting SWE-form ablation study as the work matured (see
the sections above for the current plan; the May-17 ablation finding
is documented in
[`01-subcritical-bump-1d/REPORT-ABLATION.md`](01-subcritical-bump-1d/REPORT-ABLATION.md)).
