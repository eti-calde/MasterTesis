# Experiment 1 — Sensitivity Study Report

**Date**: 2026-04-19
**Case**: Dazzi B1 (subcritical flow over parabolic bump)
**Status**: Results pending sensitivity sweep completion

---

## Summary

Inverse PINN recovers a 200 mm parabolic bump from steady-state surface observations $\eta(x)$ over a 20 m channel. Baseline with all 500 points observed, no noise, known friction: **RMSE = 6.25 mm, R² = 0.989**.

This report quantifies how inversion quality degrades under three stressors: sparse observations, measurement noise, and incomplete data types.

---

## Setup

| Parameter | Value |
|---|---|
| Domain | $x \in [-10, 10]$ m |
| Bathymetry | $z_b(x) = 0.2 - 0.05 x^2$ for $|x| < 2$, else 0 |
| Bump height | 200 mm |
| Discharge | 4.42 m²/s |
| Downstream depth | 2.0 m |
| Manning | 0 (frictionless) |
| Froude range | [0.50, 0.63] |
| Surface depression | 93 mm |

## Architecture

- Solution network: 4 hidden layers x 64 neurons, Tanh activation, Fourier feature embedding ($\sigma=1$, 16 freqs)
- Bathymetry network: 3 hidden layers x 32 neurons, Tanh activation, Fourier feature embedding
- Input $x$ normalized to $[-1, 1]$ before networks
- Loss: $\lambda_{data} \mathcal{L}_{data} + \lambda_{PDE} \mathcal{L}_{PDE} + \lambda_q \mathcal{L}_q + \lambda_{BC} \mathcal{L}_{BC} + \lambda_{TV} \mathcal{L}_{TV} + \lambda_{Tikh} \mathcal{L}_{Tikh} + \lambda_{pos} \mathcal{L}_{pos}$
- Training: 8000 Adam epochs + 300 L-BFGS steps

---

## Results

### Sweep 1 — Observation Density

*[Table filled in after run]*

| Density | $z_b$ RMSE (mm) | R² | Training time (s) |
|---|---|---|---|
| 100% | | | |
| 50% | | | |
| 20% | | | |
| 10% | | | |
| 5% | | | |

**Figure**: `figures/sensitivity_density.png` and `figures/sensitivity_density_profiles.png`

**Interpretation**: *[To be written after results available]*

---

### Sweep 2 — Noise Robustness

| Noise (% of signal) | $z_b$ RMSE (mm) | R² | Training time (s) |
|---|---|---|---|
| 0% | | | |
| 1% | | | |
| 2% | | | |
| 5% | | | |

**Figure**: `figures/sensitivity_noise.png` and `figures/sensitivity_noise_profiles.png`

**Interpretation**: *[To be written after results available]*

---

### Sweep 3 — Observation Type

| Type | $z_b$ RMSE (mm) | R² | Training time (s) |
|---|---|---|---|
| $\eta$ only | | | |
| $u$ only | | | |
| $\eta + u$ | | | |

**Figure**: `figures/sensitivity_obstype.png` and `figures/sensitivity_obstype_profiles.png`

**Interpretation**: *[To be written after results available]*

---

## Cross-cutting Findings

*[Written after all sweeps complete]*

- **Practical minimum observations**: What density is "good enough" (e.g., < 2x baseline RMSE)?
- **Noise tolerance**: At what noise level does inversion degrade beyond useful?
- **Value of velocity data**: How much does adding $u$ help vs $\eta$ alone?
- **Comparison with literature**:
  - Ruppenthal 2026 reports RMSE robust to 5% noise using optimal control + TV regularization
  - Liu 2024 CNN surrogate: similar sparsity tolerance but requires pretraining on 1000+ simulations

---

## Limitations and Next Steps

- Fixed Manning (not co-inverted) — to be tested in follow-up
- Single bump geometry — richer bathymetries would test generalization
- Steady state only — Experiment 2 (Thacker basin) will test transient benefit

---

## Raw Data

- JSON results: `results/sensitivity_results.json`
- Training log: `results/sensitivity_run.log`
- Ground truth: `data/ground_truth_dazzi_B1.npz`
