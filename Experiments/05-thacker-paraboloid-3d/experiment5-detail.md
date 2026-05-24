# Experiment 5 — Thacker Axisymmetric Paraboloid ("3D Thacker")

**Reference**: Thacker (1981), J. Fluid Mech. 107, 499–508, equations for the axisymmetric paraboloidal basin; reproduced in Delestre et al. (2016) SWASHES, Section 4.2.2.

## What "3D" means here

The full 3D incompressible Navier–Stokes equations with a free surface and arbitrary bathymetry have **no known analytical solution**. So the community uses the term "3D Thacker" to refer to the **axisymmetric paraboloid** test case: 2D horizontal flow $(u, v)$ plus a time-varying free surface $\eta(x, y, t)$ in a genuinely 3D bowl-shaped bathymetry (paraboloid of revolution). The water surface + velocities form a 2D+time problem, but the underlying domain is a true 3D basin with a curved moving shoreline.

This is the most natural "3D" extension of the experiments done so far and is directly comparable with Experiment 2 (1D planar Thacker).

## What's new vs Experiments 1–4

| Aspect | Exp 2 (1D Thacker) | Exp 4 (2D tidal) | Exp 5 (3D Thacker) |
|---|---|---|---|
| Spatial dim | 1D (x) | 2D (x, y) flat bed below mean | 2D (x, y) **in 3D bowl** |
| Shoreline | 2 moving points | stationary | **circular moving shoreline** |
| Symmetry | reflective | ortho-Cartesian | **radial / axisymmetric** |
| Bathymetry | $z_b(x)$ parabola (1D) | $z_b(x,y)$ cosine (smooth) | $z_b(r)$ **paraboloid of revolution** |
| Test of 2D dynamics | — | ridges/troughs | **coupled $(u,v)$ rotation + inflow/outflow** |

## Physical setup

**Bathymetry** — paraboloid of revolution:
$$z_b(x, y) = -h_0 \left(1 - \frac{r^2}{a^2}\right), \qquad r = \sqrt{(x - x_c)^2 + (y - y_c)^2}$$

- $h_0 = 0.1$ m: maximum rest depth at the center
- $a = 1$ m: radius where $z_b = 0$ (the "shore at rest")
- $(x_c, y_c) = (L/2, L/2)$: center of basin (default $L = 4$ m, so $(2, 2)$)

At center ($r=0$): $z_b = -h_0 = -0.1$ m. At $r = a$: $z_b = 0$. Beyond $r = a$ (inside the square domain): $z_b > 0$ — dry rim rising above mean sea level.

**Analytical solution (SWASHES 4.2.2, axisymmetric Thacker)**:

Let $A = (a^2 - r_0^2) / (a^2 + r_0^2)$ with $r_0 \in (0, a)$ controlling oscillation amplitude, and angular frequency $\omega = \sqrt{8 g h_0} / a$. Then:

$$
h(r, t) = h_0 \left[
  \frac{\sqrt{1 - A^2}}{1 - A \cos(\omega t)} - 1
  - \frac{r^2}{a^2} \left( \frac{1 - A^2}{(1 - A \cos(\omega t))^2} - 1 \right)
\right] - z_b(r)
$$

with $h$ clipped to zero where it is negative (wet/dry).

Velocities are pure radial expansion/contraction:
$$
u(x, y, t) = \frac{1}{1 - A \cos(\omega t)} \cdot \frac{1}{2} \omega \, A \sin(\omega t) \cdot (x - x_c)
$$
$$
v(x, y, t) = \frac{1}{1 - A \cos(\omega t)} \cdot \frac{1}{2} \omega \, A \sin(\omega t) \cdot (y - y_c)
$$

(velocities zero where $h = 0$).

SWASHES reference parameters: $a = 1$ m, $r_0 = 0.8$ m, $h_0 = 0.1$ m, $L = 4$ m, run for three periods $T_{end} = 3 \cdot 2\pi / \omega$.

With $g = 9.81$: $\omega = \sqrt{8 \cdot 9.81 \cdot 0.1} / 1 \approx 2.801$ rad/s, period $\approx 2.244$ s, total time ≈ 6.73 s.

## Why this case matters

1. **True 3D bathymetry**: bowl-shaped (paraboloid of revolution), not a "bumpy floor"
2. **Analytical ground truth** — machine precision, no solver uncertainty
3. **Curved moving shoreline** — hardest wetting/drying test yet
4. **Tests full 2D horizontal coupling**: both $u$ and $v$ evolve nontrivially (radial expansion/contraction)
5. **Direct extension of Exp 2** — compare 1D and "3D" Thacker for a full identifiability arc
6. **Literature-standard** — used by many papers; Ruppenthal 2026 2D benchmark is derived from this family

## Key research questions

1. **Does axisymmetry confuse the Cartesian-gridded PINN?** The ground truth has $u, v$ fields that change sign across the center.
2. **Can we recover the paraboloid parameters $h_0, a, x_c, y_c$** from surface observations of $\eta$?
3. **How many tidal phases are enough in this fully-2D wet/dry case?** (Expecting Exp 2 / Exp 4 result to hold: 2–4 phases should suffice.)
4. **Does the Exp 2 optimization pathology return** (large $N_t$ degrades due to loss imbalance)?

## Experiment design

### Phase 5.0 — Infrastructure
- [ ] `ground_truth.py`: analytical solution implemented per SWASHES formulas
- [ ] Verify $z_b$ paraboloid matches, $h, u, v$ match at key times (e.g., against our own formula derivative check — $h_t + \nabla \cdot (h\mathbf{u}) = 0$ up to machine precision in the wet region)
- [ ] Dataset generator + visualizations (radial profile, 2D snapshots at key phases, space-time center line)

### Phase 5.1 — Inverse PINN
- [ ] Reuse Experiment 4 architecture (2D tidal PINN) with these adaptations:
  - **No external tidal BC**: closed basin, energy is internal (IC alone drives dynamics)
  - **Sign-varying z_b**: can be negative (basin below 0) or positive (rim above 0) — same as Exp 4
  - Wetting-drying: softplus on h, dry-cell velocity loss
  - Closed-domain boundary: set zero-flux condition naturally (Neumann) or let domain be big enough that outer dry rim enforces it
  - Increase Fourier σ for bathymetry net to capture the paraboloidal curvature near center accurately

### Phase 5.2 — Baseline inversion
- [ ] Full space-time $\eta$ observations (eta-only, following Exp 4 success pattern)
- [ ] Target: z_b RMSE < 10 mm on ever-wet region (≈ 10% of h_0)

### Phase 5.3 — Key experiment: tidal phases sweep (analog to Exp 2 and Exp 4)
- [ ] N_t = 1, 2, 4, 8, 20 observation time instants (spanning 3 periods)
- [ ] Expected: 2–4 phases break equifinality; larger N_t degrades due to loss balance

### Phase 5.4 — Optional: parametric recovery
- [ ] Can we recover $(h_0, a, x_c, y_c)$ as 4 scalars from radial average of $\eta$?
- [ ] This is the "simplest possible" inversion: fit 4 parameters instead of a full z_b field

### Phase 5.5 — Report
- [ ] Summary + figures
- [ ] Cross-experiment synthesis: 1D → 2D → 3D Thacker progression

## Anticipated challenges

1. **Radial singularity at center**: velocity field is smooth at $r=0$ (vanishes) but derivatives may be stiff there. Should be handleable.
2. **Curved shoreline**: moving circle, intersects grid cells obliquely. The PINN doesn't care about grid alignment, but the FV verification would need subcell treatment — here we skip the FV and use the analytical solution directly.
3. **Limited wet area**: with $a = 1$ and domain $L = 4$, only the inner disk of area $\pi a^2 \approx 3.14$ m² out of $16$ m² is ever wet. Most collocation points will be in dry regions where z_b is not directly observed. We should focus loss + regularization on ever-wet region + smooth extrapolation in dry.
4. **Paraboloid has infinite extent in math sense** — in our Cartesian grid, the formula $z_b \propto r^2 - a^2$ continues rising past the domain edges. We'll clip or use the formula as-is and only evaluate RMSE on the ever-wet disk.

## Scope cuts

- Use the analytical solution directly as ground truth — skip the numerical FV verification (would require subcell shoreline treatment, over-engineering).
- Single seed per config.
- Skip primitive-conservative SWE form comparison.
- Skip parametric recovery (5.4) unless time permits.

## Files to produce

- `experiment5-detail.md` — this file
- `ground_truth.py` — analytical solution (SWASHES 4.2.2)
- `generate_and_plot.py` — dataset + visualizations (radial profile, snapshots, shoreline animation frames)
- `pinn_inverse.py` — inverse PINN adapted from Exp 4
- `n_t_sweep.py` — key experiment (number of temporal snapshots $N_t$ sweep; closed basin, not tidal)
- `REPORT.md` — findings, cross-experiment synthesis

## References

- Thacker (1981), J. Fluid Mech. 107, 499–508 — original derivation
- Delestre et al. (2016) SWASHES §4.2.2 — formulas + reference parameters
- Sampson, Easton, Singh (2006) — linear-friction extension (for later sensitivity)
- Already-done: Exp 2 (1D Thacker planar) and Exp 4 (2D tidal) for comparison baselines
