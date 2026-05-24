# Experiment 2 — Thacker Parabolic Basin (1D, Transient)

**Reference**: Thacker (1981), Dazzi (2024) case T1, SWASHES library.

## What's new vs Experiment 1

| Aspect | Experiment 1 (bump) | Experiment 2 (Thacker) |
|---|---|---|
| Time | Steady state | Transient (oscillating, period ~2s) |
| Bathymetry | Convex bump, positive above datum | Concave basin, below datum |
| Water state | Always wet | **Wetting-drying** (shoreline moves) |
| BCs | Inflow/outflow (Q, h_down) | **Closed basin** (no flow through boundaries) |
| Solution | Algebraic cubic (Bernoulli) | Closed-form $h(x, t)$ |
| PINN input | $x$ only | $(x, t)$ |

## Physical setup (T1 configuration)

**Parameters**:
- Basin half-width: $a = 1.0$ m
- Maximum depth at rest: $h_0 = 0.5$ m
- Angular frequency: $\omega = \sqrt{2 g h_0} / a = \sqrt{2 \cdot 9.81 \cdot 0.5} / 1 \approx 3.132$ rad/s
- Period: $T = 2\pi/\omega \approx 2.006$ s
- Domain: $x \in [-2, 2]$ m (total width 4 m, $L = 2a$ wet at rest)
- Simulation time: one full period, $t \in [0, T]$
- Gravity: $g = 9.81$ m/s²

**Bathymetry** (concave parabola — the "basin"):
$$z_b(x) = h_0 \left( \frac{x^2}{a^2} - 1 \right) = 0.5 (x^2 - 1)$$

At $x = 0$: $z_b = -0.5$ m (lowest point). At $x = \pm 1$: $z_b = 0$ (the "shore" at rest).

**Analytical solution** (planar-surface Thacker):

Water depth:
$$h(x, t) = \max\left(0, \; h_0 \left[1 - \left(\frac{x + 0.5 \cos(\omega t)}{a}\right)^2\right]\right)$$

Depth-averaged velocity (only where wet):
$$u(x, t) = \begin{cases} \frac{1}{2} \omega \sin(\omega t) & \text{if } h(x,t) > 0 \\ 0 & \text{otherwise} \end{cases}$$

Free surface elevation:
$$\eta(x, t) = h(x, t) + z_b(x)$$

**Physical interpretation**: the free surface stays planar (linear in $x$) at all times, tilting back and forth. The whole water mass sloshes coherently in the basin.

## Why this case matters for the thesis

1. **Analytical solution** — zero uncertainty in ground truth
2. **Transient dynamics** — tests whether temporal sampling improves inversion (core hypothesis from `Observations-for-Bathymetry-Inversion.md`)
3. **Wetting-drying** — moving shoreline is critical for real Chilean tidal flats
4. **Pure oscillation** — no forcing/BCs to confuse the PINN, isolates the inversion question
5. **Literature comparison** — Dazzi (2024) case T1, direct baseline available

## Key research question

> Can temporal data break the $z_b$-equifinality that plagued Experiment 1?

In Experiment 1, we needed boundary conditions ($z_b = 0$ at boundaries, $h = h_{down}$ at outlet) to anchor the inversion. In Experiment 2, the basin is **closed** — no such anchors exist. But the temporal richness ($N$ phases of the oscillation probe the same bathymetry under different flow states) should provide an alternative constraint.

**Expected result**: 
- Single snapshot → ill-posed, similar to Exp 1 without BCs
- Full time series → well-posed, inversion succeeds

## Experiment design

### Phase 2.0 — Infrastructure
- [ ] Ground truth generator (analytical Thacker solution)
- [ ] Verification against Dazzi's `thacker_problems.py`
- [ ] Reuse plotting/metrics utilities from Experiment 1

### Phase 2.1 — Transient PINN
- [ ] Network: $(x, t) \to (h, u)$, plus $z_b(x)$ net
- [ ] SWE residual with time derivatives: $\partial h/\partial t$, $\partial u/\partial t$
- [ ] Causal training weights (Wang 2024) for temporal consistency
- [ ] Wetting/drying: positivity loss $\mathcal{L}_{pos}$ + dry-cell velocity loss ($u = 0$ where $h = 0$)
- [ ] Initial condition loss (known at $t = 0$)

### Phase 2.2 — Baseline inversion
- [ ] All space-time observations of $\eta$
- [ ] No noise, known physics
- [ ] Target: $z_b$ RMSE < 10 mm on a 500 mm bathymetry range

### Phase 2.3 — Key experiment: snapshot vs time series
- [ ] Train with observations at $N_t = 1, 2, 4, 8, 16$ time instants
- [ ] Measure: does the inversion improve monotonically with $N_t$?
- [ ] Benchmark: at what $N_t$ does the inversion stabilize?

### Phase 2.4 — Sensitivity studies
- [ ] Observation density within each snapshot
- [ ] Noise on $\eta$
- [ ] Observation type ($\eta$, $u$, both)

### Phase 2.5 — Report
- [ ] Summary comparing with Experiment 1
- [ ] Figures: bathymetry recovery vs $N_t$, space-time error maps
- [ ] Interpretation in terms of identifiability theory

## Anticipated challenges

1. **Wetting/drying discontinuity**: $h$ transitions from 0 to positive across the moving shoreline. Classic PINN failure mode (spectral bias, can't represent sharp transitions). Mitigations: dry-cell loss, softplus output, Fourier features.
2. **Temporal causality**: standard PINNs can violate causality. Wang 2024 causal weighting solves this.
3. **Bathymetry $z_b < 0$**: our Experiment 1 softplus forced $z_b \geq 0$. Here the basin is below datum. Need to remove that constraint or redefine datum.
4. **Closed basin BCs**: no flow through boundaries means $u(\pm 2, t) = 0$ for all $t$. Need to enforce this.

## References

- Thacker (1981), "Some exact solutions to the nonlinear shallow-water wave equations", J. Fluid Mech., 107, 499-508
- Dazzi et al. (2024), WRR, e2023WR036589 — case T1
- Delestre et al. (2013), SWASHES library
- Wang (2024), "Respecting causality is all you need" — causal PINN training
