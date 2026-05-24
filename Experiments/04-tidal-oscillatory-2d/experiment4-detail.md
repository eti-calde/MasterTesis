# Experiment 4 — Tian dT10 (variable-topography tidal problem)

**Reference**: Tian et al. (2025), *Physics-Informed Neural Networks for
Solving the Two-Dimensional Shallow Water Equations With Terrain Topography
and Rainfall Source Terms*, Water Resources Research 61, e2025WR040052,
Section 3.3 (dynamic case **dT10**, equations 54–56).

> **Naming caveat.** Tian calls dT10 the "tidal problem" because of the
> shape of the bottom topography (derived from a benchmark in Supei et al.
> 2022), not because of an external tidal forcing. dT10 has periodic
> boundaries and no time-dependent forcing — the dynamics come purely from
> an unbalanced initial condition that relaxes under gravity.

This file specifies the case as it is currently implemented in
`ground_truth.py` and reflected in `Report/sections/07-apendice-casos-sinteticos.tex` (A.4).

## Physical setup

| Parameter | Value |
|---|---|
| Domain | $(x, y) \in [-2, 2]^2$ m |
| Grid | $100 \times 100$ cell centers ($\Delta x = \Delta y = 0.04$ m) |
| Topography | $z(x, y) = 1 + 0.01\,\cos(\pi x / 2)\,\cos(\pi y / 2)$ m |
| $z$ range | $[0.99, 1.01]$ m (max at center and four corners; min at midpoints of edges) |
| Initial depth | $h(x, y, 0) = z(x, y)$ (Tian eq. 56) |
| Initial velocity | $u(x, y, 0) = v(x, y, 0) = 0$ |
| Initial free surface | $\eta(x, y, 0) = h + z = 2 z$ — same maxima/minima pattern as $z$ |
| Boundary conditions | **periodic** in $x$ and $y$ |
| Simulation time | $T = 0.5$ s |
| Reference snapshots | $t = 0, 0.25, 0.5$ s (matches Tian Figure 5) |

Because $\eta(0) = 2 z$ is not flat, water flows under gravity from the
high regions (center + four corners) toward the low regions (midpoints of
the four edges). The transient is short ($T = 0.5$ s) but multi-lobed and
symmetric.

## Ground truth solver

Reference solution from the FV-HLL solver (`ground_truth.py`) on a
$100 \times 100$ cartesian grid with **periodic** ghost cells (left ghost
= rightmost interior, etc.). Source term from topography by central
differences (smooth bathymetry, so well-balancing is not critical at this
amplitude).

Tian solves dT10 with a high-order entropy-stable scheme (ES1, Fjordholm
et al. 2011) on a much finer grid ($\Delta x = 0.01$ m, CFL $= 0.25$). The
two discretizations differ in dispersion and entropy properties but both
converge to the same SWE solution at sufficient resolution; we do not
attempt bit-for-bit reproduction of Tian's numbers — only the geometry,
IC, and BC are replicated faithfully.

## Why this case matters

1. **Dynamic 2D**: complements Exp 3 (sharp localized features) with a
   smooth, distributed bathymetry.
2. **Periodic boundary conditions**: exercises the
   `pinn_bath.losses.periodic_bc_loss` term and validates that
   `pinn_bath.trainers.AdamLBFGSTrainer` dispatches the right BC loss
   from `case.metadata.bc_type`.
3. **Relaxation dynamics**: the multi-lobed flow pattern is a richer
   transient than the steady-state Exp 1 and the closed-basin oscillation
   of Exp 2 — useful for diagnosing how A1/A2/A3 handle 2D space-time
   structure.

## PINN pipeline

Exp 4 is not part of the canonical §5.1 architecture-scaling grid (that
uses Exp 1, 2, 3). It is invoked directly through
`pinn_bath.trainers.AdamLBFGSTrainer` from a `RunConfig` that sets
`loss=LossWeights(..., bc=10.0)` (the periodic BC term is otherwise
inactive). A smoke run was verified in `runs/smoke_bc/`: `loss_bc`
descends from $1.7\times 10^{-2}$ to $4.5\times 10^{-5}$ in 200 Adam epochs
on A1 small.

## Files

- `ground_truth.py` — FV-HLL solver + dT10 topography.
- `generate_and_plot.py` — dataset + figure regeneration; reproduces Tian
  Figure 5 panels at $t = 0, 0.25, 0.5$ s.
- `data/ground_truth_dT10.npz` — unified-schema dataset.
- `figures/bathymetry.png` — true $z(x, y)$ map.
- `figures/ground_truth_snapshots.png` — $h$, $u$, $v$ at $t = 0, 0.25, 0.5$ s.
- `figures/relaxation_timeseries.png` — $\eta$ at center, corner, mid-edge.

## References

- Tian et al. (2025), *Water Resources Research* 61, e2025WR040052,
  doi:10.1029/2025WR040052 (case dT10, §3.3, equations 54–56).
- Fjordholm, Mishra, Tadmor (2011) — entropy-stable scheme used as Tian's
  reference solver.
- LeVeque (2002), *Finite Volume Methods for Hyperbolic Problems*,
  Cambridge University Press.
