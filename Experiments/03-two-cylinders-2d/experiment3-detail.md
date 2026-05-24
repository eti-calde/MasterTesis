# Experiment 3 — Two cylinders 2D (Ruppenthal §7.2)

**Reference**: Ruppenthal & Kuzmin (2026), arXiv:2603.11813 — *Bathymetry reconstruction via optimal control in well-balanced finite element methods for the shallow water equations*, Section 7.2 (two-dimensional benchmark test).

This file specifies the case as it is currently implemented in
`ground_truth.py` and reflected in the thesis annex
(`Report/sections/07-apendice-casos-sinteticos.tex`, A.3). It is the
authoritative source for the parameters used by the canonical pipeline
(`pinn_bath` + `studies/arch_scaling.py`).

## Physical setup

| Parameter | Value |
|---|---|
| Domain | $[0, 25]^2$ m² |
| Grid | $50 \times 50$ cell centers ($\Delta x = \Delta y = 0.5$ m) |
| Cylinder 1 | center $(8, 8)$ m, radius $4$ m, height $0.2$ m |
| Cylinder 2 | center $(15, 15)$ m, radius $2$ m, height $0.3$ m |
| Bathymetry | **vertical-walled** indicators (no smoothing): $z_b = 0.2$ inside cyl 1, $0.3$ inside cyl 2, $0$ elsewhere |
| Initial free surface | $\eta(x, y, 0) = h + z_b = 2$ m uniform |
| Initial velocity | $\mathbf{v}(x, y, 0) = (2.21, 2.21)$ m/s uniform |
| Initial depth | $h(x, y, 0) = 2 - z_b(x, y)$ (shallower over cylinders) |
| Boundary (inflow) | $x = 0$ and $y = 0$ faces: Dirichlet $(\eta, u, v) = (2, 2.21, 2.21)$ m via ghost cells |
| Boundary (outflow) | $x = L_x$ and $y = L_y$ faces: zero-gradient Neumann |
| Simulation time | $T = 60$ s |
| Time step | $\Delta t = 10^{-2}$ s |

The two cylinders are aligned along the diagonal $x = y$, parallel to the
direction of the initial uniform flow. Ruppenthal & Kuzmin §7.2 does not
specify boundary conditions; the Dirichlet-inflow / Neumann-outflow split
above is the natural choice to preserve the prescribed uniform free-stream
along the 60 s of simulation.

## Ground truth solver

The reference solution is obtained with the FV-HLL scheme implemented in
`ground_truth.py`: standard cell-centered finite volumes on the
$50 \times 50$ cartesian grid, Harten–Lax–van Leer Riemann solver applied
to states reconstructed by Audusse's hydrostatic technique
(Audusse et al. 2004), which is well-balanced for discontinuous
bathymetry: the topography source term is absorbed into asymmetric
pressure corrections at each interface, preserving "lake-at-rest"
($u = v = 0$, $\eta = h + z_b = \text{const}$) to machine precision across
the sharp cylinder walls. The rest of the discretization follows LeVeque
(2002).

Ruppenthal & Kuzmin use a flux-corrected finite-element scheme (MCL) on the
same grid. Both are standard convergent SWE discretizations; we do not
attempt a bit-for-bit comparison with Ruppenthal's numbers — only the
geometry and initial/boundary conditions are replicated faithfully.

## Why this case matters

1. **First 2D test** of the inverse PINN pipeline; checks that the 2D
   architectures (A1/A2/A3) and 2D SWE residual scale from the 1D cases.
2. **Localized sharp features**: vertical-walled cylinders are the
   adversarial case for spectral bias — sharp gradients at the cylinder
   edges stress the bathymetry network.
3. **Multi-object recovery**: both cylinders must be located simultaneously
   (positions, radii, heights).
4. **Comparison benchmark**: Ruppenthal's optimal-control + FEM result on
   the same geometry provides a non-ML reference, useful for thesis §6.

## PINN pipeline

The PINN is launched via the canonical harness:

```bash
python -m studies.arch_scaling --study-dir runs/arch_scaling
```

which iterates over `A1 × A2 × A3 × {small, medium, large} × {seed 0, 1, 2}`
on the three §5.1 cases (Exp 1, 2, 3). The per-config trainer
(`pinn_bath.trainers.AdamLBFGSTrainer`) runs 12 000 Adam epochs + 600
L-BFGS steps per the §3.10 protocol.

Results land in `runs/arch_scaling/<run_id>/summary.json`; aggregated tables
come from `studies/aggregate.py`.

## Files

- `ground_truth.py` — FV-HLL solver + bathymetry definition.
- `generate_and_plot.py` — regenerate the dataset and figures.
- `data/ground_truth_cylinders.npz` — unified-schema dataset, loadable with `pinn_bath.data.Case.load`.
- `figures/bathymetry.png` — true $z_b$ map.
- `figures/ground_truth_snapshots.png` — $\eta$ and $|\mathbf{v}|$ at four time instants.

## References

- Ruppenthal & Kuzmin (2026), arXiv:2603.11813.
- Audusse, Bouchut, Bristeau, Klein, Perthame (2004), *A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows*, SIAM J. Sci. Comput. 25(6), 2050–2065, doi:10.1137/S1064827503431090.
- LeVeque (2002), *Finite Volume Methods for Hyperbolic Problems*, Ch. 13 (SWE), Cambridge University Press.
