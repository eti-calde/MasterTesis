# Experiment 3 — Two Cylinders (2D, Transient) REPORT

**Case**: Ruppenthal & Kuzmin (2026) §7.2 — two vertical-walled solid
cylinders on a flat bed, uniform diagonal inflow.
**Status**: **Pending — re-run on `azirafel` via `pinn_bath`**.

---

> **Re-baseline note (post-migration).** The historical 40 mm RMSE
> baseline came from the legacy `pinn_inverse.py` running on a
> subsampled 40×40 grid for 2 s. Multiple physical bugs in the
> ground-truth solver and the PINN itself were fixed before this
> baseline was usable:
> - Audusse hydrostatic reconstruction replaces the centered-difference
>   topography source (batch #5; was generating a 5 m/s² spurious
>   force at cylinder edges, comparable to gravity).
> - Dirichlet inflow on the small-x/small-y boundaries (batch #6; was
>   Neumann everywhere, letting the uniform flow decay).
> - PINN `BathymetryNet2D` `softplus(zb_raw) - 0.1` shift removed
>   (batch #12; allowed unphysical $z_b \in [-0.1, \infty)$ when
>   Ruppenthal's cylinders are $z_b \in \{0, 0.2, 0.3\}$).
> - PINN momentum residual rewritten in well-balanced form
>   $g \, \partial_x \eta$ instead of $g(\partial_x h + \partial_x z_b)$
>   (batch #12 cosmetic, numerically identical).
>
> The ground truth `data/ground_truth_cylinders.npz` is regenerated
> with the Audusse + Dirichlet-inflow solver. Final cifras land here
> after the azirafel sweep.

---

## Setup

- Domain: $[0, 25]^2$ m, $t \in [0, 60]$ s.
- Two cylinders along the diagonal: $(x_c, y_c, r, H) = (8, 8, 4, 0.2)$
  and $(15, 15, 2, 0.3)$. Sharp indicators.
- Uniform IC: $\eta = h + z_b = 2$ m, $(u, v) = (2.21, 2.21)$ m/s.
- Linear drag-free FV-HLL reference solver with Audusse HR; Dirichlet
  inflow on $x = 0$, $y = 0$; outflow (zero-gradient) on $x = L_x$,
  $y = L_y$.
- Inverse PINN: `pinn_bath` A1/small (or larger), 2D transient SWE
  residual, IC loss (uniform IC is known), eta+u+v observations.

## Reproducible study

```bash
python -m studies.arch_scaling --study-dir runs/arch_scaling
```

Exp 3 is one of the cases swept by the §5.1 architecture-scaling
study (A1×A2×A3 × small×medium×large × 3 seeds).

## Results

**TODO**. Filled from `runs/arch_scaling`. Expected:

- Cylinder localisation visual check (both cylinders correctly placed?).
- Per-cylinder height recovery (historical legacy underestimated by
  ~40%; check if Audusse + new BC + softplus fix change that).
- RMSE_zb across arch × budget cells (§5.1 part).

## Files

- `ground_truth.py` — FV-HLL with Audusse + Dirichlet inflow.
- `data/ground_truth_cylinders.npz` — Case (`bc_type="open_uniform"`).
- `figures/` — historical figures.
- `results/baseline.log` — legacy log (kept as audit trail).
