"""Preview the bathymetry case generator (F1).

Produces two figures under analysis/:
  * cases_generator_gallery.png — for each difficulty tier, several sampled
    bathymetries + one solved η(x,t) hovmöller (visual easy→hard progression
    and a check that the solver yields rich, finite dynamics).
  * cases_difficulty_hist.png — difficulty-score distributions per tier
    (computed from zb alone, no solve) to confirm the tiers separate.

Run:  .venv/bin/python scripts/preview_generator.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinn_bath.datasets.generator import (
    Grid,
    bathymetry,
    difficulty_components,
    difficulty_score,
    generate_record,
    sample_case,
)

TIER_ORDER = ["easy", "medium", "hard"]
TIER_COLOR = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728"}


def gallery(grid: Grid) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for r, tier in enumerate(TIER_ORDER):
        # Left: a few sampled bathymetries.
        axL = axes[r, 0]
        rng = np.random.default_rng(100 + r)
        for _ in range(4):
            spec = sample_case(tier, rng, grid)
            zb = bathymetry(spec.features, grid)
            axL.plot(grid.centers, zb, lw=1.3, alpha=0.8)
        axL.axhline(grid.sea_level, color="b", ls="--", lw=1, alpha=0.6, label="nivel de reposo η₀")
        axL.set_title(
            f"{tier} — batimetrías muestreadas (4)",
            fontsize=10,
            color=TIER_COLOR[tier],
            fontweight="bold",
        )
        axL.set_xlabel("x [m]")
        axL.set_ylabel("$z_b$ [m]")
        axL.set_ylim(-0.6 * grid.sea_level, 1.15 * grid.sea_level)
        axL.grid(alpha=0.3)
        if r == 0:
            axL.legend(fontsize=8, loc="lower right")

        # Right: one solved η(x,t) hovmöller.
        axR = axes[r, 1]
        rng2 = np.random.default_rng(500 + r)
        spec = sample_case(tier, rng2, grid)
        rec = generate_record(spec, grid, kernel="aug", cfl_desired=0.45)
        eta = rec["eta"]
        pcm = axR.pcolormesh(grid.centers, rec["t"], eta, shading="auto", cmap="viridis")
        axR.set_title(
            f"{tier} — η(x,t) resuelto (score={rec['score']:.2f})",
            fontsize=10,
            color=TIER_COLOR[tier],
            fontweight="bold",
        )
        axR.set_xlabel("x [m]")
        axR.set_ylabel("t [s]")
        fig.colorbar(pcm, ax=axR, fraction=0.046, pad=0.04, label="η [m]")
        finite = np.isfinite(eta).all()
        if not finite:
            axR.text(
                0.5, 0.5, "NaN!", color="red", fontsize=20, transform=axR.transAxes, ha="center"
            )

    fig.suptitle(
        "Generador de casos 1D — progresión de complejidad (régimen transitorio libre)",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig("analysis/cases_generator_gallery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved analysis/cases_generator_gallery.png")


def difficulty_hist(grid: Grid, n_per_tier: int = 400) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    scores = {}
    for r, tier in enumerate(TIER_ORDER):
        rng = np.random.default_rng(7000 + r)
        s = []
        for _ in range(n_per_tier):
            spec = sample_case(tier, rng, grid)
            zb = bathymetry(spec.features, grid)
            s.append(difficulty_score(difficulty_components(zb, grid)))
        scores[tier] = np.array(s)

    ax = axes[0]
    for tier in TIER_ORDER:
        ax.hist(
            scores[tier],
            bins=30,
            alpha=0.55,
            color=TIER_COLOR[tier],
            label=f"{tier} (μ={scores[tier].mean():.2f})",
            density=True,
        )
    ax.set_xlabel("score de dificultad")
    ax.set_ylabel("densidad")
    ax.set_title(f"Separación de tiers ({n_per_tier} muestras c/u)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Component means per tier (what drives the score).
    ax = axes[1]
    comp_names = ["amp_ratio", "bandwidth", "sign_changes", "emergent_frac"]
    width = 0.25
    xpos = np.arange(len(comp_names))
    for i, tier in enumerate(TIER_ORDER):
        rng = np.random.default_rng(8000 + i)
        comps = [
            difficulty_components(bathymetry(sample_case(tier, rng, grid).features, grid), grid)
            for _ in range(200)
        ]
        means = [np.mean([c[n] for c in comps]) for n in comp_names]
        ax.bar(xpos + i * width, means, width, color=TIER_COLOR[tier], alpha=0.7, label=tier)
    ax.set_xticks(xpos + width)
    ax.set_xticklabels(comp_names, rotation=15, fontsize=8)
    ax.set_title("Componentes promedio por tier")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("analysis/cases_difficulty_hist.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved analysis/cases_difficulty_hist.png")
    # Print overlap diagnostics.
    print("\n=== Score por tier ===")
    for tier in TIER_ORDER:
        s = scores[tier]
        print(
            f"  {tier:7s}: mean={s.mean():.3f}  sd={s.std():.3f}  "
            f"[{np.percentile(s, 5):.2f}, {np.percentile(s, 95):.2f}] (P5-P95)"
        )


if __name__ == "__main__":
    grid = Grid()
    print(
        f"Grid: x∈[{grid.xlower},{grid.xupper}] Nx={grid.nx}, "
        f"t∈[0,{grid.t_end}] Nt={grid.n_t}, sea_level={grid.sea_level}"
    )
    gallery(grid)
    difficulty_hist(grid)
