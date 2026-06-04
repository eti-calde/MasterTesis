"""Visualize generator cases: animated GIFs + a 3x3 dataset gallery.

- GIFs: 2 easy + 2 medium + 2 hard, each animating the free surface eta(x,t)
  as the incident wave train enters and propagates over the (fixed) bed zb(x).
  For showing the advisor / visual sanity checks — NOT for the PDF. Saved to
  analysis/gifs/.
- Gallery: a 3x3 grid (3 easy / 3 medium / 3 hard) of bathymetry + a couple of
  eta snapshots, a static overview of what the dataset contains.

Run:  .venv/bin/python scripts/visualize_cases.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
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

TIER_COLOR = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728"}
OUT = Path("analysis")
GIF_DIR = OUT / "gifs"


def _pick_cases(tier: str, n: int, base_seed: int) -> list:
    """Sample n reproducible cases for a tier (distinct rng seeds)."""
    cases = []
    for i in range(n):
        rng = np.random.default_rng(base_seed + i)
        cases.append(sample_case(tier, rng, Grid()))
    return cases


def make_gif(tier: str, idx: int, spec, grid: Grid) -> None:
    rec = generate_record(spec, grid, kernel="aug", cfl_desired=0.45)
    x, t, eta, zb = rec["x"], rec["t"], rec["eta"], rec["zb"]
    sea = grid.sea_level

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(x, -0.2, zb, color="#8c6d3f", alpha=0.9, zorder=2)  # bed
    ax.plot(x, zb, color="#5b4424", lw=1.5, zorder=3)
    (water,) = ax.plot([], [], color="#1f77b4", lw=2, zorder=4)
    fill = [None]
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(min(-0.1, zb.min() - 0.1), max(eta.max(), sea) + 0.15)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("altura [m]")
    title = ax.set_title("")
    score = rec["score"]

    def update(k):
        if fill[0] is not None:
            fill[0].remove()
        fill[0] = ax.fill_between(x, zb, eta[k], color="#aac9e8", alpha=0.6, zorder=1)
        water.set_data(x, eta[k])
        title.set_text(f"{tier} (score={score:.2f}) — t = {t[k]:.2f} s")
        return water, fill[0], title

    anim = animation.FuncAnimation(fig, update, frames=range(0, len(t), 2), blit=False)
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    path = GIF_DIR / f"{tier}_{idx}.gif"
    anim.save(path, writer=animation.PillowWriter(fps=15))
    plt.close(fig)
    print(f"  saved {path}  (score={score:.2f})")


def make_gallery(grid: Grid) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for r, tier in enumerate(["easy", "medium", "hard"]):
        specs = _pick_cases(tier, 3, base_seed=2000 + 100 * r)
        for c, spec in enumerate(specs):
            ax = axes[r, c]
            zb = bathymetry(spec.features, grid)
            sc = difficulty_score(difficulty_components(zb, grid))
            ax.fill_between(grid.centers, min(-0.2, zb.min() - 0.1), zb,
                            color="#8c6d3f", alpha=0.85)
            ax.plot(grid.centers, zb, color="#5b4424", lw=1.3)
            ax.axhline(grid.sea_level, color="#1f77b4", ls="--", lw=1, alpha=0.6)
            ax.set_ylim(min(-0.3, zb.min() - 0.1), 1.25 * grid.sea_level)
            ax.set_title(f"{tier} (score={sc:.2f})", color=TIER_COLOR[tier],
                         fontsize=10, fontweight="bold")
            if r == 2:
                ax.set_xlabel("x [m]")
            if c == 0:
                ax.set_ylabel("$z_b$ [m]")
            ax.grid(alpha=0.3)
    fig.suptitle(
        "Casos representativos del banco de pruebas (3 por nivel de dificultad)\n"
        "marrón: batimetría $z_b$ · línea azul punteada: nivel de reposo $\\eta_0$",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "dataset_gallery_3x3.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    grid = Grid()
    print("=== GIFs (2 por nivel) ===")
    gif_seeds = {"easy": 1000, "medium": 1100, "hard": 1200}
    for tier in ["easy", "medium", "hard"]:
        specs = _pick_cases(tier, 2, base_seed=gif_seeds[tier])
        for i, spec in enumerate(specs):
            make_gif(tier, i, spec, grid)
    print("=== galería 3x3 ===")
    make_gallery(grid)
