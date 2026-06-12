#!/usr/bin/env python
"""Visual validation of the 2D datagen pipeline: solve showcase cases, render GIFs.

2D counterpart of ``preview_cases.py``: a few cases through
:class:`IncidentWaveFjord2D` + :class:`PyClawSWE2D`, each pinned to a distinct
background slope. Two render modes:

- ``--mode 3d`` (default): a 3D scene with the bed as a solid terrain surface
  and the animated free surface above it, coloured by the anomaly delta-eta.
  The wave field is vertically exaggerated (waves are cm on a ~1 m depth) by
  a factor capped so the water never visually intersects the bed crests; the
  factor is printed in the title.
- ``--mode topdown``: top-down panels (static bed + animated delta-eta heatmap
  with bed contours). ``--mode both`` renders both files per case.

The inflow edge is marked in red. Headless-safe (Agg backend, PillowWriter;
no ffmpeg / imagemagick).

Usage::

    .venv/bin/python scripts/preview_cases_2d.py            # all 3 showcases, 3D
    .venv/bin/python scripts/preview_cases_2d.py --mode both --seed 7 --out /tmp/g
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation, PillowWriter

from pinn_bath.datagen import (
    BathymetrySampler2D,
    Grid2D,
    IncidentWaveFjord2D,
    PyClawSWE2D,
    SimResult,
)

log = logging.getLogger("preview_cases_2d")

# (name, difficulty tier, pinned background slope d(zb)/dx [m/m]).
SHOWCASES: list[tuple[str, str, float]] = [
    ("2d_flat_easy", "easy", 0.0),
    ("2d_shoaling_medium", "medium", +0.0125),
    ("2d_deepening_hard", "hard", -0.0125),
]

INFLOW_COLOR = "crimson"


def render_gif(res: SimResult, path: Path, fps: int = 12, dpi: int = 80) -> None:
    """Top-down animation: static bed (left) + animated delta-eta (right)."""
    x, y, t, zb, eta = res.x, res.y, res.t, res.zb, res.eta
    still = float(res.meta["water_level"])
    side = str(res.meta["inflow_side"])
    deta = eta - still
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

    fig, (ax_b, ax_w) = plt.subplots(
        1, 2, figsize=(12.0, 4.2), sharey=True, gridspec_kw={"width_ratios": [1, 1.15]}
    )
    fig.suptitle(
        f"{res.meta['difficulty']}  |  slope={res.meta['slope']:+.4f}  "
        f"|  score={res.meta['score']:.2f}  |  tide={still:.2f} m  "
        f"|  f={res.meta['spring_neap']:.2f}  |  inflow: {side}",
        fontsize=10,
    )

    # Left: bathymetry (static).
    im_b = ax_b.imshow(zb, origin="lower", extent=extent, cmap="terrain", aspect="equal")
    fig.colorbar(im_b, ax=ax_b, shrink=0.85, label=r"$z_b$ [m]")
    ax_b.set_title("bathymetry")
    ax_b.set_xlabel("x [m]")
    ax_b.set_ylabel("y [m]")

    # Right: surface anomaly (animated) + bed contours for refraction context.
    amax = max(float(np.abs(deta).max()), 1e-3)
    im_w = ax_w.imshow(
        deta[0],
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-amax,
        vmax=amax,
        aspect="equal",
    )
    fig.colorbar(im_w, ax=ax_w, shrink=0.85, label=r"$\delta\eta$ [m]")
    levels = np.linspace(float(zb.min()), float(zb.max()), 6)[1:-1]
    if np.ptp(zb) > 1e-6:
        ax_w.contour(x, y, zb, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
    ax_w.set_xlabel("x [m]")
    title_w = ax_w.set_title("t = 0.00 s")

    # Inflow edge marker + direction arrow.
    x_edge = float(x[0] if side == "left" else x[-1])
    for ax in (ax_b, ax_w):
        ax.axvline(x_edge, color=INFLOW_COLOR, lw=2.0, alpha=0.9)
    ax_w.annotate(
        "",
        xy=(0.12 if side == "left" else 0.88, 1.06),
        xytext=(0.02 if side == "left" else 0.98, 1.06),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": INFLOW_COLOR, "lw": 1.6},
    )

    def update(k: int):
        im_w.set_data(deta[k])
        title_w.set_text(f"$\\delta\\eta$   t = {t[k]:.2f} s")
        return im_w, title_w

    anim = FuncAnimation(fig, update, frames=len(t), interval=1000.0 / fps)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def render_gif_3d(
    res: SimResult,
    path: Path,
    fps: int = 12,
    dpi: int = 80,
    exag: float = 5.0,
    stride: int = 2,
) -> None:
    """3D scene: solid terrain bed + animated free surface coloured by delta-eta.

    The wave field is exaggerated vertically by ``exag``, capped so the lowest
    exaggerated trough still clears the highest bed crest (no surface
    intersection artifacts in mplot3d's painter-style depth sort). The mesh is
    downsampled by ``stride`` for render speed.
    """
    x, y, t, zb, eta = res.x, res.y, res.t, res.zb, res.eta
    still = float(res.meta["water_level"])
    side = str(res.meta["inflow_side"])

    xs, ys = x[::stride], y[::stride]
    X, Y = np.meshgrid(xs, ys)
    Zb = zb[::stride, ::stride]
    D = (eta - still)[:, ::stride, ::stride]
    amax = max(float(np.abs(D).max()), 1e-3)
    # Cap the exaggeration: keep half the rest column over the highest crest.
    clearance = 0.5 * (still - float(Zb.max()))
    exag = float(min(exag, max(clearance / amax, 1.0)))
    norm = colors.Normalize(-amax, amax)
    cmap = matplotlib.colormaps["RdBu_r"]

    fig = plt.figure(figsize=(10.0, 6.5))
    ax = fig.add_subplot(projection="3d")
    fig.suptitle(
        f"{res.meta['difficulty']}  |  slope={res.meta['slope']:+.4f}  "
        f"|  score={res.meta['score']:.2f}  |  tide={still:.2f} m  "
        f"|  f={res.meta['spring_neap']:.2f}  |  inflow: {side}  "
        f"|  superficie: $\\delta\\eta \\times {exag:.1f}$",
        fontsize=10,
    )
    ax.plot_surface(X, Y, Zb, cmap="terrain", linewidth=0, antialiased=False, alpha=1.0)
    water = ax.plot_surface(
        X,
        Y,
        still + exag * D[0],
        facecolors=cmap(norm(D[0])),
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.85,
    )
    # Inflow edge: red line at the still water level.
    x_edge = float(x[0] if side == "left" else x[-1])
    ax.plot([x_edge] * len(ys), ys, [still] * len(ys), color=INFLOW_COLOR, lw=2.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_zlim(float(Zb.min()) - 0.1, still + exag * amax * 1.3)
    ax.set_box_aspect((2.0, 1.0, 0.55))
    ax.view_init(elev=28, azim=-65)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.08, label=r"$\delta\eta$ [m]")
    time_text = ax.text2D(0.02, 0.94, "", transform=ax.transAxes)

    def update(k: int):
        nonlocal water
        water.remove()
        water = ax.plot_surface(
            X,
            Y,
            still + exag * D[k],
            facecolors=cmap(norm(D[k])),
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.85,
        )
        time_text.set_text(f"t = {t[k]:.2f} s")
        return water, time_text

    anim = FuncAnimation(fig, update, frames=len(t), interval=1000.0 / fps)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=repo_root / "analysis" / "gifs")
    p.add_argument("--seed", type=int, default=0, help="base seed (case i uses seed+i)")
    p.add_argument("--mode", choices=["3d", "topdown", "both"], default="3d")
    p.add_argument("--exag", type=float, default=5.0, help="vertical wave exaggeration (3d mode)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=80)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    log.info("output directory: %s", args.out)

    grid = Grid2D()
    backend = PyClawSWE2D()
    n_fail = 0
    for i, (name, difficulty, slope) in enumerate(SHOWCASES):
        env = IncidentWaveFjord2D(
            grid=grid, bathymetry=BathymetrySampler2D(slope_range=(slope, slope))
        )
        rng = np.random.default_rng(args.seed + i)
        spec = env.sample_case(difficulty, rng)
        log.info(
            "[%d/%d] %s: difficulty=%s slope=%+.4f score=%.2f tide=%.2f m f=%.2f "
            "inflow=%s features=%s",
            i + 1,
            len(SHOWCASES),
            name,
            difficulty,
            spec.bathymetry.slope,
            spec.score,
            spec.water_level,
            spec.spring_neap,
            spec.forcing.side,
            [f.kind for f in spec.bathymetry.features],
        )

        t0 = time.perf_counter()
        res = env.simulate(spec, backend)
        log.info("    solved in %.1fs: eta %s", time.perf_counter() - t0, res.eta.shape)
        if not res.ok:
            log.error("    non-finite fields, skipping GIF for %s", name)
            n_fail += 1
            continue

        jobs = []
        if args.mode in ("3d", "both"):
            jobs.append((args.out / f"{i:02d}_{name}_3d.gif", "3d"))
        if args.mode in ("topdown", "both"):
            jobs.append((args.out / f"{i:02d}_{name}.gif", "topdown"))
        for gif_path, mode in jobs:
            t0 = time.perf_counter()
            if mode == "3d":
                render_gif_3d(res, gif_path, fps=args.fps, dpi=args.dpi, exag=args.exag)
            else:
                render_gif(res, gif_path, fps=args.fps, dpi=args.dpi)
            log.info(
                "    rendered %s (%.1f MB) in %.1fs",
                gif_path.name,
                gif_path.stat().st_size / 1e6,
                time.perf_counter() - t0,
            )

    if n_fail:
        log.error("%d/%d cases failed", n_fail, len(SHOWCASES))
        return 1
    log.info("done: %d GIFs in %s", len(SHOWCASES), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
