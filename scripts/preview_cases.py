#!/usr/bin/env python
"""Visual validation of the datagen pipeline: solve showcase cases, render GIFs.

Generates a handful of cases through the new modular pipeline
(:mod:`pinn_bath.datagen`), each pinned to a distinct background slope
(flat / steep shoaling / steep deepening), solves them with the PyClaw
backend, and saves one GIF per case animating the free surface over the
sloped bathymetry. The inflow boundary is marked so the wave direction and
the bed interaction can be eyeballed.

Headless-safe (Agg backend, no display) and GIF encoding uses Matplotlib's
PillowWriter, so no ffmpeg / imagemagick is required.

Usage::

    .venv/bin/python scripts/preview_cases.py                # all 5 showcases
    .venv/bin/python scripts/preview_cases.py --seed 7 --fps 20 --out /tmp/gifs
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
from matplotlib.animation import FuncAnimation, PillowWriter

from pinn_bath.datagen import (
    BathymetrySampler,
    Grid1D,
    IncidentWaveFjord1D,
    PyClawSWE1D,
    SimResult,
)

log = logging.getLogger("preview_cases")

# (name, difficulty tier, pinned background slope d(zb)/dx [m/m]).
# slope > 0: bed rises toward +x (shoaling right); slope < 0: deepening right.
# +/-0.0125 are the v2 bank extremes (+/-1.25%, +/-0.25 H0 relief over 40 m).
SHOWCASES: list[tuple[str, str, float]] = [
    ("flat_easy", "easy", 0.0),
    ("shoaling_medium", "medium", +0.0125),
    ("deepening_medium", "medium", -0.0125),
    ("shoaling_hard", "hard", +0.0125),
    ("deepening_hard", "hard", -0.0125),
]

BED_COLOR = "#9c7a55"
WATER_COLOR = "#7fb3d5"
ETA_COLOR = "tab:blue"
INFLOW_COLOR = "crimson"


def render_gif(res: SimResult, path: Path, fps: int = 15, dpi: int = 80) -> None:
    """Animate one solved case: wave anomaly (zoom) + full water column."""
    x, t, zb, eta = res.x, res.t, res.zb, res.eta
    still = float(res.meta["water_level"])
    side = str(res.meta["inflow_side"])
    deta = eta - still  # surface anomaly about the case's tidal stage

    fig, (ax_w, ax_f) = plt.subplots(
        2, 1, figsize=(8.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )
    fig.suptitle(
        f"{res.meta['difficulty']}  |  slope={res.meta['slope']:+.4f}  "
        f"|  score={res.meta['score']:.2f}  |  tide={still:.2f} m  "
        f"|  f={res.meta['spring_neap']:.2f}  |  inflow: {side}",
        fontsize=10,
    )

    # Top panel: surface anomaly at wave scale (amps are cm on a ~1 m depth,
    # invisible at full-column scale).
    (line_d,) = ax_w.plot(x, deta[0], color=ETA_COLOR, lw=1.2)
    amax = max(float(np.abs(deta).max()), 1e-3) * 1.15
    ax_w.set_ylim(-amax, amax)
    ax_w.axhline(0.0, color="gray", lw=0.5)
    ax_w.set_ylabel(r"$\delta\eta$ [m]")
    # Time readout on the side opposite the inflow marker (no overlap).
    t_right = side == "left"
    time_text = ax_w.text(
        0.985 if t_right else 0.015,
        0.92,
        "",
        transform=ax_w.transAxes,
        ha="right" if t_right else "left",
        va="top",
    )

    # Bottom panel: full column. Static bed; animated water fill + surface.
    floor = float(zb.min()) - 0.2
    ax_f.fill_between(x, floor, zb, color=BED_COLOR, zorder=2)
    ax_f.plot(x, zb, color="k", lw=1.0, zorder=3)
    ax_f.axhline(still, color="steelblue", ls="--", lw=0.8, zorder=1, label="still water")
    water = ax_f.fill_between(x, zb, eta[0], color=WATER_COLOR, alpha=0.6, zorder=1)
    (line_e,) = ax_f.plot(x, eta[0], color=ETA_COLOR, lw=1.2, zorder=4)
    ax_f.set_ylim(floor, max(float(eta.max()), still) + 0.15)
    ax_f.set_xlim(float(x[0]), float(x[-1]))
    ax_f.set_xlabel("x [m]")
    ax_f.set_ylabel("elevation [m]")

    # Mark the inflow boundary on both panels, with the wave direction.
    x_edge = float(x[0] if side == "left" else x[-1])
    for ax in (ax_w, ax_f):
        ax.axvline(x_edge, color=INFLOW_COLOR, lw=1.5, alpha=0.8)
    ax_w.annotate(
        "incident wave",
        xy=(0.04 if side == "left" else 0.96, 0.86),
        xycoords="axes fraction",
        ha="left" if side == "left" else "right",
        color=INFLOW_COLOR,
        fontsize=9,
        arrowprops=None,
    )
    ax_w.annotate(
        "",
        xy=(0.14 if side == "left" else 0.86, 0.72),
        xytext=(0.04 if side == "left" else 0.96, 0.72),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": INFLOW_COLOR, "lw": 1.5},
    )
    fig.tight_layout()

    def update(k: int):
        nonlocal water
        line_d.set_ydata(deta[k])
        line_e.set_ydata(eta[k])
        water.remove()
        water = ax_f.fill_between(x, zb, eta[k], color=WATER_COLOR, alpha=0.6, zorder=1)
        time_text.set_text(f"t = {t[k]:.2f} s")
        return line_d, line_e, water, time_text

    anim = FuncAnimation(fig, update, frames=len(t), interval=1000.0 / fps)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=repo_root / "analysis" / "gifs")
    p.add_argument("--seed", type=int, default=0, help="base seed (case i uses seed+i)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--dpi", type=int, default=80)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    log.info("output directory: %s", args.out)

    grid = Grid1D()
    backend = PyClawSWE1D()
    n_fail = 0
    for i, (name, difficulty, slope) in enumerate(SHOWCASES):
        # Pin the slope via a degenerate slope_range so the deep-water cap is
        # still enforced against the forced trend (no post-hoc overrides).
        env = IncidentWaveFjord1D(
            grid=grid, bathymetry=BathymetrySampler(slope_range=(slope, slope))
        )
        rng = np.random.default_rng(args.seed + i)
        spec = env.sample_case(difficulty, rng)
        log.info(
            "[%d/%d] %s: difficulty=%s slope=%+.3f score=%.2f tide=%.2f m inflow=%s periods=%s",
            i + 1,
            len(SHOWCASES),
            name,
            difficulty,
            spec.bathymetry.slope,
            spec.score,
            spec.water_level,
            spec.forcing.side,
            [f"{p_:.1f}s" for p_ in spec.forcing.periods],
        )

        t0 = time.perf_counter()
        res = env.simulate(spec, backend)
        log.info("    solved in %.2fs: eta %s", time.perf_counter() - t0, res.eta.shape)
        if not res.ok:
            log.error("    non-finite fields, skipping GIF for %s", name)
            n_fail += 1
            continue

        gif_path = args.out / f"{i:02d}_{name}.gif"
        t0 = time.perf_counter()
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
