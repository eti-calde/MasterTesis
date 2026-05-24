"""Angel et al. 2024 Hamburg flume → :class:`pinn_bath.data.Case` adapter.

Loads the processed flume `.npz` (mean of 20 runs or a single per-run
file), windows to the established-wave interval, decimates 100 Hz →
``target_hz``, snaps the requested sensors to a uniform spatial grid,
and packages the result as:

- a :class:`Case` carrying the GT bathymetry on the case grid + the
  case metadata (domain, friction κ, H_rest, BC type), and
- explicit sparse observation tensors ``(obs_coords, obs_values)`` for
  the active sensors over the decimated time grid.

The Exp 6 study passes ``obs_coords`` / ``obs_values`` directly to
:class:`pinn_bath.trainers.AdamLBFGSTrainer`, bypassing the trainer's
default random sampler. The sparse pattern is therefore explicit, not
encoded inside the ``Case`` metadata.

Default sensor layout (mirrors the legacy ``data_angel.py``):

- S0 = S1 (x = 1.5 m): soft Dirichlet inlet, included in observations.
- S1 = S2 (x = 3.5 m): canonical interior observation.
- S2 = S3 (x = 5.5 m), S3 = S4 (x = 7.5 m): optional extra sensors.

The bump peak (~ x = 3.99 m) sits between S2 and S3 — no sensor lands
on it, which is the source of the documented negative result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pinn_bath.data import Case, CaseMetadata

S1, S2, S3, S4 = 0, 1, 2, 3
INLET_SENSOR = S1


@dataclass
class AngelObservations:
    """Sparse observation tensors + bookkeeping for a windowed Angel run.

    Pass ``obs_coords`` and ``obs_values`` to
    :class:`pinn_bath.trainers.AdamLBFGSTrainer` to bypass the default
    random sampler. ``snap`` records the sensor-to-grid mapping for
    diagnostics; ``obs_sensors`` and ``bc_sensor`` echo the input.
    """

    obs_coords: dict[str, torch.Tensor]
    obs_values: dict[str, torch.Tensor]
    snap: dict[int, dict[str, float]]
    obs_sensors: tuple[int, ...]
    bc_sensor: int


def case_from_angel_flume(
    flume_npz: str | Path,
    *,
    run: int | None = None,
    t_window: tuple[float, float] = (40.0, 60.0),
    target_hz: float = 10.0,
    nx: int = 136,
    obs_sensors: tuple[int, ...] = (S2,),
    bc_sensor: int = INLET_SENSOR,
    snap_tol_m: float = 1.0e-3,
    case_id: str = "exp6_angel",
    dtype: torch.dtype = torch.float64,
) -> tuple[Case, AngelObservations]:
    """Build a ``(Case, AngelObservations)`` pair from the processed `.npz`.

    Parameters mirror the legacy ``data_angel.load_angel_windowed``. Key
    defaults:

    - ``nx = 136`` → ``dx = 100 mm``, all sensors snap exactly (err = 0).
      Other safe values: 271 (50 mm), 541 (25 mm).
    - ``run = None`` → 20-run mean; ``run = NN`` → single per-run file.
    - ``t_window = (40, 60)`` s re-zeroed to ``t ∈ [0, 20]`` (SWE +
      linear drag are time-translation invariant).
    """
    flume_npz = Path(flume_npz)
    flume = np.load(flume_npz, allow_pickle=False)
    src_path = flume_npz if run is None else flume_npz.parent / f"angel2024_run{run:02d}.npz"
    src = flume if run is None else np.load(src_path, allow_pickle=False)

    time_raw = src["time"].astype(np.float64)
    eta_raw = src["eta_obs"].astype(np.float64)
    x_sensors = flume["x_sensors"].astype(np.float64)

    xmin = float(flume["xmin"])
    xmax = float(flume["xmax"])
    kappa = float(flume["kappa"])
    H_rest = float(flume["H_rest"])
    x_bathymetry = flume["x_bathymetry"].astype(np.float64)
    zb_true = flume["zb_true"].astype(np.float64)

    # Temporal window + decimation.
    t0, t1 = t_window
    dt_dec = 1.0 / target_hz
    t_phys = np.arange(t0, t1 + 0.5 * dt_dec, dt_dec)
    eta_dec = np.empty((t_phys.size, eta_raw.shape[1]), dtype=np.float64)
    for j in range(eta_raw.shape[1]):
        eta_dec[:, j] = np.interp(t_phys, time_raw, eta_raw[:, j])
    t_grid = t_phys - t_phys[0]
    Nt = t_grid.size

    # Spatial collocation grid + sensor snap.
    x_grid = np.linspace(xmin, xmax, nx)
    dx = x_grid[1] - x_grid[0]
    active = (bc_sensor, *tuple(obs_sensors))
    snap: dict[int, dict[str, float]] = {}
    for s in active:
        xi = int(np.argmin(np.abs(x_grid - x_sensors[s])))
        err = abs(x_grid[xi] - x_sensors[s])
        if err > snap_tol_m:
            raise ValueError(
                f"Sensor {s} snap error {err * 1000:.2f} mm exceeds "
                f"snap_tol_m={snap_tol_m * 1000:.2f} mm (dx={dx * 1000:.2f} mm). "
                f"Use nx=136, 271 or 541 to align all sensors, or raise snap_tol_m."
            )
        snap[s] = {
            "x_target": float(x_sensors[s]),
            "x_node": float(x_grid[xi]),
            "node_idx": float(xi),
            "snap_err_m": float(err),
            "dx": float(dx),
        }

    # Resample GT bathymetry onto the case grid.
    zb_on_grid = np.interp(x_grid, x_bathymetry, zb_true)

    # Synthesise h, u for the Case (only used for the bathymetry RMSE +
    # to satisfy the Case schema; the actual training data is the
    # AngelObservations tensors).
    h_at_rest = np.maximum(H_rest - zb_on_grid, 1.0e-6)
    h_field = np.tile(h_at_rest, (Nt, 1))
    u_field = np.zeros((Nt, nx), dtype=np.float64)
    eta_field = h_field + zb_on_grid[None, :]

    metadata = CaseMetadata(
        case_id=case_id,
        spatial_dim=1,
        has_t=True,
        bc_type="soft_inlet_outlet",
        constants={
            "g": 9.81,
            "kappa": kappa,
            "H_rest": H_rest,
            "eps_dry": 1.0e-4,
        },
        domain={"x": [float(xmin), float(xmax)], "t": [0.0, float(t_grid[-1])]},
        gt_source="angel2024_flume",
        description=(
            f"Angel et al. 2024 Hamburg flume, window={t_window}s, "
            f"target_hz={target_hz}, sensors=S1+{list(obs_sensors)}, nx={nx}."
        ),
    )
    case = Case(
        metadata=metadata,
        coords={"x": x_grid, "t": t_grid},
        fields={
            "h": h_field,
            "u": u_field,
            "zb": zb_on_grid,
            "eta": eta_field,
        },
    )

    # Build sparse observation tensors: every (sensor_x, decimated_t)
    # pair becomes a row. Total rows = len(active_sensors) * Nt.
    rows_x: list[float] = []
    rows_t: list[float] = []
    rows_eta: list[float] = []
    for s in active:
        xi = int(snap[s]["node_idx"])
        x_node = x_grid[xi]
        for ti in range(Nt):
            rows_x.append(float(x_node))
            rows_t.append(float(t_grid[ti]))
            rows_eta.append(float(eta_dec[ti, s]))

    obs_coords = {
        "x": torch.as_tensor(rows_x, dtype=dtype).reshape(-1, 1),
        "t": torch.as_tensor(rows_t, dtype=dtype).reshape(-1, 1),
    }
    obs_values = {
        "eta": torch.as_tensor(rows_eta, dtype=dtype).reshape(-1, 1),
    }
    return case, AngelObservations(
        obs_coords=obs_coords,
        obs_values=obs_values,
        snap=snap,
        obs_sensors=tuple(obs_sensors),
        bc_sensor=bc_sensor,
    )
