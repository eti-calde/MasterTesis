"""
Data loader for Exp. 6 — Angel et al. 2024 real flume data.

Responsibilities
----------------
- Load `angel2024_flume.npz` (mean of 20 runs + ground truth) or a single
  `angel2024_runNN.npz`.
- Window to the established-wave interval (wavemaker is silent until t~34 s),
  decimate 100 Hz -> ~target_hz, re-zero time to window start (SWE + linear
  drag are time-translation invariant).
- Build the (Nt, Nx) `eta_obs` array, `x_obs_indices`, `t_obs_indices` so the
  existing `ThackerInversePINN` sparse-observation machinery can be reused
  verbatim. Sensor S1 (x=1.5 m) is fed as an observation node = inlet
  Dirichlet eta(t); S2/S3/S4 are sparse interior observations.
- NRMSE helper, comparable to Angel's 10-14 %.

No scipy in the venv -> interpolation uses numpy.interp (data is native
100 Hz, target <= 40 Hz, so this is near-exact).
"""

from pathlib import Path
import numpy as np

DATASET = (Path(__file__).parents[1] / "datasets" / "angel2024" / "processed")
FLUME_NPZ = DATASET / "angel2024_flume.npz"

# Sensor layout (fixed by the experiment): indices into x_sensors
S1, S2, S3, S4 = 0, 1, 2, 3   # x = 1.5, 3.5, 5.5, 7.5 m
INLET_SENSOR = S1             # used as Dirichlet eta(t) at x=1.5 m


def load_angel_windowed(
    run=None,
    t_window=(40.0, 60.0),
    target_hz=10.0,
    nx=136,
    obs_sensors=(S2,),          # interior observation sensors (POC: S2 only)
    bc_sensor=INLET_SENSOR,     # inlet Dirichlet node
    snap_tol_m=1e-3,
):
    """Return everything needed to instantiate the Angel PINN.

    Parameters
    ----------
    run : None | int
        None -> flume.npz (mean of 20 runs, has ground truth).
        int  -> angel2024_run{NN}.npz (per-run; GT pulled from flume.npz).
    t_window : (t0, t1) seconds in the original (un-zeroed) timeline.
    target_hz : collocation/observation sampling rate after decimation.
    nx : number of uniform spatial collocation nodes over [xmin, xmax].
        Default 136 places dx = 100 mm so every sensor in {1.5, 3.5, 5.5,
        7.5} m lands exactly on a grid node (snap error = 0). Other safe
        choices are 271 (dx = 50 mm) and 541 (dx = 25 mm). The historical
        default nx=120 was wrong: it gave a 42 mm snap error on sensor S2,
        comparable to the ~8 mm wave amplitude, while the old guard
        ``err > 0.5*dx`` (~ 57 mm) failed to fire.
    obs_sensors : tuple of sensor indices used as interior observations.
    bc_sensor : sensor index fed as the inlet Dirichlet eta(t) node.
    snap_tol_m : maximum allowed sensor snap error in meters. Default
        1 mm (= ~ 1% of the typical Angel wave amplitude). Set higher
        only if you intentionally want a non-aligned grid (e.g., wall-time
        benchmarks where physical fidelity is not the goal).
    """
    flume = np.load(FLUME_NPZ)
    if run is None:
        src = flume
    else:
        src = np.load(DATASET / f"angel2024_run{run:02d}.npz")

    time_raw = src["time"].astype(np.float64)          # (10000,)
    eta_raw = src["eta_obs"].astype(np.float64)        # (10000, 4) water surface
    x_sensors = flume["x_sensors"].astype(np.float64)  # [1.5, 3.5, 5.5, 7.5]

    xmin = float(flume["xmin"])      # 1.5
    xmax = float(flume["xmax"])      # 15.0
    kappa = float(flume["kappa"])    # 0.2 linear drag
    H_rest = float(flume["H_rest"])  # 0.3
    x_bathymetry = flume["x_bathymetry"].astype(np.float64)  # (500,)
    zb_true = flume["zb_true"].astype(np.float64)            # (500,)

    # --- temporal window + decimation -------------------------------------
    t0, t1 = t_window
    dt_dec = 1.0 / target_hz
    t_phys = np.arange(t0, t1 + 0.5 * dt_dec, dt_dec)        # uniform target grid
    # interpolate every sensor column onto the decimated grid
    eta_dec = np.empty((t_phys.size, eta_raw.shape[1]), dtype=np.float64)
    for j in range(eta_raw.shape[1]):
        eta_dec[:, j] = np.interp(t_phys, time_raw, eta_raw[:, j])
    t_grid = t_phys - t_phys[0]                              # re-zero to window start
    Nt = t_grid.size

    # --- spatial collocation grid -----------------------------------------
    x_grid = np.linspace(xmin, xmax, nx)
    dx = x_grid[1] - x_grid[0]

    # snap each active sensor to nearest grid node
    active = (bc_sensor,) + tuple(obs_sensors)
    snap = {}
    for s in active:
        xi = int(np.argmin(np.abs(x_grid - x_sensors[s])))
        err = abs(x_grid[xi] - x_sensors[s])
        snap[s] = {"x_target": float(x_sensors[s]), "x_node": float(x_grid[xi]),
                   "node_idx": xi, "snap_err_m": float(err), "dx": float(dx)}
        if err > snap_tol_m:
            raise ValueError(
                f"Sensor {s} snap error {err*1000:.2f} mm exceeds "
                f"snap_tol_m={snap_tol_m*1000:.2f} mm (dx={dx*1000:.2f} mm). "
                f"Use nx=136, 271 or 541 to align all sensors exactly, or "
                f"raise snap_tol_m if you intentionally want a non-aligned grid."
            )

    # --- build (Nt, Nx) eta_obs; only snapped sensor columns are used ------
    eta_obs = np.zeros((Nt, nx), dtype=np.float64)
    for s in active:
        eta_obs[:, snap[s]["node_idx"]] = eta_dec[:, s]

    x_obs_indices = np.array(sorted(snap[s]["node_idx"] for s in active), dtype=int)
    t_obs_indices = np.arange(Nt, dtype=int)   # all decimated stamps observed

    return {
        "x_grid": x_grid,
        "t_grid": t_grid,
        "eta_obs": eta_obs,
        "x_obs_indices": x_obs_indices,
        "t_obs_indices": t_obs_indices,
        # ground truth + physics
        "x_bathymetry": x_bathymetry,
        "zb_true": zb_true,
        "kappa": kappa,
        "H_rest": H_rest,
        "xmin": xmin, "xmax": xmax,
        # bookkeeping
        "snap": snap,
        "x_sensors": x_sensors,
        "obs_sensors": tuple(obs_sensors),
        "bc_sensor": bc_sensor,
        "t_window": t_window,
        "target_hz": target_hz,
        "n_coll": Nt * nx,
        "Nt": Nt, "Nx": nx,
        # raw (for BC-sanity plot)
        "_time_raw": time_raw,
        "_eta_raw": eta_raw,
        "_eta_dec": eta_dec,
        "_t_phys": t_phys,
    }


def eval_bathymetry(pinn, x_query):
    """Evaluate the trained bath_net at arbitrary x (m). Uses the PINN's own
    x-normalization so it is consistent with training."""
    import torch
    xq = torch.tensor(np.asarray(x_query, dtype=np.float32).reshape(-1, 1),
                       device=pinn.device)
    with torch.no_grad():
        zb = pinn.bath_net(pinn._normalize_x(xq)).cpu().numpy().flatten()
    return zb


def nrmse(zb_pred, zb_true, x=None, span=None):
    """NRMSE = RMSE / (max - min), Angel-comparable.

    If `span=(a, b)` and `x` given, restrict to a <= x <= b.
    """
    zb_pred = np.asarray(zb_pred); zb_true = np.asarray(zb_true)
    if span is not None and x is not None:
        m = (np.asarray(x) >= span[0]) & (np.asarray(x) <= span[1])
        zb_pred, zb_true = zb_pred[m], zb_true[m]
    rmse = float(np.sqrt(np.mean((zb_pred - zb_true) ** 2)))
    rng = float(zb_true.max() - zb_true.min())
    return rmse / rng if rng > 0 else float("nan"), rmse


if __name__ == "__main__":
    d = load_angel_windowed()
    print(f"window={d['t_window']}s  target_hz={d['target_hz']}  "
          f"Nt={d['Nt']} Nx={d['Nx']}  n_coll={d['n_coll']}")
    print(f"kappa={d['kappa']}  H_rest={d['H_rest']}  "
          f"domain=[{d['xmin']},{d['xmax']}] m")
    for s, info in d["snap"].items():
        print(f"  sensor {s}: x={info['x_target']} -> node {info['node_idx']} "
              f"(x={info['x_node']:.4f}, snap_err={info['snap_err_m']*1000:.1f} mm)")
    print(f"x_obs_indices={d['x_obs_indices']}  "
          f"eta_obs range=[{d['eta_obs'][d['eta_obs']!=0].min():.4f},"
          f"{d['eta_obs'].max():.4f}]")
