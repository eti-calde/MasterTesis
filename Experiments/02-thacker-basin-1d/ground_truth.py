"""
Ground truth generator for Experiment 2: Thacker oscillating basin (1D, transient).

Analytical solution (planar-surface Thacker):

  z_b(x)   = h_0 * ((x/a)^2 - 1)                    [concave basin]
  h(x,t)   = max(0, h_0 * (1 - ((x + 0.5 cos(ωt))/a)^2))
  u(x,t)   = 0.5 * ω * sin(ωt)   where h > 0, else 0
  ω        = sqrt(2 g h_0) / a
  T        = 2π / ω

Reference: Dazzi et al. (2024), Thacker (1981), Delestre et al. (2013).
"""

import numpy as np
from pathlib import Path


# ============================================================
# Bathymetry and solution
# ============================================================

def bathymetry_thacker(x, a=1.0, h_0=0.5):
    """Concave parabolic basin: z_b(x) = h_0 * ((x/a)^2 - 1)."""
    return h_0 * ((x / a) ** 2 - 1.0)


def omega_thacker(a=1.0, h_0=0.5, g=9.81):
    """Angular frequency of oscillation."""
    return np.sqrt(2.0 * g * h_0) / a


def period_thacker(a=1.0, h_0=0.5, g=9.81):
    """Period of oscillation."""
    return 2.0 * np.pi / omega_thacker(a, h_0, g)


def solution_thacker(x, t, a=1.0, h_0=0.5, g=9.81):
    """Analytical Thacker solution at given space-time points.

    Parameters
    ----------
    x : np.ndarray, shape (Nx,) or scalar
    t : scalar or np.ndarray, shape (Nt,)
    a, h_0 : basin parameters
    g : gravity

    Returns
    -------
    h : water depth (max(0, ...))
    u : velocity (0 where dry)
    eta : free surface elevation = h + z_b
    zb : bed elevation (time-independent)
    """
    omega = omega_thacker(a, h_0, g)
    x = np.asarray(x, dtype=float)

    if np.isscalar(t):
        shift = 0.5 * np.cos(omega * t)
        h = h_0 * (1.0 - ((x + shift) / a) ** 2)
        h = np.maximum(h, 0.0)
        u_raw = 0.5 * omega * np.sin(omega * t)
        u = np.where(h > 0, u_raw, 0.0)
    else:
        t = np.asarray(t, dtype=float)
        # Broadcast: result shape (Nt, Nx)
        X, T = np.meshgrid(x, t)
        shift = 0.5 * np.cos(omega * T)
        h = h_0 * (1.0 - ((X + shift) / a) ** 2)
        h = np.maximum(h, 0.0)
        u_raw = 0.5 * omega * np.sin(omega * T)
        u = np.where(h > 0, u_raw, 0.0)

    zb = bathymetry_thacker(x, a, h_0)
    eta = h + zb

    return h, u, eta, zb


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(
    a=1.0, h_0=0.5,
    L=4.0, n_points_x=200,
    n_periods=1.0, n_points_t=100,
    g=9.81,
    save_path=None,
):
    """Generate a space-time ground truth dataset for Thacker T1.

    Parameters
    ----------
    a, h_0 : basin parameters
    L : domain total length (centered at x=0)
    n_points_x : spatial resolution
    n_periods : how many oscillation periods to simulate
    n_points_t : temporal resolution
    g : gravity
    save_path : if set, save as .npz

    Returns
    -------
    data : dict with x, t, h, u, eta, zb, params
        h, u, eta are shape (Nt, Nx); zb is shape (Nx,)
    """
    x = np.linspace(-L / 2, L / 2, n_points_x)

    T = period_thacker(a, h_0, g)
    t = np.linspace(0.0, n_periods * T, n_points_t)

    h, u, eta, zb = solution_thacker(x, t, a, h_0, g)

    params = {
        "a": a, "h_0": h_0, "L": L,
        "n_points_x": n_points_x, "n_points_t": n_points_t,
        "n_periods": n_periods,
        "g": g,
        "omega": float(omega_thacker(a, h_0, g)),
        "period": float(T),
    }
    data = {
        "x": x, "t": t,
        "h": h, "u": u, "eta": eta, "zb": zb,
        "params": params,
    }

    if save_path is not None:
        save_path = Path(save_path)
        _save_case(save_path, data, params)
        print(f"Dataset saved to {save_path}")

    return data


def _save_case(save_path, data, params) -> None:
    """Build a pinn_bath.data.Case and save it in the unified schema."""
    from pinn_bath.data import Case, CaseMetadata

    metadata = CaseMetadata(
        case_id="exp2_thacker_1d",
        spatial_dim=1,
        has_t=True,
        bc_type="closed",
        constants={
            "g": float(params["g"]),
            "h_0": float(params["h_0"]),
            "a": float(params["a"]),
            "omega": float(params["omega"]),
            "period": float(params["period"]),
        },
        domain={
            "x": [float(data["x"].min()), float(data["x"].max())],
            "t": [float(data["t"].min()), float(data["t"].max())],
        },
        gt_source="analytical_thacker",
        description="Exp 2: Thacker 1D parabolic basin oscillation (analytic, planar surface).",
    )
    case = Case(
        metadata=metadata,
        coords={"x": data["x"], "t": data["t"]},
        fields={"h": data["h"], "u": data["u"], "zb": data["zb"], "eta": data["eta"]},
    )
    case.save(save_path)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Thacker T1 — oscillating parabolic basin")
    print("=" * 60)
    data = generate_dataset(
        a=1.0, h_0=0.5,
        L=4.0, n_points_x=200,
        n_periods=1.0, n_points_t=100,
    )
    p = data["params"]
    print(f"  a = {p['a']} m, h_0 = {p['h_0']} m")
    print(f"  ω = {p['omega']:.6f} rad/s")
    print(f"  T = {p['period']:.6f} s")
    print(f"  domain x: [{data['x'][0]:.2f}, {data['x'][-1]:.2f}] m, Nx = {p['n_points_x']}")
    print(f"  domain t: [{data['t'][0]:.2f}, {data['t'][-1]:.4f}] s, Nt = {p['n_points_t']}")
    print()

    # Solution snapshot at t=0 vs t=T/4 (max velocity)
    print(f"At t=0 (max tilt):")
    i0 = 0
    print(f"  h range:   [{data['h'][i0].min():.4f}, {data['h'][i0].max():.4f}] m")
    print(f"  u (const): {data['u'][i0, 100]:.6f} m/s  (wet cells)")
    print(f"  wet fraction: {(data['h'][i0] > 0).mean():.2%}")

    it4 = data["t"].size // 4
    print(f"At t=T/4 (max velocity):")
    print(f"  h range:   [{data['h'][it4].min():.4f}, {data['h'][it4].max():.4f}] m")
    print(f"  u (const): {data['u'][it4, 100]:.6f} m/s  (wet cells)")
    print(f"  wet fraction: {(data['h'][it4] > 0).mean():.2%}")

    # Check conservation of bathymetry (no time dependence)
    print()
    print(f"Bathymetry range: [{data['zb'].min():.4f}, {data['zb'].max():.4f}] m")
    print(f"  (bump height from z_b floor: {data['zb'].max() - data['zb'].min():.4f} m)")
