"""
Ground truth generator for Experiment 5: Thacker axisymmetric paraboloid ("3D Thacker").

Reference: Thacker (1981), JFM 107, 499–508 / SWASHES §4.2.2 (Delestre et al. 2016).

Bathymetry:
    z_b(r) = -h_0 (1 - r^2 / a^2),  r = sqrt((x - x_c)^2 + (y - y_c)^2)

Analytical solution:
    omega = sqrt(8 g h_0) / a
    A     = (a^2 - r_0^2) / (a^2 + r_0^2)
    h(r, t) = h_0 [
        sqrt(1 - A^2) / (1 - A cos(omega t)) - 1
        - (r^2 / a^2) * ((1 - A^2) / (1 - A cos(omega t))^2 - 1)
    ] - z_b(r)
    u(x, y, t) = [(1/2) omega A sin(omega t) / (1 - A cos(omega t))] * (x - x_c)
    v(x, y, t) = [(1/2) omega A sin(omega t) / (1 - A cos(omega t))] * (y - y_c)

h is clipped to 0 where negative (dry bed). u, v set to 0 there.
"""

import numpy as np
from pathlib import Path


# ============================================================
# Bathymetry
# ============================================================

def bathymetry_paraboloid(X, Y, x_c=2.0, y_c=2.0, h_0=0.1, a=1.0):
    """Paraboloid of revolution: z_b = -h_0 (1 - r^2/a^2)."""
    r = np.sqrt((X - x_c) ** 2 + (Y - y_c) ** 2)
    return -h_0 * (1.0 - (r / a) ** 2)


def omega_thacker3d(h_0=0.1, a=1.0, g=9.81):
    return np.sqrt(8.0 * g * h_0) / a


def period_thacker3d(h_0=0.1, a=1.0, g=9.81):
    return 2.0 * np.pi / omega_thacker3d(h_0, a, g)


def analytical_thacker3d(x, y, t, x_c=2.0, y_c=2.0, h_0=0.1, a=1.0, r_0=0.8, g=9.81):
    """Compute analytical (h, u, v, eta, z_b) at space-time points.

    Parameters
    ----------
    x, y : np.ndarray, shape (Ny, Nx) or (N,)
    t : scalar or np.ndarray, shape (Nt,)
        If array, results are broadcast to shape (Nt, Ny, Nx) or (Nt, N).
    """
    omega = omega_thacker3d(h_0, a, g)
    A = (a ** 2 - r_0 ** 2) / (a ** 2 + r_0 ** 2)

    # bathymetry (time-independent)
    zb = -h_0 * (1.0 - ((x - x_c) ** 2 + (y - y_c) ** 2) / a ** 2)

    if np.isscalar(t):
        cos_w = np.cos(omega * t); sin_w = np.sin(omega * t)
        denom = 1.0 - A * cos_w
        ratio = np.sqrt(1.0 - A ** 2) / denom
        term1 = ratio - 1.0
        term2 = ((x - x_c) ** 2 + (y - y_c) ** 2) / a ** 2 * ((1.0 - A ** 2) / denom ** 2 - 1.0)
        h = h_0 * (term1 - term2) - zb
        h = np.maximum(h, 0.0)
        vel_coeff = 0.5 * omega * A * sin_w / denom
        u_raw = vel_coeff * (x - x_c)
        v_raw = vel_coeff * (y - y_c)
        mask_wet = h > 0
        u = np.where(mask_wet, u_raw, 0.0)
        v = np.where(mask_wet, v_raw, 0.0)
    else:
        t_arr = np.asarray(t, dtype=float)
        # Broadcasting: (Nt, 1, 1) vs (Ny, Nx)
        cos_w = np.cos(omega * t_arr)[:, None, None] if x.ndim == 2 else np.cos(omega * t_arr)[:, None]
        sin_w = np.sin(omega * t_arr)[:, None, None] if x.ndim == 2 else np.sin(omega * t_arr)[:, None]
        denom = 1.0 - A * cos_w
        ratio = np.sqrt(1.0 - A ** 2) / denom
        term1 = ratio - 1.0
        r2 = (x - x_c) ** 2 + (y - y_c) ** 2
        term2 = (r2 / a ** 2) * ((1.0 - A ** 2) / denom ** 2 - 1.0)
        h = h_0 * (term1 - term2) - zb[None, ...]
        h = np.maximum(h, 0.0)
        vel_coeff = 0.5 * omega * A * sin_w / denom
        u_raw = vel_coeff * (x - x_c)
        v_raw = vel_coeff * (y - y_c)
        mask_wet = h > 0
        u = np.where(mask_wet, u_raw, 0.0)
        v = np.where(mask_wet, v_raw, 0.0)

    eta = h + zb

    return h, u, v, eta, zb


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(
    L=4.0, Nx=40, Ny=40,
    x_c=2.0, y_c=2.0, h_0=0.1, a=1.0, r_0=0.8,
    n_periods=3, n_save=60,
    g=9.81,
    save_path=None,
):
    """Generate 3D Thacker ground truth on a Cartesian grid."""
    x = np.linspace(0.5 * L / Nx, L - 0.5 * L / Nx, Nx)
    y = np.linspace(0.5 * L / Ny, L - 0.5 * L / Ny, Ny)
    X, Y = np.meshgrid(x, y)

    T = period_thacker3d(h_0, a, g)
    t = np.linspace(0.0, n_periods * T, n_save)

    h, u, v, eta, zb = analytical_thacker3d(
        X, Y, t, x_c=x_c, y_c=y_c, h_0=h_0, a=a, r_0=r_0, g=g
    )

    params = {
        "L": L, "Nx": Nx, "Ny": Ny,
        "x_c": x_c, "y_c": y_c,
        "h_0": h_0, "a": a, "r_0": r_0,
        "n_periods": n_periods, "n_save": n_save,
        "g": g,
        "omega": float(omega_thacker3d(h_0, a, g)),
        "period": float(T),
    }
    data = {
        "x": x, "y": y, "t": t, "X": X, "Y": Y,
        "zb": zb, "h": h, "u": u, "v": v, "eta": eta,
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
        case_id="exp5_thacker_2d",
        spatial_dim=2,
        has_t=True,
        bc_type="closed",
        constants={
            "g": float(params["g"]),
            "h_0": float(params["h_0"]),
            "a": float(params["a"]),
            "r_0": float(params["r_0"]),
            "x_c": float(params["x_c"]),
            "y_c": float(params["y_c"]),
            "omega": float(params["omega"]),
            "period": float(params["period"]),
        },
        domain={
            "x": [float(data["x"].min()), float(data["x"].max())],
            "y": [float(data["y"].min()), float(data["y"].max())],
            "t": [float(data["t"].min()), float(data["t"].max())],
        },
        gt_source="analytical_thacker",
        description=(
            "Exp 5: Thacker 2D axisymmetric paraboloid oscillation (analytic, "
            "planar surface with moving dry-wet boundary)."
        ),
    )
    case = Case(
        metadata=metadata,
        coords={"x": data["x"], "y": data["y"], "t": data["t"]},
        fields={
            "h": data["h"],
            "u": data["u"],
            "v": data["v"],
            "zb": data["zb"],
            "eta": data["eta"],
        },
    )
    case.save(save_path)


# ============================================================
# Verification: check analytical solution against SWE residuals
# ============================================================

def verify_sw_residual(data):
    """Numerically check: does the analytical solution satisfy continuity + momentum?"""
    x, y, t = data["x"], data["y"], data["t"]
    h, u, v, eta, zb = data["h"], data["u"], data["v"], data["eta"], data["zb"]
    g = data["params"]["g"]

    dx = x[1] - x[0]; dy = y[1] - y[0]
    Nt = len(t)

    # Central differences (interior cells, excluding time boundaries)
    cont_res = []; momx_res = []; momy_res = []

    for k in range(1, Nt - 1):
        dt_k = 0.5 * (t[k + 1] - t[k - 1])
        h_t = (h[k + 1] - h[k - 1]) / (2 * dt_k)
        u_t = (u[k + 1] - u[k - 1]) / (2 * dt_k)
        v_t = (v[k + 1] - v[k - 1]) / (2 * dt_k)

        h_k = h[k]; u_k = u[k]; v_k = v[k]
        # Space derivatives (central)
        h_x = (h_k[:, 2:] - h_k[:, :-2]) / (2 * dx)
        h_y = (h_k[2:, :] - h_k[:-2, :]) / (2 * dy)
        u_x = (u_k[:, 2:] - u_k[:, :-2]) / (2 * dx)
        u_y = (u_k[2:, :] - u_k[:-2, :]) / (2 * dy)
        v_x = (v_k[:, 2:] - v_k[:, :-2]) / (2 * dx)
        v_y = (v_k[2:, :] - v_k[:-2, :]) / (2 * dy)
        zb_x = (zb[:, 2:] - zb[:, :-2]) / (2 * dx)
        zb_y = (zb[2:, :] - zb[:-2, :]) / (2 * dy)

        # Trim to common interior [1:-1, 1:-1]
        interior = (slice(1, -1), slice(1, -1))
        h_t_i = h_t[interior]; u_t_i = u_t[interior]; v_t_i = v_t[interior]
        h_i = h_k[interior]; u_i = u_k[interior]; v_i = v_k[interior]
        h_x_i = h_x[1:-1, :]; h_y_i = h_y[:, 1:-1]
        u_x_i = u_x[1:-1, :]; u_y_i = u_y[:, 1:-1]
        v_x_i = v_x[1:-1, :]; v_y_i = v_y[:, 1:-1]
        zb_x_i = zb_x[1:-1, :]; zb_y_i = zb_y[:, 1:-1]

        wet = h_i > 1e-4
        cont = h_t_i + h_x_i * u_i + h_i * u_x_i + h_y_i * v_i + h_i * v_y_i
        mx = u_t_i + u_i * u_x_i + v_i * u_y_i + g * (h_x_i + zb_x_i)
        my = v_t_i + u_i * v_x_i + v_i * v_y_i + g * (h_y_i + zb_y_i)

        if wet.any():
            cont_res.append(np.abs(cont[wet]).mean())
            momx_res.append(np.abs(mx[wet]).mean())
            momy_res.append(np.abs(my[wet]).mean())

    return (
        float(np.mean(cont_res)),
        float(np.mean(momx_res)),
        float(np.mean(momy_res)),
    )


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("3D Thacker paraboloid — ground truth")
    print("=" * 60)
    data = generate_dataset(
        L=4.0, Nx=40, Ny=40,
        x_c=2.0, y_c=2.0, h_0=0.1, a=1.0, r_0=0.8,
        n_periods=3, n_save=40,
    )
    p = data["params"]
    print(f"  a = {p['a']} m, h_0 = {p['h_0']} m, r_0 = {p['r_0']} m")
    print(f"  domain: [0, {p['L']}]^2 m, grid {p['Nx']}x{p['Ny']}")
    print(f"  omega = {p['omega']:.6f} rad/s, T = {p['period']:.6f} s")
    print(f"  simulation: {p['n_periods']} periods, {p['n_save']} snapshots, t_end = {data['t'][-1]:.3f} s")
    print()
    print(f"  zb range: [{data['zb'].min():.4f}, {data['zb'].max():.4f}] m")
    print(f"  h max:    {data['h'].max():.4f} m")
    print(f"  u range:  [{data['u'].min():.4f}, {data['u'].max():.4f}] m/s")
    print(f"  v range:  [{data['v'].min():.4f}, {data['v'].max():.4f}] m/s")
    print(f"  eta range: [{data['eta'].min():.4f}, {data['eta'].max():.4f}] m")

    # wet coverage
    wet_ever = (data["h"] > 1e-4).any(axis=0)
    print(f"  ever-wet fraction: {wet_ever.mean():.2%}")

    # SWE residual check
    print()
    print("Verifying analytical solution satisfies SWE (central differences on the grid)...")
    r_c, r_mx, r_my = verify_sw_residual(data)
    print(f"  mean |continuity residual|:  {r_c:.4e}")
    print(f"  mean |x-momentum residual|:  {r_mx:.4e}")
    print(f"  mean |y-momentum residual|:  {r_my:.4e}")
    print("  (expected small — these are central-difference numerical errors, not analytic ones)")
