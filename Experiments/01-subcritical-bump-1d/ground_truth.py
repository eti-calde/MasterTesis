"""
Ground truth generator for Experiment 1: Subcritical flow over a bump (1D, steady).

Solves the Bernoulli energy equation analytically:
    q^2 / (2*g*h^2) + h + z_b(x) = C

where C is determined by the downstream boundary condition.

References:
    - Dazzi et al. (2024), WRR, e2023WR036589
    - Delestre et al. (2013), SWASHES library
    - Ruppenthal & Kuzmin (2026), arXiv:2603.11813
"""

import numpy as np
from pathlib import Path


# --- Bathymetry profiles ---

def bump_parabolic(x, x0=0.0, height=0.2, half_width=2.0):
    """Parabolic bump as in Dazzi (2024) and SWASHES.

    Formula (matching Dazzi's bump_problems.py exactly):
        z_b(x) = height - (height/half_width^2) * (x - x0)^2  for |x - x0| < half_width
        z_b(x) = 0  elsewhere

    Dazzi B1 defaults: x0=0 (center of [-10,10] domain), height=0.2, half_width=2.
    SWASHES defaults: x0=10 (center of [0,25] domain), height=0.2, half_width=2.
    """
    zb = np.zeros_like(x)
    # Dazzi clips with z[z<0] = 0, equivalent to |x-x0| < half_width check
    zb_candidate = height - (height / half_width**2) * (x - x0)**2
    mask = zb_candidate > 0
    zb[mask] = zb_candidate[mask]
    return zb


def bump_gaussian(x, x0=0.0, amplitude=0.2, sigma=1.0):
    """Gaussian bump for smoother bathymetry.
    z_b(x) = A * exp(-(x - x0)^2 / (2*sigma^2))
    """
    return amplitude * np.exp(-((x - x0)**2) / (2 * sigma**2))


# --- Bernoulli solver ---

def solve_bernoulli_subcritical(x, zb, q, h_downstream, g=9.81):
    """Solve steady 1D Bernoulli equation for subcritical flow over known bathymetry.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates (1D array, sorted).
    zb : np.ndarray
        Bed elevation at each x.
    q : float
        Specific discharge (m^2/s), constant along the channel.
    h_downstream : float
        Water depth at the downstream boundary (last point of x).
    g : float
        Gravitational acceleration.

    Returns
    -------
    h : np.ndarray
        Water depth at each x.
    u : np.ndarray
        Depth-averaged velocity at each x.
    eta : np.ndarray
        Free surface elevation (h + zb) at each x.
    """
    # Bernoulli constant from downstream BC
    C = q**2 / (2 * g * h_downstream**2) + h_downstream + zb[-1]

    # Critical depth (minimum possible depth for this discharge)
    h_crit = (q**2 / g) ** (1.0 / 3.0)

    h = np.zeros_like(x)

    for i in range(len(x)):
        # Cubic: h^3 + (zb - C)*h^2 + q^2/(2g) = 0
        # Rewritten as: h^3 + a*h^2 + b = 0
        a = zb[i] - C
        b = q**2 / (2 * g)
        coeffs = [1.0, a, 0.0, b]
        roots = np.roots(coeffs)

        # Select the subcritical root: real, positive, and > h_crit
        real_roots = roots[np.isreal(roots)].real
        valid = real_roots[real_roots > h_crit]

        if len(valid) == 0:
            raise ValueError(
                f"No subcritical root found at x={x[i]:.3f}, zb={zb[i]:.4f}. "
                f"Flow may be supercritical or bump is too high for this discharge."
            )
        h[i] = np.min(valid)  # closest to critical = the subcritical solution

    u = q / h
    eta = h + zb

    return h, u, eta


# --- Friction extension (Manning) ---

def solve_with_manning(x, zb, q, h_downstream, n_manning, g=9.81, tol=1e-10, max_iter=100):
    """Solve steady 1D SWE with Manning friction via iterative backwater computation.

    Uses the gradually varied flow (GVF) equation integrated from downstream:
        dh/dx = (S_0 - S_f) / (1 - Fr^2)
    where S_0 = -dz_b/dx and S_f = n^2 * q^2 / h^(10/3).

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates (sorted, uniform or non-uniform spacing).
    zb : np.ndarray
        Bed elevation at each x.
    q : float
        Specific discharge (m^2/s).
    h_downstream : float
        Water depth at the downstream boundary.
    n_manning : float
        Manning roughness coefficient.
    g : float
        Gravitational acceleration.

    Returns
    -------
    h, u, eta : np.ndarray
        Water depth, velocity, and free surface elevation.
    """
    n_pts = len(x)
    h = np.zeros(n_pts)
    h[-1] = h_downstream

    # Integrate backwards from downstream using 4th-order Runge-Kutta
    for i in range(n_pts - 2, -1, -1):
        dx = x[i + 1] - x[i]

        def dhdx(x_loc, h_loc, zb_loc, dzb_dx):
            S_0 = -dzb_dx
            S_f = n_manning**2 * q**2 / h_loc**(10.0 / 3.0)
            Fr2 = q**2 / (g * h_loc**3)
            return (S_0 - S_f) / (1 - Fr2)

        # Bed slope at midpoint and endpoints
        if i + 2 < n_pts:
            dzb_ip1 = (zb[i + 2] - zb[i]) / (x[i + 2] - x[i])
        else:
            dzb_ip1 = (zb[i + 1] - zb[i]) / (x[i + 1] - x[i])
        dzb_i = (zb[i + 1] - zb[i]) / dx
        dzb_mid = 0.5 * (dzb_i + dzb_ip1)

        # RK4 (backwards: from i+1 to i, so step is -dx)
        k1 = dhdx(x[i + 1], h[i + 1], zb[i + 1], dzb_ip1)
        k2 = dhdx(x[i + 1] - 0.5 * dx, h[i + 1] - 0.5 * dx * k1, 0.5 * (zb[i] + zb[i + 1]), dzb_mid)
        k3 = dhdx(x[i + 1] - 0.5 * dx, h[i + 1] - 0.5 * dx * k2, 0.5 * (zb[i] + zb[i + 1]), dzb_mid)
        k4 = dhdx(x[i], h[i + 1] - dx * k3, zb[i], dzb_i)

        h[i] = h[i + 1] - dx * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    u = q / h
    eta = h + zb

    return h, u, eta


# --- Dataset generation ---

def generate_dataset(
    L=25.0,
    n_points=500,
    x_start=None,
    bump_type="parabolic",
    bump_params=None,
    q=4.42,
    h_downstream=2.0,
    n_manning=0.0,
    g=9.81,
    save_path=None,
):
    """Generate a complete ground truth dataset for the subcritical bump case.

    Parameters
    ----------
    L : float
        Channel length (m).
    n_points : int
        Number of spatial points.
    x_start : float or None
        Start of domain. If None, defaults to 0.
        Dazzi B1 uses x_start=-10 (domain [-10, 10]).
        SWASHES uses x_start=0 (domain [0, 25]).
    bump_type : str
        "parabolic" or "gaussian".
    bump_params : dict or None
        Parameters for the bump function. If None, uses defaults.
    q : float
        Specific discharge (m^2/s).
    h_downstream : float
        Downstream water depth (m).
    n_manning : float
        Manning coefficient. 0.0 for frictionless.
    g : float
        Gravitational acceleration.
    save_path : str or Path or None
        If provided, saves the dataset as .npz file.

    Returns
    -------
    data : dict
        Dictionary with keys: x, zb, h, u, eta, params.
    """
    if x_start is None:
        x_start = 0.0
    x = np.linspace(x_start, x_start + L, n_points)

    # Build bathymetry
    if bump_params is None:
        bump_params = {}
    if bump_type == "parabolic":
        zb = bump_parabolic(x, **bump_params)
    elif bump_type == "gaussian":
        zb = bump_gaussian(x, **bump_params)
    else:
        raise ValueError(f"Unknown bump_type: {bump_type}")

    # Solve
    if n_manning == 0.0:
        h, u, eta = solve_bernoulli_subcritical(x, zb, q, h_downstream, g)
    else:
        h, u, eta = solve_with_manning(x, zb, q, h_downstream, n_manning, g)

    # Froude number
    Fr = u / np.sqrt(g * h)

    # Package
    params = {
        "L": L,
        "n_points": n_points,
        "x_start": x_start,
        "bump_type": bump_type,
        "bump_params": bump_params,
        "q": q,
        "h_downstream": h_downstream,
        "n_manning": n_manning,
        "g": g,
    }

    data = {"x": x, "zb": zb, "h": h, "u": u, "eta": eta, "Fr": Fr, "params": params}

    if save_path is not None:
        save_path = Path(save_path)
        _save_case(save_path, data, params)
        print(f"Dataset saved to {save_path}")

    return data


def _save_case(save_path, data, params) -> None:
    """Build a pinn_bath.data.Case and save it in the unified schema."""
    from pinn_bath.data import Case, CaseMetadata

    bump_type = params["bump_type"]
    n_manning = float(params["n_manning"])
    gt_source = "analytical_bernoulli" if n_manning == 0.0 else "analytical_manning"
    description_parts = [
        f"Exp 1: subcritical flow over a {bump_type} bump (1D, steady).",
        "Bernoulli analytic." if n_manning == 0.0 else f"Manning friction (n={n_manning}).",
    ]
    # Bump support: x_0 (center) and w (half-width of the region where z_b != 0).
    # The flat-bed prior `pinn_bath.losses.flat_bed_loss` enforces z_b = 0
    # outside [x_0 - w, x_0 + w]. For parabolic bumps this matches the
    # piecewise definition; for Gaussian bumps we use 3*sigma as a soft
    # support where the bump has decayed to < 1% of its peak.
    bump_params = params.get("bump_params") or {}
    x_0 = float(bump_params.get("x0", 0.0))
    if bump_type == "parabolic":
        w = float(bump_params.get("half_width", 2.0))
    elif bump_type == "gaussian":
        w = 3.0 * float(bump_params.get("sigma", 1.0))
    else:
        w = 2.0  # generic fallback
    metadata = CaseMetadata(
        case_id=f"exp1_{bump_type}",
        spatial_dim=1,
        has_t=False,
        bc_type="open_dirichlet",
        constants={
            "g": float(params["g"]),
            "q": float(params["q"]),
            "h_down": float(params["h_downstream"]),
            "n_manning": n_manning,
            "x_0": x_0,
            "w": w,
        },
        domain={"x": [float(data["x"].min()), float(data["x"].max())]},
        gt_source=gt_source,
        description=" ".join(description_parts),
    )
    case = Case(
        metadata=metadata,
        coords={"x": data["x"]},
        fields={"h": data["h"], "u": data["u"], "zb": data["zb"], "eta": data["eta"]},
    )
    case.save(save_path)


# --- Quick sanity check ---

if __name__ == "__main__":
    # Dazzi B1 exact case: domain [-10, 10], bump at x=0
    print("=" * 60)
    print("Dazzi B1 exact (q=4.42, h_down=2.0, no friction)")
    print("=" * 60)
    data = generate_dataset(
        L=20.0, n_points=500, x_start=-10.0,
        bump_type="parabolic",
        bump_params={"x0": 0.0, "height": 0.2, "half_width": 2.0},
        q=4.42, h_downstream=2.0, n_manning=0.0,
    )
    print(f"  x range: [{data['x'][0]:.1f}, {data['x'][-1]:.1f}] m")
    print(f"  zb max:  {data['zb'].max():.4f} m")
    print(f"  h range: [{data['h'].min():.4f}, {data['h'].max():.4f}] m")
    print(f"  u range: [{data['u'].min():.4f}, {data['u'].max():.4f}] m/s")
    print(f"  eta range: [{data['eta'].min():.4f}, {data['eta'].max():.4f}] m")
    print(f"  Fr range: [{data['Fr'].min():.4f}, {data['Fr'].max():.4f}]")
    print(f"  Fr max < 1? {data['Fr'].max() < 1.0}  (subcritical check)")

    # SWASHES standard: domain [0, 25], bump at x=10
    print()
    print("=" * 60)
    print("SWASHES standard (q=4.42, h_down=2.0, no friction)")
    print("=" * 60)
    data2 = generate_dataset(
        L=25.0, n_points=500,
        bump_type="parabolic",
        bump_params={"x0": 10.0, "height": 0.2, "half_width": 2.0},
        q=4.42, h_downstream=2.0, n_manning=0.0,
    )
    print(f"  x range: [{data2['x'][0]:.1f}, {data2['x'][-1]:.1f}] m")
    print(f"  h range: [{data2['h'].min():.4f}, {data2['h'].max():.4f}] m")
    print(f"  Fr max < 1? {data2['Fr'].max() < 1.0}  (subcritical check)")
