"""
Ground truth generator for Experiment 3: Two cylinders in 2D.

Finite-volume solver for the 2D shallow water equations with topography,
following a standard HLL-type scheme on a Cartesian grid.

Conservative form:
    U_t + F(U)_x + G(U)_y = S(U, z_b)

where U = (h, hu, hv), F and G are the flux functions, S is the topography
source term.

Reference case: Ruppenthal & Kuzmin (2026), Section 7.2.
    Domain:      [0, 25] x [0, 25] m
    Grid:        50 x 50  (dx = dy = 0.5 m)
    Cylinders:   sharp indicators (vertical walls, no smoothing)
                   - cyl 1: center (8, 8), radius 4, height 0.2 m
                   - cyl 2: center (15, 15), radius 2, height 0.3 m
                 placed sequentially along the (2.21, 2.21) flow diagonal.
    IC:          free surface eta = h + zb = 2.0 m (uniform);
                 velocity (u, v) = (2.21, 2.21) m/s (uniform);
                 depth h(x, y, 0) = max(eta - zb, h_min) varies with bathymetry.
    Time:        T = 60 s, dt = 1e-2 s.
    Solver:      FV-HLL (LeVeque 2002); Ruppenthal uses FEM-MCL on the same grid.
"""

import numpy as np
from pathlib import Path


# ============================================================
# Bathymetry
# ============================================================

def bathymetry_two_cylinders(X, Y,
                              cyls=((8.0, 8.0, 4.0, 0.2),
                                    (15.0, 15.0, 2.0, 0.3)),
                              smooth=0.0):
    """Two solid cylindrical bumps on a flat bed.

    Defaults reproduce Ruppenthal & Kuzmin (2026) §7.2: vertical-walled
    cylinders along the flow diagonal. ``smooth > 0`` would add tanh-smoothed
    edges; the paper uses sharp indicators (``smooth = 0``).

    Parameters
    ----------
    X, Y : np.ndarray, shape (Ny, Nx)
    cyls : tuple of (x_center, y_center, radius, height)
    smooth : width of the tanh edge smoothing (0 for sharp)
    """
    zb = np.zeros_like(X)
    for (xc, yc, r, H) in cyls:
        dist = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
        if smooth <= 0:
            zb += H * (dist <= r).astype(float)
        else:
            # smooth step: 1 inside, 0 outside, tanh transition
            zb += H * 0.5 * (1.0 - np.tanh((dist - r) / smooth))
    return zb


# ============================================================
# Finite Volume SWE Solver (HLL)
# ============================================================

def physical_flux_x(h, hu, hv, g):
    """Physical flux in x direction."""
    eps = 1e-8
    u = hu / (h + eps)
    F_h = hu
    F_hu = hu * u + 0.5 * g * h * h
    F_hv = hv * u
    return F_h, F_hu, F_hv


def physical_flux_y(h, hu, hv, g):
    """Physical flux in y direction."""
    eps = 1e-8
    v = hv / (h + eps)
    G_h = hv
    G_hu = hu * v
    G_hv = hv * v + 0.5 * g * h * h
    return G_h, G_hu, G_hv


def hll_flux_x(hL, huL, hvL, hR, huR, hvR, g):
    """HLL Riemann flux in x."""
    eps = 1e-8
    uL = huL / (hL + eps)
    uR = huR / (hR + eps)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))

    # Wave speed estimates (Einfeldt)
    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)

    FL_h, FL_hu, FL_hv = physical_flux_x(hL, huL, hvL, g)
    FR_h, FR_hu, FR_hv = physical_flux_x(hR, huR, hvR, g)

    # HLL formula
    denom = (sR - sL) + 1e-12
    F_h = np.where(sL >= 0, FL_h,
          np.where(sR <= 0, FR_h,
                   (sR * FL_h - sL * FR_h + sL * sR * (hR - hL)) / denom))
    F_hu = np.where(sL >= 0, FL_hu,
           np.where(sR <= 0, FR_hu,
                    (sR * FL_hu - sL * FR_hu + sL * sR * (huR - huL)) / denom))
    F_hv = np.where(sL >= 0, FL_hv,
           np.where(sR <= 0, FR_hv,
                    (sR * FL_hv - sL * FR_hv + sL * sR * (hvR - hvL)) / denom))
    return F_h, F_hu, F_hv


def hll_flux_y(hL, huL, hvL, hR, huR, hvR, g):
    """HLL Riemann flux in y (L/R interpreted as bottom/top)."""
    eps = 1e-8
    vL = hvL / (hL + eps)
    vR = hvR / (hR + eps)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))

    sL = np.minimum(vL - cL, vR - cR)
    sR = np.maximum(vL + cL, vR + cR)

    GL_h, GL_hu, GL_hv = physical_flux_y(hL, huL, hvL, g)
    GR_h, GR_hu, GR_hv = physical_flux_y(hR, huR, hvR, g)

    denom = (sR - sL) + 1e-12
    G_h = np.where(sL >= 0, GL_h,
          np.where(sR <= 0, GR_h,
                   (sR * GL_h - sL * GR_h + sL * sR * (hR - hL)) / denom))
    G_hu = np.where(sL >= 0, GL_hu,
           np.where(sR <= 0, GR_hu,
                    (sR * GL_hu - sL * GR_hu + sL * sR * (huR - huL)) / denom))
    G_hv = np.where(sL >= 0, GL_hv,
           np.where(sR <= 0, GR_hv,
                    (sR * GL_hv - sL * GR_hv + sL * sR * (hvR - hvL)) / denom))
    return G_h, G_hu, G_hv


def _pad_x(h, hu, hv, zb, *, eta_inflow=None, u_inflow=None, v_inflow=None):
    """Build x-padded arrays for the Audusse reconstruction.

    Default: ``mode="edge"`` (Neumann zero-gradient) on both x-faces. When
    ``eta_inflow`` is given, override column 0 (small-x face) with Dirichlet
    inflow values ``(h = max(eta_inflow - zb_ghost, 0), hu = u_inflow * h,
    hv = v_inflow * h)``; column -1 (large-x face) remains Neumann (outflow).
    ``zb`` is always padded ``mode="edge"`` since the boundary cells of
    Ruppenthal §7.2 sit on the flat bed (``zb = 0``).
    """
    h_pad = np.pad(h, ((0, 0), (1, 1)), mode="edge")
    hu_pad = np.pad(hu, ((0, 0), (1, 1)), mode="edge")
    hv_pad = np.pad(hv, ((0, 0), (1, 1)), mode="edge")
    zb_pad = np.pad(zb, ((0, 0), (1, 1)), mode="edge")
    if eta_inflow is not None:
        h_in = np.maximum(eta_inflow - zb_pad[:, 0], 0.0)
        h_pad[:, 0] = h_in
        hu_pad[:, 0] = u_inflow * h_in
        hv_pad[:, 0] = v_inflow * h_in
    return h_pad, hu_pad, hv_pad, zb_pad


def _pad_y(h, hu, hv, zb, *, eta_inflow=None, u_inflow=None, v_inflow=None):
    """Build y-padded arrays. Mirror of :func:`_pad_x` along the y axis.

    Default: Neumann on both y-faces. When ``eta_inflow`` is given, row 0
    (small-y face) becomes Dirichlet inflow; row -1 stays Neumann (outflow).
    """
    h_pad = np.pad(h, ((1, 1), (0, 0)), mode="edge")
    hu_pad = np.pad(hu, ((1, 1), (0, 0)), mode="edge")
    hv_pad = np.pad(hv, ((1, 1), (0, 0)), mode="edge")
    zb_pad = np.pad(zb, ((1, 1), (0, 0)), mode="edge")
    if eta_inflow is not None:
        h_in = np.maximum(eta_inflow - zb_pad[0, :], 0.0)
        h_pad[0, :] = h_in
        hu_pad[0, :] = u_inflow * h_in
        hv_pad[0, :] = v_inflow * h_in
    return h_pad, hu_pad, hv_pad, zb_pad


def _audusse_x(h_pad, hu_pad, hv_pad, zb_pad, g):
    """Audusse hydrostatic reconstruction at x-interfaces (well-balanced).

    Takes already-padded arrays (see :func:`_pad_x` for BC dispatch). For
    each interface ``k`` (separating cell ``k-1`` to the left from cell
    ``k`` to the right), reconstructs ``h*_L = max(0, h_{k-1} + zb_{k-1} -
    z*)`` and ``h*_R = max(0, h_k + zb_k - z*)`` with ``z* = max(zb_{k-1},
    zb_k)``; preserves velocities; applies HLL on the reconstructed states;
    adds the asymmetric pressure correction ``(g/2)*(h^2 - h*^2)`` to the
    momentum flux. The correction on the LEFT-cell side and the RIGHT-cell
    side differ because the local depths ``h_{k-1}`` and ``h_k`` differ.

    Returns ``(F_h, F_L_hu, F_R_hu, F_hv)`` of shape ``(Ny, Nx+1)``. ``F_h``
    and ``F_hv`` are symmetric (no pressure correction in the continuity or
    cross-momentum components); only ``F_L_hu`` vs ``F_R_hu`` carry the
    asymmetric topography contribution.
    """
    eps = 1e-8
    hL = h_pad[:, :-1];  huL = hu_pad[:, :-1];  hvL = hv_pad[:, :-1]
    hR = h_pad[:, 1:];   huR = hu_pad[:, 1:];   hvR = hv_pad[:, 1:]
    zbL = zb_pad[:, :-1]
    zbR = zb_pad[:, 1:]

    z_star = np.maximum(zbL, zbR)
    hL_star = np.maximum(0.0, hL + zbL - z_star)
    hR_star = np.maximum(0.0, hR + zbR - z_star)
    uL = huL / (hL + eps)
    uR = huR / (hR + eps)
    vL = hvL / (hL + eps)
    vR = hvR / (hR + eps)
    huL_star = hL_star * uL
    huR_star = hR_star * uR
    hvL_star = hL_star * vL
    hvR_star = hR_star * vR

    F_h, F_hu_star, F_hv = hll_flux_x(
        hL_star, huL_star, hvL_star,
        hR_star, huR_star, hvR_star,
        g,
    )

    Phi_L = 0.5 * g * (hL * hL - hL_star * hL_star)
    Phi_R = 0.5 * g * (hR * hR - hR_star * hR_star)
    F_L_hu = F_hu_star + Phi_L
    F_R_hu = F_hu_star + Phi_R
    return F_h, F_L_hu, F_R_hu, F_hv


def _audusse_y(h_pad, hu_pad, hv_pad, zb_pad, g):
    """Audusse HR at y-interfaces (well-balanced).

    Takes already-padded arrays (see :func:`_pad_y`). Mirror of
    :func:`_audusse_x` along the y axis. Asymmetric pressure correction
    lands on the y-momentum component (``hv``); ``G_h`` and ``G_hu``
    (cross-momentum) stay symmetric.

    Returns ``(G_h, G_hu, G_B_hv, G_T_hv)`` of shape ``(Ny+1, Nx)``. At
    interface ``k`` separating cell ``k-1`` (bottom = "B") from cell ``k``
    (top = "T"), ``G_B_hv`` is the flux seen by the bottom cell and
    ``G_T_hv`` the flux seen by the top cell.
    """
    eps = 1e-8
    hB = h_pad[:-1, :];  huB = hu_pad[:-1, :];  hvB = hv_pad[:-1, :]
    hT = h_pad[1:, :];   huT = hu_pad[1:, :];   hvT = hv_pad[1:, :]
    zbB = zb_pad[:-1, :]
    zbT = zb_pad[1:, :]

    z_star = np.maximum(zbB, zbT)
    hB_star = np.maximum(0.0, hB + zbB - z_star)
    hT_star = np.maximum(0.0, hT + zbT - z_star)
    uB = huB / (hB + eps); uT = huT / (hT + eps)
    vB = hvB / (hB + eps); vT = hvT / (hT + eps)
    huB_star = hB_star * uB; huT_star = hT_star * uT
    hvB_star = hB_star * vB; hvT_star = hT_star * vT

    G_h, G_hu, G_hv_star = hll_flux_y(
        hB_star, huB_star, hvB_star,
        hT_star, huT_star, hvT_star,
        g,
    )

    Phi_B = 0.5 * g * (hB * hB - hB_star * hB_star)
    Phi_T = 0.5 * g * (hT * hT - hT_star * hT_star)
    G_B_hv = G_hv_star + Phi_B
    G_T_hv = G_hv_star + Phi_T
    return G_h, G_hu, G_B_hv, G_T_hv


def run_fv_simulation(zb, h0, hu0, hv0, dx, dy, t_end, cfl=0.3, g=9.81,
                      n_save=60, verbose=False, *,
                      eta_inflow=None, u_inflow=None, v_inflow=None):
    """Run a 2D SWE FV simulation with Audusse hydrostatic reconstruction.

    Uses HLL Riemann fluxes on Audusse-reconstructed interface states
    (see :func:`_audusse_x`, :func:`_audusse_y`), so the topography source
    term is absorbed into asymmetric pressure corrections at each
    interface. This is well-balanced for discontinuous ``zb`` — lake-at-rest
    states (``u = v = 0``, ``h + zb`` constant) are preserved to machine
    precision even across the sharp cylinder walls of Ruppenthal §7.2.

    Boundary conditions: by default all four faces use Neumann zero-gradient
    (``mode="edge"``). When ``eta_inflow`` is supplied, the small-x and
    small-y faces become Dirichlet inflow, prescribing the uniform state
    ``(h = eta_inflow - zb_ghost, hu = u_inflow * h, hv = v_inflow * h)`` via
    ghost cells; the large-x and large-y faces remain Neumann (outflow). All
    three (``eta_inflow``, ``u_inflow``, ``v_inflow``) must be provided when
    any is given.

    Parameters
    ----------
    zb : (Ny, Nx) bathymetry
    h0, hu0, hv0 : (Ny, Nx) initial conditions
    dx, dy : grid spacing
    t_end : final time
    cfl : Courant number
    g : gravity
    n_save : number of time slices to save
    verbose : print progress
    eta_inflow, u_inflow, v_inflow : Dirichlet inflow state at small-x and
        small-y faces. If ``None`` (default), all faces are Neumann.

    Returns
    -------
    t_saved : (n_save,) times of saved snapshots
    h_hist, hu_hist, hv_hist : (n_save, Ny, Nx) saved states
    """
    if eta_inflow is not None and (u_inflow is None or v_inflow is None):
        raise ValueError(
            "u_inflow and v_inflow are required when eta_inflow is given"
        )
    Ny, Nx = zb.shape
    h = h0.copy()
    hu = hu0.copy()
    hv = hv0.copy()

    t = 0.0
    save_times = np.linspace(0, t_end, n_save)
    t_saved = [0.0]
    h_hist = [h.copy()]
    hu_hist = [hu.copy()]
    hv_hist = [hv.copy()]
    next_save_idx = 1

    step = 0
    while t < t_end:
        eps = 1e-8
        u = hu / (h + eps)
        v = hv / (h + eps)
        c = np.sqrt(g * np.maximum(h, 0.0))
        max_speed = max(np.max(np.abs(u) + c), np.max(np.abs(v) + c), 1e-6)
        dt = cfl * min(dx, dy) / max_speed
        if t + dt > t_end:
            dt = t_end - t

        h_padx, hu_padx, hv_padx, zb_padx = _pad_x(
            h, hu, hv, zb,
            eta_inflow=eta_inflow, u_inflow=u_inflow, v_inflow=v_inflow,
        )
        h_pady, hu_pady, hv_pady, zb_pady = _pad_y(
            h, hu, hv, zb,
            eta_inflow=eta_inflow, u_inflow=u_inflow, v_inflow=v_inflow,
        )
        F_h, F_L_hu, F_R_hu, F_hv = _audusse_x(h_padx, hu_padx, hv_padx, zb_padx, g)
        G_h, G_hu, G_B_hv, G_T_hv = _audusse_y(h_pady, hu_pady, hv_pady, zb_pady, g)

        # Cell j (col, x): right interface k=j+1 -> cell j is LEFT  -> F_L
        #                  left  interface k=j   -> cell j is RIGHT -> F_R
        # Cell j (row, y): top   interface k=j+1 -> cell j is BOTTOM -> G_B
        #                  bot   interface k=j   -> cell j is TOP    -> G_T
        h = h - dt / dx * (F_h[:, 1:]    - F_h[:, :-1])    \
              - dt / dy * (G_h[1:, :]    - G_h[:-1, :])
        hu = hu - dt / dx * (F_L_hu[:, 1:] - F_R_hu[:, :-1]) \
                - dt / dy * (G_hu[1:, :]   - G_hu[:-1, :])
        hv = hv - dt / dx * (F_hv[:, 1:]   - F_hv[:, :-1])   \
                - dt / dy * (G_B_hv[1:, :] - G_T_hv[:-1, :])

        # Positivity guard
        h = np.maximum(h, 0.0)

        t += dt
        step += 1

        # Save snapshot if we cross the next save time
        while next_save_idx < n_save and t >= save_times[next_save_idx]:
            t_saved.append(t)
            h_hist.append(h.copy())
            hu_hist.append(hu.copy())
            hv_hist.append(hv.copy())
            next_save_idx += 1

        if verbose and step % 100 == 0:
            print(f"  step {step:5d}  t = {t:.4f}/{t_end:.2f}  max|u| = {np.max(np.abs(u)):.3f}")

    # Pad out if needed
    while len(h_hist) < n_save:
        t_saved.append(t)
        h_hist.append(h.copy())
        hu_hist.append(hu.copy())
        hv_hist.append(hv.copy())

    return (np.array(t_saved),
            np.array(h_hist), np.array(hu_hist), np.array(hv_hist))


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(
    Lx=25.0, Ly=25.0, Nx=50, Ny=50,
    t_end=60.0, n_save=60,     # Paper config: T = 60 s, snapshots every 1 s.
    eta_init=2.0, u_init=2.21, v_init=2.21,
    h_min=1.0e-3,
    cyls=((8.0, 8.0, 4.0, 0.2), (15.0, 15.0, 2.0, 0.3)),
    smooth=0.0,
    g=9.81,
    save_path=None,
    verbose=False,
):
    """Generate a full 2D ground truth dataset.

    Replicates Ruppenthal & Kuzmin (2026) §7.2: uniform free surface at rest
    (``eta_init``) and uniform velocity, so depth ``h(x, y, 0)`` is locally
    shallower over each cylinder.
    """
    # Grid (cell centers)
    x = np.linspace(0.5 * Lx / Nx, Lx - 0.5 * Lx / Nx, Nx)
    y = np.linspace(0.5 * Ly / Ny, Ly - 0.5 * Ly / Ny, Ny)
    X, Y = np.meshgrid(x, y)
    dx = Lx / Nx
    dy = Ly / Ny

    # Bathymetry
    zb = bathymetry_two_cylinders(X, Y, cyls=cyls, smooth=smooth)

    # IC: uniform free surface (paper §7.2) -> depth h = eta - zb >= h_min.
    h0 = np.maximum(eta_init - zb, h_min)
    hu0 = u_init * h0
    hv0 = v_init * h0

    # Run solver with Dirichlet inflow at small-x/small-y, Neumann outflow.
    t, h_hist, hu_hist, hv_hist = run_fv_simulation(
        zb, h0, hu0, hv0, dx, dy, t_end, g=g, n_save=n_save, verbose=verbose,
        eta_inflow=eta_init, u_inflow=u_init, v_inflow=v_init,
    )

    # Derive u, v, eta
    eps = 1e-8
    u_hist = hu_hist / (h_hist + eps)
    v_hist = hv_hist / (h_hist + eps)
    eta_hist = h_hist + zb[None, :, :]

    params = {
        "Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny,
        "t_end": t_end, "n_save": n_save,
        "eta_init": eta_init, "u_init": u_init, "v_init": v_init,
        "h_min": h_min,
        "cylinders": cyls, "smooth": smooth, "g": g,
        "bc": {
            "type": "dirichlet_inflow",
            "sides": ["small_x", "small_y"],
            "eta": eta_init, "u": u_init, "v": v_init,
        },
    }
    data = {
        "x": x, "y": y, "t": t, "X": X, "Y": Y,
        "zb": zb,
        "h": h_hist, "u": u_hist, "v": v_hist, "eta": eta_hist,
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
        case_id="exp3_two_cylinders",
        spatial_dim=2,
        has_t=True,
        bc_type="open_uniform",
        constants={
            "g": float(params["g"]),
            "eta_init": float(params["eta_init"]),
            "u_init": float(params["u_init"]),
            "v_init": float(params["v_init"]),
        },
        domain={
            "x": [float(data["x"].min()), float(data["x"].max())],
            "y": [float(data["y"].min()), float(data["y"].max())],
            "t": [float(data["t"].min()), float(data["t"].max())],
        },
        gt_source="fv_hll",
        description=(
            "Two-cylinder benchmark from Ruppenthal & Kuzmin 2026 §7.2, "
            "FV-HLL solver on cartesian grid."
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
# Sanity
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Two cylinders 2D — FV ground truth")
    print("=" * 60)
    data = generate_dataset(
        Lx=25.0, Ly=25.0, Nx=50, Ny=50,
        t_end=2.0, n_save=20,    # short sanity-check run; production uses t_end=60
        eta_init=2.0, u_init=2.21, v_init=2.21,
        smooth=0.0, verbose=True,
    )
    p = data["params"]
    print()
    print(f"  grid: {p['Nx']} x {p['Ny']}, dx = {p['Lx']/p['Nx']} m")
    print(f"  cyls: {p['cylinders']}")
    print(f"  t: {data['t'][0]:.3f} -> {data['t'][-1]:.3f} s, {len(data['t'])} snapshots")
    print(f"  zb range: [{data['zb'].min():.4f}, {data['zb'].max():.4f}] m")
    print(f"  h range:  [{data['h'].min():.4f}, {data['h'].max():.4f}] m")
    print(f"  u range:  [{data['u'].min():.4f}, {data['u'].max():.4f}] m/s")
    print(f"  v range:  [{data['v'].min():.4f}, {data['v'].max():.4f}] m/s")
    print(f"  eta range: [{data['eta'].min():.4f}, {data['eta'].max():.4f}] m")
