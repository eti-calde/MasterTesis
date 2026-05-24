"""
Ground truth generator for Experiment 4: Tian dT10 (variable-topography tidal problem).

Faithful replication of the dynamic case dT10 from Tian et al. (2025) §3.3
(originally derived from the "tide" benchmark of Supei et al. 2022). Despite
its name, dT10 has no tidal external forcing -- the dynamics come from an
unbalanced initial condition that relaxes under gravity with periodic edges.

Setup (Tian eqs. 54-56):
    Domain:     (x, y) in [-2, 2]^2 m
    Topography: z(x, y) = 1 + 0.01 * cos(pi*x/2) * cos(pi*y/2)
    IC:         h(x, y, 0) = z(x, y),  u(x, y, 0) = v(x, y, 0) = 0
    BC:         periodic in x and y
    Time:       T = 0.5 s

Reference solver: Tian uses entropy-stable ES1 (Fjordholm 2011) on a very fine
grid (dx = 0.01 m, CFL = 0.25). Here we use the FV-HLL scheme from Exp 3
(LeVeque 2002) on a coarser cartesian grid; both are convergent SWE schemes.

Conservative form:
    U_t + F(U)_x + G(U)_y = S(U, z_b)   with U = (h, hu, hv).
"""

import numpy as np
from pathlib import Path


# ============================================================
# Bathymetry (Tian eq. 54)
# ============================================================


def bathymetry_dT10(X, Y):
    """Tian dT10 variable topography: z = 1 + 0.01 cos(pi x / 2) cos(pi y / 2).

    Maxima 1.01 m at the center (0, 0) and the four corners (+-2, +-2);
    minima 0.99 m at the mid-edges (+-2, 0) and (0, +-2).
    """
    return 1.0 + 0.01 * np.cos(np.pi * X / 2.0) * np.cos(np.pi * Y / 2.0)


# ============================================================
# HLL finite-volume fluxes (same kernels as Exp 3)
# ============================================================


def physical_flux_x(h, hu, hv, g):
    eps = 1.0e-8
    u = hu / (h + eps)
    return hu, hu * u + 0.5 * g * h * h, hv * u


def physical_flux_y(h, hu, hv, g):
    eps = 1.0e-8
    v = hv / (h + eps)
    return hv, hu * v, hv * v + 0.5 * g * h * h


def hll_flux_x(hL, huL, hvL, hR, huR, hvR, g):
    eps = 1.0e-8
    uL = huL / (hL + eps)
    uR = huR / (hR + eps)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)
    FL = physical_flux_x(hL, huL, hvL, g)
    FR = physical_flux_x(hR, huR, hvR, g)
    denom = (sR - sL) + 1.0e-12
    out = []
    for FL_i, FR_i, UL_i, UR_i in zip(FL, FR, (hL, huL, hvL), (hR, huR, hvR)):
        flux = np.where(
            sL >= 0,
            FL_i,
            np.where(sR <= 0, FR_i, (sR * FL_i - sL * FR_i + sL * sR * (UR_i - UL_i)) / denom),
        )
        out.append(flux)
    return tuple(out)


def hll_flux_y(hL, huL, hvL, hR, huR, hvR, g):
    eps = 1.0e-8
    vL = hvL / (hL + eps)
    vR = hvR / (hR + eps)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    sL = np.minimum(vL - cL, vR - cR)
    sR = np.maximum(vL + cL, vR + cR)
    GL = physical_flux_y(hL, huL, hvL, g)
    GR = physical_flux_y(hR, huR, hvR, g)
    denom = (sR - sL) + 1.0e-12
    out = []
    for GL_i, GR_i, UL_i, UR_i in zip(GL, GR, (hL, huL, hvL), (hR, huR, hvR)):
        flux = np.where(
            sL >= 0,
            GL_i,
            np.where(sR <= 0, GR_i, (sR * GL_i - sL * GR_i + sL * sR * (UR_i - UL_i)) / denom),
        )
        out.append(flux)
    return tuple(out)


# ============================================================
# FV simulation with periodic BCs (Tian dT10)
# ============================================================


def run_fv_periodic(
    zb, h0, hu0, hv0, dx, dy, t_end,
    cfl=0.3, g=9.81, n_save=40, h_min=1.0e-3, verbose=False,
):
    """2D SWE with periodic boundary conditions on all four edges.

    Periodic ghost cells: left ghost = rightmost interior, right ghost =
    leftmost interior (and similarly for y). Source term from topography
    via centered differences (NOT well-balanced; sufficient here because the
    bathymetry is smooth and the perturbations are O(1%) of the depth).
    """
    Ny, Nx = zb.shape
    h = h0.copy()
    hu = hu0.copy()
    hv = hv0.copy()

    t = 0.0
    save_times = np.linspace(0.0, t_end, n_save)
    t_saved = [0.0]
    h_hist = [h.copy()]
    hu_hist = [hu.copy()]
    hv_hist = [hv.copy()]
    next_save_idx = 1

    step = 0
    while t < t_end:
        eps = 1.0e-8
        u = hu / (h + eps)
        v = hv / (h + eps)
        c = np.sqrt(g * np.maximum(h, 0.0))
        max_speed = max(np.max(np.abs(u) + c), np.max(np.abs(v) + c), 1.0e-6)
        dt = cfl * min(dx, dy) / max_speed
        if t + dt > t_end:
            dt = t_end - t

        # x-direction with periodic ghost cells
        h_padx = np.empty((Ny, Nx + 2))
        hu_padx = np.empty((Ny, Nx + 2))
        hv_padx = np.empty((Ny, Nx + 2))
        h_padx[:, 1:-1] = h
        hu_padx[:, 1:-1] = hu
        hv_padx[:, 1:-1] = hv
        # left ghost = rightmost interior, right ghost = leftmost interior
        h_padx[:, 0] = h[:, -1]
        hu_padx[:, 0] = hu[:, -1]
        hv_padx[:, 0] = hv[:, -1]
        h_padx[:, -1] = h[:, 0]
        hu_padx[:, -1] = hu[:, 0]
        hv_padx[:, -1] = hv[:, 0]

        hL = h_padx[:, :-1]; huL = hu_padx[:, :-1]; hvL = hv_padx[:, :-1]
        hR = h_padx[:, 1:];  huR = hu_padx[:, 1:];  hvR = hv_padx[:, 1:]
        Fh, Fhu, Fhv = hll_flux_x(hL, huL, hvL, hR, huR, hvR, g)

        # y-direction with periodic ghost cells
        h_pady = np.empty((Ny + 2, Nx))
        hu_pady = np.empty((Ny + 2, Nx))
        hv_pady = np.empty((Ny + 2, Nx))
        h_pady[1:-1, :] = h
        hu_pady[1:-1, :] = hu
        hv_pady[1:-1, :] = hv
        h_pady[0, :] = h[-1, :]
        hu_pady[0, :] = hu[-1, :]
        hv_pady[0, :] = hv[-1, :]
        h_pady[-1, :] = h[0, :]
        hu_pady[-1, :] = hu[0, :]
        hv_pady[-1, :] = hv[0, :]

        hB = h_pady[:-1, :]; huB = hu_pady[:-1, :]; hvB = hv_pady[:-1, :]
        hT = h_pady[1:, :];  huT = hu_pady[1:, :];  hvT = hv_pady[1:, :]
        Gh, Ghu, Ghv = hll_flux_y(hB, huB, hvB, hT, huT, hvT, g)

        # Topography source via centered differences (periodic wrap)
        zb_padx2 = np.concatenate([zb[:, -1:], zb, zb[:, :1]], axis=1)
        zb_pady2 = np.concatenate([zb[-1:, :], zb, zb[:1, :]], axis=0)
        dzb_dx = (zb_padx2[:, 2:] - zb_padx2[:, :-2]) / (2.0 * dx)
        dzb_dy = (zb_pady2[2:, :] - zb_pady2[:-2, :]) / (2.0 * dy)

        S_hu = -g * h * dzb_dx
        S_hv = -g * h * dzb_dy

        # Update
        h = h - dt / dx * (Fh[:, 1:] - Fh[:, :-1]) - dt / dy * (Gh[1:, :] - Gh[:-1, :])
        hu = hu - dt / dx * (Fhu[:, 1:] - Fhu[:, :-1]) - dt / dy * (Ghu[1:, :] - Ghu[:-1, :]) + dt * S_hu
        hv = hv - dt / dx * (Fhv[:, 1:] - Fhv[:, :-1]) - dt / dy * (Ghv[1:, :] - Ghv[:-1, :]) + dt * S_hv

        h = np.maximum(h, h_min)
        t += dt
        step += 1

        while next_save_idx < n_save and t >= save_times[next_save_idx]:
            t_saved.append(t)
            h_hist.append(h.copy())
            hu_hist.append(hu.copy())
            hv_hist.append(hv.copy())
            next_save_idx += 1

        if verbose and step % 50 == 0:
            print(
                f"  step {step:5d}  t = {t:.4f}/{t_end:.3f}  "
                f"max|u| = {np.max(np.abs(u)):.4f}  max|v| = {np.max(np.abs(v)):.4f}"
            )

    while len(h_hist) < n_save:
        t_saved.append(t)
        h_hist.append(h.copy())
        hu_hist.append(hu.copy())
        hv_hist.append(hv.copy())

    return (
        np.array(t_saved),
        np.array(h_hist),
        np.array(hu_hist),
        np.array(hv_hist),
    )


# ============================================================
# Dataset generation
# ============================================================


def generate_dataset(
    Lx=4.0, Ly=4.0, Nx=100, Ny=100,    # domain [-2, 2]^2, Tian-comparable resolution
    t_end=0.5, n_save=51,              # Tian dT10: T = 0.5 s
    h_min=1.0e-3,
    g=9.81,
    save_path=None,
    verbose=False,
):
    """Generate 2D ground-truth dataset for Tian dT10 (eqs. 54-56).

    The domain is centered at the origin: x in [-Lx/2, Lx/2], y in [-Ly/2, Ly/2].
    For the canonical Tian dT10, ``Lx = Ly = 4`` (so [-2, 2]^2).
    """
    x = np.linspace(-Lx / 2 + 0.5 * Lx / Nx, Lx / 2 - 0.5 * Lx / Nx, Nx)
    y = np.linspace(-Ly / 2 + 0.5 * Ly / Ny, Ly / 2 - 0.5 * Ly / Ny, Ny)
    X, Y = np.meshgrid(x, y)
    dx = Lx / Nx
    dy = Ly / Ny

    zb = bathymetry_dT10(X, Y)

    # IC: h(x, y, 0) = z(x, y),  u = v = 0  (Tian eqs. 55-56)
    h0 = zb.copy()
    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)

    t, h_hist, hu_hist, hv_hist = run_fv_periodic(
        zb, h0, hu0, hv0, dx, dy, t_end,
        g=g, n_save=n_save, h_min=h_min, verbose=verbose,
    )

    eps = 1.0e-8
    u_hist = hu_hist / (h_hist + eps)
    v_hist = hv_hist / (h_hist + eps)
    eta_hist = h_hist + zb[None, :, :]

    params = {
        "Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny,
        "t_end": t_end, "n_save": n_save, "h_min": h_min,
        "g": g, "case": "Tian dT10",
    }
    data = {
        "x": x, "y": y, "t": t, "X": X, "Y": Y,
        "zb": zb, "h": h_hist, "u": u_hist, "v": v_hist, "eta": eta_hist,
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
        case_id="exp4_tian_dT10",
        spatial_dim=2,
        has_t=True,
        bc_type="periodic",
        constants={"g": float(params["g"])},
        domain={
            "x": [float(data["x"].min()), float(data["x"].max())],
            "y": [float(data["y"].min()), float(data["y"].max())],
            "t": [float(data["t"].min()), float(data["t"].max())],
        },
        gt_source="fv_hll",
        description=(
            "Tian (2025) dT10 variable-topography tidal problem (§3.3, eqs. 54-56). "
            "FV-HLL with periodic BCs."
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


if __name__ == "__main__":
    print("=" * 60)
    print("Tian dT10 (variable-topography tidal problem) — FV ground truth")
    print("=" * 60)
    data = generate_dataset(
        Lx=4.0, Ly=4.0, Nx=100, Ny=100,
        t_end=0.5, n_save=51,
        verbose=True,
    )
    p = data["params"]
    print()
    print(f"  domain:    [{-p['Lx']/2}, {p['Lx']/2}] x [{-p['Ly']/2}, {p['Ly']/2}] m")
    print(f"  grid:      {p['Nx']} x {p['Ny']} (dx = {p['Lx']/p['Nx']:.4f} m)")
    print(f"  t:         {data['t'][0]:.4f} -> {data['t'][-1]:.4f} s, {len(data['t'])} snapshots")
    print(f"  zb range:  [{data['zb'].min():.4f}, {data['zb'].max():.4f}] m  (expect 0.99-1.01)")
    print(f"  h(0)=zb:   [{data['h'][0].min():.4f}, {data['h'][0].max():.4f}] m  (expect 0.99-1.01)")
    print(f"  h final:   [{data['h'][-1].min():.4f}, {data['h'][-1].max():.4f}] m")
    print(f"  u range:   [{data['u'].min():.4f}, {data['u'].max():.4f}] m/s")
    print(f"  eta range: [{data['eta'].min():.4f}, {data['eta'].max():.4f}] m")
