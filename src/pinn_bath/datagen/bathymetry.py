r"""Parametric bathymetry sampling: background slope + localized features.

The bed is decomposed as

.. math::
    z_b(x) = \underbrace{s\,(x - x_{mid})}_{\text{trend}}
             + \underbrace{\textstyle\sum_k f_k(x)}_{\text{features}}

where the *trend* is a per-case linear background slope (oceanographer
feedback: realistic shelves are not flat) and the *features* are the signed
gaussian / parabolic bumps and holes of the legacy generator. The trend is
centred at the domain midpoint so the mean bed level (and hence the mean
depth at a given tidal stage) is slope-independent.

Orthogonality of the slope axis
-------------------------------
The slope is sampled from the *same* distribution in every difficulty tier,
and the difficulty score is computed on the **detrended** profile (features
only). The OOD-by-difficulty split therefore stays purely about feature
complexity; the network simply has to cope with whatever background gradient
a case happens to have, in train and test alike. For ``slope = 0`` the score
is identical to the legacy ``datasets.generator`` score.

Deep-water cap
--------------
``BathymetrySampler.sample`` rescales the *feature* amplitudes (never the
trend, which is an orthogonal axis) so that the full bed
``trend + features`` stays below ``max_bed_elevation`` everywhere. The caller
(the environment) derives that cap from its lowest tidal stage minus the
minimum rest column, exactly as before; the cap is simply enforced pointwise
now that the trend makes headroom position-dependent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

Difficulty = Literal["easy", "medium", "hard"]
Kind = Literal["gaussian", "parabolic"]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Feature:
    kind: Kind
    amplitude: float  # signed; >0 bump (toward the surface), <0 hole (deeper)
    center: float
    width: float  # gaussian sigma, or parabolic half-width


def feature_profile(f: Feature, x: np.ndarray) -> np.ndarray:
    if f.kind == "gaussian":
        return f.amplitude * np.exp(-(((x - f.center) / f.width) ** 2))
    if f.kind == "parabolic":
        z = f.amplitude * (1.0 - ((x - f.center) / f.width) ** 2)
        # Clip to the feature's support (parabola only where it has the
        # feature's sign), like the SWASHES/Dazzi bump.
        return np.where(np.sign(z) == np.sign(f.amplitude), z, 0.0)
    raise ValueError(f"unknown feature kind: {f.kind!r}")


# --------------------------------------------------------------------------- #
# Bathymetry field: trend + features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BathymetryField:
    """One case's bed: linear background trend plus parametric features.

    Fully parametric and frozen, so a ``CaseSpec`` carrying it is exactly
    regenerable from its parameters (and evaluable on any grid, 1D today).
    """

    features: tuple[Feature, ...]
    slope: float = 0.0  # background gradient d(zb)/dx [m/m]
    x_mid: float = 0.0  # trend pivot; mean bed level is slope-independent

    def trend(self, x: np.ndarray) -> np.ndarray:
        return self.slope * (np.asarray(x, dtype=float) - self.x_mid)

    def detrended(self, x: np.ndarray) -> np.ndarray:
        """Features-only profile (what the difficulty score sees)."""
        x = np.asarray(x, dtype=float)
        zb = np.zeros_like(x)
        for f in self.features:
            zb = zb + feature_profile(f, x)
        return zb

    def profile(self, x: np.ndarray) -> np.ndarray:
        """Full bed elevation ``zb(x) = trend + features``."""
        return self.trend(x) + self.detrended(x)


# --------------------------------------------------------------------------- #
# Difficulty tiers (sampling ranges; amplitudes are fractions of sea_level)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tier:
    k_choices: tuple[int, ...]
    amp_frac: tuple[float, float]  # |A| / sea_level
    width: tuple[float, float]  # metres
    allow_holes: bool
    allow_drying: bool  # if False, the bed is capped below the lowest tide


# Feature widths are scaled with the incident *wavelength* (x2 vs the legacy
# 10 m bank), not with the domain (x4), preserving the width/lambda ratios
# (~0.03-0.26) that define the published difficulty axis: features stay
# sub-wavelength scatterers, so inversion must exploit celerity/phase rather
# than direct reflection. Amplitude fractions are unchanged (fjord sills and
# basins genuinely occupy large depth fractions).
TIERS: dict[Difficulty, Tier] = {
    "easy": Tier(
        k_choices=(1,),
        amp_frac=(0.15, 0.35),
        width=(2.4, 4.0),
        allow_holes=True,
        allow_drying=False,
    ),
    "medium": Tier(
        k_choices=(2, 3),
        amp_frac=(0.25, 0.55),
        width=(1.4, 2.8),
        allow_holes=True,
        allow_drying=False,
    ),
    "hard": Tier(
        k_choices=(4, 5, 6),
        amp_frac=(0.40, 0.85),
        width=(0.8, 1.8),
        allow_holes=True,
        allow_drying=False,  # no emergence: target application is deep water
    ),
}

# Background slope axis: signed gradient range [m/m], sampled uniformly and
# *identically in every tier* (orthogonal to difficulty, like the tidal axes).
# +/-1.25% over the 40 m domain keeps the trend relief at +/-0.25 H0 at the
# ends (a clearly felt celerity gradient) while staying physically realistic:
# along-thalweg fjord/estuary gradients are O(0.1-1%), up to a few % at
# fjord-head deltas and sills. Slope is dimensionless, hence Froude-invariant.
SLOPE_RANGE: tuple[float, float] = (-0.0125, 0.0125)


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BathymetrySampler:
    """Draws a :class:`BathymetryField` for a difficulty tier.

    Ports the legacy ``datasets.generator.sample_case`` bathymetry block and
    adds the background-slope axis. All sampling is driven by the caller's
    ``rng``, so datasets remain exactly regenerable from seeds.
    """

    tiers: Mapping[Difficulty, Tier] = None  # type: ignore[assignment]
    slope_range: tuple[float, float] = SLOPE_RANGE
    margin: float = 4.0  # keep centres >= one max feature width from the edges
    gaussian_prob: float = 0.7

    def __post_init__(self) -> None:
        if self.tiers is None:
            object.__setattr__(self, "tiers", TIERS)

    def sample(
        self,
        difficulty: Difficulty,
        rng: np.random.Generator,
        x: np.ndarray,
        *,
        sea_level: float,
        max_bed_elevation: float,
    ) -> BathymetryField:
        """Draw one bed. ``max_bed_elevation`` is the deep-water cap: the
        highest bed elevation allowed anywhere (the environment ties it to the
        lowest tidal stage minus the minimum rest column).
        """
        tier = self.tiers[difficulty]
        x = np.asarray(x, dtype=float)
        xlo, xhi = float(x[0]), float(x[-1])

        slope = float(rng.uniform(*self.slope_range))
        x_mid = 0.5 * (xlo + xhi)

        k = int(rng.choice(tier.k_choices))
        feats: list[Feature] = []
        for _ in range(k):
            kind: Kind = "gaussian" if rng.random() < self.gaussian_prob else "parabolic"
            mag = rng.uniform(*tier.amp_frac) * sea_level
            sign = 1.0 if (not tier.allow_holes or rng.random() < 0.5) else -1.0
            feats.append(
                Feature(
                    kind=kind,
                    amplitude=sign * mag,
                    center=rng.uniform(xlo + self.margin, xhi - self.margin),
                    width=rng.uniform(*tier.width),
                )
            )
        field = BathymetryField(features=tuple(feats), slope=slope, x_mid=x_mid)

        if not tier.allow_drying:
            field = self._cap_features(field, x, max_bed_elevation)
        return field

    def _cap_features(self, field: BathymetryField, x: np.ndarray, cap: float) -> BathymetryField:
        """Rescale feature amplitudes so ``trend + features <= cap`` pointwise.

        The trend is never rescaled (it is an orthogonal axis); the available
        headroom ``cap - trend(x)`` shrinks toward the shallow end, so the
        binding constraint is wherever a positive feature meets least headroom:
        ``scale = min over {feat > 0} of headroom / feat`` (exact, no search).
        """
        headroom = cap - field.trend(x)
        if headroom.min() <= 0.0:
            raise ValueError(
                f"slope {field.slope:+.4f} leaves no feature headroom under the "
                f"deep-water cap {cap:.3f}; shrink slope_range or raise the cap"
            )
        feat = field.detrended(x)
        pos = feat > 0
        if not np.any(feat[pos] > headroom[pos]):
            return field
        scale = float(np.min(headroom[pos] / feat[pos]))
        return replace(
            field,
            features=tuple(replace(f, amplitude=f.amplitude * scale) for f in field.features),
        )


# --------------------------------------------------------------------------- #
# Difficulty scoring (on the *detrended* profile)
# --------------------------------------------------------------------------- #
def difficulty_components(zb_detrended: np.ndarray, sea_level: float) -> dict[str, float]:
    """Interpretable scalars characterising how hard a case is to invert.

    Must be fed the **detrended** profile (``BathymetryField.detrended``) so
    the background slope does not leak into the difficulty label.
    """
    H0 = float(sea_level)
    zb = np.asarray(zb_detrended, dtype=float)
    amp_ratio = float(np.max(np.abs(zb)) / H0)  # proximity to emergence
    emergent_frac = float(np.mean(zb > H0))  # fraction dry at rest
    # Spectral "wiggliness": energy-weighted mean wavenumber, normalised.
    zc = zb - zb.mean()
    spec = np.abs(np.fft.rfft(zc)) ** 2
    k = np.arange(spec.size)
    bandwidth = float((spec @ k) / (spec.sum() + 1e-12) / max(spec.size - 1, 1))
    # Count distinct sign-coherent features (zero-crossings of zb proxy).
    sign_changes = int(np.count_nonzero(np.diff(np.sign(zc[np.abs(zc) > 0.02 * H0]))))
    return {
        "amp_ratio": amp_ratio,
        "emergent_frac": emergent_frac,
        "bandwidth": bandwidth,
        "sign_changes": float(sign_changes),
    }


def difficulty_score(components: dict[str, float]) -> float:
    """Scalar in ~[0, 1] combining the components (higher = harder).

    Identical to the legacy formula (paper Eq. for the score D); with
    ``slope = 0`` the new pipeline reproduces legacy scores exactly.
    """
    return float(
        0.44 * min(components["amp_ratio"], 1.2)
        + 0.31 * min(components["bandwidth"] * 4.0, 1.0)
        + 0.25 * min(components["sign_changes"] / 6.0, 1.0)
    )


# =========================================================================== #
# 2D extension (additive; the 1D classes above are untouched and in
# production). Same tier tables, slope range and deep-water cap logic; the
# features become rotated anisotropic Gaussians: elongation ~ 1 reads as a
# seamount / depression, elongation >> 1 as a ridge / trench.
# =========================================================================== #

Kind2D = Literal["seamount", "ridge"]

# Ridge major/minor axis ratio range (seamounts are isotropic, ratio 1).
RIDGE_ELONGATION: tuple[float, float] = (2.5, 5.0)


@dataclass(frozen=True)
class Feature2D:
    kind: Kind2D
    amplitude: float  # signed; >0 toward the surface, <0 deeper
    cx: float
    cy: float
    width: float  # minor-axis Gaussian sigma [m] (tier "width" ranges)
    elongation: float = 1.0  # major/minor ratio (1 = seamount, >1 = ridge)
    theta: float = 0.0  # major-axis orientation [rad]


def feature2d_profile(f: Feature2D, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rotated anisotropic Gaussian on the cell-centre meshgrid."""
    ct, st = np.cos(f.theta), np.sin(f.theta)
    xr = (X - f.cx) * ct + (Y - f.cy) * st  # along the major axis
    yr = -(X - f.cx) * st + (Y - f.cy) * ct  # along the minor axis
    w_major = f.width * f.elongation
    return f.amplitude * np.exp(-((xr / w_major) ** 2) - ((yr / f.width) ** 2))


@dataclass(frozen=True)
class BathymetryField2D:
    """2D bed: linear trend dipping in x plus rotated Gaussian features."""

    features: tuple[Feature2D, ...]
    slope: float = 0.0  # background gradient d(zb)/dx [m/m] (x only)
    x_mid: float = 0.0  # trend pivot

    def trend(self, X: np.ndarray) -> np.ndarray:
        return self.slope * (np.asarray(X, dtype=float) - self.x_mid)

    def detrended(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        zb = np.zeros_like(np.asarray(X, dtype=float))
        for f in self.features:
            zb = zb + feature2d_profile(f, X, Y)
        return zb

    def profile(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.trend(X) + self.detrended(X, Y)


@dataclass(frozen=True)
class BathymetrySampler2D:
    """2D analogue of :class:`BathymetrySampler` (same tiers / slope / cap).

    Each feature is a seamount (isotropic) with probability ``seamount_prob``,
    otherwise a ridge (elongated, random orientation). The deep-water cap is
    enforced pointwise against the x-trend exactly as in 1D.
    """

    tiers: Mapping[Difficulty, Tier] = None  # type: ignore[assignment]
    slope_range: tuple[float, float] = SLOPE_RANGE
    margin: float = 4.0  # feature centres stay this far from every edge
    seamount_prob: float = 0.5
    ridge_elongation: tuple[float, float] = RIDGE_ELONGATION

    def __post_init__(self) -> None:
        if self.tiers is None:
            object.__setattr__(self, "tiers", TIERS)

    def sample(
        self,
        difficulty: Difficulty,
        rng: np.random.Generator,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        sea_level: float,
        max_bed_elevation: float,
    ) -> BathymetryField2D:
        tier = self.tiers[difficulty]
        xlo, xhi = float(X.min()), float(X.max())
        ylo, yhi = float(Y.min()), float(Y.max())

        slope = float(rng.uniform(*self.slope_range))
        x_mid = 0.5 * (xlo + xhi)

        k = int(rng.choice(tier.k_choices))
        feats: list[Feature2D] = []
        for _ in range(k):
            kind: Kind2D = "seamount" if rng.random() < self.seamount_prob else "ridge"
            mag = rng.uniform(*tier.amp_frac) * sea_level
            sign = 1.0 if (not tier.allow_holes or rng.random() < 0.5) else -1.0
            elong = 1.0 if kind == "seamount" else float(rng.uniform(*self.ridge_elongation))
            feats.append(
                Feature2D(
                    kind=kind,
                    amplitude=sign * mag,
                    cx=rng.uniform(xlo + self.margin, xhi - self.margin),
                    cy=rng.uniform(ylo + self.margin, yhi - self.margin),
                    width=rng.uniform(*tier.width),
                    elongation=elong,
                    theta=float(rng.uniform(0.0, np.pi)),
                )
            )
        field = BathymetryField2D(features=tuple(feats), slope=slope, x_mid=x_mid)

        if not tier.allow_drying:
            field = self._cap_features(field, X, Y, max_bed_elevation)
        return field

    def _cap_features(
        self, field: BathymetryField2D, X: np.ndarray, Y: np.ndarray, cap: float
    ) -> BathymetryField2D:
        """Rescale feature amplitudes so ``trend + features <= cap`` pointwise."""
        headroom = cap - field.trend(X)
        if headroom.min() <= 0.0:
            raise ValueError(
                f"slope {field.slope:+.4f} leaves no feature headroom under the "
                f"deep-water cap {cap:.3f}; shrink slope_range or raise the cap"
            )
        feat = field.detrended(X, Y)
        pos = feat > 0
        if not np.any(feat[pos] > headroom[pos]):
            return field
        scale = float(np.min(headroom[pos] / feat[pos]))
        return replace(
            field,
            features=tuple(replace(f, amplitude=f.amplitude * scale) for f in field.features),
        )


def difficulty_components_2d(zb_detrended: np.ndarray, sea_level: float) -> dict[str, float]:
    """2D analogue of :func:`difficulty_components` (PROVISIONAL).

    Same three descriptors on the detrended ``(Ny, Nx)`` field: relative
    amplitude, spectral bandwidth (energy-weighted mean *radial* wavenumber,
    normalised by the maximum resolvable radius) and sign changes counted on
    the central x- and y-transects. Weights/caps reuse the 1D score so easy /
    medium / hard land in comparable ranges; to be re-calibrated when the
    production 2D bank is designed (Phase 2).
    """
    H0 = float(sea_level)
    zb = np.asarray(zb_detrended, dtype=float)
    ny, nx = zb.shape
    amp_ratio = float(np.max(np.abs(zb)) / H0)
    emergent_frac = float(np.mean(zb > H0))
    zc = zb - zb.mean()
    spec = np.abs(np.fft.rfft2(zc)) ** 2
    ky = np.fft.fftfreq(ny, d=1.0 / ny)  # index units, matches 1D convention
    kx = np.arange(spec.shape[1])
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    kr_max = float(np.sqrt((ny // 2) ** 2 + (spec.shape[1] - 1) ** 2))
    bandwidth = float((spec * kr).sum() / (spec.sum() + 1e-12) / max(kr_max, 1.0))
    sign_changes = 0
    for transect in (zc[ny // 2, :], zc[:, nx // 2]):
        live = transect[np.abs(transect) > 0.02 * H0]
        sign_changes += int(np.count_nonzero(np.diff(np.sign(live))))
    return {
        "amp_ratio": amp_ratio,
        "emergent_frac": emergent_frac,
        "bandwidth": bandwidth,
        "sign_changes": float(sign_changes),
    }
