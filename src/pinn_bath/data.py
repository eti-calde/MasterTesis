"""Case bundle: unified representation of a ground-truth dataset.

A :class:`Case` packs together (i) the ground-truth coordinates and field
values for one of the experiments, (ii) physical/geometric metadata
(boundary type, constants, domain extents), and (iii) reproducible samplers
for observations and PDE collocation points.

The on-disk schema is a single ``.npz`` file with:

- 1-D coordinate arrays: ``x`` (always), ``y`` (2-D cases), ``t`` (transient).
- Field arrays whose shape encodes the case kind:

  ====================  ================  ===========================
  Case kind             flow fields       ``zb`` shape
  ====================  ================  ===========================
  1-D steady            ``(Nx,)``         ``(Nx,)``
  1-D transient         ``(Nt, Nx)``      ``(Nx,)``
  2-D steady            ``(Ny, Nx)``      ``(Ny, Nx)``
  2-D transient         ``(Nt, Ny, Nx)``  ``(Ny, Nx)``
  ====================  ================  ===========================

- A single ``metadata_json`` key holding the rest (case_id, BC type,
  constants, domain, provenance).

Calling :meth:`Case.load` validates the schema; the file's SHA-256 is
cached on the resulting object for pre-flight checks (S14).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

BCType = Literal[
    "open_dirichlet",  # Exp 1: q upstream + h downstream
    "closed",  # Exp 2, Exp 5: closed basin (wall BCs)
    "tidal",  # Exp 4: tidal water-level BC on all sides
    "open_uniform",  # Exp 3: uniform inflow / free outflow
    "real_sensor",  # Exp 6: Dirichlet from sensor time-series
]

GTSource = Literal[
    "analytical_bernoulli",
    "analytical_thacker",
    "fv_hll",
    "sensor",
]

Coord = Literal["x", "y", "t"]
Field = Literal["h", "u", "v", "zb", "eta"]


@dataclass
class CaseMetadata:
    """Case-level metadata (everything except the GT tensors)."""

    case_id: str
    spatial_dim: int
    has_t: bool
    bc_type: str
    constants: dict[str, float]
    domain: dict[str, list[float]]  # {"x": [xmin, xmax], "y": ..., "t": ...}
    gt_source: str
    description: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> CaseMetadata:
        return cls(**json.loads(payload))


@dataclass
class Case:
    """Unified ground-truth bundle for one experiment configuration."""

    metadata: CaseMetadata
    coords: dict[str, np.ndarray]  # 1-D arrays of unique coord values
    fields: dict[str, np.ndarray]  # natural-shape ground-truth tensors
    source_path: Path | None = None
    file_hash: str | None = None
    eval_dtype: torch.dtype = field(default_factory=lambda: torch.float64)

    def __post_init__(self) -> None:
        # Run schema validation on every construction path (not just load /
        # save), so bad metadata (e.g., spatial_dim=2 with has_t=False) is
        # caught early at the call site.
        self._validate()

    # --- IO ---------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> Case:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        with np.load(p, allow_pickle=False) as data:
            if "metadata_json" not in data:
                raise ValueError(f"{p}: missing 'metadata_json'; not a pinn_bath unified .npz")
            metadata = CaseMetadata.from_json(str(data["metadata_json"]))
            coords: dict[str, np.ndarray] = {}
            for axis in ("x", "y", "t"):
                if axis in data.files:
                    coords[axis] = data[axis]
            fields: dict[str, np.ndarray] = {}
            for fname in ("h", "u", "v", "zb", "eta"):
                if fname in data.files:
                    fields[fname] = data[fname]
        instance = cls(
            metadata=metadata,
            coords=coords,
            fields=fields,
            source_path=p,
            file_hash=_sha256(p),
        )
        instance._validate()
        return instance

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._validate()
        arrays: dict[str, np.ndarray] = dict(self.coords)
        arrays.update(self.fields)
        # JSON encoded as a fixed-length unicode array (np.savez accepts that).
        arrays["metadata_json"] = np.array(self.metadata.to_json())
        # Atomic write: np.savez auto-appends ".npz", so we use a tmp name
        # that already ends in ".npz" to avoid double-extension surprises.
        import secrets

        tmp = p.parent / f".{p.stem}.{secrets.token_hex(4)}.npz"
        try:
            np.savez(tmp, **arrays)
            tmp.replace(p)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    # --- Samplers ---------------------------------------------------------

    def eval_grid(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Full ground-truth grid as flattened ``(N, 1)`` torch tensors.

        For transient cases, ``zb`` (which has no time axis on disk) is
        broadcast across time so all returned fields share the same length
        as the coordinate tensors.
        """
        mesh = self._mesh_coords()
        coords_t = {
            a: torch.as_tensor(v, dtype=self.eval_dtype).reshape(-1, 1) for a, v in mesh.items()
        }
        fields_t: dict[str, torch.Tensor] = {}
        for f, v in self.fields.items():
            if f == "zb" and self.metadata.has_t:
                v_full = self._broadcast_zb()
            else:
                v_full = v
            fields_t[f] = torch.as_tensor(v_full.reshape(-1, 1), dtype=self.eval_dtype)
        return coords_t, fields_t

    def sample_observations(
        self,
        *,
        seed: int,
        n_obs: int,
        fields: tuple[str, ...] = ("eta",),
        noise_std: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Pick ``n_obs`` GT points uniformly at random (reproducible by seed).

        Returns a dict with keys for each axis (``x``, ``y``, ``t``) and each
        requested observable field, plus ``"index"`` (the linear indices into
        the full grid) for diagnostics. All tensors are shape ``(n_obs, 1)``.
        """
        for f in fields:
            if f not in self.fields and not (
                f == "eta" and "h" in self.fields and "zb" in self.fields
            ):
                raise KeyError(f"field {f!r} not in case; available: {list(self.fields)}")
        mesh = self._mesh_coords()
        n_total = next(iter(mesh.values())).size
        if n_obs > n_total:
            raise ValueError(f"n_obs={n_obs} > grid size {n_total}")
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=n_obs, replace=False))

        out: dict[str, torch.Tensor] = {}
        for axis, arr in mesh.items():
            out[axis] = torch.as_tensor(arr.reshape(-1)[idx], dtype=self.eval_dtype).reshape(-1, 1)
        for f in fields:
            if f == "eta" and "eta" not in self.fields:
                vals = (self.fields["h"] + self._broadcast_zb()).reshape(-1)[idx]
            else:
                vals = self.fields[f].reshape(-1)[idx]
            t_vals = torch.as_tensor(vals, dtype=self.eval_dtype).reshape(-1, 1)
            if noise_std > 0.0:
                noise = torch.as_tensor(
                    rng.normal(0.0, noise_std, size=t_vals.shape[0]),
                    dtype=self.eval_dtype,
                ).reshape(-1, 1)
                t_vals = t_vals + noise
            out[f] = t_vals
        out["index"] = torch.as_tensor(idx, dtype=torch.long).reshape(-1, 1)
        return out

    def sample_collocation(
        self,
        *,
        seed: int,
        n_coll: int,
        requires_grad: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Uniform random points inside the continuous domain.

        Returns shapes ``(n_coll, 1)`` with ``requires_grad=True`` so the
        trainer can compute SWE residuals via autograd.
        """
        rng = np.random.default_rng(seed)
        out: dict[str, torch.Tensor] = {}
        for axis, lo_hi in self.metadata.domain.items():
            lo, hi = float(lo_hi[0]), float(lo_hi[1])
            samples = rng.uniform(lo, hi, size=n_coll).astype(np.float64)
            t = torch.as_tensor(samples, dtype=self.eval_dtype).reshape(-1, 1)
            if requires_grad:
                t.requires_grad_(True)
            out[axis] = t
        return out

    # --- Helpers ----------------------------------------------------------

    @property
    def n_grid_points(self) -> int:
        return int(np.prod([self.coords[a].size for a in self._axes()]))

    def _axes(self) -> list[str]:
        axes = ["x"]
        if self.metadata.spatial_dim == 2:
            axes.append("y")
        if self.metadata.has_t:
            axes.append("t")
        return axes

    def _mesh_coords(self) -> dict[str, np.ndarray]:
        """Broadcast 1-D coord arrays to the natural N-D mesh shape.

        Output arrays share the natural field shape so reshape(-1) lines them
        up with reshape(-1) of the fields.
        """
        axes = self._axes()
        arrays = [self.coords[a] for a in axes]
        # We use 'ij' indexing so the result shape matches (Nt, Ny, Nx) when
        # axes = ('t', 'y', 'x'). The natural field shape is time-leading,
        # then y, then x.
        if axes == ["x"]:
            return {"x": arrays[0]}
        if axes == ["x", "t"]:
            X, T = np.meshgrid(arrays[0], arrays[1], indexing="xy")  # (Nt, Nx)
            return {"x": X, "t": T}
        if axes == ["x", "y"]:
            X, Y = np.meshgrid(arrays[0], arrays[1], indexing="xy")  # (Ny, Nx)
            return {"x": X, "y": Y}
        if axes == ["x", "y", "t"]:
            x, y, t = arrays
            T, Y, X = np.meshgrid(t, y, x, indexing="ij")  # (Nt, Ny, Nx)
            return {"x": X, "y": Y, "t": T}
        raise NotImplementedError(f"unsupported axis set: {axes}")

    def _broadcast_zb(self) -> np.ndarray:
        """Broadcast ``zb`` to the natural flow-field shape (for eta = h + zb)."""
        zb = self.fields["zb"]
        h = self.fields["h"]
        if zb.shape == h.shape:
            return zb
        # ``zb`` lacks the time axis.
        if self.metadata.has_t and self.metadata.spatial_dim == 1:
            return np.broadcast_to(zb[None, :], h.shape)
        if self.metadata.has_t and self.metadata.spatial_dim == 2:
            return np.broadcast_to(zb[None, :, :], h.shape)
        raise ValueError(
            f"cannot broadcast zb {zb.shape} to h {h.shape} for case {self.metadata.case_id}"
        )

    def _validate(self) -> None:
        # 2D-steady has no SWE residual implemented in pinn_bath; catch
        # it at construction so the trainer never sees the impossible
        # combination.
        if self.metadata.spatial_dim == 2 and not self.metadata.has_t:
            raise ValueError(
                f"{self.metadata.case_id}: 2D steady cases are not implemented "
                f"(spatial_dim=2 requires has_t=True). Either add the t axis or "
                f"reduce to spatial_dim=1."
            )
        axes = self._axes()
        for a in axes:
            if a not in self.coords:
                raise ValueError(f"missing coord {a!r} for case {self.metadata.case_id}")
            if self.coords[a].ndim != 1:
                raise ValueError(f"coord {a!r} must be 1-D")
        required_fields: list[str] = ["h", "u", "zb"]
        if self.metadata.spatial_dim == 2:
            required_fields.append("v")
        for f in required_fields:
            if f not in self.fields:
                raise ValueError(f"missing field {f!r} for case {self.metadata.case_id}")
        for axis in self.metadata.domain:
            if axis not in axes:
                raise ValueError(
                    f"domain has axis {axis!r} that the case does not declare; axes={axes}"
                )

        # Field shape consistency: prevent silent zb-broadcast bugs at
        # the boundary between data.py and losses/ic.py. ``h``, ``u``
        # (and ``v`` in 2D) must share the natural full-grid shape
        # ``(Nt?, Ny?, Nx)`` derived from the coords. ``zb`` is time-
        # independent in the schema — its shape must match the *spatial*
        # part of that grid.
        spatial_shape: tuple[int, ...] = (
            (self.coords["x"].size,)
            if self.metadata.spatial_dim == 1
            else (self.coords["y"].size, self.coords["x"].size)
        )
        full_shape: tuple[int, ...] = (
            (self.coords["t"].size, *spatial_shape) if self.metadata.has_t else spatial_shape
        )
        for f in required_fields:
            arr = self.fields[f]
            expected = spatial_shape if f == "zb" else full_shape
            if arr.shape != expected:
                raise ValueError(
                    f"{self.metadata.case_id}: field {f!r} has shape {arr.shape}, "
                    f"expected {expected}. Check axis ordering "
                    f"(time first, then y, then x)."
                )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
