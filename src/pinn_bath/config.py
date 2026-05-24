"""RunConfig: YAML-backed config with schema validation (S2).

A single :class:`RunConfig` describes one training run end-to-end. Two runs
with the same config and seed share the same :attr:`RunConfig.run_id` (S14),
so the registry can skip or resume them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Arch = Literal["A1", "A2", "A3"]
Budget = Literal["small", "medium", "large"]
Form = Literal["primitive", "prim_cons", "conservative"]
Observation = Literal["eta", "u", "v"]


class LossWeights(BaseModel):
    """Weights for the composite PINN loss."""

    model_config = ConfigDict(extra="forbid")

    data: float = 10.0
    data_u: float = 0.0
    pde: float = 1.0
    ic: float = 0.0
    bc: float = 0.0
    pos: float = 1.0
    tv: float = 1.0e-4
    tikh: float = 0.0
    dry: float = 0.0


class OptimizerCfg(BaseModel):
    """Adam + L-BFGS schedule (protocolo §3.10)."""

    model_config = ConfigDict(extra="forbid")

    adam_epochs: int = 12_000
    adam_lr: float = 1.0e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    lbfgs_steps: int = 600
    lbfgs_lr: float = 1.0
    lbfgs_history: int = 50


class CheckpointCfg(BaseModel):
    """How often to checkpoint and how many snapshots to keep (S9)."""

    model_config = ConfigDict(extra="forbid")

    every_epochs: int = 500
    keep_last_k: int = 3
    keep_best: bool = True


class DataCfg(BaseModel):
    """Where the case dataset lives and which fields are observed."""

    model_config = ConfigDict(extra="forbid")

    case_path: str
    observations: list[Observation] = Field(default_factory=lambda: ["eta"])
    n_obs_points: int | None = None
    obs_noise_std: float = 0.0


class RunConfig(BaseModel):
    """Top-level configuration for a single PINN training run."""

    model_config = ConfigDict(extra="forbid")

    case: str
    arch: Arch
    budget: Budget
    form: Form = "primitive"
    seed: int = 0
    deterministic: bool = True
    loss: LossWeights = Field(default_factory=LossWeights)
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    checkpoint: CheckpointCfg = Field(default_factory=CheckpointCfg)
    data: DataCfg

    @property
    def run_id(self) -> str:
        """Deterministic 12-char hash of the canonical config (S14)."""
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: Path | str) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)

    def to_yaml(self, path: Path | str) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True))
