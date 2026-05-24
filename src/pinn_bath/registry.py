"""Run registry: deterministic run_id, idempotent re-launch, manifest log (S14).

A study (e.g., the §5.1 architecture scaling barrido) generates many runs:
``3 archs x 3 budgets x 3 cases x 3 seeds = 81 corridas``. The registry makes
that grid safe to interrupt and re-launch:

- :func:`run_id_for` derives a deterministic 12-char hash from a
  :class:`~pinn_bath.config.RunConfig` so two configs map to the same id.
- :class:`Manifest` is an append-only JSONL log of every attempt
  (``started``, ``ok``, ``diverged``, ``interrupted``).
- :class:`Registry` answers ``should_run(cfg)`` / ``mark_*`` and walks an
  iterator of configs deciding what to launch, skip, or resume.

The registry is content-agnostic; the actual training is performed elsewhere
(``studies/arch_scaling.py``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pinn_bath.config import RunConfig


def run_id_for(cfg: RunConfig) -> str:
    """Deterministic 12-char id for a run config (hash includes the seed)."""
    return cfg.run_id


def run_dir_for(study_dir: Path | str, cfg: RunConfig) -> Path:
    """Path to a run's output directory: ``study_dir/<run_id>/``."""
    return Path(study_dir) / run_id_for(cfg)


@dataclass
class ManifestEntry:
    """One row in :class:`Manifest`."""

    run_id: str
    status: str  # "started" | "ok" | "diverged" | "interrupted"
    ts: float
    case: str
    arch: str
    budget: str
    seed: int
    form: str
    wall_time_s: float | None = None
    final_loss: float | None = None
    error: str | None = None
    machine: str | None = None


class Manifest:
    """Append-only JSONL log of run attempts under a study directory."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: ManifestEntry) -> None:
        row = {k: v for k, v in entry.__dict__.items() if v is not None}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def latest_status(self, run_id: str) -> str | None:
        """Most recent status of ``run_id``, or ``None`` if never seen."""
        latest: tuple[float, str] | None = None
        for row in self.rows():
            if row.get("run_id") != run_id:
                continue
            ts = float(row.get("ts", 0.0))
            if latest is None or ts > latest[0]:
                latest = (ts, str(row.get("status", "")))
        return latest[1] if latest is not None else None


@dataclass
class RunDecision:
    """Result of :meth:`Registry.decide` for one config."""

    cfg: RunConfig
    run_id: str
    run_dir: Path
    action: str  # "run" | "resume" | "skip"
    reason: str


class Registry:
    """Decides which runs in a sweep need execution vs. skip vs. resume."""

    def __init__(
        self,
        study_dir: Path | str,
        manifest_path: Path | str | None = None,
        *,
        retry_errors: bool = False,
    ) -> None:
        self.study_dir = Path(study_dir)
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(manifest_path or self.study_dir / "manifest.jsonl")
        self.retry_errors = retry_errors

    # --- Decision -------------------------------------------------------

    def decide(self, cfg: RunConfig) -> RunDecision:
        rid = run_id_for(cfg)
        rdir = self.study_dir / rid
        summary_path = rdir / "summary.json"
        last_ckpt = rdir / "checkpoints" / "last.pt"

        # 1. Already completed OK -> skip.
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                summary = {}
            status = summary.get("status")
            if status == "ok":
                return RunDecision(cfg, rid, rdir, "skip", "already completed ok")
            if status == "diverged":
                return RunDecision(
                    cfg, rid, rdir, "skip", "diverged previously; needs manual inspection"
                )
            # "interrupted" or unknown -> fall through to resume/run logic.

        # 1b. Poisoned: previous attempt errored. Don't auto-retry unless
        # the user opted in with retry_errors=True; otherwise the sweep
        # would loop forever on a config that always crashes.
        if not self.retry_errors:
            last_status = self.manifest.latest_status(rid)
            if last_status == "error":
                return RunDecision(
                    cfg,
                    rid,
                    rdir,
                    "skip",
                    "previous attempt errored (manifest); pass --retry-errors to retry",
                )

        # 2. Checkpoint present -> resume.
        if last_ckpt.exists():
            return RunDecision(cfg, rid, rdir, "resume", "checkpoint exists")

        # 3. Otherwise -> fresh run.
        return RunDecision(cfg, rid, rdir, "run", "no prior artifacts")

    def plan(self, configs: Iterable[RunConfig]) -> list[RunDecision]:
        """Decide actions for every config in ``configs``."""
        return [self.decide(cfg) for cfg in configs]

    # --- Manifest helpers ----------------------------------------------

    def mark_started(self, cfg: RunConfig, *, machine: str | None = None) -> None:
        self.manifest.append(
            ManifestEntry(
                run_id=run_id_for(cfg),
                status="started",
                ts=time.time(),
                case=cfg.case,
                arch=cfg.arch,
                budget=cfg.budget,
                seed=cfg.seed,
                form=cfg.form,
                machine=machine,
            )
        )

    def mark_finished(
        self,
        cfg: RunConfig,
        *,
        status: str,
        wall_time_s: float | None = None,
        final_loss: float | None = None,
        error: str | None = None,
        machine: str | None = None,
    ) -> None:
        self.manifest.append(
            ManifestEntry(
                run_id=run_id_for(cfg),
                status=status,
                ts=time.time(),
                case=cfg.case,
                arch=cfg.arch,
                budget=cfg.budget,
                seed=cfg.seed,
                form=cfg.form,
                wall_time_s=wall_time_s,
                final_loss=final_loss,
                error=error,
                machine=machine,
            )
        )
