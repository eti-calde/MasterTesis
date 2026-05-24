"""Structured logging for training runs (S5, S6, S15).

A :class:`RunRecorder` owns a run directory and writes:

- ``env.json``: capture of the runtime environment at start.
- ``config.yaml``: the :class:`pinn_bath.config.RunConfig` driving the run.
- ``metrics.jsonl``: one JSON-encoded row per logged event (append-only).
- ``heartbeat.json``: atomically rewritten periodically so a watchdog can
  detect stuck runs.
- ``summary.json``: written when the run finishes (or aborts).

All non-append writes are atomic (write to a tmp file, then rename).
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any, TextIO

import torch

from pinn_bath.config import RunConfig
from pinn_bath.env import capture


class RunRecorder:
    """Collects logs and artifacts produced during a single training run."""

    def __init__(
        self,
        run_dir: Path | str,
        cfg: RunConfig | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.start_time: float = time.time()

        if cfg is not None:
            cfg.to_yaml(self.run_dir / "config.yaml")
        capture(repo_root=repo_root).write(self.run_dir / "env.json")

        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._heartbeat_path = self.run_dir / "heartbeat.json"
        self._summary_path = self.run_dir / "summary.json"
        self._fh: TextIO = self._metrics_path.open("a", encoding="utf-8")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    def log_epoch(self, **fields: Any) -> None:
        """Append a metrics row to ``metrics.jsonl``."""
        row = {"t_elapsed_s": round(self.elapsed_s, 4), **fields}
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._fh.flush()

    def heartbeat(self, **fields: Any) -> None:
        """Rewrite ``heartbeat.json`` atomically (S15)."""
        row = {"ts": time.time(), "t_elapsed_s": round(self.elapsed_s, 4), **fields}
        _atomic_write_json(self._heartbeat_path, row, indent=2)

    def write_summary(self, status: str = "ok", **fields: Any) -> None:
        """Rewrite ``summary.json`` atomically with a final report."""
        summary: dict[str, Any] = {
            "status": status,
            "wall_time_s": round(self.elapsed_s, 4),
            **fields,
        }
        if torch.cuda.is_available():
            summary["peak_vram_mb"] = int(torch.cuda.max_memory_allocated() // (1024 * 1024))
        if self.cfg is not None:
            from pinn_bath.config import SCHEMA_VERSION

            summary["schema_version"] = SCHEMA_VERSION
            summary["run_id"] = self.cfg.run_id
            summary["case"] = self.cfg.case
            summary["arch"] = self.cfg.arch
            summary["budget"] = self.cfg.budget
            summary["form"] = self.cfg.form
            summary["seed"] = self.cfg.seed
        _atomic_write_json(self._summary_path, summary, indent=2, sort_keys=True)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _atomic_write_json(
    path: Path, payload: Any, *, indent: int = 2, sort_keys: bool = False
) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (write tmp, then rename)."""
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n"
    tmp = path.parent / f".{path.stem}.{secrets.token_hex(4)}.json"
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
