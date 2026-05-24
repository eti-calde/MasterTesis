"""Aggregate study results into tables + statistical comparisons (S4, S13).

The orchestrators (``arch_scaling.py``, ``ablation_forms.py``) write one
``summary.json`` per run. This module:

1. :func:`collect` walks a study directory and returns one flat row per run.
2. :func:`aggregate_cell` reduces a group of rows to ``mean ± std``,
   plus a bootstrap 95% CI for the requested metric.
3. :func:`wilcoxon_pairs` runs pairwise Wilcoxon signed-rank tests across
   the axis of interest (e.g. ``arch`` at fixed ``case`` and ``budget``).
4. :func:`format_text_table` and :func:`emit_latex_table` produce
   human-readable and ``\\input``-able outputs for the §5 LaTeX scaffold.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - scipy is pinned in pyproject
    _scipy_stats = None


# --- Loading ---------------------------------------------------------------


def collect(study_dir: Path | str) -> list[dict[str, Any]]:
    """Walk ``study_dir`` for ``*/summary.json`` and return one row per run.

    Each row flattens ``final_losses`` and ``final_metrics`` under ``loss_*``
    and ``metric_*`` keys so callers can reference them uniformly.
    """
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(Path(study_dir).glob("*/summary.json")):
        data = json.loads(summary_path.read_text())
        row: dict[str, Any] = {
            "run_id": data.get("run_id"),
            "case": data.get("case"),
            "arch": data.get("arch"),
            "budget": data.get("budget"),
            "seed": data.get("seed"),
            "status": data.get("status"),
            "wall_time_s": data.get("wall_time_s"),
            "adam_time_s": data.get("adam_time_s"),
            "lbfgs_time_s": data.get("lbfgs_time_s"),
            "peak_vram_mb": data.get("peak_vram_mb"),
        }
        for k, v in (data.get("final_losses") or {}).items():
            row[f"loss_{k}"] = v
        for k, v in (data.get("final_metrics") or {}).items():
            row[f"metric_{k}"] = v
        for k, v in (data.get("baseline_metrics") or {}).items():
            row[f"baseline_{k}"] = v
        rows.append(row)
    return rows


# --- Grouping --------------------------------------------------------------


def group_by(
    rows: Iterable[dict[str, Any]], axes: Sequence[str] = ("case", "arch", "budget")
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Group rows by the values at ``axes``."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(a) for a in axes)
        grouped.setdefault(key, []).append(row)
    return grouped


# --- Aggregation -----------------------------------------------------------


@dataclass
class AggregatedMetric:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def aggregate_cell(
    rows: Sequence[dict[str, Any]],
    metric_key: str,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> AggregatedMetric:
    """Mean ± std + bootstrap CI for ``metric_key`` across the given rows."""
    values = np.asarray(
        [row[metric_key] for row in rows if row.get(metric_key) is not None], dtype=float
    )
    if values.size == 0:
        return AggregatedMetric(0, float("nan"), float("nan"), float("nan"), float("nan"))
    if values.size == 1:
        v = float(values[0])
        return AggregatedMetric(1, v, 0.0, v, v)
    rng = np.random.default_rng(seed)
    n = values.size
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return AggregatedMetric(
        n=int(n),
        mean=float(values.mean()),
        std=float(values.std(ddof=1)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )


# --- Paired statistical tests ----------------------------------------------


@dataclass
class PairedComparison:
    a: Any
    b: Any
    n: int
    statistic: float
    p_value: float
    effect_size: float  # paired Cohen's d


def wilcoxon_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    axis: str = "arch",
    value_key: str = "metric_rmse_zb",
    pair_axis: str = "seed",
) -> list[PairedComparison]:
    """Pairwise Wilcoxon signed-rank tests across ``axis``.

    Within each (case, budget) cell, group runs by ``axis`` (e.g. arch),
    pair them by ``pair_axis`` (seed), and compare every pair. Returns one
    :class:`PairedComparison` per (a, b) pair.
    """
    if _scipy_stats is None:
        raise RuntimeError("scipy is required for wilcoxon_pairs")

    by_axis: dict[Any, dict[Any, float]] = {}
    for row in rows:
        a_val = row.get(axis)
        k = row.get(pair_axis)
        v = row.get(value_key)
        if a_val is None or k is None or v is None:
            continue
        by_axis.setdefault(a_val, {})[k] = float(v)

    levels = sorted(by_axis.keys(), key=repr)
    out: list[PairedComparison] = []
    for i, a in enumerate(levels):
        for b in levels[i + 1 :]:
            shared = sorted(set(by_axis[a]) & set(by_axis[b]))
            if len(shared) < 2:
                continue
            xa = np.asarray([by_axis[a][k] for k in shared])
            xb = np.asarray([by_axis[b][k] for k in shared])
            diff = xa - xb
            try:
                w_stat, w_p = _scipy_stats.wilcoxon(diff, zero_method="zsplit")
                stat, p = float(w_stat), float(w_p)
            except ValueError:  # all-zero differences
                stat, p = float("nan"), 1.0
            sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
            cohen_d = float(diff.mean()) / sd if sd > 0 else 0.0
            out.append(PairedComparison(a, b, len(shared), stat, p, cohen_d))
    return out


# --- Formatting ------------------------------------------------------------


def format_text_table(
    rows: Sequence[dict[str, Any]],
    *,
    axes: Sequence[str] = ("case", "arch", "budget"),
    metric_key: str = "metric_rmse_zb",
    fmt: str = "{:.4e}",
) -> str:
    """Render aggregated metrics as a fixed-width text table."""
    groups = group_by(rows, axes)
    header: list[str] = [*axes, "n", "mean", "std", "CI95"]
    out_rows: list[list[str]] = [header]
    for key, cell_rows in sorted(groups.items(), key=lambda kv: tuple(map(repr, kv[0]))):
        agg = aggregate_cell(cell_rows, metric_key)
        out_rows.append(
            [
                *(str(k) for k in key),
                str(agg.n),
                fmt.format(agg.mean),
                fmt.format(agg.std),
                f"[{fmt.format(agg.ci_low)}, {fmt.format(agg.ci_high)}]",
            ]
        )
    widths = [max(len(r[c]) for r in out_rows) for c in range(len(header))]
    lines = [" | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in out_rows]
    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([lines[0], sep, *lines[1:]])


def emit_latex_table(
    rows: Sequence[dict[str, Any]],
    *,
    axes: Sequence[str] = ("case", "arch", "budget"),
    metric_key: str = "metric_rmse_zb",
    fmt: str = "{:.3e}",
    caption: str = "",
    label: str = "",
) -> str:
    """Emit a LaTeX ``tabular`` for ``\\input`` in the §5 scaffold."""
    groups = group_by(rows, axes)
    cols = "l" * len(axes) + "rr"  # axes + mean + std
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}" if caption else "",
        rf"\label{{{label}}}" if label else "",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join([*axes, r"mean", r"std"]) + r" \\",
        r"\midrule",
    ]
    for key, cell_rows in sorted(groups.items(), key=lambda kv: tuple(map(repr, kv[0]))):
        agg = aggregate_cell(cell_rows, metric_key)
        lines.append(
            " & ".join([*map(str, key), fmt.format(agg.mean), fmt.format(agg.std)]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(filter(None, lines))
