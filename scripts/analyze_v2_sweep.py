#!/usr/bin/env python
"""Rigorous analysis of the v2 operator sweep -> stats JSON + figures.

Aggregates runs/op_sweep_v2 (per-cell summary.json + test_per_case.npz) into:
  - per (size, lambda) mean+/-std of val / OOD RMSE over seeds,
  - seed-paired physics tests (lambda>0 vs lambda=0) for the small operator,
  - seed-paired scaling test (small vs medium) at lambda=0,
  - per-case OOD error vs difficulty score (cliff + Spearman),
  - generalization gap.
Writes analysis/v2_stats.json and the two regenerable figures.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "runs" / "op_sweep_v2"
OUTDIR = ROOT / "analysis"
FIGDIR = ROOT / "Paper" / "figures"


def load_cells() -> dict:
    cells: dict[tuple[str, float], dict[int, dict]] = defaultdict(dict)
    for d in sorted(SWEEP.glob("*_lam*_s*")):
        sumf = d / "summary.json"
        if not sumf.exists():
            continue
        s = json.loads(sumf.read_text())
        key = (s["size"], float(s["lambda_phys"]))
        cells[key][int(s["seed"])] = {"dir": d, **s}
    return cells


def agg(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": len(a),
        "vals": [float(v) for v in a],
    }


def paired(a: list[float], b: list[float]) -> dict:
    """Paired test a (treatment) vs b (baseline): positive delta = a worse."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {
        "n": len(d),
        "mean_delta": float(d.mean()),
        "pct_change": float(d.mean() / b.mean() * 100.0),
    }
    if len(d) >= 2 and d.std() > 0:
        _t, p = stats.ttest_rel(a, b)
        out["t_p"] = float(p)
        out["cohen_dz"] = float(d.mean() / d.std(ddof=1))
        try:
            _w, pw = stats.wilcoxon(a, b)
            out["wilcoxon_p"] = float(pw)
        except ValueError:
            out["wilcoxon_p"] = None
    return out


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    cells = load_cells()
    stats_out: dict = {"table": {}, "physics": {}, "scaling": {}, "difficulty": {}, "gap": {}}

    # ---- table: per (size, lambda) val/OOD ----
    for (size, lam), seeds in sorted(cells.items()):
        vals = [r["val_rmse"] for r in seeds.values()]
        oods = [r["test_rmse_ood"] for r in seeds.values()]
        floors = [r["physics_floor_true_zb"] for r in seeds.values()]
        beps = [r["best_epoch"] for r in seeds.values()]
        stats_out["table"][f"{size}|{lam:g}"] = {
            "size": size,
            "lambda": lam,
            "seeds": sorted(seeds),
            "val": agg(vals),
            "ood": agg(oods),
            "phys_floor": agg(floors),
            "best_epoch": agg([float(b) for b in beps]),
        }

    # ---- physics effect (small): each lambda>0 vs lambda=0, paired by seed ----
    base = cells[("small", 0.0)]
    for lam in (1e-3, 1e-2, 1e-1):
        treat = cells[("small", lam)]
        common = sorted(set(base) & set(treat))
        a = [treat[s]["test_rmse_ood"] for s in common]
        b = [base[s]["test_rmse_ood"] for s in common]
        stats_out["physics"][f"small|{lam:g}_vs_0"] = paired(a, b)

    # ---- scaling (lambda=0): medium vs small, paired by seed ----
    if ("medium", 0.0) in cells:
        sm, md = cells[("small", 0.0)], cells[("medium", 0.0)]
        common = sorted(set(sm) & set(md))
        a = [md[s]["test_rmse_ood"] for s in common]
        b = [sm[s]["test_rmse_ood"] for s in common]
        stats_out["scaling"]["medium_vs_small_lam0"] = paired(a, b)
        stats_out["scaling"]["medium_val"] = agg([md[s]["val_rmse"] for s in common])
        stats_out["scaling"]["medium_ood"] = agg([md[s]["test_rmse_ood"] for s in common])
        stats_out["scaling"]["medium_params"] = md[common[0]]["params"]
        stats_out["scaling"]["small_params"] = sm[common[0]]["params"]
        stats_out["scaling"]["note"] = (
            f"medium partial: lambda=0 only, seeds {common}; medium lambda-sweep pending"
        )

    # ---- generalization gap (small lam0) ----
    sm = cells[("small", 0.0)]
    gv = np.mean([sm[s]["val_rmse"] for s in sm])
    go = np.mean([sm[s]["test_rmse_ood"] for s in sm])
    stats_out["gap"] = {"val": float(gv), "ood": float(go), "factor": float(go / gv)}

    # ---- difficulty cliff: pool per-case OOD over the 5 small lam0 seeds ----
    scores, rmses = [], []
    for s in sm:
        d = np.load(sm[s]["dir"] / "test_per_case.npz")
        scores.append(d["score"])
        rmses.append(d["rmse"])
    score = np.concatenate(scores)
    rmse = np.concatenate(rmses)
    rho, prho = stats.spearmanr(score, rmse)
    # deciles by score
    qs = np.quantile(score, np.linspace(0, 1, 11))
    dec_rmse = []
    for i in range(10):
        m = (score >= qs[i]) & (score <= qs[i + 1] if i == 9 else score < qs[i + 1])
        dec_rmse.append(float(rmse[m].mean()))
    stats_out["difficulty"] = {
        "spearman_rho": float(rho),
        "spearman_p": float(prho),
        "decile_rmse": dec_rmse,
        "decile_edges": [float(q) for q in qs],
        "ratio_hardest_easiest_decile": float(dec_rmse[-1] / dec_rmse[0]),
        "score_range": [float(score.min()), float(score.max())],
        "n_cases": len(score),
    }

    (OUTDIR / "v2_stats.json").write_text(json.dumps(stats_out, indent=2))
    print(json.dumps(stats_out, indent=2))

    # ================= figures =================
    # error_vs_score (small lam0 vs lam1e-2 pooled) + in-dist reference
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for lam, col, lab in [
        (0.0, "tab:blue", r"$\lambda_{\rm phys}=0$"),
        (1e-2, "tab:orange", r"$\lambda_{\rm phys}=10^{-2}$"),
    ]:
        ss, rr = [], []
        for s in cells[("small", lam)]:
            d = np.load(cells[("small", lam)][s]["dir"] / "test_per_case.npz")
            ss.append(d["score"])
            rr.append(d["rmse"])
        ss = np.concatenate(ss)
        rr = np.concatenate(rr)
        order = np.argsort(ss)
        # rolling mean over score
        k = 200
        sm_s = np.convolve(ss[order], np.ones(k) / k, mode="valid")
        sm_r = np.convolve(rr[order], np.ones(k) / k, mode="valid")
        ax.plot(sm_s, sm_r * 100, color=col, lw=1.6, label=lab)
    ax.axhline(gv * 100, ls=":", color="gray", lw=1.0, label="val (in-dist)")
    ax.set_xlabel("score de dificultad $D$")
    ax.set_ylabel("RMSE de $z_b$ [cm]")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "error_vs_score.png", dpi=200)
    plt.close(fig)

    # ood bars: small (4 lambda) + medium (lam0 only)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    lams = [0.0, 1e-3, 1e-2, 1e-1]
    xs = np.arange(len(lams))
    sm_means = [stats_out["table"][f"small|{lm:g}"]["ood"]["mean"] * 100 for lm in lams]
    sm_stds = [stats_out["table"][f"small|{lm:g}"]["ood"]["std"] * 100 for lm in lams]
    ax.bar(
        xs - 0.2, sm_means, 0.4, yerr=sm_stds, capsize=3, label="pequeño (0.28M)", color="tab:blue"
    )
    if ("medium", 0.0) in cells:
        md_m = stats_out["table"]["medium|0"]["ood"]["mean"] * 100
        md_s = stats_out["table"]["medium|0"]["ood"]["std"] * 100
        ax.bar(
            [0 + 0.2],
            [md_m],
            0.4,
            yerr=[md_s],
            capsize=3,
            label="mediano (1.48M), $\\lambda{=}0$",
            color="tab:orange",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(["0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    ax.set_xlabel(r"$\lambda_{\rm phys}$")
    ax.set_ylabel("RMSE OOD de $z_b$ [cm]")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "operator_ood_bars.png", dpi=200)
    plt.close(fig)
    print(f"\nFigures written to {FIGDIR}")


if __name__ == "__main__":
    main()
