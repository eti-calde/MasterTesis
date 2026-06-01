"""Training loop + evaluation for the inverse operator (F3).

Loss = MSE(zb_pred, zb_true) [normalized] + ``lambda_phys`` * SWE residual
(physical units). The headline experiment runs ``lambda_phys = 0`` (pure
supervised) vs ``> 0`` (physics-informed) and compares OOD (hard-test) RMSE.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pinn_bath.operator.architectures import build_operator, count_parameters
from pinn_bath.operator.data import make_loaders
from pinn_bath.operator.physics import physics_loss
from pinn_bath.seed import set_seed


@torch.no_grad()
def evaluate(model, loader, norm, device, *, by_tier: bool = False):
    """RMSE of zb in physical units over a loader.

    Returns a float, or — when ``by_tier`` — a dict ``{"all", "easy", ...}``
    of per-tier RMSE (tiers present in the loader only).
    """
    model.eval()
    se, n = 0.0, 0
    tier_se: dict[int, float] = {}
    tier_n: dict[int, int] = {}
    for b in loader:
        eta, u, zb = b["eta"].to(device), b["u"].to(device), b["zb"].to(device)
        zb_pred = norm.denorm_zb(model(norm.input_tensor(eta, u)))
        sq = ((zb_pred - zb) ** 2).sum(dim=-1)  # per-case SSE
        se += float(sq.sum())
        n += zb.numel()
        if by_tier:
            diff = b["difficulty"].numpy()
            nx = zb.shape[-1]
            for c in set(diff.tolist()):
                m = diff == c
                tier_se[c] = tier_se.get(c, 0.0) + float(sq.cpu().numpy()[m].sum())
                tier_n[c] = tier_n.get(c, 0) + int(m.sum()) * nx
    allrmse = (se / max(n, 1)) ** 0.5
    if not by_tier:
        return allrmse
    names = {0: "easy", 1: "medium", 2: "hard"}
    out = {"all": allrmse}
    for c in sorted(tier_se):
        out[names[c]] = (tier_se[c] / max(tier_n[c], 1)) ** 0.5
    return out


@torch.no_grad()
def evaluate_per_case(model, loader, norm, device) -> dict[str, np.ndarray]:
    """Per-case RMSE + difficulty score (for the RMSE-vs-difficulty plot)."""
    model.eval()
    rmse, score, diff = [], [], []
    for b in loader:
        eta, u, zb = b["eta"].to(device), b["u"].to(device), b["zb"].to(device)
        zb_pred = norm.denorm_zb(model(norm.input_tensor(eta, u)))
        per = ((zb_pred - zb) ** 2).mean(dim=-1).sqrt().cpu().numpy()
        rmse.append(per)
        score.append(b["score"].numpy())
        diff.append(b["difficulty"].numpy())
    return {
        "rmse": np.concatenate(rmse),
        "score": np.concatenate(score),
        "difficulty": np.concatenate(diff),
    }


def train_operator(
    dataset_dir: str | Path,
    *,
    arch: str = "cnn",
    size: str = "small",
    lambda_phys: float = 0.0,
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    out_dir: str | Path | None = None,
    log_every: int = 10,
) -> dict[str, Any]:
    set_seed(seed, deterministic=False)  # conv kernels lack deterministic impls
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(dataset_dir, batch_size=batch_size)
    norm, dx, dt = loaders["normalizer"], loaders["dx"], loaders["dt"]
    model = build_operator(arch, size=size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_fh = (out_dir / "metrics.jsonl").open("w")
    else:
        metrics_fh = None

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        ep_loss = ep_mse = ep_phys = ep_cont = ep_mom = ep_gnorm = 0.0
        nb = 0
        for b in loaders["train"]:
            eta, u, zb = b["eta"].to(device), b["u"].to(device), b["zb"].to(device)
            zb_pred_n = model(norm.input_tensor(eta, u))
            mse = F.mse_loss(zb_pred_n, norm.norm_zb(zb))
            loss = mse
            phys_val = cont = mom = 0.0
            if lambda_phys > 0:
                lp, parts = physics_loss(eta, u, norm.denorm_zb(zb_pred_n), dx, dt)
                loss = mse + lambda_phys * lp
                phys_val, cont, mom = float(lp), parts["cont"], parts["mom"]
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
            opt.step()
            ep_loss += float(loss)
            ep_mse += float(mse)
            ep_phys += phys_val
            ep_cont += cont
            ep_mom += mom
            ep_gnorm += float(gnorm)
            nb += 1

        # Cheap per-epoch train metrics, always logged for smooth curves.
        row = {
            "epoch": epoch,
            "t_s": round(time.time() - t0, 1),
            "lr": opt.param_groups[0]["lr"],
            "train_loss": ep_loss / nb,
            "train_mse": ep_mse / nb,
            "train_phys": ep_phys / nb,
            "train_phys_cont": ep_cont / nb,
            "train_phys_mom": ep_mom / nb,
            "grad_norm": ep_gnorm / nb,
        }
        # Expensive eval (val + OOD test, per tier) only every eval_every epochs.
        do_eval = epoch % log_every == 0 or epoch == epochs - 1
        if do_eval:
            val = evaluate(model, loaders["val"], norm, device, by_tier=True)
            test = evaluate(model, loaders["test"], norm, device, by_tier=True)
            row["val_rmse"] = val["all"]
            row["test_rmse_ood"] = test["all"]
            for k, v in val.items():
                if k != "all":
                    row[f"val_rmse_{k}"] = v
            if val["all"] < best_val:
                best_val, best_epoch = val["all"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if metrics_fh:
            metrics_fh.write(json.dumps(row) + "\n")
            metrics_fh.flush()
        if do_eval:
            print(
                f"[{epoch:4d}] loss={row['train_loss']:.4f} mse={row['train_mse']:.4f} "
                f"phys={row['train_phys']:.3e} val={row['val_rmse']:.4f} "
                f"OOD={row['test_rmse_ood']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    result = {
        "arch": arch,
        "size": size,
        "lambda_phys": lambda_phys,
        "params": count_parameters(model),
        "device": device,
        "epochs": epochs,
        "seed": seed,
        "val_rmse": evaluate(model, loaders["val"], norm, device),
        "test_rmse_ood": evaluate(model, loaders["test"], norm, device),
        "best_val_rmse": best_val,
        "best_epoch": best_epoch,
        "physics_floor_true_zb": _physics_floor(loaders, norm, dx, dt, device),
    }
    if out_dir:
        if metrics_fh:
            metrics_fh.close()
        torch.save({"model": model.state_dict(), "norm": norm.as_dict()}, out_dir / "best.pt")
        per = evaluate_per_case(model, loaders["test"], norm, device)
        np.savez(out_dir / "test_per_case.npz", **per)
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def _physics_floor(loaders, norm, dx, dt, device) -> float:
    """SWE residual evaluated on the TRUE zb of the val set (signal floor)."""
    b = next(iter(loaders["val"]))
    eta, u, zb = b["eta"].to(device), b["u"].to(device), b["zb"].to(device)
    lp, _ = physics_loss(eta, u, zb, dx, dt)
    return float(lp)
