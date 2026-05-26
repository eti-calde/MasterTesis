"""M10: fix for the h>0 collapse bug.

Verifies that:
1. ``softplus_positive(x, floor=0)`` reproduces the original behaviour.
2. ``softplus_positive(x, floor>0)`` guarantees ``h >= floor`` structurally.
3. ``positivity(h, eps=0)`` is identically zero when h>0 (preserves the
   pre-M10 "dead loss" behaviour for backward-compat).
4. ``positivity(h, eps>0)`` penalises ``h < eps`` (the actual fix).
5. ``RunConfig.run_id`` is stable when the new fields are at their default
   (so existing run directories remain valid on resume).
6. A1/A2/A3 forward passes respect ``h_floor`` end-to-end.
"""

from __future__ import annotations

import pytest
import torch

from pinn_bath.config import DataCfg, LossWeights, RunConfig
from pinn_bath.losses.components import positivity
from pinn_bath.models.base import softplus_positive
from pinn_bath.models.factory import build

# --- Activation ------------------------------------------------------------


@pytest.mark.fast
def test_softplus_positive_floor_zero_matches_original() -> None:
    raw = torch.tensor([-30.0, -1.0, 0.0, 1.0, 5.0])
    out = softplus_positive(raw, floor=0.0)
    expected = torch.nn.functional.softplus(raw)
    torch.testing.assert_close(out, expected)


@pytest.mark.fast
def test_softplus_positive_floor_lifts_minimum() -> None:
    # softplus(-30) ≈ 9e-14 — practically zero. With floor=0.05 the output
    # must be >= 0.05 everywhere, regardless of how negative raw is.
    raw = torch.tensor([-30.0, -10.0, -1.0, 0.0, 1.0])
    out = softplus_positive(raw, floor=0.05)
    assert (out >= 0.05).all()
    # And for very negative raw, out should be very close to floor.
    assert out[0].item() == pytest.approx(0.05, abs=1e-6)


# --- Loss component --------------------------------------------------------


@pytest.mark.fast
def test_positivity_eps_zero_is_dead_loss_for_positive_h() -> None:
    # Old behaviour: with eps=0 (default) the penalty is zero for any h>0,
    # so attaching a non-zero pos weight to a softplus-bounded model is a
    # no-op. This is the documented bug that motivates the M10 fix.
    h = torch.tensor([1e-12, 0.001, 1.0, 100.0])  # all > 0
    assert positivity(h, eps=0.0).item() == 0.0


@pytest.mark.fast
def test_positivity_eps_active_penalises_shallow_h() -> None:
    # h = [0, 0.05, 0.10] with eps=0.10:
    #   relu(0.10 - h) = [0.10, 0.05, 0.00]
    #   squared       = [0.01, 0.0025, 0.0]
    #   mean          = 0.0125/3
    h = torch.tensor([0.0, 0.05, 0.10])
    expected = (0.01 + 0.0025 + 0.0) / 3.0
    assert positivity(h, eps=0.10).item() == pytest.approx(expected)


@pytest.mark.fast
def test_positivity_default_arg_preserves_legacy_signature() -> None:
    # Callers that still invoke ``positivity(h)`` (no eps arg) must get the
    # same value as before the M10 change.
    h = torch.tensor([-0.5, 0.0, 1.0])
    assert positivity(h).item() == pytest.approx(0.25 / 3.0)


# --- Config: backward-compatible run_id ------------------------------------


def _make_cfg(**overrides):
    base = dict(
        case="exp1",
        arch="A1",
        budget="small",
        form="primitive",
        seed=0,
        data=DataCfg(case_path="data/exp1.npz", observations=["eta"]),
    )
    base.update(overrides)
    return RunConfig(**base)


@pytest.mark.fast
def test_run_id_unchanged_when_h_floor_default() -> None:
    # Two configs that only differ in whether h_floor was passed explicitly
    # at its default (0.0) must produce the same run_id, so existing run
    # directories survive the M10 schema change.
    cfg_default = _make_cfg()
    cfg_explicit = _make_cfg(h_floor=0.0)
    assert cfg_default.run_id == cfg_explicit.run_id


@pytest.mark.fast
def test_run_id_unchanged_when_pos_eps_default() -> None:
    cfg_default = _make_cfg()
    cfg_explicit = _make_cfg(loss=LossWeights(pos_eps=0.0))
    assert cfg_default.run_id == cfg_explicit.run_id


@pytest.mark.fast
def test_run_id_changes_when_h_floor_nonzero() -> None:
    cfg_a = _make_cfg(h_floor=0.0)
    cfg_b = _make_cfg(h_floor=0.05)
    assert cfg_a.run_id != cfg_b.run_id


@pytest.mark.fast
def test_run_id_changes_when_pos_eps_nonzero() -> None:
    cfg_a = _make_cfg(loss=LossWeights(pos_eps=0.0))
    cfg_b = _make_cfg(loss=LossWeights(pos_eps=0.01))
    assert cfg_a.run_id != cfg_b.run_id


# --- Architecture: end-to-end h_floor honoured -----------------------------


@pytest.mark.fast
@pytest.mark.parametrize("arch", ["A1", "A2", "A3"])
def test_model_output_respects_h_floor(arch: str) -> None:
    # Build the model with an aggressive floor and confirm the forward
    # output of "h" is >= floor at randomly-sampled coords, no matter how
    # the (random-init) weights happened to produce the raw value.
    floor = 0.05
    model = build(
        arch,
        "small",
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        h_floor=floor,
    )
    # 100 random x in [-10, 10] — wide enough to exercise tanh saturation.
    x = torch.linspace(-10.0, 10.0, 100).unsqueeze(-1)
    with torch.no_grad():
        out = model({"x": x})
    h = out["h"].squeeze()
    assert (h >= floor - 1e-6).all(), f"{arch}: min h = {h.min().item()} < floor {floor}"


@pytest.mark.fast
@pytest.mark.parametrize("arch", ["A1", "A2", "A3"])
def test_model_with_floor_zero_matches_no_floor(arch: str) -> None:
    # Reproducibility check: h_floor=0.0 produces exactly the same output
    # as if h_floor had never been threaded through the constructor.
    torch.manual_seed(123)
    model_a = build(
        arch,
        "small",
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=0,
        h_floor=0.0,
    )
    torch.manual_seed(123)
    model_b = build(
        arch,
        "small",
        spatial_dim=1,
        has_t=False,
        output_fields=("h", "u", "zb"),
        ff_seed=0,
    )
    x = torch.linspace(-5.0, 5.0, 30).unsqueeze(-1)
    with torch.no_grad():
        a = model_a({"x": x})["h"]
        b = model_b({"x": x})["h"]
    torch.testing.assert_close(a, b)
