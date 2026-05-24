"""Tests for the A1/A2/A3 architectures and the factory."""

import pytest
import torch

from pinn_bath.models import BUDGET_TARGETS, BUDGET_TOL, build, count_parameters

CASES: list[tuple[str, dict]] = [
    ("1d_steady", dict(spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"))),
    ("1d_trans", dict(spatial_dim=1, has_t=True, output_fields=("h", "u", "zb"))),
    ("2d_trans", dict(spatial_dim=2, has_t=True, output_fields=("h", "u", "v", "zb"))),
]

ARCH_BUDGETS = [
    ("A1", "small"),
    ("A1", "medium"),
    ("A1", "large"),
    ("A2", "small"),
    ("A2", "medium"),
    ("A2", "large"),
    ("A3", "small"),
    ("A3", "medium"),
    ("A3", "large"),
]


def _build_coords(n: int, spatial_dim: int, has_t: bool) -> dict[str, torch.Tensor]:
    coords = {"x": torch.randn(n, 1)}
    if spatial_dim == 2:
        coords["y"] = torch.randn(n, 1)
    if has_t:
        coords["t"] = torch.randn(n, 1)
    return coords


@pytest.mark.fast
@pytest.mark.parametrize("arch_budget", ARCH_BUDGETS, ids=lambda p: f"{p[0]}_{p[1]}")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_param_count_within_budget(arch_budget: tuple[str, str], case: tuple[str, dict]) -> None:
    arch, budget = arch_budget
    _, kwargs = case
    model = build(arch, budget, **kwargs)
    n = count_parameters(model)
    target = BUDGET_TARGETS[budget]
    lo = int(target * (1 - BUDGET_TOL))
    hi = int(target * (1 + BUDGET_TOL))
    assert lo <= n <= hi, (
        f"{arch}/{budget}/{case[0]}: {n} params not in [{lo}, {hi}] (target {target})"
    )


@pytest.mark.fast
@pytest.mark.parametrize("arch_budget", ARCH_BUDGETS, ids=lambda p: f"{p[0]}_{p[1]}")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_forward_outputs(arch_budget: tuple[str, str], case: tuple[str, dict]) -> None:
    arch, budget = arch_budget
    _, kwargs = case
    model = build(arch, budget, **kwargs)
    coords = _build_coords(7, kwargs["spatial_dim"], kwargs["has_t"])
    out = model(coords)
    for field in kwargs["output_fields"]:
        assert field in out, f"{arch}/{budget}: missing field {field}"
        assert out[field].shape == (7, 1), f"{arch}/{budget}/{field}: {out[field].shape}"


@pytest.mark.fast
@pytest.mark.parametrize("arch_budget", ARCH_BUDGETS, ids=lambda p: f"{p[0]}_{p[1]}")
def test_h_is_positive(arch_budget: tuple[str, str]) -> None:
    arch, budget = arch_budget
    model = build(arch, budget, spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"))
    x = torch.linspace(-50.0, 50.0, 20).unsqueeze(-1)
    out = model({"x": x})
    assert (out["h"] > 0).all()


@pytest.mark.fast
@pytest.mark.parametrize("arch", ["A1", "A2", "A3"])
def test_gradient_flows_to_all_params(arch: str) -> None:
    model = build(arch, "small", spatial_dim=1, has_t=True, output_fields=("h", "u", "zb"))
    coords = {
        "x": torch.randn(8, 1, requires_grad=True),
        "t": torch.randn(8, 1, requires_grad=True),
    }
    out = model(coords)
    loss = sum(v.pow(2).mean() for v in out.values())
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad reaches {name} in {arch}"


@pytest.mark.fast
def test_a1_bathnet_ignores_t() -> None:
    """A1 imposes ∂zb/∂t = 0 structurally; zb must be invariant to t."""
    model = build("A1", "small", spatial_dim=1, has_t=True, output_fields=("h", "u", "zb"))
    x = torch.linspace(0.0, 1.0, 10).unsqueeze(-1)
    out_t0 = model({"x": x, "t": torch.zeros_like(x)})
    out_t5 = model({"x": x, "t": torch.full_like(x, 5.0)})
    assert torch.equal(out_t0["zb"], out_t5["zb"])


@pytest.mark.fast
def test_a3_zb_net_ignores_t() -> None:
    """A3's z_b net only sees spatial coords."""
    model = build("A3", "small", spatial_dim=1, has_t=True, output_fields=("h", "u", "zb"))
    x = torch.linspace(0.0, 1.0, 10).unsqueeze(-1)
    out_t0 = model({"x": x, "t": torch.zeros_like(x)})
    out_t5 = model({"x": x, "t": torch.full_like(x, 5.0)})
    assert torch.equal(out_t0["zb"], out_t5["zb"])


@pytest.mark.fast
def test_a2_zb_can_depend_on_t() -> None:
    """A2 has no structural constraint: zb may vary with t (loss handles it)."""
    model = build("A2", "small", spatial_dim=1, has_t=True, output_fields=("h", "u", "zb"))
    x = torch.linspace(0.0, 1.0, 10).unsqueeze(-1)
    out_t0 = model({"x": x, "t": torch.zeros_like(x)})
    out_t5 = model({"x": x, "t": torch.full_like(x, 5.0)})
    # A randomly initialized A2 should not give identical zb at different t
    # unless the network is degenerate, which is vanishingly unlikely.
    assert not torch.equal(out_t0["zb"], out_t5["zb"])


@pytest.mark.fast
def test_factory_rejects_unknown_arch() -> None:
    with pytest.raises(ValueError):
        build("A4", "small", spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"))


@pytest.mark.fast
def test_factory_rejects_unknown_budget() -> None:
    with pytest.raises(ValueError):
        build("A1", "xl", spatial_dim=1, has_t=False, output_fields=("h", "u", "zb"))


@pytest.mark.fast
def test_a1_requires_h_in_output() -> None:
    with pytest.raises(ValueError):
        build("A1", "small", spatial_dim=1, has_t=False, output_fields=("u", "zb"))
