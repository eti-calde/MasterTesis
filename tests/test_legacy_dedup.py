"""Byte-equivalence check for the legacy_blocks dedup refactor.

For each of Exps 01/02/03/05, re-instantiate the (now-aliased)
``SolutionNet`` and ``BathymetryNet`` under a fixed seed and confirm that
every parameter / buffer matches the golden ``state_dict`` captured
before the refactor (see ``tests/fixtures/legacy_golden/capture.py``).

If this passes, the refactor has not perturbed any historical baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

FIX = Path(__file__).parent / "fixtures" / "legacy_golden"
sys.path.insert(0, str(FIX))
from capture import EXPS, get_state_dicts  # noqa: E402


@pytest.mark.fast
@pytest.mark.parametrize("label", list(EXPS.keys()))
def test_legacy_blocks_byte_equivalent(label: str) -> None:
    exp_dir, sol_k, bath_k, cls_names = EXPS[label]
    golden = torch.load(FIX / f"{label}.pt", weights_only=True)
    sol_sd, bath_sd = get_state_dicts(exp_dir, sol_k, bath_k, cls_names)

    assert set(sol_sd.keys()) == set(golden["sol"].keys()), (
        f"{label}: sol state_dict keys mismatch: "
        f"got {sorted(sol_sd.keys())} vs golden {sorted(golden['sol'].keys())}"
    )
    assert set(bath_sd.keys()) == set(golden["bath"].keys())

    for k, v in golden["sol"].items():
        torch.testing.assert_close(
            sol_sd[k],
            v,
            atol=0,
            rtol=0,
            msg=f"{label}: sol[{k}] drift after refactor",
        )
    for k, v in golden["bath"].items():
        torch.testing.assert_close(
            bath_sd[k],
            v,
            atol=0,
            rtol=0,
            msg=f"{label}: bath[{k}] drift after refactor",
        )
