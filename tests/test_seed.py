"""Tests for pinn_bath.seed."""

import numpy as np
import pytest
import torch

from pinn_bath.seed import get_rng_state, set_rng_state, set_seed


@pytest.mark.fast
def test_set_seed_reproduces_python_numpy_torch() -> None:
    import random

    set_seed(42, deterministic=False)
    samples = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    set_seed(42, deterministic=False)
    samples_again = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert samples == samples_again


@pytest.mark.fast
def test_rng_state_roundtrip() -> None:
    set_seed(0, deterministic=False)
    state = get_rng_state()
    a = np.random.rand(3)
    b = torch.rand(3)
    set_rng_state(state)
    a2 = np.random.rand(3)
    b2 = torch.rand(3)
    np.testing.assert_array_equal(a, a2)
    assert torch.equal(b, b2)
