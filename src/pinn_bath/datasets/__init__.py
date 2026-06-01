"""Dataset adapters that produce :class:`~pinn_bath.data.Case` from
external benchmark files (Angel et al. 2024, etc.), plus sampling
helpers used by sweep studies (N_t restrictions).

The legacy adapters (``angel``, ``sampling``) import torch; the
operator-pivot modules (``generator``, ``operator_dataset``) do not. To let
the forward solver + case generator run in a torch-free environment (e.g.
generating data on a host where torch lives only in a container), the
torch-dependent names are imported lazily via ``__getattr__`` rather than at
package import time.
"""

from typing import Any

__all__ = [
    "case_from_angel_flume",
    "evenly_spaced_indices",
    "subsample_t_observations",
]


def __getattr__(name: str) -> Any:
    # Lazy re-export: only pull in the torch-dependent adapters on first use.
    if name == "case_from_angel_flume":
        from pinn_bath.datasets.angel import case_from_angel_flume

        return case_from_angel_flume
    if name in ("evenly_spaced_indices", "subsample_t_observations"):
        from pinn_bath.datasets import sampling

        return getattr(sampling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
