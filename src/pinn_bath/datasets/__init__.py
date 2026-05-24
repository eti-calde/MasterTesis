"""Dataset adapters that produce :class:`~pinn_bath.data.Case` from
external benchmark files (Angel et al. 2024, etc.), plus sampling
helpers used by sweep studies (N_t restrictions)."""

from pinn_bath.datasets.angel import case_from_angel_flume
from pinn_bath.datasets.sampling import evenly_spaced_indices, subsample_t_observations

__all__ = [
    "case_from_angel_flume",
    "evenly_spaced_indices",
    "subsample_t_observations",
]
