"""Dataset adapters that produce :class:`~pinn_bath.data.Case` from
external benchmark files (Angel et al. 2024, etc.)."""

from pinn_bath.datasets.angel import case_from_angel_flume

__all__ = ["case_from_angel_flume"]
