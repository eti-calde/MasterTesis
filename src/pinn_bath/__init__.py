"""pinn_bath — physics-informed neural networks for inverse bathymetry."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pinn-bath")
except PackageNotFoundError:
    __version__ = "0.1.0+dev"

__all__ = ["__version__"]
