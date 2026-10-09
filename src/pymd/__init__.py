"""pymd: Lennard-Jones (and Gupta/EAM) molecular dynamics on the GPU from pure Python.

A Python successor to the C++ code yamd (github.com/walidhemati/yamd).
"""

from .engine import Berendsen, Engine, Langevin
from .system import LJParams, MDConfig, fcc_lattice, maxwell_velocities

__all__ = [
    "Berendsen",
    "Engine",
    "LJParams",
    "Langevin",
    "MDConfig",
    "fcc_lattice",
    "maxwell_velocities",
    "make_engine",
]
__version__ = "0.1.0"


def make_engine(backend: str = "numpy", variant: str | None = None, **kwargs) -> Engine:
    """Construct an engine from the backend registry (see :mod:`pymd.backends`)."""
    from . import backends

    return backends.get(backend)(variant=variant, **kwargs)
