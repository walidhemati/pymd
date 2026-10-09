"""Shared fixtures: GPU tests are skipped when no CUDA device is present."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pymd import backends  # noqa: E402
from pymd.system import fcc_lattice, maxwell_velocities  # noqa: E402


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        try:
            from numba import cuda

            return bool(cuda.is_available())
        except Exception:
            return False


HAS_CUDA = _has_cuda()


def pytest_collection_modifyitems(config, items):
    if HAS_CUDA:
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def gpu_variants() -> list[tuple[str, str]]:
    """Every (backend, variant) of the available GPU backends."""
    if not HAS_CUDA:
        return []
    return [(b, v) for b, v in backends.all_variants() if b != "numpy"]


@pytest.fixture
def liquid_config():
    """2048 atoms: shuffled FCC at rho = 0.8442 with random displacements, T = 1.44."""
    pos, box = fcc_lattice(8, shuffle=True, seed=3)
    rng = np.random.default_rng(7)
    pos = pos + 0.05 * rng.standard_normal(pos.shape)
    vel = maxwell_velocities(len(pos), 1.44, seed=5)
    return pos, vel, box
