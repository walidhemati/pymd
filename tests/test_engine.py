"""Integrator, thermostats and bookkeeping on the NumPy engine (CPU, runs in CI)."""

from __future__ import annotations

import numpy as np
import pytest

from pymd import Berendsen, Langevin, LJParams, MDConfig, fcc_lattice, make_engine
from pymd.system import maxwell_velocities


def _engine(n=4, temp=1.44, **cfg):
    pos, box = fcc_lattice(n, shuffle=True)
    vel = maxwell_velocities(len(pos), temp)
    return make_engine("numpy", "cell-verlet", pos=pos, vel=vel, box=box, cfg=MDConfig(**cfg))


def test_constant_force_verlet():
    """yamd tests/test_verlet.cpp: free particles under unit force for 100 steps of 0.01."""
    from pymd.backends.numpy_backend import NumpyEngine

    eng = NumpyEngine(np.zeros((3, 3)) + 5, np.zeros((3, 3)), np.full(3, 30.0),
                      params=LJParams(rc=0.1), cfg=MDConfig(dt=0.01, skin=0.1), variant="allpairs")
    eng.f = np.ones((3, 3))
    eng._forces_valid = True
    eng._compute_forces = lambda energy=False: setattr(eng, "f", np.ones((3, 3)))  # type: ignore
    eng.run(100)
    x, v = eng.state()
    np.testing.assert_allclose(x - 5, 0.5 * 1.0**2, atol=1e-10)
    np.testing.assert_allclose(v, 1.0, atol=1e-10)


def test_nve_energy_conservation_and_momentum():
    eng = _engine(temp=0.8, check=True, rebuild_every=1)
    e0 = eng.thermo()["etot"]
    eng.run(400)
    t = eng.thermo()
    assert abs(t["etot"] - e0) < 2e-4
    _, v = eng.state()
    assert np.abs(v.sum(0)).max() < 1e-9


def test_rebuild_schedule():
    eng = _engine(rebuild_every=20, check=False)
    eng.run(100)
    assert eng.n_builds == 1 + 5
    eng2 = _engine(rebuild_every=1, check=True, temp=0.1)
    eng2.run(20)
    assert eng2.n_builds < 10


def test_berendsen_reaches_target():
    """yamd tests/test_thermostat.cpp analogue: rescaling drives T to the target."""
    eng = _engine(temp=0.3, check=True, rebuild_every=5)
    eng.run(600, thermostat=Berendsen(temperature=1.2, tau=0.05))
    assert eng.thermo()["temp"] == pytest.approx(1.2, rel=0.02)


def test_langevin_temperature():
    eng = _engine(n=5, temp=0.5, check=True, rebuild_every=5)
    eng.run(300, thermostat=Langevin(temperature=1.0, gamma=5.0, seed=1))
    temps = []
    for _ in range(20):
        eng.run(20, thermostat=Langevin(temperature=1.0, gamma=5.0))
        temps.append(eng.thermo()["temp"])
    assert np.mean(temps) == pytest.approx(1.0, rel=0.05)


def test_state_roundtrip_keeps_original_order():
    pos, box = fcc_lattice(4, shuffle=True)
    vel = maxwell_velocities(len(pos), 1.0)
    eng = make_engine("numpy", pos=pos, vel=vel, box=box)
    x, v = eng.state()
    np.testing.assert_allclose(x, pos)
    np.testing.assert_allclose(v, vel)
