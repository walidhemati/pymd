"""CPU-only checks of the LAMMPS/HOOMD head-to-head driver (bench/counterparts.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("counterparts", ROOT / "bench" / "counterparts.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

LOG = """Loop time of 0.115059 on 1 procs for 100 steps with 32000 atoms
Performance: 375459.586 tau/day, 869.119 timesteps/s, 27.812 Matom-step/s
Loop time of 0.453055 on 1 procs for 500 steps with 32000 atoms
Performance: 476763.617 tau/day, 1103.619 timesteps/s, 35.316 Matom-step/s
Loop time of 2.5 on 1 procs for 20 steps with 4000 atoms
Performance: 3456.0 tau/day, 8.0 timesteps/s, 32.000 katom-step/s
"""


def test_parse_runs_units_and_rates():
    rows = cp.parse_runs(LOG)
    assert [r["steps"] for r in rows] == [100, 500, 20]
    assert rows[1]["reported"] == 35.316e6
    assert rows[2]["reported"] == 32.0e3
    rate = rows[1]["atoms"] * rows[1]["steps"] / rows[1]["seconds"]
    assert abs(rate - rows[1]["reported"]) / rate < 1e-3


def test_round_steps_multiple_of_rebuild_interval():
    assert cp.round_steps(1) == cp.EVERY
    assert cp.round_steps(41) == 60
    assert all(cp.round_steps(x) % cp.EVERY == 0 for x in (5, 99, 1234.5))


def test_lammps_data_file_roundtrip(tmp_path):
    pos = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
    vel = np.array([[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0]])
    path = tmp_path / "x.data"
    cp.write_lammps_data(path, pos, vel, np.array([4.0, 4.0, 4.0]))
    text = path.read_text()
    assert "2 atoms" in text and "Atoms # atomic" in text and "Velocities" in text
    atoms = text.split("Atoms # atomic")[1].split("Velocities")[0].split()
    np.testing.assert_allclose(np.array(atoms, dtype=float).reshape(2, 5)[:, 2:], pos)
