"""NumPy float64 reference: ports of yamd's GoogleTest cases plus internal consistency."""

from __future__ import annotations

import numpy as np
import pytest

from pymd.reference import build_pairs, cell_grid, lj_allpairs, lj_pairs, morton3, stencil
from pymd.system import LJParams, fcc_lattice, lj_tail_corrections


def test_lj_direct_summation_finite_differences():
    """yamd tests/test_lj_direct_summation.cpp: 10 random atoms, eps=0.7, sigma=0.3, no cutoff."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(-1, 1, (10, 3))
    params = LJParams(epsilon=0.7, sigma=0.3, rc=np.inf, shift=False)
    f0, e0, _ = lj_allpairs(pos, None, params)
    delta = 1e-4
    for i in range(10):
        for k in range(3):
            p = pos.copy()
            p[i, k] += delta
            ep = lj_allpairs(p, None, params)[1]
            p[i, k] -= 2 * delta
            em = lj_allpairs(p, None, params)[1]
            fd = -(ep - em) / (2 * delta)
            assert fd == pytest.approx(f0[i, k], rel=1e-5, abs=1e-8)


def test_periodic_forces_match_finite_differences():
    pos, box = fcc_lattice(3, density=0.8)
    pos = pos + 0.1 * np.random.default_rng(1).standard_normal(pos.shape)
    params = LJParams(rc=2.5, shift=True)
    f0, _, _ = lj_allpairs(pos, box, params)
    delta = 1e-6
    for i in (0, 17, 50):
        for k in range(3):
            p = pos.copy()
            p[i, k] += delta
            ep = lj_allpairs(p, box, params)[1]
            p[i, k] -= 2 * delta
            em = lj_allpairs(p, box, params)[1]
            assert -(ep - em) / (2 * delta) == pytest.approx(f0[i, k], rel=1e-5, abs=1e-6)


def test_cell_list_equals_allpairs():
    pos, box = fcc_lattice(6, shuffle=True)
    pos = pos + 0.08 * np.random.default_rng(2).standard_normal(pos.shape)
    params = LJParams()
    fa, ea, wa = lj_allpairs(pos, box, params)
    i, j = build_pairs(pos, box, params.rc + 0.3)
    assert np.all(i < j)
    fc, ec, wc = lj_pairs(pos, box, i, j, params)
    np.testing.assert_allclose(fc, fa, atol=1e-10)
    assert ec == pytest.approx(ea, rel=1e-12)
    assert wc == pytest.approx(wa, rel=1e-12)
    # brute-force pair count
    d = pos[:, None] - pos[None]
    d -= box * np.rint(d / box)
    r2 = (d**2).sum(-1)
    assert len(i) == (np.triu(r2 < (params.rc + 0.3) ** 2, 1)).sum()


def _yamd_pairs(positions: np.ndarray, cutoff: float) -> dict[int, set[int]]:
    """Open boundaries emulated with a box much larger than the system."""
    box = np.full(3, 100.0)
    i, j = build_pairs(positions + 50.0, box, cutoff)
    out: dict[int, set[int]] = {k: set() for k in range(len(positions))}
    for a, b in zip(i, j):
        out[a].add(b)
        out[b].add(a)
    return out


def test_neighbors_yamd_test1():
    """yamd tests/test_neighbors.cpp Test1: 4 atoms, cutoff 1.5 -> 10 (directed) pairs."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    nb = _yamd_pairs(pos, 1.5)
    assert sum(len(v) for v in nb.values()) == 10
    assert [len(nb[k]) for k in range(4)] == [3, 3, 2, 2]
    assert nb[0] == {1, 2, 3} and nb[1] == {0, 2, 3} and nb[2] == {0, 1} and nb[3] == {0, 1}


def test_neighbors_yamd_first_atom_has_none():
    pos = np.array([[0, 0, 0], [7, 0, 0], [0, 7, 0], [0, 6, 0]], dtype=float)
    nb = _yamd_pairs(pos, 5.0)
    assert [len(nb[k]) for k in range(4)] == [0, 0, 1, 1]
    assert _yamd_pairs(pos, 0.5) == {0: set(), 1: set(), 2: set(), 3: set()}


def test_grid_and_stencil():
    assert list(cell_grid(np.array([10.0, 5.0, 8.4]), 2.8)) == [3, 1, 3]
    assert len(stencil(np.array([3, 1, 3]))) == 9
    assert len(stencil(np.array([4, 4, 4]))) == 27
    assert morton3(np.array([1]), np.array([0]), np.array([0]))[0] == 1
    assert morton3(np.array([0]), np.array([1]), np.array([0]))[0] == 2
    assert morton3(np.array([1]), np.array([1]), np.array([1]))[0] == 7


def test_lammps_in_lj_lattice_energy():
    """Step 0 of LAMMPS bench/in.lj: E_pair = -6.7733681 for fcc 0.8442, lj/cut 2.5 (no shift)."""
    pos, box = fcc_lattice(5, density=0.8442)
    _, e, _ = lj_allpairs(pos, box, LJParams(shift=False))
    assert e / len(pos) == pytest.approx(-6.7733681, abs=1e-7)


def test_tail_correction_signs():
    u, p = lj_tail_corrections(0.8442, LJParams(rc=2.5))
    assert u < 0 and p < 0
    assert u == pytest.approx(8 / 3 * np.pi * 0.8442 * (1 / 3 * 2.5**-9 - 2.5**-3))
