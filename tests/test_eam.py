"""Gupta / second-moment EAM for gold (yamd tests/test_ducastelle.cpp) and the GPU version."""

from __future__ import annotations

import numpy as np
import pytest

from pymd.eam import AU_MASS, GuptaParams, gupta_allpairs, mackay_icosahedron


def test_gupta_finite_differences_yamd_case():
    """yamd: 2x2x2 cubic lattice (a = 1.5) + 0.1 noise, cutoff 5, |F_fd - F| < 1e-5."""
    rng = np.random.default_rng(0)
    g = np.stack(np.meshgrid(*[np.arange(2)] * 3, indexing="ij"), -1).reshape(-1, 3) * 1.5
    pos = g + 0.1 * rng.uniform(-1, 1, g.shape)
    gp = GuptaParams(cutoff=5.0)
    f0, _ = gupta_allpairs(pos, None, gp)
    delta = 1e-4
    for i in range(len(pos)):
        for k in range(3):
            p = pos.copy()
            p[i, k] += delta
            ep = gupta_allpairs(p, None, gp)[1]
            p[i, k] -= 2 * delta
            em = gupta_allpairs(p, None, gp)[1]
            assert abs(-(ep - em) / (2 * delta) - f0[i, k]) < 1e-5


def test_mackay_counts():
    assert [len(mackay_icosahedron(n)) for n in range(5)] == [1, 13, 55, 147, 309]
    x = mackay_icosahedron(2)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() == pytest.approx(4.079 / np.sqrt(2), rel=0.06)


def test_gold_cohesive_energy_reasonable():
    """Perfect fcc Au with Cleri-Rosato parameters: E_coh close to the 3.78 eV fit target."""
    from pymd.system import fcc_lattice

    pos, box = fcc_lattice(5, density=4 / 4.079**3)
    _, e = gupta_allpairs(pos, box, GuptaParams(cutoff=10.0))
    assert -3.9 < e / len(pos) < -3.7
    assert AU_MASS == pytest.approx(196.96657 / 9.64853321e-3)


@pytest.mark.gpu
def test_gupta_gpu_matches_reference():
    pytest.importorskip("cupy")
    from pymd.eam import make_gupta_engine

    x = mackay_icosahedron(4)
    x = x + 0.05 * np.random.default_rng(1).standard_normal(x.shape) + 40.0
    box = np.full(3, 80.0)
    gp = GuptaParams()
    fref, eref = gupta_allpairs(x, None, gp)
    eng = make_gupta_engine(x, np.zeros_like(x), box, gp)
    f, e, _ = eng.forces()
    assert np.abs(f - fref).max() / np.abs(fref).max() < 1e-4
    assert e == pytest.approx(eref, rel=1e-5)
    e0 = eng.thermo()["etot"]
    eng.run(200)
    assert eng.thermo()["etot"] == pytest.approx(e0, abs=2e-4)
