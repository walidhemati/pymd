"""Every GPU backend/variant against the NumPy float64 reference."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import gpu_variants
from pymd import Berendsen, LJParams, MDConfig, backends
from pymd.reference import lj_allpairs

VARIANTS = gpu_variants()


@pytest.mark.gpu
@pytest.mark.parametrize(("backend", "variant"), VARIANTS)
def test_forces_match_reference(backend, variant, liquid_config):
    pos, vel, box = liquid_config
    params = LJParams()
    fref, eref, wref = lj_allpairs(pos, box, params)
    eng = backends.get(backend)(pos=pos, vel=vel, box=box, params=params, variant=variant)
    f, e, w = eng.forces()
    tol = 1e-12 if variant.endswith("fp64") else 5e-5
    assert np.abs(f - fref).max() / np.abs(fref).max() < tol
    assert e == pytest.approx(eref, rel=max(tol, 1e-6))
    assert w == pytest.approx(wref, rel=max(tol, 1e-6))


@pytest.mark.gpu
@pytest.mark.parametrize(("backend", "variant"), VARIANTS)
def test_nve_short_run_matches_numpy(backend, variant, liquid_config):
    """200 steps: energy conserved and the trajectory stays close to the float64 one."""
    pos, vel, box = liquid_config
    cfg = MDConfig(dt=0.002, rebuild_every=10, check=False)
    ref = backends.get("numpy")(pos=pos, vel=vel, box=box, cfg=cfg)
    eng = backends.get(backend)(pos=pos, vel=vel, box=box, cfg=cfg, variant=variant)
    e0 = eng.thermo()["etot"]
    ref.run(50)
    eng.run(50)
    assert abs(eng.thermo()["etot"] - e0) < 1e-3
    x_ref, _ = ref.state()
    x, _ = eng.state()
    d = x - x_ref
    d -= box * np.rint(d / box)
    assert np.abs(d).max() < 1e-3


@pytest.mark.gpu
@pytest.mark.parametrize("backend", sorted({b for b, _ in VARIANTS}))
def test_berendsen_on_gpu(backend, liquid_config):
    pos, vel, box = liquid_config
    cls = backends.get(backend)
    variant = next(v for v in cls.variants if v not in ("allpairs", "hashgrid"))
    eng = cls(pos=pos, vel=vel, box=box, cfg=MDConfig(check=True, rebuild_every=5),
              variant=variant)
    eng.run(300, thermostat=Berendsen(temperature=1.0, tau=0.05))
    assert eng.thermo()["temp"] == pytest.approx(1.0, rel=0.03)


def test_registry_lists_numpy():
    assert "numpy" in backends.available()
    assert set(backends.names()) >= {"numpy", "triton", "warp", "numba", "cupy-cuda-c"}
