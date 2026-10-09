"""NumPy float64 CPU backend (the correctness oracle and the CPU baseline)."""

from __future__ import annotations

import numpy as np

from ..engine import Engine
from ..reference import build_pairs, lj_allpairs, lj_pairs, wrap


class NumpyEngine(Engine):
    """Vectorized NumPy: half neighbor list from a sorted cell list, ``bincount`` scatter."""

    backend = "numpy"
    variants = {
        "cell-verlet": "float64 half Verlet list from an argsort cell list, bincount scatter",
        "allpairs": "float64 O(N^2) direct summation with minimum image",
    }

    def _setup(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.uses_list = self.variant != "allpairs"
        self.x = pos.copy()
        self.v = vel.copy()
        self.f = np.zeros_like(pos)
        self.ids = np.arange(self.n)
        self.x_ref = self.x.copy()
        self.pi = self.pj = np.zeros(0, dtype=np.int64)

    def _rebuild(self) -> None:
        self.x = wrap(self.x, self.box)
        self.pi, self.pj = build_pairs(self.x, self.box, self.rlist)
        self.x_ref = self.x.copy()

    def _compute_forces(self, energy: bool = False):
        if self.variant == "allpairs":
            f, e, w = lj_allpairs(self.x, self.box, self.params)
        else:
            f, e, w = lj_pairs(self.x, self.box, self.pi, self.pj, self.params)
        self.f = f
        return (e, w) if energy else None

    def _kick_drift(self, kick: float, drift: float) -> None:
        if kick != 0.0:
            self.v += kick * self.f
        if drift != 0.0:
            self.x += drift * self.v

    def _max_disp2(self) -> float:
        d = self.x - self.x_ref
        return float(np.max(np.einsum("ij,ij->i", d, d)))

    def _ekin(self) -> float:
        return 0.5 * self.cfg.mass * float(np.sum(self.v * self.v))

    def _scale_v(self, lam: float) -> None:
        self.v *= lam

    def _get_v_raw(self) -> np.ndarray:
        return self.v

    def _set_v_raw(self, v: np.ndarray) -> None:
        self.v = np.ascontiguousarray(v)

    def _download(self):
        return self.x.copy(), self.v.copy(), self.ids

    def _download_forces(self):
        return self.f.copy(), self.ids

    def mean_neighbors(self) -> float:
        return float(len(self.pi)) / self.n
