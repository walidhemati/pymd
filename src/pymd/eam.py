"""Gupta / Ducastelle / Cleri-Rosato second-moment (EAM-type) potential for gold, as in yamd.

Units follow yamd: energies in eV, lengths in Angstrom, time in fs, masses in units of
``1 / 9.64853321e-3`` u (so that eV, Angstrom and fs are consistent).

Per-atom energy::

    E_i = sum_j A exp(-p (r_ij / r0 - 1))  -  sqrt( sum_j xi^2 exp(-2 q (r_ij / r0 - 1)) )

with the Au parameters used by yamd (A = 0.2061 eV, xi = 1.790 eV, p = 10.229, q = 4.036,
r0 = 4.079 / sqrt(2) Angstrom) and a hard cutoff (yamd default 10 Angstrom).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import min_image
from .system import LJParams, MDConfig

#: yamd mass unit conversion: 1 u in units of eV fs^2 / Angstrom^2
U_PER_MASS_UNIT = 1.0 / 9.64853321e-3
AU_MASS = 196.96657 * U_PER_MASS_UNIT
KB_EV_K = 8.617333262e-5


@dataclass(frozen=True)
class GuptaParams:
    """Second-moment tight-binding parameters (defaults: Au, as in yamd's ducastelle.h)."""

    A: float = 0.2061
    xi: float = 1.790
    p: float = 10.229
    q: float = 4.036
    r0: float = 4.079 / np.sqrt(2.0)
    cutoff: float = 10.0


#: Au parameters used by yamd (defaults of ``ducastelle()``)
AU_GUPTA = GuptaParams()


def gupta_allpairs(
    pos: np.ndarray, box: np.ndarray | None, gp: GuptaParams = AU_GUPTA
) -> tuple[np.ndarray, float]:
    """Float64 O(N^2) reference: returns ``(forces, potential energy)``."""
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    d = pos[:, None, :] - pos[None, :, :]
    min_image(d, box)
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    np.fill_diagonal(r, np.inf)
    m = r < gp.cutoff
    rr = np.where(m, r, 1.0)
    rep = np.where(m, 2 * gp.A * np.exp(-gp.p * (rr / gp.r0 - 1.0)), 0.0)
    dens = np.where(m, gp.xi**2 * np.exp(-2 * gp.q * (rr / gp.r0 - 1.0)), 0.0)
    rho = dens.sum(1)
    emb = -np.sqrt(rho)
    dfe = np.where(rho > 0, -0.5 / np.sqrt(np.where(rho > 0, rho, 1.0)), 0.0)
    drep = -rep * gp.p / gp.r0
    ddens = -2 * gp.q / gp.r0 * dens
    pair = drep + ddens * (dfe[:, None] + dfe[None, :])  # dE/dr_ij
    fr = np.where(m, -pair / rr, 0.0)
    forces = np.einsum("ij,ijk->ik", fr, d)
    energy = emb.sum() + 0.5 * rep.sum()
    del n
    return forces, float(energy)


def mackay_icosahedron(layers: int, nn_distance: float = 4.079 / np.sqrt(2.0)) -> np.ndarray:
    """Mackay icosahedron with ``layers`` shells around a central atom (port of yamd's ih.C).

    Atom counts: 1, 13, 55, 147, 309, 561, 923, 1415, 2057, ...
    """
    ht = (np.sqrt(5.0) + 1.0) / 4.0
    b = np.array([
        [ht, 0, 0.5], [ht, 0, -0.5], [0.5, ht, 0], [-0.5, ht, 0], [0, 0.5, ht], [0, -0.5, ht],
        [0.5, -ht, 0], [0, 0.5, -ht], [-ht, 0, 0.5], [0, -0.5, -ht], [-ht, 0, -0.5],
        [-0.5, -ht, 0],
    ])
    edges = [(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (10, 3), (10, 7), (10, 8), (10, 9),
             (10, 11), (1, 2), (1, 6), (1, 7), (1, 9), (8, 3), (8, 4), (8, 5), (8, 11), (2, 3),
             (2, 4), (2, 7), (11, 5), (11, 6), (11, 9), (6, 5), (6, 9), (3, 4), (3, 7), (7, 9),
             (5, 4)]
    faces = [(0, 1, 2), (0, 2, 4), (0, 4, 5), (0, 5, 6), (0, 1, 6), (10, 3, 7), (10, 3, 8),
             (10, 8, 11), (10, 9, 11), (10, 7, 9), (1, 2, 7), (1, 7, 9), (1, 6, 9), (8, 5, 11),
             (8, 4, 5), (8, 3, 4), (2, 3, 7), (2, 3, 4), (11, 5, 6), (11, 6, 9)]
    pts = [np.zeros(3)]
    for n in range(1, layers + 1):
        pts.extend(b * n)
        for i, j in edges:
            e1, e2 = b[i] * n, b[j] * n
            pts.extend(e1 + (e2 - e1) * k / n for k in range(1, n))
        for i, j, k in faces:
            e1, e2, e3 = b[i] * n, b[j] * n, b[k] * n
            for a in range(1, n - 1):
                v1 = e1 + (e2 - e1) * (a + 1) / n
                v2 = e1 + (e3 - e1) * (a + 1) / n
                pts.extend(v1 + (v2 - v1) * c / (a + 1) for c in range(1, a + 1))
    return np.array(pts) * nn_distance


def make_gupta_engine(
    pos: np.ndarray,
    vel: np.ndarray,
    box: np.ndarray,
    gp: GuptaParams = AU_GUPTA,
    cfg: MDConfig | None = None,
):
    """GPU (CuPy CUDA C) Gupta engine; open clusters use a large periodic box."""
    from .backends.cupy_backend import GuptaCupyEngine

    cfg = cfg or MDConfig(dt=1.0, skin=1.0, rebuild_every=10, check=True, mass=AU_MASS)
    return GuptaCupyEngine(pos=pos, vel=vel, box=box, params=LJParams(rc=gp.cutoff),
                           cfg=cfg, variant="full", gupta=gp)
