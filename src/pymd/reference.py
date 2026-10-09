"""Pure NumPy float64 reference: the correctness oracle for every GPU backend.

* :func:`lj_allpairs` -- O(N^2) direct summation (periodic minimum image or open boundaries).
* :func:`cell_grid`, :func:`build_pairs` -- cell-list neighbor search (sort atoms by cell index).
* :func:`lj_pairs` -- forces / energy / virial over an explicit half pair list.
"""

from __future__ import annotations

import itertools

import numpy as np

from .system import LJParams


def wrap(pos: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrap positions into ``[0, L)``."""
    return pos - box * np.floor(pos / box)


def min_image(d: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Apply the minimum-image convention in place (no-op for open boundaries)."""
    if box is not None:
        d -= box * np.rint(d / box)
    return d


def _lj_scalar(r2: np.ndarray, params: LJParams) -> tuple[np.ndarray, np.ndarray]:
    """Return (pair energy incl. shift, f/r) for squared distances inside the cutoff."""
    sr2 = params.sigma**2 / r2
    sr6 = sr2 * sr2 * sr2
    e = 4.0 * params.epsilon * (sr6 * sr6 - sr6) - params.eshift
    ff = 24.0 * params.epsilon * (2.0 * sr6 * sr6 - sr6) / r2
    return e, ff


def lj_allpairs(
    pos: np.ndarray,
    box: np.ndarray | None,
    params: LJParams,
    chunk: int = 1024,
) -> tuple[np.ndarray, float, float]:
    """Direct O(N^2) summation in float64.

    ``box=None`` means open boundaries. ``params.rc = inf`` gives the untruncated potential
    (what yamd's ``lj_direct_summation`` computes). Returns ``(forces, epot, virial)`` with
    ``virial = sum_pairs r_ij . f_ij``.
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    forces = np.zeros_like(pos)
    epot = 0.0
    virial = 0.0
    rc2 = params.rc**2
    for s in range(0, n, chunk):
        blk = pos[s : s + chunk]
        d = blk[:, None, :] - pos[None, :, :]
        min_image(d, box)
        r2 = np.einsum("ijk,ijk->ij", d, d)
        idx = np.arange(s, s + len(blk))
        r2[np.arange(len(blk)), idx] = np.inf
        mask = r2 < rc2
        e = np.zeros_like(r2)
        ff = np.zeros_like(r2)
        e[mask], ff[mask] = _lj_scalar(r2[mask], params)
        forces[s : s + chunk] = np.einsum("ij,ijk->ik", ff, d)
        epot += 0.5 * e.sum()
        virial += 0.5 * float(np.sum(np.where(mask, ff * np.where(mask, r2, 0.0), 0.0)))
    return forces, float(epot), float(virial)


def cell_grid(box: np.ndarray, rlist: float) -> np.ndarray:
    """Number of cells per dimension (cell edge >= ``rlist``).

    Dimensions that would get fewer than 3 cells collapse to a single cell, so that the
    27-cell stencil never visits the same neighbor cell twice.
    """
    n = np.floor(np.asarray(box) / rlist).astype(np.int64)
    n[n < 3] = 1
    return n


def stencil(ncell: np.ndarray) -> np.ndarray:
    """Unique neighbor-cell offsets (27 for a full 3-D grid)."""
    ranges = [(-1, 0, 1) if k >= 3 else (0,) for k in ncell]
    return np.array(list(itertools.product(*ranges)), dtype=np.int64)


def morton3(cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> np.ndarray:
    """Interleave the bits of three (<= 10-bit) integer coordinates (Z-order key)."""

    def spread(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64) & 0x3FF
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    return (spread(cx) | (spread(cy) << 1) | (spread(cz) << 2)).astype(np.int64)


def cell_coords(pos: np.ndarray, box: np.ndarray, ncell: np.ndarray) -> np.ndarray:
    """Integer cell coordinates (N, 3) of wrapped positions."""
    c = np.floor(wrap(pos, box) / box * ncell).astype(np.int64)
    return np.minimum(c, ncell - 1)


def build_pairs(
    pos: np.ndarray, box: np.ndarray, rlist: float, cell_chunk: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Half neighbor list ``(i, j)`` with ``i < j`` and ``|r_ij| < rlist`` via a cell list.

    Atoms are sorted by linear cell index (``argsort``); each cell's atoms are then laid out in
    a padded ``(ncell, max_per_cell)`` table and candidate pairs are generated for every
    stencil offset at once.
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    ncell = cell_grid(box, rlist)
    c3 = cell_coords(pos, box, ncell)
    cid = (c3[:, 0] * ncell[1] + c3[:, 1]) * ncell[2] + c3[:, 2]
    n_cells = int(np.prod(ncell))
    order = np.argsort(cid, kind="stable")
    counts = np.bincount(cid, minlength=n_cells)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    maxpc = int(counts.max())
    table = np.full((n_cells, maxpc), -1, dtype=np.int64)
    sorted_cid = cid[order]
    table[sorted_cid, np.arange(n) - starts[sorted_cid]] = order

    all_cells = np.arange(n_cells)
    ccx = all_cells // (ncell[1] * ncell[2])
    ccy = (all_cells // ncell[2]) % ncell[1]
    ccz = all_cells % ncell[2]
    rl2 = rlist * rlist
    out_i: list[np.ndarray] = []
    out_j: list[np.ndarray] = []
    for off in stencil(ncell):
        nb = (
            ((ccx + off[0]) % ncell[0]) * ncell[1] + (ccy + off[1]) % ncell[1]
        ) * ncell[2] + (ccz + off[2]) % ncell[2]
        for s in range(0, n_cells, cell_chunk):
            a = table[s : s + cell_chunk]  # (C, P)
            b = table[nb[s : s + cell_chunk]]  # (C, P)
            ai = np.broadcast_to(a[:, :, None], (len(a), maxpc, maxpc))
            bj = np.broadcast_to(b[:, None, :], (len(a), maxpc, maxpc))
            valid = (ai >= 0) & (bj >= 0) & (ai < bj)
            ii = ai[valid]
            jj = bj[valid]
            d = pos[ii] - pos[jj]
            min_image(d, box)
            keep = np.einsum("ij,ij->i", d, d) < rl2
            out_i.append(ii[keep])
            out_j.append(jj[keep])
    return np.concatenate(out_i), np.concatenate(out_j)


def lj_pairs(
    pos: np.ndarray, box: np.ndarray | None, i: np.ndarray, j: np.ndarray, params: LJParams
) -> tuple[np.ndarray, float, float]:
    """LJ forces, energy and virial over a half pair list (float64, ``np.bincount`` scatter)."""
    n = len(pos)
    d = pos[i] - pos[j]
    min_image(d, box)
    r2 = np.einsum("ij,ij->i", d, d)
    m = r2 < params.rc**2
    d, r2, ii, jj = d[m], r2[m], i[m], j[m]
    e, ff = _lj_scalar(r2, params)
    fij = ff[:, None] * d
    forces = np.empty((n, 3))
    for k in range(3):
        forces[:, k] = np.bincount(ii, fij[:, k], minlength=n) - np.bincount(
            jj, fij[:, k], minlength=n
        )
    return forces, float(e.sum()), float(np.sum(ff * r2))
