"""PyTorch backends: eager tensor code and ``torch.compile`` (Inductor-generated Triton).

The neighbor list is a padded, atom-major ``(N, M)`` index matrix (padding = the atom
itself, masked out by ``r2 > 0``). Forces are a *gather* formulation:
``x[nl] -> (N, M, 3)`` displacements, masked LJ, reduction over ``M``. Eager mode
materialises those intermediates (so it is processed in chunks); ``torch.compile`` can fuse
gather, pair math and reduction into a single generated kernel.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..engine import Engine
from ..reference import cell_grid, stencil


def is_available() -> bool:
    """PyTorch with a CUDA device."""
    return torch is not None and torch.cuda.is_available()


def _lj_gather(x, xn, box, rc2: float, lj1: float, lj2: float, lj3: float, lj4: float,
               eshift: float, energy: bool):
    """Force (and per-atom energy/virial) on atoms ``x`` (C,3) from neighbor coords ``xn``."""
    d = x[:, None, :] - xn
    d = d - box * torch.round(d / box)
    r2 = (d * d).sum(-1)
    m = (r2 < rc2) & (r2 > 0)
    r2i = torch.where(m, 1.0 / torch.where(m, r2, torch.ones_like(r2)), torch.zeros_like(r2))
    r6i = r2i * r2i * r2i
    ff = r6i * (lj1 * r6i - lj2) * r2i
    f = (ff[..., None] * d).sum(1)
    if not energy:
        return f, None, None
    e = torch.where(m, r6i * (lj3 * r6i - lj4) - eshift, torch.zeros_like(r2)).sum(1)
    w = (ff * r2).sum(1)
    return f, 0.5 * e, 0.5 * w


def _force_full(x, nl, box, rc2: float, lj1: float, lj2: float, lj3: float, lj4: float,
                eshift: float, energy: bool, rows=None):
    """Gather formulation per Cartesian component (SoA), which Inductor fuses into one kernel.

    ``rows`` optionally restricts the computation to a slice of atoms (eager chunking).
    """
    ds = []
    for k in range(3):
        xk = x[:, k]
        xi = xk if rows is None else xk[rows]
        d = xi[:, None] - xk[nl]
        ds.append(d - box[k] * torch.round(d / box[k]))
    r2 = ds[0] * ds[0] + ds[1] * ds[1] + ds[2] * ds[2]
    m = (r2 < rc2) & (r2 > 0)
    r2i = torch.where(m, 1.0 / r2, 0.0)
    r6i = r2i * r2i * r2i
    ff = r6i * (lj1 * r6i - lj2) * r2i
    f = torch.stack([(ff * ds[k]).sum(1) for k in range(3)], 1)
    if not energy:
        return f, None, None
    e = torch.where(m, r6i * (lj3 * r6i - lj4) - eshift, 0.0).sum(1)
    w = (ff * r2).sum(1)
    return f, 0.5 * e, 0.5 * w


def _cand_mask(x, idx, cand, box, rl2: float):
    """Neighbor candidates ``cand`` (C, W) of atoms ``idx`` within ``rl2``; returns (mask, j)."""
    valid = (cand >= 0) & (cand != idx[:, None])
    cj = torch.where(valid, cand, idx[:, None])
    d = x[idx][:, None, :] - x[cj]
    d = d - box * torch.round(d / box)
    return valid & ((d * d).sum(-1) < rl2), cj


def _kick_drift(x, v, f, kick: float, drift: float):
    v2 = v + kick * f
    return x + drift * v2, v2


class TorchEagerEngine(Engine):
    """Plain PyTorch tensor operations (no custom kernels)."""

    backend = "torch-eager"
    variants = {
        "gather": "padded full neighbor list, gather-based forces, chunked over atoms",
        "allpairs": "dense O(N^2) distance blocks with cutoff, chunked",
    }
    compiled = False
    #: target elements per (chunk x M) intermediate in eager mode
    chunk_elems = 1 << 23

    def _setup(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.dev = torch.device("cuda")
        self.uses_list = self.variant != "allpairs"
        self.x = torch.tensor(pos, dtype=torch.float32, device=self.dev)
        self.v = torch.tensor(vel, dtype=torch.float32, device=self.dev)
        self.f = torch.zeros_like(self.x)
        self.ids = torch.arange(self.n, device=self.dev, dtype=torch.int64)
        self.boxt = torch.tensor(self.box, dtype=torch.float32, device=self.dev)
        self.x_ref = self.x.clone()
        p = self.params
        s6 = p.sigma**6
        self.lj = (p.rc**2, 48 * p.epsilon * s6 * s6, 24 * p.epsilon * s6,
                   4 * p.epsilon * s6 * s6, 4 * p.epsilon * s6, p.eshift)
        self.ncell = cell_grid(self.box, self.rlist)
        nc = self.ncell
        cells = np.arange(int(np.prod(nc)))
        cc = np.stack([cells // (nc[1] * nc[2]), (cells // nc[2]) % nc[1], cells % nc[2]], 1)
        nb = []
        for off in stencil(nc):
            a = (cc + off) % nc
            nb.append((a[:, 0] * nc[1] + a[:, 1]) * nc[2] + a[:, 2])
        self.cell_nb = torch.tensor(np.stack(nb, 1), device=self.dev)
        self.nl = torch.zeros((self.n, 1), dtype=torch.int64, device=self.dev)
        self._force_fn = _force_full
        self._kd_fn = _kick_drift
        self._mask_fn = _cand_mask
        if self.compiled:
            self._mask_fn = torch.compile(_cand_mask, dynamic=True)
            self._force_fn = torch.compile(_force_full, dynamic=False)
            self._kd_fn = torch.compile(_kick_drift, dynamic=False)

    # ---------------------------------------------------------------- neighbor list
    def _cell_key(self, x):
        nc = torch.tensor(self.ncell, device=self.dev)
        c = torch.floor(x / self.boxt * nc).long()
        c = torch.minimum(torch.clamp(c, min=0), nc - 1)
        return (c[:, 0] * nc[1] + c[:, 1]) * nc[2] + c[:, 2]

    def _rebuild(self) -> None:
        n = self.n
        self.x = self.x - self.boxt * torch.floor(self.x / self.boxt)
        key = self._cell_key(self.x)
        order = torch.argsort(key)
        if self.cfg.sort != "none":
            self.x, self.v = self.x[order], self.v[order]
            self.ids, key = self.ids[order], key[order]
            order = torch.arange(n, device=self.dev)
        skey = key[order]
        ncells = int(np.prod(self.ncell))
        counts = torch.bincount(skey, minlength=ncells)
        starts = torch.cumsum(counts, 0) - counts
        p = int(counts.max())
        table = torch.full((ncells, p), -1, dtype=torch.int64, device=self.dev)
        table[skey, torch.arange(n, device=self.dev) - starts[skey]] = order
        rl2 = self.rlist**2
        width = self.cell_nb.shape[1] * p
        chunk = max(1, self.chunk_elems // width)
        # keep the padded width monotone and 16-aligned so compiled graphs are reused
        rows, mmax = [], max(16, self.nl.shape[1])
        for s in range(0, n, chunk):
            idx = torch.arange(s, min(n, s + chunk), device=self.dev)
            cand = table[self.cell_nb[key[idx]]].reshape(len(idx), width)
            m, cj = self._mask_fn(self.x, idx, cand, self.boxt, rl2)
            cnt = m.sum(1)
            mc = max(1, int(cnt.max()))
            pos = torch.where(m, torch.cumsum(m, 1) - 1, mc)
            out = torch.full((len(idx), mc + 1), -1, dtype=torch.int64, device=self.dev)
            out.scatter_(1, pos, cj)
            rows.append(out[:, :mc])
            mmax = max(mmax, -(-mc // 16) * 16)
        self.nl = torch.cat(
            [torch.nn.functional.pad(r, (0, mmax - r.shape[1]), value=-1) for r in rows]
        )
        pad = self.nl < 0
        self._nnb = float((~pad).sum()) / n
        self.nl[pad] = torch.arange(n, device=self.dev)[:, None].expand_as(self.nl)[pad]
        self.x_ref = self.x.clone()

    # ---------------------------------------------------------------- forces
    def _compute_forces(self, energy: bool = False):
        n = self.n
        if self.variant == "allpairs":
            return self._allpairs(energy)
        if self.compiled:
            f, e, w = self._force_fn(self.x, self.nl, self.boxt, *self.lj, energy)
            self.f = f
        else:
            chunk = max(1, self.chunk_elems // self.nl.shape[1])
            f = torch.empty_like(self.x)
            e = torch.empty(n, device=self.dev) if energy else None
            w = torch.empty(n, device=self.dev) if energy else None
            for s in range(0, n, chunk):
                sl = slice(s, s + chunk)
                fc, ec, wc = _force_full(self.x, self.nl[sl], self.boxt, *self.lj, energy,
                                         rows=sl)
                f[sl] = fc
                if energy:
                    e[sl], w[sl] = ec, wc
            self.f = f
        if energy:
            return float(e.double().sum()), float(w.double().sum())
        return None

    def _allpairs(self, energy: bool):
        n = self.n
        chunk = max(1, self.chunk_elems // n)
        f = torch.empty_like(self.x)
        etot = wtot = 0.0
        for s in range(0, n, chunk):
            sl = slice(s, s + chunk)
            xn = self.x[None, :, :].expand(len(self.x[sl]), n, 3)
            fc, ec, wc = _lj_gather(self.x[sl], xn, self.boxt, *self.lj, energy)
            f[sl] = fc
            if energy:
                etot += float(ec.double().sum())
                wtot += float(wc.double().sum())
        self.f = f
        return (etot, wtot) if energy else None

    # ---------------------------------------------------------------- integrator
    def _kick_drift(self, kick: float, drift: float) -> None:
        self.x, self.v = self._kd_fn(self.x, self.v, self.f, kick, drift)

    def _max_disp2(self) -> float:
        return float(((self.x - self.x_ref) ** 2).sum(1).max())

    def _ekin(self) -> float:
        return 0.5 * self.cfg.mass * float((self.v.double() ** 2).sum())

    def _scale_v(self, lam: float) -> None:
        self.v = self.v * lam

    def _ou(self, c1: float, c2: float, rng) -> None:
        self.v = c1 * self.v + c2 * torch.randn_like(self.v)

    def _download(self):
        return (self.x.double().cpu().numpy(), self.v.double().cpu().numpy(),
                self.ids.cpu().numpy())

    def _download_forces(self):
        return self.f.double().cpu().numpy(), self.ids.cpu().numpy()

    def mean_neighbors(self) -> float:
        return getattr(self, "_nnb", float("nan")) if self.uses_list else float("nan")

    def synchronize(self) -> None:
        torch.cuda.synchronize()


class TorchCompileEngine(TorchEagerEngine):
    """The same gather formulation, compiled by TorchInductor into fused Triton kernels."""

    backend = "torch-compile"
    variants = {"gather": "padded full neighbor list, gather forces fused by torch.compile"}
    compiled = True
