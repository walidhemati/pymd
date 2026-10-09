"""Numba-CUDA backend: the same kernels as the CUDA C reference, written in Python syntax.

All device code is ``@cuda.jit`` Python compiled through NVVM. Kernels are kept strictly in
float32 (``float32`` casts, ``libdevice`` ``floorf``/``rintf``, float32 scalar arguments),
otherwise Numba's NumPy-style type promotion silently turns the force loop into fp64.
"""

from __future__ import annotations

import warnings

import numpy as np

# numba-cuda 0.30 still references ``np.row_stack`` (removed in NumPy 2.5). Restore the alias
# in-process before importing it.
if not hasattr(np, "row_stack"):  # pragma: no cover - depends on the installed NumPy
    setattr(np, "row_stack", np.vstack)  # noqa: B010

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from numba import cuda, float32, float64, int32
        from numba.cuda import libdevice
except Exception:  # pragma: no cover - import guard
    cuda = None

from ..engine import Engine
from ..reference import cell_grid, morton3

BLOCK = 128
SCAN_THREADS = 1024


def is_available() -> bool:
    """Numba-CUDA importable and a CUDA device present."""
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


if cuda is not None:

    @cuda.jit(device=True, inline=True)
    def _spread10(v):
        v &= 0x3FF
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    @cuda.jit(device=True, inline=True)
    def _cell_key(cx, cy, cz, ncx, ncy, ncz, morton):
        if morton:
            return _spread10(cx) | (_spread10(cy) << 1) | (_spread10(cz) << 2)
        return (cx * ncy + cy) * ncz + cz

    @cuda.jit(device=True, inline=True)
    def _cell_of(x, il, nc):
        c = int32(libdevice.floorf(x * il * float32(nc)))
        if c < 0:
            c = 0
        if c >= nc:
            c = nc - 1
        return c

    @cuda.jit
    def _fill_i32(a, value):
        i = cuda.grid(1)
        if i < a.size:
            a[i] = value

    @cuda.jit
    def _wrap_key(x, n, lx, ly, lz, ncx, ncy, ncz, morton, key):
        i = cuda.grid(1)
        if i >= n:
            return
        px = x[i, 0] - lx * libdevice.floorf(x[i, 0] / lx)
        py = x[i, 1] - ly * libdevice.floorf(x[i, 1] / ly)
        pz = x[i, 2] - lz * libdevice.floorf(x[i, 2] / lz)
        x[i, 0] = px
        x[i, 1] = py
        x[i, 2] = pz
        one = float32(1.0)
        cx = _cell_of(px, one / lx, ncx)
        cy = _cell_of(py, one / ly, ncy)
        cz = _cell_of(pz, one / lz, ncz)
        key[i] = _cell_key(cx, cy, cz, ncx, ncy, ncz, morton)

    @cuda.jit
    def _histogram(key, n, cnt):
        i = cuda.grid(1)
        if i < n:
            cuda.atomic.add(cnt, key[i], 1)

    @cuda.jit
    def _exclusive_scan(cnt, m, start):
        """Single-block chunked exclusive scan: start[0..m] (start[m] = total)."""
        sh = cuda.shared.array(SCAN_THREADS, dtype=int32)
        t = cuda.threadIdx.x
        chunk = (m + SCAN_THREADS - 1) // SCAN_THREADS
        lo = t * chunk
        hi = min(lo + chunk, m)
        s = 0
        for k in range(lo, hi):
            s += cnt[k]
        sh[t] = s
        cuda.syncthreads()
        off = 1
        while off < SCAN_THREADS:  # Hillis-Steele inclusive scan of the chunk sums
            val = sh[t - off] if t >= off else 0
            cuda.syncthreads()
            sh[t] += val
            cuda.syncthreads()
            off *= 2
        run = sh[t] - s  # exclusive prefix of this chunk
        for k in range(lo, hi):
            start[k] = run
            run += cnt[k]
        if t == SCAN_THREADS - 1:
            start[m] = sh[t]

    @cuda.jit
    def _copy_i32(src, dst, m):
        i = cuda.grid(1)
        if i < m:
            dst[i] = src[i]

    @cuda.jit
    def _scatter(key, n, cursor, order):
        i = cuda.grid(1)
        if i < n:
            order[cuda.atomic.add(cursor, key[i], 1)] = i

    @cuda.jit
    def _permute(order, n, x, v, ids, x2, v2, ids2):
        k = cuda.grid(1)
        if k >= n:
            return
        i = order[k]
        for c in range(4):
            x2[k, c] = x[i, c]
            v2[k, c] = v[i, c]
        ids2[k] = ids[i]

    @cuda.jit
    def _build_nlist(x, n, cell_start, cell_atoms, is_sorted, lx, ly, lz, ncx, ncy, ncz, morton,
                     rlist2, maxnb, nl, nnb, overflow):
        i = cuda.grid(1)
        if i >= n:
            return
        px = x[i, 0]
        py = x[i, 1]
        pz = x[i, 2]
        one = float32(1.0)
        ilx = one / lx
        ily = one / ly
        ilz = one / lz
        cx = _cell_of(px, ilx, ncx)
        cy = _cell_of(py, ily, ncy)
        cz = _cell_of(pz, ilz, ncz)
        rx = 1 if ncx >= 3 else 0
        ry = 1 if ncy >= 3 else 0
        rz = 1 if ncz >= 3 else 0
        cnt = 0
        for dx in range(-rx, rx + 1):
            ax = (cx + dx + ncx) % ncx
            for dy in range(-ry, ry + 1):
                ay = (cy + dy + ncy) % ncy
                for dz in range(-rz, rz + 1):
                    az = (cz + dz + ncz) % ncz
                    c = _cell_key(ax, ay, az, ncx, ncy, ncz, morton)
                    for k in range(cell_start[c], cell_start[c + 1]):
                        j = k if is_sorted else cell_atoms[k]
                        if j == i:
                            continue
                        ddx = px - x[j, 0]
                        ddy = py - x[j, 1]
                        ddz = pz - x[j, 2]
                        ddx -= lx * libdevice.rintf(ddx * ilx)
                        ddy -= ly * libdevice.rintf(ddy * ily)
                        ddz -= lz * libdevice.rintf(ddz * ilz)
                        if ddx * ddx + ddy * ddy + ddz * ddz < rlist2:
                            if cnt < maxnb:
                                nl[cnt * n + i] = j
                            cnt += 1
        nnb[i] = cnt
        if cnt > maxnb:
            cuda.atomic.max(overflow, 0, cnt)

    def _make_force(energy: bool):
        @cuda.jit
        def force_full(x, n, nl, nnb, lx, ly, lz, rc2, lj1, lj2, lj3, lj4, eshift, f, e_at, w_at):
            i = cuda.grid(1)
            if i >= n:
                return
            px = x[i, 0]
            py = x[i, 1]
            pz = x[i, 2]
            one = float32(1.0)
            ilx = one / lx
            ily = one / ly
            ilz = one / lz
            fx = float32(0.0)
            fy = float32(0.0)
            fz = float32(0.0)
            e = float32(0.0)
            w = float32(0.0)
            for k in range(nnb[i]):
                j = nl[k * n + i]
                dx = px - x[j, 0]
                dy = py - x[j, 1]
                dz = pz - x[j, 2]
                dx -= lx * libdevice.rintf(dx * ilx)
                dy -= ly * libdevice.rintf(dy * ily)
                dz -= lz * libdevice.rintf(dz * ilz)
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < rc2:
                    r2i = one / r2
                    r6i = r2i * r2i * r2i
                    ff = r6i * (lj1 * r6i - lj2) * r2i
                    fx += ff * dx
                    fy += ff * dy
                    fz += ff * dz
                    if energy:
                        e += r6i * (lj3 * r6i - lj4) - eshift
                        w += ff * r2
            f[i, 0] = fx
            f[i, 1] = fy
            f[i, 2] = fz
            if energy:
                half = float32(0.5)
                e_at[i] = half * e
                w_at[i] = half * w

        return force_full

    _force_full_e = _make_force(True)
    _force_full = _make_force(False)

    @cuda.jit
    def _fused_step(x, n, nl, nnb, lx, ly, lz, rc2, lj1, lj2, dtm, dt, v, xout):
        i = cuda.grid(1)
        if i >= n:
            return
        px = x[i, 0]
        py = x[i, 1]
        pz = x[i, 2]
        one = float32(1.0)
        ilx = one / lx
        ily = one / ly
        ilz = one / lz
        fx = float32(0.0)
        fy = float32(0.0)
        fz = float32(0.0)
        for k in range(nnb[i]):
            j = nl[k * n + i]
            dx = px - x[j, 0]
            dy = py - x[j, 1]
            dz = pz - x[j, 2]
            dx -= lx * libdevice.rintf(dx * ilx)
            dy -= ly * libdevice.rintf(dy * ily)
            dz -= lz * libdevice.rintf(dz * ilz)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < rc2:
                r2i = one / r2
                r6i = r2i * r2i * r2i
                ff = r6i * (lj1 * r6i - lj2) * r2i
                fx += ff * dx
                fy += ff * dy
                fz += ff * dz
        vx = v[i, 0] + dtm * fx
        vy = v[i, 1] + dtm * fy
        vz = v[i, 2] + dtm * fz
        v[i, 0] = vx
        v[i, 1] = vy
        v[i, 2] = vz
        xout[i, 0] = px + dt * vx
        xout[i, 1] = py + dt * vy
        xout[i, 2] = pz + dt * vz
        xout[i, 3] = x[i, 3]

    @cuda.jit
    def _kick_drift(x, v, f, n, kick, drift):
        i = cuda.grid(1)
        if i >= n:
            return
        vx = v[i, 0] + kick * f[i, 0]
        vy = v[i, 1] + kick * f[i, 1]
        vz = v[i, 2] + kick * f[i, 2]
        v[i, 0] = vx
        v[i, 1] = vy
        v[i, 2] = vz
        if drift != float32(0.0):
            x[i, 0] += drift * vx
            x[i, 1] += drift * vy
            x[i, 2] += drift * vz

    @cuda.jit
    def _scale_v(v, n, lam):
        i = cuda.grid(1)
        if i < n:
            v[i, 0] *= lam
            v[i, 1] *= lam
            v[i, 2] *= lam

    @cuda.jit(device=True, inline=True)
    def _block_sum_atomic(val, out):
        sh = cuda.shared.array(BLOCK, dtype=float64)
        t = cuda.threadIdx.x
        sh[t] = val
        cuda.syncthreads()
        s = BLOCK // 2
        while s > 0:
            if t < s:
                sh[t] += sh[t + s]
            cuda.syncthreads()
            s //= 2
        if t == 0:
            cuda.atomic.add(out, 0, sh[0])

    @cuda.jit
    def _sum_sq3(v, n, out):
        i = cuda.grid(1)
        val = float64(0.0)
        if i < n:
            val = float64(v[i, 0]) ** 2 + float64(v[i, 1]) ** 2 + float64(v[i, 2]) ** 2
        _block_sum_atomic(val, out)

    @cuda.jit
    def _sum1(a, n, out):
        i = cuda.grid(1)
        val = float64(0.0)
        if i < n:
            val = float64(a[i])
        _block_sum_atomic(val, out)

    @cuda.jit
    def _sum_i32(a, n, out):
        i = cuda.grid(1)
        val = float64(0.0)
        if i < n:
            val = float64(a[i])
        _block_sum_atomic(val, out)

    @cuda.jit
    def _max_disp2(x, xref, n, out):
        i = cuda.grid(1)
        if i < n:
            dx = x[i, 0] - xref[i, 0]
            dy = x[i, 1] - xref[i, 1]
            dz = x[i, 2] - xref[i, 2]
            cuda.atomic.max(out, 0, dx * dx + dy * dy + dz * dz)

    @cuda.jit
    def _copy4(src, dst, n):
        i = cuda.grid(1)
        if i < n:
            for c in range(4):
                dst[i, c] = src[i, c]


class NumbaEngine(Engine):
    """Numba-CUDA kernels (Python syntax, NVVM backend), float32."""

    backend = "numba"
    variants = {
        "full": "fp32, full Verlet list (no atomics), cell-sorted atoms",
        "full-fused": "fp32 full list, one fused force+kick+drift kernel per step (leapfrog)",
    }

    def _setup(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.supports_fused = self.variant == "full-fused"
        self.sort = self.cfg.sort
        n = self.n
        self.grid = (n + BLOCK - 1) // BLOCK
        x = np.zeros((n, 4), np.float32)
        v = np.zeros((n, 4), np.float32)
        x[:, :3] = pos
        v[:, :3] = vel
        self.x = cuda.to_device(x)
        self.v = cuda.to_device(v)
        self.f = cuda.to_device(np.zeros((n, 4), np.float32))
        self.ids = cuda.to_device(np.arange(n, dtype=np.int32))
        self.e_at = cuda.device_array(n, np.float32)
        self.w_at = cuda.device_array(n, np.float32)
        self.x_ref = cuda.device_array((n, 4), np.float32)
        _copy4[self.grid, BLOCK](self.x, self.x_ref, n)
        self.ncell = cell_grid(self.box, self.rlist)
        if self.sort == "morton":
            if self.ncell.max() > 1024:
                raise ValueError("Morton keys support at most 1024 cells per dimension")
            nc = self.ncell - 1
            nkeys = int(morton3(np.array([nc[0]]), np.array([nc[1]]), np.array([nc[2]]))[0]) + 1
        else:
            nkeys = int(np.prod(self.ncell))
        self.nkeys = nkeys
        self.key = cuda.device_array(n, np.int32)
        self.counts = cuda.device_array(nkeys, np.int32)
        self.cell_start = cuda.device_array(nkeys + 1, np.int32)
        self.cursor = cuda.device_array(nkeys, np.int32)
        self.order = cuda.device_array(n, np.int32)
        self.nnb = cuda.to_device(np.zeros(n, np.int32))
        self.overflow = cuda.device_array(1, np.int32)
        self.acc = cuda.device_array(1, np.float64)
        self.accf = cuda.device_array(1, np.float32)
        dens = n / np.prod(self.box)
        est = dens * 4.0 / 3.0 * np.pi * self.rlist**3
        self.maxnb = int(np.ceil(est * 1.3 / 8) * 8) + 8
        self.nl = cuda.device_array(self.maxnb * n, np.int32)
        if self.supports_fused:
            self.x_alt = cuda.device_array((n, 4), np.float32)
        self._box = tuple(np.float32(b) for b in self.box)
        p = self.params
        s6 = p.sigma**6
        self._lj = tuple(
            np.float32(c)
            for c in (
                p.rc**2,
                48.0 * p.epsilon * s6 * s6,
                24.0 * p.epsilon * s6,
                4.0 * p.epsilon * s6 * s6,
                4.0 * p.epsilon * s6,
                p.eshift,
            )
        )

    # ------------------------------------------------------------------ neighbor list
    def _rebuild(self) -> None:
        n, g = self.n, self.grid
        ncx, ncy, ncz = (np.int32(c) for c in self.ncell)
        morton = np.int32(self.sort == "morton")
        n32 = np.int32(n)
        _wrap_key[g, BLOCK](self.x, n32, *self._box, ncx, ncy, ncz, morton, self.key)
        gk = (self.nkeys + BLOCK - 1) // BLOCK
        _fill_i32[gk, BLOCK](self.counts, np.int32(0))
        _histogram[g, BLOCK](self.key, n32, self.counts)
        _exclusive_scan[1, SCAN_THREADS](self.counts, np.int32(self.nkeys), self.cell_start)
        _copy_i32[gk, BLOCK](self.cell_start, self.cursor, np.int32(self.nkeys))
        _scatter[g, BLOCK](self.key, n32, self.cursor, self.order)
        is_sorted = np.int32(0)
        if self.sort != "none":
            x2 = cuda.device_array_like(self.x)
            v2 = cuda.device_array_like(self.v)
            ids2 = cuda.device_array_like(self.ids)
            _permute[g, BLOCK](self.order, n32, self.x, self.v, self.ids, x2, v2, ids2)
            self.x, self.v, self.ids = x2, v2, ids2
            is_sorted = np.int32(1)
        rl2 = np.float32(self.rlist**2)
        while True:
            _fill_i32[1, 32](self.overflow, np.int32(0))
            _build_nlist[g, BLOCK](self.x, n32, self.cell_start, self.order, is_sorted,
                                   *self._box, ncx, ncy, ncz, morton, rl2,
                                   np.int32(self.maxnb), self.nl, self.nnb, self.overflow)
            ov = int(self.overflow.copy_to_host()[0])
            if ov == 0:
                break
            self.maxnb = int(np.ceil(ov * 1.2 / 8) * 8)
            self.nl = cuda.device_array(self.maxnb * n, np.int32)
        _copy4[g, BLOCK](self.x, self.x_ref, n32)
        self._forces_valid = False

    # ------------------------------------------------------------------ forces/integration
    def _dsum(self, kernel, arr) -> float:
        self.acc.copy_to_device(np.zeros(1, np.float64))
        kernel[self.grid, BLOCK](arr, np.int32(self.n), self.acc)
        return float(self.acc.copy_to_host()[0])

    def _compute_forces(self, energy: bool = False):
        k = _force_full_e if energy else _force_full
        k[self.grid, BLOCK](self.x, np.int32(self.n), self.nl, self.nnb, *self._box, *self._lj,
                            self.f, self.e_at, self.w_at)
        if energy:
            return self._dsum(_sum1, self.e_at), self._dsum(_sum1, self.w_at)
        return None

    def _kick_drift(self, kick: float, drift: float) -> None:
        _kick_drift[self.grid, BLOCK](self.x, self.v, self.f, np.int32(self.n),
                                      np.float32(kick), np.float32(drift))

    def _fused_step(self, dt: float) -> None:
        im = 1.0 / self.cfg.mass
        _fused_step[self.grid, BLOCK](self.x, np.int32(self.n), self.nl, self.nnb, *self._box,
                                      *self._lj[:3], np.float32(dt * im), np.float32(dt), self.v,
                                      self.x_alt)
        self.x, self.x_alt = self.x_alt, self.x

    def _max_disp2(self) -> float:
        self.accf.copy_to_device(np.zeros(1, np.float32))
        _max_disp2[self.grid, BLOCK](self.x, self.x_ref, np.int32(self.n), self.accf)
        return float(self.accf.copy_to_host()[0])

    def _ekin(self) -> float:
        return 0.5 * self.cfg.mass * self._dsum(_sum_sq3, self.v)

    def _scale_v(self, lam: float) -> None:
        _scale_v[self.grid, BLOCK](self.v, np.int32(self.n), np.float32(lam))

    def _get_v_raw(self) -> np.ndarray:
        return self.v.copy_to_host()[:, :3].astype(np.float64)

    def _set_v_raw(self, v: np.ndarray) -> None:
        h = np.zeros((self.n, 4), np.float32)
        h[:, :3] = v
        self.v = cuda.to_device(h)

    def _download(self):
        return (
            self.x.copy_to_host()[:, :3].astype(np.float64),
            self.v.copy_to_host()[:, :3].astype(np.float64),
            self.ids.copy_to_host(),
        )

    def _download_forces(self):
        return self.f.copy_to_host()[:, :3].astype(np.float64), self.ids.copy_to_host()

    def mean_neighbors(self) -> float:
        return self._dsum(_sum_i32, self.nnb) / self.n

    def synchronize(self) -> None:
        cuda.synchronize()
