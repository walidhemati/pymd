"""NVIDIA Warp backend.

Three variants:

* ``full`` -- hand-written cell list (counting sort with ``wp.atomic_add`` + ``array_scan``,
  atoms permuted into cell order), Verlet list over the 27-cell stencil, full-list force
  kernel without atomics. Same algorithm as the CUDA C reference.
* ``hashgrid`` -- Warp's built-in :class:`warp.HashGrid`, rebuilt every step, forces computed
  directly from grid queries (no Verlet list).
* ``hashgrid-verlet`` -- ``HashGrid`` used only to build the Verlet list at each rebuild, then
  the same full-list force kernel as ``full``.

Periodic boundaries with ``HashGrid``: Warp maps a world cell ``(cx, cy, cz)`` to the physical
cell ``(cx + 2^20) % dim`` per axis. Choosing ``dim = floor(L / r)`` and cell width ``L / dim``
makes cells ``c`` and ``c + dim`` coincide, i.e. the hash wrap *is* the periodic wrap, and a
query near a face returns atoms from the opposite face; minimum image then fixes the distance.
This needs a single cell width for all axes, so the hash-grid variants require a cubic box.
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp

    wp.config.quiet = True  # older Warp releases
    wp.config.log_level = getattr(wp, "LOG_WARNING", 30)  # Warp >= 1.10: silence init banner
except Exception:  # pragma: no cover - import guard
    wp = None

from ..engine import Engine
from ..reference import cell_grid, morton3

DEVICE = "cuda:0"


def is_available() -> bool:
    """Warp importable and a CUDA device present."""
    if wp is None:
        return False
    try:
        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


if wp is not None:

    @wp.func
    def _cell_of(x: float, il: float, nc: int) -> int:
        c = int(wp.floor(x * il * float(nc)))
        return wp.clamp(c, 0, nc - 1)

    @wp.func
    def _spread10(v: int) -> int:
        v = v & 0x3FF
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    @wp.func
    def _cell_key(cx: int, cy: int, cz: int, nc: wp.vec3i, morton: int) -> int:
        if morton != 0:
            return _spread10(cx) | (_spread10(cy) << 1) | (_spread10(cz) << 2)
        return (cx * nc[1] + cy) * nc[2] + cz

    @wp.func
    def _min_image(d: wp.vec3, box: wp.vec3) -> wp.vec3:
        return wp.vec3(
            d[0] - box[0] * wp.round(d[0] / box[0]),
            d[1] - box[1] * wp.round(d[1] / box[1]),
            d[2] - box[2] * wp.round(d[2] / box[2]),
        )

    @wp.kernel
    def _wrap(x: wp.array(dtype=wp.vec3), box: wp.vec3):
        i = wp.tid()
        p = x[i]
        x[i] = wp.vec3(
            p[0] - box[0] * wp.floor(p[0] / box[0]),
            p[1] - box[1] * wp.floor(p[1] / box[1]),
            p[2] - box[2] * wp.floor(p[2] / box[2]),
        )

    @wp.kernel
    def _key_hist(
        x: wp.array(dtype=wp.vec3),
        box: wp.vec3,
        nc: wp.vec3i,
        morton: int,
        key: wp.array(dtype=int),
        cnt: wp.array(dtype=int),
    ):
        i = wp.tid()
        p = x[i]
        cx = _cell_of(p[0], 1.0 / box[0], nc[0])
        cy = _cell_of(p[1], 1.0 / box[1], nc[1])
        cz = _cell_of(p[2], 1.0 / box[2], nc[2])
        k = _cell_key(cx, cy, cz, nc, morton)
        key[i] = k
        wp.atomic_add(cnt, k, 1)

    @wp.kernel
    def _scatter(key: wp.array(dtype=int), cursor: wp.array(dtype=int), order: wp.array(dtype=int)):
        i = wp.tid()
        slot = wp.atomic_add(cursor, key[i], 1)
        order[slot] = i

    @wp.kernel
    def _permute(
        order: wp.array(dtype=int),
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        ids: wp.array(dtype=int),
        x2: wp.array(dtype=wp.vec3),
        v2: wp.array(dtype=wp.vec3),
        ids2: wp.array(dtype=int),
    ):
        k = wp.tid()
        i = order[k]
        x2[k] = x[i]
        v2[k] = v[i]
        ids2[k] = ids[i]

    @wp.kernel
    def _build_nlist(
        x: wp.array(dtype=wp.vec3),
        n: int,
        cell_start: wp.array(dtype=int),
        order: wp.array(dtype=int),
        sorted_flag: int,
        box: wp.vec3,
        nc: wp.vec3i,
        morton: int,
        rlist2: float,
        maxnb: int,
        nl: wp.array(dtype=int),
        nnb: wp.array(dtype=int),
        overflow: wp.array(dtype=int),
    ):
        i = wp.tid()
        p = x[i]
        cx = _cell_of(p[0], 1.0 / box[0], nc[0])
        cy = _cell_of(p[1], 1.0 / box[1], nc[1])
        cz = _cell_of(p[2], 1.0 / box[2], nc[2])
        rx = int(0)  # noqa: UP018 - Warp needs typed mutable locals
        ry = int(0)  # noqa: UP018 - Warp needs typed mutable locals
        rz = int(0)  # noqa: UP018 - Warp needs typed mutable locals
        if nc[0] >= 3:
            rx = 1
        if nc[1] >= 3:
            ry = 1
        if nc[2] >= 3:
            rz = 1
        cnt = int(0)  # noqa: UP018 - Warp needs typed mutable locals
        for dx in range(-rx, rx + 1):
            ax = (cx + dx + nc[0]) % nc[0]
            for dy in range(-ry, ry + 1):
                ay = (cy + dy + nc[1]) % nc[1]
                for dz in range(-rz, rz + 1):
                    az = (cz + dz + nc[2]) % nc[2]
                    c = _cell_key(ax, ay, az, nc, morton)
                    for k in range(cell_start[c], cell_start[c + 1]):
                        j = k
                        if sorted_flag == 0:
                            j = order[k]
                        if j != i:
                            d = _min_image(p - x[j], box)
                            if wp.dot(d, d) < rlist2:
                                if cnt < maxnb:
                                    nl[cnt * n + i] = j
                                cnt += 1
        nnb[i] = cnt
        if cnt > maxnb:
            wp.atomic_max(overflow, 0, cnt)

    @wp.kernel
    def _build_nlist_hash(
        grid: wp.uint64,
        x: wp.array(dtype=wp.vec3),
        n: int,
        box: wp.vec3,
        rlist: float,
        maxnb: int,
        nl: wp.array(dtype=int),
        nnb: wp.array(dtype=int),
        overflow: wp.array(dtype=int),
    ):
        i = wp.tid()
        p = x[i]
        rlist2 = rlist * rlist
        cnt = int(0)  # noqa: UP018 - Warp needs typed mutable locals
        query = wp.hash_grid_query(grid, p, rlist)
        j = int(0)  # noqa: UP018 - Warp needs typed mutable locals
        while wp.hash_grid_query_next(query, j):
            if j != i:
                d = _min_image(p - x[j], box)
                if wp.dot(d, d) < rlist2:
                    if cnt < maxnb:
                        nl[cnt * n + i] = j
                    cnt += 1
        nnb[i] = cnt
        if cnt > maxnb:
            wp.atomic_max(overflow, 0, cnt)

    @wp.kernel
    def _force_full(
        x: wp.array(dtype=wp.vec3),
        n: int,
        nl: wp.array(dtype=int),
        nnb: wp.array(dtype=int),
        box: wp.vec3,
        rc2: float,
        lj1: float,
        lj2: float,
        lj3: float,
        lj4: float,
        eshift: float,
        energy: int,
        f: wp.array(dtype=wp.vec3),
        e_at: wp.array(dtype=float),
        w_at: wp.array(dtype=float),
    ):
        i = wp.tid()
        p = x[i]
        fi = wp.vec3(0.0, 0.0, 0.0)
        e = float(0.0)  # noqa: UP018 - Warp needs typed mutable locals
        w = float(0.0)  # noqa: UP018 - Warp needs typed mutable locals
        for k in range(nnb[i]):
            j = nl[k * n + i]
            d = _min_image(p - x[j], box)
            r2 = wp.dot(d, d)
            if r2 < rc2:
                r2i = 1.0 / r2
                r6i = r2i * r2i * r2i
                ff = r6i * (lj1 * r6i - lj2) * r2i
                fi = fi + ff * d
                if energy != 0:
                    e += r6i * (lj3 * r6i - lj4) - eshift
                    w += ff * r2
        f[i] = fi
        if energy != 0:
            e_at[i] = 0.5 * e
            w_at[i] = 0.5 * w

    @wp.kernel
    def _force_hash(
        grid: wp.uint64,
        x: wp.array(dtype=wp.vec3),
        box: wp.vec3,
        rc: float,
        lj1: float,
        lj2: float,
        lj3: float,
        lj4: float,
        eshift: float,
        energy: int,
        f: wp.array(dtype=wp.vec3),
        e_at: wp.array(dtype=float),
        w_at: wp.array(dtype=float),
    ):
        i = wp.tid()
        p = x[i]
        rc2 = rc * rc
        fi = wp.vec3(0.0, 0.0, 0.0)
        e = float(0.0)  # noqa: UP018 - Warp needs typed mutable locals
        w = float(0.0)  # noqa: UP018 - Warp needs typed mutable locals
        query = wp.hash_grid_query(grid, p, rc)
        j = int(0)  # noqa: UP018 - Warp needs typed mutable locals
        while wp.hash_grid_query_next(query, j):
            if j != i:
                d = _min_image(p - x[j], box)
                r2 = wp.dot(d, d)
                if r2 < rc2:
                    r2i = 1.0 / r2
                    r6i = r2i * r2i * r2i
                    ff = r6i * (lj1 * r6i - lj2) * r2i
                    fi = fi + ff * d
                    if energy != 0:
                        e += r6i * (lj3 * r6i - lj4) - eshift
                        w += ff * r2
        f[i] = fi
        if energy != 0:
            e_at[i] = 0.5 * e
            w_at[i] = 0.5 * w

    @wp.kernel
    def _kick_drift(
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        f: wp.array(dtype=wp.vec3),
        kick: float,
        drift: float,
    ):
        i = wp.tid()
        vi = v[i] + kick * f[i]
        v[i] = vi
        if drift != 0.0:
            x[i] = x[i] + drift * vi

    @wp.kernel
    def _disp2_max(
        x: wp.array(dtype=wp.vec3), x_ref: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)
    ):
        i = wp.tid()
        d = x[i] - x_ref[i]
        wp.atomic_max(out, 0, wp.dot(d, d))

    @wp.kernel
    def _ke_at(v: wp.array(dtype=wp.vec3), half_m: float, out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        vi = v[i]
        out[i] = wp.float64(half_m * wp.dot(vi, vi))

    @wp.kernel
    def _scale(v: wp.array(dtype=wp.vec3), lam: float):
        i = wp.tid()
        v[i] = lam * v[i]

    @wp.kernel
    def _ou_kernel(v: wp.array(dtype=wp.vec3), c1: float, c2: float, seed: int):
        i = wp.tid()
        st = wp.rand_init(seed, i)
        xi = wp.vec3(wp.randn(st), wp.randn(st), wp.randn(st))
        v[i] = c1 * v[i] + c2 * xi


class WarpEngine(Engine):
    """NVIDIA Warp kernels (fp32, ``wp.vec3`` arrays)."""

    backend = "warp"
    variants = {
        "full": "fp32 full Verlet list, hand-written counting-sort cell list, no atomics",
        "hashgrid": "fp32, Warp's built-in HashGrid rebuilt every step, no Verlet list",
        "hashgrid-verlet": "fp32 full Verlet list built from Warp's HashGrid queries",
    }

    def _setup(self, pos: np.ndarray, vel: np.ndarray) -> None:
        wp.init()
        self.device = wp.get_device(DEVICE)
        v = self.variant
        self.uses_list = v != "hashgrid"
        self.use_hash = v.startswith("hashgrid")
        n = self.n
        if self.use_hash:
            if not np.allclose(self.box, self.box[0]):
                raise ValueError("warp hashgrid variants need a cubic box")
            r = self.rlist if v == "hashgrid-verlet" else self.params.rc
            self.hdim = int(np.floor(self.box[0] / r))
            if self.hdim < 3:
                raise ValueError("warp hashgrid variants need at least 3 cells per side")
            self.hcell = float(self.box[0] / self.hdim)
            self.hquery = float(r)
        self.sort = self.cfg.sort
        self.ncell = cell_grid(self.box, self.rlist)
        if self.sort == "morton":
            if self.ncell.max() > 1024:
                raise ValueError("Morton keys support at most 1024 cells per dimension")
            nc = self.ncell - 1
            self.nkeys = int(morton3(nc[:1], nc[1:2], nc[2:3])[0]) + 1
        else:
            self.nkeys = int(np.prod(self.ncell))
        d = self.device
        with wp.ScopedDevice(d):
            self.x = wp.array(pos.astype(np.float32), dtype=wp.vec3)
            self.v = wp.array(vel.astype(np.float32), dtype=wp.vec3)
            self.f = wp.zeros(n, dtype=wp.vec3)
            self.ids = wp.array(np.arange(n, dtype=np.int32), dtype=int)
            self.x_ref = wp.clone(self.x)
            self.e_at = wp.zeros(n, dtype=float)
            self.w_at = wp.zeros(n, dtype=float)
            self.ke = wp.zeros(n, dtype=wp.float64)
            self.dmax = wp.zeros(1, dtype=float)
            self.key = wp.zeros(n, dtype=int)
            self.counts = wp.zeros(self.nkeys + 1, dtype=int)
            self.cell_start = wp.zeros(self.nkeys + 1, dtype=int)
            self.cursor = wp.zeros(self.nkeys + 1, dtype=int)
            self.order = wp.zeros(n, dtype=int)
            self.nnb = wp.zeros(n, dtype=int)
            self.overflow = wp.zeros(1, dtype=int)
            self.x2 = wp.empty_like(self.x)
            self.v2 = wp.empty_like(self.v)
            self.ids2 = wp.empty_like(self.ids)
            if self.use_hash:
                self.grid = wp.HashGrid(self.hdim, self.hdim, self.hdim, device=d)
        dens = n / np.prod(self.box)
        est = dens * 4.0 / 3.0 * np.pi * self.rlist**3
        self.maxnb = int(np.ceil(est * 1.3 / 8) * 8) + 8
        if self.uses_list:
            self.nl = wp.zeros(self.maxnb * n, dtype=int, device=d)
        self.boxv = wp.vec3(*[float(b) for b in self.box])
        self.ncv = wp.vec3i(*[int(c) for c in self.ncell])
        s6 = self.params.sigma**6
        eps = self.params.epsilon
        self.lj = (48.0 * eps * s6 * s6, 24.0 * eps * s6, 4.0 * eps * s6 * s6, 4.0 * eps * s6,
                   float(self.params.eshift))
        self._hash_calls = 0
        self._seed = 0

    # ------------------------------------------------------------- neighbor search
    def _launch(self, kernel, dim, inputs) -> None:
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def _sort_atoms(self) -> None:
        """Wrap into the box and (unless ``sort == 'none'``) permute atoms into cell order."""
        n = self.n
        self._launch(_wrap, n, [self.x, self.boxv])
        self.counts.zero_()
        morton = int(self.sort == "morton")
        self._launch(_key_hist, n, [self.x, self.boxv, self.ncv, morton, self.key, self.counts])
        wp.utils.array_scan(self.counts, self.cell_start, inclusive=False)
        wp.copy(self.cursor, self.cell_start)
        self._launch(_scatter, n, [self.key, self.cursor, self.order])
        if self.sort != "none":
            self._launch(_permute, n, [self.order, self.x, self.v, self.ids, self.x2, self.v2,
                                       self.ids2])
            self.x, self.x2 = self.x2, self.x
            self.v, self.v2 = self.v2, self.v
            self.ids, self.ids2 = self.ids2, self.ids

    def _rebuild(self) -> None:
        n = self.n
        self._sort_atoms()
        if self.use_hash:
            self.grid.build(self.x, self.hcell)
        rl2 = float(self.rlist**2)
        while True:
            self.overflow.zero_()
            if self.use_hash:
                self._launch(_build_nlist_hash, n, [self.grid.id, self.x, n, self.boxv,
                                                    self.hquery, self.maxnb, self.nl, self.nnb,
                                                    self.overflow])
            else:
                self._launch(_build_nlist, n, [self.x, n, self.cell_start, self.order,
                                               int(self.sort != "none"), self.boxv, self.ncv,
                                               int(self.sort == "morton"), rl2, self.maxnb,
                                               self.nl, self.nnb, self.overflow])
            ov = int(self.overflow.numpy()[0])
            if ov == 0:
                break
            self.maxnb = int(np.ceil(ov * 1.2 / 8) * 8)
            self.nl = wp.zeros(self.maxnb * n, dtype=int, device=self.device)
        wp.copy(self.x_ref, self.x)
        self._forces_valid = False

    # ------------------------------------------------------------- forces / integration
    def _compute_forces(self, energy: bool = False):
        n = self.n
        lj = self.lj
        if self.variant == "hashgrid":
            every = max(self.cfg.rebuild_every, 1)
            if self._hash_calls % every == 0:
                self._sort_atoms()  # periodic wrap + spatial sort, as in the list variants
            self._hash_calls += 1
            self.grid.build(self.x, self.hcell)
            self._launch(_force_hash, n, [self.grid.id, self.x, self.boxv, self.hquery, *lj,
                                          int(energy), self.f, self.e_at, self.w_at])
        else:
            self._launch(_force_full, n, [self.x, n, self.nl, self.nnb, self.boxv,
                                          float(self.params.rc**2), *lj, int(energy), self.f,
                                          self.e_at, self.w_at])
        if energy:
            e = float(self.e_at.numpy().sum(dtype=np.float64))
            w = float(self.w_at.numpy().sum(dtype=np.float64))
            return e, w
        return None

    def _kick_drift(self, kick: float, drift: float) -> None:
        self._launch(_kick_drift, self.n, [self.x, self.v, self.f, float(kick), float(drift)])

    def _max_disp2(self) -> float:
        self.dmax.zero_()
        self._launch(_disp2_max, self.n, [self.x, self.x_ref, self.dmax])
        return float(self.dmax.numpy()[0])

    def _ekin(self) -> float:
        self._launch(_ke_at, self.n, [self.v, 0.5 * self.cfg.mass, self.ke])
        return float(wp.utils.array_sum(self.ke))

    def _scale_v(self, lam: float) -> None:
        self._launch(_scale, self.n, [self.v, float(lam)])

    def _ou(self, c1: float, c2: float, rng) -> None:
        self._seed = int(rng.integers(2**31 - 1))
        self._launch(_ou_kernel, self.n, [self.v, float(c1), float(c2), self._seed])

    def _get_v_raw(self) -> np.ndarray:
        return self.v.numpy().astype(np.float64)

    def _set_v_raw(self, v: np.ndarray) -> None:
        self.v = wp.array(v.astype(np.float32), dtype=wp.vec3, device=self.device)

    def _download(self):
        return (
            self.x.numpy().astype(np.float64),
            self.v.numpy().astype(np.float64),
            self.ids.numpy(),
        )

    def _download_forces(self):
        return self.f.numpy().astype(np.float64), self.ids.numpy()

    def mean_neighbors(self) -> float:
        if not self.uses_list:
            return float("nan")
        return float(self.nnb.numpy().sum(dtype=np.float64)) / self.n

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)
