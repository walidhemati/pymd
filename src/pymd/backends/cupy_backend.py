"""Hand-written CUDA C reference backend, compiled at runtime with NVRTC via CuPy ``RawModule``.

This is the "what would a CUDA programmer write" rung of the ladder. Variants cover the
classic GPU-MD design space: full vs half (atomic) neighbor lists, atom sorting (none, cell
order, Morton order), precision (fp32, mixed fp32-pair/fp64-accumulate, fp64), a fused
one-kernel-per-step leapfrog, and a shared-memory tiled all-pairs baseline.
"""

from __future__ import annotations

import numpy as np

from .._cuda_env import preload_nvrtc

try:
    import warnings

    preload_nvrtc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp
except Exception:  # pragma: no cover - import guard
    cp = None

from ..engine import Engine
from ..reference import cell_grid

BLOCK = 128

CUDA_SRC = r"""
typedef REAL real;
typedef PREAL preal;
struct __align__(4 * sizeof(REAL)) real4_t { real x, y, z, w; };

struct Box { real lx, ly, lz, ilx, ily, ilz; };

__device__ __forceinline__ unsigned int spread10(unsigned int v) {
    v &= 0x3FF;
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

__device__ __forceinline__ int cell_key(int cx, int cy, int cz, int ncx, int ncy, int ncz,
                                        int morton) {
    if (morton) return (int)(spread10(cx) | (spread10(cy) << 1) | (spread10(cz) << 2));
    return (cx * ncy + cy) * ncz + cz;
}

__device__ __forceinline__ int cell_of(real x, real il, int nc) {
    int c = (int)floor(x * il * nc);
    return c < 0 ? 0 : (c >= nc ? nc - 1 : c);
}

extern "C" __global__ void wrap_key(real4_t* __restrict__ x, int n, real lx, real ly, real lz,
                                    int ncx, int ncy, int ncz, int morton,
                                    int* __restrict__ key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real4_t p = x[i];
    p.x -= lx * floor(p.x / lx);
    p.y -= ly * floor(p.y / ly);
    p.z -= lz * floor(p.z / lz);
    x[i] = p;
    int cx = cell_of(p.x, (real)1 / lx, ncx);
    int cy = cell_of(p.y, (real)1 / ly, ncy);
    int cz = cell_of(p.z, (real)1 / lz, ncz);
    key[i] = cell_key(cx, cy, cz, ncx, ncy, ncz, morton);
}

extern "C" __global__ void histogram(const int* __restrict__ key, int n, int* __restrict__ cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&cnt[key[i]], 1);
}

// counting-sort scatter: cursor starts as the exclusive prefix sum of the histogram
extern "C" __global__ void scatter(const int* __restrict__ key, int n, int* __restrict__ cursor,
                                   int* __restrict__ order) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) order[atomicAdd(&cursor[key[i]], 1)] = i;
}

extern "C" __global__ void permute(const int* __restrict__ order, int n,
                                   const real4_t* __restrict__ x, const real4_t* __restrict__ v,
                                   const int* __restrict__ ids, real4_t* __restrict__ x2,
                                   real4_t* __restrict__ v2, int* __restrict__ ids2) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int i = order[k];
    x2[k] = x[i];
    v2[k] = v[i];
    ids2[k] = ids[i];
}

// One thread per atom: scan the 27 stencil cells, keep j with |r_ij| < rlist.
// List layout is neighbor-major: nl[k * n + i] (coalesced for one-thread-per-atom kernels).
extern "C" __global__ void build_nlist(const real4_t* __restrict__ x, int n,
                                       const int* __restrict__ cell_start,
                                       const int* __restrict__ cell_atoms, int sorted,
                                       real lx, real ly, real lz, int ncx, int ncy, int ncz,
                                       int morton, real rlist2, int half, int maxnb,
                                       int* __restrict__ nl, int* __restrict__ nnb,
                                       int* __restrict__ overflow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real4_t p = x[i];
    int cx = cell_of(p.x, (real)1 / lx, ncx);
    int cy = cell_of(p.y, (real)1 / ly, ncy);
    int cz = cell_of(p.z, (real)1 / lz, ncz);
    int rx = ncx >= 3 ? 1 : 0, ry = ncy >= 3 ? 1 : 0, rz = ncz >= 3 ? 1 : 0;
    int cnt = 0;
    for (int dx = -rx; dx <= rx; ++dx) {
        int ax = (cx + dx + ncx) % ncx;
        for (int dy = -ry; dy <= ry; ++dy) {
            int ay = (cy + dy + ncy) % ncy;
            for (int dz = -rz; dz <= rz; ++dz) {
                int az = (cz + dz + ncz) % ncz;
                int c = cell_key(ax, ay, az, ncx, ncy, ncz, morton);
                int s = cell_start[c], e = cell_start[c + 1];
                for (int k = s; k < e; ++k) {
                    int j = sorted ? k : cell_atoms[k];
                    if (j == i || (half && j < i)) continue;
                    real4_t q = x[j];
                    real ddx = p.x - q.x, ddy = p.y - q.y, ddz = p.z - q.z;
                    ddx -= lx * rint(ddx / lx);
                    ddy -= ly * rint(ddy / ly);
                    ddz -= lz * rint(ddz / lz);
                    if (ddx * ddx + ddy * ddy + ddz * ddz < rlist2) {
                        if (cnt < maxnb) nl[cnt * n + i] = j;
                        ++cnt;
                    }
                }
            }
        }
    }
    nnb[i] = cnt;
    if (cnt > maxnb) atomicMax(overflow, cnt);
}

#define LJ_ARGS preal rc2, preal lj1, preal lj2, preal lj3, preal lj4, preal eshift

// Full neighbor list: every pair is evaluated twice, no atomics, one store per atom.
template <bool ENERGY>
__global__ void force_full(const real4_t* __restrict__ x, int n, const int* __restrict__ nl,
                           const int* __restrict__ nnb, real lx, real ly, real lz, LJ_ARGS,
                           real4_t* __restrict__ f, real* __restrict__ epair,
                           real* __restrict__ wpair) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real4_t p = x[i];
    real ilx = (real)1 / lx, ily = (real)1 / ly, ilz = (real)1 / lz;
    real fx = 0, fy = 0, fz = 0, e = 0, w = 0;
    int m = nnb[i];
    for (int k = 0; k < m; ++k) {
        int j = __ldg(&nl[k * n + i]);
        real4_t q = x[j];
        real ddx = p.x - q.x, ddy = p.y - q.y, ddz = p.z - q.z;
        ddx -= lx * rint(ddx * ilx);
        ddy -= ly * rint(ddy * ily);
        ddz -= lz * rint(ddz * ilz);
        preal dx = (preal)ddx, dy = (preal)ddy, dz = (preal)ddz;
        preal r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2) {
            preal r2i = (preal)1 / r2;
            preal r6i = r2i * r2i * r2i;
            preal ff = r6i * (lj1 * r6i - lj2) * r2i;
            fx += (real)(ff * dx);
            fy += (real)(ff * dy);
            fz += (real)(ff * dz);
            if (ENERGY) {
                e += (real)(r6i * (lj3 * r6i - lj4) - eshift);
                w += (real)(ff * r2);
            }
        }
    }
    real4_t out; out.x = fx; out.y = fy; out.z = fz; out.w = 0;
    f[i] = out;
    if (ENERGY) { epair[i] = (real)0.5 * e; wpair[i] = (real)0.5 * w; }
}

// Half neighbor list: each pair once, Newton's third law via atomicAdd on f[j].
template <bool ENERGY>
__global__ void force_half(const real4_t* __restrict__ x, int n, const int* __restrict__ nl,
                           const int* __restrict__ nnb, real lx, real ly, real lz, LJ_ARGS,
                           real4_t* __restrict__ f, real* __restrict__ epair,
                           real* __restrict__ wpair) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real4_t p = x[i];
    real ilx = (real)1 / lx, ily = (real)1 / ly, ilz = (real)1 / lz;
    real fx = 0, fy = 0, fz = 0, e = 0, w = 0;
    int m = nnb[i];
    for (int k = 0; k < m; ++k) {
        int j = __ldg(&nl[k * n + i]);
        real4_t q = x[j];
        real ddx = p.x - q.x, ddy = p.y - q.y, ddz = p.z - q.z;
        ddx -= lx * rint(ddx * ilx);
        ddy -= ly * rint(ddy * ily);
        ddz -= lz * rint(ddz * ilz);
        preal dx = (preal)ddx, dy = (preal)ddy, dz = (preal)ddz;
        preal r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2) {
            preal r2i = (preal)1 / r2;
            preal r6i = r2i * r2i * r2i;
            preal ff = r6i * (lj1 * r6i - lj2) * r2i;
            real gx = (real)(ff * dx), gy = (real)(ff * dy), gz = (real)(ff * dz);
            fx += gx; fy += gy; fz += gz;
            atomicAdd(&f[j].x, -gx);
            atomicAdd(&f[j].y, -gy);
            atomicAdd(&f[j].z, -gz);
            if (ENERGY) {
                e += (real)(r6i * (lj3 * r6i - lj4) - eshift);
                w += (real)(ff * r2);
            }
        }
    }
    atomicAdd(&f[i].x, fx);
    atomicAdd(&f[i].y, fy);
    atomicAdd(&f[i].z, fz);
    if (ENERGY) { epair[i] = e; wpair[i] = w; }
}

// Fused leapfrog: force at x, v += dt*f/m, x_out = x + dt*v. One launch per time step.
extern "C" __global__ void fused_step(const real4_t* __restrict__ x, int n,
                                      const int* __restrict__ nl, const int* __restrict__ nnb,
                                      real lx, real ly, real lz, LJ_ARGS, real dtm, real dt,
                                      real4_t* __restrict__ v, real4_t* __restrict__ xout) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real4_t p = x[i];
    real ilx = (real)1 / lx, ily = (real)1 / ly, ilz = (real)1 / lz;
    real fx = 0, fy = 0, fz = 0;
    int m = nnb[i];
    for (int k = 0; k < m; ++k) {
        int j = __ldg(&nl[k * n + i]);
        real4_t q = x[j];
        real ddx = p.x - q.x, ddy = p.y - q.y, ddz = p.z - q.z;
        ddx -= lx * rint(ddx * ilx);
        ddy -= ly * rint(ddy * ily);
        ddz -= lz * rint(ddz * ilz);
        preal dx = (preal)ddx, dy = (preal)ddy, dz = (preal)ddz;
        preal r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2) {
            preal r2i = (preal)1 / r2;
            preal r6i = r2i * r2i * r2i;
            preal ff = r6i * (lj1 * r6i - lj2) * r2i;
            fx += (real)(ff * dx);
            fy += (real)(ff * dy);
            fz += (real)(ff * dz);
        }
    }
    real4_t vi = v[i];
    vi.x += dtm * fx; vi.y += dtm * fy; vi.z += dtm * fz;
    v[i] = vi;
    p.x += dt * vi.x; p.y += dt * vi.y; p.z += dt * vi.z;
    xout[i] = p;
}

// Shared-memory tiled O(N^2) baseline (GPU Gems 3 n-body style) with cutoff + minimum image.
template <bool ENERGY>
__global__ void force_allpairs(const real4_t* __restrict__ x, int n, real lx, real ly, real lz,
                               LJ_ARGS, real4_t* __restrict__ f, real* __restrict__ epair,
                               real* __restrict__ wpair) {
    __shared__ real4_t tile[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    real4_t p = i < n ? x[i] : x[0];
    real ilx = (real)1 / lx, ily = (real)1 / ly, ilz = (real)1 / lz;
    real fx = 0, fy = 0, fz = 0, e = 0, w = 0;
    for (int t = 0; t < n; t += BLOCK) {
        int jl = t + threadIdx.x;
        if (jl < n) tile[threadIdx.x] = x[jl];
        __syncthreads();
        int m = min(BLOCK, n - t);
        for (int k = 0; k < m; ++k) {
            real4_t q = tile[k];
            real ddx = p.x - q.x, ddy = p.y - q.y, ddz = p.z - q.z;
            ddx -= lx * rint(ddx * ilx);
            ddy -= ly * rint(ddy * ily);
            ddz -= lz * rint(ddz * ilz);
            preal dx = (preal)ddx, dy = (preal)ddy, dz = (preal)ddz;
            preal r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < rc2 && t + k != i) {
                preal r2i = (preal)1 / r2;
                preal r6i = r2i * r2i * r2i;
                preal ff = r6i * (lj1 * r6i - lj2) * r2i;
                fx += (real)(ff * dx);
                fy += (real)(ff * dy);
                fz += (real)(ff * dz);
                if (ENERGY) {
                    e += (real)(r6i * (lj3 * r6i - lj4) - eshift);
                    w += (real)(ff * r2);
                }
            }
        }
        __syncthreads();
    }
    if (i >= n) return;
    real4_t out; out.x = fx; out.y = fy; out.z = fz; out.w = 0;
    f[i] = out;
    if (ENERGY) { epair[i] = (real)0.5 * e; wpair[i] = (real)0.5 * w; }
}

extern "C" __global__ void kick_drift(real4_t* __restrict__ x, real4_t* __restrict__ v,
                                      const real4_t* __restrict__ f, int n, real kick,
                                      real drift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real4_t vi = v[i], fi = f[i], p = x[i];
    vi.x += kick * fi.x; vi.y += kick * fi.y; vi.z += kick * fi.z;
    v[i] = vi;
    if (drift != (real)0) {
        p.x += drift * vi.x; p.y += drift * vi.y; p.z += drift * vi.z;
        x[i] = p;
    }
}
"""

_TEMPLATED = [
    "force_full<true>",
    "force_full<false>",
    "force_half<true>",
    "force_half<false>",
    "force_allpairs<true>",
    "force_allpairs<false>",
]
_PLAIN = ["wrap_key", "histogram", "scatter", "permute", "build_nlist", "fused_step", "kick_drift"]

_PRECISION = {  # name -> (storage/accumulation type, pair-math type)
    "fp32": ("float", "float"),
    "mixed": ("double", "float"),
    "fp64": ("double", "double"),
}
_modules: dict[str, object] = {}


def is_available() -> bool:
    """CuPy importable and a CUDA device present."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _module(prec: str):
    if prec not in _modules:
        real, preal = _PRECISION[prec]
        mod = cp.RawModule(
            code=CUDA_SRC,
            options=(f"-DREAL={real}", f"-DPREAL={preal}", f"-DBLOCK={BLOCK}", "-std=c++14"),
            name_expressions=_TEMPLATED,
        )
        _modules[prec] = mod
    return _modules[prec]


def _lj_args(params, preal_np) -> tuple:
    """LJ coefficients (rc^2, 48 e s^12, 24 e s^6, 4 e s^12, 4 e s^6, shift) as kernel scalars."""
    s6 = params.sigma**6
    vals = [
        params.rc**2,
        48.0 * params.epsilon * s6 * s6,
        24.0 * params.epsilon * s6,
        4.0 * params.epsilon * s6 * s6,
        4.0 * params.epsilon * s6,
        params.eshift,
    ]
    return tuple(preal_np(v) for v in vals)


class CupyEngine(Engine):
    """CUDA C kernels via CuPy/NVRTC (hand-written reference implementation)."""

    backend = "cupy-cuda-c"
    variants = {
        "full": "fp32, full Verlet list (no atomics), cell-sorted atoms",
        "full-fused": "fp32 full list, one fused force+kick+drift kernel per step (leapfrog)",
        "half-atomic": "fp32 half Verlet list, Newton's 3rd law with atomicAdd scatter",
        "full-morton": "fp32 full list, atoms sorted along a Morton (Z-order) curve of cells",
        "full-nosort": "fp32 full list, atoms never reordered (cell list by indirection)",
        "full-mixed": "fp32 pair math, fp64 positions/velocities/force accumulation",
        "full-fp64": "fp64 everywhere (consumer GPU fp64 rate is 1/64 of fp32)",
        "allpairs": "fp32 O(N^2) shared-memory tiled direct summation with cutoff",
    }

    def _setup(self, pos: np.ndarray, vel: np.ndarray) -> None:
        v = self.variant
        self.prec = "mixed" if v.endswith("mixed") else ("fp64" if v.endswith("fp64") else "fp32")
        self.dtype = cp.float32 if self.prec == "fp32" else cp.float64
        self.np_real = np.float32 if self.prec == "fp32" else np.float64
        self.np_preal = np.float64 if self.prec == "fp64" else np.float32
        self.uses_list = v != "allpairs"
        self.half = v == "half-atomic"
        self.supports_fused = v == "full-fused"
        self.sort = {"full-morton": "morton", "full-nosort": "none"}.get(v, self.cfg.sort)
        mod = _module(self.prec)
        self.k = {name: mod.get_function(name) for name in _PLAIN}
        for name in _TEMPLATED:
            self.k[name] = mod.get_function(name)
        self.lj = _lj_args(self.params, self.np_preal)
        n = self.n
        self.grid = ((n + BLOCK - 1) // BLOCK,)
        self.x = cp.zeros((n, 4), self.dtype)
        self.v = cp.zeros((n, 4), self.dtype)
        self.x[:, :3] = cp.asarray(pos)
        self.v[:, :3] = cp.asarray(vel)
        self.f = cp.zeros((n, 4), self.dtype)
        self.ids = cp.arange(n, dtype=cp.int32)
        self.e_at = cp.zeros(n, self.dtype)
        self.w_at = cp.zeros(n, self.dtype)
        self.x_ref = self.x.copy()
        self.ncell = cell_grid(self.box, self.rlist)
        if self.sort == "morton" and self.ncell.max() > 1024:
            raise ValueError("Morton keys support at most 1024 cells per dimension")
        if self.sort == "morton":
            from ..reference import morton3

            nc = self.ncell - 1
            nkeys = int(morton3(np.array([nc[0]]), np.array([nc[1]]), np.array([nc[2]]))[0]) + 1
        else:
            nkeys = int(np.prod(self.ncell))
        self.nkeys = nkeys
        self.key = cp.zeros(n, cp.int32)
        self.counts = cp.zeros(nkeys, cp.int32)
        self.cell_start = cp.zeros(nkeys + 1, cp.int32)
        self.order = cp.zeros(n, cp.int32)
        self.nnb = cp.zeros(n, cp.int32)
        self.overflow = cp.zeros(1, cp.int32)
        dens = n / np.prod(self.box)
        est = dens * 4.0 / 3.0 * np.pi * self.rlist**3
        self.maxnb = int(np.ceil(est * (0.65 if self.half else 1.3) / 8) * 8) + 8
        self.nl = cp.zeros(self.maxnb * n, cp.int32)
        if self.supports_fused:
            self.x_alt = cp.empty_like(self.x)
        self._box_args = tuple(self.np_real(b) for b in self.box)

    def _rebuild(self) -> None:
        n, g, b = self.n, self.grid, (BLOCK,)
        ncx, ncy, ncz = (np.int32(c) for c in self.ncell)
        morton = np.int32(self.sort == "morton")
        self.k["wrap_key"](g, b, (self.x, np.int32(n), *self._box_args, ncx, ncy, ncz, morton,
                                  self.key))
        self.counts.fill(0)
        self.k["histogram"](g, b, (self.key, np.int32(n), self.counts))
        self.cell_start[0] = 0
        cp.cumsum(self.counts, out=self.cell_start[1:])
        cursor = self.cell_start[:-1].copy()
        self.k["scatter"](g, b, (self.key, np.int32(n), cursor, self.order))
        sorted_flag = np.int32(0)
        if self.sort != "none":
            x2, v2, ids2 = cp.empty_like(self.x), cp.empty_like(self.v), cp.empty_like(self.ids)
            self.k["permute"](g, b, (self.order, np.int32(n), self.x, self.v, self.ids, x2, v2,
                                     ids2))
            self.x, self.v, self.ids = x2, v2, ids2
            sorted_flag = np.int32(1)
        rl2 = self.np_real(self.rlist**2)
        while True:
            self.overflow.fill(0)
            self.k["build_nlist"](g, b, (self.x, np.int32(n), self.cell_start, self.order,
                                         sorted_flag, *self._box_args, ncx, ncy, ncz, morton,
                                         rl2, np.int32(self.half), np.int32(self.maxnb), self.nl,
                                         self.nnb, self.overflow))
            ov = int(self.overflow.get()[0])
            if ov == 0:
                break
            self.maxnb = int(np.ceil(ov * 1.2 / 8) * 8)
            self.nl = cp.zeros(self.maxnb * n, cp.int32)
        self.x_ref = self.x.copy()
        self._forces_valid = False

    def _compute_forces(self, energy: bool = False):
        n, g, b = self.n, self.grid, (BLOCK,)
        tag = "true" if energy else "false"
        if self.variant == "allpairs":
            self.k[f"force_allpairs<{tag}>"](g, b, (self.x, np.int32(n), *self._box_args, *self.lj,
                                                     self.f, self.e_at, self.w_at))
        elif self.half:
            self.f.fill(0)
            self.k[f"force_half<{tag}>"](g, b, (self.x, np.int32(n), self.nl, self.nnb,
                                                 *self._box_args, *self.lj, self.f, self.e_at,
                                                 self.w_at))
        else:
            self.k[f"force_full<{tag}>"](g, b, (self.x, np.int32(n), self.nl, self.nnb,
                                                 *self._box_args, *self.lj, self.f, self.e_at,
                                                 self.w_at))
        if energy:
            return float(self.e_at.sum(dtype=cp.float64)), float(self.w_at.sum(dtype=cp.float64))
        return None

    def _kick_drift(self, kick: float, drift: float) -> None:
        self.k["kick_drift"](self.grid, (BLOCK,), (self.x, self.v, self.f, np.int32(self.n),
                                                   self.np_real(kick), self.np_real(drift)))

    def _fused_step(self, dt: float) -> None:
        im = 1.0 / self.cfg.mass
        self.k["fused_step"](self.grid, (BLOCK,), (self.x, np.int32(self.n), self.nl, self.nnb,
                                                   *self._box_args, *self.lj,
                                                   self.np_real(dt * im), self.np_real(dt),
                                                   self.v, self.x_alt))
        self.x, self.x_alt = self.x_alt, self.x

    def _max_disp2(self) -> float:
        d = self.x[:, :3] - self.x_ref[:, :3]
        return float(cp.max(cp.sum(d * d, axis=1)))

    def _ekin(self) -> float:
        return 0.5 * self.cfg.mass * float(cp.sum(self.v[:, :3] ** 2, dtype=cp.float64))

    def _scale_v(self, lam: float) -> None:
        self.v *= self.dtype(lam)

    def _ou(self, c1: float, c2: float, rng) -> None:
        if not hasattr(self, "_crng"):
            self._crng = cp.random.default_rng(int(rng.integers(2**31)))
        xi = self._crng.standard_normal((self.n, 3), dtype=self.dtype)
        self.v[:, :3] = self.dtype(c1) * self.v[:, :3] + self.dtype(c2) * xi

    def _download(self):
        return (
            cp.asnumpy(self.x[:, :3]).astype(np.float64),
            cp.asnumpy(self.v[:, :3]).astype(np.float64),
            cp.asnumpy(self.ids),
        )

    def _download_forces(self):
        return cp.asnumpy(self.f[:, :3]).astype(np.float64), cp.asnumpy(self.ids)

    def mean_neighbors(self) -> float:
        if not self.uses_list:
            return float("nan")
        return float(self.nnb.sum(dtype=cp.float64)) / self.n

    def synchronize(self) -> None:
        cp.cuda.Device().synchronize()
