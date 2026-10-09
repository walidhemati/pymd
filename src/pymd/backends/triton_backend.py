"""Triton backend: the hot path (neighbor-list build, forces, integration) as Triton kernels.

State lives in PyTorch CUDA tensors (positions/velocities/forces as padded ``(N, 4)`` float32).
The per-step work is entirely Triton: the force kernel (full list, or half list with
``tl.atomic_add`` scatter), the kick/drift kernel and a fused leapfrog kernel that computes
forces, updates velocities and writes the new positions to a second buffer in one launch.
The neighbor-list build is a Triton kernel too (one lane per atom, 27-cell stencil, per-lane
counters, masked stores into the neighbor-major list ``nl[k * N + i]``).

Rebuild-time glue -- ``torch.sort`` of the int32 cell keys (computed by a Triton kernel that
also wraps positions), ``torch.searchsorted`` for the cell offsets and the ``index_select``
permutation gather -- uses PyTorch ops. It runs once
per neighbor-list rebuild (every 20 steps in the benchmark), not every step.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover - import guard
    torch = None
    triton = None

from ..engine import Engine
from ..reference import cell_grid, morton3

BLOCK_FORCE = 128
WARPS_FORCE = 4
BLOCK_BUILD = 128
WARPS_BUILD = 4
BLOCK_EW = 1024


def is_available() -> bool:
    """Triton importable and a CUDA device present."""
    return torch is not None and triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _spread10(v):
        v = v & 0x3FF
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    @triton.jit
    def _wrap_key_kernel(x_ptr, c_ptr, key_ptr, n, lx, ly, lz, ncx, ncy, ncz,
                         MORTON: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        px = tl.load(x_ptr + offs * 4 + 0, mask=mask, other=0.0)
        py = tl.load(x_ptr + offs * 4 + 1, mask=mask, other=0.0)
        pz = tl.load(x_ptr + offs * 4 + 2, mask=mask, other=0.0)
        px -= lx * tl.floor(px / lx)
        py -= ly * tl.floor(py / ly)
        pz -= lz * tl.floor(pz / lz)
        tl.store(x_ptr + offs * 4 + 0, px, mask=mask)
        tl.store(x_ptr + offs * 4 + 1, py, mask=mask)
        tl.store(x_ptr + offs * 4 + 2, pz, mask=mask)
        cx = tl.minimum(tl.maximum(tl.floor(px * (1.0 / lx) * ncx).to(tl.int32), 0), ncx - 1)
        cy = tl.minimum(tl.maximum(tl.floor(py * (1.0 / ly) * ncy).to(tl.int32), 0), ncy - 1)
        cz = tl.minimum(tl.maximum(tl.floor(pz * (1.0 / lz) * ncz).to(tl.int32), 0), ncz - 1)
        if MORTON:
            key = _spread10(cx) | (_spread10(cy) << 1) | (_spread10(cz) << 2)
        else:
            key = (cx * ncy + cy) * ncz + cz
        tl.store(c_ptr + offs * 4 + 0, cx, mask=mask)
        tl.store(c_ptr + offs * 4 + 1, cy, mask=mask)
        tl.store(c_ptr + offs * 4 + 2, cz, mask=mask)
        tl.store(key_ptr + offs, key, mask=mask)

    @triton.jit
    def _build_kernel(
        x_ptr, c_ptr, cell_start_ptr, cell_atoms_ptr, n,
        lx, ly, lz, ncx, ncy, ncz, rx, ry, rz, rlist2, maxnb,
        nl_ptr, nnb_ptr, ovf_ptr,
        SORTED: tl.constexpr, MORTON: tl.constexpr, HALF: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        px = tl.load(x_ptr + offs * 4 + 0, mask=mask, other=0.0)
        py = tl.load(x_ptr + offs * 4 + 1, mask=mask, other=0.0)
        pz = tl.load(x_ptr + offs * 4 + 2, mask=mask, other=0.0)
        cx = tl.load(c_ptr + offs * 4 + 0, mask=mask, other=0)
        cy = tl.load(c_ptr + offs * 4 + 1, mask=mask, other=0)
        cz = tl.load(c_ptr + offs * 4 + 2, mask=mask, other=0)
        cnt = tl.zeros([BLOCK], dtype=tl.int32)
        for dx in range(-rx, rx + 1):
            ax = (cx + dx + ncx) % ncx
            for dy in range(-ry, ry + 1):
                ay = (cy + dy + ncy) % ncy
                for dz in range(-rz, rz + 1):
                    az = (cz + dz + ncz) % ncz
                    if MORTON:
                        key = _spread10(ax) | (_spread10(ay) << 1) | (_spread10(az) << 2)
                    else:
                        key = (ax * ncy + ay) * ncz + az
                    s = tl.load(cell_start_ptr + key, mask=mask, other=0)
                    e = tl.load(cell_start_ptr + key + 1, mask=mask, other=0)
                    m = e - s
                    mmax = tl.max(m, axis=0)
                    for t in range(0, mmax):
                        valid = mask & (t < m)
                        k = s + t
                        if SORTED:
                            j = k
                        else:
                            j = tl.load(cell_atoms_ptr + k, mask=valid, other=0)
                        qx = tl.load(x_ptr + j * 4 + 0, mask=valid, other=0.0)
                        qy = tl.load(x_ptr + j * 4 + 1, mask=valid, other=0.0)
                        qz = tl.load(x_ptr + j * 4 + 2, mask=valid, other=0.0)
                        ddx = px - qx
                        ddy = py - qy
                        ddz = pz - qz
                        ddx -= lx * libdevice.rint(ddx / lx)
                        ddy -= ly * libdevice.rint(ddy / ly)
                        ddz -= lz * libdevice.rint(ddz / lz)
                        r2 = ddx * ddx + ddy * ddy + ddz * ddz
                        ok = valid & (j != offs) & (r2 < rlist2)
                        if HALF:
                            ok = ok & (j > offs)
                        tl.store(nl_ptr + cnt * n + offs, j, mask=ok & (cnt < maxnb))
                        cnt += ok.to(tl.int32)
        tl.store(nnb_ptr + offs, cnt, mask=mask)
        tl.atomic_max(ovf_ptr, tl.max(tl.where(mask, cnt, 0), axis=0))

    @triton.jit
    def _force_kernel(
        x_ptr, nl_ptr, nnb_ptr, n, lx, ly, lz,
        rc2, lj1, lj2, lj3, lj4, eshift,
        f_ptr, e_ptr, w_ptr, v_ptr, xout_ptr, dtm, dt,
        ENERGY: tl.constexpr, FUSED: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        px = tl.load(x_ptr + offs * 4 + 0, mask=mask, other=0.0)
        py = tl.load(x_ptr + offs * 4 + 1, mask=mask, other=0.0)
        pz = tl.load(x_ptr + offs * 4 + 2, mask=mask, other=0.0)
        ilx = 1.0 / lx
        ily = 1.0 / ly
        ilz = 1.0 / lz
        m = tl.load(nnb_ptr + offs, mask=mask, other=0)
        mmax = tl.max(m, axis=0)
        fx = tl.zeros([BLOCK], dtype=tl.float32)
        fy = tl.zeros([BLOCK], dtype=tl.float32)
        fz = tl.zeros([BLOCK], dtype=tl.float32)
        ea = tl.zeros([BLOCK], dtype=tl.float32)
        wa = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(0, mmax):
            valid = mask & (k < m)
            j = tl.load(nl_ptr + k * n + offs, mask=valid, other=0)
            qx = tl.load(x_ptr + j * 4 + 0, mask=valid, other=0.0)
            qy = tl.load(x_ptr + j * 4 + 1, mask=valid, other=0.0)
            qz = tl.load(x_ptr + j * 4 + 2, mask=valid, other=0.0)
            dx = px - qx
            dy = py - qy
            dz = pz - qz
            dx -= lx * libdevice.rint(dx * ilx)
            dy -= ly * libdevice.rint(dy * ily)
            dz -= lz * libdevice.rint(dz * ilz)
            r2 = dx * dx + dy * dy + dz * dz
            inr = valid & (r2 < rc2)
            r2i = 1.0 / tl.where(inr, r2, 1.0)
            r6i = r2i * r2i * r2i
            ff = tl.where(inr, r6i * (lj1 * r6i - lj2) * r2i, 0.0)
            gx = ff * dx
            gy = ff * dy
            gz = ff * dz
            fx += gx
            fy += gy
            fz += gz
            if HALF:
                tl.atomic_add(f_ptr + j * 4 + 0, -gx, mask=inr)
                tl.atomic_add(f_ptr + j * 4 + 1, -gy, mask=inr)
                tl.atomic_add(f_ptr + j * 4 + 2, -gz, mask=inr)
            if ENERGY:
                ea += tl.where(inr, r6i * (lj3 * r6i - lj4) - eshift, 0.0)
                wa += ff * r2
        if FUSED:
            vx = tl.load(v_ptr + offs * 4 + 0, mask=mask, other=0.0) + dtm * fx
            vy = tl.load(v_ptr + offs * 4 + 1, mask=mask, other=0.0) + dtm * fy
            vz = tl.load(v_ptr + offs * 4 + 2, mask=mask, other=0.0) + dtm * fz
            tl.store(v_ptr + offs * 4 + 0, vx, mask=mask)
            tl.store(v_ptr + offs * 4 + 1, vy, mask=mask)
            tl.store(v_ptr + offs * 4 + 2, vz, mask=mask)
            tl.store(xout_ptr + offs * 4 + 0, px + dt * vx, mask=mask)
            tl.store(xout_ptr + offs * 4 + 1, py + dt * vy, mask=mask)
            tl.store(xout_ptr + offs * 4 + 2, pz + dt * vz, mask=mask)
        elif HALF:
            tl.atomic_add(f_ptr + offs * 4 + 0, fx, mask=mask)
            tl.atomic_add(f_ptr + offs * 4 + 1, fy, mask=mask)
            tl.atomic_add(f_ptr + offs * 4 + 2, fz, mask=mask)
        else:
            tl.store(f_ptr + offs * 4 + 0, fx, mask=mask)
            tl.store(f_ptr + offs * 4 + 1, fy, mask=mask)
            tl.store(f_ptr + offs * 4 + 2, fz, mask=mask)
        if ENERGY:
            scale = 1.0 if HALF else 0.5
            tl.store(e_ptr + offs, scale * ea, mask=mask)
            tl.store(w_ptr + offs, scale * wa, mask=mask)

    @triton.jit
    def _kick_drift_kernel(x_ptr, v_ptr, f_ptr, n4, kick, drift, DRIFT: tl.constexpr,
                           BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n4
        v = tl.load(v_ptr + offs, mask=mask) + kick * tl.load(f_ptr + offs, mask=mask)
        tl.store(v_ptr + offs, v, mask=mask)
        if DRIFT:
            x = tl.load(x_ptr + offs, mask=mask) + drift * v
            tl.store(x_ptr + offs, x, mask=mask)


class TritonEngine(Engine):
    """Triton kernels on PyTorch tensors (fp32, neighbor-major Verlet list)."""

    backend = "triton"
    variants = {
        "full": "fp32 full Verlet list (no atomics), cell-sorted atoms, Triton kernels",
        "full-fused": "fp32 full list, fused force+kick+drift Triton kernel per step (leapfrog)",
        "half-atomic": "fp32 half Verlet list, Newton's 3rd law with tl.atomic_add scatter",
    }

    def _setup(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.dev = torch.device("cuda")
        self.half = self.variant == "half-atomic"
        self.supports_fused = self.variant == "full-fused"
        self.sort = self.cfg.sort
        n = self.n
        self.x = torch.zeros((n, 4), dtype=torch.float32, device=self.dev)
        self.v = torch.zeros((n, 4), dtype=torch.float32, device=self.dev)
        self.x[:, :3] = torch.as_tensor(pos, device=self.dev)
        self.v[:, :3] = torch.as_tensor(vel, device=self.dev)
        self.f = torch.zeros((n, 4), dtype=torch.float32, device=self.dev)
        self.ids = torch.arange(n, dtype=torch.int32, device=self.dev)
        self.e_at = torch.zeros(n, dtype=torch.float32, device=self.dev)
        self.w_at = torch.zeros(n, dtype=torch.float32, device=self.dev)
        self.x_ref = self.x.clone()
        self.ncell = cell_grid(self.box, self.rlist)
        if self.sort == "morton":
            if self.ncell.max() > 1024:
                raise ValueError("Morton keys support at most 1024 cells per dimension")
            nc = self.ncell - 1
            self.nkeys = int(morton3(nc[:1], nc[1:2], nc[2:3])[0]) + 1
        else:
            self.nkeys = int(np.prod(self.ncell))
        self.key = torch.zeros(n, dtype=torch.int32, device=self.dev)
        self.cell = torch.zeros((n, 4), dtype=torch.int32, device=self.dev)
        self.key_range = torch.arange(self.nkeys + 1, dtype=torch.int32, device=self.dev)
        self.nnb = torch.zeros(n, dtype=torch.int32, device=self.dev)
        self.ovf = torch.zeros(1, dtype=torch.int32, device=self.dev)
        dens = n / np.prod(self.box)
        est = dens * 4.0 / 3.0 * np.pi * self.rlist**3
        self.maxnb = int(np.ceil(est * (0.65 if self.half else 1.3) / 8) * 8) + 8
        self.nl = torch.zeros(self.maxnb * n, dtype=torch.int32, device=self.dev)
        if self.supports_fused:
            self.x_alt = torch.empty_like(self.x)
        p = self.params
        s6 = p.sigma**6
        self.lj = (
            p.rc**2,
            48.0 * p.epsilon * s6 * s6,
            24.0 * p.epsilon * s6,
            4.0 * p.epsilon * s6 * s6,
            4.0 * p.epsilon * s6,
            p.eshift,
        )
        self._boxf = tuple(float(np.float32(b)) for b in self.box)
        self._dummy = torch.zeros(1, dtype=torch.float32, device=self.dev)

    # ------------------------------------------------------------------ rebuild
    def _rebuild(self) -> None:
        n = self.n
        nc = [int(v) for v in self.ncell]
        morton = self.sort == "morton"
        grid = (triton.cdiv(n, BLOCK_EW),)
        _wrap_key_kernel[grid](self.x, self.cell, self.key, n, *self._boxf, *nc,
                               MORTON=morton, BLOCK=BLOCK_EW)
        skey, order = torch.sort(self.key)
        cell_start = torch.searchsorted(skey, self.key_range, out_int32=True)
        sorted_flag = self.sort != "none"
        if sorted_flag:
            self.x = self.x.index_select(0, order)
            self.v = self.v.index_select(0, order)
            self.ids = self.ids.index_select(0, order)
            cell = self.cell.index_select(0, order)
            cell_atoms = self._dummy
        else:
            cell = self.cell
            cell_atoms = order.to(torch.int32)
        r = [1 if v >= 3 else 0 for v in nc]
        rl2 = float(self.rlist**2)
        grid = (triton.cdiv(n, BLOCK_BUILD),)
        while True:
            self.ovf.zero_()
            _build_kernel[grid](
                self.x, cell, cell_start, cell_atoms, n,
                *self._boxf, *nc, *r, rl2, self.maxnb,
                self.nl, self.nnb, self.ovf,
                SORTED=sorted_flag, MORTON=morton, HALF=self.half,
                BLOCK=BLOCK_BUILD, num_warps=WARPS_BUILD,
            )
            mx = int(self.ovf.item())
            if mx <= self.maxnb:
                break
            self.maxnb = int(np.ceil(mx * 1.2 / 8) * 8)
            self.nl = torch.zeros(self.maxnb * n, dtype=torch.int32, device=self.dev)
        self.x_ref = self.x.clone()
        self._forces_valid = False

    # ------------------------------------------------------------------ per step
    def _launch_force(self, energy: bool, fused: bool, dtm: float = 0.0, dt: float = 0.0):
        n = self.n
        grid = (triton.cdiv(n, BLOCK_FORCE),)
        _force_kernel[grid](
            self.x, self.nl, self.nnb, n, *self._boxf, *self.lj,
            self.f, self.e_at, self.w_at,
            self.v, self.x_alt if fused else self._dummy, dtm, dt,
            ENERGY=energy, FUSED=fused, HALF=self.half and not fused,
            BLOCK=BLOCK_FORCE, num_warps=WARPS_FORCE,
        )

    def _compute_forces(self, energy: bool = False):
        if self.half:
            self.f.zero_()
        self._launch_force(energy, False)
        if energy:
            return (
                float(self.e_at.sum(dtype=torch.float64)),
                float(self.w_at.sum(dtype=torch.float64)),
            )
        return None

    def _fused_step(self, dt: float) -> None:
        self._launch_force(False, True, dt / self.cfg.mass, dt)
        self.x, self.x_alt = self.x_alt, self.x

    def _kick_drift(self, kick: float, drift: float) -> None:
        n4 = self.n * 4
        grid = (triton.cdiv(n4, BLOCK_EW),)
        _kick_drift_kernel[grid](self.x, self.v, self.f, n4, float(kick), float(drift),
                                 DRIFT=drift != 0.0, BLOCK=BLOCK_EW)

    def _max_disp2(self) -> float:
        d = self.x[:, :3] - self.x_ref[:, :3]
        return float((d * d).sum(dim=1).max())

    def _ekin(self) -> float:
        return 0.5 * self.cfg.mass * float((self.v[:, :3].double() ** 2).sum())

    def _scale_v(self, lam: float) -> None:
        self.v.mul_(lam)

    def _ou(self, c1: float, c2: float, rng) -> None:
        if not hasattr(self, "_tgen"):
            self._tgen = torch.Generator(device=self.dev)
            self._tgen.manual_seed(int(rng.integers(2**31)))
        xi = torch.randn((self.n, 3), generator=self._tgen, device=self.dev)
        self.v[:, :3] = c1 * self.v[:, :3] + c2 * xi

    def _download(self):
        return (
            self.x[:, :3].double().cpu().numpy(),
            self.v[:, :3].double().cpu().numpy(),
            self.ids.cpu().numpy(),
        )

    def _download_forces(self):
        return self.f[:, :3].double().cpu().numpy(), self.ids.cpu().numpy()

    def mean_neighbors(self) -> float:
        return float(self.nnb.sum(dtype=torch.float64)) / self.n

    def synchronize(self) -> None:
        torch.cuda.synchronize()
