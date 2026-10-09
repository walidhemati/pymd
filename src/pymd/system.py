"""System setup: Lennard-Jones parameters, run configuration, lattices and velocities.

Everything here is in Lennard-Jones reduced units (sigma = epsilon = m = k_B = 1) unless
stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Argon mapping used only to quote "ns/day" (sigma = 0.3405 nm, eps/kB = 119.8 K, m = 39.948 u):
ARGON_SIGMA_NM = 0.3405
ARGON_EPS_OVER_KB_K = 119.8
ARGON_MASS_AMU = 39.948
_KB = 1.380649e-23
_AMU = 1.66053906660e-27
#: LJ time unit tau = sigma * sqrt(m / epsilon) for argon, in picoseconds (~2.156 ps).
ARGON_TAU_PS = (
    ARGON_SIGMA_NM * 1e-9 * np.sqrt(ARGON_MASS_AMU * _AMU / (ARGON_EPS_OVER_KB_K * _KB)) * 1e12
)


@dataclass(frozen=True)
class LJParams:
    """Truncated (and optionally shifted) Lennard-Jones pair potential."""

    epsilon: float = 1.0
    sigma: float = 1.0
    rc: float = 2.5
    shift: bool = True

    @property
    def eshift(self) -> float:
        """Energy shift so that u(rc) = 0 (zero if ``shift`` is False)."""
        if not self.shift:
            return 0.0
        sr6 = (self.sigma / self.rc) ** 6
        return 4.0 * self.epsilon * (sr6 * sr6 - sr6)


@dataclass
class MDConfig:
    """Integrator and neighbor-list settings.

    The defaults reproduce the LAMMPS ``bench/in.lj`` benchmark: dt = 0.005 tau,
    skin 0.3 sigma, neighbor list rebuilt every 20 steps without a displacement check.
    """

    dt: float = 0.005
    skin: float = 0.3
    rebuild_every: int = 20
    check: bool = False
    mass: float = 1.0
    sort: str = "cell"  # "none" | "cell" | "morton"
    extra: dict = field(default_factory=dict)


def fcc_lattice(
    n: int | tuple[int, int, int], density: float = 0.8442, shuffle: bool = False, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Face-centred cubic lattice of ``4 * nx * ny * nz`` atoms at reduced number ``density``.

    Returns ``(positions (N, 3) float64, box (3,) float64)``. With ``shuffle`` the atom order
    is randomly permuted (destroys the memory locality of the lattice order).
    """
    nx, ny, nz = (n, n, n) if isinstance(n, int) else n
    a = (4.0 / density) ** (1.0 / 3.0)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=np.float64)
    gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)
    pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    pos += 0.25 * a  # keep atoms off the box faces
    box = np.array([nx, ny, nz], dtype=np.float64) * a
    if shuffle:
        pos = pos[np.random.default_rng(seed).permutation(len(pos))]
    return pos, box


def fcc_cells_for(n_atoms: int) -> int:
    """Number of FCC unit cells per side giving the cubic lattice closest to ``n_atoms``."""
    return max(1, round((n_atoms / 4) ** (1.0 / 3.0)))


def maxwell_velocities(
    n: int, temperature: float, mass: float = 1.0, seed: int = 0, dist: str = "gaussian"
) -> np.ndarray:
    """Random velocities with zero total momentum, rescaled to exactly ``temperature``.

    ``dist="uniform"`` mimics LAMMPS ``velocity create`` (its default distribution).
    The kinetic temperature uses ``3N - 3`` degrees of freedom (momentum removed).
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3)) if dist == "gaussian" else rng.uniform(-0.5, 0.5, (n, 3))
    v -= v.mean(axis=0)
    t_now = mass * np.sum(v * v) / (3 * n - 3)
    if t_now > 0:
        v *= np.sqrt(temperature / t_now)
    return v


def lj_tail_corrections(density: float, params: LJParams) -> tuple[float, float]:
    """Standard long-range corrections for a *truncated* (unshifted) LJ potential.

    Returns ``(u_tail per particle, p_tail)`` (Allen & Tildesley, Frenkel & Smit).
    """
    s3 = params.sigma**3
    x3 = (params.sigma / params.rc) ** 3
    x9 = x3**3
    u = 8.0 / 3.0 * np.pi * density * params.epsilon * s3 * (x9 / 3.0 - x3)
    p = 16.0 / 3.0 * np.pi * density**2 * params.epsilon * s3 * (2.0 / 3.0 * x9 - x3)
    return float(u), float(p)


def ns_per_day(steps_per_s: float, dt_tau: float) -> float:
    """Simulated nanoseconds per wall-clock day with the argon mapping of the LJ time unit."""
    return steps_per_s * dt_tau * ARGON_TAU_PS * 1e-3 * 86400.0


def tau_per_day(steps_per_s: float, dt_tau: float) -> float:
    """Simulated LJ time units per day (what LAMMPS prints for ``units lj``)."""
    return steps_per_s * dt_tau * 86400.0
