"""Backend-independent MD driver.

Every backend subclasses :class:`Engine` and implements a handful of device primitives
(rebuild the neighbor list, compute forces, kick/drift, reductions). The time-stepping
logic -- velocity Verlet, neighbor-list rebuild schedule, thermostats -- lives here once,
so all backends integrate exactly the same equations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .system import LJParams, MDConfig


@dataclass
class Berendsen:
    """Berendsen weak-coupling thermostat (velocity rescaling every step)."""

    temperature: float
    tau: float


@dataclass
class Langevin:
    """Langevin thermostat integrated with the BAOAB splitting."""

    temperature: float
    gamma: float
    seed: int = 1234


Thermostat = Berendsen | Langevin


class Engine(ABC):
    """One simulation on one backend. Subclasses implement the ``_device`` primitives.

    Arrays passed in and returned are float64 NumPy arrays in the *original* atom order;
    backends are free to permute atoms internally (spatial sorting).
    """

    backend: ClassVar[str] = "abstract"
    #: variant name -> one-line description
    variants: ClassVar[dict[str, str]] = {}
    #: does this variant use a Verlet neighbor list?
    uses_list: bool = True
    #: can this variant run a fused force+kick+drift kernel (one launch per step)?
    supports_fused: bool = False

    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        box: np.ndarray,
        params: LJParams | None = None,
        cfg: MDConfig | None = None,
        variant: str | None = None,
        **opts,
    ) -> None:
        self.opts = opts
        self.params = params or LJParams()
        self.cfg = cfg or MDConfig()
        self.variant = variant or next(iter(self.variants))
        if self.variant not in self.variants:
            raise ValueError(f"{self.backend}: unknown variant {self.variant!r}")
        self.box = np.asarray(box, dtype=np.float64).copy()
        self.n = len(pos)
        self.rlist = self.params.rc + self.cfg.skin
        if np.any(self.box < 2 * self.rlist) and self.uses_list:
            raise ValueError("box must be at least 2*(rc+skin) wide for minimum image")
        self.step_count = 0
        self.n_builds = 0
        self._built_step: int | None = None
        self._forces_valid = False
        self._pending_half = False
        self._setup(np.asarray(pos, np.float64), np.asarray(vel, np.float64))

    # ------------------------------------------------------------------ device API
    @abstractmethod
    def _setup(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Allocate device arrays and upload the initial state."""

    @abstractmethod
    def _rebuild(self) -> None:
        """Wrap positions, (optionally) sort atoms, bin into cells, build the neighbor list."""

    @abstractmethod
    def _compute_forces(self, energy: bool = False) -> tuple[float, float] | None:
        """Compute forces at the current positions; with ``energy`` return (epot, virial)."""

    @abstractmethod
    def _kick_drift(self, kick: float, drift: float) -> None:
        """``v += kick * f / m`` then ``x += drift * v`` (one fused elementwise kernel)."""

    @abstractmethod
    def _max_disp2(self) -> float:
        """Largest squared displacement since the last neighbor-list build."""

    @abstractmethod
    def _ekin(self) -> float:
        """Total kinetic energy ``sum m v^2 / 2``."""

    @abstractmethod
    def _scale_v(self, lam: float) -> None:
        """Multiply all velocities by ``lam``."""

    @abstractmethod
    def _download(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions, velocities, original ids) in internal order."""

    @abstractmethod
    def _download_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (forces, original ids) in internal order."""

    def _ou(self, c1: float, c2: float, rng: np.random.Generator) -> None:
        """Ornstein-Uhlenbeck velocity update ``v = c1 v + c2 xi`` (host fallback)."""
        v = self._get_v_raw()
        v = c1 * v + c2 * rng.standard_normal(v.shape)
        self._set_v_raw(v)

    def _get_v_raw(self) -> np.ndarray:
        raise NotImplementedError

    def _set_v_raw(self, v: np.ndarray) -> None:
        raise NotImplementedError

    def _fused_step(self, dt: float) -> None:
        """Leapfrog step: forces at x, ``v += dt f/m``, ``x += dt v`` in one kernel."""
        raise NotImplementedError

    def mean_neighbors(self) -> float:
        """Average stored neighbor-list entries per atom (for the traffic model)."""
        return float("nan")

    def synchronize(self) -> None:  # noqa: B027 - intentionally a no-op on the CPU
        """Block until queued device work has finished."""

    # ------------------------------------------------------------------ driver
    def _maybe_rebuild(self, force: bool = False) -> None:
        if not self.uses_list:
            return
        if self._built_step is None or force:
            self._do_rebuild()
            return
        every = self.cfg.rebuild_every
        if every <= 0 or (self.step_count - self._built_step) % every != 0:
            return
        if self.step_count == self._built_step:
            return
        if self.cfg.check and self._max_disp2() <= (0.5 * self.cfg.skin) ** 2:
            return
        self._do_rebuild()

    def _do_rebuild(self) -> None:
        self._rebuild()
        self.n_builds += 1
        self._built_step = self.step_count

    def _ensure_forces(self) -> None:
        if not self._forces_valid:
            self._maybe_rebuild(force=self._built_step is None)
            self._compute_forces(False)
            self._forces_valid = True

    def run(self, nsteps: int, thermostat: Thermostat | None = None, fused: bool | None = None):
        """Advance ``nsteps`` velocity-Verlet steps (NVE, or with a thermostat)."""
        dt = self.cfg.dt
        im = 1.0 / self.cfg.mass
        self._ensure_forces()
        if isinstance(thermostat, Langevin):
            self._run_langevin(nsteps, thermostat)
            return
        if fused is None:
            fused = self.supports_fused and thermostat is None
        if fused and thermostat is None and self.supports_fused:
            self._run_fused(nsteps)
            return
        for _ in range(nsteps):
            # finishing half kick of the previous step fused with this step's half kick + drift
            self._kick_drift((dt if self._pending_half else 0.5 * dt) * im, dt)
            self.step_count += 1
            self._maybe_rebuild()
            self._compute_forces(False)
            self._pending_half = True
            if isinstance(thermostat, Berendsen):
                self._kick_drift(0.5 * dt * im, 0.0)
                self._pending_half = False
                t_now = 2.0 * self._ekin() / self.dof
                if t_now > 0:
                    lam = np.sqrt(1.0 + dt / thermostat.tau * (thermostat.temperature / t_now - 1))
                    self._scale_v(float(lam))
        if self._pending_half:
            self._kick_drift(0.5 * dt * im, 0.0)
            self._pending_half = False

    def _run_fused(self, nsteps: int) -> None:
        dt = self.cfg.dt
        im = 1.0 / self.cfg.mass
        if self._pending_half:
            self._kick_drift(0.5 * dt * im, 0.0)
            self._pending_half = False
        # v(0) -> v(-dt/2): leapfrog state (x_n, v_{n-1/2})
        self._kick_drift(-0.5 * dt * im, 0.0)
        for _ in range(nsteps):
            self._maybe_rebuild()
            self._fused_step(dt)
            self.step_count += 1
        self._maybe_rebuild()
        self._compute_forces(False)
        self._kick_drift(0.5 * dt * im, 0.0)

    def _run_langevin(self, nsteps: int, th: Langevin) -> None:
        dt = self.cfg.dt
        m = self.cfg.mass
        rng = getattr(self, "_lrng", None) or np.random.default_rng(th.seed)
        self._lrng = rng
        c1 = float(np.exp(-th.gamma * dt))
        c2 = float(np.sqrt((1 - c1 * c1) * th.temperature / m))
        for _ in range(nsteps):
            self._kick_drift(0.5 * dt / m, 0.5 * dt)  # B A
            self._ou(c1, c2, rng)  # O
            self._kick_drift(0.0, 0.5 * dt)  # A
            self.step_count += 1
            self._maybe_rebuild()
            self._compute_forces(False)
            self._kick_drift(0.5 * dt / m, 0.0)  # B

    # ------------------------------------------------------------------ observables
    @property
    def dof(self) -> int:
        return 3 * self.n - 3

    @property
    def volume(self) -> float:
        return float(np.prod(self.box))

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions and velocities (float64) in the original atom order."""
        pos, vel, ids = self._download()
        out_p = np.empty_like(pos, dtype=np.float64)
        out_v = np.empty_like(vel, dtype=np.float64)
        out_p[ids] = pos
        out_v[ids] = vel
        return out_p, out_v

    def forces(self) -> tuple[np.ndarray, float, float]:
        """Forces (original order), potential energy and virial at the current positions."""
        self._maybe_rebuild(force=True)
        ev = self._compute_forces(True)
        assert ev is not None
        f, ids = self._download_forces()
        out = np.empty_like(f, dtype=np.float64)
        out[ids] = f
        self._forces_valid = True
        return out, ev[0], ev[1]

    def thermo(self) -> dict[str, float]:
        """Instantaneous potential/kinetic/total energy per atom, temperature, pressure."""
        if self._pending_half:
            self._kick_drift(0.5 * self.cfg.dt / self.cfg.mass, 0.0)
            self._pending_half = False
        self._maybe_rebuild(force=self._built_step is None)
        ev = self._compute_forces(True)
        assert ev is not None
        self._forces_valid = True
        epot, vir = ev
        ekin = self._ekin()
        temp = 2.0 * ekin / self.dof
        # LAMMPS convention: P = (dof k T / 3 + W / 3) / V = (2 E_kin / 3 + W / 3) / V
        pressure = (2.0 * ekin / 3.0 + vir / 3.0) / self.volume
        return {
            "epot": epot / self.n,
            "ekin": ekin / self.n,
            "etot": (epot + ekin) / self.n,
            "temp": temp,
            "pressure": pressure,
            "virial": vir,
        }
