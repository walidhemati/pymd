"""Physics validation suite: writes ``results/validation.json`` (read by the figure script).

1. Forces of every backend/variant vs the NumPy float64 all-pairs oracle.
2. LAMMPS ``bench/in.lj``: step-0 thermo (exact) and step-100 thermo (statistical).
3. NVE energy conservation vs time step (fp32 / mixed / fp64) and long-run drift.
4. Radial distribution function of the dense LJ liquid and its thermodynamic consistency
   (energy and virial routes vs direct averages, energy route vs the JZG equation of state).
5. Energy and pressure vs the NIST SRSW Monte Carlo reference data and the JZG EOS.
6. Gupta/EAM gold: GPU fp32 vs float64 forces, NVE conservation of a Mackay cluster.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import backends
from .engine import Langevin
from .eos import LAMMPS_IN_LJ, NIST_MC, jzg_energy, jzg_pressure
from .reference import build_pairs, lj_allpairs
from .system import LJParams, MDConfig, fcc_lattice, lj_tail_corrections, maxwell_velocities


def _fast_backend() -> tuple[str, str]:
    av = backends.available()
    for b, v in (("cupy-cuda-c", "full"), ("warp", "full"), ("triton", "full"),
                 ("numba", "full")):
        if b in av:
            return b, v
    return "numpy", "cell-verlet"


def _engine(pos, vel, box, params, cfg, backend=None, variant=None):
    b, v = (backend, variant) if backend else _fast_backend()
    return backends.get(b)(pos=pos, vel=vel, box=box, params=params, cfg=cfg, variant=v)


def _block_err(x: np.ndarray, nblocks: int = 10) -> float:
    x = np.asarray(x)
    m = len(x) // nblocks
    if m < 1:
        return float("nan")
    b = x[: m * nblocks].reshape(nblocks, m).mean(1)
    return float(b.std(ddof=1) / np.sqrt(nblocks))


def equilibrated_liquid(cells: int, rho: float, temp: float, params: LJParams,
                        steps: int = 4000, seed: int = 0):
    """Melt an FCC lattice and equilibrate with Langevin at ``temp``; returns pos, vel, box."""
    pos, box = fcc_lattice(cells, density=rho, shuffle=True, seed=seed)
    t_melt = max(2.0 * temp, 3.0)  # well above melting even at rho = 0.9
    vel = maxwell_velocities(len(pos), t_melt, seed=seed)
    eng = _engine(pos, vel, box, params, MDConfig(dt=0.005, check=True, rebuild_every=5))
    eng.run(steps // 4, thermostat=Langevin(t_melt, gamma=2.0, seed=seed))
    eng.run(steps // 4, thermostat=Langevin(temp, gamma=2.0, seed=seed + 1))
    eng.run(steps // 2, thermostat=Langevin(temp, gamma=0.5, seed=seed + 2))
    p, v = eng.state()
    return p, v, box


# ----------------------------------------------------------------------------- 1. forces
def force_errors(quick: bool = False) -> list[dict]:
    params = LJParams(rc=2.5, shift=True)
    pos, vel, box = equilibrated_liquid(8 if quick else 10, 0.8442, 0.722, params,
                                        steps=1000 if quick else 3000)
    fref, eref, wref = lj_allpairs(pos, box, params)
    out = []
    for b, v in backends.all_variants():
        try:
            eng = backends.get(b)(pos=pos, vel=vel, box=box, params=params, variant=v)
            f, e, w = eng.forces()
        except Exception as exc:  # pragma: no cover - reporting only
            out.append({"backend": b, "variant": v, "error": repr(exc)[:200]})
            continue
        df = np.linalg.norm(f - fref, axis=1)
        fn = np.linalg.norm(fref, axis=1)
        out.append({
            "backend": b, "variant": v, "n": len(pos),
            "max_abs_err": float(df.max()), "max_force": float(fn.max()),
            "rel_max_err": float(df.max() / fn.max()),
            "rel_rms_err": float(np.sqrt(np.mean(df**2) / np.mean(fn**2))),
            "rel_energy_err": abs(e - eref) / abs(eref),
            "rel_virial_err": abs(w - wref) / abs(wref),
        })
    return out


# ----------------------------------------------------------------------------- 2. in.lj
def lammps_in_lj(quick: bool = False) -> dict:
    """Reproduce LAMMPS in.lj thermo output (step 0 exactly, step 100 statistically)."""
    params = LJParams(rc=2.5, shift=False)  # lj/cut, no pair_modify shift
    cfg = MDConfig(dt=0.005, skin=0.3, rebuild_every=20, check=False)
    pos, box = fcc_lattice(20, density=0.8442)
    seeds = [1, 2] if quick else list(range(1, 9))
    s0, s100, neigh, builds = [], [], [], []
    for s in seeds:
        vel = maxwell_velocities(len(pos), 1.44, seed=s, dist="uniform")  # LAMMPS default
        eng = _engine(pos, vel, box, params, cfg)
        t0 = eng.thermo()
        eng.run(100)
        t1 = eng.thermo()
        s0.append(t0)
        s100.append(t1)
        nb = eng.mean_neighbors()
        if np.isfinite(nb):
            neigh.append(nb / 2 if "half" not in eng.variant else nb)
        builds.append(eng.n_builds - 1)  # LAMMPS does not count the setup build

    def stat(rows, key):
        a = np.array([r[key] for r in rows])
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}

    return {
        "n": len(pos), "seeds": seeds,
        "lammps": {k: v for k, v in LAMMPS_IN_LJ.items() if not isinstance(k, int)}
        | {"step0": LAMMPS_IN_LJ[0], "step100": LAMMPS_IN_LJ[100]},
        "step0": {k: stat(s0, k) for k in ("temp", "epot", "etot", "pressure")},
        "step100": {k: stat(s100, k) for k in ("temp", "epot", "etot", "pressure")},
        "ave_neighs_half": float(np.mean(neigh)) if neigh else None,
        "neighbor_builds_after_setup": int(np.median(builds)),
    }


# ----------------------------------------------------------------------------- 3. NVE
def energy_conservation(quick: bool = False) -> dict:
    params = LJParams(rc=2.5, shift=True)
    pos, vel, box = equilibrated_liquid(10, 0.8442, 0.722, params, steps=2000)
    total_time = 5.0 if quick else 20.0
    dts = [0.002, 0.005] if quick else [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01]
    av = backends.available()
    variants = [("cupy-cuda-c", "full"), ("cupy-cuda-c", "full-mixed"),
                ("cupy-cuda-c", "full-fp64")] if "cupy-cuda-c" in av else [_fast_backend()]
    sweep = []
    for b, v in variants:
        for dt in dts:
            cfg = MDConfig(dt=dt, skin=0.3, rebuild_every=1, check=True)
            eng = backends.get(b)(pos=pos, vel=vel, box=box, params=params, cfg=cfg, variant=v)
            nsteps = int(round(total_time / dt))
            every = max(1, nsteps // 400)
            es, ts = [], []
            for k in range(nsteps // every):
                eng.run(every)
                es.append(eng.thermo()["etot"])
                ts.append((k + 1) * every * dt)
            es = np.array(es)
            slope = float(np.polyfit(ts, es, 1)[0])
            sweep.append({"backend": b, "variant": v, "dt": dt, "nsteps": nsteps,
                          "rms_fluct": float(es.std()), "drift_per_tau": slope,
                          "e_mean": float(es.mean())})
    # long run: drift of fp32 vs mixed vs fp64 at the benchmark time step
    long = []
    t_long = 25.0 if quick else 500.0
    for b, v in variants:
        cfg = MDConfig(dt=0.005, skin=0.3, rebuild_every=5, check=True)
        eng = backends.get(b)(pos=pos, vel=vel, box=box, params=params, cfg=cfg, variant=v)
        e0 = eng.thermo()["etot"]
        nsteps = int(t_long / 0.005)
        chunk = nsteps // 100
        ts, es = [0.0], [0.0]
        t_wall = time.perf_counter()
        for k in range(100):
            eng.run(chunk)
            es.append(eng.thermo()["etot"] - e0)
            ts.append((k + 1) * chunk * 0.005)
        long.append({"backend": b, "variant": v, "t": ts, "de": es,
                     "drift_per_tau": float(np.polyfit(ts, es, 1)[0]),
                     "wall_s": time.perf_counter() - t_wall})
    return {"n": len(pos), "rho": 0.8442, "temp": 0.722, "total_time_tau": total_time,
            "sweep": sweep, "long": long}


# ----------------------------------------------------------------------------- 4. g(r)
def _gr(frames, box, rmax, nbins):
    n = len(frames[0])
    rho = n / np.prod(box)
    edges = np.linspace(0, rmax, nbins + 1)
    hist = np.zeros(nbins)
    for x in frames:
        i, j = build_pairs(x, box, rmax)
        d = x[i] - x[j]
        d -= box * np.rint(d / box)
        hist += np.histogram(np.sqrt((d * d).sum(1)), bins=edges)[0]
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (len(frames) * 0.5 * n * rho * shell)
    return 0.5 * (edges[1:] + edges[:-1]), g


def radial_distribution(quick: bool = False) -> dict:
    """g(r) at rho = 0.8442, T = 0.722 with rc = 3.0 (NIST-style truncation)."""
    rho, temp = 0.8442, 0.722
    params = LJParams(rc=3.0, shift=False)
    pos, vel, box = equilibrated_liquid(8 if quick else 10, rho, temp, params, steps=3000)
    cfg = MDConfig(dt=0.005, check=True, rebuild_every=5)
    eng = _engine(pos, vel, box, params, cfg)
    th = Langevin(temp, gamma=0.5, seed=11)
    nframes = 20 if quick else 100
    frames, us, ps = [], [], []
    for _ in range(nframes):
        eng.run(50, thermostat=th)
        x, _ = eng.state()
        frames.append(x)
        t = eng.thermo()
        us.append(t["epot"])
        ps.append(t["pressure"])
    rmax = min(4.5, 0.5 * box.min())
    r, g = _gr(frames, box, rmax, 450)
    dr = r[1] - r[0]
    u_full = 4 * (r**-12 - r**-6)
    rdu = -24 * (2 * r**-12 - r**-6)  # r u'(r)
    inside = r < params.rc
    n_tail = lj_tail_corrections(rho, LJParams(rc=params.rc))
    u_route_rc = 2 * np.pi * rho * np.sum((u_full * g * r**2)[inside]) * dr
    p_route_rc = rho * temp - 2 * np.pi / 3 * rho**2 * np.sum((rdu * g * r**2)[inside]) * dr
    # energy route with the full potential out to rmax plus the g = 1 tail beyond
    u_route_full = (2 * np.pi * rho * np.sum(u_full * g * r**2) * dr
                    + lj_tail_corrections(rho, LJParams(rc=rmax))[0])
    ipk = int(np.argmax(g))
    return {
        "rho": rho, "temp": temp, "rc": params.rc, "n": len(pos), "frames": nframes,
        "r": r.tolist(), "g": g.tolist(), "peak_r": float(r[ipk]), "peak_g": float(g[ipk]),
        "u_direct": float(np.mean(us)), "u_direct_err": _block_err(us),
        "p_direct": float(np.mean(ps)), "p_direct_err": _block_err(ps),
        "u_energy_route": float(u_route_rc), "p_virial_route": float(p_route_rc),
        "u_direct_plus_lrc": float(np.mean(us) + n_tail[0]),
        "p_direct_plus_lrc": float(np.mean(ps) + n_tail[1]),
        "u_energy_route_full_lj": float(u_route_full),
        "jzg_u": jzg_energy(rho, temp), "jzg_p": jzg_pressure(rho, temp),
    }


# ----------------------------------------------------------------------------- 5. EOS
def equation_of_state(quick: bool = False) -> dict:
    """NVT (Langevin) at the NIST SRSW Monte Carlo state points, rc = 3.0 + LRC, N = 500."""
    params = LJParams(rc=3.0, shift=False)
    rows = []
    pts = NIST_MC[:2] if quick else NIST_MC
    for temp, rho, u_ref, du_ref, p_ref, dp_ref in pts:
        pos, vel, box = equilibrated_liquid(5, rho, temp, params, steps=2000 if quick else 4000)
        eng = _engine(pos, vel, box, params, MDConfig(dt=0.005, check=True, rebuild_every=5))
        th = Langevin(temp, gamma=1.0, seed=int(rho * 1000))
        us, ps = [], []
        nsamp = 200 if quick else 2000
        for _ in range(nsamp):
            eng.run(10, thermostat=th)
            t = eng.thermo()
            us.append(t["epot"])
            ps.append(t["pressure"])
        ut, pt = lj_tail_corrections(rho, params)
        rows.append({
            "temp": temp, "rho": rho, "u": float(np.mean(us) + ut), "u_err": _block_err(us),
            "p": float(np.mean(ps) + pt), "p_err": _block_err(ps),
            "nist_u": u_ref, "nist_u_err": du_ref, "nist_p": p_ref, "nist_p_err": dp_ref,
            "jzg_u": jzg_energy(rho, temp), "jzg_p": jzg_pressure(rho, temp),
            "production_tau": nsamp * 10 * 0.005,
        })
    return {"n": 500, "rc": 3.0, "rows": rows}


# ----------------------------------------------------------------------------- 6. EAM
def eam_gold(quick: bool = False) -> dict:
    if "cupy-cuda-c" not in backends.available():
        return {"skipped": "needs the CuPy backend"}
    from .eam import AU_GUPTA, gupta_allpairs, mackay_icosahedron, make_gupta_engine

    x0 = mackay_icosahedron(4 if quick else 6)
    rng = np.random.default_rng(3)
    x = x0 + 0.05 * rng.standard_normal(x0.shape) + 40.0
    box = np.full(3, 80.0)
    fref, eref = gupta_allpairs(x, None, AU_GUPTA)
    eng = make_gupta_engine(x, np.zeros_like(x), box, AU_GUPTA)
    f, e, _ = eng.forces()
    df = np.linalg.norm(f - fref, axis=1)
    # NVE of the cluster heated with small random velocities (~300 K)
    from .eam import AU_MASS, KB_EV_K

    v = rng.standard_normal(x.shape) * np.sqrt(KB_EV_K * 600 / AU_MASS)
    v -= v.mean(0)
    eng = make_gupta_engine(x, v, box, AU_GUPTA)
    e0 = eng.thermo()["etot"]
    es = []
    for _ in range(20 if quick else 100):
        eng.run(20)
        es.append(eng.thermo()["etot"] - e0)
    return {"n": len(x), "rel_max_force_err": float(df.max() / np.linalg.norm(fref, axis=1).max()),
            "rel_energy_err": abs(e - eref) / abs(eref), "nve_de_per_atom_eV": es,
            "nve_dt_fs": 1.0}


def run_all(out: Path, quick: bool = False, parts: list[str] | None = None) -> dict:
    """Run the requested validation parts and write JSON to ``out``."""
    todo = {
        "forces": force_errors, "lammps_in_lj": lammps_in_lj,
        "energy_conservation": energy_conservation, "rdf": radial_distribution,
        "eos": equation_of_state, "eam": eam_gold,
    }
    doc = json.loads(out.read_text()) if out.exists() else {}
    doc["provisional"] = True
    doc["quick"] = quick
    for name, fn in todo.items():
        if parts and name not in parts:
            continue
        t = time.perf_counter()
        print(f"[validate] {name} ...", flush=True)
        doc[name] = fn(quick)
        doc.setdefault("wall_seconds", {})[name] = time.perf_counter() - t
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1))
    return doc
