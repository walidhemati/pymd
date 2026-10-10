"""Same-GPU head-to-head with established MD codes: LAMMPS (GPU and KOKKOS packages), HOOMD-blue.

All codes run the LAMMPS ``bench/in.lj`` melt with pymd's benchmark settings (``bench/run.py``):
FCC at rho* = 0.8442, T0* = 1.44, truncated (unshifted) LJ with rc = 2.5 sigma, NVE velocity
Verlet with dt = 0.005 tau, neighbor skin 0.3 sigma, list rebuilt every 20 steps without a
displacement check, on the same ladder of FCC cells per side as ``bench/run.py``. pymd's best
variants are timed in the same session so that every row of the output shares the GPU state.

The counterparts live outside the repository (default ``~/python-hpc/_ref``)::

    git clone --depth 1 --branch stable_30Sep2026 https://github.com/lammps/lammps _ref/lammps
    bench/lammps/build_lammps.sh gpu-mixed      # and gpu-single, kokkos-double, kokkos-single
    conda create -p _ref/hoomdenv -c conda-forge --override-channels 'hoomd=7.2.0=gpu*' \\
        python=3.13 numpy

Before timing, every code runs the *identical* initial state (pymd's lattice and velocities;
LAMMPS via ``read_data``, HOOMD via a snapshot) for 100 steps and its thermo output at steps 0
and 100 is compared with pymd in float64.

    python bench/counterparts.py --quick   # smoke run, N = 32,000 -> results/counterparts.json
    python bench/counterparts.py --full    # publishable: serial, idle GPU, N up to 4,000,000

LAMMPS rates come from LAMMPS' own ``Loop time`` of each timed ``run`` command (the number its
``Performance:`` line is computed from; setup and the neighbor build at run start excluded);
HOOMD and pymd rates from wall-clock time around ``Simulation.run`` / ``Engine.run``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = Path(os.environ.get("PYHPC_REF", ROOT.parent / "_ref"))
LAMMPS_SRC = Path(os.environ.get("LAMMPS_SRC", REF / "lammps"))
LAMMPS_BUILDS = Path(os.environ.get("LAMMPS_BUILDS", REF / "lammps-builds"))
HOOMD_PYTHON = Path(os.environ.get("HOOMD_PYTHON", REF / "hoomdenv" / "bin" / "python"))
HOOMD_SRC = Path(os.environ.get("HOOMD_SRC", REF / "hoomd-blue"))
#: name -> (interpreter, extra sys.path entry or None, precision, description)
HOOMD_VARIANTS: dict[str, tuple[Path, Path | None, str, str]] = {
    "gpu-double": (HOOMD_PYTHON, None, "double",
                   "conda-forge GPU build (LongReal = 64 bit, ShortReal = 32 bit): pair "
                   "forces evaluated in float64"),
    "gpu-single": (REF / "hoomdbuild" / "bin" / "python", HOOMD_SRC / "build-single", "single",
                   "source build of the same release with HOOMD_LONGREAL_SIZE=32 "
                   "(bench/hoomd/build_hoomd_single.sh)"),
}
IN_BENCH = ROOT / "bench" / "lammps" / "in.lj.bench"
IN_VERIFY = ROOT / "bench" / "lammps" / "in.lj.verify"

DENSITY, T0, RC, SKIN, DT, EVERY = 0.8442, 1.44, 2.5, 0.3, 0.005, 20
FULL_CELLS = [10, 20, 30, 40, 50, 64, 80, 100]  # same ladder as bench/run.py
QUICK_CELLS = [20]
VERIFY_CELLS = 20

_KK = ["-k", "on", "g", "1", "-sf", "kk"]
_KK_HALF = ["-pk", "kokkos", "neigh", "half", "newton", "on"]
_GPU = ["-sf", "gpu", "-pk", "gpu", "1"]
#: name -> (build, command-line accelerator flags, precision, neighbor/offload description)
LAMMPS_VARIANTS: dict[str, tuple[str, list[str], str, str]] = {
    "kokkos-double": ("kokkos-double", _KK, "double",
                      "KOKKOS CUDA defaults: full list, newton off; build, force and "
                      "integration on the GPU"),
    "kokkos-double-half": ("kokkos-double", _KK + _KK_HALF, "double",
                           "KOKKOS CUDA: half list, newton on (atomic force scatter)"),
    "kokkos-single": ("kokkos-single", _KK, "single",
                      "KOKKOS CUDA (KOKKOS_PREC=single): full list, newton off"),
    "kokkos-single-half": ("kokkos-single", _KK + _KK_HALF, "single",
                           "KOKKOS CUDA (KOKKOS_PREC=single): half list, newton on"),
    "gpu-mixed": ("gpu-mixed", _GPU, "mixed",
                  "GPU package: pair + neighbor list on the GPU (host binning); integration "
                  "and atom sorting on the CPU, positions/forces cross PCIe every step"),
    "gpu-single": ("gpu-single", _GPU, "single", "GPU package (GPU_PREC=single), as gpu-mixed"),
    "cpu-1core": ("gpu-mixed", [], "double", "plain lj/cut on one CPU core (no accelerator)"),
}
#: one accelerated variant per build is used for the identical-state check
VERIFY_VARIANTS = ["kokkos-double", "kokkos-single", "gpu-mixed", "gpu-single"]
CPU_MAX_CELLS = {"quick": 20, "full": 30}
#: pymd variants timed in the same session (backend, variant)
PYMD_QUICK = [("cupy-cuda-c", "full"), ("cupy-cuda-c", "full-fused"), ("triton", "full-fused")]
PYMD_FULL = PYMD_QUICK + [("warp", "full"), ("numba", "full-fused")]


def round_steps(x: float) -> int:
    """Multiple of the rebuild interval, at least one interval."""
    return max(EVERY, int(math.ceil(x / EVERY)) * EVERY)


# ----------------------------------------------------------------------------- LAMMPS
def lammps_exe(build: str) -> Path:
    return LAMMPS_BUILDS / build / f"lmp_{build}"


_LOOP = re.compile(r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms")
_PERF = re.compile(r"Performance:.*?([\d.]+) (k|M|G)?atom-step/s")
_UNIT = {"k": 1e3, "M": 1e6, "G": 1e9, None: 1.0, "": 1.0}


def run_lammps(exe: Path, flags: list[str], infile: Path, variables: dict, cwd: Path,
               timeout: float = 900) -> str:
    cmd = [str(exe), *flags, "-nocite", "-log", "none", "-in", str(infile)]
    for k, v in variables.items():
        cmd += ["-var", k, str(v)]
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0 or "ERROR" in res.stdout:
        tail = "\n".join((res.stdout + res.stderr).strip().splitlines()[-6:])
        raise RuntimeError(f"LAMMPS failed ({res.returncode}): {tail}")
    return res.stdout


def parse_runs(out: str) -> list[dict]:
    """One dict per ``run`` command: loop seconds, steps, atoms and LAMMPS' reported rate."""
    loops = _LOOP.findall(out)
    perfs = _PERF.findall(out)
    rows = []
    for (sec, steps, atoms), (val, unit) in zip(loops, perfs):
        rows.append({"seconds": float(sec), "steps": int(steps), "atoms": int(atoms),
                     "reported": float(val) * _UNIT[unit or None]})
    return rows


def _last_int(pattern: str, out: str) -> float | None:
    m = re.findall(pattern, out)
    return float(m[-1]) if m else None


def bench_lammps(variant: str, cells: int, repeats: int, min_s: float, tmp: Path) -> dict:
    build, flags, prec, neigh = LAMMPS_VARIANTS[variant]
    exe = lammps_exe(build)
    n = 4 * cells**3
    cal = parse_runs(run_lammps(exe, flags, IN_BENCH,
                                {"nc": cells, "nwarm": 40, "nsteps": 100, "nrep": 1}, tmp))
    rate0 = cal[-1]["atoms"] * cal[-1]["steps"] / cal[-1]["seconds"]
    nsteps = round_steps(min_s * rate0 / n)
    out = run_lammps(exe, flags, IN_BENCH,
                     {"nc": cells, "nwarm": 100, "nsteps": nsteps, "nrep": repeats}, tmp)
    timed = parse_runs(out)[1:]  # first run command is the warm-up
    rates = [r["atoms"] * r["steps"] / r["seconds"] for r in timed]
    med = statistics.median(rates)
    return {
        "code": "lammps", "variant": variant, "precision": prec, "n": n, "size": [n],
        "metric": "atom-steps/s", "median": med, "min": min(rates), "max": max(rates),
        "value": med, "repeats": len(rates), "steps_per_repeat": nsteps,
        "rate_source": "N * steps / LAMMPS 'Loop time' of each timed run command",
        "lammps_reported_atom_steps_per_s": [r["reported"] for r in timed],
        "ave_neighs_per_atom": _last_int(r"Ave neighs/atom = ([\d.eE+-]+)", out),
        "neighbor_builds_last_run": _last_int(r"Neighbor list builds = (\d+)", out),
        "neighbor": neigh, "build": build, "flags": " ".join(flags),
        "initial_state": "LAMMPS lattice fcc + velocity create 1.44 87287 loop geom",
    }


def write_lammps_data(path: Path, pos: np.ndarray, vel: np.ndarray, box: np.ndarray) -> None:
    n = len(pos)
    idx = np.arange(1, n + 1)
    with open(path, "w") as fh:
        fh.write(f"pymd identical-state check\n\n{n} atoms\n1 atom types\n\n")
        for lo, hi, ax in zip((0.0,) * 3, box, "xyz"):
            fh.write(f"{lo:.16g} {hi:.16g} {ax}lo {ax}hi\n")
        fh.write("\nMasses\n\n1 1.0\n\nAtoms # atomic\n\n")
        np.savetxt(fh, np.column_stack([idx, np.ones(n), pos]), fmt="%d %d %.17g %.17g %.17g")
        fh.write("\nVelocities\n\n")
        np.savetxt(fh, np.column_stack([idx, vel]), fmt="%d %.17g %.17g %.17g")


def verify_lammps(variant: str, data: Path, tmp: Path) -> dict:
    build, flags, _, _ = LAMMPS_VARIANTS[variant]
    out = run_lammps(lammps_exe(build), flags, IN_VERIFY, {"data": data}, tmp)
    rows = {}
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if line.split()[:1] == ["Step"]:
            for row in lines[i + 1:]:
                parts = row.split()
                if not parts or not parts[0].isdigit():
                    break
                step, temp, epair, etot, press = (float(x) for x in parts[:5])
                rows[int(step)] = {"temp": temp, "epot": epair, "etot": etot,
                                   "pressure": press}
    return {"step0": rows[0], "step100": rows[100]}


def lammps_info() -> dict:
    info = {"source": str(LAMMPS_SRC), "builds": {}}
    try:
        info["commit"] = subprocess.run(["git", "-C", str(LAMMPS_SRC), "rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip()
        info["tag"] = subprocess.run(["git", "-C", str(LAMMPS_SRC), "describe", "--tags"],
                                     capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    keys = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS_RELEASE", "GPU_PREC",
            "GPU_ARCH", "GPU_API", "KOKKOS_PREC", "Kokkos_ARCH_AMPERE86", "Kokkos_ENABLE_CUDA",
            "BUILD_MPI", "BUILD_OMP", "CUDA_NVCC_FLAGS")
    for build in sorted({v[0] for v in LAMMPS_VARIANTS.values()}):
        exe = lammps_exe(build)
        if not exe.exists():
            continue
        b = {"exe": str(exe)}
        cache = exe.parent / "CMakeCache.txt"
        if cache.exists():
            for line in cache.read_text().splitlines():
                k = line.split(":", 1)[0]
                if k in keys and "=" in line:
                    b[k] = line.split("=", 1)[1]
        cmd = exe.parent / "cmake_command.txt"
        if cmd.exists():
            b["cmake"] = cmd.read_text().strip()
        hlp = subprocess.run([str(exe), "-h"], capture_output=True, text=True).stdout
        b["version"] = hlp.splitlines()[1].split(" - ")[-1].strip() if hlp else None
        b["compiler"] = next((ln.split(":", 1)[1].strip() for ln in hlp.splitlines()
                              if ln.startswith("Compiler:")), None)
        acc = hlp.split("Accelerator configuration:", 1)[-1].split("FFT information", 1)[0]
        b["accelerator"] = " | ".join(ln.strip() for ln in acc.splitlines() if ln.strip())
        info["builds"][build] = b
        info["version"] = b["version"]
    return info


# ----------------------------------------------------------------------------- HOOMD-blue
def hoomd_worker(spec_path: str) -> int:
    """Runs inside the HOOMD conda env (numpy + hoomd only); writes JSON to spec['out']."""
    spec = json.loads(Path(spec_path).read_text())
    if spec.get("hoomd_path"):
        sys.path.insert(0, spec["hoomd_path"])
    import hoomd

    dev = hoomd.device.GPU(notice_level=1)
    results = {"hoomd_version": hoomd.version.version, "device": dev.device,
               "precision": list(hoomd.version.floating_point_precision),
               "gpu_api_version": hoomd.version.gpu_api_version,
               "compile_flags": hoomd.version.compile_flags, "module": hoomd.__file__,
               "verify": None, "runs": []}

    def make_sim(npz: str, pressure: bool = False, every: int | None = None):
        d = np.load(npz)
        pos, vel, box = d["pos"], d["vel"], d["box"]
        snap = hoomd.Snapshot()
        snap.configuration.box = [*box, 0, 0, 0]
        snap.particles.N = len(pos)
        snap.particles.types = ["A"]
        snap.particles.position[:] = pos - box / 2
        snap.particles.velocity[:] = vel
        snap.particles.typeid[:] = 0
        snap.particles.mass[:] = 1.0
        sim = hoomd.Simulation(device=dev, seed=1)
        sim.create_state_from_snapshot(snap)
        nl = hoomd.md.nlist.Cell(buffer=spec["skin"],
                                 rebuild_check_delay=every or spec["every"], check_dist=False)
        lj = hoomd.md.pair.LJ(nlist=nl, default_r_cut=spec["rc"], mode="none")
        lj.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        sim.operations.integrator = hoomd.md.Integrator(dt=spec["dt"], methods=[nve],
                                                        forces=[lj])
        thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        sim.operations.computes.append(thermo)
        sim.always_compute_pressure = pressure
        return sim, thermo, nl

    def thermo_dict(th) -> dict:
        n = th.num_particles
        return {"temp": th.kinetic_temperature, "epot": th.potential_energy / n,
                "etot": (th.potential_energy + th.kinetic_energy) / n,
                "pressure": th.pressure, "dof": th.translational_degrees_of_freedom}

    if spec.get("verify"):
        sim, th, _ = make_sim(spec["verify"], pressure=True)
        sim.run(0)
        s0 = thermo_dict(th)
        sim.run(100)
        results["verify"] = {"step0": s0, "step100": thermo_dict(th)}
        del sim
        sim, th, _ = make_sim(spec["verify"], pressure=True, every=1)  # list never stale
        sim.run(0)
        sim.run(100)
        results["verify"]["step100_rebuild_every_1"] = thermo_dict(th)
        del sim
    for case in spec["cases"]:
        try:
            sim, _, nl = make_sim(case["npz"])
            n = case["n"]
            sim.run(100)
            tune_steps = 100
            t_tune = time.perf_counter()
            while (not sim.operations.is_tuning_complete and tune_steps < 20000
                   and time.perf_counter() - t_tune < spec["tune_seconds"]):
                sim.run(200)
                tune_steps += 200
            t = time.perf_counter()
            sim.run(100)
            per_step = (time.perf_counter() - t) / 100
            nsteps = round_steps(spec["min_s"] / per_step)
            rates, tps = [], []
            for _ in range(spec["repeats"]):
                t = time.perf_counter()
                sim.run(nsteps)
                dt = time.perf_counter() - t
                rates.append(n * nsteps / dt)
                tps.append(sim.tps)
            results["runs"].append({
                "n": n, "rates": rates, "steps_per_repeat": nsteps, "hoomd_tps": tps,
                "tuning_complete": bool(sim.operations.is_tuning_complete),
                "warmup_steps": tune_steps + 100,
                "builds_per_step": nl.num_builds / nsteps})  # num_builds counts the last run
            del sim
        except Exception as exc:  # pragma: no cover - reporting only
            results["runs"].append({"n": case["n"], "error": repr(exc)[:300]})
        Path(spec["out"]).write_text(json.dumps(results))
    Path(spec["out"]).write_text(json.dumps(results))
    return 0


def run_hoomd(variant: str, cases: list[tuple[int, Path]], verify_npz: Path | None,
              repeats: int, min_s: float, tmp: Path, quick: bool) -> dict:
    python, path, _, _ = HOOMD_VARIANTS[variant]
    spec = {"cases": [{"n": n, "npz": str(p)} for n, p in cases],
            "hoomd_path": str(path) if path else None,
            "verify": str(verify_npz) if verify_npz else None, "repeats": repeats,
            "min_s": min_s, "skin": SKIN, "rc": RC, "dt": DT, "every": EVERY,
            "tune_seconds": 10.0 if quick else 60.0,
            "out": str(tmp / f"hoomd_{variant}.json")}
    sp = tmp / f"hoomd_spec_{variant}.json"
    sp.write_text(json.dumps(spec))
    res = subprocess.run([str(python), "-I", str(Path(__file__).resolve()),
                          "--hoomd-worker", str(sp)], capture_output=True, text=True,
                         timeout=3600)
    out = Path(spec["out"])
    if not out.exists():
        raise RuntimeError(f"HOOMD worker failed: {(res.stdout + res.stderr)[-600:]}")
    return json.loads(out.read_text())


def hoomd_package() -> dict:
    meta = sorted((HOOMD_PYTHON.parents[1] / "conda-meta").glob("hoomd-*.json"))
    if not meta:
        return {}
    m = json.loads(meta[-1].read_text())
    return {"conda_package": f"{m['name']}={m['version']}={m['build']}",
            "channel": m.get("channel"), "env": str(HOOMD_PYTHON.parents[1])}


def hoomd_source() -> dict:
    if not (HOOMD_SRC / ".git").exists():
        return {}
    git = ["git", "-C", str(HOOMD_SRC)]
    out = {"path": str(HOOMD_SRC),
           "commit": subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
           "tag": subprocess.run([*git, "describe", "--tags"], capture_output=True,
                                 text=True).stdout.strip()}
    cmd = HOOMD_SRC / "build-single" / "cmake_command.txt"
    if cmd.exists():
        out["cmake"] = cmd.read_text().strip()
    return out


# ----------------------------------------------------------------------------- driver
def save_state(path: Path, pos, vel, box) -> Path:
    np.savez(path, pos=pos, vel=vel, box=box)
    return path


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def verify_all(tmp: Path, log) -> dict:
    """Identical initial state in every code; thermo at steps 0 and 100 vs pymd float64."""
    from pymd import LJParams, MDConfig, backends
    from pymd.system import fcc_lattice, maxwell_velocities

    pos, box = fcc_lattice(VERIFY_CELLS, density=DENSITY)
    vel = maxwell_velocities(len(pos), T0, seed=1, dist="uniform")
    rows = []
    params = LJParams(rc=RC, shift=False)
    cfg = MDConfig(dt=DT, skin=SKIN, rebuild_every=EVERY, check=False)
    for v in ("full-fp64", "full"):
        try:
            eng = backends.get("cupy-cuda-c")(pos=pos, vel=vel, box=box, params=params,
                                              cfg=cfg, variant=v)
            s0 = eng.thermo()
            eng.run(100)
            rows.append({"code": "pymd", "variant": f"cupy-cuda-c/{v}", "step0": s0,
                         "step100": eng.thermo()})
        except Exception as exc:
            rows.append({"code": "pymd", "variant": f"cupy-cuda-c/{v}", "error": repr(exc)})
    try:
        cfg1 = MDConfig(dt=DT, skin=SKIN, rebuild_every=1, check=False)
        eng = backends.get("cupy-cuda-c")(pos=pos, vel=vel, box=box, params=params, cfg=cfg1,
                                          variant="full-fp64")
        eng.run(100)
        every1 = {k: float(v) for k, v in eng.thermo().items()}
    except Exception:
        every1 = None
    data = tmp / "verify.data"
    write_lammps_data(data, pos, vel, box)
    for v in VERIFY_VARIANTS:
        if not lammps_exe(LAMMPS_VARIANTS[v][0]).exists():
            continue
        try:
            rows.append({"code": "lammps", "variant": v, **verify_lammps(v, data, tmp)})
        except Exception as exc:
            rows.append({"code": "lammps", "variant": v, "error": repr(exc)[:300]})
    ref = rows[0] if "error" not in rows[0] else None
    keys = ("temp", "epot", "etot", "pressure")
    for r in rows:
        if ref is None or "error" in r:
            continue
        for s in ("step0", "step100"):
            r[s] = {k: float(r[s][k]) for k in (*keys, "dof") if k in r[s]}
            r[f"rel_diff_{s}"] = {k: rel(r[s][k], ref[s][k]) for k in keys}
        log(f"verify {r['code']:7s} {r['variant']:24s} step100 rel diff "
            + " ".join(f"{k}={r['rel_diff_step100'][k]:.1e}" for k in keys))
    return {"n": len(pos), "reference": "pymd cupy-cuda-c/full-fp64",
            "state": "fcc_lattice(20) in lattice order, uniform velocities seed 1, T = 1.44",
            "reference_step100_rebuild_every_1": every1,
            "rows": rows, "_pos": pos, "_vel": vel, "_box": box}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="smoke run at N = 32,000")
    g.add_argument("--full", action="store_true", help="publishable run (idle GPU)")
    g.add_argument("--hoomd-worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--only", default=None, help="comma list of codes: pymd,lammps,hoomd")
    ap.add_argument("--provisional", action="store_true", help="mark a --full run provisional")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "counterparts.json")
    args = ap.parse_args(argv)
    if args.hoomd_worker:
        return hoomd_worker(args.hoomd_worker)

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "bench"))
    import run as pymd_bench  # bench/run.py
    from pymd import backends
    from pymd.system import fcc_lattice, maxwell_velocities

    mode = "quick" if args.quick else "full"
    cells = QUICK_CELLS if args.quick else FULL_CELLS
    repeats, min_s = 5, (0.1 if args.quick else 0.5)
    only = set(args.only.split(",")) if args.only else {"pymd", "lammps", "hoomd"}
    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t_start:6.0f} s] {msg}", flush=True)

    runs: list[dict] = []
    notes: list[str] = []

    def add(rec: dict) -> None:
        runs.append(rec)
        if "error" in rec:
            log(f"{rec['code']:7s} {rec['variant']:20s} N={rec['n']:>8d} FAILED {rec['error']}")
        else:
            log(f"{rec['code']:7s} {rec['variant']:20s} N={rec['n']:>8d} "
                f"{rec['median'] / 1e6:9.2f} Matom-steps/s (min {rec['min'] / 1e6:.2f}, "
                f"max {rec['max'] / 1e6:.2f})")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ver = verify_all(tmp, log)
        vnpz = save_state(tmp / "verify.npz", ver.pop("_pos"), ver.pop("_vel"), ver.pop("_box"))

        # --- HOOMD-blue (separate env, one process per variant for the whole ladder)
        hoomd_meta: dict = {}
        if "hoomd" in only:
            cases = []
            for c in cells:
                pos, box = fcc_lattice(c, density=DENSITY, shuffle=True, seed=c)
                vel = maxwell_velocities(len(pos), T0, seed=87287, dist="uniform")
                cases.append((len(pos), save_state(tmp / f"bench_{c}.npz", pos, vel, box)))
            for hv, (python, path, prec, desc) in HOOMD_VARIANTS.items():
                if not python.exists() or (path is not None and not path.exists()):
                    notes.append(f"HOOMD-blue {hv} skipped: {python} / {path} not found")
                    log(notes[-1])
                    continue
                try:
                    h = run_hoomd(hv, cases, vnpz, repeats, min_s, tmp, args.quick)
                except Exception as exc:
                    notes.append(f"HOOMD-blue {hv} skipped: {str(exc)[-400:]}")
                    log(notes[-1])
                    continue
                hoomd_meta[hv] = {k: h[k] for k in ("hoomd_version", "device", "precision",
                                                    "gpu_api_version", "compile_flags",
                                                    "module")} | {"description": desc}
                if h.get("verify"):
                    ref = ver["rows"][0]
                    r = {"code": "hoomd", "variant": hv, **h["verify"]}
                    for st in ("step0", "step100"):
                        r[f"rel_diff_{st}"] = {k: rel(r[st][k], ref[st][k])
                                               for k in ("temp", "epot", "etot", "pressure")}
                    e1, e1ref = r.get("step100_rebuild_every_1"), ver.get(
                        "reference_step100_rebuild_every_1")
                    if e1 and e1ref:
                        r["rel_diff_step100_rebuild_every_1"] = {
                            k: rel(e1[k], e1ref[k]) for k in ("temp", "epot", "etot", "pressure")}
                    ver["rows"].append(r)
                    log(f"verify hoomd   {hv:24s} step100 rel diff " + " ".join(
                        f"{k}={v:.1e}" for k, v in r["rel_diff_step100"].items()))
                for hr in h["runs"]:
                    base = {"code": "hoomd", "variant": hv, "precision": prec, "n": hr["n"],
                            "size": [hr["n"]]}
                    if "error" in hr:
                        add({**base, "error": hr["error"]})
                        continue
                    med = statistics.median(hr["rates"])
                    add({**base, "metric": "atom-steps/s", "median": med, "value": med,
                         "min": min(hr["rates"]), "max": max(hr["rates"]),
                         "repeats": len(hr["rates"]), "steps_per_repeat": hr["steps_per_repeat"],
                         "rate_source": "N * steps / wall time of Simulation.run",
                         "hoomd_tps": hr["hoomd_tps"], "tuning_complete": hr["tuning_complete"],
                         "warmup_steps": hr["warmup_steps"],
                         "builds_per_step": hr["builds_per_step"],
                         "neighbor": "md.nlist.Cell(buffer=0.3, rebuild_check_delay=20, "
                                     "check_dist=False); autotuned kernels; default sorter",
                         "initial_state": "pymd fcc_lattice (shuffled) + uniform velocities"})

        # --- LAMMPS
        if "lammps" in only:
            for v, (build, *_rest) in LAMMPS_VARIANTS.items():
                if not lammps_exe(build).exists():
                    notes.append(f"LAMMPS {v} skipped: {lammps_exe(build)} not built")
                    log(notes[-1])
                    continue
                for c in cells:
                    if v == "cpu-1core" and c > CPU_MAX_CELLS[mode]:
                        continue
                    try:
                        add(bench_lammps(v, c, repeats, min_s, tmp))
                    except Exception as exc:
                        add({"code": "lammps", "variant": v, "n": 4 * c**3, "size": [4 * c**3],
                             "error": repr(exc)[:300]})

    # --- pymd, same session
    if "pymd" in only:
        ceiling = {"copy_bandwidth_GBs": pymd_bench.COORDINATOR_COPY_GBS,
                   "spec_bandwidth_GBs": 448.0, "fp32_tflops": None}
        avail = backends.available()
        for b, v in PYMD_QUICK if args.quick else PYMD_FULL:
            if b not in avail:
                continue
            for c in cells:
                try:
                    r = pymd_bench.bench_one(b, v, c, repeats, min_s, ceiling)
                    add({**r, "code": "pymd", "variant": f"{b}/{v}", "backend": b,
                         "precision": "single", "n": r["size"][0],
                         "neighbor": "full list, GPU cell list (counting sort), atoms sorted "
                                     "at each rebuild",
                         "initial_state": "pymd fcc_lattice (shuffled) + uniform velocities"})
                except Exception as exc:
                    add({"code": "pymd", "variant": f"{b}/{v}", "n": 4 * c**3,
                         "size": [4 * c**3], "error": repr(exc)[:300]})
                pymd_bench._free_gpu()

    machine = pymd_bench.machine_info()
    for tool, cmd in (("nvcc", ["/usr/local/cuda/bin/nvcc", "--version"]),
                      ("gcc", ["gcc", "--version"])):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
            machine[tool] = out.splitlines()[-1] if tool == "nvcc" else out.splitlines()[0]
        except Exception:
            machine[tool] = None
    doc = {
        "provisional": bool(args.quick or args.provisional),
        "mode": mode,
        "benchmark": "LAMMPS bench/in.lj melt: fcc 0.8442, T0 1.44, lj/cut 2.5 (no shift), "
                     "NVE dt 0.005, skin 0.3, rebuild every 20 steps (no check)",
        "settings": {"density": DENSITY, "t0": T0, "rc": RC, "skin": SKIN, "dt": DT,
                     "rebuild_every": EVERY, "check": False, "cells": cells,
                     "repeats": repeats, "min_seconds_per_repeat": min_s},
        "machine": machine,
        "codes": {"lammps": lammps_info() if "lammps" in only else None,
                  "hoomd": {"conda": hoomd_package(), "source": hoomd_source(),
                            "variants": hoomd_meta} if "hoomd" in only else None,
                  "pymd": {"source": "this repository", "variants": [
                      f"{b}/{v}" for b, v in (PYMD_QUICK if args.quick else PYMD_FULL)]}},
        "lammps_variants": {k: {"build": b, "flags": " ".join(f), "precision": p,
                                "neighbor": nb} for k, (b, f, p, nb) in LAMMPS_VARIANTS.items()},
        "verify": ver,
        "notes": notes,
        "wall_seconds": time.perf_counter() - t_start,
        "runs": runs,
    }
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, default=float))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
