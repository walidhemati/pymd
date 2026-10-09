"""Same-machine head-to-head against the C++ predecessor yamd (github.com/walidhemati/yamd).

yamd is compiled from its own sources (single thread, ``g++ -O3 -march=native``, Eigen 3.4.0)
together with ``bench/yamd/yamd_driver.cpp``. yamd's meson build needs MPI and a vendored Eigen
that is incomplete in the upstream repository, so the MPI-free library sources are compiled
directly. Both codes then run the *same* open-boundary systems:

* LJ: FCC blocks at rho* = 0.8442, truncated-shifted LJ, rc = 2.5 sigma, dt = 0.005 tau.
* EAM: Mackay icosahedral Au clusters with yamd's Gupta parameters (cutoff 10 Angstrom), dt = 1 fs.

Open boundaries are emulated in pymd by a periodic box with >= 2 r_list of vacuum.

usage: python bench/yamd_compare.py [--yamd-src PATH] [--eigen PATH] [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pymd import LJParams, MDConfig, backends  # noqa: E402
from pymd.eam import AU_GUPTA, AU_MASS, gupta_allpairs, mackay_icosahedron  # noqa: E402
from pymd.reference import lj_allpairs  # noqa: E402
from pymd.system import fcc_lattice, maxwell_velocities  # noqa: E402

YAMD_SOURCES = ["lj.cpp", "lj_direct_summation.cpp", "neighbors.cpp", "ducastelle.cpp",
                "verlet.cpp"]


def build_yamd(src: Path, eigen: Path, out: Path) -> Path:
    """Compile yamd's library sources and the driver into ``out``/yamd_driver."""
    exe = out / "yamd_driver"
    cmd = ["g++", "-O3", "-march=native", "-DNDEBUG", "-std=c++17", f"-I{src / 'src'}",
           f"-I{eigen}", *[str(src / "src" / f) for f in YAMD_SOURCES],
           str(ROOT / "bench" / "yamd" / "yamd_driver.cpp"), "-o", str(exe)]
    subprocess.run(cmd, check=True)
    return exe


def run_yamd(exe: Path, mode: str, pos: np.ndarray, vel: np.ndarray, nsteps: int, tmp: Path):
    """Run the driver; returns (timing dict, forces at t = 0)."""
    inp = tmp / f"{mode}.bin"
    with open(inp, "wb") as fh:
        fh.write(np.int64(len(pos)).tobytes())
        fh.write(np.ascontiguousarray(pos, dtype=np.float64).tobytes())  # Eigen 3xN col-major
        fh.write(np.ascontiguousarray(vel, dtype=np.float64).tobytes())
    res = subprocess.run([str(exe), mode, str(inp), str(nsteps), str(tmp / mode)],
                         check=True, capture_output=True, text=True)
    info = json.loads(res.stdout.strip().splitlines()[-1])
    f = np.fromfile(tmp / f"{mode}.forces").reshape(len(pos), 3)
    return info, f


def time_yamd(exe, mode, pos, vel, tmp, min_seconds: float) -> dict:
    info, _ = run_yamd(exe, mode, pos, vel, 2, tmp)
    per = max(info["seconds"] / 2, 1e-6)
    nsteps = int(np.clip(min_seconds / per, 3, 20000))
    info, _ = run_yamd(exe, mode, pos, vel, nsteps, tmp)
    info["atom_steps_per_s"] = info["n"] * info["nsteps"] / info["seconds"]
    return info


def time_pymd(eng, min_seconds: float, warm: int = 20) -> float:
    """atom-steps/s of ``eng.run`` (warm-up excluded)."""
    eng.run(warm)
    eng.synchronize()
    nsteps = 10
    while True:
        t = time.perf_counter()
        eng.run(nsteps)
        eng.synchronize()
        dt = time.perf_counter() - t
        if dt >= min_seconds or nsteps > 10**6:
            return eng.n * nsteps / dt
        nsteps = int(nsteps * max(2.0, 1.2 * min_seconds / max(dt, 1e-4)))


def in_vacuum(pos: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    lo = pos.min(0)
    ext = pos.max(0) - lo
    box = ext + 2 * margin
    return pos - lo + margin, box


def lj_cases(exe, tmp, sizes, min_seconds, gpu_backends) -> list[dict]:
    out = []
    params = LJParams(rc=2.5, shift=True)
    for nc in sizes:
        pos0, _ = fcc_lattice(nc)
        vel = maxwell_velocities(len(pos0), 1.44, seed=1)
        n = len(pos0)
        _, fy = run_yamd(exe, "lj", pos0, vel, 0, tmp)
        fref, _, _ = lj_allpairs(pos0, None, params) if n <= 8000 else (None, 0, 0)
        row = {"system": "lj-fcc-block", "n": n}
        if fref is not None:
            row["max_abs_force_diff_yamd_vs_pymd_fp64"] = float(np.abs(fy - fref).max())
            row["max_abs_force"] = float(np.abs(fref).max())
        y = time_yamd(exe, "lj", pos0, vel, tmp, min_seconds)
        row["yamd_cpp_1thread"] = y["atom_steps_per_s"]
        for label, cfg in {
            "same-algorithm": MDConfig(skin=0.0, rebuild_every=1),
            "pymd-default": MDConfig(skin=0.3, rebuild_every=20),
        }.items():
            pos, box = in_vacuum(pos0, 2 * (2.5 + cfg.skin) + 1.0)
            cands = [("numpy", "cell-verlet")] + gpu_backends
            for b, v in cands:
                try:
                    eng = backends.get(b)(pos=pos, vel=vel, box=box, params=params, cfg=cfg,
                                          variant=v)
                    row[f"{b}/{v}/{label}"] = time_pymd(eng, min_seconds)
                except Exception as exc:  # pragma: no cover - reporting only
                    row[f"{b}/{v}/{label}"] = f"error: {exc!r}"[:200]
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def eam_cases(exe, tmp, layers, min_seconds) -> list[dict]:
    out = []
    try:
        from pymd.eam import make_gupta_engine
    except Exception:  # pragma: no cover
        return out
    for nl in layers:
        pos0 = mackay_icosahedron(nl)
        n = len(pos0)
        vel = np.zeros_like(pos0)
        _, fy = run_yamd(exe, "eam", pos0, vel, 0, tmp)
        row = {"system": "au-mackay", "layers": nl, "n": n}
        if n <= 3000:
            fref, _ = gupta_allpairs(pos0, None, AU_GUPTA)
            row["max_abs_force_diff_yamd_vs_pymd_fp64"] = float(np.abs(fy - fref).max())
            row["max_abs_force"] = float(np.abs(fref).max())
        y = time_yamd(exe, "eam", pos0, vel, tmp, min_seconds)
        row["yamd_cpp_1thread"] = y["atom_steps_per_s"]
        for label, cfg in {
            "same-algorithm": MDConfig(dt=1.0, skin=0.0, rebuild_every=1, mass=AU_MASS),
            "pymd-default": MDConfig(dt=1.0, skin=1.0, rebuild_every=10, check=True,
                                     mass=AU_MASS),
        }.items():
            pos, box = in_vacuum(pos0, 2 * (10.0 + cfg.skin) + 1.0)
            try:
                eng = make_gupta_engine(pos, vel, box, AU_GUPTA, cfg)
                if n <= 3000 and label == "same-algorithm":
                    f, _, _ = eng.forces()
                    row["max_abs_force_diff_gpu_fp32_vs_fp64"] = float(np.abs(f - fref).max())
                row[f"cupy-cuda-c/gupta/{label}"] = time_pymd(eng, min_seconds)
            except Exception as exc:  # pragma: no cover
                row[f"cupy-cuda-c/gupta/{label}"] = f"error: {exc!r}"[:200]
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_ref = ROOT.parent / "_ref"
    ap.add_argument("--yamd-src", type=Path, default=default_ref / "yamd")
    ap.add_argument("--eigen", type=Path, default=default_ref / "eigen" / "eigen-3.4.0")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "yamd-compare.json")
    args = ap.parse_args(argv)
    if not (args.yamd_src / "src" / "lj.cpp").exists():
        print(f"yamd sources not found at {args.yamd_src}; skipping (git clone "
              "https://github.com/walidhemati/yamd)")
        return 0
    if not (args.eigen / "Eigen" / "Core").exists():
        print(f"Eigen 3.4.0 not found at {args.eigen}; extract yamd/subprojects/packagecache/"
              "eigen-3.4.0.tar.bz2 there")
        return 0
    min_s = 0.3 if args.quick else 0.6
    gpu = [(b, "full") for b in ("triton", "warp", "numba", "cupy-cuda-c")
           if b in backends.available()]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        exe = build_yamd(args.yamd_src, args.eigen, tmp)
        lj = lj_cases(exe, tmp, [5, 8] if args.quick else [5, 8, 12, 16, 20], min_s, gpu)
        eam = eam_cases(exe, tmp, [4] if args.quick else [4, 6, 8, 10], min_s)
    rev = subprocess.run(["git", "-C", str(args.yamd_src), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    cpu = ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                   if line.startswith("model name"))
    except Exception:
        cpu = platform.processor()
    doc = {"provisional": True, "yamd_commit": rev, "cpu": cpu,
           "compiler": subprocess.run(["g++", "--version"], capture_output=True,
                                      text=True).stdout.splitlines()[0],
           "flags": "-O3 -march=native -DNDEBUG, single thread",
           "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "lj": lj, "eam": eam}
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
