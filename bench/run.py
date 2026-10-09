"""The only benchmark entry point (Python-HPC suite contract).

Benchmark problem: the LAMMPS ``bench/in.lj`` Lennard-Jones melt -- FCC lattice at reduced
density 0.8442, initial T* = 1.44, truncated LJ with rc = 2.5 sigma, NVE, dt = 0.005 tau,
neighbor skin 0.3 sigma, list rebuilt every 20 steps without a displacement check -- scaled from
4,000 to 4,000,000 atoms. The atom order is randomly shuffled at t = 0 so that spatial sorting
has to earn its keep.

Metric: atom-steps per second (LAMMPS' "atom-step/s"); also tau/day and ns/day (argon mapping).

    python bench/run.py --quick   # < 2 min smoke run  -> results/quick.json (provisional)
    python bench/run.py --full    # publishable run    -> results/rtx3070-laptop.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from pymd import LJParams, MDConfig, backends  # noqa: E402
from pymd.system import (  # noqa: E402
    fcc_lattice,
    maxwell_velocities,
    ns_per_day,
    tau_per_day,
)

COORDINATOR_COPY_GBS = 361.0  # SPEC.md: measured torch copy_ on this GPU, 1 GiB, idle

#: FCC cells per side -> 4 n^3 atoms
FULL_CELLS = [10, 20, 30, 40, 50, 64, 80, 100]  # 4k ... 4M
QUICK_CELLS = [10, 20]

#: (backend, variant, max cells per side) of the main ladder
MAIN = [
    ("numpy", "cell-verlet", 20),
    ("torch-eager", "gather", 50),
    ("torch-compile", "gather", 64),
    ("triton", "full", 100),
    ("triton", "full-fused", 100),
    ("warp", "full", 100),
    ("warp", "hashgrid", 100),
    ("numba", "full", 100),
    ("numba", "full-fused", 100),
    ("cupy-cuda-c", "full", 100),
    ("cupy-cuda-c", "full-fused", 100),
    ("cupy-cuda-c", "half-atomic", 100),
    ("cupy-cuda-c", "allpairs", 30),
]
#: ablations run at two sizes only
ABLATION = [
    ("cupy-cuda-c", "full-morton"),
    ("cupy-cuda-c", "full-tpa2"),
    ("cupy-cuda-c", "full-tpa4"),
    ("cupy-cuda-c", "full-nosort"),
    ("cupy-cuda-c", "full-mixed"),
    ("cupy-cuda-c", "full-fp64"),
    ("triton", "half-atomic"),
    ("warp", "hashgrid-verlet"),
]
ABLATION_CELLS = [40, 80]

FLOP_ENTRY = 20  # per neighbor-list entry: 3 sub, 3x min-image (mul, rint, fma), r2 (mul+2 fma)
FLOP_PAIR = 13  # per pair inside rc: rcp, 2 mul (r^-6), fma + 2 mul (f/r), 3 fma (accumulate)


def traffic_model(variant: str, n_list: float, rho: float, rc: float, rlist: float,
                  n_atoms: int) -> tuple[float | None, float]:
    """Minimal bytes and flops per atom-step (stated in the paper).

    Bytes: compulsory DRAM traffic of one Verlet-list step -- read x, read+write v, write x
    (4 x 3 words) plus one 4-byte index per stored list entry. Neighbor positions are assumed
    to come from cache. Rebuild traffic is ignored (never inflates the effective bandwidth).
    """
    word = 8 if variant.endswith("fp64") or variant.endswith("mixed") else 4
    frac_in = (rc / rlist) ** 3
    if variant == "allpairs":
        n_in = rho * 4.0 / 3.0 * np.pi * rc**3
        return None, (n_atoms - 1) * FLOP_ENTRY + n_in * FLOP_PAIR
    if variant == "hashgrid":
        # only pairs inside rc are counted (the cells visited per query are not known exactly)
        n_in = rho * 4.0 / 3.0 * np.pi * rc**3
        return None, n_in * (FLOP_ENTRY + FLOP_PAIR)
    bytes_ = 12 * word + 4 * n_list
    flops = n_list * (FLOP_ENTRY + frac_in * (FLOP_PAIR + (3 if "half" in variant else 0)))
    return bytes_, flops


def machine_info() -> dict:
    info = {"python": platform.python_version(), "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for mod in ("torch", "triton", "warp", "cupy", "numba", "numpy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        q = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        q = q.strip().split(", ")
        info["gpu"], info["driver"] = q[0], q[1]
    except Exception:
        info["gpu"], info["driver"] = None, None
    try:
        info["cpu"] = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                           if line.startswith("model name"))
    except Exception:
        info["cpu"] = platform.processor()
    return info


def bench_one(backend: str, variant: str, cells: int, repeats: int, min_seconds: float,
              ceiling: dict) -> dict:
    """Time one (backend, variant, size); returns a run record of the JSON contract."""
    params = LJParams(rc=2.5, shift=False)
    cfg = MDConfig(dt=0.005, skin=0.3, rebuild_every=20, check=False)
    pos, box = fcc_lattice(cells, density=0.8442, shuffle=True, seed=cells)
    n = len(pos)
    vel = maxwell_velocities(n, 1.44, seed=87287, dist="uniform")
    eng = backends.get(backend)(pos=pos, vel=vel, box=box, params=params, cfg=cfg,
                                variant=variant)
    eng.run(40)  # JIT/compile + first neighbor builds
    eng.synchronize()
    t0 = time.perf_counter()
    eng.run(20)
    eng.synchronize()
    per_step = (time.perf_counter() - t0) / 20
    nsteps = max(20, int(np.ceil(min_seconds / per_step / 20)) * 20)
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        eng.run(nsteps)
        eng.synchronize()
        rates.append(n * nsteps / (time.perf_counter() - t0))
    th = eng.thermo()
    n_list = eng.mean_neighbors()
    rho = n / float(np.prod(box))
    b, fl = traffic_model(variant, n_list if np.isfinite(n_list) else 0.0, rho, params.rc,
                          params.rc + cfg.skin, n)
    med = statistics.median(rates)
    rec = {
        "backend": backend, "variant": variant, "size": [n],
        "dtype": "float64" if variant.endswith("fp64") or backend == "numpy" else (
            "mixed" if variant.endswith("mixed") else "float32"),
        "metric": "atom-steps/s", "value": med, "repeats": repeats, "median": med,
        "min": min(rates), "max": max(rates), "steps_per_repeat": nsteps,
        "steps_per_s": med / n, "tau_per_day": tau_per_day(med / n, cfg.dt),
        "ns_per_day_argon": ns_per_day(med / n, cfg.dt),
        "mean_list_entries_per_atom": None if not np.isfinite(n_list) else n_list,
        "bytes_per_unit": b, "flops_per_unit": fl,
        "effective_GBs": None, "pct_copy_bw": None, "pct_spec_bw": None,
        "pct_peak_flops": None,
        "final_temp": th["temp"], "final_epot": th["epot"], "final_pressure": th["pressure"],
        "neighbor_builds": eng.n_builds,
    }
    if b is not None:
        gbs = med * b / 1e9
        rec["effective_GBs"] = gbs
        rec["pct_copy_bw"] = 100 * gbs / ceiling["copy_bandwidth_GBs"]
        rec["pct_spec_bw"] = 100 * gbs / ceiling["spec_bandwidth_GBs"]
    if fl and ceiling.get("fp32_tflops"):
        peak = ceiling["fp32_tflops"] * 1e12
        if rec["dtype"] == "float64":
            rec["notes"] = "fp64 run; % of fp32 FMA peak shown for reference only"
        rec["pct_peak_flops"] = 100 * med * fl / peak
    del eng
    gc.collect()
    _free_gpu()
    return rec


def _free_gpu() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from pymd.backends.cupy_backend import cp

        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pymd benchmark (LAMMPS in.lj melt)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="< 2 min smoke run")
    g.add_argument("--full", action="store_true", help="publishable run")
    ap.add_argument("--only", default=None, help="comma-separated backends to run")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--provisional", action="store_true",
                    help="mark a --full run as provisional (dev machine, shared GPU)")
    ap.add_argument("--yamd", action="store_true", help="also run bench/yamd_compare.py")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="dev only: cap FCC cells per side (memory-limited shared GPU)")
    args = ap.parse_args(argv)

    import roofline

    ceiling_meas = roofline.measure(quick=args.quick)
    ceiling = {
        "copy_bandwidth_GBs": COORDINATOR_COPY_GBS,
        "spec_bandwidth_GBs": roofline.SPEC_BANDWIDTH_GBS,
        "fp32_tflops": ceiling_meas.get("fp32_tflops"),
        "measured_copy_bandwidth_GBs_this_run": ceiling_meas.get("copy_bandwidth_GBs"),
    }
    avail = backends.available()
    only = set(args.only.split(",")) if args.only else None
    runs = []
    if args.quick:
        plan = [(b, v, c) for b, v, _ in MAIN for c in QUICK_CELLS
                if c <= 20 and not (v == "allpairs" and c > 10)]
        plan = [p for p in plan if p[0] != "numpy" or p[2] == 10]
        repeats, min_s = 5, 0.1
    else:
        plan = [(b, v, c) for b, v, cmax in MAIN for c in FULL_CELLS if c <= cmax]
        plan += [(b, v, c) for b, v in ABLATION for c in ABLATION_CELLS]
        repeats, min_s = 5, 0.5
    plan = [p for p in plan if p[0] in avail and (only is None or p[0] in only)]
    if args.max_cells:
        plan = [p for p in plan if p[2] <= args.max_cells]
    t_start = time.perf_counter()
    for b, v, c in plan:
        try:
            rec = bench_one(b, v, c, repeats, min_s, ceiling)
            runs.append(rec)
            print(f"{b:14s} {v:16s} N={rec['size'][0]:>8d} {rec['median'] / 1e6:9.2f} "
                  f"Matom-step/s  ({rec['pct_copy_bw'] or 0:5.1f}% copy BW)  "
                  f"[{time.perf_counter() - t_start:6.0f} s]", flush=True)
        except Exception as exc:
            msg = repr(exc)[:300]
            print(f"{b:14s} {v:16s} cells={c}: FAILED {msg}", flush=True)
            runs.append({"backend": b, "variant": v, "size": [4 * c**3], "error": msg})
            _free_gpu()
    doc = {
        "provisional": bool(args.quick or args.provisional),
        "benchmark": "LAMMPS bench/in.lj melt: fcc 0.8442, T0 1.44, lj/cut 2.5, skin 0.3, "
                     "rebuild every 20 steps (no check), NVE, dt 0.005, shuffled atom order",
        "machine": machine_info(),
        "ceiling": ceiling,
        "unavailable_backends": backends.unavailable(),
        "wall_seconds": time.perf_counter() - t_start,
        "runs": runs,
    }
    out = args.out or ROOT / "results" / ("quick.json" if args.quick else "rtx3070-laptop.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out} ({doc['wall_seconds']:.0f} s)")
    if args.yamd:
        import yamd_compare

        yamd_compare.main(["--quick"] if args.quick else [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
