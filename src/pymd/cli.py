"""Command line interface: ``pymd run | bench | validate | backends``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import backends
from .engine import Berendsen, Langevin
from .system import LJParams, MDConfig, fcc_lattice, maxwell_velocities, ns_per_day


def _make(args):
    pos, box = fcc_lattice(args.cells, density=args.density, shuffle=True)
    vel = maxwell_velocities(len(pos), args.temp, seed=args.seed)
    params = LJParams(rc=args.rc, shift=not args.no_shift)
    cfg = MDConfig(dt=args.dt, skin=args.skin, rebuild_every=args.every, check=args.check)
    return backends.get(args.backend)(pos=pos, vel=vel, box=box, params=params, cfg=cfg,
                                      variant=args.variant)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="cupy-cuda-c", choices=backends.names())
    p.add_argument("--variant", default=None)
    p.add_argument("--cells", type=int, default=20, help="FCC cells per side (N = 4 cells^3)")
    p.add_argument("--density", type=float, default=0.8442)
    p.add_argument("--temp", type=float, default=1.44)
    p.add_argument("--rc", type=float, default=2.5)
    p.add_argument("--no-shift", action="store_true", help="do not shift the LJ energy")
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--skin", type=float, default=0.3)
    p.add_argument("--every", type=int, default=20, help="neighbor rebuild interval")
    p.add_argument("--check", action="store_true", help="rebuild only if atoms moved > skin/2")
    p.add_argument("--seed", type=int, default=87287)


def cmd_run(args) -> int:
    eng = _make(args)
    th = None
    if args.thermostat == "berendsen":
        th = Berendsen(args.temp, args.tau)
    elif args.thermostat == "langevin":
        th = Langevin(args.temp, args.gamma)
    print(f"{eng.backend}/{eng.variant}: N={eng.n}  box={eng.box[0]:.3f}")
    print(f"{'step':>8} {'T':>9} {'E_pot':>11} {'E_tot':>11} {'P':>10}")
    for k in range(0, args.steps + 1, args.thermo):
        if k:
            eng.run(args.thermo, thermostat=th)
        t = eng.thermo()
        print(f"{k:8d} {t['temp']:9.5f} {t['epot']:11.6f} {t['etot']:11.6f} "
              f"{t['pressure']:10.5f}")
    if args.xyz:
        x, _ = eng.state()
        with open(args.xyz, "w") as fh:
            fh.write(f"{len(x)}\n\n" + "".join(f"Ar {a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in x))
    return 0


def cmd_bench(args) -> int:
    eng = _make(args)
    eng.run(40)
    eng.synchronize()
    t = time.perf_counter()
    eng.run(args.steps)
    eng.synchronize()
    dt = time.perf_counter() - t
    rate = eng.n * args.steps / dt
    print(f"{eng.backend}/{eng.variant} N={eng.n}: {rate / 1e6:.2f} Matom-step/s, "
          f"{args.steps / dt:.1f} steps/s, {ns_per_day(args.steps / dt, args.dt):.1f} ns/day "
          "(argon mapping; dev timing, see bench/run.py for the publishable protocol)")
    return 0


def cmd_validate(args) -> int:
    from .validate import run_all

    parts = args.parts.split(",") if args.parts else None
    doc = run_all(Path(args.out), quick=args.quick, parts=parts)
    if "forces" in doc:
        for r in doc["forces"]:
            if "rel_max_err" in r:
                print(f"  {r['backend']:14s} {r['variant']:16s} max|dF|/max|F| = "
                      f"{r['rel_max_err']:.2e}")
    print(f"wrote {args.out}")
    return 0


def cmd_backends(_args) -> int:
    for name, cls in backends.available().items():
        for v, desc in cls.variants.items():
            print(f"{name:14s} {v:16s} {desc}")
    for name, why in backends.unavailable().items():
        print(f"{name:14s} (unavailable: {why[:80]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pymd", description="GPU Lennard-Jones MD from Python")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run an LJ simulation and print thermo output")
    _common(p)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--thermo", type=int, default=100)
    p.add_argument("--thermostat", choices=["none", "berendsen", "langevin"], default="none")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--xyz", default=None, help="write final positions as .xyz")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("bench", help="quick throughput of one backend (dev timing)")
    _common(p)
    p.add_argument("--steps", type=int, default=400)
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("validate", help="physics validation suite -> JSON")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--parts", default=None,
                   help="comma list of forces,lammps_in_lj,energy_conservation,rdf,eos,eam")
    p.add_argument("--out", default="results/validation.json")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("backends", help="list backends and variants")
    p.set_defaults(fn=cmd_backends)
    args = ap.parse_args(argv)
    np.set_printoptions(precision=5)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
