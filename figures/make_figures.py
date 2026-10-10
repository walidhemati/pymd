"""Read results/*.json and produce figures/*.pdf|png and paper/results.tex.

Benchmark source, in order of preference: results/rtx3070-laptop.json (the publishable
``--full`` run), results/dev.json, results/quick.json. Every number in the paper and README
comes from here.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pymd.eos import jzg_pressure  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "figures"
PAPER = ROOT / "paper"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
COL_W, FULL_W = 3.4, 7.0
COLORS = {"numpy": "#7f7f7f", "torch-eager": "#9467bd", "torch-compile": "#8c564b",
          "triton": "#1f77b4", "warp": "#2ca02c", "numba": "#ff7f0e", "cupy-cuda-c": "#d62728",
          "yamd": "#000000"}
#: same-GPU counterparts (black/grey); hatching marks float64 arithmetic
CP_STYLE = {"lammps/kokkos": ("#000000", "o", "-"), "lammps/gpu": ("#555555", "s", "--"),
            "lammps/cpu": ("#b0b0b0", "x", ":"), "hoomd": ("#8c8c8c", "D", "-.")}
LADDER = ["numpy", "torch-eager", "torch-compile", "numba", "warp", "triton", "cupy-cuda-c"]
MARKERS = {"full": "o", "full-fused": "s", "half-atomic": "^", "gather": "D", "hashgrid": "v",
           "cell-verlet": "o", "allpairs": "x"}


def load(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def bench_doc():
    for name in ("rtx3070-laptop.json", "dev.json", "quick.json"):
        d = load(name)
        if d:
            d["_source"] = name
            return d
    raise SystemExit("no benchmark JSON in results/")


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def runs_ok(doc):
    return [r for r in doc["runs"] if "error" not in r]


def series(doc):
    out: dict[tuple[str, str], list[dict]] = {}
    for r in runs_ok(doc):
        out.setdefault((r["backend"], r["variant"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["size"][0])
    return out


def ceiling_rate(doc, n_list: float) -> float:
    """Atom-steps/s if the minimal traffic model ran at the copy-bandwidth ceiling."""
    return doc["ceiling"]["copy_bandwidth_GBs"] * 1e9 / (48 + 4 * n_list)


# ----------------------------------------------------------------------------- performance
def fig_perf_vs_size(doc) -> None:
    s = series(doc)
    fig, ax = plt.subplots(figsize=(FULL_W * 0.74, 3.3))
    for (b, v), rs in sorted(s.items(), key=lambda kv: LADDER.index(kv[0][0])):
        if len(rs) < 2 or v in ("full-morton", "full-nosort", "full-mixed", "full-fp64",
                                "full-tpa2", "full-tpa4",
                                "hashgrid-verlet") or (b == "triton" and v == "half-atomic"):
            continue
        n = [r["size"][0] for r in rs]
        y = [r["median"] / 1e6 for r in rs]
        lo = [r["min"] / 1e6 for r in rs]
        hi = [r["max"] / 1e6 for r in rs]
        ax.plot(n, y, marker=MARKERS.get(v, "o"), ms=3.5, lw=1.2, color=COLORS[b],
                ls="-" if v in ("full", "gather", "cell-verlet") else "--", label=f"{b} / {v}")
        ax.fill_between(n, lo, hi, color=COLORS[b], alpha=0.12, lw=0)
    nl = [r["mean_list_entries_per_atom"] for r in runs_ok(doc)
          if r["variant"] == "full" and r.get("mean_list_entries_per_atom")]
    if nl:
        c = ceiling_rate(doc, float(np.median(nl)))
        ax.axhline(c / 1e6, color="k", ls=":", lw=1)
        ax.text(4e3, c / 1e6 * 1.08, f"copy-bandwidth roofline, full list "
                f"({doc['ceiling']['copy_bandwidth_GBs']:.0f} GB/s)", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("atoms N")
    ax.set_ylabel("Matom-steps / s")
    ax.legend(fontsize=6.5, ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("LJ melt (LAMMPS in.lj), RTX 3070 Laptop", fontsize=9)
    save(fig, "perf_vs_size")


def ref_size(doc) -> int:
    """Largest size that the main GPU variants all ran."""
    s = series(doc)
    sizes = None
    for (b, v), rs in s.items():
        if b in ("numpy", "torch-eager", "torch-compile") or v == "allpairs" or len(rs) < 2:
            continue
        ns = {r["size"][0] for r in rs}
        sizes = ns if sizes is None else sizes & ns
    return max(sizes) if sizes else max(r["size"][0] for r in runs_ok(doc))


def at_size(doc, n):
    return [r for r in runs_ok(doc) if r["size"][0] == n]


def fig_roofline_bars(doc) -> None:
    n = ref_size(doc)
    rs = [r for r in at_size(doc, n) if r.get("pct_copy_bw") is not None]
    rs.sort(key=lambda r: r["pct_copy_bw"])
    fig, ax = plt.subplots(figsize=(COL_W, 0.22 * len(rs) + 0.8))
    y = np.arange(len(rs))
    ax.barh(y, [r["pct_copy_bw"] for r in rs], color=[COLORS[r["backend"]] for r in rs])
    ax.set_yticks(y, [f"{r['backend']} / {r['variant']}" for r in rs], fontsize=7)
    ax.axvline(100, color="k", ls="--", lw=1)
    for yi, r in zip(y, rs):
        ax.text(r["pct_copy_bw"] + 1, yi, f"{r['median'] / 1e6:.0f}", va="center", fontsize=6)
    ax.set_xlabel("% of copy bandwidth (minimal traffic model)")
    ax.set_xlim(0, 105)
    ax.set_title(f"N = {n:,} (labels: Matom-steps/s)", fontsize=8)
    save(fig, "roofline_bars")


def fig_roofline(doc) -> None:
    n = ref_size(doc)
    bw = doc["ceiling"]["copy_bandwidth_GBs"]
    peak = (doc["ceiling"].get("fp32_tflops") or 0) * 1e3
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    ai = np.logspace(-1, 3, 200)
    if peak:
        ax.plot(ai, np.minimum(ai * bw, peak), "k--", lw=1)
        ax.text(150, peak * 1.1, f"FMA peak {peak / 1e3:.1f} TFLOP/s", fontsize=6.5, ha="center")
    else:
        ax.plot(ai, ai * bw, "k--", lw=1)
    ax.text(0.15, 0.15 * bw * 1.6, f"{bw:.0f} GB/s", fontsize=6.5, rotation=33)
    for r in at_size(doc, n):
        if r.get("flops_per_unit") is None or r.get("dtype") == "float64":
            continue
        gflops = r["median"] * r["flops_per_unit"] / 1e9
        if r.get("bytes_per_unit"):
            x = r["flops_per_unit"] / r["bytes_per_unit"]
        else:
            continue
        ax.plot(x, gflops, MARKERS.get(r["variant"], "o"), color=COLORS[r["backend"]], ms=4,
                mfc="none" if "fused" in r["variant"] else COLORS[r["backend"]])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (flop / byte, minimal traffic)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Roofline, N = {n:,}", fontsize=8)
    save(fig, "roofline")


def fig_ablation(doc) -> None:
    sizes = sorted({r["size"][0] for r in runs_ok(doc) if r["variant"] in
                    ("full-morton", "full-nosort", "full-mixed", "full-fp64")})
    if not sizes:
        return
    labels = [("cupy-cuda-c", "full"), ("cupy-cuda-c", "full-tpa2"), ("cupy-cuda-c", "full-tpa4"),
              ("cupy-cuda-c", "full-morton"),
              ("cupy-cuda-c", "full-nosort"), ("cupy-cuda-c", "full-fused"),
              ("cupy-cuda-c", "half-atomic"), ("triton", "half-atomic"),
              ("cupy-cuda-c", "full-mixed"), ("cupy-cuda-c", "full-fp64"),
              ("warp", "hashgrid-verlet"), ("warp", "full"), ("warp", "hashgrid")]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.6))
    w = 0.8 / len(sizes)
    for k, n in enumerate(sizes):
        d = {(r["backend"], r["variant"]): r for r in at_size(doc, n)}
        ys = [d[lab]["median"] / 1e6 if lab in d else 0 for lab in labels]
        ax.bar(np.arange(len(labels)) + k * w, ys, w, color=[COLORS[b] for b, _ in labels],
               alpha=1.0 if k == 0 else 0.55, label=f"N = {n:,}")
    ax.set_xticks(np.arange(len(labels)) + 0.4 - w / 2,
                  [f"{b}\n{v}" for b, v in labels], fontsize=6)
    ax.set_ylabel("Matom-steps / s")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Ablations: sorting, fusion, Newton/atomics, precision, neighbor search",
                 fontsize=8)
    save(fig, "ablation")


# ----------------------------------------------------------------------------- validation
def fig_validation(val) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(FULL_W, 5.0))
    # (a) force errors
    ax = axs[0, 0]
    rows = [r for r in val.get("forces", []) if "rel_max_err" in r]
    y = np.arange(len(rows))
    ax.barh(y, [max(r["rel_max_err"], 1e-17) for r in rows],
            color=[COLORS[r["backend"]] for r in rows])
    ax.set_yticks(y, [f"{r['backend']}/{r['variant']}" for r in rows], fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\max_i|F_i-F_i^{\rm ref}|\,/\,\max_i|F_i^{\rm ref}|$")
    ax.set_title("(a) forces vs NumPy float64 all-pairs", fontsize=8)
    ax.invert_yaxis()
    # (b) energy fluctuation vs dt
    ax = axs[0, 1]
    ec = val.get("energy_conservation", {})
    for v, mk in (("full", "o"), ("full-mixed", "s"), ("full-fp64", "^")):
        pts = [r for r in ec.get("sweep", []) if r["variant"] == v]
        if pts:
            ax.loglog([p["dt"] for p in pts], [p["rms_fluct"] for p in pts], mk + "-", ms=3.5,
                      color={"full": "#d62728", "full-mixed": "#ff9896",
                             "full-fp64": "#7f7f7f"}[v], label=v.replace("full", "cupy"))
    if ec.get("sweep"):
        dts = np.array(sorted({r["dt"] for r in ec["sweep"]}))
        ref = [r for r in ec["sweep"] if r["dt"] == dts[-1]][0]["rms_fluct"]
        ax.loglog(dts, ref * (dts / dts[-1]) ** 2, "k--", lw=0.8, label=r"$\propto\Delta t^2$")
    ax.set_xlabel(r"time step $\Delta t$ ($\tau$)")
    ax.set_ylabel(r"rms fluctuation of $E/N$ ($\epsilon$)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("(b) NVE energy conservation", fontsize=8)
    # (c) long run drift
    ax = axs[1, 0]
    for r in ec.get("long", []):
        ax.plot(r["t"], r["de"], lw=1, label=r["variant"].replace("full", "cupy"),
                color={"full": "#d62728", "full-mixed": "#ff9896", "full-fp64": "#7f7f7f"}.get(
                    r["variant"], "k"))
    ax.set_xlabel(r"time ($\tau$), $\Delta t = 0.005$")
    ax.set_ylabel(r"$[E(t)-E(0)]/N$ ($\epsilon$)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("(c) long NVE run, N = 4000", fontsize=8)
    # (d) EOS vs NIST
    ax = axs[1, 1]
    eos = val.get("eos", {}).get("rows", [])
    for t, mk in ((0.85, "o"), (0.90, "s")):
        rs = [r for r in eos if r["temp"] == t]
        if not rs:
            continue
        rho = [r["rho"] for r in rs]
        ax.errorbar(rho, [r["p"] for r in rs], yerr=[r["p_err"] for r in rs], fmt=mk,
                    color="#d62728", ms=3.5, mfc="none", label=f"pymd T*={t}")
        ax.plot(rho, [r["nist_p"] for r in rs], mk, color="k", ms=2, label=f"NIST MC T*={t}")
        rr = np.linspace(0.77, 0.905, 50)
        ax.plot(rr, [jzg_pressure(x, t) for x in rr], "-", color="#1f77b4", lw=0.8,
                alpha=0.8, label="JZG EOS" if t == 0.85 else None)
    ax.set_xlabel(r"density $\rho^*$")
    ax.set_ylabel(r"pressure $P^*$")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("(d) equation of state vs NIST SRSW and JZG", fontsize=8)
    fig.tight_layout()
    save(fig, "validation")


def fig_rdf(val) -> None:
    r = val.get("rdf")
    if not r:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.plot(r["r"], r["g"], color="#d62728", lw=1.2)
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_xlabel(r"$r/\sigma$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_xlim(0.8, max(r["r"]))
    ax.set_title(rf"LJ liquid $\rho^*={r['rho']}$, $T^*={r['temp']}$, N={r['n']}", fontsize=8)
    ax.text(0.97, 0.9, f"U (energy route) {r['u_energy_route']:.3f}\n"
            f"U (direct)        {r['u_direct']:.3f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, family="monospace")
    save(fig, "rdf")


def fig_yamd(ym) -> None:
    if not ym:
        return
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    for ax, key, title in ((axs[0], "lj", "LJ FCC block (open)"),
                           (axs[1], "eam", "Au Mackay cluster (Gupta)")):
        rows = ym.get(key, [])
        if not rows:
            continue
        n = [r["n"] for r in rows]
        ax.plot(n, [r["yamd_cpp_1thread"] / 1e6 for r in rows], "k-o", ms=3,
                label="yamd C++ (1 thread)")
        keys = sorted({k for r in rows for k in r if k.count("/") == 2})
        for k in keys:
            b, v, mode = k.split("/")
            ys = [r.get(k) for r in rows]
            if any(not isinstance(y, float) for y in ys):
                continue
            ax.plot(n, [y / 1e6 for y in ys], marker="o" if mode == "pymd-default" else "x",
                    ms=3, color=COLORS.get(b, "k"), ls="-" if mode == "pymd-default" else ":",
                    lw=1, label=f"pymd {b} ({mode})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("atoms N")
        ax.set_ylabel("Matom-steps / s")
        ax.set_title(title, fontsize=8)
    axs[0].legend(fontsize=5.5, frameon=False, ncol=2)
    axs[1].legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    save(fig, "yamd_compare")


# ----------------------------------------------------------------------------- counterparts
def cp_family(r) -> str:
    if r["code"] == "pymd":
        return r.get("backend", r["variant"].split("/")[0])
    if r["code"] == "hoomd":
        return "hoomd"
    return "lammps/" + r["variant"].split("-")[0]


def cp_color(r) -> str:
    fam = cp_family(r)
    return COLORS[fam] if fam in COLORS else CP_STYLE[fam][0]


def cp_label(r) -> str:
    if r["code"] == "pymd":
        return "pymd " + r["variant"]
    return f"{'LAMMPS' if r['code'] == 'lammps' else 'HOOMD'} {r['variant']}"


def cp_ok(cp) -> list[dict]:
    return [r for r in cp.get("runs", []) if "error" not in r]


def cp_ref_size(cp) -> int | None:
    """Largest N at which pymd, LAMMPS KOKKOS (or GPU) and HOOMD all have a rate."""
    ok = cp_ok(cp)
    if not ok:
        return None
    groups = [{r["n"] for r in ok if r["code"] == "pymd"},
              {r["n"] for r in ok if r["code"] == "lammps" and r["variant"] != "cpu-1core"},
              {r["n"] for r in ok if r["code"] == "hoomd"}]
    groups = [g for g in groups if g]
    common = set.intersection(*groups) if groups else set()
    return max(common) if common else max(r["n"] for r in ok)


def fig_counterparts(cp, doc) -> None:
    if not cp or not cp_ok(cp):
        return
    ok = cp_ok(cp)
    n = cp_ref_size(cp)
    sizes = sorted({r["n"] for r in ok})
    multi = len(sizes) > 1
    fig, axs = plt.subplots(1, 2 if multi else 1, figsize=(FULL_W if multi else COL_W * 1.25,
                                                         3.0), squeeze=False)
    ax = axs[0, 0]
    rs = sorted([r for r in ok if r["n"] == n], key=lambda r: r["median"])
    y = np.arange(len(rs))
    ax.barh(y, [r["median"] / 1e6 for r in rs], color=[cp_color(r) for r in rs],
            hatch=None, edgecolor="white", lw=0.3)
    for yi, r in zip(y, rs):
        if r.get("precision") == "double":
            ax.barh(yi, r["median"] / 1e6, color="none", edgecolor="white", hatch="////",
                    lw=0)
        ax.text(r["median"] / 1e6 * 1.03, yi, f"{r['median'] / 1e6:.1f}", va="center",
                fontsize=5.5)
    ax.set_yticks(y, [cp_label(r) for r in rs], fontsize=6)
    nl = [r["mean_list_entries_per_atom"] for r in ok
          if r.get("variant") == "cupy-cuda-c/full" and r.get("mean_list_entries_per_atom")]
    if nl:
        c = ceiling_rate(doc, float(np.median(nl))) / 1e6
        ax.axvline(c, color="k", ls="--", lw=0.8)
        ax.text(c * 0.95, -0.4, "copy-BW roofline\n(pymd full list)", fontsize=5.5,
                va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Matom-steps / s")
    prov = "provisional, shared GPU; " if cp.get("provisional") else ""
    ax.set_title(f"N = {n:,} ({prov}hatched: float64)", fontsize=7)
    if multi:
        ax = axs[0, 1]
        keys = sorted({(r["code"], r["variant"]) for r in ok})
        for code, var in keys:
            rr = sorted([r for r in ok if (r["code"], r["variant"]) == (code, var)],
                        key=lambda r: r["n"])
            if len(rr) < 2:
                continue
            fam = cp_family(rr[0])
            if code == "pymd":
                st = dict(color=COLORS[fam], marker="o", ls="-")
            else:
                col, mk, ls = CP_STYLE[fam]
                st = dict(color=col, marker="^" if var.endswith("-half") else mk,
                          ls=":" if var.endswith("-half") else ls,
                          mfc="none" if rr[0].get("precision") == "double" else col)
            ax.plot([r["n"] for r in rr], [r["median"] / 1e6 for r in rr], ms=3, lw=1,
                    label=cp_label(rr[0]), **st)
        if nl:
            ax.axhline(c, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("atoms N")
        ax.set_ylabel("Matom-steps / s")
        ax.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_title("vs size (open: float64; triangles: half list)", fontsize=7)
    fig.tight_layout()
    save(fig, "counterparts")


def cp_macros(cp) -> dict:
    m: dict[str, str] = {}
    ok = cp_ok(cp)
    if not ok:
        return m
    n = cp_ref_size(cp)
    at = [r for r in ok if r["n"] == n]
    m["cpProvisional"] = "1" if cp.get("provisional", True) else "0"
    m["cpSource"] = "counterparts.json"
    m["cpN"] = fmt(n, 0)
    m["cpRepeats"] = str(cp.get("settings", {}).get("repeats", "?"))

    def best(pred):
        c = [r for r in at if pred(r)]
        return max(c, key=lambda r: r["median"]) if c else None

    pick = {
        "Pymd": best(lambda r: r["code"] == "pymd"),
        "PymdPython": best(lambda r: r["code"] == "pymd" and r.get("backend") != "cupy-cuda-c"),
        "Kokkos": best(lambda r: r["code"] == "lammps" and r["variant"].startswith("kokkos")),
        "KokkosDouble": best(lambda r: r["code"] == "lammps"
                             and r["variant"].startswith("kokkos-double")),
        "KokkosSingle": best(lambda r: r["code"] == "lammps"
                             and r["variant"].startswith("kokkos-single")),
        "LmpGpu": best(lambda r: r["code"] == "lammps" and r["variant"].startswith("gpu")),
        "HoomdDouble": best(lambda r: r["code"] == "hoomd" and r["variant"] == "gpu-double"),
        "HoomdSingle": best(lambda r: r["code"] == "hoomd" and r["variant"] == "gpu-single"),
        "Counter": best(lambda r: r["code"] != "pymd" and r["variant"] != "cpu-1core"),
    }
    for tag, r in pick.items():
        if r is None:
            continue
        m[f"cp{tag}"] = fmt(r["median"] / 1e6, 1)
        m[f"cp{tag}Variant"] = r["variant"].replace("_", r"\_")
    if "cpCounter" in m:
        r = pick["Counter"]
        m["cpCounterPrec"] = r.get("precision", "?")
        m["cpCounterCode"] = "LAMMPS" if r["code"] == "lammps" else "HOOMD-blue"
    pm = pick["Pymd"]
    for tag in ("Kokkos", "KokkosDouble", "KokkosSingle", "LmpGpu", "HoomdDouble",
                "HoomdSingle", "Counter"):
        r = pick[tag]
        if pm and r:
            m[f"cpRatio{tag}"] = fmt(pm["median"] / r["median"], 2)
    for a, b, tag in (("HoomdSingle", "HoomdDouble", "Hoomd"),
                      ("KokkosSingle", "KokkosDouble", "Kokkos")):
        if pick[a] and pick[b]:
            m[f"cp{tag}PrecRatio"] = fmt(pick[a]["median"] / pick[b]["median"], 1)
    pp = pick["PymdPython"]
    if pp and pick["Counter"]:
        m["cpRatioPythonCounter"] = fmt(pp["median"] / pick["Counter"]["median"], 2)
    cpu = [r for r in ok if r["variant"] == "cpu-1core"]
    if cpu:
        r = max(cpu, key=lambda r: r["n"])
        m["cpLmpCpu"] = fmt(r["median"] / 1e6, 2)
        m["cpLmpCpuN"] = fmt(r["n"], 0)
    hs = [r for r in ok if r["code"] == "hoomd"]
    if hs:
        m["cpHoomdTuned"] = "1" if all(r.get("tuning_complete") for r in hs) else "0"
    lmp = (cp.get("codes") or {}).get("lammps") or {}
    m["lammpsVersion"] = lmp.get("version") or "?"
    m["lammpsTag"] = (lmp.get("tag") or "?").replace("_", r"\_")
    hd = (cp.get("codes") or {}).get("hoomd") or {}
    vers = [v.get("hoomd_version") for v in (hd.get("variants") or {}).values()]
    m["hoomdVersion"] = vers[0] if vers else "?"
    mach = cp.get("machine", {})
    nv = re.search(r"cuda_(\d+\.\d+)", mach.get("nvcc") or "")
    m["cpNvcc"] = nv.group(1) if nv else "?"
    ver = cp.get("verify") or {}
    rows = [r for r in ver.get("rows", []) if "rel_diff_step0" in r and r["code"] != "pymd"]
    if rows:
        m["cpVerN"] = fmt(ver["n"], 0)
        for code, tag in (("lammps", "Lammps"), ("hoomd", "Hoomd")):
            rr = [r for r in rows if r["code"] == code]
            if not rr:
                continue
            for st, stag in (("step0", "Zero"), ("step100", "Hundred")):
                for k, ktag in (("epot", "Epot"), ("pressure", "Press"), ("etot", "Etot")):
                    v = max(r[f"rel_diff_{st}"][k] for r in rr)
                    m[f"cpVer{tag}{stag}{ktag}"] = f"{v:.0e}"
        e1 = [r["rel_diff_step100_rebuild_every_1"] for r in rows
              if "rel_diff_step100_rebuild_every_1" in r]
        if e1:
            m["cpVerHoomdEveryOneEpot"] = f"{max(x['epot'] for x in e1):.0e}"
            m["cpVerHoomdEveryOnePress"] = f"{max(x['pressure'] for x in e1):.0e}"
        f32 = [r for r in ver.get("rows", []) if r.get("variant") == "cupy-cuda-c/full"
               and "rel_diff_step100" in r]
        if f32:
            m["cpVerPymdSingleHundredEpot"] = f"{f32[0]['rel_diff_step100']['epot']:.0e}"
            m["cpVerPymdSingleHundredPress"] = f"{f32[0]['rel_diff_step100']['pressure']:.0e}"
    return m


def cp_table(cp) -> str:
    """Markdown table of the same-GPU comparison at the reference size (README)."""
    ok = cp_ok(cp)
    n = cp_ref_size(cp)
    rows = sorted([r for r in ok if r["n"] == n], key=lambda r: -r["median"])
    out = ["| code | variant | precision | Matom-steps/s |", "|---|---|---|---:|"]
    for r in rows:
        out.append(f"| {r['code']} | {r['variant']} | {r.get('precision', '?')} | "
                   f"{r['median'] / 1e6:.1f} |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- results.tex
def fmt(x, nd=1):
    return f"{x:,.{nd}f}".replace(",", "{,}")


def write_tex(doc, val, ym, cp=None) -> dict:
    ok = [r for r in runs_ok(doc) if r["backend"] != "numpy"]
    best = max(ok, key=lambda r: r["median"])
    m = {}
    m["provisional"] = "1" if doc.get("provisional", True) else "0"
    m["benchSource"] = doc["_source"].replace("_", r"\_")
    m["gpuName"] = doc["machine"].get("gpu") or "GPU"
    m["copyBW"] = fmt(doc["ceiling"]["copy_bandwidth_GBs"], 0)
    m["fmaPeak"] = fmt(doc["ceiling"].get("fp32_tflops") or float("nan"), 1)
    m["bestRate"] = fmt(best["median"] / 1e6, 0)
    m["bestBackend"] = best["backend"]
    m["bestVariant"] = best["variant"]
    m["bestN"] = fmt(best["size"][0], 0)
    m["bestPctCopy"] = fmt(best["pct_copy_bw"] or 0, 0)
    m["bestPctSpec"] = fmt(best["pct_spec_bw"] or 0, 0)
    m["bestPctFlops"] = fmt(best["pct_peak_flops"] or 0, 0)
    m["bestTauDay"] = fmt(best["tau_per_day"], 0)
    m["bestNsDay"] = fmt(best["ns_per_day_argon"], 0)
    m["bestBytes"] = fmt(best["bytes_per_unit"] or 0, 0)
    m["bestListLen"] = fmt(best["mean_list_entries_per_atom"] or 0, 1)
    n_ref = ref_size(doc)
    m["refN"] = fmt(n_ref, 0)
    by = {(r["backend"], r["variant"]): r for r in at_size(doc, n_ref)}
    for b, tag in (("numpy", "Numpy"), ("torch-eager", "TorchEager"),
                   ("torch-compile", "TorchCompile"), ("triton", "Triton"), ("warp", "Warp"),
                   ("numba", "Numba"), ("cupy-cuda-c", "Cupy")):
        cand = [r for (bb, _), r in by.items() if bb == b]
        if not cand:  # backend stops below n_ref: quote it at its own largest size
            own = [r for r in runs_ok(doc) if r["backend"] == b and r["variant"] != "allpairs"]
            if own:
                n_own = max(r["size"][0] for r in own)
                cand = [r for r in own if r["size"][0] == n_own]
        if cand:
            r = max(cand, key=lambda r: r["median"])
            m[f"ref{tag}"] = fmt(r["median"] / 1e6, 1)
            m[f"ref{tag}Pct"] = fmt(r["pct_copy_bw"] or 0, 0)
            m[f"ref{tag}N"] = fmt(r["size"][0], 0)
    cu = by.get(("cupy-cuda-c", "full"))
    for b, tag in (("triton", "Triton"), ("warp", "Warp"), ("numba", "Numba")):
        r = by.get((b, "full"))
        if cu and r:
            m[f"ratio{tag}"] = fmt(100 * r["median"] / cu["median"], 0)
    small = [r for r in runs_ok(doc) if r["size"][0] == 4000 and r["backend"] == "numpy"]
    if small:
        m["numpySmall"] = fmt(small[0]["median"] / 1e6, 2)
    abl_sizes = [r["size"][0] for r in runs_ok(doc) if r["variant"] == "full-nosort"]
    n_abl = max(abl_sizes) if abl_sizes else n_ref
    m["ablN"] = fmt(n_abl, 0)
    sp = {(r["backend"], r["variant"]): r for r in at_size(doc, n_abl)}
    cu = sp.get(("cupy-cuda-c", "full"))
    for v, tag in (("full-nosort", "NoSort"), ("full-morton", "Morton"), ("half-atomic", "Half"),
                   ("full-tpa2", "TpaTwo"), ("full-tpa4", "TpaFour"),
                   ("full-fused", "Fused"), ("full-mixed", "Mixed"), ("full-fp64", "FpSixtyFour")):
        r = sp.get(("cupy-cuda-c", v))
        if r and cu:
            m[f"abl{tag}"] = fmt(r["median"] / 1e6, 0)
            m[f"abl{tag}Ratio"] = fmt(r["median"] / cu["median"], 2)
    if val:
        fr = [r for r in val.get("forces", []) if "rel_max_err" in r]
        f32 = [r["rel_max_err"] for r in fr if r["variant"] not in ("full-fp64", "full-mixed")
               and r["backend"] != "numpy"]
        if f32:
            m["forceErrMax"] = f"{max(f32):.1e}"
            m["forceErrMin"] = f"{min(f32):.1e}"
        for r in fr:
            if r["variant"] == "full-fp64":
                m["forceErrFpSixtyFour"] = f"{r['rel_max_err']:.0e}"
            if r["variant"] == "full-mixed":
                m["forceErrMixed"] = f"{r['rel_max_err']:.1e}"
        m["forceN"] = fmt(fr[0]["n"], 0) if fr else "?"
        lj = val.get("lammps_in_lj")
        if lj:
            for k, tag in (("temp", "Temp"), ("epot", "Epot"), ("etot", "Etot"),
                           ("pressure", "Press")):
                m[f"inljZero{tag}"] = f"{lj['step0'][k]['mean']:.7f}"
                m[f"inljHundred{tag}"] = f"{lj['step100'][k]['mean']:.4f}"
                m[f"inljHundred{tag}Std"] = f"{lj['step100'][k]['std']:.4f}"
            m["inljSeeds"] = str(len(lj["seeds"]))
            m["inljNeigh"] = f"{lj['ave_neighs_half']:.3f}" if lj.get("ave_neighs_half") else "?"
            m["inljBuilds"] = str(lj["neighbor_builds_after_setup"])
        ec = val.get("energy_conservation")
        if ec:
            for r in ec["long"]:
                tag = {"full": "Single", "full-mixed": "Mixed", "full-fp64": "Double"}.get(
                    r["variant"])
                if tag:
                    m[f"drift{tag}"] = f"{r['drift_per_tau']:.1e}"
            m["driftTime"] = fmt(ec["long"][0]["t"][-1], 0) if ec["long"] else "?"
            sw = [r for r in ec["sweep"] if r["variant"] == "full" and r["dt"] >= 0.003]
            if len(sw) >= 2:
                k = np.polyfit(np.log([r["dt"] for r in sw]), np.log([r["rms_fluct"] for r in sw]),
                               1)[0]
                m["fluctSlope"] = f"{k:.2f}"
        rdf = val.get("rdf")
        if rdf:
            for k, tag in (("peak_r", "PeakR"), ("peak_g", "PeakG"), ("u_direct", "Udirect"),
                           ("u_energy_route", "Uroute"), ("p_direct", "Pdirect"),
                           ("p_virial_route", "Proute"), ("u_energy_route_full_lj", "UrouteFull"),
                           ("u_direct_plus_lrc", "UdirectLrc"), ("p_direct_plus_lrc",
                                                                  "PdirectLrc"),
                           ("jzg_u", "Ujzg"), ("jzg_p", "Pjzg")):
                m[f"rdf{tag}"] = f"{rdf[k]:.3f}"
            m["rdfUdirectErr"] = f"{rdf['u_direct_err']:.3f}"
            m["rdfPdirectErr"] = f"{rdf['p_direct_err']:.3f}"
        eos = val.get("eos", {}).get("rows", [])
        if eos:
            du = [abs(r["u"] - r["nist_u"]) / abs(r["nist_u"]) for r in eos]
            dp = [abs(r["p"] - r["nist_p"]) for r in eos]
            m["eosMaxRelU"] = f"{100 * max(du):.2f}"
            m["eosMaxAbsP"] = f"{max(dp):.3f}"
            m["eosNpts"] = str(len(eos))
            z = [abs(r["p"] - r["nist_p"]) / np.hypot(r["p_err"], r["nist_p_err"]) for r in eos]
            m["eosMaxZP"] = f"{max(z):.1f}"
            zu = [abs(r["u"] - r["nist_u"]) / np.hypot(r["u_err"], r["nist_u_err"]) for r in eos]
            m["eosMaxZU"] = f"{max(zu):.1f}"
        eam = val.get("eam")
        if eam and "rel_max_force_err" in eam:
            m["eamForceErr"] = f"{eam['rel_max_force_err']:.1e}"
            m["eamN"] = str(eam["n"])
            m["eamDrift"] = f"{max(abs(x) for x in eam['nve_de_per_atom_eV']):.1e}"
    if ym:
        lj = ym.get("lj", [])
        if lj:
            fd = max(r.get("max_abs_force_diff_yamd_vs_pymd_fp64", 0) for r in lj)
            m["yamdForceDiff"] = f"{fd:.0e}"
            big = max(lj, key=lambda r: r["n"])
            m["yamdN"] = fmt(big["n"], 0)
            m["yamdRate"] = fmt(big["yamd_cpp_1thread"] / 1e6, 2)
            np_ = big.get("numpy/cell-verlet/same-algorithm")
            if isinstance(np_, float):
                m["yamdNumpySame"] = fmt(np_ / 1e6, 3)
                m["yamdNumpySameRatio"] = fmt(big["yamd_cpp_1thread"] / np_, 0)
            npd = big.get("numpy/cell-verlet/pymd-default")
            if isinstance(npd, float):
                m["yamdNumpyDefault"] = fmt(npd / 1e6, 3)
            gpu = {k: v for k, v in big.items() if k.startswith("cupy-cuda-c/full/")}
            for k, v in gpu.items():
                tag = "Same" if k.endswith("same-algorithm") else "Default"
                if isinstance(v, float):
                    m[f"yamdCupy{tag}"] = fmt(v / 1e6, 1)
                    m[f"yamdCupy{tag}Ratio"] = fmt(v / big["yamd_cpp_1thread"], 0)
        eam = ym.get("eam", [])
        if eam:
            fd = max(r.get("max_abs_force_diff_yamd_vs_pymd_fp64", 0) for r in eam)
            m["yamdEamForceDiff"] = f"{fd:.0e}"
            big = max(eam, key=lambda r: r["n"])
            m["yamdEamN"] = fmt(big["n"], 0)
            m["yamdEamRate"] = fmt(big["yamd_cpp_1thread"] / 1e6, 3)
            v = big.get("cupy-cuda-c/gupta/pymd-default")
            if isinstance(v, float):
                m["yamdEamGpu"] = fmt(v / 1e6, 1)
                m["yamdEamGpuRatio"] = fmt(v / big["yamd_cpp_1thread"], 0)
        m["yamdCpu"] = ym.get("cpu", "?")
        m["yamdProvisional"] = "1" if ym.get("provisional", True) else "0"
    if cp:
        m.update(cp_macros(cp))
    lines = ["% GENERATED by figures/make_figures.py from results/*.json -- do not edit"]
    for k, v in m.items():
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    (PAPER / "results.tex").write_text("\n".join(lines) + "\n")
    return m


def perf_table(doc) -> str:
    """Markdown table of the main ladder at the reference size (for the README)."""
    n = ref_size(doc)
    rows = sorted(at_size(doc, n), key=lambda r: -r["median"])
    out = ["| backend | variant | Matom-steps/s | % copy BW | ns/day (Ar) |",
           "|---|---|---:|---:|---:|"]
    for r in rows:
        pct = f"{r['pct_copy_bw']:.0f}" if r.get("pct_copy_bw") is not None else "-"
        out.append(f"| {r['backend']} | {r['variant']} | {r['median'] / 1e6:.1f} | {pct} | "
                   f"{r['ns_per_day_argon']:.0f} |")
    return "\n".join(out)


def _replace_block(text: str, tag: str, body: str) -> str:
    a, b = f"<!-- {tag}:BEGIN -->", f"<!-- {tag}:END -->"
    if a not in text or b not in text:
        return text
    head, rest = text.split(a, 1)
    _, tail = rest.split(b, 1)
    return f"{head}{a}\n{body}\n{b}{tail}"


def update_readme(doc, m) -> None:
    """Fill the README's generated blocks (headline box and performance table) from JSON."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    prov = " **(provisional: development run on a shared GPU)**" if doc.get("provisional") else ""
    unescape = {k: str(v).replace("{,}", ",") for k, v in m.items()}
    head = (f"> **{unescape['bestRate']} million atom-steps/s** on the LAMMPS `in.lj` LJ melt "
            f"with {unescape['bestN']} atoms ({unescape['bestBackend']} / "
            f"{unescape['bestVariant']}), **{unescape['bestPctCopy']}% of the measured copy "
            f"bandwidth** ({unescape['copyBW']} GB/s, minimal-traffic model) on an "
            f"{unescape['gpuName']}; {unescape['bestNsDay']} ns/day for argon.{prov}  \n"
            f"> Source: `results/{doc['_source']}`.")
    table = (f"Throughput at N = {ref_size(doc):,} (source `results/{doc['_source']}`"
             f"{', provisional' if doc.get('provisional') else ''}):\n\n" + perf_table(doc))
    if doc.get("provisional"):
        # SPEC: development timings never go into the README; wait for the final serial run.
        head = ("> Pending the final benchmark (`python bench/run.py --full` on an idle GPU). "
                "The figures below currently show provisional development data "
                f"(`results/{doc['_source']}`).")
        table = "Table pending the final benchmark run (`results/rtx3070-laptop.json`)."
    text = readme.read_text()
    text = _replace_block(text, "HEADLINE", head)
    text = _replace_block(text, "PERF", table)
    readme.write_text(text)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    doc = bench_doc()
    val = load("validation.json")
    ym = load("yamd-compare.json")
    cp = load("counterparts.json")
    fig_perf_vs_size(doc)
    fig_roofline_bars(doc)
    fig_roofline(doc)
    fig_ablation(doc)
    if val:
        fig_validation(val)
        fig_rdf(val)
    fig_yamd(ym)
    fig_counterparts(cp, doc)
    m = write_tex(doc, val, ym, cp)
    (RES / "perf_table.md").write_text(
        f"Source: results/{doc['_source']} (provisional: {doc.get('provisional')}), "
        f"N = {ref_size(doc):,}\n\n" + perf_table(doc) + "\n")
    update_readme(doc, m)
    if cp and cp_ok(cp):
        (RES / "counterparts_table.md").write_text(
            f"Source: results/counterparts.json (provisional: {cp.get('provisional')}), "
            f"N = {cp_ref_size(cp):,}\n\n" + cp_table(cp) + "\n")
    print(f"figures from {doc['_source']}; best {m['bestBackend']}/{m['bestVariant']} "
          f"{m['bestRate']} Matom-steps/s at N={m['bestN']}")


if __name__ == "__main__":
    main()
