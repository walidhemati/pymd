"""Hero animation for the README: a Mackay-icosahedral gold cluster (Gupta potential, as in
yamd) heated on the GPU by a Berendsen thermostat until it melts.

    python figures/make_hero.py            # -> docs/media/hero.gif (< 4 MB)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from pymd.eam import AU_GUPTA, AU_MASS, KB_EV_K, mackay_icosahedron, make_gupta_engine  # noqa: E402
from pymd.engine import Berendsen  # noqa: E402
from pymd.system import MDConfig  # noqa: E402


def render(x: np.ndarray, temp: float, t_ps: float, center: np.ndarray, lim: float) -> Image:
    """Orthographic view with depth shading (back atoms darker and smaller)."""
    th = 0.35
    rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    p = (x - center) @ rot.T
    order = np.argsort(p[:, 2])
    p = p[order]
    depth = (p[:, 2] - p[:, 2].min()) / max(np.ptp(p[:, 2]), 1e-9)
    fig = plt.figure(figsize=(3.6, 3.6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    cmap = plt.get_cmap("YlOrBr_r")
    ax.scatter(p[:, 0], p[:, 1], s=10 + 16 * depth, c=cmap(0.15 + 0.55 * (1 - depth)),
               edgecolors="none")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.03, 0.96, f"Au$_{{{len(x)}}}$  Gupta EAM on GPU", color="w", fontsize=9,
            transform=ax.transAxes, va="top", family="serif")
    ax.text(0.03, 0.04, f"T = {temp:5.0f} K   t = {t_ps:5.0f} ps", color="w", fontsize=9,
            transform=ax.transAxes, family="monospace")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64)


def main() -> None:
    layers = 8
    x0 = mackay_icosahedron(layers)
    box = np.full(3, 2 * np.ptp(x0, axis=0).max() + 60.0)
    x0 = x0 + box / 2
    n = len(x0)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(x0.shape) * np.sqrt(KB_EV_K * 300 / AU_MASS)
    v -= v.mean(0)
    dt = 2.0  # fs
    cfg = MDConfig(dt=dt, skin=1.0, rebuild_every=5, check=True, mass=AU_MASS)
    eng = make_gupta_engine(x0, v, box, AU_GUPTA, cfg)
    frames = []
    nframes, steps_per_frame = 72, 500
    lim = 0.74 * np.ptp(x0, axis=0).max()
    for k in range(nframes):
        target = 300 + (1500 - 300) * min(1.0, k / (0.8 * nframes))
        eng.run(steps_per_frame, thermostat=Berendsen(target * KB_EV_K, tau=500.0))
        x, _ = eng.state()
        t = eng.thermo()["temp"]
        temp = t / KB_EV_K  # thermo() temperature is in energy units (k_B T, eV)
        frames.append(render(x, temp, (k + 1) * steps_per_frame * dt / 1000, x.mean(0), lim))
        print(f"frame {k:3d}  T = {temp:6.0f} K", flush=True)
    out = ROOT / "docs" / "media" / "hero.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * 8,
                   duration=90, loop=0, optimize=True)
    frames[-1].convert("RGB").save(out.with_suffix(".png"))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, N = {n})")


if __name__ == "__main__":
    main()
