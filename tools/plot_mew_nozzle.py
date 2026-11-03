#!/usr/bin/env python3
"""Frames and time series for demonstrator/mew_nozzle.

    python3 tools/plot_mew_nozzle.py frames <prefix> <out.png> [t1 t2 ...]
    python3 tools/plot_mew_nozzle.py series <out.png> <label>=<prefix> [...]
    python3 tools/plot_mew_nozzle.py grid <out.png> <t1,t2,..> [zoom=x0,x1,y0,y1] <label>=<prefix> ...

`frames` draws the melt (fill >= 1/2, dark) and the nozzle (grey) with
equipotential lines, one panel per time, like the paper's Fig. 5 sequence.
`series` overlays the melt held in the 1 mm under the tip ("tip zone") and the
lowest point of the melt for several runs.
"""
import csv
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DX_MM = None  # set from the frame width


def frame_times(prefix):
    out = []
    for f in sorted(glob.glob(prefix + "_t*_fill.npy")):
        out.append(float(f[len(prefix) + 2:-len("_fill.npy")]))
    return out


def frames(prefix, out, times, width_mm=6.0):
    avail = frame_times(prefix)
    if not times:
        times = avail
    times = [min(avail, key=lambda a: abs(a - t)) for t in times]
    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n + 0.6, 2.9), facecolor="#fcfcfb",
                             sharey=True)
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, times):
        fl = np.load(f"{prefix}_t{t:06.2f}_fill.npy")
        ph = np.load(f"{prefix}_t{t:06.2f}_phi.npy")
        ny, nx = fl.shape
        dx = width_mm / nx
        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) - 0.5) * dx
        img = np.full(fl.shape + (3,), 0.988)
        img[fl >= 0.5] = (0.165, 0.471, 0.839)          # melt
        img[fl < -0.5] = (0.55, 0.55, 0.53)            # nozzle, collector
        ax.imshow(img, origin="lower", extent=(0, nx * dx, -0.5 * dx, (ny - 0.5) * dx),
                  interpolation="nearest")
        gas = np.where((fl > -0.5) & (fl < 0.5), ph, np.nan)
        ax.contour(x, y, gas, levels=np.linspace(0.1, 0.9, 9), colors=INK2, linewidths=0.4)
        ax.set_title(f"t = {t:.2f} s", fontsize=8.5, color=INK)
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("height above collector [mm]", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    print("wrote", out)


def grid(out, times, runs, width_mm=8.0, zoom=None):
    """Rows = runs, columns = times. zoom = (x0, x1, y0, y1) in mm, or None."""
    nr, nc = len(runs), len(times)
    fig, axes = plt.subplots(nr, nc, figsize=(1.55 * nc + 0.9, 1.75 * nr + 0.5),
                             facecolor="#fcfcfb", squeeze=False)
    for r, (lab, prefix) in enumerate(runs):
        avail = frame_times(prefix)
        for c, t in enumerate(times):
            ax = axes[r][c]
            tt = min(avail, key=lambda a: abs(a - t))
            fl = np.load(f"{prefix}_t{tt:06.2f}_fill.npy")
            ph = np.load(f"{prefix}_t{tt:06.2f}_phi.npy")
            ny, nx = fl.shape
            dx = width_mm / nx
            x = (np.arange(nx) + 0.5) * dx
            y = (np.arange(ny) - 0.5) * dx
            img = np.full(fl.shape + (3,), 0.988)
            img[fl >= 0.5] = (0.165, 0.471, 0.839)
            img[fl < -0.5] = (0.55, 0.55, 0.53)
            ax.imshow(img, origin="lower", extent=(0, nx * dx, -0.5 * dx, (ny - 0.5) * dx),
                      interpolation="nearest")
            gas = np.where((fl > -0.5) & (fl < 0.5), ph, np.nan)
            # A field-off run still writes the potential solved at setup; it acts on
            # nothing, so drawing it would show a field that is not there.
            if "off" not in lab and np.nanmax(gas) > np.nanmin(gas):
                ax.contour(x, y, gas, levels=np.linspace(0.1, 0.9, 9), colors=INK2,
                           linewidths=0.35)
            if zoom:
                ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {tt:.1f} s", fontsize=8.5, color=INK)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8, color=INK)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    print("wrote", out)


def read(path):
    with open(path) as f:
        r = csv.DictReader(f)
        cols = {k: [] for k in r.fieldnames}
        for row in r:
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def series(out, runs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.4), facecolor="#fcfcfb")
    for k, (lab, prefix) in enumerate(runs):
        s = read(prefix + "_series.csv")
        c = SERIES[k % len(SERIES)]
        a.plot(s["t_s"], s["tip_zone_mm2"], color=c, lw=2, label=lab)
        b.plot(s["t_s"], s["z_low_mm"], color=c, lw=2, label=lab)
    a.set_xlabel("time [s]"); a.set_ylabel("melt within 1 mm of the tip [mm$^2$]")
    b.set_xlabel("time [s]"); b.set_ylabel("lowest melt point above collector [mm]")
    for ax in (a, b):
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if sys.argv[1] == "frames":
        frames(sys.argv[2], sys.argv[3], [float(t) for t in sys.argv[4:]])
    elif sys.argv[1] == "grid":
        # grid <out.png> <t1,t2,...> [zoom=x0,x1,y0,y1] label=prefix ...
        rest = sys.argv[4:]
        zoom = None
        if rest and rest[0].startswith("zoom="):
            zoom = [float(v) for v in rest[0][5:].split(",")]
            rest = rest[1:]
        grid(sys.argv[2], [float(t) for t in sys.argv[3].split(",")],
             [tuple(a.split("=", 1)) for a in rest], zoom=zoom)
    elif sys.argv[1] == "series":
        series(sys.argv[2], [tuple(a.split("=", 1)) for a in sys.argv[3:]])
    else:
        sys.exit(__doc__)
