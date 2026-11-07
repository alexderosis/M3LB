#!/usr/bin/env python3
"""Figures for demonstrator/mew_axi, the r-z melt-electrowriting model.

Reads results/mew_axi/ (series CSVs and logs, and for `shapes` the .npy frames,
which git ignores and the runs rewrite), plus the planar and 3-D siblings'
series for `compare`:

    python3 tools/plot_mew_axi.py compare results/mew_axi/mew_axi_compare.png
    python3 tools/plot_mew_axi.py sens    results/mew_axi/mew_axi_sens.png
    python3 tools/plot_mew_axi.py shapes  results/mew_axi/mew_axi_shapes.png

compare: the approach to the collector, the melt held at the tip and the melt's
mean temperature, for the isothermal r-z model on two grids, the model with
heat, and the planar and 3-D siblings. sens: the same three panels for the
model with heat under each of its unmeasured parameters. shapes: the drop,
mirrored about the axis, coloured by temperature.
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
AXI = os.path.join(RES, "mew_axi")

INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TMAP = LinearSegmentedColormap.from_list("T", ["#fde3d6", "#f6a882", "#eb6834", "#b8461b", "#6e2a10"])


def contact(log):
    try:
        for line in open(log):
            m = re.search(r"first contact with the collector at t = ([0-9.]+)", line)
            if m:
                return float(m.group(1))
    except OSError:
        pass
    return None


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def three_panels(out, runs, title):
    """runs: (label, series csv, log, colour slot, line style)"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), facecolor=SURF)
    for lab, csv, log, k, ls in runs:
        if not os.path.exists(csv):
            print("missing", csv)
            continue
        s = np.genfromtxt(csv, delimiter=",", names=True)
        c = SERIES[k]
        tc = contact(log)
        t, z = s["t_s"], s["z_low_mm"]
        keep = t <= (tc if tc is not None else 1e9)
        tz, zz = t[keep], z[keep]
        if tc is not None:                       # after contact the lowest point is the collector
            tz, zz = np.append(tz, tc), np.append(zz, 0.0)
        axes[0].plot(tz, zz, color=c, ls=ls, lw=1.8, label=lab)
        if tc is not None:
            axes[0].plot([tc], [0.0], "o", color=c, ms=5.5, mec=SURF, mew=1.2, clip_on=False)
        if "tip_zone_mm3" in s.dtype.names:      # the planar sibling's are areas (mm^2): panel a only
            axes[1].plot(t[keep], s["tip_zone_mm3"][keep], color=c, ls=ls, lw=1.8)
        if "T_gap_mean_C" in s.dtype.names and np.isfinite(s["T_gap_mean_C"]).any():
            axes[2].plot(t[keep], s["T_gap_mean_C"][keep], color=c, ls=ls, lw=1.8)
    for ax in axes:
        style(ax)
        ax.set_xlabel("time since the field and feed start [s]", color=INK2, fontsize=8.5)
    axes[0].set_ylabel("lowest point of the melt above the collector [mm]", color=INK2, fontsize=8.5)
    axes[0].set_ylim(0, 5.6)
    axes[1].set_ylabel("melt within 1 mm of the tip face [mm$^3$]", color=INK2, fontsize=8.5)
    axes[2].set_ylabel("mean temperature of the melt below the tip face [C]", color=INK2, fontsize=8.5)
    axes[2].axhspan(30, 40, color=GRID, alpha=0.7, lw=0)
    axes[2].text(0.98, 35, "nominal freezing band", transform=axes[2].get_yaxis_transform(),
                 ha="right", va="center", fontsize=7, color=INK2)
    axes[2].set_ylim(20, 125)
    for ax, ttl in zip(axes, ("a  the approach to the collector (dot: first contact)",
                              "b  melt held at the tip, to contact",
                              "c  the melt's mean temperature, to contact")):
        ax.set_title(ttl, loc="left", fontsize=9, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.14, 1, 0.97))
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


def a(name):
    return os.path.join(AXI, name + "_series.csv"), os.path.join(AXI, name + ".log")


def compare(out):
    noz = os.path.join(RES, "mew_nozzle", "R4_E_feed_0p0208_6p5kV_lc5_mu820")
    c3d = os.path.join(RES, "mew_cone3d", "R4_feed_0p0208_6p5kV_lc5_mu820")
    runs = [
        ("r-z, isothermal 120 C, 50 um", *a("R4_iso_nu6"), 0, "-"),
        ("r-z, isothermal 120 C, 25 um", *a("R8_iso_nu6"), 0, "--"),
        ("r-z with heat, 50 um", *a("R4_heat_nu6_150s"), 1, "-"),
        ("r-z with heat, 25 um", *a("R8_heat_nu6"), 1, "--"),
        ("3-D, isothermal, melt conducting to 0.5 mm", c3d + "_series.csv", c3d + ".log", 6, ":"),
        ("planar, isothermal, melt conducting to 0.5 mm", noz + "_series.csv", noz + ".log", 4, ":"),
    ]
    three_panels(out, runs, "The three MEW models at 6.5 kV, 5.5 mm, 0.0208 mm$^3$/s, PCL at 820 Pa s")


def sens(out):
    runs = [
        ("nominal: h 90 W/m$^2$K, air 25 C, freezing 30-40 C", *a("R4_heat_best"), 1, "-"),
        ("no freezing drag", *a("R4_heat_nofreeze"), 2, "-"),
        ("air at 50 C", *a("R4_heat_air50"), 3, "-"),
        ("air at 40 C", *a("R4_heat_air40"), 3, "--"),
        ("air at 35 C", *a("R4_heat_air35"), 3, ":"),
        ("freezing 20-30 C", *a("R4_heat_freeze20"), 6, "-"),
        ("h 30 W/m$^2$K", *a("R4_heat_h30"), 0, "-"),
        ("h 300 W/m$^2$K", *a("R4_heat_h300"), 0, "--"),
    ]
    three_panels(out, runs, "The r-z model with heat: what its unmeasured parameters are worth (50 um)")


def nearest(prefix, t):
    best, bt = None, 1e9
    for f in glob.glob(prefix + "_t*_fill.npy"):
        m = re.search(r"_t([0-9.]+)_fill\.npy$", f)
        if m and abs(float(m.group(1)) - t) < bt:
            bt, best = abs(float(m.group(1)) - t), f
    return best


def shapes(out):
    specs = [("isothermal, 50 um", "R4_iso_nu6", (5, 10, 14, 15.5), 0.05),
             ("isothermal, 25 um", "R8_iso_nu6", (3, 6, 8.5, 9.5), 0.025),
             ("heat, the best model", "R4_heat_best", (10, 20, 40, 60), 0.05),
             ("heat, no freezing", "R4_heat_nofreeze", (10, 20, 30, 34), 0.05),
             ("heat, air at 50 C", "R4_heat_air50", (10, 20, 25, 28), 0.05)]
    fig, axes = plt.subplots(len(specs), 4, figsize=(9.3, 2.6 * len(specs)), facecolor=SURF, squeeze=False)
    last = None
    for i, (lab, name, ts, dx) in enumerate(specs):
        for j, t in enumerate(ts):
            ax = axes[i][j]
            ax.set_facecolor(SURF)
            f = nearest(os.path.join(AXI, name), t)
            if f is None:
                ax.set_axis_off()
                continue
            fill = np.load(f)
            ny, nx = fill.shape
            full = np.concatenate([fill[:, ::-1], fill], axis=1)
            x = (np.arange(-nx, nx) + 0.5) * dx
            y = (np.arange(ny) - 0.5) * dx
            ax.pcolormesh(x, y, np.where(full < 0, 1.0, np.nan),
                          cmap=LinearSegmentedColormap.from_list("s", ["#b9b8b2", "#b9b8b2"]), shading="nearest")
            liq = np.where(full > 0, 1.0, np.nan)
            tf = f.replace("_fill.npy", "_T.npy")
            if os.path.exists(tf):
                T = np.load(tf)
                TT = np.concatenate([T[:, ::-1], T], axis=1)
                last = ax.pcolormesh(x, y, np.where(np.isfinite(liq), TT, np.nan), cmap=TMAP, vmin=25, vmax=120,
                                     shading="nearest")
            else:
                ax.pcolormesh(x, y, liq, cmap=LinearSegmentedColormap.from_list("m", [SERIES[0], SERIES[0]]),
                              shading="nearest")
            ax.contour(x, y, np.clip(full, 0, 1), levels=[0.5], colors=INK, linewidths=0.6)
            tt = float(re.search(r"_t([0-9.]+)_fill", f).group(1))
            ax.set_title("%s, t = %.1f s" % (lab, tt), fontsize=7.5, color=INK)
            ax.set_aspect("equal")
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(0, 6.8)
            ax.tick_params(colors=INK2, labelsize=6)
            for s in ax.spines.values():
                s.set_color(MUTED)
            if j == 0:
                ax.set_ylabel("height above the collector [mm]", fontsize=6.5, color=INK2)
            if i == len(specs) - 1:
                ax.set_xlabel("r [mm], mirrored", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    if last is not None:
        cax = fig.add_axes([0.935, 0.3, 0.012, 0.4])
        cb = fig.colorbar(last, cax=cax)
        cb.set_label("melt temperature [C]   (blue: isothermal at 120 C)", fontsize=7, color=INK2)
        cb.ax.tick_params(labelsize=6, colors=INK2)
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("compare", "sens", "shapes"):
        print(__doc__)
        sys.exit(2)
    {"compare": compare, "sens": sens, "shapes": shapes}[sys.argv[1]](sys.argv[2])
