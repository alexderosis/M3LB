#!/usr/bin/env python3
"""Figures for demonstrator/mew_axi, the r-z melt-electrowriting model.

Reads results/mew_axi/ (series CSVs and logs, and for `shapes` the .npy frames,
which git ignores and the runs rewrite), plus the planar and 3-D siblings'
series for `compare`:

    python3 tools/plot_mew_axi.py compare results/mew_axi/mew_axi_compare.png
    python3 tools/plot_mew_axi.py sens    results/mew_axi/mew_axi_sens.png
    python3 tools/plot_mew_axi.py shapes  results/mew_axi/mew_axi_shapes.png
    python3 tools/plot_mew_axi.py jet     results/mew_axi/mew_axi_jet.png
    python3 tools/plot_mew_axi.py cycle   results/mew_axi/mew_axi_cycle.png

compare: the approach to the collector, the melt held at the tip and the melt's
mean temperature, for the isothermal r-z model on two grids, the model with
heat, and the planar and 3-D siblings. sens: the same three panels for the
model with heat under each of its unmeasured parameters. shapes: the drop,
mirrored about the axis, coloured by temperature. jet: the same three panels
with the jet sink (mew_axi -jet) at the fibres' flow without and with the fan,
isothermal and with heat, against the runs without it. cycle: the heater's
on-off cycle (mew_axi -cycle, -breathe) with the measured activation energy --
what the nozzle does on top, what the drop does below.
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


def jet(out):
    """The jet sink (mew_axi -jet): the fibres' flow taken at the apex, without and with the fan."""
    runs = [
        ("isothermal, no jet", *a("R4_iso_nu6"), 0, "-"),
        ("isothermal, fan's jet 0.0036 mm$^3$/s", *a("R4_iso_jet036"), 0, "--"),
        ("isothermal, no-fan jet 0.0116 mm$^3$/s", *a("R4_iso_jet116"), 0, ":"),
        ("heat, no jet", *a("R4_heat_best"), 1, "-"),
        ("heat, fan's jet 0.0036 mm$^3$/s", *a("R4_heat_jet036"), 1, "--"),
        ("heat, no-fan jet 0.0116 mm$^3$/s", *a("R4_heat_jet116"), 1, ":"),
        ("heat at h 300 (a fan's cooling), no jet", *a("R4_heat_h300"), 2, "-"),
        ("heat at h 300, fan's jet 0.0036 mm$^3$/s", *a("R4_heat_jet036_h300"), 2, "--"),
    ]
    three_panels(out, runs, "The r-z model with the jet sink: the fibres' flow taken at the apex (50 um)")


def cycle(out):
    """The heater's cycle on the nozzle (mew_axi -cycle 110 128 13 53, -breathe): six panels, one x axis."""
    runs = [
        ("steady 120 C", "R4_heat_Ea41", 0, "-"),
        ("cycle, temperature only", "R4_heat_Ea41_cycT", 1, "-"),
        ("cycle with the melt's expansion, from switch-on", "R4_heat_Ea41_cyc", 2, "-"),
        ("cycle with the melt's expansion, from switch-off", "R4_heat_Ea41_cyc_ph13", 3, "-"),
    ]
    ref = ("steady 120 C with the earlier 31.3 kJ/mol (to 60 s)", "R4_heat_best")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), facecolor=SURF, sharex=True,
                             gridspec_kw={"height_ratios": [0.62, 1.0]})
    top, bot = axes
    tmax = 0.0
    for lab, name, k, ls in runs:
        csv, log = a(name)
        if not os.path.exists(csv):
            print("missing", csv)
            continue
        s = np.genfromtxt(csv, delimiter=",", names=True)
        c = SERIES[k]
        t = s["t_s"]
        tmax = max(tmax, t[-1])
        tc = contact(log)
        keep = t <= (tc if tc is not None else 1e9)
        names = s.dtype.names
        Tn = s["T_nozzle_C"] if "T_nozzle_C" in names else np.full_like(t, 120.0)
        qo = s["out_of_bore_mm3_s"] if "out_of_bore_mm3_s" in names else np.full_like(t, 0.0208)
        fed = s["fed_mm3"] if "fed_mm3" in names else 0.0208 * t
        # Lines that coincide by construction are drawn dashed over their twin: the two
        # cycles from switch-on share a temperature, the two steady feeds a flow.
        dash = (0, (5, 4))
        top[0].plot(t, Tn, color=c, ls=dash if k == 1 else ls, lw=1.8, zorder=5 - k)
        top[1].plot(t, qo, color=c, ls=dash if k == 0 else ls, lw=1.8, zorder=5 - k)
        top[2].plot(t, fed - 0.0208 * t, color=c, ls=dash if k == 0 else ls, lw=1.8, zorder=5 - k)
        bot[0].plot([], [], color=c, ls=ls, lw=1.8, label=lab)
        bot[0].plot(t[keep], s["z_low_mm"][keep], color=c, ls=ls, lw=1.8)
        if tc is not None:
            bot[0].plot([tc], [0.0], "o", color=c, ms=5.5, mec=SURF, mew=1.2, clip_on=False)
        bot[1].plot(t[keep], s["tip_zone_mm3"][keep], color=c, ls=ls, lw=1.8)
        bot[2].plot(t[keep], s["T_gap_mean_C"][keep], color=c, ls=ls, lw=1.8)
    csv, _ = a(ref[1])
    if os.path.exists(csv):
        s = np.genfromtxt(csv, delimiter=",", names=True)
        bot[0].plot(s["t_s"], s["z_low_mm"], color=MUTED, ls="--", lw=1.4, label=ref[0])
        bot[1].plot(s["t_s"], s["tip_zone_mm3"], color=MUTED, ls="--", lw=1.4)
        bot[2].plot(s["t_s"], s["T_gap_mean_C"], color=MUTED, ls="--", lw=1.4)
    for ax in axes.ravel():
        style(ax)
        ax.set_xlim(0, tmax)
    for ax in bot:
        ax.set_xlabel("time since the field and feed start [s]", color=INK2, fontsize=8.5)
    top[0].set_ylabel("nozzle and the melt it feeds [C]", color=INK2, fontsize=8.5)
    top[1].set_ylabel("melt leaving the bore [mm$^3$/s]", color=INK2, fontsize=8.5)
    top[1].set_ylim(0, None)
    top[2].set_ylabel("delivered, ahead of a steady feed [mm$^3$]", color=INK2, fontsize=8.5)
    bot[0].set_ylabel("lowest point of the melt above the collector [mm]", color=INK2, fontsize=8.5)
    bot[0].set_ylim(0, 5.6)
    bot[1].set_ylabel("melt within 1 mm of the tip face [mm$^3$]", color=INK2, fontsize=8.5)
    bot[2].set_ylabel("mean temperature of the melt below the tip face [C]", color=INK2, fontsize=8.5)
    bot[2].axhspan(30, 40, color=GRID, alpha=0.7, lw=0)
    bot[2].text(0.98, 35, "nominal freezing band", transform=bot[2].get_yaxis_transform(),
                ha="right", va="center", fontsize=7, color=INK2)
    for ax, ttl in zip(axes.ravel(), ("a  the heater's cycle, as measured: 110 to 128 C in 13 s, back in 53 s",
                                      "b  what leaves the bore (feed 0.0208 mm$^3$/s)",
                                      "c  melt delivered, ahead of the feed",
                                      "d  the approach to the collector",
                                      "e  melt held at the tip",
                                      "f  the melt's mean temperature")):
        ax.set_title(ttl, loc="left", fontsize=9, color=INK)
    h1, l1 = bot[0].get_legend_handles_labels()
    fig.legend(h1, l1, frameon=False, fontsize=8, labelcolor=INK, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The r-z model with heat, the measured 40.9 kJ/mol and the heater's on-off cycle (50 um, no jet)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


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
    if len(sys.argv) != 3 or sys.argv[1] not in ("compare", "sens", "shapes", "jet", "cycle"):
        print(__doc__)
        sys.exit(2)
    {"compare": compare, "sens": sens, "shapes": shapes, "jet": jet, "cycle": cycle}[sys.argv[1]](sys.argv[2])
