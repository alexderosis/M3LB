"""Figures for the r-z MEW note (paper/mew_axi/main.tex).

usage: python3 make_figs.py [results/mew_axi]

Reads the series and logs that demonstrator/mew_axi wrote into results/mew_axi
(tracked) and, for the drop shapes, the .npy frames beside them (git-ignored:
rerun the cases in the footer of main.tex first). Writes fig/fig1_heat.pdf and
fig/fig2_decides.pdf. Melt held at 120 C is blue, as in the planar note;
cooling melt is orange; the nominal cooled run is ink in Fig. 2 so that every
panel varies one thing against the same line.
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

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "results", "mew_axi")
OUT = os.path.join(HERE, "fig")

INK, INK2, INK3, RULE = "#111110", "#4f4e4a", "#807f79", "#dedcd5"
ISO, HEAT = "#2a78d6", "#d4572a"                      # held at 120 C / allowed to cool
AIR = {35: "#f2a477", 40: "#e0703e", 50: "#a8401a"}    # warmer air, one hue, light to dark
HCOL = {30: "#b3a8ee", 300: "#4a3bb0"}                 # the surface's heat-transfer coefficient
NOFRZ, BAND, OFF, ON60 = "#1baf7a", "#e87ba4", "#8a8984", "#eda100"
DASH, DOT = (0, (4, 2.5)), (0, (1, 1.6))
SOLID_RGB = (0.62, 0.62, 0.60)
TMAP = LinearSegmentedColormap.from_list("T", ["#fde3d6", "#f6a882", "#eb6834", "#b8461b", "#6e2a10"])
TMIN, TMAX = 25.0, 120.0

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
    "xtick.color": INK3, "ytick.color": INK3, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def series(name):
    return np.genfromtxt(os.path.join(RES, name + "_series.csv"), delimiter=",", names=True)


def contact(name):
    for line in open(os.path.join(RES, name + ".log")):
        m = re.search(r"first contact with the collector at t = ([0-9.]+)", line)
        if m:
            return float(m.group(1))
    return None


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=RULE, lw=0.5)
    ax.set_axisbelow(True)


def approach(ax, name, col, ls, lab, lw=1.4):
    """The lowest point of the melt against time, to first contact (a dot there)."""
    s = series(name)
    t, z = s["t_s"], s["z_low_mm"]
    tc = contact(name)
    keep = t <= (tc if tc is not None else 1e9)
    tz, zz = t[keep], z[keep]
    if tc is not None:                       # after contact the lowest point is the collector
        tz, zz = np.append(tz, tc), np.append(zz, 0.0)
    ax.plot(tz, zz, color=col, ls=ls, lw=lw, label=lab)
    if tc is not None:
        ax.plot([tc], [0.0], "o", ms=4, color=col, mec="white", mew=0.8, zorder=5, clip_on=False)


def nearest(prefix, t):
    files = glob.glob(prefix + "_t*_fill.npy")
    ts = np.array([float(re.search(r"_t([0-9.]+)_fill\.npy$", f).group(1)) for f in files])
    k = int(np.argmin(abs(ts - t)))
    return ts[k], files[k]


def drop(ax, name, t, dx):
    """The melt mirrored about the axis: blue if held at 120 C, else coloured by temperature."""
    tt, f = nearest(os.path.join(RES, name), t)
    fill = np.load(f)
    ny, nx = fill.shape
    full = np.concatenate([fill[:, ::-1], fill], axis=1)
    x = (np.arange(-nx, nx) + 0.5) * dx
    y = (np.arange(ny) - 0.5) * dx
    img = np.ones(full.shape + (4,))
    img[..., 3] = 0.0
    img[full < 0] = SOLID_RGB + (1.0,)
    img[0, :] = SOLID_RGB + (1.0,)                         # the collector row
    liq = full > 0
    tf = f.replace("_fill.npy", "_T.npy")
    if os.path.exists(tf):
        T = np.load(tf)
        TT = np.concatenate([T[:, ::-1], T], axis=1)
        rgba = TMAP(np.clip((TT - TMIN) / (TMAX - TMIN), 0, 1))
        img[liq] = rgba[liq]
    else:
        img[liq] = matplotlib.colors.to_rgba(ISO)
    ext = (x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[0] - 0.5 * dx, y[-1] + 0.5 * dx)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest", rasterized=True)
    ax.contour(x, y, np.clip(full, 0, 1), levels=[0.5], colors=INK, linewidths=0.45)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(0, 6.8)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    for s in ax.spines.values():
        s.set_color(INK3)
    return tt


def fig1():
    """(a) The approach to the collector, held at 120 C and cooling, on two grids, field on and
    off; (b-e) the drop just before contact or at its stall, coloured by temperature."""
    fig = plt.figure(figsize=(6.5, 2.45))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.05, 3.0], wspace=0.16)
    gs = outer[1].subgridspec(1, 5, width_ratios=[0.62, 0.62, 0.62, 0.62, 0.07], wspace=0.12)
    ax = fig.add_subplot(outer[0])
    approach(ax, "R4_iso_nu6", ISO, "-", "120 °C, 50 µm")
    approach(ax, "R8_iso_nu6", ISO, DASH, "120 °C, 25 µm")
    approach(ax, "R4_heat_nu6_150s", HEAT, "-", "cooling, 50 µm")
    approach(ax, "R8_heat_nu6", HEAT, DASH, "cooling, 25 µm")
    approach(ax, "R4_off_iso_300s", ISO, DOT, "120 °C, field off", lw=1.2)
    approach(ax, "R4_off_heat", HEAT, DOT, "cooling, field off", lw=1.2)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 5.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lowest point of the melt [mm]")
    style(ax)
    ax.legend(loc="lower left", frameon=False, handlelength=2.3, labelspacing=0.22, ncol=1,
              bbox_to_anchor=(0.13, 0.0))
    ax.set_title("(a)", loc="left", color=INK)
    shots = [("R4_iso_nu6", 15.5, "(b) 120 °C"), ("R4_heat_best", 60.0, "(c) cooling"),
             ("R4_heat_air50", 27.5, "(d) air 50 °C"), ("R4_heat_nofreeze", 33.4, "(e) no freezing")]
    for k, (name, t, lab) in enumerate(shots):
        axp = fig.add_subplot(gs[k])
        tt = drop(axp, name, t, 0.05)
        axp.set_title(f"{lab}\nt = {tt:.1f} s", loc="left", color=INK, fontsize=7.2, linespacing=1.1)
        axp.set_xlabel("r [mm]", fontsize=7)
        if k:
            axp.set_yticklabels([])
        else:
            axp.set_ylabel("height [mm]")
    cax = fig.add_subplot(gs[4])
    sm = matplotlib.cm.ScalarMappable(cmap=TMAP, norm=matplotlib.colors.Normalize(TMIN, TMAX))
    cb = fig.colorbar(sm, cax=cax, ticks=[25, 40, 60, 80, 100, 120])
    cb.set_label("melt temperature [°C]", fontsize=7, color=INK2, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, colors=INK3, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.07, right=0.925, bottom=0.17, top=0.87)
    fig.savefig(os.path.join(OUT, "fig1_heat.pdf"), dpi=300)
    plt.close(fig)


def fig2():
    """What decides it, on 50 um cells: (a) the air at the tip and the freezing drag, (b) the
    surface's cooling and the freezing band, (c) the field."""
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.15), sharey=True)
    ax = axes[0]
    approach(ax, "R4_heat_best", INK, "-", "25 °C", lw=1.6)
    for T in (35, 40, 50):
        approach(ax, f"R4_heat_air{T}", AIR[T], "-", f"{T} °C")
    approach(ax, "R4_heat_nofreeze", NOFRZ, DASH, "no freezing")
    ax.set_xlim(0, 60)
    ax.set_title("(a) the air at the tip", loc="left", color=INK)
    ax = axes[1]
    approach(ax, "R4_heat_best", INK, "-", "90", lw=1.6)
    approach(ax, "R4_heat_h30", HCOL[30], "-", "30")
    approach(ax, "R4_heat_h300", HCOL[300], "-", "300")
    approach(ax, "R4_heat_freeze20", BAND, DASH, "freezing at 20–30 °C")
    ax.set_xlim(0, 60)
    ax.set_title("(b) the cooling, h in W/m²K", loc="left", color=INK)
    ax = axes[2]
    approach(ax, "R4_heat_nu6_150s", INK, "-", "field on", lw=1.6)
    approach(ax, "R4_off_heat", OFF, DOT, "field off", lw=1.4)
    approach(ax, "R4_heat_fieldon60", ON60, DASH, "on after 60 s")
    ax.axvline(60, color=INK3, lw=0.5, ls=DOT)
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_title("(c) the field", loc="left", color=INK)
    for k, ax in enumerate(axes):
        ax.set_ylim(0, 5.6)
        ax.set_xlabel("time [s]")
        style(ax)
        ax.legend(loc="lower right" if k < 2 else "upper right", frameon=False,
                  handlelength=1.2 if k == 0 else 2.1, handletextpad=0.4, labelspacing=0.22, ncol=2 if k == 1 else 1,
                  columnspacing=0.9)
    axes[0].set_ylabel("lowest point of the melt [mm]")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.9, wspace=0.08)
    fig.savefig(os.path.join(OUT, "fig2_decides.pdf"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig2()
    print("wrote", os.path.join(OUT, "fig1_heat.pdf"), "and fig2_decides.pdf")
