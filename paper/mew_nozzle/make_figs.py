"""Figures for the MEW nozzle note (paper/mew_nozzle/main.tex).

usage: python3 make_figs.py [results/mew_nozzle]

Reads the series (tracked) and frames (the .npy frames are git-ignored: rerun the
cases first, see the footer of main.tex) that demonstrator/mew_nozzle writes for
the corrected feed, 0.0208 mm^3/s, and writes fig/fig1_real.pdf and
fig/fig2_cut.pdf. Colours follow tools/anim_mew_nozzle.py, so the note and the
animations read the same: melt blue, field off orange; the depth to which the
melt conducts is a light-to-dark blue ramp. The figures of the two earlier
readings of the feed were dropped with the correction; git has them.
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "results", "mew_nozzle")
OUT = os.path.join(HERE, "fig")

W_MM, GAP_MM, LC_MM = 8.0, 5.5, 2.0          # box width, nozzle gap, conducting depth
INK, INK2, INK3, RULE = "#111110", "#4f4e4a", "#807f79", "#dedcd5"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CUT = ("#86b6ef", "#2a78d6", "#104281")      # how deep the melt conducts: 1, 2, 3 mm
DASH = (0, (4, 2.5))
MELT_RGB = (0.165, 0.471, 0.839)
SOLID_RGB = (0.55, 0.55, 0.53)
LIGHT_RGB = (0.70, 0.81, 0.94)               # melt outside the conducting zone

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
    "xtick.color": INK3, "ytick.color": INK3, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def frame(prefix, t):
    """The saved frame nearest t: (time, fill, phi)."""
    files = glob.glob(prefix + "_t*_fill.npy")
    ts = np.array([float(f.split("_t")[-1][:-9]) for f in files])
    f = files[int(np.argmin(abs(ts - t)))]
    return float(f.split("_t")[-1][:-9]), np.load(f), np.load(f.replace("_fill", "_phi"))


def draw_melt(ax, fl, ph, field=True, cut=None, xlim=(-4, 4), ylim=(0, 6.8)):
    ny, nx = fl.shape
    dx = W_MM / nx
    img = np.ones(fl.shape + (3,))
    img[fl >= 0.5] = MELT_RGB
    img[fl < -0.5] = SOLID_RGB
    img[0, :] = SOLID_RGB                                   # the collector row
    ext = (-W_MM / 2, W_MM / 2, -0.5 * dx, (ny - 0.5) * dx)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest", rasterized=True)
    if field:
        x = (np.arange(nx) + 0.5) * dx - W_MM / 2
        y = (np.arange(ny) - 0.5) * dx
        gas = np.where((fl > -0.5) & (fl < 0.5), ph, np.nan)
        ax.contour(x, y, gas, levels=np.linspace(0.1, 0.9, 9), colors=INK2, linewidths=0.3)
    if cut is not None:
        ax.axhline(cut, color=INK3, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")


def series(name):
    rows = [r for r in csv.DictReader(open(os.path.join(RES, name + "_series.csv")))
            if all(v not in (None, "") for v in r.values())]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def fig1():
    """The corrected feed: (a) the lowest point against time, field on and off on two grids;
    (b, c) the drop at 100 s on 25 um cells, field off and on."""
    fig = plt.figure(figsize=(6.5, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1, 1], wspace=0.30)
    ax = fig.add_subplot(gs[0])
    runs = [("F_feed_0p0208_nofield", ORANGE, "-", "off, 25 µm"), ("R4_F_feed_0p0208_nofield", ORANGE, DASH, "off, 50 µm"),
            ("E_feed_0p0208_6p5kV", BLUE, "-", "6.5 kV, 25 µm"), ("R4_E_feed_0p0208_6p5kV", BLUE, DASH, "6.5 kV, 50 µm")]
    for name, col, ls, lab in runs:
        s = series(name)
        t, z = s["t_s"], s["z_low_mm"]
        dx = 0.05 if name.startswith("R4") else 0.025
        hit = np.nonzero(z <= 0.5 * dx * 1.0001)[0]
        n = hit[0] + 1 if len(hit) else len(t)
        ax.plot(t[:n], z[:n], color=col, ls=ls, lw=1.4, label=lab)
        if len(hit):
            ax.plot([t[hit[0]]], [0], "o", ms=4, color=col, mec="white", mew=0.8, zorder=5, clip_on=False)
    ax.axhline(GAP_MM - LC_MM, color=INK3, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_xlim(0, 125); ax.set_ylim(0, 5.6)
    ax.set_xlabel("time [s]"); ax.set_ylabel("lowest point of the melt [mm]")
    style(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=2.2, labelspacing=0.25, fontsize=6.8)
    ax.set_title("(a)", loc="left", color=INK)
    for k, (name, field, lab) in enumerate((("F_feed_0p0208_nofield", False, "off"), ("E_feed_0p0208_6p5kV", True, "6.5 kV"))):
        axp = fig.add_subplot(gs[k + 1])
        tt, fl, ph = frame(os.path.join(RES, name), 100.0)
        if field:
            draw_conducting(axp, fl, ph, marks=0)
        else:
            draw_melt(axp, fl, ph, field=False, xlim=(-2.0, 2.0), ylim=(0, 6.3))
        axp.set_title(f"({'bc'[k]}) {lab}, t = {tt:.0f} s", loc="left", color=INK)
        axp.set_xlabel("x [mm]")
        if k == 0:
            axp.set_ylabel("height [mm]")
        else:
            axp.set_yticklabels([])
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.9)
    fig.savefig(os.path.join(OUT, "fig1_real.pdf"), dpi=300)
    plt.close(fig)


def fig2():
    """How deep the melt conducts, 50 um cells: (a) the lowest point against time, field off and
    on with the melt conducting to 1, 2 and 3 mm below the tip; (b) the 2 mm run at 80 s, where
    the strongest pull sits."""
    fig = plt.figure(figsize=(6.5, 2.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.16)
    ax = fig.add_subplot(gs[0])
    runs = [("R4_F_feed_0p0208_nofield", ORANGE, None, "field off"),
            ("R4_E_feed_0p0208_6p5kV_lc1", CUT[0], 1.0, "on, conducts to 1 mm"),
            ("R4_E_feed_0p0208_6p5kV", CUT[1], 2.0, "on, to 2 mm (Fig. 1)"),
            ("R4_E_feed_0p0208_6p5kV_lc3", CUT[2], 3.0, "on, to 3 mm")]
    for name, col, lc, lab in runs:
        s = series(name)
        t, z = s["t_s"], s["z_low_mm"]
        hit = np.nonzero(z <= 0.025 * 1.0001)[0]
        n = hit[0] + 1 if len(hit) else len(t)
        ax.plot(t[:n], z[:n], color=col, lw=1.4, label=lab)
        if len(hit):
            ax.plot([t[hit[0]]], [0], "o", ms=4, color=col, mec="white", mew=0.8, zorder=5, clip_on=False)
        if lc is not None:
            h = GAP_MM - lc
            ax.axhline(h, color=col, lw=0.7, ls=(0, (1, 1.5)))
            j = np.nonzero(z <= h)[0][0]
            ax.plot([t[j]], [z[j]], "D", ms=3.6, color=col, mec="white", mew=0.6, zorder=6)
    ax.set_xlim(0, 140); ax.set_ylim(0, 5.6)
    ax.set_xlabel("time [s]"); ax.set_ylabel("lowest point of the melt [mm]")
    style(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=2.0, labelspacing=0.25, fontsize=6.8)
    ax.set_title("(a)", loc="left", color=INK)
    # (b) the default run below its cut: conducting melt dark, the rest light, and the strongest pull
    axp = fig.add_subplot(gs[1])
    tt, fl, ph = frame(os.path.join(RES, "R4_E_feed_0p0208_6p5kV"), 80.0)
    draw_conducting(axp, fl, ph, marks=4)
    axp.set_xlim(-2.0, 2.0); axp.set_ylim(0, 6.3); axp.set_aspect("equal")
    axp.set_title(f"(b) 6.5 kV, 2 mm, t = {tt:.0f} s", loc="left", color=INK)
    axp.set_xlabel("x [mm]")
    axp.set_ylabel("height [mm]")
    axp.set_anchor("W")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.9)
    fig.savefig(os.path.join(OUT, "fig2_cut.pdf"), dpi=300)
    plt.close(fig)


def draw_conducting(ax, fl, ph, marks=0):
    """A field-on frame by what the potential sees: melt that conducts (at the nozzle's potential)
    dark, melt below the conducting depth light, equipotentials every 10 % through everything the
    potential treats as gas, the depth dotted, and the `marks` surface cells pulled hardest (the
    3x3 mean of |grad phi|^2 over the gas, as validation/FsElectric.hpp's apply() takes it)."""
    ny, nx = fl.shape
    dx = W_MM / nx
    V = ph.max()
    cond = np.isclose(ph, V) & (fl > -0.5)
    melt = fl >= 0.5
    img = np.ones(fl.shape + (3,))
    img[melt] = LIGHT_RGB
    img[melt & cond] = MELT_RGB
    img[fl < -0.5] = SOLID_RGB
    img[0, :] = SOLID_RGB
    ax.imshow(img, origin="lower", extent=(-W_MM / 2, W_MM / 2, -0.5 * dx, (ny - 0.5) * dx),
              interpolation="nearest", rasterized=True)
    x = (np.arange(nx) + 0.5) * dx - W_MM / 2
    y = (np.arange(ny) - 0.5) * dx
    bulk = ~cond & (fl > -0.5)
    ax.contour(x, y, np.where(bulk, ph / V, np.nan), levels=np.linspace(0.1, 0.9, 9), colors=INK2, linewidths=0.3)
    if marks:
        gy, gx = np.gradient(ph)
        e2 = np.where(bulk, gx ** 2 + gy ** 2, 0.0)
        iface = (fl > 0) & (fl < 1) & cond
        pulls = []
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                b = bulk[j - 1:j + 2, i - 1:i + 2]
                if iface[j, i] and b.any():
                    pulls.append((e2[j - 1:j + 2, i - 1:i + 2][b].mean(), j, i))
        pulls.sort(reverse=True)
        for _, j, i in pulls[:marks]:
            ax.plot([x[i]], [y[j]], "o", ms=4.2, color=ORANGE, mec="white", mew=0.6, zorder=5)
    ax.axhline(GAP_MM - LC_MM, color=INK3, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(0, 6.3)
    ax.set_aspect("equal")


def style(ax):
    ax.grid(True, color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig2()
    print("wrote", ", ".join(sorted(os.listdir(OUT))))
