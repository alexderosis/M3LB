"""Figures for the MEW nozzle note (paper/mew_nozzle/main.tex).

usage: python3 make_figs.py [results/mew_nozzle]

Reads the frames and series demonstrator/mew_nozzle writes (the .npy frames are
git-ignored: rerun the four cases first, see the footer of main.tex) and
data/after_contact.csv, and writes fig/fig1_columns.pdf, fig/fig2_tip.pdf and
fig/fig3_contact.pdf. Colours follow tools/anim_mew_nozzle.py, so the note and
the animations read the same: melt blue, field off orange, per-minute aqua and
yellow.
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
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CUT = {"cut1": "#86b6ef", "cut2": "#2a78d6", "cut3": "#104281"}
MELT_RGB = (0.165, 0.471, 0.839)
SOLID_RGB = (0.55, 0.55, 0.53)

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


def fig1():
    """Printed feed to first contact: the field-on melt, with the field-off outline over it."""
    A = os.path.join(RES, "A_feed_s_6p5kV")
    B = os.path.join(RES, "B_feed_s_nofield")
    times = (1.0, 2.0, 3.0, 4.0, 4.65)
    fig, axes = plt.subplots(1, len(times), figsize=(6.5, 2.45), sharey=True)
    for c, t in enumerate(times):
        ax = axes[c]
        tt, fl, ph = frame(A, t)
        draw_melt(ax, fl, ph, field=True, cut=GAP_MM - LC_MM, xlim=(-2.0, 2.0), ylim=(0, 6.6))
        tb, fb, _ = frame(B, t)
        ny, nx = fb.shape
        dx = W_MM / nx
        x = (np.arange(nx) + 0.5) * dx - W_MM / 2
        y = (np.arange(ny) - 0.5) * dx
        ax.contour(x, y, np.ma.masked_less(fb, -0.5), levels=[0.5], colors=ORANGE,
                   linewidths=0.9, linestyles=[(0, (3, 2))])
        ax.set_title(f"t = {tt:.2f} s" if c == len(times) - 1 else f"t = {tt:.1f} s", color=INK)
        ax.set_xlabel("x [mm]")
        if c == 0:
            ax.set_ylabel("height [mm]")
    fig.tight_layout(pad=0.3, w_pad=0.5)
    fig.savefig(os.path.join(OUT, "fig1_columns.pdf"), dpi=300)
    plt.close(fig)


def series(name):
    rows = [r for r in csv.DictReader(open(os.path.join(RES, name + "_series.csv")))
            if all(v not in (None, "") for v in r.values())]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def fig2():
    """(a) melt within 1 mm under the tip, all four runs; (b, c) the per-minute meniscus at 8 s."""
    fig = plt.figure(figsize=(6.5, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    runs = [("A_feed_s_6p5kV", "mm³/s, 6.5 kV", BLUE, "-", 4.69),
            ("B_feed_s_nofield", "mm³/s, field off", ORANGE, (0, (4, 2.5)), 4.68),
            ("C_feed_min_6p5kV", "mm³/min, 6.5 kV", AQUA, "-", None),
            ("D_feed_min_12kV", "mm³/min, 12 kV", YELLOW, (0, (4, 2.5)), None)]
    for name, lab, col, ls, tcut in runs:
        s = series(name)
        m = s["t_s"] <= (tcut if tcut else 1e9)
        ax.plot(s["t_s"][m], s["tip_zone_mm2"][m], color=col, ls=ls, lw=1.5, label=lab)
    ax.set_yscale("log")
    ax.set_xlim(0, 8)
    ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("melt within 1 mm under the tip [mm²]")
    ax.grid(True, which="major", color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # Direct labels; the caption keys the colours (the pairs overlap to the eye).
    ax.text(4.9, 0.62, "0.255 mm³/s\n6.5 kV and field off", va="center", fontsize=7, color=INK2)
    ax.text(1.1, 0.0035, "0.255 mm³/min, 6.5 and 12 kV", fontsize=7, color=INK2)
    ax.set_title("(a)", loc="left", color=INK)
    zoom = dict(xlim=(-0.7, 0.7), ylim=(4.75, 5.75))
    for k, (name, lab) in enumerate((("C_feed_min_6p5kV", "6.5 kV"), ("D_feed_min_12kV", "12 kV"))):
        axz = fig.add_subplot(gs[k + 1])
        tt, fl, ph = frame(os.path.join(RES, name), 8.0)
        draw_melt(axz, fl, ph, field=True, **zoom)
        axz.set_title(f"({'bc'[k]}) {lab}, t = {tt:.1f} s", loc="left", color=INK)
        axz.set_xlabel("x [mm]")
        if k == 0:
            axz.set_ylabel("height [mm]")
        else:
            axz.set_yticklabels([])
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.89)
    fig.savefig(os.path.join(OUT, "fig2_tip.pdf"), dpi=300)
    plt.close(fig)


def fig3():
    """(a, b) the ropes at 6.0 s, field off and on; (c) the lowest attached point after contact."""
    fig = plt.figure(figsize=(6.5, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.75], wspace=0.28)
    for k, (name, lab, field) in enumerate((("B_feed_s_nofield", "field off", False),
                                            ("A_feed_s_6p5kV", "field on", True))):
        ax = fig.add_subplot(gs[k])
        tt, fl, ph = frame(os.path.join(RES, name), 6.0)
        draw_melt(ax, fl, ph, field=field, cut=(GAP_MM - LC_MM) if field else None, ylim=(0, 6.6))
        ax.set_title(f"({'ab'[k]}) {lab}, t = {tt:.1f} s", loc="left", color=INK)
        ax.set_xlabel("x [mm]")
        if k == 0:
            ax.set_ylabel("height [mm]")
        else:
            ax.set_yticklabels([])
    ax = fig.add_subplot(gs[2])
    rows = [r for r in csv.reader(l for l in open(os.path.join(HERE, "data", "after_contact.csv"))
                                  if not l.startswith("#"))]
    head, rows = rows[0], rows[1:]
    t = np.array([float(r[0]) for r in rows])
    style = {"off": ("off", ORANGE, (0, (4, 2.5))),
             "cut1": ("on, 1 mm", CUT["cut1"], "-"),
             "cut2": ("on, 2 mm (b)", CUT["cut2"], "-"),
             "cut3": ("on, 3 mm", CUT["cut3"], "-")}
    for j, key in enumerate(head[1:], start=1):
        z = np.array([float(r[j]) if r[j] else np.nan for r in rows])
        lab, col, ls = style[key]
        ax.plot(t, z, color=col, ls=ls, lw=1.5, label=lab)
    ax.axvline(4.69, color=INK3, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(4.72, 2.42, "first contact", fontsize=6.5, color=INK2, va="top")
    ax.plot([5.87], [0.013], "o", ms=4, color=ORANGE, mec="white", mew=0.8)
    ax.text(5.93, 0.1, "touches again", fontsize=6.5, color=INK2)
    ax.set_xlim(4.55, 6.65)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lowest attached point [mm]")
    ax.grid(True, color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0, bbox_to_anchor=(1.0, 0.12),
              labelspacing=0.3)
    ax.set_title("(c)", loc="left", color=INK)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.9)
    fig.savefig(os.path.join(OUT, "fig3_contact.pdf"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig2()
    fig3()
    print("wrote", ", ".join(sorted(os.listdir(OUT))))
