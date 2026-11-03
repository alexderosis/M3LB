#!/usr/bin/env python3
"""Figure for demonstrator/mew_liquefier: where a fed PCL filament melts.

Reads the CSVs that `mew_liquefier -sweep -out <prefix>` writes and draws three
panels: (a) the steady temperature field with its f_l = 1/2 contour for four
configurations, (b) the melt-front height against the cold-end wall temperature,
(c) the melt length after a fresh reload against time.

    python3 tools/plot_mew_liquefier.py results/mew_liquefier/run results/mew_liquefier/mew_liquefier.png
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Nominal hot end, as in the driver's Physical struct.
LH, LB, LTOT = 10.0, 3.0, 25.0
TS, TL, TH = 57.0, 63.0, 120.0

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read(path):
    with open(path) as f:
        r = csv.DictReader(f)
        cols = {k: [] for k in r.fieldnames}
        for row in r:
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def field(prefix, tag):
    d = read(f"{prefix}_{tag}_field.csv")
    xs, zs = np.unique(d["x_mm"]), np.unique(d["z_mm"])
    T = np.full((zs.size, xs.size), np.nan)
    F = np.full_like(T, np.nan)
    ix = np.searchsorted(xs, d["x_mm"])
    iz = np.searchsorted(zs, d["z_mm"])
    T[iz, ix], F[iz, ix] = d["T_C"], d["f_l"]
    return xs, zs, T, F


def z_eq(tc):
    tm = 0.5 * (TS + TL)
    return LTOT if tc >= tm else LH + LB * (TH - tm) / (TH - tc)


def main(prefix, out):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(10.5, 6.6), facecolor="#fcfcfb")
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 0.35, 4.2],
                          height_ratios=[1, 1, 0.06], wspace=0.30, hspace=0.55)

    # (a) fields
    cases = [("MEW_Tc30", "MEW\nT_c 30 C"), ("MEW_Tc55", "MEW\nT_c 55 C"),
             ("MEW_Tc65", "MEW\nT_c 65 C"), ("FDM_Tc30", "FDM\nT_c 30 C")]
    im = None
    for k, (tag, title) in enumerate(cases):
        ax = fig.add_subplot(gs[0:2, k])
        xs, zs, T, F = field(prefix, tag)
        im = ax.pcolormesh(xs, zs, T, cmap="Oranges", vmin=25, vmax=120, shading="nearest",
                           rasterized=True)
        ax.contour(xs, zs, F, levels=[0.5], colors=INK, linewidths=1.0)
        for zl in (LH, LH + LB):
            ax.axhline(zl, color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(0, LTOT)
        ax.set_xticks([0, 1.75]); ax.set_xticklabels(["0", "1.75"])
        ax.set_title(title, fontsize=8.5, color=INK)
        if k == 0:
            ax.set_ylabel("height above nozzle exit [mm]")
            box = dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85)
            ax.text(0.06, LH - 0.35, "heater top", fontsize=7, color=INK, va="top", bbox=box)
            ax.text(0.06, LH + LB + 0.35, "break top", fontsize=7, color=INK, va="bottom",
                    bbox=box)
        else:
            ax.set_yticklabels([])
        if k == 0:
            ax.set_xlabel("x [mm]", fontsize=8)
    cax = fig.add_subplot(gs[2, 0:4])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("T [C]    black line: liquid fraction f_l = 1/2    (x not to scale)",
                 color=INK, fontsize=8)
    cb.outline.set_linewidth(0.4)
    fig.text(0.02, 0.965, "(a)", fontsize=10, weight="bold")

    # (b) front height vs T_c
    axb = fig.add_subplot(gs[0, 5])
    tcs = np.array([30.0, 45.0, 55.0, 65.0])
    zax = {}
    for feed in ("MEW", "FDM"):
        zax[feed] = []
        for tc in tcs:
            s = read(f"{prefix}_{feed}_Tc{tc:.0f}_series.csv")
            zax[feed].append(s["z_axis_mm"][-1])
    tt = np.linspace(30, 65, 200)
    axb.plot(tt, [z_eq(t) for t in tt], color=INK2, lw=1.0, ls=(0, (4, 2)))
    axb.text(38, z_eq(38) + 0.7, "where the wall reaches 60 C", fontsize=7.5, color=INK2)
    # Markers only: four sampled T_c, and the step between 55 and 65 C is a
    # threshold, which a line drawn through it would hide.
    axb.plot(tcs, zax["MEW"], ls="none", color=BLUE, marker="o", ms=7, mec="#fcfcfb", mew=1.5,
             label="MEW feed (0.255 mm^3/s)")
    axb.plot(tcs, zax["FDM"], ls="none", color=ORANGE, marker="o", ms=7, mec="#fcfcfb",
             mew=1.5, label="FDM feed (4.09 mm^3/s)")
    axb.text(38, zax["MEW"][1] - 2.3, "MEW feed", color=INK, fontsize=8)
    axb.text(38, zax["FDM"][1] + 1.1, "FDM feed", color=INK, fontsize=8)
    axb.legend(frameon=False, fontsize=7.5, loc="upper left", title="front on the axis",
               title_fontsize=7.5, alignment="left")
    axb.axhline(LH, color=GRID, lw=0.8, zorder=0)
    axb.set_xlabel("cold-end wall temperature T_c [C]")
    axb.set_ylabel("melt front, f_l = 1/2 [mm]")
    axb.set_ylim(0, 26)
    axb.grid(axis="y", color=GRID, lw=0.5)
    axb.spines[["top", "right"]].set_visible(False)
    axb.text(-0.12, 1.04, "(b)", transform=axb.transAxes, fontsize=10, weight="bold")

    # (c) melt length after a fresh reload
    axc = fig.add_subplot(gs[1, 5])
    for tag, col, lab in [("MEW_Tc30", BLUE, "MEW, T_c 30 C"),
                          ("MEW_Tc65", AQUA, "MEW, T_c 65 C"),
                          ("FDM_Tc30", ORANGE, "FDM, T_c 30 C")]:
        s = read(f"{prefix}_{tag}_series.csv")
        t = np.concatenate([[0.0], s["t_s"]])
        l = np.concatenate([[0.0], s["l_melt_mm"]])
        # A run stops once steady (field change < 1e-9 per quarter flush), which
        # is before 60 s for FDM; the steady value is held to the edge.
        if t[-1] < 60.0:
            t, l = np.append(t, 60.0), np.append(l, l[-1])
        axc.plot(t, l, color=col, lw=2, label=lab)
        axc.text(44, l[np.searchsorted(t, 44.0)] + 0.6, lab, color=INK, fontsize=8)
    axc.set_xlim(0, 60)
    axc.set_ylim(0, 25)
    axc.set_xlabel("time after reloading fresh filament [s]")
    axc.set_ylabel("melt area / W [mm]")
    axc.grid(axis="y", color=GRID, lw=0.5)
    axc.spines[["top", "right"]].set_visible(False)
    axc.legend(frameon=False, fontsize=7.5, loc="lower right")
    axc.text(-0.12, 1.04, "(c)", transform=axc.transAxes, fontsize=10, weight="bold")

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
