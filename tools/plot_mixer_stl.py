#!/usr/bin/env python3
"""Concentration slices from demonstrator/mixer_stl's field dump.

    <venv python> tools/plot_mixer_stl.py PREFIX --dx-um 5 --out fig.png

Reads PREFIX_c.npy, PREFIX_ux.npy (shape (nz, ny, nx), NaN in walls) and
PREFIX_series.dat (its last row is the converged M at each station). Draws two
plan views -- inside the obstacle layer and above it -- and the cross-section
at every station, c on a diverging map whose neutral midpoint is c = 1/2
(fully mixed). Walls are charcoal; nodes OUTSIDE [-0.05, 1.05] are yellow,
because c must lie in [0, 1] and a node that leaves it is a numerical defect,
not mixing. Needs numpy and matplotlib (not in the system python).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
import numpy as np

CMAP = LinearSegmentedColormap.from_list(
    "c", ["#104281", "#5598e7", "#f0efec", "#e66767", "#9e1f1f"])
WALL, FLAG = "#3a3a38", "#fab219"


def show(ax, c, extent, title):
    ax.set_facecolor(WALL)
    im = ax.imshow(c, origin="lower", cmap=CMAP, vmin=0, vmax=1, extent=extent,
                   interpolation="nearest", aspect="equal")
    bad = np.where((c < -0.05) | (c > 1.05), 1.0, np.nan)
    ax.imshow(bad, origin="lower", cmap=ListedColormap([FLAG]), extent=extent,
              interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=9, loc="left", color="#3a3a38")
    ax.tick_params(labelsize=7, colors="#6b6b68")
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--dx-um", type=float, required=True)
    ap.add_argument("--zlo", type=int, help="plan-view node z inside the obstacles")
    ap.add_argument("--zhi", type=int, help="plan-view node z above them")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c = np.load(a.prefix + "_c.npy")
    nz, ny, nx = c.shape
    L = ny - 2
    rows = [l.split() for l in open(a.prefix + "_series.dat") if l.strip()]
    st_um = [float(v) for v in rows[0][rows[0].index("(um):") + 1:]]
    last = [float(v) for v in rows[-1][3:]]
    Mf, Ma = last[0::2], last[1::2]
    dx = a.dx_um
    zlo = a.zlo if a.zlo is not None else 1 + L // 4
    zhi = a.zhi if a.zhi is not None else 1 + (3 * L) // 4
    nbad = int(np.sum((c < -0.05) | (c > 1.05)))
    nflu = int(np.sum(~np.isnan(c)))

    fig = plt.figure(figsize=(11, 6.6), facecolor="#fcfcfb")
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.25)
    ext_xy = [-0.5 * dx, (nx - 0.5) * dx, -dx, (ny - 1) * dx]   # wall planes at 0, W
    for r, (z, what) in enumerate(((zlo, "inside the obstacle layer"), (zhi, "above it"))):
        ax = fig.add_subplot(gs[r, :])
        im = show(ax, c[z], ext_xy,
                  "plan view at z = %.0f um, %s" % ((z - 0.5) * dx, what))
        for s in st_um:
            ax.axvline(s, color="#3a3a38", lw=0.6, ls=":")
        ax.set_ylabel("y [um]", fontsize=8)
        if r == 1:
            ax.set_xlabel("x from the inlet [um]", fontsize=8)
    for k, s in enumerate(st_um):
        ax = fig.add_subplot(gs[2, k])
        x = int(round(s / dx))
        show(ax, c[:, :, x].copy(), [-dx, (ny - 1) * dx, -dx, (nz - 1) * dx],
             "x = %.0f um\nM = %.3f (flux)\n      %.3f (area)" % (s, Mf[k], Ma[k]))
        ax.set_xlabel("y [um]", fontsize=7)
        if k == 0:
            ax.set_ylabel("z [um]", fontsize=8)
        else:
            ax.set_yticklabels([])
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.22])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("c", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.legend(handles=[Patch(color=WALL, label="wall"),
                        Patch(color=FLAG, label="c outside [-0.05, 1.05]: %d of %d nodes"
                              % (nbad, nflu))],
               loc="lower left", bbox_to_anchor=(0.06, -0.02), ncol=2, fontsize=7, frameon=False)
    fig.suptitle(a.title, fontsize=10, x=0.06, ha="left", color="#1a1a19")
    fig.savefig(a.out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
