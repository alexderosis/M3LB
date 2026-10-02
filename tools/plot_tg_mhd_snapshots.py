#!/usr/bin/env python3
"""Snapshots of where the confined-MHD boxes dissipate: the paper's field figure.

    plot_tg_mhd_snapshots.py A_DIR B_DIR [C_DIR] [--t 2.3] [-o PREFIX] [--fmt png,pdf]
                             [--section y128] [--wall z3] [--range 1e-2 1e2]

Each DIR is a run directory that tools/tg_mhd_slices.py has sliced
(GPU/csf3/tg_mhd_snap.sub writes them; results/P_tg_mhd/snap/<run>/ once copied
back), holding slices/index.json and the planes it lists. The figure, at the dump
nearest --t, is one row per box (free slip, no slip, no field) and four columns:

  1, 2  the vertical section --section (default the y-plane nearest pi/4): the
        viscous and the Ohmic dissipation, nu|w|^2 and eta|j|^2, each over the box
        mean eps at that time -- the same normalisation as the paper's Fig. 1(a)
        and (b), whose shares are integrals of these fields. Dashed lines lie at
        delta = Re^-1/2 from the four walls of the section: f_1's layer.
  3, 4  the wall-parallel plane --wall (default the z-plane nearest delta/2 from
        the z = 0 wall), viscous and Ohmic, on the same scale.

The no-field box has no Ohmic panels. One logarithmic colour scale, --range,
serves every panel, so a colour means the same multiple of the mean everywhere.

Needs numpy and matplotlib (a scratch venv; see tools/plot_tg_mhd_ladder.py).
"""
import argparse
import json
import math
import os
import sys

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
ROWNAME = {"A": "A  free slip", "B": "B  no slip", "C": "C  no slip, no field"}


def setup_of(tag):
    if tag.startswith("hydro_"):
        return "C"
    return "B" if "_noslip_" in tag else "A"


def records(run_dir):
    with open(os.path.join(run_dir, "slices", "index.json")) as f:
        return json.load(f)


def pick(recs, t, plane=None, axis=None, near_d=None):
    """The record at the dump nearest t, by plane name or by distance from the wall."""
    times = sorted({r["t"] for r in recs})
    tt = min(times, key=lambda x: abs(x - t))
    cand = [r for r in recs if r["t"] == tt]
    if plane:
        hit = [r for r in cand if r["plane"] == plane]
        if not hit:
            raise SystemExit("no plane %s at t = %.3f (have %s)" % (plane, tt, ", ".join(r["plane"] for r in cand)))
        return hit[0]
    cand = [r for r in cand if r["axis"] == axis and r["index"] > 0] if near_d is not None else \
           [r for r in cand if r["axis"] == axis]
    if near_d is not None:
        return min(cand, key=lambda r: abs(r["d_over_delta"] - near_d))
    N = cand[0]["N"]
    return min(cand, key=lambda r: abs(r["index"] - 0.25 * (N - 1)))


def load(np, run_dir, rec):
    a = np.fromfile(os.path.join(run_dir, "slices", rec["file"]), dtype="<f4")
    n = rec["N"]
    a = a.reshape(4, n, n)
    return {k: a[i] / rec["eps_mean"] for i, k in enumerate(("visc", "ohm"))}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", help="run directories of A, B and (optionally) C")
    ap.add_argument("--t", type=float, default=2.3, help="the dump nearest this time")
    ap.add_argument("-o", "--out", default="tg_mhd_snapshots")
    ap.add_argument("--fmt", default="png")
    ap.add_argument("--section", default=None, help="yK; default the y-plane nearest pi/4")
    ap.add_argument("--wall", default=None, help="zK; default the z-plane nearest delta/2")
    ap.add_argument("--range", type=float, nargs=2, default=[1e-2, 1e2], metavar=("LO", "HI"))
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    runs = {}
    for d in args.dirs:
        recs = records(d)
        runs[setup_of(recs[0]["tag"])] = (d, recs)
    order = [s for s in ("A", "B", "C") if s in runs]
    if "A" not in runs or "B" not in runs:
        raise SystemExit("need at least the free-slip (A) and no-slip (B) runs")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "text.color": INK})
    norm = LogNorm(vmin=args.range[0], vmax=args.range[1])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(cmap(0.0))
    nr = len(order)
    fig = plt.figure(figsize=(10.5, 2.5 * nr + 0.6))
    fig.patch.set_facecolor(SURFACE)
    # a spacer column keeps the wall-plane group's axis labels off the section group
    gs = fig.add_gridspec(nr, 5, width_ratios=[1, 1, 0.16, 1, 1], left=0.075, right=0.905,
                          bottom=0.075, top=0.93, wspace=0.07, hspace=0.12)
    axs = [[fig.add_subplot(gs[r, c]) for c in (0, 1, 3, 4)] for r in range(nr)]
    meta = None
    for r, s in enumerate(order):
        d, recs = runs[s]
        sec = pick(recs, args.t, plane=args.section, axis="y")
        wal = pick(recs, args.t, plane=args.wall, axis="z", near_d=0.5)
        meta = sec
        S, W = load(np, d, sec), load(np, d, wal)
        delta = sec["delta"] / math.pi                           # in units of pi
        for c, (fld, src, kind) in enumerate((("visc", S, "sec"), ("ohm", S, "sec"),
                                             ("visc", W, "wal"), ("ohm", W, "wal"))):
            ax = axs[r][c]
            ax.set_facecolor(SURFACE)
            if s == "C" and fld == "ohm":
                ax.axis("off")
                ax.text(0.5, 0.5, "no field", ha="center", va="center", color=MUTED,
                        transform=ax.transAxes, fontsize=9)
                continue
            im = ax.imshow(np.clip(src[fld], args.range[0] * 1e-3, None), origin="lower",
                           extent=[0, 1, 0, 1], cmap=cmap, norm=norm, interpolation="nearest")
            if kind == "sec":
                for v in (delta, 1 - delta):
                    ax.axhline(v, color="#9fd7ff", lw=0.6, ls=(0, (3, 2)))
                    ax.axvline(v, color="#9fd7ff", lw=0.6, ls=(0, (3, 2)))
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=7, length=2, colors=INK2)
            if r == nr - 1:
                ax.set_xlabel(r"$x/\pi$", fontsize=8)
            else:
                ax.set_xticklabels([])
            if c in (0, 2):
                ax.set_ylabel(r"$z/\pi$" if kind == "sec" else r"$y/\pi$", fontsize=8)
            else:
                ax.set_yticklabels([])
            if r == 0:
                where = ("section $y=%.2f\\pi$" % (sec["index"] / (sec["N"] - 1)) if kind == "sec"
                         else "wall plane, $d=%.2f\\delta$" % wal["d_over_delta"])
                ax.set_title("%s: %s" % (where, "viscous" if fld == "visc" else "Ohmic"),
                             fontsize=8.5, loc="left")
        bb = axs[r][0].get_position()
        fig.text(0.014, 0.5 * (bb.y0 + bb.y1), ROWNAME[s], rotation=90, ha="center", va="center",
                 fontsize=9)
    cax = fig.add_axes([0.925, 0.12, 0.011, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("local dissipation / box mean", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Re = %g,  t = %.2f,  N = %d" % (meta["Re"], meta["t"], meta["N"]), x=0.08, ha="left",
                 fontsize=9.5, y=0.995)
    for f in args.fmt.split(","):
        fig.savefig("%s.%s" % (args.out, f), dpi=200, facecolor=SURFACE,
                    metadata={"CreationDate": None} if f == "pdf" else None)
    print("wrote " + ", ".join("%s.%s" % (args.out, f) for f in args.fmt.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
