#!/usr/bin/env python3
"""Plot validation/vortex_disruption series, up to four runs on one figure.

    python3 tools/plot_vortex_disruption.py -o fig.png RUN_DIR [RUN_DIR ...]
    python3 tools/plot_vortex_disruption.py -o fig.png --labels "vector" "flux" A B

Eight panels against time (in R0/eps): the survival of the vortex (projection
on the initial pattern over its exact viscous decay; 1 = untouched), its angular
momentum, the kinetic energy, the peak field, the mean field energy, the peak
current, the field microscale, and the odd-even share of the current, which says
whether the current sheets are still on the grid.

Each run is labelled by its directory name unless --labels is given. Colours are
the first four slots of the tree's categorical palette, in the order the runs
are given, and each run also has its own dash pattern, so no line is told apart
by colour alone. A run that went non-finite ends at its last finite probe.
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DASHES = ["-", "--", "-.", ":"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def read(run):
    cols, rows, head = None, [], ""
    with open(os.path.join(run, "series.dat")) as f:
        for line in f:
            if line.startswith("#"):
                if cols is None and line.split()[1:2] == ["t"]:
                    cols = line[1:].split()
                elif not head:
                    head = line[1:].strip()
                continue
            v = line.split()
            if len(v) != len(cols):
                continue
            row = [float(x) for x in v]
            # The driver stops at a non-finite field or Ma > 0.5 and prints that
            # row: it is not a measurement, and the series ends before it.
            if not all(math.isfinite(x) for x in row) or row[cols.index("Ma")] > 0.5:
                break
            rows.append(row)
    return {"data": {c: [r[i] for r in rows] for i, c in enumerate(cols)}, "head": head}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    if len(a.runs) > len(COLOURS):
        sys.exit("at most %d runs per figure: use small multiples for more" % len(COLOURS))
    if a.labels and len(a.labels) != len(a.runs):
        sys.exit("--labels needs one label per run")
    labels = a.labels or [os.path.basename(os.path.normpath(r)) for r in a.runs]
    runs = [read(r) for r in a.runs]

    panels = [
        ("surv", "survival of the vortex", "linear"),
        ("ang", "angular momentum / initial", "linear"),
        ("EK", "kinetic energy / initial", "linear"),
        ("bmax", "peak field / B0", "linear"),
        ("B2", "mean field energy / initial", "linear"),
        ("jmax", "peak current (B0 / R0)", "log"),
        ("Lb", "field microscale / R0", "log"),
        ("osc", "odd-even share of the current", "log"),
    ]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4), sharex=True)
    for ax, (key, name, scale) in zip(axes.flat, panels):
        for i, r in enumerate(runs):
            t, y = r["data"]["t"], r["data"][key]
            if scale == "log":
                # 0 stands for "undefined" there, and a uniform field's current
                # is round-off: neither belongs on a log axis.
                floor = 1e-9 * max(y) if y else 0.0
                pts = [(p, q) for p, q in zip(t, y) if q > floor]
                t, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(t, y, DASHES[i], color=COLOURS[i], lw=1.6, label=labels[i])
        ax.set_yscale(scale)
        ax.set_title(name, loc="left", fontsize=9, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("t  (R0 / eps)")
    # One legend for the figure, in the title row, so it never sits on a curve.
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper right", ncol=len(runs), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.995, 0.925), handlelength=3.2)
    fig.suptitle(a.title or runs[0]["head"][:110], x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(a.out, dpi=130, facecolor="#fcfcfb")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
