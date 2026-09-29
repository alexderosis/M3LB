#!/usr/bin/env python3
"""The confined-MHD ladder as one figure: where the dissipation sits, and how the
wall's share of it moves with Re.

    plot_tg_mhd_ladder.py RUNS_DIR [-o ladder.png] [--re 1000]

RUNS_DIR is what GPU/csf3/tg_mhd_ladder.sub writes (runs/tg_mhd_ladder/ on CSF3,
results/P_tg_mhd/ladder/ once copied back): one <A|B|C>_re<Re>_n<N>/ directory
per run with series.dat and profile.dat. Every quantity is read AT EACH RUN'S
TURBULENT PEAK, found and interpolated exactly as tools/tg_mhd_ladder.py does
(this imports it), and each (setup, Re) is drawn from its finest grid.

  (a) the dissipation density against the mean, (layer share of eps) / (layer
      share of volume), against wall distance d / delta at one Re (default the
      top rung), for A (free slip + conducting), B (no slip + conducting) and C
      (no slip, no field). Log scale: C's wall layer runs at ~14x the mean and
      A's at a third of it.
  (b) that density AT the wall, d = 0, against Re, for A and B.
  (c) the plan's observable, fw1 -- the share of eps within delta = Re^-1/2 of
      the walls -- against Re for A and B, with B - A written at each rung.
      The two grids of a rung agree to ~1 %, too small to draw as a bar.

Runs with no turbulent peak (B at Re = 125, C below Re = 1000 in the first
ladders) have no f_w and are left out rather than read at another time.

Needs numpy and matplotlib; the system python on the development Mac has
neither -- a scratch venv does (python3 -m venv v && v/bin/pip install numpy
matplotlib), and nothing in the tree depends on it.
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tg_mhd_ladder as T  # noqa: E402

# Categorical slots 1-3 of the dataviz reference palette, validated all-pairs in
# light mode (worst CVD dE 9.2, normal-vision dE 24.0); C's aqua is under 3:1 on
# the surface, so every curve is also labelled directly.
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
NAME = {"A": "A  free slip, conducting", "B": "B  no slip, conducting", "C": "C  no slip, no field"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def profile_at(run_dir, t):
    """(d/delta list, density list) at time t from a run's profile.dat."""
    lines = open(os.path.join(run_dir, "profile.dat")).read().splitlines()
    head = lines[0]
    N = int(head.split("N=")[1].split()[0])
    Re = float(head.split("Re=")[1].split()[0])
    vol = [float(x) for x in next(l for l in lines if l.startswith("# vol")).split()[2:]]
    rows = [[float(x) for x in l.split()] for l in lines if l and not l.startswith("#")]
    for a, b in zip(rows, rows[1:]):
        if a[0] <= t <= b[0]:
            w = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
            share = [(1 - w) * x + w * y for x, y in zip(a[1:], b[1:])]
            break
    else:
        raise ValueError("t = %g outside %s" % (t, run_dir))
    r = (N - 1) / (math.pi * math.sqrt(Re))              # delta / h
    keep = [k for k in range(len(vol)) if vol[k] > 0]
    return [k / r for k in keep], [share[k] / vol[k] for k in keep]


def collect(runs_dir):
    """{(setup, Re): (N, run_dir, run, (t_peak, eps_peak), [fw1, fw2, fw4])}, finest grid."""
    best = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "series.dat"))):
        run = T.load(p)
        if run is None or run["setup"] not in COL:
            continue
        pk = T.turbulent_peak(run)
        if not isinstance(pk, tuple):
            continue
        key = (run["setup"], run["Re"])
        if key not in best or run["N"] > best[key][0]:
            best[key] = (run["N"], os.path.dirname(p), run, pk, T.fw_at(run, pk[0]))
    return best


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6, length=3)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_dir")
    ap.add_argument("-o", "--out", default="tg_mhd_ladder.png")
    ap.add_argument("--re", type=float, default=None, help="Re of panel (a); default the largest")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = collect(args.runs_dir)
    if not best:
        raise SystemExit("no run with a turbulent peak under %s" % args.runs_dir)
    re_a = args.re if args.re is not None else max(Re for _, Re in best)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "text.color": INK})
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.patch.set_facecolor(SURFACE)

    # (a) where the dissipation sits, at one Re
    ax = axs[0]
    style(ax)
    ax.axvspan(0, 1, color="#f0efec", zorder=0, linewidth=0)
    ax.text(0.5, 0.21, "fw1 band", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.axhline(1.0, color="#b8b7b0", linewidth=0.8, zorder=1)
    ax.text(9.9, 1.06, "mean", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    for s in ("C", "B", "A"):
        if (s, re_a) not in best:
            continue
        N, d, run, pk, fw = best[(s, re_a)]
        x, rho = profile_at(d, pk[0])
        pts = [(a, b) for a, b in zip(x, rho) if a <= 10.0]
        ax.plot([a for a, _ in pts], [b for _, b in pts], color=COL[s], linewidth=1.8,
                solid_joinstyle="round", solid_capstyle="round", label=NAME[s], zorder=3)
        ax.plot([pts[0][0]], [pts[0][1]], "o", color=COL[s], markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.annotate("%s  %.2f×" % (s, pts[0][1]), (pts[0][0], pts[0][1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 25)
    ax.set_xlim(-0.15, 10)
    ax.set_xlabel(r"distance from the wall  $d/\delta$,  $\delta = Re^{-1/2}$")
    ax.set_ylabel("dissipation density / mean")
    ax.set_title("(a)  where it dissipates, Re = %g, at the peak" % re_a, loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))

    res = sorted({Re for s, Re in best if s in ("A", "B") and ("A", Re) in best and ("B", Re) in best})

    # (b) the density at the wall against Re
    ax = axs[1]
    style(ax)
    for s in ("A", "B"):
        xs, ys = [], []
        for Re in sorted(Re for t, Re in best if t == s):
            N, d, run, pk, fw = best[(s, Re)]
            _, rho = profile_at(d, pk[0])
            xs.append(Re); ys.append(rho[0])
        ax.plot(xs, ys, color=COL[s], linewidth=1.8, marker="o", markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAME[s], zorder=3)
        ax.annotate("%.2f×" % ys[-1], (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.axhline(1.0, color="#b8b7b0", linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([125, 250, 500, 1000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(100, 1500)
    ax.set_ylim(0, 3.0)
    ax.set_xlabel("Re")
    ax.set_ylabel("dissipation density at the wall / mean")
    ax.set_title("(b)  the wall layer intensifies", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK2)

    # (c) the observable: the share within delta
    ax = axs[2]
    style(ax)
    for s in ("A", "B"):
        pts = sorted((Re, best[(s, Re)][4][0]) for t, Re in best if t == s)
        ax.plot([a for a, _ in pts], [b for _, b in pts], color=COL[s], linewidth=1.8, marker="o",
                markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAME[s], zorder=3)
    for Re in res:
        a, b = best[("A", Re)][4][0], best[("B", Re)][4][0]
        ax.plot([Re, Re], [a, b], color="#c3c2b7", linewidth=0.8, zorder=2)
        ax.annotate("+%.3f" % (b - a), (Re, 0.5 * (a + b)), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([125, 250, 500, 1000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(100, 1500)
    ax.set_ylim(0, 0.18)
    ax.set_xlabel("Re")
    ax.set_ylabel(r"fw1: share of dissipation within $\delta$")
    ax.set_title("(c)  but its share barely moves  (B − A)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK2)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
