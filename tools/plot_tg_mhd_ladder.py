#!/usr/bin/env python3
"""The confined-MHD ladder's figures: where the dissipation sits, how the wall's
share of it moves with Re, and why reading that share at one instant is fragile.

    plot_tg_mhd_ladder.py RUNS_DIR [RUNS_DIR ...] [-o PREFIX] [--fmt png,pdf] [--re RE]
                          [--window 2 10] [--windows 2-6,2-10,3-10,4-10,5-10]

A RUNS_DIR is what a CSF3 job writes (results/P_tg_mhd/round2/ and ladder/ once
copied back): one <A|B|C>_re<Re>_n<N>/ directory per run with series.dat and
profile.dat, and since 2026-09-30 profile_visc.dat / profile_ohm.dat. Several may
be given, and at a (setup, Re, N) found in more than one the FIRST wins -- so list
round2/, which has the split, before ladder/, which has C below Re = 2000. Peaks,
f_w and the integrated share are found exactly as tools/tg_mhd_ladder.py finds
them (this imports it), and each (setup, Re) is drawn from its finest grid. Three
figures, PREFIX_<name>.<fmt> for each --fmt:

  ladder   (a) the dissipation density against the mean -- each wall-distance
           layer's share of ALL the dissipation over --window, int s_k eps dt /
           int eps dt, over its share of the volume -- against d / delta for
           A (free slip + conducting), B (no slip + conducting) and C (no slip, no
           field) at one Re, by default the largest at which all three have a
           turbulent peak; (b) that density AT the wall against Re, for A and B;
           (c) the plan's pre-registered observable, fw1 at the peak, against Re,
           with B - A written at each rung. Until 2026-10-02 (a) and (b) were also
           read at each run's peak; the peaks figure is why they no longer are --
           B's wall density at Re = 2000 came from t = 4.53, the other rungs' from
           t = 1.9-2.3.
  peaks    eps(t) and fw1(t) for A and B at every rung where both have a turbulent
           peak, each peak marked. This is the figure that shows the peak SWITCHING:
           A's moves from its early maximum to its later one between Re = 500 and
           1000, B's between 1000 and 2000, and fw1 moves fast through both -- so
           the ladder's (c) compares the two setups at different times.
  windows  (a) B - A against Re, read at the peak (pre-registered) and integrated
           over each --windows interval, F = int fw1 eps dt / int eps dt (post hoc):
           flat over the early window, falling over the late ones; (b) F1 of A and B
           over --window, divided into its viscous and Ohmic parts.

The integrated quantities are printed too. Their grid bands -- the two finest
grids' difference -- are at most ~0.001 and are not drawn; tools/tg_mhd_ladder.py
prints them. A run without a turbulent peak inside it has no f_w and is left out
rather than read at another time: B at Re = 125 and C below Re = 1000 have none,
and C at Re = 2000 has not reached its own by t = 10. The integrated panels keep
the same rule, so that no line joins a laminar rung to a turbulent one. The peaks
and windows figures use only the rungs where A and B both have a peak.

A PDF is written without a creation date, so a figure regenerated from the same
data is byte-identical and a tracked one shows in a diff only when it changes.

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
WIN = "#6f4fbf"      # the integrated windows' hue, kept apart from A, B and C


def ramp(hexcol, n):
    """n shades of one colour, light to dark -- toward white, the colour, toward black.
    The lightest are faint on the surface, so whatever uses them is also labelled."""
    rgb = [int(hexcol[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    if n == 1:
        return [rgb]
    out = []
    for k in range(n):
        x = -0.55 + 0.9 * k / (n - 1)
        out.append([v + (1 - v) * -x for v in rgb] if x < 0 else [v * (1 - x) for v in rgb])
    return out


def profile_integrated(run_dir, run, t0, t1):
    """(d/delta list, density list) over [t0, t1]: each layer's share of ALL the
    dissipation in the window, int s_k eps dt / int eps dt -- the per-layer form of
    tg_mhd_ladder.integrated, with the same edge interpolation -- over its share
    of the volume."""
    lines = open(os.path.join(run_dir, "profile.dat")).read().splitlines()
    vol = [float(x) for x in next(l for l in lines if l.startswith("# vol")).split()[2:]]
    rows = [[float(x) for x in l.split()] for l in lines if l and not l.startswith("#")]
    ts, eps = T.col(run, "t"), T.col(run, "eps")
    if len(rows) != len(ts) or any(abs(r[0] - t) > 1e-6 for r, t in zip(rows, ts)):
        raise ValueError("%s: profile.dat is not sampled at the series' probes" % run_dir)
    if ts[0] > t0 + 1e-2 or ts[-1] < t1 - 1e-2:
        raise ValueError("%s covers t = %.2f..%.2f only" % (run_dir, ts[0], ts[-1]))
    q = [(t, [e] + [s * e for s in r[1:]]) for t, e, r in zip(ts, eps, rows)]
    tot = T.trapz_window(q, max(t0, ts[0]), min(t1, ts[-1]))
    r = (run["N"] - 1) / (math.pi * math.sqrt(run["Re"]))      # delta / h
    keep = [k for k in range(len(vol)) if vol[k] > 0]
    return [k / r for k in keep], [tot[1 + k] / tot[0] / vol[k] for k in keep]


def load_all(dirs):
    """{(setup, Re): {N: run}} for A, B and C; the first directory wins a repeat."""
    out = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*", "series.dat"))):
            run = T.load(p)
            if run is not None and run["setup"] in COL:
                out.setdefault((run["setup"], run["Re"]), {}).setdefault(run["N"], run)
    return out


def collect(runs):
    """{(setup, Re): (N, run_dir, run, (t_peak, eps_peak), [fw1, fw2, fw4])} from the
    finest grid of each, for those with a turbulent peak."""
    best = {}
    for key, by_n in runs.items():
        N = max(by_n)
        run = by_n[N]
        pk = T.turbulent_peak(run)
        if isinstance(pk, tuple):
            best[key] = (N, os.path.dirname(run["path"]), run, pk, T.fw_at(run, pk[0]))
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


def re_axis(ax, matplotlib, res):
    ax.set_xscale("log")
    ax.set_xticks(res)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(min(res) / 1.35, max(res) * 1.35)
    ax.set_xlabel("Re")


def save(fig, base, fmts, plt):
    for f in fmts:
        fig.savefig("%s.%s" % (base, f), dpi=200, facecolor=SURFACE,
                    metadata={"CreationDate": None} if f == "pdf" else None)
    plt.close(fig)


def fig_ladder(best, re_a, window, base, fmts, plt, matplotlib):
    t0, t1 = window
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.patch.set_facecolor(SURFACE)

    # (a) where the dissipation sits, at one Re, over the window
    ax = axs[0]
    style(ax)
    ax.axvspan(0, 1, color="#f0efec", zorder=0, linewidth=0)
    ax.axhline(1.0, color="#b8b7b0", linewidth=0.8, zorder=1)
    lo, hi = 1.0, 1.0
    for s in ("C", "B", "A"):
        if (s, re_a) not in best:
            continue
        N, d, run, pk, fw = best[(s, re_a)]
        x, rho = profile_integrated(d, run, t0, t1)
        pts = [(a, b) for a, b in zip(x, rho) if a <= 10.0]
        lo, hi = min(lo, min(b for _, b in pts)), max(hi, max(b for _, b in pts))
        ax.plot([a for a, _ in pts], [b for _, b in pts], color=COL[s], linewidth=1.8,
                solid_joinstyle="round", solid_capstyle="round", label=NAME[s], zorder=3)
        ax.plot([pts[0][0]], [pts[0][1]], "o", color=COL[s], markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.annotate("%s  %.2f×" % (s, pts[0][1]), (pts[0][0], pts[0][1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(lo / 1.6, hi * 1.8)
    ax.set_xlim(-0.15, 10)
    ax.text(0.5, lo / 1.55, "fw1 band", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.text(9.9, 1.06, "mean", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xlabel(r"distance from the wall  $d/\delta$,  $\delta = Re^{-1/2}$")
    ax.set_ylabel("dissipation density / mean")
    ax.set_title("(a)  where it dissipates, Re = %g, t = %g–%g" % (re_a, t0, t1), loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))

    res = sorted({Re for s, Re in best if s in ("A", "B")})
    pairs = sorted(Re for s, Re in best if s == "A" and ("B", Re) in best)

    # (b) the density at the wall against Re, over the window
    ax = axs[1]
    style(ax)
    top = 0.0
    print("dissipation density at the wall / mean, t = %g-%g:" % (t0, t1))
    for s in ("A", "B"):
        xs, ys = [], []
        for Re in sorted(Re for t, Re in best if t == s):
            N, d, run, pk, fw = best[(s, Re)]
            _, rho = profile_integrated(d, run, t0, t1)
            xs.append(Re); ys.append(rho[0])
        print("  %s  " % s + "  ".join("Re %g: %.3f" % (a, b) for a, b in zip(xs, ys)))
        top = max(top, max(ys))
        ax.plot(xs, ys, color=COL[s], linewidth=1.8, marker="o", markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAME[s], zorder=3)
        ax.annotate("%.2f×" % ys[-1], (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.axhline(1.0, color="#b8b7b0", linewidth=0.8, zorder=1)
    re_axis(ax, matplotlib, res)
    ax.set_ylim(0, 1.2 * top)
    ax.set_ylabel("dissipation density at the wall / mean")
    ax.set_title("(b)  the wall layer, t = %g–%g" % (t0, t1), loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK2)

    # (c) the pre-registered observable: the share within delta, at the peak
    ax = axs[2]
    style(ax)
    top = 0.0
    for s in ("A", "B"):
        pts = sorted((Re, best[(s, Re)][4][0]) for t, Re in best if t == s)
        top = max(top, max(b for _, b in pts))
        ax.plot([a for a, _ in pts], [b for _, b in pts], color=COL[s], linewidth=1.8, marker="o",
                markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAME[s], zorder=3)
    for Re in pairs:
        a, b = best[("A", Re)][4][0], best[("B", Re)][4][0]
        ax.plot([Re, Re], [a, b], color="#c3c2b7", linewidth=0.8, zorder=2)
        ax.annotate("+%.3f" % (b - a), (Re, 0.5 * (a + b)), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK2)
    re_axis(ax, matplotlib, res)
    ax.set_ylim(0, 1.25 * top)
    ax.set_ylabel(r"fw1: share of dissipation within $\delta$")
    ax.set_title("(c)  pre-registered: fw1 at the peak  (B − A)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK2)

    fig.tight_layout(w_pad=2.0)
    save(fig, base, fmts, plt)


def fig_peaks(best, rungs, base, fmts, plt):
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    lim = {"A": 0.11, "B": 0.35}           # fw1; B starts at 0.95-0.98 and is clipped
    for c, s in enumerate(("A", "B")):
        shades = ramp(COL[s], len(rungs))
        for i, Re in enumerate(rungs):
            N, d, run, pk, fw = best[(s, Re)]
            t = T.col(run, "t")
            rows = ((T.col(run, "eps"), pk[1]), ([f[0] for _, f in T.fw_rows(run)], fw[0]))
            for r, (ys, at) in enumerate(rows):
                ax = axs[r][c]
                ax.plot(t, ys, color=shades[i], linewidth=1.6, zorder=3,
                        label="Re = %g   peak at t = %.2f" % (Re, pk[0]))
                # No surface-coloured halo here: on a continuous curve it cuts gaps in
                # this line and in any other passing the dot, which read as dashes.
                ax.plot([pk[0]], [at], "o", color=shades[i], markersize=6.5, zorder=4,
                        markeredgecolor=[0.65 * v for v in shades[i]], markeredgewidth=0.8)
        ax = axs[0][c]
        style(ax)
        ax.set_ylim(0, 0.058)
        ax.set_ylabel(r"dissipation rate  $\varepsilon$")
        ax.set_title("(%s)  %s: dissipation, each turbulent peak marked" % ("ab"[c], NAME[s]),
                     loc="left", fontsize=9.5)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK2)
        ax = axs[1][c]
        style(ax)
        ax.set_ylim(0, lim[s])
        ax.set_xlim(0, 10)
        ax.set_xlabel("t")
        ax.set_ylabel(r"fw1: share within $\delta$ of the walls")
        ax.set_title("(%s)  %s: fw1, read at each peak" % ("cd"[c], NAME[s]), loc="left", fontsize=9.5)
    axs[0][1].text(0.45, 0.0565, r"← impulsive start, $\varepsilon(0)$ = 0.09–0.30",
                   va="top", fontsize=7.5, color=MUTED)
    axs[1][1].text(1.6, 0.343, "← start-up wall layers: fw1(0) = 0.95–0.98, off scale",
                   va="top", fontsize=7.5, color=MUTED)
    fig.tight_layout(h_pad=1.2, w_pad=2.0)
    save(fig, base, fmts, plt)


def integrated_fine(runs, key, t0, t1):
    """([F1, F2, F4], [visc, ohm] or None, band on F1 or None) on the finest grid of key."""
    by_n = runs[key]
    Ns = sorted(by_n)[-2:]
    got = [T.integrated(by_n[n], t0, t1) for n in Ns]
    if isinstance(got[-1], str):
        return None
    band = abs(got[-1][0][0] - got[0][0][0]) if len(Ns) == 2 and not isinstance(got[0], str) else None
    return got[-1][0], got[-1][1], band


def fig_windows(runs, best, rungs, windows, split, base, fmts, plt, matplotlib):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.1))
    fig.patch.set_facecolor(SURFACE)

    # (a) B - A against Re, at the peak and over each window
    ax = axs[0]
    style(ax)
    peak = [best[("B", Re)][4][0] - best[("A", Re)][4][0] for Re in rungs]
    ax.plot(rungs, peak, color=INK, linewidth=1.6, linestyle=(0, (4, 2)), marker="o", markersize=6,
            markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.4, zorder=4,
            label="at the peak  (pre-registered)")
    print("B - A in F1     Re: " + " ".join("%8g" % Re for Re in rungs) + "   largest band")
    print("  %-14s" % "at the peak" + "    " + " ".join("%+8.4f" % v for v in peak))
    for (t0, t1), shade in zip(windows, ramp(WIN, len(windows))):
        ys, worst = [], 0.0
        for Re in rungs:
            a, b = integrated_fine(runs, ("A", Re), t0, t1), integrated_fine(runs, ("B", Re), t0, t1)
            ys.append(b[0][0] - a[0][0] if a and b else float("nan"))
            if a and b and a[2] is not None and b[2] is not None:
                worst = max(worst, a[2] + b[2])
        ax.plot(rungs, ys, color=shade, linewidth=1.8, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                label="integrated over t = %g–%g" % (t0, t1))
        print("  %-14s" % ("t = %g-%g" % (t0, t1)) + "    " + " ".join("%+8.4f" % v for v in ys)
              + "   %.4f" % worst)
    re_axis(ax, matplotlib, rungs)
    ax.set_ylim(0, 0.12)
    ax.set_ylabel("B − A in the share within δ")
    ax.set_title("(a)  the trend in Re depends on when it is read", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK2)

    # (b) the split of F1 over one window
    ax = axs[1]
    style(ax)
    t0, t1 = split
    print("F1 over t = %g-%g, viscous / Ohmic:" % (t0, t1))
    for s in ("B", "A"):
        parts = [integrated_fine(runs, (s, Re), t0, t1) for Re in rungs]
        if not all(p and p[1] for p in parts):
            print("  %s: no split profiles on every rung -- left out" % s)
            continue
        for k, (what, ls) in enumerate((("viscous", "-"), ("Ohmic", (0, (4, 2))))):
            ys = [p[1][k] for p in parts]
            ax.plot(rungs, ys, color=COL[s], linewidth=1.8, linestyle=ls, marker="o", markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label="%s  %s" % (s, what))
            print("  %s %-8s" % (s, what) + " ".join("%8.4f" % v for v in ys))
    re_axis(ax, matplotlib, rungs)
    ax.set_ylim(0, None)
    ax.set_ylabel("share of all the dissipation, within δ")
    ax.set_title("(b)  the excess is viscous  (integrated, t = %g–%g)" % (t0, t1), loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK2, ncol=2)

    fig.tight_layout(w_pad=2.0)
    save(fig, base, fmts, plt)


def parse_window(s):
    a, b = (float(x) for x in s.split("-"))
    if not a < b:
        raise argparse.ArgumentTypeError("a window is T0-T1 with T0 < T1, not %r" % s)
    return a, b


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_dirs", nargs="+")
    ap.add_argument("-o", "--out", default="tg_mhd", help="prefix of the files (default tg_mhd)")
    ap.add_argument("--fmt", default="png", help="comma-separated formats, png and/or pdf (default png)")
    ap.add_argument("--re", type=float, default=None,
                    help="Re of the ladder's panel (a); default the largest with A, B and C all peaked")
    ap.add_argument("--window", type=float, nargs=2, default=[2.0, 10.0], metavar=("T0", "T1"),
                    help="the integration window of the ladder's (a), (b) and the split (default 2 10)")
    ap.add_argument("--windows", default="2-6,2-10,3-10,4-10,5-10",
                    type=lambda s: [parse_window(w) for w in s.split(",")],
                    help="the windows compared in the windows figure's (a)")
    args = ap.parse_args(argv)
    fmts = args.fmt.split(",")
    if not fmts or any(f not in ("png", "pdf") for f in fmts):
        ap.error("--fmt takes png, pdf or png,pdf")
    if not args.window[0] < args.window[1]:
        ap.error("--window needs T0 < T1")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_all(args.runs_dirs)
    best = collect(runs)
    if not best:
        raise SystemExit("no run with a turbulent peak under %s" % ", ".join(args.runs_dirs))
    full = [Re for _, Re in best if all((s, Re) in best for s in COL)]
    re_a = args.re if args.re is not None else max(full or [Re for _, Re in best])
    rungs = sorted(Re for s, Re in best if s == "A" and ("B", Re) in best)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "text.color": INK})
    bases = ["%s_%s" % (args.out, n) for n in ("ladder", "peaks", "windows")]
    fig_ladder(best, re_a, tuple(args.window), bases[0], fmts, plt, matplotlib)
    fig_peaks(best, rungs, bases[1], fmts, plt)
    fig_windows(runs, best, rungs, args.windows, tuple(args.window), bases[2], fmts, plt, matplotlib)
    print("wrote " + ", ".join("%s.%s" % (b, f) for b in bases for f in fmts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
