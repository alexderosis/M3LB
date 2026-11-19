#!/usr/bin/env python3
"""Read a map of validation/vortex_disruption runs in Rm and M, and ask which
combination of the two the disruption of the vortex depends on.

    python3 tools/vortex_disruption_map.py MAP_DIR [--obs surv|dow|ens]
                                           [--by Rm|Re] [-o fig.png] [--reading c3]

MAP_DIR holds one directory per run, each with a series.dat. Re, Rm, M and R0
are read from the series headers, never from the directory names. --by names
the number that varies across the directory: Rm (the default: a map at Pm = 1,
or a slice at fixed Re) or Re (a slice at fixed Rm, which asks whether the
viscosity matters at all).

THE OBSERVABLE D, chosen by --obs, is 0 for a vortex the field has not touched:

    surv   D = 1 - surv   the projection of the velocity on the initial vortex,
                          over its exact viscous decay: a measure of the MEAN
                          flow, which is what the M^2 Rm^(2/3) argument is about
    dow    D = dow        Mak, Griffiths & Hughes's Delta from the Okubo-Weiss
                          field, against the exact vortex: a measure built from
                          velocity GRADIENTS, which is what M^2 Rm is about
    ens    D = 1 - ens    the enstrophy lost against the exact vortex's

WHEN IT IS READ IS A CHOICE, AND THE SCRIPT DOES NOT MAKE IT FOR YOU.  A decaying
run has no steady state, the evolution slows as Rm rises, and this tree has
already been caught once reading a trend off one instant (CLAUDE.md, "a quantity
read at the peak of a transient"). So every table is printed for several
readings side by side:

    t = 50              a fixed time
    t = c Rm^(1/3)      c = 3, 6, 12: a fixed phase of Weiss's expulsion
    largest D           over the whole run

and --reading only picks which one the figure draws.

THE QUESTION, ASKED TWICE.  First without a model: along a row of the ladder,
at fixed M^2 Rm, does D move with Rm? D ~ Rm^q with q = 0 means D depends on
M^2 Rm alone; q = -1/3 means on M^2 Rm^(2/3), while D is small. Then with one:
the field M_c(Rm) at which D crosses a level (interpolated linearly in log M
between the two runs that bracket it) and p in M_c ~ Rm^p, against p = -1/2 and
p = -1/3. Beside each p is printed the most it could move if every M_c sat
anywhere between its two bracketing runs: a bound set by the spacing of the
ladder, not a statistical error. It says nothing about the scatter of a chaotic
run, which only a second grid or a second realisation can show. A run that
stopped short of t = 12 Rm^(1/3) has no "largest D".
"""
import argparse
import math
import os
import sys

READINGS = ["t50", "c3", "c6", "c12", "max"]
NAMES = {"t50": "t = 50", "c3": "t = 3 Rm^(1/3)", "c6": "t = 6 Rm^(1/3)",
         "c12": "t = 12 Rm^(1/3)", "max": "largest D in the run"}
LEVELS = [0.1, 0.25, 0.5, 0.7, 0.8, 0.9]
OBS = {"surv": ("surv", lambda v: 1.0 - v, "D = 1 - survival of the vortex"),
       "dow": ("dow", lambda v: v, "D = Okubo-Weiss Delta (after Mak et al.)"),
       "ens": ("ens", lambda v: 1.0 - v, "D = 1 - enstrophy / exact vortex's")}


def read(run):
    path = os.path.join(run, "series.dat")
    if not os.path.isfile(path):
        return None
    cols, rows, head = None, [], ""
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if cols is None and line.split()[1:2] == ["t"]:
                    cols = line[1:].split()
                elif not head:
                    head = line[1:].split()
                continue
            v = line.split()
            if cols is None or len(v) != len(cols):
                continue
            row = [float(x) for x in v]
            if not all(math.isfinite(x) for x in row) or row[cols.index("Ma")] > 0.5:
                break                       # the driver's last, non-finite row
            rows.append(row)
    if not rows:
        return None
    get = lambda k: float(head[head.index(k) + 1])
    d = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    return {"Rm": get("Rm"), "Re": get("Re"), "M": get("M"), "R0": get("R0"), "d": d,
            "name": os.path.basename(os.path.normpath(run))}


def at(run, t, col):
    """Column `col` at time t, linearly interpolated; None if not reached."""
    T, S = run["d"]["t"], run["d"][col]
    # A run asked for t = 12 Rm^(1/3) to one decimal can stop a few hundredths
    # short of it: within half a probe of the end, the last probe is the reading.
    if t > T[-1] + 0.3:
        return None
    t = min(t, T[-1])
    for i in range(1, len(T)):
        if T[i] >= t:
            w = (t - T[i - 1]) / (T[i] - T[i - 1])
            return S[i - 1] + w * (S[i] - S[i - 1])
    return S[-1]


def ended_early(run):
    """A run that stopped short of t = 12 Rm^(1/3): it went non-finite."""
    return run["d"]["t"][-1] < 12 * run["Rm"] ** (1.0 / 3.0) - 1.0


def readings(run, col, fn):
    c = run["Rm"] ** (1.0 / 3.0)
    out = {}
    for key, t in (("t50", 50.0), ("c3", 3 * c), ("c6", 6 * c), ("c12", 12 * c)):
        v = at(run, t, col)
        out[key] = None if v is None else fn(v)
    # The largest value of a run that was cut short is not its largest value.
    out["max"] = None if ended_early(run) else max(fn(v) for v in run["d"][col])
    return out


def crossing(pts, level):
    """(M_c, h): the M at which D first crosses `level`, from (M, D) pairs, and
    half the log-spacing of the two runs that bracket it -- how far M_c could be
    from the interpolated value if D were anything monotonic between them.
    None if the level is not bracketed."""
    pts = sorted(p for p in pts if p[1] is not None)
    for (m0, d0), (m1, d1) in zip(pts, pts[1:]):
        if (d0 - level) * (d1 - level) <= 0 and d0 != d1:
            w = (level - d0) / (d1 - d0)
            span = math.log(m1) - math.log(m0)
            return math.exp(math.log(m0) + w * span), 0.5 * span
    return None


def slope_bound(xh):
    """The most a fitted exponent can move if each point moves by its own h."""
    n = len(xh)
    if n < 2:
        return None
    lx = [math.log(x) for x, _ in xh]
    mx = sum(lx) / n
    den = sum((a - mx) ** 2 for a in lx)
    return sum(abs(a - mx) * h for a, (_, h) in zip(lx, xh)) / den if den > 0 else None


def slope(xy):
    n = len(xy)
    if n < 2:
        return None
    lx = [math.log(x) for x, _ in xy]
    ly = [math.log(y) for _, y in xy]
    mx, my = sum(lx) / n, sum(ly) / n
    den = sum((a - mx) ** 2 for a in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den if den > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("map_dir")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--obs", default="surv", choices=sorted(OBS))
    ap.add_argument("--reading", default="c3", choices=READINGS)
    ap.add_argument("--by", default="Rm", choices=["Rm", "Re"],
                    help="the number that varies across the directory (default Rm)")
    a = ap.parse_args()
    col, fn, ylabel = OBS[a.obs]
    by = a.by

    runs = [read(os.path.join(a.map_dir, n)) for n in sorted(os.listdir(a.map_dir))]
    runs = [r for r in runs if r]
    if not runs:
        sys.exit("no series.dat under " + a.map_dir)
    if any(col not in r["d"] for r in runs):
        sys.exit("these series have no '%s' column: they predate it. Rerun the map." % col)
    vals = sorted({r[by] for r in runs})
    other = "Re" if by == "Rm" else "Rm"
    oth = sorted({r[other] for r in runs})
    pms = sorted({round(r["Rm"] / r["Re"], 6) for r in runs})
    print("map: %d runs, %s = %s, R0 = %s cells" %
          (len(runs), by, ", ".join("%g" % x for x in vals),
           ", ".join(sorted({"%g" % r["R0"] for r in runs}))))
    print("%s = %s;  Pm = Rm/Re = %s" % (other, ", ".join("%g" % x for x in oth),
                                        ", ".join("%g" % x for x in pms)))
    if by == "Re" and len(oth) > 1:
        sys.exit("--by Re wants one Rm in the directory: the field's clock and M^2 Rm are Rm's")
    print("observable: %s" % ylabel)

    # ---- every run, every reading ------------------------------------------
    print("\n'-' = the run did not reach that time (see t_end).")
    print("%-6s %-6s %-9s %-8s %-9s | %-7s %-7s %-7s %-7s %-7s | %-6s %-6s %-7s %-8s %-7s" %
          ("Re", "Rm", "M", "M2Rm", "M2Rm2/3", "t=50", "c=3", "c=6", "c=12", "max",
           "bmax", "t_pk", "t_end", "divb/j", "osc"))
    table = {}                                   # (x, M) -> (readings, Rm)
    for r in sorted(runs, key=lambda r: (r[by], r["M"])):
        D = readings(r, col, fn)
        table[(r[by], r["M"])] = (D, r["Rm"])
        d = r["d"]
        ib = max(range(len(d["bmax"])), key=lambda i: d["bmax"][i])
        # The first probes of a uniform field have a current that is round-off;
        # a ratio taken there is noise, not a divergence error.
        jtop = max(d["jmax"])
        dj = max((x / y for x, y in zip(d["divb"], d["jmax"]) if y > 1e-6 * jtop), default=0.0)
        fmt = lambda v: "%-7s" % "-" if v is None else "%-7.4f" % v
        print("%-6g %-6g %-9.5f %-8.4g %-9.4g | %s %s %s %s %s | %-6.2f %-6.1f %-7.1f %-8.1e "
              "%-7.4f%s" %
              (r["Re"], r["Rm"], r["M"], r["M"] ** 2 * r["Rm"],
               r["M"] ** 2 * r["Rm"] ** (2.0 / 3.0),
               fmt(D["t50"]), fmt(D["c3"]), fmt(D["c6"]), fmt(D["c12"]), fmt(D["max"]),
               d["bmax"][ib], d["t"][ib], d["t"][-1], dj, max(d["osc"]),
               "   ENDED EARLY" if ended_early(r) else ""))

    # ---- at fixed M^2 Rm, does D move with the varied number? ---------------
    if by == "Rm":
        print("\nAlong each row of the ladder (fixed M^2 Rm): D at each Rm, and q in D ~ Rm^q."
              "\nq = 0 if D depends on M^2 Rm alone; q = -0.333 if on M^2 Rm^(2/3) "
              "(while D is small).")
    else:
        print("\nAlong each row of the ladder (fixed M^2 Rm, and here fixed M): D at each Re, "
              "and q in D ~ Re^q.\nq = 0 if the viscosity does not matter.")
    # M is printed to eight digits, so M^2 Rm comes back as 9.9999 or 10: group
    # the rows in bins 2 % wide in M^2 Rm.
    row = lambda m, rm: round(math.log(m * m * rm) * 50)
    xs = sorted({row(m, v[1]) for (x, m), v in table.items()})
    for key in READINGS:
        print("\n  reading: %s" % NAMES[key])
        for x in xs:
            pts = sorted((q, v[0][key]) for (q, m), v in table.items()
                         if row(m, v[1]) == x and v[0][key] is not None and v[0][key] > 0)
            sl = slope(pts)
            mem = [m * m * v[1] for (q, m), v in table.items() if row(m, v[1]) == x]
            print("    M^2 Rm = %-7.4g %s   ->   q = %s over %d %s" %
                  (sum(mem) / len(mem), "  ".join("%.4f" % d for _, d in pts),
                   "%+.3f" % sl if sl is not None else "n/a", len(pts), by))

    # ---- the exponent, reading by reading ----------------------------------
    if by == "Rm":
        print("\nM_c(Rm), the field at which D crosses a level, and p in M_c ~ Rm^p."
              "\nPredictions: p = -0.500 (M^2 Rm) and p = -0.333 (M^2 Rm^(2/3)).")
    else:
        print("\nM_c(Re) at fixed Rm, the field at which D crosses a level, and p in M_c ~ Re^p."
              "\np = 0 if the viscosity does not matter; p = -0.500 if the threshold were M^2 Re.")
    for key in READINGS:
        print("\n  reading: %s" % NAMES[key])
        for lv in LEVELS:
            xy, xh, rmof = [], [], {}
            for v_ in vals:
                sel = [(m, v[0][key]) for (q, m), v in table.items() if q == v_]
                got = crossing(sel, lv)
                if got:
                    xy.append((v_, got[0]))
                    xh.append((v_, got[1]))
                    rmof[v_] = [v[1] for (q, m), v in table.items() if q == v_][0]
            p = slope(xy)
            bd = slope_bound(xh)
            cells = "  ".join("%s %g: %.4f" % (by, x, y) for x, y in xy) or "(never crossed)"
            print("    D = %.2f   %s   ->   p = %s over %d %s%s" %
                  (lv, cells, "%+.3f" % p if p is not None else "n/a", len(xy), by,
                   "  (ladder allows +/- %.3f)" % bd if bd is not None else ""))
            if lv >= 0.7 and xy:
                print("             as M_c^2 Rm:  %s" %
                      "  ".join("%s %g: %.2f" % (by, x, y * y * rmof[x]) for x, y in xy))

    if not a.out:
        return
    # ---- the figure: the same points against each candidate variable -------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    marks = ["o", "s", "^", "D", "v", "P"]
    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e4e3dd"
    if len(vals) > len(ramp):
        sys.exit("more values of %s than the ramp has steps: draw fewer" % by)
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    key = a.reading
    if by == "Rm":
        panels = [("M^2 Rm   (Mak, Griffiths & Hughes)", lambda m, x, rm: m * m * rm),
                  ("M^2 Rm^(2/3)   (Gilbert, Mason & Tobias)",
                   lambda m, x, rm: m * m * rm ** (2.0 / 3.0))]
    else:
        panels = [("M^2 Rm   (no dependence on the viscosity)", lambda m, x, rm: m * m * rm),
                  ("M^2 Re   (what a viscous threshold would collapse on)",
                   lambda m, x, rm: m * m * x)]
    for ax, (label, fx) in zip(axes, panels):
        for i, v_ in enumerate(vals):
            pts = sorted((fx(m, q, v[1]), v[0][key]) for (q, m), v in table.items()
                         if q == v_ and v[0][key] is not None)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=ramp[i], lw=1.6,
                        marker=marks[i], ms=6, mec="#fcfcfb", mew=1.0,
                        label="%s = %g" % (by, v_))
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.grid(True, color=grid, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper right", ncol=len(vals), frameon=False, fontsize=9,
               bbox_to_anchor=(0.995, 0.915))
    r0s = ", ".join(sorted({"%g" % r["R0"] for r in runs}))
    fixed = ("Pm = 1" if pms == [1.0] else
             "%s = %s" % (other, ", ".join("%g" % x for x in oth)))
    fig.suptitle("Single vortex, %s, R0 = %s cells: read at %s, against two candidate "
                 "variables" % (fixed, r0s, NAMES[key]), x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(a.out, dpi=130, facecolor="#fcfcfb")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
