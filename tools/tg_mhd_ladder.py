#!/usr/bin/env python3
"""Tabulate the confined-MHD Reynolds ladder and apply the plan's Stage 3 gate.

    tg_mhd_ladder.py RUNS_DIR [--tol 0.05] [--half 1.0]

Reads every RUNS_DIR/*/series.dat written by demonstrator/tg_mhd or
GPU/src/tg_mhd.cu (GPU/csf3/tg_mhd_ladder.sub puts them in
runs/tg_mhd_ladder/), and groups the runs by Re and by setup, read from the tag
in each header: A free slip + conducting, B no slip + conducting, C no slip
without a field (P, a periodic box, is listed but never gated).

THE OBSERVABLE is the plan's, fixed before any run: f_w, the share of the
viscous plus Ohmic dissipation within 1, 2, 4 x Re^-1/2 of the walls (series.dat
columns fw1 fw2 fw4), AT THE DISSIPATION PEAK. For a no-slip box that phrase
needs one clarification, because its largest dissipation is the impulsive start:
the Taylor-Green velocity does not vanish on the walls, so t = 0 puts a vortex
sheet on every one of them. Measured in tgc_noslip_cond_n129_re300: eps = 0.129
at t = 0 with fw1 = 0.91, falling to ~0.036 near t = 0.7, then a TURBULENT
maximum of 0.0485 at t = 2 with fw1 = 0.15. The peak meant is the second one.
So a run's peak is the maximum of eps after its first minimum that eps later
rises 5 % above; a free-slip run rises at once, so for it this is simply the
global maximum. The peak time is refined by a parabola through the three samples
around the discrete maximum and f_w is interpolated to it, because near a no-slip
peak fw1 moves ~0.2 per unit time and the probe spacing alone would otherwise
decide a few per cent of the answer. A run whose eps is still rising at its last
sample has not reached its peak, and prints "--" rather than its last value.

Beside it, fw1 averaged over +-half (default 1) around the same peak, as a check
that the instantaneous value is not an accident of one time.

THE GATE: for every (setup, Re) run at two resolutions, fw1, fw2 and fw4 at the
peak must each agree within --tol (5 %, the plan's number). A rung that fails is
flagged and must not go into a figure; the exit status is 1 if any rung fails.
The tolerance is RELATIVE for every setup -- the author's decision, 2026-09-27 --
and that is strict on A, whose fw1 is ~0.01, so 5 % of it is 5e-4 absolute: the
N = 65 / 97 pair at Re = 1000 fails it by 30 % while its fw4 agrees to 0.8 %.
A's small absolute error is what the headline's band weighs.
THE HEADLINE is, per Re, B - A (the wall) and B - C (the field) in fw1 on the
finer grids, beside the resolution band -- the two setups' fw1 differences
between grids, added -- and marked when it clears the band. The PRL claim needs
B - A outside the band and monotonic over at least three rungs; that is checked
too.

Pure stdlib -- it runs on a CSF3 login node with no numpy.
"""
import argparse
import glob
import math
import os
import re
import sys

COLS = ("t E_V E_M E_T H_C Omega_V Omega_M eps EM/EV OmM/OmV j_max w_max skew "
        "divb/j divb_wall/j mass_drift umax_lat fw1 fw2 fw4 tau_wall").split()
IX = {c: i for i, c in enumerate(COLS)}
FW = ("fw1", "fw2", "fw4")


def load(path):
    head, rows = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                head.append(line[1:].strip())
                continue
            p = line.split()
            if len(p) == len(COLS):
                try:
                    rows.append([float(v) for v in p])
                except ValueError:
                    pass                      # a torn last line of a live run
    if not rows or not head:
        return None
    h = head[0]
    tag = next((t for t in h.split() if re.match(r"^(tgc|tgi|tga|hydro)_", t)), None)
    mN = re.search(r"\bN=(\d+)", h)
    mRe = re.search(r"\bRe=([0-9.eE+-]+)", h)
    if not (tag and mN and mRe):
        return None
    if "_periodic_" in tag:
        setup = "P"
    elif "_slip_" in tag:
        setup = "A"
    elif "_noslip_" in tag and "_nofield_" in tag:
        setup = "C"
    elif "_noslip_" in tag:
        setup = "B"
    else:
        setup = "?"
    return {"path": path, "tag": tag, "setup": setup, "N": int(mN.group(1)),
            "Re": float(mRe.group(1)),
            "prec": "FP32" if "FP32" in h else ("FP64" if "FP64" in h else "?"),
            "rows": rows, "diverged": any("DIVERGED" in x for x in head)}


def col(run, name):
    return [r[IX[name]] for r in run["rows"]]


def finite_max(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return max(xs) if xs else None


def interp(run, name, t):
    """Linear interpolation of a column at time t (inside the sampled range)."""
    rows = run["rows"]
    for a, b in zip(rows, rows[1:]):
        if a[0] <= t <= b[0]:
            w = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
            return (1 - w) * a[IX[name]] + w * b[IX[name]]
    return None


def turbulent_peak(run, rise=1.05):
    """(t_peak, eps_peak) of the maximum after the impulsive start, or None.

    The start ends at the first running minimum of eps that eps later exceeds by
    the factor `rise`; a wiggle smaller than that during the initial decay is not
    taken for the end of it."""
    t, e = col(run, "t"), col(run, "eps")
    if len(e) < 3 or not all(math.isfinite(x) for x in e):
        return None
    k = 0
    for i in range(1, len(e)):
        if e[i] < e[k]:
            k = i
        elif e[i] > rise * e[k]:
            break
    else:
        return None                           # decays throughout: no turbulent peak
    j = max(range(k, len(e)), key=lambda i: e[i])
    if j == len(e) - 1:
        return None                           # still rising at the last sample
    if j == 0:
        return t[0], e[0]
    # vertex of the parabola through (t, e) at j-1, j, j+1 -- general spacing
    (t0, t1, t2), (e0, e1, e2) = t[j - 1:j + 2], e[j - 1:j + 2]
    den = (t0 - t1) * (t0 - t2) * (t1 - t2)
    A = (t2 * (e1 - e0) + t1 * (e0 - e2) + t0 * (e2 - e1)) / den
    B = (t2 * t2 * (e0 - e1) + t1 * t1 * (e2 - e0) + t0 * t0 * (e1 - e2)) / den
    if A >= 0:
        return t1, e1
    tp = min(max(-B / (2 * A), t0), t2)
    return tp, interp(run, "eps", tp)


def window_mean(run, name, t0, t1):
    """Trapezoidal mean of a column over [t0, t1]; None if the run does not cover it."""
    pts = [(r[0], r[IX[name]]) for r in run["rows"] if t0 - 1e-9 <= r[0] <= t1 + 1e-9]
    if len(pts) < 2 or pts[0][0] > t0 + 0.2 or pts[-1][0] < t1 - 0.2:
        return None
    area = sum(0.5 * (a[1] + b[1]) * (b[0] - a[0]) for a, b in zip(pts, pts[1:]))
    return area / (pts[-1][0] - pts[0][0])


def fmt(v, spec="%.4f", width=7):
    return "--".rjust(width) if v is None else (spec % v).rjust(width)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_dir")
    ap.add_argument("--tol", type=float, default=0.05, help="resolution gate on f_w (default 0.05)")
    ap.add_argument("--half", type=float, default=1.0, help="half-width of the fw1 window")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.runs_dir, "*", "series.dat")))
    runs = [r for r in (load(p) for p in paths) if r is not None]
    if not runs:
        raise SystemExit("no tg_mhd series.dat under %s" % args.runs_dir)

    print("%-6s %-2s %4s %4s | %6s %9s | %7s %7s %7s | %7s | %8s %8s %8s  %s"
          % ("Re", "S ", "N", "prec", "t_peak", "eps_peak", "fw1", "fw2", "fw4",
             "fw1+-%g" % args.half, "minEM/EV", "|mass|", "divb/j", "tag"))
    res = {}
    for Re in sorted({r["Re"] for r in runs}):
        for r in sorted((r for r in runs if r["Re"] == Re), key=lambda r: (r["setup"], r["N"])):
            pk = turbulent_peak(r)
            fw = [interp(r, m, pk[0]) for m in FW] if pk else [None] * 3
            fwin = window_mean(r, "fw1", pk[0] - args.half, pk[0] + args.half) if pk else None
            em = [x for x in col(r, "EM/EV")[1:] if math.isfinite(x) and x > 0]
            mass = finite_max([abs(x) for x in col(r, "mass_drift")])
            divb = finite_max(col(r, "divb/j")[1:])
            note = " DIVERGED" if r["diverged"] else ("" if pk else "  (no peak yet)")
            print("%-6g %-2s %4d %4s | %s %s | %s %s %s | %s | %s %s %s  %s%s"
                  % (Re, r["setup"], r["N"], r["prec"],
                     fmt(pk[0] if pk else None, "%.2f", 6), fmt(pk[1] if pk else None, "%.3e", 9),
                     fmt(fw[0]), fmt(fw[1]), fmt(fw[2]), fmt(fwin),
                     fmt(min(em) if em else None, "%.3f", 8), fmt(mass, "%.1e", 8),
                     fmt(divb, "%.1e", 8), r["tag"], note))
            res.setdefault((Re, r["setup"]), []).append((r["N"], fw, pk))
        print()

    print("RESOLUTION GATE: f_w at the peak from the two finest grids of each (setup, Re),"
          " tolerance %.0f %%" % (100 * args.tol))
    band, fine, failed = {}, {}, []
    for (Re, setup), lst in sorted(res.items()):
        if setup == "P":
            continue
        lst = sorted(lst, key=lambda x: x[0])
        fine[(Re, setup)] = lst[-1][1][0]
        grids = sorted({n for n, _, _ in lst})
        if len(grids) < len(lst):
            print("  Re %-6g %s  more than one run at N = %s: the gate compares the two finest"
                  " DISTINCT grids" % (Re, setup, ", ".join(str(n) for n in grids
                                                          if sum(m == n for m, _, _ in lst) > 1)))
            lst = [next(x for x in reversed(lst) if x[0] == n) for n in grids]
        if len(lst) < 2:
            print("  Re %-6g %s  one grid only (N = %d) -- not gated" % (Re, setup, lst[0][0]))
            continue
        (n1, f1, p1), (n2, f2, p2) = lst[-2], lst[-1]
        if None in f1 or None in f2:
            print("  Re %-6g %s  N = %d vs %d: a run has no peak yet -- not gated" % (Re, setup, n1, n2))
            failed.append((Re, setup))
            continue
        rel = [abs(a - b) / max(abs(a), abs(b)) if max(abs(a), abs(b)) > 0 else 0.0
               for a, b in zip(f1, f2)]
        ok = max(rel) <= args.tol
        band[(Re, setup)] = abs(f2[0] - f1[0])
        if not ok:
            failed.append((Re, setup))
        print("  Re %-6g %s  N = %d vs %d: fw1/fw2/fw4 differ %.1f/%.1f/%.1f %%, t_peak %.2f vs %.2f  %s"
              % (Re, setup, n1, n2, 100 * rel[0], 100 * rel[1], 100 * rel[2], p1[0], p2[0],
                 "ok" if ok else "FAIL"))

    print("\nHEADLINE: fw1 at the peak, finer grid of each; band = the two setups' grid differences added")
    wall = []
    for Re in sorted({k[0] for k in fine}):
        fb = fine.get((Re, "B"))
        if fb is None:
            continue
        line = "  Re %-6g" % Re
        for other, what in (("A", "wall"), ("C", "field")):
            fo = fine.get((Re, other))
            if fo is None:
                line += "   B - %s (%s)    --      " % (other, what)
                continue
            d = fb - fo
            bb, bo = band.get((Re, "B")), band.get((Re, other))
            gated = (Re, "B") not in failed and (Re, other) not in failed
            if bb is None or bo is None:
                verdict = "band unknown"
            else:
                verdict = ("clears %.4f" if abs(d) > bb + bo else "inside %.4f") % (bb + bo)
            if not gated:
                verdict += ", GATE FAILED"
            line += "   B - %s (%s) %+.4f  %-24s" % (other, what, d, verdict)
            if other == "A" and gated and bb is not None and bo is not None and abs(d) > bb + bo:
                wall.append((Re, d))
        print(line.rstrip())
    if len(wall) >= 3:
        ds = [d for _, d in wall]
        mono = all(a < b for a, b in zip(ds, ds[1:])) or all(a > b for a, b in zip(ds, ds[1:]))
        print("  B - A clears its band on %d rungs and is %s in Re" % (len(wall), "MONOTONIC" if mono else "NOT monotonic"))
    else:
        print("  B - A clears its band on %d rung(s); the claim needs three" % len(wall))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
