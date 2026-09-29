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
decide a few per cent of the answer. Two runs have no f_w, and say which: one
whose eps never rises 5 % above its post-start minimum has NO TURBULENT PEAK
inside the run (B at Re = 125 and C at Re <= 500 in the first CSF3 ladder) --
the author's decision, 2026-09-28, is that such a run has none, rather than
being read at another run's time -- and one still rising at its last sample has
not REACHED its peak.

HOW f_w IS MEASURED, and the series that got it wrong. A series whose header
carries fw=profile (both drivers since 2026-09-28) reads f_w off the
wall-distance profile with the band edge interpolated, and is taken as it is.
Older series COUNTED WHOLE NODE LAYERS -- every node with k h < m delta -- which
with the trapezoidal weights is a band of (ceil(m delta/h) - 1/2) h, not
m delta. At the first CSF3 ladder's grid pairs that is 0.91 against 1.07 delta
at Re = 250 and 1000, and fw1 differed by 8-25 % between the two grids of a rung
while the peak dissipation agreed to 1 %; the gap the band predicts and the gap
measured had the same sign in all 24 comparisons (8 pairs x 3 bands). For such
series this tool places each column at the band it really measured and
interpolates the three, with the origin, piecewise-linearly to exactly 1, 2 and
4 delta; their rows are marked. That is a reconstruction from three points --
enough to show what the gate was measuring, not a number for a figure.

Beside f_w: fw1/vol, fw1 over the delta shell's share of the box volume,
1 - (1 - 2 delta/pi)^3 -- the shell's dissipation density against the mean,
where a wall's effect shows as a trend instead of as a share that shrinks with
the shell -- and fw1 averaged over +-half (default 1) around the peak, as a check
that the instantaneous value is not an accident of one time.

THE GATE: for every (setup, Re) run at two resolutions, fw1, fw2 and fw4 at the
peak must each agree within --tol (5 %, the plan's number). A rung that fails is
flagged and must not go into a figure; the exit status is 1 if any rung fails.
The tolerance is RELATIVE for every setup -- the author's decision, 2026-09-27 --
which is strict on A, whose fw1 is a few 1e-2. THE HEADLINE is, per Re, B - A
(the wall) and B - C (the field) in fw1 on the finer grids, beside the
resolution band -- the two setups' fw1 differences between grids, added -- and
marked when it clears the band. The PRL claim needs B - A outside the band and
monotonic over at least three rungs; that is checked too.

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
            "fwdef": "profile" if "fw=profile" in h.split() else "count",
            "rows": rows, "diverged": any("DIVERGED" in x for x in head)}


def col(run, name):
    return [r[IX[name]] for r in run["rows"]]


def finite_max(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return max(xs) if xs else None


def band_corrected(N, Re, f):
    """fw1/fw2/fw4 from a series that counted whole node layers, moved to exactly
    1, 2, 4 delta: each sample sits at the band it really measured,
    (ceil(m r) - 1/2) / r delta with r = delta/h, and the three with the origin
    are interpolated piecewise-linearly (the last segment extended if needed)."""
    r = (N - 1) / (math.pi * math.sqrt(Re))
    pts = sorted([(0.0, 0.0)] + [((math.ceil(m * r) - 0.5) / r, v) for m, v in zip((1, 2, 4), f)])
    out = []
    for x in (1.0, 2.0, 4.0):
        (x0, y0), (x1, y1) = next(((a, b) for a, b in zip(pts, pts[1:]) if a[0] <= x <= b[0]),
                                  (pts[-2], pts[-1]))
        out.append(y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return out


def fw_rows(run):
    """[(t, [fw1, fw2, fw4])] at every probe, band-corrected for a counted series."""
    out = []
    for r in run["rows"]:
        f = [r[IX[m]] for m in FW]
        out.append((r[0], band_corrected(run["N"], run["Re"], f) if run["fwdef"] == "count" else f))
    return out


def fw_at(run, t):
    rows = fw_rows(run)
    for (ta, fa), (tb, fb) in zip(rows, rows[1:]):
        if ta <= t <= tb:
            w = 0.0 if tb == ta else (t - ta) / (tb - ta)
            return [(1 - w) * a + w * b for a, b in zip(fa, fb)]
    return None


def interp(run, name, t):
    """Linear interpolation of a column at time t (inside the sampled range)."""
    rows = run["rows"]
    for a, b in zip(rows, rows[1:]):
        if a[0] <= t <= b[0]:
            w = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
            return (1 - w) * a[IX[name]] + w * b[IX[name]]
    return None


def turbulent_peak(run, rise=1.05):
    """(t_peak, eps_peak) of the maximum after the impulsive start, or the reason
    there is none. The start ends at the first running minimum of eps that eps
    later exceeds by the factor `rise`; a wiggle smaller than that during the
    initial decay is not taken for the end of it."""
    t, e = col(run, "t"), col(run, "eps")
    if len(e) < 3 or not all(math.isfinite(x) for x in e):
        return "too short"
    k = 0
    for i in range(1, len(e)):
        if e[i] < e[k]:
            k = i
        elif e[i] > rise * e[k]:
            break
    else:
        return "no turbulent peak by t = %.1f" % t[-1]
    j = max(range(k, len(e)), key=lambda i: e[i])
    if j == len(e) - 1:
        return "peak not reached by t = %.1f" % t[-1]
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


def window_fw1(run, t0, t1):
    """Trapezoidal mean of fw1 over [t0, t1]; None if the run does not cover it."""
    pts = [(t, f[0]) for t, f in fw_rows(run) if t0 - 1e-9 <= t <= t1 + 1e-9]
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

    print("%-6s %-2s %4s %4s | %6s %9s | %7s %7s %7s | %7s %7s | %8s %8s %8s  %s"
          % ("Re", "S ", "N", "prec", "t_peak", "eps_peak", "fw1", "fw2", "fw4", "fw1/vol",
             "fw1+-%g" % args.half, "minEM/EV", "|mass|", "divb/j", "tag"))
    res, counted = {}, False
    for Re in sorted({r["Re"] for r in runs}):
        vol1 = 1.0 - (1.0 - 2.0 / (math.pi * math.sqrt(Re))) ** 3
        for r in sorted((r for r in runs if r["Re"] == Re), key=lambda r: (r["setup"], r["N"])):
            pk = turbulent_peak(r)
            ok = isinstance(pk, tuple)
            fw = fw_at(r, pk[0]) if ok else None
            fw = fw if fw is not None else [None] * 3
            fwin = window_fw1(r, pk[0] - args.half, pk[0] + args.half) if ok else None
            em = [x for x in col(r, "EM/EV")[1:] if math.isfinite(x) and x > 0]
            mass = finite_max([abs(x) for x in col(r, "mass_drift")])
            divb = finite_max(col(r, "divb/j")[1:])
            note = ""
            if r["diverged"]:
                note += "  DIVERGED"
            if not ok:
                note += "  (%s)" % pk
            if r["fwdef"] == "count":
                note += "  *"
                counted = True
            print("%-6g %-2s %4d %4s | %s %s | %s %s %s | %s %s | %s %s %s  %s%s"
                  % (Re, r["setup"], r["N"], r["prec"],
                     fmt(pk[0] if ok else None, "%.2f", 6), fmt(pk[1] if ok else None, "%.3e", 9),
                     fmt(fw[0]), fmt(fw[1]), fmt(fw[2]),
                     fmt(fw[0] / vol1 if fw[0] is not None else None, "%.3f"), fmt(fwin),
                     fmt(min(em) if em else None, "%.3f", 8), fmt(mass, "%.1e", 8),
                     fmt(divb, "%.1e", 8), r["tag"], note))
            res.setdefault((Re, r["setup"]), []).append((r["N"], fw, pk if ok else None))
        print()
    if counted:
        print("* f_w in this series counted whole node layers (no fw=profile in its header);"
              " shown band-corrected from its three columns -- a reconstruction, not a"
              " measurement. Re-run with the current drivers for figures.\n")

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
        if None in f1 and None in f2:
            # No turbulent peak on either grid: no f_w exists, by the author's
            # decision -- the grids AGREE, so this is not a failure of the gate.
            print("  Re %-6g %s  N = %d vs %d: no f_w on either grid (no turbulent peak) -- not gated"
                  % (Re, setup, n1, n2))
            continue
        if None in f1 or None in f2:
            # One grid has a peak and the other not: that IS a disagreement.
            print("  Re %-6g %s  N = %d vs %d: a turbulent peak on one grid only  FAIL" % (Re, setup, n1, n2))
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
