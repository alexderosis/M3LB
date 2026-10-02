#!/usr/bin/env python3
"""Every number the paper quotes from the runs, and its four tables.

    python3 paper/tg_mhd/numbers.py

Reads results/P_tg_mhd/{round2,ladder,mach,fp64check} through
tools/tg_mhd_ladder.py and tools/plot_tg_mhd_ladder.py -- so the peak, f_w and
the integrated share are exactly those of the figures -- prints the quantities
the text cites, section by section, and writes tab_runs.tex, tab_verify.tex,
tab_results.tex and tab_windows.tex beside this file for main.tex to \\input.
A number in the text that this script does not print is a number to distrust.
Pure stdlib, like the tools it imports.
"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import tg_mhd_ladder as T          # noqa: E402
import plot_tg_mhd_ladder as P     # noqa: E402  (matplotlib is imported only by its main)

U0, TMAX = 0.05, 10.0
GRIDS = {125: (128, 192), 250: (192, 256), 500: (256, 384), 1000: (384, 512), 2000: (512, 640)}
RUNGS = (250, 500, 1000, 2000)
WINDOWS = ((2.0, 6.0), (2.0, 10.0), (3.0, 10.0), (4.0, 10.0), (5.0, 10.0))
DEFAULT = (2.0, 10.0)
FRACTIONS = ((0.9, 0.5), (0.8, 0.5), (0.7, 0.4), (0.6, 0.4))


def res(d):
    return os.path.join(ROOT, "results", "P_tg_mhd", d)


def runs_of(*dirs):
    return P.load_all([res(d) for d in dirs])


RUNS = runs_of("round2", "ladder")
BEST = P.collect(RUNS)


def fine(s, Re):
    return RUNS[(s, Re)][max(RUNS[(s, Re)])]


def peak_fw(run):
    pk = T.turbulent_peak(run)
    return (pk, T.fw_at(run, pk[0])) if isinstance(pk, tuple) else (pk, None)


def dissipated(run, t0, t1):
    """int eps dt over [t0, t1], edges interpolated as in T.integrated."""
    q = [(t, [e]) for t, e in zip(T.col(run, "t"), T.col(run, "eps"))]
    return T.trapz_window(q, t0, min(t1, q[-1][0]))[0]


def t_at_fraction(run, f):
    """The first time E_T/E_T(0) falls to f, interpolated; None if it never does."""
    t, E = T.col(run, "t"), T.col(run, "E_T")
    for i in range(1, len(E)):
        if E[i] / E[0] <= f:
            w = (E[i - 1] / E[0] - f) / ((E[i - 1] - E[i]) / E[0])
            return t[i - 1] + w * (t[i] - t[i - 1])
    return None


def first_max_after_fall(xs):
    """The first local maximum of a series after its first local minimum."""
    i = 1
    while i < len(xs) and xs[i] <= xs[i - 1]:
        i += 1
    while i < len(xs) and xs[i] >= xs[i - 1]:
        i += 1
    return xs[i - 1]


def pct(a, b):
    return 100.0 * abs(a - b) / max(abs(a), abs(b))


def write(name, text):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(text)


def main():
    out = []

    def say(s=""):
        out.append(s)
        print(s)

    # ---- Sec. II: parameters -----------------------------------------------------
    say("II. PARAMETERS (u0 = %g, t = 0..%g)" % (U0, TMAX))
    rows = []
    for Re, (nc, nf) in sorted(GRIDS.items()):
        dh = [(n - 1) / (math.pi * math.sqrt(Re)) for n in (nc, nf)]
        nu_lat = [U0 * (n - 1) / (math.pi * Re) for n in (nc, nf)]
        steps = int(round(TMAX * (nf - 1) / (math.pi * U0)))      # as the driver: 40680 at N = 640
        ma = max(max(T.col(fine(s, Re), "umax_lat")) * math.sqrt(3.0) for s in ("A", "B"))
        prec = fine("A", Re)["prec"].replace("FP", "")
        say("  Re %5d  N %d/%d  delta/h %.2f/%.2f  tau-1/2 (fluid) %.4f/%.4f  (magnetic, D3Q7 cs2=1/4) %.4f  "
            "steps %d  peak Ma %.3f  FP%s" % (Re, nc, nf, dh[0], dh[1], 3 * nu_lat[0], 3 * nu_lat[1],
                                             4 * nu_lat[1], steps, ma, prec))
        rows.append("%d & %d, %d & %.2f, %.2f & %.4f, %.4f & %d & %.3f & %s \\\\" % (
            Re, nc, nf, dh[0], dh[1], 3 * nu_lat[0], 3 * nu_lat[1], steps, ma, prec))
    write("tab_runs.tex", "\\begin{tabular}{rcccccc}\n"
          "$\\mathrm{Re}$ & $N$ & $\\delta/h$ & $\\tau-\\tfrac12$ & steps & $\\mathrm{Ma}_{\\max}$ & bits \\\\\n"
          "\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n")

    # ---- Sec. III: verification ----------------------------------------------------
    say()
    say("III. VERIFICATION")
    for N in (384, 512):
        r = RUNS[("A", 1000)][N]
        em, t = T.col(r, "EM/EV"), T.col(r, "t")
        k = min(range(1, len(em)), key=lambda i: em[i])
        say("  C2 (A, Re 1000, N %d): min E_M/E_V %.4f at t = %.2f; first max Om_M/Om_V after its fall %.3f"
            % (N, em[k], t[k], first_max_after_fall(T.col(r, "OmM/OmV"))))
    r = fine("A", 2000)
    say("  A at Re 2000: min E_M/E_V %.3f" % min(x for x in T.col(r, "EM/EV")[1:] if x > 0))
    gate, worst = {}, (0.0, "")
    for (s, Re), by_n in sorted(RUNS.items()):
        if s not in ("A", "B") or len(by_n) < 2:
            continue
        nc, nf = sorted(by_n)[-2:]
        (pc, fc), (pf, ff) = peak_fw(by_n[nc]), peak_fw(by_n[nf])
        if fc is None or ff is None:
            say("  gate %s Re %d: no turbulent peak (%s)" % (s, Re, pf))
            continue
        d = [pct(a, b) for a, b in zip(fc, ff)]
        gate[(s, Re)] = (d, abs(ff[0] - fc[0]))
        worst = max(worst, (max(d), "%s Re %d" % (s, Re)))
    say("  gate: worst %.1f %% (%s)" % worst)
    mach = runs_of("mach")
    ctrl = {}
    for Re in (500, 1000):
        N = max(mach[("A", Re)])
        d_lo = peak_fw(mach[("B", Re)][N])[1][0] - peak_fw(mach[("A", Re)][N])[1][0]
        d_hi = peak_fw(RUNS[("B", Re)][N])[1][0] - peak_fw(RUNS[("A", Re)][N])[1][0]
        ctrl[Re] = ("Mach", d_lo - d_hi)
        ta = peak_fw(mach[("A", Re)][N])[0][0]
        say("  Mach halving, Re %d N %d: B-A %+.4f -> %+.4f, shift %+.5f; A's peak at halved Mach t = %.2f"
            % (Re, N, d_hi, d_lo, d_lo - d_hi, ta))
    fp = runs_of("fp64check")
    d64 = peak_fw(fp[("B", 2000)][512])[1][0] - peak_fw(fp[("A", 2000)][512])[1][0]
    d32 = peak_fw(RUNS[("B", 2000)][512])[1][0] - peak_fw(RUNS[("A", 2000)][512])[1][0]
    ctrl[2000] = ("FP64", d64 - d32)
    worst_rel = 0.0
    for s in ("A", "B"):
        a, b = RUNS[(s, 2000)][512], fp[(s, 2000)][512]
        for c in ("eps", "fw1", "E_V", "E_M"):
            worst_rel = max(worst_rel, max(abs(x - y) / max(abs(y), 1e-300)
                                           for x, y in zip(T.col(a, c), T.col(b, c))))
    say("  FP32 vs FP64, Re 2000 N 512: B-A %+.4f vs %+.4f, shift %+.5f; series differ up to %.0e relative"
        % (d32, d64, d64 - d32, worst_rel))
    mass = max(abs(x) for by_n in RUNS.values() for r in by_n.values() for x in T.col(r, "mass_drift"))
    divb = max(max(T.col(r, "divb/j")[1:]) for by_n in RUNS.values() for r in by_n.values())
    say("  over every run: |mass drift| <= %.1e, divb/|j| <= %.1e" % (mass, divb))
    rows = []
    for Re in RUNGS:
        (da, ba), (db, bb) = gate[("A", Re)], gate[("B", Re)]
        c = ctrl.get(Re)
        cs = "--" if c is None else "%s, %s" % (c[0], "$<10^{-4}$" if abs(c[1]) < 1e-4 else "$%+.4f$" % c[1])
        rows.append("%d & %d, %d & %.1f/%.1f/%.1f & %.1f/%.1f/%.1f & %.4f & %s \\\\" % (
            Re, GRIDS[Re][0], GRIDS[Re][1], da[0], da[1], da[2], db[0], db[1], db[2], ba + bb, cs))
    write("tab_verify.tex", "\\begin{tabular}{rccccl}\n"
          "$\\mathrm{Re}$ & $N$ & A: $\\Delta f_1/\\Delta f_2/\\Delta f_4$ (\\%) & B: $\\Delta f_1/\\Delta f_2/\\Delta f_4$ (\\%)"
          " & band & control shift \\\\\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n")

    # ---- Sec. IV: results ------------------------------------------------------------
    say()
    say("IV.A WHERE IT DISSIPATES (integrated over t = %g-%g)" % DEFAULT)
    for s in ("C", "B", "A"):
        if (s, 1000) in BEST:
            N, d, r, pk, fw = BEST[(s, 1000)]
            say("  Re 1000 %s: density at the wall %.2f x mean" % (s, P.profile_integrated(d, r, *DEFAULT)[1][0]))
    for s in ("A", "B"):
        v = [P.profile_integrated(BEST[(s, Re)][1], BEST[(s, Re)][2], *DEFAULT)[1][0] for Re in RUNGS]
        say("  %s wall density, Re %s: %s" % (s, "/".join(str(Re) for Re in RUNGS), " / ".join("%.3f" % x for x in v)))
    for w in WINDOWS:
        v = [P.profile_integrated(BEST[("B", Re)][1], BEST[("B", Re)][2], *w)[1][0] for Re in RUNGS]
        a = [P.profile_integrated(BEST[("A", Re)][1], BEST[("A", Re)][2], *w)[1][0] for Re in RUNGS]
        say("    window %g-%g: B slope %+.2f, A slope %+.2f (log over Re 250 -> 2000)"
            % (w[0], w[1], math.log(v[-1] / v[0]) / math.log(8), math.log(a[-1] / a[0]) / math.log(8)))
    c = P.integrated_fine(RUNS, ("C", 1000), *DEFAULT)
    b = P.integrated_fine(RUNS, ("B", 1000), *DEFAULT)
    pkc = BEST[("C", 1000)]
    say("  Re 1000: F1(C) %.4f vs F1(B) %.4f (ratio %.1f); C's turbulent peak t = %.2f eps %.3e fw1 %.3f"
        % (c[0][0], b[0][0], c[0][0] / b[0][0], pkc[3][0], pkc[3][1], pkc[4][0]))
    say("  B's impulsive start: eps(0) = %s; fw1(0) = %s" % (
        " / ".join("%.3f" % T.col(fine("B", Re), "eps")[0] for Re in RUNGS),
        " / ".join("%.3f" % T.col(fine("B", Re), "fw1")[0] for Re in RUNGS)))

    say()
    say("IV.B THE NO-SLIP EXCESS (integrated over t = %g-%g)" % DEFAULT)
    rows = []
    for Re in RUNGS:
        a, b = P.integrated_fine(RUNS, ("A", Re), *DEFAULT), P.integrated_fine(RUNS, ("B", Re), *DEFAULT)
        ea = dissipated(fine("A", Re), *DEFAULT) / T.col(fine("A", Re), "E_T")[0]
        eb = dissipated(fine("B", Re), *DEFAULT) / T.col(fine("B", Re), "E_T")[0]
        say("  Re %4d: F1 A %.4f B %.4f  B-A %+.4f (band %.4f) | visc A %.4f B %.4f (B-A %+.4f) | ohm A %.4f B %.4f"
            " (B-A %+.4f) | energy dissipated in the window / E_T(0): A %.3f B %.3f | absolute near-wall Ohmic"
            " B/A %.2f" % (Re, a[0][0], b[0][0], b[0][0] - a[0][0], a[2] + b[2], a[1][0], b[1][0],
                           b[1][0] - a[1][0], a[1][1], b[1][1], b[1][1] - a[1][1], ea, eb,
                           b[1][1] * eb / (a[1][1] * ea)))
        pa, pb = BEST[("A", Re)], BEST[("B", Re)]
        rows.append("%d & %.2f & %.2f & $%+.3f$ & %.3f & %.3f & $%+.3f$ & $%+.3f$ & $%+.3f$ \\\\" % (
            Re, pa[3][0], pb[3][0], pb[4][0] - pa[4][0], a[0][0], b[0][0], b[0][0] - a[0][0],
            b[1][0] - a[1][0], b[1][1] - a[1][1]))
    write("tab_results.tex", "\\begin{tabular}{rcccccccc}\n"
          " & \\multicolumn{3}{c}{at the peak (pre-registered)} & \\multicolumn{5}{c}{integrated, $t=2$--$10$} \\\\\n"
          "\\cline{2-4}\\cline{5-9}\n"
          "$\\mathrm{Re}$ & $t_p^{\\mathrm{A}}$ & $t_p^{\\mathrm{B}}$ & B$-$A & $F^{\\mathrm{A}}$ & $F^{\\mathrm{B}}$"
          " & B$-$A & viscous & Ohmic \\\\\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n")
    sa = [P.integrated_fine(RUNS, ("A", Re), *DEFAULT)[0][0] for Re in RUNGS]
    sb = [P.integrated_fine(RUNS, ("B", Re), *DEFAULT)[0][0] for Re in RUNGS]
    say("  F1 slopes over Re 250 -> 2000: A %+.2f, B %+.2f" % (
        math.log(sa[-1] / sa[0]) / math.log(8), math.log(sb[-1] / sb[0]) / math.log(8)))

    say()
    say("IV.C THE PRE-REGISTERED VERDICT")
    vals = []
    for Re in RUNGS:
        a, b = BEST[("A", Re)], BEST[("B", Re)]
        band = gate[("A", Re)][1] + gate[("B", Re)][1]
        vals.append((Re, b[4][0] - a[4][0], band))
        say("  Re %4d: t_peak A %.2f B %.2f; B-A %+.4f, band %.4f" % (Re, a[3][0], b[3][0], b[4][0] - a[4][0], band))
    (r1, d1, b1), (r2, d2, b2) = vals[-2], vals[-1]
    say("  Re %d -> %d: %+.4f against bands %.4f (ratio %.1f): %s" % (
        r1, r2, d2 - d1, b1 + b2, abs(d2 - d1) / (b1 + b2), "walls hold" if abs(d2 - d1) <= b1 + b2 else "cascade"))

    say()
    say("IV.D THE PEAK SWITCHES")
    for s in ("A", "B"):
        for Re in RUNGS:
            r = fine(s, Re)
            t, e, f = T.col(r, "t"), T.col(r, "eps"), T.col(r, "fw1")
            mx = [(t[i], e[i]) for i in range(1, len(e) - 1) if e[i] > e[i - 1] and e[i] >= e[i + 1] and t[i] > 0.3]
            late = [x for tt, x in zip(t, f) if tt >= 1.5]
            say("  %s Re %4d: eps maxima %s; fw1 over t >= 1.5 in [%.3f, %.3f]" % (
                s, Re, ", ".join("t %.2f eps %.4f" % m for m in mx), min(late), max(late)))
    for tt in (1.5, 2.0, 2.3, 2.5, 3.0, 3.5, 3.7, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0):
        say("  equal-time B-A at t = %4.1f: %s" % (tt, "  ".join(
            "%+.4f" % (T.fw_at(fine("B", Re), min(tt, T.col(fine("B", Re), "t")[-1]))[0]
                       - T.fw_at(fine("A", Re), min(tt, T.col(fine("A", Re), "t")[-1]))[0]) for Re in RUNGS)))
    say("  B's fw1 at t = 1.5 and 3.7, Re 2000: %.3f, %.3f" % tuple(T.fw_at(fine("B", 2000), x)[0] for x in (1.5, 3.7)))

    say()
    say("IV.E THE WINDOWS")
    rows = []
    peak = [v[1] for v in vals]
    rows.append("at the peak (pre-registered) & " + " & ".join("%.3f" % v for v in peak)
                + " & $%+.0f$ \\\\" % (100 * (peak[-1] - peak[0]) / peak[0]))
    for w in WINDOWS:
        d, bands = [], []
        for Re in RUNGS:
            a, b = P.integrated_fine(RUNS, ("A", Re), *w), P.integrated_fine(RUNS, ("B", Re), *w)
            d.append(b[0][0] - a[0][0]); bands.append(a[2] + b[2])
        steps = ["%+.4f (%s)" % (d[i + 1] - d[i], "out" if abs(d[i + 1] - d[i]) > bands[i] + bands[i + 1] else "in")
                 for i in range(len(d) - 1)]
        say("  t = %g-%g: %s  (250 -> 2000 %+.0f %%; steps %s; largest band %.4f)" % (
            w[0], w[1], " / ".join("%.4f" % x for x in d), 100 * (d[-1] - d[0]) / d[0], ", ".join(steps), max(bands)))
        rows.append("integrated, $t=%g$--$%g$ & " % w + " & ".join("%.3f" % x for x in d)
                    + " & $%+.0f$ \\\\" % (100 * (d[-1] - d[0]) / d[0]))
    for fa, fb in FRACTIONS:
        d, bands = [], []
        for Re in RUNGS:
            F = {}
            for s in ("A", "B"):
                by_n = RUNS[(s, Re)]
                got = []
                for n in sorted(by_n)[-2:]:
                    ta, tb = t_at_fraction(by_n[n], fa), t_at_fraction(by_n[n], fb)
                    got.append(T.integrated(by_n[n], ta, tb)[0][0])
                F[s] = got
            d.append(F["B"][-1] - F["A"][-1])
            bands.append(abs(F["A"][1] - F["A"][0]) + abs(F["B"][1] - F["B"][0]))
        say("  E_T/E_T(0) %.1f -> %.1f: %s  (largest band %.4f)" % (fa, fb, " / ".join("%.4f" % x for x in d), max(bands)))
        if (fa, fb) in ((0.9, 0.5), (0.6, 0.4)):
            rows.append("integrated, $E/E_0=%.1f$--$%.1f$ & " % (fa, fb) + " & ".join("%.3f" % x for x in d)
                        + " & $%+.0f$ \\\\" % (100 * (d[-1] - d[0]) / d[0]))
    write("tab_windows.tex", "\\begin{tabular}{lccccc}\n"
          "B$-$A in the share within $\\delta$ & 250 & 500 & 1000 & 2000 & change (\\%) \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\end{tabular}\n")

    say()
    say("IV.F THE INSULATING PAIR")
    pv = {}                    # plot_tg_mhd_ladder.load_all keeps A, B and C only
    for p in sorted(glob.glob(os.path.join(res("round2"), "*", "series.dat"))):
        r = T.load(p)
        if r is not None and r["setup"] in ("A'", "B'"):
            pv.setdefault((r["setup"], r["Re"]), {})[r["N"]] = r
    for Re in (500, 1000):
        for s in ("A'", "B'"):
            by_n = pv.get((s, Re), {})
            if not by_n:
                continue
            r = by_n[max(by_n)]
            F, sp = T.integrated(r, *DEFAULT)
            pk = T.turbulent_peak(r)
            say("  %s Re %4d N %d: F1 %.4f (visc %.4f, ohm %.4f, Ohmic %.0f %%); peak: %s" % (
                s, Re, r["N"], F[0], sp[0], sp[1], 100 * sp[1] / F[0],
                "t %.2f" % pk[0] if isinstance(pk, tuple) else pk))

    # ---- Sec. IV.C: the field snapshots (GPU/csf3/tg_mhd_snap.sub) -----------------
    # The plane means come from slices/index.json, which is tracked; the .f32 planes
    # themselves are not, and are needed only to redraw the figures.
    say()
    say("IV.C THE FIELDS (results/P_tg_mhd/snap, Re 1000, N 512)")
    snap, prod = runs_of("snap"), {"A": RUNS[("A", 1000)][512], "B": RUNS[("B", 1000)][512],
                                   "C": RUNS[("C", 1000)][512]}
    for s in ("A", "B", "C"):
        a, b = snap[(s, 1000)][512], prod[s]
        tb = {round(r[0], 4): r for r in b["rows"]}
        common = [(r, tb[round(r[0], 4)]) for r in a["rows"][:-1] if round(r[0], 4) in tb]
        worst = max(abs(x[T.IX[c]] - y[T.IX[c]]) / max(abs(y[T.IX[c]]), 1e-300)
                    for x, y in common for c in ("eps", "fw1", "E_V", "E_M"))
        say("  %s: the snapshot run against the production run, %d probe rows to t = %.1f: worst %.1e"
            % (s, len(common), common[-1][0][0], worst))
    idx = {}
    for s in ("A", "B", "C"):
        with open(os.path.join(res("snap"), "%s_re1000_n512" % s, "slices", "index.json")) as f:
            idx[s] = json.load(f)

    def plane(s, name, t):
        return min((r for r in idx[s] if r["plane"] == name), key=lambda r: abs(r["t"] - t))
    sec, wal = "y128", "z3"
    r0 = plane("A", sec, 2.3)
    say("  planes: section %s at y = %.4f pi; wall plane %s at d = %.2f delta (delta = %.2f cells)"
        % (sec, r0["index"] / (r0["N"] - 1), wal, plane("A", wal, 2.3)["d_over_delta"],
           r0["delta"] / r0["h"]))
    for t in (2.3, 4.6):
        say("  t = %.2f (box mean eps: %s)" % (plane("A", sec, t)["t"], ", ".join(
            "%s %.4e" % (s, plane(s, sec, t)["eps_mean"]) for s in ("A", "B", "C"))))
        for name, what in ((sec, "section"), (wal, "wall plane")):
            say("    %-10s / box mean: %s" % (what, "   ".join(
                "%s visc %.2f ohm %.2f" % (s, plane(s, name, t)["mean"]["visc"] / plane(s, name, t)["eps_mean"],
                                         plane(s, name, t)["mean"]["ohm"] / plane(s, name, t)["eps_mean"])
                for s in ("A", "B", "C"))))
        a, b = plane("A", wal, t)["mean"], plane("B", wal, t)["mean"]
        say("    wall plane, absolute B/A: viscous %.2f, Ohmic %.2f" % (b["visc"] / a["visc"], b["ohm"] / a["ohm"]))
        c = plane("C", wal, t)
        bb = plane("B", wal, t)
        say("    wall plane, C's total over B's total (each over its own box mean): %.1f" % (
            (c["mean"]["visc"] + c["mean"]["ohm"]) / c["eps_mean"]
            / ((bb["mean"]["visc"] + bb["mean"]["ohm"]) / bb["eps_mean"])))
    write("numbers.txt", "\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
