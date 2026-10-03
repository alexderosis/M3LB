#!/usr/bin/env python3
"""Every number the paper quotes from the runs, and its four tables.

    python3 paper/tg_mhd/numbers.py

Reads results/P_tg_mhd/{round2,ladder,mach,fp64check} through
tools/tg_mhd_ladder.py and tools/plot_tg_mhd_ladder.py -- so the peak, f_w and
the integrated share are exactly those of the figures -- prints the quantities
the text cites, section by section, and writes tab_runs.tex, tab_verify.tex,
tab_results.tex, tab_windows.tex, tab_layers.tex and tab_protocol.tex beside this
file for main.tex and supplementary.tex to \\input.
A number in the text that this script does not print is a number to distrust.
Pure stdlib, like the tools it imports.
"""
import datetime
import glob
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import tg_mhd_ladder as T          # noqa: E402
import plot_tg_mhd_ladder as P     # noqa: E402  (matplotlib is imported only by its main)
import tg_mhd_budget as BU         # noqa: E402
import plot_tg_mhd_budget as PB    # noqa: E402  (likewise)
import tg_mhd_controls as CT       # noqa: E402

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
    hours = {}
    for d in ("ladder", "mach", "round2", "fp64check", "snap", "ladder_count", "budget"):
        tot = 0.0
        for lt in glob.glob(os.path.join(res(d), "*", "log.txt")):
            with open(lt) as f:
                got = [float(x.split()[0]) for x in f if x.strip().endswith("probes included)")]
            tot += got[-1]
        hours[d] = tot / 3600
    say("  GPU time (each log.txt's wall clock): ladder %.1f + Mach %.1f + round 2 %.1f + FP64 %.1f = %.1f h; "
        "snapshot reruns %.1f h; the budget's reruns %.1f h; the first, node-counting ladder %.1f h" % (
            hours["ladder"], hours["mach"], hours["round2"], hours["fp64check"],
            hours["ladder"] + hours["mach"] + hours["round2"] + hours["fp64check"], hours["snap"],
            hours["budget"], hours["ladder_count"]))
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
    for N in (65, 97):
        r = T.load(res("tgc_slip_cond_n%d_re1000/series.dat" % N))
        em, t = T.col(r, "EM/EV"), T.col(r, "t")
        k = min(range(1, len(em)), key=lambda i: em[i])
        say("  C2 under-resolved (A, Re 1000, N %d): min E_M/E_V %.4f at t = %.2f%s"
            % (N, em[k], t[k], " (the last probe: still falling)" if k == len(em) - 1 else ""))
    # Pouquet et al. (2010), Table 1 and Sec. 4.3, read from the paper on 2026-10-02.
    say("  REFERENCE: C2 of Pouquet et al. 2010 is pseudo-spectral on an equivalent 128^3 over 2 pi "
        "(64 points across [0, pi]); Table 1: E_M/E_V min 0.35, Om_M/Om_V first max 2., delta k_max 0.8 "
        "(I1 1.9, A2 2.1 at 128^3; their 2048^3 runs 2.2-3, 'well resolved ... since delta k_max >= 2'; "
        "an ideal run 'has to be stopped once delta ~ 3/N', i.e. delta k_max ~ 1, k_max = N/3)")
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
    dev, bdev, jobs = [], [], glob.glob(os.path.join(res("ladder"), "tg_mhd_verify-*.log"))
    for lg in jobs:
        with open(lg) as f:
            for x in f:
                if "worst column" in x:
                    dev.append((float(x.split("divb/j")[1].split()[0]), x.rstrip().endswith("PASS")))
                elif "energy budget worst term" in x:
                    bdev.append((float(x.split("worst term")[1].split()[1]), x.rstrip().endswith("PASS")))
    say("  device against the Kokkos parent (GPU/csf3/tg_mhd_verify.sub): %d jobs, %d checks, %d PASS; "
        "worst column %.1e of its scale" % (len(jobs), len(dev), sum(ok for _, ok in dev), max(v for v, _ in dev)))
    say("  the device's energy budget against the host's (since 2026-10-02): %d checks, %d PASS; worst term"
        " %.1e of its scale" % (len(bdev), sum(ok for _, ok in bdev), max(v for v, _ in bdev)))
    say("  RECORDED: validation/mhd_parity_wall.cpp -- the box against the periodic box it mirrors, both parities, "
        "1e-14 node for node with the transient; B.n on the wall nodes 1e-32; the conducting-wall eigenmode decays at "
        "order 2.00 and 2.00 over N = 17, 33, 65")
    say("  RECORDED: results/P_tg_mhd/xcheck_n65_re200 -- the CUDA host build against the Kokkos driver, FP64, "
        "free and no slip, N = 65, 815 steps: every printed digit")
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
    say("  A's turbulent peak at Re 1000 on N = %s: t = %s" % (
        " / ".join(str(n) for n in sorted(RUNS[("A", 1000)])),
        " / ".join("%.2f" % T.turbulent_peak(RUNS[("A", 1000)][n])[0] for n in sorted(RUNS[("A", 1000)]))))

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
        steps = ["%+.4f vs %.4f (%s)" % (d[i + 1] - d[i], bands[i] + bands[i + 1],
                                         "out" if abs(d[i + 1] - d[i]) > bands[i] + bands[i + 1] else "in")
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

    # The same readings for the layers 2 delta and 4 delta thick. A band here is the
    # two grid differences of that layer's share, added, as for delta.
    say()
    say("IV.E' THE LAYER: B-A within m delta, m = 1, 2, 4 (band: both grid differences, added)")
    rows, signs = [], []

    def ba_peak(Re):
        got = []
        for n in sorted(RUNS[("A", Re)])[-2:]:
            fa, fb = peak_fw(RUNS[("A", Re)][n])[1], peak_fw(RUNS[("B", Re)][n])[1]
            got.append(([y - x for x, y in zip(fa, fb)], fa, fb))
        (_, fa0, fb0), (d, fa1, fb1) = got
        return d, [abs(a1 - a0) + abs(b1 - b0) for a0, a1, b0, b1 in zip(fa0, fa1, fb0, fb1)]

    def ba_window(Re, w):
        a, b = P.integrated_fine(RUNS, ("A", Re), *w), P.integrated_fine(RUNS, ("B", Re), *w)
        ga = [T.integrated(RUNS[("A", Re)][n], *w)[0] for n in sorted(RUNS[("A", Re)])[-2:]]
        gb = [T.integrated(RUNS[("B", Re)][n], *w)[0] for n in sorted(RUNS[("B", Re)])[-2:]]
        return ([y - x for x, y in zip(a[0], b[0])],
                [abs(ga[1][m] - ga[0][m]) + abs(gb[1][m] - gb[0][m]) for m in range(3)])

    for label, tex, get in [("at the peak", "at the peak (pre-registered)", ba_peak)] + [
            ("t = %g-%g" % w, "integrated, $t=%g$--$%g$" % w, (lambda Re, w=w: ba_window(Re, w))) for w in WINDOWS]:
        got = [get(Re) for Re in RUNGS]
        for m, mm in enumerate((1, 2, 4)):
            d = [g[0][m] for g in got]
            bands = [g[1][m] for g in got]
            signs.extend(d)
            say("  %-12s m = %d: %s  (250 -> 2000 %+.0f %%; largest band %.4f)" % (
                label, mm, " / ".join("%+.4f" % x for x in d), 100 * (d[-1] - d[0]) / d[0], max(bands)))
            rows.append("%s & %d & %s & $%+.0f$ \\\\" % (tex if m == 0 else "", mm, " & ".join(
                "%.3f" % x for x in d), 100 * (d[-1] - d[0]) / d[0]))
    say("  every reading: B-A in [%+.4f, %+.4f]" % (min(signs), max(signs)))
    write("tab_layers.tex", "\\begin{tabular}{lcccccc}\n"
          "reading & $m$ & 250 & 500 & 1000 & 2000 & change (\\%) \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\end{tabular}\n")

    say()
    say("IV.G THE INSULATING PAIR")
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

    for Re in (500, 1000):
        f = {x: T.integrated(pv[(x, Re)][max(pv[(x, Re)])], *DEFAULT)[0][0] for x in ("A'", "B'")}
        say("  B' - A' in F1, Re %4d: %+.4f" % (Re, f["B'"] - f["A'"]))
    gw = 0.0
    for Re in (500, 1000):
        by_n = pv[("A'", Re)]
        (pc, fc), (pf, ff) = (peak_fw(by_n[n]) for n in sorted(by_n))
        gw = max([gw] + [pct(a, b) for a, b in zip(fc, ff)])
    say("  A' gate, Re 500 and 1000: worst %.2f %%" % gw)
    for Re in (500, 1000):
        by_n = pv.get(("B'", Re), {})
        if len(by_n) == 2:
            f = [T.integrated(by_n[n], *DEFAULT)[0][0] for n in sorted(by_n)]
            say("  B' Re %4d: F1 %.4f / %.4f on N = %d / %d, %.1f %% apart" % (
                Re, f[0], f[1], sorted(by_n)[0], sorted(by_n)[1], pct(f[0], f[1])))

    # ---- Sec. V: the scales the discussion compares ---------------------------------
    # Kato's layer is nu = 1/Re thick; a Hartmann layer is sqrt(nu eta)/|B_n| = 1/(Re |B_n|)
    # at Pm = 1. B_I is normal to every face at t = 0, with |B_n| at most 2 b0 on the
    # z faces and b0 on the x and y faces (Eq. bi); B_C has B.n = 0 on every face.
    say()
    say("V. SCALES (finer grid of each rung; h = pi/(N-1))")
    for Re in RUNGS:
        h = math.pi / (GRIDS[Re][1] - 1)
        say("  Re %4d: Kato's layer 1/Re = %.4f = %.2f h = %.3f delta" % (Re, 1 / Re, 1 / (Re * h), 1 / math.sqrt(Re)))
    b0 = 1 / math.sqrt(3)
    for Re in (500, 1000):
        h = math.pi / (GRIDS[Re][1] - 1)
        say("  B' Re %4d, t = 0: Hartmann layer 1/(Re |B_n|) = %.2f h on the z faces (|B_n| <= 2 b0 = %.3f), "
            "%.2f h on the x, y faces (b0)" % (Re, 1 / (Re * 2 * b0 * h), 2 * b0, 1 / (Re * b0 * h)))
    vol = [1 - (1 - 2 / (math.sqrt(Re) * math.pi)) ** 3 for Re in RUNGS]
    say("  volume within delta of the walls: %s of the box; slope %+.3f over Re 250 -> 2000"
        % (" / ".join("%.4f" % v for v in vol), math.log(vol[-1] / vol[0]) / math.log(8)))

    # ---- Sec. IV.H: the energy budget (GPU/csf3/tg_mhd_budget.sub, Phase 1) --------
    # The budget runs are round 2's runs again, with budget.dat; the test is the one
    # fixed in that job's header, computed by tools/tg_mhd_budget.py.
    say()
    say("IV.H THE BUDGET (results/P_tg_mhd/budget; within delta, t = %g-%g)" % DEFAULT)
    bud = BU.find([res("budget")])
    same = 0
    for (s_, Re), by_n in sorted(bud.items()):
        for N in by_n:
            d = res("budget/%s_re%d_n%d" % (s_, Re, N))
            a = [l for l in open(os.path.join(d, "series.dat")) if not l.startswith("#")]
            b = [l for l in open(res("round2/%s_re%d_n%d/series.dat" % (s_, Re, N))) if not l.startswith("#")]
            same += a == b
    say("  %d runs; series.dat identical to round 2's in %d" % (sum(len(v) for v in bud.values()), same))
    rows, calls = [], []
    for Re in RUNGS:
        Ns = sorted(set(bud[("A", Re)]) & set(bud[("B", Re)]))
        L = {N: (BU.layer(bud[("A", Re)][N], *DEFAULT, 1.0), BU.layer(bud[("B", Re)][N], *DEFAULT, 1.0))
             for N in Ns}
        ph = {N: BU.phi_of(*L[N]) for N in Ns}
        la, lb = L[Ns[-1]]
        band = abs(ph[Ns[-1]] - ph[Ns[-2]])
        dO = la["ohm"] - lb["ohm"]
        tr, st = (la["Tres"] - lb["Tres"]) / dO, -(la["deM"] - lb["deM"]) / dO
        call = "stretching" if ph[Ns[-1]] - band > 0.5 else "transport" if ph[Ns[-1]] + band < 0.5 else "undecided"
        calls.append(call)
        sgn = all(L[N][0]["S"] > 0 and L[N][1]["S"] < 0 for N in Ns)
        lsg = all(L[N][0]["lor"] < 0 and L[N][1]["lor"] > 0 for N in Ns)
        say("  Re %4d N %d/%d: phi %.3f / %.3f, band %.3f -> %s; parts stretching %+.3f transport %+.3f"
            " storage %+.3f; Ohmic B/A %.3f; S_A/Omega_A %.2f, S_B/Omega_B %+.2f; S_A > 0 > S_B on both grids: %s;"
            " lor_A < 0 < lor_B on both grids: %s" % (
                Re, Ns[0], Ns[1], ph[Ns[-1]], ph[Ns[0]], band, call, ph[Ns[-1]], tr, st, lb["ohm"] / la["ohm"],
                la["S"] / la["ohm"], lb["S"] / lb["ohm"], sgn, lsg))
        cl = [abs(x["Tres"] - x["Texp"]) / x["ohm"] for x in (la, lb)]
        ck = [abs(x["Kres"] - x["Kexp"]) / x["visc"] for x in (la, lb)]
        say("           closure within delta, finer grid: magnetic A %.1f %% B %.1f %%, kinetic A %.1f %% B %.1f %%"
            % (100 * cl[0], 100 * cl[1], 100 * ck[0], 100 * ck[1]))
        v = lb["visc"]
        say("           B's viscous dissipation within delta: viscous transport %.2f, Lorentz work %.2f,"
            " pressure work %.2f, advection %.2f, its kinetic energy %.2f" % (
                lb["difK"] / v, lb["lor"] / v, lb["pres"] / v, lb["advK"] / v, -lb["deK"] / v))
        rows.append("%d & %d, %d & %.2f & %.2f & %.3f & $%+.2f$ & $%+.2f$ & %.2f & $%+.2f$ & %.1f, %.1f \\\\" % (
            Re, Ns[0], Ns[1], ph[Ns[-1]], ph[Ns[0]], band, tr, st, la["S"] / la["ohm"], lb["S"] / lb["ohm"],
            100 * cl[0], 100 * cl[1]))
    say("  verdict: %s" % ("STRETCHING" if all(c == "stretching" for c in calls) else
                            "TRANSPORT" if all(c == "transport" for c in calls) else "UNDECIDED"))
    bx = []
    for (s_, Re), by_n in sorted(bud.items()):
        for N, b in by_n.items():
            x = BU.layer(b, *DEFAULT, None)
            loss = x["visc"] + x["ohm"]
            bx.append((abs(x["Tres"] - x["Texp"]) / loss, abs(x["Kres"] - x["Kexp"]) / loss,
                       abs(x["S"] + x["lor"]) / loss))
    for Re in (1000,):
        prof = {}
        for s_ in ("A", "B"):
            b = bud[(s_, Re)][max(bud[(s_, Re)])]
            e = PB.box_eps(b, *DEFAULT)
            for term in ("ohm", "S"):
                x, y = PB.layer_density(b, term, *DEFAULT)
                prof[(s_, term)] = (x, [v / e for v in y])
        (xa, oa), (xb, ob) = prof[("A", "ohm")], prof[("B", "ohm")]
        (_, sa), (_, sb) = prof[("A", "S")], prof[("B", "S")]
        ka, kb = max(range(len(oa)), key=lambda k: oa[k]), max(range(len(ob)), key=lambda k: ob[k])
        km = min(range(len(sb)), key=lambda k: sb[k])
        kz = next(k for k in range(km, len(sb)) if sb[k] > 0)
        z = xb[kz - 1] + (xb[kz] - xb[kz - 1]) * (-sb[kz - 1]) / (sb[kz] - sb[kz - 1])
        ka0 = next(k for k in range(len(sa)) if sa[k] <= 0)
        say("  Fig. budget (a), Re %d, finer grid, per unit volume / box-mean eps: at the wall node A S %.2f ohm %.2f,"
            " B S %+.3f ohm %.2f; ohm peaks at d = %.1f delta (A, %.2f) and %.1f delta (B, %.2f); S_B's minimum %+.3f"
            " at %.1f delta, crossing zero at %.1f delta; S_A first reaches zero at %.1f delta"
            % (Re, sa[0], oa[0], sb[0], ob[0], xa[ka], oa[ka], xb[kb], ob[kb], sb[km], xb[km], z, xa[ka0]))
    say("  RECORDED: previews at Re = 200 (laptop, 2026-10-02, not rungs), magnetic closure within delta over"
        " t = 2-10: A 14.7 % (N 65) -> 3.1 % (N 129), B 11.1 % -> 1.7 %; the N 65 preview's phi 2.47")
    say("  box over the window, every run: magnetic closure <= %.1f %%, kinetic <= %.1f %%, |<S> + <lor>| <= %.1f %%"
        " of int eps" % tuple(100 * max(c[i] for c in bx) for i in range(3)))
    write("tab_budget.tex", "\\begin{tabular}{rccccccccc}\n"
          "$\\Rey$ & $N$ & $\\phi$ & $\\phi_{\\mathrm{coarse}}$ & band & transport & storage"
          " & $S_A/\\Omega_A$ & $S_B/\\Omega_B$ & closure (\\%) \\\\\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n")

    # ---- Sec. III': the controls (GPU/csf3/tg_mhd_controls.sub, Phase 2) -----------
    # Each control is judged by the rule fixed in that job's header; the numbers are
    # tools/tg_mhd_controls.py's, read here through its own helpers.
    say()
    say("III'. THE CONTROLS (results/P_tg_mhd/controls; within delta, t = %g-%g)" % DEFAULT)
    ctl = lambda s_, Re, N, lab: CT.run_at(res("controls/%s_re%d_n%d_%s" % (s_, Re, N, lab)))
    r2 = lambda s_, Re, N: CT.run_at(res("round2/%s_re%d_n%d" % (s_, Re, N)))
    bd = lambda s_, Re, N: CT.run_at(res("budget/%s_re%d_n%d" % (s_, Re, N)))
    F1 = lambda r: CT.F(r, DEFAULT)[0]
    def band2(Re):
        nc, nf = GRIDS[Re]
        return abs(F1(r2("A", Re, nf)) - F1(r2("A", Re, nc))) + abs(F1(r2("B", Re, nf)) - F1(r2("B", Re, nc)))
    trows, c1 = [], []
    for Re in (500, 2000):
        nf = GRIDS[Re][1]
        fa = F1(r2("A", Re, nf))
        d0 = F1(r2("B", Re, nf)) - fa
        for lab in ("ramp0.2", "ramp1.0"):
            d = F1(ctl("B", Re, nf, lab)) - fa
            c1.append(abs(d - d0) / d0 < 0.10 and d > band2(Re))
            say("  C1 start, Re %4d %s: B - A %.4f against %.4f impulsive, change %+.1f %%" % (
                Re, lab, d, d0, 100 * (d - d0) / d0))
            trows.append("C1, start & %d & %s & B$-$A & %.4f & %.4f & $%+.1f$\\%% \\\\" % (
                Re, lab.replace("ramp", "ramp "), d0, d, 100 * (d - d0) / d0))
    say("  C1 verdict: %s" % ("not an imprint of the start" if all(c1) else "the start is a factor"))
    c2 = []
    for Re in RUNGS:
        nc, nf = GRIDS[Re]
        fc, fcc = F1(ctl("C", Re, nf, "vamp")), F1(ctl("C", Re, nc, "vamp"))
        fb, fbc = F1(r2("B", Re, nf)), F1(r2("B", Re, nc))
        band = abs(fc - fcc) + abs(fb - fbc)
        c2.append(fc - fb > band)
        pk = T.turbulent_peak(ctl("C", Re, nf, "vamp"))
        say("  C2 field at matched energy, Re %4d: F1(C) %.4f, F1(B) %.4f, ratio %.2f, C - B %.4f (bands %.4f);"
            " turbulent peak %s" % (Re, fc, fb, fc / fb, fc - fb, band, "t %.2f" % pk[0] if isinstance(pk, tuple)
                                     else "none"))
        trows.append("C2, field & %d & C at $E_V(0)=\\frac14$ & $F_1$ of B, of C & %.4f & %.4f & \\\\" % (Re, fb, fc))
    say("  C2 verdict: %s" % ("holds at every rung" if all(c2) else "not at every rung"))
    a0, b0 = r2("A", 2000, 640), r2("B", 2000, 640)
    a1, b1 = ctl("A", 2000, 640, "u0p04"), ctl("B", 2000, 640, "u0p04")
    d0, d1 = F1(b0) - F1(a0), F1(b1) - F1(a1)
    p0 = CT.phi(bd("A", 2000, 640), bd("B", 2000, 640), DEFAULT)
    p0c = CT.phi(bd("A", 2000, 512), bd("B", 2000, 512), DEFAULT)
    p1 = CT.phi(a1, b1, DEFAULT)
    say("  C3 Mach, Re 2000 N 640, u0 0.05 -> 0.04: B - A %.4f -> %.4f (shift %+.5f, band %.4f); phi %.3f -> %.3f"
        " (shift %+.3f, band %.3f); peak Ma %.3f -> %.3f" % (
            d0, d1, d1 - d0, band2(2000), p0, p1, p1 - p0, abs(p0 - p0c),
            max(max(T.col(r, "umax_lat")) for r in (a0, b0)) * math.sqrt(3.0),
            max(max(T.col(r, "umax_lat")) for r in (a1, b1)) * math.sqrt(3.0)))
    trows.append("C3, Mach & 2000 & $u_0=0.04$ & B$-$A & %.4f & %.4f & $%+.5f$ \\\\" % (d0, d1, d1 - d0))
    trows.append("C3, Mach & 2000 & $u_0=0.04$ & $\\phi$ & %.3f & %.3f & $%+.3f$ \\\\" % (p0, p1, p1 - p0))
    for pm, (nc, nf) in (("pm0.5", (384, 512)), ("pm2", (512, 640))):
        a, ac, b, bc = ctl("A", 1000, nf, pm), ctl("A", 1000, nc, pm), ctl("B", 1000, nf, pm), ctl("B", 1000, nc, pm)
        (fa, pa), (fb, pb) = CT.F(a, DEFAULT), CT.F(b, DEFAULT)
        band = abs(fa - F1(ac)) + abs(fb - F1(bc))
        ph, phc = CT.phi(a, b, DEFAULT), CT.phi(ac, bc, DEFAULT)
        say("  C4 %s, Re 1000 N %d/%d: B - A %.4f (band %.4f), viscous %+.4f, Ohmic %+.4f; phi %.3f (band %.3f)" % (
            pm, nc, nf, fb - fa, band, pb[0] - pa[0], pb[1] - pa[1], ph, abs(ph - phc)))
        trows.append("C4, Pm & 1000 & Pm $=%s$ & B$-$A & %.4f & %.4f & \\\\" % (
            pm[2:], F1(r2("B", 1000, 512)) - F1(r2("A", 1000, 512)), fb - fa))
        trows.append("C4, Pm & 1000 & Pm $=%s$ & $\\phi$ & %.3f & %.3f & \\\\" % (
            pm[2:], CT.phi(bd("A", 1000, 512), bd("B", 1000, 512), DEFAULT), ph))
    write("tab_controls.tex", "\\begin{tabular}{lcccccc}\n"
          "control & $\\Rey$ & change & quantity & reference & control & shift \\\\\n\\hline\n"
          + "\n".join(trows) + "\n\\end{tabular}\n")

    # ---- Sec. IV.F: the field snapshots (GPU/csf3/tg_mhd_snap.sub) -----------------
    # The plane means come from slices/index.json, which is tracked; the .f32 planes
    # themselves are not, and are needed only to redraw the figures.
    say()
    say("IV.F THE FIELDS (results/P_tg_mhd/snap, Re 1000, N 512)")
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

    # ---- Sec. IV F: the 3-D views (GPU/csf3/tg_mhd_vol.sub) ------------------------
    say()
    say("IV.F THE 3-D VIEWS (results/P_tg_mhd/snap/*/vol, block maximum over 2^3 -> 256^3)")
    for s in ("A", "B", "C"):
        with open(os.path.join(res("snap"), "%s_re1000_n512" % s, "vol", "index.json")) as f:
            v = json.load(f)
        for t in sorted({r["t"] for r in v}):
            rr = [r for r in v if r["t"] == t]
            tot = sum(r["box_mean"] for r in rr)
            say("  %s t = %.2f: M = %d, box mean of nu|w|^2 + eta|j|^2 %.5e against series eps %.5e"
                " (interpolated between probes), rel %.1e" % (s, t, rr[0]["M"], tot, rr[0]["eps_mean"],
                                                              abs(tot - rr[0]["eps_mean"]) / rr[0]["eps_mean"]))
    # Recorded, not regenerable from tracked data: these checks read raw dumps and
    # .f32 planes, which git does not keep. Measured 2026-10-02.
    say("  RECORDED: tools/tg_mhd_slices.py --check, N = 65 Re = 200, A B C at t = 0 and 2: "
        "box mean against series eps <= 1.7e-8 relative")
    say("  RECORDED: tools/tg_mhd_vol.cpp, the same six dumps (each on a probe): <= 3.3e-9 relative, gated")
    say("  RECORDED: the 256^3 volumes against the slice planes, 80 (volume, plane) pairs at t = 2.3 and 4.6: "
        "no plane's 2x2 block maximum exceeds its volume's, and they are equal bit for bit where the plane holds it")
    # ---- Supplementary S3: the first ladder, which counted whole node layers -----------
    # results/P_tg_mhd/ladder_count keeps those series. tg_mhd_ladder.py moves a counted
    # series to the exact band before using it (band_corrected); the gate as it failed
    # read the columns as written, which is what is computed here.
    say()
    say("S3. THE FIRST LADDER, COUNTING NODE LAYERS (results/P_tg_mhd/ladder_count, raw columns)")
    cnt = {}
    for p in sorted(glob.glob(os.path.join(res("ladder_count"), "*", "series.dat"))):
        r = T.load(p)
        cnt.setdefault((r["setup"], r["Re"]), {})[r["N"]] = r

    def raw_fw(run, t):
        return T.lerp([(x[0], [x[T.IX[m]] for m in T.FW]) for x in run["rows"]], t)
    worst, f1, ep, nfail, fixed = [], [], [], 0, []
    for (s_, Re), by_n in sorted(cnt.items()):
        nc, nf = sorted(by_n)
        pc, pf = T.turbulent_peak(by_n[nc]), T.turbulent_peak(by_n[nf])
        if not (isinstance(pc, tuple) and isinstance(pf, tuple)):
            continue
        d = [pct(a, b) for a, b in zip(raw_fw(by_n[nc], pc[0]), raw_fw(by_n[nf], pf[0]))]
        worst.append(max(d)); f1.append(d[0]); ep.append(pct(pc[1], pf[1])); nfail += max(d) > 5
        fixed.append(max(pct(a, b) for a, b in zip(T.fw_at(by_n[nc], pc[0]), T.fw_at(by_n[nf], pf[0]))))
        say("  %s Re %4d: f1/f2/f4 %.1f/%.1f/%.1f %% apart; eps at the peak %.2f %%" % (s_, Re, d[0], d[1], d[2], ep[-1]))
    for N in GRIDS[1000]:
        r = (N - 1) / (math.pi * math.sqrt(1000))
        say("  Re 1000, N %d: counting the nodes with k h < delta measures (ceil(delta/h) - 1/2) h = %.2f delta"
            % (N, (math.ceil(r) - 0.5) / r))
    say("  %d of %d (box, Re) fail the 5 %% gate; worst of the three %.1f-%.1f %%, f1 %.1f-%.1f %%, eps at the peak <= %.1f %%"
        % (nfail, len(worst), min(worst), max(worst), min(f1), max(f1), max(ep)))
    say("  the same series moved to the exact band (tg_mhd_ladder.band_corrected): worst of the three %.1f %%, "
        "so all %d pass" % (max(fixed), len(fixed)))
    say("  RECORDED: the sweep N = 33..49 at Re = 100 (2026-09-28): f4 by the count jumps 0.52 -> 0.60 each time "
        "4 delta/h crosses a node; by Eq. (within) 0.561-0.566")

    # ---- Supplementary S2: when each rule reached GitHub ------------------------------
    # The commit time is the author's clock; the push time is GitHub's, from the
    # server-side log captured in provenance/ (its README says how and when).
    say()
    say("S2. THE PROTOCOL'S RECORD (git log; GitHub's push log, every provenance/github_activity_*.json)")
    log = []
    for cap in sorted(glob.glob(os.path.join(HERE, "provenance", "github_activity_*.json"))):
        with open(cap) as f:
            log += json.load(f)

    def utc(iso):
        return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(datetime.timezone.utc)

    rows = []
    for sha, plain, tex in (("663929c", "observable, peak rule, gate", "observable, peak rule, gate"),
                            ("9f65706", "estimator of f_m corrected", "estimator of $f_m$ corrected"),
                            ("cd07f45", "Mach control", "Mach control"),
                            ("d061ee8", "decision rule, FP64 control (Re 2000)", "decision rule, FP64 control ($\\Rey=2000$)"),
                            ("8376e66", "the budget test (phi)", "the budget test, $\\phi$"),
                            ("f7f55ba", "the four controls (Phase 2)", "the four controls")):
        got = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%cI", sha],
                             capture_output=True, text=True)
        committed = utc(got.stdout.strip())
        pushed = min(utc(a["timestamp"]) for a in log if a["after"].startswith(sha))
        lag = (pushed - committed).total_seconds()
        say("  %s %-38s committed %s UTC, pushed %s UTC (%+.0f s)" % (
            sha, plain, committed.strftime("%Y-%m-%d %H:%M:%S"), pushed.strftime("%Y-%m-%d %H:%M:%S"), lag))
        rows.append("%s & \\texttt{%s} & %s & %s & %.0f \\\\" % (
            tex, sha, committed.strftime("%d %b, %H:%M:%S"), pushed.strftime("%H:%M:%S"), lag))
    write("tab_protocol.tex", "\\begin{tabular}{lcccc}\n"
          "rule & commit & committed (UTC) & pushed (UTC) & lag (s) \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\end{tabular}\n")

    write("numbers.txt", "\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
