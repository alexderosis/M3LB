#!/usr/bin/env python3
"""Phase 3 of the confined-MHD paper: the forced, statistically steady boxes.

    tg_mhd_forced.py pilot PILOT_DIR [--round 1|2] [--window T0 T1] [--round2 DIR]
    tg_mhd_forced.py shares RUN_DIR... --window T0 T1 --batches NB
    tg_mhd_forced.py stretch RUNS_DIR... --window T0 T1 --batches NB [--registered COMMIT]
    tg_mhd_forced.py test RUNS_DIR --registered COMMIT

PILOT_DIR holds what a pilot job writes -- GPU/csf3/tg_mhd_pilot.sub (round 1,
runs/tg_mhd_pilot/ or results/P_tg_mhd/pilot/) or tg_mhd_pilot2.sub (round 2,
runs/tg_mhd_pilot2/ or results/P_tg_mhd/pilot2/): one
<A|B>_re<Re>_n<N>_f<F0>_m<FM>[_fp32]/ per run, each driven by the bulk
Taylor-Green force F0 (-drive) and the bulk TG-C source FM on the field
(-bdrive). The pilot exists to fix the forcing amplitudes, and the transient and
averaging scales, BEFORE the forced test is registered -- and to do it without
seeing the test's observable. So `pilot` reads only E_V, E_M, eps, umax_lat,
div B and the divergence flag of each series. It never reads f_w, the
wall-distance profiles or the budget, and prints nothing derived from them.

THE SELECTION RULES (stated and argued in each job's header, fixed before its
runs). Both take, among the pairs (F0, FM) that qualify, the one whose A at
Re = 1000 lies closest to that box's turbulent peak in the decaying ladder
(round 2's A, Re = 1000, N = 512), in the distance

    D = |ln(<eps> / eps_pk)| + |ln((<E_M>/<E_V>) / (E_M/E_V)_pk)|

with <.> the trapezoidal mean over the window. A pair qualifies when its four
runs -- A and B at Re = 250 and 1000 -- are finite to the window's end and its A
at Re = 1000 is unsteady (the r.m.s. of eps over the window above 1 % of its
mean), and
  ROUND 1 (window t = 25..50): umax_lat <= 0.08 over the WHOLE run; and if the
     closest pair sits on the scan's edge with D > 0.5, the pilot is
     INCONCLUSIVE and is extended. It was (2026-10-04).
  ROUND 2 (window t = 100..150): umax_lat <= 0.08 over the WINDOW, and every
     one of the four runs STATIONARY -- E_V and E_M averaged over the window's
     two halves differ by less than 25 % of their window means. The closest
     qualifying pair is taken whatever its D; none qualifying ends the pilot
     without a selection.
Runs at Re = 2000 (round 2's top-rung pair) gate nothing: they are reported, with
their stationarity, for the production's length.

Reported beside it, per run: the window means, the r.m.s. of eps, the integral
time of eps (its autocorrelation summed to the first zero), when the run settles
(the last time E_T was more than three r.m.s. from its window mean), the largest
umax_lat, and the largest div B/|j| in the window -- the field's source is
solenoidal analytically and only to O(h^2) on the nodes; and for the FP32 pair,
its means against FP64's.

`shares` is the machinery the forced test will be judged with, kept apart
from its rule until that is registered: per run, the share of the dissipation
within 1, 2 and 4 delta of the walls over [T0, T1] -- the ratio of time
integrals, read as the decaying ladder's integrated share is
(tg_mhd_ladder.integrated) -- with its standard error from NB batch means, the
lag-1 correlation of the batches (which should be small, or the batches are too
short) and the two halves' shares (stationarity).

THE STRETCHING PART OF THE FORCED TEST (`stretch`) -- REGISTERED 2026-11-17,
BEFORE THE S OF ANY FORCED RUN WAS READ.

Why. Over t = 2..10 of the decaying ladder the stretching S = B_i B_j d_j u_i
within delta of the no-slip conducting wall is negative at every rung, and the
paper first read that as "the no-slip wall reverses the stretching". Against
time the sign belongs to the window (tg_mhd_budget.py's POST HOC section; the
paper's Sec. IV G): it changes in both boxes, and over t = 5..10 S_B > 0 at
three rungs of four. A decaying flow cannot say whether a no-slip wall layer
takes energy from the field ON AVERAGE. A statistically steady one can, because
it has no "when" -- which is what the forced boxes are for.

The observable. For a forced run of box X (A free slip, B no slip; conducting
walls), with <.> the trapezoidal time integral over the production window
[T0, T1] of a row of budget.dat read within delta = Re^-1/2 of the walls, as
tg_mhd_budget.within reads it (m = 1),

    sigma_X = <S> / <ohm>:

the share of the layer's Ohmic dissipation that the stretching inside the layer
supplies. The forces vanish within 0.15 of every wall (tg_mhd's -gap default)
and delta <= 0.064 at Re >= 250, so the layer is unforced, and in a steady
state <ohm> = <S> + transport. sigma < 0 says the layer's field loses energy to
the flow as well as to Ohmic dissipation, transport supplying both.

Its uncertainty u_X is the largest of three, the last two on the finer grid:
  (a) the grid: |sigma(finer) - sigma(coarser)| on the rung's two grids;
  (b) the statistics: twice the standard error of sigma from NB equal batches
      of the window, each batch's sigma the ratio of its own two integrals;
  (c) the drift: |sigma(first half of the window) - sigma(second half)|.

The rule. Each question needs every rung of the forced ladder, which has at
least three Reynolds numbers with two grids each; the verdicts are read on the
finer grids.
  Q1, the sign at the no-slip wall:
      "TAKES"  if sigma_B + u_B < 0 at every rung -- a no-slip conducting wall
               layer takes energy from the field on average;
      "FEEDS"  if sigma_B - u_B > 0 at every rung -- it does not, and the
               reversal read in the decaying runs was the window's;
      "MIXED"  otherwise: reported rung by rung, and neither is claimed.
  Q2, the removal:
      "REMOVES"   if (sigma_A - sigma_B) - (u_A + u_B) > 0 at every rung -- the
                  stretching feeds the free-slip layer's Ohmic dissipation more
                  than the no-slip layer's;
      "NOT SHOWN" otherwise.
No trend in Re is tested: the decaying ladder showed that this plan cannot
carry one. sigma within 2 delta (0.127 at Re = 250, still unforced) is printed
beside the test and judges nothing. Only "TAKES" would be claimed as a
reversal, and a letter would be written on it; that is the reason the rule is
fixed first.

What is NOT fixed here, and must be, in the production job's header, before
its runs and from the pilot's E_V, E_M, eps and umax_lat alone: the forcing
pair (the pilot's own rule), the rungs and their grids, the window [T0, T1],
NB, and the production's stationarity test. Whoever fixes them must not have
read the S, the profiles or the f_w of any forced run. `stretch` refuses forced
runs unless --registered names the commit that fixed them.

What had been seen when this was written. Of the decaying runs, everything.
Of the forced runs: round 1 of the pilot (38 runs, CSF3, 2026-10-04) through
`pilot`, which reads E_V, E_M, eps, umax_lat and div B; of that round's
budget.dat files, the box mean of injM and nothing nearer a wall than the whole
box (GPU/csf3/tg_mhd_pilot2.sub says so); on 2026-11-17 the directory listing
and one series header with its t = 0 row, the initial condition. The driven
twin-check references at N = 65, Re = 200 (results/P_tg_mhd/xcheck_n65_re200/
*_drive0.1_bdrive0.1_*, 1.4 cells in delta, not rungs) had their budgets
compared between the two codes and with the periodic twin, and their box
closure read, on 2026-10-03; tg_mhd_budget.py's table has an S-within-delta
column, so a value may have been on screen then. None was recorded, and none
was read for this rule. Pilot round 2 had not been run.

THE FORCED TEST (`test`), REGISTERED 2026-11-19 IN GPU/csf3/tg_mhd_forced.sub,
whose header is the rule; the PROD_* constants below are its numbers, and a
difference between the two is a bug here. On the production's directory, and
only with --registered naming the commit that added that file, it
  1. gates every run on stationarity (finite to the window's end, umax_lat
     <= 0.08 over the window, E_V and E_M over the window's halves within 25 %
     of their means) and stops with NO VERDICT if any run fails -- before any
     f_w, profile or budget.dat is read;
  2. applies Q0, the wall share: Delta = F_B - F_A within delta on the finer
     grid of each rung, its uncertainty the largest of the grid difference,
     twice the batch standard error and the halves' difference; NO EXCESS if
     Delta - u <= 0 at any rung, else the exponent p = ln(Delta_hi/Delta_lo) /
     ln(Re_hi/Re_lo) with its range from Delta +- u at the ends, against -1/4:
     WALLS HOLD, CASCADE or UNDECIDED;
  3. applies Q1 and Q2 through `stretch` over the registered window and batches.

Exit status 0 when a pair was selected (pilot), every run was read (shares),
both verdicts were issued (stretch) or all three were (test), 1 otherwise, and
2 when `stretch` or `test` read nothing -- no runs, several runs for one box, Re
and N, runs that are not the production's, or forced runs without
--registered. Pure stdlib.
"""
import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tg_mhd_ladder as T     # noqa: E402
import tg_mhd_budget as BU    # noqa: E402

ROOT =os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
UMAX = 0.08                    # the Mach bound on umax_lat, Ma = 0.139
UNSTEADY = 0.01                # r.m.s./mean of eps above which a run is unsteady
EDGE_D = 0.5                   # round 1: a selection on the scan's edge further than this is no selection
STATIONARY = 0.25              # round 2: largest drift of E_V or E_M between the window's halves
# The production and its wall-share test, as GPU/csf3/tg_mhd_forced.sub registers them.
PROD_PAIR = (0.2, 0.1)                                     # (F0, FM), pilot round 2's selection
PROD_GRIDS = {250: (192, 256), 500: (256, 384), 1000: (384, 512)}   # Re: (coarser, finer)
PROD_WINDOW = (100.0, 250.0)
PROD_BATCHES = 10
PROD_THRESHOLD = -0.25                                     # halfway in ln between Re^-1/2 and Re^0
DIR_RE = re.compile(r"^(A|B)_re(\d+)_n(\d+)_f([0-9.]+)_m([0-9.]+)(_fp32)?$")


def load(d):
    """The run in directory d, with its pilot coordinates, or None."""
    m = DIR_RE.match(os.path.basename(os.path.normpath(d)))
    if not m:
        return None
    s = T.load(os.path.join(d, "series.dat"))
    if s is None:
        return None
    s.update(box=m.group(1), re=int(m.group(2)), n=int(m.group(3)), F0=float(m.group(4)),
             FM=float(m.group(5)), fp32=bool(m.group(6)))
    return s


def window_stats(run, t0, t1):
    """Means over [t0, t1] (trapezoidal), the r.m.s. of eps and its integral time,
    the settling time of E_T, the largest umax_lat over the run and over the
    window, and the drift of E_V and E_M between the window's two halves; or a
    string saying why there are none."""
    if run["diverged"]:
        return "diverged"
    t = T.col(run, "t")
    # Probes fall on whole steps, so a complete run's last one sits just short of
    # t1 (49.999987 at N = 256); the same 1e-2 as tg_mhd_ladder.integrated.
    if t[-1] < t1 - 1e-2:
        return "reached only t = %.2f" % t[-1]
    t1 = min(t1, t[-1])
    cols = ("E_V", "E_M", "E_T", "eps")
    q = [(r[0], [r[T.IX[c]] for c in cols]) for r in run["rows"]]
    if not all(math.isfinite(v) for _, vs in q for v in vs):
        return "non-finite"
    mean = [x / (t1 - t0) for x in T.trapz_window(q, t0, t1)]
    m = dict(zip(cols, mean))
    # samples in the window, equally spaced by the probe interval
    e = [r[T.IX["eps"]] for r in run["rows"] if t0 - 1e-9 <= r[0] <= t1 + 1e-9]
    et = [r[T.IX["E_T"]] for r in run["rows"] if t0 - 1e-9 <= r[0] <= t1 + 1e-9]
    dt = (t1 - t0) / max(len(e) - 1, 1)
    me = sum(e) / len(e)
    var = sum((x - me) ** 2 for x in e) / len(e)
    m["eps_rms"] = math.sqrt(var)
    # integral time: the autocorrelation of eps summed to its first zero
    tau = 0.5
    if var > 0:
        for k in range(1, len(e) // 2):
            rho = sum((e[i] - me) * (e[i + k] - me) for i in range(len(e) - k)) / ((len(e) - k) * var)
            if rho <= 0:
                break
            tau += rho
    m["tau_int"] = tau * dt
    # settling: the last probe before the window's end at which E_T sat more than
    # three r.m.s. away from its window mean
    mt = sum(et) / len(et)
    st = 3.0 * math.sqrt(sum((x - mt) ** 2 for x in et) / len(et))
    settle = t[0]
    for r in run["rows"]:
        if r[0] <= t1 and abs(r[T.IX["E_T"]] - mt) > st:
            settle = r[0]
    m["settle"] = settle
    m["umax"] = max(r[T.IX["umax_lat"]] for r in run["rows"])
    m["umax_win"] = max(r[T.IX["umax_lat"]] for r in run["rows"] if t0 - 1e-9 <= r[0] <= t1 + 1e-9)
    m["divb"] = max(r[T.IX["divb/j"]] for r in run["rows"] if t0 - 1e-9 <= r[0] <= t1 + 1e-9)
    # stationarity: each energy's mean over the window's two halves, as a share of
    # its window mean
    h = 0.5 * (t0 + t1)
    m1 = [x / (h - t0) for x in T.trapz_window(q, t0, h)]
    m2 = [x / (t1 - h) for x in T.trapz_window(q, h, t1)]
    m["drift_V"] = abs(m2[0] - m1[0]) / m["E_V"]
    m["drift_M"] = abs(m2[1] - m1[1]) / m["E_M"]
    return m


def target(round2):
    """eps, E_M/E_V and the time of round 2's A at Re = 1000 (N = 512), at its
    turbulent peak."""
    run = T.load(os.path.join(round2, "A_re1000_n512", "series.dat"))
    if run is None:
        raise SystemExit("no round-2 run at %s" % os.path.join(round2, "A_re1000_n512"))
    pk = T.turbulent_peak(run)
    if not isinstance(pk, tuple):
        raise SystemExit("round 2's A at Re = 1000 has no turbulent peak: %s" % pk)
    tp, ep = pk
    return ep, T.interp(run, "E_M", tp) / T.interp(run, "E_V", tp), tp


def pilot(args):
    rnd = args.round
    t0, t1 = args.window if args.window else ((25.0, 50.0) if rnd == 1 else (100.0, 150.0))
    eps_pk, ratio_pk, t_pk = target(args.round2)
    runs = []
    for name in sorted(os.listdir(args.pilot_dir)):
        r = load(os.path.join(args.pilot_dir, name))
        if r is not None:
            runs.append(r)
    if not runs:
        print("no pilot runs in %s" % args.pilot_dir)
        return 1
    print("Phase 3 pilot, round %d (GPU/csf3/%s): window t = %g..%g"
          % (rnd, "tg_mhd_pilot.sub" if rnd == 1 else "tg_mhd_pilot2.sub", t0, t1))
    print("target, round 2's A at Re = 1000 (N = 512) at its turbulent peak, t = %.2f: "
          "eps %.4f, E_M/E_V %.3f\n" % (t_pk, eps_pk, ratio_pk))
    print("  %-30s %8s %8s %8s %7s %8s %7s %7s %7s %7s %8s %13s" % (
        "run", "<E_V>", "<E_M>", "<eps>", "EM/EV", "eps rms", "t_int", "settle", "umax", "in win",
        "divb/j", "drift V / M"))
    stats = {}
    for r in runs:
        tag = "%s Re %4d N %d F0 %g FM %g%s" % (r["box"], r["re"], r["n"], r["F0"], r["FM"],
                                               " FP32" if r["fp32"] else "")
        st = window_stats(r, t0, t1)
        stats[(r["box"], r["re"], r["F0"], r["FM"], r["fp32"])] = st
        if isinstance(st, str):
            print("  %-30s %s" % (tag, st))
            continue
        print("  %-30s %8.4f %8.4f %8.4f %7.3f %7.1f%% %7.2f %7.2f %7.4f %7.4f %8.1e %5.1f%% / %4.1f%%" % (
            tag, st["E_V"], st["E_M"], st["eps"], st["E_M"] / st["E_V"], 100 * st["eps_rms"] / st["eps"],
            st["tau_int"], st["settle"], st["umax"], st["umax_win"], st["divb"],
            100 * st["drift_V"], 100 * st["drift_M"]))

    # ---- FP32 against FP64 at the same pair, where the pilot ran it ----------
    for (b, Re, f0, fm, p32), st in sorted(stats.items(), key=lambda kv: kv[0][:4]):
        if p32 and not isinstance(st, str):
            s64 = stats.get((b, Re, f0, fm, False))
            if isinstance(s64, dict):
                print("  FP32 %s Re %d F0 %g FM %g: <E_T> %+.2f %%, <eps> %+.2f %% against FP64"
                      " (whose eps r.m.s. is %.1f %%)" % (b, Re, f0, fm,
                                                         100 * (st["E_T"] / s64["E_T"] - 1),
                                                         100 * (st["eps"] / s64["eps"] - 1),
                                                         100 * s64["eps_rms"] / s64["eps"]))

    # ---- the top rung (round 2): reported, gating nothing ---------------------
    for (b, Re, f0, fm, p32), st in sorted(stats.items(), key=lambda kv: kv[0][:4]):
        if Re == 2000:
            if isinstance(st, str):
                print("  TOP RUNG %s Re 2000 F0 %g FM %g: %s" % (b, f0, fm, st))
            else:
                print("  TOP RUNG %s Re 2000 F0 %g FM %g: %s over the window (drift V %.1f %%, M %.1f %%),"
                      " settles by t = %.2f, umax_lat %.4f in the window"
                      % (b, f0, fm, "STATIONARY" if max(st["drift_V"], st["drift_M"]) < STATIONARY
                         else "NOT STATIONARY", 100 * st["drift_V"], 100 * st["drift_M"], st["settle"],
                         st["umax_win"]))

    # ---- the selection -------------------------------------------------------
    F0s = sorted({r["F0"] for r in runs if not r["fp32"] and r["re"] in (250, 1000)})
    FMs = sorted({r["FM"] for r in runs if not r["fp32"] and r["re"] in (250, 1000)})
    print("\nSELECTION over F0 = %s, FM = %s" % (", ".join("%g" % v for v in F0s),
                                                  ", ".join("%g" % v for v in FMs)))
    best = None
    for F0 in F0s:
        for FM in FMs:
            four = [stats.get((b, Re, F0, FM, False)) for b in ("A", "B") for Re in (250, 1000)]
            why = None
            if any(s is None for s in four):
                why = "a run is missing"
            elif any(isinstance(s, str) for s in four):
                why = "a run " + next(s for s in four if isinstance(s, str))
            elif rnd == 1 and any(s["umax"] > UMAX for s in four):
                why = "umax_lat %.4f > %g" % (max(s["umax"] for s in four), UMAX)
            elif rnd == 2 and any(s["umax_win"] > UMAX for s in four):
                why = "umax_lat %.4f > %g in the window" % (max(s["umax_win"] for s in four), UMAX)
            elif rnd == 2 and any(max(s["drift_V"], s["drift_M"]) >= STATIONARY for s in four):
                why = "not stationary (drift %.1f %%)" % (100 * max(max(s["drift_V"], s["drift_M"])
                                                                   for s in four))
            a = stats.get(("A", 1000, F0, FM, False))
            if why is None and a["eps_rms"] / a["eps"] <= UNSTEADY:
                why = "A at Re = 1000 is steady (eps r.m.s. %.2f %%)" % (100 * a["eps_rms"] / a["eps"])
            if why is not None:
                print("  F0 %-5g FM %-5g excluded: %s" % (F0, FM, why))
                continue
            D = abs(math.log(a["eps"] / eps_pk)) + abs(math.log(a["E_M"] / a["E_V"] / ratio_pk))
            print("  F0 %-5g FM %-5g A at Re = 1000: eps %.4f, E_M/E_V %.3f -> D = %.3f"
                  % (F0, FM, a["eps"], a["E_M"] / a["E_V"], D))
            if best is None or D < best[0]:
                best = (D, F0, FM)
    if best is None:
        print("  verdict: NO PAIR QUALIFIES; the pilot is inconclusive")
        return 1
    D, F0, FM = best
    edge = F0 in (F0s[0], F0s[-1]) or FM in (FMs[0], FMs[-1])
    if rnd == 1 and edge and D > EDGE_D:
        print("  verdict: INCONCLUSIVE -- the closest pair, F0 = %g, FM = %g, is on the scan's edge"
              " with D = %.3f > %g; extend the scan" % (F0, FM, D, EDGE_D))
        return 1
    print("  verdict: F0 = %g, FM = %g (D = %.3f%s)" % (F0, FM, D, ", on the scan's edge" if edge else ""))

    return 0


def integrand(run):
    """[(t, [eps, eps fw1, eps fw2, eps fw4, eps visc1, eps ohm1])]: what
    tg_mhd_ladder.integrated integrates, built once so that a run's batches do
    not reread its profiles; the last two are absent for a run without them."""
    fw, eps = T.fw_rows(run), T.col(run, "eps")
    ts = [t for t, _ in fw]
    parts = T.profile_parts(run)
    if parts is not None and (len(parts) != len(ts) or
                              any(abs(p[0] - t) > 1e-6 for p, t in zip(parts, ts))):
        parts = None
    return [(t, [e] + [x * e for x in f] + ([x * e for x in parts[i][1]] if parts else []))
            for i, ((t, f), e) in enumerate(zip(fw, eps))]


def shares_over(q, a, b):
    """[F1, F2, F4, viscous part of F1, Ohmic part of F1] over [a, b] (the parts
    only where the run wrote them): ratios of time integrals."""
    tot = T.trapz_window(q, a, b)
    return [x / tot[0] for x in tot[1:]]


def batch_means(run, t0, t1, nb):
    """The window's shares, F1's standard error from nb equal batches, the
    batches' lag-1 correlation, and F1 over each half of the window; or a string
    saying why there are none."""
    if run["diverged"]:
        return "DIVERGED"
    q = integrand(run)
    if q[0][0] > t0 + 1e-2 or q[-1][0] < t1 - 1e-2:
        return "run covers t = %.2f..%.2f only" % (q[0][0], q[-1][0])
    t1 = min(t1, q[-1][0])
    F = shares_over(q, t0, t1)
    L = (t1 - t0) / nb
    Fb = [shares_over(q, t0 + k * L, t0 + (k + 1) * L)[0] for k in range(nb)]
    mb = sum(Fb) / nb
    var = sum((x - mb) ** 2 for x in Fb) / (nb - 1)
    r1 = (sum((Fb[k] - mb) * (Fb[k + 1] - mb) for k in range(nb - 1)) / ((nb - 1) * var)
          if var > 0 else 0.0)
    h = 0.5 * (t0 + t1)
    return {"F": F, "se": math.sqrt(var / nb), "batches": Fb, "r1": r1,
            "halves": (shares_over(q, t0, h)[0], shares_over(q, h, t1)[0])}


def shares(args):
    t0, t1 = args.window
    rc = 0
    print("shares within 1, 2, 4 delta over t = %g..%g, F1's error from %d batch means"
          % (t0, t1, args.batches))
    for d in args.run_dirs:
        run = T.load(os.path.join(d, "series.dat"))
        if run is None:
            print("  %s: no series" % d); rc = 1
            continue
        bm = batch_means(run, t0, t1, args.batches)
        if isinstance(bm, str):
            print("  %s: %s" % (run["tag"], bm)); rc = 1
            continue
        F = bm["F"]
        parts = (" (viscous %.4f, Ohmic %.4f)" % (F[3], F[4])) if len(F) > 3 else ""
        print("  %-44s F1 %.4f +- %.4f%s  F2 %.4f  F4 %.4f  halves %.4f / %.4f  r1 %+.2f"
              % (run["tag"], F[0], bm["se"], parts, F[1], F[2], bm["halves"][0], bm["halves"][1],
                 bm["r1"]))
    return rc


def sigma_of(b, t0, t1, nb, m=1.0):
    """sigma = int S / int ohm within m delta over [t0, t1], twice its standard error
    from nb equal batches, the batches' lag-1 correlation and sigma over each half of
    the window; or a string saying why there is none. The rule is in the docstring."""
    r = m * b["dh"]
    q = [(t, [BU.within(cs, r), BU.within(co, r)])
         for t, cs, co in zip(b["t"], b["rows"]["S"], b["rows"]["ohm"])]
    if q[0][0] > t0 + 1e-2 or q[-1][0] < t1 - 1e-2:
        return "covers t = %.2f..%.2f only" % (q[0][0], q[-1][0])
    t1 = min(t1, q[-1][0])

    def sig(a, z):
        s, o = T.trapz_window(q, a, z)
        return s / o

    L = (t1 - t0) / nb
    sb = [sig(t0 + k * L, t0 + (k + 1) * L) for k in range(nb)]
    mb = sum(sb) / nb
    var = sum((x - mb) ** 2 for x in sb) / (nb - 1)
    r1 = (sum((sb[k] - mb) * (sb[k + 1] - mb) for k in range(nb - 1)) / ((nb - 1) * var)
          if var > 0 else 0.0)
    h = 0.5 * (t0 + t1)
    return {"sigma": sig(t0, t1), "se2": 2.0 * math.sqrt(var / nb), "r1": r1,
            "halves": (sig(t0, h), sig(h, t1))}


def stretch(args):
    """The registered stretching test (the docstring's rule), on the runs under
    args.runs_dirs: one directory per run, each with series.dat and budget.dat."""
    t0, t1 = args.window
    if args.batches < 2:
        print("stretch: --batches must be at least 2")
        return 2
    found = []
    for d in args.runs_dirs:
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name, "budget.dat")
            if not os.path.exists(p):
                continue
            s = T.load(os.path.join(d, name, "series.dat"))
            if s is not None and s["setup"] in ("A", "B"):
                found.append((s, p))
    if not found:
        print("stretch: no A or B run with a budget.dat under %s" % " ".join(args.runs_dirs))
        return 2
    # The guard comes BEFORE any budget.dat is opened: a forced run's S is not read,
    # not even into memory, until its window is registered.
    forced = sorted(s["tag"] for s, _ in found if "drive" in s["tag"])
    if forced and not args.registered:
        print("stretch: %d forced run(s) here (%s, ...). Their S is not to be read before the window,\n"
              "the batches, the rungs and the stationarity test are fixed and pushed -- from the pilot's\n"
              "E_V, E_M, eps and umax_lat alone. Then pass --registered COMMIT, the commit that fixed them."
              % (len(forced), forced[0]))
        return 2
    runs, clash = {}, []
    for s, p in found:
        b = BU.load_budget(p)
        if b is None:
            continue
        b["tag"] = s["tag"]
        if b["N"] in runs.setdefault((s["setup"], s["Re"]), {}):
            clash.append(s["tag"])
        runs[(s["setup"], s["Re"])][b["N"]] = b
    if not runs:
        print("stretch: no readable budget.dat under %s" % " ".join(args.runs_dirs))
        return 2
    if clash:
        print("stretch: several runs for one box, Re and N (%s, ...): give it the production's"
              " directory, one forcing pair" % clash[0])
        return 2

    print("THE STRETCHING PART OF THE FORCED TEST (rule registered 2026-11-17, in this file's docstring):")
    print("sigma = <S>/<ohm> within delta over t = %g..%g, %d batches%s" % (
        t0, t1, args.batches, "; window and batches registered in %s" % args.registered if args.registered else ""))
    if not forced:
        print("NOTE: these runs are NOT forced. A decaying flow has no steady state, so what follows is a"
              " check of the code, not the test, and its verdicts mean nothing.")
    rc, q1, q2 = 0, [], []
    rungs = sorted({Re for (_, Re) in runs})
    for Re in rungs:
        got = {}
        for s in ("A", "B"):
            by = runs.get((s, Re), {})
            if len(by) < 2:
                print("  Re %5g %s: %d grid(s); the rule needs two" % (Re, s, len(by)))
                continue
            (nc, bc), (nf, bf) = sorted(by.items())[-2:]
            f, c = sigma_of(bf, t0, t1, args.batches), sigma_of(bc, t0, t1, args.batches)
            if isinstance(f, str) or isinstance(c, str):
                print("  Re %5g %s: %s" % (Re, s, f if isinstance(f, str) else "the coarser grid " + c))
                continue
            grid, drift = abs(f["sigma"] - c["sigma"]), abs(f["halves"][0] - f["halves"][1])
            u = max(grid, f["se2"], drift)
            got[s] = (f["sigma"], u)
            f2 = sigma_of(bf, t0, t1, args.batches, 2.0)
            print("  Re %5g %s N %d/%d: sigma %+.3f (coarser %+.3f); u = %.3f = max(grid %.3f, 2 s.e. %.3f,"
                  " halves %.3f: %+.3f / %+.3f); batches' r1 %+.2f; within 2 delta %s" % (
                      Re, s, nc, nf, f["sigma"], c["sigma"], u, grid, f["se2"], drift, f["halves"][0],
                      f["halves"][1], f["r1"], f2 if isinstance(f2, str) else "%+.3f" % f2["sigma"]))
        if "B" in got:
            sg, u = got["B"]
            q1.append("TAKES" if sg + u < 0 else "FEEDS" if sg - u > 0 else "inside its band")
        else:
            q1.append("no reading")
        if "A" in got and "B" in got:
            d, u = got["A"][0] - got["B"][0], got["A"][1] + got["B"][1]
            q2.append("REMOVES" if d - u > 0 else "not shown")
            print("  Re %5g: Q1 %s; sigma_A - sigma_B = %+.3f against u_A + u_B = %.3f -> Q2 %s"
                  % (Re, q1[-1], d, u, q2[-1]))
        else:
            q2.append("no reading")
            print("  Re %5g: Q1 %s; Q2 no reading" % (Re, q1[-1]))
    if len(rungs) < 3 or "no reading" in q1 + q2:
        print("  NO VERDICT: the rule needs every rung of at least three, each with two grids covering the window")
        return 1
    v1 = ("TAKES: within delta of the no-slip wall the field loses energy to the flow on average"
          if all(c == "TAKES" for c in q1) else
          "FEEDS: within delta of the no-slip wall the flow feeds the field on average"
          if all(c == "FEEDS" for c in q1) else "MIXED (%s): neither is claimed" % ", ".join(q1))
    v2 = ("REMOVES: the stretching feeds the free-slip layer's Ohmic dissipation more than the no-slip layer's"
          if all(c == "REMOVES" for c in q2) else "NOT SHOWN (%s)" % ", ".join(q2))
    print("  Q1 verdict: %s" % v1)
    print("  Q2 verdict: %s" % v2)
    return rc


def test(args):
    """The forced test registered in GPU/csf3/tg_mhd_forced.sub (the docstring's summary):
    stationarity first, then Q0, then Q1 and Q2 through `stretch`."""
    if not args.registered:
        print("test: --registered COMMIT is required: the commit that added GPU/csf3/tg_mhd_forced.sub,\n"
              "whose header fixes this test. Nothing was read.")
        return 2
    t0, t1 = PROD_WINDOW
    runs = {}
    for name in sorted(os.listdir(args.runs_dir)):
        r = load(os.path.join(args.runs_dir, name))
        if r is None:
            continue
        if (r["F0"], r["FM"]) != PROD_PAIR or r["fp32"] or r["re"] not in PROD_GRIDS \
                or r["n"] not in PROD_GRIDS[r["re"]]:
            print("test: %s is not one of the production's runs (F0 %g, FM %g, FP64, the grids of %s);"
                  " give it the production's directory. Nothing was judged." % (
                      name, PROD_PAIR[0], PROD_PAIR[1], "GPU/csf3/tg_mhd_forced.sub"))
            return 2
        runs[(r["box"], r["re"], r["n"])] = r
    if not runs:
        print("test: no production run under %s" % args.runs_dir)
        return 2
    keys = [(b, Re, N) for Re, Ns in sorted(PROD_GRIDS.items()) for N in Ns for b in ("A", "B")]

    print("THE FORCED TEST (GPU/csf3/tg_mhd_forced.sub, registered in %s): F0 = %g, FM = %g, window t = %g..%g,"
          " %d batches" % (args.registered, PROD_PAIR[0], PROD_PAIR[1], t0, t1, PROD_BATCHES))
    print()
    print("STATIONARITY, a gate on every run (energies, eps, Mach and div B only):")
    print("  %-22s %8s %8s %8s %8s %13s %8s %8s %8s  %s" % (
        "run", "<E_V>", "<E_M>", "<eps>", "eps rms", "drift V / M", "umax win", "divb/j", "t_int", "gate"))
    failed = []
    for key in keys:
        tag = "%s Re %4d N %d" % key
        r = runs.get(key)
        st = window_stats(r, t0, t1) if r is not None else "missing"
        if isinstance(st, str):
            why = st
        elif st["umax_win"] > UMAX:
            why = "umax_lat %.4f > %g in the window" % (st["umax_win"], UMAX)
        elif max(st["drift_V"], st["drift_M"]) >= STATIONARY:
            why = "drift %.1f %% >= %g %%" % (100 * max(st["drift_V"], st["drift_M"]), 100 * STATIONARY)
        else:
            why = None
        if isinstance(st, str):
            print("  %-22s %s  FAIL" % (tag, st))
        else:
            print("  %-22s %8.4f %8.4f %8.4f %7.1f%% %5.1f%% / %4.1f%% %8.4f %8.1e %8.2f  %s" % (
                tag, st["E_V"], st["E_M"], st["eps"], 100 * st["eps_rms"] / st["eps"], 100 * st["drift_V"],
                100 * st["drift_M"], st["umax_win"], st["divb"], st["tau_int"], "pass" if why is None else
                "FAIL: " + why))
        if why is not None:
            failed.append(tag)
    if failed:
        print("  NO VERDICT on Q0, Q1 or Q2: %d run(s) failed the gate (%s). Nothing near the walls was read."
              % (len(failed), ", ".join(failed)))
        return 1

    # ---- Q0, the wall share -------------------------------------------------
    bm = {key: batch_means(runs[key], t0, t1, PROD_BATCHES) for key in keys}
    bad = [k for k in keys if isinstance(bm[k], str)]
    if bad:
        print("  NO VERDICT: no shares for %s (%s)" % ("%s Re %d N %d" % bad[0], bm[bad[0]]))
        return 1
    print()
    print("Q0, THE WALL SHARE within delta: Delta = F_B - F_A on the finer grid; u = max(grid, 2 s.e., halves)")
    rungs = sorted(PROD_GRIDS)
    D, U, FB, calls = {}, {}, {}, []
    for Re in rungs:
        nc, nf = PROD_GRIDS[Re]
        a, b, ac, bc = bm[("A", Re, nf)], bm[("B", Re, nf)], bm[("A", Re, nc)], bm[("B", Re, nc)]
        d, dc = b["F"][0] - a["F"][0], bc["F"][0] - ac["F"][0]
        grid = abs(d - dc)
        stat = 2.0 * math.sqrt(a["se"] ** 2 + b["se"] ** 2)
        halves = abs((b["halves"][0] - a["halves"][0]) - (b["halves"][1] - a["halves"][1]))
        u = max(grid, stat, halves)
        D[Re], U[Re], FB[Re] = d, u, b["F"][0]
        calls.append(d - u > 0)
        parts = ("; viscous %+.4f, Ohmic %+.4f" % (b["F"][3] - a["F"][3], b["F"][4] - a["F"][4])
                 if len(a["F"]) > 3 and len(b["F"]) > 3 else "")
        print("  Re %4d N %d/%d: F_A %.4f, F_B %.4f, Delta %+.4f (coarser %+.4f); u = %.4f = max(grid %.4f,"
              " 2 s.e. %.4f, halves %.4f) -> %s" % (Re, nc, nf, a["F"][0], b["F"][0], d, dc, u, grid, stat, halves,
                                                    "EXCESS" if d - u > 0 else "no excess"))
        print("           within 2 and 4 delta: Delta %+.4f, %+.4f%s; batches' r1 A %+.2f, B %+.2f" % (
            b["F"][1] - a["F"][1], b["F"][2] - a["F"][2], parts, a["r1"], b["r1"]))
    lo, mid, hi = rungs[0], rungs[len(rungs) // 2], rungs[-1]
    L = math.log(hi / lo)
    if not all(calls):
        q0 = "NO EXCESS at Re = %s: no exponent is judged" % ", ".join(
            "%d" % Re for Re, c in zip(rungs, calls) if not c)
        if D[lo] > 0 and D[hi] > 0:
            print("  (reported, judging nothing) p = %+.3f" % (math.log(D[hi] / D[lo]) / L))
    else:
        p = math.log(D[hi] / D[lo]) / L
        p_min = math.log((D[hi] - U[hi]) / (D[lo] + U[lo])) / L
        p_max = math.log((D[hi] + U[hi]) / (D[lo] - U[lo])) / L
        print("  exponent of Delta in Re: p = %+.3f, range [%+.3f, %+.3f], against %g" % (
            p, p_min, p_max, PROD_THRESHOLD))
        print("  (reported, judging nothing) the middle rung off the ends' line: %+.3f in ln Delta;"
              " F_B's own exponent %+.3f" % (math.log(D[mid]) - 0.5 * (math.log(D[lo]) + math.log(D[hi]))
                                              if D[mid] > 0 else float("nan"), math.log(FB[hi] / FB[lo]) / L))
        q0 = ("WALLS HOLD" if p_min > PROD_THRESHOLD else "CASCADE" if p_max < PROD_THRESHOLD else "UNDECIDED")
    print("  Q0 verdict: %s" % q0)
    print()

    # ---- Q1 and Q2, the stretching -------------------------------------------
    rc = stretch(argparse.Namespace(runs_dirs=[args.runs_dir], window=[t0, t1], batches=PROD_BATCHES,
                                    registered=args.registered))
    return rc


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot", help="select the forcing amplitudes from the pilot")
    p.add_argument("pilot_dir")
    p.add_argument("--round", type=int, choices=(1, 2), default=1)
    p.add_argument("--window", type=float, nargs=2, default=None,
                   help="default: 25 50 in round 1, 100 150 in round 2")
    p.add_argument("--round2", default=os.path.join(ROOT, "results", "P_tg_mhd", "round2"))
    q = sub.add_parser("shares", help="time-averaged near-wall shares with batch-mean errors")
    q.add_argument("run_dirs", nargs="+")
    q.add_argument("--window", type=float, nargs=2, required=True)
    q.add_argument("--batches", type=int, required=True)
    s = sub.add_parser("stretch", help="the registered stretching test: sigma = <S>/<ohm> within delta")
    s.add_argument("runs_dirs", nargs="+")
    s.add_argument("--window", type=float, nargs=2, required=True)
    s.add_argument("--batches", type=int, required=True)
    s.add_argument("--registered", default=None, metavar="COMMIT",
                   help="the commit that fixed the window, the batches, the rungs and the stationarity"
                        " test; forced runs are refused without it")
    w = sub.add_parser("test", help="the forced test registered in GPU/csf3/tg_mhd_forced.sub")
    w.add_argument("runs_dir")
    w.add_argument("--registered", default=None, metavar="COMMIT",
                   help="the commit that added GPU/csf3/tg_mhd_forced.sub; required")
    args = ap.parse_args(argv)
    if args.cmd == "pilot":
        return pilot(args)
    if args.cmd == "shares":
        return shares(args)
    if args.cmd == "stretch":
        return stretch(args)
    if args.cmd == "test":
        return test(args)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
