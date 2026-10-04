#!/usr/bin/env python3
"""Phase 3 of the confined-MHD paper: the forced, statistically steady boxes.

    tg_mhd_forced.py pilot PILOT_DIR [--round 1|2] [--window T0 T1] [--round2 DIR]
    tg_mhd_forced.py shares RUN_DIR... --window T0 T1 --batches NB

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

Exit status 0 when a pair was selected (pilot) or every run was read (shares),
1 otherwise. Pure stdlib.
"""
import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tg_mhd_ladder as T     # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
UMAX = 0.08                    # the Mach bound on umax_lat, Ma = 0.139
UNSTEADY = 0.01                # r.m.s./mean of eps above which a run is unsteady
EDGE_D = 0.5                   # round 1: a selection on the scan's edge further than this is no selection
STATIONARY = 0.25              # round 2: largest drift of E_V or E_M between the window's halves
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
    args = ap.parse_args(argv)
    if args.cmd == "pilot":
        return pilot(args)
    if args.cmd == "shares":
        return shares(args)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
