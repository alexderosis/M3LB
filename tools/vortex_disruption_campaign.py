#!/usr/bin/env python3
"""Apply the rule of GPU/csf3/vortex_disruption_map.sub to its runs.

    python3 tools/vortex_disruption_campaign.py runs/vortex_disruption

The directory holds up to three groups, each one run per subdirectory:
    map/       the main ladder: Pm = 1, five Rm, eight fields each
    grid/      the threshold rows again on a 1.5x finer grid, at two Rm
    scatter/   the threshold rows again with a 1e-5 elliptical perturbation
A missing group is reported as not run; the rule then says what it cannot
decide.

THE RULE, FIXED ON 2026-11-19, BEFORE ANY RUN OF THE CAMPAIGN. The job's header
states the same rule and is authoritative if the two ever differ; the push of
the commit that carries both is its timestamp. It was written after the laptop
map (results/Q_vortex_disruption: D2Q9 + D2Q5, Rm = 250..4000, which gave
p = -0.506 with the ladder allowing 0.038), and that is the expectation it
tests.

  OBSERVABLE. Delta, the `dow` column: the Okubo-Weiss disruption measure
  against the exact viscous vortex. For each run, its LARGEST value over the
  run; a run that did not reach t = 12 Rm^(1/3) has none and is excluded (and
  listed).

  THRESHOLD. M_c(Rm): the field at which that largest Delta first crosses 0.8
  along the ladder at that Rm, interpolated linearly in ln M between the two
  runs that bracket it. An Rm at which 0.8 is not bracketed is dropped and
  listed.

  EXPONENT. p: the least-squares slope of ln M_c against ln Rm over map/. FEWER
  THAN FOUR Rm WITH A THRESHOLD: NO VERDICT.

  UNCERTAINTY. U, the LARGEST of
    (a) the ladder's bound: how far p could move if each M_c lay anywhere
        between its two bracketing runs (tools/vortex_disruption_map.py's
        slope_bound);
    (b) the grid's shift: |p_grid - p|, where p_grid is the same fit with the
        M_c of grid/ put in place of map/'s at the Rm grid/ covers;
    (c) the scatter's shift: |p_scat - p|, likewise with scatter/'s M_c.
  A group not run contributes nothing to U, and the verdict says which were
  missing.

  VERDICT, on p and U:
      |p + 1/2| <= U < |p + 1/3|    "M^2 Rm DECIDES DESTRUCTION"
      |p + 1/3| <= U < |p + 1/2|    "M^2 Rm^(2/3) DECIDES DESTRUCTION"
      both within U                 "BOTH ALLOWED"
      neither within U              "NEITHER"

  DESCRIPTIVE, NO GATE.
    - The wind-up: q in 1 - surv ~ Rm^q at t = 3 Rm^(1/3) along the rows
      M^2 Rm = 0.625 and 1.25 of map/. The laptop's Pm = 1 map gave -0.354 and
      -0.347 for these rows, of which about -0.04 was Re's.
    - The lattices: M_c at Rm = 2000 and 4000 from map/ (D3Q27 + D3Q7) beside
      the laptop map's (D2Q9 + D2Q5), which the magnetic lattice can move
      (CLAUDE.md, "two-dimensional means nz = 1").
    - The level: p again at Delta = 0.7 and 0.9, as the laptop's banner
      records it moving with the level.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vortex_disruption_map as vm     # noqa: E402  (read, readings, crossing, slope)

LEVEL = 0.8
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAPTOP = os.path.join(ROOT, "results", "Q_vortex_disruption", "map_r020")


def load(d):
    if not os.path.isdir(d):
        return None
    runs = [vm.read(os.path.join(d, n)) for n in sorted(os.listdir(d))]
    return [r for r in runs if r]


def thresholds(runs, level, col="dow", fn=lambda v: v, key="max"):
    """{Rm: (M_c, h)} and the lists of excluded runs and unbracketed Rm."""
    out, ended, open_ = {}, [], []
    for rm in sorted({r["Rm"] for r in runs}):
        sel = [r for r in runs if r["Rm"] == rm]
        pts = []
        for r in sel:
            v = vm.readings(r, col, fn)[key]
            if v is None:
                ended.append("Rm %g M %g (t_end %.1f)" % (rm, r["M"], r["d"]["t"][-1]))
            else:
                pts.append((r["M"], v))
        got = vm.crossing(pts, level)
        if got:
            out[rm] = got
        else:
            open_.append(rm)
    return out, ended, open_


def fit(mc):
    xy = sorted((rm, v[0]) for rm, v in mc.items())
    return vm.slope(xy)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    base = sys.argv[1]
    groups = {g: load(os.path.join(base, g)) for g in ("map", "grid", "scatter")}
    for g, runs in groups.items():
        print("%-8s %s" % (g + "/", "not run" if runs is None else
                           "%d runs, Rm = %s" % (len(runs), ", ".join(
                               "%g (R0 %s)" % (rm, "/".join(sorted({"%g" % r["R0"]
                                                for r in runs if r["Rm"] == rm})))
                               for rm in sorted({r["Rm"] for r in runs})))))
    main_runs = groups["map"]
    if not main_runs:
        sys.exit("\nNO VERDICT: map/ has no runs.")
    if any("dow" not in r["d"] for r in main_runs):
        sys.exit("\nthese series have no dow column")

    mc, ended, open_ = thresholds(main_runs, LEVEL)
    print("\nTHRESHOLD (largest Delta crosses %.1f), map/:" % LEVEL)
    for rm, (m, h) in sorted(mc.items()):
        print("  Rm %-6g  M_c %.5f   M_c^2 Rm %.3f   M_c^2 Rm^(2/3) %.4f   (bracket +/- %.1f %% in M)"
              % (rm, m, m * m * rm, m * m * rm ** (2.0 / 3.0), 100 * (math.exp(h) - 1)))
    for e in ended:
        print("  EXCLUDED, did not reach t = 12 Rm^(1/3): %s" % e)
    for rm in open_:
        print("  DROPPED, %.1f not bracketed: Rm %g" % (LEVEL, rm))
    if len(mc) < 4:
        print("\nNO VERDICT: %d Rm with a threshold, the rule asks for four." % len(mc))
        return 2

    p = fit(mc)
    bound = vm.slope_bound(sorted((rm, v[1]) for rm, v in mc.items()))
    parts = [("ladder bound", bound)]
    missing = []
    for g in ("grid", "scatter"):
        runs = groups[g]
        if not runs:
            missing.append(g)
            continue
        sub, s_ended, s_open = thresholds(runs, LEVEL)
        for e in s_ended:
            print("  %s/: EXCLUDED %s" % (g, e))
        for rm in s_open:
            print("  %s/: not bracketed at Rm %g" % (g, rm))
        swapped = dict(mc)
        used = []
        for rm, v in sub.items():
            if rm in swapped:
                print("  %s/ at Rm %g: M_c %.5f against map/'s %.5f (%+.1f %%)"
                      % (g, rm, v[0], mc[rm][0], 100 * (v[0] / mc[rm][0] - 1)))
                swapped[rm] = v
                used.append(rm)
        if used:
            parts.append(("%s shift" % g, abs(fit(swapped) - p)))
        else:
            missing.append(g)
    U = max(v for _, v in parts)

    print("\nEXPONENT  p = %+.3f over %d Rm (%g .. %g)" % (p, len(mc), min(mc), max(mc)))
    for name, v in parts:
        print("  %-14s %.3f" % (name, v))
    print("  U = %.3f%s" % (U, ("   (not run, so not in U: %s)" % ", ".join(missing))
                            if missing else ""))
    a, b = abs(p + 0.5), abs(p + 1.0 / 3.0)
    print("  |p + 1/2| = %.3f   |p + 1/3| = %.3f" % (a, b))
    if a <= U < b:
        verdict = "M^2 Rm DECIDES DESTRUCTION"
    elif b <= U < a:
        verdict = "M^2 Rm^(2/3) DECIDES DESTRUCTION"
    elif a <= U and b <= U:
        verdict = "BOTH ALLOWED"
    else:
        verdict = "NEITHER"
    print("\nVERDICT: %s" % verdict)

    # ---- descriptive, no gate -------------------------------------------------
    print("\nDESCRIPTIVE (no gate)")
    for x in (0.625, 1.25):
        pts = []
        for r in main_runs:
            if abs(r["M"] ** 2 * r["Rm"] / x - 1) < 0.02:
                v = vm.readings(r, "surv", lambda s: 1.0 - s)["c3"]
                if v and v > 0:
                    pts.append((r["Rm"], v))
        q = vm.slope(sorted(pts))
        print("  wind-up, 1 - surv at t = 3 Rm^(1/3), row M^2 Rm = %-5g q = %s over %d Rm"
              "   (laptop, Pm = 1: %s)" % (x, "%+.3f" % q if q is not None else "n/a",
                                          len(pts), "-0.354" if x == 0.625 else "-0.347"))
    for lv in (0.7, 0.9):
        sub, _, _ = thresholds(main_runs, lv)
        pl = fit(sub) if len(sub) >= 2 else None
        print("  level %.1f: p = %s over %d Rm" % (lv, "%+.3f" % pl if pl is not None else "n/a",
                                                  len(sub)))
    lap = load(LAPTOP)
    if lap:
        lmc, _, _ = thresholds(lap, LEVEL)
        for rm in (2000.0, 4000.0):
            if rm in mc and rm in lmc:
                print("  lattices at Rm %g: M_c %.5f (D3Q27 + D3Q7) against %.5f (D2Q9 + D2Q5, "
                      "laptop), %+.1f %%" % (rm, mc[rm][0], lmc[rm][0],
                                             100 * (mc[rm][0] / lmc[rm][0] - 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
