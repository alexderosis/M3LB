#!/usr/bin/env python3
"""A control run against the ladder -- the Mach-halving check first -- and its verdict.

    tg_mhd_mach.py VARIED_DIR REF_DIR [REF_DIR ...]

VARIED_DIR holds runs that differ from their references in ONE setting: the
runs of GPU/csf3/tg_mhd_mach.sub (u0 = 0.025 against the ladder's 0.05), or the
FP64 controls of tg_mhd_round2.sub (FP64 against FP32 at Re = 2000). The REF_DIRs
together hold the reference runs AND the grids that give each rung its
resolution band -- e.g. `results/P_tg_mhd/fp64check results/P_tg_mhd/round2
results/P_tg_mhd/ladder`. Runs are matched on setup, Re and N, with u0 and the
precision read from each series header, and every quantity is read at the run's
own turbulent peak exactly as tools/tg_mhd_ladder.py reads it (this imports it).

WHAT HALVING u0 CHANGES: the Mach number, and -- at fixed N and Re, since
nu_lat = u0 (N - 1) / (pi Re) -- tau - 1/2 with it. So a shift below is the
Ma^2 error and the tau error together; the job's header says why that is the
test there is.

THE VERDICT, fixed in the job's header before any run. Per rung, the shift in
B - A (fw1 at the peak) between the two u0 is
    PASS      if |shift| <= that rung's resolution band from the ladder -- the
              fw1 differences between its two grids for A and for B, added;
    MARGINAL  if |shift| <= half the ladder's B - A step to the rung below:
              the error is smaller than the trend it could fake;
    FAIL      otherwise: the Mach/tau error is the size of the Re trend, and
              the ladder must be rerun at a lower u0 before the trend is read.
Also printed: each run's fw1/fw2/fw4 shift against the gate's 5 %, and B - A
extrapolated to Ma -> 0 on the assumption that the error goes as Ma^2,
X0 = X(u0/2) - (X(u0) - X(u0/2)) / 3 -- an estimate from two points, not a
measurement. A run that diverged or has no turbulent peak gives its rung no
verdict, and says so.

Exit status 0 when every rung with a verdict passes or is marginal, 1 when any
fails or no rung has a verdict. Pure stdlib.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tg_mhd_ladder as T  # noqa: E402


def load_dir(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*", "series.dat"))):
        r = T.load(p)
        if r is None or r["setup"] not in ("A", "B", "C"):
            continue
        with open(p) as f:
            m = re.search(r"\bu0=([0-9.eE+-]+)", f.readline())
        r["u0"] = float(m.group(1)) if m else None
        out.append(r)
    return out


def at_peak(r):
    """(t_peak, eps_peak, [fw1, fw2, fw4]) or the reason there is none."""
    if r["diverged"]:
        return "DIVERGED"
    pk = T.turbulent_peak(r)
    if not isinstance(pk, tuple):
        return pk
    return pk[0], pk[1], T.fw_at(r, pk[0])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("varied_dir")
    ap.add_argument("ref_dirs", nargs="+")
    args = ap.parse_args(argv)

    mach = load_dir(args.varied_dir)
    ladder = [r for d in args.ref_dirs for r in load_dir(d)]
    if not mach:
        raise SystemExit("no tg_mhd series.dat under %s" % args.varied_dir)

    # the ladder at each (setup, Re): its runs by N, and each run's peak reading
    lad = {}
    for r in ladder:
        lad.setdefault((r["setup"], r["Re"]), {})[r["N"]] = r

    print("PER RUN: each varied run against its reference at the same setup, Re and N")
    print("%-2s %6s %4s %6s %4s | %6s %9s %7s %7s %7s | shift in fw1/fw2/fw4 (5 %% gate)"
          % ("S", "Re", "N", "u0", "prec", "t_peak", "eps_peak", "fw1", "fw2", "fw4"))
    pair = {}
    for m in sorted(mach, key=lambda r: (r["Re"], r["setup"])):
        base = lad.get((m["setup"], m["Re"]), {}).get(m["N"])
        if base is None:
            print("%-2s %6g %4d  no reference run at this setup, Re and N -- skipped" % (m["setup"], m["Re"], m["N"]))
            continue
        pm, pb = at_peak(m), at_peak(base)
        for r, p in ((base, pb), (m, pm)):
            if isinstance(p, tuple):
                print("%-2s %6g %4d %6g %4s | %6.2f %9.3e %7.4f %7.4f %7.4f |"
                      % (r["setup"], r["Re"], r["N"], r["u0"], r["prec"], p[0], p[1], p[2][0], p[2][1], p[2][2]),
                      end="")
            else:
                print("%-2s %6g %4d %6g %4s | (%s)" % (r["setup"], r["Re"], r["N"], r["u0"], r["prec"], p), end="")
            if r is m and isinstance(pm, tuple) and isinstance(pb, tuple):
                rel = [abs(a - b) / max(abs(a), abs(b)) for a, b in zip(pm[2], pb[2])]
                print(" %.2f / %.2f / %.2f %%  %s" % (100 * rel[0], 100 * rel[1], 100 * rel[2],
                                                    "ok" if max(rel) <= 0.05 else "OVER 5 %"))
            else:
                print()
        pair[(m["setup"], m["Re"])] = (pb, pm, m["N"], "%g %s" % (base["u0"], base["prec"]),
                                       "%g %s" % (m["u0"], m["prec"]), base["u0"] != m["u0"])
        print()

    def fw1_fine(setup, Re):
        """fw1 at the peak on the ladder's finest grid at this setup and Re."""
        runs = lad.get((setup, Re), {})
        if not runs:
            return None
        p = at_peak(runs[max(runs)])
        return p[2][0] if isinstance(p, tuple) else None

    def band(Re):
        tot = 0.0
        for s in ("A", "B"):
            runs = lad.get((s, Re), {})
            Ns = sorted(runs)[-2:]
            if len(Ns) < 2:
                return None
            ps = [at_peak(runs[N]) for N in Ns]
            if not all(isinstance(p, tuple) for p in ps):
                return None
            tot += abs(ps[1][2][0] - ps[0][2][0])
        return tot

    print("VERDICT per rung: the shift in B - A (fw1 at the peak) from the reference to the varied runs")
    verdicts = {}
    res_all = sorted({Re for s, Re in lad})
    for Re in sorted({Re for s, Re in pair}):
        if ("A", Re) not in pair or ("B", Re) not in pair:
            print("  Re %-6g needs both A and B -- no verdict" % Re)
            continue
        (a_hi, a_lo, N, u_hi, u_lo, mach), (b_hi, b_lo, _, _, _, _) = pair[("A", Re)], pair[("B", Re)]
        bad = [what for what, p in (("A at %s" % u_hi, a_hi), ("A at %s" % u_lo, a_lo),
                                    ("B at %s" % u_hi, b_hi), ("B at %s" % u_lo, b_lo))
               if not isinstance(p, tuple)]
        if bad:
            print("  Re %-6g N = %d: no verdict -- no reading for %s" % (Re, N, ", ".join(bad)))
            continue
        d_hi, d_lo = b_hi[2][0] - a_hi[2][0], b_lo[2][0] - a_lo[2][0]
        shift = d_lo - d_hi
        bd = band(Re)
        lower = [x for x in res_all if x < Re]
        step = None
        if lower:
            f_a, f_b = fw1_fine("A", lower[-1]), fw1_fine("B", lower[-1])
            if f_a is not None and f_b is not None:
                step = abs((f_b - f_a) - d_hi)
        if bd is not None and abs(shift) <= bd:
            v = "PASS"
        elif step is not None and abs(shift) <= 0.5 * step:
            v = "MARGINAL"
        elif bd is None and step is None:
            v = "no yardstick"
        else:
            v = "FAIL"
        x0 = d_lo - (d_hi - d_lo) / 3.0
        print("  Re %-6g N = %d: B - A = %+.4f at %s, %+.4f at %s -> shift %+.4f;"
              " band %s, half-step %s  %s" % (Re, N, d_hi, u_hi, d_lo, u_lo, shift,
                                              "%.4f" % bd if bd is not None else "--",
                                              "%.4f" % (0.5 * step) if step is not None else "--", v))
        if mach:     # only a u0 change has an Ma^2 to extrapolate
            print("  %-9s Ma -> 0 estimate (error ~ Ma^2): B - A = %+.4f" % ("", x0))
        verdicts[Re] = v
    if not verdicts:
        print("  no rung has a verdict")
        return 1
    if any(v not in ("PASS", "MARGINAL") for v in verdicts.values()):
        return 1
    top = max(Re for s, Re in pair)
    if top not in verdicts:
        # The plan's gate is AT THE TOP RUNG: a verdict from the rung below is the
        # fallback's, and the exit status says the gate itself was not reached.
        print("  the top rung (Re = %g) has no verdict: the plan's gate is not met there;"
              " the rung below is the fallback" % top)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
