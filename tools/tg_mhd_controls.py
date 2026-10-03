#!/usr/bin/env python3
"""Phase 2's four controls, judged by the rules fixed in GPU/csf3/tg_mhd_controls.sub.

    tg_mhd_controls.py CONTROLS_DIR [--round2 DIR] [--budget DIR] [--window 2 10]

CONTROLS_DIR holds what that job writes (runs/tg_mhd_controls/, or
results/P_tg_mhd/controls/ once copied back): one <setup>_re<Re>_n<N>_<label>/
per run. The references are round 2's runs -- --round2 (default
results/P_tg_mhd/round2) for series, --budget (default results/P_tg_mhd/budget,
the same runs with budget.dat) for phi. Every observable is the paper's, read by
the paper's own code: F1, the share of the dissipation within delta over the
window, and its viscous and Ohmic parts (tg_mhd_ladder.integrated), and phi
(tg_mhd_budget). A band is the difference between a rung's two grids.

The rules, in short (the job's header states and argues them):
  C1 the start      B - A with a smooth start (-ramp 0.2, 1.0) at Re = 500 and
                    2000 differs from round 2's by < 10 % of it, and stays above
                    round 2's band -> "not an imprint of the start";
  C2 the field      F1(C at B's initial energy) - F1(B) > both bands, at every
                    rung -> "the field moves dissipation away from the walls"
                    holds at matched energy;
  C3 the Mach       u0 0.05 -> 0.04 at Re = 2000, N = 640: B - A moves by less
                    than round 2's band there, and phi by less than phi's band;
  C4 the Pm         at Pm = 0.5 and 2 (Re = 1000): B - A > band, viscous part
                    > 0, Ohmic part < 0; and phi - band > 1/2.

Exit status 0 when every control could be judged, 1 if a run is missing.
Pure stdlib.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tg_mhd_budget as BU    # noqa: E402
import tg_mhd_ladder as T     # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GRIDS = {250: (192, 256), 500: (256, 384), 1000: (384, 512), 2000: (512, 640)}


class Missing(Exception):
    pass


def run_at(d):
    s = T.load(os.path.join(d, "series.dat"))
    if s is None:
        raise Missing(d)
    b = os.path.join(d, "budget.dat")
    s["budget"] = BU.load_budget(b) if os.path.exists(b) else None
    return s


def F(run, w):
    got = T.integrated(run, *w)
    if isinstance(got, str):
        raise Missing("%s: %s" % (run["tag"], got))
    return got[0][0], got[1]                     # F1, [viscous, Ohmic] or None


def phi(a, b, w):
    if a["budget"] is None or b["budget"] is None:
        raise Missing("no budget.dat for %s / %s" % (a["tag"], b["tag"]))
    return BU.phi_of(BU.layer(a["budget"], *w, 1.0), BU.layer(b["budget"], *w, 1.0))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("controls_dir")
    ap.add_argument("--round2", default=os.path.join(ROOT, "results", "P_tg_mhd", "round2"))
    ap.add_argument("--budget", default=os.path.join(ROOT, "results", "P_tg_mhd", "budget"))
    ap.add_argument("--window", type=float, nargs=2, default=[2.0, 10.0])
    args = ap.parse_args(argv)
    w = tuple(args.window)
    ctl = lambda s, Re, N, lab: run_at(os.path.join(args.controls_dir, "%s_re%d_n%d_%s" % (s, Re, N, lab)))
    r2 = lambda s, Re, N: run_at(os.path.join(args.round2, "%s_re%d_n%d" % (s, Re, N)))
    bud = lambda s, Re, N: run_at(os.path.join(args.budget, "%s_re%d_n%d" % (s, Re, N)))
    rc = 0
    print("Phase 2 controls (GPU/csf3/tg_mhd_controls.sub), within delta, t = %g..%g" % w)

    def ref_band(Re):
        nc, nf = GRIDS[Re]
        return (abs(F(r2("A", Re, nf), w)[0] - F(r2("A", Re, nc), w)[0])
                + abs(F(r2("B", Re, nf), w)[0] - F(r2("B", Re, nc), w)[0]))

    # ---- C1: the start ---------------------------------------------------------
    print("\nC1 THE START: B - A with a smooth start against round 2's impulsive one")
    try:
        ok = True
        for Re in (500, 2000):
            nf = GRIDS[Re][1]
            fa, fb = F(r2("A", Re, nf), w)[0], F(r2("B", Re, nf), w)[0]
            d0, band = fb - fa, ref_band(Re)
            for lab in ("ramp0.2", "ramp1.0"):
                d = F(ctl("B", Re, nf, lab), w)[0] - fa
                rel = (d - d0) / d0
                good = abs(rel) < 0.10 and d > band
                ok = ok and good
                print("  Re %4d N %d %-8s B - A %+.4f against %+.4f impulsive: change %+.1f %% (band %.4f) -> %s"
                      % (Re, nf, lab, d, d0, 100 * rel, band, "within 10 %" if good else "BEYOND"))
        print("  verdict: %s" % ("NOT AN IMPRINT OF THE START" if ok else "THE START IS A FACTOR OF THE EXCESS"))
    except Missing as e:
        print("  missing: %s" % e); rc = 1

    # ---- C2: the field at matched energy --------------------------------------
    print("\nC2 THE FIELD AT MATCHED ENERGY: F1(C, E_V(0) = 1/4) against F1(B)")
    try:
        held = []
        for Re in sorted(GRIDS):
            nc, nf = GRIDS[Re]
            cf, cc = ctl("C", Re, nf, "vamp"), ctl("C", Re, nc, "vamp")
            Fc, Fcc = F(cf, w)[0], F(cc, w)[0]
            Fb, Fbc = F(r2("B", Re, nf), w)[0], F(r2("B", Re, nc), w)[0]
            band = abs(Fc - Fcc) + abs(Fb - Fbc)
            pk = T.turbulent_peak(cf)
            good = Fc - Fb > band
            held.append(good)
            print("  Re %4d: F1(C) %.4f, F1(B) %.4f, C - B %+.4f (bands %.4f) -> %s; C's turbulent peak: %s"
                  % (Re, Fc, Fb, Fc - Fb, band, "holds" if good else "DOES NOT", "t %.2f" % pk[0]
                     if isinstance(pk, tuple) else pk))
        print("  verdict: %s" % ("HOLDS AT EVERY RUNG" if all(held) else
                                  "holds at Re = %s only" % ", ".join(str(Re) for Re, g in zip(sorted(GRIDS), held) if g)
                                  if any(held) else "DOES NOT HOLD AT MATCHED ENERGY"))
    except Missing as e:
        print("  missing: %s" % e); rc = 1

    # ---- C3: the Mach number at the top rung ----------------------------------
    print("\nC3 THE MACH NUMBER AT Re = 2000: u0 0.05 -> 0.04, N = 640")
    try:
        nf = 640
        A0, B0 = r2("A", 2000, nf), r2("B", 2000, nf)
        A1, B1 = ctl("A", 2000, nf, "u0p04"), ctl("B", 2000, nf, "u0p04")
        d0 = F(B0, w)[0] - F(A0, w)[0]
        d1 = F(B1, w)[0] - F(A1, w)[0]
        band = ref_band(2000)
        p0 = phi(bud("A", 2000, nf), bud("B", 2000, nf), w)
        p0c = phi(bud("A", 2000, 512), bud("B", 2000, 512), w)
        p1 = phi(A1, B1, w)
        pband = abs(p0 - p0c)
        ok = abs(d1 - d0) < band and abs(p1 - p0) < pband
        print("  B - A %+.4f -> %+.4f, shift %+.5f against band %.4f; phi %.3f -> %.3f, shift %+.3f against %.3f"
              % (d0, d1, d1 - d0, band, p0, p1, p1 - p0, pband))
        print("  verdict: %s" % ("PASS" if ok else "FAIL"))
    except Missing as e:
        print("  missing: %s" % e); rc = 1

    # ---- C4: the magnetic Prandtl number --------------------------------------
    print("\nC4 THE MAGNETIC PRANDTL NUMBER at Re = 1000")
    for pm, grids in (("pm0.5", (384, 512)), ("pm2", (512, 640))):
        try:
            nc, nf = grids
            a, ac = ctl("A", 1000, nf, pm), ctl("A", 1000, nc, pm)
            b, bc = ctl("B", 1000, nf, pm), ctl("B", 1000, nc, pm)
            (Fa, pa), (Fb, pb) = F(a, w), F(b, w)
            band = abs(Fa - F(ac, w)[0]) + abs(Fb - F(bc, w)[0])
            ph, phc = phi(a, b, w), phi(ac, bc, w)
            pband = abs(ph - phc)
            vis, ohm = pb[0] - pa[0], pb[1] - pa[1]
            robust = (Fb - Fa > band) and vis > 0 and ohm < 0
            mech = ph - pband > 0.5
            print("  %s N %d/%d: B - A %+.4f (band %.4f), viscous %+.4f, Ohmic %+.4f -> %s; phi %.3f (band %.3f)"
                  " -> %s" % (pm, nc, nf, Fb - Fa, band, vis, ohm, "robust statements HOLD" if robust else
                              "robust statements DO NOT hold", ph, pband, "mechanism HOLDS" if mech else
                              "mechanism DOES NOT hold"))
        except Missing as e:
            print("  %s missing: %s" % (pm, e)); rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
