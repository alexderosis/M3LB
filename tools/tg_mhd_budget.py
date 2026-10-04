#!/usr/bin/env python3
"""The energy budget against wall distance, and the test of the Ohmic-deficit conjecture.

    tg_mhd_budget.py RUNS_DIR [RUNS_DIR ...] [--window 2 10] [--m 1]

Reads every RUNS_DIR/*/budget.dat written by demonstrator/tg_mhd or
GPU/src/tg_mhd.cu (since 2026-10-02; GPU/csf3/tg_mhd_budget.sub puts them in
runs/tg_mhd_budget/), beside each run's series.dat, which says which box it is.
The drivers' banners define the twelve terms; in short, per unit volume and in
the paper's units, with eK = |u|^2/2 and eM = |B|^2/2,

    d eK/dt = lor + advK + pres + difK - visc
    d eM/dt = S   + advM + difM + cmpM - ohm

with S = B_i B_j d_j u_i the stretching of the field by the flow and lor = u.(j x B)
the Lorentz work, each written layer by layer out to 8 delta (delta = Re^-1/2).
Within m delta of the walls every term is read as f_w is -- the layer sums with the
band edge interpolated (within_cells) -- and integrated over --window with every
integrand interpolated to the window's edges, as tg_mhd_ladder.integrated does.

THE TEST, fixed in GPU/csf3/tg_mhd_budget.sub before its runs (2026-10-02). Near a
no-slip conducting wall the Ohmic dissipation within delta is 0.55-0.59 of what it
is at a free-slip one over t = 2..10, in absolute terms; the paper conjectures that
the no-slip wall stops the tangential flow and with it the stretching that builds
current sheets along the wall. Over the window and within delta, per run,

    Omega = int ohm,   S = int S,   dE = eM(t1) - eM(t0),   T = dE - S + Omega,

so that Omega = S + T - dE EXACTLY: T is whatever else feeds the layer --
advection, resistive transport, the compressible terms and the discretisation's
own residual. The share of the Ohmic deficit that the stretching deficit accounts
for is

    phi = (S_A - S_B) / (Omega_A - Omega_B),

on the finer grid of each rung, with the band |phi(finer) - phi(coarser)|. The
verdict is "stretching" if phi - band > 1/2 at every rung, "transport" if
phi + band < 1/2 at every rung, and "undecided" otherwise. phi uses only S and
ohm, both pointwise first-derivative quantities; it does NOT depend on the
explicit transport terms, which a weakly compressible scheme closes only to a
residual, and which this tool reports beside the test as a diagnostic: the
budget's closure (T against advM + difM + cmpM), the box identity <S> = -<lor>,
the shares within 2 and 4 delta and over other windows, and the kinetic budget
of each box's layer -- what supplies its viscous dissipation.

Exit status: 0 with a verdict, 1 if a rung lacks a pair of grids, 2 if nothing
was found. Pure stdlib, like tg_mhd_ladder.py, whose loader and integrator it uses.
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tg_mhd_ladder as T     # noqa: E402

TERMS = ("eK", "eM", "visc", "ohm", "S", "lor", "advK", "pres", "difK", "advM", "difM", "cmpM")
# Written only by a driven run (Phase 3): the bulk forces' work, u.F (-drive) and
# B.S (-bdrive). A run without one reads it as zero.
OPTIONAL = ("injK", "injM")


def load_budget(path):
    """{'N','Re','dh','nbw','vol','t':[...],'rows':{term:[[cols] per probe]}} or None."""
    head, vol, rows, ts = {}, None, {k: [] for k in TERMS + OPTIONAL}, []
    with open(path) as f:
        for line in f:
            if line.startswith("# tg_mhd"):
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        head[k] = v
            elif line.startswith("# vol"):
                vol = [float(x) for x in line.split()[2:]]
            elif line.startswith("#") or not line.strip():
                continue
            else:
                p = line.split()
                if len(p) < 3 or p[1] not in rows:
                    continue
                try:
                    vals = [float(x) for x in p[2:]]
                    t = float(p[0])
                except ValueError:
                    continue                      # a torn last line of a live run
                if vol is not None and len(vals) != len(vol):
                    continue
                if p[1] == TERMS[0]:
                    ts.append(t)
                rows[p[1]].append(vals)
    n = min(len(rows[k]) for k in TERMS)          # whole probes only
    if not ts or n == 0 or vol is None:
        return None
    for k in OPTIONAL:                            # absent, or torn: zero
        if len(rows[k]) < n:
            rows[k] = [[0.0] * len(vol) for _ in range(n)]
    return {"N": int(head["N"]), "Re": float(head["Re"]), "dh": float(head["delta/h"]),
            "nbw": int(head["written"]), "vol": vol, "t": ts[:n],
            "rows": {k: v[:n] for k, v in rows.items()}}


def within(cols, r):
    """The share of a row within r cells of the walls -- within_cells on the written
    layers, never reaching into the last column, which is the rest of the box."""
    nl = len(cols) - 1
    if r >= nl - 0.5:
        raise ValueError("budget.dat stops at %d layers, %.2f cells asked" % (nl, r))
    if r <= 0.0:
        return 0.0
    if r < 0.5:
        return cols[0] * r / 0.5
    k = int(math.floor(r + 0.5))
    return sum(cols[:k]) + cols[k] * (r - (k - 0.5))


def series(b, term, m):
    """[(t, [value within m delta])] -- m = None for the whole box."""
    r = None if m is None else m * b["dh"]
    return [(t, [sum(c) if r is None else within(c, r)]) for t, c in zip(b["t"], b["rows"][term])]


def layer(b, t0, t1, m):
    """Every term integrated over [t0, t1] within m delta, and the two energies'
    changes over it; None if the run does not cover the window."""
    if b["t"][0] > t0 + 1e-2 or b["t"][-1] < t1 - 1e-2:
        return None
    a, z = max(t0, b["t"][0]), min(t1, b["t"][-1])
    out = {k: T.trapz_window(series(b, k, m), a, z)[0] for k in TERMS + OPTIONAL if k not in ("eK", "eM")}
    for e in ("eK", "eM"):
        q = series(b, e, m)
        out["d" + e] = T.lerp(q, z)[0] - T.lerp(q, a)[0]
    # T is what feeds the layer besides the stretching and a forced run's source,
    # which is zero within its envelope's gap of the walls (tg_mhd's TGForce).
    out["Tres"] = out["deM"] - out["S"] - out["injM"] + out["ohm"]      # exact by construction
    out["Texp"] = out["advM"] + out["difM"] + out["cmpM"]
    out["Kres"] = out["deK"] - out["lor"] - out["injK"] + out["visc"]   # the kinetic analogue
    out["Kexp"] = out["advK"] + out["pres"] + out["difK"]
    return out


def find(dirs):
    """{(setup, Re): {N: budget}} from every */budget.dat beside a series.dat."""
    runs = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*", "budget.dat"))):
            s = T.load(os.path.join(os.path.dirname(p), "series.dat"))
            b = load_budget(p)
            if s is None or b is None:
                continue
            b["setup"], b["tag"] = s["setup"], s["tag"]
            runs.setdefault((s["setup"], s["Re"]), {}).setdefault(b["N"], b)     # first dir wins
    return runs


def phi_of(la, lb):
    dO = la["ohm"] - lb["ohm"]
    return (la["S"] - lb["S"]) / dO if dO > 0 else None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_dirs", nargs="+")
    ap.add_argument("--window", type=float, nargs=2, default=[2.0, 10.0], metavar=("T0", "T1"))
    ap.add_argument("--m", type=float, default=1.0, help="the layer, in delta (the test uses 1)")
    args = ap.parse_args(argv)
    runs = find(args.runs_dirs)
    if not runs:
        print("no budget.dat found under", " ".join(args.runs_dirs))
        return 2
    t0, t1 = args.window
    rungs = sorted({Re for (s, Re) in runs if s in ("A", "B")})

    print("PER RUN, within %g delta, integrated over t = %g..%g (paper units, per unit box volume)"
          % (args.m, t0, t1))
    print("  %-26s %9s %9s %9s %9s %9s %9s %9s %9s" % (
        "run", "ohm", "S", "T", "dE_M", "T-Texp", "visc", "lor", "K-Kexp"))
    got = {}
    for (s, Re), by_n in sorted(runs.items()):
        for N, b in sorted(by_n.items()):
            la = layer(b, t0, t1, args.m)
            if la is None:
                print("  %-26s does not cover the window" % b["tag"])
                continue
            got[(s, Re, N)] = la
            print("  %-26s %9.3e %+9.2e %+9.2e %+9.2e %+9.2e %9.3e %+9.2e %+9.2e" % (
                b["tag"], la["ohm"], la["S"], la["Tres"], la["deM"], la["Tres"] - la["Texp"],
                la["visc"], la["lor"], la["Kres"] - la["Kexp"]))

    print()
    print("BOX CHECKS over the window (whole box, so every transport should vanish)")
    for (s, Re), by_n in sorted(runs.items()):
        for N, b in sorted(by_n.items()):
            bx = layer(b, t0, t1, None)
            if bx is None:
                continue
            loss = bx["visc"] + bx["ohm"]
            print("  %-26s <S>+<lor> %+.1e, advK %+.1e, advM %+.1e, difK %+.1e, difM %+.1e, pres %+.1e,"
                  " cmpM %+.1e, magnetic closure %+.1e, kinetic %+.1e  (all / int eps)" % (
                      b["tag"], (bx["S"] + bx["lor"]) / loss, bx["advK"] / loss, bx["advM"] / loss,
                      bx["difK"] / loss, bx["difM"] / loss, bx["pres"] / loss, bx["cmpM"] / loss,
                      (bx["Tres"] - bx["Texp"]) / loss, (bx["Kres"] - bx["Kexp"]) / loss))

    print()
    print("THE TEST (GPU/csf3/tg_mhd_budget.sub): phi = (S_A - S_B) / (Omega_A - Omega_B), "
          "within %g delta, t = %g..%g" % (args.m, t0, t1))
    rc, calls = 0, []
    for Re in rungs:
        Ns = sorted(set(runs.get(("A", Re), {})) & set(runs.get(("B", Re), {})))
        Ns = [N for N in Ns if ("A", Re, N) in got and ("B", Re, N) in got]
        if not Ns:
            print("  Re %5g: no A/B pair" % Re)
            rc = 1
            continue
        phis = {N: phi_of(got[("A", Re, N)], got[("B", Re, N)]) for N in Ns}
        nf = Ns[-1]
        la, lb = got[("A", Re, nf)], got[("B", Re, nf)]
        dO = la["ohm"] - lb["ohm"]
        if phis[nf] is None:
            print("  Re %5g N %d: Omega_A - Omega_B = %+.2e is not positive; no phi" % (Re, nf, dO))
            calls.append("none")
            continue
        band = abs(phis[nf] - phis[Ns[-2]]) if len(Ns) > 1 and phis[Ns[-2]] is not None else None
        parts = ((la["S"] - lb["S"]) / dO, (la["Tres"] - lb["Tres"]) / dO, -(la["deM"] - lb["deM"]) / dO)
        if band is None:
            call = "no band"
            rc = 1
        elif phis[nf] - band > 0.5:
            call = "stretching"
        elif phis[nf] + band < 0.5:
            call = "transport"
        else:
            call = "inside the band"
        calls.append(call)
        print("  Re %5g N %s: phi %s; Ohmic B/A %.3f; deficit = stretching %+.3f + transport %+.3f"
              " + storage %+.3f; band %s -> %s" % (
                  Re, "/".join(str(N) for N in Ns), " / ".join("%+.3f" % phis[N] if phis[N] is not None
                                                             else "--" for N in Ns),
                  lb["ohm"] / la["ohm"], parts[0], parts[1], parts[2],
                  "%.3f" % band if band is not None else "--", call))
    if calls and all(c == "stretching" for c in calls):
        verdict = "STRETCHING: the Ohmic deficit is mainly the stretching the no-slip wall removes"
    elif calls and all(c == "transport" for c in calls):
        verdict = "TRANSPORT: the Ohmic deficit is mainly not the stretching -- the conjecture falls"
    else:
        verdict = "UNDECIDED (%s)" % ", ".join(calls)
    print("  verdict: %s" % verdict)

    print()
    print("THE KINETIC SIDE, within %g delta, t = %g..%g, finer grids: what supplies the viscous dissipation"
          % (args.m, t0, t1))
    for Re in rungs:
        for s in ("A", "B"):
            by_n = runs.get((s, Re), {})
            if not by_n or (s, Re, max(by_n)) not in got:
                continue
            la = got[(s, Re, max(by_n))]
            v = la["visc"]
            print("  %s Re %5g: visc %.3e = lor %+.3f + advK %+.3f + pres %+.3f + difK %+.3f - dE_K %+.3f"
                  " + residual %+.3f  (shares of visc)" % (
                      s, Re, v, la["lor"] / v, la["advK"] / v, la["pres"] / v, la["difK"] / v,
                      la["deK"] / v, (la["Kres"] - la["Kexp"]) / v))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
