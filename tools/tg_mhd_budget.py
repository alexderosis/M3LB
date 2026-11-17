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

POST HOC (2026-11-17, after the verdict; no rule was fixed for it). The test reads
S within delta over ONE window, and over t = 2..10 S_B is negative at every rung,
which is why phi exceeds 1 there. THAT SIGN BELONGS TO THE WINDOW. The last section
prints, for every run, the times after t = 1 at which S within delta and S in the
whole box change sign, and then S/Omega of each box and phi over POST_HOC's
windows. Read it before writing that a no-slip wall "reverses" the stretching: the
layer at the free-slip wall and the whole of either box have S < 0 early in the
decay as well, the no-slip layer's S turns positive between t = 4 and 6, and over
t = 5..10 S_B is positive at Re = 500, 1000 and 2000. What does not depend on the
window is the test: phi clears 1/2 in every one of them. This is the same trap as
tg_mhd_ladder's reading at a peak -- a sign integrated over one window of an
unsteady flow is a statement about the window.

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
# The windows of the POST HOC section: the paper's (paper/tg_mhd/numbers.py, WINDOWS).
POST_HOC = ((2.0, 6.0), (2.0, 10.0), (3.0, 10.0), (4.0, 10.0), (5.0, 10.0))


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


def crossings(q, t_from=1.0):
    """[(t, +1 or -1)]: the times after t_from at which the one-column series
    q = [(t, [v])] changes sign, linear between probes, and the sign it changes TO.
    t_from = 1 leaves out the start, where S is still near zero and flickers."""
    out = []
    for (ta, (a,)), (tb, (b,)) in zip(q, q[1:]):
        if a == 0.0 or (a > 0) == (b > 0):
            continue
        tc = ta + (tb - ta) * a / (a - b)
        if tc >= t_from:
            out.append((tc, 1 if b > 0 else -1))
    return out


def probes_above(qa, qb, t_from=1.0):
    """(n, of): the probes of qb after t_from at which qa - qb > 0, qa interpolated
    to them; a probe outside qa's range is not counted."""
    n = of = 0
    for t, (v,) in qb:
        a = T.lerp(qa, t) if t >= t_from else None
        if a is None:
            continue
        of += 1
        n += a[0] - v > 0
    return n, of


def episodes(q, t_from=1.0, min_len=0.5):
    """[(t_start, +1 or -1)]: the sign of a one-column series from t_from on, with every
    episode shorter than min_len merged into its neighbours, the shortest first. The
    first episode is kept whatever its length (t_from cuts it); the last is judged by
    what the run holds of it, so one cut short by the end of the run is merged too.
    A description for a text, not a measurement: crossings() is the unmerged list."""
    seg = [(t_from, 1 if T.lerp(q, t_from)[0] > 0 else -1)] + crossings(q, t_from)
    end = q[-1][0]
    while True:
        length = [(seg[k + 1][0] if k + 1 < len(seg) else end) - seg[k][0] for k in range(len(seg))]
        short = [k for k in range(1, len(seg)) if length[k] < min_len]
        if not short:
            return seg
        k = min(short, key=lambda j: length[j])
        del seg[k]                                   # its span falls to the episode before
        if k < len(seg) and seg[k][1] == seg[k - 1][1]:
            del seg[k]                               # and the one after, of the same sign, joins it


def crossing_text(q, t_from=1.0):
    """'< 0 at t = 1, then > 0 from 4.53, < 0 from 7.66' for a one-column series."""
    s0 = T.lerp(q, t_from)[0]
    return "%s 0 at t = %g%s" % ("<" if s0 < 0 else ">", t_from, "".join(
        ", %s 0 from %.2f" % (">" if sg > 0 else "<", tc) for tc, sg in crossings(q, t_from)))


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

    print()
    print("POST HOC (no rule was fixed for this; see the docstring): THE SIGN OF S AGAINST TIME, within %g delta"
          % args.m)
    for s in ("A", "B"):
        for Re in rungs:
            for N, b in sorted(runs.get((s, Re), {}).items()):
                print("  %s Re %5g N %4d: S within %g delta %s" % (
                    s, Re, N, args.m, crossing_text(series(b, "S", args.m))))
                print("  %s            the whole box's S %s" % (" " * len("%s Re %5g N %4d" % (s, Re, N)),
                                                                crossing_text(series(b, "S", None))))
    for Re in rungs:
        by_a, by_b = runs.get(("A", Re), {}), runs.get(("B", Re), {})
        if by_a and by_b:
            n, of = probes_above(series(by_a[max(by_a)], "S", args.m), series(by_b[max(by_b)], "S", args.m))
            print("  Re %5g, finer grids: S_A - S_B > 0 at %d of B's %d probes with t >= 1" % (Re, n, of))
    print("  and over other windows (S / Omega within %g delta, finer / coarser grid; phi with its band):" % args.m)
    for w0, w1 in POST_HOC:
        for Re in rungs:
            Ns = sorted(set(runs.get(("A", Re), {})) & set(runs.get(("B", Re), {})))
            L = {N: (layer(runs[("A", Re)][N], w0, w1, args.m), layer(runs[("B", Re)][N], w0, w1, args.m))
                 for N in Ns}
            Ns = [N for N in Ns if L[N][0] is not None and L[N][1] is not None]
            if len(Ns) < 2:
                continue
            ph = [phi_of(*L[N]) for N in Ns[-2:]]
            (lac, lbc), (la, lb) = L[Ns[-2]], L[Ns[-1]]
            print("    t = %g..%g  Re %5g: S_A/Omega_A %+.3f / %+.3f, S_B/Omega_B %+.3f / %+.3f, phi %s" % (
                w0, w1, Re, la["S"] / la["ohm"], lac["S"] / lac["ohm"], lb["S"] / lb["ohm"], lbc["S"] / lbc["ohm"],
                "no deficit" if None in ph else "%+.3f band %.3f" % (ph[1], abs(ph[1] - ph[0]))))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
