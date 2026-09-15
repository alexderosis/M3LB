#!/usr/bin/env python3
"""
D3Q27 in the SHIFTED central-moment basis, with the body force expanded to SIXTH
Hermite order.  Symbolic, exact, and self-checking.

    python3 d3q27_shifted_force6.py

THE THREE-DIMENSIONAL HALF of d2q9_shifted_force4.py, and the reason it needed
writing: FOURTH ORDER IS A TWO-DIMENSIONAL STATEMENT.  On D2Q9 the product basis
tops out at (2,2), total order 4, so a fourth-order expansion is COMPLETE there
and the 2-D script's title is accurate.  On D3Q27 it tops out at (2,2,2), total
order 6.  Truncating at four leaves four slots wrong -- k_212, k_221, k_122 at
O(u^4) and k_222 at O(u^5) -- and truncating at five still leaves k_222.  The
invariant is not "fourth order"; it is "complete to 2D in D dimensions".

NO sympy.  The sibling file needs it and is therefore unrunnable on a machine
without it (this one, as of 2026-09-15).  The polynomial algebra here is 60 lines
of exact Fraction arithmetic over the six variables (ux, uy, uz, Fx, Fy, Fz), so
every identity below is proved as a polynomial identity rather than sampled at
random points.  A printed 0 means EXACTLY zero, not "below a tolerance".

WHAT IS BEING EXPANDED.  The source is -F . grad_xi f^eq, whose Hermite
coefficients are the fully symmetric tensors a_n = sym(F (x) u^(n-1)).  Collected
per multi-index (p,q,r) with each exponent in {0,1,2} -- which IS the D3Q27
product basis, not a truncation choice -- that is

    S_i = w_i  sum_{pqr}  B_pqr / (p! q! r! cs2^(p+q+r))  H_p(cx) H_q(cy) H_r(cz)

    B_pqr = p Fx ux^(p-1) uy^q uz^r
          + q Fy ux^p uy^(q-1) uz^r
          + r Fz ux^p uy^q uz^(r-1)
          = d/de [ (ux+e Fx)^p (uy+e Fy)^q (uz+e Fz)^r ]_{e=0}

H_0 = 1, H_1 = c, H_2 = c^2 - cs2 per axis.  Per-axis degree cannot exceed 2: on
{-1,0,1} the lattice has H_3 == 0 identically and H_4 == -H_2, so a term with an
exponent of 3 or more either vanishes or ALIASES into a lower moment.  That is
why the index set is closed at 2 and why the maximum order is 3 x 2 = 6.

WHAT THIS SETTLES, all printed by the script rather than asserted in prose:

  * K_F = [0, Fx, Fy, Fz, 0 x 23] EXACTLY, for all u, in the shifted basis.  Not
    to a tolerance and not up to O(Ma^2): the residual polynomial is the zero
    polynomial in all six variables.
  * READ AS MONOMIAL central moments the SAME populations give cs^(2m) F_a --
    3 slots at F_a, 6 at cs^2 F_a, 3 at cs^4 F_a, 15 exact zeros.  This is the
    closed form validation/forcing_cm.cpp hardcodes, derived here instead.
  * THE TRUNCATION LADDER, exactly.  Order 4 leaves
        k_212 = -(2 ux uy uz^2 Fx + ux^2 uz^2 Fy + 2 ux^2 uy uz Fz)
        k_221 = -(2 ux uy^2 uz Fx + 2 ux^2 uy uz Fy + ux^2 uy^2 Fz)
        k_122 = -(uy^2 uz^2 Fx + 2 ux uy uz^2 Fy + 2 ux uy^2 uz Fz)
        k_222 = +8 X,  X = ux uy^2 uz^2 Fx + ux^2 uy uz^2 Fy + ux^2 uy^2 uz Fz
    and order 5 leaves k_222 = -2 X and nothing else.  The -4 ratio between the
    two k_222 residues is exact.
  * THE TREE'S Guo IS THE ORDER-2 TRUNCATION, exactly and symbolically --
    guo_source_raw (src/forcing/Forcing.hpp:28) minus the N=2 sum is the zero
    polynomial.  So the whole ladder is a statement about running code: BGK and
    TRT carry the order-2 residue, which is ~2 u^2 |F| in the third-order slots.
  * D2Q9 IS THE SAME THEOREM one axis down: order 4 is exact there and order 3
    is not, which is what makes "fourth order" correct in 2-D and wrong in 3-D.
  * THE ORDER-6 SOURCE IS UNIQUE.  The basis matrix is a Kronecker product of
    three 3x3 blocks with det 2 at every u, so the 27 conditions K_F = [0,F,0..]
    have exactly one solution and this source IS it.  Nothing may be added.

WHAT THIS FILE DOES NOT DO, and why nothing needs changing because of it.

  It does not implement a forcing policy, and none is wanted.  A central-moment
  operator in the shifted basis never builds source populations: it writes
  k[mi(1,0,0)] = a/2 and lets the basis deliver the rest (see
  MhdCentralMomentsShifted.hpp's banner).  That write IS this source -- the two
  agree to 1.7e-16 on populations through the real operator -- so adding the
  expansion would cost 27 population evaluations per node, change nothing, and
  introduce two new ways to be wrong (the placement, and the half-shift pairing).
  The value of the expansion is as the PROOF that the three-line write is exact,
  and as the instrument that sizes what BGK and TRT give up by not having it.

  It also does not touch the D3Q19 case.  That lattice is not a product lattice
  and has no such basis; see CLAUDE.md.
"""
import itertools
import sys
from fractions import Fraction

NV = 6                                    # ux uy uz Fx Fy Fz
CS2 = Fraction(1, 3)


# ---------------------------------------------------------------------------
# Exact multivariate polynomials over Fraction.  Terms are {exponent tuple: coef}
# with the zero polynomial represented by an empty dict, so `is_zero` is `not t`.
# ---------------------------------------------------------------------------
class P:
    __slots__ = ("t",)

    def __init__(self, t=None):
        self.t = {k: v for k, v in (t or {}).items() if v}

    @staticmethod
    def const(c):
        c = Fraction(c)
        return P({(0,) * NV: c}) if c else P()

    @staticmethod
    def var(i):
        e = [0] * NV
        e[i] = 1
        return P({tuple(e): Fraction(1)})

    def __add__(self, o):
        r = dict(self.t)
        for k, v in o.t.items():
            n = r.get(k, Fraction(0)) + v
            if n:
                r[k] = n
            else:
                r.pop(k, None)
        return P(r)

    def __neg__(self):
        return P({k: -v for k, v in self.t.items()})

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        if isinstance(o, (int, Fraction)):
            return P({k: v * o for k, v in self.t.items()})
        r = {}
        for k1, v1 in self.t.items():
            for k2, v2 in o.t.items():
                k = tuple(a + b for a, b in zip(k1, k2))
                n = r.get(k, Fraction(0)) + v1 * v2
                if n:
                    r[k] = n
                else:
                    r.pop(k, None)
        return P(r)

    __rmul__ = __mul__

    def is_zero(self):
        return not self.t

    def __str__(self):
        if not self.t:
            return "0"
        names = ["ux", "uy", "uz", "Fx", "Fy", "Fz"]
        out = []
        for k in sorted(self.t, reverse=True):
            c = self.t[k]
            mono = "".join(names[i] if e == 1 else "%s^%d" % (names[i], e)
                           for i, e in enumerate(k) if e)
            if not mono:
                out.append(str(c))
            elif c == 1:
                out.append(mono)
            elif c == -1:
                out.append("-" + mono)
            else:
                out.append("%s %s" % (c, mono))
        s = " + ".join(out).replace("+ -", "- ")
        return s


U = [P.var(0), P.var(1), P.var(2)]
F = [P.var(3), P.var(4), P.var(5)]
ONE = P.const(1)


def upow(a, n):
    r = ONE
    for _ in range(n):
        r = r * U[a]
    return r


# ---------------------------------------------------------------------------
# The lattice.  Ordering is irrelevant here -- every quantity below is a sum over
# all velocities -- so the natural product order is used rather than the tree's.
# ---------------------------------------------------------------------------
def lattice(D):
    """velocities and weights for D2Q9 (D=2) or D3Q27 (D=3)"""
    ax = (0, 1, -1)
    w1 = {0: Fraction(2, 3), 1: Fraction(1, 6), -1: Fraction(1, 6)}
    if D == 2:
        vel = [(x, y, 0) for x in ax for y in ax]
        wts = [w1[v[0]] * w1[v[1]] for v in vel]
    else:
        vel = [(x, y, z) for x in ax for y in ax for z in ax]
        wts = [w1[v[0]] * w1[v[1]] * w1[v[2]] for v in vel]
    return vel, wts


def H(p, c):
    """1-D Hermite on the lattice: H_0 = 1, H_1 = c, H_2 = c^2 - cs2. Fraction."""
    return (Fraction(1), Fraction(c), Fraction(c * c) - CS2)[p]


def phi(p, C, shifted):
    """basis function of the shifted (phi_2 = C^2 - cs2) or monomial (C^2) basis"""
    if p == 0:
        return ONE
    if p == 1:
        return C
    return C * C - (P.const(CS2) if shifted else P())


def indices(D):
    """the product index set: every exponent in {0,1,2}, on D axes"""
    if D == 2:
        return [(p, q, 0) for p in range(3) for q in range(3)]
    return [(p, q, r) for p in range(3) for q in range(3) for r in range(3)]


FACT = (1, 1, 2)


def source(D, N):
    """S_i for every velocity, truncated at total Hermite order N."""
    vel, wts = lattice(D)
    out = []
    for i, c in enumerate(vel):
        s = P()
        for (p, q, r) in indices(D):
            n = p + q + r
            if n == 0 or n > N:
                continue
            # B_pqr: F in one slot of the multi-index, u in the others
            e = (p, q, r)
            B = P()
            for a in range(3):
                if e[a] == 0:
                    continue
                term = P.const(e[a]) * F[a]
                for b in range(3):
                    term = term * upow(b, e[b] - (1 if b == a else 0))
                B = B + term
            den = Fraction(FACT[p] * FACT[q] * FACT[r]) * CS2 ** n
            herm = H(p, c[0]) * H(q, c[1]) * H(r, c[2])
            s = s + B * (herm / den)
        out.append(s * wts[i])
    return out


def guo_raw(D):
    """src/forcing/Forcing.hpp:28, transcribed. S_i = w_i [(c-u)/cs2 + (c.u)c/cs4].F

    The axis loop runs over D, not 3.  On D2Q9 the header's cz is identically 0
    and every caller passes uz = 0, so the z term is absent rather than zero-by-
    cancellation: carrying uz and Fz as free symbols here would leave a spurious
    -uz Fz / cs2 and make this check fail for a reason that does not exist on the
    lattice.  (It did, on the first run of this file.)
    """
    vel, wts = lattice(D)
    ics2 = 1 / CS2
    out = []
    for i, c in enumerate(vel):
        cu = P()
        for a in range(D):
            cu = cu + U[a] * Fraction(c[a])
        s = P()
        for a in range(D):
            ba = (P.const(c[a]) - U[a]) * ics2 + cu * (ics2 * ics2 * Fraction(c[a]))
            s = s + ba * F[a]
        out.append(s * wts[i])
    return out


def moments(D, S, shifted):
    """shifted (or monomial) central moments of the source, about u"""
    vel, _ = lattice(D)
    ks = {}
    for e in indices(D):
        acc = P()
        for i, c in enumerate(vel):
            C = [P.const(c[a]) - U[a] for a in range(3)]
            b = phi(e[0], C[0], shifted) * phi(e[1], C[1], shifted) * phi(e[2], C[2], shifted)
            acc = acc + b * S[i]
        ks[e] = acc
    return ks


def target(D, e, shifted):
    """K_F: F_a on first-order slots (shifted); cs^(2m) F_a (monomial)"""
    ones = [a for a in range(3) if e[a] == 1]
    twos = sum(1 for a in range(3) if e[a] == 2)
    if len(ones) != 1:
        return P()
    if shifted and twos:
        return P()
    return F[ones[0]] * (CS2 ** twos if not shifted else Fraction(1))


def label(e, D):
    return "k_%d%d" % e[:2] if D == 2 else "k_%d%d%d" % e


# ---------------------------------------------------------------------------
def main():
    bad = 0
    print(__doc__.split("\n\n")[1].strip())
    print("=" * 78)

    # ---- 1. the identity, exactly ----------------------------------------
    print("\n1. K_F in the SHIFTED basis, order 6, as an exact polynomial identity")
    print("   (every entry is the residual K_F - target; 0 means the zero polynomial)\n")
    S6 = source(3, 6)
    k6 = moments(3, S6, True)
    nz = []
    for e in indices(3):
        d = k6[e] - target(3, e, True)
        if not d.is_zero():
            nz.append((e, d))
    print("     first-order slots   k_100 = %s   k_010 = %s   k_001 = %s"
          % (k6[(1, 0, 0)], k6[(0, 1, 0)], k6[(0, 0, 1)]))
    print("     the other 24 slots  %s" % ("all exactly zero" if not nz else "NONZERO"))
    if nz:
        bad += 1
        for e, d in nz[:6]:
            print("       %s = %s" % (label(e, 3), d))
    print("     verdict: %s" % ("PASS" if not nz else "*** FAIL ***"))

    # ---- 2. the same populations read as monomial moments -----------------
    print("\n2. The SAME populations in the MONOMIAL basis: cs^(2m) F_a")
    print("   (this is the closed form validation/forcing_cm.cpp hardcodes)\n")
    km = moments(3, S6, False)
    census = {0: 0, 1: 0, 2: 0, -1: 0}
    mbad = 0
    for e in indices(3):
        d = km[e] - target(3, e, False)
        if not d.is_zero():
            mbad += 1
        ones = [a for a in range(3) if e[a] == 1]
        twos = sum(1 for a in range(3) if e[a] == 2)
        census[twos if len(ones) == 1 else -1] += 1
    print("     %2d slots at F_a      %2d at cs^2 F_a      %2d at cs^4 F_a      %2d exact zeros"
          % (census[0], census[1], census[2], census[-1]))
    print("     mismatches: %d      verdict: %s" % (mbad, "PASS" if not mbad else "*** FAIL ***"))
    bad += 1 if mbad else 0

    # ---- 3. the truncation ladder ----------------------------------------
    print("\n3. THE LADDER. Slots left wrong by truncating at order N (D3Q27)\n")
    print("     N   wrong slots   which")
    ladder = {}
    for N in range(1, 7):
        kN = moments(3, source(3, N), True)
        wrong = [e for e in indices(3) if not (kN[e] - target(3, e, True)).is_zero()]
        ladder[N] = (kN, wrong)
        names = ", ".join(label(e, 3) for e in wrong)
        print("     %d   %2d            %s" % (N, len(wrong), names if len(wrong) <= 5 else "(many)"))
    if len(ladder[6][1]) or len(ladder[4][1]) != 4 or len(ladder[5][1]) != 1:
        print("     *** FAIL: expected 4 wrong at N=4, 1 at N=5, 0 at N=6")
        bad += 1

    print("\n   The four residues at N = 4, in closed form:")
    k4 = ladder[4][0]
    for e in ((2, 1, 2), (2, 2, 1), (1, 2, 2), (2, 2, 2)):
        print("     %s = %s" % (label(e, 3), k4[e] - target(3, e, True)))
    print("\n   The one residue at N = 5:")
    k5 = ladder[5][0]
    d5 = k5[(2, 2, 2)] - target(3, (2, 2, 2), True)
    print("     k_222 = %s" % d5)
    d4 = k4[(2, 2, 2)] - target(3, (2, 2, 2), True)
    ratio_ok = (d4 + P.const(4) * d5).is_zero()
    print("   ratio k_222(N=4) / k_222(N=5) = -4 exactly: %s" % ("PASS" if ratio_ok else "*** FAIL ***"))
    bad += 0 if ratio_ok else 1

    # ---- 4. the tree's Guo IS the order-2 truncation -----------------------
    print("\n4. src/forcing/Forcing.hpp's guo_source_raw vs the N = 2 truncation")
    print("   (transcribed, then differenced as polynomials -- D3Q27 and D2Q9)\n")
    for D, nm in ((3, "D3Q27"), (2, "D2Q9")):
        g, s2 = guo_raw(D), source(D, 2)
        diff = [a - b for a, b in zip(g, s2)]
        ok = all(d.is_zero() for d in diff)
        print("     %-6s  guo_source_raw - S(N=2)  =  %s     %s"
              % (nm, "0 for all %d velocities" % len(g) if ok else "NONZERO",
                 "PASS" if ok else "*** FAIL ***"))
        bad += 0 if ok else 1
    kg = moments(3, guo_raw(3), True)
    print("\n   So BGK and TRT carry the N = 2 residue. Its largest slots:")
    for e in ((2, 1, 0), (1, 2, 0), (2, 2, 0)):
        print("     %s = %s" % (label(e, 3), kg[e] - target(3, e, True)))

    # HermiteForce4's banner (src/forcing/Forcing.hpp) quotes three of these, in
    # the Fx-only reading -- it was derived with Fy = 0. Confirm that rather than
    # leave two statements of the same fact side by side and unrelated.
    def fx_only(p):
        return P({k: v for k, v in p.t.items() if k[4] == 0 and k[5] == 0})

    banner = {(2, 1, 0): P.const(-2) * U[0] * U[1] * F[0],
              (1, 2, 0): -(U[1] * U[1] * F[0]),
              (2, 2, 0): P.const(4) * U[0] * U[1] * U[1] * F[0]}
    bok = all((fx_only(kg[e] - target(3, e, True)) - want).is_zero()
              for e, want in banner.items())
    print("   at Fy = Fz = 0 these ARE the three residues HermiteForce4's banner")
    print("   quotes (dk_21, dk_12, dk_22):  %s" % ("PASS" if bok else "*** FAIL ***"))
    bad += 0 if bok else 1

    # ---- 5. D2Q9 is the same theorem one axis down ------------------------
    print("\n5. D2Q9: the complete order is 4, not 6 -- which is why the 2-D script")
    print("   is right to be called ...force4 and this one cannot be\n")
    print("     N   wrong slots (D2Q9)")
    for N in range(1, 5):
        kN = moments(2, source(2, N), True)
        wrong = [e for e in indices(2) if not (kN[e] - target(2, e, True)).is_zero()]
        print("     %d   %2d   %s" % (N, len(wrong),
                                      ", ".join(label(e, 2) for e in wrong) or "none"))
    k2d = moments(2, source(2, 4), True)
    ok2 = all((k2d[e] - target(2, e, True)).is_zero() for e in indices(2))
    k2d3 = moments(2, source(2, 3), True)
    ok3 = all((k2d3[e] - target(2, e, True)).is_zero() for e in indices(2))
    print("\n     order 4 complete in 2-D: %s      order 3 complete: %s"
          % ("yes" if ok2 else "NO", "yes" if ok3 else "no (as expected)"))
    if not ok2 or ok3:
        bad += 1

    # ---- 6. uniqueness ----------------------------------------------------
    print("\n6. UNIQUENESS. The basis matrix factorises as a Kronecker product of")
    print("   three 3x3 blocks A[p][c] = phi_p(c - u); det of one block:\n")
    # det of [[phi_p(c-u)]] over c in (0,1,-1), p in (0,1,2), symbolically in ux
    blk = [[phi(p, P.const(c) - U[0], True) for c in (0, 1, -1)] for p in range(3)]
    det = (blk[0][0] * (blk[1][1] * blk[2][2] - blk[1][2] * blk[2][1])
           - blk[0][1] * (blk[1][0] * blk[2][2] - blk[1][2] * blk[2][0])
           + blk[0][2] * (blk[1][0] * blk[2][1] - blk[1][1] * blk[2][0]))
    print("     det = %s      (u-independent, nonzero)" % det)
    duok = str(det) == "-2" or str(det) == "2"
    print("     so det(T) = det(block)^3 and T is invertible at EVERY u: the 27")
    print("     conditions K_F = [0,F,0...] have exactly ONE solution, and section 1")
    print("     shows this source is it. Nothing may be added to it.      %s"
          % ("PASS" if duok else "*** FAIL ***"))
    bad += 0 if duok else 1

    print("\n" + "=" * 78)
    print("ALL CHECKS PASS" if bad == 0 else "*** %d CHECK(S) FAILED ***" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
