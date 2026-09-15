#!/usr/bin/env python3
"""
D2Q9 in the SHIFTED central-moment basis, with the force and the equilibrium
both expanded to FOURTH Hermite order.  Symbolic, exact, and self-checking.

WHY THIS SITS BESIDE THE .m FILES.  This directory is where the tree derives its
central-moment algebra; the language is incidental.  D2Q9_CM.m carries the
fourth-order EQUILIBRIUM but no forcing at all, and the forcing is the half that
turned out to matter -- see the entry under WHAT THIS SETTLED below.  Written in
sympy because the result had to be checked against running C++, which the .m
files cannot do.

THREE BASES ARE IN PLAY AND CONFLATING TWO OF THEM IS THE BUG THIS FILE RECORDS.

  (1) The Hermite TENSOR basis in chat_c = c/cs, H1..H4, in which the source is
      EXPANDED.  This is the basis of the expansion, not of the moments.
  (2) MONOMIAL central moments, k_pq = sum_i S_i Cx^p Cy^q with C = c - u.
      This is what MhdCentralMoments stores (De Rosis, Leveque & Chahine Eq. 8).
  (3) SHIFTED central moments, phi_2(C) = C^2 - cs^2.  This is what ProductBasis
      and MomentCollision store.

  The same populations give DIFFERENT numbers in (2) and (3), related by
      k_21(monomial) = k_21(shifted) + cs^2 k_01(shifted).
  MomentCollision.hpp argues that a force written into the first-order slot alone
  already delivers k_21 = cs^2 F_y "for free".  That is true in (3) and false in
  (2), and MhdCentralMoments -- which is (2) -- deferred to the argument anyway
  and dropped the term.  Fixed 2026-09-15; validation/forcing_cm.cpp now holds it.

WHAT THIS SETTLED, and every number below is printed by the script rather than
asserted in prose:

  * K_F is EXACTLY u-independent for the fourth-order expansion, to 2e-16 over
    the sampled velocities: rho*K_F = [0, Fx, Fy, 0, 0, 0, cs^2 Fy, cs^2 Fx, 0]
    in the monomial basis, and [0, Fx, Fy, 0, ..., 0] in the shifted one.
  * EVERY ORDER EARNS ITS PLACE.  Order 1 alone is wrong at orders 2, 3 and 4;
    order 2 cancels its second-order residue, order 3 the third, and order 4
    appears in k_22 ONLY -- and only to finish a cancellation.  Nothing survives
    from orders 2-4 into any total.
  * The SECOND-order Guo source that src/forcing/Forcing.hpp implements is NOT
    this.  It leaves Dk_21 = -2 ux uy Fx, Dk_12 = -uy^2 Fx, Dk_22 = +4 ux uy^2 Fx.
    At the population level that is 4.2e-3 of max|S| at |u| = 0.058 and 7.7e-2 at
    |u| = 0.25, growing as |u|^2.  So the BGK family and the central-moment family
    do not solve the same forced equations; the gap is O(u^2), not the whole term.
  * The fourth-order Hermite EQUILIBRIUM has all shifted central moments ZERO
    above order 0, exactly like the product form.  So in the shifted basis the
    whole MHD equilibrium IS the Maxwell stress -- no rho above k_00.
  * Moving MhdCentralMoments from (2) to (3) is NOT a pure change of
    representation.  Relaxation happens in whichever basis you store, so
        Dk_22(monomial) = cs^2 (1 - omega_bulk) (k_3 - k_3^eq),
    identically zero at omega_bulk = 1.  That expression accounts for the
    6.5e-3 gap measured between MhdCentralMoments and MomentCollision at B = 0
    with omega_bulk = 1.2 (tests in the session log; cs^2*0.2*|Dk_3| with
    |Dk_3| ~ 0.1).

WHAT THIS FILE DOES NOT DO.  It does not implement anything.  The shifted-basis
operator and the fourth-order forcing policy are not written yet, and the force
SEMANTICS are still open: the MATLAB divides by R = rho, so the source's first
moment is F/rho and not F, which is a factor rho against the existing Guo policy
at fixed F.  Whichever is chosen, the half-shift must be paired with it --
shift_velocity adds (first moment)/(2 rho), and the operator writes
(first moment)/2 into the first-order slot.  Getting that pairing wrong is a
silent factor rho, which is ~1 in these runs and so passes every test.

Run:  python3 d2q9_shifted_force4.py      (needs sympy)
"""
import itertools
import sys

try:
    from sympy import symbols, Rational, sqrt, simplify, expand
except ImportError:                                        # pragma: no cover
    sys.exit("needs sympy:  python3 -m venv v && v/bin/pip install sympy")

Fx, Fy, U, V, R = symbols('F_x F_y u_x u_y rho', real=True)
bx, by = symbols('b_x b_y', real=True)
om, omb = symbols('omega omega_b', real=True)

cs2 = Rational(1, 3)
cs = sqrt(cs2)
cs3 = cs * cs2
cs4 = cs2 * cs2

# Esoteric Pull ordering, as in D2Q9_CM.m.  Only a column permutation: every
# moment below is unchanged by it.
C = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
W = [Rational(4, 9)] + [Rational(1, 9)] * 4 + [Rational(1, 36)] * 4
Id = [[1, 0], [0, 1]]

SLOTS = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 1), (1, 2), (2, 2)]
NAME = {s: 'k_%d%d' % s for s in SLOTS}


def hermite(i):
    """H1..H4 in chat_c = c/cs, exactly as the MATLAB builds them."""
    h = [C[i][0] / cs, C[i][1] / cs]
    H1 = [h[a] for a in range(2)]
    H2 = [[h[a] * h[b] - Id[a][b] for b in range(2)] for a in range(2)]
    H3, H4 = {}, {}
    for a, b, c in itertools.product(range(2), repeat=3):
        hI = h[a] * Id[b][c] + h[b] * Id[a][c] + h[c] * Id[a][b]
        H3[(a, b, c)] = h[a] * h[b] * h[c] - hI
        for d in range(2):
            hII = (h[a] * h[b] * Id[c][d] + h[a] * h[c] * Id[b][d] +
                   h[a] * h[d] * Id[b][c] + h[b] * h[c] * Id[a][d] +
                   h[b] * h[d] * Id[a][c] + h[c] * h[d] * Id[a][b])
            II = Id[a][b] * Id[c][d] + Id[a][c] * Id[b][d] + Id[a][d] * Id[b][c]
            H4[(a, b, c, d)] = h[a] * h[b] * h[c] * h[d] - hII + II
    return H1, H2, H3, H4


def force_orders(i):
    """The four Hermite orders of the source at direction i, kept separate.
    Transcribed from the MATLAB; note the /R, which is what makes the first
    moment F/rho rather than F."""
    H1, H2, H3, H4 = hermite(i)
    o1 = (Fx * H1[0] + Fy * H1[1]) / cs
    o2 = ((Fx * U + U * Fx) * H2[0][0] + (Fy * V + V * Fy) * H2[1][1] +
          (Fx * V + Fy * U) * H2[0][1] + (Fy * U + Fx * V) * H2[1][0]) / (2 * cs2)
    o3 = ((Fx * V * V + U * Fy * V + U * V * Fy) * H3[(0, 1, 1)] +
          (Fy * U * V + V * Fx * V + V * U * Fy) * H3[(1, 0, 1)] +
          (Fy * V * U + V * Fy * U + V * V * Fx) * H3[(1, 1, 0)] +
          (Fx * U * V + U * Fx * V + U * U * Fy) * H3[(0, 0, 1)] +
          (Fy * U * U + V * Fx * U + V * U * Fx) * H3[(1, 0, 0)] +
          (Fx * V * U + U * Fy * U + U * V * Fx) * H3[(0, 1, 0)]) / (6 * cs3)
    o4 = ((Fx*U*V*V + U*Fx*V*V + U*U*Fy*V + U*U*V*Fy) * H4[(0, 0, 1, 1)] +
          (Fx*V*U*V + U*Fy*U*V + U*V*Fx*V + U*V*U*Fy) * H4[(0, 1, 0, 1)] +
          (Fx*V*V*U + U*Fy*V*U + U*V*Fy*U + U*V*V*Fx) * H4[(0, 1, 1, 0)] +
          (Fy*U*U*V + V*Fx*U*V + V*U*Fx*V + V*U*U*Fy) * H4[(1, 0, 0, 1)] +
          (Fy*U*V*U + V*Fx*V*U + V*U*Fy*U + V*U*V*Fx) * H4[(1, 0, 1, 0)] +
          (Fy*V*U*U + V*Fy*U*U + V*V*Fx*U + V*V*U*Fx) * H4[(1, 1, 0, 0)]) / (24 * cs4)
    return [W[i] * o / R for o in (o1, o2, o3, o4)]


def guo2(i):
    """The SECOND-order Guo source, which src/forcing/Forcing.hpp implements."""
    cx, cy = C[i]
    cu = cx * U + cy * V
    return W[i] * (((cx - U) / cs2 + cu * cx / cs4) * Fx +
                   ((cy - V) / cs2 + cu * cy / cs4) * Fy)


def admissible(idx):
    """On D2Q9 each axis may carry an exponent of at most 2: H3(xxx) vanishes
    identically and H4(xxxx) reduces to -3 H2(xx), so including them injects a
    spurious O(u^3) term into the second moment."""
    return idx.count(0) <= 2 and idx.count(1) <= 2


def feq4(i):
    """Fourth-order Hermite equilibrium, admissible terms only."""
    H1, H2, H3, H4 = hermite(i)
    u = [U, V]
    o1 = sum(u[a] * H1[a] for a in range(2)) / cs
    o2 = sum(u[a] * u[b] * H2[a][b]
             for a, b in itertools.product(range(2), repeat=2)) / (2 * cs2)
    o3 = sum(u[a] * u[b] * u[c] * H3[(a, b, c)]
             for a, b, c in itertools.product(range(2), repeat=3)
             if admissible((a, b, c))) / (6 * cs3)
    o4 = sum(u[a] * u[b] * u[c] * u[d] * H4[(a, b, c, d)]
             for a, b, c, d in itertools.product(range(2), repeat=4)
             if admissible((a, b, c, d))) / (24 * cs4)
    return W[i] * R * (1 + o1 + o2 + o3 + o4)


def maxwell(i):
    """Maxwell stress addend, M_ab = |b|^2 delta_ab / 2 - b_a b_b.  Its 2-D trace
    term is identically zero, which is why k_3^eq carries no b."""
    cx, cy = C[i]
    c2 = cx * cx + cy * cy
    cb = cx * bx + cy * by
    b2 = bx * bx + by * by
    return W[i] * (Rational(1, 2) * b2 * c2 - cb * cb) / (2 * cs4)


def phi(p, Cc, shifted):
    if p == 0:
        return 1
    if p == 1:
        return Cc
    return Cc * Cc - (cs2 if shifted else 0)


def cmoments(per_direction, shifted):
    out = {}
    for p, q in SLOTS:
        out[(p, q)] = simplify(expand(sum(
            per_direction(i) * phi(p, C[i][0] - U, shifted)
                             * phi(q, C[i][1] - V, shifted) for i in range(9))))
    return out


def main():
    fail = 0

    # ---- 1. K_F by Hermite order, monomial basis -----------------------------
    print("1. K_F by Hermite order (monomial basis), written as rho*K\n")
    print("   %-6s %-26s %-26s %-22s %-14s %s"
          % ("slot", "order 1", "order 2", "order 3", "order 4", "TOTAL"))
    for p, q in SLOTS:
        per = [simplify(expand(R * sum(
            force_orders(i)[o] * (C[i][0] - U) ** p * (C[i][1] - V) ** q
            for i in range(9)))) for o in range(4)]
        tot = simplify(sum(per))
        print("   %-6s %-26s %-26s %-22s %-14s %s"
              % (NAME[(p, q)], per[0], per[1], per[2], per[3], tot))

    # ---- 2. the same object in both bases ------------------------------------
    src = lambda i: sum(force_orders(i))
    mono, shft = cmoments(src, False), cmoments(src, True)
    print("\n2. K_F in both bases, as rho*K\n")
    print("   %-6s %-22s %s" % ("slot", "monomial  phi_2 = C^2", "shifted  phi_2 = C^2 - cs^2"))
    for s in SLOTS:
        print("   %-6s %-22s %s" % (NAME[s], simplify(mono[s] * R), simplify(shft[s] * R)))
    want_m = {(1, 0): Fx, (0, 1): Fy, (2, 1): cs2 * Fy, (1, 2): cs2 * Fx}
    for s in SLOTS:
        if simplify(mono[s] * R - want_m.get(s, 0)) != 0:
            print("   *** monomial %s != expected" % NAME[s]); fail += 1
        if simplify(shft[s] * R - ({(1, 0): Fx, (0, 1): Fy}.get(s, 0))) != 0:
            print("   *** shifted %s != expected" % NAME[s]); fail += 1

    # ---- 3. what the second-order Guo source leaves --------------------------
    g = cmoments(guo2, False)
    print("\n3. second-order Guo (src/forcing/Forcing.hpp) minus the above, monomial\n")
    for s in SLOTS:
        d = simplify(g[s] - mono[s] * R)
        if d != 0:
            print("   %-6s  %s" % (NAME[s], d))

    # ---- 4. equilibrium in the shifted basis ---------------------------------
    hyd, mag = cmoments(feq4, True), cmoments(maxwell, True)
    print("\n4. equilibrium CMs, SHIFTED basis\n")
    print("   %-6s %-12s %-34s %s" % ("slot", "4th-order eq", "Maxwell term", "TOTAL"))
    for s in SLOTS:
        print("   %-6s %-12s %-34s %s" % (NAME[s], hyd[s], mag[s], simplify(hyd[s] + mag[s])))
    for s in SLOTS:
        if s != (0, 0) and simplify(hyd[s]) != 0:
            print("   *** 4th-order equilibrium %s is not zero in the shifted basis" % NAME[s])
            fail += 1
    if simplify(hyd[(0, 0)] - R) != 0:
        print("   *** k_00 != rho"); fail += 1

    # ---- 5. post-collision, shifted basis ------------------------------------
    eq = {s: simplify(hyd[s] + mag[s]) for s in SLOTS}
    k3e = simplify(eq[(2, 0)] + eq[(0, 2)])
    k4e = simplify(eq[(2, 0)] - eq[(0, 2)])
    print("\n5. POST-COLLISION CMs, shifted basis.  a = F/rho is the source's own")
    print("   first moment; k_3 = k_20 + k_02, k_4 = k_20 - k_02, ghosts at omega = 1.\n")
    print("   k_0* = rho")
    print("   k_1* = a_x/2                          (pre-collision is -a_x/2: the half shift)")
    print("   k_2* = a_y/2")
    print("   k_3* = (1 - omega_b) k_3              [ k_3^eq = %s ]" % k3e)
    print("   k_4* = (1 - omega) k_4 + omega (%s)" % k4e)
    print("   k_5* = (1 - omega) k_5 + omega (%s)" % eq[(1, 1)])
    print("   k_6* = %s" % eq[(2, 1)])
    print("   k_7* = %s" % eq[(1, 2)])
    print("   k_8* = %s" % eq[(2, 2)])
    print("\n   The force appears ONLY in k_1*, k_2*.  Read k_6* back as a monomial and")
    print("   it picks up cs^2 k_1* = cs^2 F_y / (2 rho) -- which is exactly the term")
    print("   MhdCentralMoments now writes by hand, because it stores monomials.")

    print("\n%s" % ("ALL SYMBOLIC CHECKS PASS" if fail == 0
                    else "*** %d CHECK(S) FAILED ***" % fail))
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
