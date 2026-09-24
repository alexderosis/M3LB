#!/usr/bin/env python3
"""Onset of convection in a layer of fluid with a DENSITY MAXIMUM -- the exact
reference that validation/density_anomaly.cpp brackets.

STDLIB ONLY, like tools/osm_city.py: the system python on the development
machine has no numpy, and a reference that needs a venv to regenerate is a
reference nobody regenerates.

THE PROBLEM (this is the contract with the LBM case -- change one, change both)
  Layer 0 <= z <= 1, rigid no-slip plates, fixed temperatures: theta = 1 on the
  bottom plate, theta = 0 on the top, so the conduction state is theta_b = 1 - z.
  Boussinesq, with the buoyancy (upward acceleration, nondimensional)

      b(theta) = |theta - lam|^q - |theta_ref - lam|^q ,

  i.e. the density falls as theta moves AWAY from lam in either direction. That
  is Gebhart & Mollendorf's rho = rho_m (1 - w |T - T_m|^q) with theta measured
  in units of the plate difference, lam the density maximum's position in that
  scale, and theta_ref dropping out of the linear problem.

      Ra = g w dT^q H^3 / (nu kappa)

  For q = 1 and lam < 0 this IS classical Rayleigh-Benard with a hot bottom, and
  the script refuses to report anything unless it reproduces Ra_c = 1707.762 and
  k_c = 3.117 there. For 0 < lam < 1 the layer below z = 1 - lam is unstably
  stratified and the layer above it stably (penetrative convection, Veronis 1963).

THE MARGINAL EQUATIONS (exchange of stabilities, sigma = 0; Pr drops out)
  Linearise about theta_b, take normal modes exp(i k x):

      (D^2 - k^2)^2 W = Ra k^2 b'(theta_b) Theta ,     (D^2 - k^2) Theta = -W ,
      W = DW = Theta = 0 on z = 0 and z = 1 ,

  with b'(theta) = q |theta - lam|^(q-1) sign(theta - lam). For b' = 1 this
  collapses to Chandrasekhar's (D^2 - k^2)^3 W = -Ra k^2 W.

THE METHOD: SHOOTING. Three solutions leave z = 0 satisfying its three
conditions, with (W'', W''', Theta') = the unit vectors; the marginal Ra is where
the 3x3 matrix of (W, W', Theta) at z = 1 is singular. RK4 on a mesh that is
SPLIT AT THE KINK z_k = 1 - lam and GRADED toward it: for q < 2 the coefficient
b' is continuous but its derivative is infinite there (|s|^0.894816 at the
Gebhart-Mollendorf q), and a uniform RK4 step straddling it drops to low order.
The q = 1 kink is a JUMP in b', which the split handles exactly by evaluating
each side with its own one-sided sign.

USAGE
  python3 tools/penetrative_onset.py --check            # classical limits
  python3 tools/penetrative_onset.py --q 1.894816 --lam 0.5 --k 3.8
  python3 tools/penetrative_onset.py --q 1.894816 --lam 0.5 --kmin
Every result is printed at two mesh resolutions and the difference is the error
bar; nothing is quoted from the finer one alone.
"""
import argparse
import math
import sys

Q_GM = 1.894816          # Gebhart & Mollendorf (1977), pure water, 1 atm


def bprime(z, q, lam, side):
    """b'(theta_b(z)); `side` is the one-sided sign of theta - lam on the
    current segment, so the value AT the kink is the limit from that side."""
    s = abs((1.0 - z) - lam)
    return q * (s ** (q - 1.0)) * side      # 0.0 ** 0.0 == 1.0, which is q = 1


def rhs(z, y, k2, ra, q, lam, side):
    W, W1, W2, W3, T, T1 = y
    return (W1, W2, W3,
            2.0 * k2 * W2 - k2 * k2 * W + ra * k2 * bprime(z, q, lam, side) * T,
            T1,
            k2 * T - W)


def mesh(a, b, n, kink_at):
    """n steps on [a, b], graded toward the end named by kink_at ('a', 'b' or
    None). Cubic grading puts the first step at (b-a)/n^3."""
    if kink_at is None:
        return [a + (b - a) * j / n for j in range(n + 1)]
    if kink_at == 'b':
        return [b - (b - a) * (1.0 - j / n) ** 3 for j in range(n + 1)]
    return [a + (b - a) * (j / n) ** 3 for j in range(n + 1)]


def segments(lam, n):
    zk = 1.0 - lam
    if zk <= 0.0:                        # lam >= 1: whole layer above the kink
        return [(mesh(0.0, 1.0, n, 'a' if zk == 0.0 else None), -1.0)]
    if zk >= 1.0:                        # lam <= 0: whole layer below it
        return [(mesh(0.0, 1.0, n, 'b' if zk == 1.0 else None), +1.0)]
    return [(mesh(0.0, zk, n, 'b'), +1.0), (mesh(zk, 1.0, n, 'a'), -1.0)]


def shoot(ra, k, q, lam, n):
    k2 = k * k
    sols = []
    for e in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        y = [0.0, 0.0, e[0], e[1], 0.0, e[2]]
        for zs, side in segments(lam, n):
            for i in range(len(zs) - 1):
                z0, h = zs[i], zs[i + 1] - zs[i]
                f1 = rhs(z0, y, k2, ra, q, lam, side)
                y2 = [y[j] + 0.5 * h * f1[j] for j in range(6)]
                f2 = rhs(z0 + 0.5 * h, y2, k2, ra, q, lam, side)
                y3 = [y[j] + 0.5 * h * f2[j] for j in range(6)]
                f3 = rhs(z0 + 0.5 * h, y3, k2, ra, q, lam, side)
                y4 = [y[j] + h * f3[j] for j in range(6)]
                f4 = rhs(z0 + h, y4, k2, ra, q, lam, side)
                y = [y[j] + h / 6.0 * (f1[j] + 2 * f2[j] + 2 * f3[j] + f4[j])
                     for j in range(6)]
        sols.append((y[0], y[1], y[4]))          # W, W', Theta at z = 1
    a, b, c = sols
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - b[0] * (a[1] * c[2] - a[2] * c[1])
            + c[0] * (a[1] * b[2] - a[2] * b[1]))


def brent(f, a, b, fa, fb, tol=1e-12):
    """Plain Brent root finder on a bracket [a, b] with f(a) f(b) < 0."""
    if fa * fb > 0:
        raise ValueError("not a bracket")
    c, fc, d = a, fa, b - a
    e = d
    for _ in range(200):
        if fb * fc > 0:
            c, fc, d = a, fa, b - a
            e = d
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2e-16 * abs(b) + 0.5 * tol * abs(b)
        m = 0.5 * (c - b)
        if abs(m) <= tol1 or fb == 0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, qq = 2 * m * s, 1 - s
            else:
                qq, r = fa / fc, fb / fc
                p = s * (2 * m * qq * (qq - r) - (b - a) * (r - 1))
                qq = (qq - 1) * (r - 1) * (s - 1)
            if p > 0:
                qq = -qq
            p = abs(p)
            if 2 * p < min(3 * m * qq - abs(tol1 * qq), abs(e * qq)):
                e, d = d, p / qq
            else:
                d = m
                e = d
        else:
            d = m
            e = d
        a, fa = b, fb
        b += d if abs(d) > tol1 else math.copysign(tol1, m)
        fb = f(b)
    raise RuntimeError("brent did not converge")


def marginal_ra(k, q, lam, n, guess=None, ra_max=1e8):
    """Lowest positive Ra on the marginal curve at wavenumber k: scan upward in
    geometric steps for the first sign change of the determinant, then Brent.
    A `guess` (a neighbouring k's answer) starts the scan just below it; the
    cold scan starts at Ra = 50, far under any onset in this problem family."""
    f = lambda ra: shoot(ra, k, q, lam, n)
    ra_lo, grow = (0.9 * guess, 1.02) if guess else (50.0, 1.08)
    a, fa = ra_lo, f(ra_lo)
    while a < ra_max:
        b = a * grow
        fb = f(b)
        if fa * fb < 0:
            return brent(f, a, b, fa, fb)
        a, fa = b, fb
    raise RuntimeError("no marginal Ra below %g at k = %g" % (ra_max, k))


def critical(q, lam, n, k_lo=1.5, k_hi=8.0):
    """Minimise Ra(k) by golden section; returns (k_c, Ra_c)."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = k_lo, k_hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc = marginal_ra(c, q, lam, n)
    fd = marginal_ra(d, q, lam, n, guess=fc)
    while b - a > 1e-6:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = marginal_ra(c, q, lam, n, guess=fd)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = marginal_ra(d, q, lam, n, guess=fc)
    k = 0.5 * (a + b)
    return k, marginal_ra(k, q, lam, n, guess=min(fc, fd))


def two_meshes(fn, n):
    lo, hi = fn(n), fn(2 * n)
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--q", type=float, default=Q_GM)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--k", type=float, action="append",
                    help="wavenumber k H at which to report the marginal Ra "
                         "(repeatable)")
    ap.add_argument("--kmin", action="store_true",
                    help="also minimise over k (slow: ~40 marginal solves)")
    ap.add_argument("--n", type=int, default=400,
                    help="RK4 steps per segment on the coarse mesh; the fine "
                         "mesh doubles it")
    ap.add_argument("--check", action="store_true",
                    help="reproduce the classical rigid-rigid onset first")
    a = ap.parse_args()

    if a.check:
        # lam < 0 makes b' = 1 exactly: Chandrasekhar's 1707.762 at k = 3.117.
        k, ra = critical(1.0, -0.5, a.n)
        print("classical check (q = 1, lam = -0.5): k_c = %.5f  Ra_c = %.4f"
              % (k, ra))
        ok = abs(ra - 1707.762) < 0.01 and abs(k - 3.117) < 2e-3
        print("  against 1707.762 / 3.117: %s" % ("PASS" if ok else "FAIL"))
        if not ok:
            sys.exit(1)

    print("q = %.6f   lam = %.4f   (density maximum at z = %.4f)"
          % (a.q, a.lam, 1.0 - a.lam))
    for k in (a.k or []):
        lo, hi = two_meshes(lambda n: marginal_ra(k, a.q, a.lam, n), a.n)
        print("  k H = %.6f   Ra = %.6f   (n = %d: %.6f, diff %.2e)"
              % (k, hi, a.n, lo, abs(hi - lo)))
    if a.kmin:
        (k1, r1), (k2, r2) = two_meshes(lambda n: critical(a.q, a.lam, n), a.n)
        print("  critical: k_c H = %.5f  Ra_c = %.5f   (n = %d: k %.5f Ra %.5f)"
              % (k2, r2, a.n, k1, r1))


if __name__ == "__main__":
    main()
