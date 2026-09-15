#!/usr/bin/env python3
"""What Eq. (6) implies for E/A and for Re, checked against Table I.

Neither question needs a run: both are properties of the stated initial spectrum
and of the stated definitions, so they are settled analytically here and the
solver is not involved.

    python3 spectrum_check.py

WHY E/A FOLLOWS FROM THE SPECTRUM ALONE. b = curl a, so |b_k|^2 = k^2 |a_k|^2 and
the shell spectrum of a is E_b(k)/k^2. With E_u/E_B = alf fixed by the regime,

    E/A = (1 + alf) * sum_k E_b(k) / sum_k E_b(k)/k^2,

a weighted mean square wavenumber. The paper's box is 2*pi and the mode index is
therefore the wavenumber itself, so no unit conversion enters and the result is
directly comparable with Table I's first column.

WHY Re NEEDS SETTLING. Eq. (7) writes E as an INTEGRAL over V_f, but
Re = 2 r sqrt(2 E_u) / nu is a Reynolds number only if sqrt(2 E_u) is a VELOCITY.
In two dimensions an integral of |u|^2 over an area has units L^4/T^2, whose
square root is L^2/T, and 2 r (L^2/T) / nu then has units of LENGTH rather than
being dimensionless. The two readings cannot both be literal, and which one holds
decides whether this tree's Re = 1000 is 3.9x below the reference or already
above it -- a factor 5.3 either way.
"""
import math

G, K0 = 0.98, 0.75 * math.sqrt(2) * math.pi     # Eq. (6)'s constants
R_PAPER, NU_PAPER = 19 * math.pi / 20, 1e-3
KMAX = 170                                       # 512^2 dealiased, as they ran

# Table I: (E_u/E_B, E/A, Re)
T1 = {"I":   (0.3,    16.0,    3868),
      "II":  (1.9e4,  3.4e5,   7920),
      "III": (1.3,    31.0,    5176),
      "IV":  (1.0,    16.0,    5725)}

# what this tree's own initial condition measures at t = 0, N = 321, kmax = 24
OURS = {"I": 5.486, "II": 7.964e4, "III": 10.61, "IV": 8.507}


def spec(k):
    """Eq. (6) exactly as printed: E(k) ~ k / [gamma + k/k0]^4, i.e. k^-3 at large k."""
    d = G + k / K0
    return k / (d ** 4)


def e_over_a(alf, kmax=KMAX, p=0.0, kmin=1):
    """E/A for E_b(k) = k^p * Eq.(6).  p = 0 is Eq. (6) as written."""
    ks = range(kmin, kmax + 1)
    num = sum(k ** p * spec(k) for k in ks)
    den = sum(k ** (p - 2) * spec(k) for k in ks)
    return (1.0 + alf) * num / den


def main():
    print(__doc__.split("\n\n")[0])
    print("=" * 74)

    # ---- 1. is E/A converged in the spectrum cut? -------------------------
    print("\n1. E/A from Eq. (6), regime I, against the spectrum cut")
    print("   (this is why -kmax is NOT the explanation: it converges)\n")
    print("     kmax       E/A")
    for kmax in (8, 16, 24, 40, 80, 170, 400):
        print("     %4d   %7.2f" % (kmax, e_over_a(0.3, kmax)))

    # ---- 2. Eq. (6) vs Table I, and vs us --------------------------------
    print("\n2. Eq. (6)'s own prediction against Table I, and against this tree")
    print("   (ours measured at the run's kmax = 24, so Eq. (6) is evaluated there too)\n")
    print("   reg      ours   Eq.(6)   ours/Eq.6   Table I   Eq.6/Table I")
    print("   " + "-" * 62)
    for reg, (alf, ea, _re) in T1.items():
        pred24 = e_over_a(alf, 24)
        print("   %-4s %9.4g %8.4g %11.2f %9.4g %14.2f"
              % (reg, OURS[reg], pred24, OURS[reg] / pred24, ea, pred24 / ea))
    print("\n   We sit 1.2-1.3x above Eq. (6) -- finite-mode realisation scatter, and")
    print("   consistent across all four. Eq. (6) itself sits 0.2-0.4x of Table I.")
    print("   So the disagreement is between Table I and Eq. (6), not between us")
    print("   and the paper.")

    # ---- 3. can any reading of Eq. (6) reach Table I? ---------------------
    print("\n3. Four readings of what Eq. (6) describes, at kmax = 170")
    print("   (E/A is a mean square wavenumber, so raising it needs MORE small-scale")
    print("   magnetic power; moving the spectrum onto omega and j removes two powers")
    print("   of k from b and makes it worse)\n")
    variants = (("(a) the ENERGY spectrum E_u, E_b   [what we implement]", 0.0),
                ("(b) the ENSTROPHY / current spectrum Z, J", -2.0),
                ("(c) the per-MODE amplitude of omega, j", -1.0),
                ("(d) the spectrum of the POTENTIALS psi, a", 2.0))
    for name, p in variants:
        row = "   " + name + "\n     "
        for reg, (alf, ea, _re) in T1.items():
            v = e_over_a(alf, KMAX, p)
            row += "%s %8.4g (x%.2f)  " % (reg, v, v / ea)
        print(row)
    print("\n   None reaches Table I: (a) is ~3x low, (d) ~12x high. Note the gauge")
    print("   cannot close it either -- a non-zero mean in a only ever INCREASES A,")
    print("   and zero mean already minimises it, so no gauge raises E/A.")

    # ---- 4. what slope would Table I need? -------------------------------
    print("\n4. Then what spectrum WOULD give Table I's E/A?\n")
    print("   reg    p in E_b = k^p Eq.(6)    implied large-k slope")
    print("   " + "-" * 52)
    ps = []
    for reg, (alf, ea, _re) in T1.items():
        lo, hi = -1.0, 3.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if e_over_a(alf, KMAX, mid) < ea: lo = mid
            else:                             hi = mid
        p = 0.5 * (lo + hi); ps.append(p)
        print("   %-4s %18.3f %20s" % (reg, p, "k^%.2f" % (p - 3)))
    print("\n   mean p = %.3f over four independent rows, spread %.3f."
          % (sum(ps) / len(ps), max(ps) - min(ps)))
    print("   Eq. (6) as written is p = 0, slope k^-3 -- which the paper states in")
    print("   the sentence immediately after the equation. Table I's E/A column wants")
    print("   about k^-2.1 instead.")

    # ---- 5. the Reynolds normalisation -----------------------------------
    print("\n5. Is sqrt(2 E_u) an rms velocity (E_u per unit area) or not?")
    area = math.pi * R_PAPER * R_PAPER
    print("   disc area = %.3f, so the two readings differ by sqrt(area) = %.3f\n"
          % (area, math.sqrt(area)))
    print("   reg   Re quoted   u_rms if MEAN   u_rms if INTEGRAL   physical Re if INTEGRAL")
    print("   " + "-" * 76)
    for reg, (_alf, _ea, re) in T1.items():
        v = re * NU_PAPER / (2 * R_PAPER)
        print("   %-4s %10d %15.4f %19.4f %22.0f"
              % (reg, re, v, v / math.sqrt(area), v / math.sqrt(area) * 2 * R_PAPER / NU_PAPER))
    print("\n   The MEAN reading gives u_rms ~ 0.6-1.3 and reproduces the quoted Re by")
    print("   construction. The INTEGRAL reading gives u_rms ~ 0.12-0.25 and a physical")
    print("   Reynolds number of 731-1497, far too low for the k^-3 turbulence of their")
    print("   Figs. 7-8. So E_u is per unit area and this tree's Re means the same thing.")

    # ---- 6. Table I's Re is not derivable from Table I --------------------
    print("\n6. A caution on Table I. Eq. (10) normalises to E_B = 1/2 and E_u = a^2/2,")
    print("   so E_u/E_B alone would fix E_u -- and therefore Re. It does not:\n")
    print("   reg   E_u from Re   E_u from the ratio   implied E_B")
    print("   " + "-" * 52)
    for reg, (alf, _ea, re) in T1.items():
        eu_re = (re * NU_PAPER / (2 * R_PAPER)) ** 2 / 2
        print("   %-4s %13.4g %20.4g %13.4g" % (reg, eu_re, alf * 0.5, eu_re / alf))
    print("\n   Regime II is off by four orders. So the Re column carries independent")
    print("   information -- each regime is its own realisation and the absolute")
    print("   amplitude was not held fixed across the four -- and it is NOT evidence")
    print("   about the normalisation. This is the second place Table I does not close")
    print("   against the paper's own equations; E/A above is the first.")


if __name__ == "__main__":
    main()
