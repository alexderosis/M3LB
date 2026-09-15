"""Quantitative comparison: mhd_decay -regime 1..4 against Neffaa, Bos & Schneider (2008).

Column order in the run logs:
  step t/T_e E_kin E_mag E_k/E_m E/A enstrophy <j^2> L_u L_b H_c |<b>| max|b| dv/cl bulk Bn/B
H_c is Eq. (12)'s cos(theta). E/A is already in the paper's units (scaled by (N/2pi)^2)
and uses the ZERO-MEAN gauge over the fluid disc.

THE TIME MAPPING IS PER REGIME, AND GETTING IT WRONG WAS THE LARGEST SINGLE ERROR IN
THE FIRST VERSION OF THIS COMPARISON.

E_u is the TOTAL kinetic energy, (1/2) int |u|^2 d2x over V_f -- Eq. (7) says so
plainly. The velocity scale is therefore u_rms = sqrt(2 E_u / A) with A = pi r^2 =
27.983, NOT sqrt(2 E_u). One eddy turnover is 2r/u_rms in the paper's units and
2R/u0 in ours; those are the same dimensionless turnover, so crossings map
one-for-one and only the SCALE differs:

    regime        I        II       III       IV
    T_e(paper)   57.65    0.229    27.69     31.58      paper time units per crossing

An earlier version used 9.0 for all four. That came from reading Table I's Re column
as the velocity scale -- but that column is the printed formula 2 r sqrt(2 E_u)/nu
taken literally with E_u a total, which carries units of LENGTH and equals the
physical Reynolds number times sqrt(A) = 5.29. The consequence was that the runs
were read 6.4x, 3.1x and 3.5x too LATE for regimes I, III and IV, and 39x too early
for II, which is why the agreement looked far worse than it is.

REYNOLDS, on the same correction. The reference's PHYSICAL Reynolds numbers are 618,
155536, 1287 and 1128 against this tree's 1000 -- so the campaign sits within a
factor 1.3 of regimes III and IV and ABOVE regime I, not 4-8x below as first
reported. Regime II is irreconcilable on any reading of Table I.
"""
import json, math, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(os.path.join(SP, "neffaa_reference.json")))
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}
ALF = {"I": 0.3, "II": 1.9e4, "III": 1.3, "IV": 1.0}

R_PAPER = 19 * math.pi / 20
AREA = math.pi * R_PAPER * R_PAPER
NU, R_CELLS, T_E_STEPS = 0.01525, 0.475 * 321, 6099.0

C_EK, C_EM, C_RATIO, C_EA, C_COS = 2, 3, 4, 5, 10


def tpaper(reg):
    """paper time units per domain crossing = 2r / sqrt(2 E_u / A), E_u = alf/2"""
    return 2 * R_PAPER / math.sqrt(2 * (ALF[reg] * 0.5) / AREA)


def rows(reg):
    p = os.path.join(SP, "regime%d.dat" % {v: k for k, v in ROMAN.items()}[reg])
    if not os.path.exists(p): return []
    out = []
    for l in open(p):
        t = l.split()
        if len(t) >= 16 and t[0].isdigit(): out.append([float(x) for x in t])
    return out


def at(rs, reg, col, tp):
    """column `col` at paper time nearest tp, or None if the run never got there"""
    if not rs: return None
    te = tpaper(reg)
    if tp > rs[-1][1] * te: return None
    return min(rs, key=lambda r: abs(r[1] * te - tp))[col]


def slope(xs, ys):
    n = len(xs)
    if n < 3: return None
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    d = n * sxx - sx * sx
    return (n * sxy - sx * sy) / d if d else None


def cell(v, ref):
    return " %7s %7.3g" % ("--" if v is None else "%.3g" % v,
                           ref if ref is not None else float("nan"))


def main():
    print("=" * 96)
    print("mhd_decay -regime 1..4  vs  Neffaa, Bos & Schneider, Phys. Plasmas 15, 092304 (2008)")
    print("N = 321, r = (19/20)pi, Re = 1000, Pm = 1, seed 99, 50 domain crossings")
    print("=" * 96)

    runs = {r: rows(r) for r in ROMAN.values()}

    print("\nTHE CLOCK. Crossings map one-for-one; the scale is per regime.\n")
    print("  reg   u_rms(paper)  T_e(paper)   50 crossings reach   their t=450 needs")
    print("  " + "-" * 72)
    for reg in ("I", "II", "III", "IV"):
        u = math.sqrt(2 * (ALF[reg] * 0.5) / AREA); te = tpaper(reg)
        print("  %-4s %12.4f %11.3f   t = %-16.0f %6.1f crossings"
              % (reg, u, te, 50 * te, 450 / te))

    print("\nTABLE I -- the initial condition. E_u/E_B and cos(theta) are IMPOSED, so")
    print("agreement is a check on the construction. E/A is an outcome of the spectrum.\n")
    print("  reg   quantity        ours        paper      ratio")
    print("  " + "-" * 52)
    for reg in ("I", "II", "III", "IV"):
        if not runs[reg]: continue
        r0 = runs[reg][0]; t1 = REF["table1"][reg]
        for name, mine, theirs in (("E_u/E_B", r0[C_RATIO], t1["Eu_over_EB"]),
                                   ("cos(theta)", r0[C_COS], t1["cos_theta"]),
                                   ("E/A", r0[C_EA], t1["E_over_A"])):
            print("  %-4s  %-10s %10.4g %10.4g %10.3f"
                  % (reg, name, mine, theirs, mine / theirs if theirs else float("nan")))

    for key, col, lab in (("fig2_top_Eu_over_EB", C_RATIO, "Fig. 2 top   E_u/E_B"),
                          ("fig2_mid_E_over_A",   C_EA,    "Fig. 2 mid   E/A"),
                          ("fig4_cos_theta",      C_COS,   "Fig. 4       cos(theta)")):
        print("\n%s   [corrected clock]" % lab)
        print("  reg      t=40            t=100           t=200           t=450     our reach")
        print("           ours   paper    ours   paper    ours   paper    ours   paper")
        print("  " + "-" * 80)
        for reg in ("I", "II", "III", "IV"):
            if not runs[reg]: continue
            line = "  %-4s" % reg
            for tp in (40, 100, 200, 450):
                line += cell(at(runs[reg], reg, col, tp), REF[key][reg].get(str(tp)))
            print(line + "   t=%.0f" % (runs[reg][-1][1] * tpaper(reg)))

    print("\nFig. 2 bot   E/|H_c|   (the paper: 'minimum absolute value 2 for case III')")
    print("  reg      t=40            t=100           t=200           t=450")
    print("  " + "-" * 66)
    for reg in ("I", "II", "III", "IV"):
        if not runs[reg]: continue
        line = "  %-4s" % reg
        for tp in (40, 100, 200, 450):
            ref = REF["fig2_bot_E_over_absHc"][reg].get(str(tp))
            te = tpaper(reg)
            if tp > runs[reg][-1][1] * te:
                line += cell(None, ref); continue
            b = min(runs[reg], key=lambda r: abs(r[1] * te - tp))
            hc = b[C_COS] * math.sqrt(b[C_EK] * b[C_EM])
            line += cell((b[C_EK] + b[C_EM]) / abs(hc) if hc else None, ref)
        print(line)

    print("\nFig. 6   energy decay.  power law over the paper's own window t in [20, 140];")
    print("late exponential over the last 20 crossings, as alpha in E ~ exp(-2 alpha nu t)")
    print("  reg   exponent  paper      alpha   paper   (circle's Stokes mode: 1.64)")
    print("  " + "-" * 62)
    tc = REF["text_claims"]
    for reg in ("I", "II", "III", "IV"):
        rs = runs[reg]
        if len(rs) < 10: continue
        te = tpaper(reg)
        t = [r[1] * te for r in rs]; e = [r[C_EK] + r[C_EM] for r in rs]
        mid = [(a, b) for a, b in zip(t, e) if 20 <= a <= 140 and b > 0]
        pw = slope([math.log(x[0]) for x in mid], [math.log(x[1]) for x in mid]) if len(mid) > 3 else None
        lat = [(a / te * T_E_STEPS, math.log(b)) for a, b in zip(t, e) if a / te >= 30 and b > 0]
        ex = slope([x[0] for x in lat], [x[1] for x in lat]) if len(lat) > 3 else None
        alpha = (-ex * R_CELLS * R_CELLS / (2 * NU)) / (R_PAPER ** 2) if ex else None
        pref = tc["decay_exponent_I"] if reg == "I" else (tc["decay_exponent_IV"] if reg == "IV" else None)
        aref = tc["late_alpha_I"] if reg == "I" else tc["late_alpha_II_III_IV"]
        f = lambda v: "     n/a" if v is None else "%8.4g" % v
        print("  %-4s  %s  %s   %s  %s" % (reg, f(pw), f(pref), f(alpha), f(aref)))

    print("\n  '--' means the reference time lies beyond where the run reached. Regime II")
    print("  is unreachable throughout: its velocity scale puts t = 450 at 1964 crossings,")
    print("  where the campaign ran 50, so every regime-II entry is out of range.")


if __name__ == "__main__":
    main()
