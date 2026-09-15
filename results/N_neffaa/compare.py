"""Quantitative comparison: mhd_decay -regime 1..4 against Neffaa, Bos & Schneider (2008).

Column order in the run logs:
  step t/T_e E_kin E_mag E_k/E_m E/A enstrophy <j^2> L_u L_b H_c |<b>| max|b| dv/cl bulk Bn/B
H_c is Eq. (12)'s cos(theta). E/A is already in the paper's units (scaled by (N/2pi)^2)
and uses the ZERO-MEAN gauge over the fluid disc.

TIME MAPPING. 50 domain crossings is quoted as the paper's t = 450, so t_paper = 9 * (t/T_e)
on the ADVECTIVE clock. That is the clock the run was sized on; the diffusive clocks differ
by Re, which is 1000 here against their 3868-7920, and that is reported rather than hidden.
"""
import json, math, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(os.path.join(SP, "reference.json")))
LOGS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SP, "..", "camp", "logs")
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}
TPAPER = 9.0            # paper time units per domain crossing

def rows(p):
    out = []
    for l in open(p):
        t = l.split()
        if len(t) >= 16 and t[0].isdigit():
            out.append([float(x) for x in t])
    return out

def at(rs, col, tpaper):
    """value of column `col` at the paper time nearest tpaper"""
    if not rs: return None
    best = min(rs, key=lambda r: abs(r[1] * TPAPER - tpaper))
    return best[col], best[1] * TPAPER

def slope(xs, ys):
    n = len(xs)
    if n < 3: return None
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    d = n * sxx - sx * sx
    return (n * sxy - sx * sy) / d if d else None

C_EK, C_EM, C_RATIO, C_EA, C_COS = 2, 3, 4, 5, 10
NU, R_CELLS, RPHYS = 0.01525, 0.475 * 321, 19 * math.pi / 20

def fmt(v, w=10):
    if v is None: return " " * (w - 3) + "n/a"
    return ("%*.4g" % (w, v))

print("=" * 108)
print("mhd_decay -regime 1..4  vs  Neffaa, Bos & Schneider, Phys. Plasmas 15, 092304 (2008)")
print("N = 321, r = (19/20)pi, Re = 1000, Pm = 1, seed 99, 50 crossings = their t = 450")
print("=" * 108)

runs = {}
for k in (1, 2, 3, 4):
    p = os.path.join(LOGS, "regime%d.log" % k)
    runs[k] = rows(p) if os.path.exists(p) else []

# ---------------------------------------------------------------- Table I ---
print("\nTABLE I -- the initial condition. These are IMPOSED, so agreement is a check")
print("on the construction, not a result. E/A is the exception: it is an OUTCOME of")
print("the spectrum and is the one initial quantity that can disagree.\n")
print("  reg   quantity        ours        paper      ratio")
print("  " + "-" * 52)
for k in (1, 2, 3, 4):
    if not runs[k]: continue
    r0 = runs[k][0]; t1 = REF["table1"][ROMAN[k]]
    for name, mine, theirs in (
            ("E_u/E_B", r0[C_RATIO], t1["Eu_over_EB"]),
            ("cos(theta)", r0[C_COS], t1["cos_theta"]),
            ("E/A", r0[C_EA], t1["E_over_A"])):
        ratio = mine / theirs if theirs else float('nan')
        print("  %-4s  %-10s %s %s %10.3f" % (ROMAN[k], name, fmt(mine), fmt(theirs), ratio))

# ------------------------------------------------------- time series -------
for key, col, label in (("fig2_top_Eu_over_EB", C_RATIO, "Fig. 2 top   E_u/E_B"),
                        ("fig2_mid_E_over_A",   C_EA,    "Fig. 2 mid   E/A"),
                        ("fig4_cos_theta",      C_COS,   "Fig. 4       cos(theta)")):
    print("\n%s\n  reg      t=40            t=100           t=200           t=450" % label)
    print("           ours   paper    ours   paper    ours   paper    ours   paper")
    print("  " + "-" * 70)
    for k in (1, 2, 3, 4):
        if not runs[k]: continue
        line = "  %-4s" % ROMAN[k]
        for tp in (40, 100, 200, 450):
            got = at(runs[k], col, tp)
            ref = REF[key][ROMAN[k]].get(str(tp))
            line += " %7.3g %7.3g" % (got[0] if got else float('nan'),
                                      ref if ref is not None else float('nan'))
        print(line)

# ------------------------------------------------- E/|H_c|, Fig. 2 bottom ---
print("\nFig. 2 bot   E/|H_c|   (the paper: 'attains its minimum absolute value 2 for case III')")
print("  reg      t=40            t=100           t=200           t=450")
print("  " + "-" * 70)
for k in (1, 2, 3, 4):
    if not runs[k]: continue
    line = "  %-4s" % ROMAN[k]
    for tp in (40, 100, 200, 450):
        best = min(runs[k], key=lambda r: abs(r[1] * TPAPER - tp))
        ek, em, cs = best[C_EK], best[C_EM], best[C_COS]
        hc = cs * 2.0 * math.sqrt(ek * em) / 2.0      # undo the printed normalisation
        mine = (ek + em) / abs(hc) if hc else float('nan')
        ref = REF["fig2_bot_E_over_absHc"][ROMAN[k]].get(str(tp))
        line += " %7.3g %7.3g" % (mine, ref if ref is not None else float('nan'))
    print(line)

# ------------------------------------------------------------ decay laws ---
print("\nFig. 6   energy decay.  power law fitted over t_paper in [20, 140]; late")
print("exponential over the last 20 crossings, reported as alpha in E ~ exp(-2 alpha nu t)")
print("  reg   exponent  paper      alpha   paper   (circle's Stokes mode: 1.64)")
print("  " + "-" * 62)
tc = REF["text_claims"]
for k in (1, 2, 3, 4):
    if len(runs[k]) < 10: continue
    t = [r[1] * TPAPER for r in runs[k]]
    e = [r[C_EK] + r[C_EM] for r in runs[k]]
    mid = [(a, b) for a, b in zip(t, e) if 20 <= a <= 140 and b > 0]
    pw = slope([math.log(x[0]) for x in mid], [math.log(x[1]) for x in mid]) if len(mid) > 3 else None
    lat = [(a / TPAPER * 6099.0, math.log(b)) for a, b in zip(t, e) if a / TPAPER >= 30 and b > 0]
    ex = slope([x[0] for x in lat], [x[1] for x in lat]) if len(lat) > 3 else None
    alpha = (-ex * R_CELLS * R_CELLS / (2 * NU)) / (RPHYS * RPHYS) if ex else None
    pref = tc["decay_exponent_I"] if k == 1 else (tc["decay_exponent_IV"] if k == 4 else None)
    aref = tc["late_alpha_I"] if k == 1 else tc["late_alpha_II_III_IV"]
    print("  %-4s  %s  %s   %s  %s" % (ROMAN[k], fmt(pw, 8), fmt(pref, 8), fmt(alpha, 8), fmt(aref, 6)))
print("\n  note: the fit windows sit at nu*t/R^2 well beyond the paper's last frame")
print("  (0.0505), because Re = 1000 against their 3868-7920 makes 50 crossings")
print("  about 4x further into diffusive time. Both clocks cannot be matched at once.")
