"""Mo removal-law test. The rule is in mo_test_PREREG.txt and was written first.

Steady surface temperature of a Gaussian source (intensity ~ exp(-r^2/a^2)) on a
semi-infinite body:  dT(r) = dT_peak * g(r/a),  g(rho) = exp(-z) I0(z),
z = rho^2/2,  dT_peak = A P / (2 sqrt(pi) k a).  Standard library only."""
import math, sys

R_GAS, T0, V_SCAN = 8.314, 300.0, 2.0e-3
A_ABS, K_MO = 0.38, 138.0

def g_of(rho):
    z = 0.5 * rho * rho
    if z >= 50.0:  # asymptotic exp(-z) I0(z), 4 terms; < 1e-7 relative here
        return (1 + 1/(8*z) + 9/(128*z*z) + 225/(3072*z**3)) / math.sqrt(2*math.pi*z)
    n = 4000       # (1/pi) int_0^pi exp(-z(1-cos t)) dt, trapezoid, all terms <= 1
    h = math.pi / n
    s = 0.5 * (1.0 + math.exp(-2.0 * z))
    for i in range(1, n):
        s += math.exp(-z * (1.0 - math.cos(i * h)))
    return s * h / math.pi

# one universal table of g on a radial grid in units of a
RHO = [i * 0.002 for i in range(5001)] + [10.0 + i * 0.05 for i in range(1, 1001)]
G = [g_of(r) for r in RHO]

def g_interp(rho):
    if rho <= 10.0:
        i = min(int(rho / 0.002), 4999); f = (rho - RHO[i]) / 0.002
    else:
        i = min(5000 + int((rho - 10.0) / 0.05), len(RHO) - 2); f = (rho - RHO[i]) / 0.05
    return G[i] + f * (G[i + 1] - G[i])

class Cond:
    def __init__(s, name, P, d86_um, meas, scale=1.0):
        s.name, s.P, s.meas = name, P, meas
        s.a = d86_um * 1e-6 / (2.0 * math.sqrt(2.0))
        s.dTp = scale * A_ABS * P / (2.0 * math.sqrt(math.pi) * K_MO * s.a)

    def logF(s, E, eta=0.0):
        """log of 2 a int_0^rhomax exp(-E/(R T)) d rho, at lateral offset y = a*eta."""
        terms = []
        for k in range(len(RHO)):
            rho = math.hypot(RHO[k], eta) if eta else RHO[k]
            T = T0 + s.dTp * (g_interp(rho) if eta else G[k])
            terms.append(-E / (R_GAS * T))
        m = max(terms)
        acc = 0.0
        for k in range(1, len(RHO)):
            acc += 0.5 * (math.exp(terms[k-1] - m) + math.exp(terms[k] - m)) * (RHO[k] - RHO[k-1])
        return m + math.log(2.0 * s.a * acc)

def depth(c, E, lnV0, eta=0.0):           # metres
    return math.exp(lnV0 + c.logF(E, eta)) / V_SCAN

def fit(ci, cj):
    """(E, ln V0) reproducing both measured depths exactly; None if impossible."""
    target = math.log(ci.meas / cj.meas)
    f = lambda E: (ci.logF(E) - cj.logF(E)) - target
    lo, hi = 1e3, 5e6                     # J/mol
    if f(lo) * f(hi) > 0: return None
    for _ in range(70):                   # bisection in log E
        mid = math.sqrt(lo * hi)
        (lo, hi) = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    E = math.sqrt(lo * hi)
    return E, math.log(V_SCAN * ci.meas) - ci.logF(E)

def width(c, E, lnV0, thr=2.5e-6):
    """full groove width where the predicted depth exceeds thr (her flatness)."""
    if depth(c, E, lnV0) < thr: return 0.0
    lo, hi = 0.0, 20.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if depth(c, E, lnV0, mid) >= thr else (lo, mid)
    return 2.0 * lo * c.a

def run(scale, verbose):
    G92  = Cond("G  91.8", 91.8, 142, 11.1e-6, scale)
    G151 = Cond("G  151",  151,  143, 52.1e-6, scale)
    G202 = Cond("G  202",  202,  155, 91.4e-6, scale)
    LG = [Cond("LG3 91.8", 91.8, 300, None, scale), Cond("LG3 151", 151, 305, None, scale),
          Cond("LG3 202", 202, 277, 26.85e-6, scale), Cond("LG5 91.8", 91.8, 424, None, scale),
          Cond("LG5 151", 151, 434, None, scale), Cond("LG5 202", 202, 425, None, scale)]
    LG_OBS = ["none", "none", "26.9/26.8", "none", "none", "0 / 7.4 (marginal)"]
    if verbose:
        print("steady peak T (K), A/k x %.2f:  G %.0f/%.0f/%.0f   LG3 %s   LG5 %s" % (
            scale, T0+G92.dTp, T0+G151.dTp, T0+G202.dTp,
            "/".join("%.0f" % (T0+c.dTp) for c in LG[:3]),
            "/".join("%.0f" % (T0+c.dTp) for c in LG[3:])))
    passes = 0
    for (ci, cj, held) in ((G92, G151, G202), (G92, G202, G151), (G151, G202, G92)):
        r = fit(ci, cj)
        tag = "fit on %s + %s, hold out %s" % (ci.name.strip(), cj.name.strip(), held.name.strip())
        if r is None:
            print("  %s: NO SOLUTION" % tag); continue
        E, lnV0 = r
        dh = depth(held, E, lnV0)
        ok1 = abs(dh / held.meas - 1.0) <= 0.25
        lg = [depth(c, E, lnV0) for c in LG]
        ok2 = all(lg[i] < 2.5e-6 for i in (0, 1, 3, 4))
        ok3 = lg[2] >= 2.5e-6
        ok = ok1 and ok2 and ok3
        passes += ok
        print("  %s:  E = %.0f kJ/mol   -> %s" % (tag, E / 1e3, "PASS" if ok else "FAIL"))
        print("     held-out %-8s pred %6.1f um   meas %5.1f um   (%+.0f %%)  %s" % (
            held.name, dh * 1e6, held.meas * 1e6, 100 * (dh / held.meas - 1), "ok" if ok1 else "MISS"))
        for c, d, obs in zip(LG, lg, LG_OBS):
            want = "removal" if c.name == "LG3 202" else ("--" if c.name == "LG5 202" else "none")
            got = "removal" if d >= 2.5e-6 else "none"
            mark = "" if want == "--" else ("ok" if got == want else "WRONG")
            print("     %-8s pred %8.3f um  (%-7s)   meas %-18s %s" % (c.name, d * 1e6, got, obs, mark))
        if verbose:
            print("     groove widths at 2.5 um: G %s um (meas 77 / 133 / 145)" % " / ".join(
                "%.0f" % (width(c, E, lnV0) * 1e6) for c in (G92, G151, G202)))
    verdict = {3: "EXTENDS", 0: "DOES NOT EXTEND"}.get(passes, "PARTLY")
    print("  => %d of 3 fits pass: %s\n" % (passes, verdict))

print("Mo removal-law test (rule: mo_test_PREREG.txt)\n")
print("== primary: A = 0.38, k = 138 W/m/K ==")
run(1.0, True)
for s in (0.8, 1.25):
    print("== sensitivity: A/k x %.2f ==" % s)
    run(s, False)
