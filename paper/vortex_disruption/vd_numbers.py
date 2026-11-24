#!/usr/bin/env python3
"""Every number the vortex-disruption letter quotes.

    python3 paper/vortex_disruption/vd_numbers.py         # prints, and writes vd_numbers.txt

Reads results/Q_vortex_disruption/ through tools/vortex_disruption_map.py and
tools/vortex_disruption_campaign.py -- the same reading, crossing, fit and
spacing bound the tools print -- and prints the quantities the text cites,
section by section. A number in main.tex that
vd_numbers.txt does not contain is a number to distrust. Everything here is
computed from tracked series; nothing is typed in. Pure stdlib.

EVERY RUN READ HERE IS D3Q27 + D3Q7 WITH nz = 1 (the letter names one lattice
pair). The map is gpu_campaign/map (Rm = 2000..16000, the registered CSF3 job)
together with d3q27/map (Rm = 250..1000, and two more fields at 2000 and 4000,
GPU/csf3/vortex_disruption_d3q27.sub); the slices, controls, grid check, exact
check, gallery and Fig. 1 pair are d3q27/; the field-off control and the twin of
the production run are d3q27_laptop/ (the Kokkos driver, -lat d3q27). The D2Q9 +
D2Q5 runs elsewhere in results/Q_vortex_disruption are a record and are not read.

Not called numbers.py: that name shadows the standard library's `numbers`,
which matplotlib imports, so figures.py beside it would fail (CLAUDE.md, on
paper/tg_mhd).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import vortex_disruption_map as vm         # noqa: E402
import vortex_disruption_campaign as vc    # noqa: E402

Q = os.path.join(ROOT, "results", "Q_vortex_disruption")
D3 = os.path.join(Q, "d3q27")
LEVEL = 0.8
LEVELS = (0.1, 0.25, 0.5, 0.7, 0.8, 0.9)
OBS = {"Delta": ("dow", lambda v: v), "1-surv": ("surv", lambda v: 1.0 - v)}
X = lambda r: r["M"] ** 2 * r["Rm"]
SIX = (0.625, 1.25, 1.4865, 1.7678, 2.1022, 2.5)
OUT = []


def load(*parts):
    return vc.load(os.path.join(*parts)) or []


def series(*parts):
    p = os.path.join(*parts)
    return vm.read(p) if os.path.isdir(p) else None


def near(x, y):
    return abs(math.log(x / y)) < 0.02


MAP = load(D3, "map") + load(Q, "gpu_campaign", "map")
GRID = load(Q, "gpu_campaign", "grid")
SCAT = load(Q, "gpu_campaign", "scatter")
CENTRE = [r for r in MAP if r["Rm"] == 1000 and r["Re"] == 1000
          and any(near(X(r), x) for x in SIX)]          # where the slices cross
FIXRE = load(D3, "fixRe1000") + CENTRE
FIXRM = load(D3, "fixRm1000") + CENTRE


def say(s=""):
    OUT.append(s)
    print(s)


def thresholds(runs, level, obs="Delta", key="max", by="Rm"):
    col, fn = OBS[obs]
    out, ended = {}, []
    for g in sorted({r[by] for r in runs}):
        pts = []
        for r in (r for r in runs if r[by] == g):
            v = vm.readings(r, col, fn)[key]
            if v is None:
                ended.append((r["Rm"], X(r), r["d"]["t"][-1]))
            else:
                pts.append((r["M"], v))
        got = vm.crossing(pts, level)
        if got:
            out[g] = got
    return out, ended


def fit(mc):
    xy = sorted((g, v[0]) for g, v in mc.items())
    xh = sorted((g, v[1]) for g, v in mc.items())
    return vm.slope(xy), vm.slope_bound(xh)


def row_q(runs, x, obs, key="c3", by="Rm"):
    col, fn = OBS[obs]
    pts = {}
    for r in runs:
        if near(X(r), x):
            v = vm.readings(r, col, fn)[key]
            if v is not None:
                pts[r[by]] = v
    q = vm.slope(sorted(pts.items())) if len(pts) > 1 else None
    return pts, q


def rows(runs):
    xs = []
    for r in runs:
        if not any(near(X(r), x) for x in xs):
            xs.append(X(r))
    return sorted(xs)


def f3(v):
    return "n/a" if v is None else "%+.3f" % v


def pick(runs, rm, x):
    r = [g for g in runs if g["Rm"] == rm and near(X(g), x)]
    return r[0] if r else None


# ------------------------------------------------------------------------------
say("NUMBERS FOR THE VORTEX-DISRUPTION LETTER  (paper/vortex_disruption/vd_numbers.py)")
say("All runs D3Q27 + D3Q7, nz = 1.")
say("")
say("DATA")
for name, runs in (("map (d3q27/map + gpu_campaign/map)", MAP), ("grid (gpu_campaign)", GRID),
                   ("seed (gpu_campaign/scatter)", SCAT), ("Re = 1000, Rm varied", FIXRE),
                   ("Rm = 1000, Re varied", FIXRM)):
    rms = sorted({r["Rm"] for r in runs})
    r0s = sorted({int(r["R0"]) for r in runs})
    say("  %-36s %3d runs  Rm %s  Re %s  R0 %s" % (
        name, len(runs), "/".join("%g" % v for v in rms),
        "/".join("%g" % v for v in sorted({r["Re"] for r in runs})), "/".join(map(str, r0s))))
r0_of = {}
for r in MAP:
    r0_of.setdefault(r["Rm"], set()).add(int(r["R0"]))
say("  map R0 by Rm: %s" % "  ".join("%g: %s" % (k, "/".join(map(str, sorted(v))))
                                     for k, v in sorted(r0_of.items())))
nonfin = [(r["Rm"], X(r), r["d"]["t"][-1]) for r in MAP if vm.ended_early(r)]
say("  non-finite before t = 12 Rm^(1/3): %d of %d map runs: %s" % (
    len(nonfin), len(MAP), ", ".join("Rm %g X %.4g at t %.1f" % v for v in nonfin) or "none"))
all_rm = sorted({r["Rm"] for r in MAP})
say("  Rm span: %g .. %g, a factor %g; %d values" % (all_rm[0], all_rm[-1],
                                                    all_rm[-1] / all_rm[0], len(all_rm)))
say("  fields per Rm: %s" % "  ".join("%g: %d" % (rm, sum(1 for r in MAP if r["Rm"] == rm))
                                      for rm in all_rm))
eps = 0.05
say("  eps = %g lattice units: Mach number eps sqrt(3) = %.3f" % (eps, eps * math.sqrt(3)))
# tau - 1/2 = nu / cs^2: the flow on D3Q27 (cs^2 = 1/3), the field on D3Q7 (cs^2 = 1/4)
tf = min(3 * eps * r["R0"] / r["Re"] for r in MAP)
tm = min(4 * eps * r["R0"] / r["Rm"] for r in MAP)
say("  smallest tau - 1/2 over the map: flow %.2e, field %.2e" % (tf, tm))
nx = sorted({round(X(r), 4) for r in MAP})
gaps = [math.sqrt(b / a) for a, b in zip(nx, nx[1:]) if 1.2 < a < 2.6]
say("  field spacing near the threshold: M ratios %s" % ", ".join("%.3f" % g for g in gaps))

# ------------------------------------------------------------------------------
say("")
say("1. THE WIND-UP: read at t = 3 Rm^(1/3), along each row of fixed M^2 Rm")
for obs in ("1-surv", "Delta"):
    qs = []
    for x in rows(MAP):
        pts, q = row_q(MAP, x, obs)
        if q is not None and len(pts) >= 3:
            qs.append((x, q, len(pts), min(pts), max(pts)))
    say("  %-6s q per row: %s" % (obs, "  ".join("%.4g:%+.3f(%d Rm, %g-%g)" % v for v in qs)))
    small = [q for x, q, n, a, b in qs if x <= 2.2]
    say("  %-6s q range over the rows with M^2 Rm <= 2.1: %+.3f .. %+.3f (%d rows)"
        % (obs, min(small), max(small), len(small)))
    big = [(x, q) for x, q, n, a, b in qs if x > 2.2]
    say("  %-6s rows above: %s" % (obs, "  ".join("%.4g: %+.3f" % v for v in big)))
ratios = []
for r in MAP:
    if X(r) <= 1.3:
        v = vm.readings(r, "surv", lambda v: 1.0 - v)["c3"]
        if v is not None:
            ratios.append(v / (r["M"] ** 2 * r["Rm"] ** (2.0 / 3.0)))
say("  M^2 Rm <= 1.25: (1-surv)/(M^2 Rm^(2/3)) = %.2f .. %.2f over %d runs"
    % (min(ratios), max(ratios), len(ratios)))
xs = rows(MAP)[0]
pk = sorted((r["Rm"], max(r["d"]["bmax"]) / r["Rm"] ** (1 / 3)) for r in MAP if near(X(r), xs))
say("  weakest row M^2 Rm = %.4g: peak |b| / (B0 Rm^(1/3)) = %s  (range %.2f .. %.2f)" % (
    xs, "  ".join("Rm %g: %.2f" % v for v in pk), min(v for _, v in pk), max(v for _, v in pk)))

# ------------------------------------------------------------------------------
say("")
say("2. DESTRUCTION: the field at which the largest Delta crosses %.1f" % LEVEL)
mc, ended = thresholds(MAP, LEVEL)
for rm, (m, h) in sorted(mc.items()):
    say("  Rm %-6g M_c %.4f  M_c^2 Rm %.2f  M_c^2 Rm^(2/3) %.3f  (+/- %.1f %% in M)"
        % (rm, m, m * m * rm, m * m * rm ** (2 / 3), 100 * (math.exp(h) - 1)))
p_all, b_all = fit(mc)
say("  all Rm (%g..%g, %d values): p = %+.3f, spacing bound %.3f" % (
    min(mc), max(mc), len(mc), p_all, b_all))
mcGr, _ = thresholds(GRID, LEVEL)
mcSc, _ = thresholds(SCAT, LEVEL)
for rm in sorted(mcGr):
    say("  finer grid Rm %-6g M_c %.4f against %.4f (%+.1f %%)" % (
        rm, mcGr[rm][0], mc[rm][0], 100 * (mcGr[rm][0] / mc[rm][0] - 1)))
for rm in sorted(mcSc):
    say("  seed 1e-5  Rm %-6g M_c %.4f against %.4f (%+.1f %%)" % (
        rm, mcSc[rm][0], mc[rm][0], 100 * (mcSc[rm][0] / mc[rm][0] - 1)))


def shifted(sub, base):
    d = dict(base)
    d.update(sub)
    return fit(d)[0]


for name, keep in (("all Rm", lambda rm: True), ("registered, Rm 2000..16000", lambda rm: rm >= 2000)):
    base = {k: v for k, v in mc.items() if keep(k)}
    p, b = fit(base)
    g = abs(shifted(mcGr, base) - p)
    s = abs(shifted(mcSc, base) - p)
    U = max(b, g, s)
    say("  %-28s p = %+.3f; spacing %.3f, grid shift %.3f, seed shift %.3f -> U = %.3f;"
        "  |p + 1/2| = %.3f, |p + 1/3| = %.3f" % (name, p, b, g, s, U, abs(p + 0.5),
                                                  abs(p + 1 / 3)))
allx = [m * m * rm for rm, (m, _) in mc.items()]
say("  M_c^2 Rm over all thresholds: %.2f .. %.2f (%d)" % (min(allx), max(allx), len(allx)))
say("  excluded (stopped before t = 12 Rm^(1/3)): %s" % (
    ", ".join("Rm %g X %.3g" % e[:2] for e in ended) or "none"))

# ------------------------------------------------------------------------------
say("")
say("3. THE EXPONENT AGAINST THE LEVEL (largest value in the run), all Rm")
for obs in ("Delta", "1-surv"):
    cells = []
    for lv in LEVELS:
        m_, _ = thresholds(MAP, lv, obs)
        if len(m_) >= 3:
            p, b = fit(m_)
            cells.append("%.2f: %+.3f +/- %.3f (%d Rm)" % (lv, p, b, len(m_)))
    say("  %-6s %s" % (obs, "   ".join(cells)))

# ------------------------------------------------------------------------------
say("")
say("4. THE BRAKING AT THE DESTRUCTION THRESHOLD: 1 - surv at t = 3 Rm^(1/3)")
for x in (1.7678, 2.1022):
    pts, q = row_q(MAP, x, "1-surv")
    say("  row M^2 Rm = %.4g: %s   q = %s" % (x, "  ".join(
        "Rm %g: %.3f" % kv for kv in sorted(pts.items())), f3(q)))

# ------------------------------------------------------------------------------
say("")
say("5. THE WINDOW: where in t = 0 .. 12 Rm^(1/3) the largest Delta falls")
for rm in (8000, 16000):
    for r in sorted((r for r in MAP if r["Rm"] == rm and 1.2 < X(r) < 2.6), key=X):
        T, D = r["d"]["t"], r["d"]["dow"]
        te = T[-1]
        k = max(range(len(D)), key=lambda i: D[i])
        i8 = max(i for i in range(len(T)) if T[i] <= 0.8 * te)
        say("  Rm %-6g X %-7.4g max Delta %.3f at %.2f t_end; Delta(0.8 t_end) %.3f, "
            "Delta(t_end) %.3f" % (rm, X(r), D[k], T[k] / te, D[i8], D[-1]))
late = [r for r in MAP if r["Rm"] == 16000 and 1.2 < X(r) < 2.6]
nlate = sum(1 for r in late if max(range(len(r["d"]["dow"])), key=lambda i: r["d"]["dow"][i])
            >= 0.9 * (len(r["d"]["dow"]) - 1))
say("  Rm 16000, M^2 Rm 1.25..2.5: %d of %d runs have their largest Delta in the last "
    "tenth of the window" % (nlate, len(late)))
diffs = []
for r in SCAT:
    tw = pick(MAP, r["Rm"], X(r))
    diffs.append((X(r), vm.readings(tw, "dow", lambda v: v)["max"],
                  vm.readings(r, "dow", lambda v: v)["max"]))
say("  seed 1e-5 at Rm 8000, largest Delta map -> seeded: %s" % "  ".join(
    "X %.4g: %.3f -> %.3f" % d for d in diffs))
say("  largest single-run change %.3f" % max(abs(a - b) for _, b, a in diffs))
row8 = sorted((X(r), vm.readings(r, "dow", lambda v: v)["max"]) for r in MAP
              if r["Rm"] == 8000 and 1.2 < X(r) < 2.6)
say("  Rm 8000 largest Delta along M^2 Rm 1.25..2.5: %s" % ", ".join("%.2f" % v for _, v in row8))

# ------------------------------------------------------------------------------
say("")
say("6. Re AGAINST Rm: the two slices through Re = Rm = 1000")
for name, runs, by in (("Re = 1000, Rm varied", FIXRE, "Rm"), ("Rm = 1000, Re varied", FIXRM, "Re")):
    if len({r[by] for r in runs}) < 3:
        say("  %-22s not run yet" % name)
        continue
    _, q1 = row_q(runs, 0.625, "Delta", by=by)
    _, q2 = row_q(runs, 0.625, "1-surv", by=by)
    m8, _ = thresholds(runs, 0.8, by=by)
    m9, _ = thresholds(runs, 0.9, by=by)
    p8, b8 = fit(m8)
    p9, b9 = fit(m9)
    say("  %-22s early Delta q %s, early 1-surv q %s; threshold p(0.8) %s +/- %.3f, "
        "p(0.9) %s +/- %.3f" % (name, f3(q1), f3(q2), f3(p8), b8, f3(p9), b9))
    say("  %-22s M_c^2 Rm at 0.8: %s" % (name, "  ".join(
        "%s %g: %.2f" % (by, g, m * m * (1000.0 if by == "Re" else g))
        for g, (m, _) in sorted(m8.items()))))

# ------------------------------------------------------------------------------
say("")
say("7. CONTROLS AND CHECKS (Re = Rm = 2000, R0 = 20, M^2 Rm = 2.5 where there is a field)")
for d, what in (("kin_noseed", "no Lorentz force, no seed"),
                ("kin_ellip1e-3", "no Lorentz force, ellipticity 1e-3"),
                ("kin_ellip1e-2", "no Lorentz force, ellipticity 1e-2")):
    r = series(D3, "controls", d)
    if not r:
        say("  %-36s not run yet" % what)
        continue
    S, T = r["d"]["surv"], r["d"]["t"]
    k = min(range(len(S)), key=lambda i: S[i])
    say("  %-36s surv(%.1f) = %.4f, lowest %.4f at t = %.1f" % (what, T[-1], S[-1], S[k], T[k]))
kept = pick(MAP, 2000, 2.5)
S, T = kept["d"]["surv"], kept["d"]["t"]
k = min(range(len(S)), key=lambda i: S[i])
say("  field kept (map, CUDA)               surv(50) = %.3f, surv(%.1f) = %.4f, lowest %.3f at t = %.1f"
    % (vm.at(kept, 50.0, "surv"), T[-1], S[-1], S[k], T[k]))
off = series(Q, "d3q27_laptop", "field_off_at_50")
if off:
    S, T = off["d"]["surv"], off["d"]["t"]
    k = min(range(len(S)), key=lambda i: S[i])
    say("  field removed at t = 50 (Kokkos)     surv(%.1f) = %.4f, lowest %.3f at t = %.1f"
        % (T[-1], S[-1], S[k], T[k]))
else:
    say("  field removed at t = 50: not run yet")
r = series(D3, "exact", "kinematic_re500_r30")
if r:
    T, C, S = r["d"]["t"], r["d"]["coh"], r["d"]["surv"]
    nu = r["Re"] * (C[-1] ** -0.5 - 1.0) / T[-1]
    say("  kinematic, Re = Rm = %g, R0 = %g: delivered / set viscosity %.4f from coh(%g); "
        "max |surv - 1| %.1e" % (r["Re"], r["R0"], nu, T[-1], max(abs(s - 1) for s in S)))
else:
    say("  exact check: not run yet")
gc = []
for d in sorted(os.listdir(os.path.join(D3, "grid"))) if os.path.isdir(os.path.join(D3, "grid")) else []:
    r = series(D3, "grid", d)
    tw = pick(MAP, r["Rm"], X(r))
    for obs, (col, fn) in OBS.items():
        for key in ("c3", "c6"):
            a, b = vm.readings(r, col, fn)[key], vm.readings(tw, col, fn)[key]
            if a is not None and b is not None:
                gc.append((r["Rm"], obs, key, 100 * (a / b - 1)))
for r in GRID:
    if near(X(r), 1.25):
        tw = pick(MAP, r["Rm"], X(r))
        for obs, (col, fn) in OBS.items():
            a, b = vm.readings(r, col, fn)["c3"], vm.readings(tw, col, fn)["c3"]
            gc.append((r["Rm"], obs, "c3", 100 * (a / b - 1)))
for obs in OBS:
    v = [g for g in gc if g[1] == obs]
    if v:
        say("  wind-up grid check, %-6s at 3 (and 6) Rm^(1/3), M^2 Rm 1.25, 1.5x finer grid: %s;"
            " worst %.2f %%" % (obs, "  ".join("Rm %g %s %+.2f%%" % (a, k, d) for a, _, k, d in v),
                               max(abs(g[3]) for g in v)))

# ------------------------------------------------------------------------------
say("")
say("8. THE TWO CODES AGAINST EACH OTHER")
for other, what in (("twin/gpu64_M", "CUDA code, host build"),
                    ("device_verify/twin_M", "CUDA code, A100")):
    worst = 0.0
    for m in ("0.05", "0"):
        a = series(Q, "gpu_port", "twin", "kokkos_M" + m)
        b = series(Q, "gpu_port", *(other + m).split("/"))
        for col in ("EK", "B2", "bmax", "coh", "surv", "ang", "wmax", "jmax"):
            sc = max(abs(v) for v in a["d"][col]) or 1.0
            worst = max(worst, max(abs(u - v) for u, v in zip(a["d"][col], b["d"][col])) / sc)
    say("  R0 = 10, Re = Rm = 500, t = 0..20: Kokkos against %-22s worst %.1e (0 = identical "
        "to the ten printed digits)" % (what, worst))
tw = series(Q, "d3q27_laptop", "field_kept")
if tw:
    worst = 0.0
    n = min(len(tw["d"]["t"]), len(kept["d"]["t"]))
    for col in ("EK", "B2", "bmax", "coh", "surv", "ang", "wmax", "jmax", "dow"):
        sc = max(abs(v) for v in kept["d"][col]) or 1.0
        worst = max(worst, max(abs(u - v) for u, v in zip(tw["d"][col][:n], kept["d"][col][:n])) / sc)
    say("  production run Re = Rm = 2000, M^2 Rm = 2.5, R0 = 20, t = 0..%.1f: Kokkos (laptop) "
        "against CUDA (A100) worst %.1e of a column's scale" % (tw["d"]["t"][n - 1], worst))
else:
    say("  production twin: not run yet")

# ------------------------------------------------------------------------------
say("")
say("10. ONE FIELD ACROSS Rm (Fig. 1b): M^2 Rm = 2.5, time in Rm^(1/3)")
for rm in all_rm:
    r = pick(MAP, rm, 2.5)
    if r is None or vm.ended_early(r):
        say("  Rm %-6g no complete run" % rm)
        continue
    c = rm ** (1 / 3)
    T, D = r["d"]["t"], r["d"]["dow"]
    k = max(range(len(D)), key=lambda i: D[i])
    say("  Rm %-6g R0 %-3d Delta(3 Rm^1/3) %.3f  S(3 Rm^1/3) %.3f  largest Delta %.3f at %.1f "
        "Rm^1/3  Delta(end) %.3f" % (rm, r["R0"], vm.at(r, 3 * c, "dow"), vm.at(r, 3 * c, "surv"),
                                     D[k], T[k] / c, D[-1]))

say("")
say("11. SAME BRAKING, DIFFERENT FATE (Fig. 1a): M^2 Rm^(2/3) = 0.198, Re = 1000, R0 = 40")
a = series(D3, "pair", "re1000_rm250_r40_x1.25")
b = series(D3, "pair", "re1000_rm4000_r40_x3.15")
if a and b:
    ca, cb = a["Rm"] ** (1 / 3), b["Rm"] ** (1 / 3)
    say("  Rm %g / %g, M^2 Rm %.3g / %.3g, M^2 Rm^(2/3) %.3f / %.3f" % (
        a["Rm"], b["Rm"], X(a), X(b), X(a) / ca, X(b) / cb))
    for upto in (3.0, 5.0):
        d = max(abs(vm.at(a, s * ca, "surv") - vm.at(b, s * cb, "surv"))
                for s in [i * 0.05 for i in range(1, int(round(upto / 0.05)) + 1)])
        say("    max |S(Rm 250) - S(Rm 4000)| for t <= %g Rm^(1/3): %.3f" % (upto, d))
    for s in (3, 6, 12):
        say("    t = %2d Rm^(1/3): S %.3f / %.3f   Delta %.3f / %.3f" % (
            s, vm.at(a, s * ca, "surv"), vm.at(b, s * cb, "surv"),
            vm.at(a, s * ca, "dow"), vm.at(b, s * cb, "dow")))
    say("    largest Delta %.3f / %.3f" % (max(a["d"]["dow"]), max(b["d"]["dow"])))
else:
    say("  not run yet")

say("")
say("12. THE GALLERY (Figs. 4-6): Re = Rm, R0 = 20")
gal = load(D3, "gallery")
for g in gal:
    tw = pick(load(Q, "gpu_campaign", "map"), g["Rm"], X(g))
    if tw:
        dev = max(abs(u - v) for c in ("surv", "dow") for u, v in zip(g["d"][c], tw["d"][c]))
        say("  gallery %s against the map's run: S, Delta within %.1e" % (g["name"], dev))
for rm in (250, 1000, 4000):
    cells = []
    # the map's own runs: at Rm = 4000 the gallery's are the same runs again
    # (equal series, checked above), so counting both would list each field twice
    for r in sorted((r for r in MAP if r["Rm"] == rm and r["name"].startswith("re%d_" % rm)
                     and any(near(X(r), x) for x in (0.625, 1.25, 1.7678, 2.5, 5))), key=X):
        T, D = r["d"]["t"], r["d"]["dow"]
        k = max(range(len(D)), key=lambda i: D[i])
        cells.append("%.4g: %.2f at %.1f" % (X(r), D[k], T[k] / rm ** (1 / 3)))
    say("  Rm %-5d largest Delta (at t / Rm^1/3): %s" % (rm, "   ".join(cells)))
for x in (1.25, 1.7678):
    r = pick(MAP, 4000, x)
    c = 4000 ** (1 / 3)
    say("  Rm 4000, M^2 Rm %-6g Delta at 6 Rm^(1/3) %.3f, at 12 Rm^(1/3) %.3f; S at the end %.3f"
        % (x, vm.at(r, 6 * c, "dow"), vm.at(r, 12 * c, "dow"), r["d"]["surv"][-1]))

open(os.path.join(HERE, "vd_numbers.txt"), "w").write("\n".join(OUT) + "\n")
