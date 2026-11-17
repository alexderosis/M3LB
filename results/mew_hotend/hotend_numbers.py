#!/usr/bin/env python3
"""What Murat Kilic's measurements of 2026-11-17 say, as numbers: the viscosity
table, the heater's on-off cycle and the E3D-v6 drawings.

    python3 results/mew_hotend/hotend_numbers.py > results/mew_hotend/hotend_numbers.txt

Standard library only. Three kinds of input, kept apart:
  MEASURED  viscosity_pcl.csv (his spreadsheet, cell for cell); the cycle, 110 ->
            128 C in 13 s with the heater on and back in 53 s with it off; the
            heater's 30 W.
  DRAWN     E3D-Online's drawings (GPL): V6-NOZZLE-175-400 edition 8,
            V6-175-BREAK 6, V6-BLOCK-CARTRIDGE-ALU 8.1. Each dimension below
            names its drawing. The drawing lists nine orifices; the 0.4 mm
            one is fitted (confirmed by A. De Rosis, 2026-11-17), as every
            model in this tree had taken it to be.
  NOMINAL   handbook values, marked where used. The melt's thermal expansion is
            the one that matters and it is NOT verified for PCL: 5e-4 to 7e-4
            per kelvin is the range of polymer melts, 6.4e-4 the value used.
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RGAS = 8.314462618

# ---------------------------------------------------------------- measured
T_LO, T_HI, T_ON, T_OFF, P_HEAT = 110.0, 128.0, 13.0, 53.0, 30.0
FEED = 0.0208                      # mm^3/s, the true feed (Murat Kilic, 2026-11-03)
FIBRE_NOFAN, FIBRE_FAN = 0.011552, 0.0036354   # mm^3/s the fibres carry at 35 mm/s (mew_axi's banner)

# ---------------------------------------------------------------- drawn (mm)
NOZ_A, NOZ_B, NOZ_C, NOZ_D = 0.40, 1.00, 0.60, 2.0     # V6-NOZZLE-175-400: orifice, tip flat, land, bore
NOZ_LEN, NOZ_THREAD, NOZ_TO_HEX, NOZ_TO_CONE = 12.5, 6.0, 7.5, 10.5   # same drawing: lengths from the top
NOZ_HEX_AF, NOZ_NECK = 7.0, 5.0                         # hexagon across flats, the neck under the thread
NOZ_CONE_INCL = 60.0                                    # the bore's cone, included angle (VIEW B)
NOZ_CONE_BASE = 4.5                # NOT dimensioned: the outer cone's base, read off the drawing (+-0.3)
BRK_BORE, BRK_NECK_OD, BRK_NECK_LEN, BRK_LOWER = 2.0, 2.95, 2.1, 5.0   # V6-175-BREAK, S/S 303
BLK = (23.0, 16.0, 11.5)           # V6-BLOCK-CARTRIDGE-ALU, aluminium
BLK_HEATER, BLK_SENSOR, BLK_M6 = 6.0, 3.1, 6.0          # bores: heater and sensor along 16, M6 along 11.5

# ---------------------------------------------------------------- nominal
BETA_MELT = (5.0e-4, 6.4e-4, 7.0e-4)   # 1/K: low, used, high (polymer melts; NOT verified for PCL)
ALPHA_BRASS, ALPHA_SS = 19e-6, 17.3e-6  # 1/K, linear
K_SS = 16.2                             # W/m/K, 303 stainless
RHO_AL, C_AL = 2.70e-3, 0.897           # g/mm^3, J/g/K
RHO_BRASS, C_BRASS = 8.5e-3, 0.38
RHO_SS, C_SS = 8.0e-3, 0.50
RHO_MELT, C_MELT, K_MELT = 1.08e-3, 2.0, 0.20   # the models' own: 1080 kg/m^3, 2000 J/kg/K, 0.20 W/m/K
BULK_MELT = 1.0e9                       # Pa, a polymer melt's bulk modulus (order only)
ALPHA_T_BRASS, ALPHA_T_AL = 3.4e-5, 8.4e-5      # m^2/s, thermal diffusivities
T_AMB = 25.0


def out(s=""):
    print(s)


# ================================================================ 1. viscosity
rows = [r for r in csv.DictReader(l for l in open(os.path.join(HERE, "viscosity_pcl.csv")) if l[0] != "#")]
temps = sorted({float(r["T_C"]) for r in rows})
curve = {T: [(float(r["shear_rate_1_s"]), float(r["viscosity_Pa_s"])) for r in rows if float(r["T_C"]) == T]
         for T in temps}
PLATEAU_MAX = 10.01                 # 1/s: the points that define the plateau

out("1. THE VISCOSITY TABLE (viscosity_pcl.csv: %d temperatures x %d shear rates, %.0f to %.0f 1/s)"
    % (len(temps), len(curve[temps[0]]), curve[temps[0]][0][0], curve[temps[0]][-1][0]))
out("   plateau = the mean over the %d points up to 10 1/s" % sum(1 for g, _ in curve[temps[0]] if g <= PLATEAU_MAX))
out("   T [C]   plateau [Pa s]   spread on it    at 15.8 1/s   largest stress [kPa] at [1/s]   stress at 100 1/s [kPa]")
mu0 = {}
for T in temps:
    pts = curve[T]
    pl = [m for g, m in pts if g <= PLATEAU_MAX]
    mu0[T] = sum(pl) / len(pl)
    stress = [(g * m, g) for g, m in pts]
    smax, gmax = max(stress)
    m158 = [m for g, m in pts if 15 < g < 17][0]
    out("   %5.0f   %10.1f      %+5.1f / %+5.1f %%   %+6.1f %%        %6.2f at %5.1f              %6.2f"
        % (T, mu0[T], 100 * (min(pl) / mu0[T] - 1), 100 * (max(pl) / mu0[T] - 1), 100 * (m158 / mu0[T] - 1),
           smax * 1e-3, gmax, stress[-1][0] * 1e-3))
out("   Up to 10 1/s the melt is Newtonian to within %.1f %% at every temperature."
    % max(100 * abs(m / mu0[T] - 1) for T in temps for g, m in curve[T] if g <= PLATEAU_MAX))
falls = [max(g * m for g, m in curve[T]) / (curve[T][-1][0] * curve[T][-1][1]) for T in temps]
out("   Past its largest value the STRESS falls as the rate rises, %.0f- to %.0f-fold by 100 1/s. A melt in steady"
    % (min(falls), max(falls)))
out("   uniform shear cannot do that, so those readings are not a viscosity (the sample leaving the gap, or")
out("   slipping, is what a rotational rheometer shows this way). Nothing above the largest stress is used here.")

# ================================================================ 2. Arrhenius
xs = [1.0 / (T + 273.15) for T in temps]
ys = [math.log(mu0[T]) for T in temps]
n = len(xs)
xb, yb = sum(xs) / n, sum(ys) / n
sxx = sum((x - xb) ** 2 for x in xs)
slope = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sxx
res = [y - (yb + slope * (x - xb)) for x, y in zip(xs, ys)]
se = math.sqrt(sum(r * r for r in res) / (n - 2) / sxx)
T95 = 4.303                         # Student's t, 97.5 %, 2 degrees of freedom
Ea, Ea_se = slope * RGAS * 1e-3, se * RGAS * 1e-3


def mu_fit(T):
    return math.exp(yb + slope * (1.0 / (T + 273.15) - xb))


out()
out("2. ITS TEMPERATURE DEPENDENCE (ln plateau against 1/T, least squares over the %d temperatures)" % n)
out("   Ea = %.1f kJ/mol, standard error %.1f; with %d points the 95 %% interval is %.0f to %.0f kJ/mol"
    % (Ea, Ea_se, n, Ea - T95 * Ea_se, Ea + T95 * Ea_se))
out("   residuals: " + ", ".join("%+.1f %% at %.0f C" % (100 * (math.exp(r) - 1), T) for r, T in zip(res, temps)))
out("   between neighbours: " + ", ".join(
    "%.1f (%.0f-%.0f C)" % ((ys[i] - ys[i + 1]) / (xs[i] - xs[i + 1]) * RGAS * 1e-3, temps[i], temps[i + 1])
    for i in range(n - 1)) + " kJ/mol")
out("   the models' 31.3 kJ/mol (Kelly et al. 2012, 80-120 C) is %.1f standard errors below it, and inside the"
    % ((Ea - 31.3) / Ea_se))
out("   95 % interval: four temperatures do not exclude it." if Ea - T95 * Ea_se < 31.3 else
    "   95 % interval excludes it.")
out("   the fit at 120 C: %.0f Pa s (measured plateau %.0f; the models use 820)" % (mu_fit(120), mu0[120.0]))
out("   at the cycle's ends: %.0f Pa s at %.0f C (measured plateau %.0f) and %.0f at %.0f C, %.0f C beyond the data:"
    % (mu_fit(T_LO), T_LO, mu0[T_LO], mu_fit(T_HI), T_HI, T_HI - temps[-1]))
out("   a factor %.2f across the cycle (%.2f with 31.3 kJ/mol)"
    % (mu_fit(T_LO) / mu_fit(T_HI),
       math.exp(31.3e3 / RGAS * (1 / (T_LO + 273.15) - 1 / (T_HI + 273.15)))))
for Tc in (60.0, 40.0):
    f = lambda E: math.exp(E * 1e3 / RGAS * (1 / (Tc + 273.15) - 1 / 393.15))
    out("   cooled to %.0f C (an extrapolation, %.0f C below the data) the melt is %.0f times its 120 C viscosity;"
        " 31.3 kJ/mol gave %.0f" % (Tc, temps[0] - Tc, f(Ea), f(31.3)))

# ================================================================ 3. the cycle
period = T_ON + T_OFF
dT = T_HI - T_LO
up, dn = dT / T_ON, dT / T_OFF
cap = P_HEAT / (up + dn)
loss = cap * dn
tau_c = T_OFF / math.log((T_HI - T_AMB) / (T_LO - T_AMB))
out()
out("3. THE HEATER'S CYCLE (%.0f -> %.0f C in %.0f s at %.0f W, back in %.0f s with it off)"
    % (T_LO, T_HI, T_ON, P_HEAT, T_OFF))
out("   period %.0f s, heater on %.1f %% of it: mean power %.2f W" % (period, 100 * T_ON / period, P_HEAT * T_ON / period))
out("   the block warms at %.3f K/s and cools at %.3f K/s" % (up, dn))
out("   as one lump with a steady loss: heat capacity %.1f J/K, loss %.2f W" % (cap, loss))
out("   if the cooling leg is an exponential towards %.0f C its time constant is %.0f s, and its rate runs from"
    % (T_AMB, tau_c))
out("   %.3f to %.3f K/s: within %.0f %% of the straight leg the model uses"
    % ((T_HI - T_AMB) / tau_c, (T_LO - T_AMB) / tau_c, 100 * max((T_HI - T_AMB) / tau_c / dn - 1, 1 - (T_LO - T_AMB) / tau_c / dn)))
out("   time-mean temperature, two straight legs: %.1f C (the models' nozzle is at 120)" % (0.5 * (T_LO + T_HI)))

# ================================================================ 4. the hot end
cone_h = (NOZ_D - NOZ_A) / 2 / math.tan(math.radians(NOZ_CONE_INCL / 2))
bore_len = NOZ_LEN - cone_h - NOZ_C
v_bore = math.pi / 4 * NOZ_D ** 2 * bore_len
v_cone = math.pi * cone_h / 3 * ((NOZ_D / 2) ** 2 + NOZ_D / 2 * NOZ_A / 2 + (NOZ_A / 2) ** 2)
v_land = math.pi / 4 * NOZ_A ** 2 * NOZ_C
v_brk = math.pi / 4 * BRK_BORE ** 2 * BRK_LOWER
v_hot = v_bore + v_cone + v_land + v_brk
v_neck = math.pi / 4 * BRK_BORE ** 2 * BRK_NECK_LEN
out()
out("4. THE HOT END, FROM THE DRAWINGS (the 0.4 mm nozzle)")
out("   tip: orifice %.2f mm, tip flat %.2f mm, land %.2f mm, a %.0f-degree cone %.2f mm tall from the %.1f mm bore"
    % (NOZ_A, NOZ_B, NOZ_C, NOZ_CONE_INCL, cone_h, NOZ_D))
out("        (the models' 0.4 mm bore and 1.0 mm tip face are the drawing's A and B)")
out("   outside: the tip is the end of a cone %.1f mm tall, about %.0f degrees from the axis (its base is not"
    % (NOZ_LEN - NOZ_TO_CONE,
       math.degrees(math.atan((NOZ_CONE_BASE - NOZ_B) / 2 / (NOZ_LEN - NOZ_TO_CONE)))))
out("        dimensioned), under a %.0f mm hexagon %.1f mm tall; the block's underside, %.0f x %.0f mm, is"
    % (NOZ_HEX_AF, NOZ_TO_CONE - NOZ_TO_HEX, BLK[0], BLK[1]))
out("        %.1f mm above the tip if the hexagon seats on it (a disc of the same area: radius %.1f mm;"
    % (NOZ_LEN - NOZ_TO_HEX, math.sqrt(BLK[0] * BLK[1] / math.pi)))
out("        the nozzle sits 8 mm from three of its edges)")
out("   melt at the block's temperature: nozzle bore %.1f + cone %.2f + land %.3f + the heat break's lower %.0f mm"
    % (v_bore, v_cone, v_land, BRK_LOWER))
out("        %.1f = %.1f mm^3 (the %.1f mm neck above holds %.1f more, on a gradient: not counted)"
    % (v_brk, v_hot, BRK_NECK_LEN, v_neck))
out("   the feed takes %.0f min to pass through it (%.0f cycles of the heater)"
    % (v_hot / FEED / 60, v_hot / FEED / period))
alpha_m = K_MELT / (RHO_MELT * 1e6 * C_MELT * 1e3) * 1e0   # m^2/s: k / (rho c_p), rho in kg/m^3, c_p in J/kg/K
tau_melt = (NOZ_D / 2 * 1e-3) ** 2 / (5.783 * alpha_m)
out("   the melt in the %.0f mm bore follows its wall with a lag of %.1f s (R^2 / 5.78 alpha), keeping %.1f %% of"
    % (NOZ_D, tau_melt, 100 / math.sqrt(1 + (2 * math.pi * tau_melt / period) ** 2)))
out("   a %.0f s swing; the brass tip follows the block %.1f s behind (4 L^2 / pi^2 alpha over %.1f mm), and"
    % (period, 4 * (NOZ_LEN * 1e-3) ** 2 / (math.pi ** 2 * ALPHA_T_BRASS), NOZ_LEN))
out("   the block evens out in %.1f s (L^2 / alpha over 12 mm). So the melt leaves at the block's temperature."
    % ((12e-3) ** 2 / ALPHA_T_AL))
# heat capacities of what the drawings show
v_blk = BLK[0] * BLK[1] * BLK[2] - math.pi / 4 * (BLK_M6 ** 2 * BLK[2] + BLK_HEATER ** 2 * BLK[1]
                                                  + BLK_SENSOR ** 2 * BLK[1])
hex_area = math.sqrt(3) / 2 * NOZ_HEX_AF ** 2
v_noz = (math.pi / 4 * 6.0 ** 2 * NOZ_THREAD + math.pi / 4 * NOZ_NECK ** 2 * (NOZ_TO_HEX - NOZ_THREAD)
         + hex_area * (NOZ_TO_CONE - NOZ_TO_HEX)
         + math.pi * (NOZ_LEN - NOZ_TO_CONE) / 12 * (NOZ_CONE_BASE ** 2 + NOZ_CONE_BASE * NOZ_B + NOZ_B ** 2)
         - v_bore - v_cone - v_land)
v_brk_metal = math.pi / 4 * (6.0 ** 2 - BRK_BORE ** 2) * BRK_LOWER
c_parts = (("the block", v_blk * RHO_AL * C_AL), ("the nozzle", v_noz * RHO_BRASS * C_BRASS),
           ("the heat break's lower end", v_brk_metal * RHO_SS * C_SS), ("the melt", v_hot * RHO_MELT * C_MELT))
c_sum = sum(c for _, c in c_parts)
out("   heat capacity of what is drawn: " + ", ".join("%s %.1f" % p for p in c_parts) + " = %.1f J/K" % c_sum)
out("   against the cycle's %.1f J/K: %.0f %% of it. The heater cartridge, the sensor, the clamp screws and a"
    % (cap, 100 * c_sum / cap))
out("   sock, if fitted, are not on the drawings; a bang-bang lump also reads high if the sensor lags the heater.")
a_neck = math.pi / 4 * (BRK_NECK_OD ** 2 - BRK_BORE ** 2)
for Ts in (30.0, 50.0):
    out("   up the heat break's neck (%.2f mm^2 of steel, %.1f mm long) to a heat sink at %.0f C: %.1f W of the %.1f W lost"
        % (a_neck, BRK_NECK_LEN, Ts, K_SS * a_neck * 1e-6 * (0.5 * (T_LO + T_HI) - Ts) / (BRK_NECK_LEN * 1e-3), loss))

# ================================================================ 5. breathing
out()
out("5. THE MELT EXPANDS AND CONTRACTS WITH THE CYCLE, AND THE ONLY WAY OUT IS THE ORIFICE")
out("   (the filament above is solid and driven at a fixed rate; the metal's bore grows too, by 3 alpha)")
net_metal = (3 * ALPHA_BRASS * (v_bore + v_cone + v_land) + 3 * ALPHA_SS * v_brk) / v_hot
breathe = {}
for tag, b in zip(("low", "used", "high"), BETA_MELT):
    bnet = b - net_metal
    B = bnet * v_hot
    breathe[tag] = B
    out("   beta %.1e /K (%s; net of the metal %.2e): %.4f mm^3/K, %.2f mm^3 a cycle = %.0f %% of the %.2f mm^3 fed;"
        % (b, tag, bnet, B, B * dT, 100 * B * dT / (FEED * period), FEED * period))
    out("        out of the orifice %.4f mm^3/s while heating (%.1f x the feed) and %.4f while cooling (%.2f x)"
        % (FEED + B * up, (FEED + B * up) / FEED, FEED - B * dn, (FEED - B * dn) / FEED))
B = breathe["used"]
q_hot, q_cold = FEED + B * up, FEED - B * dn
out("   with the value used: while the block cools the orifice delivers %.4f mm^3/s against the %.4f the no-fan"
    % (q_cold, FIBRE_NOFAN))
out("   fibres carry and the %.4f of the fan's; while it heats, %.4f. Per cycle the tip is left with"
    % (FIBRE_FAN, q_hot))
out("   %.2f mm^3 (no fan) or %.2f (fan) more than the fibres take, %.0f %% and %.0f %% of it arriving in the %.0f s of heating."
    % (FEED * period - FIBRE_NOFAN * period, FEED * period - FIBRE_FAN * period,
       100 * (q_hot - FIBRE_NOFAN) * T_ON / (FEED * period - FIBRE_NOFAN * period),
       100 * (q_hot - FIBRE_FAN) * T_ON / (FEED * period - FIBRE_FAN * period), T_ON))
R = NOZ_A / 2 * 1e-3
theta = math.radians(NOZ_CONE_INCL / 2)
for tag, q, T in (("nominal feed, 120 C", FEED, 120.0), ("heating leg, 110 C", q_hot, T_LO), ("cooling leg, 128 C", q_cold, T_HI)):
    mu = mu_fit(T)
    qm = q * 1e-9
    gdot = 4 * qm / (math.pi * R ** 3)
    dp = 8 * mu * NOZ_C * 1e-3 * qm / (math.pi * R ** 4) \
        + 8 * mu * qm / (3 * math.pi * math.tan(theta)) * (1 / R ** 3 - 1 / (NOZ_D / 2 * 1e-3) ** 3)
    out("   %-20s wall shear rate in the orifice %5.2f 1/s, wall stress %5.2f kPa, pressure over cone and land %.2f bar"
        % (tag + ":", gdot, mu * gdot * 1e-3, dp * 1e-5))
    dp_last = dp
dp_hot = 8 * mu_fit(T_LO) * NOZ_C * 1e-3 * q_hot * 1e-9 / (math.pi * R ** 4) \
    + 8 * mu_fit(T_LO) * q_hot * 1e-9 / (3 * math.pi * math.tan(theta)) * (1 / R ** 3 - 1 / (NOZ_D / 2 * 1e-3) ** 3)
out("   so the orifice stays on the Newtonian plateau (largest stress measured: %.0f kPa), and the pressure that"
    % (min(max(g * m for g, m in curve[T]) for T in temps) * 1e-3))
out("   drives the heating leg compresses the melt by %.4f mm^3 (bulk modulus %.0f GPa, order only): %.1f %% of"
    % (v_hot * dp_hot / BULK_MELT, BULK_MELT * 1e-9, 100 * v_hot * dp_hot / BULK_MELT / (B * dT)))
out("   what the cycle moves. The melt's compressibility does not smooth the pulse.")
out("   NOT CHECKED: that the filament holds (a drive that slips, or melt creeping back past the filament in")
out("   the heat break, would take part of the pulse), and the expansion coefficient itself.")
out()
out("6. WHAT GOES INTO THE MODELS")
out("   mew_axi -Ea %.1f -cycle %.0f %.0f %.0f %.0f -breathe %.4f   (mu stays 820 Pa s at 120 C)"
    % (Ea, T_LO, T_HI, T_ON, T_OFF, B))
RUN_B = 0.0293                      # what the runs of 2026-11-17 were given (a hand estimate made before this script)
out("   the runs of 2026-11-17 in results/mew_axi (R4_heat_Ea41_cyc*) were given -breathe %.4f, %.1f %% below that:"
    % (RUN_B, 100 * (1 - RUN_B / B)))
out("   out of the orifice %.4f mm^3/s while heating and %.4f while cooling."
    % (FEED + RUN_B * up, FEED - RUN_B * dn))
