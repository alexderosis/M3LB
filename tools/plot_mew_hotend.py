#!/usr/bin/env python3
"""One figure for the measurements of 2026-11-17 (results/mew_hotend) and what the
r-z model makes of them: the viscosity table, its temperature dependence, the
heater's on-off cycle, what that cycle does to the flow out of the orifice, and
the melt hanging below the tip with the fibres' flow taken away (mew_axi -cycle
-breathe -jet) and how far it comes down.

    python3 tools/plot_mew_hotend.py results/mew_hotend/mew_hotend.png

Reads results/mew_hotend/viscosity_pcl.csv and, for the last two panels, the
series of results/mew_axi/R4_heat_Ea41[_cyc[_ph13]]_jet116 (without the jet if
those are missing). The numbers written on it are computed here from the same
inputs as results/mew_hotend/hotend_numbers.py; -breathe is the runs' 0.0293.
Needs numpy and matplotlib.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOT = os.path.join(ROOT, "results", "mew_hotend")
AXI = os.path.join(ROOT, "results", "mew_axi")

INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
SERIES = ["#2a78d6", "#1baf7a", "#eda100"]                 # steady, from switch-on, from switch-off: as plot_mew_axi cycle
TEMPS = {90.0: "#f0a070", 100.0: "#e0703e", 110.0: "#b8461b", 120.0: "#6e2a10"}   # one hue, darker = hotter
RGAS = 8.314462618
T_LO, T_HI, T_ON, T_OFF = 110.0, 128.0, 13.0, 53.0
FEED, BREATHE = 0.0208, 0.0293
FIBRE_NOFAN, FIBRE_FAN = 0.011552, 0.0036354


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8, which="both")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def cycle_T(t, phase=0.0):
    tc = np.mod(t + phase, T_ON + T_OFF)
    return np.where(tc < T_ON, T_LO + (T_HI - T_LO) * tc / T_ON, T_HI - (T_HI - T_LO) * (tc - T_ON) / T_OFF)


def cycle_Q(t, phase=0.0):
    tc = np.mod(t + phase, T_ON + T_OFF)
    return np.where(tc < T_ON, FEED + BREATHE * (T_HI - T_LO) / T_ON, FEED - BREATHE * (T_HI - T_LO) / T_OFF)


def main(out):
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(HOT, "viscosity_pcl.csv")) if l[0] != "#")]
    curve = {}
    for r in rows:
        curve.setdefault(float(r["T_C"]), []).append((float(r["shear_rate_1_s"]), float(r["viscosity_Pa_s"])))
    temps = sorted(curve)
    mu0 = {T: np.mean([m for g, m in curve[T] if g <= 10.01]) for T in temps}
    x = np.array([1.0 / (T + 273.15) for T in temps])
    y = np.log([mu0[T] for T in temps])
    slope, icpt = np.polyfit(x, y, 1)
    Ea = slope * RGAS * 1e-3
    fit = lambda T: np.exp(icpt + slope / (np.asarray(T) + 273.15))
    old = lambda T: 820.0 * np.exp(31.3e3 / RGAS * (1.0 / (np.asarray(T) + 273.15) - 1.0 / 393.15))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=SURF)
    (a, b, c), (d, e, f) = axes

    # a: the table
    for T in temps:
        g = np.array([p[0] for p in curve[T]])
        m = np.array([p[1] for p in curve[T]])
        k = int(np.argmax(g * m))                     # the largest stress: past it the readings are not a viscosity
        a.plot(g[:k + 1], m[:k + 1], "-o", color=TEMPS[T], lw=1.8, ms=5, mec=SURF, mew=0.8)
        a.plot(g[k:], m[k:], "--", color=TEMPS[T], lw=1.0)
        a.plot(g[k + 1:], m[k + 1:], "o", color=SURF, ms=5, mec=TEMPS[T], mew=1.2)
        a.text(0.93, mu0[T], "%.0f °C  %.0f Pa s" % (T, mu0[T]), fontsize=8, color=INK2, ha="right", va="center")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(0.2, 130)
    a.set_ylim(8, 4000)
    a.axvspan(20, 130, color=GRID, alpha=0.55, lw=0)
    a.text(52, 2600, "stress falls as the\nrate rises: not a\nviscosity, not used", fontsize=7.5, color=INK2, ha="center", va="top")
    a.set_xlabel("shear rate [1/s]", color=INK2, fontsize=8.5)
    a.set_ylabel("viscosity [Pa s]", color=INK2, fontsize=8.5)
    a.set_title("a  measured: Newtonian to 10 1/s at every temperature", loc="left", fontsize=9.5, color=INK)

    # b: temperature dependence
    TT = np.linspace(85, 132, 200)
    b.set_xlim(83, 141)
    b.axvspan(T_LO, T_HI, color=GRID, alpha=0.55, lw=0)
    b.text(0.5 * (T_LO + T_HI), 2650, "the heater's\ncycle", fontsize=7.5, color=INK2, ha="center", va="top")
    b.plot(TT, old(TT), color=MUTED, lw=1.4, ls="--", label="31.3 kJ/mol through 820 Pa s (used until now)")
    b.plot(TT, fit(TT), color=INK, lw=1.6, label="fit to the four plateaus: %.1f kJ/mol" % Ea)
    for T in temps:
        b.plot([T], [mu0[T]], "o", color=TEMPS[T], ms=7, mec=SURF, mew=1.0, zorder=5)
    b.plot([T_HI], [fit(T_HI)], "o", color=SURF, ms=7, mec=INK, mew=1.2, zorder=5)
    b.annotate("%.0f Pa s\nat %.0f °C\n(extrapolated)" % (fit(T_HI), T_HI), (T_HI, fit(T_HI)), xytext=(9, 4),
               textcoords="offset points", fontsize=7.5, color=INK2, ha="left", va="bottom")
    b.annotate("%.0f Pa s at %.0f °C" % (mu0[T_LO], T_LO), (T_LO, mu0[T_LO]), xytext=(-8, -16),
               textcoords="offset points", fontsize=7.5, color=INK2, ha="right")
    b.set_yscale("log")
    b.set_ylim(500, 2800)
    b.set_yticks([500, 700, 1000, 1500, 2000, 2500])
    b.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    b.minorticks_off()
    b.set_xlabel("temperature [°C]", color=INK2, fontsize=8.5)
    b.set_ylabel("plateau viscosity [Pa s]", color=INK2, fontsize=8.5)
    b.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left")
    b.set_title("b  it stiffens faster on cooling than assumed; x%.2f across the cycle"
                % (fit(T_LO) / fit(T_HI)), loc="left", fontsize=9.5, color=INK)

    # c, d: the cycle and the outflow, two periods
    t = np.linspace(0, 2 * (T_ON + T_OFF) - 1e-6, 2000)
    for ax in (c, d):
        for k in range(2):
            ax.axvspan(k * (T_ON + T_OFF), k * (T_ON + T_OFF) + T_ON, color="#eb6834", alpha=0.13, lw=0)
    c.plot(t, cycle_T(t), color=INK, lw=1.8)
    c.axhline(120, color=MUTED, lw=1.0, ls="--")
    c.text(64, 120.4, "as modelled", fontsize=7.5, color=INK2, ha="right", va="bottom")
    c.text(T_ON + 2.5, 129.4, "heater on: %.0f s at 30 W" % T_ON, fontsize=7.5, color=INK2, ha="left", va="top")
    c.text(T_ON + T_OFF / 2, 108.6, "heater off: %.0f s" % T_OFF, fontsize=7.5, color=INK2, ha="center", va="bottom")
    c.set_ylim(108, 130)
    c.set_xlim(0, 2 * (T_ON + T_OFF))
    c.set_xlabel("time [s]", color=INK2, fontsize=8.5)
    c.set_ylabel("heater block [°C]", color=INK2, fontsize=8.5)
    c.set_title("c  measured: the nozzle is not at a steady 120 °C", loc="left", fontsize=9.5, color=INK)

    d.plot(t, cycle_Q(t), color=INK, lw=1.8)
    for q, lab in ((FEED, "fed by the filament"), (FIBRE_NOFAN, "carried by the fibres, no fan"),
                   (FIBRE_FAN, "carried by the fibres, fan")):
        d.axhline(q, color=MUTED, lw=1.0, ls="--")
        d.text(131, q + 0.0012, lab, fontsize=7.5, color=INK2, ha="right", va="bottom")
    d.set_ylim(0, 0.07)
    d.set_xlim(0, 2 * (T_ON + T_OFF))
    d.set_xlabel("time [s]", color=INK2, fontsize=8.5)
    d.set_ylabel("melt leaving the orifice [mm$^3$/s]", color=INK2, fontsize=8.5)
    d.set_title("d  estimated: the melt in the hot end expands and contracts", loc="left", fontsize=9.5, color=INK)

    # e, f: the r-z model, with the fibres' flow taken at the apex if those runs exist
    stems = ["R4_heat_Ea41", "R4_heat_Ea41_cyc", "R4_heat_Ea41_cyc_ph13"]
    jet = all(os.path.exists(os.path.join(AXI, s + "_jet116_series.csv")) for s in stems)
    labels = ["steady feed at 120 °C", "the cycle, from switch-on", "the cycle, from switch-off"]
    for s, lab, col in zip(stems, labels, SERIES):
        p = os.path.join(AXI, s + ("_jet116" if jet else "") + "_series.csv")
        if not os.path.exists(p):
            continue
        ser = np.genfromtxt(p, delimiter=",", names=True)
        e.plot(ser["t_s"], ser["gap_mm3"], color=col, lw=1.8, label=lab)
        f.plot(ser["t_s"], ser["z_low_mm"], color=col, lw=1.8)
    e.set_xlabel("time since the field and feed start [s]", color=INK2, fontsize=8.5)
    e.set_ylabel("melt hanging below the tip face [mm$^3$]", color=INK2, fontsize=8.5)
    e.set_ylim(0, None)
    e.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    f.set_xlabel("time since the field and feed start [s]", color=INK2, fontsize=8.5)
    f.set_ylabel("lowest point of the melt above the collector [mm]", color=INK2, fontsize=8.5)
    f.set_ylim(0, 5.6)
    what = "it grows only while the heater is on" if jet else "no jet"
    e.set_title("e  model, fibres' flow taken away: " + what, loc="left", fontsize=9.5, color=INK)
    f.set_title("f  model: how far the drop comes down", loc="left", fontsize=9.5, color=INK)

    for ax in axes.ravel():
        style(ax)
    fig.suptitle("Melt electrowriting: what the viscosity table and the heater's cycle change (PCL, 0.4 mm E3D-v6 nozzle, "
                 "0.0208 mm$^3$/s)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.030, "a-c measured (Z. M. Kilic, 17 Nov 2026).  d an estimate: 50.6 mm$^3$ of melt in the hot end and a "
             "nominal expansion of 6.4e-4 /K, not verified for PCL.", fontsize=8, color=INK2)
    fig.text(0.01, 0.010, "e-f the r-z model (50 µm cells, heat, 40.9 kJ/mol). The jet is not resolved: the no-fan fibres' "
             "flow, 0.0116 mm$^3$/s, is taken out at the drop's apex.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
