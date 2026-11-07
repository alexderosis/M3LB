#!/usr/bin/env python3
"""Figure for demonstrator/mew_jet, the MEW jet as a slender viscous thread.

Reads results/mew_jet/ (the sweeps and profiles mew_jet writes with -out; the
commands are in mew_jet's banner) and draws

    python3 tools/plot_mew_jet.py results/mew_jet/mew_jet.png
    python3 tools/plot_mew_jet.py --lag results/mew_jet/mew_jet_lag.png

The first: (a) the speed at which the jet lands on its own against the current
it carries, nominal and under each unmeasured input changed alone, with the
collector's 35 mm/s; (b) the jet's diameter and (c) its temperature along its
flight, at the current that lands it at 35 mm/s, cooling and held at 120 C.
--lag: the jet behind the moving collector (mew_jet -drag): (a) its side view
at half the current I* that lands it at 35 mm/s on its own, for collectors
1.02 to 4 times faster than its free landing; (b) the lag at 35 mm/s against
the current; (c) the lag against the collector's speed -- each in the mean gap
field and in the field mew_axi computes below the oozing drop at 60 s.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "mew_jet")

INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
ISO, HEAT = "#2a78d6", "#d4572a"
VAR = [("sweep_Nu0p2", "Nu 0.2", "#b3a8ee", "-"), ("sweep_Nu0p8", "Nu 0.8", "#4a3bb0", "-"),
       ("sweep_a25", "apex radius 25 um", "#9fd9c0", "-"), ("sweep_a100", "apex radius 100 um", "#0f6e4c", "-"),
       ("sweep_T60", "jet starts at 60 C", "#eda100", "-"), ("sweep_Ea50", "Ea 50 kJ/mol", "#e87ba4", "-")]


def load(name):
    return np.genfromtxt(os.path.join(RES, name + ".csv"), delimiter=",", names=True)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, which="both")
    ax.set_axisbelow(True)


def main(out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURF, gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    for name, lab, col, ls in VAR:
        s = load(name)
        ax.plot(s["I_nA"], s["U_mm_s"], color=col, ls=ls, lw=1.3, label=lab)
    s = load("sweep_isothermal")
    ax.plot(s["I_nA"], s["U_mm_s"], color=ISO, ls="--", lw=1.6, label="held at 120 C")
    s = load("sweep_nominal")
    ax.plot(s["I_nA"], s["U_mm_s"], color=INK, lw=2.0, label="nominal")
    ax.axhline(35.0, color=MUTED, lw=0.9, ls=":")
    ax.text(0.011, 38, "collector, 35 mm/s", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_xlabel("current carried by the jet [nA]", color=INK2, fontsize=8.5)
    ax.set_ylabel("speed at which the jet lands on its own [mm/s]", color=INK2, fontsize=8.5)
    ax.set_title("a  above the line, lines laid at 35 mm/s would wave", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", labelcolor=INK)
    for a, ttl in zip(axes[1:], ("b  diameter along the flight", "c  temperature along the flight")):
        a.set_title(ttl, loc="left", fontsize=9, color=INK)
    p = load("profile_nominal")
    q = load("profile_isothermal")
    axes[1].plot(p["z_mm"], p["d_um"], color=HEAT, lw=1.8, label="cooling, 0.86 nA")
    axes[1].plot(q["z_mm"], q["d_um"], color=ISO, ls="--", lw=1.6, label="held at 120 C, 0.14 nA")
    axes[1].axhline(20.5, color=MUTED, lw=0.9, ls=":")
    axes[1].text(3.2, 23, "the 20.5 um fibre", fontsize=7.5, color=INK2)
    axes[1].set_ylabel("jet diameter [um]", color=INK2, fontsize=8.5)
    axes[1].set_ylim(0, 105)
    axes[2].plot(p["z_mm"], p["T_C"], color=HEAT, lw=1.8, label="cooling")
    axes[2].plot(q["z_mm"], q["T_C"], color=ISO, ls="--", lw=1.6, label="held at 120 C")
    k = np.argmax(p["mu_Pa_s"] >= 10 * p["mu_Pa_s"][0])
    axes[2].plot([p["z_mm"][k]], [p["T_C"][k]], "o", color=HEAT, ms=5, mec=SURF, mew=1.0)
    axes[2].annotate("viscosity 10x", (p["z_mm"][k], p["T_C"][k]), xytext=(8, 6), textcoords="offset points",
                     fontsize=7.5, color=INK2)
    axes[2].set_ylabel("jet temperature [C]", color=INK2, fontsize=8.5)
    axes[2].set_ylim(0, 130)
    for a in axes[1:]:
        a.set_xlabel("distance from the cone's apex [mm]", color=INK2, fontsize=8.5)
        a.set_xlim(0, 5)
        a.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK)
    for a in axes:
        style(a)
    fig.suptitle("The MEW jet as a slender viscous thread: 0.0116 mm$^3$/s (a 20.5 um fibre at 35 mm/s), "
                 "6.5 kV over 5.5 mm, 5 mm of flight", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


RAMP = ["#bfdcff", "#93c2fb", "#66a5f2", "#3f86df", "#2a6cc0", "#1d519a", "#123873"]  # v_c/U, light to dark


def lag(out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURF, gridspec_kw={"width_ratios": [1.45, 1, 1]})
    ax = axes[0]
    vc = load("drag_nominal_lag_vc")
    L = None
    for k in range(len(vc["vc_mm_s"])):
        p = np.genfromtxt(os.path.join(RES, "drag_nominal_path_%d.csv" % k), delimiter=",", names=True)
        L = p["z_mm"][-1]
        ax.plot(p["x_mm"], L - p["z_mm"], color=RAMP[k], lw=1.8,
                label="%.2f U, %.0f mm/s" % (vc["vc_over_U"][k], vc["vc_mm_s"][k]))
    ax.plot([0], [L], "v", color=INK, ms=7, mec=SURF)
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.annotate("the collector moves this way", (6.6, 4.75), xytext=(2.6, 4.75), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8), va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.3, L + 0.5)
    ax.set_xlabel("distance along the collector's motion [mm]", color=INK2, fontsize=8.5)
    ax.set_ylabel("height above the collector [mm]", color=INK2, fontsize=8.5)
    li = load("drag_nominal_lag_I")
    I_half = float(li["I_nA"][np.argmin(np.abs(li["I_over_Istar"] - 0.5))])
    U = float(vc["vc_mm_s"][0] / vc["vc_over_U"][0])
    ax.set_title("a  the jet at %.2f nA, landing on its own at %.1f mm/s, behind faster collectors" % (I_half, U),
                 loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper right", labelcolor=INK, title="collector", title_fontsize=7.5)
    for name, lab, col, ls in (("drag_nominal", "mean gap field", INK, "-"),
                               ("drag_heat60s", "field below the oozing drop (mew_axi)", HEAT, "--")):
        a = load(name + "_lag_I")
        axes[1].plot(a["I_over_Istar"], a["lag_mm"], color=col, ls=ls, lw=1.8, marker="o", ms=3.5, label=lab)
        b = load(name + "_lag_vc")
        axes[2].plot(b["vc_over_U"], b["lag_mm"], color=col, ls=ls, lw=1.8, marker="o", ms=3.5, label=lab)
    axes[1].set_xlabel("current / the current that lands the jet at 35 mm/s on its own", color=INK2, fontsize=8.5)
    axes[1].set_title("b  the lag at the collector's 35 mm/s", loc="left", fontsize=9, color=INK)
    axes[1].set_xlim(0, 1)
    axes[2].set_xlabel("collector speed / the jet's free-landing speed", color=INK2, fontsize=8.5)
    axes[2].set_title("c  the lag at half that current", loc="left", fontsize=9, color=INK)
    axes[2].set_xlim(1, 4.1)
    for a in axes[1:]:
        a.set_ylabel("lag: landing point behind the cone [mm]", color=INK2, fontsize=8.5)
        a.set_ylim(0, None)
        a.legend(frameon=False, fontsize=7.5, loc="upper right" if a is axes[1] else "lower right", labelcolor=INK)
    for a in axes:
        style(a)
    fig.suptitle("The MEW jet behind the moving collector (string limit: no bending, tangential landing); "
                 "0.0116 mm$^3$/s, 6.5 kV", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--lag":
        lag(sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(2)
