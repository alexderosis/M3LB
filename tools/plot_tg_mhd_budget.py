#!/usr/bin/env python3
"""The confined-MHD paper's budget figure: what feeds the near-wall Ohmic dissipation.

    plot_tg_mhd_budget.py RUNS_DIR [-o PREFIX] [--fmt png,pdf] [--re 1000] [--window 2 10]

RUNS_DIR holds what GPU/csf3/tg_mhd_budget.sub writes (results/P_tg_mhd/budget/ once
copied back): one <A|B>_re<Re>_n<N>/ per run with budget.dat beside series.dat.
Everything is read through tools/tg_mhd_budget.py -- the same within-delta reading,
window integration and phi as the pre-registered test -- and drawn from the finer
grid of each rung. One figure, PREFIX.<fmt>:

  (a) at one Re (default 1000), over the window: the stretching S = B_i B_j d_j u_i
      and the Ohmic dissipation eta |j|^2 per unit volume, each wall-distance
      layer's time integral over its share of the volume, against d / delta, for
      A and B, relative to the box-mean dissipation over the window. Near the
      free-slip wall the flow feeds the field (S > 0); near the no-slip wall S is
      negative -- the field does work on the flow there.
  (b) phi, the share of the near-wall Ohmic deficit that the stretching deficit
      accounts for, against Re, with its two-grid band and the pre-registered
      threshold 1/2; and its two complements, the transport and storage shares,
      the three adding to one.
  (c) what supplies B's viscous dissipation within delta: viscous transport, the
      Lorentz work, the pressure work, advection, and the layer's own kinetic
      energy, as shares, against Re.

A PDF is written without a creation date, so the figure regenerated from the same
data is byte-identical. Needs numpy and matplotlib (see plot_tg_mhd_ladder.py).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tg_mhd_budget as BU            # noqa: E402
import tg_mhd_ladder as T             # noqa: E402
import plot_tg_mhd_ladder as P        # noqa: E402

# Shares of the kinetic supply, panel (c): one hue each, labelled directly.
SUPPLY = (("difK", "viscous transport", "#3d3c39"), ("lor", "Lorentz work", P.COL["B"]),
          ("pres", "pressure work", "#8a8984"), ("advK", "advection", "#c3c2b7"),
          ("store", "the layer's own kinetic energy", P.WIN))


def layer_density(b, term, t0, t1):
    """x = d/delta of each written layer, and that layer's time integral of term over
    its volume share -- a density, per unit volume, in the box mean's units."""
    nl = len(b["vol"]) - 1
    a, z = max(t0, b["t"][0]), min(t1, b["t"][-1])     # as tg_mhd_budget.layer clamps
    out = []
    for k in range(nl):
        q = [(t, [c[k]]) for t, c in zip(b["t"], b["rows"][term])]
        out.append(T.trapz_window(q, a, z)[0] / b["vol"][k])
    return [k / b["dh"] for k in range(nl)], out


def box_eps(b, t0, t1):
    q = [(t, [sum(v) + sum(o)]) for t, v, o in zip(b["t"], b["rows"]["visc"], b["rows"]["ohm"])]
    return T.trapz_window(q, max(t0, b["t"][0]), min(t1, b["t"][-1]))[0]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_dir")
    ap.add_argument("-o", default="tg_mhd_budget")
    ap.add_argument("--fmt", default="png")
    ap.add_argument("--re", type=float, default=1000.0)
    ap.add_argument("--window", type=float, nargs=2, default=[2.0, 10.0])
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fmts = args.fmt.split(",")
    t0, t1 = args.window
    runs = BU.find([args.runs_dir])
    rungs = sorted({Re for (s, Re) in runs if s in ("A", "B") and ("A", Re) in runs and ("B", Re) in runs})
    fine = {k: v[max(v)] for k, v in runs.items()}

    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.patch.set_facecolor(P.SURFACE)

    # (a) the stretching and the Ohmic dissipation against wall distance, one Re
    ax = axs[0]
    P.style(ax)
    ax.axvspan(0, 1, color="#f0efec", zorder=0, linewidth=0)
    ax.axhline(0.0, color="#b8b7b0", linewidth=0.8, zorder=1)
    print("(a) Re %g, t = %g-%g: density / box-mean dissipation at the wall node" % (args.re, t0, t1))
    for s in ("A", "B"):
        b = fine[(s, args.re)]
        e = box_eps(b, t0, t1)
        for term, ls, lab in (("ohm", "-", r"Ohmic  $\eta|j|^2$"), ("S", (0, (4, 2)), r"stretching  $S$")):
            x, y = layer_density(b, term, t0, t1)
            y = [v / e for v in y]
            ax.plot(x, y, color=P.COL[s], linewidth=1.8, linestyle=ls, solid_capstyle="round",
                    label="%s  %s" % (s, lab), zorder=3)
            print("  %s %-4s %+.3f" % (s, term, y[0]))
    ax.set_xlim(0, 8)
    ax.set_xlabel(r"distance from the wall  $d/\delta$,  $\delta = Re^{-1/2}$")
    ax.set_ylabel("per unit volume / mean dissipation")
    ax.set_title("(a)  the field's budget near the wall, Re = %g, t = %g–%g" % (args.re, t0, t1),
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.45),
              labelcolor=P.INK2, ncol=2)

    # (b) phi against Re, with its band, the threshold and its two complements
    ax = axs[1]
    P.style(ax)
    ax.axhline(0.5, color=P.INK2, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    ax.axhline(0.0, color="#b8b7b0", linewidth=0.8, zorder=1)
    ax.text(rungs[0] / 1.3, 0.55, "pre-registered threshold  ½", fontsize=7.5, color=P.INK2, va="bottom")
    phis, bands, tr, st = [], [], [], []
    for Re in rungs:
        Ns = sorted(set(runs[("A", Re)]) & set(runs[("B", Re)]))
        got = {N: (BU.layer(runs[("A", Re)][N], t0, t1, 1.0), BU.layer(runs[("B", Re)][N], t0, t1, 1.0))
               for N in Ns}
        ph = {N: BU.phi_of(*got[N]) for N in Ns}
        la, lb = got[Ns[-1]]
        dO = la["ohm"] - lb["ohm"]
        phis.append(ph[Ns[-1]]); bands.append(abs(ph[Ns[-1]] - ph[Ns[-2]]))
        tr.append((la["Tres"] - lb["Tres"]) / dO); st.append(-(la["deM"] - lb["deM"]) / dO)
    print("(b) phi %s, bands %s, transport %s, storage %s" % (
        " ".join("%.3f" % v for v in phis), " ".join("%.3f" % v for v in bands),
        " ".join("%+.3f" % v for v in tr), " ".join("%+.3f" % v for v in st)))
    ax.plot(rungs, tr, color="#8a8984", linewidth=1.2, marker="s", markersize=4.5,
            markeredgecolor=P.SURFACE, markeredgewidth=1.0, zorder=2, label="transport share")
    ax.plot(rungs, st, color="#c3c2b7", linewidth=1.2, marker="D", markersize=4.0,
            markeredgecolor=P.SURFACE, markeredgewidth=1.0, zorder=2, label="storage share")
    ax.errorbar(rungs, phis, yerr=bands, color=P.INK, linewidth=1.8, marker="o", markersize=6.5,
                markeredgecolor=P.SURFACE, markeredgewidth=1.5, capsize=3, elinewidth=1.0, zorder=4,
                label=r"$\phi$, stretching share")
    P.re_axis(ax, matplotlib, rungs)
    ax.set_ylim(-1.2, 2.3)
    ax.set_ylabel("share of the Ohmic deficit within δ")
    ax.set_title(r"(b)  $\phi=(S_A-S_B)/(\Omega_A-\Omega_B)$", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.62),
              labelcolor=P.INK2)

    # (c) what supplies B's viscous dissipation within delta
    ax = axs[2]
    P.style(ax)
    ax.axhline(0.0, color="#b8b7b0", linewidth=0.8, zorder=1)
    shares = {k: [] for k, _, _ in SUPPLY}
    for Re in rungs:
        lb = BU.layer(fine[("B", Re)], t0, t1, 1.0)
        v = lb["visc"]
        for k, _, _ in SUPPLY:
            shares[k].append((-lb["deK"] if k == "store" else lb[k]) / v)
    # Labels at the last rung, pushed apart where their lines end close together.
    ends = sorted((shares[k][-1], k) for k, _, _ in SUPPLY)
    gap, ylab, last = 0.038, {}, None
    for y, k in ends:
        ylab[k] = y if last is None else max(y, last + gap)
        last = ylab[k]
    for k, lab, col in SUPPLY:
        ax.plot(rungs, shares[k], color=col, linewidth=1.6, marker="o", markersize=5.5,
                markeredgecolor=P.SURFACE, markeredgewidth=1.2, zorder=3)
        ax.text(rungs[-1] * 1.12, ylab[k], lab, va="center", fontsize=7.5,
                color=P.INK2 if col == "#c3c2b7" else col)
        print("(c) B %-5s %s" % (k, " ".join("%.3f" % x for x in shares[k])))
    P.re_axis(ax, matplotlib, rungs)
    ax.set_xlim(rungs[0] / 1.35, rungs[-1] * 3.6)
    ax.set_ylabel("share of B's viscous dissipation within δ")
    ax.set_title("(c)  what feeds the no-slip layer, t = %g–%g" % (t0, t1), loc="left", fontsize=9.5)

    fig.tight_layout(w_pad=2.0)
    P.save(fig, args.o, fmts, plt)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
