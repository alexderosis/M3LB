#!/usr/bin/env python3
"""The letter's figures, PDF and PNG, into this directory.

    python3 paper/vortex_disruption/figures.py [--only pair|collapse|threshold|gallery|fixed]

Letter:
fig_pair        Fig. 1. The same braking, M^2 Rm^(2/3) = 0.198, at Rm = 250 and 4000
                with Re = 1000 for both (R0 = 40): (a) snapshots, (b) S against
                t / Rm^(1/3). One field, M^2 Rm = 2.5, across Rm = 250..32000: (c) S
                and (d) Delta against t / Rm^(1/3).
fig_collapse    Fig. 2. the wind-up and the destruction against M^2 Rm^(2/3) and M^2 Rm.
fig_threshold   Fig. 3. M_c against Rm, and the exponent against the level.
fig_gallery_t6, fig_gallery_t12   Figs. 4, 5. Rm = 250, 1000, 4000 against
                M^2 Rm = 0.625..5 at t = 6 and 12 Rm^(1/3) (Re = Rm, R0 = 20).
fig_fixed       Fig. 6. M^2 Rm = 2.5 at Rm = 250, 1000, 4000, at matched wind-up times.

EVERY RUN IS D3Q27 + D3Q7 WITH nz = 1: the map is d3q27/map (Rm = 250..1000, and
two more fields at 2000 and 4000) with gpu_campaign/map (Rm = 2000..32000).
Snapshots read the frames of GPU/csf3/vortex_disruption_d3q27.sub in
runs/vortex_disruption_d3q27/, which git does not keep; a figure whose frames are
missing is skipped and the rest are made. Their series are tracked in
results/Q_vortex_disruption/d3q27/.

The numbers on the figures are vd_numbers.py's: the same readers, the same
crossing and the same fits. Colours: the reference palette's first two
categorical slots (validated: CVD dE 24.7, normal dE 33.6), blue for the map and
for Delta, orange for the checks and for 1 - S. Six Rm in one panel take six
steps of one blue ramp (validated as ordinal: the palette's steps 250-650 and a
sixth darker one, #0a2a55, since its step 700 is too close to 650); eight do not
fit one hue, so Fig. 2 labels its lines of one Rm directly.
"""
import glob
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import vortex_disruption_map as vm                  # noqa: E402
import vortex_disruption_campaign as vc             # noqa: E402

Q = os.path.join(ROOT, "results", "Q_vortex_disruption")
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
RAMP6 = RAMP5 + ["#0a2a55"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.0, "savefig.dpi": 300})

X = lambda r: r["M"] ** 2 * r["Rm"]
D3 = os.path.join(Q, "d3q27")
FR = os.path.join(ROOT, "runs", "vortex_disruption_d3q27")   # frames of the D3Q27 job
# Fig. 1(a): the same M^2 Rm^(2/3) = 0.198 at Re = 1000, R0 = 40, either side of
# the destruction threshold (Rm 250 at M^2 Rm = 1.25, Rm 4000 at 3.15)
PAIR = ("re1000_rm250_r40_x1.25", "re1000_rm4000_r40_x3.15")


def the_map():
    """The D3Q27 map: d3q27/map (Rm 250..1000 and two more fields at 2000, 4000)
    with the registered CSF3 map (Rm 2000..32000). Every run D3Q27 + D3Q7, nz = 1."""
    return (vc.load(os.path.join(D3, "map")) or []) + vc.load(os.path.join(Q, "gpu_campaign", "map"))


def save(fig, name):
    # cropped to the content (pad 0.01 in, not matplotlib's 0.1), as the author
    # cropped them by hand on 2026-11-24 -- here so the PDFs stay vector, which a
    # crop in a viewer does not guarantee (that one came back as 300-dpi bitmaps)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "%s.%s" % (name, ext)), bbox_inches="tight",
                    pad_inches=0.01)
    plt.close(fig)
    print("  wrote %s.pdf, .png" % name)


def label(ax, s):
    ax.text(0.03, 0.96, s, transform=ax.transAxes, va="top", ha="left", fontsize=8)


# ------------------------------------------------------------------------------
def raw(path):
    with open(path, "rb") as fh:
        nx, ny = np.fromfile(fh, dtype=np.int32, count=2)
        return np.fromfile(fh, dtype=np.float32, count=nx * ny).reshape(ny, nx)


def potential(j, R0):
    """A with B = (d_y A, -d_x A), from j = -lap A, plus A = y for the applied field."""
    n = j.shape[0]
    L = n / R0
    ky = 2 * np.pi * np.fft.fftfreq(n, 1.0 / n)[:, None] / L
    kx = 2 * np.pi * np.fft.rfftfreq(n, 1.0 / n)[None, :] / L
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    ah = np.fft.rfft2(j.astype(np.float64)) / k2
    ah[0, 0] = 0.0
    y = (np.arange(n) - 0.5 * (n - 1)) / R0
    return np.fft.irfft2(ah, s=(n, n)) + y[:, None]


# ------------------------------------------------------------------------------
def ticks(axis, at):
    """Major ticks at `at`, printed plainly; minor ticks unlabelled."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    axis.set_major_locator(FixedLocator(at))
    axis.set_major_formatter(FixedFormatter(["%g" % v for v in at]))
    axis.set_minor_formatter(NullFormatter())


def by_rm(runs):
    out = {}
    for r in runs:
        out.setdefault(r["Rm"], []).append(r)
    return {k: sorted(v, key=X) for k, v in sorted(out.items())}


def fig_collapse():
    runs_all = the_map()
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.2), sharey="row")
    fig.subplots_adjust(wspace=0.08, hspace=0.42, left=0.15, right=0.97, top=0.92,
                        bottom=0.12)
    early = lambda r: vm.readings(r, "surv", lambda v: 1.0 - v)["c3"]
    late = lambda r: vm.readings(r, "dow", lambda v: v)["max"]
    xs = (lambda r: r["M"] ** 2 * r["Rm"] ** (2.0 / 3.0), X)
    xl = (r"$M^2\mathrm{Rm}^{2/3}$", r"$M^2\mathrm{Rm}$")
    for row, (get, ylab) in enumerate(((early, r"$1-S$ at $t=3\,\mathrm{Rm}^{1/3}$"),
                                       (late, r"largest $\Delta$"))):
        for col in (0, 1):
            ax = axs[row, col]
            for runs, c, mk, mfc in ((runs_all, BLUE, "o", "white"),):
                for rm, rs in by_rm(runs).items():
                    pts = [(xs[col](r), get(r)) for r in rs if get(r) is not None]
                    if row == 0:
                        pts = [p for p in pts if p[1] < 0.5]       # the wind-up while small
                    if not pts:
                        continue
                    a, b = zip(*pts)
                    ax.plot(a, b, "-", color=c, lw=0.5, alpha=0.6)
                    ax.plot(a, b, mk, color=c, mfc=mfc, ms=2.6, mew=0.6)
                    # Name each Rm where the fan spreads them: early, against M^2 Rm.
                    if row == 0 and col == 1 and rm in (250, 32000):
                        # the lowest line's label goes under its end: beside it,
                        # it would cross the frame
                        lo = rm == 32000
                        ax.annotate(r"$%g$" % rm, (a[-1], b[-1]),
                                    xytext=(3, -13) if lo else (2, 0),
                                    textcoords="offset points", fontsize=6,
                                    va="top" if lo else "center", ha="right" if lo else "left",
                                    color=INK2)
            ax.set_xscale("log")
            ticks(ax.xaxis, ((0.03, 0.1, 0.3, 1.0), (0.3, 1.0, 3.0, 10.0))[col])
            if row == 0:
                ax.set_yscale("log")
                ticks(ax.yaxis, (0.02, 0.05, 0.1, 0.2, 0.5))
            else:
                ax.axhline(0.8, color=INK2, lw=0.5, ls="--")
                ax.set_ylim(0, 1.25)
            ax.set_xlabel(xl[col], labelpad=1)
            ax.grid(color=GRID, lw=0.3)
            if col == 0:
                ax.set_ylabel(ylab)
            label(ax, "(%s)" % "abcd"[2 * row + col])
    axs[0, 0].set_title("collapses: wind-up", pad=3)
    axs[0, 1].set_title(r"spreads by $\mathrm{Rm}^{-1/3}$", pad=3)
    axs[1, 0].set_title(r"spreads", pad=3)
    axs[1, 1].set_title("collapses near 0.8: destruction", pad=3)
    save(fig, "fig_collapse")


# ------------------------------------------------------------------------------
def thresholds(runs, level, col="dow", fn=lambda v: v):
    mc, _, _ = vc.thresholds(runs, level, col, fn)
    return mc


def fig_threshold():
    mp = the_map()
    grid = vc.load(os.path.join(Q, "gpu_campaign", "grid"))
    scat = vc.load(os.path.join(Q, "gpu_campaign", "scatter"))
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.1))
    fig.subplots_adjust(hspace=0.36, left=0.17, right=0.97, top=0.97, bottom=0.1)
    ax = axs[0]
    pts = []
    for runs, c, mk, mfc, name, dx in ((mp, BLUE, "o", BLUE, "map", 1.0),
                                       (grid, ORANGE, "D", "white", r"$1.5\times$ finer grid", 1.05),
                                       (scat, ORANGE, "^", "white", r"seed $10^{-5}$", 1.1)):
        mc = thresholds(runs, 0.8)
        rm = np.array(sorted(mc))
        m = np.array([mc[k][0] for k in rm])
        h = np.array([mc[k][1] for k in rm])
        y = m * m * rm
        lo, hi = y * (1 - np.exp(-2 * h)), y * (np.exp(2 * h) - 1)
        ax.errorbar(rm * dx, y, yerr=[lo, hi], fmt=mk, color=c, mfc=mfc, ms=3.4, mew=0.7,
                    elinewidth=0.5, capsize=1.2, label=name)
        if runs is mp:
            pts += list(zip(rm, m))
    rr = np.array([200, 45000])
    lm = np.mean([math.log(m) + 0.5 * math.log(r) for r, m in pts])
    ax.plot(rr, np.exp(lm) ** 2 * np.ones_like(rr), color=INK2, lw=0.7, ls="-",
            label=r"$M_c\propto\mathrm{Rm}^{-1/2}$")
    l3 = np.mean([math.log(m) + math.log(r) / 3 for r, m in pts])
    ax.plot(rr, np.exp(2 * l3) * rr ** (1 / 3), color=INK2, lw=0.7, ls=":",
            label=r"$M_c\propto\mathrm{Rm}^{-1/3}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(rr)
    ax.set_ylim(0.8, 6)
    ticks(ax.yaxis, (1, 2, 4))
    ax.set_xlabel(r"$\mathrm{Rm}$", labelpad=1)
    ax.set_ylabel(r"$M_c^2\,\mathrm{Rm}$  (largest $\Delta = 0.8$)")
    ax.grid(color=GRID, lw=0.3, which="both")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=6, handletextpad=0.3,
              columnspacing=0.8)
    label(ax, "")
    ax.text(0.97, 0.06, "(a)", transform=ax.transAxes, ha="right")

    ax = axs[1]
    levels = (0.1, 0.25, 0.5, 0.7, 0.8, 0.9)
    for runs, mk, off in ((mp, "o", 0.0),):
        for obs, c, mfc, sub, o2 in (("dow", BLUE, BLUE, r"$\Delta$", -0.008),
                                     ("surv", ORANGE, "white", r"$1-S$", 0.008)):
            fn = (lambda v: v) if obs == "dow" else (lambda v: 1.0 - v)
            L, P, B = [], [], []
            for lv in levels:
                mc = thresholds(runs, lv, obs, fn)
                if len(mc) >= 3:
                    xy = sorted((k, v[0]) for k, v in mc.items())
                    xh = sorted((k, v[1]) for k, v in mc.items())
                    L.append(lv + off + o2)
                    P.append(vm.slope(xy))
                    B.append(vm.slope_bound(xh))
            ax.errorbar(L, P, yerr=B, fmt=mk + "-", color=c, mfc=mfc, ms=3.2, mew=0.7,
                        lw=0.6, elinewidth=0.5, capsize=1.2, label=sub)
    ax.axhline(-1 / 3, color=INK2, lw=0.7, ls=":")
    ax.axhline(-0.5, color=INK2, lw=0.7, ls="-")
    ax.text(0.37, -1 / 3 + 0.012, r"$-1/3$", fontsize=7, color=INK2)
    ax.text(0.37, -0.5 - 0.04, r"$-1/2$", fontsize=7, color=INK2)
    ax.set_xlim(0.03, 0.97)
    ax.set_ylim(-0.72, -0.05)
    ax.set_xlabel(r"level of disruption $D_c$", labelpad=1)
    ax.set_ylabel(r"$p$ in $M_c\propto\mathrm{Rm}^{p}$")
    ax.grid(color=GRID, lw=0.3)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=6, handletextpad=0.3,
              columnspacing=0.8)
    ax.text(0.97, 0.06, "(b)", transform=ax.transAxes, ha="right")
    save(fig, "fig_threshold")


# ------------------------------------------------------------------------------
#  Snapshots at matched wind-up times.
# ------------------------------------------------------------------------------
def frames_ok(d):
    """The run's log (frame_at reads its dt) and the frames the figures read:
    k = 3, 6, 12, 18, 24 of 24, i.e. t = 1.5, 3, 6, 9, 12 Rm^(1/3). Only those
    need copying off the cluster; a missing one fails loudly in raw()."""
    need = ["log.txt"] + ["%s_%04d.bin" % (f, k) for f in "wj" for k in (3, 6, 12, 18, 24)]
    return all(os.path.exists(os.path.join(d, n)) for n in need)


def frame_at(d, t):
    """The stored frame nearest t, and its time: 24 frames over the run's n_max steps."""
    log = open(os.path.join(d, "log.txt")).read()
    import re
    m = re.search(r"dt = ([\d.eE+-]+)\s+t_max = [\d.eE+-]+ \((\d+) steps\)", log)
    dt, nmax = float(m.group(1)), int(m.group(2))
    dtf = (nmax // 24) * dt
    k = min(int(round(t / dtf)), 24)
    return k, k * dtf


def snap(ax, d, k, half=5.0, lim=1.5):
    """Vorticity and field lines of frame k of run d, the central 2 half R0."""
    R0 = int(vm.read(d)["R0"])
    w = raw(os.path.join(d, "w_%04d.bin" % k)).astype(np.float64)
    A = potential(raw(os.path.join(d, "j_%04d.bin" % k)), R0)
    n = w.shape[0]
    c = 0.5 * (n - 1)
    i0, i1 = int(round(c - half * R0)), int(round(c + half * R0)) + 1
    ext = [(i0 - c) / R0, (i1 - 1 - c) / R0] * 2
    im = ax.imshow(w[i0:i1, i0:i1], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=ext, interpolation="bilinear")
    g = (np.arange(i0, i1) - c) / R0
    ax.contour(g, g, A[i0:i1, i0:i1], levels=np.arange(-half - 0.25, half + 0.5, 0.5),
               colors="#202020", linewidths=0.3, alpha=0.8, linestyles="solid")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    return im


def colorbar(fig, im, rect):
    cb = fig.colorbar(im, cax=fig.add_axes(rect), extend="both")
    cb.set_label(r"$\omega R_0/\varepsilon$", labelpad=2)
    cb.outline.set_linewidth(0.4)


CS = (1.5, 3, 6, 9, 12)                          # snapshot times, in Rm^(1/3)


def fig_pair():
    dirs = [os.path.join(FR, "pair", p) for p in PAIR]
    if not all(frames_ok(d) for d in dirs):
        print("  fig_pair skipped: no frames in runs/vortex_disruption_d3q27/pair (made by:\n    %s)"
              % "GPU/csf3/vortex_disruption_d3q27.sub")
        return
    runs = [vm.read(d) for d in dirs]
    fig = plt.figure(figsize=(7.0, 4.75))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.2, left=0.075,
                             right=0.9, top=0.925, bottom=0.085)
    top = outer[0].subgridspec(2, 5, hspace=0.06, wspace=0.04)
    for i, (d, run) in enumerate(zip(dirs, runs)):
        c = run["Rm"] ** (1 / 3)
        for j, s in enumerate(CS):
            k, t = frame_at(d, s * c)
            ax = fig.add_subplot(top[i, j])
            im = snap(ax, d, k)
            if i == 0:
                ax.set_title(r"$t=%g\,\mathrm{Rm}^{1/3}$" % s, pad=2)
            if j == 0:
                ax.set_ylabel(r"$\mathrm{Rm}=%g$" "\n" r"$M^2\mathrm{Rm}=%.3g$" % (run["Rm"], X(run)),
                              labelpad=3, fontsize=7.5)
    colorbar(fig, im, [0.912, 0.44, 0.012, 0.49])
    fig.text(0.075, 0.975, r"(a) the same $M^2\mathrm{Rm}^{2/3}=0.198$, $\mathrm{Re}=1000$",
             fontsize=8, ha="left")
    bot = outer[1].subgridspec(1, 3, wspace=0.3)
    ax = fig.add_subplot(bot[0])
    for run, col in zip(runs, (BLUE, ORANGE)):
        ax.plot(np.array(run["d"]["t"]) / run["Rm"] ** (1 / 3), run["d"]["surv"], color=col,
                lw=1.1, label=r"$\mathrm{Rm}=%g$" % run["Rm"])
    ax.set_ylabel(r"$S$", labelpad=1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    ax.text(0.97, 0.94, "(b)", transform=ax.transAxes, ha="right", va="top")
    mp = the_map()
    sets = [(250, mp), (1000, mp), (4000, mp), (8000, mp), (16000, mp), (32000, mp)]
    axS, axD = fig.add_subplot(bot[1]), fig.add_subplot(bot[2])
    for (rm, runs_), col in zip(sets, RAMP6):
        r = [g for g in runs_ if g["Rm"] == rm and abs(math.log(X(g) / 2.5)) < 0.02][0]
        t = np.array(r["d"]["t"]) / rm ** (1 / 3)
        axS.plot(t, r["d"]["surv"], color=col, lw=1.0, label=r"$%g$" % rm)
        axD.plot(t, r["d"]["dow"], color=col, lw=1.0)
    axS.set_ylabel(r"$S$", labelpad=1)
    axS.set_ylim(-0.05, 1.05)
    axD.set_ylabel(r"$\Delta$", labelpad=1)
    axD.axhline(0.8, color=INK2, lw=0.5, ls="--")
    axS.legend(frameon=False, loc="lower left", fontsize=6, title=r"$\mathrm{Rm}$",
               title_fontsize=6, ncol=2, handlelength=1.2, columnspacing=0.8)
    axS.text(0.97, 0.94, r"(c) $M^2\mathrm{Rm}=2.5$", transform=axS.transAxes, ha="right", va="top")
    axD.text(0.03, 0.94, "(d)", transform=axD.transAxes, ha="left", va="top")
    for a in (ax, axS, axD):
        a.set_xlim(0, 12)
        a.set_xticks([0, 3, 6, 9, 12])
        a.set_xlabel(r"$t\varepsilon/(R_0\,\mathrm{Rm}^{1/3})$", labelpad=1)
        a.grid(color=GRID, lw=0.3)
    save(fig, "fig_pair")


def gdir(rm, x):
    group = "gallery" if rm == 4000 else "map"
    return os.path.join(FR, group, "re%d_rm%d_r20_x%s" % (rm, rm, x))


def fig_gallery(frac):
    rms, xs = (250, 1000, 4000), ("0.625", "1.25", "1.7678", "2.5", "5")
    if not all(frames_ok(gdir(rm, x)) for rm in rms for x in xs):
        print("  fig_gallery skipped: no frames in runs/vortex_disruption_d3q27 (made by:\n    %s)"
              % "GPU/csf3/vortex_disruption_d3q27.sub")
        return
    fig, axs = plt.subplots(3, 5, figsize=(7.0, 4.35))
    fig.subplots_adjust(hspace=0.06, wspace=0.04, left=0.07, right=0.9, top=0.94, bottom=0.02)
    for i, rm in enumerate(rms):
        for j, x in enumerate(xs):
            d = gdir(rm, x)
            k, t = frame_at(d, frac * rm ** (1 / 3))
            im = snap(axs[i, j], d, k)
            axs[i, j].text(0.03, 0.04, r"$\Delta_\mathrm{max}=%.2f$" % max(vm.read(d)["d"]["dow"]),
                           transform=axs[i, j].transAxes, fontsize=6)
            if i == 0:
                axs[i, j].set_title(r"$M^2\mathrm{Rm}=%s$" % ("1.77" if x == "1.7678" else x),
                                    pad=2)
            if j == 0:
                axs[i, j].set_ylabel(r"$\mathrm{Rm}=%d$" % rm)
    colorbar(fig, im, [0.912, 0.3, 0.012, 0.45])
    save(fig, "fig_gallery_t%d" % frac)


def fig_fixed():
    rms, x = (250, 1000, 4000), "2.5"
    if not all(frames_ok(gdir(rm, x)) for rm in rms):
        print("  fig_fixed skipped: no frames in runs/vortex_disruption_d3q27")
        return
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 0.9], hspace=0.08, wspace=0.04,
                          left=0.07, right=0.9, top=0.95, bottom=0.08)
    for i, rm in enumerate(rms):
        d = gdir(rm, x)
        for j, s in enumerate(CS):
            k, t = frame_at(d, s * rm ** (1 / 3))
            ax = fig.add_subplot(gs[i, j])
            im = snap(ax, d, k)
            if i == 0:
                ax.set_title(r"$t=%g\,\mathrm{Rm}^{1/3}$" % s, pad=2)
            if j == 0:
                ax.set_ylabel(r"$\mathrm{Rm}=%d$" % rm)
    colorbar(fig, im, [0.912, 0.42, 0.012, 0.45])
    ax = fig.add_subplot(gs[3, :])
    for rm, col in zip(rms, (RAMP5[0], RAMP5[2], RAMP5[4])):
        r = vm.read(gdir(rm, x))
        ax.plot(np.array(r["d"]["t"]) / rm ** (1 / 3), r["d"]["surv"], color=col, lw=1.1,
                label=r"$\mathrm{Rm}=%d$" % rm)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$t\varepsilon/(R_0\,\mathrm{Rm}^{1/3})$", labelpad=1)
    ax.set_ylabel(r"$S$")
    ax.grid(color=GRID, lw=0.3)
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper right")
    save(fig, "fig_fixed")


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for name, f in (("pair", fig_pair), ("collapse", fig_collapse),
                    ("threshold", fig_threshold), ("gallery", lambda: fig_gallery(6)),
                    ("gallery", lambda: fig_gallery(12)), ("fixed", fig_fixed)):
        if only in (None, name):
            f()
