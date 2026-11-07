#!/usr/bin/env python3
"""Animation of the MEW jet behind the moving collector (demonstrator/mew_jet).

    ./build/demonstrator/mew_jet -anim 120 -out DIR/jet
    python3 tools/anim_mew_jet.py DIR/jet results/mew_jet/mew_jet_lag.mp4 [--fps 30]

Reads PREFIX_anim.csv, PREFIX_anim_<k>.csv (the jet at one current, from just
above its free-landing speed U to 4 U) and PREFIX_anim_free.csv (the vertical
jet landing freely). The collector speeds up from U/2 to 4 U, pausing at the
experiment's 35 mm/s. Left: the jet seen from the side, DRAWN TO SCALE -- its
line width is its diameter at the axes' millimetre scale -- and coloured by its
temperature, with tracer dots carried along it and then by the collector,
emitted at equal intervals of time so that their spacing shows the stretching
(the motion is slowed by a constant factor, printed in the frame). Below U the
collector is slower than the jet lands: it would compress the jet, which
buckles, and the model says nothing about that, so the frame says so. Right:
the lag against the collector's speed, with the frame's point on it. Writes the
mp4 and, beside it, a still of the 35 mm/s pause (_35mms.png).
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
SOLID = "#b9b8b2"
TMAP = LinearSegmentedColormap.from_list("T", ["#f0a070", "#e0703e", "#b8461b", "#6e2a10"])
TN = Normalize(25.0, 120.0)
LIVE = "#2a78d6"     # the frame's point on the lag curve
VC_EXP = 35.0        # the experiment's collector speed [mm/s]


def load(name):
    return np.genfromtxt(name, delimiter=",", names=True)


def path_array(p):
    return np.stack([p["x_mm"], p["z_mm"], p["d_um"], p["u_mm_s"], p["T_C"]], axis=1)


def travel_time(P):
    """cumulative time [s] along a path (x, z in mm, u in mm/s)"""
    ds = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    um = 0.5 * (P[1:, 3] + P[:-1, 3])
    return np.concatenate([[0.0], np.cumsum(ds / um)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    idx = load(a.prefix + "_anim.csv")
    ms = np.asarray(idx["vc_over_U"])
    lags = np.asarray(idx["lag_mm"])
    th0 = np.asarray(idx["theta0_deg"])
    U = float(idx["U_mm_s"][0])
    I = float(idx["I_nA"][0])
    paths = np.stack([path_array(load(a.prefix + "_anim_%d.csv" % int(k))) for k in idx["k"]])
    free = path_array(load(a.prefix + "_anim_free.csv"))
    L = float(free[-1, 1])
    m35 = VC_EXP / U

    def path_at(m):
        if m < ms[0]:
            return free
        k = int(np.clip(np.searchsorted(ms, m) - 1, 0, len(ms) - 2))
        w = (m - ms[k]) / (ms[k + 1] - ms[k])
        return (1 - w) * paths[k] + w * paths[k + 1]

    def lag_at(m):
        return 0.0 if m < ms[0] else float(np.interp(m, ms, lags))

    def angle_at(m):
        return 0.0 if m < ms[0] else float(np.interp(m, ms, th0))

    # the timeline: U/2 held, ramp to U, ramp to 35 mm/s, hold, ramp to 4 U, hold
    fps = a.fps
    seg = [(0.5, 0.5, 2.5), (0.5, 1.0, 2.5), (ms[0], m35, 4.0), (m35, m35, 3.0), (m35, 4.0, 6.0), (4.0, 4.0, 2.5)]
    mt = []
    for m0, m1, sec in seg:
        nf = int(round(sec * fps))
        mt += list(np.linspace(m0, m1, nf, endpoint=False)) if m0 != m1 else [m0] * nf
    mt.append(4.0)
    mt = np.asarray(mt)
    hold35 = int(round((2.5 + 2.5 + 4.0 + 1.5) * fps))   # the frame mid-way through the 35 mm/s pause

    # one slow-motion factor for the whole film: the jet at 35 mm/s takes ~2.4 s to cross the frame
    t35 = travel_time(path_at(m35))[-1]
    slow = 2.4 / t35
    dt = 1.0 / fps / slow                                 # physical seconds per frame
    emit = t35 / 26.0                                     # a tracer every so often, physically
    print("U %.3f mm/s at %.4f nA; the jet crosses in %.3f s at 35 mm/s (free landing %.3f s); slowed %.1fx"
          % (U, I, t35, travel_time(free)[-1], slow))

    tmp = tempfile.mkdtemp(prefix="anim_mew_jet_")
    try:
        fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURF)
        ax = fig.add_axes([0.045, 0.12, 0.6, 0.72])
        ax2 = fig.add_axes([0.715, 0.2, 0.265, 0.55])
        xlim, ylim = (-1.2, 10.8), (-0.45, L + 0.95)
        # points per mm along the axes (equal aspect): the jet's width is drawn to that scale
        bb = ax.get_position()
        sx = bb.width * 16 * 72 / (xlim[1] - xlim[0])
        sy = bb.height * 9 * 72 / (ylim[1] - ylim[0])
        pt_per_mm = min(sx, sy)
        # a magnified inset of the first millimetre below the cone, where the jet thins and cools
        axi = fig.add_axes([0.19, 0.545, 0.14, 0.19])
        ixl, iyl = (-0.25, 0.95), (L - 1.05, L + 0.12)
        bbi = axi.get_position()
        pt_per_mm_i = min(bbi.width * 16 * 72 / (ixl[1] - ixl[0]), bbi.height * 9 * 72 / (iyl[1] - iyl[0]))

        # tracers: [emission time, landed?, x on the collector]
        tracers = [[-k * emit, False, 0.0] for k in range(0, 160)]
        tphys = 0.0
        coll_off = 0.0
        # start them in place: the ones older than the crossing are already on the collector
        P0 = path_at(mt[0])
        tt0 = travel_time(P0)
        for tr in tracers:
            age = tphys - tr[0]
            if age > tt0[-1]:
                tr[1] = True
                tr[2] = P0[-1, 0] + mt[0] * U * (age - tt0[-1])
        for n, m in enumerate(mt):
            P = path_at(m)
            vc = m * U
            tt = travel_time(P)
            ax.cla()
            ax2.cla()
            ax.set_facecolor(SURF)
            # the collector, its ticks moving with it, and the nozzle
            ax.add_patch(plt.Rectangle((xlim[0], -0.32), xlim[1] - xlim[0], 0.32, color=SOLID, lw=0, zorder=1))
            for j in range(13):
                xt = xlim[0] + ((j + coll_off) % 13.0)
                ax.plot([xt, xt], [-0.32, -0.12], color=SURF, lw=1.2, zorder=2)
            ax.add_patch(plt.Rectangle((-0.5, L + 0.02), 1.0, 0.9, color=SOLID, lw=0, zorder=1))
            ax.add_patch(plt.Rectangle((-0.2, L + 0.02), 0.4, 0.9, color="#d6d5cf", lw=0, zorder=1))
            # the jet, to scale and coloured by temperature
            xy = np.stack([P[:, 0], L - P[:, 1]], axis=1)
            segs = np.stack([xy[:-1], xy[1:]], axis=1)
            dmid = 0.5 * (P[1:, 2] + P[:-1, 2]) * 1e-3          # mm
            Tmid = 0.5 * (P[1:, 4] + P[:-1, 4])
            lc = LineCollection(segs, linewidths=np.maximum(dmid * pt_per_mm, 0.6), colors=TMAP(TN(Tmid)),
                                capstyle="round", zorder=4)
            ax.add_collection(lc)
            # the laid fibre on the collector
            xl = float(P[-1, 0])
            ax.plot([xl, xlim[1]], [0.0, 0.0], color=TMAP(TN(P[-1, 4])),
                    lw=max(P[-1, 2] * 1e-3 * pt_per_mm, 0.6), solid_capstyle="butt", zorder=3)
            # tracers
            pts = []
            for tr in tracers:
                age = tphys - tr[0]
                if age < 0:
                    continue
                if not tr[1]:
                    if age <= tt[-1]:
                        pts.append((np.interp(age, tt, xy[:, 0]), np.interp(age, tt, xy[:, 1])))
                        continue
                    tr[1], tr[2] = True, xl + vc * (age - tt[-1])
                if tr[2] <= xlim[1]:
                    pts.append((tr[2], 0.0))
            pts = np.asarray(pts).reshape(-1, 2)
            if len(pts):
                ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.0, color=INK, mec=SURF, mew=0.5, zorder=6)
            ax.plot([0], [L], "o", ms=4, color=INK2, zorder=5)
            ax.add_patch(plt.Rectangle((ixl[0], iyl[0]), ixl[1] - ixl[0], iyl[1] - iyl[0], fill=False,
                                       ec=MUTED, lw=0.8, ls="--", zorder=7))
            axi.cla()
            axi.set_facecolor(SURF)
            axi.add_patch(plt.Rectangle((-0.5, L + 0.02), 1.0, 0.5, color=SOLID, lw=0, zorder=1))
            axi.add_collection(LineCollection(segs, linewidths=np.maximum(dmid * pt_per_mm_i, 0.6),
                                              colors=TMAP(TN(Tmid)), capstyle="round", zorder=4))
            if len(pts):
                inn = (pts[:, 0] > ixl[0]) & (pts[:, 0] < ixl[1]) & (pts[:, 1] > iyl[0]) & (pts[:, 1] < iyl[1])
                axi.plot(pts[inn, 0], pts[inn, 1], "o", ms=2.6, color=INK, mec=SURF, mew=0.4, zorder=6)
            s_arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))])
            d1, T1 = np.interp(1.0, s_arc, P[:, 2]), np.interp(1.0, s_arc, P[:, 4])
            axi.set_xlim(*ixl)
            axi.set_ylim(*iyl)
            axi.set_aspect("equal")
            axi.set_xticks([])
            axi.set_yticks([])
            for s_ in axi.spines.values():
                s_.set_color(MUTED)
                s_.set_linestyle("--")
                s_.set_linewidth(0.8)
            axi.set_title("the first mm, magnified: %.0f um and %.0f C at the cone,\n%.0f um and %.0f C 1 mm along"
                          % (P[0, 2], P[0, 4], d1, T1), fontsize=9, color=INK2, loc="left")
            if m < ms[0]:
                ax.annotate("slower than the jet lands: the collector compresses it,\n"
                            "and it buckles here (the model does not follow that)",
                            (0.0, 0.05), xytext=(0.8, 1.4), fontsize=12, color="#b8461b",
                            arrowprops=dict(arrowstyle="->", color="#b8461b", lw=1.2))
            ax.annotate("collector", (xlim[1] - 0.2, -0.22), fontsize=11, color=INK2, ha="right", va="center")
            ax.annotate("", (xlim[1] - 0.15, -0.4), xytext=(xlim[1] - 1.6, -0.4),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0), annotation_clip=False)
            ax.text(0.65, L + 0.45, "nozzle", fontsize=11, color=INK2, va="center")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_color(MUTED)
            ax.tick_params(colors=INK2, labelsize=11)
            ax.set_xlabel("along the collector's motion [mm]", color=INK2, fontsize=12)
            ax.set_ylabel("height above the collector [mm]", color=INK2, fontsize=12)
            # the colour key
            cax = fig.add_axes([0.42, 0.715, 0.18, 0.016])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=TN, cmap=TMAP), cax=cax, orientation="horizontal")
            cb.set_ticks([25, 60, 90, 120])
            cb.ax.tick_params(labelsize=9, colors=INK2)
            cb.outline.set_linewidth(0.4)
            cax.set_title("jet temperature [C]; width to scale", fontsize=10, color=INK2, loc="left")
            # the lag curve
            ax2.set_facecolor(SURF)
            ax2.axvspan(0.45, 1.0, color=GRID, lw=0)
            ax2.text(0.73, 0.6, "jet\ncompressed:\nbuckles", ha="center", fontsize=10, color=INK2)
            ax2.plot(ms, lags, color=INK, lw=2.0)
            ax2.plot([0.5, 1.0], [0, 0], color=INK, lw=2.0, ls=":")
            ax2.axvline(m35, color=MUTED, lw=1.0, ls="--")
            ax2.text(m35 + 0.06, 0.4, "35 mm/s,\nthe experiment's", fontsize=10, color=INK2)
            ax2.plot([m], [lag_at(m)], "o", ms=11, color=LIVE, mec=SURF, mew=1.5, zorder=5)
            ax2.set_xlim(0.45, 4.1)
            ax2.set_ylim(-0.3, max(lags) * 1.1)
            ax2.set_xlabel("collector speed / the jet's free-landing speed (%.1f mm/s)" % U, color=INK2, fontsize=11)
            ax2.set_ylabel("lag: landing point behind the cone [mm]", color=INK2, fontsize=11)
            ax2.tick_params(colors=INK2, labelsize=10)
            ax2.grid(True, color=GRID, lw=0.6)
            for s_ in ("top", "right"):
                ax2.spines[s_].set_visible(False)
            for t_ in list(fig.texts):
                t_.remove()
            fig.text(0.045, 0.955, "The MEW jet behind the moving collector: a slender viscous thread carrying %.2f nA "
                     "and the fibres' 0.0116 mm$^3$/s" % I, fontsize=16, color=INK)
            if m < ms[0]:
                state = "collector %.1f mm/s, %.2f x the jet's free landing" % (vc, m)
            else:
                state = ("collector %.1f mm/s, %.2f x the free landing  |  lag %.2f mm  |  the jet leaves the cone %.1f deg"
                         " from vertical  |  fibre %.1f um" % (vc, m, lag_at(m), angle_at(m), P[-1, 2]))
            fig.text(0.045, 0.915, state, fontsize=13, color=INK2)
            fig.text(0.715, 0.775, "String limit: no bending stiffness, so the jet lands\n"
                     "tangentially. The flow is the measured fibres'; the\n"
                     "current is not measured. Motion slowed %.0fx." % slow,
                     fontsize=10, color=INK2, va="bottom")
            fig.savefig(os.path.join(tmp, "f%05d.png" % n), facecolor=SURF)
            if n == hold35:
                fig.savefig(os.path.splitext(a.out)[0] + "_35mms.png", facecolor=SURF)
            cax.remove()
            # advance: physical time, the collector, the landed tracers; new tracers at the apex
            tphys += dt
            coll_off += vc * dt
            for tr in tracers:
                if tr[1]:
                    tr[2] += vc * dt
            tracers = [tr for tr in tracers if not (tr[1] and tr[2] > xlim[1] + 0.5)]
            while tracers[0][0] + emit <= tphys:
                tracers.insert(0, [tracers[0][0] + emit, False, 0.0])
        plt.close(fig)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%05d.png"),
               "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-movflags", "+faststart",
               a.out]
        subprocess.run(cmd, check=True)
        print("wrote", a.out, "and", os.path.splitext(a.out)[0] + "_35mms.png", "(%d frames)" % len(mt))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
