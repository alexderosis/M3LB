#!/usr/bin/env python3
"""Animation of demonstrator/mew_nozzle for sharing: one MP4 from four runs.

    python3 tools/anim_mew_nozzle.py <results dir> <out.mp4> [--fps 15] [--max N]

Reads, from <results dir>, the frames and series written by

    A_feed_s_6p5kV    feed 0.255 mm^3/s (Table 1 as printed), 6.5 kV
    B_feed_s_nofield  the same, field off       (drawn as an outline on A)
    C_feed_min_6p5kV  feed 0.255 mm^3/min, 6.5 kV
    D_feed_min_12kV   feed 0.255 mm^3/min, 12 kV

and renders 1600 x 900 frames with matplotlib (ffmpeg here has no drawtext, so
every word is drawn by matplotlib), then encodes H.264 with ffmpeg. Frames go
to a temporary directory beside the output and are deleted afterwards.

The corrected feed, 0.0208 mm^3/s, has two layouts of its own:

    --real    E_feed_0p0208_6p5kV and F_feed_0p0208_nofield (25 um), side by side
    --depth   R4_F_feed_0p0208_nofield and R4_E_feed_0p0208_6p5kV[_lc1|_lc3|_lc5]
              (50 um): field off, and on with the melt conducting to 1, 2, 3, 5 mm

A box freezes at its run's first contact with the collector: after it the 8 mm
box, not the physics, decides what happens at this feed (mew_nozzle's banner).
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

SURF = "#fcfcfb"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
METAL = (0.47, 0.47, 0.45)
MELT = np.array([0.165, 0.471, 0.839])
WIDTH_MM = 8.0
TIP_MM = 5.5            # tip face above the collector
XC_MM = 4.0             # nozzle axis
CONDUCT_MM = 3.5        # mew_nozzle's -lc 2 mm: melt above this height conducts

# Field-strength shading: one hue, light to dark (dataviz: sequential = one hue).
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "field", ["#fcfcfb", "#ece9f6", "#cfc8ea", "#a99bd8", "#7d68c2", "#5440a6"])

GAP_MM = 5.5
# "kv": field in kV/mm, scale 0..4.  "rel": field / (V / gap), scale 0..5 -- the
# same shading at any voltage, which is what lets a 6.5 kV and a 12 kV panel sit
# side by side without one of them saturating.
FIELD_MODE = {"mode": "kv"}

RUNS = {
    "A": ("A_feed_s_6p5kV", 6.5),
    "B": ("B_feed_s_nofield", 0.0),
    "C": ("C_feed_min_6p5kV", 6.5),
    "D": ("D_feed_min_12kV", 12.0),
}

# The corrected feed, 0.0208 mm^3/s (--real, --depth): stem, volts, conducting depth -lc [mm].
REAL = {
    "E": ("E_feed_0p0208_6p5kV", 6.5, 2.0),            # 25 um
    "F": ("F_feed_0p0208_nofield", 0.0, None),         # 25 um
    "G": ("R4_F_feed_0p0208_nofield", 0.0, None),      # 50 um
    "H1": ("R4_E_feed_0p0208_6p5kV_lc1", 6.5, 1.0),    # 50 um
    "H2": ("R4_E_feed_0p0208_6p5kV", 6.5, 2.0),        # 50 um
    "H3": ("R4_E_feed_0p0208_6p5kV_lc3", 6.5, 3.0),    # 50 um
    "H5": ("R4_E_feed_0p0208_6p5kV_lc5", 6.5, 5.0),    # 50 um: the whole drop conducts
}
# First contact with the collector as the driver prints it (the banner's real-feed
# table); None where the run ended at 120 s first. After contact the 8 mm box decides
# what happens at this feed, so a panel freezes there.
REAL_TOUCH = {"E": None, "F": 104.968, "G": 110.325, "H1": 117.712, "H2": None, "H3": 127.296,
              "H5": 62.409}
DEPTH = {"H1": "#83c1ff", "H2": "#4f8bd6", "H3": "#1e5ba1", "H5": "#002c6f"}   # the note's ramp
LIGHT = np.array([0.70, 0.81, 0.94])    # melt below the conducting depth: an insulator


def frame_index(prefix):
    out = {}
    for f in glob.glob(prefix + "_t*_fill.npy"):
        t = float(f[len(prefix) + 2:-len("_fill.npy")])
        out[t] = f[:-len("_fill.npy")]
    return dict(sorted(out.items()))


def nearest(index, t):
    if not index:
        return None
    k = min(index, key=lambda a: abs(a - t))
    return index[k]


def frame_upto(index, t, stop=None):
    """The frame nearest t, never one after `stop` (a run's first contact)."""
    keys = [k for k in index if stop is None or k <= stop + 1e-9]
    return index[min(keys, key=lambda a: abs(a - t))]


def read_series(path):
    with open(path) as f:
        r = csv.DictReader(f)
        cols = {k: [] for k in r.fieldnames}
        for row in r:
            # a run still writing its CSV leaves a half-flushed last line
            if any(row.get(k) in (None, "") for k in cols):
                continue
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def layers(stem, volts, cut_mm=None):
    """RGB image (melt, metal, field shading) and the potential in the gas. With
    cut_mm, melt more than cut_mm below the tip -- an insulator to the field -- is
    drawn light."""
    fl = np.load(stem + "_fill.npy")
    ph = np.load(stem + "_phi.npy").astype(float)
    ny, nx = fl.shape
    dx = WIDTH_MM / nx
    gas = (fl > -0.5) & (fl < 0.5)
    # The open sides and the top beside the nozzle are zero-flux cells, whose
    # reported potential is a STRUCTURAL ZERO (ScalarAdiabatic), not a field
    # value. Mirror the neighbour into them, as the boundary condition does,
    # or the gradient there draws a wall of field that is not in the solution.
    ph[:, 0] = ph[:, 1]
    ph[:, -1] = ph[:, -2]
    top = gas[-1, :]
    ph[-1, top] = ph[-2, top]
    # field strength [kV/mm] from the potential, gas only
    gy, gx = np.gradient(ph, dx)
    if volts == 0:
        emag = np.zeros_like(ph)                    # field off: nothing to shade
        top = 1.0
    elif FIELD_MODE["mode"] == "rel":
        emag = np.hypot(gx, gy) * GAP_MM            # field / (V / gap)
        top = 5.0
    else:
        emag = np.hypot(gx, gy) * volts             # kV/mm
        top = 4.0
    emag[~gas] = np.nan
    rgb = FIELD_CMAP(np.clip(np.nan_to_num(emag) / top, 0, 1))[..., :3]
    rgb[~gas] = 0.988
    f = np.clip(fl, 0, 1)[..., None]
    melt = np.broadcast_to(MELT, rgb.shape).copy()
    if cut_mm is not None and volts > 0:
        zc = (np.arange(ny) - 0.5) * dx                 # cell-centre height
        melt[zc < TIP_MM - cut_mm - 1e-9] = LIGHT
    rgb = rgb * (1 - f) + melt * f
    rgb[fl < -0.5] = METAL
    phg = np.where(gas, ph, np.nan)
    return rgb, phg, fl, dx, (nx, ny)


def draw_box(ax, stem, volts, outline_stem=None, zoom=None, contours=True, cut_mm=None):
    rgb, phg, fl, dx, (nx, ny) = layers(stem, volts, cut_mm)
    ext = (0, nx * dx, -0.5 * dx, (ny - 0.5) * dx)
    ax.imshow(rgb, origin="lower", extent=ext, interpolation="bilinear")
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) - 0.5) * dx
    if contours and volts > 0 and np.nanmax(phg) > np.nanmin(phg):
        ax.contour(x, y, phg, levels=np.linspace(0.1, 0.9, 9), colors=INK2,
                   linewidths=0.45, alpha=0.8)
    if outline_stem:
        fo = np.load(outline_stem + "_fill.npy")
        ax.contour(x, y, np.where(fo < -0.5, 0, fo), levels=[0.5], colors=ORANGE,
                   linewidths=1.4, linestyles="dashed")
    if zoom:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
    else:
        ax.set_xlim(0, WIDTH_MM); ax.set_ylim(-0.1, (ny - 0.5) * dx)
    for s in ax.spines.values():
        s.set_color(INK3); s.set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)


def single_frame(fig, t, t_end, idx, ser):
    """Run A alone: the box on the left, its tip melt and lowest point on the right."""
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], left=0.045, right=0.985,
                          top=0.865, bottom=0.105, wspace=0.16, hspace=0.42)
    axA = fig.add_subplot(gs[:, 0])
    stemA = nearest(idx["A"], t)
    stemB = nearest(idx["B"], t) if idx["B"] else None
    draw_box(axA, stemA, RUNS["A"][1], outline_stem=stemB)
    axA.set_title("Feed as printed in Table 1:  0.255 mm³/s", fontsize=13.5, color=INK,
                  loc="left", pad=8)
    axA.set_xlabel("x [mm]", fontsize=10, color=INK2)
    axA.set_ylabel("height above the collector [mm]", fontsize=10, color=INK2)
    axA.axhline(CONDUCT_MM, color=INK3, lw=0.8, ls=(0, (1, 2)))
    axA.text(0.12, CONDUCT_MM + 0.06, "melt treated as a conductor above this line",
             fontsize=8.5, color=INK2)
    axA.annotate("", xy=(7.6, 0.28), xytext=(5.9, 0.28),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    axA.text(5.9, 0.42, "collector  35 mm/s", fontsize=9.5, color=INK)
    axA.legend(handles=[Line2D([], [], color=BLUE, lw=6, label="melt"),
                        Line2D([], [], color=INK2, lw=0.8, label="equipotentials, every 0.65 kV")],
               loc="upper left", fontsize=9, frameon=True, facecolor=SURF, edgecolor=GRID,
               framealpha=0.92)
    apos = axA.get_position()
    cax = fig.add_axes([apos.x1 - 0.165, apos.y1 - 0.045, 0.15, 0.011])
    cax.imshow(np.linspace(0, 4.0, 256)[None, :], aspect="auto", cmap=FIELD_CMAP,
               extent=(0, 4.0, 0, 1), vmin=0, vmax=4.0)
    cax.set_yticks([]); cax.set_xticks([0, 1, 2, 3, 4])
    cax.tick_params(labelsize=8, colors=INK2, length=2)
    cax.set_xlabel("field strength in the air [kV/mm]", fontsize=8.5, color=INK2, labelpad=1)
    for sp in cax.spines.values():
        sp.set_visible(False)

    s = ser.get("A")
    for row, (col, ylab, ttl, lim) in enumerate([
            ("tip_zone_mm2", "melt within 1 mm of the tip [mm²]",
             "Melt hanging under the nozzle (within 1 mm of the tip)", None),
            ("z_low_mm", "lowest melt point above the collector [mm]",
             "How far the melt has fallen", (0, 5.8))]):
        ax = fig.add_subplot(gs[row, 1])
        if s is not None:
            m = s["t_s"] <= t + 1e-9
            ax.plot(s["t_s"], s[col], color=BLUE, lw=1.0, alpha=0.25)
            ax.plot(s["t_s"][m], s[col][m], color=BLUE, lw=2.4)
            if m.any():
                ax.plot(s["t_s"][m][-1], s[col][m][-1], "o", color=BLUE, ms=8, mec=SURF, mew=1.5)
            if lim is None:
                w = s["t_s"] <= t_end + 1e-9
                lim = (0, 1.15 * float(np.nanmax(s[col][w])))
        ax.axvline(t, color=INK3, lw=0.8)
        ax.set_xlim(0, t_end); ax.set_ylim(*lim)
        ax.set_xlabel("time [s]", fontsize=10, color=INK2)
        ax.set_ylabel(ylab, fontsize=10, color=INK2)
        ax.set_title(ttl, fontsize=11.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5, colors=INK2)


def overlay(ax, lines):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=SURF, alpha=0.6,
                               zorder=5))
    for k, (txt, big) in enumerate(lines):
        ax.text(0.5, 0.86 - 0.055 * k, txt, transform=ax.transAxes, ha="center",
                fontsize=12 if big else 10.5, color=INK if big else INK2,
                weight="bold" if big else "normal", zorder=6)


def pair_frame(fig, t, t_end, idx, ser, a):
    """Printed feed with the field on and off side by side, the per-minute tip
    zooms, and the melt hanging under the tip. A freezes at a.cut_a, B at a.cut_b."""
    gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1.25, 0.55, 0.55], height_ratios=[1.0, 0.82],
                          left=0.035, right=0.99, top=0.83, bottom=0.115, wspace=0.2,
                          hspace=0.42)
    boxes = [("A", "Field on, 6.5 kV", a.cut_a), ("B", "Field off", a.cut_b)]
    axes = []
    for col, (key, ttl, cut) in enumerate(boxes):
        ax = fig.add_subplot(gs[:, col])
        ax.set_anchor("N")
        axes.append(ax)
        tk = min(t, cut) if cut > 0 else t
        draw_box(ax, nearest(idx[key], tk), RUNS[key][1])
        ax.set_title(ttl, fontsize=13, color=INK, loc="left", pad=6)
        ax.set_xlabel("x [mm]", fontsize=10, color=INK2)
        if col == 0:
            ax.set_ylabel("height above the collector [mm]", fontsize=10, color=INK2)
        ax.annotate("", xy=(7.6, 0.28), xytext=(5.6, 0.28),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        ax.text(5.6, 0.42, "collector 35 mm/s", fontsize=9, color=INK)
    axA, axB = axes
    axA.axhline(CONDUCT_MM, color=INK3, lw=0.8, ls=(0, (1, 2)))
    axA.text(0.12, CONDUCT_MM + 0.06, "melt conducts above this line", fontsize=8.5, color=INK2)
    fig.canvas.draw()
    fig.text(axA.get_position().x0, axA.get_position().y1 + 0.052,
             "Feed as printed in Table 1:  0.255 mm³/s", fontsize=14, color=INK, weight="bold")
    if a.cut_a > 0 and t > a.cut_a:
        overlay(axA, [(f"Shown to contact at {a.cut_a:.1f} s", True),
                      ("After contact this rope lifts off the collector.", False),
                      ("On 12.5 µm cells the field-off rope does too:", False),
                      ("the return on the right is this 25 µm grid's.", False)])
    if a.cut_b > 0 and t > a.cut_b:
        overlay(axB, [(f"Shown to {a.cut_b:.1f} s", True),
                      ("The drawn thread is now one 25 µm cell thick", False),
                      ("and breaks up below the grid.", False)])

    # per-minute tip zooms
    zoom = (XC_MM - 0.7, XC_MM + 0.7, TIP_MM - 0.8, TIP_MM + 0.25)
    zax = []
    for col, (key, ttl) in enumerate([("C", "6.5 kV"), ("D", "12 kV *")]):
        ax = fig.add_subplot(gs[0, 2 + col])
        zax.append(ax)
        draw_box(ax, nearest(idx[key], t), RUNS[key][1], zoom=zoom)
        ax.set_title(f"0.255 mm³/min, {ttl}", fontsize=11, color=INK, loc="left")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("x [mm]", fontsize=9, color=INK2)
    ybot = min(z.get_position().y0 for z in zax)
    fig.text(zax[0].get_position().x0, ybot - 0.075,
             "* 12 kV: about the pull a round 1 mm tip feels at 6.5 kV.", fontsize=8.5, color=INK2)

    axS = fig.add_subplot(gs[1, 2:])
    styles = [("A", BLUE, "0.255 mm³/s, 6.5 kV", "-", 3.2, a.cut_a),
              ("B", ORANGE, "0.255 mm³/s, field off", (0, (4, 3)), 2.0, a.cut_b),
              ("C", AQUA, "0.255 mm³/min, 6.5 kV", "-", 3.2, 0.0),
              ("D", YELLOW, "0.255 mm³/min, 12 kV", (0, (4, 3)), 2.0, 0.0)]
    for key, colr, lab, lsty, lw, cut in styles:
        if key not in ser:
            continue
        s = ser[key]
        tk = min(t, cut) if cut > 0 else t
        full = (s["t_s"] <= cut + 1e-9) if cut > 0 else np.ones_like(s["t_s"], bool)
        m = s["t_s"] <= tk + 1e-9
        y = np.maximum(s["tip_zone_mm2"], 1e-4)
        axS.plot(s["t_s"][full], y[full], color=colr, lw=1.0, alpha=0.25)
        axS.plot(s["t_s"][m], y[m], color=colr, lw=lw, ls=lsty, label=lab)
        if m.any():
            axS.plot(s["t_s"][m][-1], y[m][-1], "o", color=colr, ms=7, mec=SURF, mew=1.5)
    axS.axvline(t, color=INK3, lw=0.8)
    axS.set_yscale("log"); axS.set_ylim(3e-3, 3.0); axS.set_xlim(0, t_end)
    axS.set_xlabel("time [s]", fontsize=10, color=INK2)
    axS.set_ylabel("mm²", fontsize=9.5, color=INK2)
    axS.set_title("Melt hanging within 1 mm under the tip", fontsize=11,
                  color=INK, loc="left")
    axS.grid(axis="y", color=GRID, lw=0.6)
    axS.spines[["top", "right"]].set_visible(False)
    axS.tick_params(labelsize=8.5, colors=INK2)
    axS.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)

    fig.canvas.draw()
    apos, bpos = axA.get_position(), axB.get_position()
    # the key, in the band under the boxes: colour scale left, line legend right
    kax = fig.add_axes([bpos.x0, apos.y0 - 0.215, bpos.width, 0.08])
    kax.axis("off")
    kax.legend(handles=[Line2D([], [], color=BLUE, lw=6, label="melt"),
                        Line2D([], [], color=INK2, lw=0.8,
                               label="equipotentials, every 10 % of the voltage"),
                        Line2D([], [], color=INK3, lw=0.8, ls=(0, (1, 2)),
                               label="below this line the melt is treated as an insulator")],
               loc="upper left", fontsize=9, frameon=False)
    cax = fig.add_axes([apos.x0 + 0.01, apos.y0 - 0.155, 0.17, 0.012])
    cax.imshow(np.linspace(0, 5.0, 256)[None, :], aspect="auto", cmap=FIELD_CMAP,
               extent=(0, 5.0, 0, 1), vmin=0, vmax=5.0)
    cax.set_yticks([]); cax.set_xticks([0, 1, 2, 3, 4, 5])
    cax.tick_params(labelsize=7.5, colors=INK2, length=2)
    cax.set_xlabel("field strength in the air / mean field (voltage / 5.5 mm)", fontsize=8.5,
                   color=INK2, labelpad=2)
    for sp in cax.spines.values():
        sp.set_visible(False)

    fig.text(0.04, 0.034, "2-D planar model, nominal viscosity 500 Pa·s and surface tension"
             " 0.03 N/m. Until first contact the state is kept mirror-symmetric, as the set-up is;"
             " melt the collector carries to the box's edge leaves there.", fontsize=9, color=INK3)
    fig.text(0.04, 0.013, "Equipotentials every 10 % of the voltage. The 10–20 µm jet of the"
             " experiment is below the 25 µm cells.", fontsize=9, color=INK3)


def freeze_note(ax, lines):
    """A small card at the top of a box whose run has stopped (contact or the run's end)."""
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=SURF, alpha=0.5,
                               zorder=5))
    for k, (txt, big) in enumerate(lines):
        ax.text(0.5, 0.95 - 0.045 * k, txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=11 if big else 9, color=INK if big else INK2,
                weight="bold" if big else "normal", zorder=6)


def real_series_axes(ax, t, t_end, ser, keys, col, ylab, ttl, lim, cuts=(), marks=True):
    """One time series per run up to t (faint beyond), contact dots and the cut depths."""
    for key, colr, lab, lsty, lw in keys:
        if key not in ser:
            continue
        s = ser[key]
        touch = REAL_TOUCH[key]
        stop = min(touch, t) if touch else t
        full = s["t_s"] <= (touch if touch else 1e9) + 1e-9
        m = s["t_s"] <= stop + 1e-9
        ax.plot(s["t_s"][full], s[col][full], color=colr, lw=1.0, alpha=0.22, ls=lsty)
        ax.plot(s["t_s"][m], s[col][m], color=colr, lw=lw, ls=lsty, label=lab)
        if m.any():
            ax.plot(s["t_s"][m][-1], s[col][m][-1], "o", color=colr, ms=7, mec=SURF, mew=1.5)
        if marks and touch and t >= touch:
            ax.plot([touch], [0], "o", color=colr, ms=9, mec=SURF, mew=1.5, clip_on=False, zorder=6)
    for h, colr, txt in cuts:
        ax.axhline(h, color=colr, lw=0.9, ls=(0, (1, 2)))
        ax.text(t_end * 0.985, h + 0.07, txt, ha="right", fontsize=8.5, color=INK2)
    ax.axvline(t, color=INK3, lw=0.8)
    ax.set_xlim(0, t_end); ax.set_ylim(*lim)
    ax.set_xlabel("time [s]", fontsize=10, color=INK2)
    ax.set_ylabel(ylab, fontsize=10, color=INK2)
    ax.set_title(ttl, fontsize=11.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=INK2)


def box_key(fig, ax, field_scale=True):
    """Legend and field scale in a band under a box."""
    pos = ax.get_position()
    kax = fig.add_axes([pos.x0, pos.y0 - 0.115, pos.width * 2.1, 0.06])
    kax.axis("off")
    kax.legend(handles=[Line2D([], [], color=BLUE, lw=6, label="melt that conducts"),
                        Line2D([], [], color=LIGHT, lw=6, label="melt below the conducting depth"),
                        Line2D([], [], color=INK2, lw=0.8, label="equipotentials, every 10 % of the voltage")],
               loc="upper left", fontsize=9, frameon=False, ncol=3, handlelength=1.6, columnspacing=1.4)
    if field_scale:
        cax = fig.add_axes([pos.x0 + 0.005, pos.y0 - 0.135, 0.15, 0.010])
        cax.imshow(np.linspace(0, 4.0, 256)[None, :], aspect="auto", cmap=FIELD_CMAP,
                   extent=(0, 4.0, 0, 1), vmin=0, vmax=4.0)
        cax.set_yticks([]); cax.set_xticks([0, 1, 2, 3, 4])
        cax.tick_params(labelsize=7.5, colors=INK2, length=2)
        cax.set_xlabel("field strength in the air [kV/mm]", fontsize=8.5, color=INK2, labelpad=1)
        for sp in cax.spines.values():
            sp.set_visible(False)


def real_frame(fig, t, t_end, idx, ser):
    """The corrected feed on 25 um cells: field on and off side by side, then the lowest
    point and the melt under the tip against time. A box freezes at its first contact."""
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25], left=0.04, right=0.985, top=0.83,
                          bottom=0.21, wspace=0.16, hspace=0.5)
    zoom = (1.5, 6.5, -0.1, 6.3)
    axes = []
    for col, (key, ttl) in enumerate((("E", "Field on, 6.5 kV"), ("F", "Field off"))):
        stem, volts, lc = REAL[key]
        touch = REAL_TOUCH[key]
        ax = fig.add_subplot(gs[:, col])
        ax.set_anchor("N")
        axes.append(ax)
        draw_box(ax, frame_upto(idx[key], t, touch), volts, zoom=zoom, cut_mm=lc)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=13, color=INK, loc="left", pad=6)
        ax.set_xlabel("x [mm]", fontsize=10, color=INK2)
        if col == 0:
            ax.set_ylabel("height above the collector [mm]", fontsize=10, color=INK2)
            ax.axhline(TIP_MM - lc, color=INK3, lw=0.8, ls=(0, (1, 2)))
            ax.text(1.6, TIP_MM - lc + 0.07, "melt conducts above", fontsize=8.5, color=INK2)
        ax.annotate("", xy=(2.9, 0.18), xytext=(1.65, 0.18),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
        ax.text(1.65, 0.33, "collector 35 mm/s", fontsize=8.5, color=INK)
        if touch and t >= touch:
            freeze_note(ax, [(f"Touches the collector at {touch:.1f} s", True),
                             ("After contact the 8 mm box decides", False),
                             ("what happens at this feed: not shown.", False)])
    fig.canvas.draw()
    box_key(fig, axes[0])
    keys = [("E", BLUE, "field on", "-", 2.6), ("F", ORANGE, "field off", (0, (4, 3)), 2.2)]
    ax1 = fig.add_subplot(gs[0, 2])
    real_series_axes(ax1, t, t_end, ser, keys, "z_low_mm", "mm above the collector",
                     "How far the melt has fallen", (0, 5.8),
                     cuts=[(TIP_MM - 2.0, INK3, "conducting depth (field on)")])
    ax1.legend(loc="upper right", fontsize=9, frameon=False)
    ax2 = fig.add_subplot(gs[1, 2])
    real_series_axes(ax2, t, t_end, ser, keys, "tip_zone_mm2", "mm²",
                     "Melt within 1 mm under the tip", (0, 1.0), marks=False)
    fig.text(0.04, 0.034, "2-D planar model, 25 µm cells, nominal viscosity 500 Pa·s and surface"
             " tension 0.03 N/m. Nothing in a planar model can carry the melt away as a jet: a sheet"
             " carrying this feed at 35 mm/s would be 1.9 µm thick.", fontsize=9, color=INK3)
    fig.text(0.04, 0.013, "The melt conducts only within 2 mm of the tip, so it cannot short the gap;"
             " that depth is a numerical choice, and it sets how much the field delays the drop.",
             fontsize=9, color=INK3)


def depth_frame(fig, t, t_end, idx, ser):
    """The corrected feed on 50 um cells: field off, and field on with the melt conducting
    to 1, 2, 3 and 5 mm below the tip; then the lowest point against time."""
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 0.22, 2.3], left=0.04, right=0.985, top=0.83,
                          bottom=0.21, wspace=0.12)
    zoom = (2.2, 5.8, -0.1, 6.3)
    boxes = (("G", "Field off"), ("H1", "On, to 1 mm"), ("H2", "On, to 2 mm"), ("H3", "On, to 3 mm"),
             ("H5", "On, to 5 mm"))
    axes = []
    for col, (key, ttl) in enumerate(boxes):
        stem, volts, lc = REAL[key]
        touch = REAL_TOUCH[key]
        ax = fig.add_subplot(gs[0, col])
        ax.set_anchor("N")
        axes.append(ax)
        draw_box(ax, frame_upto(idx[key], t, touch), volts, zoom=zoom, cut_mm=lc)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=12, color=INK, loc="left", pad=6)
        ax.set_xlabel("x [mm]", fontsize=9, color=INK2)
        ax.tick_params(labelsize=7.5)
        if col == 0:
            ax.set_ylabel("height above the collector [mm]", fontsize=10, color=INK2)
        else:
            ax.set_yticklabels([])
        if lc is not None:
            ax.axhline(TIP_MM - lc, color=INK3, lw=0.8, ls=(0, (1, 2)))
        last = max(idx[key])
        if touch and t >= touch:
            freeze_note(ax, [(f"Touches at {touch:.1f} s", True)])
        elif touch is None and t > last + 1e-9:
            freeze_note(ax, [("Run ends at 120 s", True), ("0.075 mm above the collector", False)])
    fig.canvas.draw()
    box_key(fig, axes[0])
    keys = [("G", ORANGE, "field off", (0, (4, 3)), 2.2), ("H1", DEPTH["H1"], "on, conducts to 1 mm", "-", 2.6),
            ("H2", DEPTH["H2"], "on, to 2 mm", "-", 2.6), ("H3", DEPTH["H3"], "on, to 3 mm", "-", 2.6),
            ("H5", DEPTH["H5"], "on, to 5 mm", "-", 2.6)]
    ax1 = fig.add_subplot(gs[0, 6])
    real_series_axes(ax1, t, t_end, ser, keys, "z_low_mm", "mm above the collector",
                     "How far the melt has fallen", (0, 5.8),
                     cuts=[(TIP_MM - 1.0, DEPTH["H1"], "1 mm"), (TIP_MM - 2.0, DEPTH["H2"], "2 mm"),
                           (TIP_MM - 3.0, DEPTH["H3"], "3 mm"), (TIP_MM - 5.0, DEPTH["H5"], "5 mm")])
    ax1.legend(loc="upper right", fontsize=9, frameon=False)
    fig.text(0.04, 0.034, "2-D planar model, 50 µm cells. The melt conducts only down to a chosen depth"
             " (dotted), so it cannot short the gap; below it the melt is an insulator to the field.",
             fontsize=9, color=INK3)
    fig.text(0.04, 0.013, "Each field-on drop follows the others until it passes its own depth; then it slows"
             " (1-3 mm) or runs to the collector (5 mm). That numerical choice sets the field's effect.",
             fontsize=9, color=INK3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max", type=int, default=0, help="render only the first N frames")
    ap.add_argument("--tmax", type=float, default=0.0,
                    help="stop at this time [s]; 0 = the whole run")
    ap.add_argument("--endnote", default="",
                    help="text for a closing card over the last frame, held 4 s")
    ap.add_argument("--cut-ab", type=float, default=0.0,
                    help="freeze runs A and B at this time (contact with the collector)")
    ap.add_argument("--clock", default="A",
                    help="which run's frame times drive the animation (A or D)")
    ap.add_argument("--pair", action="store_true",
                    help="field on and off side by side, with the per-minute zooms")
    ap.add_argument("--cut-a", type=float, default=0.0, help="freeze run A here [s]")
    ap.add_argument("--cut-b", type=float, default=0.0, help="freeze run B here [s]")
    ap.add_argument("--single", action="store_true",
                    help="run A alone: the box and its own two time series")
    ap.add_argument("--relative-field", action="store_true",
                    help="shade field / (V / gap) instead of kV/mm")
    ap.add_argument("--real", action="store_true",
                    help="the corrected feed on 25 um cells: field on and off side by side")
    ap.add_argument("--depth", action="store_true",
                    help="the corrected feed on 50 um cells: field off, and on with the melt"
                         " conducting to 1, 2 and 3 mm below the tip")
    a = ap.parse_args()
    if a.relative_field:
        FIELD_MODE["mode"] = "rel"

    if a.real or a.depth:
        keys = ("E", "F") if a.real else ("G", "H1", "H2", "H3", "H5")
        idx = {k: frame_index(os.path.join(a.dir, REAL[k][0])) for k in keys}
        ser = {k: read_series(os.path.join(a.dir, REAL[k][0] + "_series.csv")) for k in keys}
        missing = [REAL[k][0] for k in keys if not idx[k]]
        if missing:
            sys.exit("no frames in " + a.dir + " for " + ", ".join(missing))
        clock = "E" if a.real else "H1"
    else:
        idx = {k: frame_index(os.path.join(a.dir, v[0])) for k, v in RUNS.items()}
        ser = {}
        for k, v in RUNS.items():
            p = os.path.join(a.dir, v[0] + "_series.csv")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                ser[k] = read_series(p)
        if not idx["A"]:
            sys.exit("no frames for run A in " + a.dir)
        clock = a.clock if idx.get(a.clock) else "A"
    times = list(idx[clock].keys())
    if a.tmax > 0:
        times = [t for t in times if t <= a.tmax + 1e-9]
    if a.max:
        times = times[: a.max]
    t_end = max(times)

    tmp = tempfile.mkdtemp(prefix="mew_anim_", dir=os.path.dirname(os.path.abspath(a.out)))
    try:
        for n, t in enumerate(times):
            fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURF)
            gs = None if (a.single or a.pair) else fig.add_gridspec(
                2, 3, width_ratios=[1.45, 0.62, 0.62], height_ratios=[1.0, 0.80],
                left=0.045, right=0.985, top=0.865, bottom=0.105, wspace=0.17, hspace=0.42)
            # header
            fig.text(0.045, 0.955, "Melt electrowriting nozzle, simulated in 2-D (M3LB, lattice Boltzmann)",
                     fontsize=19, color=INK, weight="bold")
            if a.real or a.depth:
                fig.text(0.045, 0.915, "PCL at 120 °C  ·  0.4 mm nozzle  ·  6.5 kV  ·  5.5 mm gap  ·"
                         "  collector 35 mm/s  ·  feed 0.0208 mm³/s (Table 1, corrected)",
                         fontsize=11.5, color=INK2)
            else:
                fig.text(0.045, 0.915, "PCL at 120 °C  ·  0.4 mm nozzle  ·  6.5 kV  ·  5.5 mm to the collector"
                         "  ·  collector 2100 mm/min  ·  set-up from Ju, Omar, Mirihanage & Kilic (preprint)",
                         fontsize=11.5, color=INK2)
            fig.text(0.985, 0.952, f"t = {t:5.2f} s", fontsize=22, color=INK, ha="right",
                     family="monospace", weight="bold")
            # progress bar
            bar = fig.add_axes([0.80, 0.918, 0.185, 0.008])
            bar.barh(0, t / t_end, color=INK2, height=1); bar.set_xlim(0, 1)
            bar.axis("off")

            if a.real or a.depth:
                (real_frame if a.real else depth_frame)(fig, t, t_end, idx, ser)
                fig.savefig(os.path.join(tmp, f"f{n:05d}.png"), facecolor=SURF)
                plt.close(fig)
                if n % 20 == 0:
                    print(f"  frame {n + 1}/{len(times)}", flush=True)
                continue
            if a.pair:
                pair_frame(fig, t, t_end, idx, ser, a)
                fig.savefig(os.path.join(tmp, f"f{n:05d}.png"), facecolor=SURF)
                plt.close(fig)
                if n % 20 == 0:
                    print(f"  frame {n + 1}/{len(times)}", flush=True)
                continue
            if a.single:
                single_frame(fig, t, t_end, idx, ser)
                fig.text(0.045, 0.018, "2-D planar model: melt treated as a conductor near the tip,"
                         " nominal viscosity 500 Pa·s and surface tension 0.03 N/m. The 10–20 µm jet"
                         " is far below the 25 µm cells and is not resolved.", fontsize=9, color=INK3)
                fig.savefig(os.path.join(tmp, f"f{n:05d}.png"), facecolor=SURF)
                plt.close(fig)
                if n % 20 == 0:
                    print(f"  frame {n + 1}/{len(times)}", flush=True)
                continue
            # (A) full box, field on, field-off outline; frozen at contact
            tab = min(t, a.cut_ab) if a.cut_ab > 0 else t
            axA = fig.add_subplot(gs[:, 0])
            stemA = nearest(idx["A"], tab)
            stemB = nearest(idx["B"], tab) if idx["B"] else None
            draw_box(axA, stemA, RUNS["A"][1], outline_stem=stemB)
            axA.set_title("Feed as printed in Table 1:  0.255 mm³/s", fontsize=13.5, color=INK,
                          loc="left", pad=8)
            axA.set_xlabel("x [mm]", fontsize=10, color=INK2)
            axA.set_ylabel("height above the collector [mm]", fontsize=10, color=INK2)
            # THE MODEL'S EDGE, DRAWN: below this height the melt is treated as an
            # insulator (a conductor touching the collector would short the gap),
            # and the field crowds the end of the conducting part as it crosses.
            axA.axhline(CONDUCT_MM, color=INK3, lw=0.8, ls=(0, (1, 2)))
            axA.text(0.12, CONDUCT_MM + 0.06, "melt treated as a conductor above this line",
                     fontsize=8.5, color=INK2)
            axA.annotate("", xy=(7.6, 0.28), xytext=(5.9, 0.28),
                         arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
            axA.text(5.9, 0.42, "collector  35 mm/s", fontsize=9.5, color=INK)
            if a.cut_ab > 0 and t > a.cut_ab:
                axA.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axA.transAxes, color=SURF,
                                            alpha=0.55, zorder=5))
                axA.text(0.5, 0.68, f"Stopped at {a.cut_ab:.1f} s: the melt has reached the collector",
                         transform=axA.transAxes, ha="center", fontsize=13, color=INK,
                         weight="bold", zorder=6)
                axA.text(0.5, 0.63, "Both then lift off it; what follows depends on the grid, not on the field.",
                         transform=axA.transAxes, ha="center", fontsize=11.5, color=INK2, zorder=6)
            handles = [Line2D([], [], color=BLUE, lw=6, label="melt, field on"),
                       Line2D([], [], color=INK2, lw=0.8,
                              label="equipotentials, every 10 % of the voltage")]
            if stemB:
                handles.insert(1, Line2D([], [], color=ORANGE, lw=1.6, ls=(0, (3, 2)),
                                         label="melt outline, field OFF"))
            axA.legend(handles=handles, loc="upper left", fontsize=9, frameon=True,
                       facecolor=SURF, edgecolor=GRID, framealpha=0.92)

            # (C), (D) tip zooms at the per-minute feed
            zoom = (XC_MM - 0.7, XC_MM + 0.7, TIP_MM - 0.8, TIP_MM + 0.25)
            zax = []
            for col, (key, ttl) in enumerate([("C", "6.5 kV"), ("D", "12 kV *")]):
                ax = fig.add_subplot(gs[0, 1 + col])
                zax.append(ax)
                stem = nearest(idx[key], t)
                if stem:
                    draw_box(ax, stem, RUNS[key][1], zoom=zoom)
                else:
                    ax.text(0.5, 0.5, "run pending", transform=ax.transAxes, ha="center",
                            color=INK3)
                    ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
                ax.set_title(f"Feed 0.255 mm³/min,  {ttl}", fontsize=11.5, color=INK, loc="left")
                ax.set_aspect("equal")
                ax.tick_params(labelsize=8)
                ax.set_xlabel("x [mm]", fontsize=9, color=INK2)
                if col == 0:
                    ax.set_ylabel("height [mm]", fontsize=9, color=INK2)
            ybot = min(z.get_position().y0 for z in zax)
            fig.text(zax[0].get_position().x0, ybot - 0.085,
                     "* 12 kV: about the pull a round 1 mm tip feels at 6.5 kV;"
                     " a flat 2-D tip concentrates the field less.", fontsize=8.5, color=INK2)

            # field-strength scale with air breakdown marked, inside panel A
            apos = axA.get_position()
            cax = fig.add_axes([apos.x1 - 0.165, apos.y1 - 0.045, 0.15, 0.011])
            top = 5.0 if FIELD_MODE["mode"] == "rel" else 4.0
            grad = np.linspace(0, top, 256)[None, :]
            cax.imshow(grad, aspect="auto", cmap=FIELD_CMAP, extent=(0, top, 0, 1), vmin=0, vmax=top)
            cax.set_yticks([]); cax.set_xticks(list(range(int(top) + 1)))
            cax.tick_params(labelsize=8, colors=INK2, length=2)
            cax.set_xlabel("field / mean field (voltage / 5.5 mm)" if FIELD_MODE["mode"] == "rel"
                           else "field strength in the air [kV/mm]",
                           fontsize=8.5, color=INK2, labelpad=1)
            cax.patch.set_alpha(0)
            for s in cax.spines.values():
                s.set_visible(False)

            # time series: melt hanging under the tip, with a cursor
            axS = fig.add_subplot(gs[1, 1:])
            # B lies on A and D on C -- that coincidence IS the result -- so the
            # second of each pair is dashed over a solid first, or it hides it.
            styles = [("A", BLUE, "0.255 mm³/s, 6.5 kV", "-", 3.2),
                      ("B", ORANGE, "0.255 mm³/s, field off", (0, (4, 3)), 2.0),
                      ("C", AQUA, "0.255 mm³/min, 6.5 kV", "-", 3.2),
                      ("D", YELLOW, "0.255 mm³/min, 12 kV", (0, (4, 3)), 2.0)]
            for key, colr, lab, lsty, lw in styles:
                if key not in ser:
                    continue
                s = ser[key]
                tk = min(t, a.cut_ab) if (a.cut_ab > 0 and key in "AB") else t
                full = (s["t_s"] <= a.cut_ab + 1e-9) if (a.cut_ab > 0 and key in "AB") \
                    else np.ones_like(s["t_s"], bool)
                m = s["t_s"] <= tk + 1e-9
                axS.plot(s["t_s"][full], np.maximum(s["tip_zone_mm2"][full], 1e-4), color=colr,
                         lw=1.0, alpha=0.25)
                axS.plot(s["t_s"][m], np.maximum(s["tip_zone_mm2"][m], 1e-4), color=colr, lw=lw,
                         ls=lsty, label=lab)
                if m.any():
                    axS.plot(s["t_s"][m][-1], max(s["tip_zone_mm2"][m][-1], 1e-4), "o", color=colr,
                             ms=7, mec=SURF, mew=1.5)
            axS.axvline(t, color=INK3, lw=0.8)
            axS.set_yscale("log")
            axS.set_ylim(3e-3, 3.0)
            axS.set_xlim(0, t_end)
            axS.set_xlabel("time [s]", fontsize=10, color=INK2)
            axS.set_ylabel("melt within 1 mm of the tip [mm²]", fontsize=10, color=INK2)
            axS.set_title("Melt hanging within 1 mm under the nozzle",
                          fontsize=11.5, color=INK, loc="left")
            axS.grid(axis="y", color=GRID, lw=0.6, which="major")
            axS.spines[["top", "right"]].set_visible(False)
            axS.tick_params(labelsize=8.5, colors=INK2)
            axS.legend(loc="lower right", fontsize=9, frameon=False, ncol=2)

            fig.text(0.045, 0.030, "2-D planar model: melt treated as a conductor near the tip,"
                     " nominal viscosity 500 Pa·s and surface tension 0.03 N/m. The 10–20 µm jet"
                     " is far below the 25 µm cells and is not resolved.",
                     fontsize=9, color=INK3)
            # A NUMERICAL DEVICE, SO SAID ON SCREEN: without the projection both columns
            # drift apart from ~1.5 s (field on -0.42 mm, field off +0.27 mm at 4.5 s);
            # demonstrator/mew_nozzle's banner, and its -nosym switch.
            fig.text(0.045, 0.010, "The set-up is mirror-symmetric, and the melt is held so until it"
                     " touches the collector: without that, round-off pushes the columns sideways"
                     " (up to 0.4 mm).", fontsize=9, color=INK3)
            fig.savefig(os.path.join(tmp, f"f{n:05d}.png"), facecolor=SURF)
            plt.close(fig)
            if n % 20 == 0:
                print(f"  frame {n + 1}/{len(times)}", flush=True)

        hold = 2
        if a.endnote:
            import matplotlib.image as mpimg
            last = mpimg.imread(os.path.join(tmp, f"f{len(times) - 1:05d}.png"))
            fig = plt.figure(figsize=(16, 9), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(last); ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=SURF,
                                       alpha=0.88))
            fig.text(0.5, 0.56, a.endnote.split("|")[0], ha="center", va="center",
                     fontsize=22, color=INK, weight="bold")
            for k, line in enumerate(a.endnote.split("|")[1:]):
                fig.text(0.5, 0.47 - 0.05 * k, line, ha="center", va="center", fontsize=15,
                         color=INK2)
            fig.savefig(os.path.join(tmp, f"f{len(times):05d}.png"))
            plt.close(fig)
            hold = 4
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
               "-i", os.path.join(tmp, "f%05d.png"),
               "-vf", f"tpad=stop_mode=clone:stop_duration={hold},format=yuv420p",
               "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-movflags", "+faststart",
               a.out]
        subprocess.run(cmd, check=True)
        last = os.path.join(tmp, f"f{len(times) - 1:05d}.png")
        shutil.copy(last, os.path.splitext(a.out)[0] + "_last.png")
        print("wrote", a.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
