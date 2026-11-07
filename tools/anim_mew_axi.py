#!/usr/bin/env python3
"""Animations of demonstrator/mew_axi's jet sink (-jet): the round MEW drop in r-z,
with the fibres' flow taken out at its apex, without and with the fan.

    python3 tools/anim_mew_axi.py iso  DIR results/mew_axi/mew_axi_jet_iso.mp4  [--fps 20]
    python3 tools/anim_mew_axi.py heat DIR results/mew_axi/mew_axi_jet_heat.mp4 [--fps 20]
    python3 tools/anim_mew_axi.py apex DIR results/mew_axi/mew_axi_jet_apex.mp4 [--fps 20]

DIR holds the frames, series and logs of these runs, made with -tframe 0.1
(isothermal) or 0.25 (heat): the runs kept in results/mew_axi wrote one frame a
second, too few to animate. The series agree with those runs.

    iso_none   iso_fan   iso_nofan                     mew_axi [-jet Q]
    heat_none  heat_fan  heat_nofan  heat_fan_h300     mew_axi -heat -nu 0.0416666667 [-h 300] [-jet Q]

Q = 0.0036354 mm^3/s is the fan's 11.5 um fibres laid at 35 mm/s, 0.011552 the
20.5 um fibres without the fan.

iso   the three drops at 120 C side by side, mirrored about the axis, with the
      equipotentials in the air; on the right their lowest point and the melt
      held within 1 mm of the tip against time. A drop stops at its first contact
      with the collector: after it the static plate, not the experiment, decides
      (mew_axi's banner).
heat  the four drops coloured by temperature over 60 s, the fourth with the
      fan's cooling as well (h = 300 W/m^2/K, nominal).
apex  a close-up of the no-fan isothermal drop's apex, cell by cell, with the box
      the jet is drawn from (rebuilt from the frame as mew_axi places it). It
      shows the artefact recorded in mew_axi's banner: the suction dimples the
      surface on the axis, and once that surface is within about a cell of the
      box the free surface creates mass.

Frames are drawn by matplotlib (ffmpeg here has no drawtext) into a temporary
directory beside the output, encoded H.264 by ffmpeg, and deleted.
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

INK, INK2, INK3, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
SURF_RGB = np.array([0.988, 0.988, 0.984])
METAL = np.array([0.47, 0.47, 0.45])
MELT = np.array([0.165, 0.471, 0.839])         # isothermal melt, as in anim_mew_nozzle
TMAP = LinearSegmentedColormap.from_list("T", ["#f0a070", "#e0703e", "#b8461b", "#6e2a10"])   # anim_mew_jet's
TN = Normalize(25.0, 120.0)
HALF_W = 1.5                                     # mm shown either side of the axis
FEED = 0.0208                                    # mm^3/s, the piston's
JET_R, JET_DEPTH = 2, 4                          # mew_axi's default box at 50 um: 0.1 mm, four cells in

# One colour per jet, in a fixed order (checked with the dataviz validator; the
# aqua is below 3:1 on the page, so every line is also labelled in ink).
ISO = [("iso_none", "no jet", "", "#2a78d6", "no jet"),
       ("iso_fan", "the fan's jet", "0.0036 mm³/s: 11.5 µm fibres", "#eb6834", "fan"),
       ("iso_nofan", "the no-fan jet", "0.0116 mm³/s: 20.5 µm fibres", "#1baf7a", "no fan")]
HEAT = [("heat_none", "no jet", "still air: h 90 W/m²K", "#2a78d6", "no jet"),
        ("heat_fan", "the fan's jet", "0.0036 mm³/s, h 90", "#eb6834", "fan"),
        ("heat_nofan", "the no-fan jet", "0.0116 mm³/s, h 90", "#1baf7a", "no fan"),
        ("heat_fan_h300", "the fan's jet + cooling", "0.0036 mm³/s, h 300", "#6250d6", "fan + cooling")]


def frame_index(prefix):
    out = {}
    for f in glob.glob(prefix + "_t*_fill.npy"):
        m = re.search(r"_t([0-9.]+)_fill\.npy$", f)
        if m:
            out[float(m.group(1))] = f[:-len("_fill.npy")]
    return dict(sorted(out.items()))


def read_series(path):
    with open(path) as f:
        r = csv.DictReader(f)
        cols = {k: [] for k in r.fieldnames}
        for row in r:
            if any(row.get(k) in (None, "") for k in cols):      # a half-flushed last line
                continue
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


class Run:
    def __init__(self, d, stem, label, sub, colour, short):
        self.stem, self.label, self.sub, self.colour, self.short = os.path.join(d, stem), label, sub, colour, short
        self.frames = frame_index(self.stem)
        if not self.frames:
            raise SystemExit("no frames for " + self.stem)
        self.ser = read_series(self.stem + "_series.csv")
        log = open(self.stem + ".log").read()
        m = re.search(r"first contact with the collector at t = ([0-9.]+)", log)
        self.contact = float(m.group(1)) if m else None
        self.dx = float(re.search(r"dx ([0-9.]+) um", log).group(1)) * 1e-3     # mm

    def frame(self, t):
        """The frame nearest t, never one after the run's first contact."""
        stop = self.contact if self.contact is not None else 1e9
        keys = [k for k in self.frames if k <= stop + 1e-9]
        return self.frames[min(keys, key=lambda a: abs(a - t))]

    def at(self, key, t):
        s = self.ser
        if key not in s:                                  # a series from before the column existed
            return float("nan")
        tt = min(t, self.contact) if self.contact is not None else t
        return float(s[key][np.argmin(np.abs(s["t_s"] - tt))])

    def touched(self, t):
        return self.contact is not None and t >= self.contact


def fill_nan(T, liquid):
    """Liquid cells the heat solver has not reached yet (a fresh surface cell) take their neighbours' mean."""
    T = T.copy()
    for _ in range(4):
        hole = liquid & ~np.isfinite(T)
        if not hole.any():
            break
        P = np.pad(T, 1, constant_values=np.nan)
        ny, nx = T.shape
        stack = np.stack([P[1 + a:1 + a + ny, 1 + b:1 + b + nx] for a in (-1, 0, 1) for b in (-1, 0, 1)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(stack, axis=0)
        T[hole] = m[hole]
    return T


def mirrored(a):
    return np.concatenate([a[:, ::-1], a], axis=1)


def drop_layers(stem, heat):
    fl = np.load(stem + "_fill.npy").astype(float)
    ny, nx = fl.shape
    f = np.clip(fl, 0, 1)[..., None]
    if heat:
        T = fill_nan(np.load(stem + "_T.npy").astype(float), fl > 0)
        col = TMAP(TN(np.nan_to_num(T, nan=60.0)))[..., :3]
    else:
        col = np.broadcast_to(MELT, fl.shape + (3,))
    rgb = SURF_RGB * (1 - f) + col * f
    rgb[fl < -0.5] = METAL
    ph = np.load(stem + "_phi.npy").astype(float)
    gas = fl == 0
    # the top and the far side are zero-field frames, whose stored potential is a
    # structural zero: mirror the neighbour in, as the boundary condition does
    ph[-1, gas[-1]] = ph[-2, gas[-1]]
    ph[:, -1] = ph[:, -2]
    phg = np.where(gas, ph, np.nan)
    return mirrored(rgb), mirrored(phg), mirrored(fl), nx, ny


def style_box(ax):
    for s in ax.spines.values():
        s.set_color(INK3)
        s.set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=3)


def draw_drop(ax, run, t, heat, first):
    rgb, phg, full, nx, ny = drop_layers(run.frame(t), heat)
    dx = run.dx
    ax.imshow(rgb, origin="lower", extent=(-nx * dx, nx * dx, -dx, (ny - 1) * dx), interpolation="bilinear")
    x = (np.arange(-nx, nx) + 0.5) * dx
    y = (np.arange(ny) - 0.5) * dx
    if np.isfinite(phg).any():
        ax.contour(x, y, phg, levels=np.linspace(0.1, 0.9, 9), colors=INK3, linewidths=0.5, alpha=0.75)
    ax.contour(x, y, np.clip(full, 0, 1), levels=[0.5], colors=INK, linewidths=0.6)
    ax.set_xlim(-HALF_W, HALF_W)
    ax.set_ylim(-dx, (ny - 1) * dx)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks(range(0, 7))
    if first:
        ax.set_ylabel("height above the collector [mm]", fontsize=9, color=INK2)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("r [mm], mirrored", fontsize=8.5, color=INK2)
    style_box(ax)
    ax.add_patch(Rectangle((0.0, 1.058), 0.045, 0.022, transform=ax.transAxes, color=run.colour, clip_on=False))
    ax.text(0.065, 1.052, run.label, transform=ax.transAxes, fontsize=11, color=INK, va="bottom", weight="bold")
    ax.text(0.0, 1.012, run.sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    lines = ["lowest point %.2f mm" % run.at("z_low_mm", t),
             "melt at the tip %.3f mm³" % run.at("tip_zone_mm3", t)]
    if run.at("jet_asked_mm3", t) > 0:
        lines.append("taken by the jet %.3f mm³" % run.at("jet_mm3", t))
    if heat:
        lines.append("melt below the tip %.0f °C" % run.at("T_gap_mean_C", t))
    ax.apply_aspect()                                 # the equal aspect shrinks the box: place below its real edge
    pos = ax.get_position()
    for k, s in enumerate(lines):
        ax.figure.text(pos.x0, pos.y0 - 0.058 - 0.022 * k, s, fontsize=8.8, color=INK2, va="top")
    if run.touched(t):
        ax.text(0.5, 0.06, "touched the collector\nat %.2f s" % run.contact, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURF, ec=INK3, lw=0.6))


def spread(ys, lo, hi, gap):
    """Label heights pushed apart to at least `gap` (data units), order kept, inside [lo, hi]."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for _ in range(50):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            d = out[b] - out[a]
            if d < gap:
                push = 0.5 * (gap - d)
                out[a] -= push
                out[b] += push
                moved = True
        out = np.clip(out, lo, hi)
        if not moved:
            break
    return out


def series(ax, runs, t, key, ylab, ylim, tmax, title):
    ends = []
    for r in runs:
        s = r.ser
        stop = min(t, r.contact) if r.contact is not None else t
        keep = s["t_s"] <= stop + 1e-9
        tt, v = s["t_s"][keep], s[key][keep]
        if len(tt) == 0:
            continue
        if key == "z_low_mm" and r.touched(t):
            tt, v = np.append(tt, r.contact), np.append(v, 0.0)
        ax.plot(tt, v, color=r.colour, lw=1.9, solid_capstyle="round")
        ax.plot([tt[-1]], [v[-1]], "o", color=r.colour, ms=6, mec=SURF, mew=1.2, clip_on=False)
        ends.append((tt[-1], v[-1], r.short))
    if ends:
        span = ylim[1] - ylim[0]
        ys = spread([e[1] for e in ends], ylim[0] + 0.03 * span, ylim[1] - 0.03 * span, 0.075 * span)
        for (x, _, lab), y in zip(ends, ys):
            ax.text(x + 0.012 * tmax, y, lab, fontsize=8.5, color=INK2, va="center")
    ax.axvline(t, color=INK3, lw=0.7, ls=":")
    ax.set_xlim(0, tmax)
    ax.set_ylim(*ylim)
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylab, fontsize=9, color=INK2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")


def side_by_side(fig, runs, t, tmax, heat, title, sub, foot):
    n = len(runs)
    left, right = 0.045, 0.585
    gs = fig.add_gridspec(1, n, left=left, right=right, top=0.80, bottom=0.21, wspace=0.10)
    for k, r in enumerate(runs):
        draw_drop(fig.add_subplot(gs[0, k]), r, t, heat, k == 0)
    g2 = fig.add_gridspec(2, 1, left=0.655, right=0.935, top=0.80, bottom=0.21, hspace=0.42)
    series(fig.add_subplot(g2[0, 0]), runs, t, "z_low_mm", "mm", (0, 5.7), tmax,
           "lowest point of the melt above the collector")
    tipmax = max(float(np.max(r.ser["tip_zone_mm3"])) for r in runs)
    series(fig.add_subplot(g2[1, 0]), runs, t, "tip_zone_mm3", "mm³", (0, 1.12 * tipmax), tmax,
           "melt held within 1 mm of the tip face")
    fig.axes[-1].set_xlabel("time since the field and feed start [s]", fontsize=9, color=INK2)
    fig.text(0.045, 0.945, title, fontsize=17, color=INK, weight="bold")
    fig.text(0.045, 0.905, sub, fontsize=10.5, color=INK2)
    fig.text(0.935, 0.945, "t = %5.1f s" % t, fontsize=17, color=INK, ha="right", family="monospace")
    if heat:
        cax = fig.add_axes([0.655, 0.115, 0.28, 0.016])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=TN, cmap=TMAP), cax=cax, orientation="horizontal")
        cb.set_label("melt temperature [°C]; hatched: the band it freezes in, 30-40 °C (nominal)",
                     fontsize=8.5, color=INK2)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        cb.outline.set_edgecolor(INK3)
        cax.axvspan(30, 40, ymin=0, ymax=1, fill=False, hatch="////", ec=INK, lw=0)
    for k, line in enumerate(foot):
        fig.text(0.045, 0.060 - 0.024 * k, line, fontsize=9, color=INK3)


def apex_frame(fig, run, t, tmax, dimple):
    stem = run.frame(t)
    fl = np.load(stem + "_fill.npy").astype(float)
    ny, nx = fl.shape
    dx = run.dx
    solid = fl < -0.5
    ya = None
    for y in range(1, ny):
        if solid[y, :JET_R].any():
            break
        if (fl[y, :JET_R] > 0).any():
            ya = y
            break
    zc = (ya - 0.5) * dx if ya is not None else 3.0
    y0, y1 = zc - 0.30, zc + 0.90
    ax = fig.add_axes([0.045, 0.17, 0.42, 0.66])
    f = np.clip(mirrored(fl), 0, 1)[..., None]
    rgb = SURF_RGB * (1 - f) + MELT * f
    rgb[mirrored(fl) < -0.5] = METAL
    ax.imshow(rgb, origin="lower", extent=(-nx * dx, nx * dx, -dx, (ny - 1) * dx), interpolation="nearest")
    for k in range(-14, 15):
        ax.axvline(k * dx, color=GRID, lw=0.4)
    for k in range(int(y0 / dx) - 1, int(y1 / dx) + 2):
        ax.axhline(k * dx, color=GRID, lw=0.4)
    ax.axvline(0, color=INK3, lw=0.8, ls="--")
    if ya is not None and not run.touched(t):
        b0 = (ya + JET_DEPTH - 1) * dx                    # row k spans [(k-1) dx, k dx]
        ax.add_patch(Rectangle((-JET_R * dx, b0), 2 * JET_R * dx, 2 * JET_R * dx, fill=False,
                               ec="#eb6834", lw=2.0))
        ax.annotate("the jet is drawn\nfrom this box", xy=(JET_R * dx, b0 + JET_R * dx), xytext=(0.36, b0 + 0.25),
                    fontsize=9.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("r [mm], mirrored (dashed: the axis)", fontsize=9, color=INK2)
    ax.set_ylabel("height above the collector [mm]", fontsize=9, color=INK2)
    style_box(ax)
    ax.set_title("the apex, cell by cell (50 µm cells; shade = how full a cell is)", fontsize=10.5,
                 color=INK, loc="left")
    if run.touched(t):
        ax.text(0.5, 0.5, "touched the collector at %.2f s" % run.contact, transform=ax.transAxes, ha="center",
                fontsize=11, color=INK, bbox=dict(boxstyle="round,pad=0.4", fc=SURF, ec=INK3, lw=0.6))

    g2 = fig.add_gridspec(2, 1, left=0.56, right=0.935, top=0.80, bottom=0.21, hspace=0.42)
    a1 = fig.add_subplot(g2[0, 0])
    tt, dm, gap = dimple
    keep = tt <= t + 1e-9
    a1.plot(tt[keep], dm[keep], color="#1baf7a", lw=1.9)
    a1.plot(tt[keep], gap[keep], color="#eb6834", lw=1.9)
    if keep.any():
        y_dm, y_gap = spread([dm[keep][-1], gap[keep][-1]], -0.8, 4.0, 0.4)
        a1.text(tt[keep][-1] + 0.3, y_dm, "dimple", fontsize=8.5, color=INK2, va="center")
        a1.text(tt[keep][-1] + 0.3, y_gap, "box to surface", fontsize=8.5, color=INK2, va="center")
    a1.axhline(0, color=INK3, lw=0.6)
    a1.set_xlim(0, tmax)
    a1.set_ylim(-1, 4.2)
    a1.set_ylabel("cells", fontsize=9, color=INK2)
    a1.set_title("the surface on the axis: above the ring round it, and below the box", fontsize=10.5,
                 color=INK, loc="left")
    a2 = fig.add_subplot(g2[1, 0])
    s = run.ser
    stop = min(t, run.contact) if run.contact is not None else t
    keep = (s["t_s"] <= stop + 1e-9) & (s["t_s"] >= 1.0)     # before 1 s the jet taken is too small to divide by
    made = 100.0 * s["ledger"] * FEED * s["t_s"] / np.maximum(s["jet_mm3"], 1e-30)
    a2.plot(s["t_s"][keep], made[keep], color=INK2, lw=1.9)
    a2.set_xlim(0, tmax)
    a2.set_ylim(-0.3, 8)
    a2.set_ylabel("% of what the jet took", fontsize=9, color=INK2)
    a2.set_xlabel("time since the field and feed start [s]", fontsize=9, color=INK2)
    a2.set_title("melt the free surface has created", fontsize=10.5, color=INK, loc="left")
    for a in (a1, a2):
        a.axvline(t, color=INK3, lw=0.7, ls=":")
        a.set_facecolor(SURF)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(INK3)
        a.tick_params(colors=INK2, labelsize=8.5)
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.text(0.045, 0.945, "The jet sink's artefact, close up", fontsize=17, color=INK, weight="bold")
    fig.text(0.045, 0.905, "the no-fan jet (0.0116 mm³/s), isothermal at 120 °C; mew_axi -jet 0.011552",
             fontsize=10.5, color=INK2)
    fig.text(0.935, 0.945, "t = %5.1f s" % t, fontsize=17, color=INK, ha="right", family="monospace")
    foot = ["The jet is thinner than a cell, so its flow is drawn out of the melt in a box four cells above the lowest melt "
            "within 0.1 mm of the axis.",
            "Its suction pulls the surface on the axis up towards the box (a real jet pulls the apex out). Once the "
            "surface is within about a cell of the box",
            "-- the dimple from below, and from 25 s the thinning column's sides too -- the free surface creates mass: "
            "7.2 % of what the jet took by contact. Not diagnosed."]
    for k, line in enumerate(foot):
        fig.text(0.045, 0.085 - 0.024 * k, line, fontsize=9, color=INK3)


def dimple_series(run):
    """The axis surface's height above the lowest of the three columns beside it, and the box's lower
    edge above the axis surface, in cells, per frame (as read off the fill)."""
    out = []
    for t, stem in run.frames.items():
        if run.contact is not None and t > run.contact:
            break
        fl = np.load(stem + "_fill.npy").astype(float)
        solid = fl < -0.5
        ny = fl.shape[0]
        ytip = min([y for y in range(5, ny) if solid[y, :20].any()] or [ny])
        b = np.where(solid, 0.0, fl)[:ytip]
        h = []
        for x in range(4):
            col = b[:, x]
            ys = np.where(col > 0)[0]
            if len(ys) == 0:
                h.append(np.nan)
                continue
            k, s = ys[0], 0.0
            while k < len(col) and 0 < col[k] < 1:
                s += col[k]
                k += 1
            h.append(k - s)
        ya = next((y for y in range(1, ytip) if (b[y, :JET_R] > 0).any()), None)
        if ya is None or not np.isfinite(h[0]):
            continue
        out.append((t, h[0] - np.nanmin(h[1:4]), (ya + JET_DEPTH) - h[0]))
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("iso", "heat", "apex"))
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max", type=int, default=0, help="render only the first N frames (a check)")
    ap.add_argument("--still", type=float, default=None, help="write one frame at this time to OUT (a .png)")
    a = ap.parse_args()

    if a.which == "heat":
        runs = [Run(a.dir, *r) for r in HEAT]
    else:
        runs = [Run(a.dir, *r) for r in ISO]
    if a.which == "iso":
        clock = runs[2]                                 # the no-fan run reaches contact last
        tend = runs[2].contact + 0.3
    elif a.which == "heat":
        clock = runs[0]
        tend = max(clock.frames)
    else:
        runs = [runs[2]]
        clock = runs[0]
        tend = runs[0].contact + 0.3
    times = [t for t in clock.frames if t <= tend + 1e-9]
    if a.max:
        times = times[:a.max]
    if a.still is not None:
        times = [min(times, key=lambda x: abs(x - a.still))]
    tmax = 33.0 if a.which != "heat" else 60.0
    dim = dimple_series(runs[0]) if a.which == "apex" else None

    tmp = tempfile.mkdtemp(prefix="anim_mew_axi_", dir=os.path.dirname(os.path.abspath(a.out)))
    try:
        for n, t in enumerate(times):
            fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURF)
            if a.which == "iso":
                side_by_side(fig, runs, t, tmax, False,
                             "The round drop with the fibres' flow taken out at its apex",
                             "isothermal at 120 °C: PCL 820 Pa s, 6.5 kV over 5.5 mm, fed 0.0208 mm³/s; "
                             "r-z on 50 µm cells (mew_axi -jet)",
                             ["The jet is below the grid: its flow -- the fibres', laid at 35 mm/s -- is drawn "
                              "out of the melt four cells above the apex.",
                              "A drop stops at its first contact: after it the static collector, not the experiment, "
                              "decides. Grey lines: equipotentials in the air.",
                              "Isothermal contact is not grid-converged (the tip sharpens on finer cells); with heat "
                              "the drop stalls instead -- see the other video."])
            elif a.which == "heat":
                side_by_side(fig, runs, t, tmax, True,
                             "The round drop with heat: how much melt the tip keeps",
                             "PCL 820 Pa s (Arrhenius), 6.5 kV over 5.5 mm, fed 0.0208 mm³/s at 120 °C, air "
                             "25 °C; r-z on 50 µm cells (mew_axi -heat -jet)",
                             ["The jet's flow is the fibres' (20.5 µm without the fan, 11.5 µm with it, at 35 mm/s), "
                              "drawn out of the melt four cells above the apex.",
                              "h = 90 W/m²K is still air; 300 stands in for the fan's own cooling (nominal). Below "
                              "30-40 °C the melt is held by a drag (crystallising).",
                              "The jet's suction dimples the apex, and the free surface then creates 1-3 % of what "
                              "the jet takes here (mew_axi's banner)."])
            else:
                apex_frame(fig, runs[0], t, tmax, dim)
            fig.savefig(os.path.join(tmp, "f%05d.png" % n), facecolor=SURF)
            plt.close(fig)
            if n % 40 == 0:
                print("  frame %d/%d" % (n + 1, len(times)), flush=True)
        if a.still is not None:
            shutil.copy(os.path.join(tmp, "f00000.png"), a.out)
            print("wrote", a.out, "(t = %.2f s)" % times[0])
            return
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
               "-i", os.path.join(tmp, "f%05d.png"),
               "-vf", "tpad=stop_mode=clone:stop_duration=3,format=yuv420p",
               "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", a.out]
        subprocess.run(cmd, check=True)
        print("wrote", a.out, "(%d frames at %d fps, + 3 s held)" % (len(times), a.fps))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
