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


def layers(stem, volts):
    """RGB image (melt, metal, field shading) and the potential in the gas."""
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
    if FIELD_MODE["mode"] == "rel":
        emag = np.hypot(gx, gy) * GAP_MM            # field / (V / gap)
        top = 5.0
    else:
        emag = np.hypot(gx, gy) * volts             # kV/mm
        top = 4.0
    emag[~gas] = np.nan
    rgb = FIELD_CMAP(np.clip(np.nan_to_num(emag) / top, 0, 1))[..., :3]
    rgb[~gas] = 0.988
    f = np.clip(fl, 0, 1)[..., None]
    rgb = rgb * (1 - f) + MELT * f
    rgb[fl < -0.5] = METAL
    phg = np.where(gas, ph, np.nan)
    return rgb, phg, fl, dx, (nx, ny)


def draw_box(ax, stem, volts, outline_stem=None, zoom=None, contours=True):
    rgb, phg, fl, dx, (nx, ny) = layers(stem, volts)
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
             "Melt hanging under the nozzle (the paper's proposed oozing metric)", None),
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
    ap.add_argument("--single", action="store_true",
                    help="run A alone: the box and its own two time series")
    ap.add_argument("--relative-field", action="store_true",
                    help="shade field / (V / gap) instead of kV/mm")
    a = ap.parse_args()
    if a.relative_field:
        FIELD_MODE["mode"] = "rel"

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
            gs = None if a.single else fig.add_gridspec(
                2, 3, width_ratios=[1.45, 0.62, 0.62], height_ratios=[1.0, 0.80],
                left=0.045, right=0.985, top=0.865, bottom=0.105, wspace=0.17, hspace=0.42)
            # header
            fig.text(0.045, 0.955, "Melt electrowriting nozzle, simulated in 2-D (M3LB, lattice Boltzmann)",
                     fontsize=19, color=INK, weight="bold")
            fig.text(0.045, 0.915, "PCL at 120 °C  ·  0.4 mm nozzle  ·  6.5 kV  ·  5.5 mm to the collector"
                     "  ·  collector 2100 mm/min  ·  set-up from Ju, Omar, Mirihanage & Kilic (preprint)",
                     fontsize=11.5, color=INK2)
            fig.text(0.985, 0.952, f"t = {t:5.2f} s", fontsize=22, color=INK, ha="right",
                     family="monospace", weight="bold")
            # progress bar
            bar = fig.add_axes([0.80, 0.918, 0.185, 0.008])
            bar.barh(0, t / t_end, color=INK2, height=1); bar.set_xlim(0, 1)
            bar.axis("off")

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
                axA.text(0.5, 0.60, f"Stopped at {a.cut_ab:.1f} s: the melt has reached the collector",
                         transform=axA.transAxes, ha="center", fontsize=13, color=INK,
                         weight="bold", zorder=6)
                axA.text(0.5, 0.55, "and this model has no wetting law to let it stick there.",
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
            axS.set_title("Melt hanging under the nozzle: the paper's proposed oozing metric",
                          fontsize=11.5, color=INK, loc="left")
            axS.grid(axis="y", color=GRID, lw=0.6, which="major")
            axS.spines[["top", "right"]].set_visible(False)
            axS.tick_params(labelsize=8.5, colors=INK2)
            axS.legend(loc="lower right", fontsize=9, frameon=False, ncol=2)

            fig.text(0.045, 0.030, "2-D planar model: melt treated as a conductor near the tip,"
                     " nominal viscosity 500 Pa·s and surface tension 0.03 N/m. The 10–20 µm jet"
                     " is far below the 25 µm cells and is not resolved.",
                     fontsize=9, color=INK3)
            # MEASURED, NOT PHYSICS: both 0.255 mm^3/s columns start on the axis and
            # drift apart from ~1.5 s (field on -0.42 mm, field off +0.27 mm at 4.5 s)
            # in a mirror-symmetric set-up with Bo = 0.014 and no inertia.
            fig.text(0.045, 0.010, "The sideways drift of the two columns (opposite directions, up to"
                     " 0.4 mm) is numerical symmetry breaking in a mirror-symmetric set-up, not a"
                     " prediction.", fontsize=9, color=INK3)
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
