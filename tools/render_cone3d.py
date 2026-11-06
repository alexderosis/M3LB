#!/usr/bin/env python3
"""Animation for demonstrator/mew_cone3d.

    python3 tools/render_cone3d.py <prefix> <out.mp4> [--fps 8] [--tmax T] [--dx 0.05] [--lc 5]

<prefix> is the -out the run was given: it reads <prefix>_series.csv and the
frames <prefix>_tTTT.TT_{fill,phi}.npy (z, y, x; fill -1 for metal), and writes
<out.mp4> and <out>_last.png.

THE SURFACE IS FOUND ALONG RAYS FROM THE AXIS, which assumes every horizontal
slice of the melt below the tip is star-shaped about the axis: true for a drop,
a cone and a thread on the axis, false for melt the collector has dragged away
or a drop that has left the axis. Each ray stops where the bilinear fill first
falls below 1/2. Frames after first contact are drawn but their surface is
only the part still star-shaped about the nozzle's axis.

THE END IS DRAWN AGAINST TAYLOR'S CONE; NO ANGLE IS FITTED. The first version
fitted a half-angle to r(z), the surface radius averaged over the rays, between
80 % and 20 % of the neck's radius, and wanted five rows of it. On the 50 um
grid that cannot see a Taylor cone at all -- a 49.3-degree cone on a six-cell
neck spans about three such rows -- and fewer rows would not rescue it: the
same band through a rounded end 1.36 times as long as a hemisphere has
Taylor's slope exactly (its secant is 1.580 a/c; derived, not measured). So
the last panel draws the end of the melt at true scale, mirrored about the
axis, with Taylor's static cone (a perfect conductor at the critical field)
set on the same apex, which is where the fill on the axis falls through 1/2.
A cone feeding a jet need not have 49.3 degrees: the comparison describes the
shape, it does not test it.

THE EDGE OF THE MODEL'S CONDUCTOR IS DRAWN (orange, at GAP - lc). Below it the
melt is an insulator, so the field stops pulling there, and the first full run
(2026-11-06) ran its end down onto that plane and parked it there for the last
7.6 s: the frames from the first crossing on say so.
"""
import argparse
import csv
import glob
import math
import os
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

SURF = "#fcfcfb"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MELT = np.array([0.165, 0.471, 0.839])
METAL = (0.47, 0.47, 0.45)
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "field", ["#fcfcfb", "#ece9f6", "#cfc8ea", "#a99bd8", "#7d68c2", "#5440a6"])
TAYLOR = 49.29
GAP, R_BORE, R_TIP, V_KV = 5.5, 0.2, 0.5, 6.5


def frame_index(prefix):
    out = {}
    for f in glob.glob(prefix + "_t*_fill.npy"):
        t = float(f[len(prefix) + 2:-len("_fill.npy")])
        out[t] = f[:-len("_fill.npy")]
    return dict(sorted(out.items()))


def read_series(path):
    with open(path) as f:
        r = csv.DictReader(f)
        cols = {k: [] for k in r.fieldnames}
        for row in r:
            if any(row.get(k) in (None, "") for k in cols):
                continue
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def surface_rays(fl, ztip, nth=72, dr=0.25):
    """r[z, k]: the surface radius (cells) on ray k at every row below the tip."""
    nz, ny, nx = fl.shape
    xc = 0.5 * nx                        # axis on a cell corner; cell i's centre at i + 0.5
    th = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    rs = np.arange(0.0, 0.5 * nx - 1.0, dr)
    X = xc + np.outer(np.cos(th), rs) - 0.5      # in cell-index coordinates of the centres
    Y = xc + np.outer(np.sin(th), rs) - 0.5
    i0 = np.clip(np.floor(X).astype(int), 0, nx - 2); fx = np.clip(X - i0, 0, 1)
    j0 = np.clip(np.floor(Y).astype(int), 0, ny - 2); fy = np.clip(Y - j0, 0, 1)
    liq = np.clip(fl, 0.0, 1.0)
    r = np.zeros((ztip, nth))
    for z in range(1, ztip):
        s = liq[z]
        v = (s[j0, i0] * (1 - fx) * (1 - fy) + s[j0, i0 + 1] * fx * (1 - fy)
             + s[j0 + 1, i0] * (1 - fx) * fy + s[j0 + 1, i0 + 1] * fx * fy)
        for k in range(nth):
            below = np.nonzero(v[k] < 0.5)[0]
            if len(below) == 0:
                r[z, k] = rs[-1]
            elif below[0] == 0:
                r[z, k] = 0.0
            else:
                b = below[0]                     # interpolate the 1/2 crossing
                a = v[k, b - 1] - v[k, b]
                r[z, k] = rs[b - 1] + (dr * (v[k, b - 1] - 0.5) / a if a > 0 else 0.0)
    return th, r


def apex_height(fl, ztip, dx):
    """Height [mm] at which the fill on the axis falls through 1/2 below the nozzle, or None."""
    nz, ny, nx = fl.shape
    a = np.clip(fl[:, ny // 2 - 1:ny // 2 + 1, nx // 2 - 1:nx // 2 + 1], 0.0, 1.0).mean(axis=(1, 2))
    z = ztip - 1
    if a[z] < 0.5:
        return None
    while z > 1 and a[z - 1] >= 0.5:
        z -= 1
    a0, a1 = a[z - 1], a[z]
    return (z - 1.5) * dx + dx * (0.5 - a0) / (a1 - a0)


def field_slice(fl, ph, dx):
    """x-z plane beside the axis: RGB with field shading, and the potential in the gas."""
    nz, ny, nx = fl.shape
    j = ny // 2 - 1
    f = fl[:, j, :]
    p = ph[:, j, :].astype(float)
    gas = (f > -0.5) & (f < 0.5)
    # the frames (sides and top) report a structural zero, not a potential:
    # mirror the neighbour in, as anim_mew_nozzle.py does
    p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
    top = gas[-1, :]
    p[-1, top] = p[-2, top]
    gz, gx = np.gradient(p, dx)
    e = np.hypot(gx, gz) * V_KV                     # kV/mm (p in units of V, dx in mm)
    e[~gas] = np.nan
    rgb = FIELD_CMAP(np.clip(np.nan_to_num(e) / 8.0, 0, 1))[..., :3]
    rgb[~gas] = 0.988
    a = np.clip(f, 0, 1)[..., None]
    rgb = rgb * (1 - a) + MELT * a
    rgb[f < -0.5] = METAL
    return rgb, np.where(gas, p, np.nan)


def view_half_width(zoom):
    """Half-width [mm] of the view: the whole box, or a close-up from height zoom up."""
    return 2.0 if zoom <= 0 else max(0.6, 0.5 * (GAP + 1.25 - zoom))


def draw3d(ax, th, r, ztip, dx, zoom=0.0, apex=None, zcut=0.0):
    nz = r.shape[0]
    zz = (np.arange(nz) - 0.5) * dx
    rows = [z for z in range(1, ztip) if r[z].max() > 0 and zz[z] >= zoom]
    if rows:
        z0, z1 = min(rows), ztip - 1
        R = r[z0:z1 + 1] * dx
        T = np.append(th, th[0])
        R = np.concatenate([R, R[:, :1]], axis=1)
        Z = np.repeat(zz[z0:z1 + 1, None], len(T), axis=1)
        if apex is not None and zoom <= apex < zz[z0]:
            # close the end on the axis, where the fill falls through 1/2
            R = np.vstack([np.zeros((1, len(T))), R])
            Z = np.vstack([np.full((1, len(T)), apex), Z])
        X = R * np.cos(T)[None, :]
        Y = R * np.sin(T)[None, :]
        ls = LightSource(azdeg=315, altdeg=35)
        ax.plot_surface(X, Y, Z, color=MELT, shade=True, lightsource=ls, linewidth=0,
                        antialiased=True, rcount=200, ccount=200)
    # the nozzle: a grey cylinder above the tip face, and the collector
    T = np.linspace(0, 2 * np.pi, 60)
    Zc = np.array([GAP, GAP + 1.25])
    ax.plot_surface(R_TIP * np.cos(T)[None, :].repeat(2, 0), R_TIP * np.sin(T)[None, :].repeat(2, 0),
                    Zc[:, None].repeat(len(T), 1), color=METAL, shade=True, linewidth=0)
    hw = view_half_width(zoom)
    if zoom <= 0:
        g = np.linspace(-2, 2, 2)
        GX, GY = np.meshgrid(g, g)
        ax.plot_surface(GX, GY, np.zeros_like(GX), color=(0.82, 0.82, 0.80), linewidth=0, alpha=0.9)
    if zcut > max(zoom, 0.0):                     # where the model's conductor ends
        ax.plot([-hw, hw, hw, -hw, -hw], [-hw, -hw, hw, hw, -hw], [zcut] * 5, color=ORANGE, lw=0.9,
                ls=(0, (3, 2)))
    z0 = max(zoom, 0.0)
    ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw); ax.set_zlim(z0, GAP + 1.25)
    ax.set_box_aspect((2 * hw, 2 * hw, GAP + 1.25 - z0))
    ax.view_init(elev=12, azim=-60)
    ax.set_xlabel("x [mm]", fontsize=9, color=INK2, labelpad=2)
    ax.set_ylabel("y [mm]", fontsize=9, color=INK2, labelpad=2)
    ax.set_zlabel("height [mm]", fontsize=9, color=INK2, labelpad=2)
    ax.tick_params(labelsize=7, colors=INK2, pad=0)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor(SURF); a.pane.set_edgecolor(GRID)


def draw_tip(ax, rbar, apex, dx, half=0.45, below=0.08, above=0.52):
    """The end of the melt at true scale, mirrored about the axis, with Taylor's
    cone (dashed) set on the same apex."""
    zz = (np.arange(len(rbar)) - 0.5) * dx
    lo, hi = (0.0, below + above) if apex is None else (apex - below, apex + above)
    ax.set_xlim(-half, half); ax.set_ylim(lo, hi)
    ax.set_aspect("equal"); ax.set_anchor("W")
    if hi > GAP:                                  # the nozzle, while the end is near it
        for s in (-1, 1):
            ax.fill_between([s * R_BORE, s * R_TIP], GAP, hi, color=METAL, lw=0)
        ax.fill_between([-R_BORE, R_BORE], GAP, hi, color=MELT, lw=0)
    if apex is not None:
        rows = [z for z in range(1, len(rbar)) if rbar[z] > 0 and apex < zz[z] <= hi + dx]
        ys = [apex] + [zz[z] for z in rows]
        xs = [0.0] + [rbar[z] * dx for z in rows]
        if rows and rows[-1] == len(rbar) - 1:         # the last row reaches the tip face
            ys.append(GAP); xs.append(xs[-1])
        ys, xs = np.array(ys), np.array(xs)
        ax.fill_betweenx(ys, -xs, xs, color=MELT, lw=0)
        w = (hi - apex) * math.tan(math.radians(TAYLOR))
        ax.plot([-w, 0.0, w], [hi, apex, hi], color=INK, lw=1.1, ls=(0, (3, 2)))
    ax.text(1.04, 0.97, "blue: the melt,\naveraged around\nthe axis\n\ndashed: Taylor's\n49.3° cone on\nthe same apex",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="top", linespacing=1.3)
    ax.set_title("The end of the melt at true scale", fontsize=11.5, color=INK, loc="left")
    ax.set_xlabel("x [mm]", fontsize=9, color=INK2, labelpad=1)
    ax.set_ylabel("height [mm]", fontsize=9, color=INK2)
    ax.tick_params(labelsize=8, colors=INK2)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--tmax", type=float, default=0.0)
    ap.add_argument("--dx", type=float, default=0.05, help="cell size [mm]")
    ap.add_argument("--touch", type=float, default=0.0, help="first contact [s], to mark it")
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="close-up: show heights from this [mm] up to the nozzle's top")
    ap.add_argument("--snap", action="store_true", help="draw only the latest frame, to <out> as a PNG")
    ap.add_argument("--lc", type=float, default=5.0, help="the run's -lc [mm]: the melt conducts this far below the tip")
    a = ap.parse_args()

    idx = frame_index(a.prefix)
    if not idx:
        raise SystemExit("no frames for " + a.prefix)
    ser = read_series(a.prefix + "_series.csv")
    times = [t for t in idx if a.tmax <= 0 or t <= a.tmax + 1e-9]
    t_end = max(times)
    if a.snap:
        t_end = max(t_end, float(ser["t_s"][-1]))
    dx = a.dx
    ztip = int(round(GAP / dx)) + 1
    zcut = GAP - a.lc                             # the conductor's lower edge [mm]
    below = ser["z_low_mm"] < zcut
    t_cut = float(ser["t_s"][below][0]) if zcut > 0 and below.any() else None

    if a.snap:
        times = times[-1:]
    tmp = tempfile.mkdtemp(prefix="cone3d_", dir=os.path.dirname(os.path.abspath(a.out)))
    try:
        for n, t in enumerate(times):
            fl = np.load(idx[t] + "_fill.npy")
            ph = np.load(idx[t] + "_phi.npy")
            th, r = surface_rays(fl, ztip)
            rbar = r.mean(axis=1)
            apex = apex_height(fl, ztip, dx)

            fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURF)
            fig.text(0.04, 0.955, "Melt electrowriting nozzle in 3-D (M3LB, lattice Boltzmann)",
                     fontsize=19, color=INK, weight="bold")
            fig.text(0.04, 0.915, "PCL at 120 °C, 820 Pa·s  ·  0.4 mm round nozzle  ·  6.5 kV  ·  5.5 mm gap"
                     "  ·  feed 0.0208 mm³/s  ·  whole drop conducting  ·  50 µm cells", fontsize=11.5,
                     color=INK2)
            fig.text(0.985, 0.952, f"t = {t:5.2f} s", fontsize=22, color=INK, ha="right",
                     family="monospace", weight="bold")
            bar = fig.add_axes([0.80, 0.918, 0.185, 0.008])
            bar.barh(0, t / t_end, color=INK2, height=1); bar.set_xlim(0, 1); bar.axis("off")

            ax3 = fig.add_axes([0.0, 0.08, 0.36, 0.80], projection="3d")
            ax3.set_facecolor(SURF)
            draw3d(ax3, th, r, ztip, dx, a.zoom, apex, zcut)
            fig.text(0.04, 0.81, "The melt, seen from outside", fontsize=13, color=INK)

            axs = fig.add_axes([0.41, 0.14, 0.25, 0.70] if a.zoom > 0 else [0.37, 0.14, 0.27, 0.70])
            rgb, phg = field_slice(fl, ph, dx)
            nz, ny, nx = fl.shape
            ext = (-0.5 * nx * dx, 0.5 * nx * dx, -0.5 * dx, (nz - 0.5) * dx)
            axs.imshow(rgb, origin="lower", extent=ext, interpolation="bilinear")
            xs = (np.arange(nx) + 0.5) * dx - 0.5 * nx * dx
            zs = (np.arange(nz) - 0.5) * dx
            if np.nanmax(phg) > np.nanmin(phg):
                axs.contour(xs, zs, phg, levels=np.linspace(0.1, 0.9, 9), colors=INK2, linewidths=0.5)
            hw = view_half_width(a.zoom)
            axs.set_xlim(-hw, hw); axs.set_ylim(max(a.zoom, -0.05), GAP + 1.25); axs.set_aspect("equal")
            if zcut > max(a.zoom, 0.0):
                axs.axhline(zcut, color=ORANGE, lw=0.9, ls=(0, (3, 2)))
            if t_cut is not None and t >= t_cut - 1e-9:
                axs.text(0.03, 0.10, f"From t = {t_cut:.1f} s the end rests on the\nedge of the model's conductor"
                         " (orange):\nthe model, not the melt", transform=axs.transAxes, fontsize=9,
                         color=INK, va="bottom", bbox=dict(facecolor=SURF, edgecolor=ORANGE, lw=0.8, pad=4))
            axs.set_title("Slice through the axis: melt and field", fontsize=13, color=INK, loc="left")
            axs.set_xlabel("x [mm]", fontsize=10, color=INK2)
            axs.set_ylabel("height above the collector [mm]", fontsize=10, color=INK2)
            axs.tick_params(labelsize=8.5, colors=INK2)
            cax = fig.add_axes([0.40, 0.085, 0.2, 0.012])
            cax.imshow(np.linspace(0, 8, 256)[None, :], aspect="auto", cmap=FIELD_CMAP,
                       extent=(0, 8, 0, 1))
            cax.set_yticks([]); cax.set_xticks([0, 2, 4, 6, 8])
            cax.tick_params(labelsize=8, colors=INK2, length=2)
            cax.set_xlabel("field strength in the air [kV/mm]  (air breaks down near 3)", fontsize=8.5,
                           color=INK2, labelpad=1)
            for sp in cax.spines.values():
                sp.set_visible(False)

            m = ser["t_s"] <= t + 1e-9
            ax1 = fig.add_axes([0.70, 0.64, 0.285, 0.22])
            ax1.plot(ser["t_s"], ser["z_low_mm"], color=BLUE, lw=1.0, alpha=0.2)
            ax1.plot(ser["t_s"][m], ser["z_low_mm"][m], color=BLUE, lw=2.4)
            ax1.set_ylim(0, 5.8); ax1.set_title("Lowest point of the melt [mm]", fontsize=11.5,
                                                 color=INK, loc="left")
            if zcut > 0:
                ax1.axhline(zcut, color=ORANGE, lw=0.9, ls=(0, (3, 2)))
                ax1.text(0.01 * t_end, zcut + 0.18, "below this the model's melt does not conduct",
                         fontsize=8, color=INK2)
            ax2 = fig.add_axes([0.70, 0.40, 0.285, 0.16])
            ax2.plot(ser["t_s"], ser["max_E_kV_per_mm"], color="#5440a6", lw=1.0, alpha=0.2)
            ax2.plot(ser["t_s"][m], ser["max_E_kV_per_mm"][m], color="#5440a6", lw=2.0)
            ax2.set_ylim(0, 1.1 * float(ser["max_E_kV_per_mm"][ser["t_s"] <= t_end + 1e-9].max()))
            ax2.axhline(3.0, color=INK3, lw=0.8, ls=(0, (2, 2)))
            ax2.text(0.01 * t_end, 3.15, "air breaks down in a uniform field", fontsize=8, color=INK2)
            ax2.set_title("Strongest field below the tip [kV/mm]", fontsize=11.5, color=INK, loc="left")
            for ax in (ax1, ax2):
                ax.set_xlim(0, t_end)
                ax.axvline(t, color=INK3, lw=0.8)
                if a.touch > 0:
                    ax.axvline(a.touch, color=ORANGE, lw=0.8, ls=(0, (1, 2)))
                ax.grid(axis="y", color=GRID, lw=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=8.5, colors=INK2)
            ax2.set_xlabel("time [s]", fontsize=10, color=INK2)
            draw_tip(fig.add_axes([0.70, 0.095, 0.285, 0.205]), rbar, apex, dx)
            fig.text(0.04, 0.028, "3-D model on one grid (50 µm, 8 cells across the bore), not grid-converged."
                     " The melt leaves the nozzle about 0.6 mm wide, the bore plus four cells: the width the"
                     " 2-D runs had on every grid.", fontsize=9, color=INK3)
            fig.text(0.04, 0.009, "The 10-20 µm jet of the experiment is far below the cells. No air breakdown"
                     " in the model. Surface shown where it is star-shaped about the nozzle's axis;"
                     " equipotentials every 10 % of the voltage.", fontsize=9, color=INK3)
            if a.snap:
                fig.savefig(a.out, facecolor=SURF)
                plt.close(fig)
                print("wrote", a.out)
                return
            fig.savefig(os.path.join(tmp, f"f{n:05d}.png"), facecolor=SURF)
            plt.close(fig)
            if n % 10 == 0:
                print(f"  frame {n + 1}/{len(times)}", flush=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
               "-i", os.path.join(tmp, "f%05d.png"),
               "-vf", "tpad=stop_mode=clone:stop_duration=2,format=yuv420p",
               "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", a.out]
        subprocess.run(cmd, check=True)
        shutil.copy(os.path.join(tmp, f"f{len(times) - 1:05d}.png"), os.path.splitext(a.out)[0] + "_last.png")
        print("wrote", a.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
