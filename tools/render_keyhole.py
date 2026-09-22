#!/usr/bin/env python3
"""
Render the four-panel melt-pool animation from validation/keyhole's -frames dump.

WHAT THIS IS A PICTURE OF, because the distinction decides how the figure may be
used. validation/keyhole.cpp is a port of the Laser project's
keyhole_validation/solver.py, whose own first line calls it a "3D THERMAL-ONLY
Lattice Boltzmann solver".  So:

  - There is NO FLOW PANEL, and its absence is not an omission in this script.
    The reference movie slm_melt_pool_3d.mp4 was made by a DIFFERENT solver in
    that folder, lbm_slm.py, which couples a D3Q19 fluid to the thermal lattice
    with Boussinesq buoyancy, a Marangoni-like surface shear and a recoil body
    force, and whose thermal equilibrium takes the velocity -- so its
    temperature is genuinely advected.  M3LB has not ported that model.  A quiver
    panel here would have nothing to draw.

  - THE MODEL IS NOT GRID-CONVERGED. keyhole.cpp's own banner measures depth
    falling 12-16 % over a 2x refinement in every recession-cap mode, monotone
    and not settling.  Every frame here is a visualisation of a 13-parameter
    calibrated model at one grid, never a prediction, and the figure says so on
    its face rather than in a caption someone will crop off.

Needs numpy + matplotlib, which the SYSTEM python on this machine does not have:

    python3 -m venv ~/.venvs/m3lb-viz
    ~/.venvs/m3lb-viz/bin/pip install numpy matplotlib
    ~/.venvs/m3lb-viz/bin/python tools/render_keyhole.py <frames-dir> [out.mp4]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm, colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BG = "#0b0d12"

D = sys.argv[1] if len(sys.argv) > 1 else "frames"
OUT = sys.argv[2] if len(sys.argv) > 2 else "keyhole_melt_pool_3d.mp4"

ld = lambda n: np.load(os.path.join(D, n))
T_top, fl_top, surf = ld("T_top.npy"), ld("fl_top.npy"), ld("surf.npy")
T_xz, fl_xz = ld("T_xz.npy"), ld("fl_xz.npy")
t, depth, meta = ld("t.npy"), ld("depth.npy"), ld("meta.npy")
_m = [float(v) for v in meta]
dx, dt, nx, ny, nz, yc, T_s, T_l, T_0, P, spot, thick, steps = _m[:13]
# T_boil and L_vap were added to meta when the evaporation field was; older
# dumps do not have them, so the evaporation panel is optional.
T_b = _m[13] if len(_m) > 13 else None
L_v = _m[14] if len(_m) > 14 else None
evap = np.load(os.path.join(D, "evap.npy")) if os.path.exists(
    os.path.join(D, "evap.npy")) else None
nx, ny, nz = int(nx), int(ny), int(nz)
F = len(t)
um = dx * 1e6                      # micrometres per cell

# ---- crop the centreline panel in z.  The domain is nz cells deep but the pool
# ---- lives in the top few tens: at dx = 5 um a 300-cell domain is 1500 um
# ---- against a 134 um hole, so an uncropped panel is 91 % cold solid and the
# ---- keyhole is four pixels tall.  Crop to the deepest melt over the WHOLE run
# ---- plus a margin, so the axes never move between frames.
melt = np.nan_to_num(fl_xz, nan=0.0) > 0.0
zmin_i = int(np.argmax(melt.any(axis=(0, 1)))) if melt.any() else nz - 40
z_lo = max(0, zmin_i - 8)
T_xz, fl_xz = T_xz[:, :, z_lo:], fl_xz[:, :, z_lo:]
nzc = T_xz.shape[2]

Tmax = float(np.nanmax(T_top))
# ---- HOW MUCH MATERIAL HAS GONE. surf is the per-column surface index, so
# ---- (nz-1 - surf) cells have been removed from that column and the sum times
# ---- dx^3 is the cavity volume. Plotted as a mass so it is comparable with
# ---- anything weighed: rho_liquid 4130 kg/m^3 for Ti-6Al-4V.
rec_cells = (nz - 1) - surf                      # (frames, nx, ny)
vol_um3 = rec_cells.reshape(len(t), -1).sum(axis=1) * (um ** 3)
mass_ng = vol_um3 * 4130.0 * 1e-18 * 1e12        # um^3 -> m^3 -> kg -> ng

norm = mcolors.Normalize(vmin=T_0, vmax=Tmax)
cmap = cm.inferno

X, Y = np.meshgrid((np.arange(nx) + 0.5) * um, (np.arange(ny) + 0.5) * um, indexing="ij")

fig = plt.figure(figsize=(9.5, 11.5), facecolor=BG)
gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.15, 1.15, 0.85], hspace=0.42)
ax3d = fig.add_subplot(gs[0], projection="3d")
ax_top = fig.add_subplot(gs[1])
ax_xz = fig.add_subplot(gs[2])
ax_d = fig.add_subplot(gs[3])

for ax in (ax_top, ax_xz, ax_d):
    ax.set_facecolor(BG)
    ax.tick_params(colors="w", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#39405a")
ax3d.set_facecolor(BG)

im_top = ax_top.imshow(T_top[0].T, origin="lower", extent=[0, nx * um, 0, ny * um],
                       cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")
ax_top.set_title("Top surface (follows the receding free surface): T, melt boundary"
                 + (f", boiling ({T_b:.0f} K)" if T_b else ""),
                 color="w", fontsize=9)
ax_top.set_xlabel("x  [$\\mu$m]", color="w", fontsize=8)
ax_top.set_ylabel("y  [$\\mu$m]", color="w", fontsize=8)
cax = fig.add_axes([0.905, 0.175, 0.017, 0.315])
cb = fig.colorbar(im_top, cax=cax)
cb.set_label("T  [K]", color="w", fontsize=8)
cb.ax.tick_params(colors="w", labelsize=7)
cb.outline.set_color("#39405a")

im_xz = ax_xz.imshow(T_xz[0].T, origin="lower",
                     extent=[0, nx * um, (z_lo - (nz - 1)) * um, um],
                     cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")
ax_xz.set_title("Centreline section (y = y$_c$): melt depth.  "
                "NO FLOW -- this model is thermal only", color="w", fontsize=9)
ax_xz.set_xlabel("x  [$\\mu$m]", color="w", fontsize=8)
ax_xz.set_ylabel("z below original surface  [$\\mu$m]", color="w", fontsize=8)

ax_d.plot(t * 1e3, depth * 1e6, color="#ffb703", lw=1.6, label="depth")
mark, = ax_d.plot([t[0] * 1e3], [depth[0] * 1e6], "o", color="w", ms=5)
ax_d.set_title("Vapour-depression depth vs time", color="w", fontsize=9)
ax_d.set_xlabel("t  [ms]", color="w", fontsize=8)
ax_d.set_ylabel("depth  [$\\mu$m]", color="w", fontsize=8)
ax_d.grid(alpha=0.18, color="w")
_sc = float(depth.max() * 1e6) / max(float(mass_ng.max()), 1e-30)
ax_d.plot(t * 1e3, mass_ng * _sc, color="#c8ff8f", lw=1.4, ls="--")
ax_d.text(t[int(0.62 * len(t))] * 1e3, mass_ng[int(0.62 * len(t))] * _sc,
          f"  material removed, {mass_ng[-1]:.1f} ng by {t[-1]*1e3:.2f} ms",
          color="#c8ff8f", fontsize=6.5, va="bottom")

# ---- THE EVAPORATION, ON ITS OWN AXIS. `evap` is the per-column recession
# ---- rate, i.e. the melt that recoil pressure has just ejected -- the only
# ---- field here that IS the evaporation rather than a consequence of it. It is
# ---- plotted against the CAP, because the census says the cap binds on 48.5 %
# ---- of receding column-steps and a rate sitting on its limiter is a property
# ---- of the integrator rather than of the alloy.
if evap is not None:
    ax_e = ax_d.twinx()
    ax_e.set_facecolor("none")
    peak_e = evap.reshape(len(t), -1).max(axis=1)
    cap = float(peak_e.max())
    ax_e.plot(t * 1e3, peak_e, color="#ff6b9d", lw=1.3, label="peak evaporation rate")
    ax_e.axhline(cap, color="#ff6b9d", ls=":", lw=1.0, alpha=0.8)
    ax_e.text(t[-1] * 1e3, cap, f"  cap {cap:.2f} m/s", color="#ff6b9d",
              fontsize=6.5, va="center")
    ax_e.set_ylabel("evaporation rate  [m/s]", color="#ff6b9d", fontsize=8)
    ax_e.tick_params(colors="#ff6b9d", labelsize=7)
    ax_e.set_ylim(0, cap * 1.45)
    for sp in ax_e.spines.values():
        sp.set_color("#39405a")
    frac = 100.0 * (evap > 0).mean()
    ax_d.set_title(f"Depth, the evaporation driving it, and the material removed "
                   f"(evaporating in {frac:.2f} % of surface cells)",
                   color="w", fontsize=9)

# mplot3d pads its gridspec cell heavily -- the reference script positions this
# axis by hand for the same reason -- so claim the space back explicitly.
fig.subplots_adjust(top=0.905, bottom=0.045, left=0.095, right=0.875)
ax3d.set_position([0.02, 0.535, 0.95, 0.368])

fig.suptitle(f"M3LB keyhole: Clausius-Clapeyron evaporation + recoil recession"
             f"   {P:.0f} W, {spot:.0f} $\mu$m spot, dx = {um:.1f} $\mu$m",
             color="w", fontsize=10.5, y=0.991)
fig.text(0.105, 0.9135, "Free surface, coloured by temperature", color="w", fontsize=9)
fig.text(0.5, 0.9635,
         "CALIBRATED REDUCED-ORDER MODEL, NOT GRID CONVERGED "
         "(depth falls 12-16 % over a 2x refinement) -- not a prediction",
         color="#ff8fa3", fontsize=8, ha="center")
if evap is not None:
    fig.text(0.5, 0.9405,
             "The evaporation rate CHATTERS between zero and its cap -- the surface energy "
             "balance is forward Euler on a stiff feedback,\nso the clamp is part of the "
             "integrator rather than a safety valve, and it binds on 48.5 % of receding "
             "column-steps.",
             color="#ff6b9d", fontsize=7, ha="center")

holder = {}


def draw3d(i):
    if "s" in holder:
        holder["s"].remove()
    Z = (surf[i] - (nz - 1)) * um                  # depth below the original surface
    fc = cmap(norm(T_top[i]))
    holder["s"] = ax3d.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                                    linewidth=0, antialiased=False, shade=False)
    ax3d.set_zlim(min(-1.0, float((surf.min() - (nz - 1)) * um) * 1.05), um)
    ax3d.set_xlabel("x [$\\mu$m]", color="w", fontsize=7)
    ax3d.set_ylabel("y [$\\mu$m]", color="w", fontsize=7)
    ax3d.set_zlabel("z [$\\mu$m]", color="w", fontsize=7)
    # The x and y tick labels are dropped: they repeat the two panels below,
    # which carry the same x and y in micrometres, and at this box aspect they
    # collide with the next panel's title. The z scale stays -- the depth is the
    # one number this panel is for.
    ax3d.tick_params(colors="w", labelsize=6, pad=-1)
    ax3d.set_xticklabels([]); ax3d.set_yticklabels([])
    ax3d.tick_params(axis="z", labelsize=6.5)
    ax3d.view_init(elev=26, azim=-58)
    try:
        ax3d.set_box_aspect((nx * um, ny * um, 0.55 * nx * um), zoom=1.28)
    except TypeError:
        pass
    for pane in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        pane.pane.set_facecolor(BG)
        pane.pane.set_edgecolor("#39405a")


cont = {"top": None, "xz": None, "cb": None}


def update(i):
    im_top.set_data(T_top[i].T)
    im_xz.set_data(T_xz[i].T)
    for k, ax, fld, ext in (("top", ax_top, fl_top, None), ("xz", ax_xz, fl_xz, None)):
        if cont[k] is not None:
            cont[k].remove()
        f = np.nan_to_num(fld[i], nan=0.0)
        if k == "top":
            gx = (np.arange(nx) + 0.5) * um
            gy = (np.arange(ny) + 0.5) * um
        else:
            gx = (np.arange(nx) + 0.5) * um
            gy = (np.arange(nzc) + z_lo - (nz - 1) + 0.5) * um
        cont[k] = ax.contour(gx, gy, f.T, levels=[0.5], colors="#00e5ff", linewidths=1.1)
    # the EVAPORATING region: where the surface is above the boiling point.
    if T_b:
        if "cb" in cont and cont["cb"] is not None:
            cont["cb"].remove()
        gx = (np.arange(nx) + 0.5) * um
        gy = (np.arange(ny) + 0.5) * um
        cont["cb"] = (ax_top.contour(gx, gy, T_top[i].T, levels=[T_b],
                                     colors="#ff6b9d", linewidths=1.4)
                      if float(T_top[i].max()) > T_b else None)
    mark.set_data([t[i] * 1e3], [depth[i] * 1e6])
    draw3d(i)
    return ()


if __name__ == "__main__":
    print(f"{F} frames, {nx}x{ny} top, {nx}x{nzc} centreline (cropped from {nz}), "
          f"peak T = {Tmax:.0f} K, final depth = {depth[-1]*1e6:.1f} um")
    ani = animation.FuncAnimation(fig, update, frames=F, interval=50, blit=False)
    matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
    ani.save(OUT, writer=animation.FFMpegWriter(fps=20, bitrate=3600), dpi=110)
    print("wrote", OUT)
