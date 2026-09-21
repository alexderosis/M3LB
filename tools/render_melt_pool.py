#!/usr/bin/env python3
"""
Render the melt-pool animation from validation/melt_pool's -frames dump.

WHY THIS IS A DIFFERENT SCRIPT FROM tools/render_keyhole.py, and not a flag on
it.  The two cases are different KINDS of claim and the panels have to say so:

  keyhole    thermal + recession, 13 constants fitted to experiment, NOT grid
             converged.  Its fourth panel is a depth trace with nothing to check
             it against, and its surface recedes, so panel 1 draws the surface.

  melt_pool  conduction only, ZERO fitted constants, every material property
             printed with a named source, validated against the Eagar & Tsai
             (1983) moving-distributed-source solution to 0.7 % at dx = 2 um.
             The surface is FLAT and u == 0 identically, so there is nothing to
             draw there; panel 1 draws the POOL's lower surface instead, and the
             fourth panel carries the three pool dimensions AGAINST the analytic
             values -- the thing that makes this one a prediction.

A model with more physics terms is not automatically the better picture: the
reference movie this was chased from, the Laser project's slm_melt_pool_3d.mp4,
comes from lbm_slm.py, whose own line 51 calls it "a qualitative, dimensionless
(lattice-unit) representative model" -- no kelvin, no micrometres, and about a
dozen invented coupling coefficients.  This case has fewer terms and every
number in it means something.

    ~/.venvs/m3lb-viz/bin/python tools/render_melt_pool.py <frames-dir> [out.mp4]
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
OUT = sys.argv[2] if len(sys.argv) > 2 else "melt_pool_3d.mp4"

ld = lambda n: np.load(os.path.join(D, n))
T_top, T_xz, pool = ld("T_top.npy"), ld("T_xz.npy"), ld("pool.npy")
t, w2, dd, LL, xb, meta = (ld("t.npy"), ld("w2.npy"), ld("d.npy"),
                           ld("L.npy"), ld("xb.npy"), ld("meta.npy"))
(dx, dt, nx, ny, nz, jc, T_s, T_l, T_0, P, v, spot, A,
 w2_an, d_an, L_an, t_therm, la0, steps) = [float(q) for q in meta]
nx, ny, nz = int(nx), int(ny), int(nz)
F = len(t)
um = dx * 1e6

# crop the centreline panel: the pool is ~13 cells deep in a 120-cell domain
melt = T_xz >= T_l
z_lo = max(0, int(np.argmax(melt.any(axis=(0, 1)))) - 6) if melt.any() else nz - 30
T_xzc = T_xz[:, :, z_lo:]
nzc = T_xzc.shape[2]

norm = mcolors.Normalize(vmin=T_0, vmax=float(T_top.max()))
cmap = cm.inferno
X, Y = np.meshgrid((np.arange(nx) + 0.5) * um, (np.arange(ny) + 0.5) * um, indexing="ij")

fig = plt.figure(figsize=(9.5, 11.5), facecolor=BG)
gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.15, 1.15, 0.95], hspace=0.45)
ax3d = fig.add_subplot(gs[0], projection="3d")
ax_top, ax_xz, ax_tr = (fig.add_subplot(gs[i]) for i in (1, 2, 3))
for ax in (ax_top, ax_xz, ax_tr):
    ax.set_facecolor(BG)
    ax.tick_params(colors="w", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#39405a")
ax3d.set_facecolor(BG)

fig.subplots_adjust(top=0.935, bottom=0.05, left=0.095, right=0.875)
ax3d.set_position([0.02, 0.555, 0.95, 0.375])

im_top = ax_top.imshow(T_top[0].T, origin="lower", extent=[0, nx * um, 0, ny * um],
                       cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")
ax_top.set_title("Top surface: temperature, melt isotherm", color="w", fontsize=9)
ax_top.set_xlabel("x  [$\\mu$m]", color="w", fontsize=8)
ax_top.set_ylabel("y  [$\\mu$m]", color="w", fontsize=8)
cax = fig.add_axes([0.905, 0.175, 0.017, 0.315])
cb = fig.colorbar(im_top, cax=cax)
cb.set_label("T  [K]", color="w", fontsize=8)
cb.ax.tick_params(colors="w", labelsize=7)
cb.outline.set_color("#39405a")

im_xz = ax_xz.imshow(T_xzc[0].T, origin="lower",
                     extent=[0, nx * um, (z_lo - (nz - 1)) * um, um],
                     cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")
ax_xz.set_title("Centreline section (y = y$_c$): melt depth.  Surface is FLAT and "
                "u $\\equiv$ 0 -- conduction only", color="w", fontsize=9)
ax_xz.set_xlabel("x  [$\\mu$m]", color="w", fontsize=8)
ax_xz.set_ylabel("z  [$\\mu$m]", color="w", fontsize=8)

for val, col, lab in ((w2_an, "#00e5ff", "2w"), (d_an, "#ffb703", "d"),
                      (L_an, "#b388ff", "L")):
    ax_tr.axhline(val * 1e6, color=col, ls="--", lw=1.0, alpha=0.8)
ln_w, = ax_tr.plot([], [], color="#00e5ff", lw=1.6, label="2w")
ln_d, = ax_tr.plot([], [], color="#ffb703", lw=1.6, label="d")
ln_L, = ax_tr.plot([], [], color="#b388ff", lw=1.6, label="L")
ax_tr.set_xlim(0, float(t[-1]) * 1e3)
ax_tr.set_ylim(0, max(w2_an, d_an, L_an) * 1e6 * 1.25)
ax_tr.set_title("Pool dimensions vs time.  Dashed = Eagar & Tsai (1983) analytic, "
                "on the planes the simulation samples", color="w", fontsize=9)
ax_tr.set_xlabel("t  [ms]", color="w", fontsize=8)
ax_tr.set_ylabel("[$\\mu$m]", color="w", fontsize=8)
ax_tr.grid(alpha=0.18, color="w")
lg = ax_tr.legend(loc="lower right", fontsize=7, facecolor=BG, edgecolor="#39405a",
                  labelcolor="w", ncol=3)

fig.suptitle(f"M3LB  conduction melt pool (D3Q7, EnthalpyRegularised)   "
             f"Ti-6Al-4V, P = {P:.0f} W, v = {v*1e3:.0f} mm/s, dx = {um:.1f} $\\mu$m",
             color="w", fontsize=10.5, y=0.985)
fig.text(0.5, 0.955,
         f"ZERO FITTED CONSTANTS -- 2w and d within {max(abs(1-w2[-1]/w2_an),abs(1-dd[-1]/d_an))*100:.1f} % "
         f"of the analytic solution.  This one is a PREDICTION.",
         color="#8fffa3", fontsize=8.5, ha="center")
# The case's OWN caveat, carried on the figure rather than left in its banner:
# the peak surface temperature here is far above the alloy's boiling point, so
# the LBM/analytic comparison is still exact (both sides solve the same linear
# conduction problem and neither contains vaporisation) while a comparison
# against a real single track at this power and speed would not be.
_pk = float(T_top.max())
fig.text(0.5, 0.9385,
         f"Peak surface T = {_pk:.0f} K, above Ti-6Al-4V's ~3315 K boiling point: valid as a "
         f"solver check, NOT comparable to a real track at this power",
         color="#ffd39a", fontsize=7.5, ha="center")
fig.text(0.115, 0.918, "Melt pool, lower surface, coloured by surface temperature",
         color="w", fontsize=9)

holder = {}
cont = {"top": None, "xz": None}


def update(i):
    if "s" in holder:
        holder["s"].remove()
    Z = -pool[i] * 1e6                       # pool floor, micrometres below the surface
    holder["s"] = ax3d.plot_surface(X, Y, Z, facecolors=cmap(norm(T_top[i])),
                                    rstride=1, cstride=1, linewidth=0,
                                    antialiased=False, shade=False)
    ax3d.set_zlim(-float(pool.max()) * 1e6 * 1.15 - 1e-9, 1.0)
    ax3d.view_init(elev=26, azim=-58)
    ax3d.set_xlabel("x [$\\mu$m]", color="w", fontsize=7)
    ax3d.set_ylabel("y [$\\mu$m]", color="w", fontsize=7)
    ax3d.set_zlabel("depth [$\\mu$m]", color="w", fontsize=7)
    ax3d.tick_params(colors="w", labelsize=6, pad=-1)
    ax3d.set_xticklabels([]); ax3d.set_yticklabels([])
    ax3d.tick_params(axis="z", labelsize=6.5)
    try:
        ax3d.set_box_aspect((nx * um, ny * um, 0.42 * nx * um), zoom=1.28)
    except TypeError:
        pass
    for pane in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        pane.pane.set_facecolor(BG)
        pane.pane.set_edgecolor("#39405a")

    im_top.set_data(T_top[i].T)
    im_xz.set_data(T_xzc[i].T)
    for k, ax, fld, gy in (("top", ax_top, T_top[i], (np.arange(ny) + 0.5) * um),
                           ("xz", ax_xz, T_xzc[i],
                            (np.arange(nzc) + z_lo - (nz - 1) + 0.5) * um)):
        if cont[k] is not None:
            cont[k].remove()
        cont[k] = ax.contour((np.arange(nx) + 0.5) * um, gy, fld.T,
                             levels=[T_l], colors="#00e5ff", linewidths=1.1)
    ln_w.set_data(t[:i + 1] * 1e3, w2[:i + 1] * 1e6)
    ln_d.set_data(t[:i + 1] * 1e3, dd[:i + 1] * 1e6)
    ln_L.set_data(t[:i + 1] * 1e3, LL[:i + 1] * 1e6)
    return ()


if __name__ == "__main__":
    print(f"{F} frames, {nx}x{ny} top, {nx}x{nzc} centreline (cropped from {nz}), "
          f"final 2w/d/L = {w2[-1]*1e6:.2f}/{dd[-1]*1e6:.2f}/{LL[-1]*1e6:.2f} um "
          f"against analytic {w2_an*1e6:.2f}/{d_an*1e6:.2f}/{L_an*1e6:.2f}")
    ani = animation.FuncAnimation(fig, update, frames=F, interval=50, blit=False)
    matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
    ani.save(OUT, writer=animation.FFMpegWriter(fps=20, bitrate=3600), dpi=110)
    print("wrote", OUT)
