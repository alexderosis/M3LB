#!/usr/bin/env python3
"""3-D views of where the confined-MHD boxes dissipate: the paper's volume figure.

    plot_tg_mhd_3d.py A_DIR B_DIR [C_DIR] [--t 2.3] [-o PREFIX] [--fmt png,pdf]
                      [--view 35 22] [--range 1 300] [--size 520]

Each DIR is a run directory that tools/tg_mhd_vol.cpp has processed: vol/index.json
and the block-maximum volumes of nu|w|^2 and eta|j|^2 it lists (GPU/csf3/
tg_mhd_vol.sub makes them from tg_mhd_snap.sub's dumps; results/P_tg_mhd/snap/
<run>/vol/ once copied back). The figure, at the dump nearest --t, has one column
per box -- free slip, no slip, no field -- and two rows, the viscous and the
Ohmic dissipation, each over the box mean eps at that time, as in Figs. 3 and 4.

MAXIMUM-INTENSITY PROJECTION, as results/N_mhd_sphere/render_volume.py argues for
current sheets: every ray keeps the largest value it meets, so a sheet is never
hidden behind a dimmer one, and the price is that MIP shows no occlusion. The
DEPTH CUE is the same substitute that renderer uses: the depth at which each
ray's maximum lies dims the far half by up to 35 % along the colour ramp. It
says which of two
structures is nearer and must not be read as an intensity. Rays are orthographic,
sampled trilinearly at 2.5 samples per voxel, from a camera --view yaw and
elevation degrees off the x axis and the horizontal; the box's edges are drawn,
the hidden ones dashed, and its origin corner marks x, y and z.

One logarithmic colour scale serves every panel: --range LO HI, or by default the
99.7th percentile of all the panels' rays as HI and HI/300 as LO, printed so that
a second figure can be given the same range. The no-field box has no Ohmic panel.

Needs numpy and matplotlib (a scratch venv; see tools/plot_tg_mhd_ladder.py).
"""
import argparse
import json
import math
import os
import sys

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
COLNAME = {"A": "A  free slip", "B": "B  no slip", "C": "C  no slip, no field"}


def setup_of(tag):
    if tag.startswith("hydro_"):
        return "C"
    return "B" if "_noslip_" in tag else "A"


def camera(yaw, elev):
    """(right, up, into) unit vectors of an orthographic camera looking at the box."""
    import numpy as np
    y, e = math.radians(yaw), math.radians(elev)
    eye = np.array([math.cos(e) * math.cos(y), math.cos(e) * math.sin(y), math.sin(e)])
    w = -eye                                             # into the scene
    up = np.array([0.0, 0.0, 1.0])
    v = up - np.dot(up, w) * w
    v /= np.linalg.norm(v)
    u = np.cross(w, v)                                   # right-handed: u x v = -w points at the eye
    return u, v, w


def mip(np, vol, u, v, w, size):
    """Maximum along orthographic rays through the unit cube, and the depth (0 near,
    1 far) at which each maximum lies. vol is [z][y][x] over the unit cube."""
    M = vol.shape[0]
    R = 0.5 * math.sqrt(3.0) * 1.02                      # the cube's half diagonal, with a margin
    ns = int(2.5 * M)                                    # 1.6 a voxel aliased thin oblique sheets
    a = np.linspace(-R, R, size)
    s = np.linspace(-R, R, ns)
    best = np.full((size, size), -np.inf, dtype=np.float32)
    depth = np.zeros((size, size), dtype=np.float32)
    c = np.array([0.5, 0.5, 0.5])
    for r0 in range(0, size, 16):
        rows = a[::-1][r0:r0 + 16]                       # image row 0 is the top
        B, A, S = np.meshgrid(rows, a, s, indexing="ij")
        P = (c[None, None, None, :] + A[..., None] * u + B[..., None] * v + S[..., None] * w)
        q = P * (M - 1)                                  # voxel coordinates, x y z
        inside = np.all((q >= 0) & (q <= M - 1), axis=-1)
        q = np.clip(q, 0, M - 1 - 1e-6)
        i0 = np.floor(q).astype(np.int64)
        f = q - i0
        val = np.zeros(q.shape[:-1], dtype=np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    wgt = ((f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1])
                           * (f[..., 2] if dz else 1 - f[..., 2]))
                    zi = np.minimum(i0[..., 2] + dz, M - 1)
                    yi = np.minimum(i0[..., 1] + dy, M - 1)
                    xi = np.minimum(i0[..., 0] + dx, M - 1)
                    val += (wgt * vol[zi, yi, xi]).astype(np.float32)
        val = np.where(inside, val, -np.inf)
        k = np.argmax(val, axis=-1)
        mx = np.take_along_axis(val, k[..., None], axis=-1)[..., 0]
        best[r0:r0 + len(rows)] = mx
        depth[r0:r0 + len(rows)] = k / (ns - 1)
    return best, depth


def edges(np, u, v, w):
    """The cube's 12 edges in image coordinates, each with whether it is hidden.
    An edge is hidden only when BOTH faces it borders face away from the camera --
    the three edges at the far corner, in a generic view -- not when its midpoint
    merely lies behind the centre, which would dash six."""
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=float)
    c = np.array([0.5, 0.5, 0.5])
    out = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(corners[i] - corners[j])) != 1:
                continue
            mid = 0.5 * (corners[i] + corners[j]) - c
            # the two faces at this edge: outward normals along the axes where the edge is off-centre
            normals = [np.sign(mid[k]) * np.eye(3)[k] for k in range(3) if abs(mid[k]) > 1e-9]
            hidden = all(np.dot(n, w) > 0 for n in normals)        # a face points away when n . w > 0
            pts = [(np.dot(p - c, u), np.dot(p - c, v)) for p in (corners[i], corners[j])]
            out.append((pts, hidden))
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", help="run directories of A, B and (optionally) C")
    ap.add_argument("--t", type=float, default=2.3, help="the dump nearest this time")
    ap.add_argument("-o", "--out", default="tg_mhd_3d")
    ap.add_argument("--fmt", default="png")
    ap.add_argument("--view", type=float, nargs=2, default=[35.0, 22.0], metavar=("YAW", "ELEV"))
    ap.add_argument("--range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="the colour range; default from the data (printed)")
    ap.add_argument("--size", type=int, default=520, help="pixels across a panel")
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.patheffects as pe

    runs = {}
    for d in args.dirs:
        with open(os.path.join(d, "vol", "index.json")) as f:
            recs = json.load(f)
        runs[setup_of(recs[0]["tag"])] = (d, recs)
    order = [s for s in ("A", "B", "C") if s in runs]
    if "A" not in runs or "B" not in runs:
        raise SystemExit("need at least the free-slip (A) and no-slip (B) runs")
    u, v, w = camera(*args.view)
    R = 0.5 * math.sqrt(3.0) * 1.02
    panels, meta = {}, None
    for s in order:
        d, recs = runs[s]
        for fld in ("visc", "ohm"):
            if s == "C" and fld == "ohm":
                continue
            rec = min((x for x in recs if x["field"] == fld), key=lambda x: abs(x["t"] - args.t))
            meta = rec
            M = rec["M"]
            vol = np.fromfile(os.path.join(d, "vol", rec["file"]), dtype="<f4").reshape(M, M, M)
            panels[(s, fld)] = mip(np, vol / rec["eps_mean"], u, v, w, args.size)
    if args.range is None:
        allv = np.concatenate([b[np.isfinite(b)].ravel() for b, _ in panels.values()])
        hi = float(np.percentile(allv, 99.7))
        lo = hi / 300.0
    else:
        lo, hi = args.range
    print("colour range: %.3g .. %.3g times the box mean" % (lo, hi))
    norm = LogNorm(vmin=lo, vmax=hi)
    cmap = plt.get_cmap("magma")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "text.color": INK})
    fig = plt.figure(figsize=(10.5, 7.4))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, len(order), left=0.06, right=0.9, bottom=0.03, top=0.9, wspace=0.03, hspace=0.06)
    for c, s in enumerate(order):
        for r, fld in enumerate(("visc", "ohm")):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(COLNAME[s], fontsize=10.5)
            if c == 0:
                ax.set_ylabel("viscous" if fld == "visc" else "Ohmic", fontsize=10.5)
            if (s, fld) not in panels:
                ax.set_facecolor(SURFACE)
                ax.text(0.5, 0.5, "no field", ha="center", va="center", color=MUTED,
                        transform=ax.transAxes, fontsize=10.5)
                continue
            best, depth = panels[(s, fld)]
            # The depth cue darkens along the colour ramp, not in RGB: multiplying RGB
            # turned the saturated pale yellow of the far half olive, a colour the
            # ramp does not contain.
            level = np.clip(norm(np.clip(np.where(np.isfinite(best), best, lo), lo, None)), 0, 1)
            shade = 1.0 - 0.35 * np.clip((depth - 0.5) / 0.5, 0, 1)          # dim the far half only
            rgba = cmap(level * shade)
            rgba[~np.isfinite(best)] = (0, 0, 0, 1)
            ax.imshow(rgba, extent=[-R, R, -R, R], interpolation="bilinear")
            for (p0, p1), hidden in edges(np, u, v, w):
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#bfe3ff" if not hidden else "#7a8a99",
                        lw=0.8 if not hidden else 0.6, ls="-" if not hidden else (0, (3, 2)), alpha=0.9)
            o = np.array([0.0, 0.0, 0.0]) - 0.5
            for k, lab in enumerate("xyz"):
                e = np.zeros(3)
                e[k] = 1.0
                p = o + 1.07 * e
                ax.text(np.dot(p, u), np.dot(p, v), lab, color="#bfe3ff", fontsize=9, ha="center", va="center",
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)
            ax.set_aspect("equal")
    cax = fig.add_axes([0.915, 0.1, 0.012, 0.74])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("largest local dissipation along the ray / box mean", fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    fig.suptitle("Re = %g,  t = %.2f,  N = %d (block maximum over %d$^3$)" % (
        meta["Re"], meta["t"], meta["N"], meta["stride"]), x=0.06, ha="left", fontsize=11, y=0.985)
    for f in args.fmt.split(","):
        fig.savefig("%s.%s" % (args.out, f), dpi=200, facecolor=SURFACE,
                    metadata={"CreationDate": None} if f == "pdf" else None)
    print("wrote " + ", ".join("%s.%s" % (args.out, f) for f in args.fmt.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
