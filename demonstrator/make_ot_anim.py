#!/usr/bin/env python3
"""Animate the Orszag-Tang current field beside its own maximum.

Reads what `validation/orszag_tang -tmax T -movie DIR` writes:

    DIR/ot_j_%04d.bin   int32 nx, int32 ny, then nx*ny float32, x fastest,
                        already divided by dt so the values are physical (1/s)
    DIR/ot_series.dat   t  j_max  zeta_max  Ma_max, one row per probe

and hands the frames to ffmpeg.

NOTE ON DEPENDENCIES. Needs numpy and matplotlib; the SYSTEM python has
neither. python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy matplotlib

Four things here are deliberate.

  SHARED SCALE, taken from the frame at the global peak of j_max. The whole
  point of this animation is that the current rises and then decays, and a
  self-normalising colour scale destroys exactly that: it makes a decaying field
  look steady. The same argument, and the same conclusion, as
  make_aorta_anim.py. The cost is that late frames go pale, which is the honest
  rendering of a field that really has collapsed -- the series panel carries the
  number.

  PERCENTILE, NOT PEAK, for that scale. j at a high cell Reynolds number is
  carried by thin sheets and, once under-resolved, by individual cells; scaling
  to p100 lets one cell set the ramp and washes out the sheets that are the
  structure. p99.5 of |j| over the peak frame is used instead, and printed, so
  the figure says what it clipped.

  ASINH, NOT LINEAR, BY DEFAULT -- and this is the one choice here that can
  mislead, so it is labelled on the colour bar itself. max|j| grows by more than
  an order of magnitude over a run, and under ONE fixed linear scale the early
  frames are blank white: the shared-scale rule above then hides the very thing
  it was adopted to show. An asinh map keeps a single fixed mapping for the whole
  sequence -- nothing is renormalised frame to frame -- while staying linear
  below the initial field's own amplitude and compressing above it. --norm linear
  restores the plain scale; the series panel is linear either way, so the
  quantitative growth is never read off the colours.

  DIVERGING MAP, SYMMETRIC. j = curl b is signed and its zero is physical --
  RdBu_r about 0, matching doc/fig/mhd_orszag_tang_3d27.py, so a still from this
  animation and the validation figure can be read side by side.

  --tmax CUTS BOTH THE SERIES AND THE FRAMES, and a run that blew up needs it.
  The shared scale is taken from the frame at the peak of j_max, so ONE
  non-finite probe at the end silently becomes the reference frame and every
  other frame renders as flat zero. Cutting at the last physical probe is not
  hiding the blow-up -- it is refusing to let the blow-up set the scale for the
  part that is real. Say in the caption where the cut is.

  THE FRAME CLOCK IS NOT ASSUMED. Frames and probes have independent strides in
  the solver, so the frame interval is passed in (--dt-frame, as the run's own
  header prints it) rather than inferred from the series by counting rows, which
  is right only when the two strides happen to coincide.

usage: make_ot_anim.py <dir> <out.mp4> [--fps 25] [--dt-frame S] [--pct 99.5]
                      [--tmax S] [--norm asinh|linear]
"""
import glob
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from matplotlib.gridspec import GridSpec

PCT = 99.5


def raw(path):
    with open(path, "rb") as fh:
        nx, ny = np.fromfile(fh, dtype=np.int32, count=2)
        return np.fromfile(fh, dtype=np.float32, count=nx * ny).reshape(ny, nx)


def arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    d = sys.argv[1]
    out = sys.argv[2]
    fps = arg("--fps", 25, int)
    pct = arg("--pct", PCT)
    frames = sorted(glob.glob(os.path.join(d, "ot_j_*.bin")))
    if not frames:
        sys.exit("no ot_j_*.bin in " + d)

    series = np.loadtxt(os.path.join(d, "ot_series.dat"))
    ts, js = series[:, 0], series[:, 1]

    # The frame clock. --dt-frame is what the run printed; the fallback assumes
    # the two strides coincided, which is true only when they did.
    dtf = arg("--dt-frame", ts[0] * len(ts) / (len(frames) - 1))
    tf = np.arange(len(frames)) * dtf

    tmax = arg("--tmax", ts[-1])
    keep = ts <= tmax + 1e-12
    if not keep.all():
        print("  cut at t = %.3f s: dropping %d probe(s), last kept j_max = %.4g"
              % (tmax, int((~keep).sum()), js[keep][-1]))
    ts, js = ts[keep], js[keep]

    frames = [f for i, f in enumerate(frames) if i * dtf <= tmax + 1e-12]
    tf = tf[:len(frames)]

    ipk = int(np.argmax(js))
    kpk = int(np.clip(round(ts[ipk] / dtf), 0, len(frames) - 1))
    lim = float(np.percentile(np.abs(raw(frames[kpk])), pct))
    print("  %d frames, dt_frame = %.4f s, t_end = %.2f s" % (len(frames), dtf, tf[-1]))
    print("  j_max peaks at %.4g (1/s) at t = %.2f s  (frame %d)" % (js[ipk], ts[ipk], kpk))
    print("  colour scale +-%.4g = p%.1f of |j| on that frame (p100 = %.4g)"
          % (lim, pct, float(np.abs(raw(frames[kpk])).max())))

    # The asinh knee is the initial field's own p99.5: linear where the flow
    # starts, compressive over everything the sheets later reach.
    kind = sys.argv[sys.argv.index("--norm") + 1] if "--norm" in sys.argv else "asinh"
    if kind == "asinh":
        knee = max(float(np.percentile(np.abs(raw(frames[0])), pct)), lim * 1e-3)
        norm = AsinhNorm(linear_width=knee, vmin=-lim, vmax=lim)
        clabel = r"$j$  (1/s), asinh scale, linear below %.2g" % knee
        print("  asinh colour scale, linear below %.4g" % knee)
    else:
        norm = None
        clabel = r"$j$  (1/s)"

    tmp = os.path.join(d, "_anim")
    os.makedirs(tmp, exist_ok=True)
    plt.rcParams.update({"font.size": 11})

    for i, fn in enumerate(frames):
        A = raw(fn)
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = GridSpec(1, 2, width_ratios=[1.0, 0.92], left=0.04, right=0.965,
                      top=0.88, bottom=0.11, wspace=0.34)

        ax = fig.add_subplot(gs[0, 0])
        kw = dict(norm=norm) if norm is not None else dict(vmin=-lim, vmax=lim)
        im = ax.imshow(A, origin="lower", cmap="RdBu_r",
                       extent=(0, 2 * np.pi, 0, 2 * np.pi), interpolation="nearest",
                       **kw)
        ax.set_title(r"$j = \nabla\times b$   (1/s)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(clabel, fontsize=9)

        bx = fig.add_subplot(gs[0, 1])
        bx.plot(ts, js, color="#333333", lw=1.4)
        k = np.searchsorted(ts, tf[i])
        if k > 0:
            bx.plot(ts[:k], js[:k], color="#c0392b", lw=2.0)
            bx.plot([ts[k - 1]], [js[k - 1]], "o", color="#c0392b", ms=7)
        bx.axvline(tf[i], color="#c0392b", lw=0.8, alpha=0.45)
        bx.set_xlim(0, ts[-1])
        bx.set_ylim(0, js.max() * 1.06)
        bx.set_xlabel("t (s)")
        bx.set_ylabel(r"$\max|j|$  (1/s)")
        bx.set_title(r"$\max|j|$ against time")
        bx.grid(alpha=0.25, lw=0.6)

        fig.suptitle(SUPTITLE + r"        $t$ = %6.2f s" % tf[i], y=0.975)
        fig.savefig(os.path.join(tmp, "f_%04d.png" % i))
        plt.close(fig)
        if (i + 1) % 25 == 0:
            print("    %d/%d" % (i + 1, len(frames)))

    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", os.path.join(tmp, "f_%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  wrote " + out)


SUPTITLE = os.environ.get(
    "OT_TITLE",
    "Orszag-Tang vortex, D3Q27 + D3Q7 (nz = 1), central moments")

if __name__ == "__main__":
    main()
