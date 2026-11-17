#!/usr/bin/env python3
"""Every number the -stir / -stirfeed paragraphs of demonstrator/mhd_decay.cpp's
banner quote, recomputed from the four logs beside this file.

    python3 results/N_neffaa/stir329/stir_numbers.py            # the logs only
    python3 results/N_neffaa/stir329/stir_numbers.py FRAMES     # and the rotation

FRAMES is the -bin directory of a rerun of stir.txt's command (README.md has
it): the rotation needs the vorticity frames, which are not tracked. It is the
azimuthal mean of u_theta on a circle, from Stokes: the sum of the vorticity
inside the circle over its circumference. Needs numpy for that part only.

The series columns are the driver's own: 1 step, 2 t/T_e, 3 E_kin, 4 E_mag,
8 <j^2>, 15 the bulk rms(div b)/rms(curl b). A probe every 0.32 turnovers, so
"peak" and "at every probe" mean of those eleven rows and nothing finer.
"""
import glob
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def series(name):
    rows = []
    for line in open(os.path.join(HERE, name)):
        f = line.split()
        if len(f) >= 15 and f[0].isdigit():
            rows.append([float(v) for v in f[:15]])
    return rows


def dmass(name):
    return float(re.search(r"dmass (\S+)", open(os.path.join(HERE, name)).read()).group(1))


ctrl, stir = series("ctrl.txt"), series("stir.txt")
f2, f4 = series("feed_2e-4.txt"), series("feed_4e-4.txt")
EK, EM, JJ, BULK = 2, 3, 7, 14
assert len(ctrl) == len(stir) == len(f2) == len(f4) == 11
assert all(a[0] == b[0] == c[0] == d[0] for a, b, c, d in zip(ctrl, stir, f2, f4))
# One thing changed between the four: the same initial condition.
assert ctrl[0][2:] == stir[0][2:] == f2[0][2:] == f4[0][2:]

print("turnovers at the last probe                 %.2f" % stir[-1][1])
print("E_kin, stirred / unstirred, at the end      %.0f   (first probe after t = 0: %.0f)"
      % (stir[-1][EK] / ctrl[-1][EK], stir[1][EK] / ctrl[1][EK]))
pk = max(stir, key=lambda r: r[JJ])
print("r.m.s. current, stirred: peak at %.2f turnovers, peak / end = %.2f"
      % (pk[1], math.sqrt(pk[JJ] / stir[-1][JJ])))
print("E_mag, stirred / unstirred, at the end      %.3f" % (stir[-1][EM] / ctrl[-1][EM]))
print("E_mag, stirred unfed, start / end           %.1f" % (stir[0][EM] / stir[-1][EM]))
for tag, s in (("A = 4e-4", f4), ("A = 2e-4", f2)):
    rel = [r[EM] / s[0][EM] for r in s[1:]]
    print("E_mag / its start, fed %s: %.2f to %.2f over the probes after t = 0, %.2f at the end"
          % (tag, min(rel), max(rel), rel[-1]))
print("r.m.s. current end / start: fed 4e-4 %.2f, unfed %.2f"
      % (math.sqrt(f4[-1][JJ] / f4[0][JJ]), math.sqrt(stir[-1][JJ] / stir[0][JJ])))
dev = [abs(a[EK] / b[EK] - 1.0) for a, b in zip(f4[1:], stir[1:])]
print("E_kin, fed 4e-4 against unfed: within %.1f %% at every probe (%.1f %% at the end)"
      % (100 * max(dev), 100 * dev[-1]))
print("bulk rms(div b)/rms(curl b) at the end: fed 4e-4 %.3f, unfed %.3f, unstirred %.3f"
      % (f4[-1][BULK], stir[-1][BULK], ctrl[-1][BULK]))
print("dmass: unstirred %+.2e, stirred %+.2e, fed 2e-4 %+.2e, fed 4e-4 %+.2e"
      % tuple(dmass(n) for n in ("ctrl.txt", "stir.txt", "feed_2e-4.txt", "feed_4e-4.txt")))
print("\nthe table in ../README.md, at the end of each run:")
for tag, s in (("unstirred", ctrl), ("stirred", stir), ("fed 2e-4", f2), ("fed 4e-4", f4)):
    print("  %-10s E_kin / unstirred %3.0f   E_mag / its start %.3f   r.m.s. current / its start %.2f"
          "   bulk div/curl %.3f"
          % (tag, s[-1][EK] / ctrl[-1][EK], s[-1][EM] / s[0][EM],
             math.sqrt(s[-1][JJ] / s[0][JJ]), s[-1][BULK]))

if len(sys.argv) > 1:
    import numpy as np

    log = open(os.path.join(HERE, "stir.txt")).read()
    N = int(re.search(r"^\s+N (\d+)", log, re.M).group(1))
    R = float(re.search(r"half-width ([0-9.]+)", log).group(1))
    c = 0.5 * N - 0.5
    y, x = np.mgrid[0:N, 0:N]
    r = np.hypot(x - c, y - c)
    tr = np.loadtxt(os.path.join(HERE, "hand.txt"))
    om = np.where(tr[:, 2] > 0, np.gradient(np.radians(tr[:, 1]), tr[:, 0]), 0.0)
    frames = sorted(glob.glob(os.path.join(sys.argv[1], "decay_*_w[0-9]*.bin")))
    if not frames:
        sys.exit("no decay_*_w*.bin in " + sys.argv[1])
    print("\nrotation, from %d vorticity frames; the paddle's largest angular speed in "
          "contact is %.4e rad/step" % (len(frames), om.max()))
    # The frame stride, from the log's step count, and the strokes: runs of contact > 0.
    stride = int(re.search(r"running (\d+)", log).group(1)) // (len(frames) - 1)
    fstep = np.arange(len(frames)) * stride
    starts = [tr[k, 0] for k in range(len(tr)) if tr[k, 2] > 0 and (k == 0 or tr[k - 1, 2] == 0)]
    ends = [tr[k, 0] for k in range(len(tr)) if tr[k, 2] > 0 and (k == len(tr) - 1 or tr[k + 1, 2] == 0)]
    w80 = np.array([np.nan_to_num(np.fromfile(f, dtype=np.float32, offset=8).reshape(N, N)
                                  .astype(float))[r < 0.80 * R].sum() for f in frames])
    u80 = w80 / (2 * math.pi * 0.80 * R) / (om.max() * 0.80 * R)
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else fstep[-1] + 1
        m = (fstep >= a) & (fstep < b)
        print("  stroke %d, contact over steps %5d-%5d of the track's %d: largest at 0.80 R %.3f "
              "of the paddle's largest speed there (%d frames)"
              % (i + 1, a, ends[i], tr[-1, 0], u80[m].max(), int(m.sum())))
    for rr in (0.66, 0.80, 0.94):
        ut = []
        for f in frames:
            w = np.nan_to_num(np.fromfile(f, dtype=np.float32, offset=8).reshape(N, N).astype(float))
            ut.append(w[r < rr * R].sum() / (2 * math.pi * rr * R))
        ut = np.array(ut)
        k = int(ut.argmax())
        print("  azimuthal mean of u_theta at %.2f R: largest %.5f in frame %d of %d = %.3f of the "
              "paddle's largest speed there; counter-clockwise in %d of %d frames after t = 0; "
              "last frame %.3f"
              % (rr, ut[k], k, len(ut) - 1, ut[k] / (om.max() * rr * R),
                 int((ut[1:] > 0).sum()), len(ut) - 1, ut[-1] / (om.max() * rr * R)))
