#!/usr/bin/env python3
"""Compose the three-panel 3-D Orszag-Tang animation.

    python3 results/G_orszag_tang_3d/make_ot3d_animation.py <rundir> [out.mp4]

<rundir> is what GPU/csf3/ot3d_re3040.sub leaves behind and must already hold
slices.mp4, vol.mp4 and log.txt -- i.e. render_slices.py and render_volume.py
have been run and their PNGs encoded. This script adds the J_max(t) panel, which
neither renderer draws, and stacks the three.

WHY THIS FILE EXISTS. The .mp4 is not tracked -- .gitignore's rule is that a
rendered animation is regenerable from a dump and the dump is itself ignored --
so the composition has to be tracked instead, or it lives in whoever's shell
history built it. Same argument as results/N_mhd_sphere/make_wall_comparison.py.

THE WHOLE PIPELINE, because this is the last step of four:

    # 1. the run (CSF3; ~11 min on an A100 at M = 288)
    sbatch GPU/csf3/ot3d_re3040.sub
    # 2. render -- pure stdlib, no GPU, runs on a login node
    cd ot3_re3040
    python3 ../results/N_mhd_sphere/render_slices.py anim_frames png
    python3 ../results/N_mhd_sphere/render_volume.py anim_frames
    # 3. encode the two panels at the SAME framerate, so they stay in step
    ffmpeg -y -framerate 24 -i png/frame_%04d.png -c:v libx264 -pix_fmt yuv420p \
           -crf 20 -movflags +faststart slices.mp4
    ffmpeg -y -framerate 24 -i anim_frames/png/jvol_%04d.png -c:v libx264 \
           -pix_fmt yuv420p -crf 18 jvol.mp4
    # 4. this script

LAYOUT. Slices left at 2x nearest-neighbour, the |J| cube right, the J_max strip
across the bottom. 2x on the slices is not decoration: it makes the slice panel
exactly twice its own height, the cube is then scaled square to that height, and
the two fill the row with no dead band -- which is what the first version of this
merge had, because the slice panel is landscape and the volume portrait.

THE VOLUME CROP IS DERIVED, NOT MEASURED. render_volume.py lays its frame out as
W = P + 2*M and H = P + TOP + GAP + STRIP + 26 with M = 14, TOP = 46, GAP = 30,
STRIP = 74 -- so H - W = 148 BEFORE the even-padding, and the padding adds the
same 0 or 1 to both. The encode may also have upscaled by an integer s. Hence

    s    = (H - W) / 148          exact, and 1 or 2 in practice
    crop = (W - 28 s) square at (14 s, 46 s)

which drops the header and footer text bands and leaves the cube alone. Checked
against both runs: M = 100 gives s = 2 and 1412 px, M = 288 gives s = 1 and 678.
P is ambiguous by one pixel when it is odd (the pad hides it), which is two
pixels of extra black at the panel edge after a 2x upscale and is invisible.

THE STRIP READS log.txt, NOT THE .dat. The driver prints J_max in both lattice
and dimensionless units and the dimensionless one is column 7 -- the paper's.
Matching on "seven numeric fields" rather than on the word Jmax is deliberate:
the column header and the footnote both contain it.

AND IT DRAWS THE WHOLE CURVE FROM FRAME 0, traversed part bright and the rest
dimmed, rather than growing a line out of nothing. You can see where you are in
a shape you can already take in.
"""
import importlib.util
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(HERE, os.pardir, 'N_mhd_sphere', 'render_slices.py')

BG, FG, DIM, GRID = (12, 12, 16), (228, 232, 240), (150, 150, 160), (52, 56, 70)
LINE, AHEAD, CUR = (255, 176, 96), (86, 64, 44), (120, 190, 255)
SH, L, Rm, T, B = 300, 90, 40, 54, 46          # strip height and margins
FPS = 24


def load_font():
    """render_slices.py owns the Canvas, the 5x7 bitmap font and the PNG encoder.
    Import it rather than copy any of them: THIS ffmpeg has no drawtext, so that
    font is the only way to put a glyph on a pixel, and a second copy would
    drift."""
    spec = importlib.util.spec_from_file_location('render_slices', RS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def probe(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v',
         '-show_entries', 'stream=width,height,nb_frames',
         '-of', 'csv=p=0', path], capture_output=True, text=True).stdout.strip()
    w, h, n = out.split(',')[:3]
    return int(w), int(h), int(n)


def jmax_series(log):
    """(t, J_max) in the paper's units. Seven numeric fields is the data row;
    the header and the footnote both say "Jmax" and neither is one."""
    out = []
    for ln in open(log):
        f = ln.split()
        if len(f) == 7 and re.match(r'^[0-9]+\.[0-9]+$', f[0]):
            out.append((float(f[0]), float(f[6])))
    return out


def strip_frames(rs, pts, W, outdir):
    os.makedirs(outdir, exist_ok=True)
    tmax = max(t for t, _ in pts)
    jmax = max(j for _, j in pts)
    # 5-7 labels whatever the range: a fixed 10-unit grid draws 26 of them at
    # J_max ~ 250 and the axis becomes a smear.
    raw = jmax / 5.0
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(x * mag for x in (1, 2, 2.5, 5, 10) if x * mag >= raw)
    ytop = step * (int(jmax / step) + 1)
    px = lambda t: L + (W - L - Rm) * (t / tmax)
    py = lambda j: SH - B - (SH - T - B) * (j / ytop)
    pk = max(pts, key=lambda p: p[1])
    for k in range(len(pts)):
        cv = rs.Canvas(W, SH, BG)
        for gi in range(int(round(ytop / step)) + 1):
            g = gi * step
            y = int(py(g)); cv.rect(L, y, W - L - Rm, 1, GRID)
            cv.text(12, y - 3, '%4g' % g, DIM, 1)
        for g in range(int(tmax) + 1):
            x = int(px(g)); cv.rect(x, T, 1, SH - T - B, GRID)
            cv.text(x - 3, SH - B + 8, '%d' % g, DIM, 1)
        for i in range(1, len(pts)):
            x0, y0 = px(pts[i - 1][0]), py(pts[i - 1][1])
            x1, y1 = px(pts[i][0]), py(pts[i][1])
            n = max(1, int(abs(x1 - x0) + abs(y1 - y0)))
            col = LINE if i <= k else AHEAD
            for s in range(n + 1):
                cv.rect(int(x0 + (x1 - x0) * s / n),
                        int(y0 + (y1 - y0) * s / n), 2, 2, col)
        xc = int(px(pts[k][0]))
        cv.rect(xc, T, 1, SH - T - B, CUR)
        cv.rect(xc - 3, int(py(pts[k][1])) - 3, 7, 7, CUR)
        cv.text(12, 16, 'J_max  vs  t     (dimensionless)', FG, 2)
        cv.text(W - 560, 16, 't = %5.3f     J_max = %8.3f' % pts[k], FG, 2)
        cv.text(W - 560, 34, 'peak %.2f at t = %.3f' % (pk[1], pk[0]), DIM, 1)
        rs.write_png(os.path.join(outdir, 'j_%04d.png' % k), W, SH, cv.px)
    return pk


def main(argv):
    if len(argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    run = argv[1]
    out = argv[2] if len(argv) > 2 else os.path.join(run, 'ot3d_merged_jmax.mp4')
    sl, vo, log = (os.path.join(run, f) for f in ('slices.mp4', 'vol.mp4', 'log.txt'))
    for p in (sl, vo, log):
        if not os.path.exists(p):
            sys.stderr.write('missing %s\n  Run steps 1-3 from this file\'s '
                             'docstring first.\n' % p)
            return 1

    sw, sh, sn = probe(sl)
    vw, vh, vn = probe(vo)
    if sn != vn:
        sys.stderr.write('slices has %d frames, vol has %d. They were dumped on '
                         'different intervals (-dump vs -dumpvol) and would drift '
                         'apart; re-run with -dumpvol 1.\n' % (sn, vn))
        return 1

    s = (vh - vw) / 148.0
    if abs(s - round(s)) > 1e-6 or round(s) < 1:
        sys.stderr.write('vol.mp4 is %dx%d, which is not render_volume.py\'s '
                         'layout (H - W should be 148 x an integer upscale).\n'
                         % (vw, vh))
        return 1
    s = int(round(s))
    side = vw - 28 * s
    row_h = 2 * sh
    W = 2 * sw + row_h                       # slices at 2x, cube square to the row

    pts = jmax_series(log)
    if len(pts) != sn:
        sys.stderr.write('log.txt has %d diagnostic rows but the videos have %d '
                         'frames; they are not the same run.\n' % (len(pts), sn))
        return 1

    rs = load_font()
    jdir = os.path.join(run, 'jstrip')
    pk = strip_frames(rs, pts, W, jdir)
    print('%d frames  slices %dx%d  vol %dx%d (upscale %d, crop %d)  -> %dx%d'
          % (sn, sw, sh, vw, vh, s, side, W, row_h + SH))
    print('peak J_max %.3f at t = %.3f' % (pk[1], pk[0]))

    fc = ('[0:v]scale=iw*2:ih*2:flags=neighbor[s];'
          '[1:v]crop=%d:%d:%d:%d,scale=%d:%d:flags=lanczos[v];'
          '[s][v]hstack=inputs=2[top];'
          '[top][2:v]vstack=inputs=2,pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black[o]'
          % (side, side, 14 * s, 46 * s, row_h, row_h))
    cmd = ['ffmpeg', '-y', '-i', sl, '-i', vo,
           '-framerate', str(FPS), '-i', os.path.join(jdir, 'j_%04d.png'),
           '-filter_complex', fc, '-map', '[o]', '-frames:v', str(sn),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19',
           '-movflags', '+faststart', out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        return r.returncode
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
