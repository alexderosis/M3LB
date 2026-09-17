#!/usr/bin/env python3
"""Frames for the transitional MHD jet -- GPU/src/mhd_jet.cu's mid-plane dumps.

WHY THIS EXISTS RATHER THAN render_slices.py.  That renderer is built for a
SQUARE slice of a cube: it takes one N from meta.txt and lays the panels out
against it.  This case is a 512 x 734 portrait cut through a box that is
13.96 x 20 x 13.96, so N is not a number it has.  Everything that does not
depend on the shape -- the PNG writer, the bitmap font, the canvas, the
colour ramps -- is imported from it rather than copied, so a fix there is a
fix here.

THE SCALE IS FIXED ACROSS THE WHOLE SERIES, AND THAT IS THE POINT.  |J| grows
by three orders of magnitude between t = 20 and the reconnection peak, so a
per-frame normalisation -- which is what the reference's own Fig. 1 uses, with
a different colourbar on each panel -- makes the current look equally intense
throughout and hides the entire event.  A fixed LOG scale over `-decades`
below the global maximum shows the growth as growth.  `-lin` is there for a
single frame, not for a movie.

THE DEFAULT CROP IS PHYSICS, NOT FRAMING.  The jet is sech^2(x2), which is
below 1e-4 of its peak by |x2| = 5, so better than half of the 734 rows are
quiescent fluid that never does anything.  Cropping to |x2| <= 5 keeps every
structure and turns a 512 x 734 portrait into a 512 x 366 frame.  `-crop 0`
keeps the whole box.

usage:  render_jet.py <anim_frames dir> [-field jmag|umag|bmag] [-crop X]
                      [-decades D] [-lin] [-zoom Z] [-out DIR]
then:   ffmpeg -y -framerate 24 -i png/frame_%04d.png -c:v libx264 \
               -pix_fmt yuv420p -crf 18 jet.mp4
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'N_mhd_sphere'))
from render_slices import (Canvas, write_png, read_slice, make_lut,   # noqa: E402
                           INFERNO, VIRIDIS, MAGMA, BG, FG, percentile)

RAMPS = {'jmag': (INFERNO, '|J|  current'),
         'umag': (VIRIDIS, '|u|  speed'),
         'bmag': (MAGMA,   '|b|  field')}
LX = 4.0 * math.pi / 0.9              # the driver's box, x1 and x3
PAD, CAPH, CBARH = 12, 20, 10


def read_log(run_dir):
    """(times, dt) from log.txt.  The driver dumps every `-dump`-th PROBE, so
    the times are a subsample of the table and the stride is recovered by
    counting, never assumed.

    dt MATTERS AND IS NOT DECORATION.  dump_slices() forms the curl with a
    spacing of one cell, so every value in the .raw files is in LATTICE units,
    where the reference's |J| is ~1e1 and these are ~1e-7.  The conversion is
    J = J(lat)/dt, the same one the driver applies to its own J_max column.
    Without log.txt this renderer says LATTICE on the caption rather than
    printing a number in units it cannot name -- a mislabelled current has
    already cost this project once."""
    log = os.path.join(run_dir, '..', 'log.txt')
    if not os.path.exists(log):
        log = os.path.join(run_dir, 'log.txt')
    if not os.path.exists(log):
        return None
    ts, dt = [], None
    for line in open(log):
        if dt is None:
            m = re.search(r'dt = ([0-9.eE+-]+)', line)
            if m:
                dt = float(m.group(1))
        p = line.split()
        if len(p) >= 10:
            try:
                ts.append(float(p[0]))
            except ValueError:
                pass
    return (ts or None), dt


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    run = argv[0]
    opt = {'field': 'jmag', 'crop': 5.0, 'decades': 3.5, 'zoom': 1,
           'out': None, 'log': True}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '-lin':
            opt['log'] = False
        elif a == '-field' and i + 1 < len(argv):
            i += 1; opt['field'] = argv[i]
        elif a == '-crop' and i + 1 < len(argv):
            i += 1; opt['crop'] = float(argv[i])
        elif a == '-decades' and i + 1 < len(argv):
            i += 1; opt['decades'] = float(argv[i])
        elif a == '-zoom' and i + 1 < len(argv):
            i += 1; opt['zoom'] = max(1, int(argv[i]))
        elif a == '-out' and i + 1 < len(argv):
            i += 1; opt['out'] = argv[i]
        else:
            print('unknown option %r' % a); return 2
        i += 1
    if opt['field'] not in RAMPS:
        print('field must be one of %s' % ', '.join(RAMPS)); return 2

    names = sorted(f for f in os.listdir(run)
                   if re.fullmatch(opt['field'] + r'_\d+\.raw', f))
    if not names:
        print('no %s_*.raw in %s' % (opt['field'], run)); return 1

    # Pass one: the global maximum, and the geometry, from the files themselves.
    nx = ny = None
    hi = 0.0
    frames = []
    for n in names:
        try:
            w, h, v = read_slice(os.path.join(run, n))
        except ValueError as e:
            print('  skipping %s (%s)' % (n, e)); continue
        if nx is None:
            nx, ny = w, h
        elif (w, h) != (nx, ny):
            print('  %s is %dx%d, expected %dx%d -- refusing to mix shapes'
                  % (n, w, h, nx, ny)); return 1
        hi = max(hi, max(v))
        frames.append((n, v))
    if not frames:
        print('every frame was unreadable'); return 1

    dl = LX / nx
    if opt['crop'] > 0:
        half = min(ny // 2, int(round(opt['crop'] / dl)))
        y0, y1 = ny // 2 - half, ny // 2 + half
    else:
        y0, y1 = 0, ny
    rows = y1 - y0
    lut = make_lut(RAMPS[opt['field']][0])
    Z = opt['zoom']
    lo = hi / (10.0 ** opt['decades'])

    W = nx * Z + 2 * PAD
    H = rows * Z + CAPH + CBARH + 3 * PAD
    W += W & 1                                   # libx264 wants even dimensions
    H += H & 1

    ts, dt = read_log(run)
    if ts and len(ts) >= len(frames):
        stride = max(1, round((len(ts) - 1) / max(1, len(frames) - 1)))
        ts = [ts[min(k * stride, len(ts) - 1)] for k in range(len(frames))]
    else:
        ts = None

    out = opt['out'] or os.path.join(run, 'png')
    os.makedirs(out, exist_ok=True)
    print('%d frames, %dx%d slice, dl = %.5f' % (len(frames), nx, ny, dl))
    print('crop |x2| <= %.2f -> %d rows; canvas %dx%d at zoom %d'
          % (opt['crop'], rows, W, H, Z))
    unit = 'lattice'
    hi_q = hi
    if dt:
        unit, hi_q = "reference's", hi / dt
        print('J = J(lat)/dt with dt = %.6e from log.txt' % dt)
    else:
        print('NO log.txt -- values stay in LATTICE units and the caption says so')
    print('scale %s, fixed over the series: max %.4g (%s units), floor %.4g'
          % ('log' if opt['log'] else 'linear', hi_q, unit,
             hi_q / (10.0 ** opt['decades']) if opt['log'] else 0.0))

    span = math.log10(hi) - math.log10(lo) if opt['log'] else hi
    for k, (n, v) in enumerate(frames):
        cv = Canvas(W, H, BG)
        px = []
        for y in range(y0, y1):
            row = bytearray()
            base = y * nx
            for x in range(nx):
                val = v[base + x]
                if opt['log']:
                    t = 0.0 if val <= lo else (math.log10(val) - math.log10(lo)) / span
                else:
                    t = val / hi if hi > 0 else 0.0
                c = lut[min(255, max(0, int(t * 255)))]
                row += c * Z
            px.append(bytes(row))
        cv.blit_rows(PAD, PAD, px, Z)
        cv.frame_rect(PAD - 1, PAD - 1, nx * Z + 2, rows * Z + 2, (60, 66, 78))

        cby = PAD + rows * Z + PAD
        for x in range(nx * Z):
            cv.rect(PAD + x, cby, 1, CBARH, lut[min(255, x * 256 // (nx * Z))])
        cv.frame_rect(PAD - 1, cby - 1, nx * Z + 2, CBARH + 2, (60, 66, 78))

        cap = '%s   %s   max %.4g%s' % (
            RAMPS[opt['field']][1],
            'log, %.1f decades' % opt['decades'] if opt['log'] else 'linear',
            hi_q, '' if dt else ' (LATTICE)')
        if ts:
            cap = 't = %7.2f   ' % ts[k] + cap
        cv.text(PAD, cby + CBARH + PAD - 4, cap.upper(), FG, 1)
        write_png(os.path.join(out, 'frame_%04d.png' % k), W, H, cv.px)
        if k % 20 == 0 or k == len(frames) - 1:
            print('  %4d/%d' % (k + 1, len(frames)))

    print('\nwrote %d PNGs to %s' % (len(frames), out))
    print('ffmpeg -y -framerate 24 -i %s/frame_%%04d.png -c:v libx264 '
          '-pix_fmt yuv420p -crf 18 jet_%s.mp4' % (out, opt['field']))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
