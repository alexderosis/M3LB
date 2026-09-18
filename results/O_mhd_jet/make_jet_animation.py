#!/usr/bin/env python3
"""One film from the jet run: mid-plane slice, rotating volume, and gamma(t).

Top row is the two |J| renders side by side; under them the magnetic-energy
growth rate with a cursor on the current frame, so the structure above and the
number below are always the same instant.

WHY A SCRIPT AND NOT A PASTED ffmpeg LINE. The geometry is neither obvious nor
stable: the slice canvas depends on -crop and -zoom, the volume canvas on
--vscale, --cropy and the box's own extents, and getting either wrong scales
one panel against the other by a few per cent, which reads as a rendering
fault rather than a composition mistake. Everything is DERIVED from the PNGs
and the log on disk. Nothing is typed in.

WHAT IT REFUSES, each having produced a plausible wrong film in this tree:
  * different frame counts in the two directories -- ffmpeg pads the shorter
    input by HOLDING ITS LAST FRAME, so the panels drift apart in time and
    nothing says so;
  * a log whose probe count is not a whole multiple of the frame count, which
    means the log is from a different run than the frames;
  * a missing or empty directory.

THE PANELS ARE MATCHED ON HEIGHT, EACH BY AN INTEGER FACTOR. Both renders are
one pixel per lattice cell and a fractional rescale would blur exactly the
one-cell-thick current sheets the picture is of, so each panel is scaled by the
integer nearest to the height ratio (nearest, not floor: at 432 against 802,
floor gives 1 and leaves the slice at half the volume's height) and the
difference is absorbed by padding.

GAMMA IS PLOTTED AS d(E_b/E_b0)/dt, TWICE THE LOG'S COLUMN. The driver prints
Eq. (3.1) read literally -- half that -- and the amplitude evidence says the
1/2 there is the energy prefactor rather than an extra halving: doubling moved
the minimum from 49 % of the reference's to 97 % and the peak from 41 % to
82 %. Two independent amplitudes landing together under one factor is the
argument. If that reading is ever settled the other way, -half plots the
column as printed.

usage: make_jet_animation.py <run dir> [-o out.mp4] [-fps N] [-half]
                             [-slice DIR] [-vol DIR] [-log FILE]
"""
import importlib.util
import math
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(HERE, os.pardir, 'N_mhd_sphere', 'render_slices.py')

BG, FG, DIM, GRID = (12, 12, 16), (228, 232, 240), (150, 150, 160), (52, 56, 70)
LINE, AHEAD, CUR, ZERO = (255, 176, 96), (86, 64, 44), (120, 190, 255), (96, 100, 118)
SH, L, Rm, T, B = 300, 96, 40, 56, 46


def load_rs():
    """render_slices.py owns the Canvas, the 5x7 bitmap font and the PNG
    encoder. Imported rather than copied: this ffmpeg has no drawtext, so that
    font is the only way to put a glyph on a pixel, and a second copy would
    drift from it."""
    spec = importlib.util.spec_from_file_location('render_slices', RS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def png_size(path):
    with open(path, 'rb') as f:
        d = f.read(24)
    if len(d) < 24 or d[:8] != b'\x89PNG\r\n\x1a\n':
        raise SystemExit('%s is not a PNG' % path)
    return struct.unpack('>II', d[16:24])


def frames(d, pat):
    if not os.path.isdir(d):
        raise SystemExit('no such directory: %s' % d)
    got = sorted(f for f in os.listdir(d) if re.fullmatch(pat, f))
    if not got:
        raise SystemExit('no frames matching %s in %s' % (pat, d))
    return got


def probe_rows(log):
    """(t, gamma) for every probe row. gamma is None on the t = 0 row, which
    prints '-' because the rate is a difference and there is nothing to
    difference against yet."""
    out = []
    for ln in open(log):
        f = ln.split()
        if len(f) == 10 and re.match(r'^\d+\.\d+$', f[0]):
            try:
                g = float(f[4])
            except ValueError:
                g = None
            out.append((float(f[0]), g))
    if not out:
        raise SystemExit('no ten-field data rows in %s -- is that a jet log?' % log)
    return out


def nice_step(span, want=6):
    raw = span / float(want)
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    return next((x * mag for x in (1, 2, 2.5, 5, 10) if x * mag >= raw), mag * 10)


def strip_frames(rs, rows, pick, scale, W, outdir):
    """One strip per frame. `pick[k]` indexes `rows` for frame k."""
    os.makedirs(outdir, exist_ok=True)
    pts = [(t, scale * g) for t, g in rows if g is not None]
    tmin, tmax = rows[0][0], rows[-1][0]
    gmin = min(g for _, g in pts)
    gmax = max(g for _, g in pts)
    step = nice_step(gmax - gmin)
    lo = step * math.floor(gmin / step)
    hi = step * math.ceil(gmax / step)
    px = lambda t: L + (W - L - Rm) * ((t - tmin) / (tmax - tmin))
    py = lambda g: SH - B - (SH - T - B) * ((g - lo) / (hi - lo))
    pk = max(pts, key=lambda p: p[1])
    tr = min(pts, key=lambda p: p[1])
    # zero crossing, linearly interpolated -- the saturation instant
    zc = None
    for i in range(1, len(pts)):
        if pts[i - 1][1] > 0.0 >= pts[i][1]:
            a, b = pts[i - 1], pts[i]
            zc = a[0] + (b[0] - a[0]) * a[1] / (a[1] - b[1])
            break
    lab = 'gamma  (Eq. 3.1)' if abs(scale - 1.0) < 1e-9 else 'd(E_b/E_b0)/dt'
    for k in range(len(pick)):
        cv = rs.Canvas(W, SH, BG)
        g = lo
        while g <= hi + 1e-9:
            y = int(py(g))
            cv.rect(L, y, W - L - Rm, 1, ZERO if abs(g) < 1e-9 else GRID)
            cv.text(14, y - 3, '%6g' % g, DIM, 1)
            g += step
        for t in range(0, int(tmax) + 1, 20):
            x = int(px(t))
            cv.rect(x, T, 1, SH - T - B, GRID)
            cv.text(x - 6, SH - B + 8, '%d' % t, DIM, 1)
        if zc is not None:
            xz = int(px(zc))
            for y in range(T, SH - B, 4):
                cv.rect(xz, y, 1, 2, (70, 96, 120))
        now = rows[pick[k]][0]
        for i in range(1, len(pts)):
            x0, y0 = px(pts[i - 1][0]), py(pts[i - 1][1])
            x1, y1 = px(pts[i][0]), py(pts[i][1])
            n = max(1, int(abs(x1 - x0) + abs(y1 - y0)))
            col = LINE if pts[i][0] <= now else AHEAD
            for sp in range(n + 1):
                cv.rect(int(x0 + (x1 - x0) * sp / n),
                        int(y0 + (y1 - y0) * sp / n), 2, 2, col)
        xc = int(px(now))
        cv.rect(xc, T, 1, SH - T - B, CUR)
        gnow = rows[pick[k]][1]
        if gnow is not None:
            cv.rect(xc - 3, int(py(scale * gnow)) - 3, 7, 7, CUR)
        cv.text(14, 16, ('%s  vs  t' % lab).upper(), FG, 2)
        cv.text(W - 620, 16, 't = %6.2f    rate = %+7.3f'
                % (now, scale * gnow if gnow is not None else 0.0), FG, 2)
        cv.text(W - 620, 36, 'peak %+.3f at t=%.1f   min %+.3f at t=%.1f%s'
                % (pk[1], pk[0], tr[1], tr[0],
                   '   zero at t=%.1f' % zc if zc else ''), DIM, 1)
        rs.write_png(os.path.join(outdir, 'g_%04d.png' % k), W, SH, cv.px)
    return pk, tr, zc


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    run = argv[0]
    out = os.path.join(run, 'jet_merged.mp4')
    fps, scale, sdir, vdir, log = 12, 2.0, None, None, None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '-o' and i + 1 < len(argv):
            i += 1; out = argv[i]
        elif a == '-fps' and i + 1 < len(argv):
            i += 1; fps = int(argv[i])
        elif a == '-half':
            scale = 1.0
        elif a == '-slice' and i + 1 < len(argv):
            i += 1; sdir = argv[i]
        elif a == '-vol' and i + 1 < len(argv):
            i += 1; vdir = argv[i]
        elif a == '-log' and i + 1 < len(argv):
            i += 1; log = argv[i]
        else:
            raise SystemExit('unknown option %r' % a)
        i += 1
    sdir = sdir or os.path.join(run, 'png_slice')
    vdir = vdir or os.path.join(run, 'png_vol')
    log = log or os.path.join(run, 'log.txt')

    sf = frames(sdir, r'frame_\d+\.png')
    vf = frames(vdir, r'jvol_\d+\.png')
    if len(sf) != len(vf):
        raise SystemExit(
            'frame counts differ: %d slices, %d volumes.\n'
            '  ffmpeg would hold the shorter input on its last frame and the\n'
            '  panels would drift apart in time without saying so.\n'
            '  Re-render whichever stopped early.' % (len(sf), len(vf)))
    n = len(sf)

    rows = probe_rows(log)
    stride = int(round((len(rows) - 1) / float(max(1, n - 1))))
    if stride < 1 or abs(stride * (n - 1) - (len(rows) - 1)) > stride:
        raise SystemExit(
            '%d probe rows do not subsample to %d frames (stride %d).\n'
            '  That log is from a different run than those frames.'
            % (len(rows), n, stride))
    pick = [min(k * stride, len(rows) - 1) for k in range(n)]

    sw, sh = png_size(os.path.join(sdir, sf[0]))
    vw, vh = png_size(os.path.join(vdir, vf[0]))
    ref = max(sh, vh)
    us, uv = max(1, int(round(ref / float(sh)))), max(1, int(round(ref / float(vh))))
    tsw, tsh, tvw, tvh = sw * us, sh * us, vw * uv, vh * uv
    row_h = max(tsh, tvh)
    W = tsw + tvw
    W += W & 1

    rs = load_rs()
    gdir = os.path.join(run, 'png_gamma')
    pk, tr, zc = strip_frames(rs, rows, pick, scale, W, gdir)

    H = row_h + SH
    H += H & 1
    print('%d frame triples, t = %.1f to %.1f' % (n, rows[0][0], rows[-1][0]))
    print('slice  %dx%d x%d -> %dx%d' % (sw, sh, us, tsw, tsh))
    print('volume %dx%d x%d -> %dx%d' % (vw, vh, uv, tvw, tvh))
    print('gamma  %dx%d   peak %+.3f at t=%.1f, min %+.3f at t=%.1f, zero t=%s'
          % (W, SH, pk[1], pk[0], tr[1], tr[0],
             '%.2f' % zc if zc else 'none'))
    print('canvas %dx%d (%.2f:1) at %d fps -> %.1f s' % (W, H, W / float(H), fps, n / float(fps)))

    # EVERY STAGE HERE IS ONE THAT DOES SOMETHING. A scale by 1 and a pad to the
    # size the input already has are both no-ops in arithmetic, and ffmpeg 9
    # rejects the second outright -- "Padded dimensions cannot be smaller than
    # input dimensions" -- so they are emitted only when they change the frame.
    # W and H are already forced even above, which is what the trailing
    # ceil(iw/2)*2 pad would otherwise be for.
    def chain(src, up, w, h, th, tag):
        f = '[%s]' % src
        if up > 1:
            f += 'scale=iw*%d:ih*%d:flags=neighbor,' % (up, up)
        if th != h:
            f += 'pad=%d:%d:0:%d:black,' % (w, h, (h - th) // 2)
        return f.rstrip(',') + '[%s]' % tag if f.endswith(',') else f + 'null[%s]' % tag

    fc = ';'.join([chain('0:v', us, tsw, row_h, tsh, 's'),
                   chain('1:v', uv, tvw, row_h, tvh, 'v'),
                   '[s][v]hstack=inputs=2[top]',
                   '[top][2:v]vstack=inputs=2[o]'])
    cmd = ['ffmpeg', '-y',
           '-framerate', str(fps), '-i', os.path.join(sdir, 'frame_%04d.png'),
           '-framerate', str(fps), '-i', os.path.join(vdir, 'jvol_%04d.png'),
           '-framerate', str(fps), '-i', os.path.join(gdir, 'g_%04d.png'),
           '-filter_complex', fc, '-map', '[o]',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out]
    r = subprocess.run(cmd, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode('utf-8', 'replace')[-2500:])
        return r.returncode
    print('\nwrote %s  (%.1f MB)' % (out, os.path.getsize(out) / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
