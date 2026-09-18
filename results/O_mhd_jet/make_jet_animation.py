#!/usr/bin/env python3
"""One film from the jet's two renders: the mid-plane slice beside the volume.

WHY A SCRIPT AND NOT A PASTED ffmpeg LINE. The geometry is not obvious and it
is not stable: the slice canvas depends on -crop and -zoom, the volume canvas
on --vscale and on the box's own extents, and getting either wrong scales one
panel against the other by a few per cent, which reads as a rendering fault
rather than as a composition mistake. Everything below is DERIVED from the
PNGs on disk. Nothing is typed in.

WHAT IT REFUSES TO DO, because each of these has produced a plausible wrong
film in this tree before:
  * different frame counts in the two directories -- that means one render
    stopped early and the panels would drift out of time with each other,
    silently, since ffmpeg pads the short input by holding its last frame;
  * a frame count of zero, or a directory that is not there;
  * panel widths that disagree by more than the integer upscale can fix.

HOW THE PANELS ARE MATCHED. The slice is upscaled by the largest INTEGER
factor that does not exceed the volume's width, with nearest-neighbour
sampling: the slice is one pixel per lattice cell and a fractional or smooth
rescale would blur exactly the one-cell-thick current sheets the picture is
of. It is then padded to the volume's height and centred. Integer-only is
also why the two panels end up within a pixel or two of each other rather
than exactly equal -- the pad absorbs the difference.

usage: make_jet_animation.py <run dir> [-o out.mp4] [-fps N] [-slice DIR]
                             [-vol DIR]
The run dir is the one holding png_slice/ and png_vol/; -slice and -vol
override either.
"""
import os
import re
import struct
import subprocess
import sys


def png_size(path):
    with open(path, 'rb') as f:
        d = f.read(24)
    if len(d) < 24 or d[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('%s is not a PNG' % path)
    w, h = struct.unpack('>II', d[16:24])
    return w, h


def frames(d, pat):
    if not os.path.isdir(d):
        raise SystemExit('no such directory: %s' % d)
    got = sorted(f for f in os.listdir(d) if re.fullmatch(pat, f))
    if not got:
        raise SystemExit('no frames matching %s in %s' % (pat, d))
    return got


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    run = argv[0]
    out = os.path.join(run, 'jet_merged.mp4')
    fps, sdir, vdir = 12, None, None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '-o' and i + 1 < len(argv):
            i += 1; out = argv[i]
        elif a == '-fps' and i + 1 < len(argv):
            i += 1; fps = int(argv[i])
        elif a == '-slice' and i + 1 < len(argv):
            i += 1; sdir = argv[i]
        elif a == '-vol' and i + 1 < len(argv):
            i += 1; vdir = argv[i]
        else:
            raise SystemExit('unknown option %r' % a)
        i += 1
    sdir = sdir or os.path.join(run, 'png_slice')
    vdir = vdir or os.path.join(run, 'png_vol')

    sf = frames(sdir, r'frame_\d+\.png')
    vf = frames(vdir, r'jvol_\d+\.png')
    if len(sf) != len(vf):
        raise SystemExit(
            'frame counts differ: %d slices, %d volumes.\n'
            '  ffmpeg would hold the shorter input on its last frame and the\n'
            '  two panels would drift apart in time without saying so.\n'
            '  Re-render whichever stopped early, or trim both to %d.'
            % (len(sf), len(vf), min(len(sf), len(vf))))

    sw, sh = png_size(os.path.join(sdir, sf[0]))
    vw, vh = png_size(os.path.join(vdir, vf[0]))
    up = max(1, vw // sw)                  # integer, so the sheets stay crisp
    tw, th = sw * up, sh * up
    if tw > vw:
        raise SystemExit('slice %dx%d upscales past the volume width %d'
                         % (sw, sh, vw))
    W, H = vw + vw, vh                     # each panel gets the volume's width
    pad_x = (vw - tw) // 2
    pad_y = (vh - th) // 2
    if pad_y < 0:
        raise SystemExit('slice is taller than the volume after upscaling '
                         '(%d > %d); lower the slice -zoom' % (th, vh))

    print('%d frame pairs' % len(sf))
    print('slice  %dx%d  x%d -> %dx%d, padded into %dx%d at (%d, %d)'
          % (sw, sh, up, tw, th, vw, vh, pad_x, pad_y))
    print('volume %dx%d' % (vw, vh))
    print('canvas %dx%d  (%.2f:1)  at %d fps -> %.1f s'
          % (W, H, W / float(H), fps, len(sf) / float(fps)))

    fc = ('[0:v]scale=iw*%d:ih*%d:flags=neighbor,'
          'pad=%d:%d:%d:%d:black[s];'
          '[s][1:v]hstack=inputs=2,'
          'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black[o]'
          % (up, up, vw, vh, pad_x, pad_y))
    cmd = ['ffmpeg', '-y',
           '-framerate', str(fps), '-i', os.path.join(sdir, 'frame_%04d.png'),
           '-framerate', str(fps), '-i', os.path.join(vdir, 'jvol_%04d.png'),
           '-filter_complex', fc, '-map', '[o]',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out]
    print('\n' + ' '.join(cmd) + '\n')
    r = subprocess.run(cmd, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode('utf-8', 'replace')[-2500:])
        return r.returncode
    print('wrote %s  (%.1f MB)' % (out, os.path.getsize(out) / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
