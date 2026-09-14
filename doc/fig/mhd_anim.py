"""Assemble MHD frame dumps into an animation.

Reads the compact frames written by demonstrator/mhd_cylinder and
demonstrator/mhd_decay -- two little-endian int32 then nx*ny float32, which is
mkpng.read_field's own format -- and hands the PNGs to ffmpeg.

  usage: mhd_anim.py <glob> <out.mp4> [--fps N] [--vmax V] [--stack <glob2>]
                     [--label TEXT] [--pct P]

THE COLOUR SCALE IS SHARED ACROSS THE WHOLE SEQUENCE BY DEFAULT, and that is
the point rather than a detail. A frame normalised to its own extremum shows a
decaying field as one of constant brightness -- which looks like physics and is
the opposite of it. The scale is taken as a percentile over a SAMPLE of frames
rather than over the first or the last: a decay's first frame is the brightest
it will ever be and a wake's first frame is a transient, so either choice alone
biases the whole film.

--pernorm OVERRIDES THAT, AND IS THE RIGHT CHOICE FOR EXACTLY ONE CASE. A freely
decaying field loses three to four decades of amplitude, and on a shared scale
the entire second half of the film is blank -- which is where the thing being
shown actually happens, since selective decay is a statement about the LENGTH
scale growing and not about the amplitude. So each frame is normalised to its
own percentile, the film shows structure rather than brightness, and the
amplitude that has been divided out is reported as a number instead: the
mhd_decay table's E_kin and E_mag columns are the honest record of it. Do not
use it on a wake, where the amplitude is the result.

NaN MEANS SOLID and is painted, not interpolated. The dumps write NaN into the
obstacle and into every cell whose derivative stencil would reach it, so a
staircase body does not emit a ring of false vorticity.

--stack renders a second series beneath the first in one frame, which is how
vorticity and current are meant to be read: the wake and the thing braking it.
"""
import glob, math, os, struct, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkpng

BODY = (28, 30, 38)          # solid / masked cells


def read(fn):
    d = open(fn, 'rb').read()
    nx, ny = struct.unpack('<ii', d[:8])
    v = struct.unpack('<%df' % (nx * ny), d[8:8 + 4 * nx * ny])
    return nx, ny, v


def scale_of(files, pct):
    """Symmetric diverging scale from a sample of the sequence."""
    step = max(1, len(files) // 12)
    mags = []
    for fn in files[::step]:
        _, _, v = read(fn)
        mags.extend(abs(t) for t in v if t == t)      # t == t rejects NaN
    if not mags:
        return 1.0
    mags.sort()
    return max(mags[min(len(mags) - 1, int(pct * len(mags)))], 1e-30)


def rgb_of(nx, ny, v, vmax):
    """Row-major RGB, flipped so that y = 0 is the BOTTOM of the image."""
    out = bytearray(nx * ny * 3)
    inv = 0.5 / vmax
    for y in range(ny):
        src = (ny - 1 - y) * nx
        dst = y * nx * 3
        for x in range(nx):
            t = v[src + x]
            if t != t:
                r, g, b = BODY
            else:
                r, g, b = mkpng.cmap(0.5 + t * inv)
            out[dst] = r; out[dst + 1] = g; out[dst + 2] = b
            dst += 3
    return out


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    pattern, out = args[0], args[1]
    fps, vmax, stack, pct, pernorm = 20, None, None, 0.995, False
    i = 2
    while i < len(args):
        if args[i] == '--fps': fps = int(args[i + 1]); i += 2
        elif args[i] == '--vmax': vmax = float(args[i + 1]); i += 2
        elif args[i] == '--stack': stack = args[i + 1]; i += 2
        elif args[i] == '--pct': pct = float(args[i + 1]); i += 2
        elif args[i] == '--pernorm': pernorm = True; i += 1
        else: i += 1

    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit('no frames match %s' % pattern)
    sfiles = sorted(glob.glob(stack)) if stack else []
    if stack and len(sfiles) != len(files):
        # NOT a warning to ignore: a mismatched pair silently pastes frame k of
        # one field beside frame k of another taken at a different time.
        sys.exit('stack mismatch: %d vs %d frames' % (len(files), len(sfiles)))

    v1 = vmax or (None if pernorm else scale_of(files, pct))
    # AN EXPLICIT --vmax APPLIES TO BOTH PANELS. Each series otherwise gets its
    # own percentile, which is right when the two panels are different
    # QUANTITIES (vorticity above, current below) and wrong when they are the
    # same quantity under different conditions -- a Hartmann sweep stacked to
    # show suppression is a comparison of amplitudes, and giving each panel its
    # own scale would normalise away the very thing being shown.
    v2 = (vmax or (None if pernorm else scale_of(sfiles, pct))) if sfiles else None
    print('%d frames   %s' % (len(files),
          'PER-FRAME normalisation (amplitude divided out)' if pernorm
          else 'shared scale %.4g%s' % (v1, ('   stacked %.4g' % v2) if v2 else '')))

    tmp = tempfile.mkdtemp(prefix='mhdanim_')
    for k, fn in enumerate(files):
        nx, ny, v = read(fn)
        rgb = rgb_of(nx, ny, v, v1 if v1 else scale_of([fn], pct))
        if sfiles:
            nx2, ny2, w = read(sfiles[k])
            if nx2 != nx:
                sys.exit('stacked frames have different widths')
            gap = bytes(BODY) * nx * 3            # three rows of separator
            rgb = bytes(rgb) + gap + rgb_of(nx2, ny2, w,
                                            v2 if v2 else scale_of([sfiles[k]], pct))
            ny = ny + 3 + ny2
        mkpng.write_png(os.path.join(tmp, 'f%05d.png' % k), nx, ny, rgb)
        if (k + 1) % 20 == 0:
            print('  %d/%d' % (k + 1, len(files))); sys.stdout.flush()

    # yuv420p needs even dimensions; pad rather than crop so nothing is lost.
    subprocess.run(['ffmpeg', '-y', '-framerate', str(fps),
                    '-i', os.path.join(tmp, 'f%05d.png'),
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out],
                   check=True)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
