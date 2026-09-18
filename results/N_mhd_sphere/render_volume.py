#!/usr/bin/env python3
"""Volume render of |J| for demonstrator/mhd_sphere -- current sheets in 3-D.

  usage: render_volume.py [in_dir] [out.mp4] [flags]
  e.g.:  render_volume.py results/N_mhd_sphere
         render_volume.py results/N_mhd_sphere jvol.mp4 --scale frame --size 640

  in_dir defaults to this script's own directory, and the dumps are looked for
  in <in_dir>/anim_frames, then <in_dir>/frames, then <in_dir> itself.

  --scale log|frame|linear  transfer function (default log; see below -- this is
                            the flag that decides whether the film tells the
                            truth about the decay)
  --decades D               length of the log ramp; default 0 = size it to this
                            run's own fall. PIN IT when comparing two runs.
  --yawrate D  --yaw0 D  --elev D    camera, degrees (default 3/frame, 35, 22)
  --title S                          banner line (default names the case)
  --spin z|y                         axis the camera orbits (default z)
  --ramp emissive|inferno            colour ramp (default emissive)
  --cropy N                          box mode: keep rows within N of the
                                     y centre (default 0, the whole box)
  --vscale S                MIP samples per voxel, splat is ceil(S) wide
                            (default 4; drop to 2 for ~2x the speed)
  --size PX                 target panel width (default 440)
  --every N  --limit N      stride / truncate the frame list
  --fps N                   framerate in the printed ffmpeg command (default 12)
  --png DIR                 PNG output directory (default <in_dir>/png)
  --R r                     sphere radius, if meta.txt has none (default 0.40 N)
  --nodepth --nosmooth      drop the depth cue / the display tent filter
  --selftest                write a PNG and read it back before rendering
  --quiet

STANDARD LIBRARY ONLY -- no numpy, no PIL, no matplotlib. PNG is written through
zlib and struct as in doc/fig/mkpng.py and doc/fig/vol3d.py, and `--selftest`
reads one back with the parser in this file. ffmpeg is called only by the
command printed at the end, and never with drawtext: every glyph below is
rasterised into the pixels by draw_text().

--------------------------------------------------------------------------------
WHY A VOLUME AND NOT THE SLICE
--------------------------------------------------------------------------------
jmag_%04d.raw is the z = N/2 cut, and a current sheet is a SURFACE: the slice
shows it only where it happens to cut that one plane, so a sheet lying parallel
to the cut is invisible and a sheet crossing it reads as a line whose extent is
an artefact of the cut. This projects the whole jvol_%04d.raw volume instead.

MAXIMUM-INTENSITY PROJECTION, not an alpha composite. MIP is order-independent,
so the voxels can be visited in storage order and no depth sort is needed --
which is what makes a pure-Python renderer affordable at all. What MIP does NOT
show: occlusion (a bright sheet behind a dim one still wins) and therefore true
depth ordering; the DEPTH CUE below is a substitute, not a fix. What it does
show honestly: the peak |J| along every ray, so a sheet is never hidden.

DEPTH CUE. Alongside the max, the renderer stores the DEPTH AT WHICH THAT MAX
OCCURRED, and dims the far half of the volume by up to 38%. That is a cue, not
physics: it tells you which of two structures is nearer, and it must not be read
as an intensity difference. It costs one extra store per accepted voxel.

--------------------------------------------------------------------------------
THE GEOMETRY IS A SPHERE AND THE OUTSIDE IS NOT PHYSICS
--------------------------------------------------------------------------------
mhd_sphere confines the flow by volume penalisation to r <= R about
((N-1)/2, (N-1)/2, (N-1)/2), and dump() deliberately does NOT mask the exterior
before writing (its banner: a file that has already thrown the penalised region
away cannot be used to check that it really is at rest). So the MASK IS THE
RENDERER'S JOB. Every loop here is driven by a per-(z,y) span of x that lies
inside the sphere, so exterior voxels are never read, never drive the colour
scale, and never reach a pixel. The sphere is drawn as a faint ring at its own
projected radius -- under an orthographic camera a sphere projects to a circle
of radius R at every yaw, so the ring is exact and is the viewer's frame of
reference for the rotation.

--------------------------------------------------------------------------------
THE COLOUR SCALE, WHICH IS THE PART THAT CAN LIE
--------------------------------------------------------------------------------
This is a DECAYING flow, but NOT UNIFORMLY SO ACROSS THE FIELDS, and scaling
|J| off the kinetic energy would be badly wrong. Measured on the real 129-frame
N = 64 run (results/N_mhd_sphere/anim_frames/meta.txt, t/Te 0 -> 4):

    E_u          1.250e-03 -> 3.547e-07     3524x
    E_b          1.250e-03 -> 1.496e-05       84x
    |J| P99.5    2.867e-02 -> 1.634e-03     17.5x, and it PEAKS AT FRAME 4
                 (3.814e-02), so the fall from the peak is 23.3x = 1.37 decades

The kinetic energy falls forty times harder than the magnetic, because this is
selective decay; |J| is a curl of B and follows B, not u. AND |J| RISES BEFORE
IT FALLS -- the sheets sharpen over the first few frames -- which is why vref
below is the largest per-frame value in the run and not the first frame's.

Two obvious schemes are both wrong on their own:

  * per-frame autoscale renders a decaying flow at constant brightness. It looks
    like a steady state. It is the single most misleading thing this script
    could do, and it is not the default.
  * one fixed linear scale is honest about amplitude and leaves the last 80% of
    the film black -- which is exactly where selective decay does its work,
    since that is a statement about the LENGTH scale growing, not the amplitude.

--scale log IS THE DEFAULT and is the compromise: a FIXED logarithmic scale
spanning `--decades` below a single global reference vref = the LARGEST
per-frame P99.5 in the run, so nothing clips and nothing is renormalised frame
to frame. Every frame therefore sits lower on the ramp than the brightest one,
and the walk down the ramp IS the decay.
  --decades defaults to the run's own fall plus 0.5 of a decade of headroom
  (1.87 on the run above), because a ramp much longer than the decay wastes
  most of itself: a fixed 2.5 was tried first and gave only 1.31x of dimming
  across the whole run against 1.96x once the ramp was sized to it. The chosen
  value is printed and is written on every frame -- a scale that adapts to its
  input cannot be compared between runs unless you can see what it picked, so
  PASS --decades EXPLICITLY WHEN COMPARING TWO RUNS.

MEASURED on that run, over the panel, frame 0 -> frame 128:

    mean luminance        115.3 -> 58.9      1.96x
    mean lit colour   (189,146, 91) -> (41, 86,137)    amber -> deep blue
    lit area fraction     0.735 -> 0.703     1.04x

THE DECAY READS AS HUE FIRST AND BRIGHTNESS SECOND, which is worth knowing
before anyone "improves" the ramp into a single hue's lightness ladder and
destroys most of the signal. The lit AREA barely moves, so do not expect the
sphere to empty out: it changes colour, from sheet cores at the top of the ramp
to a field an order of magnitude below them.
  What a log scale SHOWS: the geometry of the sheets over the ramp's decades of
  amplitude, and the decay, as a walk down the ramp.
  What it does NOT show: relative amplitude by eye. A voxel that looks half as
  bright as another is a DECADE weaker, not half as strong. Read the strip and
  the caption for numbers; do not read amplitude ratios off the picture.

--scale frame is per-frame P99.5 normalisation, kept because it is the right
tool for one question only -- how the sheet GEOMETRY evolves once amplitude is
divided out. It is doc/fig/mhd_anim.py's --pernorm and carries that script's
warning: do not use it where the amplitude is the result.
--scale linear is the fixed linear control. It is the honest one and it does go
dark; that is the point of keeping it.

AND WHATEVER THE SCHEME, THE AMPLITUDE IS ALSO DRAWN AS A NUMBER. Every frame
carries a decay strip -- log10 of E_u and E_b from meta.txt and of this frame's
own |J| P99.5, with a marker at the current frame -- plus the frame's peak and
its ratio to vref printed in the caption. No choice of colour map can make a
falling curve look flat, so the viewer can always tell the flow is decaying
even if the transfer function flatters it.

ROBUST AMPLITUDE, NOT THE MAX. The scale statistic is the 99.5th percentile of
|J| inside the sphere, estimated from every 7th voxel of each span (7 is coprime
with the lattice so the sample does not alias onto it). Current sheets are
extremely localised -- doc/fig/mkpng.py records a 2-D peak of 46.4 against a
99th percentile of 22.7 -- so normalising by the max leaves the field pale and
hides the structure the render exists to show. The clipping is stated on the
frame.

--------------------------------------------------------------------------------
COST, MEASURED, BECAUSE THE CAMERA ROTATES ONLY IF IT IS AFFORDABLE
--------------------------------------------------------------------------------
On this machine (CPython 3.14, one core), default --vscale 4 with its 4x4
splat, whole pipeline including BOTH reads of every volume, the PNG encode and
the read-back check:

    N = 64, 129 frames, REAL RUN, 70320 voxels in sphere  0.09 s/frame, 11.1 s
    N = 64,  24 frames, synthetic                         0.08 s/frame
    N = 64,  --vscale 2 (2x2 splat), synthetic            0.04 s/frame
    N = 128,  6 frames, synthetic, 562104 voxels          0.39 s/frame

so the camera YAWS (--yawrate, default 3 deg/frame) rather than falling back to
three fixed orthogonal panels: at N = 64 a rotating frame costs 0.08 s, which is
roughly twenty times under the one-to-two second budget that would have forced
the static compromise. Cost is O(voxels inside the sphere) x the splat area and
is independent of the output size, because the projection is a forward splat and
not a ray march -- N = 128 is 8x the voxels and 4.9x the time. If a much larger
volume ever needs this, drop --vscale to 2 before dropping the rotation.

--------------------------------------------------------------------------------
THE FILE FORMAT, AND ONE DISAGREEMENT WITH THE BRIEF
--------------------------------------------------------------------------------
validation/FieldDump.hpp::scalar_volume() writes THREE int32 -- nx, ny, nz --
then nx*ny*nz float32 with x fastest, then y, then z, little-endian. It does NOT
write a 2-int header with ny*nz folded into the second field, and
doc/fig/vol3d.py's reader agrees ('<iii'). read_volume() below therefore tries
the 3-int layout first, VALIDATES IT AGAINST THE FILE SIZE, and only then falls
back to a 2-int (nx, ny*nz) header, so it reads either without being told which.
"""
import glob
import math
import os
import struct
import sys
import time
import zlib

# ------------------------------------------------------------------ PNG output
def write_png(path, w, h, rgb):
    """8-bit truecolour PNG. One IDAT, filter 0 on every scanline."""
    raw = b''.join(b'\x00' + bytes(rgb[y * w * 3:(y + 1) * w * 3]) for y in range(h))

    def chunk(typ, data):
        return (struct.pack('>I', len(data)) + typ + data
                + struct.pack('>I', zlib.crc32(typ + data) & 0xffffffff))

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'
                + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(bytes(raw), 6))
                + chunk(b'IEND', b''))


def read_png(path):
    """Independent parser, used by --selftest to prove write_png's output really
    is a PNG: it checks the signature, every chunk CRC, and the unfiltered
    scanline length. Returns (w, h, rows) with rows as bytes of RGB."""
    d = open(path, 'rb').read()
    if d[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('bad PNG signature')
    pos, idat, w = 8, b'', None
    while pos < len(d):
        ln = struct.unpack('>I', d[pos:pos + 4])[0]
        typ = d[pos + 4:pos + 8]
        dat = d[pos + 8:pos + 8 + ln]
        crc = struct.unpack('>I', d[pos + 8 + ln:pos + 12 + ln])[0]
        if crc != (zlib.crc32(typ + dat) & 0xffffffff):
            raise ValueError('CRC mismatch in chunk %s' % typ)
        if typ == b'IHDR':
            w, h, depth, ctype, comp, filt, ilace = struct.unpack('>IIBBBBB', dat)
            if (depth, ctype, comp, filt, ilace) != (8, 2, 0, 0, 0):
                raise ValueError('unexpected IHDR %r' % (dat,))
        elif typ == b'IDAT':
            idat += dat
        elif typ == b'IEND':
            break
        pos += 12 + ln
    if w is None:
        raise ValueError('no IHDR')
    raw, stride, rows = zlib.decompress(idat), w * 3, []
    if len(raw) != h * (stride + 1):
        raise ValueError('IDAT length %d, expected %d' % (len(raw), h * (stride + 1)))
    for y in range(h):
        o = y * (stride + 1)
        if raw[o] != 0:
            raise ValueError('row %d uses filter %d, expected 0' % (y, raw[o]))
        rows.append(raw[o + 1:o + 1 + stride])
    return w, h, rows


# ----------------------------------------------------------------- dump readers
def read_volume(path):
    """FieldDump.hpp::scalar_volume -> (nx, ny, nz, values, x fastest).

    Tries the 3-int header the header file actually writes, validated against
    the file size; falls back to a 2-int (nx, ny*nz) header if that fails."""
    d = open(path, 'rb').read()
    if len(d) >= 12:
        nx, ny, nz = struct.unpack('<iii', d[:12])
        if (0 < nx and 0 < ny and 0 < nz
                and len(d) == 12 + 4 * nx * ny * nz):
            n = nx * ny * nz
            return nx, ny, nz, list(struct.unpack('<%df' % n, d[12:12 + 4 * n]))
    if len(d) >= 8:
        nx, nyz = struct.unpack('<ii', d[:8])
        if 0 < nx and 0 < nyz and len(d) == 8 + 4 * nx * nyz:
            # ny*nz folded into one field: assume a cube, which is what the
            # demonstrator writes (scalar_volume(p, N, N, N, jmag)).
            ny = int(round(math.sqrt(nyz)))
            if ny * ny != nyz:
                raise ValueError('%s: 2-int header %dx%d is not a cube; pass --ny'
                                 % (path, nx, nyz))
            n = nx * nyz
            return nx, ny, ny, list(struct.unpack('<%df' % n, d[8:8 + 4 * n]))
    raise ValueError('%s: %d bytes matches neither a 3-int nor a 2-int header'
                     % (path, len(d)))


def times_from_log(in_dir, nframes):
    """Frame times read from a driver log, for dumps whose meta.txt carries no
    per-frame rows.

    GPU/src/mhd_jet.cu writes the geometry into meta.txt but not the table --
    its times live in log.txt, which render_jet.py already reads for the
    slices. Without this the volume frames say 'frame index only' while the
    slice frames beside them are labelled, which on a case whose whole point is
    WHEN the current sheet goes three-dimensional is the one label you cannot
    do without.

    The dump stride is recovered by COUNTING, never assumed: the driver dumps
    every -dump-th probe and the log holds every probe, so 276 rows against 138
    frames gives 2. A log from a different run gives a stride that does not
    divide and the times come back None rather than wrong."""
    for cand in (os.path.join(in_dir, '..', 'log.txt'),
                 os.path.join(in_dir, 'log.txt')):
        if not os.path.exists(cand):
            continue
        rows = []
        for line in open(cand):
            f = line.split()
            if len(f) >= 10:
                try:
                    rows.append(float(f[0]))
                except ValueError:
                    pass
        if len(rows) < nframes:
            continue
        stride = max(1, int(round((len(rows) - 1) / float(max(1, nframes - 1)))))
        return [rows[min(k * stride, len(rows) - 1)] for k in range(nframes)], cand
    return None, None


def read_meta(path):
    """meta.txt: 'N <int>', 'R <float>', 'frames <int>', and per frame
    'frame <i> <t/Te> <E_u> <E_b>'. Absent or partial is not fatal -- every
    consumer below has a fallback -- because the renderer must work on a dump
    made before the driver learned to write it."""
    m = {'N': None, 'R': None, 'Te': None, 'frames': None, 'rows': {},
         'nx': None, 'ny': None, 'nz': None,
         'vstride': None, 'vnx': None, 'vny': None, 'vnz': None}
    if not os.path.exists(path):
        return m
    for line in open(path):
        p = line.split()
        if not p or p[0].startswith('#'):
            continue
        try:
            if p[0] == 'N' and len(p) > 1:
                m['N'] = int(float(p[1]))
            elif p[0] == 'R' and len(p) > 1:
                m['R'] = float(p[1])
            elif p[0] == 'Te' and len(p) > 1:
                m['Te'] = float(p[1])
            elif p[0] == 'frames' and len(p) > 1:
                m['frames'] = int(float(p[1]))
            elif p[0] in ('nx', 'ny', 'nz', 'vstride', 'vnx', 'vny', 'vnz') \
                    and len(p) > 1:
                m[p[0]] = int(float(p[1]))
            elif p[0] == 'frame' and len(p) >= 5:
                m['rows'][int(p[1])] = (float(p[2]), float(p[3]), float(p[4]))
        except ValueError:
            continue
    return m


# ------------------------------------------------------------------- 5x7 font
# Column-major, bit 0 = top row. ASCII 32..126. drawtext is missing from this
# ffmpeg build, so every character on every frame comes from here.
_F = (
    "0000000000" "00005F0000" "0007000700" "147F147F14" "242A7F2A12"  # sp ! " # $
    "2313086462" "3649552250" "0000050300" "001C224100" "0041221C00"  # % & ' ( )
    "14083E0814" "08083E0808" "0000503000" "0808080808" "0000606000"  # * + , - .
    "2010080402" "3E5149453E" "00427F4000" "4261514946" "2141454B31"  # / 0 1 2 3
    "1814127F10" "2745454539" "3C4A494930" "0171090503" "3649494936"  # 4 5 6 7 8
    "064949291E" "0036360000" "0056360000" "0814224100" "1414141414"  # 9 : ; < =
    "0041221408" "0201510906" "324979413E" "7E1111117E" "7F49494936"  # > ? @ A B
    "3E41414122" "7F4141221C" "7F49494941" "7F09090901" "3E4149497A"  # C D E F G
    "7F0808087F" "00417F4100" "2040413F01" "7F08142241" "7F40404040"  # H I J K L
    "7F020C027F" "7F0408107F" "3E4141413E" "7F09090906" "3E4151215E"  # M N O P Q
    "7F09192946" "4649494931" "01017F0101" "3F4040403F" "1F2040201F"  # R S T U V
    "3F4038403F" "6314081463" "0708700807" "6151494543" "007F414100"  # W X Y Z [
    "0204081020" "0041417F00" "0402010204" "4040404040" "0001020400"  # \ ] ^ _ `
    "2054545478" "7F48444438" "3844444420" "384444487F" "3854545418"  # a b c d e
    "087E090102" "0C5252523E" "7F08040478" "00447D4000" "204044 3D00" # f g h i j
    "7F10284400" "00417F4000" "7C04180478" "7C08040478" "3844444438"  # k l m n o
    "7C14141408" "081414187C" "7C08040408" "4854545420" "043F444020"  # p q r s t
    "3C4040207C" "1C2040201C" "3C4030403C" "4428102844" "0C5050503C"  # u v w x y
    "4464544C44" "0008364100" "00007F0000" "0041360800" "0804081008"  # z { | } ~
)
_F = _F.replace(' ', '')
assert len(_F) == 95 * 10, 'font table is %d chars, expected %d' % (len(_F), 95 * 10)
FONT = {chr(32 + _i): [int(_F[10 * _i + 2 * _k:10 * _i + 2 * _k + 2], 16)
                       for _k in range(5)] for _i in range(95)}


def text_w(s, sc=1):
    return len(s) * 6 * sc


def draw_text(buf, W, H, x0, y0, s, col, sc=1):
    r, g, b = col
    for ch in s:
        cols = FONT.get(ch, FONT['?'])
        for cx in range(5):
            bits = cols[cx]
            if bits:
                for cy in range(7):
                    if bits >> cy & 1:
                        px, py = x0 + cx * sc, y0 + cy * sc
                        for dy in range(sc):
                            yy = py + dy
                            if 0 <= yy < H:
                                o = (yy * W + px) * 3
                                for dx in range(sc):
                                    if 0 <= px + dx < W:
                                        i = o + dx * 3
                                        buf[i], buf[i + 1], buf[i + 2] = r, g, b
        x0 += 6 * sc
    return x0


# ---------------------------------------------------------------- colour ramp
# doc/fig/vol3d.py's emissive ramp, kept identical so that a still from this
# film and a vol3d.py figure of the same field are directly comparable: deep
# indigo where the current is weak, through teal and amber to near-white in the
# sheet cores, against a dark ground because an emissive volume composited over
# white washes out and no transfer function recovers it.
RAMPS = {}
RAMPS['emissive'] = [
    (0.00, (38, 44, 96)), (0.18, (52, 104, 176)), (0.38, (48, 168, 198)),
    (0.58, (110, 214, 168)), (0.74, (232, 206, 112)), (0.88, (250, 160, 88)),
    (1.00, (255, 248, 236))]

# Matplotlib's inferno, sampled at nine stops. IT SUITS A MIP BETTER THAN THE
# RAMP ABOVE, for the reason the foot comment below spends a paragraph working
# around: inferno STARTS at (0, 0, 4), which is already the ground, so a weak
# voxel contributes nothing without any fade being applied to it. The emissive
# ramp starts at a visible indigo and needs its bottom fifth faded by hand or
# the silhouette fills in solid.
#
# The cost is that inferno carries the low end in BRIGHTNESS where the emissive
# ramp carries it in HUE, so a decaying field reads as dimming rather than as
# a colour change, and two frames a factor of ten apart are harder to tell
# apart at a glance. Which is the right trade depends on whether the film is
# about the decay or about the structure at one instant.
RAMPS['inferno'] = [
    (0.000, (0, 0, 4)),     (0.125, (31, 12, 72)),   (0.250, (85, 15, 109)),
    (0.375, (136, 34, 106)), (0.500, (186, 54, 85)),  (0.625, (227, 89, 51)),
    (0.750, (249, 140, 10)), (0.875, (249, 201, 50)), (1.000, (252, 255, 164))]

RAMP = RAMPS['emissive']
BG = (10, 11, 18)          # ground / outside the sphere
RING = (74, 82, 104)       # sphere silhouette
FG = (206, 212, 226)
DIM = (128, 136, 156)


def _emit(t):
    if t <= 0.0:
        return RAMP[0][1]
    if t >= 1.0:
        return RAMP[-1][1]
    for i in range(len(RAMP) - 1):
        a, b = RAMP[i], RAMP[i + 1]
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0])
            return tuple(int(round(a[1][k] + f * (b[1][k] - a[1][k]))) for k in range(3))
    return RAMP[-1][1]


NLUT = 256
NDEP = 16          # depth quantisation for the shading table

# OPACITY, WHICH MIP DOES NOT GIVE YOU FOR FREE. vol3d.py can start its ramp at
# a visible indigo because it is an emission-ABSORPTION composite: a weak voxel
# contributes little there because its alpha is low. A maximum-intensity
# projection has no alpha at all, so the same ramp paints the whole sphere a
# solid indigo disc and the silhouette stops meaning anything. The bottom fifth
# of the scale is therefore faded into the ground here.
#
# WHAT IT IS WORTH IS SMALL, AND WAS MEASURED RATHER THAN ASSUMED. On the
# synthetic decay in the scratchpad (48.7x fall in P99.5, log scale, 2.5
# decades, one variable changed) the panel dims 1.74x with AFOOT off and 1.76x
# with it on, and the lit fraction moves 1.03x against 1.04x -- i.e. the foot
# contributes essentially NOTHING to how the decay reads. It is kept for the
# RAMP's base colour, not for the decay, and the distinction matters: a test
# field whose own noise floor scales with the amplitude (as that one's does)
# keeps every voxel above the log floor and so cannot exercise the foot at all.
# Expect it to earn more on a real field, whose quiet regions do fall through
# the floor -- but that is a prediction, not a measurement.
#
# THE DECAY IS CARRIED BY HUE, NOT BY BRIGHTNESS, and that is the number to
# quote: over the same 48.7x fall the mean lit colour goes (205, 196, 182) --
# near-white amber, the sheet cores clipping the top of the ramp -- to
# (51, 138, 157), a plain teal, while mean luminance falls only 1.76x. A viewer
# reads the colour change long before the brightness change, which is why the
# ramp runs through hue and not through a single hue's lightness.
AFOOT = 0.20       # fraction of the scale over which opacity rises 0 -> 1


def _shade(t, dfar):
    """Emission at level t, faded out at the bottom of the scale and dimmed with
    depth, composited over the ground. dfar is 0 at the near pole, 1 at the far."""
    a = t / AFOOT if t < AFOOT else 1.0
    a *= 1.0 - 0.38 * dfar            # the depth CUE: a cue, never an amplitude
    c = _emit(t)
    return tuple(int(round(BG[k] + (c[k] - BG[k]) * a)) for k in range(3))


# SHADE[d * NLUT + t] -> the final RGB. Precomputed once, so the inner blit does
# two integer indexes and no arithmetic; it is why the depth cue is free.
def build_shade():
    """Rebuild the two lookup tables from whatever RAMP currently is. They are
    built at import for the default and again after the arguments are parsed,
    because --ramp chooses between ramps and _shade() closes over RAMP through
    _emit() rather than taking it as an argument."""
    global SHADE, SHADE_FLAT
    SHADE = [_shade(t / (NLUT - 1.0), d / (NDEP - 1.0))
             for d in range(NDEP) for t in range(NLUT)]
    SHADE_FLAT = [_shade(t / (NLUT - 1.0), 0.0) for t in range(NLUT)]


def set_ramp(name):
    global RAMP
    if name not in RAMPS:
        return False
    RAMP = RAMPS[name]
    build_shade()
    return True


SHADE = SHADE_FLAT = None
build_shade()


# ------------------------------------------------------------- sphere geometry
def sphere_spans(nx, ny, nz, R):
    """Rows of x inside the sphere, as (z, y, x0, x1, base). This IS the mask:
    nothing outside it is read, so the penalised exterior cannot reach either
    the colour scale or a pixel."""
    # R <= 0 IS THE PERIODIC-BOX SENTINEL (GPU/src/orszag_tang.cu): there is no
    # sphere, so every row of x is in play and the whole cube is rendered. The
    # ring is suppressed alongside it -- see where R is read in main().
    if R is not None and R <= 0:
        return [(z, y, 0, nx - 1, (z * ny + y) * nx)
                for z in range(nz) for y in range(ny)]
    cx, cy, cz = 0.5 * (nx - 1), 0.5 * (ny - 1), 0.5 * (nz - 1)
    R2, out = R * R, []
    for z in range(nz):
        dz = z - cz
        t1 = R2 - dz * dz
        if t1 < 0.0:
            continue
        for y in range(ny):
            dy = y - cy
            t2 = t1 - dy * dy
            if t2 < 0.0:
                continue
            xr = math.sqrt(t2)
            x0 = max(0, int(math.ceil(cx - xr)))
            x1 = min(nx - 1, int(math.floor(cx + xr)))
            if x1 >= x0:
                out.append((z, y, x0, x1, (z * ny + y) * nx))
    return out


def span_stats(vol, spans, stride=7):
    """(peak, p995) of |J| inside the sphere. p995 from every `stride`-th voxel:
    a full sort of 64^3 costs ~0.3 s per frame in pure Python and the percentile
    does not need it. stride 7 is coprime with the usual N so the sample does
    not land on a lattice sub-grid."""
    peak, samp = 0.0, []
    for _z, _y, x0, x1, base in spans:
        row = vol[base + x0:base + x1 + 1]
        m = max(row)
        if m > peak:
            peak = m
        samp.extend(row[::stride])
    if not samp:
        return 0.0, 0.0
    samp.sort()
    return peak, samp[min(len(samp) - 1, int(0.995 * len(samp)))]


# -------------------------------------------------------------------- the MIP
def mip_size(R, sc, ns=2):
    """Screen buffer for an orthographic MIP of a sphere of radius R at `sc`
    samples per voxel with an ns x ns splat. A sphere projects to the same
    circle at every yaw, so ONE size serves the whole rotation and no frame can
    clip -- which is also why the inner loop needs no bounds test: the margin
    below is wide enough for the splat at the extreme of the silhouette, so the
    splat can never run off the buffer and wrap onto the opposite edge."""
    n = int(math.ceil(2.0 * R * sc)) + 2 * (ns + 2)
    return n, n


def mip_size_box(nx, ny, nz, elev, spin, sc, ns=2):
    """Screen buffer for an orthographic MIP of a BOX -- tight, not merely safe.

    mip_size() above sizes from a radius, which for a box means its
    circumscribing sphere: correct, and wasteful. At 128 x 184 x 128 spun about
    y at 22 deg that is a 1045^2 buffer for a silhouette which never exceeds
    737 x 966, so a third of the width is permanently black and the panel
    floats in it.

    The bound here is EXACT rather than generous. Let (a, b) be the half extents
    in the plane the camera orbits and c the half extent along the spin axis.
    Screen right is a unit vector in the orbit plane, so the widest the box can
    project is max over yaw of a|sin| + b|cos| = sqrt(a^2 + b^2). Screen up
    tilts out of that plane by the elevation, giving sin(e) sqrt(a^2 + b^2) +
    cos(e) c. Both maxima are ATTAINED at some yaw, so one size still serves the
    whole rotation and no frame can clip -- which is what the inner loop's
    missing bounds test depends on, and the reason this must be a bound and not
    an estimate.

    Third return is the depth half-range, the same construction against the
    into-screen axis: cos(e) sqrt(a^2 + b^2) + sin(e) c. The circumscribing
    radius overstates it by 9 % here, which flattens the depth cue."""
    hx, hy, hz = 0.5 * nx, 0.5 * ny, 0.5 * nz
    a, b, c = (hx, hz, hy) if spin == 'y' else (hx, hy, hz)
    d = math.sqrt(a * a + b * b)
    se = abs(math.sin(math.radians(elev)))
    ce = abs(math.cos(math.radians(elev)))
    pad = 2 * (ns + 2)
    return (int(math.ceil(2.0 * d * sc)) + pad,
            int(math.ceil(2.0 * (se * d + ce * c) * sc)) + pad,
            ce * d + se * c)


def _project_inner(vol, spans, val, dep, W0, H0, OX, OY, floor,
                   pax, pay, paz, pbx, pby, pbz, fx, fy, fz, cx, cy, cz, ns=2):
    """The inner loop, and the only thing in this file whose cost scales with
    the volume. Per voxel it is one compare against `floor`, two int()s and an
    ns x ns max-update; the screen position is carried along the x run by three
    additions rather than recomputed, which is what a separable orthographic
    projection buys.

    ns IS THE SPLAT WIDTH AND IT MUST COVER THE SAMPLE SPACING. Projected
    neighbouring voxels are at most `sc` pixels apart, so ns = ceil(sc) leaves
    no holes at any yaw. Dropping to ns = 1 and repairing afterwards with
    fill_holes() is cheaper but only equivalent at sc <= 2; above that the
    dilate cannot reach across a 3-pixel gap.

    `floor` is not an independent clip: it is the amplitude below which the
    transfer function returns index 0, so skipping those voxels cannot change a
    single pixel. It is what makes the late, faint frames the FASTEST to render
    rather than the slowest."""
    if ns <= 1:
        for z, y, x0, x1, base in spans:
            pxf = OX + pax * (x0 - cx) + pay * (y - cy) + paz * (z - cz)
            pyf = OY + pbx * (x0 - cx) + pby * (y - cy) + pbz * (z - cz)
            dpf = fx * (x0 - cx) + fy * (y - cy) + fz * (z - cz)
            for v in vol[base + x0:base + x1 + 1]:
                if v > floor:
                    i = int(pyf) * W0 + int(pxf)
                    if v > val[i]:
                        val[i] = v
                        dep[i] = dpf
                pxf += pax
                pyf += pbx
                dpf += fx
        return val, dep, W0, H0
    if ns == 2:                                   # the default, unrolled
        for z, y, x0, x1, base in spans:
            pxf = OX + pax * (x0 - cx) + pay * (y - cy) + paz * (z - cz)
            pyf = OY + pbx * (x0 - cx) + pby * (y - cy) + pbz * (z - cz)
            dpf = fx * (x0 - cx) + fy * (y - cy) + fz * (z - cz)
            for v in vol[base + x0:base + x1 + 1]:
                if v > floor:
                    i = int(pyf) * W0 + int(pxf)
                    if v > val[i]:
                        val[i] = v
                        dep[i] = dpf
                    i += 1
                    if v > val[i]:
                        val[i] = v
                        dep[i] = dpf
                    i += W0
                    if v > val[i]:
                        val[i] = v
                        dep[i] = dpf
                    i -= 1
                    if v > val[i]:
                        val[i] = v
                        dep[i] = dpf
                pxf += pax
                pyf += pbx
                dpf += fx
        return val, dep, W0, H0
    offs = [dy * W0 + dx for dy in range(ns) for dx in range(ns)]
    for z, y, x0, x1, base in spans:
        pxf = OX + pax * (x0 - cx) + pay * (y - cy) + paz * (z - cz)
        pyf = OY + pbx * (x0 - cx) + pby * (y - cy) + pbz * (z - cz)
        dpf = fx * (x0 - cx) + fy * (y - cy) + fz * (z - cz)
        for v in vol[base + x0:base + x1 + 1]:
            if v > floor:
                b = int(pyf) * W0 + int(pxf)
                for o in offs:
                    i = b + o
                    if v > val[i]:
                        val[i] = v
                        dep[i] = dpf
            pxf += pax
            pyf += pbx
            dpf += fx
    return val, dep, W0, H0


def fill_holes(val, dep, W0, H0):
    """One max-dilate into empty pixels only. Equivalent to having splatted each
    voxel over a 2x2 block, but paid per BUFFER pixel (~12k) instead of per
    VOXEL (~70k), and it cannot dim anything that was already hit."""
    src = list(val)
    for y in range(1, H0 - 1):
        r = y * W0
        for x in range(1, W0 - 1):
            i = r + x
            if src[i] != 0.0:
                continue
            best, bd = 0.0, 0.0
            for j in (i - W0 - 1, i - W0, i - W0 + 1, i - 1, i + 1,
                      i + W0 - 1, i + W0, i + W0 + 1):
                s = src[j]
                if s > best:
                    best, bd = s, dep[j]
            if best > 0.0:
                val[i] = best
                dep[i] = bd


def smooth3(val, W0, H0):
    """Separable 1-2-1 tent, once. The MIP is computed at `--vscale` samples per
    voxel and then magnified by an integer factor for display; without this the
    magnification shows the sample lattice rather than the field. It is a
    DISPLAY filter: it lowers the peak of a one-voxel spike slightly, so the
    numbers in the caption come from the raw volume, never from this buffer."""
    tmp = [0.0] * (W0 * H0)
    for y in range(H0):
        r = y * W0
        tmp[r] = val[r]
        tmp[r + W0 - 1] = val[r + W0 - 1]
        for x in range(1, W0 - 1):
            i = r + x
            tmp[i] = 0.25 * val[i - 1] + 0.5 * val[i] + 0.25 * val[i + 1]
    for x in range(W0):
        val[x] = tmp[x]
        val[(H0 - 1) * W0 + x] = tmp[(H0 - 1) * W0 + x]
    for y in range(1, H0 - 1):
        r = y * W0
        for x in range(W0):
            i = r + x
            val[i] = 0.25 * tmp[i - W0] + 0.5 * tmp[i] + 0.25 * tmp[i + W0]


# ---------------------------------------------------------------- the transfer
def make_transfer(mode, vref, vframe, decades):
    """value -> LUT index. Returns (fn, floor, label). `floor` is the amplitude
    below which a voxel cannot brighten any pixel, so the projector can skip it;
    it is derived from the transfer function itself and is therefore not an
    independent clip."""
    if mode == 'frame':
        s = vframe if vframe > 0.0 else 1.0
        lab = 'per-frame linear, P99.5 = %.3e (STRUCTURE ONLY: amplitude divided out)' % s
        inv = (NLUT - 1) / s
        return (lambda v: int(v * inv)), 0.02 * s, lab
    if mode == 'linear':
        s = vref if vref > 0.0 else 1.0
        lab = 'fixed linear, vref = %.3e (global P99.5) -- goes dark, and that is the decay' % s
        inv = (NLUT - 1) / s
        return (lambda v: int(v * inv)), 0.02 * s, lab
    s = vref if vref > 0.0 else 1.0
    lo = s * (10.0 ** -decades)
    k = (NLUT - 1) / decades
    lg = math.log10
    lab = 'fixed log, %.1f decades below vref = %.3e (global P99.5)' % (decades, s)
    return (lambda v: int((lg(v / lo)) * k) if v > lo else 0), lo, lab


# ------------------------------------------------------------------ the canvas
def blit_panel(buf, W, H, px, py, val, dep, W0, H0, tf, R, mag, depth_cue):
    """Colourise the low-res MIP and magnify it by the integer factor `mag` with
    nearest-neighbour sampling, which the brief permits and which costs nothing;
    smooth3() above is what keeps that from looking like a lattice. The colour
    itself is two table lookups -- see SHADE."""
    nlut = NLUT - 1
    dscale = (NDEP - 1) / (2.0 * R) if R > 0 else 0.0
    half = 0.5 * (NDEP - 1)
    flat = SHADE_FLAT
    tab = SHADE
    for y0 in range(H0):
        r0 = y0 * W0
        by = py + y0 * mag
        if by < 0 or by + mag > H:
            continue
        for x0 in range(W0):
            v = val[r0 + x0]
            if v <= 0.0:
                continue
            t = tf(v)
            if t < 0:
                t = 0
            elif t > nlut:
                t = nlut
            if depth_cue:
                d = int(dep[r0 + x0] * dscale + half)
                if d < 0:
                    d = 0
                elif d >= NDEP:
                    d = NDEP - 1
                cr, cg, cb = tab[d * NLUT + t]
            else:
                cr, cg, cb = flat[t]
            bx = px + x0 * mag
            for dy in range(mag):
                o = ((by + dy) * W + bx) * 3
                for dx in range(mag):
                    j = o + dx * 3
                    buf[j] = cr
                    buf[j + 1] = cg
                    buf[j + 2] = cb


def draw_ring(buf, W, H, ccx, ccy, rad, col, alpha=1.0):
    """The sphere silhouette. Orthographic + a sphere = a circle of radius R at
    every yaw, so this is exact and is the only fixed reference the rotating
    view has. Drawn under the field, faintly, so it frames rather than competes."""
    n = max(64, int(6.28 * rad))
    for k in range(n):
        a = 6.283185307179586 * k / n
        x = int(round(ccx + rad * math.cos(a)))
        y = int(round(ccy + rad * math.sin(a)))
        if 0 <= x < W and 0 <= y < H:
            i = (y * W + x) * 3
            for c in range(3):
                buf[i + c] = int(buf[i + c] * (1 - alpha) + col[c] * alpha)


def hline(buf, W, H, x0, x1, y, col):
    if not 0 <= y < H:
        return
    for x in range(max(0, x0), min(W, x1)):
        i = (y * W + x) * 3
        buf[i], buf[i + 1], buf[i + 2] = col


def vline(buf, W, H, x, y0, y1, col):
    if not 0 <= x < W:
        return
    for y in range(max(0, y0), min(H, y1)):
        i = (y * W + x) * 3
        buf[i], buf[i + 1], buf[i + 2] = col


def dot(buf, W, H, x, y, col, r=0):
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < W and 0 <= yy < H:
                i = (yy * W + xx) * 3
                buf[i], buf[i + 1], buf[i + 2] = col


SERIES_COL = [(232, 206, 112), (110, 190, 232), (238, 128, 108)]


def draw_strip(buf, W, H, x0, y0, sw, sh, series, cur):
    """log10 amplitude against frame, with a marker at the current frame.

    THIS IS WHAT STOPS THE COLOUR MAP FROM LYING. Every transfer function above
    trades amplitude for structure to some degree; the curves here are the
    amplitude, undivided, on a log axis, so a viewer can always read the decay
    off the frame even under --scale frame, which shows none of it in the image.
    """
    hline(buf, W, H, x0, x0 + sw, y0 + sh, (52, 58, 74))
    vline(buf, W, H, x0, y0, y0 + sh + 1, (52, 58, 74))
    lo, hi = None, None
    for _n, vals, _c in series:
        for v in vals:
            if v is not None and v > 0.0:
                l = math.log10(v)
                lo = l if lo is None or l < lo else lo
                hi = l if hi is None or l > hi else hi
    if lo is None or hi - lo < 1e-9:
        lo, hi = (0.0, 1.0) if lo is None else (lo - 0.5, lo + 0.5)
    pad = 0.06 * (hi - lo)
    lo -= pad
    hi += pad
    nf = max(1, max(len(v) for _n, v, _c in series) - 1)
    # THE MARKER GOES DOWN FIRST, so that no piece of chrome can cover a data
    # point: the curves are the honest record of the amplitude and must be the
    # topmost thing in the strip.
    if 0 <= cur <= nf:
        mx = x0 + int(round(sw * cur / nf))
        for y in range(y0, y0 + sh + 1):
            if 0 <= y < H and (y - y0) % 3 != 2:
                i = (y * W + mx) * 3
                buf[i], buf[i + 1], buf[i + 2] = (236, 240, 250)
    lx = x0 + 4
    for si, (name, vals, col) in enumerate(series):
        for i, v in enumerate(vals):
            if v is None or v <= 0.0:
                continue
            px = x0 + int(round(sw * i / nf))
            py = y0 + sh - int(round(sh * (math.log10(v) - lo) / (hi - lo)))
            dot(buf, W, H, px, py, col, 0)
        lx = draw_text(buf, W, H, lx, y0 - 9, name, col, 1) + 5
    draw_text(buf, W, H, x0 + sw - 6 * 10, y0 - 9, 'log10 amp', DIM, 1)
    draw_text(buf, W, H, x0 + 3, y0 + 1, '%+.2f' % hi, DIM, 1)          # top of axis
    draw_text(buf, W, H, x0 + 3, y0 + sh - 8, '%+.2f' % lo, DIM, 1)     # bottom


# ------------------------------------------------------------------------ main
USAGE = __doc__


def die(msg):
    sys.stderr.write('render_volume.py: %s\n' % msg)
    sys.exit(1)


def selftest(tmp):
    """Write a PNG and read it straight back with this file's own parser."""
    w, h = 37, 23
    rgb = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            rgb[i], rgb[i + 1], rgb[i + 2] = (x * 7) % 256, (y * 11) % 256, (x + y) % 256
    p = os.path.join(tmp, '_selftest.png')
    write_png(p, w, h, rgb)
    rw, rh, rows = read_png(p)
    assert (rw, rh) == (w, h), 'dimensions %dx%d != %dx%d' % (rw, rh, w, h)
    for y in range(h):
        assert rows[y] == bytes(rgb[y * w * 3:(y + 1) * w * 3]), 'row %d differs' % y
    sz = os.path.getsize(p)
    print('selftest: %dx%d PNG, %d bytes, CRCs ok, pixels round-trip exactly' % (w, h, sz))
    os.remove(p)


def main(argv):
    a = [x for x in argv[1:]]
    if '-h' in a or '--help' in a:
        print(USAGE)
        return 0

    # 'decades': 0 means "size the log ramp to this run's own decay" -- see the
    # block in main() that sets it, and pass --decades N to pin it instead.
    opt = {'title': '', 'spin': 'z', 'ramp': 'emissive', 'cropy': 0,
           'scale': 'log', 'decades': 0.0, 'yawrate': 3.0, 'yaw0': 35.0,
           'elev': 22.0, 'size': 440, 'vscale': 4.0, 'fps': 12, 'every': 1,
           'limit': 0, 'R': 0.0, 'nodepth': False, 'nosmooth': False,
           'png': '', 'selftest': False, 'quiet': False}
    pos = []
    i = 0
    while i < len(a):
        s = a[i]
        if s == '--selftest':
            opt['selftest'] = True
        elif s == '--nodepth':
            opt['nodepth'] = True
        elif s == '--nosmooth':
            opt['nosmooth'] = True
        elif s == '--quiet':
            opt['quiet'] = True
        elif s.startswith('--'):
            k = s[2:]
            if k not in opt:
                die('unknown flag --%s (see --help)' % k)
            if i + 1 >= len(a):
                die('--%s needs a value' % k)
            i += 1
            opt[k] = type(opt[k])(a[i]) if not isinstance(opt[k], bool) else a[i]
        else:
            pos.append(s)
        i += 1

    if not set_ramp(opt['ramp']):
        die('unknown --ramp %s (have %s)' % (opt['ramp'], ', '.join(sorted(RAMPS))))

    here = os.path.dirname(os.path.abspath(__file__))
    in_dir = pos[0] if len(pos) > 0 else here
    out_mp4 = pos[1] if len(pos) > 1 else os.path.join(in_dir, 'mhd_sphere_jvol.mp4')
    png_dir = opt['png'] or os.path.join(in_dir, 'png')

    if opt['selftest']:
        os.makedirs(png_dir, exist_ok=True)
        selftest(png_dir)

    # WHERE THE DUMPS ARE IS NOT FIXED, so look rather than assume.
    # mhd_sphere.cpp writes results/N_mhd_sphere/anim_frames/ (and meta.txt
    # INSIDE it, not beside it) -- `anim_*` is the tree's ignored-dump
    # convention, .gitignore:34. An earlier spelling was frames/. Both are
    # accepted, and so is being pointed straight at a dump directory.
    dump_dir, files = None, []
    for cand in (os.path.join(in_dir, 'anim_frames'), os.path.join(in_dir, 'frames'),
                 in_dir):
        g = sorted(glob.glob(os.path.join(cand, 'jvol_*.raw')))
        if g:
            dump_dir, files = cand, g
            break
    if not files:
        die('no jvol_*.raw in %s/anim_frames, %s/frames or %s itself'
            ' -- run demonstrator/mhd_sphere with -dumpevery first'
            % (in_dir, in_dir, in_dir))
    if opt['every'] > 1:
        files = files[::opt['every']]
    if opt['limit'] > 0:
        files = files[:opt['limit']]

    meta = read_meta(os.path.join(dump_dir, 'meta.txt'))
    if meta['N'] is None:
        meta = read_meta(os.path.join(in_dir, 'meta.txt'))
    nx, ny, nz, vol = read_volume(files[0])
    # meta['R'] == 0 is the PERIODIC-BOX sentinel and is FALSY, so it must not go
    # through an `or` chain -- that silently falls back to 0.40 N and renders a
    # box as a sphere, cutting 48 % of the cube away without a word. Resolve it
    # explicitly, and keep the implausible-radius guard for real spheres only.
    if opt['R']:
        R = opt['R']
    elif meta['R'] is not None:
        R = meta['R']                         # may be 0.0 -> periodic box
    else:
        R = 0.40 * nx                         # rfac default in mhd_sphere.cpp
    box = (R <= 0)
    if not box and R > 0.5 * math.sqrt(3.0) * nx:
        die('implausible sphere radius R = %g for N = %d' % (R, nx))
    # A STRIDED VOLUME IS NOT A MISMATCH. GPU/src/orszag_tang.cu's -volstride S
    # writes the |J| volume reduced to (N/S)^3 by block maximum, because a full
    # frame at N = 288 is 95.6 MB and 244 of them is 23 GB. So meta's N and the
    # volume's own dimensions legitimately differ by an exact integer factor, and
    # reporting that as a discrepancy sends the reader looking for a bug. Keep the
    # warning for the case it was written for: dimensions that do NOT divide,
    # which really is a mixed-up directory.
    # THE STRIDE IS READ WHERE IT IS STATED AND ONLY INFERRED WHERE IT IS NOT.
    # The cube inference below -- meta's N divided by the volume's nx -- cannot
    # work on a box whose three extents are independent: 512 x 734 x 512 at
    # stride 4 dumps 128 x 184 x 128, and 734/184 is not 4. GPU/src/mhd_jet.cu
    # writes vstride and the three reduced extents outright for exactly that
    # reason, so those win when present and the inference is the fallback for
    # dumps made before the driver learned to say so.
    grid_N, vstride = nx, 1
    gnx, gny, gnz = nx, ny, nz
    if meta['vstride']:
        vstride = max(1, meta['vstride'])
        if meta['vnx'] and (meta['vnx'], meta['vny'], meta['vnz']) != (nx, ny, nz):
            sys.stderr.write('  note: meta says the volume is %dx%dx%d, the file '
                             'says %dx%dx%d; trusting the file\n'
                             % (meta['vnx'], meta['vny'], meta['vnz'], nx, ny, nz))
        gnx = meta['nx'] or nx * vstride
        gny = meta['ny'] or ny * vstride
        gnz = meta['nz'] or nz * vstride
        grid_N = gnx
        if not opt['quiet'] and vstride > 1:
            print('  volume is a stride-%d dump of %dx%dx%d, rendered at %dx%dx%d'
                  % (vstride, gnx, gny, gnz, nx, ny, nz))
    elif meta['N'] and meta['N'] != nx:
        if meta['N'] % nx == 0:
            grid_N, vstride = meta['N'], meta['N'] // nx
            gnx = gny = gnz = grid_N
            if not opt['quiet']:
                print('  volume is a stride-%d dump of a %d^3 grid, rendered at %d^3'
                      % (vstride, grid_N, nx))
        else:
            sys.stderr.write('  note: meta.txt says N = %d, volume says %d and does '
                             'NOT divide it; using %d\n' % (meta['N'], nx, nx))
    spans = sphere_spans(nx, ny, nz, R)

    # --cropy TRIMS THE BOX, NOT THE CANVAS, and the two are different fixes
    # for what looks like one problem. mip_size_box() is an exact bound on
    # where the box CAN project, so the black left inside it is not slack -- it
    # is box that is genuinely EMPTY. The jet is sech^2(x2), below 1e-4 of its
    # peak by |x2| = 5, so on a 20-unit box more than half the rows never carry
    # a current above the log floor, and no canvas arithmetic recovers that
    # space. render_jet.py crops the slices for the same reason and by default;
    # here it is opt-in, because the sphere and Orszag-Tang have nothing to
    # crop.
    #
    # IT HAPPENS BEFORE THE SCAN, NOT AFTER. vref is a PERCENTILE, so it
    # depends on how many voxels are in the sample, not only on their values:
    # P99.5 of a box that is half quiescent fluid is a lower number than P99.5
    # of the jet alone. Cropping after the scan would colour the frames on a
    # scale taken from voxels that are no longer in them.
    ny_eff = ny
    if box and opt['cropy'] > 0:
        half = min(ny // 2, int(opt['cropy']))
        lo, hi = ny // 2 - half, ny // 2 + half
        spans = [t for t in spans if lo <= t[1] < hi]
        ny_eff = 2 * half
        if not opt['quiet']:
            print('  cropped to |y - ny/2| < %d rows: %d of %d kept'
                  % (half, ny_eff, ny))
    nvox = sum(x1 - x0 + 1 for _z, _y, x0, x1, _b in spans)
    cx, cy, cz = 0.5 * (nx - 1), 0.5 * (ny - 1), 0.5 * (nz - 1)

    # ---- pass 1: amplitude of every frame, so the strip can be drawn on frame 0
    if not opt['quiet']:
        shape = ('%d^3' % nx) if nx == ny == nz else ('%dx%dx%d' % (nx, ny, nz))
        if box:
            print('  %d frames, %s volume, PERIODIC BOX (no sphere mask, no ring), '
                  '%d voxels' % (len(files), shape, nvox))
        else:
            print('  %d frames, %s volume, R = %.2f, %d voxels inside the sphere (%.1f%%)'
                  % (len(files), shape, R, nvox, 100.0 * nvox / (nx * ny * nz)))
    t0 = time.time()
    peaks, p995 = [], []
    for k, fn in enumerate(files):
        v = vol if k == 0 else read_volume(fn)[3]
        pk, p5 = span_stats(v, spans)
        peaks.append(pk)
        p995.append(p5)
    t_scan = time.time() - t0

    samp = sorted(x for x in p995 if x > 0.0)
    vref = samp[-1] if samp else 1.0
    vmin = samp[0] if samp else 1.0
    # vref IS THE LARGEST per-frame P99.5 IN THE RUN, i.e. the top of the ramp is
    # the brightest robust amplitude the flow ever reaches and NOTHING CLIPS.
    #
    # An earlier version took the 80th percentile of those per-frame values, on
    # doc/fig/mhd_anim.py's argument that a decaying run's first frame is the
    # brightest it will ever be so anchoring there buries the rest. THAT
    # ARGUMENT IS ABOUT A LINEAR SCALE AND IT IS WRONG HERE. On a log scale
    # anchoring at the top is exactly what you want -- every later frame simply
    # sits lower on the ramp -- whereas the 80th percentile clipped the first
    # fifth of the run flat against white and threw away that much of the range.
    # Measured on the real 129-frame N=64 run, which falls 18.0x in |J| P99.5:
    # the 80th-percentile anchor gave a mean panel luminance of 147.2 -> 112.3,
    # only 1.31x over the whole decay, with frames 0-26 indistinguishable.

    # DECADES, SIZED TO THE RUN RATHER THAN GUESSED. The point of the log scale
    # is that the decay walks down the ramp, so the ramp has to be about as long
    # as the decay: too long and the flow barely moves on it (the 2.5 fixed
    # default used only half the ramp on a 1.26-decade fall), too short and the
    # within-frame structure clips. Default: the run's own fall plus 0.5 of a
    # decade of headroom so the last frame still has internal contrast.
    # It is PRINTED and it is on every frame, because a scale that adapts to its
    # input cannot be compared between runs unless you can see what it chose --
    # pass --decades explicitly to compare two runs on one ramp.
    if opt['decades'] <= 0.0:
        fall_dec = math.log10(vref / vmin) if vmin > 0.0 else 1.0
        opt['decades'] = min(4.0, max(1.0, fall_dec + 0.5))

    ts = [meta['rows'].get(k, (None, None, None))[0] for k in range(len(files))]
    if meta['rows']:
        idx = sorted(meta['rows'])
        ts = [meta['rows'][idx[k]][0] if k < len(idx) else None for k in range(len(files))]
        eu = [meta['rows'][idx[k]][1] if k < len(idx) else None for k in range(len(files))]
        eb = [meta['rows'][idx[k]][2] if k < len(idx) else None for k in range(len(files))]
    else:
        ts, src = times_from_log(in_dir, len(files))
        if ts and not opt['quiet']:
            print('  frame times from %s (meta.txt carries no table)'
                  % os.path.relpath(src, in_dir))
        if ts is None:
            ts = [None] * len(files)
        eu = eb = None

    series = [('|J| P99.5', p995, SERIES_COL[0])]
    if eu and any(x for x in eu if x):
        series.append(('E_u', eu, SERIES_COL[1]))
    if eb and any(x for x in eb if x):
        series.append(('E_b', eb, SERIES_COL[2]))

    # ---- layout
    sc = opt['vscale']
    ns = max(1, int(math.ceil(sc - 1e-9)))   # splat width must cover the spacing
    # mip_size and the depth cue need the radius the volume actually PROJECTS to,
    # which for a box is the circumscribing sphere -- a cube of side N reaches
    # sqrt(3) N / 2 from its centre, and at some yaw a corner sits there. R itself
    # is 0 in box mode, which would size the canvas to nothing.
    # A CUBE'S CIRCUMSCRIBING SPHERE IS THE SPECIAL CASE, NOT THE RULE. This was
    # 0.5 sqrt(3) nx, which is right only when the three extents are equal; the
    # jet's volume is 128 x 184 x 128 and that formula undersizes the canvas by
    # 17 %, clipping the box at the yaws where its long axis lies across the
    # screen. The half-diagonal below is the same number on a cube -- put
    # nx = ny = nz and it reduces to 0.5 sqrt(3) nx exactly -- so no cube result
    # moves, and it is correct for any box.
    if box:
        W0, H0, R_depth = mip_size_box(nx, ny_eff, nz, opt['elev'],
                                       opt['spin'], sc, ns)
        R_draw = R_depth
    else:
        R_draw = R_depth = R
        W0, H0 = mip_size(R_draw, sc, ns)
    mag = max(1, int(round(opt['size'] / float(W0))))
    PW, PH = W0 * mag, H0 * mag
    M = 14
    TOP, STRIP, GAP = 46, 74, 30
    W = PW + 2 * M
    H = TOP + PH + GAP + STRIP + 26
    W += W & 1
    H += H & 1

    os.makedirs(png_dir, exist_ok=True)
    fall = peaks[0] / peaks[-1] if peaks and peaks[-1] > 0 else 0.0

    if not opt['quiet']:
        print('  scan: %.2f s for %d frames (%.3f s/frame)'
              % (t_scan, len(files), t_scan / len(files)))
        print('  canvas %dx%d  (MIP %dx%d at %.1f samples/voxel, %dx%d splat,'
              ' magnified %dx)' % (W, H, W0, H0, sc, ns, ns, mag))
        print('  vref = %.4e (largest per-frame P99.5)   |J| P99.5 falls %.1fx'
              '   log ramp %.2f decades' % (vref, fall, opt['decades']))

    t0 = time.time()
    for k, fn in enumerate(files):
        v = vol if k == 0 else read_volume(fn)[3]
        tf, floor, slab = make_transfer(opt['scale'], vref, p995[k], opt['decades'])
        yaw = opt['yaw0'] + opt['yawrate'] * k

        val, dep, _w, _h = _project_inner(
            v, spans, [0.0] * (W0 * H0), [0.0] * (W0 * H0), W0, H0,
            0.5 * W0, 0.5 * H0, floor,
            *axes(yaw, opt['elev'], sc, opt['spin']), cx=cx, cy=cy, cz=cz, ns=ns)
        if ns <= 1:
            fill_holes(val, dep, W0, H0)   # only ns = 1 can leave any
        if not opt['nosmooth']:
            smooth3(val, W0, H0)

        buf = bytearray(BG * (W * H))
        px, py = M, TOP
        if not box:
            draw_ring(buf, W, H, px + 0.5 * PW - 0.5, py + 0.5 * PH - 0.5,
                      R * sc * mag, RING, 0.85)
        blit_panel(buf, W, H, px, py, val, dep, W0, H0, tf, R_depth, mag,
                   not opt['nodepth'])
        if not box:
            draw_ring(buf, W, H, px + 0.5 * PW - 0.5, py + 0.5 * PH - 0.5,
                      R * sc * mag, RING, 0.45)

        tstr = ('t/Te %6.2f' % ts[k]) if ts[k] is not None else 'frame index only'
        draw_text(buf, W, H, M, 6,
                  (opt['title'] or
                   ('3D ORSZAG-TANG, PERIODIC BOX   |J| MAX-INTENSITY PROJECTION'
                    if box else
                    'MHD DECAY IN A PENALISED SPHERE   |J| MAX-INTENSITY '
                    'PROJECTION')),
                  FG, 1)
        draw_text(buf, W, H, M, 18,
                  ('%s   frame %03d/%03d   %s%s  periodic box   '
                   'yaw %03d deg  elev %02d deg'
                   % (tstr, k + 1, len(files),
                      ('N %d' % grid_N) if gnx == gny == gnz
                      else ('%dx%dx%d' % (gnx, gny, gnz)),
                      '' if vstride == 1 else ' (vol /%d)' % vstride,
                      int(yaw) % 360, int(opt['elev'])))
                  if box else
                  ('%s   frame %03d/%03d   N %d  R %.1f   yaw %03d deg  elev %02d deg'
                   % (tstr, k + 1, len(files), nx, R, int(yaw) % 360, int(opt['elev']))),
                  DIM, 1)
        draw_text(buf, W, H, M, 30, slab, DIM, 1)

        y = TOP + PH + 6
        draw_text(buf, W, H, M, y,
                  'frame |J| peak %.3e  P99.5 %.3e  = %.3f x vref'
                  % (peaks[k], p995[k], (p995[k] / vref) if vref else 0.0), FG, 1)
        draw_text(buf, W, H, M, y + 11,
                  (('whole box shown; it is periodic, so the faces are not'
                    ' boundaries' if ny_eff == ny else
                    'periodic box, cropped to the middle %d of %d rows in y; '
                    'the faces are not boundaries' % (ny_eff, ny)) if box else
                   'ring = sphere r=R (exact under orthographic); outside it is'
                   ' penalised, not physics'), DIM, 1)

        draw_strip(buf, W, H, M, TOP + PH + GAP + 12, W - 2 * M - 2, STRIP - 24,
                   series, k)

        out = os.path.join(png_dir, 'jvol_%04d.png' % k)
        write_png(out, W, H, buf)
        if k == 0:
            rw, rh, _rows = read_png(out)      # every run proves its own output
            if (rw, rh) != (W, H):
                die('wrote a PNG that reads back as %dx%d, expected %dx%d'
                    % (rw, rh, W, H))
        if not opt['quiet'] and (k % 10 == 0 or k == len(files) - 1):
            el = time.time() - t0
            print('    frame %3d/%3d  %.2f s/frame' % (k + 1, len(files), el / (k + 1)))
    dt = time.time() - t0

    print('\n  rendered %d frames in %.1f s  =  %.2f s/frame  (%dx%d)'
          % (len(files), dt, dt / len(files), W, H))
    print('  PNGs: %s/jvol_%%04d.png' % png_dir)
    print('\n  ffmpeg -y -framerate %d -i %s/jvol_%%04d.png'
          ' -c:v libx264 -pix_fmt yuv420p -crf 18 -vf "scale=iw*2:ih*2:flags=neighbor" %s'
          % (opt['fps'], png_dir, out_mp4))
    return 0


def axes(yaw, elev, sc, spin='z'):
    """Orthographic camera basis -> the nine per-voxel coefficients the inner
    loop increments: d(px), d(py), d(depth) per unit step in x, y and z. Yaw is
    about the volume's z axis and elevation is above the z = const plane. The
    SPHERE is invariant under both, so the rotation moves only the field inside
    it and the silhouette stays put -- which is what makes the ring a usable
    frame of reference rather than a wobbling outline."""
    ce, se = math.cos(math.radians(elev)), math.sin(math.radians(elev))
    cf, sf = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    rx, ry, rz = -sf, cf, 0.0                      # screen right
    ux, uy, uz = -se * cf, -se * sf, ce            # screen up
    fx, fy, fz = -ce * cf, -ce * sf, -se           # into the screen
    # WHICH AXIS TO SPIN ABOUT IS A QUESTION A SPHERE CANNOT ASK. For the
    # penalised sphere and for Orszag-Tang's cube the three axes are alike and
    # z is as good as any. The jet is not alike: x2 is the cross-stream
    # direction the shear layer spreads along, 20 units against 13.96, and
    # spinning about z lays it across the screen so the box tumbles end over
    # end and no frame resembles the mid-plane slices. --spin y stands it
    # upright and turns the two periodic directions past the camera, which is
    # the view the physics is usually drawn in. The remap swaps the y and z
    # components of all three basis vectors, so 'z' is bit-for-bit the old
    # behaviour and no existing render moves.
    if spin == 'y':
        ry, rz = rz, ry
        uy, uz = uz, uy
        fy, fz = fz, fy
    return (sc * rx, sc * ry, sc * rz,
            -sc * ux, -sc * uy, -sc * uz,          # image y grows downward
            fx, fy, fz)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
