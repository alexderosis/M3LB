#!/usr/bin/env python3
"""Three-panel animation frames for demonstrator/mhd_sphere.

    python3 results/N_mhd_sphere/render_slices.py [indir] [outdir] [options]

defaults: indir  results/N_mhd_sphere/anim_frames
          outdir results/N_mhd_sphere/png

STANDARD LIBRARY ONLY -- no numpy, no PIL, no matplotlib. PNG is written by
hand (zlib + struct, filter 0 on every scanline) and the caption font is a 5x7
bitmap in this file, because the ffmpeg on this machine has no drawtext filter
and there is nothing else here that can put a glyph on a pixel.

==============================================================================
WHAT THE COLOUR SCALE DOES AND DOES NOT SHOW  --  read this before believing
a frame.
==============================================================================
This is a DECAYING flow. Over the reference run (N=64, Re=200, Pm=1, 4 turnover
times) E_u falls 1.25e-3 -> 3.55e-7, a factor of 3521, i.e. |u| falls 59x; E_b
falls only 84x, i.e. |b| falls 9.2x. So the two fields do not decay together
and NO single linear scale serves both:

  * a per-frame autoscale makes a decaying flow look steady. It is a lie and
    is not offered here, not even as an option.
  * one global linear scale is honest and unreadable: at t/Te = 4 the velocity
    panel is at 1.7% of full scale, which is black.

Three schemes are offered instead. All three normalise from the SPHERE
INTERIOR only -- the penalised exterior is dead fluid and must not set a scale.

  -scale log     (default) log10 over DEC decades (default 3) below a global
                 reference per field. Shows structure at every time AND dims
                 visibly as the field decays: |u| at 59x down sits at 0.41 of
                 the bar, not at 0. Shows: relative structure, and decay, in
                 compressed form. Does NOT show: linear amplitude ratios --
                 a patch twice as bright is 10^(DEC/2) times larger, not 2x.
  -scale rms     fixed at CLIP x the frame's own r.m.s. (default 3). Best for
                 reading SHAPE -- sheets, the Alfvenic alignment -- and it is
                 deliberately blind to amplitude: the last frame looks like the
                 first. The decay is then carried ONLY by the printed rms/peak
                 per panel and by the log energy history at the bottom, which
                 is why that history is drawn in every mode and not optional.
  -scale global  one linear scale from the global reference, for all frames.
                 The honest blackout, kept so the argument above can be
                 reproduced rather than taken on trust.

The global reference is, per field, the largest over frames of that frame's
99.5th percentile over the interior. Not the outright max: |J| lives in thin
sheets and one hot cell would pale the whole sequence (the same argument
doc/fig/mkpng.py makes for its percentile clip).

The energy history strip is read from meta.txt, is always log10, and carries a
cursor at the current frame. IT is the decay evidence; the panels are the
structure.

Measured on the SYNTHETIC harness (fake frames built to the .dat's own decay
rates -- this checks the renderer, not the physics), mean interior luminance of
the first rendered frame against the last, over an energy drop of 688x in E_u
and 35x in E_b:

    scale      |u|                |b|
    log        168.9 -> 92.7      179.1 -> 103.5     0.55x / 0.58x
    rms         80.9 -> 80.0       50.7 ->  49.9     0.99x / 0.98x   flat BY
                                                     CONSTRUCTION
    global     106.2 -> 34.2       95.8 ->  16.8     0.32x / 0.18x   -> black

So the default dims by about a factor two over the part of the decay these
frames cover, `rms` is provably blind to it, and `global` is provably on its way
to an unreadable frame. Those are the three claims above, as numbers.

==============================================================================
THE DUMP FORMAT, as written by validation/FieldDump.hpp -- verified against the
header, not against the prose that asked for this script.
==============================================================================
  write_raw / scalar_slice : int32 nx, int32 ny, then nx*ny float32, x fastest.
  scalar_volume            : int32 nx, int32 ny, int32 nz  <-- THREE ints, then
                             nx*ny*nz float32, x fastest then y then z.
It is NOT "ny replaced by ny*nz" in a two-int header; a reader written to that
belief would consume 4 bytes of float data as a header and shear every slice.
read_volume() below checks the size three ways before trusting it.

meta.txt, written by demonstrator/mhd_sphere.cpp (dump loop, `-dumpevery`):
      N <int>
      R <float>
      Te <float>                <-- present, and the brief did not mention it
      frame <index> <t/Te> <E_u> <E_b>      ... one per frame, flushed
      frames <int>              <-- written at CLOSE, so it is ABSENT if the
                                    run is still going or died. Never trust it
                                    as the frame count; glob the directory.
Everything here degrades to "render what is on disk" if meta.txt is missing or
truncated, because reading a live run is the normal case.

Default input dir is anim_frames/ (what mhd_sphere.cpp actually writes, see its
dump()); frames/ is accepted as a fallback spelling.
"""

import math
import os
import struct
import sys
import zlib
from array import array

# ----------------------------------------------------------------------------
# Colour maps. Stops inline, as required -- viridis, magma and inferno sampled
# at nine points each. A 256-entry LUT is built once per map: the per-pixel path
# then costs one int() and one bytes lookup, which is what makes a pure-Python
# renderer fast enough to be worth having.
# ----------------------------------------------------------------------------
VIRIDIS = [(0.000, (68, 1, 84)), (0.125, (72, 40, 120)), (0.250, (62, 74, 137)),
           (0.375, (49, 104, 142)), (0.500, (38, 130, 142)), (0.625, (31, 158, 137)),
           (0.750, (53, 183, 121)), (0.875, (109, 205, 89)), (0.940, (180, 222, 44)),
           (1.000, (253, 231, 37))]

MAGMA = [(0.000, (0, 0, 4)), (0.125, (24, 15, 62)), (0.250, (59, 15, 112)),
         (0.375, (100, 26, 128)), (0.500, (140, 41, 129)), (0.625, (183, 55, 121)),
         (0.750, (222, 73, 104)), (0.875, (249, 123, 93)), (0.940, (253, 177, 122)),
         (1.000, (252, 253, 191))]

INFERNO = [(0.000, (0, 0, 4)), (0.125, (22, 11, 57)), (0.250, (66, 10, 104)),
           (0.375, (106, 23, 110)), (0.500, (147, 38, 103)), (0.625, (188, 55, 84)),
           (0.750, (221, 81, 58)), (0.875, (243, 120, 25)), (0.940, (250, 186, 33)),
           (1.000, (252, 255, 164))]

BG = (14, 15, 20)            # frame background
FG = (222, 226, 234)         # caption text
DIM = (138, 146, 162)        # secondary text
BORDER = (70, 76, 90)        # panel border
CIRCLE = (236, 240, 248)     # the sphere boundary r = R
CUR_U = (120, 205, 255)      # E_u in the history strip
CUR_B = (255, 160, 110)      # E_b
EXT_ALPHA = 0.13             # how much of the exterior survives -- see below


def make_lut(stops):
    """256-entry RGB lookup, each entry a 3-byte bytes object."""
    lut = []
    for i in range(256):
        t = i / 255.0
        c = stops[-1][1]
        for k in range(len(stops) - 1):
            a, b = stops[k], stops[k + 1]
            if t <= b[0]:
                f = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
                c = tuple(int(round(a[1][j] + f * (b[1][j] - a[1][j]))) for j in range(3))
                break
        lut.append(bytes(c))
    return lut


# ----------------------------------------------------------------------------
# 5x7 bitmap font. '#' is ink. Drawn at an integer scale, advance 6*scale.
# ----------------------------------------------------------------------------
_G = {
 '0': ".###.|#...#|#..##|#.#.#|##..#|#...#|.###.",
 '1': "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
 '2': ".###.|#...#|....#|...#.|..#..|.#...|#####",
 '3': "#####|...#.|..#..|...#.|....#|#...#|.###.",
 '4': "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.",
 '5': "#####|#....|####.|....#|....#|#...#|.###.",
 '6': "..##.|.#...|#....|####.|#...#|#...#|.###.",
 '7': "#####|....#|...#.|..#..|.#...|.#...|.#...",
 '8': ".###.|#...#|#...#|.###.|#...#|#...#|.###.",
 '9': ".###.|#...#|#...#|.####|....#|...#.|.##..",
 'A': ".###.|#...#|#...#|#####|#...#|#...#|#...#",
 'B': "####.|#...#|#...#|####.|#...#|#...#|####.",
 'C': ".###.|#...#|#....|#....|#....|#...#|.###.",
 'D': "###..|#..#.|#...#|#...#|#...#|#..#.|###..",
 'E': "#####|#....|#....|####.|#....|#....|#####",
 'F': "#####|#....|#....|####.|#....|#....|#....",
 'G': ".###.|#...#|#....|#.###|#...#|#...#|.####",
 'H': "#...#|#...#|#...#|#####|#...#|#...#|#...#",
 'I': ".###.|..#..|..#..|..#..|..#..|..#..|.###.",
 'J': "..###|...#.|...#.|...#.|...#.|#..#.|.##..",
 'K': "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#",
 'L': "#....|#....|#....|#....|#....|#....|#####",
 'M': "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#",
 'N': "#...#|##..#|#.#.#|#..##|#...#|#...#|#...#",
 'O': ".###.|#...#|#...#|#...#|#...#|#...#|.###.",
 'P': "####.|#...#|#...#|####.|#....|#....|#....",
 'Q': ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#",
 'R': "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
 'S': ".####|#....|#....|.###.|....#|....#|####.",
 'T': "#####|..#..|..#..|..#..|..#..|..#..|..#..",
 'U': "#...#|#...#|#...#|#...#|#...#|#...#|.###.",
 'V': "#...#|#...#|#...#|#...#|#...#|.#.#.|..#..",
 'W': "#...#|#...#|#...#|#.#.#|#.#.#|##.##|#...#",
 'X': "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
 'Y': "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..",
 'Z': "#####|....#|...#.|..#..|.#...|#....|#####",
 'a': ".....|.....|.###.|....#|.####|#...#|.####",
 'b': "#....|#....|####.|#...#|#...#|#...#|####.",
 'c': ".....|.....|.###.|#....|#....|#....|.###.",
 'd': "....#|....#|.####|#...#|#...#|#...#|.####",
 'e': ".....|.....|.###.|#...#|#####|#....|.###.",
 'f': "..##.|.#...|.#...|####.|.#...|.#...|.#...",
 'g': ".....|.....|.####|#...#|.####|....#|.###.",
 'h': "#....|#....|####.|#...#|#...#|#...#|#...#",
 'i': "..#..|.....|.##..|..#..|..#..|..#..|.###.",
 'j': "...#.|.....|...#.|...#.|...#.|#..#.|.##..",
 'k': "#....|#....|#..#.|#.#..|##...|#.#..|#..#.",
 'l': ".##..|..#..|..#..|..#..|..#..|..#..|.###.",
 'm': ".....|.....|##.#.|#.#.#|#.#.#|#...#|#...#",
 'n': ".....|.....|####.|#...#|#...#|#...#|#...#",
 'o': ".....|.....|.###.|#...#|#...#|#...#|.###.",
 'p': ".....|.....|####.|#...#|#...#|####.|#....",
 'q': ".....|.....|.####|#...#|#...#|.####|....#",
 'r': ".....|.....|#.##.|##...|#....|#....|#....",
 's': ".....|.....|.####|#....|.###.|....#|####.",
 't': ".#...|.#...|####.|.#...|.#...|.#..#|..##.",
 'u': ".....|.....|#...#|#...#|#...#|#..##|.##.#",
 'v': ".....|.....|#...#|#...#|#...#|.#.#.|..#..",
 'w': ".....|.....|#...#|#...#|#.#.#|#.#.#|.#.#.",
 'x': ".....|.....|#...#|.#.#.|..#..|.#.#.|#...#",
 'y': ".....|.....|#...#|#...#|.####|....#|.###.",
 'z': ".....|.....|#####|...#.|..#..|.#...|#####",
 ' ': ".....|.....|.....|.....|.....|.....|.....",
 '.': ".....|.....|.....|.....|.....|.##..|.##..",
 ',': ".....|.....|.....|.....|.##..|.##..|.#...",
 '-': ".....|.....|.....|#####|.....|.....|.....",
 '+': ".....|..#..|..#..|#####|..#..|..#..|.....",
 '/': "....#|....#|...#.|..#..|.#...|#....|#....",
 '|': "..#..|..#..|..#..|..#..|..#..|..#..|..#..",
 '=': ".....|.....|#####|.....|#####|.....|.....",
 '(': "..##.|.#...|.#...|.#...|.#...|.#...|..##.",
 ')': ".##..|...#.|...#.|...#.|...#.|...#.|.##..",
 '[': ".###.|.#...|.#...|.#...|.#...|.#...|.###.",
 ']': ".###.|...#.|...#.|...#.|...#.|...#.|.###.",
 ':': ".....|.##..|.##..|.....|.##..|.##..|.....",
 '%': "##..#|##.#.|..#..|.#...|#.##.|..##.|.....",
 '^': "..#..|.#.#.|#...#|.....|.....|.....|.....",
 '_': ".....|.....|.....|.....|.....|.....|#####",
 '<': "...#.|..#..|.#...|#....|.#...|..#..|...#.",
 '>': ".#...|..#..|...#.|....#|...#.|..#..|.#...",
 '*': ".....|#.#.#|.###.|#####|.###.|#.#.#|.....",
 '#': ".#.#.|.#.#.|#####|.#.#.|#####|.#.#.|.#.#.",
 '?': ".###.|#...#|....#|...#.|..#..|.....|..#..",
 '!': "..#..|..#..|..#..|..#..|..#..|.....|..#..",
}
FONT = {c: [r for r in s.split('|')] for c, s in _G.items()}
CH_W, CH_H = 5, 7


# ----------------------------------------------------------------------------
# Canvas: an RGB byte buffer plus the few primitives the layout needs.
# ----------------------------------------------------------------------------
class Canvas:
    def __init__(self, w, h, bg=BG):
        self.w, self.h = w, h
        self.px = bytearray(bytes(bg) * (w * h))

    def rect(self, x0, y0, w, h, rgb):
        c = bytes(rgb)
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(self.w, x0 + w), min(self.h, y0 + h)
        if x2 <= x1 or y2 <= y1:
            return
        row = c * (x2 - x1)
        for y in range(y1, y2):
            o = (y * self.w + x1) * 3
            self.px[o:o + len(row)] = row

    def frame_rect(self, x0, y0, w, h, rgb):
        self.rect(x0, y0, w, 1, rgb)
        self.rect(x0, y0 + h - 1, w, 1, rgb)
        self.rect(x0, y0, 1, h, rgb)
        self.rect(x0 + w - 1, y0, 1, h, rgb)

    def dot(self, x, y, rgb):
        if 0 <= x < self.w and 0 <= y < self.h:
            o = (y * self.w + x) * 3
            self.px[o:o + 3] = bytes(rgb)

    def blit_rows(self, x0, y0, rows, zoom):
        """rows[i] is one already-expanded RGB scanline; each is written `zoom`
        times, which is the whole of the upscaling."""
        y = y0
        for r in rows:
            for _ in range(zoom):
                if 0 <= y < self.h:
                    o = (y * self.w + x0) * 3
                    self.px[o:o + len(r)] = r
                y += 1

    def text(self, x, y, s, rgb=FG, scale=1):
        cx = x
        for ch in s:
            g = FONT.get(ch)
            if g is None:
                g = FONT['?']
            for gy in range(CH_H):
                rowbits = g[gy]
                for gx in range(CH_W):
                    if rowbits[gx] == '#':
                        self.rect(cx + gx * scale, y + gy * scale, scale, scale, rgb)
            cx += (CH_W + 1) * scale
        return cx

    def line(self, x0, y0, x1, y1, rgb):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self.dot(x0, y0, rgb)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy


def text_w(s, scale=1):
    return len(s) * (CH_W + 1) * scale


def fit_scale(s, avail, scales=(2, 1)):
    """Largest of `scales` at which s fits in `avail` pixels. The caption is the
    only place the numbers live -- this ffmpeg has no drawtext -- so a caption
    that runs off the canvas is a lost measurement, not a cosmetic problem."""
    for sc in scales:
        if text_w(s, sc) <= avail:
            return sc
    return scales[-1]


def elide(s, avail, scale=1):
    """Truncate to what fits, so a panel's stat line cannot bleed into its
    neighbour."""
    n = max(0, avail // ((CH_W + 1) * scale))
    return s if len(s) <= n else (s[:max(0, n - 1)] + '>' if n else '')


def write_png(path, w, h, px):
    """8-bit truecolour, filter 0 on every scanline. Chunk = length, type,
    data, crc32(type+data)."""
    stride = w * 3
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += px[y * stride:(y + 1) * stride]

    def chunk(typ, data):
        return (struct.pack('>I', len(data)) + typ + data
                + struct.pack('>I', zlib.crc32(typ + data) & 0xffffffff))

    blob = (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(bytes(raw), 6))
            + chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(blob)
    return len(blob)


# ----------------------------------------------------------------------------
# Readers. Both validate the byte count against the header before returning,
# because a renderer pointed at a live run WILL meet a half-written file and
# must skip it rather than shear it.
# ----------------------------------------------------------------------------
def read_slice(path):
    with open(path, 'rb') as f:
        d = f.read()
    if len(d) < 8:
        raise ValueError('shorter than a header')
    nx, ny = struct.unpack('<ii', d[:8])
    if nx <= 0 or ny <= 0 or nx > 1 << 14 or ny > 1 << 14:
        raise ValueError('implausible header %d x %d' % (nx, ny))
    need = 8 + 4 * nx * ny
    if len(d) < need:
        raise ValueError('truncated: %d bytes, header wants %d' % (len(d), need))
    v = array('f')
    v.frombytes(d[8:need])
    if sys.byteorder != 'little':
        v.byteswap()
    return nx, ny, v


def read_volume(path):
    """int32 nx, ny, nz then nx*ny*nz floats, x fastest then y then z.
    Three ints -- confirmed in validation/FieldDump.hpp scalar_volume()."""
    with open(path, 'rb') as f:
        d = f.read()
    if len(d) < 12:
        raise ValueError('shorter than a header')
    nx, ny, nz = struct.unpack('<iii', d[:12])
    if min(nx, ny, nz) <= 0 or max(nx, ny, nz) > 1 << 12:
        raise ValueError('implausible header %d x %d x %d' % (nx, ny, nz))
    need = 12 + 4 * nx * ny * nz
    if len(d) < need:
        raise ValueError('truncated: %d bytes, header wants %d' % (len(d), need))
    v = array('f')
    v.frombytes(d[12:need])
    if sys.byteorder != 'little':
        v.byteswap()
    return nx, ny, nz, v


def read_meta(path):
    meta = {'N': None, 'R': None, 'Te': None, 'frames': None, 'rows': {}, 'case': None}
    if not os.path.exists(path):
        return meta
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if not p:
                continue
            try:
                if p[0] == 'N' and len(p) > 1:
                    meta['N'] = int(float(p[1]))
                elif p[0] == 'R' and len(p) > 1:
                    meta['R'] = float(p[1])
                elif p[0] == 'Te' and len(p) > 1:
                    meta['Te'] = float(p[1])
                elif p[0] == 'frames' and len(p) > 1:
                    meta['frames'] = int(p[1])
                elif p[0] == 'case' and len(p) > 1:
                    # optional; demonstrator/tg_mhd writes it so a closed box is
                    # not captioned as the periodic Orszag-Tang it shares a mode with
                    meta['case'] = ' '.join(p[1:])
                elif p[0] == 'frame' and len(p) >= 5:
                    meta['rows'][int(p[1])] = (float(p[2]), float(p[3]), float(p[4]))
            except ValueError:
                continue          # a torn last line during a live run
    return meta


# ----------------------------------------------------------------------------
def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    i = int(q * (len(sorted_vals) - 1))
    return sorted_vals[i]


def fmt(x, sig=3):
    if x != x:
        return 'nan'
    if x == 0:
        return '0'
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return ('%.' + str(sig - 1) + 'e') % x
    return ('%.' + str(sig) + 'g') % x


class Field:
    """One panel's identity: file stem, label, colour map, and the global
    reference the scales are built from."""

    def __init__(self, stem, label, lut):
        self.stem = stem
        self.label = label
        self.lut = lut
        self.ref = 0.0          # global reference (see banner)
        self.stats = {}         # frame -> (rms, peak, p995), interior only


# ----------------------------------------------------------------------------
def build_mask(N, R, cx, cy):
    """True where the cell centre is inside the sphere's mid-plane cut. The
    centre is ((N-1)/2, (N-1)/2, (N-1)/2) and the slice is at z = N//2, so the
    circle in this plane has radius sqrt(R^2 - dz^2), NOT R, whenever N is even
    -- dz = N//2 - (N-1)/2 = 0.5 there. At N=64, R=25.6 that is 25.5951, a
    0.02% correction: too small to see and free to get right."""
    # R <= 0 IS THE PERIODIC-BOX SENTINEL, written by GPU/src/orszag_tang.cu,
    # which has no sphere: all-true mask, rcut 0, and draw_circle() already
    # treats rcut <= 0 as 'no ring'. Distinct from R being ABSENT, which still
    # means 'assume mhd_sphere's rfac = 0.40' -- a box case must write R 0
    # explicitly or it gets cut down to a disc.
    if R is not None and R <= 0:
        return bytearray(b'\x01' * (N * N)), 0.0
    dz = float(N // 2) - 0.5 * (N - 1)
    rr = R * R - dz * dz
    rcut = math.sqrt(rr) if rr > 0 else 0.0
    m = bytearray(N * N)
    r2 = rcut * rcut
    for y in range(N):
        dy = y - cy
        for x in range(N):
            dx = x - cx
            if dx * dx + dy * dy <= r2:
                m[y * N + x] = 1
    return m, rcut


def interior_stats(v, mask, N):
    s = 0.0
    n = 0
    vals = []
    for i in range(N * N):
        if mask[i]:
            a = v[i]
            if a != a:
                continue
            s += a * a
            n += 1
            vals.append(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    vals.sort()
    return math.sqrt(s / n), vals[-1], percentile(vals, 0.995)


def panel_rows(v, N, mask, lut, mode, hi, lo_log, zoom):
    """One panel's pixel rows, already expanded horizontally by `zoom`.

    The exterior is not deleted, it is attenuated to EXT_ALPHA of its colour
    over the background. mhd_sphere.cpp's dump() banner keeps the penalised
    region in the file on purpose -- so that it can be checked to be at rest --
    and a renderer that painted it flat black would throw that check away. At
    13% it cannot compete with the interior for attention and a lit exterior is
    still visible, which is exactly the failure you want to see."""
    rows = []
    inv = 0.0
    if mode == 'log':
        dlo = math.log10(lo_log) if lo_log > 0 else -30.0
        dhi = math.log10(hi) if hi > 0 else dlo + 1.0
        span = (dhi - dlo) if dhi > dlo else 1.0
    else:
        inv = (1.0 / hi) if hi > 0 else 0.0
    for y in range(N - 1, -1, -1):          # flip: y increases upward
        base = y * N
        out = bytearray()
        for x in range(N):
            a = v[base + x]
            if a != a or a <= 0.0:
                t = 0.0
            elif mode == 'log':
                t = (math.log10(a) - dlo) / span
            else:
                t = a * inv
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            c = lut[int(t * 255.0)]
            if not mask[base + x]:
                c = bytes((int(BG[k] + EXT_ALPHA * (c[k] - BG[k])) for k in range(3)))
            out += c * zoom
        rows.append(bytes(out))
    return rows


def draw_circle(cv, x0, y0, N, zoom, rcut, cx, cy, rgb=CIRCLE):
    """The sphere boundary, one pixel thick.

    Scanned by row AND by column rather than swept in angle. Angular sampling
    looked correct and was not: near the top and bottom of the circle dy/dtheta
    goes to zero, so hundreds of samples land on the same few rows and the
    boundary grew a chunky white cap several pixels thick, while the left and
    right extremes stayed hairline. Solving for the two crossings of each scan
    line gives a uniform one-pixel curve with no gaps, at a fraction of the
    cost."""
    if rcut <= 0:
        return
    rp = rcut * zoom
    ccx = x0 + (cx + 0.5) * zoom            # circle centre, canvas coordinates
    ccy = y0 + (N - 1 - cy + 0.5) * zoom
    xlo, xhi = x0, x0 + N * zoom - 1
    ylo, yhi = y0, y0 + N * zoom - 1

    def put(px, py):
        if xlo <= px <= xhi and ylo <= py <= yhi:
            cv.dot(px, py, rgb)

    for py in range(int(math.floor(ccy - rp)), int(math.ceil(ccy + rp)) + 1):
        s = rp * rp - (py - ccy) ** 2
        if s < 0:
            continue
        dx = math.sqrt(s)
        put(int(round(ccx - dx)), py)
        put(int(round(ccx + dx)), py)
    for px in range(int(math.floor(ccx - rp)), int(math.ceil(ccx + rp)) + 1):
        s = rp * rp - (px - ccx) ** 2
        if s < 0:
            continue
        dy = math.sqrt(s)
        put(px, int(round(ccy - dy)))
        put(px, int(round(ccy + dy)))


def draw_cbar(cv, x, y, w, h, lut):
    for i in range(w):
        cv.rect(x + i, y, 1, h, lut[int(255.0 * i / max(1, w - 1))])
    cv.frame_rect(x - 1, y - 1, w + 2, h + 2, BORDER)


def draw_history(cv, x, y, w, h, rows, cur, mode_note):
    """log10 E_u and log10 E_b against t/Te, with a cursor at the current
    frame. This is the panel that proves the flow is decaying, and it is drawn
    in every colour mode for exactly that reason."""
    cv.frame_rect(x, y, w, h, BORDER)
    ks = sorted(rows)
    pts = [(rows[k][0], rows[k][1], rows[k][2]) for k in ks]
    vals = [e for _, a, b in pts for e in (a, b) if e > 0]
    if len(pts) < 2 or not vals:
        cv.text(x + 6, y + 6, 'no energy history in meta.txt', DIM, 1)
        return
    t0, t1 = pts[0][0], pts[-1][0]
    if t1 <= t0:
        t1 = t0 + 1.0
    lo = math.floor(math.log10(min(vals)))
    hi = math.ceil(math.log10(max(vals)))
    if hi <= lo:
        hi = lo + 1
    pad_l, pad_r, pad_t, pad_b = 52, 8, 10, 14
    px0, py0 = x + pad_l, y + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    def sx(t):
        return px0 + int(pw * (t - t0) / (t1 - t0))

    def sy(e):
        if e <= 0:
            return py0 + ph
        return py0 + int(ph * (hi - math.log10(e)) / (hi - lo))

    d = int(hi - lo)
    step = 1 if d <= 6 else 2
    for k in range(int(lo), int(hi) + 1, step):
        yy = sy(10.0 ** k)
        for xx in range(px0, px0 + pw, 4):
            cv.dot(xx, yy, (44, 48, 60))
        cv.text(x + 4, yy - 3, '1e%d' % k, DIM, 1)
    for series, col in ((1, CUR_U), (2, CUR_B)):
        prev = None
        for p in pts:
            q = (sx(p[0]), sy(p[series]))
            if prev:
                cv.line(prev[0], prev[1], q[0], q[1], col)
            prev = q
    if cur in rows:
        c = rows[cur]
        cxp = sx(c[0])
        for yy in range(py0, py0 + ph, 3):
            cv.dot(cxp, yy, (110, 118, 136))
        for series, col in ((1, CUR_U), (2, CUR_B)):
            yy = sy(c[series])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cv.dot(cxp + dx, yy + dy, col)
    cv.text(px0, y + h - 11, 't/Te ' + fmt(t0, 3), DIM, 1)
    rl = 't/Te ' + fmt(t1, 3)
    cv.text(px0 + pw - text_w(rl, 1), y + h - 11, rl, DIM, 1)
    lx = cv.text(px0 + 8, y + 2, 'E_u', CUR_U, 1)
    lx = cv.text(lx + 6, y + 2, 'E_b', CUR_B, 1)
    cv.text(lx + 10, y + 2, mode_note, DIM, 1)


# ----------------------------------------------------------------------------
def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    indir, outdir = None, None
    scale_mode, clip, dec, zoom, want_mip = 'log', 3.0, 3.0, 0, False
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('-scale', '--scale') and i + 1 < len(argv):
            i += 1
            scale_mode = argv[i]
        elif a in ('-clip', '--clip') and i + 1 < len(argv):
            i += 1
            clip = float(argv[i])
        elif a in ('-dec', '--dec') and i + 1 < len(argv):
            i += 1
            dec = float(argv[i])
        elif a in ('-zoom', '--zoom') and i + 1 < len(argv):
            i += 1
            zoom = int(argv[i])
        elif a in ('-mip', '--mip'):
            want_mip = True
        elif a in ('-h', '--help'):
            print(__doc__)
            return 0
        else:
            pos.append(a)
        i += 1
    if pos:
        indir = pos[0]
    if len(pos) > 1:
        outdir = pos[1]
    if scale_mode not in ('log', 'rms', 'global'):
        print('unknown -scale %r: use log, rms or global' % scale_mode)
        return 2

    if indir is None:
        indir = os.path.join(here, 'anim_frames')
        if not os.path.isdir(indir) and os.path.isdir(os.path.join(here, 'frames')):
            indir = os.path.join(here, 'frames')
    if outdir is None:
        outdir = os.path.join(here, 'png')
    if not os.path.isdir(indir):
        print('no input directory: %s' % indir)
        print('run:  ./build/demonstrator/mhd_sphere -dumpevery 500 --kokkos-num-threads=4')
        return 1
    os.makedirs(outdir, exist_ok=True)

    meta = read_meta(os.path.join(indir, 'meta.txt'))

    # Frame indices come from the DIRECTORY, never from meta's `frames` line --
    # that line is written when the file is closed and is absent on a live or
    # killed run.
    idx = []
    for fn in os.listdir(indir):
        if fn.startswith('umag_') and fn.endswith('.raw'):
            try:
                idx.append(int(fn[5:-4]))
            except ValueError:
                pass
    idx.sort()
    if not idx:
        print('no umag_*.raw in %s' % indir)
        return 1

    fields = [Field('umag', '|u|  speed', make_lut(VIRIDIS)),
              Field('bmag', '|b|  field', make_lut(MAGMA)),
              Field('jmag', '|J|  current', make_lut(INFERNO))]
    if want_mip:
        fields.append(Field('jvol', '|J|  max over z', make_lut(INFERNO)))

    def path(f, k):
        return os.path.join(indir, '%s_%04d.raw' % (f.stem, k))

    def load(f, k):
        """Panel data as an N*N slice. jvol is a volume, reduced to a maximum
        intensity projection along z -- which is the axis the other three
        panels cut, so the MIP and the slices are the same view."""
        if f.stem != 'jvol':
            nx, ny, v = read_slice(path(f, k))
            if nx != ny:
                raise ValueError('not square: %d x %d' % (nx, ny))
            return nx, v
        nx, ny, nz, vol = read_volume(path(f, k))
        if nx != ny:
            raise ValueError('not square: %d x %d' % (nx, ny))
        out = array('f', bytes(4 * nx * ny))
        for z in range(nz):
            o = z * nx * ny
            for i2 in range(nx * ny):
                a = vol[o + i2]
                if a > out[i2]:
                    out[i2] = a
        return nx, out

    # ---- pass 1: geometry, and the global reference per field ---------------
    N = meta['N']
    if N is None:
        N = read_slice(path(fields[0], idx[0]))[0]
    R = meta['R']
    r_note = ''
    if R is not None and R <= 0:
        r_note = '  (periodic box: no sphere mask, no ring)'
    elif R is None:
        R = 0.40 * N                     # Opts::rfac default in mhd_sphere.cpp
        r_note = '  (R ASSUMED from rfac=0.40: meta.txt had none)'
    cx = cy = 0.5 * (N - 1)
    mask, rcut = build_mask(N, R, cx, cy)

    print('render_slices: %s -> %s' % (indir, outdir))
    print('  N = %d   R = %.3f   mid-plane cut radius %.3f%s' % (N, R, rcut, r_note))
    print('  %d frame index(es) on disk, meta has %d row(s)%s'
          % (len(idx), len(meta['rows']),
             '' if meta['frames'] is None else ', meta says frames=%d' % meta['frames']))
    print('  pass 1: interior statistics')

    for k in idx:
        for f in fields:
            try:
                n2, v = load(f, k)
            except (OSError, ValueError) as e:
                print('    frame %04d: %s_%04d.raw skipped (%s)' % (k, f.stem, k, e))
                f.stats[k] = None
                continue
            if n2 != N:
                print('    frame %04d: %s has N=%d, expected %d -- skipped'
                      % (k, f.stem, n2, N))
                f.stats[k] = None
                continue
            st = interior_stats(v, mask, N)
            f.stats[k] = st
            if st[2] > f.ref:
                f.ref = st[2]
    live = [k for k in idx if fields[0].stats.get(k)]
    if not live:
        print('  nothing renderable')
        return 1
    for f in fields:
        if f.ref <= 0:
            f.ref = 1.0
        print('    %-16s global ref (max over frames of p99.5, interior) = %s'
              % (f.label, fmt(f.ref, 4)))

    # ---- layout -------------------------------------------------------------
    if zoom <= 0:
        zoom = max(1, int(round(320.0 / N)))
    P = N * zoom
    npan = len(fields)
    PAD, GAP = 14, 16
    W = PAD * 2 + npan * P + (npan - 1) * GAP
    # A narrow panel row must not clip the caption: guarantee room for a
    # representative header at scale 1 even when the panels alone are narrower.
    W = max(W, PAD * 2 + text_w('mhd_sphere   N=%d  R=%.1f   mid-plane z=%d   '
                                'frame 000   t/Te = 00.000' % (N, R, N // 2), 1))
    Y_H1, Y_H2 = PAD, PAD + 20
    Y_LBL = PAD + 42
    Y_PAN = Y_LBL + 18
    Y_CBAR = Y_PAN + P + 9
    CBH = 10
    Y_S1 = Y_CBAR + CBH + 6
    Y_S2 = Y_S1 + 11
    Y_HIST = Y_S2 + 18
    HH = 104
    H = Y_HIST + HH + PAD
    W += W & 1               # libx264 + yuv420p needs even dimensions, and an
    H += H & 1               # odd one fails at encode time, not here.
    xs = [PAD + p * (P + GAP) for p in range(npan)]

    mode_note = {'log': 'log %g dec' % dec,
                 'rms': 'fixed %gx frame rms' % clip,
                 'global': 'global linear'}[scale_mode]

    e0 = None
    if meta['rows']:
        k0 = min(meta['rows'])
        e0 = (meta['rows'][k0][1], meta['rows'][k0][2])

    print('  pass 2: %d x %d px, zoom %d, scale %s' % (W, H, zoom, mode_note))
    written, skipped = [], []
    seq = 0
    for k in idx:
        if not fields[0].stats.get(k):
            skipped.append(k)
            continue
        cv = Canvas(W, H)
        row = meta['rows'].get(k)

        # --- header ---------------------------------------------------------
        # In box mode there is no sphere and no radius to quote, and the case is
        # not mhd_sphere -- labelling it so is the kind of caption that outlives
        # the run and gets believed.
        if R is not None and R <= 0 and meta.get('case'):
            h1 = '%s   N=%d   mid-plane z=%d   frame %d' % (meta['case'], N, N // 2, k)
        elif R is not None and R <= 0:
            h1 = 'orszag_tang 3D   N=%d  periodic box   mid-plane z=%d   frame %d' \
                 % (N, N // 2, k)
        else:
            h1 = 'mhd_sphere   N=%d  R=%.1f   mid-plane z=%d   frame %d' % (N, R, N // 2, k)
        if row:
            h1 += '   t/Te = %.3f' % row[0]
        cv.text(PAD, Y_H1, h1, FG, fit_scale(h1, W - 2 * PAD))
        if row:
            h2 = 'E_u = %s   E_b = %s' % (fmt(row[1], 4), fmt(row[2], 4))
            if e0 and e0[0] > 0 and e0[1] > 0 and row[1] > 0 and row[2] > 0:
                h2 += '   decayed %.0fx / %.0fx from frame 0' % (e0[0] / row[1],
                                                                 e0[1] / row[2])
        else:
            h2 = 'no meta.txt row for this frame -- energies unknown'
        h2 += '   [%s]' % mode_note
        cv.text(PAD, Y_H2, h2, DIM, fit_scale(h2, W - 2 * PAD))

        # --- panels ---------------------------------------------------------
        for p, f in enumerate(fields):
            x0 = xs[p]
            st = f.stats.get(k)
            cv.text(x0, Y_LBL, f.label, FG, fit_scale(f.label, P))
            if not st:
                cv.rect(x0, Y_PAN, P, P, (26, 27, 34))
                cv.frame_rect(x0 - 1, Y_PAN - 1, P + 2, P + 2, BORDER)
                cv.text(x0 + 8, Y_PAN + P // 2, 'FRAME MISSING', DIM, 2)
                continue
            rms, peak, p995 = st
            if scale_mode == 'rms':
                hi = clip * rms if rms > 0 else (f.ref if f.ref > 0 else 1.0)
                lo = 0.0
            elif scale_mode == 'global':
                hi, lo = f.ref, 0.0
            else:
                hi = f.ref
                lo = hi / (10.0 ** dec)
            n2, v = load(f, k)
            cv.blit_rows(x0, Y_PAN,
                         panel_rows(v, N, mask, f.lut, scale_mode, hi, lo, zoom), zoom)
            draw_circle(cv, x0, Y_PAN, N, zoom, rcut, cx, cy)
            cv.frame_rect(x0 - 1, Y_PAN - 1, P + 2, P + 2, BORDER)
            draw_cbar(cv, x0, Y_CBAR, P, CBH, f.lut)
            left = fmt(lo, 2) if scale_mode == 'log' else '0'
            cv.text(x0, Y_S1, left, DIM, 1)
            rl = fmt(hi, 3)
            cv.text(x0 + P - text_w(rl, 1), Y_S1, rl, DIM, 1)
            mid = mode_note
            if text_w(left, 1) + text_w(rl, 1) + text_w(mid, 1) + 16 <= P:
                cv.text(x0 + (P - text_w(mid, 1)) // 2, Y_S1, mid, DIM, 1)
            s2 = 'rms %s   max %s   (%s)' % (fmt(rms, 3), fmt(peak, 3),
                 'whole box' if (R is not None and R <= 0) else 'in sphere')
            cv.text(x0, Y_S2, elide(s2, P), DIM, 1)

        draw_history(cv, PAD, Y_HIST, W - 2 * PAD, HH, meta['rows'], k,
                     'log10 energy, cursor = this frame')

        # Output numbering is SEQUENTIAL in the render, not copied from the
        # source index: ffmpeg's image2 demuxer stops at the first gap in a
        # %04d sequence, so one skipped input frame would silently truncate the
        # movie. doc/fig/mkpng.py records that exact trap.
        outn = os.path.join(outdir, 'frame_%04d.png' % seq)
        nb = write_png(outn, W, H, cv.px)
        written.append((seq, k, outn, nb))
        seq += 1

    for s, k, p, nb in written:
        print('  wrote %s   (src frame %04d, %d x %d, %.1f kB)'
              % (os.path.basename(p), k, W, H, nb / 1024.0))
    print('  %d PNG(s) in %s' % (len(written), outdir))
    if skipped:
        print('  skipped %d frame(s) with no readable umag: %s'
              % (len(skipped), ' '.join('%04d' % k for k in skipped)))
    if written and written[-1][0] != written[-1][1]:
        print('  NOTE: output numbering was renumbered 0..%d because input '
              'indices had gaps' % (len(written) - 1))

    fps = 12
    stem = os.path.join(outdir, 'frame_%04d.png')
    mp4 = os.path.join(outdir, 'mhd_sphere.mp4')
    gif = os.path.join(outdir, 'mhd_sphere.gif')
    pal = os.path.join(outdir, 'palette.png')
    print('\n  mp4 (no drawtext needed -- the caption is already in the pixels):')
    print('    ffmpeg -y -framerate %d -start_number 0 -i %s \\\n'
          '      -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart %s'
          % (fps, stem, mp4))
    print('\n  gif, two passes so the palette suits the colour maps:')
    print('    ffmpeg -y -framerate %d -start_number 0 -i %s \\\n'
          '      -vf "fps=%d,scale=900:-1:flags=lanczos,palettegen=stats_mode=diff" %s'
          % (fps, stem, fps, pal))
    print('    ffmpeg -y -framerate %d -start_number 0 -i %s -i %s \\\n'
          '      -lavfi "fps=%d,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse='
          'dither=bayer:bayer_scale=3" %s' % (fps, stem, pal, fps, gif))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
