#!/usr/bin/env python3
"""Decaying MHD turbulence in a disc as a luminous ball: the current density as
light, on black.

Reads what `demonstrator/mhd_decay -bin DIR -binevery K` writes, plus the run's
own stdout saved as DIR/log.txt:

    DIR/decay_<walls>_j%04d.bin   int32 N, int32 N, then N*N float32, x fastest,
                                  row y = 0 FIRST: the current density in
                                  LATTICE units, NaN in the solid and in the
                                  cells whose stencil reaches it
    DIR/log.txt                   for N, Re, the disc's radius, the turnover time,
                                  the step count and (if any) the swirl

and pipes 1920x1080 RGB frames straight into ffmpeg, the disc in the middle.
`doc/fig/mkpng.py` is the plain rendering of the same frames and stays the one
to read a value off. This one is for showing.

    python3 tools/anim_mhd_decay_glow.py runs/rasengan/n1081 out.mp4 --orb
    python3 tools/anim_mhd_decay_glow.py DIR --orb --still-t 0.2,0.4 runs/still

Needs numpy and PIL, and ffmpeg for the video.

What is drawn, and the choices that could mislead.

  ONE CELL IS ONE PIXEL when the run has N = 1080 or 1081, and nothing is
  interpolated (1081 loses its last row and column, which are solid). A run with
  N <= 545 is enlarged twofold and any other N is resized to 1080, both
  bicubically, and the strip along the bottom then says so.

  BRIGHTNESS IS |j| AND HUE IS ITS SIGN: electric blue for j > 0, indigo for
  j < 0, black at zero. One fixed mapping for the whole run, as in
  tools/anim_ot_glow.py and for its reason: asinh(|j|/knee)/asinh(lim/knee),
  saturating at `lim`, the largest 99.9th percentile of |j| over the frames
  (about 400 of them are sampled, evenly), with knee = lim/12. Both are printed. So a field
  that decays goes dark, which is what it did.

  j IS THE QUANTITY TO SHOW IN THIS CASE, and mhd_decay.cpp's banner says why:
  the run's div b grows, and j = curl b is blind to any gradient field, so the
  current is the part of the picture that does not depend on it.

  THE THIN RING IS THE WALL, drawn at the disc's radius. It is not data.

  --orb IS DECORATION AND CHANGES BRIGHTNESS WITH RADIUS. It shades the disc as
  if it were a ball: the field is multiplied by 0.5 + 0.5 sqrt(1 - (r/R)^2), so
  it is HALF as bright at the wall as at the centre, a faint glow is added at
  the centre, and a 110-pixel blur of the whole frame is added as an aura. Do
  not compare brightness across radii on an --orb frame; the strip says the
  shading is cosmetic. Without --orb none of the three is applied.

  THE GLOW (bloom) is three Gaussians of the bright part of the picture, 2.5,
  12 and 48 pixels, added back. It widens the halo of a sheet, not the sheet.
  The two widest blurs (48, and --orb's 110) are computed on a reduced copy of
  the frame; at those widths the difference is below one grey level.

  TIME IS IN TURNOVERS, 2R/u0, the unit the run's own log uses.

  --wind AND --hands DRAW THINGS THAT WERE NEVER SIMULATED, and the frame says
  so in words. --wind adds fifty-two streaks circling the ball outside the wall,
  counter-clockwise like the vortex of a -swirl run; they come from one fixed
  random draw and know nothing of the flow. --hands adds two hands, an original
  generic shape built from tapered capsules: one under the ball, rocking, and
  one sweeping back and forth round its side, both lit only by the ball. To
  make room for two arms --hands also draws the ball SMALLER and off centre
  (radius 295 pixels), so the field is no longer one cell per pixel: it is
  reduced with a Lanczos filter, and the frame gives the pixels per cell. The
  hands move in VIDEO time, so --every changes the flow's speed and not theirs.
  The hand under the ball only strains. The other STROKES the ball one way,
  pressed on, and is lifted back; and that stroke can be made to drive the
  flow: --write-stir FILE --frames K --binevery B --radius R writes the hand's
  track, `demonstrator/mhd_decay -stir FILE -binevery B` drags the fluid under
  the hand at the hand's speed, and this script then refuses to draw the hands
  on frames whose stride is not the track's, since they would not be where the
  paddle was. The paddle's speed is a lattice velocity: --write-stir prints its
  Mach number.

  --anime DRAWS THE SAME FRAMES AS A CEL, in the two-hand layout, and it is a
  LOOK, not a rendering to read from. The current is shown in FOUR FLAT BANDS
  by strength (edges at brightness 0.14, 0.30, 0.50 and 0.72 of the same asinh
  law), two hues by sign, so differences inside a band are thrown away. The
  ball's three-tone shading, its white core and halo, its outline, the hands
  (skin in two flat tones, ink lines, a pale keyline), the bold wind and the
  speed lines are all drawn. As each stroke of the hand lands, the picture is
  in NEGATIVE for two frames and everything but the text shakes for a third of
  a second; those frames show nothing about the flow. The frame lists what is
  drawn and what is applied to the simulation. --orb and --wind have no effect
  under --anime, which brings its own.

usage: anim_mhd_decay_glow.py <dir> [out.mp4] [--orb] [--wind] [--hands] [--anime]
           [--fps 60] [--tmax T]
           [--every K] [--binevery K] [--jobs 6] [--hold 1.5] [--crf 16]
           [--still i,j,k PREFIX] [--still-t t1,t2 PREFIX]
       anim_mhd_decay_glow.py --write-stir FILE --frames K --binevery B --radius R
"""
import glob
import os
import re
import subprocess
import sys
import textwrap
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
BG = np.array([0.0016, 0.0022, 0.0046], dtype=np.float32)   # linear light
EXPOSURE = 1.15

# Linear-light ramps against brightness in [0, 1]; they pass 1 on purpose, the
# excess being what the bloom spreads and the tone curve rolls off.
STOPS = np.array([0.00, 0.20, 0.40, 0.58, 0.74, 0.88, 1.00])
POS = np.array([[0, 0, 0], [0.002, 0.008, 0.040], [0.010, 0.050, 0.26],
                [0.040, 0.26, 0.95], [0.30, 0.95, 1.9], [1.3, 2.1, 2.7],
                [2.8, 3.0, 3.1]])                       # electric blue to white
NEG = np.array([[0, 0, 0], [0.006, 0.004, 0.040], [0.040, 0.020, 0.26],
                [0.20, 0.12, 0.95], [0.75, 0.60, 1.9], [1.8, 1.7, 2.7],
                [2.9, 2.9, 3.1]])                       # indigo to white
SHELL_RGB = np.array([0.35, 0.65, 1.0], dtype=np.float32) * 0.55
CORE_RGB = np.array([0.55, 0.80, 1.0], dtype=np.float32) * 0.10

FONT_TEXT = "/System/Library/Fonts/Avenir Next.ttc"     # 7 regular
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"

G = {}                   # per-process state, filled by init()


def raw(path):
    with open(path, "rb") as fh:
        nx, ny = np.fromfile(fh, dtype=np.int32, count=2)
        return np.fromfile(fh, dtype=np.float32, count=nx * ny).reshape(ny, nx)


def flag(name):
    return name in sys.argv


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055)


def tone(lin):
    """Linear HDR -> display sRGB in [0, 1]: an exponential shoulder, then sRGB."""
    return srgb(1.0 - np.exp(-EXPOSURE * np.maximum(lin, 0.0)))


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ramp(b, table):
    out = np.empty(b.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(b, STOPS, table[:, c])
    return out


def resize(a, size, how=Image.BICUBIC):
    """A float image to (width, height) = size."""
    return np.asarray(Image.fromarray(np.ascontiguousarray(a, dtype=np.float32), "F")
                      .resize(size, how))


def blur(img, parts):
    """Sum of weighted Gaussian blurs of an h x w x 3 image, parts = [(sigma,
    weight)], by FFT on a zero-padded copy so that nothing wraps round."""
    pad = int(3 * max(s for s, _ in parts)) + 1
    hp, wp = img.shape[0] + 2 * pad, img.shape[1] + 2 * pad
    ky = np.fft.fftfreq(hp)[:, None] * 2.0 * np.pi
    kx = np.fft.rfftfreq(wp)[None, :] * 2.0 * np.pi
    k2 = kx ** 2 + ky ** 2
    K = sum(w * np.exp(-0.5 * s * s * k2) for s, w in parts).astype(np.float32)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        p = np.zeros((hp, wp), dtype=np.float32)
        p[pad:pad + img.shape[0], pad:pad + img.shape[1]] = img[..., c]
        out[..., c] = np.fft.irfft2(np.fft.rfft2(p) * K, s=(hp, wp))[pad:-pad, pad:-pad]
    return out


def blur1(a, sigma):
    """A Gaussian blur of one channel."""
    return blur(a[..., None], [(sigma, 1.0)])[..., 0]


def wide_blur(img, sigma, q):
    """A Gaussian blur of width sigma computed on a copy reduced q-fold."""
    h, w = img.shape[:2]
    small = img.reshape(h // q, q, w // q, q, 3).mean(axis=(1, 3))
    small = blur(small, [(sigma / q, 1.0)])
    return np.stack([resize(small[..., c], (w, h), Image.BILINEAR) for c in range(3)], axis=-1)


# ------------------------------------------------------------------------------
#  --hands and --wind: things DRAWN round the ball. Nothing here is simulated.
# ------------------------------------------------------------------------------
SCENE = (900.0, 440.0, 295.0)       # with --hands: the ball's centre and radius on the frame
HAND0 = (960.0, 392.0, 335.0)       # the frame the hand was designed in: ball centre, radius
HAND_RGB = np.array([0.42, 0.68, 1.0], dtype=np.float32)
WIND_RGB = np.array([0.50, 0.80, 1.15], dtype=np.float32) * 0.70


def capsule(px, py, a, b, ra, rb):
    """Signed distance to a tapered capsule from a to b (radii ra, rb)."""
    bax, bay = b[0] - a[0], b[1] - a[1]
    pax, pay = px - a[0], py - a[1]
    h = np.clip((pax * bax + pay * bay) / (bax * bax + bay * bay), 0.0, 1.0)
    return np.hypot(pax - bax * h, pay - bay * h) - (ra + (rb - ra) * h)


def smin(d1, d2, k):
    """A smooth minimum: the union of two shapes with a fillet of size k."""
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)


def hand_parts0(xx, yy, curl=0.0, arm_only=False):
    """Signed distances to the pieces of an open hand in profile, palm up, its arm
    coming from the lower left, FAR TO NEAR: arm and palm, the four fingers from
    the little one, then the thumb. Tapered capsules joined smoothly; an original,
    generic hand, in the frame it was designed in (HAND0). `curl` closes it:
    degrees added at every finger joint, and taken by the thumb towards the
    fingers. With arm_only, the forearm alone."""
    wrist = np.array([672.0, 918.0])
    ang = np.deg2rad(-9.0)                               # the palm rises to the right

    def at(x, y):                                        # local (along, up) -> pixel
        return (wrist[0] + x * np.cos(ang) + y * np.sin(ang),
                wrist[1] + x * np.sin(ang) - y * np.cos(ang))

    def chain(start, a0, segs, turn=1.0):
        """A finger: segments (length, r0, r1, bend) from `start`, turning up;
        `turn` is which way, and how much, the curl bends it."""
        d, p, a = None, np.array(start), a0
        for n_seg, (length, r0, r1, bend) in enumerate(segs):
            a = a - np.deg2rad(bend + (curl if n_seg else 0.5 * curl) * turn)   # y is down: minus turns up
            q = p + length * np.array([np.cos(a), np.sin(a)])
            c = capsule(xx, yy, p, q, r0, r1)
            d = c if d is None else smin(d, c, 5.0)
            p = q
        return d

    elbow = (wrist[0] - 1500.0 * np.cos(np.deg2rad(27.0)),
             wrist[1] + 1500.0 * np.sin(np.deg2rad(27.0)))
    d = capsule(xx, yy, elbow, wrist, 128.0, 50.0)                                 # forearm
    if arm_only:
        return [d]
    d = smin(d, capsule(xx, yy, at(-6, 0), at(250, 2), 55.0, 43.0), 26.0)          # palm
    d = smin(d, capsule(xx, yy, at(24, 10), at(120, 20), 50.0, 40.0), 20.0)        # heel of the thumb
    fingers = [  # base (along, up), first bend, [(length, r0, r1, bend) ...]
        ((252, -10), 4.0, [(98, 24, 21, 0), (60, 20.5, 18, 15), (46, 17.5, 14.5, 17)]),     # index
        ((258, 2), 9.0, [(108, 24, 21, 0), (68, 20.5, 18, 17), (48, 17.5, 14.5, 19)]),      # middle
        ((248, 12), 15.0, [(98, 23, 20, 0), (62, 19.5, 17, 19), (46, 16.5, 14, 21)]),       # ring
        ((228, 20), 23.0, [(74, 20, 17.5, 0), (46, 17, 15, 21), (38, 14.5, 12.5, 23)]),     # little
    ]
    chains = [chain(at(*base), ang - np.deg2rad(first), segs) for base, first, segs in fingers]
    thumb = chain(at(62, 34), ang - np.deg2rad(58.0),
                  [(66, 29, 25, 0), (52, 24, 20, -16), (40, 19.5, 15.5, -10)], turn=-0.8)
    return [d] + chains[::-1] + [thumb]


def hand_sdf0(xx, yy, curl=0.0, arm_only=False):
    """Signed distance to the whole hand: its pieces joined smoothly."""
    parts = hand_parts0(xx, yy, curl, arm_only)
    d = parts[0]
    for finger in parts[4:0:-1]:                         # index first: the order they were always joined in
        d = smin(d, finger, 9.0)
    return d if arm_only else smin(d, parts[5], 16.0)


CLAW = 20.0                         # the closed pose: degrees of curl at each joint
HAND_BOX = (550.0, 580.0, 1290.0, 1040.0)   # in HAND0's frame: what holds palm, fingers and thumb


def hand_setup():
    """Once: the frame's pixel grid, and how the ball lights each pixel."""
    cx, cy, rb = SCENE
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    G["grid"] = (xx, yy)
    lx, ly = cx - xx, cy - yy
    dist = np.maximum(np.hypot(lx, ly), 1.0)
    G["to_ball"] = (lx / dist, ly / dist)
    G["fall"] = 1.0 / (1.0 + ((dist - rb) / (0.50 * rb)) ** 2)     # with distance from the ball


def hand(beta_deg, lift, grip):
    """(coverage, light) of the hand turned beta degrees round the ball, counter-
    clockwise on screen, from under it, held `lift` pixels further out, and
    closed by `grip` in [0, 1] (curl = grip * CLAW at every joint). The signed
    distance is evaluated afresh at each pose -- the whole hand inside the box
    that holds it, the forearm alone elsewhere -- because blending the distances
    of an open and a closed hand gives pointed fingers wherever a fingertip
    moves by more than its own width. Lit by the ball alone: a thin edge where
    the surface faces it, and a little light wrapping further in, the second
    from a smoothed distance so that the joins between capsules do not show."""
    cx, cy, rb = SCENE
    s = rb / HAND0[2]
    b = np.deg2rad(beta_deg)
    cb, sb = float(np.cos(b)), float(np.sin(b))
    xx, yy = G["grid"]

    def canon(x, y):                              # frame pixel -> the hand's design frame
        x1, y1 = (x - cx) / s, (y - cy) / s
        return x1 * cb - y1 * sb + HAND0[0], x1 * sb + y1 * cb - lift / s + HAND0[1]

    sdf = s * hand_sdf0(*canon(xx, yy), arm_only=True)
    # the box, turned with the hand, as rows and columns of the frame
    qx = np.array([HAND_BOX[0], HAND_BOX[2], HAND_BOX[2], HAND_BOX[0]]) - HAND0[0]
    qy = np.array([HAND_BOX[1], HAND_BOX[1], HAND_BOX[3], HAND_BOX[3]]) - HAND0[1] + lift / s
    fx, fy = cx + s * (qx * cb + qy * sb), cy + s * (-qx * sb + qy * cb)
    c0, c1 = int(max(0, np.floor(fx.min()))), int(min(W, np.ceil(fx.max())))
    r0, r1 = int(max(0, np.floor(fy.min()))), int(min(H, np.ceil(fy.max())))
    gy, gx = np.gradient(sdf)
    mx, my = gx, gy                               # the smoothed normal: the sharp one outside the box
    if c1 > c0 and r1 > r0:
        box = (slice(r0, r1), slice(c0, c1))
        sdf[box] = s * hand_sdf0(*canon(xx[box], yy[box]), curl=float(np.clip(grip, 0.0, 1.0)) * CLAW)
        gy, gx = np.gradient(sdf)
        soft = blur(np.repeat(sdf[box][..., None], 3, axis=2), [(9.0, 1.0)])[..., 0]
        sy, sx = np.gradient(soft)
        mx, my = gx.copy(), gy.copy()
        inner = (slice(r0 + 12, r1 - 12), slice(c0 + 12, c1 - 12))     # the blur's edge is not used
        mx[inner], my[inner] = sx[12:-12, 12:-12], sy[12:-12, 12:-12]
    ux, uy = G["to_ball"]

    def facing(vx, vy):                           # outward normal . direction to the ball
        return np.clip((vx * ux + vy * uy) / np.maximum(np.hypot(vx, vy), 1e-6), 0.0, 1.0)

    mask = np.clip(0.5 - sdf, 0.0, 1.0)
    depth = np.maximum(-sdf, 0.0)
    lum = (1.25 * facing(gx, gy) ** 1.3 * np.exp(-depth / 5.0)
           + 0.085 * facing(mx, my) ** 1.2 * np.exp(-depth / 30.0)) * G["fall"] + 0.0035
    return mask[..., None], (lum * mask)[..., None] * HAND_RGB


def wind(t, cx, cy, rb):
    """Streaks spiralling round the ball at time t (turnovers), as linear RGB.
    Fifty-two arcs from one fixed random draw, each circling counter-clockwise,
    like the vortex, at a rate that falls with radius, and flickering."""
    rng = np.random.default_rng(7)
    q = 2
    lay = Image.new("L", (W * q, H * q), 0)
    d = ImageDraw.Draw(lay)
    for _ in range(52):
        u = rng.random()
        r0 = 1.04 + 0.36 * u ** 1.6                        # radii, crowded near the ball
        th0 = rng.random() * 2.0 * np.pi
        arc = np.deg2rad(9.0 + 38.0 * rng.random() * (1.2 - 0.5 * u))
        pitch = 0.05 + 0.12 * rng.random()                 # the tail sits further out
        width = 0.8 + 1.1 * rng.random() * (1.0 - 0.5 * u)
        peak = (0.18 + 0.62 * rng.random() ** 1.5) * (1.0 - 0.80 * u)
        rate = 1.25 / r0 ** 1.5                            # revolutions per turnover
        flick = 0.65 + 0.35 * np.sin(2.0 * np.pi * (3.0 * rng.random() + (1.5 + 2.0 * rng.random()) * t))
        k = 26
        s = np.linspace(0.0, 1.0, k)                       # 0 at the head, 1 at the tail
        th = th0 + 2.0 * np.pi * rate * t - arc * s
        rr = rb * r0 * (1.0 + pitch * s + 0.003 * np.sin(5.0 * s + 6.0 * th0))
        xs, ys = cx + rr * np.cos(th), cy - rr * np.sin(th)
        for i in range(k - 1):
            a = peak * flick * np.sin(np.pi * (0.5 * (s[i] + s[i + 1])) ** 0.7) ** 1.6
            d.line([(xs[i] * q, ys[i] * q), (xs[i + 1] * q, ys[i + 1] * q)],
                   fill=int(255 * min(1.0, a)), width=max(1, int(round(width * q))))
    lay = lay.filter(ImageFilter.GaussianBlur(0.9 * q)).resize((W, H), Image.LANCZOS)
    return (np.asarray(lay, dtype=np.float32) / 255.0)[..., None] * WIND_RGB


# ------------------------------------------------------------------------------
#  --anime: the same frames drawn as a cel. Flat tones, ink, speed lines.
# ------------------------------------------------------------------------------
def lin(c):
    """sRGB 0..255 -> linear light."""
    c = np.asarray(c, dtype=np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# The current in four flat bands by strength (edges at these brightnesses), two
# hues by sign. The top bands pass 1 so that they glow.
A_EDGES = (0.14, 0.30, 0.50, 0.72)
A_GAIN = np.array([1.1, 1.2, 1.5, 2.2, 3.4], dtype=np.float32)[:, None]
A_POS = lin([(22, 58, 140), (44, 128, 232), (112, 208, 255), (208, 244, 255), (255, 255, 255)]) * A_GAIN
A_NEG = lin([(36, 44, 140), (92, 102, 234), (162, 174, 255), (226, 224, 255), (255, 255, 255)]) * A_GAIN
A_GLOW = lin([(170, 235, 255)])[0]
A_RING = lin([(200, 242, 255)])[0]
A_SKIN, A_SHADE, A_RIM, A_INK, A_KEY = (lin([c])[0] for c in (
    (246, 206, 178), (158, 108, 112), (204, 240, 255), (14, 10, 22), (96, 120, 168)))
A_SWOOSH, A_LINES = lin([(90, 200, 255)])[0], lin([(205, 235, 255)])[0]


def anime_ball(j):
    """The disc as a cel-shaded ball, native grid, linear RGB, y UP: the current
    in flat bands on a ball shaded in three flat tones, with a white core, one
    flat halo round it and a bright outline. The bands are the simulation; the
    shading, the core and the outline are drawn."""
    n = G["n"]
    fluid = np.isfinite(j)
    jz = np.where(fluid, j, 0.0)[::-1]
    b = np.minimum(np.arcsinh(np.abs(jz) / G["knee"]) / np.arcsinh(G["lim"] / G["knee"]), 1.0)
    q = sum(smoothstep(e - 0.012, e + 0.012, b) for e in A_EDGES)        # whole, except across an edge
    rgb = np.empty(q.shape + (3,), dtype=np.float32)
    for c in range(3):
        rgb[..., c] = np.where(jz > 0, np.interp(q, np.arange(5.0), A_POS[:, c]),
                               np.interp(q, np.arange(5.0), A_NEG[:, c]))
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx - 0.5 * (n - 1), yy - 0.5 * (n - 1)) / G["radius"]
    shade = 1.0 - 0.20 * smoothstep(0.62, 0.64, r) - 0.18 * smoothstep(0.86, 0.88, r)
    rgb = rgb * (shade + (1.0 - shade) * smoothstep(1.5, 2.5, q))[..., None]   # on the dim bands only
    core = 2.8 * np.exp(-(r / 0.105) ** 2) + 0.34 * (1.0 - smoothstep(0.215, 0.23, r))
    rgb = (rgb + core[..., None] * A_GLOW) * fluid.astype(np.float32)[::-1][..., None]
    ring = np.exp(-((r - 1.0) * G["radius"] / 1.8) ** 2)
    return rgb + (2.2 * ring)[..., None] * A_RING


def cel_hand(img, beta_deg, lift, grip, cx, cy):
    """Paints one hand over `img` (linear RGB), piece by piece from the far side:
    skin in two flat tones, lit where the surface faces the ball, a hard pale
    highlight inside the lit edge, a bold ink line inside every edge and a pale
    keyline outside the whole hand so that the ink reads against a dark frame.
    A finger is not drawn where it lies in the palm, so no line crosses the
    knuckles."""
    rb = SCENE[2]
    s = rb / HAND0[2]
    b = np.deg2rad(beta_deg)
    cb, sb = float(np.cos(b)), float(np.sin(b))
    xx, yy = G["grid"]
    x1, y1 = (xx - cx) / s, (yy - cy) / s
    parts = [s * d for d in hand_parts0(x1 * cb - y1 * sb + HAND0[0], x1 * sb + y1 * cb - lift / s + HAND0[1],
                                        float(np.clip(grip, 0.0, 1.0)) * CLAW)]
    union = parts[0]
    for d in parts[1:]:
        union = np.minimum(union, d)
    key = smoothstep(3.2, 2.0, union) * smoothstep(-0.5, 0.5, union) * 0.55
    img = img + (A_KEY - img) * key[..., None]
    base = parts[0]
    for k, d in enumerate(parts):
        cover = np.clip(0.5 - d, 0.0, 1.0)
        if k:
            cover = cover * smoothstep(-16.0, -3.0, base)
        rows, cols = np.flatnonzero(cover.max(axis=1) > 0.0), np.flatnonzero(cover.max(axis=0) > 0.0)
        if rows.size == 0:
            continue
        box = (slice(max(0, rows[0] - 20), min(H, rows[-1] + 21)), slice(max(0, cols[0] - 20), min(W, cols[-1] + 21)))
        dd, cv = d[box], cover[box]
        lx, ly = cx - xx[box], cy - yy[box]
        dist = np.maximum(np.hypot(lx, ly), 1.0)
        gy, gx = np.gradient(blur1(dd, 5.0))
        dot = (gx * lx + gy * ly) / (np.maximum(np.hypot(gx, gy), 1e-6) * dist)    # facing the ball: +1
        near = smoothstep(3.6 * rb, 1.5 * rb, dist)                                 # its light fades along the arm
        depth = np.maximum(-dd, 0.0)
        fill = A_SHADE + (A_SKIN - A_SHADE) * (smoothstep(-0.30, -0.22, dot) * near)[..., None]
        rw = 8.0 if k else 13.0
        rim = smoothstep(0.55, 0.63, dot) * smoothstep(rw + 1.5, rw, depth) * near
        fill = fill + (A_RIM - fill) * rim[..., None]
        ink = smoothstep(5.2, 4.0, depth)
        if k:
            ink = ink * smoothstep(-9.0, 1.0, base[box])
        fill = fill + (A_INK - fill) * ink[..., None]
        img[box] = img[box] * (1.0 - cv[..., None]) + fill * cv[..., None]
    return img


def swooshes(t, cx, cy, rb):
    """Sixteen bold, tapered streaks circling the ball counter-clockwise, flat
    cyan with a white stripe, each coming and going: linear RGB to add. Drawn."""
    rng = np.random.default_rng(11)
    q = 2
    body, stripe = Image.new("L", (W * q, H * q), 0), Image.new("L", (W * q, H * q), 0)
    db, ds = ImageDraw.Draw(body), ImageDraw.Draw(stripe)
    for _ in range(16):
        u = rng.random()
        r0 = 1.07 + 0.42 * u ** 1.3
        th0 = rng.random() * 2.0 * np.pi
        arc = np.deg2rad(26.0 + 46.0 * rng.random())
        pitch = 0.05 + 0.10 * rng.random()
        wmax = (4.0 + 7.0 * rng.random()) * (1.0 - 0.45 * u)
        rate = 1.25 / r0 ** 1.5
        live = 0.5 + 0.5 * np.sin(2.0 * np.pi * (rng.random() + (0.7 + 0.8 * rng.random()) * t))
        if live < 0.25:
            continue
        sgm = np.linspace(0.0, 1.0, 40)
        th = th0 + 2.0 * np.pi * rate * t - arc * sgm
        rr = rb * r0 * (1.0 + pitch * sgm)
        half = wmax * np.sin(np.pi * sgm) ** 0.75 * min(1.0, (live - 0.25) / 0.25)
        for lay, f in ((db, 1.0), (ds, 0.38)):
            outer = [(cx + (rr[k] + f * half[k]) * np.cos(th[k]), cy - (rr[k] + f * half[k]) * np.sin(th[k]))
                     for k in range(40)]
            inner = [(cx + (rr[k] - f * half[k]) * np.cos(th[k]), cy - (rr[k] - f * half[k]) * np.sin(th[k]))
                     for k in range(39, -1, -1)]
            lay.polygon([(x * q, y * q) for x, y in outer + inner], fill=255)
    a = np.asarray(body.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
    b = np.asarray(stripe.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
    return a[..., None] * A_SWOOSH * 0.9 + b[..., None] * 1.6


def speed_lines(seed, strength, cx, cy, rb):
    """Forty lines converging on the ball from the edges of the frame, redrawn
    from a new random draw every three frames and kept off the two blocks of
    text: linear RGB to add. Drawn."""
    rng = np.random.default_rng(seed)
    q = 2
    lay = Image.new("L", (W * q, H * q), 0)
    d = ImageDraw.Draw(lay)
    for _ in range(40):
        th = rng.random() * 2.0 * np.pi
        r_in = rb * (1.75 + 1.2 * rng.random() ** 1.5)
        half = np.deg2rad(0.12 + 0.55 * rng.random() ** 2)
        a = int(255 * (0.25 + 0.75 * rng.random()))
        pts = [(cx + r_in * np.cos(th), cy - r_in * np.sin(th)),
               (cx + 2400.0 * np.cos(th - half), cy - 2400.0 * np.sin(th - half)),
               (cx + 2400.0 * np.cos(th + half), cy - 2400.0 * np.sin(th + half))]
        d.polygon([(x * q, y * q) for x, y in pts], fill=a)
    a = np.asarray(lay.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
    xx, yy = G["grid"]
    a = a * (1.0 - smoothstep(470.0, 410.0, xx) * smoothstep(330.0, 290.0, yy))
    a = a * (1.0 - smoothstep(1500.0, 1560.0, xx) * smoothstep(120.0, 90.0, yy))
    return a[..., None] * A_LINES * strength


def anime_frame(i, j):
    """Frame i as a cel: sRGB uint8. As a stroke of the hand lands (every stroke
    but the first, which lands under the fade-in) the picture is shown in
    NEGATIVE for two frames, the speed lines flare, and everything but the text
    shakes for a third of a second."""
    cx, cy, rb = SCENE
    tau, t = i / G["fps"], G["tf"][i]
    ts, tr = STROKE["t_stroke"], STROKE["t_return"]
    n_stroke = int(np.floor(tau / (ts + tr) + 1e-9))        # not tau % cycle: 15.5 % 3.1 is 3.0999...
    t_in = max(0.0, tau - n_stroke * (ts + tr))
    landing = t_in < ts and n_stroke >= 1
    hit = float(np.exp(-t_in / 0.12)) if landing else 0.0
    impact = landing and t_in < 2.0 / G["fps"] - 1e-9
    if landing:
        shake = 9.0 * float(np.exp(-t_in / 0.10))
        cx += shake * np.cos(2.0 * np.pi * 23.0 * t_in)
        cy += shake * np.sin(2.0 * np.pi * 17.0 * t_in + 1.0)
    n, k = G["n"], rb / G["radius"]
    disc = anime_ball(j)
    if abs(k - 1.0) > 0.01:                       # not one cell per pixel: resized once
        s2 = int(round(n * k))
        disc = np.maximum(np.stack([resize(disc[..., c], (s2, s2), Image.LANCZOS) for c in range(3)], axis=-1), 0.0)
    else:
        s2, k = n, 1.0
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    x0, y0 = int(round(cx - G["c"] * k)), int(round(cy - G["c"] * k))
    ys, ye, xs, xe = max(0, y0), min(H, y0 + s2), max(0, x0), min(W, x0 + s2)
    canvas[ys:ye, xs:xe] = disc[ys - y0:ye - y0, xs - x0:xe - x0]
    canvas = canvas + swooshes(t, cx, cy, rb) + speed_lines(i // 3, 0.045 + 0.50 * hit, cx, cy, rb)
    src = canvas * smoothstep(0.20, 1.0, canvas.max(axis=-1))[..., None]
    canvas = canvas + blur(src, [(2.5, 0.40), (12.0, 0.30)]) + 0.22 * wide_blur(src, 48.0, 4)
    canvas = canvas + 0.10 * wide_blur(canvas, 110.0, 8) + BG
    for beta, lift, grip in hand_poses(tau):
        canvas = cel_hand(canvas, beta, lift, grip, cx, cy)
    img = tone(canvas)
    if impact:
        img = np.clip(1.0 - img.max(axis=-1, keepdims=True) ** 0.55, 0.0, 1.0) \
            * np.array([0.92, 0.97, 1.0], dtype=np.float32)
    if G["dither"]:
        img = img + (np.random.default_rng(i).random(img.shape, dtype=np.float32) - 0.5) / 255.0
    im = Image.fromarray((255.0 * np.clip(img, 0.0, 1.0) + 0.5).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    small = ImageFont.truetype(FONT_TEXT, 17, index=7)
    for row, line in enumerate(G["alines"]):
        d.text((56, 40 + 23 * row), line, font=small, fill=(60, 70, 90) if impact else (140, 152, 176))
    d.text((W - 56, 48), "t = %4.2f turnovers" % t, font=ImageFont.truetype(FONT_MONO, 26),
           fill=(30, 36, 50) if impact else (226, 232, 244), anchor="ra")
    return np.asarray(im)


# ------------------------------------------------------------------------------
#  One frame.
# ------------------------------------------------------------------------------
def init(cfg):
    G.update(cfg)
    side = cfg["side"]
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    c = 0.5 * (cfg["n"] - 1) * cfg["scale"] if cfg["scale"] == 1.0 else 0.5 * (side - 1)
    r = np.hypot(xx - c, yy - c) / cfg["radius"]
    G["limb"] = (0.50 + 0.50 * np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0)))[..., None]
    G["core"] = np.exp(-(r / 0.45) ** 2)[..., None] * CORE_RGB
    G["shell"] = np.exp(-((r - 1.0) * cfg["radius"] / 1.6) ** 2)[..., None] * SHELL_RGB
    G["c"] = c
    if cfg.get("hands"):
        hand_setup()


def field(j):
    """One frame of j (row y = 0 first, NaN outside the fluid) -> the disc in
    linear light, side x side, y UP."""
    n, side = G["n"], G["side"]
    fluid = np.isfinite(j)
    jz = np.where(fluid, j, 0.0)[::-1]
    alpha = fluid.astype(np.float32)[::-1]
    if G["scale"] == 1.0:
        jz, alpha = jz[:side, :side], alpha[:side, :side]
    else:
        jz = resize(jz, (side, side))
        alpha = np.clip(resize(alpha, (side, side)), 0.0, 1.0)
    b = np.minimum(np.arcsinh(np.abs(jz) / G["knee"]) / np.arcsinh(G["lim"] / G["knee"]), 1.0)
    rgb = np.where((jz > 0)[..., None], ramp(b, POS), ramp(b, NEG)) * alpha[..., None]
    if G["orb"]:
        rgb = rgb * G["limb"] + G["core"]
    return rgb + G["shell"]


def spring(u):
    """An underdamped step: 0 before u = 0, then to 1 with one overshoot."""
    u = np.maximum(u, 0.0)
    return np.where(u > 0.0, 1.0 - np.exp(-5.0 * u) * (np.cos(7.0 * u) + (5.0 / 7.0) * np.sin(7.0 * u)), 0.0)


def bump(tau, t0, t1, rise=0.07, fall=0.22):
    """1 between t0 and t1, with a quick rise and a slower fall."""
    return smoothstep(t0, t0 + rise, tau) * (1.0 - smoothstep(t1, t1 + fall, tau))


# The stirring hand strokes the ball one way, in contact, and is lifted back: degrees
# round the ball and seconds of video. PATCH is how far ahead of the hand's turn,
# in degrees, the part that touches (palm and fingers) sits.
STROKE = dict(a=58.0, b=124.0, t_stroke=2.5, t_return=0.6)
PATCH = 12.0


def _stroke_table():
    """Position along a stroke against its phase: speed rises over the first 15 %,
    is steady, and falls over the last 15 %."""
    u = np.linspace(0.0, 1.0, 2001)
    speed = smoothstep(0.0, 0.15, u) * (1.0 - smoothstep(0.85, 1.0, u))
    pos = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    return u, pos / pos[-1]


_U, _POS = _stroke_table()


def stroke(tau):
    """The stirring hand at video time tau: (turn in degrees, contact in [0, 1],
    phase of the return in [0, 1] or -1 during a stroke, which stroke it is)."""
    a, b, ts, tr = STROKE["a"], STROKE["b"], STROKE["t_stroke"], STROKE["t_return"]
    t = tau % (ts + tr)
    n = int(tau // (ts + tr))
    if t < ts:
        u = t / ts
        return (a + (b - a) * float(np.interp(u, _U, _POS)),
                float(smoothstep(0.0, 0.06, u) * (1.0 - smoothstep(0.94, 1.0, u))), -1.0, n)
    v = (t - ts) / tr
    return b + (a - b) * float(smoothstep(0.0, 1.0, v)), 0.0, v, n


def hand_poses(tau):
    """The two hands at video time tau (seconds): (turn in degrees round the ball,
    lift in pixels, grip in [0, 1]) each. THE HOLDING HAND strains: it sags under
    the ball for 2.9 s, trembling more and clenching as it goes, then heaves it
    back up with a bounce; that is made up. THE STIRRING HAND strokes the ball
    counter-clockwise, pressed against it at a steady speed, and is lifted back
    quickly for the next stroke; on every third return it shakes, as if burnt,
    and that is made up too. Its STROKE is not only drawn: `write_stir` exports
    it, and a run of demonstrator/mhd_decay -stir drags the fluid under the hand
    at the hand's speed while `contact` is on."""
    w = 2.0 * np.pi * tau
    # the holding hand
    c = (tau % 3.2) / 3.2                                    # the sag-and-heave cycle
    sag = np.where(c < 0.9, (c / 0.9) ** 1.6, 0.0) + (1.0 - spring((c - 0.9) * 3.2 / 0.16)) * (c >= 0.9)
    tremble = np.sin(8.7 * w) * (0.25 + 0.75 * sag)
    hold = (2.0 * np.sin(0.6 * w) + 0.7 * tremble, 17.0 * sag + 1.6 * tremble, 0.15 + 0.55 * sag)
    # the stirring hand
    beta, contact, back, n = stroke(tau)
    if back < 0.0:                                           # pressed on, fingers round the ball
        stir = (beta, 4.0 + 1.5 * np.sin(6.0 * w), 0.50 + 0.10 * np.sin(3.0 * w))
    else:                                                    # lifted clear and brought back
        arc = np.sin(np.pi * back)
        burnt = arc if n % 3 == 2 else 0.0
        stir = (beta + 6.0 * burnt * np.sin(11.0 * w), 5.0 + 58.0 * arc ** 1.2 + 30.0 * burnt,
                0.15 + 0.20 * burnt * (1.0 + np.sin(11.0 * w)))
    return tuple(float(v) for v in hold), tuple(float(v) for v in stir)


def write_stir(path, frames, binevery, fps, radius):
    """The stirring hand's track for demonstrator/mhd_decay -stir: one row per
    video frame of (step, polar angle of the contact in degrees, contact, feed).
    The polar angle is the simulation's, counter-clockwise from +x: the hand's
    turn is counted from under the ball, hence the -90. `feed` is the contact
    with a sign that alternates from stroke to stroke, for -stirfeed."""
    rows = [(i * binevery,) + stroke(i / fps)[:2] + (stroke(i / fps)[3],) for i in range(frames + 1)]
    with open(path, "w") as fh:
        fh.write("# step  phi_deg  contact  feed   (tools/anim_mhd_decay_glow.py --write-stir: "
                 "%d frames at %d fps, %d steps a frame)\n" % (frames, fps, binevery))
        for step, beta, contact, n in rows:
            fh.write("%d %.6f %.6f %.6f\n" % (step, beta - 90.0 + PATCH, contact,
                                              contact if n % 2 == 0 else -contact))
    om = max(abs(np.deg2rad(r1[1] - r0[1])) / binevery
             for r0, r1 in zip(rows[:-1], rows[1:]) if r0[2] > 0.0 or r1[2] > 0.0)
    print("  wrote %s: %d rows, one every %d steps; strokes of %.0f degrees in %.2f s, back in %.2f s"
          % (path, len(rows), binevery, STROKE["b"] - STROKE["a"], STROKE["t_stroke"], STROKE["t_return"]))
    print("  largest paddle speed in contact: %.4f lattice units at 0.94 R = %.1f cells (Ma %.3f)"
          % (om * 0.94 * radius, 0.94 * radius, om * 0.94 * radius * np.sqrt(3.0)))


def render(i):
    j = raw(G["frames"][i])
    if int(np.isfinite(j).sum()) < G["fluid"]:
        return None                               # a NaN inside the fluid: blown up
    if G["anime"]:
        out = anime_frame(i, j)
        fade = min(1.0, (i + 1) / max(1.0, 0.5 * G["fps"])) if G["fade"] else 1.0
        if fade < 1.0:
            out = (out * (fade * fade * (3.0 - 2.0 * fade))).astype(np.uint8)
        return out.tobytes()
    side = G["side"]
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    if G["hands"]:                                # the ball smaller, to leave room for two arms
        cx, cy, rb = SCENE
        k = rb / G["radius"]
        s2 = int(round(side * k))
        disc = field(j)
        x0, y0 = int(round(cx - G["c"] * k)), int(round(cy - G["c"] * k))
        canvas[y0:y0 + s2, x0:x0 + s2] = np.maximum(np.stack(
            [resize(disc[..., c], (s2, s2), Image.LANCZOS) for c in range(3)], axis=-1), 0.0)
    else:
        k = 1.0
        y0, x0 = (H - side) // 2, (W - side) // 2
        cx, cy, rb = x0 + G["c"], y0 + G["c"], G["radius"]
        cut = max(0, -y0)
        canvas[max(0, y0):max(0, y0) + side - 2 * cut, x0:x0 + side] = field(j)[cut:side - cut]
    if G["wind"]:
        canvas = canvas + wind(G["tf"][i], cx, cy, rb)

    src = canvas * smoothstep(0.20, 1.0, canvas.max(axis=-1))[..., None]
    canvas = canvas + blur(src, [(2.5 * k, 0.45), (12.0 * k, 0.32)]) + 0.22 * wide_blur(src, 48.0 * k, 4)
    if G["orb"]:
        canvas = canvas + 0.10 * wide_blur(canvas, 110.0 * k, 8)
    if G["hands"]:                                # in front of the glow, and lit by the ball
        for beta, lift, grip in hand_poses(i / G["fps"]):
            cover, light = hand(beta, lift, grip)
            canvas = canvas * (1.0 - 0.90 * cover) + light
    img = tone(canvas + BG)
    if G["dither"]:
        img = img + (np.random.default_rng(i).random(img.shape, dtype=np.float32) - 0.5) / 255.0
    im = Image.fromarray((255.0 * np.clip(img, 0.0, 1.0) + 0.5).astype(np.uint8), "RGB")

    d = ImageDraw.Draw(im)
    small, dim = ImageFont.truetype(FONT_TEXT, 17, index=7), (140, 152, 176)
    clock, mono = "t = %4.2f turnovers" % G["tf"][i], ImageFont.truetype(FONT_MONO, 26)
    if G["hands"]:                                # the arms leave by the bottom: text at the top
        d.text((W - 56, 48), clock, font=mono, fill=(226, 232, 244), anchor="ra")
        for row, line in enumerate(G["lines"]):
            d.text((56, 44 + 26 * row), line, font=small, fill=dim)
    else:
        d.text((56, H - 62), G["meta"], font=small, fill=dim)
        d.text((W - 56, H - 70 - (26 if G["note"] else 0)), clock, font=mono,
               fill=(226, 232, 244), anchor="ra")
        if G["note"]:
            d.text((W - 56, H - 62), G["note"], font=small, fill=dim, anchor="ra")
    out = np.asarray(im)
    fade = min(1.0, (i + 1) / max(1.0, 0.5 * G["fps"])) if G["fade"] else 1.0
    if fade < 1.0:
        out = (out * (fade * fade * (3.0 - 2.0 * fade))).astype(np.uint8)
    return out.tobytes()


# ------------------------------------------------------------------------------
#  The driver.
# ------------------------------------------------------------------------------
def main():
    if flag("--write-stir"):                      # the hand's track for a -stir run; no frames needed
        write_stir(arg("--write-stir", "", str), arg("--frames", 960, int),
                   arg("--binevery", 16, int), arg("--fps", 60, int), arg("--radius", 295.6))
        return
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else None
    fps = arg("--fps", 60, int)
    frames = sorted(glob.glob(os.path.join(d, "decay_*_j[0-9]*.bin")))
    if not frames:
        sys.exit("no decay_*_j*.bin in " + d)

    log = open(os.path.join(d, "log.txt")).read() if os.path.exists(os.path.join(d, "log.txt")) else ""
    m_n = re.search(r"N (\d+)\s+disc, half-width ([0-9.]+)", log)
    m_re = re.search(r"Re (\d+)\s+Pr_m", log)
    m_te = re.search(r"one turnover 2R/u0 = (\d+) steps; running (\d+)", log)
    m_sw = re.search(r"swirl: ([0-9.]+) of E_kin", log)
    m_wr = re.search(r"wrote (\d+) binary frames", log)
    if not (m_n and m_re and m_te):
        sys.exit("no log.txt of a disc run in %s: this script reads N, Re, the radius and "
                 "the turnover time from it" % d)
    n, radius = int(m_n.group(1)), float(m_n.group(2))
    if flag("--binevery"):
        binevery = arg("--binevery", 1, int)
    elif m_wr and int(m_wr.group(1)) > 1:
        binevery = int(m_te.group(2)) // (int(m_wr.group(1)) - 1)
    else:
        sys.exit("the run did not finish, so the frame stride is not in its log: give --binevery K")
    m_st = re.search(r"stir: paddle follows \S+, \d+ rows, one every (\d+) steps", log)
    if m_st and (flag("--hands") or flag("--anime")) and (int(m_st.group(1)) != binevery or flag("--every")):
        sys.exit("the run was stirred by a hand track with one row every %s steps, and these "
                 "frames are %d steps apart%s: the drawn hand would not be where the paddle was"
                 % (m_st.group(1), binevery, " and thinned by --every" if flag("--every") else ""))
    dtf = binevery / float(m_te.group(1))                 # turnovers per stored frame

    frames = [f for i, f in enumerate(frames) if i * dtf <= arg("--tmax", 1e30) + 1e-12]
    every = arg("--every", 1, int)
    frames, dtf = frames[::every], dtf * every
    tf = np.arange(len(frames)) * dtf

    j0 = raw(frames[0])
    if j0.shape != (n, n):
        sys.exit("frames are %dx%d and the log says N = %d" % (j0.shape + (n,)))
    if (flag("--hands") or flag("--anime")) and n <= 1081:
        side, scale, how = n, 1.0, ""             # kept as computed: the scene resizes it once
    elif n in (1080, 1081):
        side, scale, how = 1080, 1.0, ""
    elif n <= 545:
        side, scale, how = 2 * n, 2.0, " (enlarged ×2)"
    else:
        side, scale, how = 1080, 1080.0 / n, " (resized to 1080²)"

    stride = max(1, len(frames) // 400)
    p999 = [float(np.nanpercentile(np.abs(raw(f)), 99.9)) for f in frames[::stride]]
    lim = max(p999)
    knee = lim / 12.0
    print("  %d frames of %dx%d, one every %.5f turnovers, t_end = %.3f turnovers"
          % (len(frames), n, n, dtf, tf[-1]))
    print("  brightness saturates at |j| = %.4g (lattice), the largest p99.9 over the frames "
          "(at t = %.2f); knee %.4g" % (lim, tf[::stride][int(np.argmax(p999))], knee))

    # A run driven by the hand's track is not a decay, and the frame says which it is.
    fed = bool(re.search(r"feed: da/dt", log))
    what = ("magnetohydrodynamic flow in a disc, stirred%s under the stroking hand"
            % (" and fed" if fed else "")) if m_st else "decaying magnetohydrodynamic turbulence in a disc"
    meta = what + "  ·  Re = %s  ·  %d² cells%s" % (m_re.group(1), n, how)
    if m_sw:
        meta += "  ·  swirl %s" % m_sw.group(1).rstrip("0").rstrip(".")
    meta += "  ·  colour: electric current density, sign as hue  ·  M3LB"

    # What is drawn rather than simulated, said on the frame.
    anime = flag("--anime")                       # a look of its own, in the two-hand layout
    orb, hands, windy = flag("--orb"), flag("--hands") or anime, flag("--wind")
    drawn = [w for w, on in (("the ball's shading", orb), ("the wind", windy),
                             ("the hands", hands)) if on]
    if drawn == ["the ball's shading"]:
        note = "ball shading and aura are cosmetic"
    elif drawn:
        note = (", ".join(drawn[:-1]) + " and " + drawn[-1] if len(drawn) > 1 else drawn[0]) \
            + (" are" if len(drawn) > 1 else " is") + " drawn, not simulated"
    else:
        note = ""
    if m_st and hands:
        note += "; the stroke's push%s applied to the simulation" % (" and its flux are" if fed else " is")
    head, tail = meta.split("  \u00b7  colour: ")
    lines = [head + ("  \u00b7  %.2f pixels per cell" % (SCENE[2] / (radius * scale) * scale)),
             "colour: " + tail] + ([note] if note else [])

    # --anime: the text as a narrow block, clear of the swooshes round the ball.
    cells = "Re = %s  \u00b7  %d\u00b2 cells  \u00b7  M3LB" % (m_re.group(1), n)
    alines = textwrap.wrap(what, 40) + [cells, ""] \
        + textwrap.wrap("colour: electric current density, four flat bands, two hues for its sign", 40) + [""] \
        + textwrap.wrap("drawn: the hands, the lines, the wind, the ball's shading and its core", 40)
    if m_st:
        alines += textwrap.wrap("applied to the simulation: the stroke's push%s"
                                % (" and its flux" if fed else ""), 40)
    if abs(SCENE[2] / (radius * scale) - 1.0) > 0.01:
        alines.insert(alines.index(cells) + 1, "shown at %.2f pixels per cell" % (SCENE[2] / radius))

    cfg = dict(frames=frames, tf=tf, n=n, side=side, scale=scale, radius=radius * scale,
               lim=lim, knee=knee, fps=fps, orb=orb, hands=hands, wind=windy, meta=meta,
               note=note, lines=lines, anime=anime, alines=alines,
               dither=not flag("--no-dither"), fluid=int(np.isfinite(j0).sum()),
               fade=not (flag("--still") or flag("--still-t")))    # a still is not faded in

    if flag("--still") or flag("--still-t"):
        k = sys.argv.index("--still" if flag("--still") else "--still-t")
        init(cfg)
        want = sys.argv[k + 1].split(",")
        which = ([int(s) for s in want] if flag("--still") else
                 [int(np.argmin(np.abs(tf - float(s)))) for s in want])
        for i in which:
            i = min(i, len(frames) - 1)
            buf = render(i)
            path = "%s_%04d.png" % (sys.argv[k + 2], i)
            Image.frombytes("RGB", (W, H), buf).save(path)
            print("  wrote %s  (t = %.3f turnovers)" % (path, tf[i]))
        return

    if out is None:
        sys.exit("give an output .mp4, or --still i,j,k PREFIX")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(arg("--crf", 16, int)),
           "-pix_fmt", "yuv420p",
           "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
           "-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last, done = None, 0
    with Pool(arg("--jobs", 6, int), initializer=init, initargs=(cfg,)) as pool:
        for buf in pool.imap(render, range(len(frames)), chunksize=2):
            if buf is None:
                print("  non-finite frame %d: stopping there" % done)
                break
            enc.stdin.write(buf)
            last, done = buf, done + 1
            if done % 100 == 0:
                print("    %d/%d" % (done, len(frames)), flush=True)
    for _ in range(int(arg("--hold", 1.5) * fps)):
        enc.stdin.write(last)
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    print("  wrote %s: %d frames + %.1f s hold at %d fps = %.1f s"
          % (out, done, arg("--hold", 1.5), fps, done / fps + arg("--hold", 1.5)))


if __name__ == "__main__":
    main()
