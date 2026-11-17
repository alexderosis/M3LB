#!/usr/bin/env python3
"""Orszag-Tang as a dark, emissive animation: current sheets that glow, and the
magnetic field lines that reconnect through them.

Reads what `validation/orszag_tang -tmax T -movie DIR -nframes K` writes, plus
the run's own stdout saved as DIR/log.txt:

    DIR/ot_j_%04d.bin   int32 nx, int32 ny, then nx*ny float32, x fastest, row
                        y = 0 FIRST, in physical units (1/s)
    DIR/ot_series.dat   t  j_max  zeta_max  Ma_max, one row per probe
    DIR/log.txt         for dt, the frame stride, N and Re (else --dt-frame)

and pipes 1920x1080 RGB frames straight into ffmpeg. Two layouts: `panel` (the
default) puts the field on the left, 1080 pixels square, and a panel on the
right with the title, the colour key and max|j| against time; `full` fills the
frame with the field and keeps a title, the clock and a one-line key along the
bottom. `demonstrator/make_ot_anim.py` is the DIAGNOSTIC rendering of the same
files -- axes, a colour bar with numbers -- and stays the one to read a value
off. This one is for showing.

    python3 tools/anim_ot_glow.py runs/ot_glow/n1080 runs/ot_glow/ot_glow.mp4
    python3 tools/anim_ot_glow.py DIR --still 0,300,600 runs/ot_glow/still
    python3 tools/anim_ot_glow.py DIR out.mp4 --layout full --look fire --warp 4.5,11,4

Needs numpy and PIL, and ffmpeg for the video. No matplotlib, no scipy.

What is drawn, and the choices that could mislead.

  ONE CELL IS ONE PIXEL when the run has N = 1080, and nothing is interpolated.
  A coarser run (N < 1080, used to develop this script) is interpolated by
  zero-padding its Fourier transform, which is exact for a periodic field and
  rings beside a sheet the grid did not resolve; the footer then says so. A run
  with N a multiple of 1080 is rendered at N and box-averaged.

  THE FIELD LINES ARE NOT IN THE FILES; THEY ARE RECOVERED FROM j. In two
  dimensions B = (d_y A, -d_x A) and j = -lap A, so A follows from j by one
  division in Fourier space, exactly, the box being periodic; its mean is set
  to zero, which costs nothing because the mean of A is conserved by an
  incompressible periodic flow. Lines are drawn at FIXED levels of A, dA apart,
  for the whole run. That is the physics on show: A is carried by the flow
  apart from resistive slip, so a fixed level is a material line wherever the
  plasma is ideal, and where a line changes its connections it has reconnected.
  A line that shrinks to a point and vanishes is flux being annihilated at an
  O-point, not a rendering fault. The solver's j is a central difference of B
  and the inversion is spectral: at t = 0 the recovered A is checked against the
  initial condition 2 cos y + cos 2x and the difference is printed (it is
  O(dx^2): 1.35e-4 against a range of 6 at N = 540).

  ONE FIXED COLOUR MAPPING FOR THE WHOLE RUN, as in make_ot_anim.py and for its
  reason: a mapping renormalised frame by frame makes a decaying field look
  steady. Brightness is asinh(|j|/knee)/asinh(lim/knee): linear below `knee`,
  the 99.5th percentile of |j| at t = 0, and saturating at `lim`, the --pct
  percentile of |j| on the frame where max|j| peaks. Both are printed and `lim`
  is written on the colour key. The hue is the SIGN of j (warm out of the page,
  cool into it), and zero is black.

  THE GLOW IS DECORATION AND IS KEPT SMALL ENOUGH NOT TO MOVE ANYTHING. Bloom
  adds a blurred copy of the bright part of the picture back to it (two
  Gaussians, 3 and 14 pixels, periodic like the box). It widens the apparent
  halo of a sheet, not the sheet: the unblurred field is still drawn on top at
  full sharpness. Do not measure a sheet's thickness on a frame of this
  animation; --no-bloom removes it.

  --tmax CUTS BOTH THE SERIES AND THE FRAMES. A run that blew up must be cut
  before the blow-up or one non-finite probe becomes the reference frame; a
  non-finite frame stops the render either way.

  --every K USES EVERY K-th FRAME, so the same run plays K times faster without
  being rerun; the clock and the series follow. Frame indices given to --still
  count the frames of the VIDEO, after --every and --warp; --still-t takes
  times instead and picks the nearest. A still carries the video's half-second
  fade-in from black, so --still 0 is a black frame and not a failed render.

  --layout full TILES THE PERIODIC BOX. A 1920-pixel frame at one cell per pixel
  is 1.78 periods wide, so every structure within 420 cells of the left and
  right edges of the box appears twice. That is the box, not a larger
  simulation, and the strip along the bottom says so. The box's centre is put
  on the frame's centre.

  --look fire DROPS THE SIGN. It maps |j| through one ramp, violet to
  white-hot, with the same brightness law and the same `knee` and `lim` as the
  default look (`ember`, which keeps the sign as hue). Its field lines are twice
  as dense, softer, and fade out where neighbours come within a few pixels of
  each other, which is where the field is strongest -- so in this look the
  ABSENCE of lines is not weak field. Its bloom has a third, 45-pixel Gaussian.

  --warp T1,T2,SMAX IS A VARIABLE-SPEED PLAYBACK OF THE SAME FRAMES: one stored
  frame per video frame until t = T1, then a smooth rise to SMAX frames per
  video frame at t = T2, held to the end. A decaying run moves less and less
  per frame (here the frame-to-frame change falls fivefold from the first
  quarter to the last), so a uniform speed either drags at the end or stutters
  at the start. Each video frame is the NEAREST stored frame, never a blend of
  two, so nothing on screen is interpolated in time; the clock is the stored
  frame's own time and the speed is written beside the key.

  --inset tr|br PUTS max|j| AGAINST TIME IN A CORNER OF --layout full: top
  right, or bottom right above the clock. It is the panel's curve, drawn up to
  the frame's own time, on a darkened patch of the field; its time axis is
  SIMULATED time, so under --warp its head moves faster as the playback does.

usage: anim_ot_glow.py <dir> [out.mp4] [--fps 60] [--tmax S] [--pct 99.9]
           [--layout panel|full] [--look ember|fire] [--warp T1,T2,SMAX]
           [--inset tr|br]
           [--dA 0.25] [--dt-frame S] [--every K] [--jobs 6] [--hold 1.5]
           [--no-bloom] [--no-lines] [--still i,j,k PREFIX]
           [--still-t t1,t2 PREFIX] [--crf 16]
"""
import glob
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = 1080                 # the field, in pixels
W, H = 1920, 1080        # the frame
PW = W - S               # the panel's width
TWO_PI = 2.0 * np.pi

BG = np.array([0.0022, 0.0027, 0.0042])          # linear light, the whole frame

# Linear-light ramps against brightness b in [0, 1]. They pass 1 on purpose: the
# excess is what the bloom spreads and the tone curve then rolls off.
STOPS = np.array([0.00, 0.26, 0.45, 0.60, 0.75, 0.88, 1.00])
WARM = np.array([[0, 0, 0], [0.014, 0.002, 0.006], [0.075, 0.008, 0.012],
                 [0.30, 0.045, 0.016], [0.95, 0.26, 0.040], [1.80, 0.95, 0.28],
                 [2.6, 2.1, 1.3]])
COOL = np.array([[0, 0, 0], [0.002, 0.005, 0.020], [0.008, 0.022, 0.11],
                 [0.020, 0.080, 0.38], [0.040, 0.36, 1.05], [0.34, 1.10, 1.85],
                 [1.4, 2.2, 2.7]])
LINE_RGB = np.array([0.62, 0.72, 0.95]) * 0.10   # field lines, added on top
EXPOSURE = 1.15

# --look fire: |j| alone, violet to white-hot.
FIRE_STOPS = np.array([0.00, 0.22, 0.42, 0.58, 0.72, 0.86, 1.00])
FIRE = np.array([[0, 0, 0], [0.006, 0.002, 0.020], [0.050, 0.006, 0.090],
                 [0.32, 0.020, 0.20], [1.05, 0.16, 0.10], [2.0, 0.95, 0.20],
                 [3.0, 2.6, 2.0]])
THREAD_RGB = np.array([0.75, 0.80, 1.0], dtype=np.float32) * 0.045

# The bloom of each look: (sigma in pixels, weight) of each periodic Gaussian.
BLOOM = {"ember": [(3.0, 0.42), (14.0, 0.30)],
         "fire": [(2.5, 0.45), (12.0, 0.32), (45.0, 0.18)]}

X0 = S // 2 - W // 2     # --layout full: the box's centre on the frame's centre

FONT_TEXT = "/System/Library/Fonts/Avenir Next.ttc"     # 2 demi, 5 medium, 7 regular
FONT_MATH = "/System/Library/Fonts/HelveticaNeue.ttc"   # 0 regular: has the nabla
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
    """Linear light in [0, 1] -> sRGB."""
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055)


def tone(lin):
    """Linear HDR -> display sRGB in [0, 1]: an exponential shoulder, then sRGB."""
    return srgb(1.0 - np.exp(-EXPOSURE * np.maximum(lin, 0.0)))


def ramp(b, table, stops=STOPS):
    out = np.empty(b.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(b, stops, table[:, c])
    return out


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------
#  The field: j -> (j, A, |B|) on the render grid, then colour.
# ------------------------------------------------------------------------------
class Spectral:
    def __init__(self, n, r):
        self.n, self.r = n, r
        self.ky = np.fft.fftfreq(n, 1.0 / n)[:, None]
        self.kx = np.fft.rfftfreq(n, 1.0 / n)[None, :]
        k2 = self.kx ** 2 + self.ky ** 2
        k2[0, 0] = 1.0
        self.ik2 = 1.0 / k2
        self.ik2[0, 0] = 0.0                      # the gauge: <A> = 0

    def pad(self, h):
        """Spectrum on n -> spectrum on r > n, dropping n's Nyquist modes."""
        n, r = self.n, self.r
        kp = (n - 1) // 2
        out = np.zeros((r, r // 2 + 1), dtype=complex)
        out[:kp + 1, :kp + 1] = h[:kp + 1, :kp + 1]
        out[-kp:, :kp + 1] = h[-kp:, :kp + 1]
        return out * (float(r) / float(n)) ** 2

    def fields(self, j):
        n, r = self.n, self.r
        jh = np.fft.rfft2(j.astype(np.float64))
        ah = jh * self.ik2
        bxh = 1j * self.ky * ah                   # B_x =  d_y A
        byh = -1j * self.kx * ah                  # B_y = -d_x A
        if r == n:
            inv = lambda h: np.fft.irfft2(h, s=(n, n))
            jr = j.astype(np.float64)
        else:
            inv = lambda h: np.fft.irfft2(self.pad(h), s=(r, r))
            jr = inv(jh)
        return jr, inv(ah), np.hypot(inv(bxh), inv(byh))


def bloom_kernel(r, parts):
    """Fourier transform of the bloom's periodic Gaussians, weights included."""
    ky = np.fft.fftfreq(r)[:, None] * TWO_PI
    kx = np.fft.rfftfreq(r)[None, :] * TWO_PI
    k2 = kx ** 2 + ky ** 2
    return sum(w * np.exp(-0.5 * sig * sig * k2) for sig, w in parts).astype(np.float32)


def threads(A, B, dA, q, px):
    """Soft lines on the levels (m + 1/2) dA of A: Gaussian across, 0.9 output
    pixels, and faded out where neighbouring lines are fewer than 6 output
    pixels apart (gone at 2.5), where they would alias. q = render / output."""
    f = A / dA
    grad = np.maximum(B, 1e-3) * px
    dist = np.abs(f - np.floor(f) - 0.5) * dA / grad
    return (np.exp(-(dist / (0.9 * q)) ** 2)
            * smoothstep(2.5 * q, 6.0 * q, dA / grad)).astype(np.float32)


def field_rgb(j):
    """One frame of j (row y = 0 first) -> linear-light RGB, S x S, y UP."""
    sp, r = G["spec"], G["r"]
    jr, A, B = sp.fields(j)

    b = np.arcsinh(np.abs(jr) / G["knee"]) / np.arcsinh(G["lim"] / G["knee"])
    b = np.minimum(b, 1.0)
    if G["look"] == "fire":
        rgb = ramp(b, FIRE, FIRE_STOPS)
        if G["lines"]:
            rgb = rgb + threads(A, B, 0.5 * G["dA"], r / S, TWO_PI / r)[..., None] * THREAD_RGB
    else:
        rgb = np.where((jr > 0)[..., None], ramp(b, WARM), ramp(b, COOL))

    if G["lines"] and G["look"] != "fire":
        # Distance to the nearest level of A, in pixels, to first order: the
        # level is dl away in A and A changes by |B| px per pixel. A line of
        # width w is then the box-filtered indicator of dist < w/2.
        px = TWO_PI / r
        f = A / G["dA"]
        dl = np.abs(f - np.floor(f) - 0.5) * G["dA"]
        dist = dl / (np.maximum(B, 1e-3) * px)
        w = 1.25 * r / S
        mask = np.clip(0.5 + 0.5 * w - dist, 0.0, 1.0).astype(np.float32)
        rgb = rgb + mask[..., None] * LINE_RGB.astype(np.float32)

    if r != S:                                    # N a multiple of 1080
        q = r // S
        rgb = rgb.reshape(S, q, S, q, 3).mean(axis=(1, 3))

    if G["bloom"]:
        lum = rgb.max(axis=-1)
        t = np.clip((lum - 0.22) / 0.78, 0.0, 1.0)
        src = rgb * (t * t * (3.0 - 2.0 * t))[..., None]
        for c in range(3):
            rgb[..., c] += np.fft.irfft2(np.fft.rfft2(src[..., c]) * G["bk"], s=(S, S))

    return rgb[::-1] + BG.astype(np.float32)


# ------------------------------------------------------------------------------
#  The panel.
# ------------------------------------------------------------------------------
def font(path, size, index=0):
    return ImageFont.truetype(path, size, index=index)


def wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=fnt) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def spaced(draw, xy, text, fnt, fill, track):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + track


def nice_ticks(top):
    """Ticks from 0 at a 1-2-5 step, the one that comes nearest four intervals."""
    e = np.floor(np.log10(top))
    steps = [m * 10.0 ** k for k in (e - 1, e, e + 1) for m in (1.0, 2.0, 5.0)]
    step = min(steps, key=lambda s: abs(top / s - 4.0))
    return np.arange(0.0, top + 1e-9, step)


PLOT = dict(x0=84, x1=PW - 64, y0=676, y1=896)    # the series, in panel pixels


def panel_static(cfg):
    """Everything on the panel that does not change: sRGB uint8, H x PW x 3."""
    bg = tuple(int(round(255 * v)) for v in srgb(BG))
    im = Image.new("RGB", (PW, H), bg)
    d = ImageDraw.Draw(im)
    x0, x1 = 64, PW - 64
    white, soft, dim = (236, 240, 248), (176, 186, 204), (112, 122, 142)

    spaced(d, (x0, 92), "MAGNETOHYDRODYNAMICS", font(FONT_TEXT, 19, 2), (120, 150, 210), 3.2)
    d.text((x0 - 3, 124), "Orszag–Tang vortex", font=font(FONT_TEXT, 66, 2), fill=white)

    body = font(FONT_TEXT, 27, 7)
    y = 238
    for para in ("A smooth magnetic field, stirred by a smooth flow, collapses into "
                 "thin sheets of electric current.",
                 "Field lines ride with the plasma — except inside those sheets, "
                 "where they break and reconnect."):
        for line in wrap(d, para, body, x1 - x0):
            d.text((x0, y), line, font=body, fill=soft)
            y += 39
        y += 14

    # The colour key: the same ramps and tone curve as the field, so the bar IS
    # the mapping. Its ends are |j| = lim, which is written under it.
    ky, kh = 456, 12
    small = font(FONT_TEXT, 20, 7)
    if cfg["look"] == "fire":
        lin = ramp(np.linspace(0.0, 1.0, x1 - x0), FIRE, FIRE_STOPS)
        d.text((x0, ky + 24), "0", font=small, fill=soft)
        d.text((x1, ky + 24), "%.3g" % cfg["lim"], font=small, fill=soft, anchor="ra")
        d.text(((x0 + x1) // 2, ky + 24), "current  | j | = | ∇ × B |",
               font=font(FONT_MATH, 20, 0), fill=white, anchor="ma")
    else:
        b = np.abs(np.linspace(-1.0, 1.0, x1 - x0))
        lin = np.where((np.arange(x1 - x0) >= (x1 - x0) // 2)[:, None],
                       ramp(b, WARM), ramp(b, COOL))
        d.text((x0, ky + 24), "into the page", font=small, fill=soft)
        d.text((x1, ky + 24), "out of the page", font=small, fill=soft, anchor="ra")
        d.text(((x0 + x1) // 2, ky + 24), "current  j = ∇ × B",
               font=font(FONT_MATH, 20, 0), fill=white, anchor="ma")
    bar = (255 * tone(lin + BG)).astype(np.uint8)
    im.paste(Image.fromarray(np.repeat(bar[None], kh, axis=0), "RGB"), (x0, ky))
    d.text((x0, ky + 54), "asinh scale, linear below |j| = %.3g and saturated at %.3g"
           % (cfg["knee"], cfg["lim"]), font=font(FONT_TEXT, 17, 7), fill=dim)
    if cfg["lines"]:
        ly = ky + 98
        d.line([(x0, ly), (x0 + 54, ly)], fill=(150, 170, 215), width=2)
        d.text((x0 + 70, ly), "magnetic field lines", font=small, fill=soft, anchor="lm")

    # The series' frame: a title, a baseline, ticks. The curve is drawn per frame.
    p = PLOT
    d.text((x0, p["y0"] - 58), "peak current", font=font(FONT_TEXT, 24, 5), fill=white)
    d.text((x0 + 158, p["y0"] - 54), "max |j|", font=font(FONT_MATH, 20, 0), fill=soft)
    tick = font(FONT_MONO, 15)
    for v in nice_ticks(cfg["jtop"]):
        yy = p["y1"] - (p["y1"] - p["y0"]) * v / cfg["jtop"]
        d.line([(p["x0"], yy), (p["x1"], yy)], fill=(30, 34, 46), width=1)
        d.text((p["x0"] - 10, yy), "%g" % v, font=tick, fill=dim, anchor="rm")
    for v in nice_ticks(cfg["tend"]):
        xx = p["x0"] + (p["x1"] - p["x0"]) * v / cfg["tend"]
        d.text((xx, p["y1"] + 10), "%g" % v, font=tick, fill=dim, anchor="ma")
    d.text((p["x1"], p["y1"] + 34), "time (s)", font=font(FONT_TEXT, 17, 7), fill=dim, anchor="ra")

    foot = font(FONT_TEXT, 17, 7)
    for i, line in enumerate(cfg["footer"]):
        d.text((x0, H - 78 + 24 * i), line, font=foot, fill=dim)
    return np.asarray(im).copy()


def curve_ink(i, pw, ph, pad, lw=2.6, dot=5.5, blur=7):
    """max|j| up to frame i's time as light to screen onto a pw x ph plot (plus
    `pad` all round): the curve, a dot at its head, and a blurred copy under
    both. Drawn 3x oversize and reduced."""
    cfg, q = G, 3
    t = cfg["tf"][i]
    lay = Image.new("L", ((pw + 2 * pad) * q, (ph + 2 * pad) * q), 0)
    dl = ImageDraw.Draw(lay)
    ts, js = cfg["ts"], cfg["js"]
    k = int(np.searchsorted(ts, t, side="right"))
    xs = np.concatenate([[0.0], ts[:k], [t]])
    ys = np.concatenate([[cfg["j0"]], js[:k], [np.interp(t, np.r_[0.0, ts], np.r_[cfg["j0"], js])]])
    pts = [((pad + pw * a / cfg["tend"]) * q, (pad + ph * (1.0 - b / cfg["jtop"])) * q)
           for a, b in zip(xs, ys)]
    if len(pts) > 1:
        dl.line(pts, fill=255, width=int(lw * q), joint="curve")
    hx, hy = pts[-1]
    dl.ellipse([hx - dot * q, hy - dot * q, hx + dot * q, hy + dot * q], fill=255)
    lay = lay.resize((pw + 2 * pad, ph + 2 * pad), Image.LANCZOS)
    core = np.asarray(lay, dtype=np.float32) / 255.0
    halo = np.asarray(lay.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255.0
    ink = core[..., None] * np.array([1.0, 0.86, 0.66], dtype=np.float32) \
        + 0.9 * halo[..., None] * np.array([1.0, 0.45, 0.12], dtype=np.float32)
    return np.clip(ink, 0.0, 1.0)


def panel(i):
    """The panel at frame i: static layer plus the curve so far and the clock."""
    cfg, p = G, PLOT
    out = G["panel"].astype(np.float32) / 255.0
    t = cfg["tf"][i]

    pw, ph = p["x1"] - p["x0"], p["y1"] - p["y0"]
    pad = 24
    ink = curve_ink(i, pw, ph, pad)
    ya, xa = p["y0"] - pad, p["x0"] - pad
    reg = out[ya:ya + ink.shape[0], xa:xa + ink.shape[1]]
    reg[:] = 1.0 - (1.0 - reg) * (1.0 - ink)                         # screen blend

    im = Image.fromarray((255 * np.clip(out, 0, 1)).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    d.text((PW - 64, PLOT["y0"] - 62), "t = %4.2f s" % t, font=font(FONT_MONO, 30),
           fill=(236, 240, 248), anchor="ra")
    return np.asarray(im)


# ------------------------------------------------------------------------------
#  One frame, and the driver.
# ------------------------------------------------------------------------------
def init(cfg):
    G.update(cfg)
    n = cfg["n"]
    r = n if (n >= S and n % S == 0) else S
    G["r"] = r
    G["spec"] = Spectral(n, r)
    G["bk"] = bloom_kernel(S, BLOOM[cfg["look"]])
    if cfg["layout"] == "full":
        G["cols"] = (np.arange(W) + X0) % S
        # A scrim along the bottom, in display space, so that the strip reads.
        y = np.arange(H, dtype=np.float32)[:, None, None]
        G["scrim"] = 1.0 - 0.72 * smoothstep(H - 190.0, H - 20.0, y)
        if cfg["inset"]:
            G["scrim"] = G["scrim"] * inset_static(cfg)
    else:
        G["panel"] = panel_static(cfg)


# --inset: max|j| against time in a corner of --layout full. The plot area, and
# the room its backdrop leaves round it for the title and the tick labels.
INSET_W, INSET_H = 470, 150
INSET_ROOM = dict(left=52, right=22, top=44, bottom=34)


def inset_static(cfg):
    """Places the inset, keeps its unchanging ink (title, ticks, hairlines) as an
    RGBA layer in G, and returns the darkening under it, H x W x 1."""
    m = INSET_ROOM
    x1 = W - 56 - m["right"]
    x0 = x1 - INSET_W
    y0 = 40 + m["top"] if cfg["inset"] == "tr" else H - 128 - m["bottom"] - INSET_H
    y1 = y0 + INSET_H
    G["inset_xy"] = (x0, y0)

    back = Image.new("L", (W, H), 0)
    ImageDraw.Draw(back).rounded_rectangle(
        [x0 - m["left"], y0 - m["top"], x1 + m["right"], y1 + m["bottom"]], radius=16, fill=255)
    back = np.asarray(back.filter(ImageFilter.GaussianBlur(6)), dtype=np.float32) / 255.0

    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    white, soft, dim = (236, 240, 248, 255), (176, 186, 204, 255), (128, 138, 158, 255)
    d.text((x0, y0 - 34), "peak current", font=font(FONT_TEXT, 19, 5), fill=white)
    d.text((x0 + 126, y0 - 31), "max |j|", font=font(FONT_MATH, 16, 0), fill=soft)
    tick = font(FONT_MONO, 12)
    for v in nice_ticks(cfg["jtop"])[::2]:
        yy = y1 - INSET_H * v / cfg["jtop"]
        d.line([(x0, yy), (x1, yy)], fill=(255, 255, 255, 26), width=1)
        d.text((x0 - 8, yy), "%g" % v, font=tick, fill=dim, anchor="rm")
    for v in nice_ticks(cfg["tend"]):
        d.text((x0 + INSET_W * v / cfg["tend"], y1 + 8), "%g" % v, font=tick, fill=dim,
               anchor="ma")
    d.text((x1, y1 + 7), "time (s)", font=font(FONT_TEXT, 13, 7), fill=dim, anchor="ra")
    G["inset_layer"] = lay
    return (1.0 - 0.62 * back)[..., None]


def strip(img, i):
    """--layout full: the title, the clock and the key along the bottom."""
    im = Image.fromarray(img, "RGB")
    if G["inset"]:
        im.paste(G["inset_layer"], (0, 0), G["inset_layer"])
    d = ImageDraw.Draw(im)
    white, dim = (238, 241, 248), (150, 160, 180)
    d.text((56, H - 104), "Orszag–Tang vortex", font=font(FONT_TEXT, 40, 2), fill=white)
    d.text((58, H - 52), G["meta"], font=font(FONT_TEXT, 17, 7), fill=dim)
    d.text((W - 56, H - 96), "t = %5.2f s" % G["tf"][i], font=font(FONT_MONO, 34),
           fill=white, anchor="ra")
    key = G["key"]
    if G["warped"]:
        key += "  ·  playback ×%.1f" % G["speed"][i]
    d.text((W - 56, H - 52), key, font=font(FONT_TEXT, 17, 7), fill=dim, anchor="ra")
    return np.asarray(im)


def render(i):
    j = raw(G["frames"][G["src"][i]])
    if not np.isfinite(j).all():
        return None
    f = tone(field_rgb(j))
    if G["layout"] == "full":
        f = f[:, G["cols"]] * G["scrim"]
        if G["inset"]:
            pad = 16
            ink = curve_ink(i, INSET_W, INSET_H, pad, lw=2.2, dot=4.5, blur=6)
            xa, ya = G["inset_xy"][0] - pad, G["inset_xy"][1] - pad
            reg = f[ya:ya + ink.shape[0], xa:xa + ink.shape[1]]
            reg[:] = 1.0 - (1.0 - np.clip(reg, 0.0, 1.0)) * (1.0 - ink)   # screen blend
    if G["dither"]:
        f = f + (np.random.default_rng(i).random(f.shape, dtype=np.float32) - 0.5) / 255.0
    f = (255.0 * np.clip(f, 0.0, 1.0) + 0.5).astype(np.uint8)
    if G["layout"] == "full":
        out = strip(f, i)
    else:
        out = np.empty((H, W, 3), dtype=np.uint8)
        out[:, :S] = f
        out[:, S:] = panel(i)
    fade = min(1.0, (i + 1) / max(1.0, 0.5 * G["fps"]))
    if fade < 1.0:
        out = (out * (fade * fade * (3.0 - 2.0 * fade))).astype(np.uint8)
    return out.tobytes()


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else None
    fps = arg("--fps", 60, int)
    frames = sorted(glob.glob(os.path.join(d, "ot_j_*.bin")))
    if not frames:
        sys.exit("no ot_j_*.bin in " + d)
    series = np.loadtxt(os.path.join(d, "ot_series.dat"))
    ts, js = series[:, 0], series[:, 1]

    # The frame clock and the footer come from the run's own header.
    log = ""
    if os.path.exists(os.path.join(d, "log.txt")):
        log = open(os.path.join(d, "log.txt")).read()
    m_dt = re.search(r"dt = ([0-9.eE+-]+) s", log)
    m_fs = re.search(r"movie: \d+ frames every (\d+) steps", log)
    m_op = re.search(r"operator = (.*?)\s+N = (\d+)\s+Re = (\d+)", log)
    if flag("--dt-frame"):
        dtf = arg("--dt-frame", 0.0)
    elif m_dt and m_fs:
        dtf = float(m_dt.group(1)) * int(m_fs.group(1))
    else:
        sys.exit("no log.txt with dt and the frame stride in %s: give --dt-frame" % d)

    tmax = arg("--tmax", ts[-1])
    bad = ~np.isfinite(js)
    if bad.any():
        tmax = min(tmax, ts[np.argmax(bad)] - 1e-9)
        print("  non-finite probe at t = %.3f s: cut before it" % ts[np.argmax(bad)])
    keep = ts <= tmax + 1e-12
    ts, js = ts[keep], js[keep]
    frames = [f for i, f in enumerate(frames) if i * dtf <= tmax + 1e-12]
    every = arg("--every", 1, int)                # play K times faster: every K-th frame
    frames, dtf = frames[::every], dtf * every

    # Which stored frame each video frame shows, and how fast it is playing.
    warped = flag("--warp")
    if warped:
        t1, t2, smax = [float(v) for v in sys.argv[sys.argv.index("--warp") + 1].split(",")]
        if not (0.0 <= t1 < t2 and smax >= 1.0):
            sys.exit("--warp T1,T2,SMAX needs 0 <= T1 < T2 and SMAX >= 1")
        pos, speed = [0.0], []
        while True:
            u = min(max((pos[-1] * dtf - t1) / (t2 - t1), 0.0), 1.0)
            speed.append(1.0 + (smax - 1.0) * u * u * (3.0 - 2.0 * u))
            if pos[-1] + speed[-1] > len(frames) - 1:
                break
            pos.append(pos[-1] + speed[-1])
        src, speed = np.rint(pos).astype(int), np.array(speed)
    else:
        src, speed = np.arange(len(frames)), np.ones(len(frames))
    tf = src * dtf

    j0 = raw(frames[0])
    n = j0.shape[0]
    if j0.shape[0] != j0.shape[1] or not (n <= S or n % S == 0):
        sys.exit("frames are %dx%d: need a square N <= %d or a multiple of it" % (j0.shape + (S,)))
    ipk = int(np.argmax(js))
    kpk = int(np.clip(round(ts[ipk] / dtf), 0, len(frames) - 1))
    pct = arg("--pct", 99.9)
    apk = np.abs(raw(frames[kpk]))
    lim = float(np.percentile(apk, pct))
    knee = float(np.percentile(np.abs(j0), 99.5))
    print("  %d stored frames of %dx%d, dt_frame = %.5f s; %d video frames, t_end = %.3f s"
          % (len(frames), n, n, dtf, len(src), tf[-1]))
    if warped:
        print("  --warp: x1 to t = %g s (video frame %d), x%g from t = %g s (video frame %d)"
              % (t1, int(np.searchsorted(tf, t1)), smax, t2, int(np.searchsorted(tf, t2))))
    print("  max|j| peaks at %.4g (1/s) at t = %.2f s (frame %d)" % (js[ipk], ts[ipk], kpk))
    print("  brightness saturates at |j| = %.4g = p%g of that frame (its p100 = %.4g)"
          % (lim, pct, apk.max()))
    print("  asinh knee %.4g = p99.5 of |j| at t = 0" % knee)

    # The recovered potential against the initial condition it must reproduce.
    sp = Spectral(n, n)
    _, A0, _ = sp.fields(j0)
    yy, xx = np.meshgrid(np.arange(n) * TWO_PI / n, np.arange(n) * TWO_PI / n, indexing="ij")
    err = np.abs(A0 - (2.0 * np.cos(yy) + np.cos(2.0 * xx))).max()
    print("  A recovered from j at t = 0: max error %.2e against 2 cos y + cos 2x "
          "(range 6)" % err)
    if err > 0.05:
        sys.exit("  the t = 0 frame is not the Orszag-Tang initial condition in the "
                 "orientation and units this script assumes; refusing to draw field lines")

    if m_op:
        lat = "D3Q27 + D3Q7 (nz = 1)" if "D3Q27" in m_op.group(1) else "D2Q9 + D2Q5"
        op = "BGK" if "BGK" in m_op.group(1) else "central moments"
        foot = ["M3LB  ·  lattice Boltzmann, %s, %s" % (op, lat),
                "%d² cells  ·  Re = %s  ·  Pm = 1" % (n, m_op.group(3))
                + ("" if n % S == 0 else "  ·  interpolated to 1080²")]
    else:
        foot = ["M3LB  ·  lattice Boltzmann", "%d² cells" % n]

    look = arg("--look", "ember", str)
    layout = arg("--layout", "panel", str)
    if look not in BLOOM or layout not in ("panel", "full"):
        sys.exit("--look is ember or fire, --layout is panel or full")
    inset = arg("--inset", "", str)
    if inset not in ("", "tr", "br") or (inset and layout != "full"):
        sys.exit("--inset is tr or br, and belongs to --layout full")
    # The strip of --layout full: what was computed, and what the colours mean.
    meta = "magnetohydrodynamic turbulence  ·  Re = %s  ·  %d² cells%s, periodic box shown " \
           "%.2f periods wide  ·  M3LB" % (m_op.group(3) if m_op else "?", n,
                                           "" if n % S == 0 else " interpolated to 1080²",
                                           W / S)
    key = ("|j|, asinh scale saturating at %.3g" if look == "fire" else
           "current density j, asinh scale saturating at |j| = %.3g") % lim

    cfg = dict(frames=frames, src=src, speed=speed, warped=warped, tf=tf, ts=ts, js=js,
               n=n, lim=lim, knee=knee, fps=fps, look=look, layout=layout, inset=inset,
               meta=meta,
               key=key, dA=arg("--dA", 0.25), lines=not flag("--no-lines"),
               bloom=not flag("--no-bloom"), dither=not flag("--no-dither"),
               jtop=1.08 * float(js.max()), j0=float(np.abs(j0).max()),
               tend=float(tf[-1]), footer=foot)

    if flag("--still") or flag("--still-t"):
        k = sys.argv.index("--still" if flag("--still") else "--still-t")
        init(cfg)
        want = sys.argv[k + 1].split(",")
        which = ([int(s) for s in want] if flag("--still") else
                 [int(np.argmin(np.abs(tf - float(s)))) for s in want])
        for i in which:
            i = min(i, len(src) - 1)
            buf = render(i)
            path = "%s_%04d.png" % (sys.argv[k + 2], i)
            Image.frombytes("RGB", (W, H), buf).save(path)
            print("  wrote " + path + "  (t = %.3f s)" % tf[i])
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
        for buf in pool.imap(render, range(len(src)), chunksize=2):
            if buf is None:
                print("  non-finite frame %d: stopping there" % done)
                break
            enc.stdin.write(buf)
            last, done = buf, done + 1
            if done % 100 == 0:
                print("    %d/%d" % (done, len(src)), flush=True)
    for _ in range(int(arg("--hold", 1.5) * fps)):
        enc.stdin.write(last)
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    print("  wrote %s: %d frames + %.1f s hold at %d fps = %.1f s"
          % (out, done, arg("--hold", 1.5), fps, done / fps + arg("--hold", 1.5)))


if __name__ == "__main__":
    main()
