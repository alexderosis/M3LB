#!/usr/bin/env python3
"""A single vortex in a weak magnetic field, as a dark, emissive animation: the
vortex in colour and the field lines it winds up, expels and is finally torn
apart by.

Reads what `validation/vortex_disruption -frames K -out DIR` writes, plus the
run's stdout saved as DIR/log.txt:

    DIR/w_%04d.bin, DIR/j_%04d.bin   int32 nx, int32 ny, then nx*ny float32,
                       x fastest, row y = 0 FIRST: the vorticity in eps/R0 and
                       the current in B0/R0
    DIR/series.dat     the case's probes; column 6 is `surv`
    DIR/log.txt        R0, the box, Re, Rm, M, dt and the number of steps

and pipes 1920x1080 RGB frames into ffmpeg. Two layouts: `pair` puts two runs
side by side, 960 pixels square each, over a strip with the title, both
survival curves and the clock; `single` puts one run on the left, 1080 square,
and a panel on the right.

    python3 tools/anim_vortex_disruption.py runs/vortex_anim/A runs/vortex_anim/B \\
        --layout pair out.mp4
    python3 tools/anim_vortex_disruption.py DIR_A DIR_B --still-t 10,95 PREFIX
    python3 tools/anim_vortex_disruption.py DIR --sheet 10,40,85,100 sheet.png

Needs numpy and PIL, and ffmpeg for the video. No matplotlib, no scipy.

THE FILM, 2026-11-22 (runs/vortex_anim/vortex_disruption_pair_duo.mp4, 22.2 s):
two runs of about 21 min each on 4 threads, 5.8 GB of frames each,

    for X in 0.625 2.5; do
      ./build/validation/vortex_disruption -r0 40 -nr 20 -re 2000 -rm 2000 \\
          -M $(python3 -c "print((${X}/2000)**0.5)") -lat d2q9 -tmax 151.2 \\
          -probe 0.5 -frames 1200 -out runs/vortex_anim/rm2000_r040_x$X \\
          --kokkos-num-threads=4 > runs/vortex_anim/rm2000_r040_x$X/log.txt
    done

then the pair, look `duo`, about 4 min with 6 jobs. PASS -lat d2q9: the case
defaults to D3Q27 + D3Q7, about 3.7x slower and not the lattice of its map.
The looks: `vort` the vorticity, `current` the current by sign, `fire` |j|
violet to white-hot, `duo` the vorticity dimmed with |j| glowing over it.

What is drawn, and the choices that could mislead.

  THE CENTRAL --crop R0 OF THE BOX IS SHOWN (12 by default, of 20), centred on
  the vortex, and it is INTERPOLATED: by zero-padding the Fourier transform of
  the whole periodic box, which is exact for a band-limited periodic field and
  rings beside a sheet the grid did not resolve. At R0 = 40 and 960 pixels that
  is a factor of exactly 2.

  THE FIELD LINES ARE NOT IN THE FILES; THEY ARE RECOVERED FROM j. In two
  dimensions B = (d_y A, -d_x A) and j = -lap A, so the periodic part of A is
  j/k^2 in Fourier space; the applied field B0 along x adds A = y. Lines are
  drawn at FIXED levels of A, dA apart (in B0 R0), for the whole run, so far
  from the vortex they are straight, dA apart, and a line is a material line
  wherever the field is frozen in. Where neighbouring lines come within a few
  pixels they fade out -- where the field is strongest -- so there the ABSENCE
  of lines is not weak field. At start-up the largest |B| recovered from A is
  printed beside the probe's bmax for the same time.

  ONE FIXED COLOUR MAPPING FOR THE WHOLE FILM AND BOTH PANELS: brightness is
  asinh(|q|/knee)/asinh(lim/knee), saturating at `lim`. For the vorticity, lim
  is the initial peak (3.29 eps/R0) and the hue is its sign: amber turns with
  the vortex, blue against it -- the shielding ring of this zero-circulation
  vortex is blue at t = 0. For the current, lim is the --pct percentile of |j|
  at the frame where max|j| peaks, over all runs given, in B0/R0, so two runs
  with different B0 are compared RELATIVE TO THEIR OWN APPLIED FIELD.

  THE GLOW IS DECORATION: a blurred copy of the bright part of the picture added
  back (three Gaussians, wrapped at the panel's edges). It widens the halo of a
  sheet, not the sheet. --no-bloom removes it.

  THE CURVES are the probe column `surv`: the projection of the velocity on the
  initial vortex over its exact viscous decay, so 1 is a vortex only viscosity
  has touched and 0 is none of it left.

usage: anim_vortex_disruption.py DIR [DIR2] [out.mp4] [--layout pair|single]
           [--look vort|current|fire|duo] [--crop 12] [--dA 0.25] [--fps 60]
           [--frames 1200] [--wknee 0.3] [--pct 99.9] [--jobs 6] [--hold 2]
           [--tmax T] [--no-bloom] [--still-t t1,t2 PREFIX] [--sheet t1,t2 PNG]
"""
import glob
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
TWO_PI = 2.0 * np.pi
T_CORE = TWO_PI / np.sqrt(np.e)        # one turn of the core, in R0/eps

BG = np.array([0.0022, 0.0027, 0.0042], dtype=np.float32)    # linear light

STOPS = np.array([0.00, 0.26, 0.45, 0.60, 0.75, 0.88, 1.00])
WARM = np.array([[0, 0, 0], [0.014, 0.002, 0.006], [0.075, 0.008, 0.012],
                 [0.30, 0.045, 0.016], [0.95, 0.26, 0.040], [1.80, 0.95, 0.28],
                 [2.6, 2.1, 1.3]])
COOL = np.array([[0, 0, 0], [0.002, 0.005, 0.020], [0.008, 0.022, 0.11],
                 [0.020, 0.080, 0.38], [0.040, 0.36, 1.05], [0.34, 1.10, 1.85],
                 [1.4, 2.2, 2.7]])
FIRE_STOPS = np.array([0.00, 0.22, 0.42, 0.58, 0.72, 0.86, 1.00])
FIRE = np.array([[0, 0, 0], [0.006, 0.002, 0.020], [0.050, 0.006, 0.090],
                 [0.32, 0.020, 0.20], [1.05, 0.16, 0.10], [2.0, 0.95, 0.20],
                 [3.0, 2.6, 2.0]])
ELECTRIC = np.array([[0, 0, 0], [0.000, 0.004, 0.012], [0.004, 0.022, 0.07],
                     [0.020, 0.12, 0.32], [0.15, 0.55, 1.05], [0.75, 1.45, 2.1],
                     [1.9, 2.4, 2.7]])
LOOKS = ("vort", "current", "fire", "duo")
# The field lines' colour, added in linear light, per look.
THREAD = {"vort": np.array([0.72, 0.82, 1.00]) * 0.12,
          "current": np.array([0.75, 0.80, 1.00]) * 0.07,
          "fire": np.array([0.75, 0.80, 1.00]) * 0.06,
          "duo": np.array([1.00, 0.90, 0.75]) * 0.09}
BLOOM = [(2.5, 0.40), (12.0, 0.28), (40.0, 0.12)]
EXPOSURE = 1.15

FONT_TEXT = "/System/Library/Fonts/Avenir Next.ttc"     # 2 demi, 5 medium, 7 regular
FONT_MATH = "/System/Library/Fonts/HelveticaNeue.ttc"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"
INK = [(120, 205, 255), (255, 176, 96)]                  # the two runs' curves

G = {}


def flag(name):
    return name in sys.argv


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def raw(path):
    with open(path, "rb") as fh:
        nx, ny = np.fromfile(fh, dtype=np.int32, count=2)
        return np.fromfile(fh, dtype=np.float32, count=nx * ny).reshape(ny, nx)


def srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055)


def tone(lin):
    return srgb(1.0 - np.exp(-EXPOSURE * np.maximum(lin, 0.0)))


def ramp(b, table, stops=STOPS):
    out = np.empty(b.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(b, stops, table[:, c])
    return out


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def font(path, size, index=0):
    return ImageFont.truetype(path, size, index=index)


# ------------------------------------------------------------------------------
#  A run: its header, its probes and its frames.
# ------------------------------------------------------------------------------
class Run:
    def __init__(self, d, nframes):
        self.d = d
        log = open(os.path.join(d, "log.txt")).read()
        m = re.search(r"R0 = (\d+) cells\s+box (\d+) x (\d+).*?Re = ([\d.]+)\s+Rm = ([\d.]+)"
                      r".*?M = ([\d.eE+-]+)", log)
        if not m:
            sys.exit("no run header in %s/log.txt" % d)
        self.R0, self.n = int(m.group(1)), int(m.group(2))
        self.Re, self.Rm, self.M = float(m.group(4)), float(m.group(5)), float(m.group(6))
        x = re.search(r"M\^2 Rm = ([\d.eE+-]+)", log)
        self.X = float(x.group(1)) if x else self.M ** 2 * self.Rm
        s = re.search(r"dt = ([\d.eE+-]+)\s+t_max = [\d.eE+-]+ \((\d+) steps\)", log)
        self.dt, self.nmax = float(s.group(1)), int(s.group(2))
        self.lat = "D2Q9 + D2Q5" if "D2Q9" in log else "D3Q27 + D3Q7 (nz = 1)"
        self.dtf = max(1, self.nmax // nframes) * self.dt
        self.w = sorted(glob.glob(os.path.join(d, "w_*.bin")))
        self.j = sorted(glob.glob(os.path.join(d, "j_*.bin")))
        self.nf = min(len(self.w), len(self.j))
        ser = np.loadtxt(os.path.join(d, "series.dat"))
        ok = np.isfinite(ser).all(axis=1)
        self.ts, self.surv, self.jmax = ser[ok, 0], ser[ok, 5], ser[ok, 8]
        self.bmax = ser[ok, 3]


class Spec:
    """The whole periodic box -> its central P x P pixels, by zero-padding."""

    def __init__(self, run, P, crop):
        n, R0 = run.n, run.R0
        wc = crop * R0                               # cells across the crop
        r = int(round(n * P / wc))
        if r < n:
            sys.exit("the crop needs fewer pixels than cells: lower --crop")
        self.n, self.r, self.P = n, r, P
        L = n / R0
        self.ky = TWO_PI * np.fft.fftfreq(n, 1.0 / n)[:, None] / L
        self.kx = TWO_PI * np.fft.rfftfreq(n, 1.0 / n)[None, :] / L
        k2 = self.kx ** 2 + self.ky ** 2
        k2[0, 0] = 1.0
        self.ik2 = 1.0 / k2
        self.ik2[0, 0] = 0.0
        mc = 0.5 * (n - 1) * r / n                   # the vortex's centre, padded grid
        self.c0 = int(round(mc - 0.5 * P + 0.5))
        ys = (self.c0 + np.arange(P)) * n / r        # in cells
        self.Ay = ((ys - 0.5 * (n - 1)) / R0)[:, None]   # the applied field: A = y
        self.px = (n / r) / R0                       # R0 per pixel

    def pad(self, h):
        n, r = self.n, self.r
        kp = (n - 1) // 2
        out = np.zeros((r, r // 2 + 1), dtype=complex)
        out[:kp + 1, :kp + 1] = h[:kp + 1, :kp + 1]
        out[-kp:, :kp + 1] = h[-kp:, :kp + 1]
        return out * (float(r) / float(n)) ** 2

    def inv(self, h):
        c0, P = self.c0, self.P
        f = np.fft.irfft2(self.pad(h), s=(self.r, self.r))
        return f[c0:c0 + P, c0:c0 + P]


def bloom_kernel(P):
    ky = np.fft.fftfreq(P)[:, None] * TWO_PI
    kx = np.fft.rfftfreq(P)[None, :] * TWO_PI
    k2 = kx ** 2 + ky ** 2
    return sum(w * np.exp(-0.5 * s * s * k2) for s, w in BLOOM).astype(np.float32)


def fields(run, sp, k):
    """Frame k of a run on the crop: vorticity, current, A, |grad A| per pixel."""
    w, j = raw(run.w[k]), raw(run.j[k])
    if not (np.isfinite(w).all() and np.isfinite(j).all()):
        return None
    jh = np.fft.rfft2(j.astype(np.float64))
    A = sp.inv(jh * sp.ik2) + sp.Ay
    gy, gx = np.gradient(A)
    return (sp.inv(np.fft.rfft2(w.astype(np.float64))), sp.inv(jh), A, np.hypot(gx, gy))


def panel_rgb(run, sp, k, look):
    """Frame k of a run -> linear-light RGB, P x P, y UP; None if non-finite."""
    f = fields(run, sp, k)
    if f is None:
        return None
    wv, jv, A, grad = f
    bw = np.minimum(np.arcsinh(np.abs(wv) / G["wknee"]) / np.arcsinh(G["wlim"] / G["wknee"]), 1.0)
    bj = np.minimum(np.arcsinh(np.abs(jv) / G["jknee"]) / np.arcsinh(G["jlim"] / G["jknee"]), 1.0)
    if look == "vort":
        rgb = np.where((wv > 0)[..., None], ramp(bw, WARM), ramp(bw, COOL))
    elif look == "current":
        rgb = np.where((jv > 0)[..., None], ramp(bj, WARM), ramp(bj, COOL))
    elif look == "fire":
        rgb = ramp(bj, FIRE, FIRE_STOPS)
    else:
        rgb = 0.45 * np.where((wv > 0)[..., None], ramp(bw, WARM), ramp(bw, COOL)) \
            + 0.9 * ramp(bj, ELECTRIC)

    # Soft lines on the levels (m + 1/2) dA: Gaussian across, about 0.85 px,
    # and faded where neighbours are fewer than 6 px apart (gone at 2.5).
    dA = G["dA"]
    g = np.maximum(grad, 1e-6)
    fr = A / dA
    dist = np.abs(fr - np.floor(fr) - 0.5) * dA / g
    line = np.exp(-(dist / 0.85) ** 2) * smoothstep(2.5, 6.0, dA / g)
    rgb = rgb + line.astype(np.float32)[..., None] * THREAD[look].astype(np.float32)

    if G["bloom"]:
        lum = rgb.max(axis=-1)
        t = np.clip((lum - 0.22) / 0.78, 0.0, 1.0)
        src = rgb * (t * t * (3.0 - 2.0 * t))[..., None]
        for c in range(3):
            rgb[..., c] += np.fft.irfft2(np.fft.rfft2(src[..., c]) * G["bk"], s=src.shape[:2])
    return rgb[::-1] + BG


# ------------------------------------------------------------------------------
#  Text and curves.
# ------------------------------------------------------------------------------
def label(img, xy, lines, align="la"):
    """Lines of (text, font, fill) with a soft dark shadow, so they read on any field."""
    sh = Image.new("L", img.size, 0)
    ds, d = ImageDraw.Draw(sh), ImageDraw.Draw(img)
    x, y = xy
    pos = []
    for text, fnt, fill in lines:
        pos.append((x, y, text, fnt, fill))
        ds.text((x, y), text, font=fnt, fill=200, anchor=align)
        y += int(fnt.size * 1.32)
    sh = sh.filter(ImageFilter.GaussianBlur(6))
    img.paste((0, 0, 0), (0, 0), sh)
    for x, y, text, fnt, fill in pos:
        d.text((x, y), text, font=fnt, fill=fill, anchor=align)


def curves(img, box, t, runs, tend, inks):
    """surv against time up to t, one curve per run, with a glow; box = x0, y0, w, h."""
    x0, y0, pw, ph = box
    q, pad = 3, 18
    lo, hi = -0.2, 1.05
    d = ImageDraw.Draw(img)
    dim = (96, 106, 126)
    for v in (0.0, 1.0):                         # hairlines at 0 and 1
        yv = y0 + ph * (hi - v) / (hi - lo)
        d.line([(x0, yv), (x0 + pw, yv)], fill=(44, 50, 62), width=1)
        d.text((x0 - 10, yv), "%g" % v, font=font(FONT_TEXT, 17, 7), fill=dim, anchor="rm")
    for run, ink in zip(runs, inks):
        lay = Image.new("L", ((pw + 2 * pad) * q, (ph + 2 * pad) * q), 0)
        dl = ImageDraw.Draw(lay)
        k = int(np.searchsorted(run.ts, t, side="right"))
        xs = np.r_[run.ts[:k], t]
        ys = np.r_[run.surv[:k], np.interp(t, run.ts, run.surv)]
        pts = [((pad + pw * a / tend) * q, (pad + ph * (hi - b) / (hi - lo)) * q)
               for a, b in zip(xs, ys)]
        if len(pts) > 1:
            dl.line(pts, fill=255, width=int(2.4 * q), joint="curve")
        hx, hy = pts[-1]
        dl.ellipse([hx - 5 * q, hy - 5 * q, hx + 5 * q, hy + 5 * q], fill=255)
        lay = lay.resize((pw + 2 * pad, ph + 2 * pad), Image.LANCZOS)
        core = np.asarray(lay, dtype=np.float32) / 255.0
        halo = np.asarray(lay.filter(ImageFilter.GaussianBlur(6)), dtype=np.float32) / 255.0
        c = np.array(ink, dtype=np.float32) / 255.0
        add = np.clip(core[..., None] * (0.35 + 0.65 * c) + 0.8 * halo[..., None] * c, 0, 1)
        reg = np.asarray(img.crop((x0 - pad, y0 - pad, x0 + pw + pad, y0 + ph + pad)),
                         dtype=np.float32) / 255.0
        reg = 1.0 - (1.0 - reg) * (1.0 - add)
        img.paste(Image.fromarray((255 * reg + 0.5).astype(np.uint8)), (x0 - pad, y0 - pad))


# ------------------------------------------------------------------------------
#  The frame.
# ------------------------------------------------------------------------------
def init(cfg):
    G.update(cfg)
    G["specs"] = [Spec(r, cfg["P"], cfg["crop"]) for r in cfg["runs"]]
    G["bk"] = bloom_kernel(cfg["P"])


def display(rgb):
    f = tone(rgb)
    f = f + (np.random.default_rng(0).random(f.shape, dtype=np.float32) - 0.5) / 255.0
    return (255.0 * np.clip(f, 0.0, 1.0) + 0.5).astype(np.uint8)


def render(i):
    k, t = G["src"][i], G["tf"][i]
    runs, P = G["runs"], G["P"]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:] = (255 * srgb(BG)).astype(np.uint8)
    for m, (run, sp) in enumerate(zip(runs, G["specs"])):
        rgb = panel_rgb(run, sp, k, G["look"])
        if rgb is None:
            return None
        x0 = m * P if G["layout"] == "pair" else 0
        out[:P, x0:x0 + P] = display(rgb)
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img)
    white, soft, dim = (236, 240, 248), (176, 186, 204), (112, 122, 142)
    tend = G["tend"]

    if G["layout"] == "pair":
        d.line([(P, 0), (P, P)], fill=(30, 34, 44), width=2)
        for m, run in enumerate(runs):
            x0 = m * P + 34
            label(img, (x0, 30), [
                ("M = %.3f" % run.M, font(FONT_TEXT, 34, 2), INK[m]),
                ("applied field%s" % ("" if m == 0 else ",  %.0f× the left"
                                      % (run.M / runs[0].M)), font(FONT_TEXT, 21, 5), soft)])
        # The strip.
        y0 = P
        d.text((40, y0 + 22), "One vortex, two magnetic fields", font=font(FONT_TEXT, 31, 2),
               fill=white)
        d.text((40, y0 + 62), G["key"], font=font(FONT_TEXT, 19, 7), fill=soft)
        d.text((40, y0 + 90), "  ·  ".join(G["meta"]), font=font(FONT_TEXT, 14, 7), fill=dim)
        cx, cw = 1110, 400
        d.text((cx - 30, y0 + 12), "vortex left", font=font(FONT_TEXT, 17, 5), fill=soft)
        curves(img, (cx, y0 + 40, cw, 64), t, runs, tend, INK)
        d = ImageDraw.Draw(img)
        d.text((W - 40, y0 + 18), "t = %5.1f" % t, font=font(FONT_MONO, 40), fill=white,
               anchor="ra")
        d.text((W - 40, y0 + 72), "R0/ε  ·  the core turns once in %.1f" % T_CORE,
               font=font(FONT_TEXT, 18, 7), fill=soft, anchor="ra")
    else:
        run = runs[0]
        x0 = P + 64
        d.text((x0, 70), "A vortex in a", font=font(FONT_TEXT, 46, 2), fill=white)
        d.text((x0, 124), "weak magnetic field", font=font(FONT_TEXT, 46, 2), fill=white)
        y = 210
        for s in G["blurb"]:
            d.text((x0, y), s, font=font(FONT_TEXT, 22, 7), fill=soft)
            y += 32
        d.text((x0, y + 30), "M = %.3f   ·   M²Rm = %g" % (run.M, run.X),
               font=font(FONT_TEXT, 26, 5), fill=INK[1])
        d.text((x0, 620), "vortex left", font=font(FONT_TEXT, 20, 5), fill=soft)
        curves(img, (x0 + 30, 660, W - x0 - 110, 160), t, runs, tend, [INK[1]])
        d = ImageDraw.Draw(img)
        d.text((W - 64, 880), "t = %5.1f" % t, font=font(FONT_MONO, 44), fill=white, anchor="ra")
        d.text((W - 64, 940), "R0/ε  ·  the core turns once in %.1f" % T_CORE,
               font=font(FONT_TEXT, 19, 7), fill=soft, anchor="ra")
        for m, s_ in enumerate(G["meta"]):
            d.text((x0, H - 70 + 24 * m), s_, font=font(FONT_TEXT, 16, 7), fill=dim)

    out = np.asarray(img)
    fade = min(1.0, (i + 1) / max(1.0, 0.5 * G["fps"]))
    if fade < 1.0:
        out = (out * (fade * fade * (3.0 - 2.0 * fade))).astype(np.uint8)
    return out.tobytes()


def sheet(times, path):
    """Looks x times for the LAST run given, each panel reduced to 400 px."""
    run, sp = G["runs"][-1], G["specs"][-1]
    cell, top, left = 400, 56, 120
    img = Image.new("RGB", (left + cell * len(times), top + cell * len(LOOKS)), (8, 9, 13))
    d = ImageDraw.Draw(img)
    for c, tt in enumerate(times):
        d.text((left + c * cell + cell // 2, 18), "t = %g" % tt, font=font(FONT_TEXT, 24, 5),
               fill=(220, 226, 236), anchor="ma")
    for r, look in enumerate(LOOKS):
        d.text((14, top + r * cell + cell // 2), "%s\n%s" % ("ABCD"[r], look),
               font=font(FONT_TEXT, 26, 2), fill=(220, 226, 236), anchor="lm")
        for c, tt in enumerate(times):
            k = int(round(tt / run.dtf))
            rgb = panel_rgb(run, sp, min(k, run.nf - 1), look)
            im = Image.fromarray(display(rgb)).resize((cell, cell), Image.LANCZOS)
            img.paste(im, (left + c * cell, top + r * cell))
    img.save(path)
    print("  wrote " + path)


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    # Positional arguments that are option values are not run directories.
    vals = {sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a.startswith("--")
            and a not in ("--no-bloom",)}
    pos = [p for p in pos if p not in vals]
    dirs = [p for p in pos if os.path.isdir(p)]
    out = next((p for p in pos if p.endswith(".mp4")), None)
    if not dirs:
        sys.exit(__doc__)
    layout = arg("--layout", "pair" if len(dirs) == 2 else "single", str)
    if (layout == "pair") != (len(dirs) == 2):
        sys.exit("--layout pair takes two run directories, single takes one")
    look = arg("--look", "vort", str)
    if look not in LOOKS:
        sys.exit("--look is one of " + ", ".join(LOOKS))
    nframes = arg("--frames", 1200, int)
    runs = [Run(d, nframes) for d in dirs]
    if len({(r.n, r.R0, round(r.dtf, 12)) for r in runs}) != 1:
        sys.exit("the runs differ in box, R0 or frame interval")
    P = 960 if layout == "pair" else 1080
    nf = min(r.nf for r in runs)
    tmax = arg("--tmax", 1e30)
    src = np.array([k for k in range(nf) if k * runs[0].dtf <= tmax + 1e-9])
    tf = src * runs[0].dtf

    # One mapping for every frame and both panels.
    wlim = float(np.abs(raw(runs[0].w[0])).max())
    pct = arg("--pct", 99.9)
    jlim = 0.0
    for r in runs:
        kpk = int(np.clip(round(r.ts[int(np.argmax(r.jmax))] / r.dtf), 0, r.nf - 1))
        jlim = max(jlim, float(np.percentile(np.abs(raw(r.j[kpk])), pct)))
    cfg = dict(runs=runs, P=P, crop=arg("--crop", 12.0), look=look, layout=layout,
               dA=arg("--dA", 0.25), wlim=wlim, wknee=arg("--wknee", 0.3), jlim=jlim,
               jknee=jlim / 30.0, bloom=not flag("--no-bloom"), fps=arg("--fps", 60, int),
               src=src, tf=tf, tend=float(max(r.ts[-1] for r in runs)))
    r0 = runs[0]
    cfg["meta"] = ["Re = Rm = %g  ·  lattice Boltzmann, %s, %d² cells, R0 = %d"
                   % (r0.Rm, r0.lat, r0.n, r0.R0),
                   "central %g R0 of a 20 R0 periodic box  ·  M3LB"
                   % cfg["crop"]]
    cfg["key"] = {"vort": "colour: vorticity  ·  lines: magnetic field lines",
                  "current": "colour: electric current  ·  lines: magnetic field lines",
                  "fire": "colour: electric current, either sign  ·  lines: magnetic field lines",
                  "duo": "colour: vorticity  ·  glow: electric current  ·  lines: field lines"}[look]
    cfg["blurb"] = ["The vortex winds the field lines into a",
                    "spiral and expels them from its core.",
                    "The wound-up field pushes back, brakes",
                    "the vortex, and finally tears it apart."]
    print("  %d run(s), %d frames each, dt_frame = %.4f R0/eps, t to %.1f"
          % (len(runs), len(src), r0.dtf, tf[-1]))
    print("  vorticity saturates at %.3f (the t = 0 peak), knee %.2f; current at %.1f B0/R0 "
          "(p%g at max|j|), knee %.2f" % (wlim, cfg["wknee"], jlim, pct, jlim / 30.0))

    init(cfg)
    # Check: the largest |B| recovered from A against the probe's bmax.
    for r, sp in zip(runs, G["specs"]):
        tc = min(20.0, tf[-1])
        k = int(round(tc / r.dtf))
        f = fields(r, sp, k)
        b = f[3].max() / sp.px
        print("  %s: max|B| from A at t = %.1f is %.2f B0 on the crop; probe bmax %.2f"
              % (os.path.basename(r.d), k * r.dtf, b, np.interp(k * r.dtf, r.ts, r.bmax)))

    if flag("--sheet"):
        k = sys.argv.index("--sheet")
        sheet([float(s) for s in sys.argv[k + 1].split(",")], sys.argv[k + 2])
        return
    if flag("--still-t"):
        k = sys.argv.index("--still-t")
        for s in sys.argv[k + 1].split(","):
            i = int(np.argmin(np.abs(tf - float(s))))
            G["fps"] = 0                         # no fade on a still
            buf = render(i)
            path = "%s_%s_%s_t%05.1f.png" % (sys.argv[k + 2], layout, look, tf[i])
            Image.frombytes("RGB", (W, H), buf).save(path)
            print("  wrote %s" % path)
        return

    if out is None:
        sys.exit("give an output .mp4, or --still-t / --sheet")
    fps = cfg["fps"]
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
    hold = arg("--hold", 2.0)
    for _ in range(int(hold * fps)):
        enc.stdin.write(last)
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    print("  wrote %s: %d frames + %.1f s hold at %d fps = %.1f s"
          % (out, done, hold, fps, done / fps + hold))


if __name__ == "__main__":
    main()
