#!/usr/bin/env python3
"""Planes of the local dissipation from tg_mhd's raw dumps -- the paper's snapshots.

    tg_mhd_slices.py RUN_DIR [--planes z0,z3,y128] [--jobs 4] [--check]

RUN_DIR is a run's output directory from demonstrator/tg_mhd or GPU/src/tg_mhd.cu
run with -raw: series.dat names the setup, Re and Pm, and raw/fields_NNNN.raw hold
the full fields (write_raw_fields: "TGMHDRAW", int32 nx ny nz nv, double t,
double h, then nv float32 blocks u_x u_y u_z b_x b_y b_z rho, x fastest, in the
paper's units). For every dump and every requested plane this writes, under
RUN_DIR/slices/,

    <plane>_t<t>.f32   float32 [4][rows][cols]: nu|w|^2, eta|j|^2, |u|, |b|
    index.json         one record per file: plane, time, N, Re, h, delta, the box
                       mean eps at that time (series.dat, interpolated), the axes

A plane is zK -- the wall-parallel plane K nodes from the z = 0 wall, rows y and
columns x -- or yK, the vertical section K nodes from the y = 0 wall, rows z and
columns x. Default: z0 (on the wall), the planes nearest delta/2 and delta, z at
pi/4, the mid-plane, and y at pi/8, pi/4 and pi/2.

THE DERIVATIVES ARE THE DRIVERS', NOT A NEW DISCRETISATION -- HostFields::d in
demonstrator/tg_mhd.cpp, so a slice shows the same local eps whose box average
is series.dat's eps: second-order central differences, and at a wall node the
wall's own extension -- the mirror, odd or even by the parity, for a free-slip
velocity and for B, and a one-sided second-order stencil for a no-slip velocity.
--check proves the match: it sums nu|w|^2 + eta|j|^2 over the whole box with the
trapezoidal weights and compares with series.dat's eps, which must agree to the
float32 rounding of the dump. It reads every plane, so use it on a small run
whose last dump falls on its last probe (-tmax T -raw T).

Pure stdlib, like the other tg_mhd tools -- it runs on a CSF3 node next to the
dumps, so only the slices travel. A plane takes 0.4-0.5 s at N = 129 (measured
2026-10-02 on a thermally throttled M1), so about 7 s at N = 512 by its N^2
scaling.
"""
import argparse
import json
import math
import multiprocessing
import os
import re
import struct
import sys
from array import array

HEAD = 40                                   # "TGMHDRAW" + 4 int32 + 2 double
ONESIDED, EVEN, ODD = 0, 1, 2


def read_header(path):
    with open(path, "rb") as f:
        if f.read(8) != b"TGMHDRAW":
            raise SystemExit("%s: not a tg_mhd raw dump" % path)
        nx, ny, nz, nv = struct.unpack("<4i", f.read(16))
        t, h = struct.unpack("<2d", f.read(16))
    if not nx == ny == nz or nv != 7:
        raise SystemExit("%s: expected a cube with 7 fields, got %d %d %d %d" % (path, nx, ny, nz, nv))
    return nx, t, h


def run_info(run_dir):
    with open(os.path.join(run_dir, "series.dat")) as f:
        head = f.readline()
        rows = [[float(x) for x in l.split()] for l in f if l.strip() and not l.startswith("#")]
    tag = next(t for t in head.split() if re.match(r"^(tgc|tgi|tga|hydro)_", t))
    get = lambda k: float(re.search(r"\b%s=([0-9.eE+-]+)" % k, head).group(1))
    if "_periodic_" in tag:
        raise SystemExit("%s: a periodic run has no walls to slice against" % run_dir)
    return {"tag": tag, "Re": get("Re"), "Pm": get("Pm"), "noslip": "_noslip_" in tag,
            "conducting": "_cond_" in tag, "series": [(r[0], r[7]) for r in rows]}


def ext(info, c, k):
    """HostFields::ext: the extension of component c across a wall normal to k."""
    if c < 3:
        return ONESIDED if info["noslip"] else (ODD if c == k else EVEN)
    return ODD if ((c - 3 == k) == info["conducting"]) else EVEN


def eps_at(info, t):
    s = info["series"]
    for (ta, ea), (tb, eb) in zip(s, s[1:]):
        if ta <= t <= tb:
            return ea + (eb - ea) * ((t - ta) / (tb - ta) if tb > ta else 0.0)
    return min(s, key=lambda r: abs(r[0] - t))[1]


def needed(p, L):
    """The node layers a derivative at layer p reads, along the plane's normal."""
    if 0 < p < L - 1:
        return (p - 1, p, p + 1)
    return (0, 1, 2) if p == 0 else (L - 3, L - 2, L - 1)


def compute(task):
    """One plane of one dump: [nu|w|^2, eta|j|^2, |u|, |b|] as four float32 arrays."""
    path, info, axis, K = task
    L, t, h = read_header(path)
    nu = 1.0 / info["Re"]
    eta = nu / info["Pm"]
    rows = {}                                   # (c, y, z) -> the row along x
    with open(path, "rb") as f:
        def row(c, y, z):
            f.seek(HEAD + 4 * ((c * L + z) * L * L + y * L))
            a = array("f")
            a.frombytes(f.read(4 * L))
            if sys.byteorder != "little":
                a.byteswap()
            return a
        if axis == "z":
            zs, ys = needed(K, L), range(L)
        else:
            zs, ys = range(L), needed(K, L)
        for c in range(6):
            for z in zs:
                for y in ys:
                    rows[(c, y, z)] = row(c, y, z)

    def v(c, x, y, z):
        return rows[(c, y, z)][x]

    def d(c, k, x, y, z):
        p = (x, y, z)[k]
        if 0 < p < L - 1:
            if k == 0:
                return 0.5 * (v(c, x + 1, y, z) - v(c, x - 1, y, z))
            if k == 1:
                return 0.5 * (v(c, x, y + 1, z) - v(c, x, y - 1, z))
            return 0.5 * (v(c, x, y, z + 1) - v(c, x, y, z - 1))
        e = ext(info, c, k)
        s = 1 if p == 0 else -1                 # step INTO the box

        def at(o):
            q = [x, y, z]
            q[k] += o
            return v(c, q[0], q[1], q[2])
        if e == ONESIDED:
            return s * (-1.5 * at(0) + 2.0 * at(s) - 0.5 * at(2 * s))
        return 0.0 if e == EVEN else s * at(s)  # mirror ghost: odd gives s*f(in)

    ih = 1.0 / h
    out = [array("f") for _ in range(4)]
    outer = range(L)                            # rows of the image: y for a z-plane, z for a y-plane
    for o in outer:
        y, z = (o, K) if axis == "z" else (K, o)
        for x in range(L):
            du = [[d(c, k, x, y, z) * ih for k in range(3)] for c in range(3)]
            db = [[d(3 + c, k, x, y, z) * ih for k in range(3)] for c in range(3)]
            w2 = ((du[2][1] - du[1][2]) ** 2 + (du[0][2] - du[2][0]) ** 2 + (du[1][0] - du[0][1]) ** 2)
            j2 = ((db[2][1] - db[1][2]) ** 2 + (db[0][2] - db[2][0]) ** 2 + (db[1][0] - db[0][1]) ** 2)
            out[0].append(nu * w2)
            out[1].append(eta * j2)
            out[2].append(math.sqrt(v(0, x, y, z) ** 2 + v(1, x, y, z) ** 2 + v(2, x, y, z) ** 2))
            out[3].append(math.sqrt(v(3, x, y, z) ** 2 + v(4, x, y, z) ** 2 + v(5, x, y, z) ** 2))
    return out


def default_planes(L, Re, h):
    dh = 1.0 / math.sqrt(Re) / h                # delta in cells
    q = lambda frac: int(round(frac * (L - 1)))
    zs = sorted({0, max(1, int(round(0.5 * dh))), max(1, int(round(dh))), q(0.25), q(0.5)})
    return ["z%d" % k for k in zs] + ["y%d" % k for k in sorted({q(0.125), q(0.25), q(0.5)})]


def check(path, info):
    """The trapezoid-weighted box mean of nu|w|^2 + eta|j|^2 against series.dat's eps."""
    L, t, h = read_header(path)
    tot = wsum = 0.0
    for K in range(L):
        a = compute((path, info, "z", K))
        wz = 0.5 if K in (0, L - 1) else 1.0
        for yy in range(L):
            wy = 0.5 if yy in (0, L - 1) else 1.0
            for x in range(L):
                w = wz * wy * (0.5 if x in (0, L - 1) else 1.0)
                tot += w * (a[0][yy * L + x] + a[1][yy * L + x])
                wsum += w
    mean, ref = tot / wsum, eps_at(info, t)
    return t, mean, ref


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--planes", default=None, help="comma-separated zK / yK (default: see the docstring)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--check", action="store_true", help="whole-box eps against series.dat (slow; small runs)")
    args = ap.parse_args(argv)

    info = run_info(args.run_dir)
    raw = sorted(os.path.join(args.run_dir, "raw", f) for f in os.listdir(os.path.join(args.run_dir, "raw"))
                 if f.startswith("fields_") and f.endswith(".raw"))
    if not raw:
        raise SystemExit("no raw/fields_*.raw under %s -- run the driver with -raw" % args.run_dir)
    if args.check:
        bad = 0
        for p in raw:
            t, mean, ref = check(p, info)
            rel = abs(mean - ref) / max(abs(ref), 1e-300)
            ok = rel < 1e-5
            bad += 0 if ok else 1
            print("%s  t = %.4f: box mean of nu|w|^2 + eta|j|^2 = %.8e, series eps = %.8e, rel %.1e  %s"
                  % (os.path.basename(p), t, mean, ref, rel, "PASS" if ok else "FAIL"))
        return 1 if bad else 0

    L, _, h = read_header(raw[0])
    planes = args.planes.split(",") if args.planes else default_planes(L, info["Re"], h)
    for p in planes:
        if not re.match(r"^[yz]\d+$", p) or int(p[1:]) >= L:
            raise SystemExit("plane %r: expected zK or yK with K < %d" % (p, L))
    out_dir = os.path.join(args.run_dir, "slices")
    os.makedirs(out_dir, exist_ok=True)
    tasks, meta = [], []
    for path in raw:
        _, t, h = read_header(path)
        for p in planes:
            tasks.append((path, info, p[0], int(p[1:])))
            meta.append({"file": "%s_t%.3f.f32" % (p, t), "plane": p, "axis": p[0], "index": int(p[1:]),
                         "t": t, "N": L, "Re": info["Re"], "Pm": info["Pm"], "h": h,
                         "delta": 1.0 / math.sqrt(info["Re"]), "d_over_delta": int(p[1:]) * h * math.sqrt(info["Re"]),
                         "eps_mean": eps_at(info, t), "tag": info["tag"],
                         "fields": ["visc", "ohm", "umag", "bmag"],
                         "rows": "y" if p[0] == "z" else "z", "cols": "x"})
    with multiprocessing.Pool(max(1, min(args.jobs, len(tasks)))) as pool:
        for m, res in zip(meta, pool.imap(compute, tasks)):
            with open(os.path.join(out_dir, m["file"]), "wb") as f:
                for a in res:
                    if sys.byteorder != "little":
                        a.byteswap()
                    a.tofile(f)
            print("  %s  (eps mean %.4e)" % (m["file"], m["eps_mean"]), flush=True)
    idx = os.path.join(out_dir, "index.json")
    old = json.load(open(idx)) if os.path.exists(idx) else []
    keep = {m["file"] for m in meta}
    json.dump([m for m in old if m["file"] not in keep] + meta, open(idx, "w"), indent=1)
    print("wrote %d planes to %s" % (len(meta), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
