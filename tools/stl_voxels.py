#!/usr/bin/env python3
"""Voxel-union STL -> M3LB voxel array (.npy, float32, 1 = fluid, 0 = solid).

Written 2026-10-23 for Nianzhi's mixer, whose generator exports each topology as
an STL of the FLUID volume built from unit cubes: every vertex on an integer
grid, every facet axis-aligned, solid obstacles as holes. Such an STL has an
EXACT voxel representation, and this recovers it on the STL's own grid; the
solver driver (demonstrator/mixer_stl.cpp) then refines it onto the lattice.

EXACT, OR IT REFUSES. A general STL needs a general voxeliser and a choice of
how to treat cut cells; that is a modelling decision this tool does not make.
It checks, and aborts on failure:
  * every vertex lies on the grid (in units of --voxel), and every facet is
    perpendicular to an axis;
  * every ray crosses an even number of facets;
  * the fluid voxel count equals the STL's enclosed volume (divergence theorem),
    so a hole in the surface or a flipped facet cannot pass.

THE RAY IS NOT THROUGH THE FACE CENTRE, AND THAT IS NOT A DETAIL. A cube face
is two triangles split along a diagonal, so a ray through (j+1/2, k+1/2) runs
ALONG that diagonal, hits both triangles and cancels itself in the parity
count. The first version did exactly that and filled nothing: 0 fluid voxels
for a volume of 30970. The ray goes through (j+1/2, k+3/10), which no diagonal
of a unit square passes through.

Output: <out>.npy, shape (NX, NY, NZ), x slowest -- the STL's bounding box in
voxels -- and a summary on stdout. Stdlib only, like tools/osm_city.py.

    python3 tools/stl_voxels.py mixer1.stl --out geom/mixer1
"""
import argparse
import struct
import sys
from collections import defaultdict


def read_stl(path):
    """Triangles as ((x,y,z),(x,y,z),(x,y,z)) tuples; ASCII or binary."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            tris = []
            for t in range(n):
                v = struct.unpack_from("<12f", data, 84 + 50 * t)
                tris.append((v[3:6], v[6:9], v[9:12]))
            return tris
    tris, cur = [], []
    for line in data.decode("ascii", errors="replace").splitlines():
        tok = line.split()
        if tok and tok[0] == "vertex":
            cur.append(tuple(float(x) for x in tok[1:4]))
            if len(cur) == 3:
                tris.append(tuple(cur))
                cur = []
    if cur:
        sys.exit("stl_voxels: a facet with %d vertices" % len(cur))
    return tris


def to_grid(tris, voxel):
    """Vertices in voxel units, as ints; refuse anything off the grid."""
    out, worst = [], 0.0
    for tri in tris:
        g = []
        for v in tri:
            q = [c / voxel for c in v]
            r = [round(c) for c in q]
            worst = max(worst, max(abs(a - b) for a, b in zip(q, r)))
            g.append(tuple(int(c) for c in r))
        out.append(tuple(g))
    if worst > 1e-6:
        sys.exit("stl_voxels: a vertex is %.3g voxels off the grid -- not a voxel union "
                 "at --voxel %g; refusing rather than approximating" % (worst, voxel))
    return out


def signed_volume(tris):
    vol = 0.0
    for a, b, c in tris:
        vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
    return vol


def in_triangle(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def voxelise(tris):
    lo = [min(v[a] for t in tris for v in t) for a in range(3)]
    hi = [max(v[a] for t in tris for v in t) for a in range(3)]
    nx, ny, nz = (hi[a] - lo[a] for a in range(3))
    hits = defaultdict(list)
    for t in tris:
        span = [max(v[a] for v in t) - min(v[a] for v in t) for a in range(3)]
        if sorted(span)[0] != 0:
            sys.exit("stl_voxels: a facet is not perpendicular to an axis")
        if span[0] != 0:
            continue                    # only x-facets can cut a ray along x
        yz = [(v[1] - lo[1], v[2] - lo[2]) for v in t]
        for j in range(min(p[0] for p in yz), max(p[0] for p in yz)):
            for k in range(min(p[1] for p in yz), max(p[1] for p in yz)):
                if in_triangle((j + 0.5, k + 0.3), *yz):
                    hits[(j, k)].append(t[0][0] - lo[0])
    vox = bytearray(nx * ny * nz)       # 0 = solid
    for (j, k), xs in hits.items():
        xs.sort()
        if len(xs) % 2:
            sys.exit("stl_voxels: ray (y=%d, z=%d) crosses %d facets -- the surface is "
                     "not closed" % (j, k, len(xs)))
        for a, b in zip(xs[0::2], xs[1::2]):
            for i in range(a, b):
                vox[(i * ny + j) * nz + k] = 1
    return vox, (nx, ny, nz), lo


def write_npy_f32(path, vox, shape):
    hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d), }" % shape
    hdr += " " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode("latin1"))
        f.write(struct.pack("<%df" % len(vox), *vox))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stl")
    ap.add_argument("--voxel", type=float, default=1.0,
                    help="voxel edge in STL units (default 1)")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.npy")
    a = ap.parse_args()

    tris = to_grid(read_stl(a.stl), a.voxel)
    vox, (nx, ny, nz), lo = voxelise(tris)
    nf, vol = sum(vox), signed_volume(tris)
    print("stl        %s: %d facets, origin (%d, %d, %d) voxels" % (a.stl, len(tris), *lo))
    print("voxels     %d x %d x %d (x slowest)   fluid %d   enclosed volume %.6g"
          % (nx, ny, nz, nf, vol))
    if abs(nf - vol) > 1e-6 * max(1.0, vol):
        sys.exit("stl_voxels: fluid voxels %d != enclosed volume %.6g -- refusing" % (nf, vol))
    solid = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)
             if not vox[(i * ny + j) * nz + k]]
    if solid:
        ext = [(min(s[a] for s in solid), max(s[a] for s in solid) + 1) for a in range(3)]
        print("solid      %d voxels, x [%d, %d)  y [%d, %d)  z [%d, %d)"
              % (len(solid), *ext[0], *ext[1], *ext[2]))
    else:
        print("solid      none")
    write_npy_f32(a.out + ".npy", vox, (nx, ny, nz))
    print("wrote      %s.npy  float32 (%d, %d, %d), 1 = fluid" % (a.out, nx, ny, nz))


if __name__ == "__main__":
    main()
