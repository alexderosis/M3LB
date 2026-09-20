#!/usr/bin/env python3
"""OpenStreetMap city -> M3LB height field.

Turns OSM building footprints and their height tags into the
`<out>_heights.npy` + `<out>_meta.json` pair that `src/io/HeightField.hpp`
reads, so that a real city can be imported and run by `demonstrator/urban`
without leaving this repository.

STDLIB ONLY, AND THAT IS THE POINT.  The reference implementation this was
written against -- the Pollutant project's `build_voxel_city.py` -- needs
osmnx, geopandas, shapely, pyproj, pandas and numpy.  None of those are in
this machine's system python; they live in one project-local venv, which made
importing a city something M3LB could not do on its own.  What the job
actually needs is an HTTP GET, a map projection, and a polygon fill, and each
of those is under a hundred lines.  `HeightField.hpp` already hand-rolls a
.npy reader for the same reason and says so in its banner; `doc/fig/plot_urban.py`
is already numpy-free.  So: `urllib`, `json`, `math`, `array`, `zlib`, nothing else.

WHAT IT IS CHECKED AGAINST.  Two independent implementations agreeing is worth
more than either alone, which is the argument `tests/cross_colour` makes for
the colour gradient, so the same is done here:

  * The projection is checked against pyproj over a +-0.01 deg envelope around
    six places including a zone edge, the equator and 69 N.  Worst disagreement
    over 54 points is 0.95 mm -- 2e-4 of a 5 m cell.  `--selftest` re-checks it
    against hard-coded pyproj values, with no network and no pyproj.
  * The raster is checked against the Pollutant project's Manchester output,
    built by the osmnx script from a frozen 2026-07-30 Overpass snapshot that
    is still on disk.  `--osmnx-cache` reads that snapshot, so the comparison
    is of two rasterisers on identical input rather than of two downloads taken
    months apart.  `--compare-to` reports the agreement; see `doc/m3lb.tex`.

THE HEIGHT LADDER IS THE MODELLING, and most of a city rests on its lower
rungs.  A footprint's height is taken from `height`, else `building:levels`
times 3.2 m, else a fallback.  Manchester carries a `height` tag on 5.7% of
its buildings and reaches one through `building:levels` on a further 33.6%,
leaving 60.8% on the fallback; Midtown Manhattan carries one on 95%, the city
having imported LiDAR-derived heights -- that figure is the reference
implementation's, over 46,327 buildings, not a measurement taken here.  Which rung produced each cell is written to `<out>_source.npy` and
counted in the metadata, so the fraction of a domain resting on a guess is
visible rather than assumed.

`--fallback flat` puts one number on every untagged building; `--fallback type`
takes the median storey count of the buildings in this same query that DO carry
`building:levels`, per `building` value, and applies it to those that do not.
Measured on Manchester, switching between them moves 14.67% of cells by a mean
of 0.37 m and raises the median built column from 10.0 m to 12.8 m -- houses
calibrate to two storeys, apartments to three, `building:part` sections to
five -- while leaving the built set BIT IDENTICAL, 47,787 cells with not one
gained or lost.  It is a height model and nothing else.

The flat default is the worse model -- it puts a terraced house and a seven-
storey office at the same 10 m -- and it is nonetheless the DEFAULT here,
because it is what the existing Manchester and Manhattan rasters were built
with and a cross-check whose two sides differ in the model is a cross-check of
nothing.  Use `type` for new work and say which one a figure used.

BOTH `building` AND `building:part` ARE QUERIED, and the second is not
optional.  OSM's Simple 3D Buildings schema puts a tall structure's real
geometry on `building:part` children while the parent `building=yes` outline
carries no height at all.  In Midtown those parentless outlines are precisely
the large ones: the reference implementation measured that querying `building`
alone leaves 20.6% of footprints by count but 54.8% of built AREA on the
fallback.  That pair is theirs; what was measured here is Manchester, where
`building:part` supplies 621 of the 6,174 footprints.  Heights are reduced with a
max where footprints overlap, so a tower's tallest part beats its own podium.

WHAT IT DOES NOT DO.  No terrain: the ground is flat at z = 0, which is what
the solver assumes.  No roof shapes, no overhangs, no bridges -- a column is
solid from the ground to the height of the tallest footprint covering it, and
nothing can be solid above empty air.  No vegetation, no walls, no street
furniture.  Relations are assembled by stitching member ways into rings, which
handles the multipolygons in a city centre but is not a full OSM multipolygon
implementation; unclosed leftovers are reported and dropped rather than
guessed at.  It does not choose a wind, a source or a diffusivity -- those are
`demonstrator/urban`'s arguments.

Output
  <out>_heights.npy   (nx, ny) float32, C order, metres of building, 0 = open
  <out>_source.npy    (nx, ny) uint8, which rung of the ladder produced it
  <out>_meta.json     grid, CRS, provenance counts.  dz == dx always: the
                      solver is isotropic and HeightField.hpp rejects the rest.
"""
import argparse
import array
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

LEVEL_M = 3.2       # metres per storey when only building:levels is known
DEFAULT_M = 10.0    # last resort under --fallback flat
MIN_TYPE_N = 20     # tagged examples needed before trusting a per-type median
UA = "M3LB/1.0 (lattice Boltzmann urban dispersion; https://github.com/alexderosis/M3LB)"

# Which rung of the height ladder produced a cell.  Written into every
# meta.json so the codes are self-describing rather than remembered.
SRC = {0: "open", 1: "height tag", 2: "building:levels", 3: "type-calibrated",
       4: "city median", 5: "flat default"}

# =============================================================================
# WGS84 -> UTM.  Snyder's series to O(e^6) in the transverse Mercator terms and
# O(e^8) in the meridional arc; checked against pyproj at 0.95 mm worst over the
# working envelope, which is 2e-4 of a 5 m cell.  A local tangent plane would
# also have been accurate enough over 2 km, but UTM is what the reference wrote
# into meta.json and matching it keeps x0/y0 comparable between the two tools.
# =============================================================================
_A = 6378137.0                    # WGS84 semi-major axis, m
_F = 1.0 / 298.257223563          # flattening
_K0 = 0.9996                      # UTM scale on the central meridian
_E2 = _F * (2.0 - _F)
_EP2 = _E2 / (1.0 - _E2)


def utm_zone(lat, lon):
    """(zone, EPSG code).  Northern hemisphere 326xx, southern 327xx."""
    z = int((lon + 180.0) / 6.0) + 1
    return z, (32600 if lat >= 0 else 32700) + z


def to_utm(lat, lon, zone, south=False):
    lam0 = math.radians(-183.0 + 6.0 * zone)
    phi, lam = math.radians(lat), math.radians(lon)
    s, c, t = math.sin(phi), math.cos(phi), math.tan(phi)
    N = _A / math.sqrt(1.0 - _E2 * s * s)
    T, C = t * t, _EP2 * c * c
    a = (lam - lam0) * c
    M = _A * ((1 - _E2/4 - 3*_E2**2/64 - 5*_E2**3/256) * phi
              - (3*_E2/8 + 3*_E2**2/32 + 45*_E2**3/1024) * math.sin(2*phi)
              + (15*_E2**2/256 + 45*_E2**3/1024) * math.sin(4*phi)
              - (35*_E2**3/3072) * math.sin(6*phi))
    e = _K0 * N * (a + (1 - T + C) * a**3 / 6.0
                   + (5 - 18*T + T*T + 72*C - 58*_EP2) * a**5 / 120.0) + 500000.0
    n = _K0 * (M + N * t * (a*a/2.0 + (5 - T + 9*C + 4*C*C) * a**4 / 24.0
                            + (61 - 58*T + T*T + 600*C - 330*_EP2) * a**6 / 720.0))
    return e, n + (10000000.0 if south else 0.0)


def from_utm(e, n, zone, south=False):
    """Inverse, used only to put the query bbox back into lat/lon."""
    e -= 500000.0
    if south:
        n -= 10000000.0
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    mu = (n / _K0) / (_A * (1 - _E2/4 - 3*_E2**2/64 - 5*_E2**3/256))
    phi1 = (mu + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
            + (151*e1**3/96) * math.sin(6*mu))
    s, c, t = math.sin(phi1), math.cos(phi1), math.tan(phi1)
    C1, T1 = _EP2 * c * c, t * t
    N1 = _A / math.sqrt(1 - _E2 * s * s)
    R1 = _A * (1 - _E2) / (1 - _E2 * s * s) ** 1.5
    D = e / (N1 * _K0)
    phi = phi1 - (N1 * t / R1) * (D*D/2 - (5 + 3*T1 + 10*C1 - 4*C1*C1 - 9*_EP2) * D**4/24
                                  + (61 + 90*T1 + 298*C1 + 45*T1*T1 - 252*_EP2
                                     - 3*C1*C1) * D**6/720)
    lam = (D - (1 + 2*T1 + C1) * D**3/6
           + (5 - 2*C1 + 28*T1 - 3*C1*C1 + 8*_EP2 + 24*T1*T1) * D**5/120) / c
    return math.degrees(phi), -183.0 + 6.0 * zone + math.degrees(lam)


# =============================================================================
# .npy, read and write.  Same restriction as HeightField.hpp's reader: flat
# C-order arrays of one dtype, nothing pickled.  The header string is written
# with the exact spacing that reader searches for ("'fortran_order': False").
# =============================================================================
def npy_write(path, data, dtype, shape):
    typ = {"f4": "<f4", "u1": "|u1"}[dtype]
    code = {"f4": "f", "u1": "B"}[dtype]
    hdr = ("{'descr': '%s', 'fortran_order': False, 'shape': (%s), }"
           % (typ, ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "")))
    pad = 64 - (10 + len(hdr) + 1) % 64
    hdr = hdr + " " * (pad if pad != 64 else 0) + "\n"
    buf = array.array(code, data)
    if sys.byteorder == "big" and code == "f":
        buf.byteswap()
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00")
        f.write(len(hdr).to_bytes(2, "little"))
        f.write(hdr.encode("latin-1"))
        f.write(buf.tobytes())


def npy_read(path):
    """-> (list-like of values, shape).  Accepts <f4 and |u1, C order only."""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(path + ": not a .npy file")
        ver = f.read(2)
        hlen = int.from_bytes(f.read(2 if ver[0] == 1 else 4), "little")
        hdr = f.read(hlen).decode("latin-1")
        if "'fortran_order': False" not in hdr:
            raise ValueError(path + ": Fortran order is not supported")
        descr = hdr.split("'descr':")[1].split("'")[1]
        shape = tuple(int(s) for s in
                      hdr.split("'shape':")[1].split("(")[1].split(")")[0]
                      .replace(",", " ").split())
        n = 1
        for s in shape:
            n *= s
        code = {"<f4": "f", "f4": "f", "|u1": "B", "u1": "B", "<u1": "B"}.get(descr)
        if code is None:
            raise ValueError(path + ": unsupported dtype " + descr)
        buf = array.array(code)
        buf.fromfile(f, n)
        if sys.byteorder == "big" and code == "f":
            buf.byteswap()
        return buf, shape


# =============================================================================
# Fetching.  Nominatim for the place name, Overpass for the buildings, both
# cached to disk by query hash: a re-run costs nothing and works offline, and
# the raster a figure was built from can be rebuilt months later from the same
# bytes.  That is not a convenience -- it is what makes the cross-check below a
# comparison of two rasterisers rather than of two downloads.
# =============================================================================
def _cached_get(url, data, cache_dir, label, pause=1.0):
    import hashlib
    key = hashlib.sha1((url + "|" + (data or "")).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{label}_{key}.json") if cache_dir else None
    if path and os.path.exists(path):
        print(f"[cache]  {label}: {os.path.basename(path)}")
        return json.load(open(path))
    print(f"[net]    {label}: requesting ...", flush=True)
    req = urllib.request.Request(url, data=(data.encode() if data else None),
                                 headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read().decode("utf-8")
            break
        except Exception as ex:                       # noqa: BLE001
            if attempt == 2:
                raise
            print(f"         {type(ex).__name__}, retrying in {5 * (attempt + 1)}s")
            time.sleep(5 * (attempt + 1))
    out = json.loads(body)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(body)
        print(f"[cache]  {label}: saved {os.path.basename(path)} "
              f"({len(body)/1e6:.1f} MB)")
    time.sleep(pause)
    return out


def geocode(place, cache_dir):
    q = urllib.parse.urlencode({"q": place, "format": "json", "limit": 1})
    j = _cached_get("https://nominatim.openstreetmap.org/search?" + q,
                    None, cache_dir, "nominatim")
    if not j:
        raise SystemExit(f"could not geocode {place!r}")
    return float(j[0]["lat"]), float(j[0]["lon"])


def overpass(south, west, north, east, cache_dir, url):
    bbox = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
    q = (f"[out:json][timeout:300];("
         f'way["building"]({bbox});way["building:part"]({bbox});'
         f'relation["building"]({bbox});relation["building:part"]({bbox});'
         f");out geom;")
    return _cached_get(url, "data=" + urllib.parse.quote(q), cache_dir, "overpass")


# =============================================================================
# Parsing.  Two response shapes have to be understood, and supporting the
# second is what makes the cross-check possible: `out geom;` (what this tool
# asks for -- geometry inline, one pass) and the node-reference form with a
# separate node list (what osmnx asks for, and therefore what the Pollutant
# project's frozen Manchester snapshot is written in).
# =============================================================================
def _rings_from_members(members, nodes, ways_by_id):
    """Stitch member ways into closed rings.  Returns (rings, n_unclosed).

    A member must be resolved through `ways_by_id` and not read in place.  In
    the `out geom;` form a member carries its own geometry, so reading it in
    place works and every synthetic test passes; in the node-reference form --
    which is what osmnx asks for, and therefore what every cached snapshot is
    written in -- a member is a bare {type, ref, role} and reading it in place
    yields NOTHING.  Every multipolygon then vanishes without an error, and
    since the member ways of a multipolygon are untagged (the tags live on the
    relation) the way loop does not pick them up either.  That cost 39 whole
    buildings, 1.4% of the Manchester domain, and the only reason it was
    noticed is that the raster was diffed against an independent one."""
    segs = []
    for m in members:
        if m.get("type") != "way" or m.get("role") not in ("outer", "inner", ""):
            continue
        pts = _way_points(m, nodes)
        if not pts and m.get("ref") in ways_by_id:
            pts = _way_points(ways_by_id[m["ref"]], nodes)
        if len(pts) >= 2:
            segs.append(list(pts))
    rings, unclosed = [], 0
    while segs:
        cur = segs.pop(0)
        changed = True
        while changed and cur[0] != cur[-1]:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == cur[-1]:
                    cur.extend(s[1:]); segs.pop(i); changed = True; break
                if s[-1] == cur[-1]:
                    cur.extend(reversed(s[:-1])); segs.pop(i); changed = True; break
                if s[-1] == cur[0]:
                    cur = s[:-1] + cur; segs.pop(i); changed = True; break
                if s[0] == cur[0]:
                    cur = list(reversed(s[1:])) + cur; segs.pop(i); changed = True; break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append(cur)
        else:
            unclosed += 1
    return rings, unclosed


def _way_points(w, nodes):
    if "geometry" in w:
        return [(p["lat"], p["lon"]) for p in w["geometry"]]
    if "nodes" in w:
        out = []
        for nid in w["nodes"]:
            p = nodes.get(nid)
            if p is None:
                return []
            out.append(p)
        return out
    return []


def parse_osm(docs):
    """-> (footprints, stats).  A footprint is (tags, [ring, ...]) with each
    ring a list of (lat, lon).  Rings after the first are holes."""
    nodes, ways, rels = {}, [], []
    for d in docs:
        for e in d.get("elements", ()):
            t = e.get("type")
            if t == "node":
                nodes[e["id"]] = (e["lat"], e["lon"])
            elif t == "way":
                ways.append(e)
            elif t == "relation":
                rels.append(e)
    ways_by_id = {w["id"]: w for w in ways if "id" in w}
    seen_in_rel = set()
    feats, unclosed, skipped = [], 0, 0
    for r in rels:
        tags = r.get("tags", {})
        roles = {m.get("role") for m in r.get("members", ())}
        # A type=building relation (Simple 3D Buildings) groups an outline with
        # its parts; both are separately present as tagged ways, so taking the
        # relation as well would double-count the footprint and, worse, hand the
        # outline's missing height to the tower's own cells.
        if tags.get("type") == "building" or roles & {"part", "outline"}:
            skipped += 1
            continue
        rings, nu = _rings_from_members(r.get("members", ()), nodes, ways_by_id)
        unclosed += nu
        if rings:
            # Largest ring first, so the rest are treated as holes.
            rings.sort(key=lambda g: abs(_shoelace(g)), reverse=True)
            feats.append((tags, rings))
            for m in r.get("members", ()):
                if m.get("type") == "way":
                    seen_in_rel.add(m.get("ref"))
    for w in ways:
        if w.get("id") in seen_in_rel:
            continue
        tags = w.get("tags", {})
        if not ("building" in tags or "building:part" in tags):
            continue
        pts = _way_points(w, nodes)
        if len(pts) < 4 or pts[0] != pts[-1]:
            unclosed += 1
            continue
        feats.append((tags, [pts]))
    return feats, {"ways": len(ways), "relations": len(rels), "nodes": len(nodes),
                   "unclosed": unclosed, "grouping_relations": skipped}


def _shoelace(ring):
    s = 0.0
    for i in range(len(ring) - 1):
        s += ring[i][1] * ring[i + 1][0] - ring[i + 1][1] * ring[i][0]
    return 0.5 * s


# =============================================================================
# The height ladder.
# =============================================================================
def parse_height(raw):
    """(metres, ok).  Handles '62.5', '15 m', '25.0;26.2' (first value) and
    "103'" (feet).  Three height strings in Manchester are malformed this way
    and one of them is in feet, which would become 103 m -- a 3x error on that
    building -- if the apostrophe were ignored."""
    if raw is None:
        return None, False
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None, False
    feet = s.endswith("'") or "ft" in s.lower()
    head = s.split(";")[0].strip()
    num, seen_dot = "", False
    for ch in head:
        if ch.isdigit():
            num += ch
        elif ch == "." and not seen_dot and num:
            num += ch; seen_dot = True
        elif num:
            break
        elif ch in "+- \t":
            continue
        else:
            break
    if not num or num == ".":
        return None, False
    v = float(num)
    if feet:
        v *= 0.3048
    return (v, True) if 1.0 < v < 600.0 else (None, False)


def btype(tags):
    """The label a per-type median is keyed on.  A building:part way has no
    `building` tag at all; those are tower sections and get their own category
    rather than being lumped in with whatever the default would be."""
    v = tags.get("building")
    if v is None or str(v).strip() in ("", "nan"):
        return "(building:part)"
    return str(v)


def assign_heights(feats, mode, level_m, default_m, quiet=False):
    lv_by_type, all_lv = {}, []
    if mode == "type":
        for tags, _ in feats:
            lv, ok = parse_height(tags.get("building:levels"))
            if ok and lv < 200:
                lv_by_type.setdefault(btype(tags), []).append(lv)
                all_lv.append(lv)
    med = lambda v: sorted(v)[len(v) // 2] if v else 0.0          # noqa: E731
    global_lv = med(all_lv) if all_lv else default_m / level_m
    type_lv = {t: med(v) for t, v in lv_by_type.items() if len(v) >= MIN_TYPE_N}
    if mode == "type" and not quiet:
        print(f"[calib]  storeys: city median {global_lv:.1f}; per-type medians "
              f"for {len(type_lv)} types with >= {MIN_TYPE_N} examples")
        for t in sorted(type_lv, key=lambda k: -len(lv_by_type[k]))[:6]:
            print(f"         {t:22} {type_lv[t]:4.1f} storeys  "
                  f"(n={len(lv_by_type[t])})")
    out = []
    for tags, rings in feats:
        h, ok = parse_height(tags.get("height"))
        if ok:
            out.append((h, 1, rings)); continue
        lv, ok = parse_height(tags.get("building:levels"))
        if ok and lv < 200:
            out.append((lv * level_m, 2, rings)); continue
        if mode == "flat":
            out.append((default_m, 5, rings)); continue
        t = btype(tags)
        if t in type_lv:
            out.append((type_lv[t] * level_m, 3, rings)); continue
        out.append((global_lv * level_m, 4, rings))
    return out


# =============================================================================
# Rasterising.  Scanline fill on CELL CENTRES, which is the same rule
# HeightField.hpp uses vertically ("a cell is solid when its centre lies below
# the local building height"), so the horizontal and vertical discretisations
# agree about what a cell is.
#
# The reference does this the other way round -- build every cell centre as a
# point, put the polygons in an STRtree, query point-in-polygon -- which needs
# an index and a geometry library.  A scanline needs neither and is O(edges x
# rows crossed) rather than O(cells x log polygons).
#
# Holes come free: an inner ring's crossings join the same sorted list and the
# even-odd rule closes the courtyard.  Overlap is resolved by SORTING BY HEIGHT
# ASCENDING and letting later writes win, so the tallest footprint covering a
# cell supplies both its height and its provenance.  Taking a max over the
# height alone would be wrong for the provenance -- the codes are ordinal, so a
# heightless outline overlapping a real tower part would report "fallback"
# exactly where the height came from the part.  That mislabelled 48% of built
# area in the reference before it was fixed there, and the same trap is here.
#
# INDEX ORDER IS i*ny + j, EASTING SLOW -- the scanline runs along constant
# northing j and strides, rather than walking a contiguous row, because that is
# what `HeightField.hpp` stores and what the .npy of shape (nx, ny) means.  The
# contiguous-looking alternative, j*nx + i, is the same array TRANSPOSED: it
# rotates the city by 90 degrees, and every cell count, built fraction and mass
# budget downstream is identical, so nothing reports it.  This file got it wrong
# first time and the selftest below now pins it.
# =============================================================================
def rasterise(polys, nx, ny, x0, y0, dx):
    hmap = array.array("f", bytes(4 * nx * ny))
    smap = array.array("B", bytes(nx * ny))
    for h, src, rings in sorted(polys, key=lambda p: p[0]):
        ymin = min(p[1] for g in rings for p in g)
        ymax = max(p[1] for g in rings for p in g)
        j0 = max(0, int(math.ceil((ymin - y0) / dx - 0.5)))
        j1 = min(ny - 1, int(math.floor((ymax - y0) / dx - 0.5)))
        if j1 < j0:
            continue
        edges = []
        for g in rings:
            for a in range(len(g) - 1):
                if g[a][1] != g[a + 1][1]:
                    edges.append((g[a][0], g[a][1], g[a + 1][0], g[a + 1][1]))
        for j in range(j0, j1 + 1):
            y = y0 + (j + 0.5) * dx
            xs = [xa + (y - ya) * (xb - xa) / (yb - ya)
                  for xa, ya, xb, yb in edges
                  if (ya <= y < yb) or (yb <= y < ya)]
            if len(xs) < 2:
                continue
            xs.sort()
            for a in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil((xs[a] - x0) / dx - 0.5)))
                i1 = min(nx - 1, int(math.ceil((xs[a + 1] - x0) / dx - 0.5)) - 1)
                for i in range(i0, i1 + 1):
                    hmap[i * ny + j] = h
                    smap[i * ny + j] = src
    return hmap, smap


# =============================================================================
# Comparison.  Kept in the tool rather than in a scratch script because it is
# the evidence that the importer is right, and evidence that is not runnable
# stops being evidence.
# =============================================================================
def compare(mine, ref_path, nx, ny, dx, label="reference"):
    ref, shape = npy_read(ref_path)
    if len(ref) != len(mine):
        print(f"[diff]   SHAPE MISMATCH: {label} has {shape} = {len(ref)} cells, "
              f"this raster has {nx}x{ny} = {len(mine)}")
        return False
    n = len(mine)
    exact = same_built = only_mine = only_ref = 0
    worst = 0.0
    sad = 0.0
    for a, b in zip(mine, ref):
        if a == b:
            exact += 1
        d = abs(a - b)
        sad += d
        worst = max(worst, d)
        if a > 0 and b > 0:
            same_built += 1
        elif a > 0:
            only_mine += 1
        elif b > 0:
            only_ref += 1
    built_ref = same_built + only_ref
    built_mine = same_built + only_mine
    print(f"[diff]   against {label} ({os.path.basename(ref_path)})")
    print(f"         identical cells      {exact:8,} / {n:,}  ({100*exact/n:6.2f}%)")
    print(f"         built here / there   {built_mine:8,} / {built_ref:,}  "
          f"({100*(built_mine-built_ref)/max(built_ref,1):+.2f}%)")
    print(f"         built in both        {same_built:8,}  "
          f"({100*same_built/max(built_ref,1):6.2f}% of theirs)")
    print(f"         built only here      {only_mine:8,}  "
          f"({100*only_mine/n:6.2f}% of domain)")
    print(f"         built only there     {only_ref:8,}  "
          f"({100*only_ref/n:6.2f}% of domain)")
    print(f"         mean |dh| over all   {sad/n:8.3f} m      worst {worst:.1f} m")
    return exact == n


# =============================================================================
def emit_fixture(out):
    """Write a small city through the WHOLE pipeline, for the C++ side to read.

    `--selftest` proves this file is self-consistent; it cannot prove that
    `HeightField.hpp` agrees with it.  That seam is a .npy header written by
    one language and parsed by another, and the interesting way for it to fail
    is not an exception but a TRANSPOSE, which every total is blind to.  So the
    fixture is deliberately not square, its buildings are not symmetric, and
    the metadata carries the index and height of the first built column for the
    reader to check rather than print."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    lat, lon, zone = 53.0, -2.0, 30
    xc, yc = to_utm(lat, lon, zone)
    nx, ny, nz, dx = 24, 16, 10, 4.0           # nx != ny on purpose
    x0, y0 = xc, yc
    def box(i0, j0, i1, j1):                   # cell indices -> a lat/lon ring
        pts = [(x0 + i0*dx, y0 + j0*dx), (x0 + i1*dx, y0 + j0*dx),
               (x0 + i1*dx, y0 + j1*dx), (x0 + i0*dx, y0 + j1*dx)]
        r = [from_utm(e, n, zone) for e, n in pts]
        return [list(r) + [r[0]]]
    feats = [({"height": "20"}, box(2, 1, 5, 3)),        # tagged height
             ({"building:levels": "5"}, box(10, 6, 14, 9)),
             ({"building": "house"}, box(17, 11, 20, 14))]
    polys = assign_heights(feats, "flat", LEVEL_M, DEFAULT_M, quiet=True)
    proj = [(h, sc, [[to_utm(la, lo, zone) for la, lo in g] for g in rings])
            for h, sc, rings in polys]
    hmap, smap = rasterise(proj, nx, ny, x0, y0, dx)
    first = next((k for k, v in enumerate(hmap) if v > 0), -1)
    npy_write(out + "_heights.npy", hmap, "f4", (nx, ny))
    npy_write(out + "_source.npy", smap, "u1", (nx, ny))
    meta = {"place": "osm_city.py fixture", "nx": nx, "ny": ny, "nz": nz,
            "dx": dx, "dz": dx, "fallback": "flat",
            "built_fraction": sum(1 for v in hmap if v > 0) / (nx * ny),
            "check_i": first // ny, "check_j": first % ny,
            "check_h": hmap[first], "check_max_h": max(hmap),
            "generator": "M3LB tools/osm_city.py --emit-fixture"}
    with open(out + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fixture] {out}_heights.npy  {nx}x{ny}x{nz} at {dx} m, "
          f"built {100*meta['built_fraction']:.1f}%, first built column "
          f"i={meta['check_i']} j={meta['check_j']} h={meta['check_h']:.1f} m")
    return 0


# =============================================================================
def selftest():
    """No network, no numpy, no pyproj.  Every check is against something
    independently known rather than against this file's own output."""
    fails = []

    def ck(ok, what):
        print(f"   {what:<62} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(what)

    print("\n  projection, against pyproj values computed on 2026-09-19")
    # EPSG:4326 -> EPSG:32630 / 32618 / 32756, printed by pyproj 3.7.2.
    for lat, lon, zone, south, ex, en in (
            (53.4729842, -2.2502070, 30, False, 549766.8502, 5925149.8999),
            (40.7585152, -73.9746069, 18, False, 586551.7969, 4512456.1914),
            (-33.8688, 151.2093, 56, True, 334368.6336, 6250948.3454)):
        e, n = to_utm(lat, lon, zone, south)
        d = math.hypot(e - ex, n - en)
        ck(d < 0.005, f"({lat:.4f},{lon:.4f}) -> zone {zone} within 5 mm  [{d*1e3:.2f} mm]")
    for lat, lon, zone in ((53.4729842, -2.2502070, 30), (69.65, 18.95, 34)):
        e, n = to_utm(lat, lon, zone)
        rl, ro = from_utm(e, n, zone)
        # In metres on the ground.  The inverse is only used to turn the query
        # box back into lat/lon, where a millimetre is eight orders below the
        # bounding box's own slack, so this is a sanity check and not a budget.
        d = math.hypot((rl - lat) * 111320.0,
                       (ro - lon) * 111320.0 * math.cos(math.radians(lat)))
        ck(d < 0.005, f"forward/inverse round trip at {lat:.2f} N  [{d*1e3:.2f} mm]")
    ck(utm_zone(-2.25, -2.25)[1] == 32730 and utm_zone(2.25, -2.25)[1] == 32630,
       "hemisphere picks 327xx / 326xx")

    print("\n  height parsing")
    for raw, want in (("62.5", 62.5), ("15 m", 15.0), ("25.0;26.2", 25.0),
                      ("103'", 103 * 0.3048), ("35 ft", 35 * 0.3048),
                      (None, None), ("", None), ("nan", None),
                      ("0.5", None), ("900", None), ("yes", None)):
        v, ok = parse_height(raw)
        ck((v is None and want is None) or (ok and abs(v - want) < 1e-9),
           f"height {raw!r} -> {want}")

    print("\n  rasteriser, against areas known in closed form")
    # A 20 m square on a 1 m grid, corner-aligned to cell edges: exactly 400
    # cell centres fall inside, and that is a count no discretisation choice
    # can argue with.
    sq = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0), (10.0, 10.0)]
    h, s = rasterise([(12.0, 1, [sq])], 64, 64, 0.0, 0.0, 1.0)
    ck(sum(1 for v in h if v > 0) == 400, "20 m square on a 1 m grid -> 400 cells")
    ck(all(v == 12.0 for v in h if v > 0), "square carries its height")
    ck(sum(1 for v in s if v == 1) == 400, "square carries its provenance")

    # Courtyard: the same square with a 10 m hole leaves 400 - 100 = 300.
    hole = [(15.0, 15.0), (15.0, 25.0), (25.0, 25.0), (25.0, 15.0), (15.0, 15.0)]
    h, _ = rasterise([(12.0, 1, [sq, hole])], 64, 64, 0.0, 0.0, 1.0)
    ck(sum(1 for v in h if v > 0) == 300, "inner ring carves a 10 m courtyard")

    # Tallest wins, both ways round, and the PROVENANCE follows the height
    # rather than being maxed independently -- the trap named in the banner.
    tall = [(12.0, 12.0), (18.0, 12.0), (18.0, 18.0), (12.0, 18.0), (12.0, 12.0)]
    for order in ((40.0, 1, [tall]), (12.0, 5, [sq])), ((12.0, 5, [sq]), (40.0, 1, [tall])):
        h, s = rasterise(list(order), 64, 64, 0.0, 0.0, 1.0)
        ck(h[15 * 64 + 15] == 40.0 and s[15 * 64 + 15] == 1,
           "overlap: taller wins and takes its provenance with it")
    ck(h[11 * 64 + 11] == 12.0 and s[11 * 64 + 11] == 5, "podium keeps its own cells")

    # A triangle: area 200 m^2, so the centre count must land within a
    # perimeter's worth of it rather than being exactly right.
    tri = [(10.0, 10.0), (30.0, 10.0), (10.0, 30.0), (10.0, 10.0)]
    h, _ = rasterise([(5.0, 1, [tri])], 64, 64, 0.0, 0.0, 1.0)
    got = sum(1 for v in h if v > 0)
    ck(abs(got - 200) <= 24, f"triangle area 200 m2 -> {got} cells (perimeter band)")

    # Nothing outside the domain, and a polygon entirely outside writes nothing.
    far = [(100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 100.0)]
    h, _ = rasterise([(5.0, 1, [far])], 64, 64, 0.0, 0.0, 1.0)
    ck(sum(1 for v in h if v > 0) == 0, "polygon outside the domain writes nothing")
    # Straddling the edge: clipped, not wrapped.  A wrap would be invisible in
    # any total and would put a building on the wrong side of the city.
    edge = [(-10.0, 30.0), (5.0, 30.0), (5.0, 35.0), (-10.0, 35.0), (-10.0, 30.0)]
    h, _ = rasterise([(5.0, 1, [edge])], 64, 64, 0.0, 0.0, 1.0)
    ck(sum(1 for v in h if v > 0) == 25 and all(h[j * 64 + 63] == 0 for j in range(64)),
       "polygon straddling x=0 is clipped, not wrapped")

    print("\n  npy round trip, and the shape HeightField.hpp insists on")
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t_heights.npy")
    vals = [0.0, 1.5, 300.25, 7.0] * 4
    npy_write(p, vals, "f4", (4, 4))
    back, shape = npy_read(p)
    ck(shape == (4, 4) and list(back) == vals, "float32 (4,4) round trip")
    head = open(p, "rb").read(80)
    ck(b"'fortran_order': False" in head, "header carries the exact C-order string")
    ck(len(head[:10]) == 10 and head[:6] == b"\x93NUMPY", "magic and version")
    npy_write(os.path.join(d, "s.npy"), [0, 1, 5, 255], "u1", (2, 2))
    back, shape = npy_read(os.path.join(d, "s.npy"))
    ck(shape == (2, 2) and list(back) == [0, 1, 5, 255], "uint8 round trip")

    print("\n  index order: i is easting, j is northing, index = i*ny + j")
    # One cell, at a known metric offset from the origin, must land at a known
    # linear index.  A transpose here rotates the city and nothing downstream
    # complains, which is why it is pinned rather than inferred.
    one = [(21.0, 41.0), (24.0, 41.0), (24.0, 44.0), (21.0, 44.0), (21.0, 41.0)]
    h, _ = rasterise([(9.0, 1, [one])], 64, 48, 0.0, 0.0, 1.0)
    idx = [k for k, v in enumerate(h) if v > 0]
    # nx != ny deliberately: on a square grid a transpose is undetectable here,
    # and a transpose is the failure this check exists for.
    ck(idx and all(k // 48 in (21, 22, 23) and k % 48 in (41, 42, 43) for k in idx),
       "a cell at (21..24 m E, 41..44 m N) lands at i=21..23, j=41..43")
    ck(len(idx) == 9, "and covers exactly 3x3 cells")

    print("\n  osm parsing, both response shapes")
    geom = {"elements": [{"type": "way", "id": 1, "tags": {"building": "yes"},
                          "geometry": [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 1e-4},
                                       {"lat": 1e-4, "lon": 1e-4}, {"lat": 0.0, "lon": 0.0}]}]}
    refs = {"elements": [{"type": "node", "id": 10, "lat": 0.0, "lon": 0.0},
                         {"type": "node", "id": 11, "lat": 0.0, "lon": 1e-4},
                         {"type": "node", "id": 12, "lat": 1e-4, "lon": 1e-4},
                         {"type": "way", "id": 2, "tags": {"building": "yes"},
                          "nodes": [10, 11, 12, 10]}]}
    fa, _ = parse_osm([geom])
    fb, _ = parse_osm([refs])
    ck(len(fa) == 1 and len(fb) == 1, "out geom; and node-ref forms both parse")
    ck(fa[0][1][0] == fb[0][1][0], "and give identical rings")
    # A type=building grouping relation must NOT become a footprint of its own.
    grp = {"elements": [{"type": "relation", "id": 3,
                         "tags": {"type": "building", "building": "yes"},
                         "members": [{"type": "way", "ref": 2, "role": "outline"}]},
                        *refs["elements"]]}
    fc, st = parse_osm([grp])
    ck(len(fc) == 1 and st["grouping_relations"] == 1,
       "Simple 3D Buildings relation is skipped, its ways are not")
    # A multipolygon in the node-reference form: the members are bare refs and
    # the member way is UNTAGGED, so if ref resolution fails the building
    # disappears in silence.  This is the shape osmnx caches are written in.
    mp_refs = {"elements": [
        {"type": "node", "id": 20, "lat": 0.0, "lon": 0.0},
        {"type": "node", "id": 21, "lat": 0.0, "lon": 2e-4},
        {"type": "node", "id": 22, "lat": 2e-4, "lon": 2e-4},
        {"type": "node", "id": 23, "lat": 2e-4, "lon": 0.0},
        {"type": "way", "id": 30, "nodes": [20, 21, 22, 23, 20]},
        {"type": "relation", "id": 40,
         "tags": {"type": "multipolygon", "building": "yes", "height": "30"},
         "members": [{"type": "way", "ref": 30, "role": "outer"}]}]}
    fe, _ = parse_osm([mp_refs])
    ck(len(fe) == 1 and len(fe[0][1][0]) == 5,
       "multipolygon member resolved by ref, not read in place")
    ck(fe and fe[0][0].get("height") == "30",
       "and the relation's tags, not the untagged member's, carry the height")
    # Same relation with the geometry inline must give the same ring.
    mp_geom = {"elements": [
        {"type": "relation", "id": 41,
         "tags": {"type": "multipolygon", "building": "yes"},
         "members": [{"type": "way", "ref": 31, "role": "outer",
                      "geometry": [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 2e-4},
                                   {"lat": 2e-4, "lon": 2e-4}, {"lat": 2e-4, "lon": 0.0},
                                   {"lat": 0.0, "lon": 0.0}]}]}]}
    ff, _ = parse_osm([mp_geom])
    ck(ff and ff[0][1][0] == fe[0][1][0],
       "and both response shapes give the identical ring")
    # Two open ways that only form a ring once stitched end to end.
    split = {"elements": [
        {"type": "node", "id": 50, "lat": 0.0, "lon": 0.0},
        {"type": "node", "id": 51, "lat": 0.0, "lon": 2e-4},
        {"type": "node", "id": 52, "lat": 2e-4, "lon": 2e-4},
        {"type": "way", "id": 60, "nodes": [50, 51]},
        {"type": "way", "id": 61, "nodes": [51, 52, 50]},
        {"type": "relation", "id": 62, "tags": {"type": "multipolygon", "building": "y"},
         "members": [{"type": "way", "ref": 60, "role": "outer"},
                     {"type": "way", "ref": 61, "role": "outer"}]}]}
    fg, _ = parse_osm([split])
    ck(len(fg) == 1 and fg[0][1][0][0] == fg[0][1][0][-1],
       "two open member ways are stitched into one closed ring")

    unclosed = {"elements": [{"type": "way", "id": 4, "tags": {"building": "yes"},
                              "geometry": [{"lat": 0.0, "lon": 0.0},
                                           {"lat": 0.0, "lon": 1e-4}]}]}
    fd, st = parse_osm([unclosed])
    ck(not fd and st["unclosed"] == 1, "an unclosed way is dropped and counted")

    print("\n  height ladder")
    F = [({"height": "50"}, [sq]), ({"building:levels": "4"}, [sq]),
         ({"building": "house"}, [sq])]
    got = assign_heights(F, "flat", 3.2, 10.0, quiet=True)
    ck([g[0] for g in got] == [50.0, 12.8, 10.0], "flat: tag, levels x 3.2, default")
    ck([g[1] for g in got] == [1, 2, 5], "flat: provenance codes")
    F2 = F + [({"building": "house", "building:levels": "3"}, [sq])] * MIN_TYPE_N
    got = assign_heights(F2, "type", 3.2, 10.0, quiet=True)
    ck(abs(got[2][0] - 9.6) < 1e-9 and got[2][1] == 3,
       f"type: a house with no height takes the house median (3 storeys)")
    got = assign_heights([({"building": "office"}, [sq])] + F2[3:], "type", 3.2, 10.0,
                         quiet=True)
    ck(got[0][1] == 4, "type: a type with too few examples falls to the city median")

    print(f"\n  {'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILURE(S)'}")
    return 1 if fails else 0


# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="OpenStreetMap city -> M3LB height field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples
  %(prog)s --place "Manchester city centre, UK" --domain-m 2000 --dx 5 \\
      --top-m 300 --out geom/mcr
  build/demonstrator/urban -geom geom/mcr -bearing 250 -minutes 20 \\
      --kokkos-num-threads=4
  %(prog)s --selftest          # no network: projection, raster, parser, npy""")
    p.add_argument("--place", help="name to geocode with Nominatim")
    p.add_argument("--lat", type=float, help="domain centre, instead of --place")
    p.add_argument("--lon", type=float)
    p.add_argument("--domain-m", type=float, default=2000.0, help="square side, m")
    p.add_argument("--dx", type=float, default=5.0,
                   help="cell size, m.  dz == dx: the solver is isotropic")
    p.add_argument("--top-m", type=float, default=300.0,
                   help="domain height, m.  Must clear the tallest building with "
                        "room for the boundary layer above it")
    p.add_argument("--fallback", choices=("flat", "type"), default="flat",
                   help="height for a building with no height and no levels")
    p.add_argument("--level-m", type=float, default=LEVEL_M)
    p.add_argument("--default-m", type=float, default=DEFAULT_M)
    p.add_argument("--out", default="city", help="output prefix")
    p.add_argument("--cache", default=None,
                   help="directory for Nominatim/Overpass responses "
                        "(default <out>_osm_cache)")
    p.add_argument("--osmnx-cache", default=None,
                   help="read an osmnx cache directory instead of querying; "
                        "every .json in it holding OSM elements is merged")
    p.add_argument("--overpass-url", default="https://overpass-api.de/api/interpreter")
    p.add_argument("--compare-to", default=None,
                   help="a reference <name>_heights.npy to diff the result against")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--emit-fixture", default=None,
                   help="write a small synthetic city to this prefix, for the "
                        "C++ reader's cross-language check.  No network")
    a = p.parse_args()

    if a.selftest:
        return selftest()
    if a.emit_fixture:
        return emit_fixture(a.emit_fixture)
    if a.lat is None and not a.place:
        p.error("one of --place or --lat/--lon is required")

    cache = a.cache or (a.out + "_osm_cache")
    if a.lat is not None:
        lat, lon = a.lat, a.lon
        place = a.place or f"{lat:.5f},{lon:.5f}"
    else:
        lat, lon = geocode(a.place, cache)
        place = a.place

    nx = int(round(a.domain_m / a.dx))
    ny = nx
    nz = int(round(a.top_m / a.dx))
    zone, epsg = utm_zone(lat, lon)
    south = lat < 0
    xc, yc = to_utm(lat, lon, zone, south)
    x0, y0 = xc - a.domain_m / 2, yc - a.domain_m / 2
    print(f"[domain] {place}: {lat:.5f},{lon:.5f}  EPSG:{epsg}")
    print(f"         {a.domain_m:.0f} m square at {a.dx} m -> {nx}x{ny}, "
          f"{a.top_m:.0f} m tall -> {nz} layers = {nx*ny*nz:,} cells")

    # Query beyond the domain so a building straddling the edge is whole, and
    # so the per-type calibration has more than the domain's own buildings to
    # work from.  0.6 is the reference's padding and is kept for comparability.
    if a.osmnx_cache:
        docs = []
        for f in sorted(os.listdir(a.osmnx_cache)):
            if not f.endswith(".json"):
                continue
            try:
                j = json.load(open(os.path.join(a.osmnx_cache, f)))
            except Exception:                              # noqa: BLE001
                continue
            if isinstance(j, dict) and "elements" in j:
                docs.append(j)
                ts = j.get("osm3s", {}).get("timestamp_osm_base", "?")
                print(f"[cache]  {f[:16]}...: {len(j['elements']):,} elements, "
                      f"OSM snapshot {ts}")
        if not docs:
            raise SystemExit(f"no Overpass responses found in {a.osmnx_cache}")
    else:
        pad = a.domain_m * 0.6
        s_lat, w_lon = from_utm(xc - pad, yc - pad, zone, south)
        n_lat, e_lon = from_utm(xc + pad, yc + pad, zone, south)
        docs = [overpass(s_lat, w_lon, n_lat, e_lon, cache, a.overpass_url)]

    feats, st = parse_osm(docs)
    print(f"[osm]    {st['ways']:,} ways, {st['relations']:,} relations, "
          f"{st['nodes']:,} nodes -> {len(feats):,} footprints")
    if st["grouping_relations"]:
        print(f"         {st['grouping_relations']:,} Simple 3D Buildings relations "
              f"skipped (their ways are taken directly)")
    if st["unclosed"]:
        print(f"         {st['unclosed']:,} unclosed ways/rings dropped")

    polys = assign_heights(feats, a.fallback, a.level_m, a.default_m)
    counts = {}
    for _, s, _ in polys:
        counts[s] = counts.get(s, 0) + 1
    for k in sorted(counts):
        print(f"         {SRC[k]:16} {counts[k]:6,} "
              f"({100*counts[k]/max(len(polys),1):5.1f}%)")

    # Project once, here: the rasteriser works in metres and must not know
    # about the ellipsoid.
    proj = [(h, s, [[to_utm(la, lo, zone, south) for la, lo in g] for g in rings])
            for h, s, rings in polys]
    t = time.time()
    hmap, smap = rasterise(proj, nx, ny, x0, y0, a.dx)
    built = sum(1 for v in hmap if v > 0)
    n = nx * ny
    print(f"[raster] {time.time()-t:.1f} s   built fraction {100*built/n:.1f}%, "
          f"open {100*(n-built)/n:.1f}%")
    if built:
        hs = sorted(v for v in hmap if v > 0)
        print(f"         building height: median {hs[len(hs)//2]:.1f} m, "
              f"p95 {hs[int(0.95*len(hs))]:.1f} m, max {hs[-1]:.1f} m")
        over = sum(1 for v in hmap if v > a.top_m)
        if over:
            print(f"         WARNING {over:,} cells exceed the {a.top_m:.0f} m top "
                  f"and will be truncated by the solver")
    cell_src = {}
    for v in smap:
        cell_src[v] = cell_src.get(v, 0) + 1
    print("         by area: " + ", ".join(
        f"{SRC[k]} {100*v/n:.1f}%" for k, v in sorted(cell_src.items()) if k))

    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    npy_write(a.out + "_heights.npy", hmap, "f4", (nx, ny))
    npy_write(a.out + "_source.npy", smap, "u1", (nx, ny))
    meta = {"place": place, "lat": lat, "lon": lon, "utm_crs": f"EPSG:{epsg}",
            "x0": x0, "y0": y0, "domain_m": a.domain_m,
            "nx": nx, "ny": ny, "nz": nz, "dx": a.dx, "dz": a.dx,
            "top_m": a.top_m, "level_m": a.level_m, "default_m": a.default_m,
            "fallback": a.fallback, "n_buildings": len(polys),
            "built_fraction": built / n,
            "source_1_height_tag": counts.get(1, 0),
            "source_2_levels": counts.get(2, 0),
            "source_3_type_calibrated": counts.get(3, 0),
            "source_4_city_median": counts.get(4, 0),
            "source_5_flat_default": counts.get(5, 0),
            "source_codes": " ".join(f"{k}={v}" for k, v in SRC.items()),
            "generator": "M3LB tools/osm_city.py",
            "osm_source": a.osmnx_cache or a.overpass_url}
    with open(a.out + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[write]  {a.out}_heights.npy, {a.out}_source.npy, {a.out}_meta.json")

    if a.compare_to:
        same = compare(hmap, a.compare_to, nx, ny, a.dx)
        # Deliberately NOT an exit code. A difference here is only a defect when
        # the two sides read the same OSM snapshot; against a live query it is
        # editing, which is the more common use and is not a failure. The
        # caller reads the breakdown -- one-sided differences mean missing
        # footprints, two-sided ones mean the map moved.
        print("         (differences are reported, not exit-coded: against a "
              "live query\n          they are edits, not defects)"
              if not same else "         EXACT")
    print(f"\n  run it:  build/demonstrator/urban -geom {a.out} "
          f"-bearing 250 -diff 20 -minutes 12 --kokkos-num-threads=4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
