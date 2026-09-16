#!/usr/bin/env python3
# =============================================================================
#  check_vti.py -- independent validator for binary VTK ImageData (.vti) and
#  for the .pvd collection that indexes a time series of them.
#
#  WHY THIS EXISTS.  ParaView is a forgiving reader right up to the point where
#  it is not: an appended block whose offset is one byte out does not produce an
#  error dialog, it produces a field of 1e+38 garbage, or a silent load of the
#  WRONG array into the right name.  A writer that gets WholeExtent inclusive/
#  exclusive wrong opens fine and shows a grid one plane too large, filled from
#  whatever follows it in the appended section.  Neither of those is visible in
#  a screenshot of a turbulent field.  So it never trusts the file's own frame:
#  it re-derives every length from the XML header (extent -> npoints ->
#  ncomp*npoints*sizeof) and then REQUIRES the byte stream to agree exactly.
#
#  WHAT IT CHECKS
#    1. XML parses; root is VTKFile type="ImageData"; byte_order matches this
#       machine; header_type present and one of UInt32/UInt64.
#    2. WholeExtent == Piece Extent, extents are INCLUSIVE, so
#       npoints = (xmax-xmin+1)(ymax-ymin+1)(zmax-zmin+1); cells are one fewer
#       per axis (floored at 1).  Every DataArray's value count must equal
#       ncomp * npoints (PointData) or ncomp * ncells (CellData).
#    3. format="appended": the section opens with '_'; each offset lands exactly
#       on a block boundary; each block starts with a header_type-sized byte
#       count; that count == ncomp*n*sizeof(type); and the blocks TILE the
#       section -- first at 0, each starting where the previous ended, nothing
#       left over but whitespace.
#    4. Every value decodes and is finite; min/max/mean reported per array, and
#       a constant array is flagged -- that is what a field which was allocated
#       but never written looks like, and no framing check can fail on it.
#    5. Scalars=/Vectors= name arrays that exist; NumberOfComponents="3" arrays
#       really carry a multiple of 3 values.
#    6. .pvd: root type="Collection"; every referenced file exists; timesteps
#       strictly increase (per part); each referenced .vti gets all of the above.
#
#  WHAT IT DOES NOT CHECK.  Compressed appended data (compressor=...) is
#  REJECTED rather than skipped -- a checker that prints "ok" on a file it did
#  not read is worse than no checker.  It cannot see point ORDER: x-fastest,
#  then y, then z is a convention no byte in the file records, so a transposed
#  dump passes here and looks wrong only in ParaView.  It does not check that
#  Origin/Spacing are physically meaningful, only that they are three finite
#  numbers with non-zero spacing.
#
#  STDLIB ONLY -- no numpy, no lxml.  (This tree's python has neither.)
#
#  USAGE
#    check_vti.py FILE.vti [FILE2.vti ...]
#    check_vti.py FILE.pvd            # follows the collection
#    check_vti.py DIR/                # every .pvd and .vti in DIR
#    options: --limit N   check at most N datasets of a collection (evenly
#                         spaced, always including the first and the last)
#             --quiet     one line per file unless it fails
#             --no-values skip the decode/min/max pass (framing checks only)
#  Exit status 0 = every file passed, 1 = at least one failed, 2 = bad usage.
# =============================================================================

import base64
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
from array import array

# VTK type name -> (struct/array typecode, size in bytes, is_float)
VTK_TYPES = {
    "Int8": ("b", 1, False),
    "UInt8": ("B", 1, False),
    "Int16": ("h", 2, False),
    "UInt16": ("H", 2, False),
    "Int32": ("i", 4, False),
    "UInt32": ("I", 4, False),
    "Int64": ("q", 8, False),
    "UInt64": ("Q", 8, False),
    "Float32": ("f", 4, True),
    "Float64": ("d", 8, True),
}

HEADER_TYPES = {"UInt32": 4, "UInt64": 8}

MACHINE_ORDER = "LittleEndian" if sys.byteorder == "little" else "BigEndian"

# --no-values: keep every framing check, skip the decode/statistics pass.  The
# framing is what catches a broken writer; the values are what catch a broken
# simulation, and on a 500-frame series only the first is worth the wait.
NO_VALUES = False


class Fatal(Exception):
    """Raised when the file cannot be checked any further."""


class Report:
    """Collects failures for one file.  Failures accumulate; a Fatal stops."""

    def __init__(self, path, quiet=False):
        self.path = path
        self.quiet = quiet
        self.fails = []
        self.warns = []
        self.lines = []

    def info(self, msg):
        self.lines.append("        " + msg)

    def ok(self, msg):
        self.lines.append("  ok    " + msg)

    def warn(self, msg):
        self.warns.append(msg)
        self.lines.append("  WARN  " + msg)

    def fail(self, msg):
        self.fails.append(msg)
        self.lines.append("  FAIL  " + msg)

    def flush(self):
        bad = bool(self.fails)
        if self.quiet and not bad and not self.warns:
            print("PASS  %s" % self.path)
            return
        print("=== %s" % self.path)
        for ln in self.lines:
            print(ln)
        if bad:
            print("  ---> FAILED: %d problem(s)" % len(self.fails))
            for m in self.fails:
                print("       * " + m)
        else:
            print("  ---> PASS%s" % (" (%d warning(s))" % len(self.warns)
                                     if self.warns else ""))
        print("")


# -----------------------------------------------------------------------------
# XML plumbing.  The appended section is RAW BINARY sitting inside an XML
# document, which no XML parser can be handed directly: it will contain '<',
# '&' and very likely the bytes of '</AppendedData>' itself by chance.  So cut
# the payload out at the byte level first and parse what is left.
# -----------------------------------------------------------------------------
def local(tag):
    return tag.rsplit("}", 1)[-1]


def split_appended(raw, rep):
    """-> (xml_bytes, payload, payload_file_offset) ; payload None if absent."""
    open_tag = raw.find(b"<AppendedData")
    if open_tag < 0:
        return raw, None, None
    if raw.find(b"<AppendedData", open_tag + 1) >= 0:
        raise Fatal("more than one <AppendedData> element")
    gt = raw.find(b">", open_tag)
    if gt < 0:
        raise Fatal("<AppendedData> tag is not closed")
    k = gt + 1
    while k < len(raw) and raw[k:k + 1] in b" \t\r\n":
        k += 1
    if raw[k:k + 1] != b"_":
        got = raw[k:k + 12]
        raise Fatal("AppendedData does not start with the '_' marker "
                    "(first non-space byte after the tag is %r); offsets are "
                    "measured from the byte AFTER that underscore" % got)
    start = k + 1
    end = raw.rfind(b"</AppendedData>")
    if end < start:
        raise Fatal("no closing </AppendedData> after the '_' marker")
    return raw[:start] + raw[end:], raw[start:end], start


def parse_file(path, rep):
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw:
        raise Fatal("file is empty (0 bytes)")
    xml_bytes, payload, payload_at = split_appended(raw, rep)
    try:
        root = ET.fromstring(xml_bytes.decode("utf-8", "replace"))
    except ET.ParseError as e:
        raise Fatal("XML does not parse: %s" % e)
    return root, payload, payload_at, len(raw)


# -----------------------------------------------------------------------------
# value decoding and statistics
# -----------------------------------------------------------------------------
def decode_values(buf, vtk_type, byte_order, rep, who):
    """bytes -> python list of numbers, honouring the DECLARED byte order."""
    code, size, is_float = VTK_TYPES[vtk_type]
    if len(buf) % size:
        rep.fail("%s: %d bytes is not a whole number of %s values (%d B each)"
                 % (who, len(buf), vtk_type, size))
        buf = buf[:len(buf) - (len(buf) % size)]
    arr = array(code)
    if arr.itemsize != size:
        vals = list(struct.unpack(
            ("<" if byte_order == "LittleEndian" else ">")
            + "%d%s" % (len(buf) // size, code), buf))
        return vals
    arr.frombytes(buf)
    if byte_order != MACHINE_ORDER:
        arr.byteswap()
    return arr.tolist()


def stats(vals, is_float, rep, who):
    """min/max/mean, and an exact hunt for non-finite values when needed."""
    if not vals:
        return None
    bad = []
    if is_float:
        total = math.fsum(vals) if len(vals) < 4_000_000 else sum(vals)
        lo, hi = min(vals), max(vals)
        if not (math.isfinite(total) and math.isfinite(lo)
                and math.isfinite(hi)):
            for i, v in enumerate(vals):
                if not math.isfinite(v):
                    bad.append((i, v))
                    if len(bad) >= 8:
                        break
            finite = [v for v in vals if math.isfinite(v)]
            if bad:
                shown = ", ".join("[%d]=%r" % b for b in bad)
                rep.fail("%s: NON-FINITE value(s): %s%s"
                         % (who, shown, " ..." if len(bad) >= 8 else ""))
            if not finite:
                return None
            lo, hi = min(finite), max(finite)
            total = math.fsum(finite)
            return lo, hi, total / len(finite), len(bad)
        return lo, hi, total / len(vals), 0
    lo, hi = min(vals), max(vals)
    return lo, hi, sum(vals) / len(vals), 0


def fmt_stats(st):
    if st is None:
        return "min/max/mean: (no finite values)"
    lo, hi, mean, nbad = st
    return "min %-14.6g max %-14.6g mean %-14.6g%s" % (
        lo, hi, mean, "  (%d NON-FINITE)" % nbad if nbad else "")


# -----------------------------------------------------------------------------
# one .vti
# -----------------------------------------------------------------------------
TALLY = {"pass": 0, "fail": 0, "failed_paths": []}


def check_vti(path, quiet=False):
    rep = Report(path, quiet)
    try:
        _check_vti(path, rep)
    except Fatal as e:
        rep.fail(str(e))
    rep.flush()
    if rep.fails:
        TALLY["fail"] += 1
        TALLY["failed_paths"].append(path)
    else:
        TALLY["pass"] += 1
    return not rep.fails


def _check_vti(path, rep):
    root, payload, payload_at, filesize = parse_file(path, rep)

    # -- 1. root element ------------------------------------------------------
    if local(root.tag) != "VTKFile":
        raise Fatal("root element is <%s>, expected <VTKFile>"
                    % local(root.tag))
    ftype = root.get("type")
    if ftype != "ImageData":
        raise Fatal('VTKFile type="%s", expected "ImageData" for a .vti'
                    % ftype)
    byte_order = root.get("byte_order")
    if byte_order is None:
        rep.fail("VTKFile has no byte_order attribute")
        byte_order = MACHINE_ORDER
    elif byte_order not in ("LittleEndian", "BigEndian"):
        rep.fail('byte_order="%s" is neither LittleEndian nor BigEndian'
                 % byte_order)
        byte_order = MACHINE_ORDER
    elif byte_order != MACHINE_ORDER:
        rep.fail('byte_order="%s" but this machine is %s; ParaView will '
                 "byte-swap every value it reads" % (byte_order,
                                                     MACHINE_ORDER))
    compressor = root.get("compressor")
    ver = root.get("version")
    if ver is None:
        rep.warn("VTKFile has no version attribute (harmless, but unusual)")

    header_type = root.get("header_type")
    hsize = HEADER_TYPES.get(header_type or "", None)
    if payload is not None or _has_binary(root):
        if header_type is None:
            rep.fail("header_type is missing; a binary/appended file needs "
                     'header_type="UInt32" or "UInt64" (VTK assumes UInt32, '
                     "so a 64-bit writer silently misreads every block)")
            hsize = 4
            header_type = "UInt32(assumed)"
        elif hsize is None:
            rep.fail('header_type="%s" is not UInt32 or UInt64' % header_type)
            hsize = 4
    rep.ok('VTKFile type="ImageData" version=%s byte_order="%s" header_type=%s'
           % (ver, byte_order, header_type))
    rep.info("%d bytes on disk, %s"
             % (filesize, "%d of them appended payload (%.1f%% XML header)"
                % (len(payload), 100.0 * (filesize - len(payload)) / filesize)
                if payload is not None else "no appended section"))
    if compressor:
        raise Fatal('compressor="%s" is set: this checker does NOT decode '
                    "compressed appended blocks, so it cannot vouch for this "
                    "file.  Re-run the writer without compression to validate "
                    "it." % compressor)

    # -- 2. extents -----------------------------------------------------------
    img = [c for c in root if local(c.tag) == "ImageData"]
    if len(img) != 1:
        raise Fatal("expected exactly one <ImageData> child of <VTKFile>, "
                    "found %d" % len(img))
    img = img[0]
    whole = parse_extent(img.get("WholeExtent"), "WholeExtent", rep)
    check_triple(img.get("Origin"), "Origin", rep, allow_zero=True)
    sp = check_triple(img.get("Spacing"), "Spacing", rep, allow_zero=False)
    if sp is not None and any(v < 0 for v in sp):
        rep.warn("Spacing has a negative component %r; ParaView accepts it "
                 "but the image will be mirrored" % (sp,))

    pieces = [c for c in img if local(c.tag) == "Piece"]
    if len(pieces) == 0:
        raise Fatal("<ImageData> has no <Piece>")
    if len(pieces) > 1:
        rep.fail("%d <Piece> elements: a serial .vti must have exactly one "
                 "(the reader takes the first and drops the rest)"
                 % len(pieces))
    piece = pieces[0]
    pext = parse_extent(piece.get("Extent"), "Piece Extent", rep)
    if whole is None or pext is None:
        raise Fatal("cannot continue without both extents")
    if whole != pext:
        rep.fail("Piece Extent %s does not equal WholeExtent %s"
                 % (" ".join(map(str, pext)), " ".join(map(str, whole))))

    nx = whole[1] - whole[0] + 1
    ny = whole[3] - whole[2] + 1
    nz = whole[5] - whole[4] + 1
    if min(nx, ny, nz) < 1:
        raise Fatal("extent %s is empty or inverted (max < min on some axis); "
                    "extents are INCLUSIVE" % " ".join(map(str, whole)))
    npoints = nx * ny * nz
    ncells = max(nx - 1, 1) * max(ny - 1, 1) * max(nz - 1, 1)
    rep.ok("extent %s -> %d x %d x %d = %d points (%d cells); extents are "
           "INCLUSIVE" % (" ".join(map(str, whole)), nx, ny, nz, npoints,
                          ncells))

    # -- 3. gather the DataArrays --------------------------------------------
    groups = []
    for g in piece:
        if local(g.tag) in ("PointData", "CellData"):
            groups.append((local(g.tag), g))
        elif local(g.tag) in ("Points", "Coordinates", "Cells"):
            rep.fail("<%s> has no place in ImageData (that is an "
                     "UnstructuredGrid/RectilinearGrid element)" % local(g.tag))
    if not groups:
        rep.fail("the <Piece> carries neither <PointData> nor <CellData>: "
                 "the file opens in ParaView and shows nothing")

    arrays = []          # dicts describing every DataArray
    for gname, g in groups:
        n_expect = npoints if gname == "PointData" else ncells
        seen = {}
        for da in g:
            if local(da.tag) != "DataArray":
                rep.warn("<%s> inside <%s> is not a DataArray; ignored"
                         % (local(da.tag), gname))
                continue
            a = describe_array(da, gname, n_expect, rep)
            if a is None:
                continue
            if a["name"] in seen:
                rep.fail('two arrays named "%s" in <%s>; ParaView keeps only '
                         "one" % (a["name"], gname))
            seen[a["name"]] = a
            arrays.append(a)
        # -- 5. Scalars=/Vectors= must name arrays that exist -----------------
        for attr in ("Scalars", "Vectors", "Tensors", "Normals", "TCoords"):
            want = g.get(attr)
            if not want:
                continue
            if want not in seen:
                rep.fail('<%s %s="%s"> names an array that does not exist '
                         "here (present: %s)"
                         % (gname, attr, want,
                            ", ".join(sorted(seen)) or "none"))
            elif attr == "Vectors" and seen[want]["ncomp"] != 3:
                rep.fail('<%s Vectors="%s"> but that array has '
                         'NumberOfComponents="%d"; ParaView will not treat it '
                         "as a vector" % (gname, want, seen[want]["ncomp"]))
            elif attr == "Scalars" and seen[want]["ncomp"] != 1:
                rep.warn('<%s Scalars="%s"> names a %d-component array'
                         % (gname, want, seen[want]["ncomp"]))
    if not arrays:
        rep.fail("no usable DataArray in the file")
        return

    # -- 4. appended section: headers, tiling, values -------------------------
    app = [a for a in arrays if a["format"] == "appended"]
    if app and payload is None:
        rep.fail("%d array(s) declare format=\"appended\" but the file has no "
                 "<AppendedData> section" % len(app))
        app = []
    if payload is not None and not app:
        rep.warn("there is an <AppendedData> section but no array points into "
                 "it (%d bytes dead)" % len(payload))

    if app:
        enc = None
        for e in root.iter():
            if local(e.tag) == "AppendedData":
                enc = e.get("encoding")
        if enc is None:
            # vtkXMLDataParser takes raw ONLY on an explicit encoding="raw";
            # anything else, missing included, is decoded as base64, which
            # turns a raw payload into garbage without an error.
            rep.fail('<AppendedData> has no encoding attribute: VTK treats '
                     "everything that is not encoding=\"raw\" as base64, so a "
                     "raw payload is decoded as base64 and silently ruined")
            enc = "raw"
        if enc not in ("raw", "base64"):
            raise Fatal('AppendedData encoding="%s" is neither raw nor base64'
                        % enc)
        rep.ok("AppendedData: encoding=%s, '_' marker at file offset %d, "
               "%d bytes of payload" % (enc, payload_at - 1, len(payload)))
        if enc == "raw":
            read_appended_raw(app, payload, hsize, byte_order, rep)
        else:
            read_appended_base64(app, payload, hsize, byte_order, rep)

    # inline arrays (ascii / base64 "binary")
    for a in arrays:
        if a["format"] == "ascii":
            read_ascii(a, rep)
        elif a["format"] == "binary":
            read_inline_binary(a, hsize or 4, byte_order, rep)
        elif a["format"] not in ("appended",):
            rep.fail('%s: format="%s" is not ascii, binary or appended'
                     % (a["who"], a["format"]))

    # -- 5b. per-array length checks and statistics ---------------------------
    rep.info("%-12s %-8s %4s %10s   %s"
             % ("array", "type", "comp", "values", "statistics"))
    for a in arrays:
        vals = a.get("values")
        n = len(vals) if vals is not None else a.get("nvalues")
        if n is None:
            rep.info("%-12s %-8s %4d %10s   (not decoded)"
                     % (a["name"], a["type"], a["ncomp"], "-"))
            continue
        if a["ncomp"] != 1 and n % a["ncomp"]:
            rep.fail('%s: NumberOfComponents="%d" but the array holds %d '
                     "values, which is not divisible by %d (%d left over) -- "
                     "it cannot be a %d-component field"
                     % (a["who"], a["ncomp"], n, a["ncomp"],
                        n % a["ncomp"], a["ncomp"]))
        if n != a["nexpect"]:
            rep.fail("%s: holds %d values but the %s extent needs "
                     "%d component * %d %s = %d"
                     % (a["who"], n, "point" if a["group"] == "PointData"
                        else "cell", a["ncomp"], a["nelem"],
                        "points" if a["group"] == "PointData" else "cells",
                        a["nexpect"]))
        if vals is None:
            rep.info("%-12s %-8s %4d %10d   (values not decoded: --no-values)"
                     % (a["name"], a["type"], a["ncomp"], n))
            continue
        st = stats(vals, VTK_TYPES[a["type"]][2], rep, a["who"])
        rep.info("%-12s %-8s %4d %10d   %s"
                 % (a["name"], a["type"], a["ncomp"], n, fmt_stats(st)))
        # A field that was never filled in -- the mirror copy that was not
        # taken, the compute_*() that was not called -- is byte-for-byte a
        # valid array of one repeated value.  Nothing above can fail on it.
        if st is not None and st[0] == st[1] and n > 1:
            rep.warn("%s: every one of the %d values is %g; that is also what "
                     "an array that was allocated but never written looks like"
                     % (a["who"], n, st[0]))
        # VTK sometimes advertises the range; cross-check it if it did.
        if st is not None:
            for attr, idx in (("RangeMin", 0), ("RangeMax", 1)):
                sv = a["elem"].get(attr)
                if sv is None:
                    continue
                try:
                    want = float(sv)
                except ValueError:
                    rep.warn("%s: %s=%r is not a number" % (a["who"], attr, sv))
                    continue
                got = st[idx] if a["ncomp"] == 1 else None
                if got is not None and not close(want, got):
                    rep.warn("%s: %s advertises %g, data gives %g"
                             % (a["who"], attr, want, got))


def _has_binary(root):
    for e in root.iter():
        if local(e.tag) == "DataArray" and e.get("format") in ("appended",
                                                               "binary"):
            return True
    return False


def close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def parse_extent(s, what, rep):
    if s is None:
        rep.fail("%s attribute is missing" % what)
        return None
    parts = s.split()
    if len(parts) != 6:
        rep.fail("%s=%r has %d numbers, expected 6" % (what, s, len(parts)))
        return None
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        rep.fail("%s=%r is not six integers" % (what, s))
        return None


def check_triple(s, what, rep, allow_zero):
    if s is None:
        rep.fail("%s attribute is missing" % what)
        return None
    parts = s.split()
    if len(parts) != 3:
        rep.fail("%s=%r has %d numbers, expected 3" % (what, s, len(parts)))
        return None
    try:
        v = [float(p) for p in parts]
    except ValueError:
        rep.fail("%s=%r is not three numbers" % (what, s))
        return None
    if not all(math.isfinite(x) for x in v):
        rep.fail("%s=%r contains a non-finite number" % (what, s))
    if not allow_zero and any(x == 0.0 for x in v):
        rep.fail("%s=%r has a zero component; the image collapses to a plane "
                 "and ParaView renders nothing" % (what, s))
    return v


def describe_array(da, group, nelem, rep):
    name = da.get("Name")
    if not name:
        rep.fail("a DataArray in <%s> has no Name attribute" % group)
        name = "<unnamed>"
    who = '%s "%s"' % (group, name)
    vtype = da.get("type")
    if vtype not in VTK_TYPES:
        rep.fail('%s: type="%s" is not a VTK scalar type' % (who, vtype))
        return None
    try:
        ncomp = int(da.get("NumberOfComponents", "1"))
    except ValueError:
        rep.fail("%s: NumberOfComponents=%r is not an integer"
                 % (who, da.get("NumberOfComponents")))
        return None
    if ncomp < 1:
        rep.fail("%s: NumberOfComponents=%d" % (who, ncomp))
        return None
    fmt = da.get("format")
    if fmt is None:
        rep.fail("%s: no format attribute (need ascii, binary or appended)"
                 % who)
        fmt = "ascii"
    a = {"elem": da, "name": name, "who": who, "group": group, "type": vtype,
         "ncomp": ncomp, "format": fmt, "nelem": nelem,
         "nexpect": ncomp * nelem,
         "nbytes_expect": ncomp * nelem * VTK_TYPES[vtype][1],
         "values": None, "offset": None}
    if fmt == "appended":
        off = da.get("offset")
        if off is None:
            rep.fail('%s: format="appended" but no offset attribute' % who)
            return a
        try:
            a["offset"] = int(off)
        except ValueError:
            rep.fail("%s: offset=%r is not an integer" % (who, off))
            return a
        if a["offset"] < 0:
            rep.fail("%s: negative offset %d" % (who, a["offset"]))
            a["offset"] = None
    return a


# -----------------------------------------------------------------------------
# appended data
# -----------------------------------------------------------------------------
def read_appended_raw(app, payload, hsize, byte_order, rep):
    endian = "<" if byte_order == "LittleEndian" else ">"
    hcode = endian + ("I" if hsize == 4 else "Q")
    blocks = []
    for a in app:
        off = a["offset"]
        if off is None:
            continue
        if off + hsize > len(payload):
            rep.fail("%s: offset %d + %d-byte block header runs past the end "
                     "of the appended section (%d bytes) -- file truncated or "
                     "offset wrong" % (a["who"], off, hsize, len(payload)))
            continue
        nbytes = struct.unpack(hcode, payload[off:off + hsize])[0]
        a["nbytes_header"] = nbytes
        if nbytes != a["nbytes_expect"]:
            rep.fail("%s: block header at offset %d says %d bytes, but "
                     "%d component * %d %s * %d B (%s) = %d bytes"
                     % (a["who"], off, nbytes, a["ncomp"], a["nelem"],
                        "points" if a["group"] == "PointData" else "cells",
                        VTK_TYPES[a["type"]][1], a["type"],
                        a["nbytes_expect"]))
        end = off + hsize + nbytes
        if end > len(payload):
            rep.fail("%s: block at offset %d claims %d bytes of data but only "
                     "%d bytes remain in the appended section (%d short) -- "
                     "the file is truncated"
                     % (a["who"], off, nbytes, len(payload) - off - hsize,
                        end - len(payload)))
            nbytes = max(0, len(payload) - off - hsize)
            end = off + hsize + nbytes
        blocks.append((off, end, a))
        buf = payload[off + hsize:end]
        size = VTK_TYPES[a["type"]][1]
        if len(buf) % size:
            rep.fail("%s: block holds %d bytes, not a whole number of %s "
                     "values (%d B each)"
                     % (a["who"], len(buf), a["type"], size))
        a["nvalues"] = len(buf) // size
        if not NO_VALUES:
            # already reported any ragged tail; hand decode_values a whole
            # number of values so it does not say the same thing twice
            a["values"] = decode_values(buf[:a["nvalues"] * size], a["type"],
                                        byte_order, rep, a["who"])

    # -- the blocks must TILE the section: no gap, no overlap, no leftover ----
    blocks.sort(key=lambda b: b[0])
    cursor = 0
    for off, end, a in blocks:
        if off != cursor:
            d = off - cursor
            rep.fail("%s: offset %d does not land on a block boundary -- the "
                     "previous block ends at %d, so there is a %d-byte %s "
                     "(offsets are counted from the byte AFTER the '_')"
                     % (a["who"], off, cursor, abs(d),
                        "gap" if d > 0 else "overlap"))
        cursor = max(cursor, end)
    leftover = payload[cursor:]
    junk = leftover.strip(b" \t\r\n")
    if junk:
        rep.fail("%d byte(s) of the appended section are not covered by any "
                 "block (%d of them non-whitespace, first bytes %r)"
                 % (len(leftover), len(junk), junk[:16]))
    elif leftover:
        rep.info("%d trailing whitespace byte(s) after the last block (fine)"
                 % len(leftover))
    if blocks:
        rep.ok("%d appended blocks tile [0, %d) exactly%s"
               % (len(blocks), cursor, "" if not leftover else
                  " + %d whitespace" % len(leftover)))


def read_appended_base64(app, payload, hsize, byte_order, rep):
    """base64-encoded appended data: each block is encoded on its own, and VTK
    encodes the block HEADER separately from the block DATA.  Offsets index the
    base64 CHARACTER stream, so exact tiling cannot be asserted the way it can
    for raw; say so rather than implying a check that did not happen."""
    rep.warn("encoding=base64: block tiling is NOT verified (offsets index "
             "encoded characters, whose length depends on padding); lengths "
             "and values are")
    text = payload.decode("ascii", "replace")
    hchars = ((hsize + 2) // 3) * 4
    endian = "<" if byte_order == "LittleEndian" else ">"
    hcode = endian + ("I" if hsize == 4 else "Q")
    for a in app:
        off = a["offset"]
        if off is None or off >= len(text):
            rep.fail("%s: offset %s is outside the %d-character appended "
                     "section" % (a["who"], off, len(text)))
            continue
        try:
            head = b64run(text[off:off + hchars])
            nbytes = struct.unpack(hcode, head[:hsize])[0]
            need = ((a["nbytes_expect"] + 2) // 3) * 4
            body = b64run(text[off + hchars:off + hchars + need])
        except Exception as e:                       # noqa: BLE001
            rep.fail("%s: base64 block at offset %d does not decode (%s)"
                     % (a["who"], off, e))
            continue
        if nbytes != a["nbytes_expect"]:
            rep.fail("%s: base64 block header says %d bytes, expected %d"
                     % (a["who"], nbytes, a["nbytes_expect"]))
        a["values"] = decode_values(body[:a["nbytes_expect"]], a["type"],
                                    byte_order, rep, a["who"])


def b64run(s):
    """Decode one base64 run.  A run of n bytes is exactly ceil(n/3)*4
    characters INCLUDING its own '=' padding, so never add padding to a slice
    that is already a multiple of 4 -- that is what turns a valid file into a
    spurious 'Incorrect padding'."""
    s = "".join(s.split())
    if len(s) % 4:
        s += "=" * (4 - len(s) % 4)
    return base64.b64decode(s)


def read_inline_binary(a, hsize, byte_order, rep):
    """format="binary" means base64 INLINE.  VTK encodes the block header and
    the data as TWO SEPARATE base64 runs (so the text starts with 8 characters
    encoding a 4-byte count); a few writers encode them as one.  Try both, and
    say which matched -- guessing wrong shifts every value by a few bytes and
    still decodes to plausible numbers."""
    text = "".join((a["elem"].text or "").split())
    if not text:
        rep.fail('%s: format="binary" but the element is empty' % a["who"])
        return
    endian = "<" if byte_order == "LittleEndian" else ">"
    hcode = endian + ("I" if hsize == 4 else "Q")
    hchars = ((hsize + 2) // 3) * 4
    want = a["nbytes_expect"]

    # -- VTK's own convention: header run, then data run --------------------
    try:
        head = b64run(text[:hchars])
        n = struct.unpack(hcode, head[:hsize])[0] if len(head) >= hsize else -1
        if n == want:
            body = b64run(text[hchars:hchars + ((want + 2) // 3) * 4])
            a["values"] = decode_values(body[:want], a["type"], byte_order,
                                        rep, a["who"])
            return
    except Exception:                                  # noqa: BLE001
        n = -1

    # -- the other convention: one run holding header and data --------------
    try:
        whole = b64run(text)
        if len(whole) >= hsize:
            m = struct.unpack(hcode, whole[:hsize])[0]
            if m == want:
                rep.warn("%s: inline base64 encodes the block header together "
                         "with the data; VTK encodes them as two runs and "
                         "ParaView may not read this" % a["who"])
                a["values"] = decode_values(whole[hsize:hsize + want],
                                            a["type"], byte_order, rep,
                                            a["who"])
                return
    except Exception:                                  # noqa: BLE001
        pass

    rep.fail("%s: inline base64 block header says %s bytes, expected %d "
             "(tried both the separate-run and single-run conventions)"
             % (a["who"], n if n >= 0 else "?", want))
    try:
        a["values"] = decode_values(b64run(text[hchars:])[:want], a["type"],
                                    byte_order, rep, a["who"])
    except Exception as e:                             # noqa: BLE001
        rep.fail("%s: inline base64 does not decode (%s)" % (a["who"], e))


def read_ascii(a, rep):
    toks = (a["elem"].text or "").split()
    is_float = VTK_TYPES[a["type"]][2]
    vals = []
    for i, t in enumerate(toks):
        try:
            vals.append(float(t) if is_float else int(float(t)))
        except ValueError:
            rep.fail("%s: token %d is %r, not a number" % (a["who"], i, t))
            return
    a["values"] = vals


# -----------------------------------------------------------------------------
# one .pvd
# -----------------------------------------------------------------------------
CHECKED = set()          # absolute paths already validated
REFERENCED = set()       # absolute paths named by some .pvd we have read


def check_pvd(path, quiet=False, limit=0):
    rep = Report(path, quiet)
    refs = []
    try:
        refs = _check_pvd(path, rep, limit)
    except Fatal as e:
        rep.fail(str(e))
    rep.flush()
    good = not rep.fails
    if rep.fails:
        TALLY["fail"] += 1
        TALLY["failed_paths"].append(path)
    else:
        TALLY["pass"] += 1
    for r in refs:
        if r in CHECKED:
            continue
        CHECKED.add(r)
        good = check_vti(r, quiet) and good
    return good


def _check_pvd(path, rep, limit):
    root, payload, _, _ = parse_file(path, rep)
    if payload is not None:
        rep.fail("a .pvd must not carry an <AppendedData> section")
    if local(root.tag) != "VTKFile":
        raise Fatal("root element is <%s>, expected <VTKFile>"
                    % local(root.tag))
    if root.get("type") != "Collection":
        raise Fatal('VTKFile type="%s", expected "Collection" for a .pvd'
                    % root.get("type"))
    cols = [c for c in root if local(c.tag) == "Collection"]
    if len(cols) != 1:
        raise Fatal("expected exactly one <Collection>, found %d" % len(cols))
    base = os.path.dirname(os.path.abspath(path))

    sets = []
    for i, ds in enumerate(cols[0]):
        if local(ds.tag) != "DataSet":
            rep.warn("<%s> inside <Collection> ignored" % local(ds.tag))
            continue
        fn = ds.get("file")
        ts = ds.get("timestep")
        part = ds.get("part", "0")
        if not fn:
            rep.fail("DataSet %d has no file attribute" % i)
            continue
        if ts is None:
            rep.fail('DataSet %d (file="%s") has no timestep attribute; '
                     "ParaView will collapse the series to one frame"
                     % (i, fn))
            t = None
        else:
            try:
                t = float(ts)
            except ValueError:
                rep.fail("DataSet %d: timestep=%r is not a number" % (i, ts))
                t = None
            if t is not None and not math.isfinite(t):
                rep.fail("DataSet %d: timestep=%r is not finite" % (i, ts))
        full = fn if os.path.isabs(fn) else os.path.join(base, fn)
        if not os.path.isfile(full):
            rep.fail('DataSet %d: file="%s" does not exist (looked for %s)'
                     % (i, fn, full))
            continue
        if os.path.getsize(full) == 0:
            rep.fail('DataSet %d: file="%s" exists but is 0 bytes' % (i, fn))
            continue
        if not fn.lower().endswith((".vti", ".vtr", ".vts", ".vtu", ".vtp",
                                    ".pvti")):
            rep.warn('DataSet %d: file="%s" has an unusual extension' % (i, fn))
        sets.append((t, part, fn, full))

    if not sets:
        rep.fail("the collection references no readable dataset")
        return []

    # timesteps strictly increasing, per part (two parts may share a time)
    byname = {}
    for t, part, fn, full in sets:
        byname.setdefault(part, []).append((t, fn))
    for part, seq in sorted(byname.items()):
        prev_t, prev_f = None, None
        for t, fn in seq:
            if t is None or prev_t is None:
                prev_t, prev_f = t, fn
                continue
            if t < prev_t:
                rep.fail('timesteps go BACKWARDS in part %s: %g ("%s") after '
                         '%g ("%s")' % (part, t, fn, prev_t, prev_f))
            elif t == prev_t:
                rep.fail('duplicate timestep %g in part %s ("%s" and "%s"); '
                         "ParaView shows only one of them"
                         % (t, part, prev_f, fn))
            prev_t, prev_f = t, fn
    dups = {}
    for t, part, fn, full in sets:
        dups[full] = dups.get(full, 0) + 1
    for full, k in dups.items():
        if k > 1:
            rep.warn("%s is referenced %d times" % (os.path.basename(full), k))

    times = [t for t, _, _, _ in sets if t is not None]
    rep.ok("Collection of %d dataset(s), %d part(s), t = %g .. %g, all files "
           "present" % (len(sets), len(byname),
                        min(times) if times else float("nan"),
                        max(times) if times else float("nan")))

    refs = [full for _, _, _, full in sets]
    REFERENCED.update(refs)
    if limit and len(refs) > limit:
        idx = sorted({round(i * (len(refs) - 1) / (limit - 1))
                      for i in range(limit)}) if limit > 1 else [0]
        rep.info("--limit %d: checking datasets %s of 0..%d only"
                 % (limit, ", ".join(map(str, idx)), len(refs) - 1))
        refs = [refs[i] for i in idx]
    return refs


# -----------------------------------------------------------------------------
USAGE = """usage: check_vti.py [--quiet] [--no-values] [--limit N] \
FILE.vti | FILE.pvd | DIR ...
  --limit N     of a .pvd collection, check only N datasets, evenly spaced
  --quiet       one line per file unless it fails or warns
  --no-values   framing checks only; do not decode values or report min/max"""


def main(argv):
    global NO_VALUES
    args, quiet, limit = [], False, 0
    it = iter(argv[1:])
    for a in it:
        if a == "--quiet":
            quiet = True
        elif a == "--no-values":
            NO_VALUES = True
        elif a == "--limit" or a.startswith("--limit="):
            v = a.split("=", 1)[1] if "=" in a else next(it, None)
            try:
                limit = int(v)
            except (TypeError, ValueError):
                print("usage: --limit N", file=sys.stderr)
                return 2
        elif a.startswith("--"):
            print("unknown option %s\n%s" % (a, USAGE), file=sys.stderr)
            return 2
        else:
            args.append(a)
    if not args:
        print(USAGE, file=sys.stderr)
        return 2

    targets, from_dir = [], False
    for a in args:
        if os.path.isdir(a):
            from_dir = True
            names = sorted(os.listdir(a))
            pvds = [os.path.join(a, n) for n in names
                    if n.lower().endswith(".pvd")]
            vtis = [os.path.join(a, n) for n in names
                    if n.lower().endswith(".vti")]
            if not pvds and not vtis:
                print("no .pvd or .vti in %s" % a, file=sys.stderr)
                return 2
            targets += pvds + vtis
        elif os.path.isfile(a):
            targets.append(a)
        else:
            print("no such file: %s" % a, file=sys.stderr)
            return 2

    nskip = 0
    for t in targets:
        if t.lower().endswith(".pvd"):
            check_pvd(t, quiet, limit)
        else:
            ap = os.path.abspath(t)
            # in directory mode a .vti named by a .pvd we just read is not
            # checked twice; with --limit that is also how the limit survives
            if ap in CHECKED or (from_dir and ap in REFERENCED):
                nskip += 1
                continue
            CHECKED.add(ap)
            check_vti(t, quiet)
    if nskip:
        print("(%d .vti skipped: named by a .pvd that was checked here)"
              % nskip)
    print("-" * 62)
    n = TALLY["pass"] + TALLY["fail"]
    print("%d file(s) checked: %d passed, %d FAILED"
          % (n, TALLY["pass"], TALLY["fail"]))
    for p_ in TALLY["failed_paths"]:
        print("  FAILED: %s" % p_)
    return 1 if TALLY["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
