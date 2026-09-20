#!/usr/bin/env python3
"""Choose the three transfer-function levels for vol_urban from the data.

make_urban_anim.sh says the right levels "depend on the run" and then hands you
a default. That default was tuned for one Manchester run, and it is wrong for
any plume of a different strength: the three bands are ABSOLUTE iso-shells --
near-white at l0, salmon at l1, red at l2 -- so a plume whose body sits below l1
renders as a uniform white blob with no shells in it at all, which is exactly
what a first render of a longer-fetch run looks like.

So measure instead of guessing. The levels are taken as percentiles of the
concentration over the cells that actually CARRY plume, which is the population
the shells have to separate:

  l0  the 50th percentile -- the outer haze, half the plume brighter than it
  l1  the 90th            -- the salmon body
  l2  the 99th            -- the red core

WHICH CELLS COUNT IS THE WHOLE QUESTION. Over every cell in the domain the
median concentration is zero, because a 2 km box is mostly empty air, and every
percentile collapses onto the noise floor. So cells below a floor are excluded,
and the floor is a fraction of the frame's own peak (1e-4 by default, i.e. four
decades down) rather than an absolute number -- an absolute floor is the same
mistake one step further back. Solid cells carry a -1 sentinel and are dropped.

Percentiles, not a linear split of the range: concentration in a plume is
log-distributed over several decades, so the midpoint of [min, max] is far out
in the tail and the band that is supposed to show the body shows nothing.

  usage: plume_levels.py <conc.vtk> [--nx 400] [--ny 400] [--nz 60]
                         [--floor 1e-4] [--pct 50,90,99]
"""
import argparse, array, sys


def read_vtk(path):
    d = open(path, "rb").read()
    key = b"LOOKUP_TABLE default\n"
    i = d.index(key) + len(key)
    a = array.array("f")
    a.frombytes(d[i:])
    a.byteswap()                      # legacy VTK binary is big-endian
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("vtk")
    p.add_argument("--floor", type=float, default=1e-4,
                   help="fraction of the frame peak below which a cell is not "
                        "plume (default 1e-4)")
    p.add_argument("--pct", default="50,90,99")
    p.add_argument("--quiet", action="store_true",
                   help="print only the -levels argument")
    a = p.parse_args()

    v = read_vtk(a.vtk)
    peak = max(v)
    cut = peak * a.floor
    plume = sorted(x for x in v if x > cut)
    if len(plume) < 100:
        sys.exit(f"{a.vtk}: only {len(plume)} cells above {cut:.2e} -- "
                 f"is this frame empty?")
    pcts = [float(s) for s in a.pct.split(",")]
    lv = [plume[min(len(plume) - 1, int(q / 100.0 * len(plume)))] for q in pcts]
    # Bands must be strictly increasing or two shells coincide and the middle one
    # never renders. A plume can be flat enough for that at low percentiles.
    for i in range(1, len(lv)):
        if lv[i] <= lv[i - 1]:
            lv[i] = lv[i - 1] * 2.0
    out = ",".join(f"{x:.2g}" for x in lv)
    if a.quiet:
        print(out)
        return
    solid = sum(1 for x in v if x < 0)
    print(f"  {a.vtk}")
    print(f"    peak {peak:.4e}, floor {cut:.2e} ({a.floor:g} of peak), "
          f"{len(plume):,} plume cells, {solid:,} solid")
    for q, x in zip(pcts, lv):
        n = sum(1 for y in plume if y >= x)
        print(f"    p{q:<5g} {x:.4e}   {n:9,} cells at or above "
              f"({100*n/len(plume):5.1f}% of the plume)")
    print(f"\n  -levels {out}")


if __name__ == "__main__":
    main()
