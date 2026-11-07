#!/usr/bin/env python3
"""The electric field along the axis below the melt, from one demonstrator/mew_axi
frame, for demonstrator/mew_jet -Efile.

    python3 tools/mew_axi_axis_field.py results/mew_axi/R4_heat_best_t059.95 out.csv [--V 6.5] [--dx 0.05]

Reads <prefix>_fill.npy and <prefix>_phi.npy (rows from the collector up, row 0
the collector, row j spanning [(j-1) dx, j dx]; column 0 on the axis; phi in
units of the applied voltage, 1 in the melt). The melt's lowest point on the axis
is the bottom face of its lowest row with fill >= 1/2 -- where FsElectricAxi's
staircase conductor ends -- and a jet leaving the melt starts there. Writes
zeta_mm (distance below that point), E_kV_per_mm: central differences at the gas
cells on the axis, and at the two ends the derivative of the quadratic through
the end's potential (1 on the melt's face, 0 on the collector's) and the two
nearest cell centres. Prints the apex height, which mew_jet takes as the jet's
length.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--V", type=float, default=6.5, help="applied voltage [kV]")
    ap.add_argument("--dx", type=float, default=0.05, help="cell size [mm]")
    a = ap.parse_args()
    fill = np.load(a.prefix + "_fill.npy")
    phi = np.load(a.prefix + "_phi.npy").astype(float)
    col_f, col_p = fill[:, 0], phi[:, 0]
    rows = np.nonzero(col_f >= 0.5)[0]
    if len(rows) == 0:
        sys.exit("no melt on the axis")
    jl = int(rows.min())                      # the lowest conducting row on the axis
    h_apex = (jl - 1) * a.dx                  # its bottom face
    centres = np.arange(1, jl) * a.dx - 0.5 * a.dx   # gas rows 1 .. jl-1
    p = col_p[1:jl]
    if np.any(col_f[1:jl] > 0.0):
        print("warning: melt below the apex on the axis (rows %s)" % np.nonzero(col_f[1:jl] > 0)[0])
    # the downward field is +dphi/dh times the voltage (phi rises towards the melt)
    def quad_slope(h, hs, ps):                # dphi/dh [1/mm] of the quadratic through (hs, ps)
        c = np.polyfit(hs, ps, 2)
        return 2 * c[0] * h + c[1]
    E = np.empty_like(p)
    E[1:-1] = (p[2:] - p[:-2]) / (2 * a.dx) * a.V
    hs = np.array([h_apex, centres[-1], centres[-2]])      # phi = 1 on the melt's face
    ps = np.array([1.0, p[-1], p[-2]])
    E[-1] = quad_slope(centres[-1], hs, ps) * a.V
    E_apex = quad_slope(h_apex, hs, ps) * a.V
    hs = np.array([0.0, centres[0], centres[1]])           # phi = 0 on the collector's
    ps = np.array([0.0, p[0], p[1]])
    E[0] = quad_slope(centres[0], hs, ps) * a.V
    E_coll = quad_slope(0.0, hs, ps) * a.V
    zeta = np.concatenate([[0.0], h_apex - centres[::-1], [h_apex]])
    Ez = np.concatenate([[E_apex], E[::-1], [E_coll]])
    with open(a.out, "w") as f:
        f.write("zeta_mm,E_kV_per_mm\n")
        for z, e in zip(zeta, Ez):
            f.write("%.6f,%.6f\n" % (z, e))
    print("apex %.4f mm above the collector; E at the apex %.3f, at 0.5 mm %.3f, at 1 mm %.3f, at the collector %.3f kV/mm;"
          " mean below the apex %.3f (V/h %.3f)" % (h_apex, E_apex, np.interp(0.5, zeta, Ez), np.interp(1.0, zeta, Ez),
                                                  E_coll, float(np.sum(0.5 * (Ez[1:] + Ez[:-1]) * np.diff(zeta))) / h_apex, a.V / h_apex))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
