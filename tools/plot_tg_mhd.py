#!/usr/bin/env python3
"""Plots for demonstrator/tg_mhd.cpp and GPU/src/tg_mhd.cu.

    plot_tg_mhd.py RUN_DIR [RUN_DIR ...] [-o out.png] [--title TEXT]

Each RUN_DIR holds a series.dat (the drivers write one per run). Every run is
overlaid on the same six panels, labelled by its tag, so the free-slip /
no-slip / hydrodynamic trio of the confined-MHD plan reads as one figure:

  1. kinetic and magnetic energy, log scale
  2. total dissipation eps = 2 nu Omega_V + 2 eta Omega_M
  3. E_M/E_V and Omega_M/Omega_V, with the reference values of Pouquet et al.
     (2010) Table 1, run C2, drawn where they apply (min E_M/E_V = 0.35 and the
     first maximum of Omega_M/Omega_V = 2, both at nu = eta = 1e-3)
  4. the maxima of |j| and |omega|
  5. the share of dissipation within 1, 2 and 4 x Re^-1/2 of the walls
  6. health: div b / |j| (global and in the wall layer), the mass drift and the
     peak lattice speed, which is the Mach number's numerator

needs numpy and matplotlib. The system python on the development Mac has
neither; a scratch venv does (python3 -m venv v && v/bin/pip install numpy
matplotlib), and nothing in the tree depends on that venv.

load_raw(path) reads a raw/fields_NNNN.raw volume -- "TGMHDRAW", int32 nx ny nz
nvar, float64 t h, then nvar float32 blocks (ux uy uz bx by bz rho, x fastest)
-- and returns (t, h, dict of arrays shaped (nz, ny, nx)).
"""
import argparse
import os
import struct
import sys

COLS = ("t E_V E_M E_T H_C Omega_V Omega_M eps EM/EV OmM/OmV j_max w_max skew "
        "divb/j divb_wall/j mass_drift umax_lat fw1 fw2 fw4 tau_wall").split()


def load_series(run_dir):
    path = os.path.join(run_dir, "series.dat")
    head, rows = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                head.append(line[1:].strip())
                continue
            parts = line.split()
            if len(parts) == len(COLS):
                rows.append([float(v) for v in parts])
    if not rows:
        raise SystemExit("%s: no data rows" % path)
    import numpy as np
    a = np.array(rows)
    data = {c: a[:, i] for i, c in enumerate(COLS)}
    tag = os.path.basename(os.path.normpath(run_dir))
    if head:
        toks = head[0].split()
        for t in toks:
            if t.startswith(("tgc_", "tgi_", "tga_", "hydro_")):
                tag = t
                break
    return tag, data, head


def load_raw(path):
    """Read one raw/fields_NNNN.raw file. Returns (t, h, fields)."""
    import numpy as np
    with open(path, "rb") as f:
        b = f.read()
    if b[:8] != b"TGMHDRAW":
        raise ValueError("%s: not a tg_mhd raw field file" % path)
    nx, ny, nz, nv = struct.unpack("4i", b[8:24])
    t, h = struct.unpack("2d", b[24:40])
    a = np.frombuffer(b[40:], dtype=np.float32).reshape(nv, nz, ny, nx)
    names = ("ux", "uy", "uz", "bx", "by", "bz", "rho")[:nv]
    return t, h, {n: a[i] for i, n in enumerate(names)}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories holding series.dat")
    ap.add_argument("-o", "--out", default=None, help="output PNG (default: first run's dir)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [load_series(r) for r in args.runs]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    ax = ax.ravel()
    styles = ["-", "--", "-.", ":"]

    for k, (tag, d, _) in enumerate(runs):
        ls = styles[k % len(styles)]
        t = d["t"]
        ax[0].semilogy(t, d["E_V"], ls, color="C0", label="E_V  " + tag)
        if d["E_M"].max() > 0:
            ax[0].semilogy(t, d["E_M"], ls, color="C3", label="E_M  " + tag)
        ax[1].plot(t, d["eps"], ls, color="C%d" % k, label=tag)
        if d["E_M"].max() > 0:
            ax[2].plot(t, d["EM/EV"], ls, color="C0", label="E_M/E_V  " + tag)
            ax[2].plot(t, d["OmM/OmV"], ls, color="C3", label="Omega_M/Omega_V  " + tag)
            ax[3].plot(t, d["j_max"], ls, color="C3", label="|j|max  " + tag)
        ax[3].plot(t, d["w_max"], ls, color="C0", label="|w|max  " + tag)
        if "slip" in tag or "noslip" in tag:
            for m, c in zip(("fw1", "fw2", "fw4"), ("C2", "C1", "C4")):
                ax[4].plot(t, d[m], ls, color=c, label="%s  %s" % (m, tag))
        ax[5].semilogy(t[1:], d["divb/j"][1:], ls, color="C0", label="div b/|j|  " + tag)
        ax[5].semilogy(t[1:], d["divb_wall/j"][1:], ls, color="C9",
                       label="div b/|j| wall  " + tag)
        ax[5].semilogy(t[1:], abs(d["mass_drift"][1:]) + 1e-16, ls, color="C7",
                       label="|mass drift|  " + tag)

    if any("re1000" in r[0] and r[0].startswith("tgc_") for r in runs):
        ax[2].axhline(0.35, color="C0", lw=0.8, alpha=0.6)
        ax[2].axhline(2.0, color="C3", lw=0.8, alpha=0.6)
        ax[2].text(0.02, 0.37, "C2: min E_M/E_V = 0.35", color="C0",
                   transform=ax[2].get_yaxis_transform(), fontsize=8)
        ax[2].text(0.02, 2.03, "C2: first max = 2", color="C3",
                   transform=ax[2].get_yaxis_transform(), fontsize=8)

    titles = ["energies", "dissipation  eps = 2 nu Om_V + 2 eta Om_M",
              "magnetic / kinetic ratios", "maxima of |j| and |omega|",
              "share of dissipation within 1, 2, 4 Re^-1/2 of the walls",
              "health: div b, mass"]
    for a, tt in zip(ax, titles):
        a.set_title(tt, fontsize=10)
        a.set_xlabel("t  (Pouquet et al. units)")
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    if args.title:
        fig.suptitle(args.title)

    out = args.out or os.path.join(args.runs[0], "series.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
