#!/usr/bin/env python3
"""The heater-cycle runs of 2026-11-17 (mew_axi -Ea 40.9 [-cycle 110 128 13 53 [-breathe 0.0293 [-phase 13]]])
as a table: the numbers demonstrator/mew_axi.cpp's banner quotes for them.

    python3 results/mew_axi/cycle_numbers.py > results/mew_axi/cycle_numbers.txt

Standard library only. Columns are the driver's own: z_low_mm the melt's lowest point above the
collector, tip_zone_mm3 the melt within 1 mm of the tip face, gap_mm3 all the melt below it,
width_mm its widest outline, T_end_C its lowest axis cell, T_gap_mean_C its mean temperature.
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = (("R4_heat_best", "steady 120 C, 31.3 kJ/mol (the earlier run, to 60 s)"),
        ("R4_heat_Ea41", "steady 120 C, 40.9 kJ/mol"),
        ("R4_heat_Ea41_cycT", "cycle on the temperature only"),
        ("R4_heat_Ea41_cyc", "cycle with the melt's expansion, from switch-on"),
        ("R4_heat_Ea41_cyc_ph13", "cycle with the melt's expansion, from switch-off"))
TIMES = (10, 20, 40, 60, 80, 100, 120, 150)
COLS = (("z_low_mm", "%6.3f"), ("tip_zone_mm3", "%6.3f"), ("gap_mm3", "%6.3f"), ("width_mm", "%6.3f"),
        ("T_end_C", "%6.1f"), ("T_gap_mean_C", "%6.1f"))


def load(name):
    p = os.path.join(HERE, name + "_series.csv")
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(p))] if os.path.exists(p) else []


def at(rows, t):
    r = min(rows, key=lambda q: abs(q["t_s"] - t))
    return r if abs(r["t_s"] - t) < 0.5 else None


def contact(name):
    try:
        m = re.search(r"first contact with the collector at t = ([0-9.]+)", open(os.path.join(HERE, name + ".log")).read())
        return float(m.group(1)) if m else None
    except OSError:
        return None


data = {n: load(n) for n, _ in RUNS}
for n, lab in RUNS:
    rows = data[n]
    if not rows:
        print("%s: no series" % n)
        continue
    tc = contact(n)
    print("%s -- %s; series to %.1f s; %s" % (n, lab, rows[-1]["t_s"],
          "first contact at %.2f s" % tc if tc else "no contact"))
    print("    t [s]   " + "  ".join("%12s" % c for c, _ in COLS))
    for t in TIMES:
        r = at(rows, t)
        if r:
            print("    %5.0f   " % t + "  ".join("%12s" % (f % r[c]) for c, f in COLS))
    print("    largest |ledger| after 1 s: %.1e" % max(abs(r["ledger"]) for r in rows if r["t_s"] > 1))
    if "fed_mm3" in rows[-1]:
        print("    fed by the end: %.4f mm^3 (a steady feed: %.4f)" % (rows[-1]["fed_mm3"], 0.0208 * rows[-1]["t_s"]))
    print()

# What the cycle does to the drop, against the steady run at the same time.
base = data["R4_heat_Ea41"]
if base:
    print("against the steady 40.9 kJ/mol run at the same time (z_low difference in mm; tip-zone ratio):")
    for n, lab in RUNS[2:]:
        rows = data[n]
        if not rows:
            continue
        out = []
        for t in TIMES:
            r, b = at(rows, t), at(base, t)
            if r and b:
                out.append("%3.0f s: %+.3f, x%.2f" % (t, r["z_low_mm"] - b["z_low_mm"], r["tip_zone_mm3"] / b["tip_zone_mm3"]))
        print("  %-52s %s" % (lab, "; ".join(out)))
    # the swing of the tip's melt within each heater cycle, breathing runs
    print()
    for n, lab in RUNS[3:]:
        rows = data[n]
        if not rows or "out_of_bore_mm3_s" not in rows[0]:
            continue
        print("%s: the melt below the tip face (gap_mm3) over each leg of the cycle" % n)
        legs, start = [], rows[0]
        for a, b in zip(rows, rows[1:]):
            if abs(a["out_of_bore_mm3_s"] - b["out_of_bore_mm3_s"]) > 1e-4:
                legs.append((start, a))
                start = b
        legs.append((start, rows[-1]))
        for s, e in legs:
            if e["t_s"] - s["t_s"] < 1:
                continue
            print("    %6.1f - %6.1f s  %s  gap %.3f -> %.3f mm^3 (%+.3f), z_low %.3f -> %.3f mm, T_end %.1f -> %.1f C"
                  % (s["t_s"], e["t_s"], "heating" if s["out_of_bore_mm3_s"] > 0.0208 else "cooling",
                     s["gap_mm3"], e["gap_mm3"], e["gap_mm3"] - s["gap_mm3"], s["z_low_mm"], e["z_low_mm"],
                     s["T_end_C"], e["T_end_C"]))


# ---------------------------------------------------------------- with the jet
# The same three feeds with the no-fan fibres' flow taken at the apex (-jet 0.011552).
JET = (("R4_heat_Ea41_jet116", "steady feed"),
       ("R4_heat_Ea41_cyc_jet116", "cycle with the melt's expansion, from switch-on"),
       ("R4_heat_Ea41_cyc_ph13_jet116", "cycle with the melt's expansion, from switch-off"))
JET_Q, FEED_Q = 0.011552, 0.0208


def created(rows, r):
    """What the free surface itself has added: melt now + what left - melt at the start - what was fed."""
    fed = lambda q: q.get("fed_mm3", FEED_Q * q["t_s"])
    m0 = rows[0]["melt_total_mm3"] + rows[0]["drained_mm3"] - fed(rows[0])
    return r["melt_total_mm3"] + r["drained_mm3"] - m0 - fed(r)


jet = {n: load(n) for n, _ in JET}
if all(jet.values()):
    print()
    print("WITH THE NO-FAN JET (%.6f mm^3/s taken at the apex, %.0f %% of the feed)" % (JET_Q, 100 * JET_Q / FEED_Q))
    for n, lab in JET:
        rows = jet[n]
        tc = contact(n)
        print("%s -- %s; series to %.1f s; %s" % (n, lab, rows[-1]["t_s"],
              "first contact at %.2f s" % tc if tc else "no contact"))
        print("    t [s]   " + "  ".join("%12s" % c for c, _ in COLS) + "       jet_mm3   created_mm3")
        for t in TIMES:
            r = at(rows, t)
            if r:
                print("    %5.0f   " % t + "  ".join("%12s" % (f % r[c]) for c, f in COLS)
                      + "  %12.4f  %12.4f" % (r["jet_mm3"], created(rows, r)))
        e = rows[-1]
        print("    by the end: jet took %.4f of %.4f asked; the free surface created %.4f mm^3 = %.1f %% of what the"
              " jet took, %.1f %% of the melt below the tip" % (e["jet_mm3"], e["jet_asked_mm3"], created(rows, e),
              100 * created(rows, e) / e["jet_mm3"], 100 * created(rows, e) / e["gap_mm3"]))
        print()
    for n, lab in JET[1:]:
        rows = jet[n]
        print("%s: the melt below the tip face over each leg of the cycle" % n)
        legs, start = [], rows[0]
        for a, b in zip(rows, rows[1:]):
            if abs(a["out_of_bore_mm3_s"] - b["out_of_bore_mm3_s"]) > 1e-4:
                legs.append((start, a))
                start = b
        legs.append((start, rows[-1]))
        for s, e in legs:
            if e["t_s"] - s["t_s"] < 1:
                continue
            q = s["out_of_bore_mm3_s"]
            dt_ = e["t_s"] - s["t_s"]
            print("    %6.1f - %6.1f s  %s  gap %.3f -> %.3f mm^3 (%+.3f; supply less jet %+.3f, created %+.3f),"
                  " z_low %.3f -> %.3f mm"
                  % (s["t_s"], e["t_s"], "heating" if q > FEED_Q else "cooling", s["gap_mm3"], e["gap_mm3"],
                     e["gap_mm3"] - s["gap_mm3"], (q - JET_Q) * dt_, created(rows, e) - created(rows, s),
                     s["z_low_mm"], e["z_low_mm"]))
    print()
    print("the fibres' flow against what the orifice delivers while the heater is off:")
    print("    fibres %.4f mm^3/s; cooling leg %.4f with -breathe 0.0293 (0.0132 to 0.0097 over the expansion"
          " coefficient's range, results/mew_hotend)" % (JET_Q, FEED_Q - 0.0293 * 18 / 53))
    for q, lab in ((FEED_Q - 0.0293 * 18 / 53, "the cooling leg's flow"), (JET_Q, "the fibres' own")):
        import math
        print("    a fibre laid at 35 mm/s carrying %s: %.1f um across" % (lab, 1e3 * math.sqrt(4 * q / (math.pi * 35.0))))
