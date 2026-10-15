#!/usr/bin/env python3
"""Physical or dimensionless problem -> lattice units, for M3LB.  Stdlib only.

    units.py dimensionless --N 100 --re 1000 [--u0 0.05] [--lattice d3q27]
    units.py dimensionless --N 100 --pe 500  --lattice d3q7        # a scalar
    units.py physical --L 2000 --N 400 --u-phys 6 --nu-phys 1.5e-5 --u-lat-max 0.1
    units.py physical --L 2000 --N 400 --u-phys 6 --nu-phys 1.5e-5 --tau 0.6

It prints the numbers CLAUDE.md says to read BEFORE a run -- omega, tau, Mach,
the two timescales -- and flags the ones that fail silently. It is the recipe
in CLAUDE.md ("From a physical problem to lattice units") as arithmetic and
nothing more: in a driver, still take omega from `Coll::omega_from_viscosity`
(or `omega_from_diffusivity`), which reads its own lattice's cs2.

THE TRAP IT EXISTS FOR: cs2 is 1/3 on D2Q9, D3Q27 and D2Q5 but 1/4 on D3Q7, so
`tau = 3 nu + 1/2` is simply wrong for a D3Q7 scalar. `--lattice` selects it.
"""
import argparse
import math
import sys

CS2 = {"d2q9": 1 / 3, "d3q27": 1 / 3, "d2q5": 1 / 3, "d3q7": 1 / 4}


def warn(msg):
    print("  WARNING: " + msg)


def count(x):
    """Steps, readable at any size: '4,000', '1,000,000', '0.0125'."""
    return f"{x:,.0f}" if x >= 10 else f"{x:.3g}"


def report(nu_lat, cs2, u0, lattice, what, scalar=False):
    """Print omega/tau/Mach for a lattice-unit coefficient and flag the traps."""
    tau = nu_lat / cs2 + 0.5
    omega = 1.0 / tau
    mach = u0 / math.sqrt(cs2)
    print(f"  {what}_lat = {nu_lat:.6g}")
    print(f"  cs2 ({lattice})  = {cs2:.6g}")
    print(f"  tau = {tau:.6f}    tau - 1/2 = {tau - 0.5:.3g}    omega = {omega:.6f}")
    print(f"  u_lat = {u0:.6g}    Ma = u/cs = {mach:.4f}")
    if nu_lat <= 0 or tau <= 0.5:
        warn("tau <= 1/2: unstable. Raise N (cells), not u0.")
    elif tau - 0.5 < 0.012:
        warn("tau - 1/2 < 0.012: close to the stability floor"
             + ("." if scalar else
                ". The regularised fluid wall's measured floor is ~0.006-0.012 "
                "(GPU/csf3/tg_mhd_mach.sub).")
             + " Resolution is the safe knob; halving u0 pushes tau toward 1/2 twice over.")
    if scalar and omega > 1.9:
        warn("omega > 1.9 on a scalar: use ScalarRegularised / EnthalpyRegularised; "
             "BGK ghosts reflect instead of damping as omega -> 2.")
    elif omega > 1.99:
        warn("omega > 1.99: next to the omega = 2 limit.")
    if mach > 0.087 and not scalar:
        warn(f"Ma = {mach:.3f} > 0.087 (u0 > 0.05): compressibility error grows as "
             "Ma^2. Keep u0 <= 0.05 unless you have measured what a larger one costs.")
    return tau


def dimensionless(a):
    cs2 = CS2[a.lattice]
    if (a.re is None) == (a.pe is None):
        sys.exit("give exactly one of --re (momentum) or --pe (a scalar)")
    if a.re is not None:
        nu = a.u0 * a.N / a.re
        what, dimless = "nu", f"Re = {a.re:g}"
    else:
        nu = a.u0 * a.N / a.pe
        what, dimless = "D", f"Pe = {a.pe:g}"
    print(f"dimensionless: N = {a.N} cells across the Reynolds length, "
          f"u0 = {a.u0:g}, {dimless}, {a.lattice}")
    print("  (wall-bounded cases: the length is H, and the domain is ny = H + 2)")
    report(nu, cs2, a.u0, a.lattice, what, scalar=a.pe is not None)
    print(f"  one convective time  N/u0   = {count(a.N / a.u0)} steps")
    print(f"  one diffusive time   N^2/nu = {count(a.N ** 2 / nu)} steps   "
          f"(= Re x longer; quote a result after the right one)")


def physical(a):
    cs2 = CS2[a.lattice]
    if (a.u_lat_max is None) == (a.tau is None):
        sys.exit("give exactly one of --u-lat-max (cap velocity) or --tau (fix tau)")
    dx = a.L / a.N
    if a.u_lat_max is not None:
        dt = a.u_lat_max * dx / a.u_phys
        strategy = f"cap velocity: u_lat_max = {a.u_lat_max:g}"
    else:
        dt = (a.tau - 0.5) * cs2 * dx * dx / a.nu_phys
        strategy = f"fix tau = {a.tau:g}"
    u_lat = a.u_phys * dt / dx
    nu_lat = a.nu_phys * dt / (dx * dx)
    print(f"physical: L = {a.L:g} m over N = {a.N} cells, {a.lattice}, {strategy}")
    print(f"  dx = {dx:.6g} m    dt = {dt:.6g} s")
    report(nu_lat, cs2, u_lat, a.lattice, "nu", scalar=a.scalar)
    if a.tau is not None and u_lat / math.sqrt(cs2) > 0.087:
        print("  NOTE: fixing tau sets dt by DIFFUSION. This flow is advection-dominated, "
              "so use --u-lat-max instead (CLAUDE.md: cap the velocity when advection dominates).")
    if nu_lat / cs2 < 1e-4:
        print("  NOTE: at this Reynolds number the viscous term is not resolvable on "
              "this grid. demonstrator/urban prescribes the wind and sets an effective "
              "diffusivity (-diff) rather than solving it.")
    if a.accel is not None:
        print(f"  a_lat = a dt^2/dx = {a.accel * dt * dt / dx:.6g}   "
              f"(from a = {a.accel:g} m/s^2)")
    if a.t_end is not None:
        print(f"  {a.t_end:g} s = {count(a.t_end / dt)} steps")
    print(f"  one convective time L/u = {a.L / a.u_phys:.6g} s = "
          f"{count(a.L / a.u_phys / dt)} steps")
    print("  A coefficient held fixed in lattice units while dt moves silently "
          "rescales the physics; derive each from its SI value (CLAUDE.md).")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dimensionless", help="specified by Re or Pe and a resolution")
    d.add_argument("--N", type=int, required=True)
    d.add_argument("--re", type=float)
    d.add_argument("--pe", type=float)
    d.add_argument("--u0", type=float, default=0.05)
    d.add_argument("--lattice", choices=CS2, default="d3q27")
    d.set_defaults(fn=dimensionless)
    q = sub.add_parser("physical", help="the geometry carries a real length")
    q.add_argument("--L", type=float, required=True, help="characteristic length, m")
    q.add_argument("--N", type=int, required=True, help="cells across L")
    q.add_argument("--u-phys", type=float, required=True, help="fastest speed, m/s")
    q.add_argument("--nu-phys", type=float, required=True, help="kinematic viscosity "
                   "(or diffusivity), m^2/s")
    q.add_argument("--u-lat-max", type=float, help="cap the fastest cell at this")
    q.add_argument("--tau", type=float, help="or fix tau")
    q.add_argument("--accel", type=float, help="optional acceleration, m/s^2")
    q.add_argument("--t-end", type=float, help="optional physical duration, s")
    q.add_argument("--scalar", action="store_true",
                   help="the coefficient is a scalar diffusivity (adds the omega > 1.9 note)")
    q.add_argument("--lattice", choices=CS2, default="d3q27")
    q.set_defaults(fn=physical)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
