//==============================================================================
//  Where a slowly fed PCL filament melts inside a melt-electrowriting (MEW) hot
//  end, and how far the melt reaches above the heater.
//
//  It models the one mechanism Ju, Omar, Mirihanage & Kilic, "Suppressing
//  oozing during filament-based melt electrowriting" (SSRN 7582986, preprint,
//  not peer reviewed) argue for: heat conducted up the filament melts it above
//  the intended zone, and a fan above the heater block (their Strategy 2)
//  confines the melt. The paper is experimental and qualitative, so nothing
//  here is compared against it; this is a demonstrator, not a validation case.
//
//  THE MODEL. A slab of width W = the filament diameter, fed at the filament
//  speed U = Q / (pi D^2 / 4) as PLUG FLOW, inside a bore whose wall
//  temperature is PRESCRIBED: T_h over the heater, a linear ramp to T_c across
//  the heat break, T_c above it. The fan IS the parameter T_c. The scalar is the
//  total enthalpy with the latent heat ADVECTED by the moving solid --
//  EnthalpyAdvect::Material, checked against an exact solution in
//  validation/fed_rod.cpp; the default Sensible would delete the latent heat
//  from this problem. D3Q7, nz = 1, on-node walls exactly W apart, an on-node
//  inlet at T_in, a zero-gradient outflow at the nozzle exit. All material and
//  hot-end numbers are NOMINAL (struct Physical): typical PCL values, an
//  E3D-v6-like geometry, and Table 1's 120 C and feed rates.
//
//  MEASURED 2026-11-03, FP64, N = 32 (dx = 54.7 um), tau = 0.7, every row
//  steady (field change < 1e-9 per quarter flush). Lengths in mm above the
//  nozzle exit; l_melt is the melt AREA / W, l_above its part above the heater;
//  z_eq is where the WALL reaches the middle of the band (60 C).
//
//      feed  Pe_D   T_c   l_melt  l_above  z_axis  z_W/8  z_eq   t90 [s]
//      MEW   2.04   30    11.500   1.519   11.202  11.691 12.000    4.7
//      MEW          45    11.901   1.921   11.581  12.098 12.400    5.3
//      MEW          55    12.317   2.336   11.973  12.503 12.769    5.3
//      MEW          65    22.709  12.729   21.714  23.662 25.000   21.2
//      FDM  32.73   30     8.028   0.320    3.577  10.369 12.000    3.9
//      FDM          45     8.713   0.475    4.464  10.842 12.400    4.3
//      FDM          55     9.279   0.657    5.156  11.292 12.769    4.6
//      FDM          65    12.203   3.053    6.233  13.703 25.000    5.2
//
//  WHAT THE TABLE SAYS, AND ONLY THAT.
//  - At the MEW feed the filament is close to equilibrium with the bore: the
//    axis front sits 0.80 mm below z_eq at both T_c = 30 and 55. At the FDM
//    feed (16x faster) the axis front sits 8.4 mm below z_eq at T_c = 30.
//    So at MEW feed the melt volume is set by the hot end's WALL temperature
//    map, which is what a fan changes, and not by the heater.
//  - MEW puts 4.7x more melt above the heater than FDM (1.519 vs 0.320 mm at
//    T_c = 30), in the same hot end.
//  - Below the band, cooling the cold end from 55 to 30 C shortens the MEW melt
//    by 0.82 mm (6.6 %). Above the liquidus (65 C) the melt fills the cold zone
//    and l_melt grows 1.84x. In this model the fan is a THRESHOLD
//    control: decisive if the unfanned cold end exceeds PCL's melting range,
//    marginal if not. Which side the paper's machine is on is a temperature
//    nobody has measured, and the model cannot supply it.
//  - From fresh filament the melt reaches 90 % of its steady size in about
//    5 s at MEW feed below the band (21 s at T_c = 65), sampled every 0.59 s.
//    A melt zone that refills in seconds cannot by itself be a build-up over
//    minutes; if oozing develops over a print, the slow clock is elsewhere --
//    most plausibly the cold end warming, which this model holds fixed.
//
//  GRID. N = 16 / 32 / 64 at fixed tau (U_lat ~ 1/N): MEW, T_c = 55, l_melt
//  12.316 / 12.317 / 12.316, z_axis 11.983 / 11.973 / 11.973, z_W/8 12.505 /
//  12.503 / 12.502; FDM, T_c = 30, l_melt 8.043 / 8.028 / 8.020, z_axis 3.605 /
//  3.577 / 3.577. l_above is a cut at z = L_h and moves by +-0.6 % (2.308 /
//  2.336 / 2.322). The first version read the near-wall front one NODE from
//  the wall, a different place at every N, and it drifted under refinement by
//  construction; it is now at a fixed W/8. FP32 at N = 32, MEW, T_c = 30:
//  l_melt 11.499 against 11.500.
//
//  WHAT IT DOES NOT MODEL -- the reliability limits, stated per mechanism.
//  - A SLAB, NOT A CYLINDER. The slowest conduction mode is 9.87 alpha/D^2
//    across a slab and 23.1 alpha/D^2 across a cylinder of the same diameter,
//    so radial heating is ~2.3x too slow here: the MEW conclusion is
//    conservative and the FDM lag overstated. An axisymmetric scalar does not
//    exist in this tree.
//  - NO MELT FLOW. Plug flow is exact for the solid and wrong for melt sheared
//    at the wall, and the filament-bore gap (where melt can creep back up) is
//    absent. A coupled run needs a Darcy sink dragging the solid to the FEED
//    velocity; DarcyGuo drags it to rest. At Re ~ 1e-9..1e-8 and Pr ~ 1e6..1e7
//    (PCL at 120 C taken as 100..1000 Pa s, in the 0.4 mm nozzle) the flow is
//    quasi-steady Stokes, so a coupled LBM run would match Pe and keep Re << 1
//    rather than match Pr.
//  - NO RHEOLOGY. PCL is shear-thinning and viscoelastic, and die swell is one
//    of the oozing mechanisms the paper cites; neither tree has a
//    non-Newtonian model.
//  - NO RESIDUAL PRESSURE. The melt's physical compressibility -- the stored
//    energy that keeps a filament-fed head extruding after the feed stops --
//    is not the LBM's: the lattice sound speed is a numerical parameter, and
//    the physical Mach number here is 1e-7..1e-6. That mechanism needs a lumped
//    compliance-resistance model, not this solver.
//  - NO TAYLOR CONE. A cone and jet need the Maxwell stress on a free surface;
//    the EHD stack in this tree is charge injection into a bulk liquid.
//  - THE HARDWARE IS A BOUNDARY CONDITION. The heat break, the heat sink and
//    the fan are a prescribed wall profile; T_c and the ramp length L_b stand
//    in for all of it.
//==============================================================================
#include "collision/EnthalpyBGK.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ScalarSolver.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

namespace {

constexpr double kPi = 3.14159265358979323846;

//------------------------------------------------------------------------------
// The physical problem. Every number here is NOMINAL: the material values are
// typical of a semi-crystalline PCL (Mw ~ 50-80k, ~55 % crystalline) and the
// hot-end lengths are E3D-v6-like, not measured on the paper's machine.
//------------------------------------------------------------------------------
struct Physical {
  double D      = 1.75e-3;   // filament diameter = slab width W [m]
  double Q      = 0.255e-9;  // feed [m^3/s] -- Table 1, MEW
  double rho    = 1100.0;    // [kg/m^3], one value for both phases
  double cp     = 2000.0;    // [J/(kg K)], one value for both phases
  double k      = 0.20;      // [W/(m K)]
  double Lf     = 75.0e3;    // latent heat [J/kg]
  double Ts     = 57.0;      // solidus [C]
  double Tl     = 63.0;      // liquidus [C]
  double Th     = 120.0;     // heater = nozzle temperature [C] -- Table 1
  double Tc     = 30.0;      // cold-end wall [C] -- THE fan parameter
  double Tin    = 25.0;      // filament entering the hot end [C]
  double Lh     = 10.0e-3;   // heated length above the nozzle exit [m]
  double Lb     = 3.0e-3;    // heat break: wall ramps Th -> Tc over this [m]
  double Lc     = 12.0e-3;   // cold zone above the break [m]

  double U() const { return Q / (0.25 * kPi * D * D); }
  double alpha() const { return k / (rho * cp); }
  double Ltot() const { return Lh + Lb + Lc; }
  // Wall temperature at height z above the nozzle exit.
  double Twall(double z) const {
    if (z <= Lh) return Th;
    if (z <= Lh + Lb) return Th + (Tc - Th) * (z - Lh) / Lb;
    return Tc;
  }
};

struct Opts {
  int    N     = 32;      // cells across the filament
  double tau   = 0.7;
  bool   sweep = false;
  bool   ladder = false;
  std::string out;        // prefix for CSV output; empty = none
};

struct Outcome {
  double l_melt  = 0;   // melt area / W  [mm]  -- an equivalent melt length
  double l_above = 0;   // the part of it above the heater [mm]
  double z_axis  = 0;   // f_l = 1/2 on the axis, height above the exit [mm]
  double z_wall  = 0;   // f_l = 1/2 at W/8 from the wall [mm]
  double t_phys  = 0;   // simulated time [s]
  double t90     = 0;   // time for l_melt to reach 90 % of its final value [s]
  double change  = 0;   // last interval's max |dH| / range
  long   steps   = 0;
  bool   steady  = false;
};

//------------------------------------------------------------------------------
// One run. 2-D = a 3-D lattice with nz = 1 (CLAUDE.md). x across the slab with
// ON-NODE walls at x = 0 and x = N, so the wall is exactly W apart; y along the
// feed, inlet at y = 0, nozzle exit at y = ny - 1.
//------------------------------------------------------------------------------
Outcome run(const Physical& P, const Opts& o, bool verbose, const std::string& tag) {
  using Coll = EnthalpyBGK<D3Q7, EnthalpyAdvect::Material>;
  const int    N   = o.N;
  const double dx  = P.D / N;
  const int    ny  = int(std::lround(P.Ltot() / dx)) + 1;
  const double a_l = Coll::diffusivity_from_omega(Real(1.0 / o.tau));
  const double dt  = a_l * dx * dx / P.alpha();
  const double U_l = P.U() * dt / dx;
  const double Pe  = P.U() * P.D / P.alpha();

  // Lattice temperature = T in C; rho c_p = 1 lattice unit, so H is in kelvin
  // and La = L / c_p. A single dimensionless group is what that hides, and it
  // is the Stefan number, printed below.
  PhaseChange pc;
  pc.T_s = Real(P.Ts); pc.T_l = Real(P.Tl);
  pc.cp_s = pc.cp_l = Real(1);
  pc.k_s  = pc.k_l  = Real(a_l);
  pc.La   = Real(P.Lf / P.cp);
  Coll coll; coll.set_material(pc);
  const PhaseChange m = coll.material();   // the normalised copy
  auto Hof = [&](double T) {
    double f = (T - P.Ts) / (P.Tl - P.Ts);
    f = f < 0 ? 0 : (f > 1 ? 1 : f);
    return m.enthalpy_of(Real(T), Real(f));
  };
  const double Hmin = Hof(std::min(P.Tin, P.Tc)), Hmax = Hof(P.Th);
  coll.T_ref = Real(0.5 * (Hmin + Hmax));

  if (verbose) {
    std::printf("  [%s] N = %d  ny = %d  dx = %.2f um  dt = %.3e s  tau = %.2f\n",
                tag.c_str(), N, ny, dx * 1e6, dt, o.tau);
    std::printf("  [%s] U = %.4f mm/s  Pe_D = %.2f  U_lat = %.4e  U^2/cs2 = %.2e  "
                "Ste = %.2f\n", tag.c_str(), P.U() * 1e3, Pe, U_l, U_l * U_l / 0.25,
                (P.Th - P.Tl) * P.cp / P.Lf);
  }

  Domain d(N + 1, ny, 1, false, false, true);
  ScalarSolver<D3Q7, EsotericPull<D3Q7>, Coll> s(d, coll);
  s.set_geometry([&](Index x, Index y, Index) -> std::uint8_t {
    if (x == 0 || x == N) return ScalarMoment;          // bore wall
    if (y == 0) return ScalarMoment;                     // filament inlet
    if (y == ny - 1) return ScalarOutflow;               // nozzle exit
    return ScalarBulk;
  });
  s.set_wall_values([&](Index x, Index y, Index) -> Real {
    const double z = (ny - 1 - y) * dx;
    if (x == 0 || x == N) return Real(Hof(P.Twall(z)));
    return Real(Hof(P.Tin));
  });
  s.finalize_geometry();   // builds the moment masks AND the outflow donors

  View1D<Real> ux("ux", d.n_padded), uy("uy", d.n_padded), uz("uz", d.n_padded);
  Kokkos::deep_copy(ux, Real(0));
  Kokkos::deep_copy(uy, Real(U_l));
  Kokkos::deep_copy(uz, Real(0));
  s.set_velocity(ux, uy, uz);

  // Fresh filament at T_in everywhere inside a hot end already at temperature:
  // the state just after Strategy 1's purge and reload. t90 is measured from it.
  const Real Hin = Real(Hof(P.Tin));
  s.initialize_field_in_flow(KOKKOS_LAMBDA(Index) -> Real { return Hin; });

  std::vector<double> H(std::size_t(N + 1) * ny), Hp;
  auto read = [&]() {
    s.compute_field();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x <= N; ++x) H[std::size_t(y) * (N + 1) + x] = double(h(d.id(x, y, 0)));
  };
  auto fl_at = [&](int x, int y) {
    Real f, T, E, c; m.invert(Real(H[std::size_t(y) * (N + 1) + x]), f, T, E, c);
    return double(f);
  };
  auto measure = [&](Outcome& r) {
    double A = 0, Aab = 0;
    for (int y = 0; y < ny; ++y) {
      const double z  = (ny - 1 - y) * dx;
      const double wy = (y == 0 || y == ny - 1) ? 0.5 : 1.0;
      for (int x = 0; x <= N; ++x) {
        const double w = wy * ((x == 0 || x == N) ? 0.5 : 1.0) * dx * dx * fl_at(x, y);
        A += w;
        if (z > P.Lh) Aab += w;
      }
    }
    r.l_melt  = A / P.D * 1e3;
    r.l_above = Aab / P.D * 1e3;
    // Topmost f_l = 1/2 crossing in a column, scanning down from the inlet.
    auto front = [&](int x) {
      for (int y = 1; y < ny; ++y) {
        const double f0 = fl_at(x, y - 1), f1 = fl_at(x, y);
        if (f0 < 0.5 && f1 >= 0.5) {
          const double yc = (y - 1) + (0.5 - f0) / (f1 - f0);
          return (ny - 1 - yc) * dx * 1e3;
        }
      }
      return 0.0;
    };
    // At a FIXED fraction of the width. The first version read column x = 1,
    // i.e. one node from the wall, which is a different physical place at every
    // N and drifted toward the wall value under refinement by construction.
    r.z_axis = front(N / 2);
    r.z_wall = front(N / 8);
  };

  // Steady when the whole field moves by less than tol over a quarter flush.
  const double tol   = sizeof(Real) == 4 ? 1e-6 : 1e-9;
  const long   sub   = std::max<long>(10, long(0.25 * (ny - 1) / U_l) / 100);
  const long   chunk = 100 * sub;                 // ~ a quarter flush, a multiple of sub
  const long   cap   = 60 * chunk * 4;            // ~60 flush times
  std::vector<double> ts, ls;
  FILE* fs = nullptr;
  if (!o.out.empty()) {
    fs = std::fopen((o.out + "_" + tag + "_series.csv").c_str(), "w");
    if (fs) std::fprintf(fs, "t_s,l_melt_mm,l_above_mm,z_axis_mm,z_w8_mm\n");
  }
  // The series is sampled 100x more often than the convergence check, or t90
  // reads as a multiple of a quarter flush -- which it did, identically for
  // every MEW row, in the first version.
  Outcome r;
  read();
  Hp = H;
  for (long t = 0; t < cap; t += chunk) {
    for (long j = 0; j < chunk; j += sub) {
      for (long k = 0; k < sub; ++k) s.step();
      read();
      r.steps  = t + j + sub;
      r.t_phys = r.steps * dt;
      measure(r);
      ts.push_back(r.t_phys); ls.push_back(r.l_melt);
      if (fs) std::fprintf(fs, "%.4f,%.6f,%.6f,%.6f,%.6f\n", r.t_phys, r.l_melt, r.l_above,
                           r.z_axis, r.z_wall);
    }
    double dmax = 0;
    for (std::size_t n = 0; n < H.size(); ++n) dmax = std::max(dmax, std::abs(H[n] - Hp[n]));
    Hp = H;
    r.change = dmax / (Hmax - Hmin);
    if (r.change < tol) { r.steady = true; break; }
  }
  if (fs) std::fclose(fs);
  for (std::size_t i = 0; i < ls.size(); ++i)
    if (ls[i] >= 0.9 * r.l_melt) { r.t90 = ts[i]; break; }

  if (!o.out.empty()) {
    FILE* ff = std::fopen((o.out + "_" + tag + "_field.csv").c_str(), "w");
    if (ff) {
      std::fprintf(ff, "x_mm,z_mm,T_C,f_l\n");
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x <= N; ++x) {
          Real f, T, E, c; m.invert(Real(H[std::size_t(y) * (N + 1) + x]), f, T, E, c);
          std::fprintf(ff, "%.5f,%.5f,%.4f,%.5f\n", x * dx * 1e3, (ny - 1 - y) * dx * 1e3,
                       double(T), double(f));
        }
      std::fclose(ff);
    }
  }
  return r;
}

void header() {
  std::printf("  %-6s %6s %4s %9s %9s %9s %9s %9s %9s %8s %7s\n", "feed", "T_c", "N",
              "l_melt", "l_above", "z_axis", "z_W/8", "z_eq", "t90 [s]", "t [s]", "steady");
}
void row(const char* feed, const Physical& P, int N, const Outcome& r) {
  // Local equilibrium: the height at which the WALL reaches the middle of the
  // band. A filament slow enough to equilibrate across its width melts there.
  const double Tm = 0.5 * (P.Ts + P.Tl);
  double zeq;
  if (P.Tc >= Tm) zeq = P.Ltot();
  else zeq = P.Lh + P.Lb * (P.Th - Tm) / (P.Th - P.Tc);
  std::printf("  %-6s %6.1f %4d %9.3f %9.3f %9.3f %9.3f %9.3f %9.1f %8.1f %7s\n", feed,
              P.Tc, N, r.l_melt, r.l_above, r.z_axis, r.z_wall, zeq * 1e3, r.t90, r.t_phys,
              r.steady ? "yes" : "NO");
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    Physical P;
    Opts o;
    bool set_feed = false;
    for (int i = 1; i < argc; ++i) {
      auto num = [&]() {
        if (i + 1 >= argc) { std::printf("mew_liquefier: %s needs a value\n", argv[i]); std::exit(2); }
        return std::atof(argv[++i]);
      };
      if      (!std::strcmp(argv[i], "-N"))      o.N = int(num());   // a multiple of 8
      else if (!std::strcmp(argv[i], "-tau"))    o.tau = num();
      else if (!std::strcmp(argv[i], "-Q"))      { P.Q = num() * 1e-9; set_feed = true; }  // mm^3/s
      else if (!std::strcmp(argv[i], "-Tc"))     { P.Tc = num(); set_feed = true; }
      else if (!std::strcmp(argv[i], "-Th"))     P.Th = num();
      else if (!std::strcmp(argv[i], "-sweep"))  o.sweep = true;
      else if (!std::strcmp(argv[i], "-ladder")) o.ladder = true;
      else if (!std::strcmp(argv[i], "-out") && i + 1 < argc) o.out = argv[++i];
      else {
        std::printf("mew_liquefier: unknown option '%s'\n", argv[i]);
        Kokkos::finalize();
        return 2;
      }
    }
    // -sweep sets Q and T_c per row; silently overriding an explicit flag is
    // the "default applied after the argument loop" trap in CLAUDE.md.
    if (o.sweep && set_feed) {
      std::printf("mew_liquefier: -sweep sets -Q and -Tc itself; drop them or drop -sweep\n");
      Kokkos::finalize();
      return 2;
    }
    if (o.N < 8 || o.N % 8 != 0) {
      std::printf("mew_liquefier: -N must be a multiple of 8 (the W/8 front column)\n");
      Kokkos::finalize();
      return 2;
    }
    std::printf("mew_liquefier: D3Q7 EnthalpyBGK(Material), nz = 1, precision %s\n",
                sizeof(Real) == 4 ? "float" : "double");
    std::printf("  PCL (nominal): rho %.0f  cp %.0f  k %.2f  L %.0f kJ/kg  band [%.0f, %.0f] C"
                "  alpha = %.3e m^2/s\n", P.rho, P.cp, P.k, P.Lf * 1e-3, P.Ts, P.Tl, P.alpha());
    std::printf("  hot end (nominal): heater %.0f mm at %.0f C, break %.0f mm, cold zone %.0f mm,"
                " filament in at %.0f C\n", P.Lh * 1e3, P.Th, P.Lb * 1e3, P.Lc * 1e3, P.Tin);
    std::printf("  lengths in mm; l_* = melt AREA / W; z_* = f_l = 1/2 height above the nozzle"
                " exit, on the axis and at W/8 from the wall; z_eq = where the WALL reaches the band middle\n\n");

    if (o.ladder) {
      header();
      for (int N : {16, 32, 64}) {
        Opts oo = o; oo.N = N;
        const Outcome r = run(P, oo, true, "ladder_N" + std::to_string(N));
        row("cfg", P, N, r);
        if (!r.steady) rc = 1;
      }
    } else if (o.sweep) {
      header();
      const double feeds[2] = {0.255, 4.09};
      const char*  names[2] = {"MEW", "FDM"};
      for (int f = 0; f < 2; ++f)
        for (double Tc : {30.0, 45.0, 55.0, 65.0}) {
          Physical Pp = P; Pp.Q = feeds[f] * 1e-9; Pp.Tc = Tc;
          char tag[32]; std::snprintf(tag, sizeof tag, "%s_Tc%.0f", names[f], Tc);
          const Outcome r = run(Pp, o, Tc == 30.0, tag);
          row(names[f], Pp, o.N, r);
          if (!r.steady) rc = 1;
        }
    } else {
      header();
      const Outcome r = run(P, o, true, "single");
      row("cfg", P, o.N, r);
      if (!r.steady) rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
