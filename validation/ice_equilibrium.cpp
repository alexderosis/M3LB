//==============================================================================
//  ICE ON WATER AT EQUILIBRIUM -- the coupled melting model (EnthalpyBGK +
//  DarcyGuo + the density-anomaly buoyancy) against an EXACT steady state.
//
//  The configuration is Wang, Jiang, Du, Sun & Calzavarini, PRFluids 6,
//  L091501 (2021): water under a plate held below 0 C, a heated plate below,
//  ice growing down from the cold plate until the heat it conducts balances the
//  heat the water delivers. Their comparison runs at T_b = 10 C in a 24 cm cell,
//  where the water layer convects at Ra ~ 1e8 and equilibrium takes 2.5 days of
//  physical time -- a GPU computation, and not this one.
//
//  THIS CASE IS THE LIMIT IN WHICH THE ANSWER IS EXACT. With T_b <= 4 C all
//  the water sits BELOW the density maximum, so warmer water is DENSER and the
//  layer is stably stratified: nothing convects, and the equilibrium is
//  two-phase conduction with the ice front where the fluxes match,
//
//      k_i (0 - T_t) / h  =  k_w (T_b - 0) / (H - h)
//      h / H = k_i |T_t| / (k_i |T_t| + k_w T_b),
//
//  with T piecewise linear in each phase. Every piece of the coupled model is
//  in play and each has a way to miss it: the enthalpy inversion (T and f_l
//  from H), the two conductivities and capacities (k_s cp_l != k_l cp_s, so the
//  relaxation rate comes from each node's own phase), the on-node enthalpy
//  walls, the implicit mushy drag, and the anomaly's SIGN -- which is what
//  keeps the water still. The buoyancy is sized so that the water layer would
//  sit at Ra ~ 5e3 if its stratification were unstable, three times the
//  classical onset, and the water is SEEDED with a mode that a wrong sign would
//  grow; the check on the residual velocity is therefore not vacuous.
//
//  THE RATIOS ARE WATER'S, THE LATENT HEAT IS NOT. k_i / k_w = 3.944 (2.25 /
//  0.5705 W/(m K), ice at -5 C, water at 5 C: the phase-mean temperatures
//  Wang et al. evaluate at) and (rho c_p)_i / (rho c_p)_w = 0.4824 (2027 /
//  4202 J/(kg K), equal densities as in their Boussinesq model). The latent
//  heat is La = 2 in units of (rho c_p)_w K against water's 79.5: a steady
//  state cannot depend on it, and it sets only how long the front takes to get
//  there. The same argument would let the GPU runs shorten their 2.5 days.
//
//  GEOMETRY. D3Q27 central moments + D3Q7 EnthalpyBGK at nz = 1, periodic in
//  x, HALFWAY walls for both fields: bounce-back for the fluid and
//  anti-bounce-back for the enthalpy, planes at y = 0.5 and H + 0.5, exactly
//  the pair rayleigh_benard.cpp and density_anomaly.cpp run. Two things about
//  that choice were learnt the hard way:
//
//   * ANTI-BOUNCE-BACK TAKES THE SENSIBLE ENTHALPY E(T_w), NOT H. It imposes
//     its value through the NON-REST equilibrium, w_i (E - T_ref), which
//     carries only E; handed the total H, the liquid wall would sit La too
//     warm. So the walls get H_bot - La (water) and H_top (ice, f_l = 0).
//   * THE ON-NODE PAIR DIVERGED, AND THE FAULT IS THE FORCED REGULARISED WALL.
//     The first version ran ScalarMoment + regularised fluid walls on nodes 0
//     and H. It went NON-FINITE at 170k steps with the ice drag A = 1e3 and at
//     424k with A = 10, the fastest-growing velocity always at y = 1 -- next to
//     the WATER wall, where A = 0, so the drag was not involved. -frozen (no
//     fluid step) converged cleanly, and this halfway pair has run 2.7M steps
//     at A = 1e3 without a hint of it. The same regularised wall carries every
//     on-node Rayleigh-Benard number in the tree, but those are growth rates
//     over 0.75 diffusive times, too short to see a drift that takes 1e5
//     steps. The mechanism is NOT diagnosed; CLAUDE.md records it.
//
//  THE ISOTHERMAL FRONT DOES NOT HAVE A UNIQUE DISCRETE EQUILIBRIUM, AND WHERE
//  IT STOPS IS PREDICTABLE. With T_s = T_l a steady state needs only the node
//  below the front at T >= 0 and the node above at T <= 0, so a whole WINDOW of
//  link positions is an equilibrium: linearising the two-slab interface
//  temperature, T_i ~ -G (p - p_e), the window is [-slope_i/(2G),
//  +slope_w/(2G)] cells, H-independent because both slopes and G fall as 1/H.
//  Ice GROWING toward equilibrium stops at the first link inside it, i.e.
//  within one cell short of the thin edge. Predicted edges 0.906 and 1.222
//  cells; measured at H = 64, 0.903 and 1.241. The error in CELLS therefore
//  does not refine away -- the isothermal front is first order in H at best --
//  and the check here is that it lands in the predicted window, not near the
//  exact h. A MUSHY BAND [-d, d] removes the ambiguity (f_l is then a function
//  of T) and converges onto ITS exact equilibrium, which the Kirchhoff
//  transform K(T) = int k dT gives in closed form. But the band is a model
//  change: at d = 0.2 C the band's own h differs from the isothermal one by
//  0.87 cells at H = 64 here, because its reduced conductivity spans 4.6 cells
//  of the gently sloped ice. Choose d against the gradient at the front.
//
//  -conv repeats the case at H = 16, 32, 64; -band d gives the solid/liquid
//  range [-d, d]; -A, -ra and -frozen are the controls used to find the wall
//  instability recorded above. Default: H = 32, d = 0 and 0.2, 95 s.
//
//  MEASURED 2026-09-24, FP64, Threads, A_solid = 100, would-be Ra 5e3.
//  Front error in cells (T = 0 crossing; negative = ice too thin):
//
//                              H = 16    H = 32    H = 64
//    isothermal  -2 / +3 C     -0.450    -0.780    -0.903    edge 0.906
//                -1 / +3.8 C   -1.670    -1.025    -1.241    edge 1.222
//    d = 0.2 C   -2 / +3 C     -0.240    -0.049    +0.090    T err 1.8/0.79/0.18 %
//                -1 / +3.8 C   -0.110    -0.075    -0.022    T err 1.8/0.32/0.094 %
//
//  At every row the coupling contributes nothing -- the frozen twin agrees to
//  <= 1.2e-8 of the span -- the residual velocity is <= 2.5e-9 of the would-be
//  free-fall velocity, and the seeded mode has decayed by >= 1e9. H = 16 at
//  -1 / +3.8 C falls outside its window (7.85 cells of water); it is reported,
//  not asserted, since only -conv runs it.
//==============================================================================
#include "boundary/MomentDirichlet.hpp"
#include "boundary/Regularized.hpp"
#include "collision/BGK.hpp"
#include "collision/EnthalpyBGK.hpp"
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace lbm;

using FL = D3Q27;
using SL = D3Q7;
using FColl = CentralMoments<FL, DarcyGuo, ShiftedPopulations>;
using SColl = EnthalpyBGK<SL>;

namespace {

constexpr double kKi   = 2.25 / 0.5705;       // k_ice / k_water
constexpr double kCi   = 2027.0 / 4202.0;     // (rho c_p)_ice / (rho c_p)_water
constexpr double kLa   = 2.0;                 // latent heat, (rho c_p)_w K
constexpr double kAlw  = 0.02;                // water diffusivity, lattice
constexpr double kNu   = 0.02;                // water viscosity, lattice
constexpr double kQ    = DensityAnomalyGuo::q_water;
constexpr double kTm   = DensityAnomalyGuo::Tm_water_C;
double kAsol = 1.0e2;                         // Darcy drag at f_l = 0 (-A)
double kBand = 0.0;                           // -band: T_s = -d, T_l = +d
bool   kFrozen = false;                       // -frozen: no fluid step
double kRa = 5e3;                             // -ra: would-be Ra of the water
constexpr double kEps  = 1.0e-3;              // Carman-Kozeny floor

struct Result {
  double h, hT, h_exact;        // ice thickness / H: f_l integral, T = 0 crossing
  double h_exact_fl;            // exact f_l-integral thickness (differs inside a band)
  std::vector<double> T;        // final interior temperature, for the frozen twin
  double window;                // isothermal: predicted thin-side pinning edge, cells
  double T_err;                 // max |T - exact| / (T_b - T_t)
  double u_ratio;               // max |u| / would-be free-fall velocity
  double seed_ratio;            // late / early amplitude of the seeded mode
  long   steps;
  bool   converged, finite;
};

Result run(Index H, double Tt, double Tb, double ra_if_unstable, bool verbose,
           bool frozen = false) {
  const Index nx = H, ny = H + 2;              // halfway walls: H nodes deep
  Domain d(nx, ny, 1, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  // THE EXACT EQUILIBRIUM, by the Kirchhoff transform K(T) = int_0^T k:
  // the flux q = -k dT/ds is uniform, so K(T(s)) = K(T_b) - q s from the
  // bottom plane. Inside a band [-d, d] the liquid fraction is linear in T and
  // k = k_s + f (k_l - k_s) (MushMix::Parallel), so K is quadratic there; at
  // d = 0 this is the two-slab formula k_i |T_t| / (k_i |T_t| + k_w T_b).
  const double kl = 1.0, ks = kKi, dk = kl - ks, bd = kBand;
  auto K = [&](double T) {
    auto band = [&](double t) { return ks * t + (bd > 0 ? dk / (4 * bd) * ((t + bd) * (t + bd) - bd * bd) : 0.0); };
    if (T >= bd)  return band(bd) + kl * (T - bd);
    if (T <= -bd) return band(-bd) + ks * (T + bd);
    return band(T);
  };
  const double q  = (K(Tb) - K(Tt)) / double(H);
  const double he = 1.0 - K(Tb) / q / double(H);              // T = 0 front, h / H
  // f_l integral: full liquid below T = d, plus the band's int f k dT / q.
  const double liq = (K(Tb) - K(bd)) / q + (bd > 0 ? 2 * bd * (ks / 2 + dk / 3) / q : 0.0);
  const double he_fl = 1.0 - liq / double(H);
  auto T_exact = [&](double sdist) {                          // bisection on K
    const double target = K(Tb) - q * sdist;
    double lo = Tt, hi = Tb;
    for (int it = 0; it < 80; ++it) {
      const double mid = 0.5 * (lo + hi);
      (K(mid) > target ? hi : lo) = mid;
    }
    return 0.5 * (lo + hi);
  };
  const double dw = (1.0 - he) * double(H);                  // water depth
  // Buoyancy sized from the Ra the water layer WOULD have if unstable.
  const double wlat = ra_if_unstable * kNu * kAlw / (std::pow(Tb, kQ) * dw * dw * dw);
  const double u_ff = std::sqrt(wlat * std::pow(Tb, kQ) * dw);

  PhaseChange pc;
  pc.T_s = Real(-kBand); pc.T_l = Real(kBand);
  pc.cp_s = Real(kCi); pc.cp_l = Real(1);
  pc.k_s = Real(kKi * kAlw); pc.k_l = Real(kAlw);
  pc.La = Real(kLa); pc.E_datum = Real(0);
  pc.normalise();

  SColl scoll;
  scoll.set_material(pc);
  const Real H_bot = pc.enthalpy_of(Real(Tb), Real(1));
  const Real H_top = pc.enthalpy_of(Real(Tt), Real(0));
  scoll.T_ref = Real(0.5) * (H_bot + H_top);
  scoll.omega = SColl::omega_from_diffusivity(Real(kAlw));
  ScalarSolver<SL, EsotericPull<SL>, SColl> th(d, scoll);
  // Anti-bounce-back imposes its value through the NON-REST equilibrium,
  // w_i (E - T_ref): the SENSIBLE enthalpy. So it is handed E(T_w), which is
  // H - La at the liquid wall and H at the ice wall, not the total H.
  const Real E_bot = H_bot - pc.La, E_top = H_top;
  th.set_geometry([&](Index, Index y, Index) -> ScalarCell {
    return (y == 0 || y == ny - 1) ? ScalarDirichlet : ScalarBulk;
  });
  th.set_wall_values([&](Index, Index y, Index) -> Real {
    return (y == 0) ? E_bot : E_top;
  });
  th.finalize_geometry();

  // Start AWAY from the answer: a front 15 % of H too high, linear profiles on
  // either side of it, and a small mode in the water that the anomaly's sign
  // must damp.
  const double h0 = std::max(0.05, he - 0.15);
  const double kx = 2.0 * M_PI / double(nx);
  const PhaseChange m = pc;
  const Index Hc = H;
  th.initialize_field(KOKKOS_LAMBDA(Index n) -> Real {
    Index px, py, pz; d.coords(n, px, py, pz);
    const double x = double(px - d.hx), y = (double(py - d.hy) - 0.5) / double(Hc);
    const double yf = 1.0 - h0;                        // front, from the bottom
    if (y >= yf) {                                     // ice
      const double T = Tt * (y - yf) / (1.0 - yf);
      return m.enthalpy_of(Real(T), Real(0));
    }
    double T = Tb * (1.0 - y / yf);
    T += 0.02 * Tb * std::sin(kx * x) * std::sin(M_PI * y / yf);
    return m.enthalpy_of(Real(T), Real(1));
  });
  th.compute_field();

  View1D<Real> Ey("Ey", d.n_padded), Av("A", d.n_padded);
  FColl fcoll;
  fcoll.omega = FColl::omega_from_viscosity(Real(kNu));
  fcoll.omega_bulk = Real(1);
  fcoll.forcing = DarcyGuo{};
  fcoll.forcing.Ey = Ey;
  fcoll.forcing.A = Av;
  FluidSolver<FL, EsotericPull<FL>, FColl> fl(d, fcoll);
  fl.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });
  fl.initialize(Real(1));
  th.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // T0 at the mean of the water's |T - Tm|^q, so the still layer's force
  // averages to zero and the hydrostatic adjustment is as small as it can be.
  const double T0 = kTm - std::pow((std::pow(kTm, kQ + 1.0) -
                                    std::pow(kTm - Tb, kQ + 1.0)) /
                                   ((kQ + 1.0) * Tb), 1.0 / kQ);

  auto couple = [&]() {
    th.compute_field();
    auto Hf = th.temperature();                 // an ENTHALPY, see CLAUDE.md
    auto ey = Ey; auto av = Av;
    const Real w = Real(wlat), q = Real(kQ), Tm = Real(kTm), T0r = Real(T0);
    const Real Asol = Real(kAsol), eps = Real(kEps);
    const PhaseChange mm = pc;
    Kokkos::parallel_for("ice_couple", d.n_padded, KOKKOS_LAMBDA(Index n) {
      Real f, T, E, dE;
      mm.invert(Hf(n), f, T, E, dE);
      // f_l-weighted: the ice is held by the drag, and |T - Tm|^q reaches
      // 150 at -10 C, which unweighted would compress the lattice density
      // hydrostatically by far more than the flow can tolerate.
      ey(n) = f * w * (Kokkos::pow(Kokkos::fabs(T - Tm), q) -
                       Kokkos::pow(Kokkos::fabs(T0r - Tm), q));
      av(n) = Asol * eps * (Real(1) - f) * (Real(1) - f) / (f * f * f + eps);
    });
  };

  auto ice_thickness = [&]() {
    th.compute_field();
    auto hH = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    double liq = 0;
    for (Index y = 1; y <= H; ++y) {                // interior rows, one cell each
      const double wgt = 1.0;
      double row = 0;
      for (Index x = 0; x < nx; ++x) row += double(pc.liquid_fraction(hH(d.id(x, y))));
      liq += wgt * row / double(nx);
    }
    return 1.0 - liq / double(H);
  };
  auto seed_amp = [&]() {
    fl.compute_macroscopic();
    auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    double s = 0, c = 0;
    for (Index y = 1; y < ny - 1; ++y)
      for (Index x = 0; x < nx; ++x) {
        s += double(hv(d.id(x, y))) * std::sin(kx * double(x));
        c += double(hv(d.id(x, y))) * std::cos(kx * double(x));
      }
    return std::hypot(s, c);
  };

  Result r{0, 0, he, he_fl, {}, 0, 0, 0, 0, 0, false, true};
  // THE ISOTHERMAL FRONT'S PINNING WINDOW, predicted. With no band, a steady
  // state needs only the node below the front at T >= 0 and the node above at
  // T <= 0, so any link whose two-slab interface temperature T_i lies in
  // [-slope_w / 2, +slope_i / 2] is an equilibrium. Linearising T_i about the
  // exact front, T_i ~ -G (p - p_e) with G = -N'(p_e) / D(p_e) for
  // N = k_w T_b / p - k_i |T_t| / (H - p), D = k_w / p + k_i / (H - p), so a
  // front GROWING toward equilibrium stops at the first link inside the window,
  // i.e. within one cell short of its thin edge slope_w / (2 G). In cells, and
  // H-independent: both slopes and G scale as 1/H.
  {
    const double pe = (1.0 - kKi * (-Tt) / (kKi * (-Tt) + Tb)) * double(H);
    const double sw = Tb / pe;
    const double Np = -(Tb / (pe * pe) + kKi * (-Tt) / ((double(H) - pe) * (double(H) - pe)));
    const double Dp = 1.0 / pe + kKi / (double(H) - pe);
    r.window = 0.5 * sw / (-Np / Dp);
  }
  // CONVERGED MEANS THE WHOLE ENTHALPY FIELD HAS STOPPED, NOT h. With an
  // isothermal front the liquid-fraction integral is FLAT for thousands of
  // steps while the front node heats or cools sensibly toward 0 C, so "h did
  // not move" fires on a plateau -- an earlier version stopped at 36000 steps
  // three cells from the answer that way.
  auto snapshot = [&]() {
    th.compute_field();
    auto hH = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
    std::vector<double> v(std::size_t(nx * ny));
    for (Index y = 0; y < ny; ++y)
      for (Index x = 0; x < nx; ++x) v[std::size_t(y * nx + x)] = double(hH(d.id(x, y)));
    return v;
  };
  const long probe = 2000, cap = 4000000;
  const double Hspan = double(H_bot - H_top);
  std::vector<double> prevH = snapshot();
  double a_early = -1;
  for (long t = 0; t < cap; t += probe) {
    for (long k = 0; k < probe; ++k) {
      if (!(kFrozen || frozen)) { couple(); fl.step(true); }
      th.step();
    }
    r.steps = t + probe;
    if (a_early < 0) a_early = seed_amp();
    const std::vector<double> now = snapshot();
    double dmax = 0;
    for (std::size_t i = 0; i < now.size(); ++i) {
      if (!std::isfinite(now[i])) { r.finite = false; break; }
      dmax = std::max(dmax, std::abs(now[i] - prevH[i]));
    }
    if (!r.finite) break;
    if (verbose && (r.steps % 40000 == 0))
    {
      fl.compute_macroscopic();
      auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
      double um = 0; Index ym = 0;
      for (Index y = 0; y < ny; ++y)
        for (Index x = 0; x < nx; ++x) {
          const double u = std::hypot(double(hu(d.id(x, y))), double(hv(d.id(x, y))));
          if (u > um) { um = u; ym = y; }
        }
      std::printf("      step %8ld  h/H = %.7f  max dH %.2e  max|u|/u_ff %.2e at y = %d\n",
                  r.steps, ice_thickness(), dmax / Hspan, um / u_ff, int(ym));
    }
    prevH = now;
    if (dmax / Hspan < 1e-10) { r.converged = true; break; }
  }
  r.h = ice_thickness();
  r.seed_ratio = seed_amp() / std::max(a_early, 1e-300);

  // Pointwise temperature against the exact piecewise-linear profile, and the
  // largest velocity anywhere against the free-fall velocity the layer would
  // have if it convected.
  th.compute_field(); fl.compute_macroscopic();
  auto hH = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  double terr = 0, umax = 0;
  for (Index y = 1; y <= H; ++y)
    for (Index x = 0; x < nx; ++x) {
      const Index n = d.id(x, y);
      const double T = double(pc.temperature_of(hH(n)));
      const double Te = T_exact(double(y) - 0.5);     // from the plane y = 0.5
      terr = std::max(terr, std::abs(T - Te));
      r.T.push_back(T);
      umax = std::max(umax, std::hypot(double(hu(n)), double(hv(n))));
      if (!std::isfinite(T)) r.finite = false;
    }
  r.T_err = terr / (Tb - Tt);
  // The front from T: the 0 C crossing between the last node above 0 and the
  // first at or below it, interpolated, averaged over the columns.
  double y0sum = 0;
  for (Index x = 0; x < nx; ++x) {
    double y0 = 0;
    for (Index y = 1; y < H; ++y) {
      const double T1 = double(pc.temperature_of(hH(d.id(x, y))));
      const double T2 = double(pc.temperature_of(hH(d.id(x, y + 1))));
      if (T1 > 0 && T2 <= 0) { y0 = double(y) - 0.5 + T1 / (T1 - T2); break; }
    }
    y0sum += y0;
  }
  r.hT = 1.0 - (y0sum / double(nx)) / double(H);
  r.u_ratio = umax / u_ff;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  bool conv = false, verbose = false;
  int only = -1;
  bool band_set = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-conv") conv = true;
    if (std::string(argv[i]) == "-v") verbose = true;
    if (std::string(argv[i]) == "-frozen") kFrozen = true;
    if (std::string(argv[i]) == "-case" && i + 1 < argc) only = std::atoi(argv[++i]);
    if (std::string(argv[i]) == "-A" && i + 1 < argc) kAsol = std::atof(argv[++i]);
    if (std::string(argv[i]) == "-ra" && i + 1 < argc) kRa = std::atof(argv[++i]);
    if (std::string(argv[i]) == "-band" && i + 1 < argc) { kBand = std::atof(argv[++i]); band_set = true; }
  }
  int rc = 0;
  Kokkos::initialize(argc, argv);
  {
    std::printf("Ice on water at equilibrium: EnthalpyBGK + DarcyGuo + density "
                "anomaly, exact two-phase conduction\n");
    std::printf("D3Q27 CM + D3Q7 enthalpy, nz = 1, periodic x, on-node walls;  "
                "k_i/k_w = %.4f  C_i/C_w = %.4f  La = %.1f\n", kKi, kCi, kLa);
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    struct Case { double Tt, Tb; };
    const Case cases[] = {{-2.0, 3.0}, {-1.0, 3.8}};
    const std::vector<Index> grids = conv ? std::vector<Index>{16, 32, 64}
                                          : std::vector<Index>{32};
    const std::vector<double> bands = band_set ? std::vector<double>{kBand}
                                               : std::vector<double>{0.0, 0.2};
    for (double band : bands) {
    kBand = band;
    std::printf("  band d = %.3f C   A_solid = %.0f   would-be Ra = %.0f   %s\n",
                kBand, kAsol, kRa, kBand > 0
                    ? "(against the band's own exact solution)"
                    : "(isothermal: against the predicted pinning window)");
    std::printf("  %-12s %-4s %-9s %-9s %-8s %-9s %-9s %-8s %-8s %-8s %-8s %-8s %s\n",
                "T_t, T_b", "H", "T=0 ex", "T=0", "err[c]", "f_l ex", "f_l", "err[c]",
                "T err", "twin", "u/u_ff", "seed", "steps");
    for (int ci = 0; ci < 2; ++ci) {
      if (only >= 0 && ci != only) continue;
      const Case& c = cases[ci];
      double prev = NAN;
      for (Index H : grids) {
        const Result r = run(H, c.Tt, c.Tb, kRa, verbose);
        // THE FROZEN TWIN: the same case with no fluid step at all. The water
        // is stable, so the flow must contribute NOTHING to the equilibrium;
        // the difference between the two fields is that claim, measured.
        const Result z = run(H, c.Tt, c.Tb, kRa, false, /*frozen=*/true);
        double twin = 0;
        for (std::size_t i = 0; i < r.T.size() && i < z.T.size(); ++i)
          twin = std::max(twin, std::abs(r.T[i] - z.T[i]));
        twin /= (c.Tb - c.Tt);
        const double ec = (r.hT - r.h_exact) * double(H);        // in cells
        const double ef = (r.h - r.h_exact_fl) * double(H);
        // Isothermal: ice too thin by e cells, e in (edge - 1, edge], with a
        // 0.1-cell allowance for the two-slab model. Band: its exact solution.
        const double e = -ec;
        const bool front_ok = kBand > 0
            ? (std::abs(ec) <= 0.3 && r.T_err <= 0.025)
            : (H < 32 || (e >= r.window - 1.1 && e <= r.window + 0.1));
        const bool ok = r.finite && r.converged && z.converged && front_ok &&
                        twin < 1e-6 && r.u_ratio < 1e-4 && r.seed_ratio < 1e-3;
        std::printf("  %+4.1f,%+4.1f %-4d %-9.6f %-9.6f %+8.3f %-9.6f %-9.6f %+8.3f "
                    "%-8.1e %-8.1e %-8.1e %-8.1e %ld %s", c.Tt, c.Tb, int(H),
                    r.h_exact, r.hT, ec, r.h_exact_fl, r.h, ef, r.T_err, twin,
                    r.u_ratio, r.seed_ratio, r.steps, ok ? "PASS" : "FAIL");
        if (kBand == 0) std::printf("  (window edge %.3f)", r.window);
        std::printf("\n");
        prev = ec / double(H);
        (void)prev;
        if (!ok) rc = 1;
      }
    }
    std::printf("\n");
    }
    std::printf("\n  %s\n\n", rc ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return rc;
}
