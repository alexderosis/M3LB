//==============================================================================
//  WATER NEAR 4 C -- DensityAnomalyGuo (src/forcing/Forcing.hpp) against the
//  EXACT onset of penetrative convection, and against its own linear limit.
//
//  A rigid-rigid layer, hot plate below at theta = 1, cold plate above at 0,
//  whose buoyancy is b(theta) = |theta - lam|^q: the density is greatest at
//  theta = lam, so the fluid below z = 1 - lam is unstably stratified and the
//  fluid above it STABLY. That is the stratification under a melting ice
//  ceiling -- 0 C at the ice, warmer water below, 4 C in between -- and it is
//  the sign change the whole scallop problem turns on. Ra = w H^3 / (nu kappa),
//  with w the anomaly's coefficient in place of g beta dT.
//
//  WHAT IS CHECKED, and why each is a different kind of claim.
//
//   1. THE LINEAR LIMIT, AS AN IDENTITY TO ROUNDING. q = 1 with every
//      temperature above Tm is BoussinesqGuo with beta = w. The same growing
//      Rayleigh-Benard mode is run under both policies and the fields compared.
//      Not bit for bit -- |T - Tm| - |T0 - Tm| and T - T0 round differently --
//      so the criterion is a rounding-level bound, and it catches a wrong sign,
//      a wrong direction or a factor anywhere in the new at().
//
//   2. THE ONSET, AGAINST AN EXACT REFERENCE, at the Gebhart-Mollendorf
//      q = 1.894816 and two positions of the maximum:
//        lam = 0.5 -- maximum mid-layer, upper half stable: the case that
//                     needs the sign change to be right;
//        lam = 0   -- maximum on the cold plate, whole layer unstable but b'
//                     falling from 1.9 to 0 across it: the case that needs the
//                     SHAPE to be right with no sign change to hide behind.
//      The reference is tools/penetrative_onset.py -- stdlib Python, shooting
//      with RK4 on a mesh split and graded at the kink z = 1 - lam, which
//      reproduces Chandrasekhar's 1707.762 / 3.117 (it prints 1707.7618 /
//      3.11632) before it will report anything. It is independent of this code:
//      no line is shared, and it solves the marginal ODEs rather than a lattice.
//      The box is periodic with width = one wavelength, and the width is chosen
//      so that k H is the SAME at every resolution (25/16 H and 2 H), so each
//      lam has one reference number, not one per grid:
//
//        lam = 0.5, k H = 4.021239 (2 pi / 1.5625):  Ra = 16764.6422
//        lam = 0,   k H = pi       (2 pi / 2):       Ra =  1674.2916
//
//      (the script's two meshes agree to 6.9e-5 and 2.0e-5). For scale, the
//      minima are Ra_c = 16763.68 at k_c H = 3.995 and 1674.22 at 3.125, and
//      the classical layer's is 1707.76 -- so lam = 0.5 moves the onset by a
//      factor of 9.8, which a sign error in the stable half could not survive.
//
//      Ra_c is read off the ZERO OF THE GROWTH RATE of the box's mode, which
//      rayleigh_benard.cpp found to agree with bisection to 0.04 % and is far
//      cheaper: the mode's amplitude is its projection onto exp(i k x) sin(pi
//      y/H), measured at three times after a quarter diffusive time of warm-up,
//      and the two half-span rates are printed so that a mode that has not
//      separated, or is oscillating, shows as a disagreement between them.
//
//  WHY nu AND kappa ARE FIXED AND THE BUOYANCY FOLLOWS -- the opposite of
//  rayleigh_benard.cpp, and for a reason. That case fixes a free-fall velocity
//  u_c = 0.02 and derives nu from Ra. At Ra = 1.7e4 that puts tau at 0.51,
//  and raising u_c instead raises the hydrostatic compression of the base
//  state, dRho/rho ~ 3 u_c^2, which is a Rayleigh-number error that does NOT
//  refine away. Fixing nu = kappa = 0.02 and taking w = Ra nu kappa / H^3 is
//  diffusive scaling: tau stays at 0.56 and 0.58 on every grid, and the
//  compression, ~ w H / cs^2 ~ 1 / H^2, falls at the same rate as the
//  discretisation error instead of setting a floor under it.
//
//  THE DISCRETISATION. D3Q27 central moments + D3Q7 BGK at nz = 1, the tree's
//  standing pair for new work; halfway bounce-back and anti-bounce-back walls,
//  both planes midway, so the layer is exactly H deep (rayleigh_benard.cpp's
//  banner argues why the two families must not be mixed). omega_bulk = 1, as in
//  rb_high_ra: it damps the acoustic adjustment of the base state without
//  touching an incompressible mode. The conduction state's body force is NOT
//  uniform under this policy, so T0 is chosen to make its layer mean zero; that
//  only minimises the hydrostatic adjustment and drops out of the stability
//  problem.
//
//  WHAT THIS DOES NOT CHECK: anything nonlinear (no Nusselt number is
//  compared), any sidewall, and any phase change. It validates the buoyancy law
//  and its coupling, not an ice problem.
//
//  -conv repeats the onset at H = 16, 32, 64 and prints the order.
//
//  MEASURED 2026-09-24, FP64, Threads backend (4 threads).
//
//      1. linear limit          max rel diff T 1.66e-15, v 7.45e-14 against
//                               BoussinesqGuo; sigma +2.212923 in both
//      2. onset error           H = 16      H = 32      H = 64     order
//         lam = 0.5            +4.665 %    +1.059 %    +0.250 %   2.14, 2.08
//         lam = 0              +2.133 %    +0.481 %    +0.113 %   2.15, 2.09
//
//  Both converge at second order ONTO the reference rather than past it, and
//  Richardson on the last pair gives 16761.40 and 1674.13 -- 0.019 % and
//  0.010 % from it, i.e. the remaining error IS the grid's. The two half-span
//  rates agree to 1e-4 in every run, so no mode was read before it separated.
//  lam = 0.5 carries about twice lam = 0's error at every H because its
//  unstable sublayer is H/2 deep: it is resolved by half the cells. The error
//  is POSITIVE throughout, as the classical layer's is with this wall pair
//  (+0.65 % at H = 32 in rayleigh_benard.cpp, D2Q9 BGK), so the anomaly adds
//  no error of its own sign. Default mode (H = 32 only) runs in 32 s; -conv in
//  about 8 min, 90 % of it the six H = 64 runs.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace lbm;

using FL = D3Q27;
using SL = D3Q7;
template <class P> using FluidCM = CentralMoments<FL, P, ShiftedPopulations>;

namespace {

constexpr double kQ   = DensityAnomalyGuo::q_water;
constexpr double kNu  = 0.02;            // lattice viscosity, every grid
constexpr double kPr  = 1.0;             // onset is independent of Pr
constexpr double kEps = 1e-3;            // seed amplitude in theta

// T0 that makes the conduction state's body force average to zero over the
// layer: |T0 - Tm|^q = mean of |theta - Tm|^q, theta in [-1/2, 1/2].
double neutral_T0(double q, double Tm) {
  auto P = [q](double s) {                          // antiderivative of |s|^q
    return std::copysign(std::pow(std::abs(s), q + 1.0), s) / (q + 1.0);
  };
  return Tm + std::pow(P(0.5 - Tm) - P(-0.5 - Tm), 1.0 / q);
}

struct Shape { double q, Tm, T0; };    // Tm, T0 in the lattice gauge below

struct Run {
  double sigma, sigma_a, sigma_b;      // per diffusive time H^2 / kappa
  std::vector<double> T, v;            // final fields, interior rows
  double seconds;
  bool ok;
};

// One growth-rate measurement. Gauge: theta_lat = theta - 1/2, hot plate +1/2
// below, cold plate -1/2 above, so the reference's lam sits at Tm = lam - 1/2.
template <class Policy>
Run run(Index H, double aspect, double Ra, Shape sh, double warm_td,
        double span_td) {
  const Index nx = Index(std::lround(aspect * double(H)));
  const Index ny = H + 2;
  const double kappa = kNu / kPr;
  const double G = Ra * kNu * kappa / (double(H) * double(H) * double(H));

  Domain d(nx, ny, 1, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  ScalarBGK<SL> scoll;
  scoll.omega = ScalarBGK<SL>::omega_from_diffusivity(Real(kappa));
  scoll.T_ref = Real(0);
  ScalarSolver<SL, EsotericPull<SL>, ScalarBGK<SL>> th(d, scoll);
  th.set_geometry([&](Index, Index y, Index) -> ScalarCell {
    return (y == 0 || y == ny - 1) ? ScalarDirichlet : ScalarBulk;
  });
  th.set_wall_values([&](Index, Index y, Index) -> Real {
    return (y == 0) ? Real(0.5) : Real(-0.5);
  });
  const double kx = 2.0 * M_PI / double(nx);
  const Index Hc = H, nyc = ny;
  th.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Index x = px - d.hx, y = py - d.hy;
    if (y <= 0 || y >= nyc - 1) return Real(0);
    const double yy = (double(y) - 0.5) / double(Hc);
    return Real(0.5 - yy + kEps * std::sin(kx * double(x)) * std::sin(M_PI * yy));
  });
  th.finalize_geometry();
  th.compute_field();

  Policy force;
  force.T = th.temperature();
  force.gx = Real(0); force.gy = Real(1); force.gz = Real(0);
  force.rho0 = Real(1); force.T0 = Real(sh.T0);
  if constexpr (requires { force.beta; }) {
    force.beta = Real(G);
  } else {
    force.w = Real(G); force.q = Real(sh.q); force.Tm = Real(sh.Tm);
  }

  FluidCM<Policy> fcoll;
  fcoll.omega = FluidCM<Policy>::omega_from_viscosity(Real(kNu));
  fcoll.omega_bulk = Real(1);
  fcoll.forcing = force;
  FluidSolver<FL, EsotericPull<FL>, FluidCM<Policy>> fl(d, fcoll);
  fl.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });
  fl.initialize(Real(1));
  th.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // Projection onto exp(i k x) sin(pi (y - 1/2) / H): the modulus, so the
  // measurement does not depend on the mode keeping the seed's phase, and
  // x-uniform content -- the acoustic adjustment of the base state and Guo's
  // half-force shift, both functions of y alone -- is rejected exactly.
  auto amplitude = [&]() {
    fl.compute_macroscopic();
    auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    double s = 0, c = 0, n2 = 0;
    for (Index y = 1; y <= H; ++y) {
      const double wy = std::sin(M_PI * (double(y) - 0.5) / double(H));
      for (Index x = 0; x < nx; ++x) {
        const double v = double(hv(d.id(x, y)));
        s  += v * std::sin(kx * double(x)) * wy;
        c  += v * std::cos(kx * double(x)) * wy;
        n2 += std::sin(kx * double(x)) * std::sin(kx * double(x)) * wy * wy;
      }
    }
    return std::hypot(s, c) / n2;
  };

  const double t_diff = double(H) * double(H) / kappa;
  const auto warm = std::size_t(warm_td * t_diff);
  const auto half = std::size_t(0.5 * span_td * t_diff);
  const auto t0 = std::chrono::steady_clock::now();
  auto advance = [&](std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) { fl.step(true); th.step(); }
  };
  advance(warm);
  const double a0 = amplitude();
  advance(half);
  const double a1 = amplitude();
  advance(half);
  const double a2 = amplitude();
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();

  Run r{0, 0, 0, {}, {}, secs, false};
  if (!(a0 > 0) || !(a1 > 0) || !std::isfinite(a2)) return r;
  const double dt_half = double(half) / t_diff;
  r.sigma_a = std::log(a1 / a0) / dt_half;
  r.sigma_b = std::log(a2 / a1) / dt_half;
  r.sigma   = std::log(a2 / a0) / (2.0 * dt_half);

  th.compute_field();
  auto hT = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  for (Index y = 1; y <= H; ++y)
    for (Index x = 0; x < nx; ++x) {
      r.T.push_back(double(hT(d.id(x, y))));
      r.v.push_back(double(hv(d.id(x, y))));
    }
  r.ok = true;
  return r;
}

double max_abs(const std::vector<double>& a) {
  double m = 0;
  for (double x : a) m = std::max(m, std::abs(x));
  return m;
}
double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

struct Onset { double ra_c, err; int runs; bool bracketed, consistent; };

// Zero of sigma(Ra) near the reference: a +/-4 % bracket, widened in 4 % steps
// if both ends share a sign, then one regula-falsi refinement.
Onset onset(Index H, double aspect, double ra_ref, Shape sh) {
  struct P { double ra, s; };
  auto sample = [&](double ra) {
    const Run r = run<DensityAnomalyGuo>(H, aspect, ra, sh, 0.25, 0.5);
    std::printf("      Ra = %10.3f  sigma = %+9.5f  (halves %+9.5f %+9.5f)  %5.1f s\n",
                ra, r.sigma, r.sigma_a, r.sigma_b, r.seconds);
    const bool cons = std::abs(r.sigma_a - r.sigma_b) <= 0.05 + 0.1 * std::abs(r.sigma);
    return std::pair<P, bool>{P{ra, r.ok ? r.sigma : NAN}, r.ok && cons};
  };
  int runs = 0;
  bool consistent = true;
  auto take = [&](double ra) {
    auto [p, c] = sample(ra); ++runs; consistent = consistent && c; return p;
  };
  P lo = take(0.96 * ra_ref), hi = take(1.04 * ra_ref);
  for (int i = 0; i < 5 && lo.s * hi.s > 0; ++i) {
    if (lo.s > 0) lo = take(lo.ra - 0.04 * ra_ref);   // both growing: go lower
    else          hi = take(hi.ra + 0.04 * ra_ref);   // both decaying: go higher
  }
  if (!(lo.s < 0 && hi.s > 0)) return {NAN, NAN, runs, false, consistent};
  const double est = lo.ra - lo.s * (hi.ra - lo.ra) / (hi.s - lo.s);
  const P mid = take(est);
  if (mid.s < 0) lo = mid; else hi = mid;
  const double ra_c = lo.ra - lo.s * (hi.ra - lo.ra) / (hi.s - lo.s);
  return {ra_c, ra_c / ra_ref - 1.0, runs, true, consistent};
}

}  // namespace

int main(int argc, char** argv) {
  bool conv = false;
  Index H = 32;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-conv") conv = true;
    if (a == "-h" && i + 1 < argc) H = std::atoi(argv[++i]);
  }
  int rc = 0;
  Kokkos::initialize(argc, argv);
  {
    std::printf("Density-anomaly buoyancy: water near 4 C (DensityAnomalyGuo)\n");
    std::printf("D3Q27 central moments + D3Q7 BGK, nz = 1, halfway walls, "
                "nu = kappa = %.3f\n", kNu);
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    //--------------------------------------------------------------------------
    // 1. The linear limit. Classical layer at 1.2 Ra_c, H = 16, where the mode
    //    grows by e^(sigma t) and any wrong term in at() would move it.
    //--------------------------------------------------------------------------
    {
      const double Ra = 1.2 * 1707.762;
      const Run rb = run<BoussinesqGuo>(16, 2.0158, Ra, Shape{1, 0, 0}, 0.25, 0.5);
      const Run ra = run<DensityAnomalyGuo>(16, 2.0158, Ra, Shape{1.0, -1.0, 0.0},
                                            0.25, 0.5);
      const double eT = max_diff(rb.T, ra.T) / max_abs(rb.T);
      const double ev = max_diff(rb.v, ra.v) / max_abs(rb.v);
      const double tol = std::is_same_v<Real, float> ? 1e-4 : 1e-11;
      const bool ok = rb.ok && ra.ok && eT < tol && ev < tol && rb.sigma > 0;
      std::printf("  1. q = 1, Tm below every temperature, against BoussinesqGuo "
                  "(H = 16, Ra = 1.2 Ra_c)\n");
      std::printf("     sigma %+.6f vs %+.6f   max rel diff T %.2e  v %.2e  "
                  "(tol %.0e)   %s\n", ra.sigma, rb.sigma, eT, ev, tol,
                  ok ? "PASS" : "FAIL");
      std::printf("     cost: %.2f s against %.2f s for the same steps (x%.2f)\n\n",
                  ra.seconds, rb.seconds, ra.seconds / rb.seconds);
      if (!ok) rc = 1;
    }

    //--------------------------------------------------------------------------
    // 2. The onset against tools/penetrative_onset.py.
    //--------------------------------------------------------------------------
    struct Case { const char* name; double lam, aspect, ra_ref; };
    const Case cases[] = {
      {"lam = 0.5", 0.5, 1.5625, 16764.642181},   // --lam 0.5 --k 4.0212385965949354
      {"lam = 0  ", 0.0, 2.0,     1674.291614},   // --lam 0   --k 3.141592653589793
    };
    const double tol_ra = 0.015;
    std::vector<Index> grids = conv ? std::vector<Index>{16, 32, 64}
                                    : std::vector<Index>{H};
    std::printf("  2. onset at q = %.6f against the shooting reference "
                "(tol %.1f %% at H >= 32)\n", kQ, 100 * tol_ra);
    for (const Case& c : cases) {
      const double Tm = c.lam - 0.5;
      const Shape sh{kQ, Tm, neutral_T0(kQ, Tm)};
      std::printf("   %s  Tm = %+.3f  T0 = %+.5f  k H = %.6f  Ra_ref = %.4f\n",
                  c.name, Tm, sh.T0, 2 * M_PI / c.aspect, c.ra_ref);
      double prev = NAN;
      for (Index h : grids) {
        std::printf("    H = %d, box %d x %d\n", int(h),
                    int(std::lround(c.aspect * double(h))), int(h + 2));
        const Onset o = onset(h, c.aspect, c.ra_ref, sh);
        const bool pass = o.bracketed && (h < 32 || std::abs(o.err) < tol_ra);
        std::printf("    -> Ra_c = %.3f   err %+.3f %%   %d runs   halves %s   %s",
                    o.ra_c, 100 * o.err, o.runs,
                    o.consistent ? "agree" : "DISAGREE", pass ? "PASS" : "FAIL");
        if (std::isfinite(prev) && std::isfinite(o.err) && o.err != 0)
          std::printf("   order %.2f", std::log(std::abs(prev / o.err)) / std::log(2.0));
        std::printf("\n");
        prev = o.err;
        if (!pass) rc = 1;
      }
    }
    std::printf("\n  %s\n\n", rc ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return rc;
}
