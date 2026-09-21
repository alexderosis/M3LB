//==============================================================================
//  THERMOCAPILLARY (MARANGONI) SHEAR ON A FLAT FREE SURFACE, against the exact
//  parallel-flow solution.
//
//  This is STAGE 1 of a coupled melt-pool model, and it exists so that the
//  Marangoni term can be wrong HERE, against an exact answer, rather than in a
//  melt pool where nothing can check it. validation/melt_pool.cpp's banner
//  names Marangoni convection as its largest omission; adding it there first
//  would have produced a number with no reference and a picture with no claim.
//
//  THE PROBLEM. An incompressible layer of depth h on a no-slip floor, with a
//  flat, non-deformable free surface carrying a uniform tangential stress
//  tau = (dgamma/dT)(dT/dx). Steady, parallel (u = u(y) x-hat), zero net flux
//  through any cross-section -- the flux condition is what a closed pool
//  imposes, and it is what makes the surface flow return along the floor.
//
//  Solving mu u'' = dp/dx with u(0) = 0, mu u'(h) = tau and int_0^h u dy = 0
//  gives, derived with sympy rather than transcribed:
//
//      u(y) = (tau h / mu) [ (3/4) s^2 - (1/2) s ],      s = y/h
//      u(h) = tau h / (4 mu)                  surface, coefficient 1/4
//      u = 0 at s = 2/3                       interior zero crossing
//      mu u'(0) = -tau/2                      FLOOR CARRIES HALF THE SURFACE
//                                             SHEAR, OPPOSITE IN SIGN
//      u_min = -(1/12) tau h / mu at s = 1/3  return-flow extremum
//      dp/dx = 3 tau / (2 h)                  falls OUT of the flux condition,
//                                             it is not an extra assumption
//
//  FOUR INDEPENDENT NUMBERS, NOT ONE, AND THAT IS THE POINT. A wrong Guo
//  prefactor scales the whole profile and is caught by the surface coefficient;
//  a wrong half-shift or a wrong wall plane distorts the SHAPE and is caught by
//  the zero crossing and the extremum location, which are pure numbers with no
//  tau, h or mu in them at all. A validation with one check cannot separate
//  those two failures.
//
//  WHY PERIODIC IN x RATHER THAN A CLOSED CAVITY. The classical thermocapillary
//  benchmark is a shallow closed cavity, where this profile is the CORE
//  solution and holds only far from the end walls -- so the comparison carries
//  an aspect-ratio error that has to be estimated and subtracted. Periodic x
//  removes the end walls entirely, and the parallel-flow solution becomes
//  EXACT rather than asymptotic. The cost is that the return pressure gradient
//  can no longer develop on its own (a periodic box cannot sustain a mean
//  pressure gradient), so it is imposed as the uniform body force the
//  derivation gives. That is not circular: dp/dx and tau are INPUTS, and the
//  four numbers above are OUTPUTS that the scheme has to reproduce. The net
//  flux is then a consistency check on the two forces being correctly
//  normalised against each other, and it is asserted.
//
//  THE SURFACE STRESS IS APPLIED AS A BODY FORCE IN THE TOP ROW, tau/dy over a
//  cell of thickness dy -- the standard device for a flat free surface. The
//  analytic solution puts the stress ON the plane y = h; the discrete one
//  spreads it over [h-dy, h], the same trade validation/melt_pool.cpp makes for
//  its surface heat flux.
//
//  MEASURED 2026-09-21, H = 16/32/64/128, tau_lb = 0.8, surface u held at 0.01
//  so the Mach number does not vary down the ladder. AN EARLIER VERSION OF THIS
//  BANNER PREDICTED ORDER 1 AND SAID THAT ORDER 2 WOULD MEAN THE FORCE WAS IN
//  THE WRONG PLACE. THE MEASUREMENT IS 1.5, AND IT IS NEITHER:
//
//      quantity                              order      what it is
//      surface u, floor shear              1.00-1.07    FIRST  order
//      zero crossing, return extremum      1.87-2.07    SECOND order
//      relative L2 over the whole profile  1.49-1.50    3/2
//
//  The split is the answer and it is sharper than a single number would have
//  been. The first-order error is CONFINED TO THE SURFACE LAYER, where the
//  force is smeared; the interior is clean second order, which is what the
//  zero crossing and the extremum -- pure numbers carrying no tau, h or mu --
//  establish. An O(dy) error living in a layer one cell thick contributes
//  O(dy) * sqrt(dy/h) = O(dy^3/2) to an L2 normalised over the whole depth, so
//  3/2 is the predicted exponent for exactly this error structure and 1.49 is
//  what came out.
//
//  What that buys for the melt pool: the Marangoni term's error does NOT
//  pollute the interior, so a pool dimension read off an isotherm away from the
//  surface inherits second order, while the surface velocity itself does not.
//  It also means a second-order surface treatment is the one worthwhile
//  refinement here, and where it would pay.
//
//  THE TWO PLANES MUST AGREE, AND THIS IS THE TRAP THE CASE IS SIZED AROUND.
//  The no-slip floor is halfway bounce-back, plane at y = 0.5; the free surface
//  is `set_specular_walls` (SpecWall), a GHOST whose mirror plane sits half a
//  cell outside the last fluid node, at y = H + 0.5. Both halfway, so the fluid
//  depth is exactly h = H and node y sits at s = (y - 0.5)/H. `set_specular_nodes`
//  (SpecNode) would put the surface ON the node instead and silently shift the
//  depth by half a cell -- CLAUDE.md records that picking the wrong one of the
//  two is silent. Halfway is also what the SCALAR uses (its implicit
//  bounce-back zero-flux plane is the top face of the last cell), so when this
//  term reaches validation/melt_pool.cpp the fluid and thermal surfaces will
//  coincide. They would not with SpecNode.
//
//  WHAT THIS DOES NOT DO. It does not test the map from a temperature field to
//  tau -- tau is prescribed here, deliberately, so that a failure is in the
//  momentum response and nowhere else. It does not test a deformable surface;
//  the surface is flat by construction and stays flat. And it carries no
//  material properties at all: it is dimensionless, so nothing about Ti-6Al-4V
//  can be read off it.
//==============================================================================
#include "collision/BGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

namespace {

int checks = 0, failures = 0;

void verdict(const char* what, double got, double want, double tol) {
  ++checks;
  const bool ok = std::abs(got - want) <= tol;
  if (!ok) ++failures;
  std::printf("    %-46s %12.6f  (want %10.6f +/- %.6f)  %s\n",
              what, got, want, tol, ok ? "PASS" : "FAIL");
}

template <class L>
using Coll = BGK<L, SecondOrderEquilibrium<L>, FieldGuo, ShiftedPopulations>;

struct Profile {
  std::vector<double> u;      // u_x at each row, index 0 .. ny-1
  double mass_drift = 0;
};

//------------------------------------------------------------------------------
//  One run. h = H in lattice units; fluid rows y = 1..H with the no-slip plane
//  at y = 0.5 and the mirror plane at y = H + 0.5.
//------------------------------------------------------------------------------
template <class L>
Profile run(Index H, Real tau_lb, double tau_s, std::size_t steps) {
  const Index nx = 6, ny = H + 2, nz = (L::D == 3) ? 1 : 1;
  const Real rho0 = Real(1);
  const Real nu   = (tau_lb - Real(0.5)) * cs2<L, Real>();

  Domain d(nx, ny, nz, true, false, true);

  // The surface stress lives in the top fluid row only, as tau/dy with dy = 1.
  // The uniform part is the return pressure gradient -dp/dx = -3 tau / (2 h),
  // which the derivation gives and a periodic box cannot generate for itself.
  View1D<Real> Fx("Fx", d.n_padded);
  auto hFx = Kokkos::create_mirror_view(Fx);
  for (Index n = 0; n < Index(d.n_padded); ++n) hFx(n) = Real(0);
  for (Index x = 0; x < nx; ++x)
    for (Index z = 0; z < nz; ++z) hFx(d.id(x, H, z)) = Real(tau_s);
  Kokkos::deep_copy(Fx, hFx);

  Coll<L> coll;
  coll.omega = Real(1) / tau_lb;
  coll.forcing = FieldGuo{};
  coll.forcing.Ex = Fx;
  coll.forcing.fx = Real(-1.5 * tau_s / double(H));

  FluidSolver<L, EsotericPull<L>, Coll<L>> s(d, coll);
  s.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });
  s.set_specular_walls([&](Index, Index y, Index) -> std::uint8_t {
    return (y == ny - 1) ? NrmYp : NrmNone;
  });
  s.initialize(rho0);

  const Real m0 = s.total_mass();
  for (std::size_t t = 0; t < steps; ++t) s.step();
  s.compute_macroscopic();

  Profile p;
  p.mass_drift = double(s.total_mass() - m0) / double(m0);
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  p.u.assign(std::size_t(ny), 0.0);
  for (Index y = 0; y < ny; ++y)
    p.u[std::size_t(y)] = double(hu(d.id(nx / 2, y, nz / 2)));
  (void)nu;
  return p;
}

// The exact profile, in the same units: u(s) = (tau h / mu) (3/4 s^2 - 1/2 s).
double u_exact(double s, double tau_s, double H, double mu) {
  return (tau_s * H / mu) * (0.75 * s * s - 0.5 * s);
}

}  // namespace

//==============================================================================
int main(int argc, char** argv) {
  // THREE RUNGS BY DEFAULT, because this case is in the ctest list and the
  // H = 128 rung is 88 % of the wall clock (983040 steps against 245760) for
  // one extra order estimate. Three rungs still give two, and they agree to
  // 0.00. `-full` adds it back; the banner's table is the four-rung run.
  std::vector<Index> Hs = {16, 32, 64};
  Real  tau_lb = Real(0.8);
  double u_target = 0.01;          // surface velocity, sets the Mach number
  double factor = 6.0;             // run length in momentum-diffusion times

  for (int i = 1; i < argc; ++i) {
    auto next = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-tau"))  { double v; next(v); tau_lb = Real(v); }
    else if (!std::strcmp(argv[i], "-u"))    next(u_target);
    else if (!std::strcmp(argv[i], "-t"))    next(factor);
    else if (!std::strcmp(argv[i], "-full"))  Hs.push_back(128);
    else std::fprintf(stderr, "marangoni: unknown option %s\n", argv[i]);
  }

  Kokkos::initialize(argc, argv);
  {
    using L = D3Q27;
    const double nu = (double(tau_lb) - 0.5) * double(cs2<L, Real>());
    const double mu = 1.0 * nu;                    // rho0 = 1

    std::printf("thermocapillary shear on a flat free surface, against the exact\n"
                "parallel-flow solution  u/(tau h/mu) = 3/4 s^2 - 1/2 s\n");
    std::printf("  %s   tau_lb = %.3f   nu = %.6f   mu = %.6f   "
                "surface u target = %.4f\n\n",
                L::name, double(tau_lb), nu, mu, u_target);

    std::vector<double> errs;
    for (Index H : Hs) {
      // tau chosen per rung so the SURFACE VELOCITY is the same at every H:
      // holding tau fixed instead would change the Mach number down the ladder
      // and the ladder would measure compressibility as well as the wall.
      const double tau_s = 4.0 * mu * u_target / double(H);
      const std::size_t steps =
          std::size_t(factor * double(H) * double(H) / nu);

      const Profile p = run<L>(H, tau_lb, tau_s, steps);

      // ---- the four numbers -------------------------------------------------
      // Rows y = 1..H sit at s = (y - 0.5)/H.
      const double scale = tau_s * double(H) / mu;
      double l2n = 0, l2d = 0;
      for (Index y = 1; y <= H; ++y) {
        const double s = (double(y) - 0.5) / double(H);
        const double e = p.u[std::size_t(y)] - u_exact(s, tau_s, double(H), mu);
        l2n += e * e;
        l2d += u_exact(s, tau_s, double(H), mu) * u_exact(s, tau_s, double(H), mu);
      }
      const double l2 = std::sqrt(l2n / l2d);
      errs.push_back(l2);

      // surface: extrapolate the last two rows to the mirror plane s = 1
      const double uS = 1.5 * p.u[std::size_t(H)] - 0.5 * p.u[std::size_t(H - 1)];
      // floor shear: one-sided from the no-slip plane (u = 0 at s = 0)
      const double dudy0 = (p.u[std::size_t(1)] - 0.0) / 0.5;   // over half a cell
      // interior zero crossing, linearly interpolated
      double s0 = 0;
      for (Index y = 1; y < H; ++y)
        if (p.u[std::size_t(y)] < 0 && p.u[std::size_t(y + 1)] >= 0) {
          const double a = p.u[std::size_t(y)], b = p.u[std::size_t(y + 1)];
          s0 = ((double(y) - 0.5) + a / (a - b)) / double(H);
          break;
        }
      // return-flow extremum
      double umin = 0; double smin = 0;
      for (Index y = 1; y <= H; ++y)
        if (p.u[std::size_t(y)] < umin) {
          umin = p.u[std::size_t(y)]; smin = (double(y) - 0.5) / double(H);
        }
      // net flux, as a fraction of the surface-velocity scale
      double flux = 0;
      for (Index y = 1; y <= H; ++y) flux += p.u[std::size_t(y)];
      flux /= (double(H) * std::abs(scale));

      std::printf("  H = %3lld   steps = %8zu   mass drift %+.2e\n",
                  (long long)H, steps, p.mass_drift);
      verdict("surface u / (tau h / mu)          [1/4]", uS / scale, 0.25, 0.02);
      verdict("interior zero crossing s          [2/3]", s0, 2.0 / 3.0, 0.03);
      verdict("floor shear mu u'(0) / tau       [-1/2]", mu * dudy0 / tau_s, -0.5, 0.05);
      verdict("return extremum u_min/(tau h/mu)[-1/12]", umin / scale, -1.0 / 12.0, 0.01);
      verdict("extremum location s               [1/3]", smin, 1.0 / 3.0, 0.05);
      verdict("net flux (should vanish)            [0]", flux, 0.0, 2e-3);
      std::printf("    relative L2 of the whole profile        %12.6f %%\n\n",
                  100.0 * l2);
    }

    std::printf("  convergence of the profile L2:\n");
    for (std::size_t i = 1; i < errs.size(); ++i)
      std::printf("    H %3lld -> %3lld   ratio %6.3f   order %5.2f\n",
                  (long long)Hs[i - 1], (long long)Hs[i],
                  errs[i - 1] / errs[i], std::log2(errs[i - 1] / errs[i]));
    std::printf("  3/2 is the expected answer, and it is a MIXTURE rather than a\n"
                "  rate: the surface velocity and the floor shear converge at 1,\n"
                "  the zero crossing and the extremum at 2. An O(dy) error in a\n"
                "  layer one cell thick contributes O(dy^3/2) to an L2 taken over\n"
                "  the whole depth. Read the per-quantity rows above, not this\n"
                "  single number -- it is the one that hides the structure.\n");

    std::printf("\n[marangoni] %d checks, %d failures\n", checks, failures);
  }
  Kokkos::finalize();
  return failures == 0 ? 0 : 1;
}
