//==============================================================================
//  Specular reflection: a free-slip wall, and the sharpest test available for it.
//
//  THE TEST IS AN IDENTITY, NOT A CONVERGENCE RATE. Poiseuille flow in a channel
//  of width 2H, bounce-back on both sides, has a steady state that is exactly
//  mirror-symmetric about its mid-plane. Replace the upper half by a specular
//  wall placed ON that mid-plane and the half-domain must reproduce the lower
//  half of the full solution -- and not to second order, to ROUND-OFF. A mirror
//  boundary is either the mirror or it is not.
//
//  That is a much stronger statement than "the error is small", and it is the
//  reason this case does not simply fit a parabola and check the order. A
//  second-order-accurate free-slip wall that quietly leaked a little momentum
//  would pass a convergence test and fail this one.
//
//  WHERE THE PLANES ARE, since half a cell is the whole question here:
//
//     full   solid row y=0, fluid y=1..2H, solid row y=2H+1   -> ny = 2H+2
//            no-slip planes at y = 0.5 and y = 2H+0.5, width 2H,
//            mid-plane at y = H+0.5
//     half   solid row y=0, fluid y=1..H,  SPEC row y=H+1     -> ny = H+2
//            no-slip plane at y = 0.5, symmetry plane midway between the last
//            fluid node H and the ghost H+1, i.e. y = H+0.5  -- the same plane
//
//  Both the reflecting plane and the no-slip plane therefore sit halfway, which
//  is the pairing this tree's default wall family already uses (\S rb_high_ra).
//
//  SECOND CHECK, against the analytic solution rather than against another run:
//  the half-channel profile is fitted with a parabola and its roots taken. The
//  lower root must land on 0.5 (the bounce-back plane) and the vertex on H+0.5
//  (the symmetry plane). Reporting u at the ghost row instead would be
//  meaningless -- nothing there is a velocity.
//
//  THIRD CHECK: zero net momentum flux through the specular wall. A free-slip
//  wall must pass no mass and no normal momentum, so the total mass of the half
//  domain is conserved exactly, the way a closed box is. Regularised walls are
//  NOT mass conserving (CLAUDE.md records the measurement); specular is, because
//  it is a permutation of the populations and a permutation cannot change their
//  sum.
//
//    usage: specular [-h H] [-tau T] [-lat 2d|3d]
//==============================================================================
#include "collision/BGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

template <class L>
using Coll = BGK<L, SecondOrderEquilibrium<L>, Guo, ShiftedPopulations>;

//------------------------------------------------------------------------------
//  Run a channel. `spec` closes the top with a specular wall at y = ny-1 and
//  makes the domain the LOWER HALF; otherwise both sides are bounce-back.
//  Returns u_x down the column, index 0 .. ny-1.
//------------------------------------------------------------------------------
template <class L>
static std::vector<double> run(Index H, Real tau, bool spec, std::size_t steps,
                               double& mass_drift) {
  const Index ny = spec ? (H + 2) : (2 * H + 2);
  const Index nx = 6, nz = (L::D == 3) ? 6 : 1;
  const Real rho0 = Real(1);
  const Real nu   = (tau - Real(0.5)) * cs2<L, Real>();
  const Real umax = Real(0.02);
  // The FULL channel's driving force, in both cases: the half domain must solve
  // the same physical problem, not a rescaled one.
  const Real G = Real(8) * rho0 * nu * umax / Real(4 * H * H);

  Domain d(nx, ny, nz, true, false, true);
  Coll<L> coll;
  coll.omega = Real(1) / tau;
  coll.forcing = Guo{G, Real(0), Real(0)};
  FluidSolver<L, EsotericPull<L>, Coll<L>> s(d, coll);
  s.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });
  if (spec)
    s.set_specular_walls([&](Index, Index y, Index) -> std::uint8_t {
      return (y == ny - 1) ? NrmYp : NrmNone;
    });
  s.initialize(rho0);

  const Real m0 = s.total_mass();
  for (std::size_t t = 0; t < steps; ++t) s.step();
  s.compute_macroscopic();
  mass_drift = double(s.total_mass() - m0) / double(m0);

  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  std::vector<double> u(static_cast<std::size_t>(ny), 0.0);
  for (Index y = 0; y < ny; ++y) u[std::size_t(y)] = double(h(d.id(nx / 2, y, nz / 2)));
  return u;
}

template <class L>
static int check(Index H, Real tau, std::size_t steps) {
  double dm_full = 0, dm_half = 0;
  const std::vector<double> uf = run<L>(H, tau, false, steps, dm_full);
  const std::vector<double> uh = run<L>(H, tau, true,  steps, dm_half);

  // ---- the identity ------------------------------------------------------
  double worst = 0.0, scale = 0.0;
  Index yworst = 0;
  for (Index y = 1; y <= H; ++y) {
    const double e = std::abs(uh[std::size_t(y)] - uf[std::size_t(y)]);
    if (e > worst) { worst = e; yworst = y; }
    scale = std::max(scale, std::abs(uf[std::size_t(y)]));
  }
  const double rel = scale > 0 ? worst / scale : 1.0;

  // ---- where the planes actually are, from the half-domain profile --------
  // u = a + b y + c y^2 over the fluid rows; lower root = no-slip plane,
  // vertex = symmetry plane.
  double S[3][4] = {};
  for (Index y = 1; y <= H; ++y) {
    const double p[3] = {1.0, double(y), double(y) * double(y)};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) S[i][j] += p[i] * p[j];
      S[i][3] += p[i] * uh[std::size_t(y)];
    }
  }
  for (int c = 0; c < 3; ++c) {
    int piv = c;
    for (int r = c; r < 3; ++r) if (std::abs(S[r][c]) > std::abs(S[piv][c])) piv = r;
    for (int j = 0; j < 4; ++j) std::swap(S[c][j], S[piv][j]);
    const double dd = S[c][c];
    for (int j = 0; j < 4; ++j) S[c][j] /= dd;
    for (int r = 0; r < 3; ++r) if (r != c) {
      const double f = S[r][c];
      for (int j = 0; j < 4; ++j) S[r][j] -= f * S[c][j];
    }
  }
  const double a = S[0][3], b = S[1][3], c2 = S[2][3];
  const double vertex = -b / (2 * c2);
  const double disc = b * b - 4 * c2 * a;
  const double root = disc >= 0 ? (-b + std::sqrt(disc)) / (2 * c2) : std::nan("");
  const double lo = std::min(root, (-b - std::sqrt(disc)) / (2 * c2));

  const bool ok_id   = rel < 1e-12;
  const bool ok_slip = std::abs(lo - 0.5) < 0.02;
  const bool ok_sym  = std::abs(vertex - (double(H) + 0.5)) < 0.02;
  const bool ok_mass = std::abs(dm_half) < 1e-13;

  std::printf("  %-6s H = %3lld  tau = %.2f\n", L::name, (long long)H, double(tau));
  std::printf("    half vs full lower half : worst %.3e rel (at y = %lld)   %s\n",
              rel, (long long)yworst, ok_id ? "ok" : "FAIL");
  std::printf("    no-slip plane           : %8.4f   want 0.5            %s\n",
              lo, ok_slip ? "ok" : "FAIL");
  std::printf("    symmetry plane          : %8.4f   want %.1f           %s\n",
              vertex, double(H) + 0.5, ok_sym ? "ok" : "FAIL");
  std::printf("    mass drift, half domain : %+.3e                       %s\n",
              dm_half, ok_mass ? "ok" : "FAIL");
  std::printf("    (full-domain drift %+.3e, u_max %.6f)\n\n",
              dm_full, uf[std::size_t(H)]);
  return (ok_id && ok_slip && ok_sym && ok_mass) ? 0 : 1;
}

int main(int argc, char** argv) {
  Index H = 16;
  Real tau = Real(0.8);
  std::size_t steps = 40000;
  bool three_d = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h"     && i + 1 < argc) H     = Index(std::atol(argv[++i]));
    if (a == "-tau"   && i + 1 < argc) tau   = Real(std::atof(argv[++i]));
    if (a == "-steps" && i + 1 < argc) steps = std::size_t(std::atol(argv[++i]));
    if (a == "-lat"   && i + 1 < argc) three_d = (std::string(argv[++i]) == "3d");
  }
  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    std::printf("Specular reflection -- free-slip wall against an exact mirror\n\n");
    if (three_d) {
      fails += check<D3Q27>(H, tau, steps);
    } else {
      fails += check<D2Q9>(H, tau, steps);
      fails += check<D2Q9>(H, Real(1.2), steps);
      fails += check<D3Q27>(H, tau, steps);
    }
    std::printf("  %s\n", fails ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return fails ? 1 : 0;
}
