//==============================================================================
//  The on-node zero-flux scalar wall, checked as an IDENTITY.
//
//  ScalarSpecular (src/boundary/Specular.hpp, mirror_unknowns) claims to put a
//  mirror plane exactly ON a node. If that is true then a box of width M closed
//  by two of them is not merely similar to the periodic box of width 2M whose
//  mirror it is -- it is the SAME PROBLEM, and the two must agree to round-off
//  rather than to a tolerance. A convergence rate would not test this: a wall
//  that is nearly a mirror also converges.
//
//  The construction. Nodes 0..M of a periodic box of width 2M, with everything
//  even about x = 0 and about x = M -- both of which are NODES, which is what
//  "on-node" means:
//
//      psi = A sin(pi x/M) cos(2 pi y/ny)   odd about both planes
//      u = (d_y psi, -d_x psi)              so u_x is odd, u_y even, div u = 0
//      q(0) = 1 + a cos(pi x/M) cos(2 pi y/ny) + b cos(2 pi x/M)   even
//
//  Then q must stay even for all time, so the half box reproduces the full one
//  node for node. The velocity is deliberately NOT solenoidal: this tests the
//  boundary, and a scalar does not care.
//
//  ===================== WHY THE HARD CASE IS omega -> 2 =====================
//  This wall exists because the two conditions the tree already had both fail
//  in validation/ehd_cavity.cpp, and both failures are about omega_q -> 2:
//
//    ScalarAdiabatic is bounce-back, which reverses ALL components. At
//    omega -> 2 that re-injects an odd-even mode every step and it never damps
//    (CLAUDE.md's ringing entry). Measured in the cavity: NON-FINITE inside
//    0.05 t0 at both N = 41 and N = 81. It also reports ZERO at its own node.
//
//    ScalarOutflow is on-node and reports the real value, but it is an OPEN
//    boundary -- it prescribes the node from a donor -- so it neither conserves
//    nor reflects. Measured in the cavity: q reached -0.083 q0 at N = 41 and
//    -0.279 at N = 81, i.e. WORSE under refinement, and by N = 81 the
//    hydrostatic reference itself was 158 % wrong.
//
//  So the case is run at an omega chosen to sit where they fail, not at a
//  comfortable one: `-omega` defaults to 1.999.
//
//  ===================== WHAT IT DOES NOT TEST ===============================
//  Axis-aligned normals only -- there are no specular corners, and the setter
//  aborts on a non-axis normal rather than mirroring about one of two. Nothing
//  here tests a moving mirror plane or a specular condition on a curved wall.
//
//    usage: scalar_specular [-m M] [-ny N] [-steps K] [-omega W] [-tol E]
//==============================================================================
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ScalarSolver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

struct Cfg {
  Index M = 24, ny = 20;
  std::size_t steps = 400;
  double omega = 1.99, tol = 1e-12;
  double U = 0.05, a = 0.5, b = 0.3;
};

//------------------------------------------------------------------------------
//  One run. `half` closes x at both ends with the on-node mirror; otherwise the
//  box is the full periodic one of width 2M. Returns q on nodes 0..M.
//------------------------------------------------------------------------------
// bc: 0 = ScalarSpecular (the thing under test), 1 = ScalarAdiabatic (the
// control -- see check()).
template <class L>
static std::vector<double> run(const Cfg& c, bool half, int bc = 0) {
  const Index nx = half ? c.M + 1 : 2 * c.M;
  const Index ny = c.ny;
  Domain d(nx, ny, 1, /*periodic x*/ !half, /*y*/ true, /*z*/ true);

  using Coll = ScalarBGK<L>;
  using Sol  = ScalarSolver<L, EsotericPull<L>, Coll>;
  Coll coll;  coll.omega = Real(c.omega);  coll.T_ref = Real(0);
  Sol s(d, coll);
  s.set_geometry([&](Index x, Index, Index) -> ScalarCell {
    if (half && (x == 0 || x == nx - 1))
      return bc ? ScalarAdiabatic : ScalarSpecular;
    return ScalarBulk;
  });
  if (half && !bc)
    s.set_specular_walls([&](Index x, Index, Index) -> std::uint8_t {
      if (x == 0)      return NrmXm;
      if (x == nx - 1) return NrmXp;
      return NrmNone;
    });

  View1D<Real> ux("ux", d.n_padded), uy("uy", d.n_padded), uz("uz", d.n_padded);
  const double Mm = double(c.M), nyd = double(ny), U = c.U;
  Kokkos::parallel_for("uv", Range(0, d.n_padded), KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const double x = double(px - d.hx), y = double(py - d.hy);
    // From the streamfunction psi = A sin(pi x/M) cos(2 pi y/ny), so the field
    // is SOLENOIDAL and the scalar is transported without a source. A velocity
    // with div u != 0 pumps it, and a reference solution that grows by two
    // orders of magnitude is a fragile thing to regress against even when the
    // identity itself still holds. psi is odd about x = 0 and about x = M, so
    // u_x is odd (no flux through either plane) and u_y is even.
    ux(n) = Real(-U * Kokkos::sin(M_PI * x / Mm) * Kokkos::sin(2.0 * M_PI * y / nyd));
    uy(n) = Real(-U * (nyd / (2.0 * Mm)) * Kokkos::cos(M_PI * x / Mm)
                    * Kokkos::cos(2.0 * M_PI * y / nyd));
    uz(n) = Real(0);
  });
  s.set_velocity(ux, uy, uz);

  const double aa = c.a, bb = c.b;
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const double x = double(px - d.hx), y = double(py - d.hy);
    return Real(1.0 + aa * Kokkos::cos(M_PI * x / Mm) * Kokkos::cos(2.0 * M_PI * y / nyd)
                    + bb * Kokkos::cos(2.0 * M_PI * x / Mm));
  });
  s.finalize_geometry();
  s.compute_field();
  for (std::size_t t = 0; t < c.steps; ++t) s.step();
  s.compute_field();

  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());
  std::vector<double> out(static_cast<std::size_t>((c.M + 1) * ny));
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x <= c.M; ++x)
      out[std::size_t(y) * std::size_t(c.M + 1) + std::size_t(x)] =
          double(h(d.id(x, y, 0)));
  return out;
}

template <class L>
static int check(const Cfg& c, const char* name) {
  const std::vector<double> full = run<L>(c, false);
  const std::vector<double> hlf  = run<L>(c, true);
  double worst = 0.0, scale = 0.0, wall_lo = 1e300, wall_hi = -1e300;
  for (std::size_t i = 0; i < full.size(); ++i) {
    worst = std::max(worst, std::abs(full[i] - hlf[i]));
    scale = std::max(scale, std::abs(full[i]));
  }
  // DOES THIS TEST HAVE TEETH? The same comparison with the wall the tree
  // already had. If bounce-back also reproduced the mirrored box to round-off
  // then the identity above would be measuring nothing, and the new boundary
  // would not have been needed. Printed, never scored.
  double lo = 1e300, hi = -1e300;
  for (double v : full) { lo = std::min(lo, v); hi = std::max(hi, v); }
  const std::vector<double> ad = run<L>(c, true, /*bc*/1);
  double worst_ad = 0.0;
  for (std::size_t i = 0; i < full.size(); ++i)
    worst_ad = std::max(worst_ad, std::abs(full[i] - ad[i]));
  // The wall node must carry a REAL value. ScalarAdiabatic reports exactly zero
  // at its own node (CLAUDE.md lists that trap), and a case that differentiates
  // or integrates the field across the wall column cannot use such a node.
  for (Index y = 0; y < c.ny; ++y) {
    const double v = hlf[std::size_t(y) * std::size_t(c.M + 1)];
    wall_lo = std::min(wall_lo, v);  wall_hi = std::max(wall_hi, v);
  }
  const double rel = worst / scale;
  // "Reports a real value", not "reports a positive one": this scalar is
  // signed. The trap being guarded is a node that is STRUCTURALLY zero.
  const double wall_mag = std::max(std::abs(wall_lo), std::abs(wall_hi));
  // AND THE REFERENCE MUST STILL CARRY STRUCTURE. Two nearly-uniform fields
  // agree to round-off whatever the wall does, so a run whose initial
  // pattern has diffused away would pass this test vacuously. omega = 1 is
  // diffusive enough to do exactly that in 400 steps, which is why the
  // control cases below are shorter.
  const bool ok = std::isfinite(rel) && rel < c.tol && wall_mag > 0.05
                  && (hi - lo) > 0.05;
  std::printf("  %-6s  half box %lld x %lld vs periodic %lld x %lld,"
              " omega = %.4f\n", name, (long long)(c.M + 1), (long long)c.ny,
              (long long)(2 * c.M), (long long)c.ny, c.omega);
  std::printf("          max |q_half - q_full| = %.3e  (relative %.3e,"
              " tolerance %.0e)   %s\n", worst, rel, c.tol, ok ? "ok" : "FAIL");
  std::printf("          value at the mirror node in [%.4f, %.4f]  (an adiabatic"
              " node would report 0)\n", wall_lo, wall_hi);
  std::printf("          field range %.4f .. %.4f (span %.3f, must exceed 0.05);  CONTROL, ScalarAdiabatic in\n          the same half box is out by %.3e -- the field itself, because\n          such a node reports zero\n\n", lo, hi, hi - lo, worst_ad);
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  Cfg c;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if      (a == "-m"     && i + 1 < argc) c.M     = Index(std::atol(argv[++i]));
    else if (a == "-ny"    && i + 1 < argc) c.ny    = Index(std::atol(argv[++i]));
    else if (a == "-steps" && i + 1 < argc) c.steps = std::size_t(std::atol(argv[++i]));
    else if (a == "-omega" && i + 1 < argc) c.omega = std::atof(argv[++i]);
    else if (a == "-tol"   && i + 1 < argc) c.tol   = std::atof(argv[++i]);
  }

  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    std::printf("On-node zero-flux scalar wall (ScalarSpecular) -- identity"
                " against the box it mirrors\n");
    std::printf("  %s   %zu steps at omega = %.3f, %zu at the omega = 1"
                " control\n\n", sizeof(Real) == 4 ? "FP32" : "FP64", c.steps,
                c.omega, c.steps / 10);
    fails += check<D2Q5>(c, "D2Q5");
    fails += check<D2Q9>(c, "D2Q9");
    // omega = 1 is the easy case and is kept as a control: if the hard one
    // fails and this passes, the fault is the ghost modes and not the mirror.
    Cfg e = c;  e.omega = 1.0;  e.steps = c.steps / 10;
    fails += check<D2Q5>(e, "D2Q5");
    fails += check<D3Q7>(e, "D3Q7");
    std::printf("  %s\n", fails ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return fails;
}
