//==============================================================================
//  The ON-NODE specular fluid wall (SpecNode), checked as an IDENTITY.
//
//  The tree already had a free-slip fluid wall, SpecWall, and it is exact --
//  validation/specular.cpp measures its plane at 16.5000 to 5.5e-14. But it is
//  a GHOST cell, so that plane sits HALF A CELL outside the last fluid node,
//  while the scalar's zero-flux wall (ScalarSpecular) is ON-NODE, because the
//  Dirichlet family it must pair with (ScalarMoment) is on-node for the reason
//  CLAUDE.md's half-cell entry gives. The two were therefore exact about planes
//  half a cell APART, which is the gap that entry names and this file closes.
//
//  SpecNode mirrors the UNKNOWN directions and then COLLIDES, exactly as
//  ScalarSpecular does. So the claim to test is the same one, and it is an
//  identity rather than a rate: a box closed by two on-node mirrors is not
//  merely similar to the periodic box of twice the width whose mirror it is --
//  it is THE SAME PROBLEM, node for node, to round-off. A wall that is only
//  nearly a mirror still converges, so a convergence rate would not see the
//  difference.
//
//  ================= THREE CHECKS, AND WHY THE THIRD EXISTS =================
//  1. HALF BOX, 2-D. Nodes 0..M of a periodic box of width 2M, everything even
//     about x = 0 and about x = M -- both NODES, which is what "on-node" means.
//     From a streamfunction psi = A sin(pi x/M) cos(2 pi y/ny) so the field is
//     solenoidal: u_x is odd about both planes (no flux through either), u_y
//     even, and the density perturbation even. The half box must reproduce the
//     full one to round-off.
//
//  2. QUARTER BOX, 3-D. The same construction mirrored in x AND z, so the four
//     edge lines carry a TWO-FACE mask and the mirror is a composition. This is
//     the case SpecWall explicitly refuses ("two specular walls meeting at an
//     edge are not handled") and the case an on-node wall cannot refuse: in a
//     closed box the edge node is a real fluid node and there is nowhere else
//     to put it.
//
//  3. THE MOMENT IDENTITIES, on a random population vector. The identity tests
//     above would also pass for a wall that happened to be symmetric but was
//     not zero-flux -- symmetry of the SETUP can hide it. So the mirror is also
//     checked directly for what it is supposed to impose:
//         sum_i f_i c_n     = 0   on every masked axis   (no normal flux)
//         sum_i f_i c_n c_t = 0   for each masked normal (no tangential stress,
//                                 i.e. du_t/dn = 0)
//     and rho left FREE, which is what distinguishes a mirror from bounce-back.
//     At a two-face edge the momentum ALONG the edge line survives, and that is
//     correct rather than a leak: the edge is the intersection of two symmetry
//     planes and flow along it is constrained by neither. The test asserts that
//     it survives, so a mirror that quietly became bounce-back would fail here
//     even though it would still pass checks 1 and 2 on a symmetric setup.
//
//  ================= WHAT THIS DOES NOT TEST ================================
//  Nothing here says SpecNode is BETTER than SpecWall -- they are exact about
//  different planes and each is right for the wall family it pairs with. There
//  is no moving mirror, no oblique plane (the setter rejects a mask with both
//  faces of one axis, and there is no non-axis mask to give it), and no
//  measurement of the SpecNode/ScalarSpecular pair in a running EHD case; that
//  belongs in ehd_electroconvection, not here.
//
//    usage: specular_node [-m M] [-ny N] [-p P] [-steps K] [-tau T] [-tol E]
//==============================================================================
#include "boundary/Specular.hpp"
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
using Coll = BGK<L, SecondOrderEquilibrium<L>, NoForcing, ShiftedPopulations>;

struct Cfg {
  Index M = 16, ny = 20, P = 12;
  std::size_t steps = 300;
  // tau = 0.6, not 0.8. At 0.8 the seeded mode decays by e^-4 over 300 steps
  // and the reference arrives at |u| = 6e-4, which is close enough to nothing
  // that the identity would pass vacuously -- the structure guard in check2d()
  // caught exactly that. Lower nu keeps the field alive for the whole run.
  double tau = 0.6, tol = 1e-12;
  double U = 0.04, a = 0.02;
};

//==============================================================================
//  1. THE HALF BOX, 2-D.
//==============================================================================
// mode: 0 = periodic reference, 1 = half box closed by SpecNode (the thing
// under test), 2 = half box closed by the tree's existing HALFWAY mirror
// SpecWall, which is the control -- see check2d().
template <class L>
static std::vector<double> run2d(const Cfg& c, int mode, std::vector<double>* rho_out) {
  const bool half = mode != 0;
  const Index nx = half ? c.M + 1 : 2 * c.M;
  const Index ny = c.ny, nz = 1;
  Domain d(nx, ny, nz, /*periodic x*/ !half, /*y*/ true, /*z*/ true);

  Coll<L> coll;
  coll.omega = Real(1.0 / c.tau);
  FluidSolver<L, EsotericPull<L>, Coll<L>> s(d, coll);
  s.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  if (mode == 1)
    s.set_specular_nodes([&](Index x, Index, Index) -> std::uint8_t {
      if (x == 0)      return SpecXm;
      if (x == nx - 1) return SpecXp;
      return SpecNone;
    });
  else if (mode == 2)
    s.set_specular_walls([&](Index x, Index, Index) -> std::uint8_t {
      if (x == 0)      return NrmXm;
      if (x == nx - 1) return NrmXp;
      return NrmNone;
    });

  const double Mm = double(c.M), nyd = double(ny), U = c.U, aa = c.a;
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const double x = double(px - d.hx), y = double(py - d.hy);
    // psi = A sin(pi x/M) cos(2 pi y/ny): solenoidal, u_x odd about x = 0 AND
    // about x = M, u_y even about both. Both planes are nodes.
    const double ux = -U * Kokkos::sin(M_PI * x / Mm) * Kokkos::sin(2.0 * M_PI * y / nyd);
    const double uy = -U * (nyd / (2.0 * Mm)) * Kokkos::cos(M_PI * x / Mm)
                          * Kokkos::cos(2.0 * M_PI * y / nyd);
    const double rr = 1.0 + aa * Kokkos::cos(M_PI * x / Mm)
                              * Kokkos::cos(2.0 * M_PI * y / nyd);
    return FlowState{Real(rr), Real(ux), Real(uy), Real(0)};
  });

  for (std::size_t t = 0; t < c.steps; ++t) s.step();
  s.compute_macroscopic();

  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
  std::vector<double> out;
  rho_out->clear();
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x <= c.M; ++x) {
      out.push_back(double(hu(d.id(x, y, 0))));
      out.push_back(double(hv(d.id(x, y, 0))));
      rho_out->push_back(double(hr(d.id(x, y, 0))));
    }
  return out;
}

//==============================================================================
//  2. THE QUARTER BOX, 3-D -- the two-face edges.
//==============================================================================
template <class L>
static std::vector<double> run3d(const Cfg& c, bool quarter,
                                 std::vector<double>* rho_out) {
  const Index nx = quarter ? c.M + 1 : 2 * c.M;
  const Index nz = quarter ? c.P + 1 : 2 * c.P;
  const Index ny = c.ny;
  Domain d(nx, ny, nz, !quarter, true, !quarter);

  Coll<L> coll;
  coll.omega = Real(1.0 / c.tau);
  FluidSolver<L, EsotericPull<L>, Coll<L>> s(d, coll);
  s.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  if (quarter)
    s.set_specular_nodes([&](Index x, Index, Index z) -> std::uint8_t {
      std::uint8_t m = SpecNone;
      if (x == 0)      m = std::uint8_t(m | SpecXm);
      if (x == nx - 1) m = std::uint8_t(m | SpecXp);
      if (z == 0)      m = std::uint8_t(m | SpecZm);
      if (z == nz - 1) m = std::uint8_t(m | SpecZp);
      return m;
    });

  // u_x odd in x and even in z, u_z odd in z and even in x, u_y even in both,
  // rho even in both. Solenoidal by the amplitude relation below.
  const double Mm = double(c.M), Pp = double(c.P), nyd = double(ny);
  const double sc = c.U / std::max(Mm, Pp);
  const double A = sc * Mm, B = -sc * nyd * 0.5, C = sc * Pp;
  const double aa = c.a;
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const double x = double(px - d.hx), y = double(py - d.hy), z = double(pz - d.hz);
    const double sx = Kokkos::sin(M_PI * x / Mm), cx = Kokkos::cos(M_PI * x / Mm);
    const double sz = Kokkos::sin(M_PI * z / Pp), cz = Kokkos::cos(M_PI * z / Pp);
    const double sy = Kokkos::sin(2.0 * M_PI * y / nyd);
    const double cy = Kokkos::cos(2.0 * M_PI * y / nyd);
    return FlowState{Real(1.0 + aa * cx * cy * cz),
                     Real(A * sx * cy * cz), Real(B * cx * sy * cz),
                     Real(C * cx * cy * sz)};
  });

  for (std::size_t t = 0; t < c.steps; ++t) s.step();
  s.compute_macroscopic();

  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto hw = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
  std::vector<double> out;
  rho_out->clear();
  for (Index z = 0; z <= c.P; ++z)
    for (Index y = 0; y < ny; ++y)
      for (Index x = 0; x <= c.M; ++x) {
        out.push_back(double(hu(d.id(x, y, z))));
        out.push_back(double(hv(d.id(x, y, z))));
        out.push_back(double(hw(d.id(x, y, z))));
        rho_out->push_back(double(hr(d.id(x, y, z))));
      }
  return out;
}

//==============================================================================
//  3. THE MOMENT IDENTITIES, on a random vector.
//==============================================================================
//  Two properties, and the second is the one with teeth about the DENSITY.
//
//  A mirror does NOT preserve the sum of the populations it is handed: it
//  overwrites the unknown directions, so rho after the mirror is whatever the
//  symmetrised set carries. "Density free" means the condition imposes no value
//  on rho, not that it returns the one it was given -- an early version of this
//  test asserted the latter and was simply wrong about what the wall does.
//
//  What is assertable is that an ALREADY-SYMMETRIC state is a fixed point: hand
//  the mirror an equilibrium at (rho, u) whose normal components vanish on the
//  masked axes and it must return that state bit for bit, rho included. A
//  condition that clamped or rescaled the density would fail this, and so would
//  bounce-back, which would flatten u_t.
//
//  `expect_tangential` is the other guard against a mirror that quietly became
//  bounce-back: bounce-back kills EVERY momentum component, which would still
//  pass both flux tests and would still pass a symmetric identity check.
template <class L>
static int moments(std::uint8_t faces, const char* name, bool expect_tangential) {
  double f[L::Q];
  // Deterministic, and not symmetric in any axis: a symmetric input would make
  // the flux vanish for reasons that have nothing to do with the mirror.
  unsigned st = 12345u;
  for (int i = 0; i < L::Q; ++i) {
    st = st * 1664525u + 1013904223u;
    f[i] = 0.5 + double(st >> 8) / double(1u << 24);
  }
  mirror_unknowns_faces<L>(f, faces);

  double mom[3] = {0, 0, 0}, str[3][3] = {};
  for (int i = 0; i < L::Q; ++i)
    for (int a = 0; a < 3; ++a) {
      mom[a] += f[i] * double(cvel<L>(i, a));
      for (int b = 0; b < 3; ++b)
        str[a][b] += f[i] * double(cvel<L>(i, a)) * double(cvel<L>(i, b));
    }

  double worst_flux = 0, worst_shear = 0, tang = 0;
  for (int a = 0; a < 3; ++a) {
    if (face_sign(faces, a) == 0) { tang = std::max(tang, std::abs(mom[a])); continue; }
    worst_flux = std::max(worst_flux, std::abs(mom[a]));
    for (int b = 0; b < 3; ++b)
      if (b != a) worst_shear = std::max(worst_shear, std::abs(str[a][b]));
  }

  // ---- the fixed-point property, which is where rho is checked -------------
  const double rho0 = 1.3;
  double uu[3] = {0.03, -0.02, 0.01};
  for (int a = 0; a < 3; ++a)
    if (face_sign(faces, a) != 0 || a >= L::D) uu[a] = 0.0;
  double g[L::Q];
  const double cs2v = double(cs2<L, double>());
  for (int i = 0; i < L::Q; ++i) {
    const double cu = double(cvel<L>(i, 0)) * uu[0] + double(cvel<L>(i, 1)) * uu[1]
                    + double(cvel<L>(i, 2)) * uu[2];
    const double u2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];
    g[i] = double(weight<L, double>(i)) * rho0
         * (1.0 + cu / cs2v + cu * cu / (2 * cs2v * cs2v) - u2 / (2 * cs2v));
  }
  double g0[L::Q];
  for (int i = 0; i < L::Q; ++i) g0[i] = g[i];
  mirror_unknowns_faces<L>(g, faces);
  double fixed = 0, rho_a = 0, rho_b = 0;
  for (int i = 0; i < L::Q; ++i) {
    fixed = std::max(fixed, std::abs(g[i] - g0[i]));
    rho_a += g0[i];  rho_b += g[i];
  }

  const bool ok = worst_flux < 1e-13 && worst_shear < 1e-13
                  && fixed < 1e-15 && std::abs(rho_a - rho_b) < 1e-15
                  && (expect_tangential ? tang > 1e-3 : true);
  std::printf("    %-16s flux %.2e  shear %.2e  |u_t| %.4f  fixed point %.2e"
              " (rho %.6f)  %s\n",
              name, worst_flux, worst_shear, tang, fixed, rho_b, ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}

//==============================================================================
static int check2d(const Cfg& c) {
  std::vector<double> rf, rh, rc;
  const std::vector<double> uf = run2d<D2Q9>(c, 0, &rf);
  const std::vector<double> uh = run2d<D2Q9>(c, 1, &rh);
  // DOES THIS TEST HAVE TEETH? The same half box closed by the mirror the tree
  // ALREADY had. SpecWall is exact too -- validation/specular.cpp measures its
  // plane at 16.5000 -- but it is exact about a plane half a cell OUTSIDE the
  // last fluid node, so it does not solve the same problem as the periodic box
  // above and must not reproduce it. If it did, the identity here would be
  // measuring nothing and SpecNode would not have been needed. Printed, never
  // scored, and compared only on the interior nodes 1..M-1 that both
  // geometries treat as fluid.
  const std::vector<double> uc = run2d<D2Q9>(c, 2, &rc);
  double worst = 0, scale = 0, wr = 0, rs = 0, lo = 1e300, hi = -1e300;
  for (std::size_t i = 0; i < uf.size(); ++i) {
    worst = std::max(worst, std::abs(uf[i] - uh[i]));
    scale = std::max(scale, std::abs(uf[i]));
  }
  for (std::size_t i = 0; i < rf.size(); ++i) {
    wr = std::max(wr, std::abs(rf[i] - rh[i]));
    rs = std::max(rs, std::abs(rf[i] - 1.0));
    lo = std::min(lo, rf[i]);  hi = std::max(hi, rf[i]);
  }
  double worst_c = 0;
  for (Index y = 0; y < c.ny; ++y)
    for (Index x = 1; x < c.M; ++x) {
      const std::size_t i = 2 * (std::size_t(y) * std::size_t(c.M + 1) + std::size_t(x));
      worst_c = std::max(worst_c, std::abs(uf[i] - uc[i]));
      worst_c = std::max(worst_c, std::abs(uf[i + 1] - uc[i + 1]));
    }
  const double rel = worst / scale;
  // The reference must still CARRY STRUCTURE: two decayed fields agree to
  // round-off whatever the wall does, and would pass this vacuously.
  const bool ok = rel < c.tol && wr < c.tol * 10 && scale > 1e-3 && rs > 1e-5;
  std::printf("  D2Q9   half box %lld x %lld vs periodic %lld x %lld, tau = %.2f\n",
              (long long)(c.M + 1), (long long)c.ny, (long long)(2 * c.M),
              (long long)c.ny, c.tau);
  std::printf("    velocity : worst %.3e (relative %.3e, tolerance %.0e)   %s\n",
              worst, rel, c.tol, ok ? "ok" : "FAIL");
  std::printf("    density  : worst %.3e; reference still carries %.2e of"
              " structure and |u| up to %.4f\n", wr, rs, scale);
  std::printf("    CONTROL, the same box closed by the HALFWAY mirror SpecWall:"
              " worst %.3e,\n              i.e. %.1f %% of the field, on the"
              " interior nodes both geometries share\n",
              worst_c, 100.0 * worst_c / scale);
  return ok ? 0 : 1;
}

static int check3d(const Cfg& c) {
  std::vector<double> rf, rh;
  const std::vector<double> qf = run3d<D3Q27>(c, false, &rf);
  const std::vector<double> qh = run3d<D3Q27>(c, true,  &rh);
  double worst = 0, scale = 0, wr = 0, rs = 0;
  for (std::size_t i = 0; i < qf.size(); ++i) {
    worst = std::max(worst, std::abs(qf[i] - qh[i]));
    scale = std::max(scale, std::abs(qf[i]));
  }
  for (std::size_t i = 0; i < rf.size(); ++i) {
    wr = std::max(wr, std::abs(rf[i] - rh[i]));
    rs = std::max(rs, std::abs(rf[i] - 1.0));
  }
  const double rel = worst / scale;
  // Same structure guard as in 2-D. The density is normalised by 1, not by rs,
  // because rho itself is O(1) and its perturbation is the small quantity.
  const bool ok = rel < c.tol && wr < c.tol * 10 && scale > 1e-3 && rs > 1e-5;
  std::printf("  D3Q27  quarter box %lld x %lld x %lld vs periodic %lld x %lld x %lld,"
              " tau = %.2f\n", (long long)(c.M + 1), (long long)c.ny,
              (long long)(c.P + 1), (long long)(2 * c.M), (long long)c.ny,
              (long long)(2 * c.P), c.tau);
  std::printf("    four TWO-FACE edge lines : velocity worst %.3e (relative %.3e,"
              " tolerance %.0e)   %s\n", worst, rel, c.tol, ok ? "ok" : "FAIL");
  std::printf("    density worst %.3e; reference carries %.2e of structure and"
              " |u| up to %.4f\n", wr, rs, scale);
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  Cfg c;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if      (a == "-m"     && i + 1 < argc) c.M     = Index(std::atol(argv[++i]));
    else if (a == "-ny"    && i + 1 < argc) c.ny    = Index(std::atol(argv[++i]));
    else if (a == "-p"     && i + 1 < argc) c.P     = Index(std::atol(argv[++i]));
    else if (a == "-steps" && i + 1 < argc) c.steps = std::size_t(std::atol(argv[++i]));
    else if (a == "-tau"   && i + 1 < argc) c.tau   = std::atof(argv[++i]);
    else if (a == "-tol"   && i + 1 < argc) c.tol   = std::atof(argv[++i]);
  }
  if (sizeof(Real) == 4 && c.tol < 1e-5) c.tol = 1e-5;   // FP32 cannot hold 1e-12

  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    std::printf("On-node specular fluid wall (SpecNode) -- identity against the"
                " box it mirrors\n");
    std::printf("  %s   %zu steps, tolerance %.0e\n\n",
                sizeof(Real) == 4 ? "FP32" : "FP64", c.steps, c.tol);
    fails += check2d(c);
    std::printf("\n");
    fails += check3d(c);
    std::printf("\n  moment identities of the mirror itself (random populations,"
                " no solver):\n");
    fails += moments<D2Q9>(SpecXm, "D2Q9 -x face", true);
    fails += moments<D3Q27>(SpecXm, "D3Q27 -x face", true);
    fails += moments<D3Q27>(std::uint8_t(SpecXm | SpecZm), "D3Q27 -x-z edge", true);
    // A three-face corner leaves NO free axis, so every momentum component is
    // constrained and `tang` is vacuously zero -- the tangential guard is not
    // applicable there rather than expected to fail.
    fails += moments<D3Q27>(std::uint8_t(SpecXm | SpecYp | SpecZm),
                            "D3Q27 corner", false);
    std::printf("\n  %s\n", fails ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return fails;
}
