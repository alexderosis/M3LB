//==============================================================================
//  Is the HALFWAY ghost mirror SpecWall an exact mirror for an UNSTEADY flow?
//
//  PRINTED, NOT SCORED -- a measurement kept so the number in
//  boundary/Specular.hpp's banner can be reproduced, not a regression test.
//
//  validation/specular.cpp holds SpecWall to 5.5e-14, but against a STEADY,
//  wall-invariant Poiseuille state. specular_node.cpp's SpecWall control is
//  compared against ON-node planes, which is a different problem. The like-for-
//  like reference is the periodic box of 2M whose symmetry planes lie HALFWAY
//  between nodes (nodes at x = i + 1/2, planes at x = 0 and x = M), and that is
//  what this compares against, on two unsteady fields:
//
//    * a decaying shear wave u_y(x), invariant along the wall, which isolates
//      the TIMING of the reflection;
//    * a mode varying along the wall as well, which adds the lateral landing
//      point of the diagonal populations.
//
//  Measured 2026-09-25 (D2Q9 BGK, tau = 0.6, diffusive scaling to one physical
//  time): shear wave 2.4e-2 / 1.2e-2 / 6.1e-3 of the field at M = 16 / 32 / 64,
//  i.e. first order; the along-wall mode 13 % at M = 16. The larger boxes of the
//  second case keep ny = 20 while the step count grows, so that mode decays to
//  noise there and only M = 16 is a usable number -- read it as "not an
//  identity", not as a rate.
//
//    usage: specwall_unsteady
//==============================================================================
#include "collision/BGK.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace lbm;
using L = D2Q9;
using Coll = BGK<L, SecondOrderEquilibrium<L>, NoForcing, ShiftedPopulations>;

static std::vector<double> run(bool half, int M, int P, double tau, int steps, bool steady_x) {
  const Index nx = half ? M + 2 : 2 * M, ny = P;
  Domain d(nx, ny, 1, !half, true, true);
  Coll c; c.omega = Real(1.0 / tau);
  FluidSolver<L, EsotericPull<L>, Coll> s(d, c);
  s.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  if (half)
    s.set_specular_walls([&](Index x, Index, Index) -> std::uint8_t {
      if (x == 0) return NrmXm;
      if (x == nx - 1) return NrmXp;
      return NrmNone;
    });
  const double U = 0.04, a = 0.02, Mm = M, Pp = P;
  const double off = half ? -0.5 : 0.5;   // physical x of node i
  s.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const double x = double(px - d.hx) + off, y = double(py - d.hy);
    if (steady_x) {   // x-INVARIANT shear wave: u_y(x) only, symmetric about the planes
      return FlowState{Real(1), Real(0), Real(U * Kokkos::cos(M_PI * x / Mm)), Real(0)};
    }
    const double ux = U * Kokkos::sin(M_PI * x / Mm) * Kokkos::cos(2 * M_PI * y / Pp);
    const double uy = -U * (Pp / (2 * Mm)) * Kokkos::cos(M_PI * x / Mm) * Kokkos::sin(2 * M_PI * y / Pp);
    const double rr = 1.0 + a * Kokkos::cos(M_PI * x / Mm) * Kokkos::cos(2 * M_PI * y / Pp);
    return FlowState{Real(rr), Real(ux), Real(uy), Real(0)};
  });
  for (int t = 0; t < steps; ++t) s.step();
  s.compute_macroscopic();
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  std::vector<double> out;
  for (Index y = 0; y < ny; ++y)
    for (int i = 0; i < M; ++i) {
      const Index x = half ? i + 1 : i;
      out.push_back(hu(d.id(x, y, 0))); out.push_back(hv(d.id(x, y, 0)));
    }
  return out;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int P = 20;
    for (int steady = 1; steady >= 0; --steady)
      for (int M : {16, 32, 64}) {
        const int steps = 300 * (M / 16) * (M / 16);   // diffusive: same physical time
        const std::vector<double> f = run(false, M, P, 0.6, steps, steady), h = run(true, M, P, 0.6, steps, steady);
        double w = 0, sc = 0;
        for (std::size_t i = 0; i < f.size(); ++i) { w = std::max(w, std::abs(f[i] - h[i])); sc = std::max(sc, std::abs(f[i])); }
        std::printf("  %-26s M = %3d  %6d steps   SpecWall half box vs periodic (halfway planes): %.3e of %.3e (%.2e rel)\n",
                    steady ? "shear wave, wall-invariant" : "mode varying along wall", M, steps, w, sc, w / sc);
      }
  }
  Kokkos::finalize();
}
