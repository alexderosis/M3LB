//==============================================================================
//  THE GPU COLOUR OPERATOR, BEHIND A PLAIN-ARRAY INTERFACE.
//
//  Its own translation unit for one reason: both code bases put their lattice
//  and their operator in `namespace lbm`, so including src/collision/
//  ColourGradient.hpp and GPU/include/lbm/colour.cuh in ONE file collides on
//  D3Q27 and on half a dozen other names. Separated into two objects and joined
//  by extern "C" they never meet, and the test can drive both.
//
//  NOTHING FROM Kokkos REACHES THIS FILE, which is the point: GPU/ does not
//  acquire a Kokkos dependency by being cross-checked. The include path points
//  into GPU/include and stops there.
//
//  PRECISION MUST BE MATCHED BY HAND. GPU/'s Real is float unless LBM_DOUBLE is
//  defined; the parent's is double unless LBM_SINGLE_PRECISION is. The two
//  defaults are OPPOSITE, so tests/CMakeLists.txt defines LBM_DOUBLE on this
//  file exactly when the parent is FP64. Getting that wrong does not fail to
//  compile: it reports a disagreement of ~1e-7, which is FP32 epsilon wearing
//  the costume of a physics bug. That happened while this harness was being
//  written, and it cost a real detour.
//==============================================================================
#include "lbm/colour.cuh"

using namespace lbm;

extern "C" void xc_gpu_collide(double* f, double rho, const double* u, double p,
                               const double* g, const double* dr,
                               double ar, double ab, double nur, double nub,
                               double A, double ombulk, double bx, double by,
                               double bz, double rref, double rr0, double rb0,
                               int per_colour) {
  ColourModel m;
  m.alpha_r = Real(ar);  m.alpha_b = Real(ab);
  m.nu_r = Real(nur);    m.nu_b = Real(nub);
  m.A = Real(A);         m.omega_bulk = Real(ombulk);
  m.bx = Real(bx);  m.by = Real(by);  m.bz = Real(bz);
  m.rho_ref = Real(rref);
  m.rho_r0 = Real(rr0);  m.rho_b0 = Real(rb0);
  m.rest = per_colour ? ColourModel::RestTerm::PerColour
                      : ColourModel::RestTerm::AlphaBar;
  Real ff[27], uu[3], gg[3], dd[3];
  for (int i = 0; i < 27; ++i) ff[i] = Real(f[i]);
  for (int a = 0; a < 3; ++a) {
    uu[a] = Real(u[a]);  gg[a] = Real(g[a]);  dd[a] = Real(dr[a]);
  }
  colour_collide(m, ff, Real(rho), uu, Real(p), gg, dd);
  for (int i = 0; i < 27; ++i) f[i] = double(ff[i]);
}

// The velocity set, so the two lattices are matched slot by slot rather than
// assumed to share an ordering. They do share one today; a future reordering on
// either side would otherwise turn this test into nonsense that looks physical.
extern "C" void xc_gpu_velocities(int* c) {
  for (int i = 0; i < 27; ++i) {
    c[3 * i + 0] = D3Q27::cx(i);
    c[3 * i + 1] = D3Q27::cy(i);
    c[3 * i + 2] = D3Q27::cz(i);
  }
}
