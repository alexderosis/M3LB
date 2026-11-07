//==============================================================================
//  THE AXISYMMETRIC FREE SURFACE (FreeSurfaceSolver::set_axisymmetric) against
//  exact answers.
//
//  WHY IT EXISTS. demonstrator/mew_nozzle is planar: its nozzle is a slot, its
//  melt a sheet, and a straight-sided planar meniscus has no curvature to hold
//  a Taylor cone. A round nozzle needs the azimuthal curvature and the
//  r-weighted flow, and doing it in 3-D costs 6 h for 40 s at 50 um
//  (demonstrator/mew_cone3d). The r-z solver is a 2-D D2Q9 lattice with x the
//  radius (the axis on the left face of column 0) and y the axis; the member
//  `axisymmetric` of FreeSurfaceSolver derives it. Every reference here is
//  exact, and each section isolates one piece of it.
//
//   A. REST. A liquid cylinder about the axis, no surface tension: rest must
//      hold to round-off, and the r-weighted mass with it.
//   P. REST ABOVE THE LATTICE REFERENCE PRESSURE: an all-liquid tube and a free
//      cylinder with its gas, both at rho = 1.001. The first, r-weighted
//      version of the solver could hold neither (see the member).
//   B. A CYLINDER WITH SURFACE TENSION, periodic in y over 2R < 2 pi R so that
//      Rayleigh-Plateau cannot grow: p' = sigma / R, the azimuthal curvature
//      ALONE (a straight generator has none in the meridian). The pressure is
//      the mean over the last quarter of the run.
//   C. A SPHERE ON THE AXIS: p' = 2 sigma / R, meridional plus azimuthal. R from
//      the conserved r-weighted volume, as validation/surface_tension does.
//   D. HAGEN-POISEUILLE. All liquid, a halfway wall at r = R, periodic in y,
//      body force g: u = g (R^2 - r^2) / (4 nu). No radial velocity.
//   M. THE SAME PIPE WITH A VISCOSITY THAT TRIPLES TOWARDS THE WALL, through
//      FreeSurfaceSolver::omega_of: mu = mu0 (1 + 2 r^2/R^2) gives
//      u = (g R^2 / (8 nu0)) ln[3 / (1 + 2 r^2/R^2)]. What a melt whose
//      viscosity follows its temperature needs, and the local mu in the r-z
//      source is all it takes (the stress tau_rz / r is local in mu).
//   T. A THREAD THREE CELLS THICK, STRETCHED by F_z = G sin(2 pi z / 64),
//      must settle to rest. Not an exact answer: the stability of the axis
//      column, which with (mu/r) d_r u_z explicit there came apart at
//      nu = 1/2 -- what mew_axi's liquid column did just before contact. With
//      its self part implicit (AxiGuo::Az) it holds at 1/2 and at 1, and
//      comes apart at 2; see the member axisymmetric.
//   V. AN EXACT STOKES MODE WITH RADIAL VELOCITY, psi = A r J1(kr) sin(mz):
//      u_r = -A m J1 cos(mz), u_z = A k J0 sin(mz), E^2 psi = -K^2 psi with
//      K^2 = k^2 + m^2. A body force mu K^2 u holds it steady at a uniform
//      pressure; released, it decays at nu K^2. A free surface at k R_w = j_11
//      carries u_r = 0 and no tangential stress there. THIS is the section that
//      caught the radial terms: until AxiGuo had its own force_of() the
//      collision never saw the implicit drag, and the mode came out at 1.48 x
//      its amplitude and decayed at 0.68 x its rate, while D -- which has no
//      radial velocity -- converged at second order throughout, and B breathed
//      for 40000 steps instead of settling.
//   E. A PROLATE DROP RELAXING (l = 2). Stokes limit, from Lamb's interior
//      solution with zero tangential stress, normal stress -sigma kappa and the
//      kinematic condition:
//          d eps/dt = -[l (l+2) (2l+1) / (2 (2l^2 + 4l + 3))] sigma/(mu R) eps,
//      20/19 sigma/(mu R) for l = 2, the drop-in-vacuum limit of Taylor's
//      (2 lambda + 3)(19 lambda + 16) / (40 (lambda + 1)) relaxation time. Run
//      at Oh = 10, where the planar case's finite-Oh root is 1.0018 x Stokes;
//      the amplitude is a/b - 1 from the r-weighted second moments, with the
//      geometric exchange on (the planar case needs it to converge, and so
//      does this one).
//
//  MEASURED (FP64, 4 threads, 21 s, 2026-11-06), copied from the final run:
//   A  max |u| 1.4e-16, mass 1.1e-14.          P  0 and 0.
//   B  dp R / sigma 1.00002 / 1.00001 / 0.99989 at R = 8 / 16 / 32, spurious
//      current 1e-16 / 7e-16 / 6e-8.
//   C  dp R / sigma 2.0209 / 2.0028 / 2.0005 / 1.9992 at R = 8 / 12 / 16 / 24
//      (+1.0 / +0.14 / +0.03 / -0.04 %; the planar case's height functions give
//      +0.85 / +0.32 / +0.34 / -0.02 %), spurious current 3e-10 to 3e-8.
//   D  L2 2.52e-3 / 5.98e-4 / 1.45e-4 at R = 8 / 16 / 32, order 2.07 / 2.04,
//      the axis cell 1.5e-3 / 3.5e-4 / 8.4e-5.
//   M  L2 2.54e-2 / 6.55e-3 / 1.66e-3, order 1.96 / 1.98: second order, ten
//      times D's constant. It is NOT the wall's tau (2 there): with nu0 = 1/18,
//      so that the wall sits at tau = 1, it is 1.41e-2 / 3.57e-3 / 8.99e-4. Half
//      of it is the scheme's own variable-omega stress -- the same profile in a
//      PLANAR channel gives 1.25e-2 / 3.12e-3 / 7.81e-4 (order 2.00) -- and the
//      r-z terms add about as much again. The gate is 2 / R^2.
//   T  max |u| 1.2e-4 and 2.0e-4 after 24000 steps at nu = 1/2 and 1; mass
//      +2.5e-6 and +1.3e-3. The second is two conversion events in the
//      transient (one cell briefly at rho = 1.33), not a drift: nu = 1 SURVIVES
//      and nu <= 1/2 is CLEAN. Gates 2e-3 on |u|, 1e-4 and 5e-3 on mass.
//   V  steady amplitude 0.924 / 0.975 / 0.991, decay 1.105 / 1.059 / 1.041 of
//      nu K^2 at R_w = 8 / 16 / 32: converging, at about first order, which is
//      the free surface's (the same mode has a wall nowhere else). Before the
//      axis column's implicit term the decay read 1.110 / 1.061 / 1.042; no
//      other section moved.
//   E  0.983 / 0.990 of the Stokes rate at R = 8 / 16, rms of ln eps 8e-3 /
//      2e-2. With the DEFAULT exchange 0.954 / 0.812 / 0.783 at R = 8 / 16 /
//      32, slowing as R grows exactly as the planar case does. With the
//      geometric one at R = 32 the drop DRIFTS along the axis from ~1e5 steps
//      (accelerating; 22 cells by 7e5 steps, until it reaches the box) while
//      the default exchange holds its centroid at 54.500 to every printed
//      digit: the geometric exchange breaks the up-down symmetry of a free drop
//      over a long run, which a drop with no restoring force against
//      translation then follows. Not diagnosed; why R = 32 is not run here.
//  Mass: the free surface's conversions move it by up to 3e-5 here, inside the
//  band validation/surface_tension's static drop shows (-1.6e-5 under a 1e-3
//  gate); the gates are 1e-4. Every gate is a regression guard set from these
//  numbers, not a derivation.
//  FP32 (measured 2026-11-06): everything else passes with the same gates, but
//  the mass moves by up to 1.8e-4 in B and C and 7.5e-4 in E at R = 8 (gate
//  3e-3 there), and two rows reach the precision's floor and are NOT run: D
//  at R = 32 (1.1e-3, order -1.7 -- the force per step is below FP32's
//  resolution of the populations) and E at R = 16 (rate 0.74 of Stokes). M's
//  R = 32 row reads 1.2e-3, order 2.4: the floor arriving, inside its gate.
//
//   J. ON REQUEST (-sec J; not in the default list, no gate): FreeSurfaceSolver::
//      draw() in r-z. A resting cylinder, R = 16 and 32, periodic in z, has
//      0.005 (r-weighted) drawn per step for 3000 steps from a 2 x 4-cell box ON
//      the axis or at r = R/2, both deep in the liquid; (total + drained -
//      before) / drawn is -3.3e-4 / -3.3e-4 at R = 16 and -3.7e-4 / -3.5e-4 at
//      R = 32: the axis costs nothing extra. With the geometric exchange alone,
//      -3.3e-4 / -3.4e-4; with surface tension 0.005 alone, -3.3e-4 / +1.5e-3;
//      with both, -1.2e-2 / -2.2e-2. A spot of density 0.95 seeded in the axis
//      box with nothing drawn moves the total by -6.7e-7 / -2.7e-3 of its
//      deficit. Written 2026-11-07 to find why demonstrator/mew_axi's jet made
//      mass: not the axis -- the box was touching the free surface (see the JET
//      in that banner).
//
//  Usage: axisym_fs [-sec APBCDMKTVE] [-sec J] [-v]
//==============================================================================
#include "core/Types.hpp"
#include "solver/FreeSurfaceSolver.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

using L    = D2Q9;
using Coll = CentralMoments<L, AxiGuo, RawPopulations>;
using FS   = FreeSurfaceSolver<L, Coll>;

namespace {

bool g_verbose = false;
int  g_fail = 0;
constexpr double kPi = 3.14159265358979323846;
// Round-off gates scale with the precision (validation/decaying_flows.cpp's rule).
const double kEps = sizeof(Real) == 8 ? 1.0 : 1e8;
// The free surface's conversions in FP32 move the mass by up to 1.8e-4 (B, C)
// and 1.7e-3 (E at R = 16) where FP64 stays under 3.3e-5: its own gate.
const bool kFP32 = sizeof(Real) == 4;
const double kMass = kFP32 ? 3e-3 : 1e-4;

void check(const char* what, double got, double want, double tol) {
  const bool ok = std::isfinite(got) && std::fabs(got - want) <= tol;
  if (!ok) ++g_fail;
  std::printf("  %-50s %12.6g  (want %g +- %g)  %s\n", what, got, want, tol, ok ? "PASS" : "FAIL");
}

void base(FS& s, bool geom) {
  s.coll.omega = FS::omega_from_viscosity(Real(1.0 / 6.0));
  s.set_axisymmetric(true);
  if (geom) {
    s.set_exchange_geometric(true);
    s.exchange_donor = true;
    s.surface_follows_line = true;
  }
}

struct Snap {
  double vol = 0, umax = 0, p_core = 0, zc = 0, mzz = 0, mrr = 0;
  long ncore = 0;
  bool finite = true;
};

// r-weighted volume, centroid and second moments; the mean density of FLUID
// cells within `core` of the point (0, zc) (sphere) or of the axis (cylinder).
Snap snap(const FS& s, const Domain& d, double core, bool sphere, double zc_in) {
  Snap S;
  auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
  auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.fill());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  double vz = 0;
  for (Index y = 0; y < d.ny; ++y)
    for (Index x = 0; x < d.nx; ++x) {
      const Index n = d.id(x, y);
      const std::uint8_t f = hf(n);
      if (f != FsFluid && f != FsInterface) continue;
      const double r = double(x) + 0.5, z = double(y) + 0.5;
      const double c = (f == FsFluid) ? 1.0 : double(he(n));
      S.vol += c * r;
      vz += c * r * z;
      const double u = std::hypot(double(hu(n)), double(hv(n)));
      if (!std::isfinite(u)) S.finite = false;
      S.umax = std::max(S.umax, u);
    }
  S.zc = (S.vol > 0) ? vz / S.vol : 0;
  const double zc = (zc_in > 0) ? zc_in : S.zc;
  double ps = 0;
  for (Index y = 0; y < d.ny; ++y)
    for (Index x = 0; x < d.nx; ++x) {
      const Index n = d.id(x, y);
      const std::uint8_t f = hf(n);
      const double r = double(x) + 0.5, z = double(y) + 0.5;
      if (f == FsFluid || f == FsInterface) {
        const double c = (f == FsFluid) ? 1.0 : double(he(n));
        S.mzz += c * r * (z - S.zc) * (z - S.zc);
        S.mrr += c * r * r * r;
      }
      if (f != FsFluid) continue;
      if ((sphere ? std::hypot(r, z - zc) : r) > core) continue;
      ps += double(hr(n));
      ++S.ncore;
    }
  if (S.vol > 0) { S.mzz /= S.vol; S.mrr /= S.vol; }
  S.p_core = (S.ncore > 0) ? ps / double(S.ncore) : 0.0;
  if (!S.finite) S.umax = NAN;
  return S;
}

//---- A, P and B: the cylinder, and the tube ------------------------------------------
struct Res { double prel = NAN, umax = NAN, drift = NAN; };

Res cylinder(Index R, double sigma, std::size_t steps, double rho0) {
  Res C;
  const Index nx = R + 8, ny = 2 * R;              // y-period 2R < 2 pi R: stable
  Domain d(nx, ny, 1, false, true, true);
  FS s(d);
  base(s, false);
  if (sigma > 0) { s.set_surface_tension(Real(sigma)); s.set_height_function(true); }
  s.rho_G = Real(rho0);
  s.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  const Domain dd = d;
  const Index hx = d.hx;
  const Real Rr = Real(R), r0 = Real(rho0);
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real r = Real(px - hx) + Real(0.5);
    Real e = Rr + Real(0.5) - r;
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, r0};
  });
  const double m0 = double(s.total_mass());
  double psum = 0, pn = 0;
  for (std::size_t t = 1; t <= steps; ++t) {
    s.step();
    if (4 * t > 3 * steps && t % 100 == 0) {
      const Snap S = snap(s, d, 0, false, 0);
      const double Rm = std::sqrt(2.0 * S.vol / double(ny));   // pi R^2 len = 2 pi vol
      const Snap Cc = snap(s, d, Rm - 3.0, false, 0);
      psum += (Cc.p_core - rho0) / 3.0 * (sigma > 0 ? Rm / sigma : 1.0);
      pn += 1;
    }
  }
  C.drift = (double(s.total_mass()) - m0) / m0;
  C.prel = pn > 0 ? psum / pn : NAN;
  C.umax = snap(s, d, 0, false, 0).umax;
  return C;
}

double tube(Index R, double rho0, std::size_t steps) {
  Domain d(R + 1, 8, 1, false, true, true);
  FS s(d);
  base(s, false);
  s.set_geometry([&](Index x, Index, Index) -> FsCell { return x >= R ? FsSolid : FsGas; });
  const Real r0 = Real(rho0);
  s.initialize(KOKKOS_LAMBDA(Index) { return typename FS::Seed{Real(1), r0}; });
  for (std::size_t t = 0; t < steps; ++t) s.step();
  return snap(s, d, 0, false, 0).umax;
}

//---- C: the sphere -------------------------------------------------------------------
Res sphere(Index R, double sigma, std::size_t steps) {
  Res C;
  const Index nx = R + 10, ny = 2 * R + 20;
  Domain d(nx, ny, 1, false, false, true);
  FS s(d);
  base(s, false);
  s.set_surface_tension(Real(sigma));
  s.set_height_function(true);
  s.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  const Real Rr = Real(R), zc = Real(ny) * Real(0.5);
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real r = Real(px - hx) + Real(0.5), z = Real(py - hy) + Real(0.5) - zc;
    Real e = Rr + Real(0.5) - Kokkos::sqrt(r * r + z * z);
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, Real(1)};
  });
  const double m0 = double(s.total_mass());
  for (std::size_t t = 0; t < steps; ++t) s.step();
  C.drift = (double(s.total_mass()) - m0) / m0;
  const Snap S = snap(s, d, 0, true, double(zc));
  const double Rm = std::cbrt(3.0 * S.vol / 2.0);   // 2 pi vol = (4/3) pi R^3
  const Snap Cc = snap(s, d, Rm - 3.0, true, S.zc);
  C.prel = (Cc.p_core - 1.0) / 3.0 * Rm / sigma;
  C.umax = S.umax;
  if (g_verbose) std::printf("    sphere R=%d: R_vol %.4f, core cells %ld\n", int(R), Rm, Cc.ncore);
  return C;
}

//---- D: Hagen-Poiseuille ---------------------------------------------------------------
double pipe(Index R, double& axis_err) {
  Domain d(R + 1, 4, 1, false, true, true);
  FS s(d);
  base(s, false);
  const double nu = 1.0 / 6.0, U = 0.01;
  const double g = 4.0 * nu * U / double(R * R);
  s.set_gravity(Real(0), Real(g));
  s.set_geometry([&](Index x, Index, Index) -> FsCell { return x >= R ? FsSolid : FsGas; });
  s.initialize(KOKKOS_LAMBDA(Index) { return typename FS::Seed{Real(1), Real(1)}; });
  const std::size_t steps = std::size_t(12.0 * double(R * R) / nu);
  for (std::size_t t = 0; t < steps; ++t) s.step();
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  double e2 = 0, n2 = 0;
  for (Index x = 0; x < R; ++x) {
    const double r = double(x) + 0.5;
    const double ue = g * (double(R * R) - r * r) / (4.0 * nu);
    const double u = double(hv(d.id(x, 1)));
    e2 += (u - ue) * (u - ue) * r;
    n2 += ue * ue * r;
    if (x == 0) axis_err = (u - ue) / ue;
  }
  return std::sqrt(e2 / n2);
}

//---- M: Hagen-Poiseuille with a viscosity that triples towards the wall -------------------
// mu(r) = mu0 (1 + a r^2/R^2): (1/r) d_r (r mu d_r u) = -rho g gives
//   u(r) = (g R^2 / (4 nu0 a)) ln[(1 + a) / (1 + a r^2/R^2)],
// through FreeSurfaceSolver::omega_of, one rate per column.
double pipe_mu(Index R, double a, double& axis_err) {
  Domain d(R + 1, 4, 1, false, true, true);
  FS s(d);
  base(s, false);
  const double nu0 = 1.0 / 6.0, U = 0.01;
  const double g = 4.0 * nu0 * a * U / (double(R * R) * std::log(1.0 + a));
  s.set_gravity(Real(0), Real(g));
  s.set_geometry([&](Index x, Index, Index) -> FsCell { return x >= R ? FsSolid : FsGas; });
  s.initialize(KOKKOS_LAMBDA(Index) { return typename FS::Seed{Real(1), Real(1)}; });
  s.omega_of = View1D<Real>("omega_of", d.n_padded);
  auto h = Kokkos::create_mirror_view(s.omega_of);
  for (Index n = 0; n < d.n_padded; ++n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const double r = double(px - d.hx) + 0.5;
    h(n) = FS::omega_from_viscosity(Real(nu0 * (1.0 + a * r * r / double(R * R))));
  }
  Kokkos::deep_copy(s.omega_of, h);
  const std::size_t steps = std::size_t(12.0 * double(R * R) / nu0);
  for (std::size_t t = 0; t < steps; ++t) s.step();
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  double e2 = 0, n2 = 0;
  for (Index x = 0; x < R; ++x) {
    const double r = double(x) + 0.5;
    const double ue = g * double(R * R) / (4.0 * nu0 * a) * std::log((1.0 + a) / (1.0 + a * r * r / double(R * R)));
    const double u = double(hv(d.id(x, 1)));
    e2 += (u - ue) * (u - ue) * r;
    n2 += ue * ue * r;
    if (x == 0) axis_err = (u - ue) / ue;
  }
  return std::sqrt(e2 / n2);
}

//---- K: Brinkman's pipe -------------------------------------------------------------------
// A uniform implicit drag A (FreeSurfaceSolver::drag_of) in the pipe of D:
// mu (u'' + u'/r) - A u + rho g = 0, u(R) = 0, so
//   u(r) = (g / A) [1 - I0(k r) / I0(k R)],  k^2 = A / mu.
double besseli0(double x) {          // power series, ~1e-16 for the x <= 10 used here
  double s = 1, t = 1;
  for (int k = 1; k < 60; ++k) { t *= (x / (2.0 * k)) * (x / (2.0 * k)); s += t; }
  return s;
}
double pipe_drag(Index R, double kR, double& axis_err) {
  Domain d(R + 1, 4, 1, false, true, true);
  FS s(d);
  base(s, false);
  const double nu = 1.0 / 6.0, U = 0.01;
  const double k = kR / double(R), A = nu * k * k;
  const double g = A * U / (1.0 - 1.0 / besseli0(kR));
  s.set_gravity(Real(0), Real(g));
  s.set_geometry([&](Index x, Index, Index) -> FsCell { return x >= R ? FsSolid : FsGas; });
  s.initialize(KOKKOS_LAMBDA(Index) { return typename FS::Seed{Real(1), Real(1)}; });
  s.drag_of = View1D<Real>("drag_of", d.n_padded);
  Kokkos::deep_copy(s.drag_of, Real(A));
  const std::size_t steps = std::size_t(12.0 * double(R * R) / nu);
  for (std::size_t t = 0; t < steps; ++t) s.step();
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  double e2 = 0, n2 = 0;
  for (Index x = 0; x < R; ++x) {
    const double r = double(x) + 0.5;
    const double ue = g / A * (1.0 - besseli0(k * r) / besseli0(kR));
    const double u = double(hv(d.id(x, 1)));
    e2 += (u - ue) * (u - ue) * r;
    n2 += ue * ue * r;
    if (x == 0) axis_err = (u - ue) / ue;
  }
  return std::sqrt(e2 / n2);
}

//---- T: a thin thread stretched by a body force ----------------------------------------
// A liquid cylinder R = 3 about the axis, periodic over 64 cells, pulled by
// F_z = G sin(2 pi z / 64). It settles to rest, varicose, held by its surface
// tension and pressure. Its only use is that it MUST settle: with the axis
// column's (mu/r) d_r u_z explicit it came apart at nu = 1/2 (the member
// axisymmetric), which is what mew_axi's column did just before contact.
struct Thread { double umax = NAN, drift = NAN; };
Thread thread(double nu, std::size_t steps) {
  Thread T;
  const Index nx = 11, ny = 64;
  Domain d(nx, ny, 1, false, true, true);
  FS s(d);
  base(s, true);
  s.coll.omega = FS::omega_from_viscosity(Real(nu));
  s.set_surface_tension(Real(2e-5));
  s.set_height_function(true);
  s.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real r = Real(px - hx) + Real(0.5);
    Real e = Real(3.5) - r;
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, Real(1)};
  });
  s.force_of_y = View1D<Real>("fy", d.n_padded);
  auto fy = s.force_of_y;
  const Real kz = Real(2.0 * kPi / double(ny));
  Kokkos::parallel_for("thread_force", Kokkos::RangePolicy<>(0, d.n_padded), KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    fy(n) = Real(1e-4) * Kokkos::sin(kz * (Real(py - hy) + Real(0.5)));
  });
  const double m0 = double(s.total_mass());
  for (std::size_t t = 0; t < steps; ++t) s.step();
  T.drift = (double(s.total_mass()) - m0) / m0;
  T.umax = snap(s, d, 0, false, 0).umax;
  return T;
}

//---- V: an exact axisymmetric Stokes mode with radial velocity -------------------------
// Bessel J_n(x) = (1/pi) int_0^pi cos(n t - x sin t) dt, Simpson on 400 panels:
// ~1e-12 for the x <= 4 used here (libc++ has no cyl_bessel_j).
double besselj(int nn, double x) {
  const int N = 400;
  const double h = kPi / N;
  double sum = 0;
  for (int i = 0; i <= N; ++i) {
    const double t = i * h;
    const double w = (i == 0 || i == N) ? 1.0 : (i % 2 ? 4.0 : 2.0);
    sum += w * std::cos(nn * t - x * std::sin(t));
  }
  return sum * h / 3.0 / kPi;
}

struct Mode { double amp = NAN, emax = NAN, rate = NAN; };

Mode stokes_mode(Index Rw) {
  Mode M;
  const Index nx = Rw + 6, ny = Rw;                // period L = R_w: m R_w = 2 pi > 1
  Domain d(nx, ny, 1, false, true, true);
  FS s(d);
  base(s, false);
  s.set_surface_tension(Real(1e-2));               // stiff: the surface stays put
  s.set_height_function(true);
  s.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  const Domain dd = d;
  const Index hx = d.hx;
  const Real Rr = Real(Rw);
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real r = Real(px - hx) + Real(0.5);
    Real e = Rr + Real(0.5) - r;
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, Real(1)};
  });
  const double nu = 1.0 / 6.0;
  const double k = 3.8317059702 / double(Rw), m = 2.0 * kPi / double(ny);
  const double K2 = k * k + m * m, A = 1e-4;
  auto fxv = View1D<Real>("fx", d.n_padded), fyv = View1D<Real>("fy", d.n_padded);
  auto hfx = Kokkos::create_mirror_view(fxv), hfy = Kokkos::create_mirror_view(fyv);
  std::vector<double> ure(d.n_padded, 0.0), uze(d.n_padded, 0.0);
  for (Index y = 0; y < d.ny; ++y)
    for (Index x = 0; x < d.nx; ++x) {
      const Index n = d.id(x, y);
      const double r = double(x) + 0.5, z = double(y) + 0.5;
      ure[n] = -A * m * besselj(1, k * r) * std::cos(m * z);
      uze[n] = A * k * besselj(0, k * r) * std::sin(m * z);
      hfx(n) = Real(nu * K2 * ure[n]);  hfy(n) = Real(nu * K2 * uze[n]);   // rho = 1
    }
  Kokkos::deep_copy(fxv, hfx);  Kokkos::deep_copy(fyv, hfy);
  s.force_of_x = fxv;  s.force_of_y = fyv;
  for (std::size_t t = 0; t < std::size_t(40.0 / (nu * K2)); ++t) s.step();
  // the projection of u on the mode (1 if exact) and the largest deviation
  auto measure = [&](double& emax) {
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
    auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
    auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
    double num = 0, den = 0; emax = 0;
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) {
        const Index n = d.id(x, y);
        if (hf(n) != FsFluid) continue;
        const double r = double(x) + 0.5;
        emax = std::max(emax, std::hypot(double(hu(n)) - ure[n], double(hv(n)) - uze[n]));
        num += (double(hu(n)) * ure[n] + double(hv(n)) * uze[n]) * r;
        den += (ure[n] * ure[n] + uze[n] * uze[n]) * r;
      }
    return num / den;
  };
  M.amp = measure(M.emax);
  M.emax /= A;
  s.force_of_x = View1D<Real>();  s.force_of_y = View1D<Real>();
  const std::size_t span = std::size_t(1.0 / (nu * K2));
  std::vector<double> T, Y;
  for (std::size_t t = 1; t <= span; ++t) {
    s.step();
    if (t % std::max<std::size_t>(1, span / 40) == 0) {
      double e;
      T.push_back(double(t));
      Y.push_back(std::log(measure(e)));
    }
  }
  double st = 0, sy = 0, stt = 0, sty = 0;
  const double nn = double(T.size());
  for (std::size_t i = 0; i < T.size(); ++i) { st += T[i]; sy += Y[i]; stt += T[i] * T[i]; sty += T[i] * Y[i]; }
  M.rate = -(nn * sty - st * sy) / (nn * stt - st * st) / (nu * K2);
  return M;
}

//---- E: relaxation of a prolate drop ------------------------------------------------------
struct Rate { double ratio = NAN, Rm = 0, Oh = 0, eps0 = 0, epsN = 0, drift = 0, rms = 0; };

Rate relax(Index R, double Oh, double aspect, bool geom) {
  Rate Q;
  const double nu = 1.0 / 6.0;
  const double sigma = nu * nu / (Oh * Oh * double(R));   // Oh = mu / sqrt(rho sigma R), rho = 1
  const Index nx = Index(1.35 * double(R)) + 10, ny = Index(2.8 * double(R)) + 20;
  Domain d(nx, ny, 1, false, false, true);
  FS s(d);
  base(s, geom);
  s.set_surface_tension(Real(sigma));
  s.set_height_function(true);
  s.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  const double b = double(R) / std::cbrt(aspect), a = aspect * b;   // a b^2 = R^3
  const Real ra = Real(a), rb = Real(b), zc = Real(ny) * Real(0.5), Rr = Real(R);
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real r = Real(px - hx) + Real(0.5), z = Real(py - hy) + Real(0.5) - zc;
    const Real q = Kokkos::sqrt((r / rb) * (r / rb) + (z / ra) * (z / ra));
    Real e = Rr + Real(0.5) - q * Rr;
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, Real(1)};
  });
  const double m0 = double(s.total_mass());
  // a/b from <z^2> = a^2/5 and <r^2> = 2 b^2/5
  auto amp = [&]() { const Snap S = snap(s, d, 0, true, 0); return std::sqrt(S.mzz / (0.5 * S.mrr)) - 1.0; };
  Q.Rm = std::cbrt(3.0 * snap(s, d, 0, true, 0).vol / 2.0);
  Q.Oh = nu / std::sqrt(sigma * Q.Rm);
  const double lam = (20.0 / 19.0) * sigma / (nu * Q.Rm);
  Q.eps0 = amp();
  // skip two viscous times R^2/nu, then fit ln(eps) over 1.5 e-folds
  const std::size_t skip = std::size_t(2.0 * double(R * R) / nu);
  const std::size_t span = std::size_t(1.5 / lam);
  const std::size_t every = std::max<std::size_t>(1, span / 60);
  std::vector<double> T, Y;
  for (std::size_t t = 1; t <= skip + span; ++t) {
    s.step();
    if (t >= skip && (t - skip) % every == 0) {
      const double e = amp();
      if (!(e > 0) || !std::isfinite(e)) break;
      T.push_back(double(t));
      Y.push_back(std::log(e));
    }
  }
  Q.epsN = amp();
  Q.drift = (double(s.total_mass()) - m0) / m0;
  if (T.size() > 3) {
    double st = 0, sy = 0, stt = 0, sty = 0;
    const double nn = double(T.size());
    for (std::size_t i = 0; i < T.size(); ++i) { st += T[i]; sy += Y[i]; stt += T[i] * T[i]; sty += T[i] * Y[i]; }
    const double slope = (nn * sty - st * sy) / (nn * stt - st * st);
    const double icpt = (sy - slope * st) / nn;
    double rr = 0;
    for (std::size_t i = 0; i < T.size(); ++i) { const double e = Y[i] - (icpt + slope * T[i]); rr += e * e; }
    Q.rms = std::sqrt(rr / nn);
    Q.ratio = -slope / lam;
  }
  return Q;
}

}  // namespace

//---- J (on request): draw() in r-z, from the axis or from a ring ------------------------
// A resting liquid cylinder of radius R, periodic in z; dm (r-weighted) drawn per
// step from a 2 x 4-cell box on the axis or at r = R/2, or nothing drawn with a
// spot of density rho_seed in the axis box instead. Returns (total + drained -
// before) / drawn, or / the seeded deficit for the control.
double draw_cyl(Index R, int where, double dm, int steps, double rho_seed, bool geom = false, double sigma = 0) {
  const Index nx = R + 8, ny = 2 * R;
  Domain d(nx, ny, 1, false, true, true);
  FS s(d);
  base(s, geom);
  if (sigma > 0) { s.set_surface_tension(Real(sigma)); s.set_height_function(true); }
  s.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy, x0 = (where == 1) ? R / 2 : 0, y0 = ny / 2 - 2;
  const Real Rr = Real(R), rs = Real(rho_seed);
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Index x = px - hx, y = py - hy;
    const Real r = Real(x) + Real(0.5);
    Real e = Rr + Real(0.5) - r;
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    const bool inbox = x >= x0 && x < x0 + 2 && y >= y0 && y < y0 + 4;
    return typename FS::Seed{e, inbox ? rs : Real(1)};
  });
  const double m0 = double(s.total_mass());
  double deficit = 0;
  for (Index y = y0; y < y0 + 4; ++y)
    for (Index x = x0; x < x0 + 2; ++x) deficit += (1.0 - rho_seed) * (double(x) + 0.5);
  auto box = KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Index x = px - hx, y = py - hy;
    return x >= x0 && x < x0 + 2 && y >= y0 && y < y0 + 4;
  };
  for (int t = 0; t < steps; ++t) {
    s.step();
    if (where >= 0 && dm > 0) s.draw(Real(dm), box);
  }
  const double led = double(s.total_mass()) + double(s.drained_mass()) - m0;
  const double ref = (dm > 0 && where >= 0) ? double(s.sunk_mass()) : deficit;
  return led / ref;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::string sec = "APBCDMKTVE";
    for (int i = 1; i < argc; ++i) {
      if (!std::strcmp(argv[i], "-v")) g_verbose = true;
      else if (!std::strcmp(argv[i], "-sec") && i + 1 < argc) sec = argv[++i];
      else {
        std::fprintf(stderr, "axisym_fs: unknown option %s\n  usage: axisym_fs [-sec APBCDMKTVE] [-sec J] [-v]\n", argv[i]);
        Kokkos::finalize();
        return 2;
      }
    }
    std::printf("axisym_fs: FreeSurfaceSolver in r-z, D2Q9 central moments + AxiGuo, %s\n",
                sizeof(Real) == 8 ? "FP64" : "FP32");
    const auto t0 = std::chrono::steady_clock::now();
    char w[96];

    if (sec.find('J') != std::string::npos) {
      std::printf("\nJ. draw() in r-z (on request): (total + drained - before) / drawn\n");
      for (Index R : {16, 32}) {
        std::printf("  R = %2d: from the axis box %+.3e, from the ring at R/2 %+.3e;"
                    " a seeded spot on the axis, nothing drawn: %+.3e of its deficit\n", int(R),
                    draw_cyl(R, 0, 0.005, 3000, 1.0), draw_cyl(R, 1, 0.005, 3000, 1.0),
                    draw_cyl(R, -1, 0.0, 3000, 0.95));
        std::printf("          the axis box with -geom %+.3e; with surface tension 0.005 %+.3e; with both %+.3e\n",
                    draw_cyl(R, 0, 0.005, 3000, 1.0, true, 0), draw_cyl(R, 0, 0.005, 3000, 1.0, false, 0.005),
                    draw_cyl(R, 0, 0.005, 3000, 1.0, true, 0.005));
      }
    }
    if (sec.find('A') != std::string::npos) {
      std::printf("\nA. a liquid cylinder at rest, no surface tension (2000 steps)\n");
      const Res C = cylinder(16, 0.0, 2000, 1.0);
      check("max |u|", C.umax, 0.0, 1e-13 * kEps);
      check("relative mass drift", C.drift, 0.0, 1e-12 * kEps);
    }
    if (sec.find('P') != std::string::npos) {
      std::printf("\nP. rest above the lattice reference pressure (rho = 1.001, 2000 steps)\n");
      check("closed tube, all liquid: max |u|", tube(16, 1.001, 2000), 0.0, 1e-13 * kEps);
      check("free cylinder, gas at 1.001: max |u|", cylinder(16, 0.0, 2000, 1.001).umax, 0.0, 1e-13 * kEps);
    }
    if (sec.find('B') != std::string::npos) {
      std::printf("\nB. a cylinder with surface tension: dp R / sigma = 1 (azimuthal curvature alone)\n");
      for (Index R : {8, 16, 32}) {
        const Res C = cylinder(R, 2e-3, 8000, 1.0);
        std::snprintf(w, sizeof w, "R = %2d: dp R / sigma (last quarter)", int(R));
        check(w, C.prel, 1.0, 0.01);
        std::snprintf(w, sizeof w, "R = %2d: max |u| (spurious)", int(R));
        check(w, C.umax, 0.0, 1e-6);
        std::snprintf(w, sizeof w, "R = %2d: relative mass drift", int(R));
        check(w, C.drift, 0.0, kMass);
      }
    }
    if (sec.find('C') != std::string::npos) {
      std::printf("\nC. a sphere on the axis: dp R / sigma = 2\n");
      for (Index R : {8, 12, 16, 24}) {
        const Res C = sphere(R, 2e-3, 10000);
        std::snprintf(w, sizeof w, "R = %2d: dp R / sigma", int(R));
        check(w, C.prel, 2.0, 0.04);
        std::snprintf(w, sizeof w, "R = %2d: max |u| (spurious)", int(R));
        check(w, C.umax, 0.0, 1e-6);
        std::snprintf(w, sizeof w, "R = %2d: relative mass drift", int(R));
        check(w, C.drift, 0.0, kMass);
      }
    }
    if (sec.find('D') != std::string::npos) {
      std::printf("\nD. Hagen-Poiseuille, wall at r = R: relative L2 error of u(r)\n");
      double prev = 0;
      for (Index R : {8, 16, 32}) {
        if (kFP32 && R == 32) {   // FP32 floor: 1.1e-3, order -1.7 (measured)
          std::printf("  R = 32: not run in FP32 -- the force per step is below its resolution there\n");
          continue;
        }
        double ae = 0;
        const double e = pipe(R, ae);
        std::printf("  R = %2d: L2 %.4e  on the axis %+.4e", int(R), e, ae);
        if (prev > 0) std::printf("  order %.2f", std::log(prev / e) / std::log(2.0));
        std::printf("\n");
        prev = e;
        std::snprintf(w, sizeof w, "R = %2d: relative L2 error", int(R));
        check(w, e, 0.0, 0.2 / double(R * R));
      }
    }
    if (sec.find('M') != std::string::npos) {
      std::printf("\nM. Hagen-Poiseuille, viscosity 1 -> 3 towards the wall (omega_of): relative L2 error\n");
      double prev = 0;
      for (Index R : {8, 16, 32}) {
        double ae = 0;
        const double e = pipe_mu(R, 2.0, ae);
        std::printf("  R = %2d: L2 %.4e  on the axis %+.4e", int(R), e, ae);
        if (prev > 0) std::printf("  order %.2f", std::log(prev / e) / std::log(2.0));
        std::printf("\n");
        prev = e;
        std::snprintf(w, sizeof w, "R = %2d: relative L2 error", int(R));
        check(w, e, 0.0, 2.0 / double(R * R));
      }
    }
    if (sec.find('K') != std::string::npos) {
      std::printf("\nK. Brinkman's pipe, an implicit drag (drag_of) with k R = 6: relative L2 error\n");
      double prev = 0;
      for (Index R : {8, 16, 32}) {
        double ae = 0;
        const double e = pipe_drag(R, 6.0, ae);
        std::printf("  R = %2d: L2 %.4e  on the axis %+.4e", int(R), e, ae);
        if (prev > 0) std::printf("  order %.2f", std::log(prev / e) / std::log(2.0));
        std::printf("\n");
        prev = e;
        std::snprintf(w, sizeof w, "R = %2d: relative L2 error", int(R));
        check(w, e, 0.0, 1.0 / double(R * R));
      }
    }
    if (sec.find('T') != std::string::npos) {
      std::printf("\nT. a thread three cells thick, stretched, must settle (24000 steps)\n");
      for (double nu : {0.5, 1.0}) {
        const Thread T = thread(nu, 24000);
        std::printf("  nu = %.2f: max |u| %.3e  mass %+.2e\n", nu, T.umax, T.drift);
        std::snprintf(w, sizeof w, "nu = %.2f: max |u| after 24000 steps", nu);
        check(w, T.umax, 0.0, 2e-3);
        std::snprintf(w, sizeof w, "nu = %.2f: relative mass drift", nu);
        check(w, T.drift, 0.0, nu < 0.75 ? 1e-4 : 5e-3);
      }
    }
    if (sec.find('V') != std::string::npos) {
      std::printf("\nV. an exact Stokes mode with radial velocity, psi = A r J1(kr) sin(mz)\n");
      const double tol_a[3] = {0.10, 0.04, 0.02}, tol_r[3] = {0.15, 0.08, 0.06};
      int k = 0;
      for (Index R : {8, 16, 32}) {
        const Mode M = stokes_mode(R);
        std::printf("  R_w = %2d: steady amplitude %.4f, max |u - u_mode| / A %.4f, decay %.4f of nu K^2\n",
                    int(R), M.amp, M.emax, M.rate);
        std::snprintf(w, sizeof w, "R_w = %2d: steady amplitude", int(R));
        check(w, M.amp, 1.0, tol_a[k]);
        std::snprintf(w, sizeof w, "R_w = %2d: decay rate / nu K^2", int(R));
        check(w, M.rate, 1.0, tol_r[k]);
        ++k;
      }
    }
    if (sec.find('E') != std::string::npos) {
      std::printf("\nE. a prolate drop (a/b = 1.1) relaxing, Oh = 10, geometric exchange: rate / (20/19 sigma/(mu R))\n");
      for (Index R : {8, 16}) {
        if (kFP32 && R == 16) {   // FP32: rate 0.74, mass -1.7e-3 (measured)
          std::printf("  R = 16: not run in FP32 -- sigma/(mu R) per step is below its resolution there\n");
          continue;
        }
        const Rate Q = relax(R, 10.0, 1.1, true);
        std::printf("  R = %2d: R_vol %.3f  Oh %.3f  eps %.4f -> %.4f  rate ratio %.4f  rms ln %.1e  mass %+.1e\n",
                    int(R), Q.Rm, Q.Oh, Q.eps0, Q.epsN, Q.ratio, Q.rms, Q.drift);
        std::snprintf(w, sizeof w, "R = %2d: rate / Stokes rate", int(R));
        check(w, Q.ratio, 1.0, 0.03);
        std::snprintf(w, sizeof w, "R = %2d: relative mass drift", int(R));
        check(w, Q.drift, 0.0, kMass);
      }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\naxisym_fs: %s (%d failed), %.1f s\n", g_fail ? "FAIL" : "PASS", g_fail, secs);
    rc = g_fail ? 1 : 0;
  }
  Kokkos::finalize();
  return rc;
}
