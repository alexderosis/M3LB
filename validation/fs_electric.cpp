//==============================================================================
//  A conducting liquid film pulled by a non-uniform field, against the exact
//  linear response -- the acceptance test for validation/FsElectric.hpp.
//
//  MEASURED 2026-11-03, FP64, D2Q9 free surface + D2Q5 TRT potential
//  (tau_phi = 10, Lambda = 1/4), F = 0.4, kd = pi/2, delta = 0.24, 42 s.
//
//      row                                  value
//      A1. stress, flat conductor           mode 1 within 0.046 % (s=1),
//                                           0.011 % (s=2) of -2 C E0 E1; the
//                                           mean sits 1.3 % high, which is the
//                                           E1^2 term the linear theory drops
//      A2. stress, conductor at a cos kx    -1.14 % (s=1, a=2), -1.29 % (s=2,
//                                           a=4) of the feedback term alone --
//                                           the staircase conductor
//      B0. uniform field                    a flat surface stays flat: 5.8e-30
//      B1. frozen field                     -1.0654 against -1.1737: 9.2 % short
//      B2. live field                       -1.5920 against -1.9121: 16.7 % short;
//                                           live/frozen 1.494 against
//                                           1/(1-F) = 1.629
//          ... on the measured stiffness    -1.5920 against -1.6404: 2.95 %
//
//  WHAT THE ROWS SAY. A is the field and the stress map alone, no flow: it is
//  as good as the potential is converged (dphi ~ 1e-15) plus the staircase.
//  B1 is the free surface under a PRESCRIBED pattern of pull -- no field
//  dynamics at all -- so its 9 % is the free surface's, not this module's. B2's
//  last line removes it: B1 measures the restoring force the surface really
//  delivers, the field must take back F of the EXACT one, i.e. F eta_frozen /
//  eta_ow of the real one, and the live run then lands within 3 %. That is the
//  two-way coupling, verified against the surface that exists.
//
//  THE FREE-SURFACE FLOOR IS NOT THE ONE ALREADY RECORDED, AND IT IS NOT
//  DIAGNOSED. CLAUDE.md records that a free-surface deflection's error is set
//  by its size in CELLS (36 % at one cell, 1.4 % at eight, under gravity).
//  Here the frozen row read 9.2 % short at 1.07 cells (s=1) and 19.5 % short at
//  1.93 cells (s=2, every length doubled): the error GREW as the deflection in
//  cells grew. With gravity switched off, so that capillarity alone restores,
//  s=1 read 25 % short, so the excess stiffness is the CAPILLARY response to a
//  gentle, long-wave curvature (radius ~250 cells here), not the hydrostatic
//  one. The feedback check absorbs it either way -- +2.0 % with gravity off --
//  which is why B2 is gated on that and B1/B2's raw errors only as regression
//  guards. Quote this coupling's accuracy as the A rows and B2's 3 %; quote a
//  deflection only with the free surface's own error beside it.
//
//  THE FIRST VERSION OF THE FROZEN ROW READ 29 % SHORT FOR A REASON OF ITS OWN:
//  the stress map was written at the flat surface's interface cells only, and a
//  surface that moved more than a cell left its pull behind. It now carries the
//  stress down every column, as recoil.cpp does; FsElectric itself now writes
//  every cell that touches the gas, for the same reason.
//
//  FP32: all seven pass; B2's consistency reads 4.47 % against its 5 % gate,
//  the thinnest margin in the file.
//
//  -nrec re-measures the free surface with normal_reconstruction (2026-11-03):
//  the frozen row reads 21.6 % short at s = 1 and 27.3 % at s = 2 (9.2 / 19.5 %
//  without), and B2's consistency 8.9 % (2.95 %). It makes this floor worse.
//
//  C. THREE DIMENSIONS (2026-11-05), D3Q27 free surface + D3Q7 TRT potential,
//  the path FsElectric took for demonstrator/mew_cone3d. Sections A and B print
//  exactly what they printed before it, all 29 lines.
//      C1. A1 turned 90 degrees            the film's normal along z, the pattern
//          (ny = 4, uniform in y)          along x: mode and mean equal A1's to
//                                          the LAST BIT (0.0, not 1e-16), as they
//                                          must -- D3Q7 on a y-uniform field is
//                                          D2Q5's five-point problem
//      C2. conducting sphere in a uniform  e2 against the exact |E|^2 in the gas
//          field (N = 8a, sphere at 1/2)   within two cells: amplitude (least-
//                                          squares slope) 1.04123 at a = 5,
//                                          1.00131 at a = 10; mean error +0.0082
//                                          and +0.0003 of 3 E0; rms 0.0690 and
//                                          0.0707 of 3 E0
//  THE SPHERE'S SCATTER DOES NOT FALL WITH ITS RADIUS, AND THAT WAS A WRONG
//  PREDICTION. The first gates asked the rms to fall from a = 5 to 10 and to
//  sit under 5 %; it read 6.9 and 7.1 %. Each gas cell beside a staircase sees
//  the steps next to it, which do not shrink as the sphere grows, so the
//  cell-by-cell field carries a ~7 % scatter while its amplitude converges (an
//  error 31x smaller at twice the radius -- two points, so no order is
//  claimed). C2's gates are therefore regression guards set from this run: the
//  amplitude and the mean, which converge, and the scatter at its own level.
//  Deleting the z gradient from FsElectric fails all six C checks (C1 by 100 %,
//  the sphere's amplitude by 75 %). The run takes 48 s, C 9 s of it.
//
//  WHAT THIS DOES NOT TEST. A conductor whose potential is not uniform; a
//  surface sharper than the field's resolution (a cone tip); any field
//  stronger than F = 0.4 of the restoring force, where the linear theory and
//  the Tonks-Frenkel threshold at F = 1 take over; and the free surface's own
//  accuracy, which B1 measures and does not certify.
//
//  THE PROBLEM. A film of conducting liquid (potential 1) lies on the floor of
//  a periodic box under gravity and surface tension; a flat electrode a gap d
//  above its surface is held at delta cos(kx). Linearised, with eta cos(kx)
//  the surface deflection and C the stress coefficient (p_E = C |grad phi|^2),
//
//      eta (rho g + sigma k^2 - 2 C E0^2 k coth kd) = -2 C E0 k delta / sinh kd
//
//  with E0 = 1/d. Writing F = 2 C E0^2 k coth(kd) / (rho g + sigma k^2) for the
//  share of the restoring force the field takes back,
//
//      eta = - F d delta / ((1 - F) cosh kd).
//
//  The 1/(1 - F) is the TWO-WAY coupling: a crest moves closer to the
//  electrode, the field there grows, and the pull grows with it (the start of
//  the Tonks-Frenkel instability at F = 1). A field computed once on the flat
//  surface and frozen gives the one-way answer without it, so the frozen row is
//  a control the live row must differ from by exactly that factor.
//==============================================================================
#include "FsElectric.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace lbm;

using L  = D2Q9;
using PL = D2Q5;
using FS = FreeSurfaceSolver<L>;

namespace {

bool g_nrec = false;   // -nrec: FreeSurfaceSolver::normal_reconstruction

int failures = 0, checks = 0;
void verdict_le(const char* what, double got, double ceil_) {
  ++checks;
  const bool ok = got <= ceil_;
  if (!ok) ++failures;
  std::printf("  %-56s %11.4e   (want <= %.2e)   %s\n", what, got, ceil_, ok ? "PASS" : "FAIL");
}
void verdict_ge(const char* what, double got, double floor_) {
  ++checks;
  const bool ok = got >= floor_;
  if (!ok) ++failures;
  std::printf("  %-56s %11.4e   (want >= %.2e)   %s\n", what, got, floor_, ok ? "PASS" : "FAIL");
}

std::vector<double> surface(const FS& s, const Domain& d) {
  auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.fill());
  auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
  std::vector<double> h(std::size_t(d.nx), 0.0);
  for (Index x = 0; x < d.nx; ++x) {
    double col = 0;
    for (Index y = 1; y < d.ny - 1; ++y) {
      const std::uint8_t f = hf(d.id(x, y, 0));
      if (f == FsFluid) col += 1.0;
      else if (f == FsInterface) col += double(he(d.id(x, y, 0)));
    }
    h[std::size_t(x)] = col;
  }
  return h;
}

struct Result {
  double amp = 0, eta_tw = 0, eta_ow = 0, F = 0, d = 0;
  double last_change = 0, mass_drift = 0, max_pe = 0;
  long steps = 0;
  bool finite = true;
};

// s scales every length; F, kd and the gravity/capillary split are held fixed.
Result run(int s, double Ftarget, double delta, bool frozen, bool verbose,
           double gfrac = 1.0) {
  Result R;
  const Index Lx = 128 * s, h0 = 24 * s, d0 = 32 * s, Ly = h0 + d0 + 2;
  constexpr double cs2 = double(lbm::cs2<L, double>());
  const double k = 2.0 * M_PI / double(Lx);
  const double kd0 = k * double(d0);
  const double nu = 1.0 / 6.0;
  const double sigma = 1e-3;
  const double g = gfrac * sigma * k * k;       // rho g = gfrac sigma k^2
  const double restore = g + sigma * k * k;
  const double E0 = 1.0 / double(d0);
  const double C = Ftarget * restore / (2.0 * E0 * E0 * k / std::tanh(kd0));

  Domain d(Lx, Ly, 1, true, false, true);
  FS fs(d);
  fs.coll.omega = FS::omega_from_viscosity(Real(nu));
  fs.set_gravity(Real(0), Real(-g));
  fs.set_surface_tension(Real(sigma));
  fs.set_normal_reconstruction(g_nrec);
  fs.set_geometry([&](Index, Index y, Index) -> FsCell {
    return (y == 0 || y == Ly - 1) ? FsSolid : FsGas;
  });

  const Real rg0 = Real(1) - Real(C * E0 * E0 / cs2);   // the mean pull, seeded
  const Domain dd = d;
  const Index hy = d.hy;
  const Real hr = Real(h0), gr = Real(g);
  constexpr Real ics = inv_cs2<L, Real>();
  fs.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real y = Real(py - hy);
    Real e = hr - (y - Real(0.5));
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    const Real dz = hr - y;
    return typename FS::Seed{e, rg0 + (dz > Real(0) ? gr * dz * ics : Real(0))};
  });

  FsElectric<PL> el(d, Real(10));
  el.coef = Real(C);
  el.rho_G0 = Real(1);
  el.set_roles(
      [&](Index, Index y) { return (y == 0 || y == Ly - 1) ? ElecElectrode : ElecFree; },
      [&](Index x, Index y) -> Real {
        if (y == 0) return Real(1);
        return Real(delta * std::cos(k * double(x)));
      });
  el.refresh(fs);
  el.initialize(Real(0.5));
  el.relax(20000);
  el.apply(fs);
  // THE FROZEN CONTROL CARRIES ITS STRESS DOWN EVERY COLUMN, as recoil.cpp
  // does. Left at the cells around the flat surface it was computed on, a
  // surface that moved more than a cell left its own pull behind: the first
  // version of this row read 29 % short for that reason alone.
  if (frozen) {
    auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.rho_G_of);
    for (Index x = 0; x < Lx; ++x) {
      const Real v = hr(d.id(x, h0, 0));
      for (Index y = 1; y < Ly - 1; ++y) hr(d.id(x, y, 0)) = v;
    }
    Kokkos::deep_copy(fs.rho_G_of, hr);
  }

  // Overdamped capillary-gravity relaxation, softened by the field:
  // rate = omega0^2 / (2 nu k^2), omega0^2 = k tanh(kh) restore (1 - F).
  const double om2 = k * std::tanh(k * double(h0)) * restore * (1.0 - Ftarget);
  const double rate = std::fmin(2.0 * nu * k * k, om2 / (2.0 * nu * k * k));
  const long nst = long(10.0 / rate);
  const long M = 10;

  const double m0 = double(fs.total_mass());
  std::vector<double> h = surface(fs, d);
  auto amp_of = [&](const std::vector<double>& hh, double& mean) {
    mean = 0;
    for (double v : hh) mean += v;
    mean /= double(Lx);
    double a = 0;
    for (Index x = 0; x < Lx; ++x) a += (hh[std::size_t(x)] - mean) * std::cos(k * double(x));
    return 2.0 * a / double(Lx);
  };
  double mean = 0, prev = 0;
  for (long t = 0; t < nst; t += M) {
    for (long j = 0; j < M; ++j) fs.step();
    if (!frozen) {
      el.refresh(fs);
      el.relax(M);
      el.apply(fs);
    }
    if ((t / M) % 500 == 0 || t + M >= nst) {
      h = surface(fs, d);
      const double a = amp_of(h, mean);
      if (!std::isfinite(a)) { R.finite = false; return R; }
      R.last_change = std::fabs(a - prev);
      prev = a;
      if (verbose)
        std::printf("    t = %8ld   amp = %+.5f   mean h = %.4f\n", t + M, a, mean);
    }
  }
  R.steps = nst;
  h = surface(fs, d);
  R.amp = amp_of(h, mean);
  R.mass_drift = (double(fs.total_mass()) - m0) / m0;
  R.max_pe = el.max_stress();

  // Exact, at the gap the run actually settled to.
  R.d = double(Ly) - 2.0 - mean;
  const double kd = k * R.d, E = 1.0 / R.d;
  R.F = 2.0 * C * E * E * k / std::tanh(kd) / restore;
  R.eta_ow = -R.F * R.d * delta / std::cosh(kd);
  R.eta_tw = R.eta_ow / (1.0 - R.F);
  return R;
}

// THE STRESS MAP ALONE: no steps, the free surface only seeded. Two shapes.
//   shape 0: flat conductor, electrode at delta cos(kx)  -> mode 1 = -2 C E0 E1
//   shape 1: conductor surface at a cos(kx), flat electrode at 0
//            -> mode 1 = +2 C E0^2 k coth(kd) a, the feedback term by itself
// The exact values are evaluated at the conductor position the FIELD sees: the
// half link above the last cell with fill >= 1/2 (FsElectric's banner), which
// for the fill-1/2 seed of shape 0 is half a cell above the liquid's surface.
struct FieldResult { double mode = 0, exact = 0, mean = 0, mean_exact = 0, dphi = 0; };

FieldResult field_only(int s, double Ftarget, int shape, double amp) {
  const Index Lx = 128 * s, h0 = 24 * s, d0 = 32 * s, Ly = h0 + d0 + 2;
  constexpr double cs2 = double(lbm::cs2<L, double>());
  const double k = 2.0 * M_PI / double(Lx), kd0 = k * double(d0);
  const double sigma = 1e-3, g = sigma * k * k, restore = g + sigma * k * k;
  const double E0 = 1.0 / double(d0);
  const double C = Ftarget * restore / (2.0 * E0 * E0 * k / std::tanh(kd0));
  Domain d(Lx, Ly, 1, true, false, true);
  FS fs(d);
  fs.set_geometry([&](Index, Index y, Index) -> FsCell {
    return (y == 0 || y == Ly - 1) ? FsSolid : FsGas;
  });
  const Domain dd = d;
  const Index hy = d.hy, hx = d.hx;
  // shape 1 seeds the surface a quarter cell above a half-filled row, so the
  // staircase does not sit on the threshold everywhere at once
  const Real hr = Real(h0) + (shape ? Real(0.25) : Real(0));
  const Real ar = shape ? Real(amp) : Real(0), kr = Real(k);
  fs.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real Y = hr + ar * Kokkos::cos(kr * Real(px - hx));
    Real e = Y - (Real(py - hy) - Real(0.5));
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, Real(1)};
  });
  FsElectric<PL> el(d, Real(10));
  el.coef = Real(C);
  const double delta = shape ? 0.0 : amp;
  el.set_roles(
      [&](Index, Index y) { return (y == 0 || y == Ly - 1) ? ElecElectrode : ElecFree; },
      [&](Index x, Index y) -> Real {
        return y == 0 ? Real(1) : Real(delta * std::cos(k * double(x)));
      });
  el.refresh(fs);
  el.initialize(Real(0.5));
  el.relax(20000);
  el.apply(fs);
  el.relax(20000);
  FieldResult R;
  R.dphi = el.apply(fs);

  // The stress at each column's interface cell(s), averaged over them.
  auto hr2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.rho_G_of);
  auto hf  = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
  double a1 = 0, mean = 0;
  for (Index x = 0; x < Lx; ++x) {
    double pe = 0; int c = 0;
    for (Index y = 1; y < Ly - 1; ++y)
      if (hf(d.id(x, y, 0)) == FsInterface) { pe += (1.0 - double(hr2(d.id(x, y, 0)))) * cs2; ++c; }
    pe = c ? pe / c : 0.0;
    mean += pe; a1 += pe * std::cos(k * double(x));
  }
  R.mean = mean / double(Lx);
  R.mode = 2.0 * a1 / double(Lx);
  // The conductor the field sees: the mean of the staircase's half links.
  double ytop = 0;
  for (Index x = 0; x < Lx; ++x) {
    Index top = 0;
    for (Index y = 1; y < Ly - 1; ++y)
      if (el.cell(x, y) == ScalarDirichlet) top = y;
    ytop += double(top) + 0.5;
  }
  ytop /= double(Lx);
  const double df = double(Ly) - 1.5 - ytop, Ef = 1.0 / df, kdf = k * df;
  R.mean_exact = C * Ef * Ef;
  R.exact = shape ? 2.0 * C * Ef * Ef * k / std::tanh(kdf) * amp
                  : -2.0 * C * Ef * delta * k / std::sinh(kdf);
  return R;
}

//------------------------------------------------------------------------------
// C. THE THREE-DIMENSIONAL PATH (2026-11-05), D3Q27 free surface + D3Q7 TRT.
//------------------------------------------------------------------------------
using L3  = D3Q27;
using PL3 = D3Q7;
using FS3 = FreeSurfaceSolver<L3>;

// C1: A1 turned so that the film's normal is z and the electrode's pattern runs
// along x, on ny = 4 periodic planes. D3Q7's steady state on a field that does
// not depend on y is D2Q5's five-point one (anti-bounce-back included), and the
// 3 x 3 x 3 mean over three identical planes is the 3 x 3 one, so the answer
// must be A1's to round-off: the z axis, the z loops and the block in z are
// what is new, and they reproduce what the 2-D path validated.
FieldResult field_only3(int s, double Ftarget, double delta) {
  const Index Lx = 128 * s, Ny = 4, h0 = 24 * s, d0 = 32 * s, Lz = h0 + d0 + 2;
  constexpr double cs2 = double(lbm::cs2<L3, double>());
  const double k = 2.0 * M_PI / double(Lx), kd0 = k * double(d0);
  const double sigma = 1e-3, g = sigma * k * k, restore = g + sigma * k * k;
  const double E0 = 1.0 / double(d0);
  const double C = Ftarget * restore / (2.0 * E0 * E0 * k / std::tanh(kd0));
  Domain d(Lx, Ny, Lz, true, true, false);
  FS3 fs(d);
  fs.set_geometry([&](Index, Index, Index z) -> FsCell {
    return (z == 0 || z == Lz - 1) ? FsSolid : FsGas;
  });
  const Domain dd = d;
  const Index hz = d.hz;
  const Real hr = Real(h0);
  fs.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    Real e = hr - (Real(pz - hz) - Real(0.5));
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS3::Seed{e, Real(1)};
  });
  FsElectric<PL3> el(d, Real(10));
  el.coef = Real(C);
  el.set_roles(
      [&](Index, Index, Index z) { return (z == 0 || z == Lz - 1) ? ElecElectrode : ElecFree; },
      [&](Index x, Index, Index z) -> Real {
        return z == 0 ? Real(1) : Real(delta * std::cos(k * double(x)));
      });
  el.refresh(fs);
  el.initialize(Real(0.5));
  el.relax(20000);
  el.apply(fs);
  el.relax(20000);
  FieldResult R;
  R.dphi = el.apply(fs);
  auto hr2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.rho_G_of);
  auto hf  = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
  double a1 = 0, mean = 0;
  for (Index x = 0; x < Lx; ++x) {
    double pe = 0; int c = 0;
    for (Index y = 0; y < Ny; ++y)
      for (Index z = 1; z < Lz - 1; ++z)
        if (hf(d.id(x, y, z)) == FsInterface) { pe += (1.0 - double(hr2(d.id(x, y, z)))) * cs2; ++c; }
    pe = c ? pe / c : 0.0;
    mean += pe; a1 += pe * std::cos(k * double(x));
  }
  R.mean = mean / double(Lx);
  R.mode = 2.0 * a1 / double(Lx);
  double ztop = 0;
  for (Index x = 0; x < Lx; ++x) {
    Index top = 0;
    for (Index z = 1; z < Lz - 1; ++z)
      if (el.cell(x, 0, z) == ScalarDirichlet) top = z;
    ztop += double(top) + 0.5;
  }
  ztop /= double(Lx);
  const double df = double(Lz) - 1.5 - ztop, Ef = 1.0 / df;
  R.mean_exact = C * Ef * Ef;
  R.exact = -2.0 * C * Ef * delta * k / std::sinh(k * df);
  return R;
}

// C2: a conducting sphere of radius a cells in a uniform field, the one exact
// curved conductor in 3-D: |E|^2 = E0^2 [cos^2 t (1 + 2 a^3/r^3)^2
// + sin^2 t (1 - a^3/r^3)^2] at distance r from its centre, t from the field.
// Plates at z = 0 (potential 0) and z = N - 1 (1), zero-field frames on the
// four sides, the sphere at 1/2 on the mid-plane; N = 8a puts the nearest image
// 8a away, a correction of order (1/8)^3. Compared cell by cell over the gas
// within two cells of the sphere, in units of the polar surface field 3 E0.
struct SphereResult { double slope = 0, mean = 0, rms = 0, dphi = 0; long cells = 0; };

SphereResult sphere3(int a) {
  const Index N = 8 * a;
  Domain d(N, N, N, false, false, false);
  FS3 fs(d);
  fs.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  const double c = 0.5 * double(N - 1);        // the mid-plane, in cell-centre coordinates
  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy, hz = d.hz;
  const Real ar = Real(a), cr = Real(c);
  fs.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real X = Real(px - hx) - cr, Y = Real(py - hy) - cr, Z = Real(pz - hz) - cr;
    Real e = ar + Real(0.5) - Kokkos::sqrt(X * X + Y * Y + Z * Z);
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS3::Seed{e, Real(1)};
  });
  FsElectric<PL3> el(d, Real(10));
  el.coef = Real(1);
  el.liquid_value = Real(0.5);
  el.set_roles(
      [&](Index x, Index y, Index z) {
        if (z == 0 || z == N - 1) return ElecElectrode;
        if (x == 0 || x == N - 1 || y == 0 || y == N - 1) return ElecFrame;
        return ElecFree;
      },
      [&](Index, Index, Index z) -> Real { return z == N - 1 ? Real(1) : Real(0); });
  el.refresh(fs);
  el.initialize(Real(0.5));
  SphereResult R;
  R.dphi = 1;
  const double tol = sizeof(Real) == 4 ? 1e-6 : 1e-10;
  for (int it = 0; it < 40 && R.dphi > tol; ++it) { el.relax(1000); R.dphi = el.apply(fs); }
  const double E0 = 1.0 / double(N - 2), a3 = double(a) * a * a;
  double se = 0, se2 = 0, num = 0, den = 0;
  for (Index z = 1; z < N - 1; ++z)
    for (Index y = 1; y < N - 1; ++y)
      for (Index x = 1; x < N - 1; ++x) {
        if (el.cell(x, y, z) != ScalarBulk) continue;
        const double X = double(x) - c, Y = double(y) - c, Z = double(z) - c;
        const double r = std::sqrt(X * X + Y * Y + Z * Z);
        if (r > double(a) + 2.0) continue;
        const double ct2 = Z * Z / (r * r), q = a3 / (r * r * r);
        const double ex2 = E0 * E0 * (ct2 * (1 + 2 * q) * (1 + 2 * q) + (1 - ct2) * (1 - q) * (1 - q));
        const double e2 = double(el.e2(x, y, z));
        const double err = (std::sqrt(e2) - std::sqrt(ex2)) / (3.0 * E0);
        se += err; se2 += err * err; ++R.cells;
        num += e2 * ex2; den += ex2 * ex2;
      }
  R.mean = se / double(R.cells);
  R.rms = std::sqrt(se2 / double(R.cells));
  R.slope = num / den;                         // least-squares e2 = slope * exact^2
  return R;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    bool verbose = false;
    int only_s = 0, only_frozen = 0;
    double only_g = 1.0, only_delta = 0.24;
    for (int a = 1; a < argc; ++a) {
      if (!std::strcmp(argv[a], "-v")) { verbose = true; continue; }
      if (!std::strcmp(argv[a], "-nrec")) { g_nrec = true; continue; }
      if (!std::strcmp(argv[a], "-only") && a + 4 < argc) {   // s frozen gfrac delta
        only_s = std::atoi(argv[++a]); only_frozen = std::atoi(argv[++a]);
        only_g = std::atof(argv[++a]); only_delta = std::atof(argv[++a]);
        continue;
      }
      std::printf("fs_electric: unknown option '%s'\n", argv[a]);
      Kokkos::finalize();
      return 2;
    }
    std::printf("fs_electric: D2Q9 free surface + D2Q5 TRT potential, precision %s\n",
                sizeof(Real) == 4 ? "float" : "double");
    const double F = 0.4, delta = 0.24;
    if (only_s > 0) {
      Result r = run(only_s, F, only_delta, only_frozen != 0, verbose, only_g);
      const double ex = only_frozen ? r.eta_ow : r.eta_tw;
      std::printf("  s=%d frozen=%d g/(sigma k^2)=%.2f delta=%.3f: amp %+.4f exact %+.4f"
                  " err %+.4f  (one-way %+.4f two-way %+.4f F %.4f) steps %ld drift %.1e\n",
                  only_s, only_frozen, only_g, only_delta, r.amp, ex,
                  (r.amp - ex) / std::fabs(ex), r.eta_ow, r.eta_tw, r.F, r.steps, r.mass_drift);
      Kokkos::finalize();
      return 0;
    }

    const bool fp32 = sizeof(Real) == 4;

    // ---- A. the stress map alone -------------------------------------------
    std::printf("\n-- A1. flat conductor, electrode at delta cos(kx): -2 C E0 E1 --\n");
    double worstA1 = 0;
    for (int sf : {1, 2}) {
      const FieldResult f = field_only(sf, F, 0, delta);
      const double e = (f.mode - f.exact) / std::fabs(f.exact);
      worstA1 = std::max(worstA1, std::fabs(e));
      std::printf("  s=%d  mode %.5e  exact %.5e  err %+.5f   mean %.5e (C E0^2 %.5e)  dphi %.1e\n",
                  sf, f.mode, f.exact, e, f.mean, f.mean_exact, f.dphi);
    }
    verdict_le("A1: stress mode 1 vs exact, worst |err|", worstA1, fp32 ? 2e-2 : 5e-3);

    std::printf("\n-- A2. conductor surface at a cos(kx), flat electrode: 2 C E0^2 k coth(kd) a --\n");
    double worstA2 = 0;
    for (int sf : {1, 2}) {
      const FieldResult f = field_only(sf, F, 1, 2.0 * sf);
      const double e = (f.mode - f.exact) / std::fabs(f.exact);
      worstA2 = std::max(worstA2, std::fabs(e));
      std::printf("  s=%d a=%d  mode %.5e  exact %.5e  err %+.5f\n", sf, 2 * sf, f.mode, f.exact, e);
    }
    verdict_le("A2: feedback term vs exact, worst |err|", worstA2, 0.05);

    // ---- B. coupled to the free surface ------------------------------------
    std::printf("\n-- B0. uniform field: a flat surface must stay flat --\n");
    Result r0 = run(1, F, 0.0, false, verbose);
    std::printf("  amp = %+.3e cells   mass drift %.2e   steps %ld\n", r0.amp, r0.mass_drift,
                r0.steps);
    verdict_le("B0: |amp| of a flat surface in a uniform field [cells]", std::fabs(r0.amp), 1e-6);

    std::printf("\n-- B1. frozen field: the free surface under a prescribed pull --\n");
    Result r1 = run(1, F, delta, true, verbose);
    std::printf("  amp = %+.4f   one-way exact %+.4f   err %+.4f   F = %.4f  d = %.3f\n",
                r1.amp, r1.eta_ow, (r1.amp - r1.eta_ow) / std::fabs(r1.eta_ow), r1.F, r1.d);

    std::printf("\n-- B2. live field: the two-way coupling --\n");
    Result r2 = run(1, F, delta, false, verbose);
    std::printf("  amp = %+.4f   two-way exact %+.4f   err %+.4f\n",
                r2.amp, r2.eta_tw, (r2.amp - r2.eta_tw) / std::fabs(r2.eta_tw));
    // THE FEEDBACK, JUDGED AGAINST THE SURFACE THAT EXISTS. B1 measures the
    // restoring force the free surface really delivers: eta_frozen = eta_ow / s
    // with s its stiffness relative to the exact one. The field then takes back
    // the share F / s of THAT force, and the live answer must be
    //     eta_frozen / (1 - F eta_frozen / eta_ow).
    const double ratio = r1.amp / r1.eta_ow;
    const double pred = r1.amp / (1.0 - r1.F * ratio);
    const double epred = (r2.amp - pred) / std::fabs(pred);
    std::printf("  live / frozen = %.4f  (exact 1/(1-F) = %.4f)\n", r2.amp / r1.amp, 1.0 / (1.0 - r1.F));
    std::printf("  live against the frozen run's own stiffness: pred %+.4f  err %+.4f\n", pred, epred);
    verdict_le("B2: live vs feedback on the measured stiffness, |err|", std::fabs(epred), 0.05);
    verdict_ge("B2: live / frozen (the feedback is live)", r2.amp / r1.amp, 1.3);
    // Regression guards on the raw free-surface floor, NOT derived: see banner.
    verdict_le("B1: |frozen - exact| / exact (free-surface floor)",
               std::fabs((r1.amp - r1.eta_ow) / r1.eta_ow), 0.15);
    verdict_le("B2: |live - exact| / exact (free-surface floor)",
               std::fabs((r2.amp - r2.eta_tw) / r2.eta_tw), 0.25);
    std::printf("  mass drift: B0 %.1e  B1 %.1e  B2 %.1e\n", r0.mass_drift, r1.mass_drift,
                r2.mass_drift);

    // ---- C. three dimensions: D3Q27 free surface + D3Q7 potential ------------
    std::printf("\n-- C1. A1 in 3-D (normal z, pattern along x, uniform in y): A1 to round-off --\n");
    {
      const FieldResult f2 = field_only(1, F, 0, delta);
      const FieldResult f3 = field_only3(1, F, delta);
      const double e3 = (f3.mode - f3.exact) / std::fabs(f3.exact);
      const double dmode = std::fabs(f3.mode - f2.mode) / std::fabs(f2.mode);
      const double dmean = std::fabs(f3.mean - f2.mean) / std::fabs(f2.mean);
      std::printf("  3-D mode %.10e  2-D mode %.10e  exact %.5e  err %+.5f   dphi %.1e\n",
                  f3.mode, f2.mode, f3.exact, e3, f3.dphi);
      std::printf("  3-D mean %.10e  2-D mean %.10e\n", f3.mean, f2.mean);
      verdict_le("C1: |3-D - 2-D| / 2-D, stress mode 1", dmode, fp32 ? 1e-4 : 1e-10);
      verdict_le("C1: |3-D - 2-D| / 2-D, mean stress", dmean, fp32 ? 1e-4 : 1e-10);
    }

    std::printf("\n-- C2. conducting sphere in a uniform field: |E| within 2 cells, per 3 E0 --\n");
    {
      // Gates set from the first run and NOT derived (banner): the amplitude and
      // the mean converge, the cell-by-cell scatter does not.
      const SphereResult q5 = sphere3(5), q10 = sphere3(10);
      for (const auto& [a, q] : {std::pair{5, q5}, std::pair{10, q10}})
        std::printf("  a = %2d  N = %3d  cells %6ld  slope(e2 vs exact^2) %.5f  mean %+.5f  rms %.5f"
                    "  dphi %.1e\n", a, 8 * a, q.cells, q.slope, q.mean, q.rms, q.dphi);
      verdict_le("C2: a = 10, |slope - 1| (the field's amplitude)", std::fabs(q10.slope - 1.0), 0.01);
      verdict_le("C2: |slope - 1| at a = 10 over a = 5 (it falls)",
                 std::fabs(q10.slope - 1.0) / std::fabs(q5.slope - 1.0), 0.5);
      verdict_le("C2: a = 10, |mean |E| error| / (3 E0)", std::fabs(q10.mean), 0.005);
      verdict_le("C2: rms |E| error / (3 E0), both radii (staircase)", std::max(q5.rms, q10.rms), 0.08);
    }
    std::printf("\n%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
