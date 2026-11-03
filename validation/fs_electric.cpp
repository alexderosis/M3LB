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
    std::printf("\n%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
