//==============================================================================
//  A SESSILE DROP: the contact-angle law of FreeSurfaceSolver against the one
//  shape it must produce.
//
//  WHY THIS CASE EXISTS. demonstrator/mew_nozzle's grid ladder (2026-11-03)
//  found the melt leaving the nozzle through a neck that is the bore plus four
//  cells on every grid -- 12, 20 and 36 cells at 50, 25 and 12.5 um -- so the
//  column width, the front speed and the contact time all moved with the cell
//  and nothing converged. The free surface had no contact angle: its curvature
//  read every wall as gas, and where a melt that does not wet lets go of a
//  nozzle is decided by the grid. The law added for it
//  (FreeSurfaceSolver::set_contact_angle) needs a test with an answer.
//
//  THE REFERENCE IS EXACT. Without gravity a 2-D drop on a flat wall comes to
//  rest as a circular segment meeting the wall at the prescribed angle theta,
//  with area and height
//
//      A = R^2 (theta - sin theta cos theta),     h = R (1 - cos theta),
//
//  so A / h^2 = (theta - sin theta cos theta) / (1 - cos theta)^2, which falls
//  monotonically from infinity to pi/4 over (0, pi) and is inverted for theta by
//  bisection. Both inputs are integrals of the fill level -- A over the whole
//  drop, h over the column through its apex, measured from the halfway wall --
//  so neither needs an interface-position convention, and neither reads the
//  contact line itself, which a one-cell interface resolves only to a cell. A
//  circle fitted to the column heights of the R0 = 16, 60-degree drop agrees
//  (63.0 against 63.4 degrees, rms 0.07 cells).
//
//  EACH ANGLE IS APPROACHED FROM BOTH SIDES, AND THAT IS THE MEASUREMENT. The
//  first version seeded a semicircle and every angle stopped short of its
//  target on the 90-degree side. Seeded as a cap of the same area 15 degrees
//  below and 15 above, the drop stops at the two ends of a WINDOW: every
//  configuration in it is static (spurious currents 7e-9 to 6e-6 at 60-120
//  degrees). Measured, sigma = 0.01, nu = 1/6, 2500 R0 steps (low end / high
//  end, degrees):
//
//      theta    R0 = 8             R0 = 16            R0 = 32
//        60     58.630 / 62.612    57.578 / 63.408    53.383 / 67.122
//        90     83.204 / 90.932    83.924 / 93.518    81.420 / 96.420
//       120    117.142 / 121.312  115.501 / 125.225  111.263 / 132.862
//       150    137.974 / 149.730  137.140 / 163.606  134.792 / 154.238
//
//  The prescribed angle lies inside every window at 60, 90 and 120, and the
//  midpoints are within 3 degrees of it on all three grids (worst -2.93, 90
//  degrees at R0 = 8). THE WINDOW WIDENS
//  WITH RESOLUTION -- half-widths 2.0 / 2.9 / 6.9 degrees at 60, 3.9 / 4.8 /
//  7.5 at 90, 2.1 / 4.9 / 10.8 at 120 -- so this is not a discretisation error
//  that refinement removes, and a finer grid pins the contact line harder. What
//  it is: kappa is the only thing the law touches, and in a static state kappa
//  is uniform over the interface; the one-pass smoothing leaves fill patterns
//  that change the true curvature without changing kappa (the R0 = 16, 60-degree
//  drop is a circle of R 24.0 while the solver reads kappa = 1/26.5 at every
//  interface cell, contact line included), so the contact line can stop
//  wherever those patterns absorb the mismatch. A fixed absolute tolerance in
//  kappa is a relative one that grows with R, which is the trend measured. Not
//  proven here; surface_tension -rate measures the same mechanism on a free drop
//  with no wall (2026-11-04). The cure is not known.
//
//  150 DEGREES is lopsided -- midpoint -6.1 / +0.4 / -5.5 at R0 = 8 / 16 / 32,
//  and at R0 = 8 the window misses 150 by 0.27 -- and 30 DEGREES does not
//  settle: its low end is still moving when the run ends, and on exact
//  30-degree caps (-static) the curvature next to the wall reads kappa R =
//  1.16 / 1.49 / 2.05 at R = 12 / 24 / 48, growing with R. Both are reported
//  and not gated. At 60-120 degrees the same static test reads 0.97-0.99 at the
//  top of the cap, against 0.995-1.007 for a free circle.
//
//  THE PRESSURE IS REPORTED, NOT GATED. dp R / sigma, with R from the measured
//  A and theta, is 1 for a cap at rest as a cap, the reference
//  validation/surface_tension.cpp uses. Here it is -0.13 to +0.25 off at
//  60-120 degrees, high at the low end of a window and low at the high end in
//  eight of the nine pairs, because a drop stopped at the edge of its window
//  is static with a curvature the shape does not have.
//
//  LAW OFF (the default), the drop recedes from the semicircle to 149.4 /
//  151.7 / 149.0 degrees at R0 = 8 / 16 / 32: a wall read as gas is
//  non-wetting at about 150 degrees, not the 180 the solver's banner said.
//
//  Lattice: D2Q9 with D2Q9 curvature, as demonstrator/mew_nozzle and the other
//  free-surface cases. -lat d3q27 runs the same drop at nz = 1 and reproduced
//  D2Q9 to every printed digit (R0 = 8, 60 degrees, seeded at 75).
//
//  -hf (2026-11-04) runs it with FreeSurfaceSolver::height_function, the
//  curvature that stops a free drop pinning (surface_tension -rate). Cells
//  whose height stencil meets the wall keep the default curvature, and with it
//  this law, and the mixture FAILS A GATE: at 60 / 90 / 120 / 150 degrees the
//  drops stop at 52.59-63.45 / 86.41-88.72 / 116.35-121.28 / 133.69-143.66,
//  half-widths 5.4 / 1.2 / 2.5 / 5.0 against the default's 2.9 / 4.8 / 4.9 /
//  13.2, every midpoint low by 1.2 to 11.3 degrees, and 90 lies 1.3 outside
//  its window. All eight of those drops end at rest (spurious 3e-16 to 6e-10).
//  That the bias comes from the cells by the wall, where the two curvatures
//  meet, is a reading and not a measurement. A contact angle with height
//  functions needs ghost heights at the wall; until it has them -hf is a probe
//  here, not a model.
//
//  The default run (ctest) is R0 = 16 at 40000 steps; -ladder adds R0 = 8 and
//  32, ungated; -static runs the estimator test above; -theta/-seed/-R0 make
//  one run, with -series and -profile to watch it. The gates were measured in
//  FP64; the FP32 build passes every one, its window ends within 0.06 degrees
//  of FP64's and its worst mass drift 6.8e-4 against 2.7e-5.
//==============================================================================

#include "core/Types.hpp"
#include "solver/FreeSurfaceSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

namespace {

bool g_profile = false;   // -profile: print the column heights h(x) at the end
bool g_static = false;    // -static: the curvature estimator alone, nothing stepped
bool g_hf = false;        // -hf: FreeSurfaceSolver::height_function (2026-11-04)

struct Result {
  double A = 0, h = 0, theta = 0, R = 0, dp = 0, laplace = 0;
  double umax = 0, mass_drift = 0, dtheta_last = 0, base1 = 0, ktop = 0;
  bool finite = true;
};

// A / h^2 for a circular segment of contact angle t (radians).
double ratio(double t) {
  const double c = std::cos(t), s = std::sin(t);
  return (t - s * c) / ((1.0 - c) * (1.0 - c));
}

// Invert ratio() by bisection; it is monotone decreasing on (0, pi).
double theta_from(double A, double h) {
  const double q = A / (h * h);
  double lo = 1e-6, hi = M_PI - 1e-9;
  if (q >= ratio(lo)) return lo;
  if (q <= ratio(hi)) return hi;
  for (int k = 0; k < 200; ++k) {
    const double mid = 0.5 * (lo + hi);
    if (ratio(mid) > q) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

struct Shape { double A, h, base1; };

template <class FS>
Shape shape(const FS& s, const Domain& d, Index nx, Index ny) {
  auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
  auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.fill());
  Shape sh{0, 0, 0};
  for (Index x = 0; x < nx; ++x) {
    double col = 0;
    for (Index y = 1; y < ny - 1; ++y) {
      const Index n = d.id(x, y);
      const std::uint8_t f = hf(n);
      const double c = (f == FsFluid) ? 1.0 : (f == FsInterface ? double(he(n)) : 0.0);
      col += c;
      if (y == 1) sh.base1 += c;
    }
    sh.A += col;
    if (col > sh.h) sh.h = col;
  }
  return sh;
}

// theta_deg <= 0 runs with the law OFF: the wall reads as gas, the default.
template <class L>
Result run(double R0, double theta_deg, double seed_deg, double sigma, double nu,
           std::size_t nsteps, bool series) {
  using FS = FreeSurfaceSolver<L>;
  Result Rr;
  constexpr double cs2d = double(lbm::cs2<L, double>());
  const Index nx = Index(8.0 * R0), ny = Index(2.0 * R0);

  Domain d(nx, ny, 1, true, false, true);      // periodic along the wall
  FS s(d);
  s.coll.omega = FS::omega_from_viscosity(Real(nu));
  s.set_surface_tension(Real(sigma));
  s.set_height_function(g_hf);
  if (theta_deg > 0) s.set_contact_angle(Real(theta_deg * M_PI / 180.0));
  const Index nyi = ny;
  s.set_geometry([&](Index, Index y, Index) -> FsCell {
    return (y == 0 || y == nyi - 1) ? FsSolid : FsGas;
  });

  // A circular cap on the halfway wall (y = 0.5 between solid row 0 and fluid
  // row 1) meeting it at seed_deg, with the area of a semicircle of radius R0,
  // at uniform density: the pressure jump is part of the answer. seed_deg = 90
  // is the semicircle itself.
  const double ts = seed_deg * M_PI / 180.0;
  const double Rs = R0 * std::sqrt(0.5 * M_PI / (ts - std::sin(ts) * std::cos(ts)));
  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  const Real Rr0 = Real(Rs), xc = Real(nx) * Real(0.5);
  const Real yc = Real(0.5 - Rs * std::cos(ts));
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real x = Real(px - hx) - xc, y = Real(py - hy) - yc;
    const Real r = Kokkos::sqrt(x * x + y * y);
    Real e = Rr0 + Real(0.5) - r;
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, Real(1)};
  });

  const double m0 = double(s.total_mass());
  const std::size_t chunk = nsteps / 20 > 0 ? nsteps / 20 : 1;
  double th_prev = 0;
  for (std::size_t t = 0; t < nsteps; t += chunk) {
    for (std::size_t k = 0; k < chunk && t + k < nsteps; ++k) s.step();
    const Shape sh = shape(s, d, nx, ny);
    const double th = theta_from(sh.A, sh.h) * 180.0 / M_PI;
    if (series)
      std::printf("      step %7zu  A %9.3f  h %7.3f  base row 1 %8.3f  theta %8.3f\n",
                  t + chunk, sh.A, sh.h, sh.base1, th);
    Rr.dtheta_last = th - th_prev;
    th_prev = th;
  }
  const double m1 = double(s.total_mass());
  Rr.mass_drift = (m1 - m0) / m0;
  if (!std::isfinite(m1)) { Rr.finite = false; return Rr; }

  const Shape sh = shape(s, d, nx, ny);
  if (g_profile) {
    auto pf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
    auto pe = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.fill());
    for (Index x = 0; x < nx; ++x) {
      double col = 0;
      for (Index y = 1; y < ny - 1; ++y) {
        const std::uint8_t f = pf(d.id(x, y));
        col += (f == FsFluid) ? 1.0 : (f == FsInterface ? double(pe(d.id(x, y))) : 0.0);
      }
      if (col > 0) std::printf("      profile x %4d  h %8.4f\n", int(x), col);
    }
    // The topmost liquid cells of each column: flag, fill and curvature.
    auto pk = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.kappa());
    for (Index x = 0; x < nx; ++x) {
      Index top = 0;
      for (Index y = 1; y < ny - 1; ++y) {
        const std::uint8_t f = pf(d.id(x, y));
        if (f == FsFluid || f == FsInterface) top = y;
      }
      if (top == 0) continue;
      std::printf("      top x %4d:", int(x));
      for (Index y = top - 2; y <= top; ++y) {
        if (y < 1) continue;
        const Index n = d.id(x, y);
        std::printf("  y %3d %c fill %.4f kappa %+.5f", int(y),
                    pf(n) == FsFluid ? 'F' : (pf(n) == FsInterface ? 'I' : 'G'),
                    double(pe(n)), double(pk(n)));
      }
      std::printf("\n");
    }
  }
  Rr.A = sh.A;  Rr.h = sh.h;  Rr.base1 = sh.base1;
  const double th = theta_from(sh.A, sh.h);
  Rr.theta = th * 180.0 / M_PI;
  Rr.R = std::sqrt(sh.A / (th - std::sin(th) * std::cos(th)));

  // Bulk pressure: Fluid cells whose whole 5x5 block is Fluid, away from the
  // interface whose curvature is being tested and away from the wall.
  auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
  auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto huy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  double rsum = 0; long rn = 0;
  for (Index y = 3; y < ny - 3; ++y)
    for (Index x = 2; x < nx - 2; ++x) {
      bool bulk = true;
      for (Index b = -2; b <= 2 && bulk; ++b)
        for (Index a = -2; a <= 2 && bulk; ++a)
          bulk = hf(d.id(x + a, y + b)) == FsFluid;
      if (!bulk) continue;
      rsum += double(hr(d.id(x, y)));  ++rn;
    }
  Rr.dp = (rn > 0) ? cs2d * (rsum / double(rn) - 1.0) : 0.0;   // rho_G = 1
  Rr.laplace = Rr.dp * Rr.R / sigma;
  // The solver's own curvature over the upper half of the interface, times
  // the fitted R: 1 for a circular cap, whatever the pressure reads.
  {
    auto hk = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.kappa());
    double ks = 0; long kn = 0;
    for (Index y = 1; y < ny - 1; ++y)
      for (Index x = 0; x < nx; ++x) {
        const Index n = d.id(x, y);
        if (hf(n) != FsInterface || double(y) - 0.5 < 0.5 * sh.h) continue;
        ks += double(hk(n));  ++kn;
      }
    Rr.ktop = (kn > 0) ? ks / double(kn) * Rr.R : 0.0;
  }
  for (Index y = 1; y < ny - 1; ++y)
    for (Index x = 0; x < nx; ++x) {
      const Index n = d.id(x, y);
      if (hf(n) != FsFluid && hf(n) != FsInterface) continue;
      const double u = std::hypot(double(hux(n)), double(huy(n)));
      if (!std::isfinite(u)) Rr.finite = false;
      if (u > Rr.umax) Rr.umax = u;
    }
  return Rr;
}

// -static: the curvature ESTIMATOR alone, on exact area fractions (16 x 16
// samples a cell) of a cap of radius R meeting the wall at theta, or of a free
// circle of the same R (wall = false). Nothing is stepped.
template <class L>
void static_kappa(double R, double theta_deg, bool wall, double sigma) {
  using FS = FreeSurfaceSolver<L>;
  const Index nx = Index(8.0 * R), ny = wall ? Index(2.0 * R) : Index(3.0 * R);
  Domain d(nx, ny, 1, true, !wall, true);
  FS s(d);
  s.coll.omega = FS::omega_from_viscosity(Real(1.0 / 6.0));
  s.set_surface_tension(Real(sigma));
  s.set_height_function(g_hf);
  if (wall) s.set_contact_angle(Real(theta_deg * M_PI / 180.0));
  const Index nyi = ny;
  s.set_geometry([&](Index, Index y, Index) -> FsCell {
    return (wall && (y == 0 || y == nyi - 1)) ? FsSolid : FsGas;
  });
  const double th = theta_deg * M_PI / 180.0;
  const double xc = 0.5 * double(nx);
  const double yc = wall ? 0.5 - R * std::cos(th) : 0.5 * double(ny);
  std::vector<Real> frac(std::size_t(nx) * std::size_t(ny), Real(0));
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      int in = 0;
      for (int b = 0; b < 16; ++b)
        for (int a = 0; a < 16; ++a) {
          const double px = double(x) - 0.5 + (a + 0.5) / 16.0;
          const double py = double(y) - 0.5 + (b + 0.5) / 16.0;
          if (wall && py < 0.5) continue;
          if ((px - xc) * (px - xc) + (py - yc) * (py - yc) < R * R) ++in;
        }
      frac[std::size_t(y) * std::size_t(nx) + std::size_t(x)] = Real(in) / Real(256);
    }
  View1D<Real> fv("frac", d.n_padded);
  auto hfv = Kokkos::create_mirror_view(fv);
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) hfv(d.id(x, y)) = frac[std::size_t(y) * std::size_t(nx) + std::size_t(x)];
  Kokkos::deep_copy(fv, hfv);
  s.initialize(KOKKOS_LAMBDA(Index n) { return typename FS::Seed{fv(n), Real(1)}; });
  s.compute_curvature();
  auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
  auto hk = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.kappa());
  double top = 0, low = 0, all = 0; long nt = 0, nl = 0, na = 0;
  const double ytop = wall ? 0.5 + R * (1.0 - std::cos(th)) : yc + R;
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      const Index n = d.id(x, y);
      if (hf(n) != FsInterface) continue;
      const double k = double(hk(n));
      all += k; ++na;
      if (double(y) > ytop - 0.5 * R * (1.0 - std::cos(th > 1.5708 ? 1.5708 : th))) { top += k; ++nt; }
      if (wall && y <= 2) { low += k; ++nl; }
    }
  std::printf("   static %s R %.1f theta %.0f:  kappa R  top %.4f  all %.4f%s\n",
              wall ? "cap   " : "circle", R, theta_deg, nt ? top / nt * R : 0.0, na ? all / na * R : 0.0,
              wall ? (std::string("  rows 1-2 ") + std::to_string(nl ? low / nl * R : 0.0)).c_str() : "");
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t nsteps = 40000;
  double sigma = 0.01, theta1 = -1, R1 = 16, seed1 = 90;
  bool series = false, ladder = false, d3 = false;
  g_profile = false;
  for (int i = 1; i < argc; ++i) {
    if      (!std::strcmp(argv[i], "-steps") && i + 1 < argc) nsteps = std::size_t(std::atof(argv[++i]));
    else if (!std::strcmp(argv[i], "-sigma") && i + 1 < argc) sigma = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-theta") && i + 1 < argc) theta1 = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-R0") && i + 1 < argc) R1 = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-seed") && i + 1 < argc) seed1 = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-series")) series = true;
    else if (!std::strcmp(argv[i], "-profile")) g_profile = true;
    else if (!std::strcmp(argv[i], "-static")) g_static = true;
    else if (!std::strcmp(argv[i], "-ladder")) ladder = true;
    else if (!std::strcmp(argv[i], "-hf")) g_hf = true;
    else if (!std::strcmp(argv[i], "-lat") && i + 1 < argc) {
      const char* l = argv[++i];
      if (!std::strcmp(l, "d3q27")) d3 = true;
      else if (std::strcmp(l, "d2q9")) {
        std::fprintf(stderr, "sessile_drop: -lat must be d2q9 or d3q27. NOTHING WAS RUN.\n");
        return 2;
      }
    }
    else if (!std::strncmp(argv[i], "--kokkos", 8)) {}
    else {
      std::fprintf(stderr, "sessile_drop: unknown option '%s'. NOTHING WAS RUN.\n"
                           "  -steps N  -sigma S  -theta DEG (one run; 0 = law off)  -seed DEG  -R0 R\n"
                           "  -series  -ladder  -lat d2q9|d3q27  -hf\n", argv[i]);
      return 2;
    }
  }

  Kokkos::initialize(argc, argv);
  int failures = 0, checks = 0;
  {
    const double nu = 1.0 / 6.0;     // omega = 1, for the reason recoil.cpp gives
    std::printf("sessile drop: the contact angle of a 2-D drop at rest on a wall\n");
    std::printf("backend %s   precision %s   lattice %s   sigma %.4g   nu %.4f   %zu steps\n\n",
                Kokkos::DefaultExecutionSpace::name(), precision_name(),
                d3 ? "D3Q27 (nz = 1)" : "D2Q9", sigma, nu, nsteps);
    // The ladder scales the run with the drop: relaxation slows as R0 grows.
    auto go = [&](double R0, double th, double seed) {
      const std::size_t n = ladder ? std::size_t(double(nsteps) * R0 / 16.0) : nsteps;
      return d3 ? run<D3Q27>(R0, th, seed, sigma, nu, n, series)
                : run<D2Q9>(R0, th, seed, sigma, nu, n, series);
    };
    auto row = [&](double R0, double th, double seed, const Result& r) {
      std::printf("   %5.0f %5.0f %7s  %9.3f %8.3f %9.3f %+8.3f %+9.4f %+9.4f %10.3e %+10.2e %+9.4f %s\n",
                  R0, seed, th > 0 ? (std::to_string(int(th)) + "").c_str() : "off",
                  r.A, r.h, r.theta, th > 0 ? r.theta - th : 0.0, r.laplace - 1.0, r.ktop - 1.0,
                  r.umax, r.mass_drift, r.dtheta_last, r.finite ? "" : "NON-FINITE");
    };
    std::printf("   %5s %5s %7s  %9s %8s %9s %8s %9s %9s %10s %10s %9s\n", "R0", "seed", "theta",
                "A", "h", "theta_m", "error", "dpR/s-1", "kR-1 top", "spurious", "mass", "last dth");
    if (g_static) {
      for (double R : {12.0, 24.0, 48.0}) {
        static_kappa<D2Q9>(R, 90.0, false, sigma);
        for (double th : {30.0, 60.0, 90.0, 120.0}) static_kappa<D2Q9>(R, th, true, sigma);
      }
    } else if (theta1 >= 0) {
      const Result r = go(R1, theta1, seed1);
      row(R1, theta1, seed1, r);
    } else {
      auto check = [&](const char* what, bool ok, double got) {
        ++checks;
        if (!ok) ++failures;
        std::printf("  %-58s %10.4g   %s\n", what, got, ok ? "PASS" : "FAIL");
      };
      const std::vector<double> radii = ladder ? std::vector<double>{8, 16, 32}
                                               : std::vector<double>{R1};
      const double angles[5] = {30, 60, 90, 120, 150};
      for (double R0 : radii) {
        double lo[5], hi[5], mass_worst = 0;
        bool finite = true;
        for (int k = 0; k < 5; ++k) {
          const double th = angles[k];
          const Result a = go(R0, th, th - 15.0);
          row(R0, th, th - 15.0, a);
          const Result b = go(R0, th, th + 15.0);
          row(R0, th, th + 15.0, b);
          lo[k] = std::min(a.theta, b.theta);
          hi[k] = std::max(a.theta, b.theta);
          finite = finite && a.finite && b.finite;
          if (k > 0 && k < 4) mass_worst = std::max(mass_worst,
                                           std::max(std::fabs(a.mass_drift), std::fabs(b.mass_drift)));
        }
        const Result off = go(R0, -1, 90.0);
        row(R0, -1, 90.0, off);
        std::printf("\n   R0 = %.0f: where each drop stops, seeded 15 degrees below and above\n", R0);
        std::printf("   %7s %9s %9s %12s %10s\n", "theta", "low end", "high end", "mid - theta", "half");
        for (int k = 0; k < 5; ++k)
          std::printf("   %7.0f %9.3f %9.3f %+12.3f %10.3f\n", angles[k], lo[k], hi[k],
                      0.5 * (lo[k] + hi[k]) - angles[k], 0.5 * (hi[k] - lo[k]));
        if (ladder) continue;
        // THE GATES WERE SET FROM THE MEASUREMENT AND NOT BEFORE IT, at R0 = 16
        // and 40000 steps, so they are regression guards on what the law does
        // there and not a convergence claim: the window WIDENS with R0 (banner).
        // 60, 90 and 120 degrees are gated; 30 and 150 are reported only, the
        // first because its low end is still moving when the run ends, the
        // second because its window is lopsided (midpoint -6 deg at R0 = 8 and
        // 32, +0.4 at 16), so a gate there would pass here by coincidence.
        std::printf("\n");
        check("every run finite", finite, finite ? 1.0 : 0.0);
        for (int k = 1; k < 4; ++k) {
          char buf[96];
          std::snprintf(buf, sizeof buf, "%.0f deg lies inside its window", angles[k]);
          check(buf, lo[k] <= angles[k] && angles[k] <= hi[k], angles[k]);
          std::snprintf(buf, sizeof buf, "%.0f deg: window midpoint within 3 deg", angles[k]);
          const double mid = 0.5 * (lo[k] + hi[k]) - angles[k];
          check(buf, std::fabs(mid) <= 3.0, mid);
          std::snprintf(buf, sizeof buf, "%.0f deg: window half-width within 5.5 deg", angles[k]);
          check(buf, 0.5 * (hi[k] - lo[k]) <= 5.5, 0.5 * (hi[k] - lo[k]));
        }
        check("law off: the wall reads as gas, the drop sits above 140 deg",
              off.theta >= 140.0, off.theta);
        check("mass drift, 60-120 deg runs, within 1e-3", mass_worst <= 1e-3, mass_worst);
      }
    }
  }
  Kokkos::finalize();
  std::printf("\n[sessile_drop] %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
