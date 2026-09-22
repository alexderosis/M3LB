//==============================================================================
//  SURFACE TENSION ON A SHARP FREE SURFACE: Laplace's law, and the spurious
//  current that comes with it.
//
//  STAGE 6 of a coupled melt-pool model. Stage 5 (validation/recoil.cpp) gave
//  FreeSurfaceSolver a per-node gas pressure and measured it against the exact
//  hydrostatic depression; what it explicitly did NOT deliver was surface
//  tension, because p_G = p_atm + sigma kappa was expressible but nothing
//  computed kappa. This is kappa, and the case that decides whether it is worth
//  having.
//
//  WHY IT MATTERS HERE RATHER THAN IN GENERAL. A real keyhole is a hole held
//  OPEN by recoil pressure and held CLOSED by surface tension, and its radius is
//  where the two balance. A model with recoil and no surface tension has no
//  such balance: nothing sets a minimum radius, and the depression is limited
//  only by the depth of the pool. So Stage 5 alone cannot predict a keyhole
//  width however well it reproduces a hydrostatic depression, and that is the
//  gap this closes.
//
//  THE REFERENCE IS EXACT AND IT IS THE SAME ONE validation/laplace.cpp USES
//  for the multiphase models: a static drop carries
//
//      dp = sigma kappa,     kappa = 1/R in 2-D,  2/R in 3-D
//
//  so dp * R / sigma = 1 with no fitted constant. Running the SAME reference
//  against a third interface representation is the point -- this tree already
//  has that number for the phase field and the colour gradient, and a sharp
//  interface is the one case where the pressure jump is imposed at a boundary
//  rather than carried by a diffuse layer.
//
//  HOW kappa IS BUILT, and the two things that are not obvious. It is
//  kappa = -div(grad eps / |grad eps|) on the GRADIENT lattice (D2Q9 here), and
//  FreeSurfaceSolver::compute_curvature argues both choices at the source. The
//  short version: eps_ is meaningful only at interface cells, so the colour
//  function has to be BUILT (1 in fluid, eps at the interface, 0 in gas) before
//  anything is differenced; and it is mollified once before differencing,
//  because a one-cell-thick interface is nearly a step and the curvature of a
//  step is noise.
//
//  THE DROP COMPRESSES, SO R IS MEASURED AND NOT ASSUMED. The run starts at a
//  uniform density with no pressure jump and lets surface tension build one.
//  Mass is conserved, so the drop reaches the Laplace pressure by SHRINKING --
//  the radius at the end is not the radius seeded. R is therefore taken from
//  the conserved volume, R = sqrt(V/pi), which is exact for a disc and needs no
//  interface-position convention. Using the seeded R instead would fold the
//  compression into the reported error, and at these sigma it is not a small
//  effect.
//
//  THE SPURIOUS CURRENT IS REPORTED BESIDE THE ERROR, NOT INSTEAD OF IT. A
//  static drop should be at rest; whatever velocity survives is the
//  discretisation of the surface force talking. This tree already measures the
//  same quantity for the two diffuse models, and the comparison is only
//  meaningful if the parameters are matched -- see the note at the results.
//==============================================================================

#include "core/Types.hpp"
#include "solver/FreeSurfaceSolver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace lbm;

using L  = D2Q9;
using FS = FreeSurfaceSolver<L>;

namespace {

struct Result {
  double dp = 0, R_meas = 0, R_seed = 0, laplace = 0;  // laplace = dp R / sigma
  double umax = 0, mass_drift = 0, kappa_meas = 0, aspect = 0;
  bool   finite = true;
};

// ecc: 1 seeds a circle, >1 an ellipse of the SAME AREA (pi a b = pi R^2) whose
// ASPECT RATIO a/b is ecc -- so the semi-axes are R sqrt(ecc) and R/sqrt(ecc),
// not R ecc and R/ecc. Those differ by a square: the first version of this used
// the latter and called it an aspect ratio, and the t = 0 guard below read
// 1.6863 for a requested 1.30, which is 1.30^2 to three figures. The guard was
// added to catch a seeding bug and caught a naming one instead.
Result run(Index N, double R0, double sigma, double nu, std::size_t nsteps,
           bool verbose, double ecc = 1.0) {
  Result Rr;
  Rr.R_seed = R0;
  constexpr double cs2d = double(lbm::cs2<L, double>());

  Domain d(N, N, 1, true, true, true);          // periodic: no walls at all
  FS s(d);
  s.coll.omega = FS::omega_from_viscosity(Real(nu));
  s.set_surface_tension(Real(sigma));
  s.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });

  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  const Real Rr0 = Real(R0), xc = Real(N) * Real(0.5);
  const Real ea = Real(R0 * std::sqrt(ecc)), eb = Real(R0 / std::sqrt(ecc));
  // Uniform density and NO pressure jump: the jump is what is being measured,
  // so seeding it would be assuming the answer.
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real x = Real(px - hx) - xc, y = Real(py - hy) - xc;
    // For an ellipse the "radius" is the level set (x/a)^2 + (y/b)^2 = 1
    // rescaled so that the same linear cell fraction applies.
    const Real q = Kokkos::sqrt((x / ea) * (x / ea) + (y / eb) * (y / eb));
    const Real r = (ecc == Real(1)) ? Kokkos::sqrt(x * x + y * y)
                                    : q * Rr0;
    Real e = Rr0 + Real(0.5) - r;                // linear cell fraction
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, Real(1)};
  });

  const double m0 = double(s.total_mass());
  for (std::size_t t = 0; t < nsteps; ++t) s.step();
  const double m1 = double(s.total_mass());
  Rr.mass_drift = (m1 - m0) / m0;
  if (!std::isfinite(m1)) { Rr.finite = false; return Rr; }

  // ---- measure ---------------------------------------------------------------
  auto hf  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
  auto he  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.fill());
  auto hr  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
  auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto huy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());

  // R from the conserved volume: exact for a disc, no interface convention.
  double vol = 0;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const Index n = d.id(x, y);
      const std::uint8_t f = hf(n);
      if      (f == FsFluid)     vol += 1.0;
      else if (f == FsInterface) vol += double(he(n));
    }
  Rr.R_meas = std::sqrt(vol / M_PI);

  // Core pressure: FLUID cells at least two cells inside the interface, so the
  // average is of the bulk and not of the layer whose curvature is being tested.
  double rsum = 0; long rn = 0;
  const double rcore = Rr.R_meas - 3.0;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const Index n = d.id(x, y);
      const double dx0 = double(x) - 0.5 * double(N), dy0 = double(y) - 0.5 * double(N);
      if (hf(n) != FsFluid) continue;
      if (std::sqrt(dx0 * dx0 + dy0 * dy0) > rcore) continue;
      rsum += double(hr(n));  ++rn;
      if (!std::isfinite(double(hr(n)))) Rr.finite = false;
    }
  const double rho_in = (rn > 0) ? rsum / double(rn) : 0.0;
  Rr.dp = cs2d * (rho_in - 1.0);                 // rho_G = 1 outside
  Rr.laplace = (sigma != 0) ? Rr.dp * Rr.R_meas / sigma : 0.0;
  Rr.kappa_meas = (sigma != 0) ? Rr.dp / sigma : 0.0;

  // Aspect ratio from the second moments of the colour function. For a disc it
  // is 1; it is how the relaxation test knows the drop became round.
  {
    double cx = 0, cy = 0, m = 0;
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y);
        const std::uint8_t f = hf(n);
        const double c = (f == FsFluid) ? 1.0 : (f == FsInterface ? double(he(n)) : 0.0);
        cx += c * double(x);  cy += c * double(y);  m += c;
      }
    if (m > 0) { cx /= m; cy /= m; }
    double ixx = 0, iyy = 0;
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y);
        const std::uint8_t f = hf(n);
        const double c = (f == FsFluid) ? 1.0 : (f == FsInterface ? double(he(n)) : 0.0);
        ixx += c * (double(x) - cx) * (double(x) - cx);
        iyy += c * (double(y) - cy) * (double(y) - cy);
      }
    Rr.aspect = (iyy > 0) ? std::sqrt(ixx / iyy) : 0.0;
  }

  // Spurious current: a static drop should be at rest.
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const Index n = d.id(x, y);
      if (hf(n) != FsFluid && hf(n) != FsInterface) continue;
      const double u = std::hypot(double(hux(n)), double(huy(n)));
      if (u > Rr.umax) Rr.umax = u;
    }
  if (verbose)
    std::printf("      core cells %ld, rho_in %.8f, vol %.2f\n", rn, rho_in, vol);
  return Rr;
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = false;
  std::size_t nsteps = 20000;
  double sigma = 0.01;
  for (int i = 1; i < argc; ++i) {
    if      (!std::strcmp(argv[i], "-v")) verbose = true;
    else if (!std::strcmp(argv[i], "-steps") && i + 1 < argc)
      nsteps = std::size_t(std::atof(argv[++i]));
    else if (!std::strcmp(argv[i], "-sigma") && i + 1 < argc)
      sigma = std::atof(argv[++i]);
    else if (!std::strncmp(argv[i], "--kokkos", 8)) {}
    else {
      std::fprintf(stderr, "surface_tension: unknown option '%s'. NOTHING WAS RUN.\n"
                           "  -v  -steps N  -sigma S\n", argv[i]);
      return 2;
    }
  }

  Kokkos::initialize(argc, argv);
  int failures = 0, checks = 0;
  {
    std::printf("surface tension on a sharp free surface: Laplace's law\n");
    std::printf("backend %s   precision %s\n\n",
                Kokkos::DefaultExecutionSpace::name(), precision_name());
    const double nu = 1.0 / 6.0;      // omega = 1, for the reason recoil.cpp gives

    auto check = [&](const char* what, double got, double want, double tol) {
      ++checks;
      const bool ok = std::fabs(got - want) <= tol;
      if (!ok) ++failures;
      std::printf("  %-52s %12.6f  (want %.6f +/- %.4f)   %s\n",
                  what, got, want, tol, ok ? "PASS" : "FAIL");
    };

    std::printf("1. Laplace's law, sigma = %.4g, dp R / sigma must be 1\n", sigma);
    std::printf("   %-8s %-10s %-10s %-12s %-10s %-11s\n",
                "R seed", "R meas", "dp", "dp R/sigma", "err", "spurious u");
    double worst = 0, worst12 = 0, err8 = 0, umax_worst = 0;
    for (double R0 : {8.0, 12.0, 16.0, 24.0}) {
      const Index N = Index(6.0 * R0);
      const Result R = run(N, R0, sigma, nu, nsteps, verbose);
      const double err = R.laplace - 1.0;
      std::printf("   %-8.1f %-10.4f %-10.6f %-12.5f %-10.4f %-11.3e\n",
                  R.R_seed, R.R_meas, R.dp, R.laplace, err, R.umax);
      if (!R.finite) { ++checks; ++failures; std::printf("   NON-FINITE\n"); continue; }
      if (std::fabs(err) > std::fabs(worst)) worst = err;
      if (R0 == 8.0) err8 = err;
      else if (std::fabs(err) > std::fabs(worst12)) worst12 = err;
      if (R.umax > umax_worst) umax_worst = R.umax;
      char buf[128];
      std::snprintf(buf, sizeof buf, "mass drift at R = %.0f", R0);
      check(buf, R.mass_drift, 0.0, 1e-3);
    }
    std::printf("   worst relative error over the sweep: %.4f\n", worst);
    // THE THRESHOLDS BELOW WERE SET FROM THE MEASUREMENT AND NOT BEFORE IT.
    // Measured 2026-09-22: +3.39 % at R = 8, then +0.88 / -0.64 / +1.14 % at
    // R = 12/16/24. The error does NOT fall monotonically with R, and that is a
    // property of curvature on a volume-of-fluid field rather than a defect:
    // it depends on how the circle happens to sit on the grid, so two radii a
    // factor of two apart can land on opposite sides of the answer. There is
    // therefore NO ORDER QUOTED HERE -- a rate fitted through non-monotone
    // points would be arithmetic rather than a measurement. What is claimable
    // is a BAND: 1.2 % for R >= 12, and R = 8 is the resolution floor.
    check("worst Laplace error, R >= 12", worst12, 0.0, 0.020);
    check("Laplace error at R = 8 (the resolution floor)", err8, 0.0, 0.050);
    check("worst spurious current over the sweep", umax_worst, 0.0, 1e-4);

    // ---- 2. THE NULL CONTROL ------------------------------------------------
    // Laplace's law alone cannot prove the term ACTS -- a scheme that produced
    // the right jump for the wrong reason would pass it. sigma = 0 must give NO
    // jump at all, which is the one run that isolates the new term from
    // everything else in the solver.
    std::printf("\n2. null control: sigma = 0 must give no pressure jump\n");
    const Result Z = run(96, 16.0, 0.0, nu, nsteps, false);
    std::printf("   dp = %.3e  (Laplace jump at sigma = %.4g would be %.3e)\n",
                Z.dp, sigma, sigma / 16.0);
    check("sigma = 0 gives no pressure jump", Z.dp, 0.0, 1e-6);

    // ---- 3. AN ELLIPSE MUST BECOME A CIRCLE ---------------------------------
    // The dynamic test, and the one that checks the SIGN. Surface tension
    // minimises perimeter at fixed area, so an ellipse seeded at aspect 1.3
    // must relax toward 1. A sign error would make it grow instead, and no
    // static measurement above would notice.
    std::printf("\n3. an ellipse relaxes to a circle (sign and restoring action)\n");
    // THE SEED IS VERIFIED FIRST, because this test is vacuous if it is not.
    // An aspect of 1.0 at the end proves nothing unless it was 1.3 at the
    // start -- a seeding bug that produced a circle would pass the relaxation
    // check without the curvature term doing anything at all. Zero steps, so
    // this reads the initial condition and nothing else.
    const Result E0 = run(144, 20.0, sigma, nu, 0, false, 1.30);
    std::printf("   seeded aspect measured at t = 0: %.4f\n", E0.aspect);
    check("the ellipse seed really is an ellipse", E0.aspect, 1.30, 0.03);
    const Result E = run(144, 20.0, sigma, nu, nsteps, false, 1.30);
    std::printf("   after %zu steps: %.6f  (spurious u %.3e)\n",
                nsteps, E.aspect, E.umax);
    check("ellipse relaxes toward a circle", E.aspect, 1.0, 0.05);
  }
  Kokkos::finalize();

  std::printf("\n[surface_tension] %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
