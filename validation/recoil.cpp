//==============================================================================
//  RECOIL PRESSURE: a non-uniform normal stress on a free surface, against the
//  exact hydrostatic depression.
//
//  STAGE 5 of a coupled melt-pool model, and it exists for the same reason
//  validation/marangoni.cpp and validation/mushy_sink.cpp do: the device is
//  measured against something exact BEFORE it is allowed near a melt pool,
//  where nothing has a closed-form answer and a wrong constant still converges.
//
//  WHAT RECOIL PRESSURE IS. Metal boiling off a laser-heated surface leaves with
//  momentum, and the surface feels the reaction. It is the force that digs a
//  keyhole -- not the beam, which only supplies the heat. The standard closure
//  is Anisimov's,
//
//      P_r(T) = (1 + beta_r)/2 * P_v(T),
//      P_v(T) = P_0 exp[ (L_v/R_s) (1/T_b - 1/T) ]        (Clausius-Clapeyron)
//
//  with beta_r the retro-diffusion fraction. validation/melt_pool.cpp ALREADY
//  computes P_v, for the evaporative cooling of its tier (e) -- so the
//  thermodynamics is not what is new here. What is new is that the pressure has
//  somewhere to push.
//
//  THIS NEEDED ONE LINE OF SOLVER, NOT A MODEL. FreeSurfaceSolver's dynamic
//  condition already states that the normal stress at the surface EQUALS the gas
//  pressure -- that is what its population reconstruction imposes, and
//  validation/gravity_wave.cpp measures it through the dispersion relation. It
//  was only ever the UNIFORMITY of rho_G that made that one atmosphere. So
//  rho_G_of, an optional per-node field, is the whole of Stage 5 on the solver
//  side, and it follows ScalarBGK's omega_of idiom exactly: empty means the
//  scalar, set means authoritative.
//
//  THE EXACT REFERENCE, AND WHY IT IS AVAILABLE HERE. A static, connected liquid
//  under gravity has a pressure that depends only on depth. Take two columns at
//  x1 and x2 with surface heights z1, z2 under gas pressures p1, p2 and equate
//  the pressure at a common level below both:
//
//      p1 + rho g (z1 - z0) = p2 + rho g (z2 - z0)
//   => z2 - z1 = (p1 - p2) / (rho g)
//
//  so the surface sits LOWER where the gas pushes HARDER, by exactly
//
//      dz(x) = - [p_G(x) - <p_G>] / (rho g),      p_G = rho_G cs^2.
//
//  There is no surface tension in that statement and no viscosity, which is the
//  point: the reference is exact for a solver that HAS no surface tension, so
//  the missing curvature term is not a caveat here. It is also exactly the
//  mechanism that makes a keyhole, measured in the one regime where it has a
//  closed form.
//
//  A COSINE, NOT A GAUSSIAN, FOR THE HEADLINE NUMBER. The forcing is
//  rho_G(x) = rho_G0 + a cos(kx) with k = 2 pi / Lx, so the exact answer is a
//  pure cosine of amplitude cs^2 a / (rho g) and the measurement can be the
//  first Fourier mode of the surface -- the same estimator gravity_wave.cpp
//  uses, and one that rejects the discretisation noise a pointwise maximum
//  would collect. A GAUSSIAN bump is then run as a second shape, because a
//  single-mode test cannot tell a correct response from one that is right only
//  at k = 2 pi / Lx, and a keyhole is localised rather than sinusoidal.
//
//  WHAT IS DELIBERATELY NOT HERE.
//   * NO THERMAL COUPLING. The pressure field is PRESCRIBED. Driving it from
//     P_v(T) is the melt-pool case's job, and doing it here would put a fitted
//     temperature between the device and its exact answer.
//   * SURFACE TENSION IS NOW HERE, IN SECTION 4, and this entry is kept because
//     sections 1-3 still run without it deliberately: the pressure device is
//     measured alone before it is measured in company, so a failure in the sum
//     cannot hide inside a term. Section 4 adds it and tests the BALANCE,
//     h_a = p_a/(rho g + sigma k^2) -- which is the keyhole's own balance,
//     linearised. Measured at Lx = 256 with the forcing scaled to hold the
//     depression at 8.00 cells while the capillary share of the restoring force
//     goes 0 -> 32.5 -> 65.8 -> 88.5 %: errors +1.40, -2.84, +0.24, +2.18 %,
//     i.e. the same band as the sigma = 0 row. Adding surface tension costs no
//     accuracy, which is what says the two terms compose rather than merely
//     both being present.
//     WHAT IS STILL MISSING for a keyhole is the NONLINEAR regime: this is a
//     small-slope balance, and a real keyhole is a deep cavity whose walls are
//     not a perturbation of a flat surface. Do not read section 4 as a keyhole
//     depth.
//   * NO EVAPORATIVE MASS LOSS. The liquid volume is constant, so <h> is fixed
//     by conservation and the comparison is of the SHAPE about that mean.
//
//  THE COMPRESSIBILITY FLOOR, STATED BEFORE IT IS BLAMED. The lattice liquid is
//  weakly compressible: rho(z) = rho_G + g (z_s - z)/cs^2, so the "rho" in the
//  reference is not exactly 1 but the mean liquid density over the depression.
//  The relative error that costs is O(g h / cs^2), which is 9.6e-4 at the
//  settings below -- an order below the discretisation error, and it is why the
//  amplitudes here are small rather than spectacular.
//==============================================================================

#include "core/Types.hpp"
#include "solver/FreeSurfaceSolver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace lbm;

using L  = D2Q9;
using FS = FreeSurfaceSolver<L>;

namespace {

struct Result {
  double amp_meas = 0;    // measured first-mode surface amplitude
  double amp_exact = 0;   // cs^2 a / (rho g)
  double err = 0;         // relative
  double l2 = 0;          // pointwise L2 over the profile, relative to amp
  double mass_drift = 0;
  double nu = 0;          // the critically-damped viscosity this run chose
  std::size_t steps = 0;  // and the steps it needed
  bool   finite = true;
};

// Surface height of every column: the column's total fill. Exact in this
// representation -- it IS how much liquid the column holds.
std::vector<double> surface(const FS& s, const Domain& d, Index Lx, Index Ly) {
  auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.fill());
  auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());
  std::vector<double> h(std::size_t(Lx), 0.0);
  for (Index x = 0; x < Lx; ++x) {
    double col = 0;
    for (Index y = 1; y < Ly - 1; ++y) {
      const std::uint8_t f = hf(d.id(x, y));
      if      (f == FsFluid)     col += 1.0;
      else if (f == FsInterface) col += double(he(d.id(x, y)));
    }
    h[std::size_t(x)] = col;
  }
  return h;
}

// shape == 0: cosine, a cos(kx).  shape == 1: Gaussian bump of width w.
// sigma > 0 turns on surface tension, which changes BOTH the exact answer and
// the relaxation rate. It is only meaningful for the cosine: the balance is per
// MODE, so a localised bump has no single-formula answer.
Result run(Index Lx, Index Ly, double h0, double g, double nu, double a,
           std::size_t nsteps, int shape, double w, bool verbose,
           double sigma = 0.0) {
  Result R;
  constexpr double cs2 = double(lbm::cs2<L, double>());
  const double k = 2.0 * M_PI / double(Lx);
  // THE RESTORING FORCE IS NOW GRAVITY PLUS CAPILLARITY, and everything that
  // depended on gravity alone has to take the sum. Linearised Young-Laplace for
  // a surface h(x) under an applied pressure p_a cos(kx):
  //     p_a cos(kx) = -sigma h'' + rho g h  =>  h_a = p_a / (rho g + sigma k^2)
  // which is this file's original p_a/(rho g) at sigma = 0, so the sigma sweep
  // in section 4 contains the sigma = 0 row as a consistency check rather than
  // as a separate claim.
  const double restore = g + sigma * k * k;      // per unit rho, rho = 1

  // THE VISCOSITY IS FIXED AT omega = 1, AND THAT IS A MEASURED CHOICE RATHER
  // THAN A DEFAULT. The equilibrium here is hydrostatic, so viscosity cannot
  // move it -- only how fast it is reached -- which looks like it makes nu a
  // free knob to spend on getting there. It is not, and both obvious ways of
  // spending it were tried and are recorded because each fails differently.
  //
  //  * RAISING nu TO "DAMP FASTER" IS BACKWARDS. At nu = 0.5 the surface is
  //    OVERDAMPED, where the relaxation rate is omega_0^2/(2 nu k^2) and FALLS
  //    with viscosity. The run then sat at 69 % of the exact depression at every
  //    resolution and the order collapsed from 1.84 to 0.03 -- a case simply not
  //    converged in time, reading exactly like a scheme that does not converge
  //    in space. That is the trap this file exists to avoid, met while writing
  //    it.
  //  * CRITICAL DAMPING, 2 nu k^2 = omega_0, IS THE FASTEST APPROACH AND IS
  //    LESS ACCURATE. Choosing nu per grid that way (0.0696/0.1392/0.2785)
  //    gave order 1.59 and a Gaussian error of 7.1 % against 1.84 and 2.7 % at
  //    a flat nu = 1/6 -- and it did so with MORE relaxation times elapsed, so
  //    it is not a convergence effect. It is the viscosity dependence of the
  //    boundary treatment: omega = 1 is where an LBM wall sits where it claims
  //    to, and moving away from it costs accuracy that no amount of settling
  //    recovers. TWO POINTS, so this is a reason to keep omega = 1, not a
  //    measured order in nu.
  //
  // So nu is 1/6 and the STEPS are what gets derived. The relaxation is a
  // damped surface gravity wave, omega_0 = sqrt(g k tanh(k h)) damped at
  // gamma = 2 nu k^2, whose slow rate is gamma when underdamped and
  // omega_0^2/gamma when over -- i.e. min of the two, always.
  // Capillary-gravity waves: omega_0^2 = (g k + sigma k^3/rho) tanh(k h). Using
  // the gravity-only frequency here would under-count the restoring force and
  // pick a step count too small at large sigma -- the same "not converged in
  // time" failure this file already records, arriving by a different door.
  const double om0   = std::sqrt((g * k + sigma * k * k * k) * std::tanh(k * h0));
  const double nu_c  = (nu > 0.0) ? nu : 1.0 / 6.0;
  const double gam   = 2.0 * nu_c * k * k;
  const double rate  = std::fmin(gam, om0 * om0 / gam);
  // Eight relaxation times: a residual transient of e^-8 = 3.4e-4, two orders
  // under the 1.5 % discretisation error this case is trying to measure.
  const std::size_t nst = (nsteps > 0) ? nsteps : std::size_t(8.0 / rate) + 1;

  Domain d(Lx, Ly, 1, /*periodic x*/ true, /*y*/ false, /*z*/ true);
  FS s(d);
  s.coll.omega = FS::omega_from_viscosity(Real(nu_c));
  s.set_gravity(Real(0), Real(-g));
  if (sigma != 0.0) s.set_surface_tension(Real(sigma));

  const Index Lyi = Ly;
  s.set_geometry([&](Index, Index y, Index) -> FsCell {
    return (y == 0 || y == Lyi - 1) ? FsSolid : FsGas;
  });

  // ---- the applied gas pressure ---------------------------------------------
  // rho_G_of is indexed by node, and only its value at INTERFACE cells is ever
  // read -- but the interface moves, so the whole column carries the column's
  // pressure rather than only the cells that happen to be interface at step 0.
  s.rho_G_of = View1D<Real>("rho_G_of", d.n_padded);
  {
    const Domain dd = d;
    const Index hx = d.hx;
    const Real ar = Real(a), kr = Real(k), wr = Real(w);
    const Real xc = Real(Lx) * Real(0.5);
    const int  sh = shape;
    auto rgf = s.rho_G_of;
    Kokkos::parallel_for("recoil_pG", d.n_padded, KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; dd.coords(n, px, py, pz);
      const Real x = Real(px - hx);
      Real dp;
      if (sh == 0) {
        dp = ar * Kokkos::cos(kr * x);
      } else {
        const Real t = (x - xc) / wr;
        dp = ar * Kokkos::exp(Real(-0.5) * t * t);
      }
      rgf(n) = Real(1) + dp;
    });
  }
  // The mean applied pressure, which sets the level the shape is measured about.
  std::vector<double> dp(std::size_t(Lx), 0.0);
  double dp_mean = 0;
  for (Index x = 0; x < Lx; ++x) {
    double v;
    if (shape == 0) v = a * std::cos(k * double(x));
    else {
      const double t = (double(x) - 0.5 * double(Lx)) / w;
      v = a * std::exp(-0.5 * t * t);
    }
    dp[std::size_t(x)] = v;
    dp_mean += v;
  }
  dp_mean /= double(Lx);

  // ---- flat surface, hydrostatic, at the MEAN gas pressure -------------------
  // Seeding at the mean rather than at rho_G0 keeps the initial transient
  // symmetric about the final state instead of biased to one side of it.
  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  const Real hr = Real(h0), gr = Real(g), rg0 = Real(1.0 + dp_mean);
  constexpr Real ics = inv_cs2<L, Real>();
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real y = Real(py - hy);
    Real e = hr - (y - Real(0.5));
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    const Real dz = hr - y;
    const Real r = rg0 + (dz > Real(0) ? gr * dz * ics : Real(0));
    return typename FS::Seed{e, r};
  });

  const double m0 = double(s.total_mass());
  for (std::size_t t = 0; t < nst; ++t) s.step();
  const double m1 = double(s.total_mass());
  R.mass_drift = (m1 - m0) / m0;
  if (!std::isfinite(m1)) { R.finite = false; return R; }

  // ---- measure ---------------------------------------------------------------
  const std::vector<double> hsurf = surface(s, d, Lx, Ly);
  double hm = 0;
  for (double v : hsurf) hm += v;
  hm /= double(Lx);
  for (double v : hsurf) if (!std::isfinite(v)) { R.finite = false; return R; }

  // exact: dz(x) = -cs^2 (dp(x) - <dp>) / (rho g + sigma k^2), rho = 1 to
  // O(g h / cs^2). At sigma = 0 this is the hydrostatic form unchanged.
  std::vector<double> ex(std::size_t(Lx), 0.0);
  for (Index x = 0; x < Lx; ++x)
    ex[std::size_t(x)] = -cs2 * (dp[std::size_t(x)] - dp_mean) / restore;

  if (shape == 0) {
    // First Fourier mode of each, which for a cosine forcing IS the answer.
    double am = 0, ae = 0;
    for (Index x = 0; x < Lx; ++x) {
      const double c = std::cos(k * double(x));
      am += (hsurf[std::size_t(x)] - hm) * c;
      ae += ex[std::size_t(x)] * c;
    }
    R.amp_meas  = 2.0 * am / double(Lx);
    R.amp_exact = 2.0 * ae / double(Lx);
  } else {
    // Localised: the peak depression, which is what a keyhole depth is.
    double am = 0, ae = 0;
    for (Index x = 0; x < Lx; ++x) {
      if (std::fabs(hsurf[std::size_t(x)] - hm) > std::fabs(am))
        am = hsurf[std::size_t(x)] - hm;
      if (std::fabs(ex[std::size_t(x)]) > std::fabs(ae)) ae = ex[std::size_t(x)];
    }
    R.amp_meas = am;  R.amp_exact = ae;
  }
  R.err = (R.amp_exact != 0.0)
        ? (R.amp_meas - R.amp_exact) / std::fabs(R.amp_exact) : 0.0;

  double num = 0, den = 0;
  for (Index x = 0; x < Lx; ++x) {
    const double e = (hsurf[std::size_t(x)] - hm) - ex[std::size_t(x)];
    num += e * e;  den += ex[std::size_t(x)] * ex[std::size_t(x)];
  }
  R.l2 = (den > 0) ? std::sqrt(num / den) : 0.0;
  R.nu = nu_c;  R.steps = nst;

  if (verbose) {
    std::printf("      x      h-<h>      exact\n");
    for (Index x = 0; x < Lx; x += std::max<Index>(1, Lx / 16))
      std::printf("   %4d   %9.4f  %9.4f\n", int(x),
                  hsurf[std::size_t(x)] - hm, ex[std::size_t(x)]);
  }
  return R;
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = false;
  std::size_t nsteps = 0;   // 0 = choose it from the relaxation time
  for (int i = 1; i < argc; ++i) {
    if      (!std::strcmp(argv[i], "-v")) verbose = true;
    else if (!std::strcmp(argv[i], "-steps") && i + 1 < argc)
      nsteps = std::size_t(std::atof(argv[++i]));
    else if (!std::strncmp(argv[i], "--kokkos", 8)) {}
    else {
      std::fprintf(stderr, "recoil: unknown option '%s'. NOTHING WAS RUN.\n"
                           "  -v   print the surface profile\n"
                           "  -steps N\n", argv[i]);
      return 2;
    }
  }

  Kokkos::initialize(argc, argv);
  int failures = 0, checks = 0;
  {
    std::printf("recoil pressure vs the exact hydrostatic depression\n");
    std::printf("backend %s   precision %s\n\n",
                Kokkos::DefaultExecutionSpace::name(), precision_name());

    const double g = 1e-5, nu = 0.0;   // 0 = choose it: see run()
    auto check = [&](const char* what, double got, double want, double tol) {
      ++checks;
      const bool ok = std::fabs(got - want) <= tol;
      if (!ok) ++failures;
      std::printf("  %-52s %12.6f  (want %.6f +/- %.4f)   %s\n",
                  what, got, want, tol, ok ? "PASS" : "FAIL");
    };

    // ---- 1. THE ERROR IS SET BY THE DEPRESSION RESOLVED IN CELLS ------------
    // Measured before the thresholds were written, which is the only order that
    // works: a flat tolerance here would have been a guess, and this tree has
    // already had to fix exactly that once (melt_pool's acceptance thresholds).
    // Sweeping the forcing at ONE grid, the relative error falls as the
    // depression spans more cells -- 1 cell is 36 % wrong and 8 cells 1.4 %.
    // That is the one-cell-thick interface: the free-surface condition imposes
    // the gas pressure at the CELL, not at the sub-cell position of the surface
    // inside it, so the error is a fixed fraction of a cell rather than of the
    // answer. The control that proves it is not the grid: a = 6e-5 gives 13.06 %
    // at Lx = 128 and 13.16 % at Lx = 64 -- the SAME two-cell depression on
    // grids a factor of two apart.
    std::printf("1. the error is set by the depression in CELLS (Lx = 128, g = %.0e)\n", g);
    std::printf("   %-10s %-12s %-12s %-10s %-10s\n",
                "a", "cells meas", "cells exact", "rel err", "L2");
    double prev = 1e9;
    bool monotone = true;
    for (double a : {3.0e-5, 6.0e-5, 1.2e-4}) {
      const Result R = run(128, 64, 32.0, g, nu, a, nsteps, 0, 0.0, false);
      std::printf("   %-10.2e %-12.5f %-12.5f %-10.4f %-10.4f\n",
                  a, R.amp_meas, R.amp_exact, R.err, R.l2);
      if (!R.finite) { ++failures; ++checks; std::printf("   NON-FINITE\n"); continue; }
      if (R.err >= prev) monotone = false;
      prev = R.err;
      char buf[128];
      std::snprintf(buf, sizeof buf, "mass drift at a = %.1e", a);
      check(buf, R.mass_drift, 0.0, 1e-3);   // gravity_wave's convention
    }
    ++checks;
    if (!monotone) ++failures;
    std::printf("  %-52s %12s  %s\n",
                "relative error falls as the depression is resolved", "",
                monotone ? "PASS" : "FAIL");

    // ---- 2. THE ORDER -------------------------------------------------------
    // The same PHYSICAL problem on three grids: g ~ 1/Lx and the depth ~ Lx, so
    // the depression is a fixed FRACTION of the depth and the only thing that
    // changes is how many cells resolve it. THIS is the convergence statement,
    // and it is the one an acceptance threshold can be built on.
    std::printf("\n2. resolution, same physical forcing -- the order\n");
    std::printf("   %-8s %-12s %-12s %-10s\n", "Lx", "meas", "exact", "rel err");
    double e_prev = 0, e_last = 0, order_last = 0;
    int    li = 0;
    for (Index Lx : {64, 128, 256}) {
      const double sc = double(Lx) / 128.0;
      const Result R = run(Lx, Index(64 * sc), 32.0 * sc, g / sc, nu,
                           1.2e-4, nsteps, 0, 0.0, false);
      std::printf("   %-8d %-12.5f %-12.5f %-10.4f  (nu %.4f, %zu steps)\n",
                  int(Lx), R.amp_meas, R.amp_exact, R.err, R.nu, R.steps);
      const double e = std::fabs(R.err);
      if (li > 0) order_last = std::log(e_prev / e) / std::log(2.0);
      e_prev = e;  e_last = e;  ++li;
    }
    std::printf("   order over the last refinement: %.2f\n", order_last);
    // >= 1.5 rather than >= 2: the interface error is first order in the
    // sub-cell surface position and second order in the profile, and the two
    // mix. The measured pair is 1.39 then 1.84, rising toward 2 as the coarse
    // point leaves the unresolved regime -- so the FINEST pair is the one with
    // any claim to be an order, and 1.5 is the floor that pair must clear.
    check("convergence order over the finest refinement", order_last, 2.0, 0.5);
    check("relative error on the finest grid", e_last, 0.0, 0.02);

    // ---- 3. A LOCALISED BUMP -------------------------------------------------
    // A keyhole is not a cosine. One mode cannot distinguish a correct response
    // from one right only at k = 2 pi / Lx. Run at Lx = 256 with the width
    // scaled with the grid, so the bump is resolved rather than measuring the
    // cells effect again.
    std::printf("\n3. Gaussian bump (localised, keyhole-like), Lx = 256\n");
    const Result G = run(256, 128, 64.0, g / 2.0, nu, 1.2e-4, nsteps, 1, 24.0,
                         verbose);
    std::printf("   peak depression %.5f cells, exact %.5f, rel err %.4f, L2 %.4f\n",
                G.amp_meas, G.amp_exact, G.err, G.l2);
    check("Gaussian peak depression, relative error", G.err, 0.0, 0.05);
    check("Gaussian profile L2", G.l2, 0.0, 0.08);
    check("Gaussian mass drift", G.mass_drift, 0.0, 1e-3);

    // ---- 4. RECOIL AND SURFACE TENSION TOGETHER -----------------------------
    // THE INTEGRATION TEST, and the reason it is here rather than in
    // validation/surface_tension.cpp: each stage has been measured alone and
    // nothing yet says they compose. They reach the flow through the SAME
    // field -- rho_eff = rho_G + sigma kappa/cs^2 -- so a sign error or a
    // double count in that sum is invisible to either case on its own.
    //
    // It is also the keyhole balance in the one regime where it is exact. A
    // keyhole is a hole held OPEN by recoil and CLOSED by surface tension, and
    // linearised Young-Laplace gives the balance directly:
    //     h_a = p_a / (rho g + sigma k^2).
    //
    // THE AMPLITUDE IS SCALED WITH THE RESTORING FORCE SO THE ANSWER DOES NOT
    // MOVE. recoil.cpp's own finding is that the error is set by the depression
    // measured in CELLS, so a sigma sweep at fixed forcing would shrink the
    // depression and confound the coupling with the resolution. Here a is
    // raised in step with (rho g + sigma k^2), so the exact depression is 8.00
    // cells at EVERY sigma and the measured one must be too -- while the
    // capillary share of the restoring force goes from 0 to 88.5 %.
    std::printf("\n4. recoil AND surface tension: h_a = p_a/(rho g + sigma k^2)\n");
    std::printf("   %-10s %-10s %-12s %-12s %-10s\n",
                "sigma", "cap share", "depth meas", "depth exact", "rel err");
    const Index Lx4 = 256;
    const double g4 = 5e-6, k4 = 2.0 * M_PI / double(Lx4);
    double worst4 = 0;
    for (double sg : {0.0, 4.0e-3, 1.6e-2, 6.4e-2}) {
      const double restore4 = g4 + sg * k4 * k4;
      const double a4 = 8.0 * restore4 / (1.0 / 3.0);   // hold 8 cells
      const Result S = run(Lx4, 128, 64.0, g4, 0.0, a4, 0, 0, 0.0, false, sg);
      std::printf("   %-10.1e %-10.1f %-12.5f %-12.5f %-10.4f\n",
                  sg, 100.0 * sg * k4 * k4 / restore4,
                  S.amp_meas, S.amp_exact, S.err);
      if (!S.finite) { ++checks; ++failures; std::printf("   NON-FINITE\n"); continue; }
      if (std::fabs(S.err) > std::fabs(worst4)) worst4 = S.err;
    }
    // Same band as the sigma = 0 case at this resolution: adding surface
    // tension must not cost accuracy, and if it does the sum is wrong.
    check("worst error over the sigma sweep at fixed depression", worst4, 0.0, 0.04);
  }
  Kokkos::finalize();

  std::printf("\n[recoil] %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
