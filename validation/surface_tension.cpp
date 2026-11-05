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
//
//  -nrec runs it with FreeSurfaceSolver::normal_reconstruction (2026-11-03):
//  errors +3.39 / +1.08 / -0.40 / +2.99 % at R = 8/12/16/24 against the
//  default's +3.39 / +0.88 / -0.64 / +1.14 %, i.e. the same band with R = 24
//  past this file's 2 % gate, and spurious currents of ~3e-9 against ~5e-6. The
//  gates are the default's; -nrec failing one of them is the measurement.
//
//  -rate (2026-11-04) MEASURES HOW FAST AN ELLIPSE BECOMES A CIRCLE, NOT ONLY
//  THAT IT DOES, AND THE CURVATURE FAILS IT. It runs instead of the gated
//  sections, which it leaves untouched.
//  THE REFERENCE is a 2-D viscous drop in a void, mode 2, linearised:
//  psi = (A r^2 + B J_2(q r)) sin 2theta e^{-lambda t}, q^2 = lambda/nu, with
//  zero tangential stress, normal stress -sigma kappa and the kinematic
//  condition (dispersion(), exact_rate()). Its Stokes limit is
//  lambda = sigma/(mu R) -- n sigma/(2 mu R) for mode n -- which the loss of
//  surface energy against that flow's dissipation gives independently. The
//  finite-Oh root was checked at both ends before it was used: at Oh = 1000 it
//  is the Stokes rate to 2e-7, and as Oh -> 0 its frequency goes to the
//  inviscid sqrt(6) and its damping to the potential-flow 4 nu/R^2, the
//  shortfall falling as Oh^(1/2) (0.954 of it at Oh = 1e-2, 0.9995 at 1e-6),
//  a boundary-layer term. At Oh = 10 and 3 it is 1.0018 and 1.0205 x Stokes.
//  The amplitude is eps = a/R from the second moments of the colour function,
//  the same number for either orientation (-diag turns the ellipse 45 deg);
//  sigma comes from Oh at the seeded radius and the rate from the radius of the
//  seed's volume; nu = 1/6, aspect 1.1 (eps0 = 0.047), and the fit skips the
//  first 2 R^2/nu, by which the fast viscous modes have gone.
//  MEASURED (FP64, Oh = 3): the fitted rate over the exact one, the same for
//  each half of the window alone, and eps at the end over eps0 (0.108 at
//  t lambda = 2.227 if the rate were right):
//      R =  8    1.266   (1.410, 1.043)   0.065
//      R = 16    0.804   (0.817, 0.373)   0.202
//      R = 32    0.049   (0.152, 0.005)   0.618
//      R = 64    0.005   (0.008, 0.003)   0.883
//  From R = 32 the drop STOPS. Oh = 10 gives 1.303 / 0.846 / 0.120 at
//  R = 8 / 16 / 32, the R = 32 drop stopping at 0.619 against 0.618 here, so
//  the stall is set by the geometry and not by the size of sigma. -nrec gives
//  0.546 / 0.118 / 0.050 and stops sooner; -diag 0.987 / 0.662 / 0.236. No
//  variant converges; every one slows as R grows.
//  WHY, from the solver's own kappa fitted along the interface cells as
//  k0 + k2 cos 2(theta - theta0): a curvature that saw the shape would give
//  k2 = 3 eps/R. At R = 32 it gives 0.91 of that at t lambda = 0.1 and 0.02 by
//  0.7, with eps still 0.63 eps0, and stays within 0.035 of zero after; at
//  R = 64 it ends at 0.06; at R = 16 it swings between 1.62 and -0.66; at R = 8
//  it reads 1.5-1.6 from t lambda = 0.6 on, which is why that drop is fast.
//  The deformation moves into cell-scale fill patterns that the once-smoothed
//  curvature cannot tell from a circle. What is left where the runs end has a
//  curvature of its own, 3 eps/R, of 0.0018 / 0.0027 / 0.0020 per cell at
//  R = 16 / 32 / 64 -- a radius of 370-560 cells, about fixed in cells -- so a
//  finer grid pins a given shape harder. kappa's scatter about the fit,
//  0.0094 / 0.0060 / 0.0046 / 0.0026 per cell at R = 8 / 16 / 32 / 64, falls,
//  but more slowly than 1/R. This is the reading validation/sessile_drop offers
//  for its contact-angle window, measured here with no wall.
//
//  -hf (2026-11-04) runs every section with FreeSurfaceSolver::height_function,
//  the curvature from column heights. The gated sections all pass with it, and
//  better than the default: Laplace +0.85 / +0.32 / +0.34 / -0.02 % at
//  R = 8 / 12 / 16 / 24, spurious currents at round-off (5e-16 to 4e-14). Under
//  -rate it REMOVES THE STALL AND NOT THE SLOWING: rate over the exact one, the
//  same seed and Oh as the table above, 1.030 / 0.897 / 0.795 / 0.695 at
//  R = 8 / 16 / 32 / 64, every decay a clean exponential (rms of ln eps 0.006-
//  0.011), and turned 45 degrees (-diag) 0.738 / 0.696 / 0.367 / 0.284.
//  Oh = 10 gives 1.024 / 0.895 / 0.806 at R = 8 / 16 / 32. On exact fills
//  (-exact) its kappa carries 0.997 / 1.001 / 0.997 of the mode-2 curvature at
//  t = 0 at R = 8 / 16 / 32; the dynamics then ripple the fill and it keeps
//  about half -- the solver member's banner has that.
//  THE CONTROLS that make these numbers mean something. -nosigma runs the
//  solver with no surface tension against the same reference: the ellipse does
//  not move (rate 0, mass to 1e-12), so everything measured is capillary.
//  -exact seeds each cell with the exact area of the ellipse in it (the disc's
//  closed form through an affine map), which puts R from the volume at 8.0000 /
//  16.0000 / 32.0000. -fo X sets the conversion hysteresis, fill_offset: 1e-5
//  against 1e-3 moves the -hf rates by under 1 %. -dump FILE writes the last
//  drop's final state, on which a second, Python implementation of the stencil
//  agreed with the solver's kappa to 1.4e-17.
//==============================================================================

#include "core/Types.hpp"
#include "solver/FreeSurfaceSolver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace lbm;

using L  = D2Q9;
using FS = FreeSurfaceSolver<L>;

namespace {

// -nrec: FreeSurfaceSolver::normal_reconstruction, opt-in (2026-11-03). The
// default run is unchanged; this measures what the refinement costs here.
bool g_nrec = false;
// -hf: FreeSurfaceSolver::height_function, opt-in (2026-11-04), in every section.
bool g_hf = false;
// -rate -nosigma: the CONTROL. sigma and the reference are computed as usual
// but the solver runs with no surface tension, so nothing drives the shape and
// any relaxation it shows is numerical.
bool g_nosig = false;
// -rate -exact: seed each cell with the exact area of the ellipse inside it,
// rather than the linear cell fraction (aligned ellipse only).
bool g_exact = false;
// -rate -dump FILE: the last drop's final state, every cell: x y flag fill kappa nx ny.
const char* g_dump = nullptr;
// -rate -fo X: FreeSurfaceSolver::fill_offset, the conversion hysteresis (default 1e-3).
double g_fo = -1.0;

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
  s.set_normal_reconstruction(g_nrec);
  s.set_height_function(g_hf);
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

// ---- -rate: THE RELAXATION RATE OF A FREE DROP -------------------------------
// See the banner for the reference and what it tests.

// rho_n = q J_n'(q) / J_n(q) = n - q J_{n+1}(q) / J_n(q), the ratio from the
// backward recurrence t_k = J_{k+1}/J_k = 1 / (2(k+1)/q - t_{k+1}). Valid below
// the first zero of J_n (5.14 for n = 2); the overdamped root sits at
// q = sqrt(lambda/Oh), about 1/Oh, nowhere near it.
double bessel_rho(int n, double q) {
  double t = 0.0;
  for (int k = n + 60; k >= n; --k) t = 1.0 / (2.0 * (k + 1) / q - t);
  return double(n) - q * t;
}

// The linear dispersion relation of mode n at a real decay rate lambda, in
// units rho = R = sigma = 1, so that mu = nu = Oh. The flow is
// psi = (A r^n + B J_n(q r)) sin(n theta) e^{-lambda t} with q^2 = lambda/Oh;
// zero tangential stress fixes A/B, the kinematic condition fixes B per unit
// amplitude, and this returns the normal-stress balance, -p + 2 mu du_r/dr +
// sigma (n^2 - 1) a, which vanishes at a root.
double dispersion(int n, double Oh, double lam) {
  const double q = std::sqrt(lam / Oh);
  const double rho = bessel_rho(n, q);
  const double c = 2.0 * n * (n - 1);
  const double G = 2.0 * n * n - q * q - 2.0 * rho;   // tangential stress / (B J_n)
  const double b = -lam / (n * (1.0 - G / c));       // B J_n per unit amplitude
  const double A = -b * G / c;
  return -lam * A + 2.0 * Oh * (n * (n - 1) * A + n * b * (rho - 1.0)) + (n * n - 1.0);
}

// The slow overdamped root, in units of sqrt(sigma / (rho R^3)). Bisection on
// [0.5, 2] x the Stokes rate n/(2 Oh): the function is positive at the lower
// end and negative at the upper for every Oh >= 2, checked before this was
// written. Below Oh = 2 the mode approaches the oscillatory regime and the
// bracket is refused rather than trusted.
double exact_rate(int n, double Oh) {
  const double lS = double(n) / (2.0 * Oh);
  double lo = 0.5 * lS, hi = 2.0 * lS;
  if (!(dispersion(n, Oh, lo) > 0.0 && dispersion(n, Oh, hi) < 0.0)) {
    std::fprintf(stderr, "surface_tension: no overdamped root bracketed at Oh = %g\n", Oh);
    std::exit(2);
  }
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (dispersion(n, Oh, mid) > 0.0 ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

// The mode-2 amplitude from the second moments of the colour function about its
// centroid. For r = R + a cos 2(theta - theta0) the two components are
// (a/R) cos 2theta0 and (a/R) sin 2theta0, so the magnitude is a/R whichever
// way the ellipse points -- which is what lets -diag use the same number.
template <class HF, class HE>
void drop_shape(const Domain& d, Index N, const HF& hf, const HE& he,
                double& vol, double& ec, double& es, double& cxo, double& cyo) {
  auto colour = [&](Index n) {
    const std::uint8_t f = hf(n);
    return (f == FsFluid) ? 1.0 : (f == FsInterface ? double(he(n)) : 0.0);
  };
  double cx = 0, cy = 0, m = 0;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const double c = colour(d.id(x, y));
      cx += c * double(x);  cy += c * double(y);  m += c;
    }
  cx /= m;  cy /= m;
  double ixx = 0, iyy = 0, ixy = 0;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const double c = colour(d.id(x, y));
      const double X = double(x) - cx, Y = double(y) - cy;
      ixx += c * X * X;  iyy += c * Y * Y;  ixy += c * X * Y;
    }
  vol = m;
  ec = (ixx - iyy) / (2.0 * (ixx + iyy));
  es = ixy / (ixx + iyy);
  cxo = cx;  cyo = cy;
}

// WHAT THE CURVATURE SEES of that shape: the solver's own kappa at the interface
// cells, fitted as k0 + kc cos 2theta + ks sin 2theta about the centroid, and its
// mode-2 part projected on the shape's own orientation. A correct curvature
// gives 3 eps / R there (kappa = (1 + 3 eps cos 2theta)/R to first order), so
// `k2` divided by that is 1, and 0 means the deformation is invisible to the
// force that should remove it. `rms` is what the fit leaves, in 1/cells.
template <class HF, class HK>
void kappa_mode2(const Domain& d, Index N, const HF& hf, const HK& hk,
                 double cx, double cy, double ec, double es,
                 double& k0, double& k2, double& rms) {
  double M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, v[3] = {0, 0, 0};
  std::vector<double> th, kk;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const Index n = d.id(x, y);
      if (hf(n) != FsInterface) continue;
      const double t = std::atan2(double(y) - cy, double(x) - cx), k = double(hk(n));
      const double b[3] = {1.0, std::cos(2 * t), std::sin(2 * t)};
      for (int i = 0; i < 3; ++i) {
        v[i] += b[i] * k;
        for (int j = 0; j < 3; ++j) M[i][j] += b[i] * b[j];
      }
      th.push_back(t);  kk.push_back(k);
    }
  auto det3 = [](const double A[3][3]) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  };
  const double D = det3(M);
  double c[3];
  for (int col = 0; col < 3; ++col) {        // Cramer's rule
    double A[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) A[i][j] = (j == col) ? v[i] : M[i][j];
    c[col] = det3(A) / D;
  }
  const double e = std::hypot(ec, es);
  k0 = c[0];
  k2 = (e > 0) ? (c[1] * ec + c[2] * es) / e : 0.0;
  double r2 = 0;
  for (std::size_t i = 0; i < th.size(); ++i) {
    const double r = kk[i] - (c[0] + c[1] * std::cos(2 * th[i]) + c[2] * std::sin(2 * th[i]));
    r2 += r * r;
  }
  rms = th.empty() ? 0.0 : std::sqrt(r2 / double(th.size()));
}

// -v: kappa against angle in 16 bins, as kappa R, beside what the shape implies,
// 1 + 3 eps cos 2(theta - theta0). Reads the solver's kappa as last computed.
template <class HF, class HE, class HK>
void kappa_table(const Domain& d, Index N, const HF& hf, const HE& he, const HK& hk) {
  double vol, ec, es, cx, cy;
  drop_shape(d, N, hf, he, vol, ec, es, cx, cy);
  const double R = std::sqrt(vol / M_PI), th0 = 0.5 * std::atan2(es, ec), e = std::hypot(ec, es);
  double sk[16] = {0}, sk2[16] = {0}; int cnt[16] = {0};
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const Index n = d.id(x, y);
      if (hf(n) != FsInterface) continue;
      double t = std::atan2(double(y) - cy, double(x) - cx) - th0;
      t = std::fmod(t + 4 * M_PI, 2 * M_PI);
      const int b = std::min(15, int(t / (2 * M_PI) * 16));
      const double k = double(hk(n)) * R;
      sk[b] += k;  sk2[b] += k * k;  ++cnt[b];
    }
  std::printf("      kappa R by angle from the long axis (eps = %.5f):\n", e);
  std::printf("      %-8s %-6s %-10s %-10s %-10s\n", "deg", "cells", "mean", "shape", "rms");
  for (int b = 0; b < 16; ++b) {
    if (!cnt[b]) continue;
    const double tc = (b + 0.5) * 2 * M_PI / 16, m = sk[b] / cnt[b];
    std::printf("      %-8.1f %-6d %-10.5f %-10.5f %-10.5f\n", tc * 180 / M_PI, cnt[b], m,
                1.0 + 3.0 * e * std::cos(2 * tc), std::sqrt(std::max(0.0, sk2[b] / cnt[b] - m * m)));
  }
}

// -exact: the area of the disc x^2 + y^2 <= r^2 inside [x0,x1] x [y0,y1], in
// closed form -- the integral of min(y1, s) - max(y0, -s), s = sqrt(r^2 - x^2),
// split where either bound changes branch, with F the antiderivative of s. An
// ellipse of semi-axes p (x) and q (y) is this disc of radius q with x scaled by
// q/p, times p/q.
KOKKOS_INLINE_FUNCTION double disc_rect_area(double r, double x0, double x1, double y0, double y1) {
  if (r <= 0.0) return 0.0;
  const double a = x0 > -r ? x0 : -r, b = x1 < r ? x1 : r;
  if (a >= b) return 0.0;
  auto S = [&](double x) { const double v = r * r - x * x; return v > 0.0 ? Kokkos::sqrt(v) : 0.0; };
  auto F = [&](double x) {
    double u = x / r;
    u = u < -1.0 ? -1.0 : (u > 1.0 ? 1.0 : u);
    return 0.5 * (x * S(x) + r * r * Kokkos::asin(u));
  };
  double br[6];
  int nb = 0;
  br[nb++] = a;
  for (double y : {y0, y1}) {
    const double ay = y < 0.0 ? -y : y;
    if (ay < r) {
      const double xs = Kokkos::sqrt(r * r - ay * ay);
      if (-xs > a && -xs < b) br[nb++] = -xs;
      if (xs > a && xs < b) br[nb++] = xs;
    }
  }
  br[nb++] = b;
  for (int i = 1; i < nb; ++i)            // insertion sort, at most six points
    for (int j = i; j > 0 && br[j] < br[j - 1]; --j) { const double t = br[j]; br[j] = br[j - 1]; br[j - 1] = t; }
  double A = 0.0;
  for (int i = 0; i + 1 < nb; ++i) {
    const double u = br[i], v = br[i + 1];
    if (v <= u) continue;
    const double sm = S(0.5 * (u + v));
    const bool up_s = sm < y1, lo_s = -sm > y0;
    if ((up_s ? sm : y1) <= (lo_s ? -sm : y0)) continue;
    const double Is = F(v) - F(u);
    A += (up_s ? Is : y1 * (v - u)) - (lo_s ? -Is : y0 * (v - u));
  }
  return A;
}

// -v with -hf: where the height stencil fails, from the solver's own test run on
// host copies (the colour is rebuilt from flags and fill, which is exact without
// wetting). Prints each failing cell's angle, fill, normal and its three centre
// columns along the dominant axis, liquid end first.
template <class HF, class HE, class HN>
void hf_failures(const Domain& d, Index N, const HF& hf, const HE& he,
                 const HN& hnx, const HN& hny, double cx, double cy, int maxprint) {
  Kokkos::View<Real*, HostSpace> col("col", d.n_padded);
  for (Index n = 0; n < d.n_padded; ++n)
    col(n) = hf(n) == FsFluid ? Real(1) : (hf(n) == FsInterface ? he(n) : Real(0));
  int shown = 0, total = 0;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const Index n = d.id(x, y);
      if (hf(n) != FsInterface) continue;
      Real k;
      if (FS::height_curvature(d, hf, col, hnx(n), hny(n), Real(0), n, k)) continue;
      ++total;
      if (shown >= maxprint) continue;
      ++shown;
      const double th = std::atan2(double(y) - cy, double(x) - cx) * 180 / M_PI;
      const int a = std::fabs(double(hny(n))) > std::fabs(double(hnx(n))) ? 1 : 0;
      const int sg = (a ? hny(n) : hnx(n)) > 0 ? 1 : -1;
      std::printf("      fallback at (%d,%d) %7.1f deg fill %.3f n = (%.3f, %.3f), liquid end first:", int(x), int(y), th,
                  double(he(n)), double(hnx(n)), double(hny(n)));
      for (int kk = -1; kk <= 1; ++kk) {
        std::printf("  [");
        for (int j = -3; j <= 3; ++j) {
          Index q[2] = {x, y};
          q[a] += Index(-sg * j);          // liquid end first
          q[1 - a] += Index(kk);
          const Index m = d.id((q[0] % N + N) % N, (q[1] % N + N) % N);
          const char c = hf(m) == FsFluid ? 'F' : hf(m) == FsInterface ? 'i' : hf(m) == FsGas ? '.' : '#';
          std::printf("%c", c);
        }
        std::printf("]");
      }
      std::printf("\n");
    }
  std::printf("      %d interface cells fail the height stencil\n", total);
}

struct Rate {
  double R_meas = 0, Oh = 0, eps0 = 0, eps_end = 0;
  double lam_ref = 0, lam = 0, lam1 = 0, lam2 = 0, rms = 0;
  double g_end = 0, knoise_end = 0;      // k2 / (3 eps/R) and the fit residual x R, at the end
  long   hf_used = 0, hf_fell = 0;       // -hf: the last curvature pass's coverage
  double mass_drift = 0, seconds = 0;
  std::size_t steps = 0;
  int samples = 0;
  bool finite = true;
};

// Least-squares slope of y on x over [i0, i1).
double ls_slope(const std::vector<double>& x, const std::vector<double>& y,
                std::size_t i0, std::size_t i1) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double k = double(i1 - i0);
  for (std::size_t i = i0; i < i1; ++i) {
    sx += x[i];  sy += y[i];  sxx += x[i] * x[i];  sxy += x[i] * y[i];
  }
  return (k * sxy - sx * sy) / (k * sxx - sx * sx);
}

// One drop: sigma follows from Oh at the SEEDED radius, the reference from the
// MEASURED one (the volume of the seed), and the fit skips two viscous times
// R^2/nu, by which the fast viscous modes the impulsive start excites -- the
// slowest at about 26 nu/R^2 for n = 2 -- are gone.
Rate relax(Index N, double R0, double Oh, double nu, double ecc, bool diag,
           double efolds, bool verbose) {
  Rate Q;
  const double sigma = nu * nu / (Oh * Oh * R0);     // Oh = nu / sqrt(sigma R), rho = 1
  Domain d(N, N, 1, true, true, true);
  FS s(d);
  s.coll.omega = FS::omega_from_viscosity(Real(nu));
  s.set_surface_tension(g_nosig ? Real(0) : Real(sigma));
  s.set_normal_reconstruction(g_nrec);
  s.set_height_function(g_hf && !g_nosig);
  if (g_fo >= 0.0) s.fill_offset = Real(g_fo);
  s.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });

  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  const Real Rr0 = Real(R0), xc = Real(N) * Real(0.5);
  const Real ea = Real(R0 * std::sqrt(ecc)), eb = Real(R0 / std::sqrt(ecc));
  const Real is2 = Real(1.0 / std::sqrt(2.0));
  const bool dg = diag, ex = g_exact;
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    Real x = Real(px - hx) - xc, y = Real(py - hy) - xc;
    if (dg) { const Real u = (x + y) * is2, v = (y - x) * is2; x = u; y = v; }
    const Real r = Kokkos::sqrt((x / ea) * (x / ea) + (y / eb) * (y / eb)) * Rr0;
    Real e = Rr0 + Real(0.5) - r;
    if (ex) {
      const double pa = double(ea), qb = double(eb), k = qb / pa;
      double A = disc_rect_area(qb, (double(x) - 0.5) * k, (double(x) + 0.5) * k,
                                double(y) - 0.5, double(y) + 0.5) / k;
      // A cell wholly inside sums to 1 only to round-off, and anything below 1
      // seeds an Interface cell: the first run had 1490 of them at R = 32.
      if (A > 1.0 - 1e-9) A = 1.0;
      if (A < 1e-9) A = 0.0;
      e = Real(A);
    }
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, Real(1)};
  });

  auto hf = Kokkos::create_mirror_view(HostSpace{}, s.flags());
  auto he = Kokkos::create_mirror_view(HostSpace{}, s.fill());
  auto hk = Kokkos::create_mirror_view(HostSpace{}, s.kappa());
  double g = 0, kn = 0;                     // the curvature diagnostic, last sample
  auto measure = [&](double& vol) {
    Kokkos::deep_copy(hf, s.flags());
    Kokkos::deep_copy(he, s.fill());
    // kappa() is what the LAST step started from, so a cell that changed type in
    // that step would carry a stale value (zero, for one that was Fluid or Gas).
    // Recomputing it here changes nothing the next step reads: every step
    // recomputes the curvature from the state before it does anything else.
    if (!g_nosig) {
      s.compute_curvature();
      Kokkos::deep_copy(hk, s.kappa());
    }
    double ec, es, cx, cy;
    drop_shape(d, N, hf, he, vol, ec, es, cx, cy);
    const double e = std::hypot(ec, es), Rv = std::sqrt(vol / M_PI);
    double k0 = 0, k2 = 0, kr = 0;
    if (!g_nosig) kappa_mode2(d, N, hf, hk, cx, cy, ec, es, k0, k2, kr);
    g = (e > 0) ? k2 / (3.0 * e / Rv) : 0.0;
    kn = kr * Rv;
    return e;
  };

  double vol = 0;
  Q.eps0 = measure(vol);
  if (verbose) {
    std::printf("      t = 0: k2/(3eps/R) = %.4f\n", g);
    if (!g_nosig) kappa_table(d, N, hf, he, hk);
  }
  Q.R_meas = std::sqrt(vol / M_PI);
  Q.Oh = nu / std::sqrt(sigma * Q.R_meas);
  Q.lam_ref = exact_rate(2, Q.Oh) * std::sqrt(sigma / (Q.R_meas * Q.R_meas * Q.R_meas));

  const double t0 = 2.0 * Q.R_meas * Q.R_meas / nu;
  const double window = efolds / Q.lam_ref;
  const std::size_t every = std::max<std::size_t>(1, std::size_t(window / 100.0));
  const std::size_t nsteps = std::size_t(t0 + window) + 1;

  const double m0 = double(s.total_mass());
  std::vector<double> tt, ll;
  const auto clock0 = std::chrono::steady_clock::now();
  for (std::size_t t = 0; t < nsteps;) {
    const std::size_t k = std::min(every, nsteps - t);
    for (std::size_t j = 0; j < k; ++j) s.step();
    t += k;
    double v;
    const double e = measure(v);
    if (!std::isfinite(e) || !std::isfinite(v)) { Q.finite = false; break; }
    if (verbose)
      std::printf("      t = %9zu  t lam_ref = %7.4f  eps/eps0 = %.6f  k2/(3eps/R) = %8.4f"
                  "  resid x R = %.4f\n", t, double(t) * Q.lam_ref, e / Q.eps0, g, kn);
    if (double(t) >= t0) { tt.push_back(double(t)); ll.push_back(std::log(e)); }
    Q.eps_end = e;
    Q.g_end = g;  Q.knoise_end = kn;
  }
  Q.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - clock0).count();
  Q.hf_used = long(s.height_function_cells());
  Q.hf_fell = long(s.height_function_fallbacks());
  if (verbose && Q.finite && !g_nosig) kappa_table(d, N, hf, he, hk);
  if (g_dump && Q.finite && !g_nosig) {
    auto hnx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.normal_x());
    auto hny = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.normal_y());
    if (FILE* fp = std::fopen(g_dump, "w")) {
      std::fprintf(fp, "# N %d R %.6f eps %.8e\n", int(N), Q.R_meas, Q.eps_end);
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x) {
          const Index n = d.id(x, y);
          std::fprintf(fp, "%d %d %d %.17g %.17g %.17g %.17g\n", int(x), int(y), int(hf(n)),
                       double(he(n)), double(hk(n)), double(hnx(n)), double(hny(n)));
        }
      std::fclose(fp);
    }
  }
  if (verbose && Q.finite && g_hf && !g_nosig) {
    auto hnx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.normal_x());
    auto hny = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.normal_y());
    double v, ec, es, cx, cy;
    drop_shape(d, N, hf, he, v, ec, es, cx, cy);
    hf_failures(d, N, hf, he, hnx, hny, cx, cy, 12);
  }
  Q.steps = nsteps;
  Q.mass_drift = (double(s.total_mass()) - m0) / m0;
  Q.samples = int(tt.size());
  if (!Q.finite || tt.size() < 8) { Q.finite = false; return Q; }

  const std::size_t n = tt.size(), h = n / 2;
  Q.lam  = -ls_slope(tt, ll, 0, n);
  Q.lam1 = -ls_slope(tt, ll, 0, h);
  Q.lam2 = -ls_slope(tt, ll, h, n);
  double c = 0;
  for (std::size_t i = 0; i < n; ++i) c += ll[i] + Q.lam * tt[i];
  c /= double(n);
  double r2 = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double r = ll[i] - (c - Q.lam * tt[i]);
    r2 += r * r;
  }
  Q.rms = std::sqrt(r2 / double(n));
  return Q;
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = false;
  std::size_t nsteps = 20000;
  double sigma = 0.01;
  // -rate and its options: a measurement, run instead of the gated sections
  bool rate = false, diag = false;
  double oh = 10.0, ecc_r = 1.1, efolds = 2.0, rmin = 8.0, rmax = 32.0;
  for (int i = 1; i < argc; ++i) {
    if      (!std::strcmp(argv[i], "-v")) verbose = true;
    else if (!std::strcmp(argv[i], "-steps") && i + 1 < argc)
      nsteps = std::size_t(std::atof(argv[++i]));
    else if (!std::strcmp(argv[i], "-sigma") && i + 1 < argc)
      sigma = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-nrec")) g_nrec = true;
    else if (!std::strcmp(argv[i], "-hf")) g_hf = true;
    else if (!std::strcmp(argv[i], "-nosigma")) g_nosig = true;
    else if (!std::strcmp(argv[i], "-exact")) g_exact = true;
    else if (!std::strcmp(argv[i], "-dump") && i + 1 < argc) g_dump = argv[++i];
    else if (!std::strcmp(argv[i], "-fo") && i + 1 < argc) g_fo = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-rate")) rate = true;
    else if (!std::strcmp(argv[i], "-diag")) diag = true;
    else if (!std::strcmp(argv[i], "-oh") && i + 1 < argc) oh = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-ecc") && i + 1 < argc) ecc_r = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-efold") && i + 1 < argc) efolds = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-rmin") && i + 1 < argc) rmin = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-rmax") && i + 1 < argc) rmax = std::atof(argv[++i]);
    else if (!std::strncmp(argv[i], "--kokkos", 8)) {}
    else {
      std::fprintf(stderr, "surface_tension: unknown option '%s'. NOTHING WAS RUN.\n"
                           "  -v  -steps N  -sigma S  -nrec  -hf\n"
                           "  -rate [-oh X] [-ecc E] [-efold W] [-rmin R] [-rmax R] [-diag] [-nosigma] [-exact] [-dump FILE] [-fo X]\n",
                   argv[i]);
      return 2;
    }
  }
  if (g_exact && (!rate || diag)) {
    std::fprintf(stderr, "surface_tension: -exact seeds the aligned -rate ellipse only. NOTHING WAS RUN.\n");
    return 2;
  }
  if (g_nosig && !rate) {
    std::fprintf(stderr, "surface_tension: -nosigma is a control for -rate only. NOTHING WAS RUN.\n");
    return 2;
  }
  if (rate && !(oh >= 2.0 && ecc_r > 1.0 && efolds > 0.0 && rmin >= 4.0 && rmax >= rmin)) {
    std::fprintf(stderr, "surface_tension -rate: need -oh >= 2, -ecc > 1, -efold > 0,"
                         " 4 <= -rmin <= -rmax. NOTHING WAS RUN.\n");
    return 2;
  }

  Kokkos::initialize(argc, argv);
  int failures = 0, checks = 0;
  if (rate) {
    std::printf("relaxation of a free 2-D drop, mode 2, against the exact linear rate\n");
    std::printf("backend %s   precision %s   %s%s\n", Kokkos::DefaultExecutionSpace::name(),
                precision_name(), g_nrec ? "normal reconstruction ON  " : "",
                g_hf ? "height functions ON" : "");
    if (g_nosig)
      std::printf("CONTROL: surface tension OFF in the solver; the reference is the run with it on\n");
    const double nu = 1.0 / 6.0;
    std::printf("Oh = %g, aspect %.3f %s, nu = 1/6, fit over %.2f e-folds after 2 R^2/nu\n",
                oh, ecc_r, diag ? "(diagonal)" : "(along x)", efolds);
    std::printf("exact rate / Stokes rate sigma/(mu R) at this Oh: %.6f\n\n",
                exact_rate(2, oh) * oh);
    std::printf("   %-5s %-5s %-8s %-8s %-11s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-10s %-7s\n",
                "R", "N", "R meas", "eps0", "lam_ref", "lam/ref", "1st half", "2nd half",
                "rms ln", "eps end", "k2 end", "resid*R", "mass", "sec");
    for (double R0 = rmin; R0 <= rmax + 1e-9; R0 *= 2.0) {
      const Index N = Index(2.0 * R0 + 16.0);
      const Rate Q = relax(N, R0, oh, nu, ecc_r, diag, efolds, verbose);
      ++checks;
      if (!Q.finite) {
        ++failures;
        std::printf("   %-5.0f %-5ld NON-FINITE after %zu steps\n", R0, long(N), Q.steps);
        continue;
      }
      std::printf("   %-5.0f %-5ld %-8.4f %-8.5f %-11.4e %-9.5f %-9.5f %-9.5f %-9.2e %-9.2e %-9.4f %-9.4f %-10.2e %-7.1f\n",
                  R0, long(N), Q.R_meas, Q.eps0, Q.lam_ref, Q.lam / Q.lam_ref,
                  Q.lam1 / Q.lam_ref, Q.lam2 / Q.lam_ref, Q.rms, Q.eps_end / Q.eps0,
                  Q.g_end, Q.knoise_end, Q.mass_drift, Q.seconds);
      if (g_hf)
        std::printf("         height functions at the end: %ld interface cells, %ld kept pass 4's kappa\n",
                    Q.hf_used, Q.hf_fell);
    }
    std::printf("\n   lam/ref is the fitted rate over the exact one; '1st half' and '2nd half'\n"
                "   fit each half of the window alone, so a non-exponential decay shows as\n"
                "   a difference. 'k2 end' is the mode-2 part of the solver's own kappa over\n"
                "   the exact 3 eps/R for the shape present at the end: 1 if the curvature\n"
                "   sees the deformation, 0 if it is blind to it; 'resid*R' is what the\n"
                "   mode-2 fit leaves, relative to 1/R. A measurement, not a gate.\n");
  } else {
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
  if (rate) {
    std::printf("\n[surface_tension -rate] %d drops, %d non-finite\n", checks, failures);
    return failures == 0 ? 0 : 1;
  }

  std::printf("\n[surface_tension] %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
