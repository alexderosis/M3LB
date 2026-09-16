//==============================================================================
//  PHYSICS checks for the CUDA core, run WITHOUT a GPU.
//
//  host_check.cpp verifies pieces: a velocity set, a transform, one collision.
//  This file runs whole simulations and compares them against solutions that are
//  known in closed form. It can do that on a laptop because every per-node update
//  is an LBM_HD function and the CUDA kernels are three lines of index arithmetic
//  around them -- so these are the kernels, driven by a for-loop instead of by a
//  grid of threads. Under Esoteric Pull each slot has exactly one writer per
//  step, so the loop is not an approximation to the launch; it is the same
//  computation.
//
//  What it covers, and what each case would catch on its own:
//
//    1  Poiseuille flow between bounce-back walls   geometry, forcing, and where
//                                                   the no-slip plane really sits
//    1d regularised walls                     the wall ON the node, and what
//                                                   BC3 costs in mass
//    1c shifted storage                        f_i - w_i, and the three decades
//                                                   of FP32 amplitude it recovers
//    2  a closed box                                that geometry writes every
//                                                   slot exactly once
//    3  an insulating box                           the scalar is conserved
//    4  conduction between Dirichlet walls          anti-bounce-back, and where
//                                                   ITS plane sits
//    5  a decaying sinusoid                         the D3Q7 diffusivity, cs^2=1/4
//    6  an advected sinusoid                        the advective flux
//    6b an open exit                              ScalarOutflow, against the
//                                                   bounce-back it replaces
//    7  uniform buoyancy                            the whole Boussinesq path
//    8  resistive decay                             induction alone, no flow
//    8b a magnetic Dirichlet wall              where the moment condition puts
//                                                   the boundary
//    9  a shear Alfven wave                         the Lorentz coupling, the
//                                                   induction equation, and the
//                                                   ORDER in which they are
//                                                   evaluated
//
//  Case 9 is the one that matters most. It is an exact solution of the full
//  NONLINEAR incompressible MHD equations, and it fails in a distinctive way if
//  the two-way coupling is lagged by a step: the damping error then GROWS with
//  resolution while the phase speed converges cleanly.
//
//  Build:  c++ -std=c++17 -O2 -Iinclude test/host_physics.cpp -o host_physics
//==============================================================================
#include "lbm/hostsim.hpp"

#include <cmath>
#include <functional>
#include <cstdio>
#include <vector>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures = 0;
static const bool fp64 = (sizeof(Real) == 8);

static void check(bool ok, const char* what, double got, double want, const char* unit = "") {
  const double rel = (want != 0.0) ? std::fabs(got - want) / std::fabs(want)
                                   : std::fabs(got - want);
  std::printf("  %s  %-52s %12.6g vs %-12.6g %s(%.2e)\n", ok ? "PASS" : "FAIL",
              what, got, want, unit, rel);
  if (!ok) ++failures;
}

static void note(const char* s) { std::printf("        %s\n", s); }

//------------------------------------------------------------------------------
// Amplitude and phase of the fundamental x-mode, averaged over y and z.
// f(x) ~ amp * sin(k x + phase), k = 2 pi / nx.
//------------------------------------------------------------------------------
struct Mode { double amp, phase; };

static Mode fit_x(const Real* v, int nx, int ny, int nz) {
  const double k = 2.0 * M_PI / nx;
  const double inv = 1.0 / (double(ny) * nz);
  double S = 0, C = 0;
  for (int x = 0; x < nx; ++x) {
    double p = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y) p += double(v[node_id(x, y, z, nx, ny)]);
    p *= inv;
    S += p * std::sin(k * x);
    C += p * std::cos(k * x);
  }
  return {2.0 * std::sqrt(S * S + C * C) / nx, std::atan2(C, S)};
}

// Continue a phase sequence across the +-pi branch cut.
static double unwrap(double prev, double now) {
  while (now - prev >  M_PI) now -= 2.0 * M_PI;
  while (now - prev < -M_PI) now += 2.0 * M_PI;
  return now;
}

// Least-squares slope of y against x through n samples.
static double slope(const std::vector<double>& x, const std::vector<double>& y) {
  const double n = double(x.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i];
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

//==============================================================================
//  1. Poiseuille flow between halfway bounce-back walls.
//
//  H fluid nodes sit at y = 1..H; the walls are the SOLID layers at y = 0 and
//  y = H+1, so the no-slip planes are at y = 0.5 and y = H + 0.5 and
//
//      u(y) = (G / 2 rho nu) (y - 0.5)(H + 0.5 - y),   u_max = G H^2 / 8 rho nu.
//
//  Getting the plane wrong by half a lattice unit gives a parabola that still
//  looks like a parabola and is wrong by O(1/H). This case is therefore as much
//  a test of where the wall IS as of whether it holds.
//==============================================================================
static void poiseuille(Op op, const char* name, double tol_fp64, double tol_fp32) {
  const int H = 16, nx = 4, nz = 4, ny = H + 2;
  const double nu = 1.0 / 6.0;
  const double umax = 0.05;
  const double G = 8.0 * 1.0 * nu * umax / (double(H) * double(H));
  const std::size_t T = 20000;

  host::Fluid fl(nx, ny, nz, op, Real(nu));

  std::vector<std::uint8_t> flags(std::size_t(nx) * ny * nz, std::uint8_t(Fluid));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      flags[std::size_t(node_id(x, 0,      z, nx, ny))] = Solid;
      flags[std::size_t(node_id(x, ny - 1, z, nx, ny))] = Solid;
    }
  fl.set_geometry(flags);

  BodyForce b;
  b.fx = Real(G);
  fl.set_force(b, ForceUniform);
  fl.initialise_with([](int, int, int) {
    return Macro{Real(1), Real(0), Real(0), Real(0)};
  });

  for (std::size_t t = 0; t < T; ++t) fl.step();

  std::vector<Real> rho, ux, uy, uz;
  fl.macroscopic_to_host(rho, ux, uy, uz);

  // The AMPLITUDE is fitted by least squares against the shape function, not
  // read off as the largest node value. For even H the parabola peaks at
  // y = (H+1)/2, which is half-way BETWEEN two nodes, so the largest sample is
  // below the continuous maximum by O(1/H^2) for reasons that have nothing to do
  // with the solver. Comparing the two is a test of arithmetic, not of physics.
  double worst_rel = 0, num = 0, den = 0;
  for (int y = 1; y <= H; ++y) {
    double u = 0;
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x) u += double(ux[std::size_t(node_id(x, y, z, nx, ny))]);
    u /= double(nx) * nz;
    const double shape = (y - 0.5) * (H + 0.5 - y);
    const double exact = (G / (2.0 * nu)) * shape;
    worst_rel = std::fmax(worst_rel, std::fabs(u - exact) / umax);
    num += u * shape;
    den += shape * shape;
  }
  const double amp = num / den;
  const double amp_exact = G / (2.0 * nu);
  char buf[128];
  const double tol = fp64 ? tol_fp64 : tol_fp32;
  std::snprintf(buf, sizeof buf, "Poiseuille %s: fitted parabola amplitude", name);
  check(std::fabs(amp - amp_exact) / amp_exact < tol, buf, amp, amp_exact);
  std::snprintf(buf, sizeof buf, "Poiseuille %s: worst profile error / u_max", name);
  check(worst_rel < tol, buf, worst_rel, 0.0);
}

//==============================================================================
//  1b. WHERE THE BOUNCE-BACK WALL ACTUALLY IS, and what TRT buys.
//
//  Case 1 fixes the viscosity at 1/6 (tau = 1) and asks whether the profile is
//  right. That is the one viscosity at which the question is easy: BGK is TRT
//  with omega_minus == omega_plus, so its magic parameter is
//  Lambda = (tau - 1/2)^2 = 1/4, which at tau = 1 happens to sit near the 3/16
//  that puts the wall halfway. Every other viscosity is a different story, and
//  a code validated only at tau = 1 does not know that.
//
//  The measurement. Fit a quadratic through the profile at y = 1..H and take its
//  lower root: that is where the fluid thinks the no-slip plane is. Halfway
//  bounce-back puts it at exactly 0.5, between the last fluid node at y = 1 and
//  the solid at y = 0.
//
//  MEASURED IN FP64, H = 16, u_max = 0.002, 200 000 steps:
//
//      nu       tau      BGK y0      TRT y0 (Lambda = 3/16)
//      0.005    0.515     0.5156      0.500000
//      0.02     0.56      0.5153      0.500000
//      0.1667   1.00      0.4948      0.500000
//      0.5      2.00      0.3299      0.500000
//      1.5      5.00     -1.0263      0.500000
//
//  TRT at 3/16 is EXACT, to every digit FP64 carries, at every viscosity from
//  tau = 0.515 to tau = 5. That is the theoretical result and it is worth
//  seeing it come out clean rather than approximately. BGK is not: at tau = 5
//  it places its wall a lattice OUTSIDE the solid node, so the effective channel
//  is three cells wider than the geometry and a Reynolds number computed from H
//  is 18% wrong with nothing in the run to say so.
//
//  AN EARLIER VERSION OF THIS CASE REPORTED TRT AS NOT EXACT -- 0.5008 at
//  tau = 0.515 drifting to 0.5943 at tau = 5 -- and explained the drift as the
//  compressible equilibrium and Guo forcing spoiling a result derived for the
//  Stokes problem. That was wrong, and instructive about how. The velocity the
//  diagnostic pass reported was missing Guo's half-force shift (see macro_node
//  in solver.cuh), so every profile was uniformly low by F/(2 rho). A CONSTANT
//  offset is invisible in a profile shape; what it does is move the fitted
//  ROOT, and it moves it further the larger the force -- and at fixed u_max the
//  force grows with the viscosity. So the artefact grew with tau and looked
//  exactly like a physical trend. It was not a Mach effect, which is what the
//  earlier version checked and correctly ruled out; ruling out one explanation
//  is not evidence for another.
//
//  THE SECOND HALF OF THE CASE IS THE ONE THAT PINS THE SCHEME: omega_minus is
//  FREE. Sweeping Lambda over a factor of twelve moves y0 across a wide range
//  and leaves the implied viscosity at 0.1666 to a few parts in 100 000, the
//  same for every Lambda. If the split leaked into the symmetric channel the
//  viscosity would move with it, and this is the only test here that would see
//  it.
//==============================================================================
static double fit_wall(const std::vector<double>& u, int H, double& curvature) {
  // Least squares u = a + b y + c y^2 over y = 1..H, then the lower root.
  double S[5] = {0, 0, 0, 0, 0}, T[3] = {0, 0, 0};
  for (int y = 1; y <= H; ++y) {
    double pw = 1.0;
    for (int k = 0; k < 5; ++k) { S[k] += pw; pw *= double(y); }
    pw = 1.0;
    for (int k = 0; k < 3; ++k) { T[k] += u[std::size_t(y - 1)] * pw; pw *= double(y); }
  }
  double M[3][4] = {{S[0], S[1], S[2], T[0]},
                    {S[1], S[2], S[3], T[1]},
                    {S[2], S[3], S[4], T[2]}};
  for (int i = 0; i < 3; ++i) {
    int piv = i;
    for (int r = i + 1; r < 3; ++r)
      if (std::fabs(M[r][i]) > std::fabs(M[piv][i])) piv = r;
    for (int k = 0; k < 4; ++k) std::swap(M[i][k], M[piv][k]);
    for (int r = 0; r < 3; ++r)
      if (r != i) {
        const double fac = M[r][i] / M[i][i];
        for (int k = i; k < 4; ++k) M[r][k] -= fac * M[i][k];
      }
  }
  const double a = M[0][3] / M[0][0], b = M[1][3] / M[1][1], c = M[2][3] / M[2][2];
  curvature = c;
  return (-b + std::sqrt(b * b - 4.0 * c * a)) / (2.0 * c);
}

static double channel_wall(Op op, double nu, double lambda, int H, std::size_t T,
                           double& curvature) {
  const int nx = 4, nz = 4, ny = H + 2;
  const double umax = 0.002;
  const double G = 8.0 * nu * umax / (double(H) * double(H));

  host::Fluid fl(nx, ny, nz, op, Real(nu));
  if (op == Op::TRT) fl.set_magic(Real(lambda));

  std::vector<std::uint8_t> flags(std::size_t(nx) * ny * nz, std::uint8_t(Fluid));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      flags[std::size_t(node_id(x, 0,      z, nx, ny))] = Solid;
      flags[std::size_t(node_id(x, ny - 1, z, nx, ny))] = Solid;
    }
  fl.set_geometry(flags);

  BodyForce b;
  b.fx = Real(G);
  fl.set_force(b, ForceUniform);
  fl.initialise_with([](int, int, int) {
    return Macro{Real(1), Real(0), Real(0), Real(0)};
  });
  for (std::size_t t = 0; t < T; ++t) fl.step();

  std::vector<Real> rho, ux, uy, uz;
  fl.macroscopic_to_host(rho, ux, uy, uz);
  std::vector<double> prof(std::size_t(H), 0.0);   // NOT prof(std::size_t(H)) -- that declares a function
  for (int y = 1; y <= H; ++y) {
    double u = 0;
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x) u += double(ux[std::size_t(node_id(x, y, z, nx, ny))]);
    prof[std::size_t(y - 1)] = u / (double(nx) * nz);
  }
  return fit_wall(prof, H, curvature);
}

static void wall_position() {
  const int H = 16;
  const std::size_t T = 40000;      // 30 time constants at the stiffest tau here
  double c = 0;

  // tau = 2, where BGK's own magic parameter is 2.25 against the wanted 3/16.
  const double nu = 0.5;
  const double y_bgk = channel_wall(Op::BGK, nu, 0.0,        H, T, c);
  const double y_trt = channel_wall(Op::TRT, nu, 3.0 / 16.0, H, T, c);
  check(std::fabs(y_bgk - 0.5) > 0.1, "wall at tau = 2: BGK misplaces it by > 0.1",
        y_bgk, 0.5);
  // Exact in FP64; the FP32 residual is round-off at u_max = 0.002, which is
  // close to that precision's floor for this measurement.
  check(std::fabs(y_trt - 0.5) < (fp64 ? 1e-9 : 2e-3),
        "wall at tau = 2: TRT puts it at 0.5 exactly", y_trt, 0.5);
  note("and it does so at every viscosity from tau = 0.515 to 5 -- see the banner");

  // omega_minus is free: Lambda moves the wall and NOT the viscosity.
  const double nu0 = 1.0 / 6.0;
  const double G0  = 8.0 * nu0 * 0.002 / (double(H) * double(H));
  double y_lo = 0, y_hi = 0, nu_lo = 0, nu_hi = 0;
  y_lo = channel_wall(Op::TRT, nu0, 1.0 / 12.0, H, T, c);  nu_lo = -G0 / (2.0 * c);
  y_hi = channel_wall(Op::TRT, nu0, 1.0,        H, T, c);  nu_hi = -G0 / (2.0 * c);
  check(std::fabs(y_lo - y_hi) > 0.05, "Lambda moves the wall position",
        std::fabs(y_lo - y_hi), 0.076);
  check(std::fabs(nu_lo - nu0) / nu0 < 1e-3 && std::fabs(nu_hi - nu0) / nu0 < 1e-3,
        "Lambda leaves the viscosity untouched", nu_hi, nu0);
}

//==============================================================================
//  1c. SHIFTED STORAGE: the same physics, three decades more of it in FP32.
//
//  The arrays hold either f_i or g_i = f_i - w_i. Populations are O(w_i) ~ 1e-1
//  while the collision works on differences O(1e-6), and the momentum sum
//  cancels those 1e-1 terms down to the flow speed -- so in FP32 both throw away
//  most of the mantissa. Storing g_i removes both cancellations, exactly:
//  sum_i c_i w_i = 0, so sum_i c_i g_i IS sum_i c_i f_i while every summand is
//  small.
//
//  THE CASE. A shear wave u_x = A sin(2 pi y / L) decays at nu k^2, and the
//  decay rate is measured from the fitted amplitude. Shrinking A shrinks nothing
//  about the physics -- the rate is amplitude-independent -- so any dependence
//  on A is arithmetic.
//
//  MEASURED, nu = 0.02, L = 32, 2000 steps, as a relative error in the rate:
//
//      A        raw (FP32)    shifted (FP32)     both (FP64)
//      1e-2     +4.97e-03     +5.22e-03          +5.234e-03
//      1e-3     +4.47e-03     +5.22e-03          +5.234e-03
//      1e-4     -3.79e-02     +5.22e-03          +5.234e-03
//      1e-5     -9.68e-01     +5.21e-03          +5.234e-03
//      1e-6     -1.02e+00     +5.22e-03          +5.234e-03
//
//  Two things to read off it.
//
//  FIRST, the FP64 column is one number. Raw and shifted agree to every digit
//  there, at every amplitude -- so this is a change of REPRESENTATION and not of
//  scheme, and +5.234e-3 is the discretisation error of the operator, which is
//  what it should be at these parameters.
//
//  SECOND, the FP32 columns separate completely. Raw storage is fine to about
//  A = 1e-3 and then collapses: at A = 1e-5 the measured decay rate is 3% of the
//  true one, i.e. the wave barely decays at all, and at 1e-6 it is worse than
//  useless. Shifted holds +5.22e-3 at every amplitude, within 0.4% of the FP64
//  answer. That is three decades of amplitude recovered, and this tree is FP32
//  by default because FP64 on a consumer NVIDIA part runs at 1/32 to 1/64 of the
//  rate -- so it is not a refinement, it is the difference between what the
//  device is for and what it can resolve.
//
//  AND NOTE HOW IT FAILS. Raw FP32 at A = 1e-5 does not produce NaN, does not
//  blow up, and does not look wrong: it produces a shear wave that sits there.
//  A run reporting an anomalously stable low-amplitude mode is what this looks
//  like from outside.
//==============================================================================
static double shear_amplitude(const std::vector<Real>& ux, int nx, int ny, int nz) {
  const double k = 2.0 * M_PI / ny;
  double S = 0;
  for (int y = 0; y < ny; ++y) {
    double p = 0;
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x) p += double(ux[std::size_t(node_id(x, y, z, nx, ny))]);
    S += (p / (double(nx) * nz)) * std::sin(k * y);
  }
  return 2.0 * S / ny;
}

static double shear_decay_error(bool shifted, double A) {
  const int nx = 4, ny = 32, nz = 4;
  const double nu = 0.02;
  const std::size_t T = 2000;
  const double k = 2.0 * M_PI / ny;

  host::Fluid fl(nx, ny, nz, Op::BGK, Real(nu));
  fl.set_shifted(shifted);
  fl.initialise_with([&](int, int y, int) {
    Macro m;
    m.rho = Real(1);
    m.ux = Real(A * std::sin(k * y));
    return m;
  });

  std::vector<Real> rho, ux, uy, uz;
  fl.macroscopic_to_host(rho, ux, uy, uz);
  const double a0 = shear_amplitude(ux, nx, ny, nz);
  for (std::size_t t = 0; t < T; ++t) fl.step();
  fl.macroscopic_to_host(rho, ux, uy, uz);
  const double a1 = shear_amplitude(ux, nx, ny, nz);

  const double rate = -std::log(a1 / a0) / double(T);
  const double exact = nu * k * k;
  return (rate - exact) / exact;
}

static void shifted_storage() {
  // A COMFORTABLE amplitude, where both work. This is the check that shifted
  // storage changes the representation and not the scheme: if it were a
  // different operator, it would differ HERE, where neither is short of
  // precision.
  const double big_raw = shear_decay_error(false, 1e-2);
  const double big_shf = shear_decay_error(true,  1e-2);
  check(std::fabs(big_raw - big_shf) < (fp64 ? 1e-9 : 5e-4),
        "at A = 1e-2 the two storages agree: same scheme", big_shf, big_raw);

  // A HARD one, where the cancellation bites. In FP64 nothing happens; in FP32
  // raw loses the wave and shifted does not.
  const double sml_raw = shear_decay_error(false, 1e-5);
  const double sml_shf = shear_decay_error(true,  1e-5);
  check(std::fabs(sml_shf - big_shf) < 5e-3,
        "at A = 1e-5 shifted still gets the decay rate", sml_shf, big_shf);
  if (fp64) {
    check(std::fabs(sml_raw - big_raw) < 1e-9,
          "  ... and in FP64 so does raw, which is the point", sml_raw, big_raw);
    note("the shift is a precision device; in FP64 there is nothing for it to do");
  } else {
    check(std::fabs(sml_raw) > 0.5,
          "  ... while raw FP32 has lost it entirely", sml_raw, 0.0);
    note("raw FP32 at A=1e-5 gives a wave that barely decays -- no NaN, no blow-up");
  }
  std::printf("        (A=1e-2: raw %+.3e shifted %+.3e | A=1e-5: raw %+.3e shifted %+.3e)\n",
              big_raw, big_shf, sml_raw, sml_shf);
}

//==============================================================================
//  1d. REGULARISED WALLS: the wall is ON the node.
//
//  Latt et al.'s BC3 (regularized.cuh). Halfway bounce-back puts the no-slip
//  plane midway between the last fluid node and the first solid node; this puts
//  it ON the boundary node, because that is the node whose velocity is imposed.
//  A channel of nodes w0..w1 is therefore w1 - w0 wide, not w1 - w0 + 1.
//
//  THE PLANE BEYOND EACH WALL MUST BE MARKED Excluded, and that is not a
//  formality. This code's indexing is periodic on every axis -- "outside the
//  fluid" is a geometry flag and nothing else -- so without it the
//  unknown-direction mask comes out EMPTY and the wall reconstructs against
//  directions that really did stream. Measured before the setup was fixed: a
//  uniform 15%-of-u_max slip, with a profile that still looked like a parabola.
//
//  1. COUETTE IS EXACT. Two walls, one moving, no body force: the linear profile
//     is machine-exact (7e-14) and both wall nodes hold their imposed velocity
//     to the last bit. Bounce-back cannot do either -- its wall is half a cell
//     away, so the node itself is never at the imposed value.
//
//  2. FORCED POISEUILLE CONVERGES AT SECOND ORDER. l2 profile error / u_max,
//     nu = 0.5 (tau = 2), 30 000 steps, FP64:
//
//         L     BGK        order    TRT        order    CM         order
//          9    1.178e-01    --     3.681e-03    --     2.945e-02    --
//         15    4.435e-02   1.91    1.386e-03   1.91    1.109e-02   1.91
//         31    1.075e-02   1.95    3.358e-04   1.95    2.687e-03   1.95
//         63    2.645e-03   1.98    8.266e-05   1.98    6.613e-04   1.98
//
//     Second order for all three, and TRT is THIRTY-TWO TIMES more accurate
//     than BGK at every resolution, CM four times. That is the same story case
//     1b tells about bounce-back, arriving through a completely different
//     boundary condition: what the wall costs depends on the free relaxation
//     rate, not on the wall alone.
//
//     THE ERROR IS NOT A SLIP LENGTH, which is what it looked like at first. It
//     reads as a uniform velocity pedestal, and fitting the parabola's root
//     gives an apparent slip of 0.176 cells at L = 15 -- but that apparent slip
//     halves when L doubles (0.287, 0.176, 0.086, 0.042), so it is a fixed
//     absolute velocity error, i.e. O(1/L^2) against u_max. A real slip length
//     would not move with L. It is also independent of u_max over two decades,
//     so it is linear, and it grows with tau.
//
//  3. CORNERS. A closed box with every wall at rest must stay at rest, and does:
//     max|u| is 5.8e-16 with the local closure and 8.9e-15 with the
//     finite-difference route. That is the sharpest corner test available,
//     because a corner rho extrapolated wrongly or a Pi built from a wrong
//     gradient both show up immediately as a flow out of nothing.
//
//  4. AND A WARNING THAT BELONGS WITH THEM. Regularised walls OVERWRITE
//     populations, so they are NOT mass conserving. At rest that costs nothing
//     -- the closed box holds its mass exactly -- but in a driven flow it leaks
//     steadily and does NOT saturate. Measured on the Re = 80 lid-driven cavity
//     at 32x32, as relative mass drift:
//
//         steps    4000     8000    12000    16000    20000
//         local   -3.4e-3  -6.8e-3 -1.02e-2 -1.36e-2 -1.70e-2
//         FD      -4.4e-3  -9.7e-3 -1.60e-2 -2.35e-2 -3.25e-2
//
//     Linear in time for the local closure, slightly worse than linear for the
//     FD route. A long cavity run therefore drifts in density and needs either
//     a pressure anchor or a renormalisation; do not read an absolute pressure
//     off one. This is a property of BC3, not a defect in this port -- the
//     parent says the same thing about its own boundary nodes -- but it is the
//     kind of property that is only discovered at the end of a long run.
//==============================================================================
static void reg_channel(std::vector<RegWallSpec>& spec, std::vector<std::uint8_t>& geo,
                        int nx, int ny, int nz, int w0, int w1, double Utop) {
  geo.assign(std::size_t(nx) * ny * nz, std::uint8_t(Fluid));
  spec.assign(std::size_t(nx) * ny * nz, RegWallSpec{});
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      geo[std::size_t(node_id(x, w0 - 1, z, nx, ny))] = Excluded;
      geo[std::size_t(node_id(x, w1 + 1, z, nx, ny))] = Excluded;
      spec[std::size_t(node_id(x, w0, z, nx, ny))] = RegWallSpec{NrmYm, 0, 0, 0};
      spec[std::size_t(node_id(x, w1, z, nx, ny))] = RegWallSpec{NrmYp, Real(Utop), 0, 0};
    }
}

// l2 profile error / u_max for a forced channel on regularised walls.
static double reg_poiseuille(Op op, int ny, double nu, double umax) {
  const int nx = 4, nz = 4, w0 = 1, w1 = ny - 2;
  const double L = double(w1 - w0);
  const double G = 8.0 * nu * umax / (L * L);

  host::Fluid fl(nx, ny, nz, op, Real(nu));
  if (op == Op::TRT) fl.set_magic(Real(3.0 / 16.0));
  std::vector<std::uint8_t> geo;
  std::vector<RegWallSpec> spec;
  reg_channel(spec, geo, nx, ny, nz, w0, w1, 0.0);
  fl.set_geometry(geo);
  fl.set_regularized_walls(spec);

  BodyForce b;  b.fx = Real(G);
  fl.set_force(b, ForceUniform);
  fl.initialise_with([](int, int, int) { Macro m; m.rho = Real(1); return m; });
  for (int t = 0; t < 30000; ++t) fl.step();

  std::vector<Real> rho, ux, uy, uz;
  fl.macroscopic_to_host(rho, ux, uy, uz);
  double s = 0;
  int cnt = 0;
  for (int y = w0; y <= w1; ++y) {
    double u = 0;
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x) u += double(ux[std::size_t(node_id(x, y, z, nx, ny))]);
    u /= double(nx) * nz;
    const double yy = double(y - w0);
    const double e = (G / (2.0 * nu)) * yy * (L - yy);
    s += (u - e) * (u - e);  ++cnt;
  }
  return std::sqrt(s / cnt) / umax;
}

static void regularized_walls() {
  const int nx = 4, nz = 4;

  // ---- 1. Couette is exact, and the wall node holds its velocity ------------
  {
    const int ny = 18, w0 = 1, w1 = ny - 2;
    const double nu = 0.5, U = 0.01, L = double(w1 - w0);
    host::Fluid fl(nx, ny, nz, Op::BGK, Real(nu));
    std::vector<std::uint8_t> geo;
    std::vector<RegWallSpec> spec;
    reg_channel(spec, geo, nx, ny, nz, w0, w1, U);
    fl.set_geometry(geo);
    fl.set_regularized_walls(spec);
    fl.initialise_with([](int, int, int) { Macro m; m.rho = Real(1); return m; });
    for (int t = 0; t < 60000; ++t) fl.step();

    std::vector<Real> rho, ux, uy, uz;
    fl.macroscopic_to_host(rho, ux, uy, uz);
    double worst = 0;
    for (int y = w0; y <= w1; ++y) {
      double u = 0;
      for (int z = 0; z < nz; ++z)
        for (int x = 0; x < nx; ++x) u += double(ux[std::size_t(node_id(x, y, z, nx, ny))]);
      u /= double(nx) * nz;
      worst = std::fmax(worst, std::fabs(u - U * double(y - w0) / L) / U);
    }
    check(worst < (fp64 ? 1e-10 : 1e-4),
          "Couette on regularised walls is EXACT", worst, 0.0);
    const double ub = double(ux[std::size_t(node_id(0, w0, 0, nx, ny))]);
    const double ut = double(ux[std::size_t(node_id(0, w1, 0, nx, ny))]);
    check(std::fabs(ub) < 1e-12 && std::fabs(ut - U) < (fp64 ? 1e-12 : 1e-8),
          "  ... and both wall NODES hold their imposed velocity", ut, U);
    note("bounce-back can do neither: its wall is half a cell from any node");
  }

  // ---- 2. forced Poiseuille: second order, and TRT is far better ------------
  {
    const double e15b = reg_poiseuille(Op::BGK, 18, 0.5, 0.002);
    const double e31b = reg_poiseuille(Op::BGK, 34, 0.5, 0.002);
    const double e31t = reg_poiseuille(Op::TRT, 34, 0.5, 0.002);
    const double order = std::log(e15b / e31b) / std::log(31.0 / 15.0);
    check(order > 1.8 && order < 2.15,
          "forced Poiseuille converges at second order", order, 2.0);
    check(e31b / e31t > 10.0,
          "  ... and TRT beats BGK at the wall by more than 10x", e31b / e31t, 32.0);
    std::printf("        (l2/u_max: BGK %.3e -> %.3e, order %.2f; TRT %.3e at L=31)\n",
                e15b, e31b, order, e31t);
  }

  // ---- 3. corners: a closed box must stay closed ---------------------------
  {
    const int n = 34, nzz = 4, a = 1, b = n - 2;
    for (int route = 0; route < 2; ++route) {
      host::Fluid fl(n, n, nzz, Op::BGK, Real(0.02));
      fl.set_fd_corners(route == 0);
      std::vector<std::uint8_t> geo(std::size_t(n) * n * nzz, std::uint8_t(Fluid));
      std::vector<RegWallSpec> spec(std::size_t(n) * n * nzz);
      for (int z = 0; z < nzz; ++z)
        for (int y = 0; y < n; ++y)
          for (int x = 0; x < n; ++x) {
            const long id = node_id(x, y, z, n, n);
            if (x == 0 || x == n - 1 || y == 0 || y == n - 1) {
              geo[std::size_t(id)] = Excluded;  continue;
            }
            const bool xw = (x == a || x == b), yw = (y == a || y == b);
            if (xw && yw)  spec[std::size_t(id)] = RegWallSpec{NrmCorner, 0, 0, 0};
            else if (yw)   spec[std::size_t(id)] = RegWallSpec{std::uint8_t(y == a ? NrmYm : NrmYp), 0, 0, 0};
            else if (xw)   spec[std::size_t(id)] = RegWallSpec{std::uint8_t(x == a ? NrmXm : NrmXp), 0, 0, 0};
          }
      fl.set_geometry(geo);
      fl.set_regularized_walls(spec);
      fl.initialise_with([](int, int, int) { Macro m; m.rho = Real(1); return m; });
      const double m0 = fl.total_mass();
      for (int t = 0; t < 3000; ++t) fl.step();
      std::vector<Real> rho, ux, uy, uz;
      fl.macroscopic_to_host(rho, ux, uy, uz);
      double umax = 0;
      for (int y = a; y <= b; ++y)
        for (int x = a; x <= b; ++x) {
          const long id = node_id(x, y, nzz / 2, n, n);
          umax = std::fmax(umax, std::fabs(double(ux[std::size_t(id)]))
                               + std::fabs(double(uy[std::size_t(id)])));
        }
      char buf[128];
      std::snprintf(buf, sizeof buf, "closed box, %s corners: stays at rest",
                    route == 0 ? "FD   " : "local");
      check(umax < (fp64 ? 1e-12 : 1e-6), buf, umax, 0.0);
      std::snprintf(buf, sizeof buf, "  ... and holds its mass exactly");
      check(std::fabs((fl.total_mass() - m0) / m0) < (fp64 ? 1e-12 : 1e-5), buf,
            (fl.total_mass() - m0) / m0, 0.0);
    }
    note("a driven cavity does NOT hold its mass -- BC3 overwrites populations; "
         "see the banner");
  }
}

//==============================================================================
//  2. A closed box.
//
//  Six solid walls, a swirl inside, no force. Two things must hold, and they
//  test different halves of the geometry implementation:
//
//    * the sum over EVERY slot of the lattice is conserved to round-off. Under
//      Esoteric Pull each slot has one writer per step; if a solid cell were
//      visited when it should not be, or a fluid cell skipped, this is where it
//      shows;
//    * the flow decays to rest, because there is nothing driving it.
//
//  The case also measures something worth knowing rather than asserting: how
//  much of the total the FLUID cells see. A population in flight toward a wall
//  spends a step in a slot the wall owns, so a fluid-only sum always undercounts.
//  Nothing is lost; the plotted field is what is missing it.
//==============================================================================
static void closed_box() {
  const int n = 24;
  host::Fluid fl(n, n, n, Op::CentralMoments, Real(1.0 / 6.0));

  std::vector<std::uint8_t> flags(std::size_t(n) * n * n, std::uint8_t(Fluid));
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        if (x == 0 || y == 0 || z == 0 || x == n - 1 || y == n - 1 || z == n - 1)
          flags[std::size_t(node_id(x, y, z, n, n))] = Solid;
  fl.set_geometry(flags);

  const double k = 2.0 * M_PI / n;
  fl.initialise_with([k, n](int x, int y, int z) {
    Macro m;
    m.rho = Real(1);
    m.ux  = Real( 0.04 * std::sin(k * x) * std::cos(k * y));
    m.uy  = Real(-0.04 * std::cos(k * x) * std::sin(k * y));
    m.uz  = Real(0);
    (void)z; (void)n;
    return m;
  });

  auto energy = [&]() {
    std::vector<Real> rho, ux, uy, uz;
    fl.macroscopic_to_host(rho, ux, uy, uz);
    double e = 0;
    for (std::size_t i = 0; i < ux.size(); ++i)
      e += 0.5 * (double(ux[i]) * double(ux[i]) + double(uy[i]) * double(uy[i]) +
                  double(uz[i]) * double(uz[i]));
    return e;
  };

  const double m0 = fl.total_mass();
  const double e0 = energy();
  for (std::size_t t = 0; t < 200; ++t) fl.step();
  const double m1 = fl.total_mass();
  const double e1 = energy();

  check(std::fabs(m1 - m0) / m0 < (fp64 ? 1e-12 : 3e-6),
        "closed box: total mass over every slot conserved", m1, m0);
  check(e1 < 0.5 * e0 && e1 > 0.0,
        "closed box: the flow decays with nothing driving it", e1 / e0, 0.0);
  char buf[160];
  std::snprintf(buf, sizeof buf,
                "energy fell to %.2e of its initial value in 200 steps, while the mass "
                "held to %.1e -- decay is physics, the mass is bookkeeping", e1 / e0,
                std::fabs(m1 - m0) / m0);
  note(buf);
}

//==============================================================================
//  3. An insulating box conserves the scalar, exactly.
//
//  Every wall adiabatic, a blob in the middle, no flow. The total population --
//  summed over the WHOLE lattice, not over fluid cells -- must not move.
//
//  The fluid-cell sum is reported alongside, because the gap between the two is
//  the thing most likely to be mistaken for a leak. In the parent implementation
//  it reached 13% on an urban geometry and none of it was lost.
//==============================================================================
static void insulating_box() {
  const int n = 20;
  host::Scalar sc(n, n, n, Real(0.05));

  std::vector<std::uint8_t> flags(std::size_t(n) * n * n, std::uint8_t(ScalarBulk));
  std::vector<Real> wall(std::size_t(n) * n * n, Real(0));
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        if (x == 0 || y == 0 || z == 0 || x == n - 1 || y == n - 1 || z == n - 1)
          flags[std::size_t(node_id(x, y, z, n, n))] = ScalarAdiabatic;
  sc.set_geometry(flags, wall);

  const double c = 0.5 * (n - 1);
  sc.initialise_with([c](int x, int y, int z) {
    const double r2 = (x - c) * (x - c) + (y - c) * (y - c) + (z - c) * (z - c);
    return Real(std::exp(-r2 / 8.0));
  });

  const double m0 = sc.total_population();
  for (std::size_t t = 0; t < 3000; ++t) sc.step();
  const double m1 = sc.total_population();

  check(std::fabs(m1 - m0) / m0 < (fp64 ? 1e-12 : 3e-4),
        "insulating box: total population conserved", m1, m0);

  const std::vector<Real>& T = sc.field();
  double in_fluid = 0;
  for (int z = 1; z < n - 1; ++z)
    for (int y = 1; y < n - 1; ++y)
      for (int x = 1; x < n - 1; ++x) in_fluid += double(T[std::size_t(node_id(x, y, z, n, n))]);
  char buf[160];
  std::snprintf(buf, sizeof buf,
                "the field over bulk cells sees %.2f%% of it -- the rest is in wall slots, "
                "in flight, not lost", 100.0 * in_fluid / m1);
  note(buf);
}

//==============================================================================
//  4. Conduction between Dirichlet walls.
//
//  Anti-bounce-back puts T_wall half-way between the Dirichlet node and its
//  fluid neighbour, exactly as halfway bounce-back does for no-slip. So with
//  Dirichlet layers at y = 0 and y = H+1 the two PLANES are at y = 0.5 and
//  y = H + 0.5, the gap between them is H, and the steady profile is
//
//      T(y) = (y - 0.5) / H.
//
//  Measuring between the NODES instead gives a gradient wrong by (H+1)/H, an
//  O(1/H) error that shrinks under refinement and so is easily mistaken for
//  ordinary discretisation error.
//==============================================================================
//------------------------------------------------------------------------------
// 4b  DELLAR'S MOMENT CONDITION: the plane ON the node.
//
// The same conduction problem as case 4, with ScalarMoment plates instead of
// anti-bounce-back ones. The exact steady profile is then linear between the
// NODES -- T(y) = y/(ny-1) -- rather than between planes half a cell outside
// them. This is not a style preference: the EHD cases differentiate the field
// carrying the boundary value, and a halfway stencil built from interior nodes
// never sees the imposed value at all.
//------------------------------------------------------------------------------
static void on_node_dirichlet(Real D, const char* tag) {
  const int nx = 4, ny = 18, nz = 1;
  host::Scalar sc(nx, ny, nz, D, Real(0));
  sc.set_periodicity(true, false, true);

  std::vector<std::uint8_t> flags(std::size_t(nx) * ny * nz, std::uint8_t(ScalarBulk));
  std::vector<Real> wall(std::size_t(nx) * ny * nz, Real(0));
  for (int x = 0; x < nx; ++x) {
    flags[std::size_t(node_id(x, 0,      0, nx, ny))] = ScalarMoment;
    flags[std::size_t(node_id(x, ny - 1, 0, nx, ny))] = ScalarMoment;
    wall [std::size_t(node_id(x, 0,      0, nx, ny))] = Real(0);
    wall [std::size_t(node_id(x, ny - 1, 0, nx, ny))] = Real(1);
  }
  sc.set_geometry(flags, wall);
  sc.initialise_with([](int, int, int) { return Real(0.5); });
  for (std::size_t t = 0; t < 40000; ++t) sc.step();

  const std::vector<Real>& T = sc.field();
  double worst = 0;
  for (int y = 0; y < ny; ++y) {
    double t = 0;
    for (int x = 0; x < nx; ++x) t += double(T[std::size_t(node_id(x, y, 0, nx, ny))]);
    t /= double(nx);
    worst = std::fmax(worst, std::fabs(t - double(y) / double(ny - 1)));
  }
  char buf[120];
  std::snprintf(buf, sizeof buf,
                "on-node Dirichlet at omega = %.4f: deviation from T = y/(ny-1)",
                double(sc.omega()));
  (void)tag;
  check(worst < (fp64 ? 1e-9 : 2e-5), buf, worst, 0.0);
  note("anti-bounce-back would put this profile on (y-0.5)/H instead -- half a");
  note("cell out at each end, which is what makes E = -grad phi first order.");
}

//------------------------------------------------------------------------------
// 4c  THE ON-NODE ZERO-FLUX WALL, CHECKED AS AN IDENTITY.
//
// A box of width M closed by two mirrors is not merely similar to the periodic
// box of width 2M whose mirror it is; it is the SAME PROBLEM, so the two must
// agree to round-off rather than to a tolerance. A wall that is NEARLY a mirror
// also converges, so a rate would not test this.
//
// Everything is even about x = 0 and about x = M, both of which are NODES:
// the streamfunction psi = A sin(pi x/M) cos(2 pi y/ny) is odd about both, so
// u_x is odd (no flux through either plane), u_y even, and div u = 0. A
// velocity with a divergence would pump the scalar, and a reference that grows
// by two orders of magnitude is a fragile thing to regress against.
//------------------------------------------------------------------------------
static void specular_identity() {
  const int M = 12, ny = 16, nz = 1;
  const double U = 0.05;
  auto run = [&](bool half, std::vector<Real>& out) {
    const int nx = half ? M + 1 : 2 * M;
    host::Scalar sc(nx, ny, nz, Real(0.02), Real(0));
    sc.set_periodicity(!half, true, true);
    const std::size_t N = std::size_t(nx) * ny * nz;
    std::vector<std::uint8_t> flags(N, std::uint8_t(ScalarBulk)), nrm(N, std::uint8_t(NrmNone));
    std::vector<Real> wall(N, Real(0));
    if (half)
      for (int y = 0; y < ny; ++y) {
        flags[std::size_t(node_id(0,      y, 0, nx, ny))] = ScalarSpecular;
        flags[std::size_t(node_id(nx - 1, y, 0, nx, ny))] = ScalarSpecular;
        nrm  [std::size_t(node_id(0,      y, 0, nx, ny))] = NrmXm;
        nrm  [std::size_t(node_id(nx - 1, y, 0, nx, ny))] = NrmXp;
      }
    sc.set_geometry(flags, wall);
    if (half) sc.set_specular_walls(nrm);
    std::vector<Real> ux(N), uy(N), uz(N, Real(0));
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const std::size_t n = std::size_t(node_id(x, y, 0, nx, ny));
        ux[n] = Real(-U * std::sin(M_PI * x / double(M)) * std::sin(2.0 * M_PI * y / ny));
        uy[n] = Real(-U * (double(ny) / (2.0 * M)) * std::cos(M_PI * x / double(M))
                        * std::cos(2.0 * M_PI * y / ny));
      }
    sc.advect_with(ux.data(), uy.data(), uz.data());
    sc.initialise_with([&](int x, int y, int) {
      return Real(1.0 + 0.5 * std::cos(M_PI * x / double(M)) * std::cos(2.0 * M_PI * y / ny)
                      + 0.3 * std::cos(2.0 * M_PI * x / double(M)));
    });
    for (std::size_t t = 0; t < 300; ++t) sc.step();
    const std::vector<Real>& T = sc.field();
    out.assign(std::size_t(M + 1) * ny, Real(0));
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x <= M; ++x)
        out[std::size_t(y) * std::size_t(M + 1) + std::size_t(x)] =
            T[std::size_t(node_id(x, y, 0, nx, ny))];
  };
  std::vector<Real> full, half;
  run(false, full); run(true, half);
  double worst = 0, lo = 1e300, hi = -1e300;
  for (std::size_t i = 0; i < full.size(); ++i) {
    worst = std::fmax(worst, std::fabs(double(full[i]) - double(half[i])));
    lo = std::fmin(lo, double(full[i])); hi = std::fmax(hi, double(full[i]));
  }
  check(worst < (fp64 ? 1e-12 : 2e-5),
        "specular: half box vs the periodic box it mirrors", worst, 0.0);
  // Two flat fields agree to round-off whatever the wall does, so the reference
  // has to be shown still carrying structure for that number to mean anything.
  check(hi - lo > 0.05, "specular: the reference still carries structure",
        hi - lo, 0.05);
}

//------------------------------------------------------------------------------
// 3x  THE ON-NODE SPECULAR FLUID WALL (SpecNode), AS AN IDENTITY.
//
// Same claim as the scalar mirror above, applied to momentum: a box closed by
// two on-node mirrors is not merely similar to the periodic box of twice the
// width whose mirror it is, it is THE SAME PROBLEM node for node. A wall that
// is only nearly a mirror still converges, so a rate would not see it.
//
// Two geometries, and the second is the one this tree needs. A half box in x
// exercises single-face masks; a QUARTER box in x and z puts a TWO-FACE mask on
// four edge lines, which is the case the parent tree's ghost-cell mirror
// explicitly refuses and an on-node wall cannot -- in a closed box the edge node
// is a real fluid node and there is nowhere else to put it.
//
// The initial state is even about every mirror plane: u_n odd in its own
// direction, everything else even, all four planes on NODES.
//------------------------------------------------------------------------------
static void fluid_specular_node() {
  const int M = 12, ny = 16, P = 8;
  const double A = 0.03, aa = 0.02;

  auto run = [&](bool mirror_x, bool mirror_z, std::vector<double>& out) {
    const int nx = mirror_x ? M + 1 : 2 * M;
    const int nz = mirror_z ? P + 1 : 2 * P;
    host::Fluid fl(nx, ny, nz, Op::BGK, Real(0.0333));
    const std::size_t N = std::size_t(nx) * ny * nz;
    if (mirror_x || mirror_z) {
      std::vector<std::uint8_t> faces(N, std::uint8_t(SpecNone));
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            std::uint8_t m = SpecNone;
            if (mirror_x && x == 0)      m = std::uint8_t(m | SpecXm);
            if (mirror_x && x == nx - 1) m = std::uint8_t(m | SpecXp);
            if (mirror_z && z == 0)      m = std::uint8_t(m | SpecZm);
            if (mirror_z && z == nz - 1) m = std::uint8_t(m | SpecZp);
            if (m != SpecNone) faces[std::size_t(node_id(x, y, z, nx, ny))] = m;
          }
      fl.set_specular_nodes(faces);
    }
    // Solenoidal by the amplitude relation A/M + 2B/ny + C/P = 0.
    const double Ax = A, Cz = A * double(P) / double(M);
    const double By = -0.5 * double(ny) * (Ax / double(M) + Cz / double(P));
    fl.initialise_with([&](int x, int y, int z) {
      const double sx = std::sin(M_PI * x / double(M)), cx = std::cos(M_PI * x / double(M));
      const double sz = std::sin(M_PI * z / double(P)), cz = std::cos(M_PI * z / double(P));
      const double sy = std::sin(2.0 * M_PI * y / ny),  cy = std::cos(2.0 * M_PI * y / ny);
      Macro m;
      m.rho = Real(1.0 + aa * cx * cy * cz);
      m.ux  = Real(Ax * sx * cy * cz);
      m.uy  = Real(By * cx * sy * cz);
      m.uz  = Real(Cz * cx * cy * sz);
      return m;
    });
    for (std::size_t t = 0; t < 200; ++t) fl.step();
    std::vector<Real> rho, ux, uy, uz;
    fl.macroscopic_to_host(rho, ux, uy, uz);
    out.clear();
    for (int z = 0; z <= P; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x <= M; ++x) {
          const std::size_t n = std::size_t(node_id(x, y, z, nx, ny));
          out.push_back(double(ux[n])); out.push_back(double(uy[n]));
          out.push_back(double(uz[n])); out.push_back(double(rho[n]) - 1.0);
        }
  };

  std::vector<double> ref, half, quarter;
  run(false, false, ref);
  run(true,  false, half);
  run(true,  true,  quarter);

  double w_half = 0, w_quart = 0, scale = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    w_half  = std::fmax(w_half,  std::fabs(ref[i] - half[i]));
    w_quart = std::fmax(w_quart, std::fabs(ref[i] - quarter[i]));
    scale   = std::fmax(scale, std::fabs(ref[i]));
  }
  const double tol = fp64 ? 1e-13 : 3e-6;
  check(w_half  < tol, "SpecNode: half box in x vs the periodic box it mirrors",
        w_half, 0.0);
  check(w_quart < tol, "SpecNode: quarter box, four TWO-FACE edge lines",
        w_quart, 0.0);
  // Two decayed fields agree to round-off whatever the wall does.
  check(scale > 1e-3, "SpecNode: the reference still carries structure", scale, 1e-3);

  // ---- what the mirror imposes, on a random vector and with no solver -------
  // The identities above would also pass for a wall that happened to be
  // symmetric but was not zero-flux, because the SETUP is symmetric. So the
  // permutation is checked directly. `tang` guards against a mirror that
  // quietly became bounce-back: bounce-back kills EVERY momentum component and
  // would still pass both flux tests and both identities.
  auto moments = [&](std::uint8_t faces, const char* name, bool expect_tang) {
    Real f[27];
    unsigned st = 12345u;
    for (int i = 0; i < 27; ++i) {
      st = st * 1664525u + 1013904223u;
      f[i] = Real(0.5 + double(st >> 8) / double(1u << 24));
    }
    mirror_unknowns_faces<D3Q27>(f, faces);
    double mom[3] = {0, 0, 0}, str[3][3] = {};
    for (int i = 0; i < 27; ++i) {
      const int c[3] = {D3Q27::cx(i), D3Q27::cy(i), D3Q27::cz(i)};
      for (int a = 0; a < 3; ++a) {
        mom[a] += double(f[i]) * c[a];
        for (int b = 0; b < 3; ++b) str[a][b] += double(f[i]) * c[a] * c[b];
      }
    }
    double flux = 0, shear = 0, tang = 0;
    for (int a = 0; a < 3; ++a) {
      if (face_sign(faces, a) == 0) { tang = std::fmax(tang, std::fabs(mom[a])); continue; }
      flux = std::fmax(flux, std::fabs(mom[a]));
      for (int b = 0; b < 3; ++b)
        if (b != a) shear = std::fmax(shear, std::fabs(str[a][b]));
    }
    char lbl[96];
    std::snprintf(lbl, sizeof lbl, "SpecNode: %s, normal flux and shear", name);
    check(flux < 1e-6 && shear < 1e-6, lbl, std::fmax(flux, shear), 0.0);
    if (expect_tang) {
      std::snprintf(lbl, sizeof lbl, "SpecNode: %s, tangential momentum survives", name);
      check(tang > 1e-3, lbl, tang, 1e-3);
    }
  };
  moments(SpecXm, "-x face", true);
  moments(std::uint8_t(SpecXm | SpecZm), "-x-z edge", true);
  // A three-face corner leaves no free axis, so `tang` is vacuously zero there
  // and the guard does not apply.
  moments(std::uint8_t(SpecXm | SpecYp | SpecZm), "corner", false);
}

//------------------------------------------------------------------------------
// 4d  THE SOURCE REACHES A SPECULAR NODE.
//
// A specular cell is a BULK node carrying a mirror closure, not a node whose
// value is prescribed, so it must take the source like any other. Withholding
// it leaves the PDE unsolved in the wall column, and in the parent tree that
// read as a mediocre boundary condition rather than as a bug -- the Poisson
// reference came out +8.87 % against +0.51 %, second order fell to first, and
// nothing failed. With a uniform source and no diffusive gradient to drive,
// EVERY node must climb at exactly S per step, wall column included.
//------------------------------------------------------------------------------
static void scalar_source_reaches_wall() {
  const int nx = 10, ny = 6, nz = 1;
  const Real S = Real(0.01);
  host::Scalar sc(nx, ny, nz, Real(0.1), Real(0));
  sc.set_periodicity(false, true, true);
  const std::size_t N = std::size_t(nx) * ny * nz;
  std::vector<std::uint8_t> flags(N, std::uint8_t(ScalarBulk)), nrm(N, std::uint8_t(NrmNone));
  std::vector<Real> wall(N, Real(0)), src(N, S);
  for (int y = 0; y < ny; ++y) {
    flags[std::size_t(node_id(0,      y, 0, nx, ny))] = ScalarSpecular;
    flags[std::size_t(node_id(nx - 1, y, 0, nx, ny))] = ScalarSpecular;
    nrm  [std::size_t(node_id(0,      y, 0, nx, ny))] = NrmXm;
    nrm  [std::size_t(node_id(nx - 1, y, 0, nx, ny))] = NrmXp;
  }
  sc.set_geometry(flags, wall);
  sc.set_specular_walls(nrm);
  // (a) A NEGLIGIBLE SOURCE MUST BE A NO-OP, CHECKED ACROSS STEPS.
  //
  // add_source is a read-modify-write of each node's own slots. gather and
  // scatter are a STREAMING pair -- scatter puts in[i] where gather took
  // out[i+1] from -- so handing the values back unswapped advances the field an
  // extra step every time a source is added.
  //
  // TWO OBVIOUS TESTS OF THAT ARE BLIND, and both were written before this one.
  // A constant source on a flat field: a uniform field is invariant under
  // streaming. And a one-shot source compared before/after: the crossing merely
  // SWAPS each opposite pair, and the field is their SUM, so it is unchanged
  // until a step flips the parity. Only a differential test over several steps
  // sees it: the same run with and without an utterly negligible source.
  {
    auto run = [&](bool with_source, std::vector<Real>& out) {
      host::Scalar a(nx, ny, nz, Real(0.1), Real(0));
      a.set_periodicity(false, true, true);
      a.set_geometry(flags, wall);
      a.set_specular_walls(nrm);
      a.initialise_with([](int x, int y, int) {
        return Real(1.0 + 0.3 * std::sin(2.0 * M_PI * y / 6.0) + 0.2 * (x % 3));
      });
      std::vector<Real> tiny(N, Real(1e-30));
      for (int t = 0; t < 12; ++t) { if (with_source) a.add_source(tiny.data()); a.step(); }
      out = a.field();
    };
    std::vector<Real> plain, sourced;
    run(false, plain); run(true, sourced);
    double worst_one = 0;
    for (std::size_t i = 0; i < plain.size(); ++i)
      worst_one = std::fmax(worst_one, std::fabs(double(plain[i]) - double(sourced[i])));
    check(worst_one < (fp64 ? 1e-12 : 1e-6),
          "a 1e-30 source changes nothing (it is not a streaming step)",
          worst_one, 0.0);
  }

  // (b) and it must reach the specular column.
  sc.initialise_with([](int, int, int) { return Real(0); });
  const int steps = 50;
  for (int t = 0; t < steps; ++t) { sc.add_source(src.data()); sc.step(); }
  const std::vector<Real>& T = sc.field();
  const double want = double(S) * steps;
  double worst_bulk = 0, worst_wall = 0;
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      const double v = double(T[std::size_t(node_id(x, y, 0, nx, ny))]);
      if (x == 0 || x == nx - 1) worst_wall = std::fmax(worst_wall, std::fabs(v - want));
      else                       worst_bulk = std::fmax(worst_bulk, std::fabs(v - want));
    }
  check(worst_bulk < (fp64 ? 1e-12 : 1e-5),
        "uniform source: bulk climbs at exactly S per step", worst_bulk, 0.0);
  check(worst_wall < (fp64 ? 1e-12 : 1e-5),
        "uniform source: the SPECULAR column climbs at the same rate", worst_wall, 0.0);
}

//------------------------------------------------------------------------------
// 4e  EVERY OUTFLOW NODE MUST FIND A DONOR, INCLUDING THE CORNERS.
//
// The donor rule is "one step inward along every outward axis at once", so a
// face node takes its axis neighbour and a corner the diagonal. Outward was
// detected ONLY by a neighbour marked out of the transport, which is how a
// channel with bounce-back walls says it -- and a box with ON-NODE walls
// excludes nothing at all, so its corners found no outward axis, fell through
// to the axial fallback, and came out inert.
//
// This is the closed cavity's exact scalar geometry: injector on the bottom
// row, zero-gradient collector on the top, mirrors down the sides. The two top
// corners are the ones that used to fail; their four axial neighbours are two
// more outflow nodes, a specular column, and -- through the periodic wrap --
// the injector, so none is bulk and only the diagonal will do.
//------------------------------------------------------------------------------
static void outflow_corners() {
  const int nx = 9, ny = 9, nz = 1;
  const std::size_t N = std::size_t(nx) * ny * nz;
  std::vector<std::uint8_t> flags(N, std::uint8_t(ScalarBulk));
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      const std::size_t id = std::size_t(node_id(x, y, 0, nx, ny));
      if (y == 0)               flags[id] = ScalarMoment;
      else if (y == ny - 1)     flags[id] = ScalarOutflow;
      else if (x == 0 || x == nx - 1) flags[id] = ScalarSpecular;
    }
  const bool per[3] = {false, false, true};
  std::vector<long> donor;
  long degenerate = 0;
  const long nout = build_scalar_donors(flags, nx, ny, nz, per, donor, degenerate);
  check(nout == nx, "outflow: the whole collector row is found", double(nout), double(nx));
  check(degenerate == 0, "outflow: no node is left inert, corners included",
        double(degenerate), 0.0);
  // and the corner takes the DIAGONAL, which is the only bulk cell it touches
  const long c0 = donor[std::size_t(node_id(0, ny - 1, 0, nx, ny))];
  const long c1 = donor[std::size_t(node_id(nx - 1, ny - 1, 0, nx, ny))];
  check(c0 == node_id(1, ny - 2, 0, nx, ny) && c1 == node_id(nx - 2, ny - 2, 0, nx, ny),
        "outflow: each corner's donor is its inward diagonal", double(c0),
        double(node_id(1, ny - 2, 0, nx, ny)));
  // A FULLY PERIODIC BOX MUST BE UNCHANGED by all of that: with nothing
  // excluded and every axis wrapping, no direction is outward and the axial
  // fallback is what runs, exactly as before.
  const bool allper[3] = {true, true, true};
  std::vector<long> d2;
  long deg2 = 0;
  build_scalar_donors(flags, nx, ny, nz, allper, d2, deg2);
  check(deg2 == 2, "outflow: an all-periodic box still reports the two corners",
        double(deg2), 2.0);
}

static void conduction() {
  const int H = 16, nx = 4, nz = 4, ny = H + 2;
  host::Scalar sc(nx, ny, nz, Real(0.125), Real(0.5));

  std::vector<std::uint8_t> flags(std::size_t(nx) * ny * nz, std::uint8_t(ScalarBulk));
  std::vector<Real> wall(std::size_t(nx) * ny * nz, Real(0));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      flags[std::size_t(node_id(x, 0,      z, nx, ny))] = ScalarDirichlet;
      flags[std::size_t(node_id(x, ny - 1, z, nx, ny))] = ScalarDirichlet;
      wall [std::size_t(node_id(x, 0,      z, nx, ny))] = Real(0);
      wall [std::size_t(node_id(x, ny - 1, z, nx, ny))] = Real(1);
    }
  sc.set_geometry(flags, wall);
  sc.initialise_with([](int, int, int) { return Real(0.5); });

  for (std::size_t t = 0; t < 20000; ++t) sc.step();

  const std::vector<Real>& T = sc.field();
  double worst_err = 0;
  for (int y = 1; y <= H; ++y) {
    double t = 0;
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x) t += double(T[std::size_t(node_id(x, y, z, nx, ny))]);
    t /= double(nx) * nz;
    worst_err = std::fmax(worst_err, std::fabs(t - (y - 0.5) / double(H)));
  }
  check(worst_err < (fp64 ? 1e-6 : 2e-4),
        "conduction: worst deviation from the exact linear profile", worst_err, 0.0);

  // The same measurement made between the NODES, to show what the wrong
  // convention costs on this grid.
  double t1 = 0, tH = 0;
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      t1 += double(T[std::size_t(node_id(x, 1, z, nx, ny))]);
      tH += double(T[std::size_t(node_id(x, H, z, nx, ny))]);
    }
  t1 /= double(nx) * nz; tH /= double(nx) * nz;
  char buf[160];
  std::snprintf(buf, sizeof buf,
                "measured gradient %.6f per node; planes H = %d apart give %.6f, "
                "nodes H-1 apart would give %.6f",
                (tH - t1) / double(H - 1), H, 1.0 / H, 1.0 / (H - 1));
  note(buf);
}

//==============================================================================
//  5. A decaying sinusoid: the D3Q7 diffusivity.
//
//  T = T0 + A sin(kx) decays as exp(-D k^2 t) with D = cs^2 (1/omega - 1/2) and
//  cs^2 = 1/4. Run at two resolutions: the measured rate must approach the
//  analytic one, and the error must fall like the square of the wavenumber. A
//  single resolution cannot tell a wrong cs^2 from ordinary discretisation error;
//  two can, because a wrong cs^2 does not converge away.
//==============================================================================
static double diffusion_error(int L) {
  const int ny = 4, nz = 4;
  const double D = 0.125;                      // omega = 1
  host::Scalar sc(L, ny, nz, Real(D), Real(1.0));

  const double k = 2.0 * M_PI / L;
  sc.initialise_with([k](int x, int, int) { return Real(1.0 + 0.1 * std::sin(k * x)); });

  // Run for a FIXED NUMBER OF E-FOLDS, not a fixed number of steps. The decay
  // time is 1/(D k^2), which is four times shorter at L = 16 than at L = 32; a
  // step count that suits one resolution lets the other decay into round-off,
  // and the fit then measures noise. (It did, the first time: -66% and +73%.)
  const std::size_t T = std::size_t(2.0 / (D * k * k));
  const std::size_t probe = T / 20 ? T / 20 : 1;
  std::vector<double> ts, la;
  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      const Mode m = fit_x(sc.field().data(), L, ny, nz);
      ts.push_back(double(t));
      la.push_back(std::log(m.amp));
    }
    if (t < T) sc.step();
  }
  const double rate = -slope(ts, la);
  const double exact = D * k * k;
  return (rate - exact) / exact;
}

static void diffusion() {
  const double e16 = diffusion_error(16);
  const double e32 = diffusion_error(32);
  check(std::fabs(e32) < 0.02, "diffusion: rate error at L = 32", e32, 0.0);
  const double ratio = std::fabs(e16 / e32);
  check(ratio > 2.5 && ratio < 5.5,
        "diffusion: error falls as k^2 when L doubles (ratio)", ratio, 4.0);
  char buf[160];
  std::snprintf(buf, sizeof buf,
                "rate error %.3f%% at L=16, %.3f%% at L=32 -- a wrong cs^2 would not converge",
                100.0 * e16, 100.0 * e32);
  note(buf);
}

//==============================================================================
//  6. An advected sinusoid: the advective flux.
//
//  A uniform velocity is imposed directly, so this isolates the scalar from the
//  fluid entirely. The mode's phase must move at exactly u:  dphi/dt = -k u.
//
//  The first-order D3Q7 equilibrium carries an O(u^2) defect by construction --
//  the lattice cannot represent the cross terms of the uu tensor -- so the
//  tolerance here is a statement about that, not about round-off.
//==============================================================================
static void advection() {
  const int L = 32, ny = 4, nz = 4;
  const double u = 0.02, D = 0.05;
  host::Scalar sc(L, ny, nz, Real(D), Real(1.0));

  std::vector<Real> vx(std::size_t(L) * ny * nz, Real(u));
  std::vector<Real> vy(std::size_t(L) * ny * nz, Real(0));
  std::vector<Real> vz(std::size_t(L) * ny * nz, Real(0));
  sc.advect_with(vx.data(), vy.data(), vz.data());

  const double k = 2.0 * M_PI / L;
  sc.initialise_with([k](int x, int, int) { return Real(1.0 + 0.1 * std::sin(k * x)); });

  const std::size_t T = 3000, probe = 100;
  std::vector<double> ts, ph;
  double prev = 0;
  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      const Mode m = fit_x(sc.field().data(), L, ny, nz);
      const double p = ts.empty() ? m.phase : unwrap(prev, m.phase);
      prev = p;
      ts.push_back(double(t));
      ph.push_back(p);
    }
    if (t < T) sc.step();
  }
  const double measured = -slope(ts, ph) / k;
  check(std::fabs(measured - u) / u < 0.02, "advection: phase speed", measured, u);
}

//==============================================================================
//  6b. AN OPEN EXIT: a blob must leave, not pile up.
//
//  ScalarOutflow is zero-gradient by equilibrium extrapolation from an interior
//  donor, run as a SECOND pass after the main one -- see the enum banner in
//  streaming.cuh for why reading the donor's populations inside the main kernel
//  would be a race rather than merely untidy.
//
//  A boundary condition is only worth having if the alternative is visibly
//  wrong, so this runs the SAME case twice and changes one flag. A Gaussian
//  blob is advected at u = 0.05 down a closed tube whose far end is either
//  ScalarOutflow or ScalarAdiabatic. Measured, nx = 96, D = 0.005, FP32:
//
//      t       OPEN  sum  peak       CLOSED  sum  peak
//      0            1.000  1.000            1.000  1.000
//      600          1.000  0.850            1.000  0.850
//      1200         0.996  0.754            0.998  0.754
//      1500         0.389  0.661            0.666  5.246
//      1800         0.003  0.011            0.446  7.420
//      2100         0.000  0.000            0.444  7.427
//
//  Identical to four digits until the blob reaches the exit, which is the check
//  that the flag changes nothing in the interior. Then they separate completely:
//  the open exit empties the tube, and the closed one reflects the blob into a
//  pile SEVEN AND A HALF TIMES the initial peak and holds 44% of the scalar for
//  ever. That factor of 7.4 is the thing an open boundary is for, and it is why
//  a plume study run against a bounce-back exit is not merely slightly wrong.
//
//  The peak also tracks x0 + u t exactly while it is in the interior -- 20, 35,
//  50, 65, 80 at t = 0, 300, 600, 900, 1200 -- which is the advection check
//  case 6 makes on a periodic domain, repeated here against a wall.
//
//  WHAT THIS DOES NOT CHECK is the accuracy of the exit, only that it opens.
//  The condition discards the non-equilibrium part, which slightly damps the
//  diffusive flux there; that is negligible when the exit is advection
//  dominated, which is the only place an open boundary belongs, and it is a
//  known approximation rather than an accident.
//==============================================================================
static double open_exit(bool open, double& peak_out, int& peak_x_at_1200) {
  const int nx = 96, ny = 4, nz = 4;
  const double U = 0.05, D = 0.005, x0 = 20.0, sig = 4.0;

  host::Scalar sc(nx, ny, nz, Real(D));
  std::vector<std::uint8_t> fl(std::size_t(nx) * ny * nz, std::uint8_t(ScalarBulk));
  std::vector<Real> wall(std::size_t(nx) * ny * nz, Real(0));
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y) {
      // The last plane is outside the field; the one before it is the boundary.
      // Marking the outside is what tells build_donors which way is "inward".
      fl[std::size_t(node_id(nx - 1, y, z, nx, ny))] = ScalarExcluded;
      fl[std::size_t(node_id(nx - 2, y, z, nx, ny))] = open ? ScalarOutflow : ScalarAdiabatic;
      // Close the inlet too, so nothing wraps round and re-enters.
      fl[std::size_t(node_id(0, y, z, nx, ny))] = ScalarExcluded;
      fl[std::size_t(node_id(1, y, z, nx, ny))] = ScalarAdiabatic;
    }
  sc.set_geometry(fl, wall);

  std::vector<Real> ux(std::size_t(nx) * ny * nz, Real(U));
  std::vector<Real> uy(ux.size(), Real(0)), uz(ux.size(), Real(0));
  sc.advect_with(ux.data(), uy.data(), uz.data());
  sc.initialise_with([&](int x, int, int) {
    return Real(std::exp(-0.5 * (x - x0) * (x - x0) / (sig * sig)));
  });

  auto scan = [&](double& sum, double& peak, int& px) {
    std::vector<Real> Tv;
    sc.field_to_host(Tv);
    sum = 0;  peak = -1e30;  px = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 2; x < nx - 2; ++x)
          sum += double(Tv[std::size_t(node_id(x, y, z, nx, ny))]);
    sum /= double(ny) * nz;
    for (int x = 2; x < nx - 2; ++x) {
      const double v = double(Tv[std::size_t(node_id(x, ny / 2, nz / 2, nx, ny))]);
      if (v > peak) { peak = v; px = x; }
    }
  };

  double s0 = 0, p0 = 0;  int q = 0;
  scan(s0, p0, q);
  for (int t = 0; t < 1200; ++t) sc.step();
  double s = 0, p = 0;
  scan(s, p, peak_x_at_1200);
  for (int t = 0; t < 900; ++t) sc.step();          // t = 2100
  scan(s, p, q);
  peak_out = p;
  return s / s0;
}

static void open_boundary() {
  double peak_open = 0, peak_closed = 0;
  int px_open = 0, px_closed = 0;
  const double left_open   = open_exit(true,  peak_open,   px_open);
  const double left_closed = open_exit(false, peak_closed, px_closed);

  // The interior must not know which flag was set, until the blob arrives.
  check(px_open == 80 && px_closed == 80,
        "the blob advects at u = 0.05 either way (x = 80 at t = 1200)",
        double(px_open), 80.0);
  check(left_open < 1e-3, "open exit: the tube empties", left_open, 0.0);
  check(peak_open < 1e-2, "open exit: nothing is left behind", peak_open, 0.0);
  check(left_closed > 0.4, "closed exit: 44% of the scalar is trapped for ever",
        left_closed, 0.444);
  check(peak_closed > 5.0, "closed exit: and it piles up past 5x the initial peak",
        peak_closed, 7.43);
  note("that factor of 7.4 is what an open boundary is for; measured, not asserted");
}

//==============================================================================
//  7. Uniform buoyancy: the whole Boussinesq path in one number.
//
//  A periodic box at rest, held at a uniform T above the reference. The force is
//  then F = rho0 g beta (T - T0) everywhere and there is no pressure gradient to
//  balance it, so the fluid accelerates uniformly:  du/dt = F / rho, exactly.
//
//  Measuring the INCREMENT between two times rather than the value avoids the
//  half-step offset Guo's velocity carries, and makes the test independent of
//  when it is sampled.
//
//  This exercises the whole chain -- scalar populations, compute_field, the
//  BodyForce read of that field, the Guo source -- against an exact answer, in a
//  case with no closed-form convection solution to hide behind.
//==============================================================================
static void buoyancy() {
  const int n = 8;
  const double T0 = 1.0, dT = 0.05, beta = 1.0;

  host::Fluid fl(n, n, n, Op::BGK, Real(1.0 / 6.0));
  host::Scalar sc(n, n, n, Real(0.05), Real(T0));

  fl.enable_velocity_output();
  sc.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  BodyForce b;
  b.T = sc.field_device();
  b.gx = Real(0); b.gy = Real(-1); b.gz = Real(0);
  b.rho0 = Real(1); b.beta = Real(beta); b.T0 = Real(T0);
  fl.set_force(b, ForceBoussinesq);

  fl.initialise_with([](int, int, int) { return Macro{Real(1), Real(0), Real(0), Real(0)}; });
  sc.initialise_with([T0, dT](int, int, int) { return Real(T0 + dT); });

  auto mean_uy = [&]() {
    std::vector<Real> rho, ux, uy, uz;
    fl.macroscopic_to_host(rho, ux, uy, uz);
    double s = 0;
    for (Real v : uy) s += double(v);
    return s / double(uy.size());
  };

  for (int t = 0; t < 100; ++t) host::coupled_step(fl, &sc, nullptr);
  const double u1 = mean_uy();
  for (int t = 0; t < 100; ++t) host::coupled_step(fl, &sc, nullptr);
  const double u2 = mean_uy();

  const double measured = (u2 - u1) / 100.0;
  const double exact = -beta * dT;             // rho0 g beta dT / rho, with rho = 1
  check(std::fabs(measured - exact) / std::fabs(exact) < (fp64 ? 1e-6 : 2e-3),
        "buoyancy: acceleration per step", measured, exact);
}

//==============================================================================
//  8. Resistive decay: the induction equation on its own.
//
//  B = (0, B0 sin kx, 0) with u = 0 everywhere. The flow is not solved at all,
//  so this isolates the magnetic lattice and its resistivity, eta = cs^2
//  (1/omega - 1/2) with cs^2 = 1/4 again. Exact: B decays as exp(-eta k^2 t).
//
//  Note this case says NOTHING about div B: B_y depends only on x, so div B is
//  structurally zero and would report round-off whatever the scheme did.
//==============================================================================
static void resistive_decay() {
  const int L = 32, ny = 4, nz = 4;
  const double eta = 0.02;
  host::Magnetic mag(L, ny, nz, Real(eta));

  const double k = 2.0 * M_PI / L;
  const double B0 = 0.02;
  mag.initialise_with(
      [k, B0](int x, int, int, Real B[3]) {
        B[0] = Real(0); B[1] = Real(B0 * std::sin(k * x)); B[2] = Real(0);
      },
      [](int, int, int, Real u[3]) { u[0] = u[1] = u[2] = Real(0); });

  const std::size_t T = 4000, probe = 200;
  std::vector<double> ts, la;
  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      const Mode m = fit_x(mag.by().data(), L, ny, nz);
      ts.push_back(double(t));
      la.push_back(std::log(m.amp));
    }
    if (t < T) mag.step();
  }
  const double rate = -slope(ts, la);
  const double exact = eta * k * k;
  check(std::fabs(rate - exact) / exact < 0.02, "resistive decay: rate", rate, exact);
}

//==============================================================================
//  8b. A MAGNETIC WALL, AND WHERE IT IS.
//
//  Dellar's moment condition (core.cuh): the field these lattices carry is a
//  ZEROTH moment, so imposing its boundary value is one linear equation in the
//  unknown populations, and on a straight wall a cross lattice leaves exactly
//  one unknown -- the direction pointing into the domain. That equation is then
//  exact and unique, with no closure assumption and no free parameter.
//
//  WHAT THAT BUYS, AND WHY IT IS THE POINT OF THE CONDITION. The boundary value
//  is attained AT the node, not half a cell away. This case is the magnetic
//  twin of case 4 (conduction between Dirichlet walls), and it makes the same
//  measurement: a conducting slab is held at B_y = 0 on one face and 0.02 on
//  the other, with no flow at all, so the steady state is a straight line and
//  its GRADIENT says where the planes are.
//
//  Measured, 32 fluid nodes with the held nodes at x = 1 and x = 32:
//
//      worst deviation from linear   2.7e-15 (FP64) / 1.4e-06 (FP32)
//      measured gradient             0.000645161
//      planes 31 apart give          0.000645161      <- on the nodes
//      planes 32 apart would give    0.000625000      <- half a cell out
//
//  So the planes are on the nodes, to round-off, and a half-way condition would
//  be 3.2% out on the gradient of this case. A Hartmann layer is a handful of
//  cells thick, so half of one is not a rounding error there -- that is the
//  whole reason to carry a second kind of boundary condition for B rather than
//  bouncing it back with the fluid.
//
//  WHAT IS STILL MISSING, said plainly rather than left to be discovered:
//
//   * A full Hartmann flow needs an ON-NODE VELOCITY wall as well, so that the
//     fluid and the field agree about where the wall is. This code has halfway
//     bounce-back only, which puts the no-slip plane half a cell outside the
//     last fluid node while the condition tested here puts B exactly on it.
//     Mixing them is a half-cell disagreement about the channel width, which is
//     precisely the error the moment condition exists to avoid. So: the
//     magnetic wall works, and a wall-bounded MHD BENCHMARK still wants the
//     parent for now.
//   * MagOutflow is exposed and NOT validated. The parent measures it driving B
//     about 6% high over the last ten nodes of an inlet-driven channel. Keep an
//     outlet far from anything being measured.
//   * A conducting or insulating wall -- one that couples to a wall current or
//     matches onto an exterior vacuum field -- is a different piece of work and
//     is not faked by either of these.
//==============================================================================
static void magnetic_wall() {
  const int nx = 34, ny = 4, nz = 4;
  const double eta = 0.1, BL = 0.0, BR = 0.02;

  host::Magnetic mg(nx, ny, nz, Real(eta));

  std::vector<std::uint8_t> geo(std::size_t(nx) * ny * nz, std::uint8_t(Fluid));
  std::vector<std::uint8_t> kind(geo.size(), std::uint8_t(MagBulk));
  std::vector<Real> wx(geo.size(), Real(0)), wy(geo.size(), Real(0)), wz(geo.size(), Real(0));
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y) {
      // The OUTERMOST plane is solid: that is what tells the mask which
      // direction streamed in from outside. The held node is the fluid face
      // next to it -- marking the solid cell itself would leave the wall with
      // no unknown direction and the condition would silently do nothing,
      // which set_walls counts and reports.
      geo[std::size_t(node_id(0, y, z, nx, ny))] = Solid;
      geo[std::size_t(node_id(nx - 1, y, z, nx, ny))] = Solid;
      const long a = node_id(1, y, z, nx, ny);
      const long b = node_id(nx - 2, y, z, nx, ny);
      kind[std::size_t(a)] = MagDirichlet;  wy[std::size_t(a)] = Real(BL);
      kind[std::size_t(b)] = MagDirichlet;  wy[std::size_t(b)] = Real(BR);
    }
  mg.set_geometry(geo);
  mg.set_walls(kind, wx, wy, wz);
  mg.initialise_with(
      [](int, int, int, Real B[3]) { B[0] = B[1] = B[2] = Real(0); },
      [](int, int, int, Real u[3]) { u[0] = u[1] = u[2] = Real(0); });

  // Diffusive time across the slab is L^2/eta = 31^2/0.1 ~ 9600 steps.
  for (int t = 0; t < 40000; ++t) mg.step();

  std::vector<Real> bx, by, bz;
  mg.field_to_host(bx, by, bz);

  double worst = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  int n = 0;
  for (int x = 1; x <= nx - 2; ++x) {
    const double v = double(by[std::size_t(node_id(x, ny / 2, nz / 2, nx, ny))]);
    const double e = BL + (BR - BL) * double(x - 1) / double(nx - 3);
    worst = std::fmax(worst, std::fabs(v - e));
    sx += x;  sy += v;  sxx += double(x) * x;  sxy += double(x) * v;  ++n;
  }
  const double grad = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const double on_node = (BR - BL) / double(nx - 3);
  const double half_out = (BR - BL) / double(nx - 2);

  const double wl = double(by[std::size_t(node_id(1, ny / 2, nz / 2, nx, ny))]);
  const double wr = double(by[std::size_t(node_id(nx - 2, ny / 2, nz / 2, nx, ny))]);

  check(worst < (fp64 ? 1e-12 : 1e-5),
        "magnetic wall: the steady profile is linear to round-off", worst, 0.0);
  check(std::fabs(grad - on_node) / on_node < 1e-4,
        "magnetic wall: gradient says the planes are ON the nodes", grad, on_node);
  check(std::fabs(grad - half_out) / half_out > 0.02,
        "  ... and NOT half a cell outside them", grad, half_out);
  check(std::fabs(wl - BL) < (fp64 ? 1e-14 : 1e-8) &&
        std::fabs(wr - BR) < (fp64 ? 1e-14 : 1e-8),
        "magnetic wall: the imposed value is attained exactly at the node", wr, BR);
  note("a half-way condition would be 3.2% out on this gradient; a Hartmann "
       "layer is a few cells thick");
}

//==============================================================================
//  9. The shear Alfven wave.
//
//  An EXACT solution of the full nonlinear incompressible MHD equations, not
//  merely the linearised ones: u is perpendicular to B0 and everything depends
//  only on x, so (u.grad)u vanishes identically while (B.grad)B does not. Both
//  the Lorentz coupling and the induction equation are therefore driven, and both
//  must be right.
//
//      B = (B0, b sin k(x - v_A t), 0),   u = (0, -b sin k(x - v_A t) / sqrt(rho), 0)
//
//  propagating at v_A = B0 / sqrt(rho) and damping at (nu + eta) k^2 / 2.
//
//  PHASE IS MEASURED AS WELL AS AMPLITUDE, and that is the point. An error in the
//  Lorentz coupling shows up as the wrong wave SPEED. An error in the coupling
//  ORDER shows up only in the damping -- and it does not refine away, which is
//  how the parent implementation eventually found it.
//==============================================================================
//------------------------------------------------------------------------------
// MHD + A PER-NODE FORCE. The two force kinds must agree.
//
// ForceUniform and ForceField are the same force delivered two ways: one a
// constant in BodyForce, the other an array read per cell. On a uniform field
// they must put identical momentum on the lattice, with or without a magnetic
// coupling. Comparing them rather than checking either alone is what makes this
// a test and not a restatement of the implementation.
//
// IT EXISTS BECAUSE IT FAILED. launch_force() and dispatch_force() enumerate the
// template combinations by hand, and under mhd_ only ForceUniform was listed --
// ForceField fell through to ForceNone and the collision applied NOTHING. That
// alone would have been loud, but macro_force honours ForceField unconditionally,
// so the REPORTED velocity still carried Guo's half shift F/(2 rho) from a force
// that was never applied. A penalised MHD run would have shown the wall acting in
// every diagnostic while contributing nothing to the dynamics -- the same silent
// coupling failure CLAUDE.md records for ehd_cavity, where deleting the term
// changed the answer by nothing.
//------------------------------------------------------------------------------
static void mhd_field_force() {
  const int L = 16, ny = 4, nz = 4;
  const int N = L * ny * nz;
  const double nu = 0.02, eta = 0.02;
  const Real Fx = Real(1e-5), B0 = Real(0.01);

  auto momentum = [&](bool mhd, int kind) {
    host::Magnetic mag(L, ny, nz, Real(eta));
    host::Fluid    fl (L, ny, nz, Op::BGK, Real(nu));
    std::vector<Real> field(std::size_t(N), Fx);
    BodyForce b;
    if (kind == ForceUniform) b.fx = Fx;
    else { b.Fx = field.data(); b.Fy = nullptr; b.Fz = nullptr; }
    // ForceField reads all three arrays, so give the null ones somewhere to point
    std::vector<Real> zero(std::size_t(N), Real(0));
    if (kind == ForceField) { b.Fy = zero.data(); b.Fz = zero.data(); }
    fl.set_force(b, kind);
    if (mhd) {
      fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
      mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());
      mag.initialise_with(
          [B0](int, int, int, Real B[3]) { B[0] = Real(0); B[1] = B0; B[2] = Real(0); },
          [](int, int, int, Real u[3]) { u[0] = u[1] = u[2] = Real(0); });
    }
    fl.initialise_with([](int, int, int) {
      Macro m; m.rho = Real(1); m.ux = m.uy = m.uz = Real(0); return m;
    });
    const std::size_t T = 200;
    for (std::size_t t = 0; t < T; ++t) { if (mhd) mag.compute_field(); fl.step(); if (mhd) mag.step(); }
    const std::size_t NN = static_cast<std::size_t>(N);
    std::vector<Real> rho(NN), ux(NN), uy(NN), uz(NN);
    fl.macroscopic_to_host(rho, ux, uy, uz);
    double p = 0;
    for (std::size_t n = 0; n < NN; ++n) p += double(rho[n]) * double(ux[n]);
    return p / double(N);                       // momentum per cell
  };

  const double want = double(Fx) * (200.0 + 0.5);   // F*(T + 1/2): Guo's half shift
  for (int mhd = 0; mhd < 2; ++mhd) {
    const double pu = momentum(mhd != 0, ForceUniform);
    const double pf = momentum(mhd != 0, ForceField);
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s: ForceField == ForceUniform",
                  mhd ? "MHD" : "hydro");
    check(std::fabs(pf - pu) / want < 1e-9, buf, pf, pu);
    std::snprintf(buf, sizeof buf, "%s: ForceField momentum == F*(T+1/2)",
                  mhd ? "MHD" : "hydro");
    // the differential check above is exact -- same arithmetic both ways -- but
    // the absolute one accumulates 200 steps of rounding, so it takes the
    // precision-dependent tolerance the rest of this file uses.
    check(std::fabs(pf - want) / want < (fp64 ? 1e-9 : 1e-4), buf, pf, want);
  }
}

//------------------------------------------------------------------------------
// THE PER-NODE MAGNETIC SOURCE, against the identity it cannot get wrong.
//
// The source is added post-collision on the weights, so sum_i w_i = 1 puts S_a
// straight into B and sum_i w_i c_i = 0 leaves the induction flux alone. In a
// periodic box with no velocity and no resistive decay to fight -- B uniform, so
// laplacian B = 0 -- one step must therefore add EXACTLY S to B, and T steps T*S.
//
// Two things this checks that a single run cannot. A zero source must leave the
// field bit-identical, which catches a source applied where it should not be.
// And the growth must be LINEAR in T: a source added on the wrong side of the
// scatter would still move B, but it would advance the field by a step as well,
// which the parent tree records as the gather/scatter trap. Only a differential
// run over several steps separates the two.
//------------------------------------------------------------------------------
static void magnetic_source() {
  const int L = 8, ny = 4, nz = 4;
  const std::size_t N = std::size_t(L) * ny * nz;
  const Real B0 = Real(0.01), S = Real(1e-6);

  auto run = [&](Real s_amp, std::size_t T) {
    host::Magnetic mag(L, ny, nz, Real(0.02));
    std::vector<Real> sx(N, s_amp), sy(N, Real(0)), sz(N, Real(0));
    mag.set_source(sx.data(), sy.data(), sz.data());
    mag.initialise_with(
        [B0](int, int, int, Real B[3]) { B[0] = B0; B[1] = Real(0); B[2] = Real(0); },
        [](int, int, int, Real u[3]) { u[0] = u[1] = u[2] = Real(0); });
    for (std::size_t t = 0; t < T; ++t) mag.step();
    double bx = 0;
    const std::vector<Real>& b = mag.bx();
    for (std::size_t n = 0; n < N; ++n) bx += double(b[n]);
    return bx / double(N);
  };

  // a zero source must change nothing
  check(std::fabs(run(Real(0), 50) - double(B0)) < (fp64 ? 1e-12 : 1e-6),
        "magnetic source: S = 0 leaves B untouched", run(Real(0), 50), double(B0));

  // and a uniform one must add exactly T*S, linearly
  for (std::size_t T : {10u, 40u}) {
    const double got = run(S, T);
    const double want = double(B0) + double(T) * double(S);
    char buf[128];
    std::snprintf(buf, sizeof buf, "magnetic source: B grows by T*S after %zu steps", T);
    check(std::fabs(got - want) / want < (fp64 ? 1e-9 : 1e-5), buf, got, want);
  }
}

//------------------------------------------------------------------------------
// nz = 1 ON THE MAGNETIC LATTICE. Untested until now, and relied upon.
//
// In a periodic direction one cell deep, wrap(z +/- 1, 1) = z, so a +/-z pair's
// neighbour IS the node. Esoteric Pull still writes slots i and i+1 -- two
// distinct slots at one node -- so nothing races and the population stays put,
// which is what streaming into yourself means. rb_high_ra.cu:56 argues that for
// the FLUID and measures it; every magnetic driver and every magnetic test in
// this tree uses nz = 4, so the same claim on D3Q7 was assumption rather than
// measurement. It matters because a 2-D case ported here runs at nz = 1.
//
// The test is an identity, not a tolerance: a field varying only in x cannot
// know how deep the box is, so nz = 1 and nz = 4 must agree to round-off at
// every node -- through resistive decay AND through advection by a velocity.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// MagNeumann: dB/dn = 0 at a chosen face, to SECOND order.
//
// Two checks, and the second is the one that matters.
//
// EXACTNESS. B_w = (4 B_1 - B_2)/3 is the value that makes the second-order
// one-sided derivative (-3 B_w + 4 B_1 - B_2)/2 vanish. So on a field that is
// QUADRATIC in the wall-normal coordinate with zero slope at the wall,
// B(s) = a + c s^2, the reconstruction must return `a` EXACTLY -- and the
// first-order alternative, copying the nearest neighbour, returns a + c. The
// test drives c hard enough that the two cannot be confused: at c = 1 the
// first-order answer is off by 100 % of the offset.
//
// IT IS APPLIED AT ALL. A condition that is coded and never reached is this
// tree's recurring failure -- the driver next door read four uninitialised
// pointers and penalised nothing while looking healthy. So the same field is
// run with the node marked MagBulk instead, and the wall value must then come
// out DIFFERENT. A test that passes either way measures nothing.
//------------------------------------------------------------------------------
static void magnetic_neumann() {
  const int L = 12, ny = 4, nz = 4;
  const std::size_t N = std::size_t(L) * ny * nz;
  const double a0 = 0.02, c2 = 1.0e-3, m1 = 1.5e-3;

  // `By` is the profile along x; the wall sits at x = 0 with outward normal -x.
  auto wall_value = [&](std::uint8_t kind_at_wall, bool give_face,
                        const std::function<double(int)>& By) {
    host::Magnetic mg(L, ny, nz, Real(0.02));
    std::vector<std::uint8_t> kind(N, std::uint8_t(MagBulk)), face(N, std::uint8_t(MFaceNone));
    std::vector<Real> wx(N, Real(0)), wy(N, Real(0)), wz(N, Real(0));
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y) {
        const std::size_t n0 = std::size_t((z * ny + y) * L + 0);
        kind[n0] = kind_at_wall;
        if (give_face) face[n0] = std::uint8_t(MFaceXm);   // outward normal is -x
        const std::size_t nL = std::size_t((z * ny + y) * L + (L - 1));
        kind[nL] = std::uint8_t(MagDirichlet);             // the level Neumann cannot fix
        wy[nL] = Real(By(L - 1));
      }
    mg.set_walls(kind, wx, wy, wz, face);
    mg.initialise_with(
        [&By](int x, int, int, Real B[3]) {
          B[0] = Real(0); B[1] = Real(By(x)); B[2] = Real(0);
        },
        [](int, int, int, Real u[3]) { u[0] = u[1] = u[2] = Real(0); });
    mg.compute_field();
    return double(mg.by()[std::size_t((1 * ny + 1) * L + 0)]);
  };

  // (a) EXACTNESS, on a quadratic with zero slope at the wall: B = a0 + c2 x^2.
  //     (4 B_1 - B_2)/3 = a0 exactly; copying the nearest neighbour gives a0+c2.
  const std::function<double(int)> quad =
      [a0, c2](int x) { return a0 + c2 * double(x) * double(x); };
  const double got_q = wall_value(std::uint8_t(MagNeumann), true, quad);
  check(std::fabs(got_q - a0) < (fp64 ? 1e-12 : 2e-6),
        "MagNeumann: (4B1-B2)/3 is EXACT on a zero-slope quadratic", got_q, a0);
  check(std::fabs(got_q - (a0 + c2)) > 0.5 * c2,
        "MagNeumann: and it is NOT the first-order neighbour copy", got_q, a0 + c2);

  // (b) IS IT APPLIED AT ALL? The quadratic above cannot answer that: its slope
  //     at the wall is zero, so the node's OWN value already equals what Neumann
  //     reconstructs and MagBulk agrees with it by construction. That is exactly
  //     the blind control this tree keeps rediscovering, and it failed here
  //     before being noticed. A LINEAR profile separates all three: the node's
  //     own value is a0, the second-order reconstruction is a0 + 2m/3, and the
  //     first-order copy is a0 + m.
  const std::function<double(int)> lin =
      [a0, m1](int x) { return a0 + m1 * double(x); };
  const double neu  = wall_value(std::uint8_t(MagNeumann), true,  lin);
  const double bulk = wall_value(std::uint8_t(MagBulk),    false, lin);
  check(std::fabs(neu - (a0 + 2.0 * m1 / 3.0)) < (fp64 ? 1e-12 : 2e-6),
        "MagNeumann: linear profile gives a0 + 2m/3", neu, a0 + 2.0 * m1 / 3.0);
  check(std::fabs(bulk - a0) < (fp64 ? 1e-12 : 2e-6),
        "MagNeumann: the MagBulk control reads the node's own value", bulk, a0);
  check(std::fabs(neu - bulk) > 0.3 * m1,
        "MagNeumann: so marking the node MagBulk DOES change the answer", neu, bulk);
}

static void magnetic_thin() {
  const int L = 32;
  const double eta = 0.02, k = 2.0 * M_PI / L, B0 = 0.02, U = 0.01;

  auto profile = [&](int nz, std::size_t T) {
    const int ny = 4;
    host::Magnetic mag(L, ny, nz, Real(eta));
    std::vector<Real> ux(std::size_t(L) * ny * nz, Real(U)),
                      uy(std::size_t(L) * ny * nz, Real(0)),
                      uz(std::size_t(L) * ny * nz, Real(0));
    mag.advect_with(ux.data(), uy.data(), uz.data());
    mag.initialise_with(
        [k, B0](int x, int, int, Real B[3]) {
          B[0] = Real(0); B[1] = Real(B0 * std::sin(k * x)); B[2] = Real(0);
        },
        [](int, int, int, Real u[3]) { u[0] = u[1] = u[2] = Real(0); });
    for (std::size_t t = 0; t < T; ++t) mag.step();
    // one x-line, which is all the field varies along
    std::vector<double> line(static_cast<std::size_t>(L));
    const std::vector<Real>& by = mag.by();
    for (int x = 0; x < L; ++x) line[std::size_t(x)] = double(by[std::size_t(x)]);
    return line;
  };

  for (std::size_t T : {1u, 200u}) {
    const std::vector<double> a = profile(1, T), b = profile(4, T);
    double worst = 0, scale = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      worst = std::max(worst, std::fabs(a[i] - b[i]));
      scale = std::max(scale, std::fabs(b[i]));
    }
    char buf[128];
    std::snprintf(buf, sizeof buf, "magnetic nz=1 == nz=4 after %zu step%s",
                  T, T == 1 ? "" : "s");
    check(worst / scale < (fp64 ? 1e-12 : 1e-5), buf, worst / scale, 0.0);
  }
}

static void alfven(Op op, const char* name) {
  const int L = 64, ny = 4, nz = 4;
  const double nu = 0.01, eta = 0.01;
  const double B0 = 0.02, b = 0.002, rho = 1.0;
  const double k = 2.0 * M_PI / L;
  const double vA = B0 / std::sqrt(rho);

  host::Magnetic mag(L, ny, nz, Real(eta));
  host::Fluid    fl (L, ny, nz, op, Real(nu));

  fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
  mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  fl.initialise_with([k, b, rho](int x, int, int) {
    Macro m;
    m.rho = Real(rho);
    m.ux  = Real(0);
    m.uy  = Real(-b * std::sin(k * x) / std::sqrt(rho));
    m.uz  = Real(0);
    return m;
  });
  mag.initialise_with(
      [k, b, B0](int x, int, int, Real B[3]) {
        B[0] = Real(B0); B[1] = Real(b * std::sin(k * x)); B[2] = Real(0);
      },
      [k, b, rho](int x, int, int, Real u[3]) {
        u[0] = Real(0); u[1] = Real(-b * std::sin(k * x) / std::sqrt(rho)); u[2] = Real(0);
      });

  const std::size_t T = 4000, probe = 200, settle = 400;
  std::vector<double> ts, ph, la;
  double prev = 0;
  bool first = true;
  for (std::size_t t = 0; t <= T; ++t) {
    if (t % probe == 0) {
      const Mode m = fit_x(mag.by().data(), L, ny, nz);
      const double p = first ? m.phase : unwrap(prev, m.phase);
      prev = p; first = false;
      if (t >= settle) {                       // let the seeding transient leave
        ts.push_back(double(t));
        ph.push_back(p);
        la.push_back(std::log(m.amp));
      }
    }
    if (t < T) { mag.compute_field(); fl.step(); mag.step(); }
  }

  const double speed = -slope(ts, ph) / k;
  const double damp  = -slope(ts, la);
  const double damp_exact = 0.5 * (nu + eta) * k * k;

  char buf[128];
  std::snprintf(buf, sizeof buf, "Alfven %s: wave speed v_A = B0/sqrt(rho)", name);
  check(std::fabs(speed - vA) / vA < 0.03, buf, speed, vA);
  std::snprintf(buf, sizeof buf, "Alfven %s: damping rate (nu+eta) k^2 / 2", name);
  check(std::fabs(damp - damp_exact) / damp_exact < 0.15, buf, damp, damp_exact);
}

//==============================================================================
int main() {
  std::printf("Physics checks, host build, Real = %s\n\n", fp64 ? "double" : "float");

  std::printf("  -- geometry and forcing --\n");
  poiseuille(Op::BGK, "BGK", 2e-3, 3e-3);
  poiseuille(Op::CentralMoments, "CM", 2e-3, 3e-3);
  // TRT gets a HUNDREDFOLD TIGHTER tolerance, and that is the point of having
  // it. BGK and the central-moment operator both sit at 1.6e-3 here, which is
  // the wall being 0.005 out of place at tau = 1 (case 1b); TRT at Lambda =
  // 3/16 puts the wall exactly at 0.5 and its amplitude error drops to 1.4e-5.
  // Same grid, same forcing, same 20 000 steps -- the only difference is which
  // rate the antisymmetric part relaxes at.
  //
  // This tolerance is a REGRESSION GUARD, not a pass mark: it is set an order
  // of magnitude above what TRT actually achieves, so that a future change
  // which quietly reintroduces a wall-position error is caught here rather than
  // absorbed by a loose bound.
  poiseuille(Op::TRT, "TRT", 1e-4, 2e-4);
  wall_position();
  shifted_storage();
  regularized_walls();
  closed_box();
  fluid_specular_node();

  std::printf("\n  -- the passive scalar --\n");
  insulating_box();
  conduction();
  // SWEEP OMEGA. Diffusivity 0.125 on D3Q7 is exactly omega = 1, where the
  // collision wipes every non-equilibrium population and ANY consistent fill of
  // the unknown directions gives the right answer. A single test there cannot
  // see a boundary whose effective plane moves with tau, so it is not one test.
  outflow_corners();
  on_node_dirichlet(Real(0.125), "omega = 1");
  on_node_dirichlet(Real(0.05),  "omega > 1");
  on_node_dirichlet(Real(0.30),  "omega < 1");
  specular_identity();
  scalar_source_reaches_wall();
  diffusion();
  advection();
  open_boundary();

  std::printf("\n  -- coupled: buoyancy --\n");
  buoyancy();

  std::printf("\n  -- magnetohydrodynamics --\n");
  resistive_decay();
  magnetic_wall();
  mhd_field_force();
  magnetic_source();
  magnetic_thin();
  magnetic_neumann();
  alfven(Op::BGK, "BGK");
  alfven(Op::CentralMoments, "CM");

  std::printf("\n%s  (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
