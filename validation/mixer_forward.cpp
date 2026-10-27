//==============================================================================
//  THE MIXER FORWARD EVALUATOR AGAINST FOUR ANCHORS -- P1 of
//  doc/mixer_adjoint_plan.md.
//
//  src/topopt/Mixer.hpp solves the steady flow and the steady concentration in
//  Nianzhi's duct and reports a mixing index and a pressure drop. An optimiser
//  will push on every weakness that evaluator has, so before any gradient is
//  built it is held to four things it can be wrong against.
//
//  1. EMPTY DUCT, pressure gradient against the exact series solution
//     (topopt/DuctFlow.hpp): G = K nu jbar / L^2, K = 28.454 (f Re = 56.908).
//     Three grids at fixed tau = 0.8 (diffusive scaling, U = 0.4/L, so the Mach
//     error falls as 1/L^2 with the discretisation error and the ladder
//     measures the scheme -- channel3d.cpp's banner shows what acoustic
//     scaling measures instead). G is the slope of the area-mean density over
//     the middle third of a 3L duct, a full L from the inlet and the outlet.
//
//  2. PLUG-FLOW MIXING LAYER against the erf solution. No walls (y, z
//     periodic, nz = 1) and a uniform inlet, so the exact flow is uniform and
//     the exact concentration, for an inlet band smoothed as erf with width s0,
//     is the same erf with s^2 = s0^2 + D x / U -- the smoothing is a virtual
//     origin, so the comparison needs no sharp step. Axial diffusion is dropped
//     by that solution; its relative effect is ~1/Pe_x <= 2e-3 in the window
//     compared. Measured three ways: the flow's departure from uniform; the
//     pointwise error; and the GROWTH RATE of the layer's variance against the
//     same statistic of the exact profile, i.e. the effective transverse
//     diffusivity. The last is what an optimiser can exploit: numerical
//     diffusion is free mixing.
//
//  3. SOLID SLAB at A_max (the whole design box, gamma = 0). Two numbers: the
//     fraction of the flux that leaks through the slab, and Darcy's law at the
//     slab's centre, u = -grad p / A, which is the drag doing what it says.
//
//  4. ONE GREY DESIGN ON TWO GRIDS (-band, not run by ctest): slanted grooves
//     with tanh edges about two cells wide at L = 40, at L = 40 (the production
//     grid of the plan) and L = 60, at fixed Re, Pe, Da and geometry in units
//     of L. M and Dp/Dp0 must agree within a band. This is the plan's guard
//     against an evaluator that is converged in time but not in space.
//
//  5. STABILITY GUARD (ctest): the same grooves at L = 16, where the flow runs
//     at ~4 U. The evaluator's concentration must stay in [-0.05, 1.05] and
//     converge; the ScalarRegularised control must HALT there. Anchor 4 is too
//     slow for ctest, and this is what keeps the instability it found from
//     coming back unseen.
//
//  6. A UNIFORM INLET STAYS UNIFORM (ctest): the grooves at L = 16 with c = 1
//     at the inlet, whose exact answer is c = 1 everywhere. With the
//     uniform-preserving correction (ScalarSolver::set_uniform_correction,
//     calibrated by MixerEvaluator::calibrate_uniform) the field must be 1 to
//     1e-9; without it, the control must miss by at least 0.1.
//
//  THE GATES WERE REGISTERED ON 2026-10-15, BEFORE THE FIRST RUN, and are the
//  constants below. Each is derived from an estimate, written next to it. A
//  gate that fails stays failed: changing one is a new registration and must be
//  recorded here with the reason and the run that forced it.
//
//  CHANGES AFTER REGISTRATION (the dated record, with hashes, is in
//  results/mixer_forward/registration/REGISTERED.txt):
//    * THE SCALAR OPERATOR. Anchor 4's first run diverged on both grids: the
//      ScalarRegularised scalar is linearly unstable above |u| ~ 0.052 at
//      Pe = 1e3, L = 40 (collision/ScalarTRT.hpp has the numbers). The
//      evaluator now defaults to ScalarTRT at Lambda = 1/4; the gated rows of
//      anchors 2 and 4 run it, and anchor 2 also prints the regularised
//      operator's rows, ungated, so the change is measured, not asserted. No
//      threshold moved.
//    * TWO GATES ADDED, each registered before its first run: G4_PHI (the plan
//      named Phi; the original gate used M, which flattens as M -> 1) and the
//      anchor-5 guard.
//
//  PREDICTIONS (made before the first run), AND WHAT THE RUNS SAID:
//    1. second order, sign constant -- CONFIRMED (below).
//    2. "the variance-growth ratio within 1e-3 of 1, because the second moment
//       of a linear LBM grows exactly as 2 D t" -- WRONG. The theorem is about
//       the whole moment; the check takes a WINDOWED variance of a profile 2-4
//       cells wide, and the scheme's profile also differs from erf.
//    3. leak ~7e-4 -- measured 9.75e-4 / 9.81e-4 (L = 16 / 40), 1.39x the
//       estimate and grid-independent, as Darcy flow at fixed Da must be.
//    4. "no prediction of the size of the difference; the DarcyGuo surface
//       offset (~0.8 cell, tau-dependent) narrows a 10-cell groove by ~16% at
//       L = 40 and ~12% at L = 60, which could alone move M by several
//       percent" (the registered wording) -- CONTRADICTED: M moved 0.38%.
//
//  MEASURED, FP64, D3Q27 CentralMoments<DarcyGuo> + D3Q7 ScalarTRT (Lambda =
//  1/4) with the uniform-preserving correction, 2026-10-15/16 (every run and
//  every change since registration: results/mixer_forward/registration/):
//    1. G/G_exact - 1 = +1.2408e-02, +2.4213e-03, +5.6509e-04, +1.3838e-04 at
//       L = 8, 16, 32, 64; order 2.357, 2.099, 2.030. PASS.
//    2. max |c - c_exact| = 1.1647e-02, 2.9934e-03, 7.6030e-04 at L = 20, 40,
//       80, order 1.960, 1.977, c in [0, 1]; flow uniform to 5.3e-15 of U.
//       Variance growth / exact = 0.943929, 0.984564, 0.995819: FAIL at L = 40
//       (1.5% against 1%). Over the rows whose window spans 4 sigma (ungated):
//       0.968966, 0.994500, 0.998775. The regularised scalar the evaluator
//       replaced, same rows (ungated): 1.166101, 1.032700, 1.007459; pointwise
//       1.9811e-02, 5.7474e-03, 1.3221e-03, and c outside [0, 1] by 6.8e-3.
//    3. L = 16: leak 9.7528e-04 PASS; Darcy u A / (-dp/dx) = 1.141476 FAIL --
//       the drag region carries an odd-even velocity layer next to every solid
//       surface, decaying x(-0.42) per cell at tau = 0.62, and the 8-cell slab's
//       centre is 4 cells in. L = 40 (-full): leak 9.811652e-04, Darcy 0.999959,
//       both PASS (the layer decays x(-0.19) per cell at tau = 0.8).
//    4. M = 0.164867 / 0.164240, Phi = 0.1743615 / 0.1746236, Dp/Dp0 = 3.92972 /
//       3.94161 at L = 40 / 60: 0.38%, 0.15%, 0.30% apart. PASS. (Measured
//       before the second review's changes. On the final code L = 40 reproduced
//       every printed digit; the L = 60 re-run was killed by the system at 44k
//       steps, so its row is the earlier run's.) Converged c in
//       [-0.0027, 1.0927] and [-0.00002, 1.0023]. The first run (regularised
//       scalar) DIVERGED; the second (TRT, uncorrected) halted, the field reaching
//       1.42 in steady state. Converged without the correction Phi is 0.14868
//       at L = 40, 15% lower; that difference is the flow's compressibility
//       diluting the scalar (src/topopt/Mixer.hpp, "what it changes").
//    5. ScalarTRT converges at L = 16 but to c in [-0.0844, 1.7272] -- the
//       maximum at a near-stagnant drag node (divergence 310% of |u| per cell),
//       the minimum in open fluid at local cell Pe ~190. FAIL: L = 16 is under-
//       resolved at Pe = 1e3. The regularised control blows up (PASS).
//    6. Correction on: max |c - 1| = 2.319e-10, outlet Phi 1.9e-22. Off: 1.240.
//       PASS.
//
//  WHAT THIS DOES NOT TEST: the duct's walls with the scalar (anchor 2 has
//  none); unsteady behaviour; any gradient; FP32 (it prints a skip there, since
//  the convergence tolerances below are under FP32's resolution).
//==============================================================================
#include "Campaign.hpp"
#include "topopt/Mixer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;
using namespace lbm::topopt;

namespace {

//------------------------------------------------------------------------------
// GATES -- registered 2026-10-15, before the first run.
//------------------------------------------------------------------------------
// 1. empty duct
constexpr double G1_ERR   = 1e-2;   // |G/G_exact - 1| at L = 32. A second-order
                                    // scheme at 32 cells; a percent would be a
                                    // gross error in a quantity the wall and the
                                    // Mach number set at O(1/L^2) ~ 1e-3.
constexpr double G1_ORD_LO = 1.5, G1_ORD_HI = 2.5;   // order, L = 16 -> 32
// 2. plug-flow mixing layer, at L = 40 unless stated
constexpr double G2_PLUG  = 1e-10;  // max |u - U e_x| / U anywhere: uniform flow
                                    // is an exact fixed point of every closure.
constexpr double G2_DEFF  = 1e-2;   // |variance-growth ratio - 1|, both edges
constexpr double G2_POINT = 2e-2;   // max |c - c_exact| in the window: the layer
                                    // spans 5-10 cells there, (1/5)^2 / 2 ~ 2e-2
constexpr double G2_ORD_LO = 1.5;   // order of the pointwise error, 40 -> 80
// 3. solid slab, at every L run
constexpr double G3_LEAK  = 5e-3;   // flux fraction through the slab; ~7x the
                                    // Darcy estimate of 7e-4
constexpr double G3_DARCY = 1e-2;   // |u A / (-dp/dx) - 1| at the slab centre
// 4. two grids (-band)
constexpr double G4_M  = 0.10;      // |M(60) - M(40)| / M(60)
constexpr double G4_DP = 0.10;      // |r(60) - r(40)| / r(60), r = Dp/Dp0
//------------------------------------------------------------------------------
// ADDED after the first anchor-4 run, registered before their own first run
// (timestamp in REGISTERED.txt).
//------------------------------------------------------------------------------
constexpr double G4_PHI = 0.10;     // |Phi(60) - Phi(40)| / Phi(60), as the plan says
constexpr double G5_LO = -0.05, G5_HI = 1.05;   // the evaluator's own halt band
// Added with the uniform-preserving correction, registered before its first run.
constexpr double G6_EXACT   = 1e-9;  // max |c - 1| for a uniform inlet, correction on:
                                     // exact by construction, up to the calibration's
                                     // 1e-13 and the run's own convergence
constexpr double G6_CONTROL = 0.1;   // ... and at least this, correction off -- so the
                                     // check can fail (0.27-2.08 measured, L = 16, 876 steps)

bool gate(const char* what, bool ok, bool& all) {
  std::printf("    %-58s %s\n", what, ok ? "PASS" : "FAIL");
  all = all && ok;
  return ok;
}

double order(double e_coarse, double e_fine, double ratio) {
  return std::log(std::fabs(e_coarse) / std::fabs(e_fine)) / std::log(ratio);
}

//------------------------------------------------------------------------------
// 1. EMPTY DUCT
//------------------------------------------------------------------------------
double duct_error(Index L, int verbose, double& secs) {
  MixerSpec sp;
  sp.L = L;
  sp.len_in = 1.0; sp.len_des = 1.0; sp.len_out = 1.0;
  sp.H = 0.0;
  sp.U = 0.4 / double(L);
  sp.Re = sp.U * double(L) / 0.1;          // nu = 0.1, tau = 0.8 on every grid
  MixerEvaluator ev(sp);
  const Convergence cv = ev.run_flow(1e-9, 30.0, 0, verbose);
  const auto rho = ev.host_rho(), ux = ev.host_ux();
  // Least-squares slope of <rho>(x) over x in [L, 2L].
  double sx = 0, sy = 0, sxx = 0, sxy = 0, sj = 0, cnt = 0;
  for (Index x = L; x <= 2 * L; ++x) {
    double rb, jb; ev.plane_means(rho, ux, x, rb, jb);
    sx += x; sy += rb; sxx += double(x) * x; sxy += double(x) * rb; sj += jb; cnt += 1;
  }
  const double slope = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx);
  const double G = -double(cs2<D3Q27, Real>()) * slope;
  const double jbar = sj / cnt;
  const double Gex = ev.g.K * ev.g.nu * jbar / (double(L) * L);
  secs = cv.secs;
  std::printf("    L = %3lld  U = %.5f  tau = %.4f  steps %7zu  e_hat %.2e%s  G = %.8e  "
              "G_exact = %.8e  err %+.4e\n", (long long)L, sp.U, 1.0 / ev.omega_fluid(),
              cv.steps, cv.e_hat, cv.converged ? "" : " NOT CONVERGED", G, Gex, G / Gex - 1.0);
  if (!cv.converged || !cv.finite) return NAN;
  return G / Gex - 1.0;
}

bool anchor_duct(bool full, int verbose) {
  std::printf("\n  1. EMPTY DUCT -- pressure gradient against the exact series\n");
  std::printf("    K = %.6f (f Re = %.4f), u_max/Ubar = %.5f from the series\n",
              duct_K(), 2.0 * duct_K(), duct_self_check());
  std::vector<Index> Ls = {8, 16, 32};
  if (full) Ls.push_back(64);
  std::vector<double> e;
  double secs = 0, t;
  for (Index L : Ls) { e.push_back(duct_error(L, verbose, t)); secs += t; }
  for (std::size_t i = 1; i < e.size(); ++i)
    std::printf("    order %3lld -> %3lld : %.3f\n", (long long)Ls[i - 1], (long long)Ls[i],
                order(e[i - 1], e[i], double(Ls[i]) / double(Ls[i - 1])));
  bool all = true;
  bool mono = true;
  for (std::size_t i = 1; i < e.size(); ++i)
    mono = mono && std::isfinite(e[i]) && std::fabs(e[i]) < std::fabs(e[i - 1]) &&
           (e[i] > 0) == (e[i - 1] > 0);
  gate("|G/G_exact - 1| at L = 32 <= 1e-2", std::fabs(e[2]) <= G1_ERR, all);
  const double p = order(e[1], e[2], 2.0);
  gate("order 16 -> 32 in [1.5, 2.5]", p >= G1_ORD_LO && p <= G1_ORD_HI, all);
  gate("|error| falls on every rung, sign constant", mono, all);
  std::printf("    (%.1f s)\n", secs);
  return all;
}

//------------------------------------------------------------------------------
// 2. PLUG-FLOW MIXING LAYER
//------------------------------------------------------------------------------
struct LayerResult { double plug, point, ratio_up, ratio_dn, ratio_up_r, ratio_dn_r; bool ok_conv; };

// Second central moment of the forward differences of c across one edge, over
// nodes y in [ya, yb). `sign` = +1 for the rising edge, -1 for the falling one.
double edge_variance(const std::vector<double>& col, Index ya, Index yb, double sign) {
  double s0 = 0, s1 = 0, s2 = 0;
  for (Index y = ya; y + 1 < yb; ++y) {
    const double w = sign * (col[std::size_t(y + 1)] - col[std::size_t(y)]);
    const double eta = double(y) + 1.0;           // the difference sits between nodes
    s0 += w; s1 += w * eta; s2 += w * eta * eta;
  }
  const double mu = s1 / s0;
  return s2 / s0 - mu * mu;
}
double fit_slope(const std::vector<double>& x, const std::vector<double>& v) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0; const double n = double(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    sx += x[i]; sy += v[i]; sxx += x[i] * x[i]; sxy += x[i] * v[i];
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

template <class Ev>
LayerResult layer(Index L, int verbose) {
  MixerSpec sp;
  sp.L = L;
  sp.walls = false; sp.nz_open = 1;
  sp.developed = false; sp.band = true;
  sp.s0 = double(L) / 40.0;
  sp.U = 0.025; sp.Re = 10.0; sp.Pe = 1000.0;
  Ev ev(sp);
  LayerResult r{};
  const Convergence cf = ev.run_flow(1e-10, 5.0, 0, verbose);
  const auto ux = ev.host_ux(), uy = ev.host_uy(), uz = ev.host_uz();
  double plug = 0;
  for (Index n : ev.fnodes)
    plug = std::max({plug, std::fabs(ux[n] - sp.U), std::fabs(uy[n]), std::fabs(uz[n])});
  r.plug = plug / sp.U;
  const Convergence cs = ev.run_scalar(1e-9, 30.0, 0, verbose);
  r.ok_conv = cf.converged && cs.converged && cf.finite && cs.finite && cs.bounded;
  const auto c = ev.host_c();
  const Index nx = ev.g.nx, ny = ev.g.ny;
  const double D = ev.g.D;
  double point = 0;
  std::vector<double> xs, vu, vd, vue, vde;
  // The window caveat: the variance is taken over half the period, L/2, so its
  // tails are cut once the layer is wide. Rows where the half-window L/4 spans
  // at least 4 standard deviations (sqrt(2) s) are kept for a second, ungated
  // ratio that the cut cannot bias.
  const double s_lim = double(L) / (16.0 * std::sqrt(2.0));
  std::vector<double> xr, vur, vdr, vuer, vder;
  std::vector<double> col(static_cast<std::size_t>(ny), 0.0),
                      cole(static_cast<std::size_t>(ny), 0.0);
  for (Index x = L / 2; x <= nx - 1 - L / 2; ++x) {
    const double s = std::sqrt(sp.s0 * sp.s0 + D * double(x) / sp.U);
    for (Index y = 0; y < ny; ++y) {
      col[std::size_t(y)] = c[std::size_t(ev.dom.id(x, y, 0))];
      cole[std::size_t(y)] = inlet_c(ev.g.eta(y), double(L), s, true);
      point = std::max(point, std::fabs(col[std::size_t(y)] - cole[std::size_t(y)]));
    }
    xs.push_back(double(x));
    vu.push_back(edge_variance(col, 0, L / 2, +1.0));
    vd.push_back(edge_variance(col, L / 2, L, -1.0));
    vue.push_back(edge_variance(cole, 0, L / 2, +1.0));
    vde.push_back(edge_variance(cole, L / 2, L, -1.0));
    if (s <= s_lim) {
      xr.push_back(xs.back());
      vur.push_back(vu.back()); vdr.push_back(vd.back());
      vuer.push_back(vue.back()); vder.push_back(vde.back());
    }
  }
  r.point = point;
  const double su = fit_slope(xs, vu), sd = fit_slope(xs, vd);
  const double sue = fit_slope(xs, vue), sde = fit_slope(xs, vde);
  r.ratio_up = su / sue; r.ratio_dn = sd / sde;
  r.ratio_up_r = xr.size() >= 5 ? fit_slope(xr, vur) / fit_slope(xr, vuer) : NAN;
  r.ratio_dn_r = xr.size() >= 5 ? fit_slope(xr, vdr) / fit_slope(xr, vder) : NAN;
  std::printf("    L = %3lld  %-17s s0 = %.2f  omega_c = %.5f  flow %s, scalar %zu steps "
              "e_hat %.2e e_phi %.2e%s (%.1f s)\n", (long long)L, Ev::ScalarColl::name, sp.s0,
              ev.omega_scalar(), cf.converged ? "steady" : "NOT STEADY", cs.steps, cs.e_hat,
              cs.e_phi, !cs.bounded ? " HALTED (c out of bounds)" :
                        (cs.converged ? "" : " NOT CONVERGED"), cf.secs + cs.secs);
  std::printf("            plug %.2e   c in [%.5f, %.5f]   max|c - c_exact| %.4e\n",
              r.plug, cs.cmin, cs.cmax, r.point);
  std::printf("            variance growth / exact: rising %.6f  falling %.6f   (D_eff from "
              "slope %.5e, D %.5e)\n", r.ratio_up, r.ratio_dn, 0.5 * sp.U * su, D);
  std::printf("            the same over x <= %.0f, where the window holds 4 sigma "
              "(%zu rows, ungated): rising %.6f  falling %.6f\n",
              xr.empty() ? NAN : xr.back(), xr.size(), r.ratio_up_r, r.ratio_dn_r);
  return r;
}

bool anchor_layer(int verbose) {
  std::printf("\n  2. PLUG-FLOW MIXING LAYER -- against the erf solution, Pe = 1e3, U = 0.025\n");
  const Index Ls[3] = {20, 40, 80};
  LayerResult r[3];
  for (int i = 0; i < 3; ++i) r[i] = layer<MixerEvaluator>(Ls[i], verbose);
  for (int i = 1; i < 3; ++i)
    std::printf("    order of max|c - c_exact| %lld -> %lld : %.3f\n", (long long)Ls[i - 1],
                (long long)Ls[i], order(r[i - 1].point, r[i].point, 2.0));
  bool all = true;
  bool conv = r[0].ok_conv && r[1].ok_conv && r[2].ok_conv;
  double plug = std::max({r[0].plug, r[1].plug, r[2].plug});
  gate("flow and concentration converged on every grid", conv, all);
  gate("flow uniform to 1e-10 of U on every grid", plug <= G2_PLUG, all);
  gate("variance growth within 1e-2 of exact at L = 40 (both edges)",
       std::fabs(r[1].ratio_up - 1.0) <= G2_DEFF && std::fabs(r[1].ratio_dn - 1.0) <= G2_DEFF, all);
  gate("max|c - c_exact| <= 2e-2 at L = 40", r[1].point <= G2_POINT, all);
  gate("pointwise error falls 20 -> 40 -> 80, order 40 -> 80 >= 1.5",
       r[1].point < r[0].point && r[2].point < r[1].point &&
       order(r[1].point, r[2].point, 2.0) >= G2_ORD_LO, all);
  // The operator the evaluator used before (ungated): the same rows, so the
  // change of operator is a measured difference and not a remembered one.
  std::printf("    for comparison, ungated -- the regularised scalar the evaluator replaced:\n");
  for (int i = 0; i < 3; ++i) layer<MixerEvaluatorReg>(Ls[i], verbose);
  return all;
}

//------------------------------------------------------------------------------
// 3. SOLID SLAB
//------------------------------------------------------------------------------
bool slab(Index L, int verbose, bool diag) {
  MixerSpec sp;
  sp.L = L;
  MixerEvaluator ev(sp);
  ev.set_design_fn([](double, double, double) { return 0.0; });
  const Convergence cv = ev.run_flow(1e-8, 40.0, 0, verbose);
  const auto rho = ev.host_rho(), ux = ev.host_ux();
  const MixerGeom& g = ev.g;
  const Index xc = (g.x_d0 + g.x_d1) / 2;
  double qs = 0, qt = 0;
  for (Index z = g.z0; z <= g.z1; ++z)
    for (Index y = g.y0; y <= g.y1; ++y) {
      const Index n = ev.dom.id(xc, y, z);
      qt += rho[n] * ux[n];
      if (g.in_design(xc, y, z)) qs += rho[n] * ux[n];
    }
  const double leak = qs / qt;
  const Index yc = (g.y0 + g.y1) / 2, zc = g.z0 + (g.nzd - 1) / 2;
  const Index n = ev.dom.id(xc, yc, zc);
  const double dpdx = double(cs2<D3Q27, Real>()) *
                      (rho[ev.dom.id(xc + 1, yc, zc)] - rho[ev.dom.id(xc - 1, yc, zc)]) / 2.0;
  const double darcy = ux[n] * g.A_max / (-dpdx);
  std::printf("    L = %3lld  A_max = %.4f  tau = %.4f  steps %7zu  e_hat %.2e%s (%.1f s)\n",
              (long long)L, g.A_max, 1.0 / ev.omega_fluid(), cv.steps, cv.e_hat,
              cv.converged ? "" : " NOT CONVERGED", cv.secs);
  std::printf("            leak Q_slab/Q = %.4e   at the slab centre (%lld,%lld,%lld) u/U = %.4e  "
              "u A / (-dp/dx) = %.6f\n", leak, (long long)xc, (long long)yc, (long long)zc,
              ux[n] / sp.U, darcy);
  // -diag: keep stepping past the stopping point and watch the two slab
  // numbers. Diagnosis only; the gates below read the values at the stop.
  if (diag) {
    for (int k = 1; k <= 2; ++k) {
      for (std::size_t s = 0; s < std::size_t(cv.T_relax / 2.0); ++s) ev.fluid.step();
      const auto r2 = ev.host_rho(), u2 = ev.host_ux();
      double q2s = 0, q2t = 0;
      for (Index z = g.z0; z <= g.z1; ++z)
        for (Index y = g.y0; y <= g.y1; ++y) {
          const Index m = ev.dom.id(xc, y, z);
          q2t += r2[m] * u2[m];
          if (g.in_design(xc, y, z)) q2s += r2[m] * u2[m];
        }
      const double dp2 = double(cs2<D3Q27, Real>()) *
          (r2[ev.dom.id(xc + 1, yc, zc)] - r2[ev.dom.id(xc - 1, yc, zc)]) / 2.0;
      std::printf("      diag  +%5.1f T  leak %.6e  u/U %.6e  u A / (-dp/dx) %.6f\n",
                  0.5 * k, q2s / q2t, u2[n] / sp.U, u2[n] * g.A_max / (-dp2));
    }
    // Profiles through the slab centre: is the field smooth where Darcy is read?
    const auto r3 = ev.host_rho(), u3 = ev.host_ux(), v3 = ev.host_uy(), w3 = ev.host_uz();
    const double c2 = double(cs2<D3Q27, Real>());
    std::printf("      diag  along x at (y, z) = (%lld, %lld):   x   rho-1   u/U   u A/(-dp/dx, central)\n",
                (long long)yc, (long long)zc);
    for (Index x = xc - 6; x <= xc + 6; ++x) {
      const Index m = ev.dom.id(x, yc, zc);
      const double dp3 = c2 * (r3[ev.dom.id(x + 1, yc, zc)] - r3[ev.dom.id(x - 1, yc, zc)]) / 2.0;
      std::printf("        %3lld  %+.8e  %+.6e  %.6f\n", (long long)x, r3[m] - 1.0,
                  u3[m] / sp.U, u3[m] * g.A_max / (-dp3));
    }
    std::printf("      diag  along z at (x, y) = (%lld, %lld):   z   rho-1   u/U   v/U   w/U   "
                "u A/(-dp/dx)\n", (long long)xc, (long long)yc);
    for (Index z = g.z0; z <= g.z1; ++z) {
      const Index m = ev.dom.id(xc, yc, z);
      const double dp3 = c2 * (r3[ev.dom.id(xc + 1, yc, z)] - r3[ev.dom.id(xc - 1, yc, z)]) / 2.0;
      std::printf("        %3lld%s %+.8e  %+.6e  %+.3e  %+.3e  %.6f\n", (long long)z,
                  g.in_design(xc, yc, z) ? "s" : " ", r3[m] - 1.0, u3[m] / sp.U,
                  v3[m] / sp.U, w3[m] / sp.U, u3[m] * g.A_max / (-dp3));
    }
    std::printf("      diag  along y at (x, z) = (%lld, %lld):   y   u/U\n", (long long)xc, (long long)zc);
    for (Index y = g.y0; y <= g.y1; ++y)
      std::printf("        %3lld  %+.6e\n", (long long)y, u3[ev.dom.id(xc, y, zc)] / sp.U);
  }
  bool all = true;
  char what[128];
  std::snprintf(what, sizeof what, "L = %lld converged", (long long)L);
  gate(what, cv.converged && cv.finite, all);
  std::snprintf(what, sizeof what, "L = %lld leak <= 5e-3", (long long)L);
  gate(what, std::fabs(leak) <= G3_LEAK, all);
  std::snprintf(what, sizeof what, "L = %lld Darcy's law at the slab centre within 1e-2",
                (long long)L);
  gate(what, std::fabs(darcy - 1.0) <= G3_DARCY, all);
  return all;
}

bool anchor_slab(bool full, int verbose, bool diag) {
  std::printf("\n  3. SOLID SLAB -- the whole design box at A_max, H = L/2, Da = 1e-5\n");
  bool all = slab(16, verbose, diag);
  if (full) all = slab(40, verbose, diag) && all;
  return all;
}

//------------------------------------------------------------------------------
// 4. ONE GREY DESIGN ON TWO GRIDS
//------------------------------------------------------------------------------
struct BandResult { MixerResult m; bool ok; };

// Slanted grooves: solid where sin(2 pi (xi + eta) / lambda) > 0, with tanh
// edges; lambda = L/2, edges ~0.05 L wide. Defined in units of L, so every grid
// sees the same geometry.
template <class Ev>
void set_grooves(Ev& ev) {
  const double lam = 0.5, w = 0.3, pi = 3.14159265358979323846;
  ev.set_design_fn([=](double xi, double eta, double) {
    return 0.5 * (1.0 + std::tanh(-std::sin(2.0 * pi * (xi + eta) / lam) / w));
  });
}

BandResult band_run(Index L, int verbose) {
  MixerSpec sp;
  sp.L = L;
  sp.s0 = double(L) / 40.0;
  MixerEvaluator ev(sp);
  set_grooves(ev);
  std::printf("    L = %lld\n", (long long)L);
  ev.print_spec("      ");
  std::printf("      connected (gamma >= 0.5 from inlet to outlet): %s\n",
              ev.connected() ? "yes" : "NO");
  const Convergence cf = ev.run_flow(1e-6, 40.0, 0, verbose);
  std::printf("      flow   %8zu steps  t/T = %.3f  e_hat %.2e%s  (%.0f s)\n", cf.steps,
              double(cf.steps) / cf.T_relax, cf.e_hat, cf.converged ? "" : " NOT CONVERGED",
              cf.secs);
  std::vector<double> hist;
  const Convergence cs = ev.run_scalar(1e-6, 40.0, 0, verbose, &hist);
  std::printf("      scalar %8zu steps  t/T = %.3f  e_hat %.2e  e_phi %.2e%s  (%.0f s)\n",
              cs.steps, double(cs.steps) / cs.T_relax, cs.e_hat, cs.e_phi,
              !cs.bounded ? "  HALTED (c out of bounds)" : (cs.converged ? "" : " NOT CONVERGED"),
              cs.secs);
  // Phi along the run, at a tenth, a quarter, half and the end: how the slow
  // (diffusive) relaxation inside the solid reaches the reported number.
  if (!hist.empty())
    for (double f : {0.1, 0.25, 0.5, 1.0}) {
      const std::size_t i = std::min(hist.size() - 1, std::size_t(f * double(hist.size())));
      std::printf("      Phi at %3.0f%% of the scalar run: %.8e\n", 100.0 * f, hist[i]);
    }
  BandResult b;
  b.m = ev.measure();
  b.ok = cf.converged && cs.converged && cf.finite && cs.finite && cs.bounded;
  std::printf("      Phi = %.6e  M = %.6f  cbar = %.6f  Dp = %.6e  Dp0 = %.6e  Dp/Dp0 = %.5f\n",
              b.m.Phi, b.m.M, b.m.cbar, b.m.dp, b.m.dp0, b.m.dp_ratio);
  std::printf("      limits: max|u| = %.5f (%.2f U)  Ma_max = %.4f  density drop %.2f%%  "
              "c in [%.5f, %.5f]  flow r_odd = %.2e\n", b.m.umax, b.m.umax / sp.U, b.m.Ma_max,
              100.0 * b.m.drho, b.m.cmin, b.m.cmax, cf.r_odd);
  return b;
}

bool anchor_band(Index La, Index Lb, int verbose) {
  std::printf("\n  4. ONE GREY DESIGN ON TWO GRIDS -- slanted grooves, Re = 10, Pe = 1e3, "
              "Da = 1e-5\n");
  const BandResult a = band_run(La, verbose), b = band_run(Lb, verbose);
  const double dM = std::fabs(b.m.M - a.m.M) / std::fabs(b.m.M);
  const double dP = std::fabs(b.m.dp_ratio - a.m.dp_ratio) / std::fabs(b.m.dp_ratio);
  const double dF = std::fabs(b.m.Phi - a.m.Phi) / std::fabs(b.m.Phi);
  std::printf("    |M(%lld) - M(%lld)| / M(%lld) = %.4f    |r(%lld) - r(%lld)| / r(%lld) = %.4f"
              "    |Phi(%lld) - Phi(%lld)| / Phi(%lld) = %.4f\n",
              (long long)Lb, (long long)La, (long long)Lb, dM, (long long)Lb, (long long)La,
              (long long)Lb, dP, (long long)Lb, (long long)La, (long long)Lb, dF);
  bool all = true;
  gate("both grids converged", a.ok && b.ok, all);
  // A band compared between runs that did not converge compares two transients
  // -- the first TRT run passed all three on states ~1000 steps from the
  // unmixed seed. So the concentration bands are NOT EVALUATED unless both
  // concentrations converged; the flow band only needs the flows, which the
  // halt does not touch.
  if (a.ok && b.ok) {
    gate("M within 10% between the grids", dM <= G4_M, all);
    gate("Phi within 10% between the grids (added gate)", dF <= G4_PHI, all);
  } else {
    std::printf("    %-58s %s\n", "M and Phi bands: concentration not converged", "NOT EVALUATED");
  }
  gate("Dp/Dp0 within 10% between the grids", dP <= G4_DP, all);
  return all;
}

//------------------------------------------------------------------------------
// 5. STABILITY GUARD -- the grooves at L = 16
//------------------------------------------------------------------------------
template <class Ev>
Convergence guard_run(int verbose, MixerResult& m) {
  MixerSpec sp;
  sp.L = 16;
  sp.s0 = 16.0 / 40.0;
  Ev ev(sp);
  set_grooves(ev);
  const Convergence cf = ev.run_flow(1e-6, 40.0, 0, verbose);
  const Convergence cs = ev.run_scalar(1e-4, 40.0, 0, verbose);
  m = ev.measure();
  // Diagnosis, ungated: where the extreme concentrations sit, what drag and
  // speed are there, and the D3Q7 central divergence of the frozen velocity --
  // a uniform c stays uniform only where that divergence vanishes.
  {
    const auto c = ev.host_c(), ux = ev.host_ux(), uy = ev.host_uy(), uz = ev.host_uz();
    auto hA = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ev.A);
    Index nmax = ev.snodes[0], nmin = ev.snodes[0];
    for (Index n : ev.snodes) { if (c[n] > c[nmax]) nmax = n; if (c[n] < c[nmin]) nmin = n; }
    for (Index n : {nmax, nmin}) {
      Index px, py, pz; ev.dom.coords(n, px, py, pz);
      const Index x = px - ev.dom.hx, y = py - ev.dom.hy, z = pz - ev.dom.hz;
      const double div = 0.5 * (ux[ev.dom.id(x + 1, y, z)] - ux[ev.dom.id(x - 1, y, z)] +
                                uy[ev.dom.id(x, y + 1, z)] - uy[ev.dom.id(x, y - 1, z)] +
                                uz[ev.dom.id(x, y, z + 1)] - uz[ev.dom.id(x, y, z - 1)]);
      std::printf("      diag  c = %+.5f at (%lld,%lld,%lld)  A = %.4g  |u|/U = %.4f  "
                  "D3Q7 div u = %+.3e (%.2f%% of |u| per cell)\n", c[n], (long long)x,
                  (long long)y, (long long)z, double(hA(n)),
                  std::sqrt(ux[n] * ux[n] + uy[n] * uy[n] + uz[n] * uz[n]) / sp.U, div,
                  100.0 * std::fabs(div) / std::max(1e-300, std::sqrt(ux[n] * ux[n] +
                  uy[n] * uy[n] + uz[n] * uz[n])));
    }
    double dmax = 0.0, dmax_open = 0.0;
    for (Index n : ev.snodes) {
      Index px, py, pz; ev.dom.coords(n, px, py, pz);
      const Index x = px - ev.dom.hx, y = py - ev.dom.hy, z = pz - ev.dom.hz;
      if (x < 1 || x > ev.g.nx - 2) continue;
      const double div = std::fabs(0.5 * (ux[ev.dom.id(x + 1, y, z)] - ux[ev.dom.id(x - 1, y, z)] +
                                          uy[ev.dom.id(x, y + 1, z)] - uy[ev.dom.id(x, y - 1, z)] +
                                          uz[ev.dom.id(x, y, z + 1)] - uz[ev.dom.id(x, y, z - 1)]));
      dmax = std::max(dmax, div);
      if (double(hA(n)) == 0.0 && !ev.g.in_design(x, y, z)) dmax_open = std::max(dmax_open, div);
    }
    std::printf("      diag  max |D3Q7 div u| / U: %.3e anywhere, %.3e outside the design box\n",
                dmax / sp.U, dmax_open / sp.U);
  }
  std::printf("    %-17s flow %zu steps%s, max|u| = %.2f U; scalar %zu steps  c in [%.5f, %.5f]"
              "%s  (%.1f s)\n", Ev::ScalarColl::name, cf.steps, cf.converged ? "" : " NOT CONVERGED",
              m.umax / sp.U, cs.steps, cs.cmin, cs.cmax,
              !cs.bounded ? "  HALTED" : (cs.converged ? "  converged" : "  NOT CONVERGED"),
              cf.secs + cs.secs);
  return cs;
}

//------------------------------------------------------------------------------
// 6. A UNIFORM INLET STAYS UNIFORM
//------------------------------------------------------------------------------
double uniform_dev(bool fix, int verbose, Calibration& cb, double& phi) {
  MixerSpec sp;
  sp.L = 16;
  sp.c_uniform = 1.0;
  sp.uniform_fix = fix;
  MixerEvaluator ev(sp);
  set_grooves(ev);
  ev.run_flow(1e-6, 40.0, 0, verbose);
  // Correction on: converge hard. Off: the deviation shows within a thousand
  // steps, and the run is capped rather than left to drift (band widened so the
  // halt does not end it first).
  const Convergence cs = fix ? ev.run_scalar(1e-10, 40.0, 0, verbose)
                             : ev.run_scalar(1e-10, 0.05, 0, verbose, nullptr, true, 10.0);
  cb = ev.calib;
  const auto c = ev.host_c();
  double dev = 0.0;
  for (Index n : ev.snodes) dev = std::max(dev, std::fabs(c[n] - 1.0));
  phi = ev.measure().Phi;
  std::printf("    correction %-3s  scalar %zu steps%s  max|c - 1| = %.3e  outlet Phi = %.3e",
              fix ? "on" : "off", cs.steps, cs.converged ? " (converged)" : " (capped)", dev, phi);
  if (fix)
    std::printf("   [calibration: %zu steps, last change %.1e, fix in [%.4f, %.4f], %.1f s]",
                cb.steps, cb.change, cb.fix_min, cb.fix_max, cb.secs);
  std::printf("\n");
  return dev;
}

bool anchor_uniform(int verbose) {
  std::printf("\n  6. A UNIFORM INLET STAYS UNIFORM -- the grooves at L = 16, c = 1 at the inlet\n");
  Calibration cb_on, cb_off;
  double phi_on, phi_off;
  const double on  = uniform_dev(true, verbose, cb_on, phi_on);
  const double off = uniform_dev(false, verbose, cb_off, phi_off);
  bool all = true;
  gate("correction on: max|c - 1| <= 1e-9", cb_on.converged && on <= G6_EXACT, all);
  gate("control, correction off: max|c - 1| >= 0.1", off >= G6_CONTROL, all);
  return all;
}

bool anchor_guard(int verbose) {
  std::printf("\n  5. STABILITY GUARD -- the grooves at L = 16, Pe = 1e3, where the flow runs "
              "at ~4 U\n");
  MixerResult mt, mr;
  const Convergence ct = guard_run<MixerEvaluator>(verbose, mt);
  const Convergence cr = guard_run<MixerEvaluatorReg>(verbose, mr);
  bool all = true;
  gate("evaluator: concentration converged and in [-0.05, 1.05]",
       ct.converged && ct.bounded && ct.cmin >= G5_LO && ct.cmax <= G5_HI, all);
  gate("control: the regularised scalar halts (the guard can fail)", !cr.bounded, all);
  return all;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    bool full = false, band = false, diag = false;
    int only = 0, verbose = 0;
    Index La = 40, Lb = 60;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-full") full = true;
      else if (a == "-band") band = true;
      else if (a == "-diag") diag = true;
      else if (a == "-only" && i + 1 < argc) only = std::atoi(argv[++i]);
      else if (a == "-v" && i + 1 < argc) verbose = std::atoi(argv[++i]);
      else if (a == "-bandL" && i + 2 < argc) { La = std::atoi(argv[++i]); Lb = std::atoi(argv[++i]); }
      else {
        std::fprintf(stderr, "mixer_forward: unknown option '%s'\n"
                     "  options: -full  -band  -diag  -only N (1-6)  -v N  -bandL La Lb\n", a.c_str());
        rc = 2;
      }
    }
    if (rc == 0 && sizeof(Real) == 4) {
      // 77 is registered as SKIP_RETURN_CODE in validation/CMakeLists.txt, so
      // ctest reports this as skipped -- not as a pass of gates never run.
      std::printf("mixer_forward: SKIP -- the gates are FP64 gates (convergence to 1e-9..1e-10\n"
                  "  is below FP32's resolution); build with LBM_PRECISION=double.\n");
      rc = 77;
    } else if (rc == 0) {
      std::printf("MIXER FORWARD EVALUATOR -- six anchors (doc/mixer_adjoint_plan.md, P1)\n");
      std::printf("  D3Q27 CentralMoments<DarcyGuo> raw + D3Q7 %s, FP64, gates registered "
                  "2026-10-15\n", MixScalarColl::name);
      bool all = true;
      if (only == 0 || only == 1) all = anchor_duct(full, verbose) && all;
      if (only == 0 || only == 2) all = anchor_layer(verbose) && all;
      if (only == 0 || only == 3) all = anchor_slab(full, verbose, diag) && all;
      if (only == 0 || only == 5) all = anchor_guard(verbose) && all;
      if (only == 0 || only == 6) all = anchor_uniform(verbose) && all;
      if (band || only == 4) all = anchor_band(La, Lb, verbose) && all;
      std::printf("\n%s\n", all ? "PASS" : "FAIL");
      rc = all ? 0 : 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
