//==============================================================================
//  validation/jet1d -- the slender viscous jet of validation/Jet1D.hpp against
//  exact solutions, one term at a time, and all of them together by
//  self-convergence.
//
//  Each section switches on one term and has a closed form (nondimensional:
//  Q = L = 1 and 3 mu = 1 unless said):
//   J1. STRETCHING ALONE, drawn from u0 = 1 to uL = 100: N is constant, so
//       ln u is linear, u = 100^z, N0 = ln 100.
//   J2. A UNIFORM PULL f, FREE LANDING (N(L) = 0). With K = f/(3 mu) the
//       momentum balance is (u'/u)' = -K/u, whose first integral
//       (u'/u)^2/2 = K/u + C gives, for C = -2c^2 < 0,
//       u = (K/2c^2) sin^2(c(z - z0)); N(L) = 0 puts the crest at the
//       collector, c(L - z0) = pi/2, so u0 = (K/2c^2) cos^2(cL) and the landing
//       speed is K/2c^2. Run at K = 2, c = 1.2. This is the electrified jet in a
//       uniform field (f = I E/Q, see the header) and the case whose landing
//       speed estimates the critical translation speed.
//   J3. THE SAME PULL, DRAWN: C = +2c^2 > 0 gives u = (K/2c^2) sinh^2(c(z - z0));
//       K = 2, c = 1, z0 = -0.2, a draw of 56.
//   J4. A VISCOSITY RISING 55-FOLD ALONG THE JET, mu = mu0 e^{z/l} with l = L/4
//       (what a cooling jet does), no force, drawn 20-fold: N is constant again
//       and ln u follows the integral of 1/mu,
//       u = exp[ln 20 (1 - e^{-z/l})/(1 - e^{-L/l})].
//   J5. COOLING AT A CONSTANT SPEED (u0 = uL, no force, so N = 0 and the radius
//       is fixed): T = T_air + (T0 - T_air) e^{-2 pi a h z/(rho c_p Q)}, with h
//       chosen for three e-folds over L.
//   J6. INERTIA, no force: N - rho Q u is constant, so u' = u(alpha + beta u)
//       with beta = rho/(3 mu), a logistic: u = 1/(-beta/alpha +
//       (1/u0 + beta/alpha) e^{-alpha z}). rho = 1, alpha = 1, L = 1/2: the
//       speed rises 4.7-fold.
//   J7. SURFACE TENSION, no force: N = 3 mu Q u'/u + pi gamma a is constant, and
//       s = sqrt(u) obeys the LINEAR s' = (N s - gamma sqrt(pi Q))/(6 mu Q), so
//       s = c/N + (s0 - c/N) e^{N z/(6 mu Q)} with c = gamma sqrt(pi Q). Run at
//       c = 1, N = 2: s = (1 + e^z)/2.
//   J8. EVERYTHING AT ONCE -- inertia, surface tension, the pull, a viscosity
//       rising 7-fold as the jet cools, and a heat-transfer coefficient rising as
//       the jet thins (h ~ 1/a, a constant Nusselt number) -- landing freely. No
//       closed form; the order is read from the landing speed on doubled grids.
//   J9. THE DRAGGED JET (solve_dragged), a viscous catenary: uniform pull f = 2,
//       no inertia, surface tension or cooling. In the vertical tension V the
//       stretching is d(1/u)/dV = P/(3 mu f Q^2), P = sqrt(H^2 + V^2), so
//       1/u = 1/u0 - k (G(V0) - G(V)) with G = (V P + H^2 asinh(V/H))/2, and the
//       landing point is the integral of (V/P, H/P) u/(f Q) over V -- done by
//       Simpson's rule on 2e5 intervals, i.e. to round-off. H = 0.5, V0 = 2,
//       u0 = 0.5. Checked twice: the path integrated from the start, and the
//       boundary-value problem (land at that depth at that speed) recovering H,
//       V0 and the lag. And at H -> 0 it must be J2's vertical free landing --
//       the check that found a sign flip a rounding of V to -1e-16 caused there
//       (see drhs).
//  CONTROLS, which the checks must catch: J6 run without its inertia, and J7
//  without its surface tension, each against its own exact solution.
//
//  MEASURED (2026-11-07), copied from the final run. FP64 in both builds: the
//  model is host-side and always double, and the FP32 build's output is
//  byte-identical. The error at n = 200 is the larger of the profile's and the
//  starting tension's, the order is between n = 100 and 200:
//   J1 2.3e-9, order 3.97 (the profile is exact at the nodes, 8.5e-16: all of
//      the error is in N0)
//   J2 1.0e-8, 3.97; the landing speed 0.6944444433 against 0.6944444444
//   J3 6.5e-8, 3.95    J4 3.6e-8, 3.94    J5 1.3e-9, 4.02
//   J6 2.2e-10, 4.00   J7 1.9e-11, 3.99
//   J8 the landing speed 0.312777144462 at n = 800, orders 3.86 / 3.95 / 3.98 /
//      3.99 on the doubled grids; its temperature, exact because h ~ 1/a makes
//      the loss per length independent of the radius, to 1.6e-9 of e^{-pi z}.
//   J9 the path from the start 6.8e-11 at n = 200, order 4.00; the
//      boundary-value problem recovers H to 1.8e-10, V0 to 5.7e-12 and the lag
//      to 3.1e-11 (orders 4.01, 4.00); at H -> 0 the free landing 0.6944444447
//      against J2's exact 0.6944444444. Before the fix in drhs it read 0.6952,
//      the landing height of that vertical jet converging at FIRST order.
//   Controls: J6 without its inertia misses by 22 %, J7 without its surface
//   tension by 6.0 %.
//  The gates are the scheme's order, 4 +- 0.3, and a loose 1e-6 at n = 200. The
//  first run's gates were per-section guesses and one failed (J4: 3.6e-8 against
//  1e-8), so those are now regression guards, not derivations. Runs in 0.2 s.
//
//  Usage: jet1d (no options)
//==============================================================================
#include "Jet1D.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>

using namespace jet1d;

static int g_fail = 0;

static void check(const char* what, double got, double want, double tol) {
  const bool ok = std::fabs(got - want) <= tol;
  if (!ok) ++g_fail;
  std::printf("  %-46s %12.4e  (want %g +- %g)  %s\n", what, got, want, tol, ok ? "PASS" : "FAIL");
}

static double max_rel(const std::vector<double>& z, const std::vector<double>& v,
                      const std::function<double(double)>& exact) {
  double e = 0.0;
  for (std::size_t i = 0; i < z.size(); ++i) e = std::max(e, std::fabs(v[i] - exact(z[i])) / std::fabs(exact(z[i])));
  return e;
}

static Jet base() {
  Jet J;
  J.Q = 1.0;
  J.L = 1.0;
  J.mu = [](double, double) { return 1.0 / 3.0; };
  return J;
}

// One exact-solution section on four grids. The error is the larger of the
// profile's (u, or T for J5) and the starting tension's: for a linear ODE (J1)
// the shooting makes the profile exact AT THE NODES -- the step's growth factor
// is fitted so that n of them reach u_L -- and the whole discretisation error
// sits in N0. The gates are the scheme's order, 4 +- 0.3 between the two finest
// grids, and a loose absolute bound (1e-6 at n = 200) that the first run set:
// every section measured below 7e-8 there.
static void ladder(const char* name, const std::function<Solution(int)>& run,
                   const std::function<double(double)>& exact, bool temperature, double N0_exact) {
  double prev = 0.0;
  const int ns[] = {25, 50, 100, 200};
  double order = 0.0;
  for (int n : ns) {
    const Solution s = run(n);
    if (!s.ok) {
      std::printf("  %s n = %d: the shooting failed\n", name, n);
      ++g_fail;
      return;
    }
    const double eu = max_rel(s.p.z, temperature ? s.p.T : s.p.u, exact);
    const double eN = std::fabs(s.N0 - N0_exact) / std::max(1.0, std::fabs(N0_exact));
    const double e = std::max(eu, eN);
    std::printf("  %s n = %3d: profile %.3e, N0 %.3e", name, n, eu, eN);
    if (prev > 0.0) {
      order = std::log(prev / e) / std::log(2.0);
      std::printf(", order %.2f", order);
    }
    std::printf("\n");
    prev = e;
  }
  char w[96];
  std::snprintf(w, sizeof w, "%s: error at n = 200", name);
  check(w, prev, 0.0, 1e-6);
  std::snprintf(w, sizeof w, "%s: order from n = 100 to 200", name);
  check(w, order, 4.0, 0.3);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    for (int i = 1; i < argc; ++i) {
      std::fprintf(stderr, "jet1d: unknown option %s (it takes none)\n", argv[i]);
      Kokkos::finalize();
      return 2;
    }
    std::printf("jet1d: the slender viscous jet (validation/Jet1D.hpp) against exact solutions; FP64\n");
    const double nan = std::nan("");

    std::printf("\nJ1. stretching alone, drawn 100-fold: u = 100^z\n");
    {
      const Jet J = base();
      ladder("J1", [&](int n) { return solve_drawn(J, 1.0, 0.0, 100.0, n); },
             [](double z) { return std::pow(100.0, z); }, false, std::log(100.0));
    }

    std::printf("\nJ2. uniform pull, free landing: u = (K/2c^2) sin^2(c(z - z0)), K = 2, c = 1.2\n");
    {
      const double K = 2.0, c = 1.2, A = K / (2 * c * c), z0 = 1.0 - kPi / (2 * c);
      Jet J = base();
      J.f = [K](double, double) { return K; };  // f = 3 mu K, and 3 mu = 1
      const double u0 = A * std::cos(c) * std::cos(c);
      ladder("J2", [&](int n) { return solve_free(J, u0, 0.0, n); },
             [&](double z) { return A * std::pow(std::sin(c * (z - z0)), 2); }, false, 2 * c * std::tan(c));
      const Solution s = solve_free(J, u0, 0.0, 200);
      std::printf("  landing speed %.10f against K/2c^2 = %.10f\n", s.p.u.back(), A);
      check("J2: landing speed / (K/2c^2) - 1, n = 200", s.p.u.back() / A - 1.0, 0.0, 1e-8);
    }

    std::printf("\nJ3. uniform pull, drawn: u = (K/2c^2) sinh^2(c(z - z0)), K = 2, c = 1, z0 = -0.2\n");
    {
      const double K = 2.0, c = 1.0, A = K / (2 * c * c), z0 = -0.2;
      Jet J = base();
      J.f = [K](double, double) { return K; };
      const double u0 = A * std::pow(std::sinh(-c * z0), 2), uL = A * std::pow(std::sinh(c * (1.0 - z0)), 2);
      ladder("J3", [&](int n) { return solve_drawn(J, u0, 0.0, uL, n); },
             [&](double z) { return A * std::pow(std::sinh(c * (z - z0)), 2); }, false,
             2 * c / std::tanh(-c * z0));
    }

    std::printf("\nJ4. viscosity rising 55-fold along the jet, drawn 20-fold\n");
    {
      const double l = 0.25;
      Jet J = base();
      J.mu = [l](double z, double) { return std::exp(z / l) / 3.0; };
      const double D = std::log(20.0), den = 1.0 - std::exp(-1.0 / l);
      ladder("J4", [&](int n) { return solve_drawn(J, 1.0, 0.0, 20.0, n); },
             [&](double z) { return std::exp(D * (1.0 - std::exp(-z / l)) / den); }, false, D / (l * den));
    }

    std::printf("\nJ5. cooling at a constant speed: T = e^{-3z}\n");
    {
      Jet J = base();
      J.rho_cp = 1.0;
      J.T_air = 0.0;
      const double a = std::sqrt(1.0 / kPi), H = 3.0 / (2.0 * kPi * a);
      J.h = [H](double, double, double) { return H; };
      ladder("J5", [&](int n) { return solve_drawn(J, 1.0, 1.0, 1.0, n); },
             [](double z) { return std::exp(-3.0 * z); }, true, 0.0);
    }

    std::printf("\nJ6. inertia, rho = 1: u = 1/(-1 + 2 e^{-z}) on L = 1/2\n");
    auto logistic = [](double z) { return 1.0 / (-1.0 + 2.0 * std::exp(-z)); };
    {
      Jet J = base();
      J.L = 0.5;
      J.rho = 1.0;
      ladder("J6", [&](int n) { return solve_drawn(J, 1.0, 0.0, logistic(0.5), n); }, logistic, false, 2.0);
    }

    std::printf("\nJ7. surface tension, gamma sqrt(pi Q) = 1: u = ((1 + e^z)/2)^2\n");
    auto capillary = [](double z) { return std::pow(0.5 * (1.0 + std::exp(z)), 2); };
    {
      Jet J = base();
      J.gamma = 1.0 / std::sqrt(kPi);
      ladder("J7", [&](int n) { return solve_drawn(J, 1.0, 0.0, capillary(1.0), n); }, capillary, false, 2.0);
    }

    std::printf("\nJ8. everything at once, landing freely: the landing speed on doubled grids\n");
    {
      Jet J = base();
      J.rho = 0.5;
      J.gamma = 0.2;
      J.rho_cp = 1.0;
      J.T_air = 0.0;
      J.mu = [](double, double T) { return std::exp(2.0 * (1.0 - T)) / 3.0; };
      J.f = [](double, double) { return 2.0; };
      J.h = [](double a, double, double) { return 0.5 / a; };
      double U[6], N0[6];
      const int ns[] = {25, 50, 100, 200, 400, 800};
      for (int k = 0; k < 6; ++k) {
        const Solution s = solve_free(J, 0.1, 1.0, ns[k]);
        if (!s.ok) {
          std::printf("  J8 n = %d: the shooting failed\n", ns[k]);
          ++g_fail;
          U[k] = N0[k] = nan;
          continue;
        }
        U[k] = s.p.u.back();
        N0[k] = s.N0;
        std::printf("  J8 n = %3d: landing speed %.12f, N0 %.12f, T at landing %.6f, mu x %.2f", ns[k], U[k], N0[k],
                    s.p.T.back(), std::exp(2.0 * (1.0 - s.p.T.back())));
        if (k >= 2) std::printf(", order %.2f", std::log(std::fabs(U[k - 2] - U[k - 1]) / std::fabs(U[k - 1] - U[k])) / std::log(2.0));
        std::printf("\n");
      }
      const double p = std::log(std::fabs(U[2] - U[3]) / std::fabs(U[3] - U[4])) / std::log(2.0);
      check("J8: order of the landing speed, n = 100/200/400", p, 4.0, 0.3);
      check("J8: |U(200) - U(800)| / U", std::fabs(U[3] - U[5]) / U[5], 0.0, 1e-7);
      // h = Nu-like/a makes the loss per length, 2 pi a h, independent of the
      // radius, so T is EXACT however the jet thins: e^{-pi z} here.
      const Solution s = solve_free(J, 0.1, 1.0, 200);
      const double eT = s.ok ? max_rel(s.p.z, s.p.T, [](double z) { return std::exp(-kPi * z); }) : 1.0;
      std::printf("  J8 n = 200: temperature against e^{-pi z}, max rel. error %.3e\n", eT);
      check("J8: temperature, n = 200 (exact: e^{-pi z})", eT, 0.0, 1e-8);
    }

    std::printf("\nJ9. the dragged jet, a viscous catenary: uniform pull f = 2, H = 0.5, V0 = 2, u0 = 0.5\n");
    {
      // Exact: d(1/u)/dV = P/(3 mu f Q^2) with P = sqrt(H^2 + V^2), so
      // 1/u(V) = 1/u0 - k (G(V0) - G(V)), G(V) = (V P + H^2 asinh(V/H))/2, k = 1/(3 mu f Q^2);
      // z and x at the landing are the integrals of (V/P, H/P) u/(f Q) over V in [0, V0],
      // done by composite Simpson on 2e5 intervals (round-off, not truncation).
      const double H = 0.5, V0 = 2.0, u0 = 0.5, f = 2.0, k = 0.5;
      auto G = [H](double v) { return 0.5 * (v * std::sqrt(H * H + v * v) + H * H * std::asinh(v / H)); };
      auto u = [&](double v) { return 1.0 / (1.0 / u0 - k * (G(V0) - G(v))); };
      double zr = 0.0, xr = 0.0;
      const int M = 200000;
      for (int i = 0; i <= M; ++i) {
        const double v = V0 * i / M, w = (i == 0 || i == M) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        const double P = std::sqrt(H * H + v * v);
        zr += w * (v / P) * u(v) / f;
        xr += w * (H / P) * u(v) / f;
      }
      zr *= V0 / M / 3.0;
      xr *= V0 / M / 3.0;
      const double ur = u(0.0);
      std::printf("  exact: lands %.12f below and %.12f behind the start, at u = %.12f\n", zr, xr, ur);
      Jet J = base();
      J.L = zr;
      J.f = [f](double, double) { return f; };
      double prev = 0.0, order = 0.0;
      for (int n : {25, 50, 100, 200}) {
        detail::Y e{};
        detail::integrate_V(J, H, V0, u0, 0.0, n, e);
        const double err = std::max({std::fabs(e.z - zr) / zr, std::fabs(e.x - xr) / xr, std::fabs(e.u - ur) / ur});
        std::printf("  J9 n = %3d, integrated from the start: z %.3e, x %.3e, u %.3e", n, std::fabs(e.z - zr) / zr,
                    std::fabs(e.x - xr) / xr, std::fabs(e.u - ur) / ur);
        if (prev > 0.0) {
          order = std::log(prev / err) / std::log(2.0);
          std::printf(", order %.2f", order);
        }
        std::printf("\n");
        prev = err;
      }
      check("J9: the path, max rel. error at n = 200", prev, 0.0, 1e-6);
      check("J9: the path, order from n = 100 to 200", order, 4.0, 0.3);
      prev = 0.0;
      for (int n : {50, 100, 200}) {
        const Dragged d = solve_dragged(J, u0, 0.0, ur, n);
        if (!d.ok) {
          std::printf("  J9 n = %d: no dragged solution\n", n);
          ++g_fail;
          break;
        }
        const double err = std::max({std::fabs(d.H - H) / H, std::fabs(d.V0 - V0) / V0, std::fabs(d.lag - xr) / xr});
        std::printf("  J9 n = %3d, solved as the boundary-value problem: H %.3e, V0 %.3e, lag %.3e, start angle %.6f"
                    " (exact %.6f) rad, free landing %.6f", n, std::fabs(d.H - H) / H, std::fabs(d.V0 - V0) / V0,
                    std::fabs(d.lag - xr) / xr, d.theta0, std::atan2(H, V0), d.U_free);
        if (prev > 0.0) std::printf(", order %.2f", std::log(prev / err) / std::log(2.0));
        std::printf("\n");
        if (n == 200) check("J9: H, V0 and the lag recovered, n = 200", err, 0.0, 1e-6);
        prev = err;
      }
      // At H = 0 the dragged jet is J2's vertical one: with no inertia, P = 0 is N = 0.
      Jet J2 = base();
      const double K = 2.0, c = 1.2, A = K / (2 * c * c);
      J2.f = [K](double, double) { return K; };
      const Dragged d2 = solve_dragged(J2, A * std::cos(c) * std::cos(c), 0.0, 10.0 * A, 200);
      std::printf("  J9 at H -> 0: the free landing %.10f against J2's exact %.10f\n", d2.U_free, A);
      check("J9: free landing at H = 0 / J2's - 1, n = 200", d2.U_free / A - 1.0, 0.0, 1e-7);
      // control: the same landing without the drag (a vertical jet, H = 0) is not the dragged one
      std::printf("  control: without the drag the jet would land at %.6f, not %.6f\n", solve_dragged(J, u0, 0.0, ur, 200).U_free, ur);
    }

    std::printf("\nCONTROLS: the checks must see a term that is missing\n");
    {
      Jet J = base();
      J.L = 0.5;  // J6 without its inertia
      const Solution s = solve_drawn(J, 1.0, 0.0, logistic(0.5), 200);
      const double e = s.ok ? max_rel(s.p.z, s.p.u, logistic) : 1.0;
      std::printf("  J6 without inertia: max rel. error %.3e\n", e);
      check("control: J6 without inertia misses by > 1e-3", e > 1e-3 ? 1.0 : 0.0, 1.0, 0.0);
    }
    {
      const Jet J = base();  // J7 without its surface tension
      const Solution s = solve_drawn(J, 1.0, 0.0, capillary(1.0), 200);
      const double e = s.ok ? max_rel(s.p.z, s.p.u, capillary) : 1.0;
      std::printf("  J7 without surface tension: max rel. error %.3e\n", e);
      check("control: J7 without surface tension misses by > 1e-3", e > 1e-3 ? 1.0 : 0.0, 1.0, 0.0);
    }

    std::printf("\njet1d: %s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    rc = g_fail ? 1 : 0;
  }
  Kokkos::finalize();
  return rc;
}
