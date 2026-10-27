#pragma once
//==============================================================================
//  FULLY DEVELOPED LAMINAR FLOW IN A SQUARE DUCT -- the exact series solution.
//
//  For a duct -a < y < a, -b < z < b driven by G = -dp/dx (White, Viscous Fluid
//  Flow, the rectangular-duct series),
//
//      u(y,z) = (16 a^2 G / mu pi^3) sum_{i odd} (-1)^((i-1)/2)
//               [1 - cosh(i pi z/2a) / cosh(i pi b/2a)] cos(i pi y/2a) / i^3,
//
//      Q = (4 b a^3 G / 3 mu) [1 - (192 a / pi^5 b) sum_{i odd} tanh(i pi b/2a) / i^5].
//
//  Only the SQUARE duct is needed (a = b = L/2), and only that is implemented.
//  Dividing the two gives the profile normalised by its own mean,
//
//      u / Ubar = (4K / pi^3) S(y,z),   S = the sum in u(y,z),
//      G = K mu Ubar / L^2,             K = 12 / [1 - (192/pi^5) sum tanh(i pi/2)/i^5].
//
//  The mixer uses it twice: as the inlet profile, so the flow enters fully
//  developed, and as the exact answer of the empty-duct check in
//  validation/mixer_forward.cpp.
//
//  CHECKED AGAINST A PUBLISHED VALUE, NOT ONLY AGAINST ITSELF. A typo in a
//  series still gives a smooth, consistent profile. duct_self_check() therefore
//  holds K to the square-duct Darcy friction factor f Re = 2K = 56.908 on the
//  hydraulic diameter L (Shah & London's value, as quoted in arXiv:2502.07678,
//  checked 2026-10-15), and then holds the VELOCITY sum to K: the profile must
//  integrate to exactly its own mean and vanish on the wall. The first check
//  never evaluates S, so it cannot catch a typo there; the second can.
//  The often-quoted peak u_max/Ubar = 2.096 was NOT found in a checked source,
//  so it is printed by the anchors as a computed number and asserted nowhere.
//
//  The cosh ratio is evaluated as exp(k (|z| - a)) (1 + e^{-2k|z|}) /
//  (1 + e^{-2ka}) with k = i pi / 2a, because cosh itself overflows a double
//  for i > 450 and the series wants several hundred terms near the walls.
//==============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace lbm {
namespace topopt {

// The series' common factor in G = K mu Ubar / L^2 for a square duct of side L.
// The tanh sum converges as 1/i^5; 2000 odd terms leave a tail below 1e-14.
inline double duct_K() {
  const double pi = 3.14159265358979323846;
  double s = 0.0;
  for (int i = 1; i < 4000; i += 2) s += std::tanh(i * pi / 2.0) / std::pow(double(i), 5);
  return 12.0 / (1.0 - 192.0 / std::pow(pi, 5) * s);
}

// u / Ubar at (y, z), both measured in units of L from one wall, i.e. in [0, 1].
// The sum converges as 1/i^3 (the cos factor does not decay); 600 odd terms
// leave a tail below 1e-6 of the mean, far below anything a lattice of a few
// hundred cells resolves.
inline double duct_u_over_mean(double y, double z, double K) {
  const double pi = 3.14159265358979323846;
  const double a = 0.5;                      // half side, in units of L
  const double yc = y - a, zc = std::fabs(z - a);
  double S = 0.0;
  for (int i = 1; i < 1200; i += 2) {
    const double k = i * pi / (2.0 * a);     // i pi / 2a
    const double ratio = std::exp(k * (zc - a)) * (1.0 + std::exp(-2.0 * k * zc)) /
                         (1.0 + std::exp(-2.0 * k * a));
    const double sgn = ((i - 1) / 2) % 2 == 0 ? 1.0 : -1.0;
    S += sgn * (1.0 - ratio) * std::cos(k * yc) / (double(i) * i * i);
  }
  return 4.0 * K / (pi * pi * pi) * S;
}

// Midpoint quadrature of u/Ubar over the section, nq points a side.
inline double duct_mean_midpoint(int nq, double K) {
  double m = 0.0;
  for (int j = 0; j < nq; ++j)
    for (int k = 0; k < nq; ++k) m += duct_u_over_mean((j + 0.5) / nq, (k + 0.5) / nq, K);
  return m / (double(nq) * nq);
}

// Abort unless K matches the published friction factor and the velocity sum is
// consistent with K. Returns u_max / Ubar for the caller to print. Runs once per
// process; later calls return the cached value.
//
// THE MEAN IS RICHARDSON-EXTRAPOLATED, and the first version of this check is
// why. The midpoint rule alone is off by O(h^2): for a profile that is concave
// and vanishes on all four walls that is +1.2e-4 at 100 points a side -- an
// order of magnitude above what the first version estimated, so it aborted on a
// correct series (2026-10-15, before any gate of mixer_forward had been read).
// Combining 64 and 128 points a side removes the h^2 term; what is left is
// O(h^4 log h) from the corners, well below the 1e-6 asked of it.
// THE PROFILE'S SHAPE, against a second method. The first version also asked
// that u vanish on the wall, which every term of the series does by
// construction whatever its coefficient -- a check that could not fail. This
// solves the same problem a different way: del^2 v = -1 on the unit square,
// v = 0 on its edges, by cell-centred finite differences and SOR on n x n cells,
// and returns max |v/vbar - u/Ubar| over the cells together with K_fd = 1/vbar.
// Both carry the scheme's O(h^2) error, ~1e-3 at n = 64.
inline double duct_fd_check(int n, double K, double& K_fd) {
  const double h = 1.0 / n, pi = 3.14159265358979323846;
  const double w = 2.0 / (1.0 + std::sin(pi * h));             // optimal SOR
  std::vector<double> v(std::size_t(n) * n, 0.0);
  auto at = [&](int j, int k) -> double {                      // ghost = -interior
    const int jj = j < 0 ? 0 : (j >= n ? n - 1 : j), kk = k < 0 ? 0 : (k >= n ? n - 1 : k);
    const double s = (j < 0 || j >= n || k < 0 || k >= n) ? -1.0 : 1.0;
    return s * v[std::size_t(jj) * n + kk];
  };
  for (int it = 0; it < 20000; ++it) {
    double dmax = 0.0;
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const double g = 0.25 * (at(j + 1, k) + at(j - 1, k) + at(j, k + 1) + at(j, k - 1) + h * h);
        // A wall cell's ghost is minus itself, so its diagonal is 5, not 4.
        const int walls = (j == 0) + (j == n - 1) + (k == 0) + (k == n - 1);
        double& c = v[std::size_t(j) * n + k];
        const double gs = walls ? (4.0 * g + walls * c) / (4.0 + walls) : g;
        const double nv = c + w * (gs - c);
        dmax = std::max(dmax, std::fabs(nv - c));
        c = nv;
      }
    if (dmax < 1e-15) break;
  }
  double vbar = 0.0;
  for (double x : v) vbar += x;
  vbar /= double(n) * n;
  K_fd = 1.0 / vbar;
  double e = 0.0;
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < n; ++k)
      e = std::max(e, std::fabs(v[std::size_t(j) * n + k] / vbar -
                                duct_u_over_mean((j + 0.5) * h, (k + 0.5) * h, K)));
  return e;
}

inline double duct_self_check() {
  static const double umax = [] {
    const double K = duct_K();
    const double fRe = 2.0 * K;
    const double m1 = duct_mean_midpoint(64, K), m2 = duct_mean_midpoint(128, K);
    const double mean = (4.0 * m2 - m1) / 3.0;
    double K_fd = 0.0;
    const double shape = duct_fd_check(64, K, K_fd);
    const bool ok_f = std::fabs(fRe - 56.908) < 1e-3;
    const bool ok_m = std::fabs(mean - 1.0) < 1e-6;
    const bool ok_s = shape < 5e-3 && std::fabs(K_fd / K - 1.0) < 5e-3;
    if (!ok_f || !ok_m || !ok_s) {
      std::printf("duct_self_check: f Re = %.5f (published 56.908), mean of u/Ubar = %.9f "
                  "(must be 1; midpoint %.9f at 64, %.9f at 128), max |u/Ubar - FD| = %.3e, "
                  "K_fd/K - 1 = %.3e -- the series is wrong\n", fRe, mean, m1, m2, shape,
                  K_fd / K - 1.0);
      std::abort();
    }
    return duct_u_over_mean(0.5, 0.5, K);
  }();
  return umax;
}

}  // namespace topopt
}  // namespace lbm
