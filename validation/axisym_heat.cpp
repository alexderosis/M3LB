//==============================================================================
//  THE r-z HEAT MODULE (validation/FsHeatAxi.hpp) against exact answers, on
//  prescribed fields -- no lattice Boltzmann, so a failure here is the heat
//  module's and nothing else's.
//
//   H1. CONDUCTION IN A CYLINDER, wall at r = R held at 0, start at 1. The
//       slowest mode decays at lambda_1^2 alpha / R^2, lambda_1 = 2.404826
//       (J0's first zero); the rate is fitted to T on the axis over
//       alpha t / R^2 = 0.5 .. 1.0, where the next mode is below 1e-6 of it.
//       Checks the r-weighted faces, the axis and the half-cell wall.
//   H2. NEWTON COOLING THROUGH A FREE SURFACE: a liquid cylinder whose last
//       column is an interface cell of fill f, in gas at T_air = 0. Exactly,
//       T decays at lambda^2 alpha / R_s^2 with lambda J1(lambda) / J0(lambda)
//       = Bi = h R_s / k, R_s = R + f the surface. At Bi = 0.02 that is the
//       lumped 2 h / (rho c_p R_s) to 0.5 %. Checks the surface term and its
//       radius, at f = 1/2 and f = 1/4.
//   H4. A LIQUID SPHERE ON THE AXIS COOLING THE SAME WAY, every slope at once:
//       1 - lambda cot(lambda) = Bi, rate lambda^2 alpha / R^2.
//   H3. STEADY ADVECTION-DIFFUSION between a hot wall upstream and a cold one
//       downstream, flow U towards the cold wall at Pe = U L / alpha = 4:
//       T = (1 - exp(-Pe s / L)) / (1 - exp(-Pe)), s from the cold wall.
//       Upwind: FIRST order, by construction.
//
//  MEASURED (FP64, 2026-11-06, under a second), copied from the final run:
//   H1  0.998824 / 0.999708 / 0.999926 at R = 8 / 16 / 32, order 2.01 / 1.99.
//   H2  f = 1/2: 0.99465 / 0.99799 / 0.99918;  f = 1/4: 0.99013 / 0.99548 /
//       0.99785 -- first order in 1/R, from the surface's discrete volume. The
//       area is the module's sum |grad phi| r, exactly R + f here.
//   H4  1.0107 / 1.0123 / 1.0104 at R = 8 / 16 / 32: that sum's bias on a
//       curved surface, flat in R. THIS SECTION CAUGHT TWO THINGS: the first
//       version lost the loss of every surface cell touching gas only at a
//       corner (0.74-0.76 of a sphere's area), and its successor, per-cell
//       PLIC segments, summed to 0.85-0.93 of it and erratically -- see
//       FsHeatAxi::surface_area. (With 8 x 8 sampling of the fill instead of
//       32 x 32 the R = 32 row read 0.981: the fill, not the module.)
//   H3  2.66e-2 / 1.45e-2 / 7.55e-3 at L = 16 / 32 / 64, order 0.88 / 0.94.
//  Gates: H1 1/R^2, H2 0.12/R, H4 2 %, H3 1/L -- regression guards set from these.
//==============================================================================
#include "core/Types.hpp"
#include "FsHeatAxi.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace lbm;

namespace {

int g_fail = 0;

void check(const char* what, double got, double want, double tol) {
  const bool ok = std::isfinite(got) && std::fabs(got - want) <= tol;
  if (!ok) ++g_fail;
  std::printf("  %-52s %12.6g  (want %g +- %g)  %s\n", what, got, want, tol, ok ? "PASS" : "FAIL");
}

double besselj(int nn, double x) {   // (1/pi) int_0^pi cos(n t - x sin t) dt, Simpson
  const int N = 400;
  const double pi = 3.14159265358979323846, h = pi / N;
  double s = 0;
  for (int i = 0; i <= N; ++i) {
    const double t = i * h, w = (i == 0 || i == N) ? 1.0 : (i % 2 ? 4.0 : 2.0);
    s += w * std::cos(nn * t - x * std::sin(t));
  }
  return s * h / 3.0 / pi;
}

HeatFields blank(const Domain& d) {
  HeatFields F;
  const std::size_t N = std::size_t(d.n_padded);
  F.flags.assign(N, FsGas); F.fill.assign(N, 0.0); F.ux.assign(N, 0.0); F.uy.assign(N, 0.0);
  return F;
}

// H1: the slowest mode's rate / (lambda_1^2 alpha / R^2)
double h1(Index R) {
  const Index nx = R + 1, ny = 3;
  Domain d(nx, ny, 1, false, false, true);
  HeatFields F = blank(d);
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      const Index n = d.id(x, y);
      if (x < R) { F.flags[n] = FsFluid; F.fill[n] = 1.0; }
      else F.flags[n] = FsSolid;
    }
  FsHeatAxi H(d);
  H.alpha = 0.1;
  H.set_roles([&](Index x, Index) { return x >= R ? HeatWall : HeatFree; }, [](Index, Index) { return 0.0; });
  H.initialize(F, [](Index, Index) { return 1.0; });
  const double tR = double(R * R) / H.alpha;   // alpha t / R^2 = 1
  const int chunk = 10;
  double t = 0, t1 = -1, T1 = 0, t2 = -1, T2 = 0;
  while (t < 1.0 * tR + 0.5) {
    H.update(F, chunk);
    t += chunk;
    if (t1 < 0 && t >= 0.5 * tR) { t1 = t; T1 = H.T(0, 1); }
  }
  t2 = t; T2 = H.T(0, 1);
  const double rate = std::log(T1 / T2) / (t2 - t1);
  const double l1 = 2.404825557695773;
  return rate / (l1 * l1 * H.alpha / double(R * R));
}

// H2: the cooling rate / the exact one
double h2(Index R, double f, double& lumped_ratio) {
  const Index nx = R + 4, ny = 3;
  Domain d(nx, ny, 1, false, false, true);
  HeatFields F = blank(d);
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      const Index n = d.id(x, y);
      if (x < R) { F.flags[n] = FsFluid; F.fill[n] = 1.0; }
      else if (x == R) { F.flags[n] = FsInterface; F.fill[n] = f; }
    }
  FsHeatAxi H(d);
  H.alpha = 0.1;
  const double Rs = double(R) + f, Bi = 0.02;
  H.hloss = Bi * H.alpha / Rs;          // Bi = h R_s / k = hloss R_s / alpha in lattice units
  H.T_air = 0.0;
  H.set_roles([](Index, Index) { return HeatFree; }, [](Index, Index) { return 0.0; });
  H.initialize(F, [](Index, Index) { return 1.0; });
  // the exact rate: lambda J1 / J0 = Bi, by bisection on (0, 2.4048)
  double lo = 1e-9, hi = 2.4;
  for (int k = 0; k < 200; ++k) {
    const double m = 0.5 * (lo + hi);
    (m * besselj(1, m) / besselj(0, m) < Bi ? lo : hi) = m;
  }
  const double lam = 0.5 * (lo + hi);
  const double exact = lam * lam * H.alpha / (Rs * Rs);
  const double tend = 0.5 / exact;      // T falls to e^-1/2
  const int chunk = 10;
  double t = 0, t1 = -1, h1v = 0, v1 = 0;
  while (t < tend) {
    H.update(F, chunk);
    t += chunk;
    if (t1 < 0 && t >= 0.25 * tend) { t1 = t; H.totals(h1v, v1); }
  }
  double h2v = 0, v2 = 0;
  H.totals(h2v, v2);
  const double rate = std::log((h1v / v1) / (h2v / v2)) / (t - t1);
  lumped_ratio = rate / (2.0 * H.hloss * Rs / (Rs * Rs));
  return rate / exact;
}

// H4: a liquid SPHERE on the axis cooling through its surface, every slope at
// once: exactly 1 - lambda cot(lambda) = Bi, rate lambda^2 alpha / R^2; the
// lumped limit 3 h / (rho c_p R). Fill by 32 x 32 subsampling.
double h4(Index R, double& lumped_ratio) {
  const Index nx = R + 4, ny = 2 * R + 8;
  Domain d(nx, ny, 1, false, false, true);
  HeatFields F = blank(d);
  const double zc = 0.5 * double(ny), Rr = double(R);
  double vol = 0;
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      const Index n = d.id(x, y);
      double c = 0;
      constexpr int S = 32;
      for (int a = 0; a < S; ++a)
        for (int b = 0; b < S; ++b) {
          const double r = double(x) + (a + 0.5) / S, z = double(y) + (b + 0.5) / S - zc;
          // r-weighted subsample: the cell's volume per radian is r dr dz
          c += (std::hypot(r, z) < Rr ? 1.0 : 0.0) * r;
        }
      const double rc = double(x) + 0.5;
      c /= double(S * S) * rc;
      if (c >= 1.0 - 1e-12) { F.flags[n] = FsFluid; F.fill[n] = 1.0; }
      else if (c > 1e-12) { F.flags[n] = FsInterface; F.fill[n] = c; }
      vol += c * rc;
    }
  FsHeatAxi H(d);
  H.alpha = 0.1;
  const double Bi = 0.02;
  H.hloss = Bi * H.alpha / Rr;
  H.T_air = 0.0;
  H.set_roles([](Index, Index) { return HeatFree; }, [](Index, Index) { return 0.0; });
  H.initialize(F, [](Index, Index) { return 1.0; });
  double lo = 1e-9, hi = 3.1;              // 1 - lam cot lam = Bi on (0, pi)
  for (int k = 0; k < 200; ++k) {
    const double m = 0.5 * (lo + hi);
    ((1.0 - m / std::tan(m)) < Bi ? lo : hi) = m;
  }
  const double lam = 0.5 * (lo + hi);
  const double exact = lam * lam * H.alpha / (Rr * Rr);
  const double tend = 0.5 / exact;
  const int chunk = 10;
  double t = 0, t1 = -1, h1v = 0, v1 = 0;
  while (t < tend) {
    H.update(F, chunk);
    t += chunk;
    if (t1 < 0 && t >= 0.25 * tend) { t1 = t; H.totals(h1v, v1); }
  }
  double h2v = 0, v2 = 0;
  H.totals(h2v, v2);
  const double rate = std::log((h1v / v1) / (h2v / v2)) / (t - t1);
  lumped_ratio = rate / (3.0 * H.hloss / Rr);
  (void)vol;
  return rate / exact;
}

// H3: the L2 error against the exact steady profile, L cells between the walls
double h3(Index L) {
  const Index nx = 3, ny = L + 2;
  Domain d(nx, ny, 1, false, false, true);
  HeatFields F = blank(d);
  FsHeatAxi H(d);
  H.alpha = 0.1;
  const double Pe = 4.0, U = Pe * H.alpha / double(L);
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      const Index n = d.id(x, y);
      if (y == 0 || y == ny - 1) { F.flags[n] = FsSolid; continue; }
      F.flags[n] = FsFluid; F.fill[n] = 1.0; F.uy[n] = -U;
    }
  H.set_roles([&](Index, Index y) { return (y == 0 || y == ny - 1) ? HeatWall : HeatFree; },
              [&](Index, Index y) { return y == ny - 1 ? 1.0 : 0.0; });
  H.initialize(F, [](Index, Index) { return 0.5; });
  const double tend = 20.0 * double(L * L) / H.alpha;
  for (double t = 0; t < tend; t += 50) H.update(F, 50);
  double e2 = 0, n2 = 0;
  for (Index y = 1; y < ny - 1; ++y) {
    const double s = double(y) - 0.5;   // the cold wall's face is at y = 1/2
    const double Te = (1.0 - std::exp(-Pe * s / double(L))) / (1.0 - std::exp(-Pe));
    const double T = H.T(1, y);
    e2 += (T - Te) * (T - Te); n2 += Te * Te;
  }
  return std::sqrt(e2 / n2);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    for (int i = 1; i < argc; ++i) {
      std::fprintf(stderr, "axisym_heat: unknown option %s (it takes none)\n", argv[i]);
      Kokkos::finalize();
      return 2;
    }
    std::printf("axisym_heat: the r-z heat module on prescribed fields, %s\n", sizeof(Real) == 8 ? "FP64" : "FP32");
    char w[96];

    std::printf("\nH1. conduction in a cylinder, wall at 0: rate / (lambda_1^2 alpha / R^2)\n");
    double prev = 0;
    for (Index R : {8, 16, 32}) {
      const double q = h1(R);
      std::printf("  R = %2d: %.6f", int(R), q);
      if (prev != 0) std::printf("  order %.2f", std::log(std::fabs(prev - 1) / std::fabs(q - 1)) / std::log(2.0));
      std::printf("\n");
      prev = q;
      std::snprintf(w, sizeof w, "R = %2d: rate ratio", int(R));
      check(w, q, 1.0, 1.0 / double(R * R));
    }

    std::printf("\nH2. Newton cooling through a free surface at Bi = 0.02: rate / exact\n");
    for (double f : {0.5, 0.25}) {
      for (Index R : {8, 16, 32}) {
        double lumped = 0;
        const double q = h2(R, f, lumped);
        std::printf("  f = %.2f, R = %2d: %.6f  (lumped 2h/R_s: %.6f)\n", f, int(R), q, lumped);
        std::snprintf(w, sizeof w, "f = %.2f, R = %2d: rate / exact", f, int(R));
        check(w, q, 1.0, 0.12 / double(R));
      }
    }

    std::printf("\nH4. a liquid sphere cooling through its surface at Bi = 0.02: rate / exact\n");
    for (Index R : {8, 16, 32}) {
      double lumped = 0;
      const double q = h4(R, lumped);
      std::printf("  R = %2d: %.6f  (lumped 3h/R: %.6f)\n", int(R), q, lumped);
      std::snprintf(w, sizeof w, "R = %2d: rate / exact", int(R));
      check(w, q, 1.0, 0.02);
    }

    std::printf("\nH3. steady advection-diffusion, Pe = 4, upwind: relative L2 error\n");
    prev = 0;
    for (Index L : {16, 32, 64}) {
      const double e = h3(L);
      std::printf("  L = %2d: %.4e", int(L), e);
      if (prev > 0) std::printf("  order %.2f", std::log(prev / e) / std::log(2.0));
      std::printf("\n");
      prev = e;
      std::snprintf(w, sizeof w, "L = %2d: relative L2 error", int(L));
      check(w, e, 0.0, 1.0 / double(L));
    }

    std::printf("\naxisym_heat: %s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    rc = g_fail ? 1 : 0;
  }
  Kokkos::finalize();
  return rc;
}
