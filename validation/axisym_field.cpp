//==============================================================================
//  THE r-z ELECTRIC FIELD (validation/FsElectricAxi.hpp) against exact answers.
//
//   S. A CONDUCTING SPHERE IN A UNIFORM FIELD, on the axis, held at 0:
//          phi = -E0 z (1 - a^3 / rho^3),
//      the outer faces of the box held at that exact potential (on the face
//      half a cell inside each boundary cell, where FsElectricAxi puts an
//      electrode's value), so the box is not an error. The field at every gas
//      cell touching the sphere, against the exact field at that cell's
//      centre: relative L2 and the largest deviation. The conductor's surface
//      is FsElectric's staircase (fill >= 1/2), so this converges at first
//      order and no better; validation/fs_electric's 3-D sphere has the same
//      floor.
//   O. AN OHMIC COLUMN bridging two plates at 1 and 0, no gas between them:
//      the potential is exactly linear in z, inside the melt and out.
//   N. NO CURRENT BEFORE CONTACT: a drop hanging from the top plate, not
//      touching the bottom one. Ohmic and equipotential must give the same
//      field, because the ohmic melt touches one electrode only.
//
//   X. COAXIAL ELECTRODES, a rod of radius a at 1 on the axis and a shell at b
//      at 0, periodic in z: phi = ln(r/b) / ln(a/b). Both electrodes sit on
//      cell faces, so there is no staircase, and this is the r-weighted
//      operator alone.
//
//  MEASURED (FP64, 2026-11-06), copied from the run:
//   S  amplitude (slope of E^2 on the exact E^2) 0.983 / 0.973 / 0.985 at
//      a = 5 / 10 / 20, mean |E| error +0.0036 / -0.0033 / -0.0037 and rms
//      0.074 / 0.075 / 0.070 of 3 E0. Not monotone, so no order is claimed:
//      the staircase, cell by cell, is the same ~7 % scatter fs_electric's 3-D
//      sphere shows (0.069 / 0.071), and the amplitude moves inside +-3 %.
//   X  max |phi - exact| 3.32e-3 / 8.83e-4 / 2.28e-4 at a = 4 / 8 / 16 (b = 8a),
//      order 1.91 / 1.96: the operator is second order where the geometry is.
//   O  max |phi - linear| 9.1e-15.        N  4.1e-14 of max E^2.
//  The sphere's gates are regression guards set from this run.
//
//  Usage: axisym_field [-sec SXON]
//==============================================================================
#include "FsElectricAxi.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace lbm;

using L  = D2Q9;
using FS = FreeSurfaceSolver<L, CentralMoments<L, AxiGuo, RawPopulations>>;

namespace {

int g_fail = 0;

void check(const char* what, double got, double want, double tol) {
  const bool ok = std::isfinite(got) && std::fabs(got - want) <= tol;
  if (!ok) ++g_fail;
  std::printf("  %-52s %12.6g  (want %g +- %g)  %s\n", what, got, want, tol, ok ? "PASS" : "FAIL");
}

//---- S: the sphere -----------------------------------------------------------------
// fs_electric C2's measure, in r-z: over the gas within two cells of the sphere,
// each cell weighted by its radius (the ring it stands for), the least-squares
// slope of E^2 against the exact E^2 (the field's amplitude), and the mean and
// rms of |E| - |E_exact| in units of the polar surface field 3 E0.
struct SphereErr { double slope = NAN, mean = NAN, rms = NAN; int sweeps = 0; long cells = 0; };

SphereErr sphere(Index a) {
  SphereErr Q;
  const Index nx = 5 * a, ny = 10 * a;
  Domain d(nx, ny, 1, false, false, true);
  FS fs(d);
  fs.set_axisymmetric(true);
  fs.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  const double zc = 0.5 * double(ny);
  const Real ra = Real(a), zcr = Real(zc);
  fs.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real r = Real(px - hx) + Real(0.5), z = Real(py - hy) + Real(0.5) - zcr;
    Real e = ra + Real(0.5) - Kokkos::sqrt(r * r + z * z);
    e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
    return typename FS::Seed{e, Real(1)};
  });
  const double E0 = 0.01, a3 = double(a) * a * a;
  auto phx = [&](double r, double z) {
    const double zp = z - zc, rho = std::hypot(r, zp);
    return -E0 * zp * (1.0 - a3 / (rho * rho * rho));
  };
  FsElectricAxi el(d);
  el.liquid_value = Real(0);
  el.set_roles(
      [&](Index x, Index y) -> ElecRole {
        return (x == nx - 1 || y == 0 || y == ny - 1) ? ElecElectrode : ElecFree;
      },
      [&](Index x, Index y) -> Real {
        // the exact potential on the face toward the interior
        if (x == nx - 1) return Real(phx(double(nx - 1), double(y) + 0.5));
        if (y == 0) return Real(phx(double(x) + 0.5, 1.0));
        return Real(phx(double(x) + 0.5, double(ny - 1)));
      });
  el.refresh(fs);
  el.initialize(0.0);
  Q.sweeps = el.solve(200000, 1e-13);
  el.apply(fs);
  double sxy = 0, sxx = 0, sm = 0, s2 = 0, sw = 0;
  for (Index y = 1; y < ny - 1; ++y)
    for (Index x = 0; x < nx - 1; ++x) {
      if (el.cell(x, y) != 3) continue;          // gas
      const double r = double(x) + 0.5, zp = double(y) + 0.5 - zc;
      const double rho = std::hypot(r, zp);
      if (rho - double(a) > 2.0) continue;
      const double r5 = std::pow(rho, 5);
      const double Ez = E0 * (1.0 - a3 / (rho * rho * rho) + 3.0 * a3 * zp * zp / r5);
      const double Er = E0 * 3.0 * a3 * zp * r / r5;
      const double ex2 = Er * Er + Ez * Ez, e2 = el.e2(x, y);
      sxy += r * e2 * ex2;  sxx += r * ex2 * ex2;
      const double de = (std::sqrt(e2) - std::sqrt(ex2)) / (3.0 * E0);
      sm += r * de;  s2 += r * de * de;  sw += r;
      ++Q.cells;
    }
  Q.slope = sxy / sxx;
  Q.mean = sm / sw;
  Q.rms = std::sqrt(s2 / sw);
  return Q;
}

//---- X: coaxial electrodes, no staircase --------------------------------------------
// A rod of radius a on the axis at 1 (conductor cells x < a, its face at r = a),
// an outer electrode at r = b (cells x >= b, face at r = b) at 0, periodic in z:
// phi = ln(r / b) / ln(a / b), exact on faces that sit on cell faces.
double coax(Index a, Index b, int& sweeps) {
  Domain d(b + 1, 4, 1, false, true, true);
  FS fs(d);
  fs.set_axisymmetric(true);
  fs.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  fs.initialize(KOKKOS_LAMBDA(Index) { return typename FS::Seed{Real(0), Real(1)}; });
  FsElectricAxi el(d);
  el.set_roles([&](Index x, Index) -> ElecRole { return (x < a || x >= b) ? ElecElectrode : ElecFree; },
               [&](Index x, Index) -> Real { return x < a ? Real(1) : Real(0); });
  el.refresh(fs);
  el.initialize(0.5);
  sweeps = el.solve(400000, 1e-15);
  double dev = 0;
  for (Index x = a; x < b; ++x) {
    const double r = double(x) + 0.5;
    dev = std::max(dev, std::fabs(el.phi(x, 1) - std::log(r / double(b)) / std::log(double(a) / double(b))));
  }
  return dev;
}

//---- O and N: columns and drops between two plates ----------------------------------
struct Plates { double maxdev = NAN; int sweeps = 0; bool touch = false; std::vector<double> e2; };

// rr: the melt's radius; reach: how far down from the top plate it hangs
Plates plates(Index nx, Index ny, Index rr, Index reach, bool ohmic) {
  Plates P;
  Domain d(nx, ny, 1, false, false, true);
  FS fs(d);
  fs.set_axisymmetric(true);
  fs.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });
  const Domain dd = d;
  const Index hx = d.hx, hy = d.hy;
  const Index top = ny - 1;
  fs.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Index x = px - hx, y = py - hy;
    const bool in = x < rr && y >= top - reach && y < top;
    return typename FS::Seed{in ? Real(1) : Real(0), Real(1)};
  });
  FsElectricAxi el(d);
  el.ohmic = ohmic;
  el.liquid_value = Real(1);
  el.set_roles(
      [&](Index x, Index y) -> ElecRole {
        if (y == 0 || y == top) return ElecElectrode;
        if (x == nx - 1) return ElecFrame;
        return ElecFree;
      },
      [&](Index, Index y) -> Real { return y == top ? Real(1) : Real(0); });
  el.refresh(fs);
  el.initialize(0.5);
  P.sweeps = el.solve(400000, 1e-14);
  el.apply(fs);
  P.touch = el.touching();
  // linear between the plates' faces at z = 1 (phi = 0) and z = top (phi = 1)
  double dev = 0;
  for (Index y = 1; y < top; ++y)
    for (Index x = 0; x < nx - 1; ++x) {
      const double z = double(y) + 0.5;
      dev = std::max(dev, std::fabs(el.phi(x, y) - (z - 1.0) / double(top - 1)));
      P.e2.push_back(el.e2(x, y));
    }
  P.maxdev = dev;
  return P;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::string sec = "SXON";
    for (int i = 1; i < argc; ++i) {
      if (!std::strcmp(argv[i], "-sec") && i + 1 < argc) sec = argv[++i];
      else {
        std::fprintf(stderr, "axisym_field: unknown option %s\n  usage: axisym_field [-sec SON]\n", argv[i]);
        Kokkos::finalize();
        return 2;
      }
    }
    std::printf("axisym_field: FsElectricAxi, r-z finite volumes, %s\n", sizeof(Real) == 8 ? "FP64" : "FP32");
    char w[96];
    if (sec.find('S') != std::string::npos) {
      std::printf("\nS. a conducting sphere in a uniform field: the gas within two cells of it, per 3 E0\n");
      for (Index a : {5, 10, 20}) {
        const SphereErr Q = sphere(a);
        std::printf("  a = %2d: %5ld cells  slope(E^2 vs exact) %.5f  mean %+.5f  rms %.5f  (%d sweeps)\n",
                    int(a), Q.cells, Q.slope, Q.mean, Q.rms, Q.sweeps);
        std::snprintf(w, sizeof w, "a = %2d: |slope - 1|, the field's amplitude", int(a));
        check(w, std::fabs(Q.slope - 1.0), 0.0, 0.04);
        std::snprintf(w, sizeof w, "a = %2d: rms |E| error / (3 E0) (staircase)", int(a));
        check(w, Q.rms, 0.0, 0.08);
      }
    }
    if (sec.find('X') != std::string::npos) {
      std::printf("\nX. coaxial electrodes, rod a at 1, shell b at 0: phi = ln(r/b) / ln(a/b)\n");
      double prev = 0;
      for (Index k : {1, 2, 4}) {
        int sw = 0;
        const double e = coax(4 * k, 32 * k, sw);
        std::printf("  a = %2d, b = %3d: max |phi - exact| %.4e  (%d sweeps)", int(4 * k), int(32 * k), e, sw);
        if (prev > 0) std::printf("  order %.2f", std::log(prev / e) / std::log(2.0));
        std::printf("\n");
        prev = e;
      }
    }
    if (sec.find('O') != std::string::npos) {
      std::printf("\nO. an ohmic column bridging plates at 1 and 0: phi linear in z\n");
      const Plates P = plates(24, 40, 6, 38, true);
      std::printf("  touching the collector: %s, %d sweeps\n", P.touch ? "yes" : "NO", P.sweeps);
      check("max |phi - linear|", P.maxdev, 0.0, 1e-10);
    }
    if (sec.find('N') != std::string::npos) {
      std::printf("\nN. a drop hanging from the top plate, not touching: ohmic = equipotential\n");
      const Plates A = plates(24, 40, 6, 20, true), B = plates(24, 40, 6, 20, false);
      double dmax = 0, emax = 0;
      for (std::size_t k = 0; k < A.e2.size(); ++k) {
        dmax = std::max(dmax, std::fabs(A.e2[k] - B.e2[k]));
        emax = std::max(emax, B.e2[k]);
      }
      std::printf("  touching: %s / %s\n", A.touch ? "yes" : "no", B.touch ? "yes" : "no");
      check("max |E^2 ohmic - E^2 equipotential| / max E^2", dmax / emax, 0.0, 1e-9);
    }
    std::printf("\naxisym_field: %s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    rc = g_fail ? 1 : 0;
  }
  Kokkos::finalize();
  return rc;
}
