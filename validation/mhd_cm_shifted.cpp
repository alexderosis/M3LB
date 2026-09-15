//==============================================================================
//  MhdCentralMomentsShifted<D2Q9> against the algebra that defines it.
//
//  Six identities, each exact rather than approximate, so the threshold is
//  machine precision and not a tolerance anyone had to choose. Every expression
//  they check is derived symbolically in MATLAB/d2q9_shifted_force4.py; this
//  case is the same statements evaluated on the running operator.
//
//    1. The unforced post-collision state IS the shifted equilibrium at orders
//       3 and 4 (omega_6 = omega_7 = omega_8 = 1), and that equilibrium is the
//       Maxwell stress alone -- no rho above k_0.
//    2. K_F in the SHIFTED basis is [0, a_x, a_y, 0, ..., 0]: first order only.
//    3. The same populations read as MONOMIAL moments carry
//       k_21 = cs^2 a_y / 2 and k_12 = cs^2 a_x / 2 -- delivered by the basis
//       function, not written by hand. This is the term MhdCentralMoments has to
//       add explicitly (f6af63c) and the reason this operator cannot lose it.
//    4. Orders 0 to 3 agree with the monomial operator to machine precision, so
//       the change of basis is not a change of physics below fourth order.
//    5. At fourth order it IS a change, by exactly
//           d k_22(monomial) = cs^2 (1 - omega_bulk) (k_3 - k_3^eq),
//       which is zero at omega_bulk = 1. Checking the closed form rather than
//       just "they differ" is what makes this a measurement.
//    6. THE PAIRING, and the only one of the six that needs rho != 1. Through
//       the real path -- macroscopic() applies the half shift, collide() writes
//       +a/2 -- one step adds exactly a = F/rho to sum(c f). HermiteForce4's
//       first moment is F/rho where Guo's is F, so the shift is F/(2 rho^2) and
//       not F/(2 rho); a mismatched pair is a factor rho, which is 1 + O(Ma^2)
//       in any real run and would therefore pass every other test in this file.
//       That is why rho is drawn away from 1 here and only here.
//
//  WHAT THIS DOES NOT CHECK. That the fourth-order Hermite source is the right
//  physics: the closed form itself is derived and asserted in the script above.
//  validation/forcing_cm.cpp holds K_F for the MONOMIAL operators; checks 2 and 3
//  here are the shifted operator's counterpart, which is why it is not repeated
//  there. Nor anything three-dimensional: the operator is D2Q9 only.
//==============================================================================
#include "collision/MhdCentralMoments.hpp"
#include "collision/MhdCentralMomentsShifted.hpp"
#include "lattice/Lattices.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

using namespace lbm;

namespace {

constexpr double CS2 = 1.0 / 3.0;
constexpr double CS2_3D = 1.0 / 3.0;   // D3Q27, same as D2Q9 -- unlike D3Q7's 1/4

// monomial (shifted = false) or shifted central moment of f about (ux, uy)
double cmom(const Real f[9], double ux, double uy, int p, int q, bool shifted) {
  auto phi = [&](int e, double Cc) {
    return e == 0 ? 1.0 : (e == 1 ? Cc : Cc * Cc - (shifted ? CS2 : 0.0));
  };
  double s = 0;
  for (int i = 0; i < 9; ++i)
    s += double(f[i]) * phi(p, double(D2Q9::cx(i)) - ux)
                      * phi(q, double(D2Q9::cy(i)) - uy);
  return s;
}

const int SLOT[9][2] = {{0,0},{1,0},{0,1},{2,0},{0,2},{1,1},{2,1},{1,2},{2,2}};

}  // namespace

//------------------------------------------------------------------------------
// THE D3Q27 SPECIALISATION. Three identities, chosen because they are the ones
// that would break if the 2-D result had been carried across carelessly.
//
//   1. K_F in the SHIFTED basis is [0, a_x, a_y, a_z, 0, ...]: first order only,
//      in three dimensions as in two. validation/forcing_cm.cpp already asserts
//      the same thing on D3Q27 from the other side (its Hermite row), so this is
//      the operator meeting a statement that was already independently true.
//   2. The same populations read as MONOMIAL moments carry cs^2 a_b / 2 on the
//      mixed third-order slots -- delivered by the basis function, not written.
//   3. Momentum: one step adds exactly a = F/rho to sum(c f) at rho != 1, which
//      is the pairing between shift_velocity and the first-order write, and the
//      only one of the three that needs a density away from one.
//
// NOTE what is NOT claimed: that k_3^eq vanishes. It does in 2-D, where the
// Maxwell stress is traceless, and does NOT in 3-D, where its trace is |b|^2/2.
// The operator transforms its equilibrium rather than assuming a closed form, so
// nothing depends on it -- but it is the obvious thing to get wrong.
//------------------------------------------------------------------------------
static int check3d() {
  const Real omega = Real(1.3), omega_b = Real(0.7);
  View1D<Real> vbx("bx", 1), vby("by", 1), vbz("bz", 1);
  auto hbx = Kokkos::create_mirror_view(vbx);
  auto hby = Kokkos::create_mirror_view(vby);
  auto hbz = Kokkos::create_mirror_view(vbz);

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-0.06, 0.06), P(-0.01, 0.01),
                                         Fd(-2e-3, 2e-3), Rh(0.85, 1.15);
  double w_kf_s = 0, w_kf_m = 0, w_mom = 0, w_basis = 0;

  auto cm3 = [](const Real f[27], const double u[3], int p, int q, int r, bool shifted) {
    auto phi = [&](int e, double Cc) {
      return e == 0 ? 1.0 : (e == 1 ? Cc : Cc * Cc - (shifted ? CS2_3D : 0.0));
    };
    double s = 0;
    for (int i = 0; i < 27; ++i)
      s += double(f[i]) * phi(p, double(D3Q27::cx(i)) - u[0])
                        * phi(q, double(D3Q27::cy(i)) - u[1])
                        * phi(r, double(D3Q27::cz(i)) - u[2]);
    return s;
  };

  for (int t = 0; t < 200; ++t) {
    const double b[3] = {U(rng), U(rng), U(rng)};
    const double F[3] = {Fd(rng), Fd(rng), Fd(rng)};
    hbx(0) = Real(b[0]); hby(0) = Real(b[1]); hbz(0) = Real(b[2]);
    Kokkos::deep_copy(vbx, hbx); Kokkos::deep_copy(vby, hby); Kokkos::deep_copy(vbz, hbz);

    Real f0[27];
    for (int i = 0; i < 27; ++i)
      f0[i] = ProductFormEquilibrium<D3Q27>::eq(i, Real(Rh(rng)), Real(U(rng)),
                                                Real(U(rng)), Real(U(rng))) + Real(P(rng));

    MhdCentralMomentsShifted<D3Q27, HermiteForce4> fc;
    fc.omega = omega; fc.omega_bulk = omega_b;
    fc.Bx = vbx; fc.By = vby; fc.Bz = vbz;
    fc.forcing.fx = Real(F[0]); fc.forcing.fy = Real(F[1]); fc.forcing.fz = Real(F[2]);
    MhdCentralMomentsShifted<D3Q27, NoForcing> uc;
    uc.omega = omega; uc.omega_bulk = omega_b;
    uc.Bx = vbx; uc.By = vby; uc.Bz = vbz;

    Real fa[27], fb[27];
    for (int i = 0; i < 27; ++i) { fa[i] = f0[i]; fb[i] = f0[i]; }
    const Macro mac = uc.macroscopic(fb, 0);      // ONE Macro, no half shift
    const double rho = double(mac.dens);
    const double u[3] = {double(mac.ux), double(mac.uy), double(mac.uz)};
    const double a[3] = {F[0] / rho, F[1] / rho, F[2] / rho};
    const Real u3[3] = {mac.ux, mac.uy, mac.uz};
    const Real b3[3] = {Real(b[0]), Real(b[1]), Real(b[2])};
    fc.collide(fa, mac, 0);
    uc.collide(fb, mac, 0);

    Real d[27];
    for (int i = 0; i < 27; ++i) d[i] = fa[i] - fb[i];
    for (int p = 0; p < 3; ++p)
      for (int q = 0; q < 3; ++q)
        for (int r = 0; r < 3; ++r) {
          const int e[3] = {p, q, r};
          int ones = 0, twos = 0, axis = -1;
          for (int m = 0; m < 3; ++m) {
            if (e[m] == 1) { ++ones; axis = m; } else if (e[m] == 2) ++twos;
          }
          const double ws = (ones == 1 && twos == 0) ? a[axis] / 2 : 0.0;
          const double wm = (ones == 1) ? std::pow(CS2_3D, twos) * a[axis] / 2 : 0.0;
          w_kf_s = std::max(w_kf_s, std::fabs(cm3(d, u, p, q, r, true)  - ws));
          w_kf_m = std::max(w_kf_m, std::fabs(cm3(d, u, p, q, r, false) - wm));
        }

    // the pairing, through the real path
    Real fp[27];
    for (int i = 0; i < 27; ++i) fp[i] = f0[i];
    double r0 = 0, p0[3] = {0, 0, 0};
    for (int i = 0; i < 27; ++i) {
      r0 += double(fp[i]);
      p0[0] += double(fp[i]) * D3Q27::cx(i);
      p0[1] += double(fp[i]) * D3Q27::cy(i);
      p0[2] += double(fp[i]) * D3Q27::cz(i);
    }
    const Macro sh = fc.macroscopic(fp, 0);
    fc.collide(fp, sh, 0);
    double p1[3] = {0, 0, 0};
    for (int i = 0; i < 27; ++i) {
      p1[0] += double(fp[i]) * D3Q27::cx(i);
      p1[1] += double(fp[i]) * D3Q27::cy(i);
      p1[2] += double(fp[i]) * D3Q27::cz(i);
    }
    for (int m = 0; m < 3; ++m)
      w_mom = std::max(w_mom, std::fabs(p1[m] - p0[m] - F[m] / r0));

    // AND THE BASIS ITSELF, which checks 7-9 are blind to. They difference a
    // forced run against an unforced one, and phi_1 = C in BOTH bases, so the
    // basis cancels and a monomial transform passes all three. Measured: it
    // does. What distinguishes them is where orders >= 3 are sent -- this
    // operator drives them to the equilibrium's SHIFTED moments, so reading the
    // post-collision state back in the shifted basis must reproduce exactly
    // those. Under the monomial transform it does not.
    Real fe[27];
    uc.equilibrium(fe, Real(rho), u3, b3);
    for (int p = 0; p < 3; ++p)
      for (int q = 0; q < 3; ++q)
        for (int r = 0; r < 3; ++r)
          if (p + q + r >= 3)
            w_basis = std::max(w_basis, std::fabs(cm3(fb, u, p, q, r, true)
                                                - cm3(fe, u, p, q, r, true)));
  }

  const double TOL = 1e-14;
  struct { const char* what; double err; } R[] = {
    {"7. D3Q27: K_F shifted  == [0, a_x, a_y, a_z, 0 ...]              ", w_kf_s},
    {"8. D3Q27: K_F monomial == cs^(2m) a_b / 2 from the basis         ", w_kf_m},
    {"9. D3Q27: one step adds a = F/rho to sum(c f)   [rho != 1]       ", w_mom},
    {"10. D3Q27: orders >=3 ARE the shifted equilibrium (tests the BASIS)", w_basis},
  };
  int bad = 0;
  for (auto& r : R) {
    const bool ok = r.err < TOL;
    if (!ok) ++bad;
    std::printf("  %s  %.3e   %s\n", r.what, r.err, ok ? "OK" : "FAIL");
  }
  return bad;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int bad = 0;
  {
    const Real omega = Real(1.3), omega_b = Real(0.7);   // both away from 1
    View1D<Real> vbx("bx", 1), vby("by", 1), vbz("bz", 1);
    auto hbx = Kokkos::create_mirror_view(vbx);
    auto hby = Kokkos::create_mirror_view(vby);

    std::mt19937 rng(2026);
    std::uniform_real_distribution<double> U(-0.08, 0.08), P(-0.02, 0.02),
                                           Fd(-2e-3, 2e-3), Rh(0.85, 1.15);

    double w_eq = 0, w_kf_s = 0, w_kf_m = 0, w_low = 0, w_k22 = 0, w_mom = 0;

    for (int t = 0; t < 400; ++t) {
      const double bx = U(rng), by = U(rng), Fx = Fd(rng), Fy = Fd(rng);
      hbx(0) = Real(bx); hby(0) = Real(by);
      Kokkos::deep_copy(vbx, hbx);
      Kokkos::deep_copy(vby, hby);

      Real f0[9];
      for (int i = 0; i < 9; ++i)
        f0[i] = ProductFormEquilibrium<D2Q9>::eq(i, Real(Rh(rng)), Real(U(rng)),
                                                 Real(U(rng)), Real(0)) + Real(P(rng));

      MhdCentralMomentsShifted<D2Q9, HermiteForce4> fc;
      fc.omega = omega; fc.omega_bulk = omega_b;
      fc.Bx = vbx; fc.By = vby; fc.Bz = vbz;
      fc.forcing.fx = Real(Fx); fc.forcing.fy = Real(Fy);

      MhdCentralMomentsShifted<D2Q9, NoForcing> uc;
      uc.omega = omega; uc.omega_bulk = omega_b;
      uc.Bx = vbx; uc.By = vby; uc.Bz = vbz;

      Real fa[9], fb[9];
      for (int i = 0; i < 9; ++i) { fa[i] = f0[i]; fb[i] = f0[i]; }

      // ONE Macro for both, built WITHOUT the half shift. Letting macroscopic()
      // shift the forced call would collide it about a different u, and the
      // equilibrium would stop cancelling -- the trap forcing_cm.cpp records.
      const Macro mac = uc.macroscopic(fb, 0);
      const double rho = double(mac.dens), ux = double(mac.ux), uy = double(mac.uy);
      const double ax = Fx / rho, ay = Fy / rho;
      fc.collide(fa, mac, 0);
      uc.collide(fb, mac, 0);

      // --- 1. unforced post-collision == the shifted equilibrium -------------
      const double bx2 = bx * bx, by2 = by * by, bxy = bx * by;
      const double k6e = 0.5 * uy * (bx2 - by2) + 2 * ux * bxy;
      const double k7e = 0.5 * ux * (by2 - bx2) + 2 * uy * bxy;
      const double k8e = 0.5 * (ux * ux - uy * uy) * (bx2 - by2) - 4 * ux * uy * bxy;
      w_eq = std::max({w_eq,
                       std::abs(cmom(fb, ux, uy, 0, 0, true) - rho),
                       std::abs(cmom(fb, ux, uy, 2, 1, true) - k6e),
                       std::abs(cmom(fb, ux, uy, 1, 2, true) - k7e),
                       std::abs(cmom(fb, ux, uy, 2, 2, true) - k8e)});

      // --- 2 and 3. K_F in both bases ---------------------------------------
      Real d[9];
      for (int i = 0; i < 9; ++i) d[i] = fa[i] - fb[i];
      for (int j = 0; j < 9; ++j) {
        const double want_s = (j == 1) ? ax / 2 : (j == 2) ? ay / 2 : 0.0;
        const double want_m = (j == 1) ? ax / 2 : (j == 2) ? ay / 2
                            : (j == 6) ? 0.5 * CS2 * ay
                            : (j == 7) ? 0.5 * CS2 * ax : 0.0;
        w_kf_s = std::max(w_kf_s,
            std::abs(cmom(d, ux, uy, SLOT[j][0], SLOT[j][1], true) - want_s));
        w_kf_m = std::max(w_kf_m,
            std::abs(cmom(d, ux, uy, SLOT[j][0], SLOT[j][1], false) - want_m));
      }

      // --- 4 and 5. against the monomial operator, unforced ------------------
      MhdCentralMoments<D2Q9, true, NoForcing> mc;
      mc.omega = omega; mc.omega_bulk = omega_b;
      mc.Bx = vbx; mc.By = vby; mc.Bz = vbz;
      Real fm[9];
      for (int i = 0; i < 9; ++i) fm[i] = f0[i];
      mc.collide(fm, mac, 0);
      for (int j = 0; j < 8; ++j)                     // orders 0..3, i.e. not k_22
        w_low = std::max(w_low,
            std::abs(cmom(fb, ux, uy, SLOT[j][0], SLOT[j][1], false)
                   - cmom(fm, ux, uy, SLOT[j][0], SLOT[j][1], false)));
      const double k3pre = cmom(f0, ux, uy, 2, 0, true) + cmom(f0, ux, uy, 0, 2, true);
      const double pred  = CS2 * (1.0 - double(omega_b)) * k3pre;   // k_3^eq = 0
      const double meas  = cmom(fb, ux, uy, 2, 2, false) - cmom(fm, ux, uy, 2, 2, false);
      w_k22 = std::max(w_k22, std::abs(meas - pred));

      // --- 6. the pairing, through the REAL path, at rho != 1 ----------------
      Real fp[9];
      for (int i = 0; i < 9; ++i) fp[i] = f0[i];
      double r0 = 0, p0[2] = {0, 0};
      for (int i = 0; i < 9; ++i) {
        r0    += double(fp[i]);
        p0[0] += double(fp[i]) * D2Q9::cx(i);
        p0[1] += double(fp[i]) * D2Q9::cy(i);
      }
      const Macro shifted = fc.macroscopic(fp, 0);     // applies the half shift
      fc.collide(fp, shifted, 0);
      double p1[2] = {0, 0};
      for (int i = 0; i < 9; ++i) {
        p1[0] += double(fp[i]) * D2Q9::cx(i);
        p1[1] += double(fp[i]) * D2Q9::cy(i);
      }
      w_mom = std::max({w_mom, std::abs(p1[0] - p0[0] - Fx / r0),
                               std::abs(p1[1] - p0[1] - Fy / r0)});
    }

    const double TOL = 1e-14;
    struct { const char* what; double err; } R[] = {
      {"1. unforced post-collision == shifted equilibrium (Maxwell alone)", w_eq},
      {"2. K_F shifted   == [0, a_x, a_y, 0 ...]  (first order only)     ", w_kf_s},
      {"3. K_F monomial  == [.., cs^2 a_y/2, cs^2 a_x/2, 0] from the basis", w_kf_m},
      {"4. orders 0-3 == the monomial operator                           ", w_low},
      {"5. k_22 gap == cs^2 (1 - omega_b)(k_3 - k_3^eq)                  ", w_k22},
      {"6. one step adds a = F/rho to sum(c f)   [rho != 1]              ", w_mom},
    };
    std::printf("MhdCentralMomentsShifted<D2Q9> + HermiteForce4 vs the symbolic algebra\n");
    std::printf("backend %s   precision %s   400 random states, "
                "omega %.2f, omega_bulk %.2f, rho in [0.85, 1.15]\n\n",
                ExecSpace::name(), precision_name(), double(omega), double(omega_b));
    for (auto& r : R) {
      const bool ok = r.err < TOL;
      if (!ok) ++bad;
      std::printf("  %s  %.3e   %s\n", r.what, r.err, ok ? "OK" : "FAIL");
    }
    bad += check3d();
    std::printf("\n  %s\n", bad == 0
        ? "PASS -- the operator is the algebra in MATLAB/d2q9_shifted_force4.py"
        : "*** FAIL ***");
  }
  Kokkos::finalize();
  return bad == 0 ? 0 : 1;
}
