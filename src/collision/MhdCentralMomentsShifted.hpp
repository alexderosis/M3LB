#pragma once
#include <type_traits>
//==============================================================================
//  D2Q9 MHD central moments in the SHIFTED basis, with the force and the
//  equilibrium both at FOURTH Hermite order.
//
//  This is MhdCentralMoments moved from one basis to another, and it is NOT a
//  pure change of representation -- see THE ONE BEHAVIOURAL DIFFERENCE below.
//  Both operators are kept: this one because the shifted basis is what the rest
//  of the tree stores and what makes the forcing correct by construction, the
//  monomial one because it reproduces De Rosis, Leveque & Chahine Eq. (8) and
//  Eq. (11) literally and tests/test_moments.cpp pins it there.
//
//  MATLAB/d2q9_shifted_force4.py derives every expression below symbolically and
//  asserts it; run it after touching any of them.
//
//  THE BASIS. phi_0 = 1, phi_1 = C, phi_2 = C^2 - cs^2 per axis, C = c - u --
//  ProductBasis's fwd1d/inv1d, reused here rather than rewritten. The monomial
//  operator's own fwd/inv differ from these by exactly the cs^2 subtraction, and
//  that single term is the whole difference between the two schemes.
//
//  THE EQUILIBRIUM IS THE MAXWELL STRESS, AND NOTHING ELSE. The fourth-order
//  Hermite equilibrium has every shifted central moment ZERO above order zero --
//  verified symbolically -- so no rho survives past k_0 and the hydrodynamic
//  Galilean defects that Eq. (11) carries at orders 3 and 4 are absent. On D2Q9
//  that equilibrium is the product form: identical shifted moments means
//  identical populations, since nine moments determine nine populations. So
//  ProductFormEquilibrium is used for the seed, and it is exact, not a stand-in.
//
//      k_0^eq = rho                       k_1^eq = k_2^eq = 0
//      k_3^eq = 0                         <- the 2 rho cs^2 lives in the basis
//      k_4^eq = by^2 - bx^2               k_5^eq = -bx by
//      k_6^eq = uy (bx^2 - by^2)/2 + 2 ux bx by
//      k_7^eq = ux (by^2 - bx^2)/2 + 2 uy bx by
//      k_8^eq = (ux^2 - uy^2)(bx^2 - by^2)/2 - 4 ux uy bx by
//
//  with k_3 = k_20 + k_02 and k_4 = k_20 - k_02 as in the paper's Eq. (8).
//
//  THE FORCE IS ONE LINE, AND CANNOT BE FORGOTTEN. In this basis the fourth-order
//  K_F is [0, a_x, a_y, 0, ..., 0] -- first order only -- so the force is written
//  into k_1, k_2 and nowhere else, and the cs^2 a / 2 that a MONOMIAL operator
//  must add by hand at third order is delivered by the basis function instead.
//  That is the bug fixed in f6af63c made structurally impossible: read k_6* back
//  as a monomial and it picks up cs^2 k_1* = cs^2 a_y / 2, which is exactly the
//  term the monomial operator now writes explicitly.
//
//  `a` is the forcing policy's own first moment, from accel_of(). For Guo it is
//  the force density F; for HermiteForce4 it is F/rho. The operator does not
//  know or care which -- that is the point of routing it through accel_of --
//  but the policy's shift_velocity must be paired with the same choice, because
//  the pre-collision first central moment is -a/2 and the post-collision value
//  written here is +a/2. A mismatched pair is a silent factor rho.
//
//  THE ONE BEHAVIOURAL DIFFERENCE from the monomial operator, and it is bounded:
//  relaxation happens in whichever basis you store, so equilibrating k_8 here
//  rather than k_22 there changes the post-collision fourth-order moment by
//
//      d k_22(monomial) = cs^2 (1 - omega_bulk) (k_3 - k_3^eq),
//
//  identically zero at omega_bulk = 1. That expression is exactly the 6.5e-3 gap
//  measured between MhdCentralMoments and MomentCollision at B = 0 with
//  omega_bulk = 1.2; this operator closes it, at the price of no longer being
//  Eq. (11) to the letter. Orders 0 to 3 are unchanged.
//
//  WHAT THIS DOES NOT DO. D2Q9 only -- the admissible fourth-order Hermite terms
//  and the 2-D Maxwell trace (which vanishes, hence no b in k_3^eq) are both
//  two-dimensional statements. There is no HighOrder switch: the second-order
//  truncation is the monomial operator's business, and reproducing the published
//  table is what that operator is for.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/ProductBasis.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

namespace lbm {

template <class L, class Forcing = NoForcing>
struct MhdCentralMomentsShifted;

template <class Forcing>
struct MhdCentralMomentsShifted<D2Q9, Forcing> {
  using Lattice       = D2Q9;
  using ForcingPolicy = Forcing;
  using Equilibrium   = ProductFormEquilibrium<D2Q9>;
  using Storage       = RawPopulations;
  static constexpr const char* name = "MhdCMS";

  Real omega      = Real(1);    // omega_4 = omega_5, sets the viscosity
  Real omega_bulk = Real(1);    // omega_3
  View1D<Real> Bx, By, Bz;
  Forcing forcing{};

  static Real omega_from_viscosity(Real nu) {
    return Real(1) / (nu * inv_cs2<D2Q9, Real>() + Real(0.5));
  }
  static Real viscosity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<D2Q9, Real>();
  }
  KOKKOS_INLINE_FUNCTION static Real density(const Macro& m) { return m.dens; }

  KOKKOS_INLINE_FUNCTION
  Macro macroscopic(const Real f[9], Index n = 0) const {
    Real s = Real(0), mx = Real(0), my = Real(0);
    for (int i = 0; i < 9; ++i) {
      s  += f[i];
      mx += f[i] * Real(D2Q9::cx(i));
      my += f[i] * Real(D2Q9::cy(i));
    }
    const Real ir = Real(1) / s;
    Macro m{s, mx * ir, my * ir, Real(0)};
    // The half shift, whichever convention the policy carries. It must leave the
    // pre-collision first central moment at -a/2; collide() writes +a/2.
    forcing.shift_velocity(n, s, m.ux, m.uy, m.uz);
    return m;
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real f[9], const Macro& mac, Index n = 0) const {
    using B = ProductBasis<D2Q9>;
    const Real rho = mac.dens, ux = mac.ux, uy = mac.uy;
    const Real bx = Bx(n), by = By(n);

    // populations -> SHIFTED central moments m[p][q]; fwd1d carries the cs^2
    Real m[3][3];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) m[a][b] = f[B::pi(a, b)];
    for (int a = 0; a < 3; ++a) B::fwd1d(m[a][0], m[a][1], m[a][2], uy);
    for (int q = 0; q < 3; ++q) B::fwd1d(m[0][q], m[1][q], m[2][q], ux);

    // paper's basis (Eq. 8), read in the shifted frame
    const Real k3 = m[2][0] + m[0][2];
    const Real k4 = m[2][0] - m[0][2];
    const Real k5 = m[1][1];

    // equilibria -- the Maxwell stress alone; k3e is identically zero
    const Real bx2 = bx * bx, by2 = by * by, bxy = bx * by;
    const Real k4e = by2 - bx2;
    const Real k5e = -bxy;
    const Real k6e = Real(0.5) * uy * (bx2 - by2) + Real(2) * ux * bxy;
    const Real k7e = Real(0.5) * ux * (by2 - bx2) + Real(2) * uy * bxy;
    const Real k8e = Real(0.5) * (ux * ux - uy * uy) * (bx2 - by2)
                   - Real(4) * ux * uy * bxy;

    // relaxation (Eq. 12); ghosts 6,7,8 go straight to equilibrium
    const Real k3s = (Real(1) - omega_bulk) * k3;      // k3e = 0
    const Real k4s = k4 + omega * (k4e - k4);
    const Real k5s = k5 + omega * (k5e - k5);

    m[0][0] = rho;
    m[2][0] = Real(0.5) * (k3s + k4s);
    m[0][2] = Real(0.5) * (k3s - k4s);
    m[1][1] = k5s;
    m[2][1] = k6e;                    // no force term: K_F is first order here
    m[1][2] = k7e;
    m[2][2] = k8e;
    if constexpr (Forcing::active) {
      Real a[3]; accel_of(forcing, n, rho, a);
      m[1][0] = Real(0.5) * a[0];
      m[0][1] = Real(0.5) * a[1];
    } else {
      m[1][0] = Real(0);
      m[0][1] = Real(0);
    }

    for (int q = 0; q < 3; ++q) B::inv1d(m[0][q], m[1][q], m[2][q], ux);
    for (int a = 0; a < 3; ++a) B::inv1d(m[a][0], m[a][1], m[a][2], uy);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) f[B::pi(a, b)] = m[a][b];
  }

  KOKKOS_INLINE_FUNCTION
  static Real seed_value(int i, Real rho, Real ux, Real uy, Real uz) {
    // exact, not a stand-in: on D2Q9 the product form IS the fourth-order
    // Hermite equilibrium (identical shifted moments, hence identical populations)
    return ProductFormEquilibrium<D2Q9>::eq(i, rho, ux, uy, uz);
  }
};

}  // namespace lbm
