#pragma once
//==============================================================================
//  Body-force schemes.
//
//  Forcing is NOT orthogonal to collision: for the central-moment operator the
//  force enters at the moment level, not as a post-collision addend. So the
//  forcing policy is a member of the collision policy and the collision operator
//  decides how to consume it -- never applied as a separate pass.
//
//  Every policy answers `at(n, F)` for the force at node n. Uniform forces
//  ignore n; a Boussinesq buoyancy reads the temperature field there. That node
//  index is why the collision interface carries one: a body force that varies in
//  space is the normal case, not the exception -- buoyancy needs it now and the
//  Lorentz force will need it for MHD.
//==============================================================================
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// Shared Guo source term, without the (1 - omega/2) prefactor. TRT and the
// moment operators relax different modes at different rates and each applies its
// own prefactor per mode; only BGK can fold a single one in.
//------------------------------------------------------------------------------
template <class L>
KOKKOS_INLINE_FUNCTION
Real guo_source_raw(int i, const Real F[3], Real ux, Real uy, Real uz) {
  constexpr Real ics2 = inv_cs2<L, Real>();
  const Real cx = Real(cvel<L>(i, 0));
  const Real cy = Real(cvel<L>(i, 1));
  const Real cz = Real(cvel<L>(i, 2));
  const Real cu = cx * ux + cy * uy + cz * uz;
  const Real bx = (cx - ux) * ics2 + cu * ics2 * ics2 * cx;
  const Real by = (cy - uy) * ics2 + cu * ics2 * ics2 * cy;
  const Real bz = (cz - uz) * ics2 + cu * ics2 * ics2 * cz;
  return weight<L, Real>(i) * (bx * F[0] + by * F[1] + bz * F[2]);
}

//------------------------------------------------------------------------------
struct NoForcing {
  static constexpr const char* name = "None";
  static constexpr bool active = false;

  KOKKOS_INLINE_FUNCTION void at(Index, Real F[3]) const { F[0] = F[1] = F[2] = Real(0); }
  KOKKOS_INLINE_FUNCTION void shift_velocity(Index, Real, Real&, Real&, Real&) const {}
  template <class L>
  KOKKOS_INLINE_FUNCTION Real source_raw(Index, int, Real, Real, Real) const { return Real(0); }
  template <class L>
  KOKKOS_INLINE_FUNCTION Real source(Index, int, Real, Real, Real, Real) const { return Real(0); }
};

//------------------------------------------------------------------------------
// Guo et al. (2002), uniform force.
//   u   = ( sum_i c_i f_i + F/2 ) / rho
//   S_i = (1 - omega/2) w_i [ (c_i - u)/cs2 + (c_i.u) c_i / cs4 ] . F
//------------------------------------------------------------------------------
struct Guo {
  static constexpr const char* name = "Guo";
  static constexpr bool active = true;

  Real fx = Real(0), fy = Real(0), fz = Real(0);

  KOKKOS_INLINE_FUNCTION void at(Index, Real F[3]) const { F[0] = fx; F[1] = fy; F[2] = fz; }

  KOKKOS_INLINE_FUNCTION
  void shift_velocity(Index, Real rho, Real& ux, Real& uy, Real& uz) const {
    const Real h = Real(0.5) / rho;
    ux += h * fx;  uy += h * fy;  uz += h * fz;
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source_raw(Index n, int i, Real ux, Real uy, Real uz) const {
    Real F[3]; at(n, F);
    return guo_source_raw<L>(i, F, ux, uy, uz);
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source(Index n, int i, Real omega, Real ux, Real uy, Real uz) const {
    return (Real(1) - Real(0.5) * omega) * source_raw<L>(n, i, ux, uy, uz);
  }
};

//------------------------------------------------------------------------------
// Boussinesq buoyancy, delivered through the same Guo machinery.
//
//   F(n) = rho0 * g * beta * (T(n) - T0)
//
// The density is held constant everywhere except in the buoyancy term itself --
// that is the Boussinesq approximation, and it is why the thermal field couples
// back into the flow through a force rather than through the equation of state.
//------------------------------------------------------------------------------
struct BoussinesqGuo {
  static constexpr const char* name = "Boussinesq";
  static constexpr bool active = true;

  View1D<Real> T;                                   // temperature field
  Real gx = Real(0), gy = Real(-1), gz = Real(0);   // gravity direction
  Real rho0 = Real(1), beta = Real(1), T0 = Real(0);
  Real fx = Real(0), fy = Real(0), fz = Real(0);    // optional uniform part

  KOKKOS_INLINE_FUNCTION
  void at(Index n, Real F[3]) const {
    const Real b = rho0 * beta * (T(n) - T0);
    F[0] = fx + gx * b;  F[1] = fy + gy * b;  F[2] = fz + gz * b;
  }
  KOKKOS_INLINE_FUNCTION
  void shift_velocity(Index n, Real rho, Real& ux, Real& uy, Real& uz) const {
    Real F[3]; at(n, F);
    const Real h = Real(0.5) / rho;
    ux += h * F[0];  uy += h * F[1];  uz += h * F[2];
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source_raw(Index n, int i, Real ux, Real uy, Real uz) const {
    Real F[3]; at(n, F);
    return guo_source_raw<L>(i, F, ux, uy, uz);
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source(Index n, int i, Real omega, Real ux, Real uy, Real uz) const {
    return (Real(1) - Real(0.5) * omega) * source_raw<L>(n, i, ux, uy, uz);
  }
};

//------------------------------------------------------------------------------
// A uniform force plus an arbitrary per-node one, both through Guo.
//
// The uniform part is gravity; the field part is whatever another module wants
// to push the fluid with -- a penalised rigid body, most obviously. Keeping the
// two separate rather than making the caller add gravity into the field means
// the field can be rewritten every step by a module that knows nothing about
// gravity, which is exactly the arrangement PenalisedBody wants.
//
// The three views are optional: a default-constructed View has a null data
// pointer, and an absent component simply contributes nothing.
//------------------------------------------------------------------------------
struct FieldGuo {
  static constexpr const char* name = "FieldGuo";
  static constexpr bool active = true;

  View1D<Real> Ex, Ey, Ez;                          // per-node force, optional
  Real fx = Real(0), fy = Real(0), fz = Real(0);    // uniform part

  KOKKOS_INLINE_FUNCTION
  void at(Index n, Real F[3]) const {
    F[0] = fx + (Ex.data() ? Ex(n) : Real(0));
    F[1] = fy + (Ey.data() ? Ey(n) : Real(0));
    F[2] = fz + (Ez.data() ? Ez(n) : Real(0));
  }
  KOKKOS_INLINE_FUNCTION
  void shift_velocity(Index n, Real rho, Real& ux, Real& uy, Real& uz) const {
    Real F[3]; at(n, F);
    const Real h = Real(0.5) / rho;
    ux += h * F[0];  uy += h * F[1];  uz += h * F[2];
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source_raw(Index n, int i, Real ux, Real uy, Real uz) const {
    Real F[3]; at(n, F);
    return guo_source_raw<L>(i, F, ux, uy, uz);
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source(Index n, int i, Real omega, Real ux, Real uy, Real uz) const {
    return (Real(1) - Real(0.5) * omega) * source_raw<L>(n, i, ux, uy, uz);
  }
};

//------------------------------------------------------------------------------
// FOURTH-ORDER HERMITE FORCE, DIVIDED BY THE DENSITY.
//
// The companion of MATLAB/d2q9_shifted_force4.py, which derives what this is and
// asserts it. Two things distinguish it from `Guo`, and the second is the one
// that bites.
//
// THE EXPANSION. Guo's source is the SECOND-order Hermite expansion. Carried to
// FOURTH order the monomial central moments of the source become exactly
//     K_F = [0, F_x, F_y, 0, 0, 0, cs^2 F_y, cs^2 F_x, 0] / rho,
// u-INDEPENDENT, where the second-order source leaves behind
//     dk_21 = -2 ux uy Fx,  dk_12 = -uy^2 Fx,  dk_22 = +4 ux uy^2 Fx.
// That residue is 4.2e-3 of max|S| at |u| = 0.058 and 7.7e-2 at |u| = 0.25,
// growing as |u|^2. In the SHIFTED basis the fourth-order K_F collapses further,
// to [0, F_x, F_y, 0, ..., 0] -- first order only -- which is why the operator
// that consumes this policy writes the force into k_1 and nowhere else.
//
// THIS POLICY DOES NOT BUILD SOURCE POPULATIONS, and so cannot be used with BGK
// or TRT: it has no source_raw / source. That is deliberate rather than missing.
// A moment operator never needs them -- it writes K_F into the moment slots
// directly -- and supplying a second-order `source()` here would hand the BGK
// family something that is NOT the expansion this policy is named for. Pair it
// with MhdCentralMomentsShifted; anything else is a compile error.
//
// THE DENSITY. The reference expansion divides by R = rho, so the source's own
// first moment is F/rho and NOT F. The force therefore enters as an
// ACCELERATION: the momentum increment is F/rho per step, where `Guo` gives F.
// At fixed F the two policies differ by a factor rho -- which is 1 + O(Ma^2)
// here, so a run with the wrong one converges, looks healthy, and is wrong by a
// fraction of a percent. `accel()` is the single place that division happens,
// and shift_velocity is paired to it: u = (sum c f + a/2) / rho with a = F/rho,
// i.e. the half shift is a/(2 rho) = F/(2 rho^2), one power of rho more than
// Guo's. Getting that pairing wrong is silent, so it is asserted in
// validation/forcing_cm.cpp rather than left to inspection.
//
// `at()` keeps Guo's meaning -- the force DENSITY -- so that code reading it for
// other purposes (FluidSolver's open boundary) sees the same quantity from every
// policy. Only `accel()` carries the convention.
//------------------------------------------------------------------------------
struct HermiteForce4 {
  static constexpr const char* name = "Hermite4";
  static constexpr bool active = true;
  static constexpr bool per_density = true;   // first moment is F/rho, not F

  View1D<Real> Ex, Ey, Ez;                          // per-node force, optional
  Real fx = Real(0), fy = Real(0), fz = Real(0);    // uniform part

  KOKKOS_INLINE_FUNCTION
  void at(Index n, Real F[3]) const {               // force DENSITY, as in Guo
    F[0] = fx + (Ex.data() ? Ex(n) : Real(0));
    F[1] = fy + (Ey.data() ? Ey(n) : Real(0));
    F[2] = fz + (Ez.data() ? Ez(n) : Real(0));
  }

  // The source's own first moment: what the operator must write as 2 * k_1, and
  // what the momentum gains per step.
  KOKKOS_INLINE_FUNCTION
  void accel(Index n, Real rho, Real a[3]) const {
    Real F[3]; at(n, F);
    const Real ir = Real(1) / rho;
    a[0] = F[0] * ir;  a[1] = F[1] * ir;  a[2] = F[2] * ir;
  }

  KOKKOS_INLINE_FUNCTION
  void shift_velocity(Index n, Real rho, Real& ux, Real& uy, Real& uz) const {
    Real a[3]; accel(n, rho, a);
    const Real h = Real(0.5) / rho;                 // = F / (2 rho^2)
    ux += h * a[0];  uy += h * a[1];  uz += h * a[2];
  }
};

// accel_of() is the ONE place a policy's force convention is read, so that an
// operator can take any policy without knowing which convention it carries.
//
// Default: the policy's first moment is the force density itself, so there is no
// division -- true of Guo, BoussinesqGuo, FieldGuo and (trivially) NoForcing.
// HermiteForce4 overrides it, and the non-template overload wins on exact match.
template <class P>
KOKKOS_INLINE_FUNCTION void accel_of(const P& p, Index n, Real, Real a[3]) {
  p.at(n, a);
}
KOKKOS_INLINE_FUNCTION
void accel_of(const HermiteForce4& p, Index n, Real rho, Real a[3]) {
  p.accel(n, rho, a);
}

}  // namespace lbm
