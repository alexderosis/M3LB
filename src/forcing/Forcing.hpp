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

  // (gx, gy, gz) is the direction a WARMER parcel is pushed when beta > 0, i.e.
  // AGAINST gravity: every case in the tree sets (0, 1, 0) and that is what
  // their hot plates rise under. The default below is never relied on.
  View1D<Real> T;                                   // temperature field
  Real gx = Real(0), gy = Real(-1), gz = Real(0);   // buoyancy direction
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
// BUOYANCY OF WATER NEAR ITS DENSITY MAXIMUM, through the same Guo machinery.
//
//   F(n) = up * rho0 * w * ( |T(n) - Tm|^q - |T0 - Tm|^q )
//
// Gebhart & Mollendorf (1977), Deep-Sea Res. 24, 831, pure water at 1 atm:
//
//   rho(T) = rho_m (1 - w_GM |T - T_m|^q),
//   rho_m = 999.9720 kg/m^3, w_GM = 9.297173e-6 (C)^-q, q = 1.894816,
//   T_m = 4.029325 C.
//
// CHECKED AGAINST: q to all digits (as quoted by Ramilison & Gebhart 1980), and
// the rest at the precision Wang, Jiang, Du, Sun & Calzavarini, PRFluids 6,
// L091501 (2021) print -- rho_c = 999.972, alpha* = 9.30e-6 K^-q, q = 1.895,
// T_c ~ 4 C -- which is the enthalpy-LBM group this model will be compared
// with. The remaining digits of w_GM and T_m are the values usually quoted and
// were NOT read from the 1977 paper itself.
//
// WHY A FORCE AND NOT AN EQUATION OF STATE. Over 0..8 C the density moves by
// 1.2e-4 of itself, so this is Boussinesq to that accuracy exactly as
// BoussinesqGuo is: the density is constant except in the buoyancy, and the
// temperature couples back through F rather than through the populations. What
// changes is only the shape of b(T) -- it is not monotone. A parcel is LIGHTER
// the farther it sits from Tm on EITHER side, so the layer between 0 and 4 C
// heated from below is unstable and the layer above 4 C is stable, and the same
// temperature gradient drives convection on one side of Tm and suppresses it on
// the other. That sign change is the whole of the physics this policy exists
// for (pinnacles below ~5 C, scallops at 5-7 C, the other pinnacles above, in
// Weady et al., PRL 128, 044502, 2022).
//
// UNITS ARE THE CASE'S, NOT THIS STRUCT'S. T, T0 and Tm are in whatever units
// the scalar carries, and must be the same units. If the scalar is the usual
// theta = (T - T_ref) / dT, then |T - T_m|^q = dT^q |theta - theta_m|^q, so the
// lattice coefficient is
//
//   w = g_lat * w_GM * dT^q      (theta_m = (T_m - T_ref) / dT),
//
// and Ra = w H^3 / (nu kappa) is the Rayleigh number with the ANOMALY's
// coefficient in place of g beta dT. `w` plays exactly the role BoussinesqGuo's
// `beta` does in every case here, which also stores g beta rather than beta.
//
// (gx, gy, gz) IS UP: the direction a LIGHTER parcel is pushed, the same meaning
// every BoussinesqGuo case gives those members. So swapping the policy in a case
// needs `beta` -> `w` and two new members, and nothing else.
//
// q = 1 WITH T0 AND EVERY T ABOVE Tm IS BoussinesqGuo WITH beta = w -- to
// ROUNDING, NOT BIT FOR BIT, because |T - Tm| - |T0 - Tm| and T - T0 round
// differently. validation/density_anomaly.cpp measures the gap on a growing
// Rayleigh-Benard mode: 1.7e-15 in T and 7.5e-14 in v. q < 1 is not supported:
// the buoyancy gradient is then infinite at Tm, which describes no fluid.
// Nothing asserts it.
//
// VALIDATED against the exact onset of penetrative convection
// (tools/penetrative_onset.py, stdlib shooting): with the maximum mid-layer,
// so that the upper half is STABLY stratified, Ra_c is +1.06 % at H = 32 and
// +0.25 % at H = 64, second order, on D3Q27 CM + D3Q7 -- the onset moves by a
// factor of 9.8 from the classical layer's, so a sign error in the stable half
// could not pass. What is NOT validated yet is anything nonlinear: no Nusselt
// number or sidewall flow with this law has been compared with a reference.
//
// COST. at() calls pow twice. The moment operators call at() twice per node
// (shift_velocity and the order-1 source); BGK and TRT call it once per
// DIRECTION through source(), so on D3Q27 they pay 28 evaluations where a
// central-moment operator pays 2. That is one more reason the moment operator
// is the default for this physics, not a reason to cache: a precomputed
// reference term would be a second copy of T0 that can go stale.
//------------------------------------------------------------------------------
struct DensityAnomalyGuo {
  static constexpr const char* name = "DensityAnomaly";
  static constexpr bool active = true;

  // Gebhart & Mollendorf (1977), pure water, 1 atm -- see above for what was
  // checked. Material data for the CASE to convert; nothing here reads them.
  static constexpr double q_water    = 1.894816;
  static constexpr double w_water    = 9.297173e-6;   // (deg C)^-q
  static constexpr double Tm_water_C = 4.029325;      // deg C
  static constexpr double rho_water  = 999.9720;      // kg/m^3

  View1D<Real> T;                                   // temperature field
  Real gx = Real(0), gy = Real(1), gz = Real(0);    // UP: where lighter goes
  Real rho0 = Real(1), w = Real(1), T0 = Real(0);
  Real Tm = Real(0), q = Real(q_water);             // density maximum, exponent
  Real fx = Real(0), fy = Real(0), fz = Real(0);    // optional uniform part

  KOKKOS_INLINE_FUNCTION
  void at(Index n, Real F[3]) const {
    const Real b = rho0 * w * (Kokkos::pow(Kokkos::fabs(T(n) - Tm), q) -
                               Kokkos::pow(Kokkos::fabs(T0 - Tm), q));
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
// A LINEAR DRAG -A u, APPLIED IMPLICITLY -- the enthalpy-porosity (Carman-
// Kozeny) mushy sink without the lag -- plus FieldGuo's external force.
//
//   F(n) = F_ext(n) - A(n) u(n)
//
// Guo defines the physical velocity with a half shift, rho u = m + F/2, and here
// F depends on u. Because the drag is LINEAR and LOCAL the pair closes in one
// line, with no iteration and no stored velocity:
//
//   u = (m + F_ext/2) / (rho + A/2),      then F = F_ext - A u.
//
// WHY. The driver-side sink every case used until now builds -A u from the
// velocity compute_macroscopic() REPORTED, which is a step old, and that makes
// it a two-step recurrence whose spectral radius reaches 1 at A = 1.000
// (validation/mushy_sink.cpp, model iii): A <= 1, and a leakage floor of ~2 % of
// u_max in the solid that no choice of A can go under. SS316L at dx = 8 already
// asked for A = 3.855. Closing the half shift instead gives the trapezoidal rule,
// a per-step factor g(A) = (2 - A)/(2 + A) on a uniform state: |g| < 1 for EVERY
// A > 0, so the bound is gone and the solid's velocity falls as 1/A.
//
// THE PRICE, stated because it looks like a bug when met: for A > 2, g < 0, and
// as A -> infinity g -> -1. The RAW momentum sum c_i f_i inside the solid then
// flips sign every step and barely decays. Nothing reads it. The physical
// velocity is m/(rho + A/2) ~ 2m/A, so the equilibrium, the advected scalar and
// the reported field all see a velocity that is small because A is large -- the
// alternation lives in a quantity that is not a velocity. mushy_sink.cpp's
// ringing detector watches the PHYSICAL u, which is the claim worth testing.
//
// WHERE IT ENTERS, and the two places it must not be skipped:
//   * shift_velocity() is the closed form above, so every operator's
//     macroscopic() -- and so collide(), compute_macroscopic() and the reported
//     field -- sees the implicit velocity;
//   * source()/source_raw() evaluate the drag at the u they are handed, which
//     BGK, TRT and the multiphase/MHD BGK operators pass after that shift;
//   * an operator that reads "the force" directly must call force_of(), which
//     hands the policy the velocity. MomentCollision and FluidSolver's
//     regularised wall do. at() alone returns the EXTERNAL part only.
// Operators that cannot be handed a velocity refuse the policy at compile
// time: accel_of() is DELETED for it (so MhdCentralMomentsShifted does not
// compile with it) and MhdCentralMoments static_asserts. A drag that is silently
// dropped would be a porous medium that is not there.
//
// A is in lattice units: force per unit volume per unit velocity. The
// Carman-Kozeny form A = A_lat eps (1 - f_l)^2 / (f_l^3 + eps) belongs to the
// case, together with the conversion A_lat = C dt / rho; melt_pool.cpp's banner
// records what a hidden dt in that constant cost.
//
// MEASURED in validation/mushy_sink.cpp (reduced channel, D3Q27, tau = 0.8):
// where the lagged sink converges the two profiles agree to 2.5e-14, because a
// steady state cannot see a lag; every A to 1e6 is stable on BGK and CM; the
// leak falls as 1/A (2.79e-6 of u_max at A = 1e4). The stiff sink's no-slip
// plane is NOT on the first solid node: it sits at +0.736 cells past the last
// liquid node on BGK and +0.809 on CM at tau = 0.8, and moves with tau (0.57 to
// 0.90 over tau = 0.6..1.0). Where the solid surface is depends on tau.
//------------------------------------------------------------------------------
struct DarcyGuo {
  static constexpr const char* name = "Darcy";
  static constexpr bool active = true;
  static constexpr bool velocity_dependent = true;

  View1D<Real> Ex, Ey, Ez;                          // external per-node force
  View1D<Real> A;                                   // drag coefficient, >= 0
  Real fx = Real(0), fy = Real(0), fz = Real(0);    // uniform part

  KOKKOS_INLINE_FUNCTION
  void at(Index n, Real F[3]) const {               // EXTERNAL part only
    F[0] = fx + (Ex.data() ? Ex(n) : Real(0));
    F[1] = fy + (Ey.data() ? Ey(n) : Real(0));
    F[2] = fz + (Ez.data() ? Ez(n) : Real(0));
  }
  KOKKOS_INLINE_FUNCTION
  Real drag(Index n) const { return A.data() ? A(n) : Real(0); }

  // The whole force at a known physical velocity.
  KOKKOS_INLINE_FUNCTION
  void at_u(Index n, const Real u[3], Real F[3]) const {
    at(n, F);
    const Real a = drag(n);
    F[0] -= a * u[0];  F[1] -= a * u[1];  F[2] -= a * u[2];
  }
  // (ux, uy, uz) arrive as m / rho and leave as the implicit physical velocity.
  KOKKOS_INLINE_FUNCTION
  void shift_velocity(Index n, Real rho, Real& ux, Real& uy, Real& uz) const {
    Real F[3]; at(n, F);
    const Real h = Real(0.5) / rho;
    const Real s = rho / (rho + Real(0.5) * drag(n));
    ux = (ux + h * F[0]) * s;  uy = (uy + h * F[1]) * s;  uz = (uz + h * F[2]) * s;
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source_raw(Index n, int i, Real ux, Real uy, Real uz) const {
    const Real u[3] = {ux, uy, uz};
    Real F[3]; at_u(n, u, F);
    return guo_source_raw<L>(i, F, ux, uy, uz);
  }
  template <class L>
  KOKKOS_INLINE_FUNCTION
  Real source(Index n, int i, Real omega, Real ux, Real uy, Real uz) const {
    return (Real(1) - Real(0.5) * omega) * source_raw<L>(n, i, ux, uy, uz);
  }
};

// True for a policy whose force depends on the velocity it is evaluated at.
template <class P>
inline constexpr bool velocity_dependent_force_v =
    requires { requires P::velocity_dependent; };

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
// Guo's. Getting that pairing wrong is silent, so it is asserted rather than
// left to inspection -- in validation/mhd_cm_shifted.cpp, check 6 (D2Q9) and
// check 9 (D3Q27), both of which run at rho != 1 because that is the only place
// the factor is visible. NOT in validation/forcing_cm.cpp, which this comment
// named until 2026-09-15: that case never instantiates this policy, and every
// row of it forces with `Guo` at rho = 1, where a stray power of rho is exactly
// 1. A cross-reference to a case that cannot see the property is worse than
// none, because it reads as coverage.
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
// DarcyGuo's force depends on u, which accel_of() is not given: refuse it
// rather than return the external part and drop the drag.
void accel_of(const DarcyGuo&, Index, Real, Real*) = delete;

// force_of() is how an operator that HOLDS the node's physical velocity asks
// for the force. Default: the policy's at(), which ignores u -- true of every
// policy except DarcyGuo, whose drag -A u needs it. Same overload pattern as
// accel_of(): the non-template exact match wins.
template <class P>
KOKKOS_INLINE_FUNCTION void force_of(const P& p, Index n, const Real*, Real F[3]) {
  p.at(n, F);
}
KOKKOS_INLINE_FUNCTION
void force_of(const DarcyGuo& p, Index n, const Real* u, Real F[3]) {
  p.at_u(n, u, F);
}

}  // namespace lbm
