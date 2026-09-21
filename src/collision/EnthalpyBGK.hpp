#pragma once
//==============================================================================
//  Advection-diffusion collision for a melting/solidifying scalar: the
//  transported moment is the TOTAL ENTHALPY, not the temperature.
//
//      H = e(T) + La f_l,       sum_i g_i = H,
//
//  with e the sensible enthalpy (gauge e(T_s) = 0), La = rho L the volumetric
//  latent heat and f_l the liquid fraction. `ScalarBGK` transports sum_i g_i = T
//  and its banner says why that cannot carry a jump in volumetric heat capacity;
//  this operator is that missing variable.
//
//  WHY THE LATENT HEAT LIVES IN THE REST POPULATION, AND NOWHERE ELSE.
//
//      g_0^eq = H - E (1 - w_0)
//      g_i^eq = w_i E (1 + c_i.u / cs2)                        i = 1 .. Q-1
//
//  The correction sits at c_0 = 0, so it is invisible to every moment that
//  carries a velocity:
//
//      sum_i g_i^eq          = H                 (conserved, discontinuous)
//      sum_i c_ia g_i^eq     = E u_a             (advects the SENSIBLE part)
//      sum_i c_ia c_ib g_i^eq= cs2 E delta_ab    (diffuses the SENSIBLE part)
//
//  That asymmetry is the whole scheme. H is conserved and may jump across the
//  front; E and k grad T stay continuous; the Stefan condition La ds/dt =
//  [k dT/dn] is then a CONSEQUENCE of conservation rather than a jump imposed at
//  a tracked interface. Nothing here tracks an interface.
//
//  The obvious alternative -- g_i^eq = w_i H (1 + c_i.u/cs2), i.e. advect and
//  diffuse the total -- is stable, smooth, and converges cleanly to the wrong
//  front speed. It is kept selectable as `EnthalpyAdvect::Total` so that
//  validation/stefan.cpp re-falsifies it on every ctest run, rather than this
//  banner asserting it. A control that is only argued for is not a control.
//
//  THE (1 - w_0) FACTOR IS LOAD-BEARING. Drop it and sum_i g_i^eq = H - w_0 E,
//  so BGK bleeds w_0 omega E per node per step -- w_0 = 1/4 on D3Q7, 1/3 on
//  D2Q5. At La = 0 with matched properties this operator is population-for-
//  population identical to ScalarBGK, and that identity (tests/test_enthalpy.cpp
//  block 1) is the first thing the omission breaks.
//
//  UNITS. Every coefficient is VOLUMETRIC or the plain transport coefficient,
//  in lattice units:
//
//      cp_s, cp_l   rho c_p        volumetric heat capacity, NOT c_p
//      k_s,  k_l    lambda         conductivity,             NOT lambda/rho
//      La           rho L          volumetric latent heat
//      D = k / (dE/dT)             = lambda / (rho c_p), with no stray 1/rho
//
//  rho = 1 in lattice units is exactly what makes a per-mass/per-volume mix-up
//  invisible, so the distinction is spelled on every member below.
//
//  RELAXATION. D = cs2 (1/omega - 1/2) with cs2 = 1/4 on D3Q7 and 1/3 on D2Q5,
//  read from the lattice by omega_from_diffusivity(). `tau = 3 D + 1/2` is wrong
//  by 4/3 on D3Q7 and must not appear anywhere, not even in a printed
//  diagnostic.
//
//  Where alpha = k/(dE/dT) is phase-independent (k_s cp_l == k_l cp_s, under
//  parallel mixing) the rate is the solver's: the scalar `omega` and the
//  per-node `omega_of` field both behave exactly as they do for ScalarBGK.
//  Where it is not, the rate is a function of THIS node's own enthalpy and the
//  solver's argument is IGNORED. See rate_at().
//
//  WHAT THIS DOES NOT DO.
//
//  - No momentum sink in the mush. There is no Carman-Kozeny term and no
//    penalty: a mushy cell advects at whatever velocity it is handed. Building
//    the moving populations on E rather than H drops La u.grad f_l, which is
//    the enthalpy-porosity assumption and is defensible only if the velocity is
//    damped to zero in the mush -- which this does not do. PhaseFields::Report
//    reports max|u| over mushy nodes so a case cannot ignore it silently.
//  - No density change on melting, no shrinkage, no convection driven by the
//    solid/liquid density difference. rho is uniform and the datum E_datum is
//    free only where div u = 0.
//  - No material dependence. collide() has no node index, so a substrate under
//    a melt pool cannot have its own cp and k; omega_at(n) can modulate a rate
//    but not a capacity. A per-node PhaseChange field would fix it and is not
//    written.
//  - The Chapman-Enskog recovery of dH/dt + div(E u) = div(k grad T) is to the
//    usual advection-diffusion order, NOT exact. It is exact at u = 0, which is
//    the whole of the acceptance test.
//
//  THE INVERSION STAYS IN REGISTERS, AND THAT WAS NOT OBVIOUS. A three-branch
//  closed form containing a sqrt, evaluated per node per step, is exactly the
//  shape that could push the population array into per-thread local memory --
//  the mechanism MomentCollision.hpp's banner measures at 47x on a device.
//  Measured 2026-09-21 with tests/frame_check.sh, clang -O3, arm64, frame /
//  loops / regidx and the instruction count:
//
//      ScalarBGK<D3Q7>             0 / 0 / 0   (80)  FP64    0 / 0 / 0   (80)  FP32
//      EnthalpyBGK<D3Q7>           0 / 0 / 0  (104)          0 / 0 / 0  (100)
//      EnthalpyRegularised<D3Q7>   0 / 0 / 0   (86)          0 / 0 / 0   (84)
//
//  So the inversion costs 24 instructions over the sensible scalar and spills
//  nothing. Re-run that script after touching invert(); the wall clock will not
//  tell you, and neither will any test in tests/.
//
//  WHAT IT ACHIEVES. validation/stefan.cpp, measured 2026-09-21, FP64, D3Q7,
//  Ste = 1, tau = 1.3, melting against the analytic Neumann solution: the
//  integral front position is within 0.011 % at nx = 160 and within 0.006 %
//  over nx = 160..640, the liquid temperature profile within 0.114 % in
//  relative L2, and a closed periodic box conserves the total enthalpy exactly
//  IN FP64 (see the open item below for FP32).
//  The EnthalpyAdvect::Total control -- the reading that advects and diffuses
//  the latent heat -- comes in at 15.39 %, i.e. 1371x worse, so the argument at
//  the top of this banner is re-falsified on every ctest run rather than
//  asserted here.
//
//  THE CAPACITY JUMP IS TESTED, NOT ASSUMED. The same case's two-phase row runs
//  cp_s = 2 against cp_l = 1 -- the configuration ScalarBGK's banner says needs
//  a different variable -- against the two-phase Neumann solution, and lands
//  within 0.71 % (lambda = 0.3065539, tau_s = 0.9, tau_l = 1.3). That row also
//  exercises the path row 2 cannot reach: k_s cp_l != k_l cp_s, so
//  rate_from_solver() is false and each node's rate comes from its own
//  enthalpy rather than from the solver's omega.
//
//  NO CONVERGENCE ORDER IS ESTABLISHED. The refinement ladder gives apparent
//  orders 2.31 and 1.30, and repeating it with the end time scaled by 1.07
//  gives -2.99 and 5.67. At an error of a few parts in 1e5 the ladder is
//  measuring where the exact front sits inside its cell, not the
//  discretisation. The claimable result is the bound above and not a rate; see
//  that case's banner for what would be needed to measure one.
//
//  ONE OPEN ITEM, IN FP32 ONLY. A closed periodic box conserves the total
//  enthalpy exactly in FP64 and does not drift with step count. In FP32 it
//  drifts 3.05e-05 at 5000 steps and 3.55e-04 at 20000 -- superlinear, and
//  neither a random walk nor a linear leak. EnthalpyRegularised, whose sum
//  telescopes, is no better (4.35e-04), so per-slot accumulation is refuted as
//  the cause and nothing has replaced it. validation/stefan.cpp prints both
//  rows every run. Treat a long FP32 enthalpy budget as unproven.
//
//  EVERYTHING ELSE HERE IS DERIVATION, NOT MEASUREMENT.
//==============================================================================
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace lbm {

//------------------------------------------------------------------------------
// What the moving populations carry. Sensible is the model. Total is the
// reading that advects AND diffuses the latent heat -- kept so the negative
// result is re-measured rather than asserted.
//------------------------------------------------------------------------------
enum class EnthalpyAdvect : std::uint8_t { Sensible = 0, Total = 1 };

//------------------------------------------------------------------------------
// Conductivity of a partially melted cell.
//
// Parallel is the default because it is what the enthalpy-porosity literature
// uses. Series is the correct rule for a layered cell in one dimension, and the
// difference is O(dx) in the front position -- the same order as everything the
// Stefan test measures, and UNMEASURED here. validation/stefan.cpp -mix series
// runs the alternative and prints the gap.
//------------------------------------------------------------------------------
enum class MushMix : std::uint8_t { Parallel = 0, Series = 1 };

//==============================================================================
//  The material law, and its inversion.
//
//  Trivially copyable, no Views, no destructor, so it captures by value into a
//  Kokkos lambda alongside the operator.
//
//  THE DERIVED BLOCK IS PRIVATE. Assigning the public fields and leaving the
//  derived ones stale is a wrong material that runs, converges, and reports a
//  plausible front. normalise() is the only door, and ready() says whether it
//  has been through it.
//==============================================================================
struct PhaseChange {
  // Set these. See the banner's units paragraph.
  Real T_s     = Real(0);   // solidus
  Real T_l     = Real(0);   // liquidus; T_l == T_s is an isothermal front
  Real cp_s    = Real(1);   // rho c_p, solid
  Real cp_l    = Real(1);   // rho c_p, liquid
  Real k_s     = Real(1);   // conductivity, solid
  Real k_l     = Real(1);   // conductivity, liquid
  Real La      = Real(1);   // rho L, volumetric latent heat
  Real E_datum = Real(0);   // gauge: E(T_s) = E_datum. Free only where div u = 0.

  //----------------------------------------------------------------------------
  // Host. Fills the derived block. Aborts rather than returning a flag: a
  // material that is wrong is wrong before the first step, and the tree's
  // convention is to stop there.
  //----------------------------------------------------------------------------
  inline void normalise() {
    if (T_l < T_s) {
      std::printf("PhaseChange: T_l (%g) < T_s (%g)\n", double(T_l), double(T_s));
      std::abort();
    }
    if (cp_s <= Real(0) || cp_l <= Real(0)) {
      std::printf("PhaseChange: volumetric heat capacity must be positive "
                  "(cp_s = %g, cp_l = %g). These are rho*c_p, not c_p.\n",
                  double(cp_s), double(cp_l));
      std::abort();
    }
    if (k_s <= Real(0) || k_l <= Real(0)) {
      std::printf("PhaseChange: conductivity must be positive "
                  "(k_s = %g, k_l = %g)\n", double(k_s), double(k_l));
      std::abort();
    }
    if (La < Real(0)) {
      std::printf("PhaseChange: La (%g) < 0\n", double(La));
      std::abort();
    }
    dTm_    = T_l - T_s;
    d_cp_   = cp_l - cp_s;
    cp_ref_ = Real(0.5) * (cp_s + cp_l);
    B_      = cp_s * dTm_ + La;
    H_l_    = cp_ref_ * dTm_ + La;
    if (B_ <= Real(0)) {
      // dTm == 0 and La == 0: no band and no latent heat, so the mushy branch
      // is a 0/0 and there is no phase change to model. Use ScalarBGK.
      std::printf("PhaseChange: dTm == 0 and La == 0 leaves no phase change. "
                  "Use ScalarBGK for a sensible-only scalar.\n");
      std::abort();
    }
    inv_B_       = Real(1) / B_;
    linear_band_ = (d_cp_ * dTm_ == Real(0));
    uniform_     = (k_s * cp_l == k_l * cp_s);
    ready_       = true;
  }

  //----------------------------------------------------------------------------
  // Forward map. TWO arguments, and that is not a convenience.
  //
  // H(T) IS NOT A FUNCTION AT AN ISOTHERMAL FRONT. At dTm == 0 the enthalpy
  // jumps by La at T = T_m, so a node sitting exactly at the melting point has
  // every H in [0, La] and the temperature cannot say which. That is precisely
  // the configuration validation/stefan.cpp uses, and a one-argument seed there
  // would pick a branch by accident -- invisibly, because every mushy state has
  // the same temperature AND (at dTm == 0) the same sensible enthalpy.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  Real enthalpy_of(Real T, Real fl) const { return sensible_of(T) + La * fl; }

  // HOST ONLY, and it aborts at an isothermal melting point rather than
  // silently choosing. The collision never calls it.
  inline Real enthalpy_of_T(Real T) const {
    if (dTm_ == Real(0) && T == T_s) {
      std::printf("PhaseChange::enthalpy_of_T(%g): H is not a function of T at "
                  "an isothermal melting point -- every H in [0, %g] has this "
                  "temperature. Use enthalpy_of(T, fl).\n",
                  double(T), double(La));
      std::abort();
    }
    return sensible_of(T) + La * (T <= T_s ? Real(0)
                                : T >= T_l ? Real(1)
                                           : (T - T_s) / dTm_);
  }

  //----------------------------------------------------------------------------
  // The inverse: closed form, three branches, no iteration, no table, no array.
  // This is the only direction the collision uses, and unlike the forward map it
  // IS single-valued -- T(H) is continuous and non-decreasing, constant across
  // the band.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void invert(Real H, Real& fl, Real& T, Real& E, Real& dEdT) const {
    if (H <= Real(0)) {                       // SOLID
      fl   = Real(0);
      T    = T_s + H / cp_s;
      dEdT = cp_s;
    } else if (H >= H_l_) {                   // LIQUID
      fl   = Real(1);
      T    = T_l + (H - H_l_) / cp_l;
      dEdT = cp_l;
    } else {                                  // MUSHY
      // H = B f + (d_cp dTm / 2) f^2. The citardauq root
      //     f = 2H / (B + sqrt(B^2 + 2 d_cp dTm H))
      // rather than (-B + sqrt(disc))/(d_cp dTm), which divides by zero at
      // cp_s == cp_l -- the commonest case -- and loses every digit to
      // cancellation for small d_cp while staying finite and plausible.
      //
      // f is solved FIRST and T = T_s + f dTm second, so dTm is only ever a
      // multiplier and never a divisor: dTm == 0 is an ordinary point.
      fl = linear_band_
             ? H * inv_B_
             : Real(2) * H / (B_ + Kokkos::sqrt(B_ * B_ + Real(2) * d_cp_ * dTm_ * H));
      T    = T_s + fl * dTm_;
      dEdT = cp_s + d_cp_ * fl;
    }
    // f_l is NEVER clamped. It is in [0,1] by construction (disc > 0 throughout
    // the band, and f(H_l) = 1 exactly), and a clamped fraction is how a scheme
    // silently creates or destroys latent heat.
    E = E_datum + H - La * fl;
  }

  KOKKOS_INLINE_FUNCTION Real temperature_of(Real H) const {
    Real fl, T, E, dEdT; invert(H, fl, T, E, dEdT); return T;
  }
  KOKKOS_INLINE_FUNCTION Real liquid_fraction(Real H) const {
    Real fl, T, E, dEdT; invert(H, fl, T, E, dEdT); return fl;
  }

  // Mixture conductivity. The rule is the operator's compile-time policy, so it
  // is passed rather than stored.
  template <MushMix M>
  KOKKOS_INLINE_FUNCTION Real conductivity(Real fl) const {
    if constexpr (M == MushMix::Parallel) {
      return k_s + fl * (k_l - k_s);
    } else {
      return Real(1) / ((Real(1) - fl) / k_s + fl / k_l);
    }
  }

  // Read-only view of the derived block, for tests and for a driver banner.
  KOKKOS_INLINE_FUNCTION Real band_width()     const { return dTm_;  }
  KOKKOS_INLINE_FUNCTION Real band_slope()     const { return B_;    }
  KOKKOS_INLINE_FUNCTION Real liquidus_H()     const { return H_l_;  }
  KOKKOS_INLINE_FUNCTION Real capacity_mean()  const { return cp_ref_; }
  KOKKOS_INLINE_FUNCTION bool linear_band()    const { return linear_band_; }
  // alpha is phase-independent: true only under parallel mixing, which is why
  // the operator ANDs this with its own policy before trusting it.
  KOKKOS_INLINE_FUNCTION bool uniform_alpha()  const { return uniform_; }
  KOKKOS_INLINE_FUNCTION bool ready()          const { return ready_; }

 private:
  // e(T), gauge e(T_s) = 0. Quadratic across the band so that dE/dT is the
  // EXACT mixture cp_s + d_cp f rather than a band-average constant: a
  // piecewise-linear e with a constant band capacity makes D jump by cp_s/cp_ref
  // and cp_l/cp_ref at the two edges and misplaces the band's sensible content
  // by up to |d_cp| dTm / 8 at mid-band -- exactly zero at both edges, which is
  // what makes it easy to miss.
  KOKKOS_INLINE_FUNCTION Real sensible_of(Real T) const {
    const Real th = T - T_s;
    if (th <= Real(0))   return cp_s * th;
    if (th >= dTm_)      return cp_ref_ * dTm_ + cp_l * (T - T_l);
    return cp_s * th + d_cp_ * th * th / (Real(2) * dTm_);
  }

  Real dTm_ = 0, d_cp_ = 0, cp_ref_ = 1, B_ = 1, H_l_ = 1, inv_B_ = 1;
  bool linear_band_ = true;
  bool uniform_     = true;
  bool ready_       = false;
};

//==============================================================================
//  The operator.
//==============================================================================
template <class L,
          EnthalpyAdvect A = EnthalpyAdvect::Sensible,
          MushMix        M = MushMix::Parallel>
struct EnthalpyBGK {
  using Lattice = L;
  using Storage = RawPopulations;
  static constexpr const char* name =
      (A == EnthalpyAdvect::Sensible) ? "EnthalpyBGK" : "EnthalpyBGK(Total,CONTROL)";
  // Not part of the collision concept. Here so that a driver banner cannot
  // print the operator's name without printing what it actually carries.
  static constexpr const char* transported = "total enthalpy H = e(T) + La f_l";

  // NO LATTICE ASSERTION HERE, AND THAT IS A CORRECTION. An earlier version
  // asserted D*cs2 == 1 - w_0, which holds on D2Q5 and D3Q7 and FAILS on D2Q9
  // (2/3 vs 5/9) and D3Q27 (1 vs 19/27) -- so it silently banned this operator
  // from the two product lattices. That identity belongs to the REGULARISED
  // sibling, which writes h[0] as the residual dH - D cs2 dE and must agree
  // with dH - (1 - w_0) dE; it asserts it itself.
  //
  // This equilibrium needs only the three weight identities every lattice here
  // satisfies -- sum_i w_i = 1, sum_i w_i c_i = 0, sum_i w_i c_ia c_ib =
  // cs2 delta_ab -- so its moments are dH, E u and cs2 dE on ANY of them.
  // tests/test_enthalpy.cpp block 2 sweeps D3Q7, D2Q5 and D3Q27 to pin that.
  // The lattice matters for the comparison, not the algebra: D3Q27 has
  // cs2 = 1/3 like the D3Q19 this tree's reduced-keyhole port is checked
  // against, where D3Q7 has 1/4.
  //
  // Deliberately NO supports_navier_stokes assertion either: this is an
  // advection-diffusion operator and that trait is false on exactly the
  // reduced lattices it is most often used on.

  // THE STORAGE REFERENCE, AND IT IS AN ENTHALPY. The scalar concept spells it
  // T_ref and ScalarSolver reads it by that name at six sites, so the name is
  // not ours to choose. Set it to the mean enthalpy, not the mean temperature.
  Real T_ref = Real(0);
  Real omega = Real(1);      // used only where alpha is phase-independent
  View1D<Real> omega_of;     // ScalarBGK's per-node field; same caveat

  //----------------------------------------------------------------------------
  void set_material(PhaseChange m) {
    m.normalise();
    pc_   = m;
    omega = omega_from_diffusivity(m.k_s / m.cp_s);
  }
  void set_enthalpy_reference(Real H0) { T_ref = H0; }
  KOKKOS_INLINE_FUNCTION const PhaseChange& material() const { return pc_; }

  static Real omega_from_diffusivity(Real d) {
    return Real(1) / (d * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real diffusivity_from_omega(Real w) {
    return (Real(1) / w - Real(0.5)) * cs2<L, Real>();
  }
  static Real omega_from_conductivity(Real k, Real cp) {
    return omega_from_diffusivity(k / cp);
  }
  Real omega_solid()  const { return omega_from_diffusivity(pc_.k_s / pc_.cp_s); }
  Real omega_liquid() const { return omega_from_diffusivity(pc_.k_l / pc_.cp_l); }
  Real tau_solid()    const { return Real(1) / omega_solid();  }
  Real tau_liquid()   const { return Real(1) / omega_liquid(); }

  // True when the solver's rate is honoured. Parallel mixing with k_s cp_l ==
  // k_l cp_s makes k(f) = alpha dE/dT identically, so alpha is the same in both
  // phases and through the band. Series mixing breaks that even at matched
  // alpha, so the policy is ANDed in rather than trusted from the material.
  KOKKOS_INLINE_FUNCTION bool rate_from_solver() const {
    return (M == MushMix::Parallel) && pc_.uniform_alpha();
  }

  //----------------------------------------------------------------------------
  // dH = sum_i h_i, so H = T_ref + dH.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static Real deviation(const Real h[L::Q]) {
    Real t = Real(0);
    for (int i = 0; i < L::Q; ++i) t += h[i];
    return t;
  }

  // RETURNS THE TOTAL ENTHALPY, NOT A TEMPERATURE. The name belongs to the
  // scalar concept (ChargeCentralMoments is the precedent for a transported
  // variable that is not a temperature); use PhaseFields for T and f_l.
  KOKKOS_INLINE_FUNCTION Real temperature(const Real h[L::Q]) const {
    return T_ref + deviation(h);
  }

  //----------------------------------------------------------------------------
  // h_i^eq = w_i [ dE + E (c_i.u)/cs2 ]  +  (i == 0 ? dH - dE : 0)
  //
  // Deviation in the rest term, FULL value in the flux term -- the same
  // asymmetry ScalarBGK's eq() already argues, and for the same reason.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static Real eq_at(int i, Real dH, Real dE, Real Eflux, Real ux, Real uy, Real uz) {
    constexpr Real ics2 = inv_cs2<L, Real>();
    const Real cu = Real(cvel<L>(i, 0)) * ux + Real(cvel<L>(i, 1)) * uy +
                    Real(cvel<L>(i, 2)) * uz;
    return weight<L, Real>(i) * (dE + Eflux * ics2 * cu) +
           (i == 0 ? (dH - dE) : Real(0));
  }

  // The concept's single-direction equilibrium. One inversion per call, so this
  // is for the cold paths (seeding, the outflow donor) and not the hot loop.
  KOKKOS_INLINE_FUNCTION
  Real eq(int i, Real dH, Real ux, Real uy, Real uz) const {
    Real fl, T, E, dEdT;
    pc_.invert(T_ref + dH, fl, T, E, dEdT);
    const Real Eflux = (A == EnthalpyAdvect::Sensible) ? E : (T_ref + dH);
    return eq_at(i, dH, Eflux - T_ref, Eflux, ux, uy, uz);
  }

  KOKKOS_INLINE_FUNCTION Real omega_at(Index n) const {
    return omega_of.data() ? omega_of(n) : omega;
  }

  // The rate this node actually relaxes at, exposed so a test can pin the
  // difference between "the material decided" and "the solver decided".
  KOKKOS_INLINE_FUNCTION Real rate_at(Real dH, Real w_from_solver) const {
    if (rate_from_solver()) return w_from_solver;
    Real fl, T, E, dEdT;
    pc_.invert(T_ref + dH, fl, T, E, dEdT);
    return dEdT / (pc_.template conductivity<M>(fl) * inv_cs2<L, Real>() +
                   Real(0.5) * dEdT);
  }

  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dH, Real ux, Real uy, Real uz, Real w) const {
    Real fl, T, E, dEdT;
    pc_.invert(T_ref + dH, fl, T, E, dEdT);
    // The control reading advects and diffuses the whole enthalpy.
    const Real Eflux = (A == EnthalpyAdvect::Sensible) ? E : (T_ref + dH);
    const Real dE    = Eflux - T_ref;
    const Real ww    = rate_from_solver()
                         ? w
                         : dEdT / (pc_.template conductivity<M>(fl) * inv_cs2<L, Real>() +
                                   Real(0.5) * dEdT);
    for (int i = 0; i < L::Q; ++i)
      h[i] += ww * (eq_at(i, dH, dE, Eflux, ux, uy, uz) - h[i]);
  }
  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dH, Real ux, Real uy, Real uz) const {
    collide(h, dH, ux, uy, uz, omega);
  }

 protected:
  PhaseChange pc_{};
};

}  // namespace lbm
