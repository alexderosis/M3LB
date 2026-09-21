#pragma once
//==============================================================================
//  Regularised sibling of EnthalpyBGK: same equilibrium, same material law,
//  same transported variable. It relaxes the flux moment at omega and writes
//  every other moment straight to equilibrium instead of relaxing it.
//
//  Identical to EnthalpyBGK at omega = 1. It matters near omega = 2, where BGK
//  becomes a reflection and the ghost moments invert every step rather than
//  damping -- the mechanism ScalarRegularised.hpp already argues for the
//  sensible scalar, unchanged here.
//
//  THE GHOSTS GO TO cs2 dE, NOT cs2 dH. Writing dH there is the "diffuse the
//  total enthalpy" error wearing a regularised hat: it puts the latent heat
//  back into the second moment, which is the one place the whole scheme depends
//  on it not being. Setting E = H reproduces ScalarRegularised exactly, which
//  is how that mistake is spelled.
//
//  h[0] IS WRITTEN AS THE RESIDUAL, dH - D cs2 dE, so sum_i h_i = dH holds in
//  floating point and not merely to round-off -- BGK's per-slot update
//  conserves only to the accumulated error of Q separate multiply-adds. That is
//  why validation/stefan.cpp runs the closed-box conservation row on this
//  operator and prints BGK's drift beside it as the baseline rather than
//  asserting the same threshold for both.
//
//  The consistency of the two forms is a compile-time integer identity:
//  h_0^eq from here is dH - D cs2 dE, and from EnthalpyBGK it is
//  dH - (1 - w_0) dE, so D cs2 == 1 - w_0. It is asserted BELOW, in this class,
//  and not on EnthalpyBGK -- that identity holds on D2Q5 and D3Q7 and fails on
//  D2Q9 (2/3 vs 5/9) and D3Q27 (1 vs 19/27), so asserting it on the base class
//  banned the BGK operator from the two product lattices although its
//  equilibrium is correct there. Corrected 2026-09-21.
//==============================================================================
#include "collision/EnthalpyBGK.hpp"

namespace lbm {

template <class L,
          EnthalpyAdvect A = EnthalpyAdvect::Sensible,
          MushMix        M = MushMix::Parallel>
struct EnthalpyRegularised : EnthalpyBGK<L, A, M> {
  using Base    = EnthalpyBGK<L, A, M>;
  using Lattice = L;
  using Storage = RawPopulations;
  static constexpr const char* name = "EnthalpyRegularised";

  // The same rest-plus-axial-pairs contract ScalarRegularised asserts, and for
  // the same reason: the closed form below assumes one rest velocity and D
  // axial pairs, with pair a in slots (2a+1, 2a+2).
  static_assert(L::Q == 2 * L::D + 1,
                "EnthalpyRegularised needs a rest-plus-axial-pairs lattice "
                "(D2Q5 or D3Q7); D2Q9/D3Q27 have coupled axes.");
  static_assert(cvel<L>(1, 0) == 1 && cvel<L>(2, 0) == -1,
                "slots 1,2 must be the +/-x pair.");
  static_assert(cvel<L>(3, 1) == 1 && cvel<L>(4, 1) == -1,
                "slots 3,4 must be the +/-y pair.");
  static_assert(L::D < 3 || (cvel<L>(5, 2) == 1 && cvel<L>(6, 2) == -1),
                "slots 5,6 must be the +/-z pair.");
  // h[0] is written as dH - D cs2 dE and must equal the BGK form's
  // dH - (1 - w_0) dE, i.e. D*cs2 == 1 - w_0. Exact integers: D3Q7 gives
  // 3*1*8 == 6*4, D2Q5 gives 2*1*6 == 4*3. It FAILS on D2Q9 (2/3 vs 5/9) and
  // D3Q27 (1 vs 19/27), which is why this assertion lives here and not on
  // EnthalpyBGK -- the BGK equilibrium is correct on those lattices and was
  // wrongly forbidden them until 2026-09-21.
  static_assert(L::D * L::cs2_num * L::w_den == (L::w_den - L::w_num(0)) * L::cs2_den,
                "EnthalpyRegularised: D*cs2 must equal 1 - w_0, or the rest slot "
                "and the axial pairs disagree about where the latent heat lives.");

  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dH, Real ux, Real uy, Real uz, Real w) const {
    constexpr Real cs2v = cs2<L, Real>();
    Real fl, T, E, dEdT;
    this->material().invert(this->T_ref + dH, fl, T, E, dEdT);
    const Real Eflux = (A == EnthalpyAdvect::Sensible) ? E : (this->T_ref + dH);
    const Real dE    = Eflux - this->T_ref;
    const Real ww    = this->rate_from_solver()
                         ? w
                         : dEdT / (this->material().template conductivity<M>(fl) *
                                       inv_cs2<L, Real>() +
                                   Real(0.5) * dEdT);
    const Real u[3] = {ux, uy, uz};
    const Real d    = cs2v * dE;             // every ghost, at equilibrium

    // Unrolls: D is 2 or 3 with a constexpr bound, so the subscripts fold and h
    // stays in registers. tests/frame_check.sh is the instrument that says so.
    for (int a = 0; a < L::D; ++a) {
      const Real j = h[2 * a + 1] - h[2 * a + 2];
      const Real p = j + ww * (Eflux * u[a] - j);
      h[2 * a + 1] = Real(0.5) * (d + p);
      h[2 * a + 2] = Real(0.5) * (d - p);
    }
    h[0] = dH - Real(L::D) * d;
  }
  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dH, Real ux, Real uy, Real uz) const {
    collide(h, dH, ux, uy, uz, this->omega);
  }
};

}  // namespace lbm
