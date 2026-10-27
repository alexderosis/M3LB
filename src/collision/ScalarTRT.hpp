#pragma once
//==============================================================================
//  TWO-RELAXATION-TIME advection-diffusion collision for a passive scalar,
//  by default at Lambda = 1/4, the value that keeps it stable when the flow is
//  fast and the diffusivity small.
//
//  THE FAMILY. On D2Q5/D3Q7 a scalar has T, D fluxes j_a (odd) and D ghosts
//  M_a (even, the axial second moments). The flux must relax at omega, which
//  sets D = cs2 (1/omega - 1/2); the ghosts carry nothing physical and relax at
//  a free rate omega_plus. The two rates enter the scheme only through
//
//      Lambda = (1/omega_plus - 1/2) (1/omega - 1/2),
//
//  and the two scalar operators already in the tree are members of the family:
//
//      ScalarBGK          omega_plus = omega   Lambda = (1/omega - 1/2)^2
//      ScalarRegularised  omega_plus = 1       Lambda = (1/omega - 1/2) / 2
//
//  tests/test_scalar_trt.cpp asserts both reductions to round-off. Both put
//  Lambda -> 0 as omega -> 2, i.e. as D -> 0, and that is the problem.
//
//  WHY THIS EXISTS -- MEASURED 2026-10-15 on Nianzhi's mixer. At Pe = 1e3 the
//  mixer's scalar runs at omega = 1.98413 (D = 1e-3) and ScalarRegularised
//  DIVERGED on both grids of validation/mixer_forward.cpp's two-grid check:
//  Phi went 0.176 -> 1e49 -> 4.7e271, growing ~x1.017 per step, while the flow
//  was fine. A von Neumann scan of the 7x7 one-step map with a uniform velocity
//  (tools/scalar_stability.py, which needs numpy) gives the largest stable |u|:
//
//      D          ScalarRegularised (x / diag)   BGK      this, Lambda = 1/4
//      1.0e-3     0.052 / 0.087                  0.25     0.50, every direction
//      5.0e-4     0.037 / 0.061                  0.25     0.50
//      2.5e-4     0.026 / 0.042                  0.25     0.50
//
//  The regularised limit falls as sqrt(D) -- about 1.64 sqrt(D) along an axis
//  -- and at D = 1e-3 it sits BELOW the empty duct's own centreline speed
//  (2.096 x 0.025 = 0.0524). Lambda = 1/4 makes the limit independent of D and
//  ten times larger. It is the stability-optimal value of the TRT advection-
//  diffusion analysis; nearby values are not: at D = 1e-3 the same scan gives
//  0.017 at Lambda = 1/12, 0.033 at 1/6, and 0.12 along x but 0.06 on a
//  diagonal at 3/16. tests/test_scalar_trt.cpp runs it as a differential test
//  on a periodic ring at u = 0.1: the regularised control blows up, this does
//  not.
//
//  WHAT IT IS NOT.
//    * Not "more accurate". The steady-state truncation errors of a TRT scheme
//      are functions of Lambda, and 1/4 is chosen for stability, not for
//      accuracy -- the accuracy-optimal values above are the unstable ones here.
//      In particular the location of a bounce-back (ScalarAdiabatic) wall
//      depends on Lambda, so a geometry error measured with one operator does
//      not transfer to the other. Measure, per case.
//    * Not a cure for under-resolution: a layer thinner than a cell is still
//      thinner than a cell. It removes an INSTABILITY, nothing else.
//    * Near omega = 2 the ghosts relax slowly (omega_plus = 0.016 at
//      omega = 1.984) but monotonically -- 1 - omega_plus > 0 -- so they decay
//      rather than reflect: BGK's ringing (ScalarRegularised's banner) is a
//      ghost multiplied by 1 - omega ~ -1 every step, which this does not do.
//
//  Interface, storage reference, per-node omega, omega_from_diffusivity: all
//  inherited from ScalarBGK. omega_plus follows the node's own omega, so a
//  conjugate problem keeps Lambda fixed in every material.
//==============================================================================
#include "collision/ScalarBGK.hpp"

namespace lbm {

template <class L>
struct ScalarTRT : ScalarBGK<L> {
  using Base    = ScalarBGK<L>;
  using Lattice = L;
  using Storage = RawPopulations;
  static constexpr const char* name = "ScalarTRT";

  // The closed form needs one rest velocity and D axial pairs in slots
  // (2a+1, 2a+2) -- D2Q5 and D3Q7, the adjacency contract Esoteric Pull relies
  // on. Asserted, as in ScalarRegularised.
  static_assert(L::Q == 2 * L::D + 1,
                "ScalarTRT needs a rest-plus-axial-pairs lattice (D2Q5 or D3Q7).");
  static_assert(cvel<L>(1, 0) == 1 && cvel<L>(2, 0) == -1, "slots 1,2 must be the +/-x pair.");
  static_assert(cvel<L>(3, 1) == 1 && cvel<L>(4, 1) == -1, "slots 3,4 must be the +/-y pair.");
  static_assert(L::D < 3 || (cvel<L>(5, 2) == 1 && cvel<L>(6, 2) == -1),
                "slots 5,6 must be the +/-z pair.");

  Real Lambda = Real(0.25);

  // The ghost rate that gives this Lambda at flux rate w. At w = 2 (D = 0) the
  // ghosts are frozen, which is the limit and not a division by zero.
  KOKKOS_INLINE_FUNCTION Real omega_plus(Real w) const {
    const Real tm = Real(1) / w - Real(0.5);
    return tm > Real(0) ? Real(1) / (Lambda / tm + Real(0.5)) : Real(0);
  }

  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dT, Real ux, Real uy, Real uz, Real w) const {
    constexpr Real cs2v = cs2<L, Real>();
    const Real T    = this->T_ref + dT;
    const Real u[3] = {ux, uy, uz};
    const Real d    = cs2v * dT;              // every ghost's equilibrium
    const Real wp   = omega_plus(w);
    Real msum = Real(0);
    // Unrolls (D is 2 or 3, a constexpr bound), so h stays in registers -- the
    // same structure as ScalarRegularised, with the ghost relaxed at wp
    // instead of replaced.
    for (int a = 0; a < L::D; ++a) {
      const Real j = h[2 * a + 1] - h[2 * a + 2];
      const Real m = h[2 * a + 1] + h[2 * a + 2];
      const Real p = j + w * (T * u[a] - j);
      const Real g = m + wp * (d - m);
      h[2 * a + 1] = Real(0.5) * (g + p);
      h[2 * a + 2] = Real(0.5) * (g - p);
      msum += g;
    }
    // sum_i h_i = dT by construction: the rest slot takes what the pairs leave.
    h[0] = dT - msum;
  }
  KOKKOS_INLINE_FUNCTION
  void collide(Real h[L::Q], Real dT, Real ux, Real uy, Real uz) const {
    collide(h, dT, ux, uy, uz, this->omega);
  }
};

}  // namespace lbm
