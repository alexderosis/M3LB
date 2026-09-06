#pragma once
//==============================================================================
//  Specular reflection for a scalar: the ON-NODE zero-flux wall.
//
//  A population leaves with its wall-NORMAL component reversed and its
//  tangential components untouched,
//
//      h_i^out = h_m(i)^in,   c_m(i) = c_i with component `axis` negated,
//
//  against bounce-back's h_i^out = h_opp(i)^in, which reverses ALL of them.
//
//  ===================== WHY THE PARENT TREE NEEDED THIS =====================
//  Ported from src/boundary/Specular.hpp (2026-09-05), where it was written
//  because validation/ehd_cavity.cpp -- a closed square with
//  d_x q = d_x phi = 0 on the sides -- could not be run without it. With the
//  fluid frozen, so that only the charge/potential pair is in play, both
//  pre-existing conditions failed there:
//
//      lateral wall        N = 41        N = 81
//      none (periodic)     -0.026 q0     -0.022 q0     reference
//      ScalarAdiabatic     non-finite    non-finite    inside 0.05 t0
//      ScalarOutflow       -0.083 q0     -0.279 q0     I0 158 % wrong
//
//  (worst excursion of q outside its bound [0, q0]). The open boundary gets
//  WORSE under refinement, which is the signature of a condition that is wrong
//  rather than merely inaccurate.
//
//  THREE PROPERTIES, and each is why one of the alternatives fails.
//
//    * It reports the REAL value at the node. Sum_i h_i is the field there, so
//      the node may be differentiated or integrated across like any other.
//      An adiabatic node reports a structural ZERO (that is deliberate: the
//      insulated plane is half a cell away and the node is a ghost), which is
//      fatal in a case whose whole answer is grad phi and an integral of q.
//    * It reverses only the NORMAL component. At omega -> 2 a FULL reversal
//      re-injects an odd-even mode that never damps -- the mechanism
//      collide_scalar_regularised exists to fix inside the collision, arriving
//      through the boundary instead. A mirror does not excite it.
//    * The normal flux Sum_i h_i c_i.n vanishes IDENTICALLY, by the symmetry of
//      the mirrored set. Zero flux exactly, not to O(dx).
//
//  ===================== IT ACTS ON THE UNKNOWNS ONLY ========================
//  After streaming, a wall node holds real data in every direction that arrived
//  from inside and nothing usable in the directions that would have arrived
//  from outside. Mirror symmetry about the NODE says the missing population
//  equals its normal-reflected partner, and that partner is one of the arrived
//  ones. So only the unknowns are overwritten; everything else is left alone
//  and the node then COLLIDES like any other. That is what puts the plane ON
//  the node, which is what the on-node Dirichlet family (ScalarMoment) has to
//  pair with -- see that banner for why halfway is first order here.
//
//  ===================== WHY THERE IS NO MIRROR TABLE ========================
//  The parent tree builds a constexpr lookup, m[axis][i]. That is right on a
//  CPU and would be WRONG here: a runtime index into a constexpr array is
//  exactly the pattern that pushes it out of registers into per-thread LOCAL
//  memory -- off-chip DRAM, uncoalesced -- which this tree has measured at 47x
//  in the colour gradient and which CLAUDE.md names first among the invariants
//  that break silently.
//
//  So the partner is SEARCHED FOR instead, in registers, with no memory
//  touched. The search is O(Q^2) and that is affordable precisely because this
//  runs on WALL nodes only: a boundary is O(N) of an O(N^3) domain, so an inner
//  loop there is invisible in the total. Trading arithmetic for memory traffic
//  is the right way round on a device, and it is the opposite of the trade the
//  CPU version makes.
//==============================================================================
#include "regularized.cuh"     // NormalCode, normal_of
#include "streaming.cuh"

namespace lbm {

//------------------------------------------------------------------------------
// Component `a` of direction i, without a table indexed at run time.
//------------------------------------------------------------------------------
template <class L>
LBM_HD LBM_INLINE int cvel_at(int i, int a) {
  return a == 0 ? L::cx(i) : (a == 1 ? L::cy(i) : L::cz(i));
}

//------------------------------------------------------------------------------
// Overwrite the unknown directions of a streamed node with their normal
// reflections. `code` is the OUTWARD normal (regularized.cuh's NormalCode);
// axis-aligned only, because a corner has two normals and one mirror cannot
// serve both.
//------------------------------------------------------------------------------
template <class L>
LBM_HD LBM_INLINE void mirror_unknowns(Real* h, std::uint8_t code) {
  int nv[3];
  normal_of(code, nv);
  const int ax = nv[0] ? 0 : (nv[1] ? 1 : 2);
  const int sg = nv[ax];

  Real in[L::Q];
  for (int i = 0; i < L::Q; ++i) in[i] = h[i];

  for (int i = 0; i < L::Q; ++i) {
    // Unknown <=> it points INTO the domain, i.e. against the outward normal.
    if (cvel_at<L>(i, ax) * sg >= 0) continue;
    // The partner has the same tangential components and the opposite normal
    // one. Search rather than look up -- see the banner.
    const int want[3] = {(ax == 0 ? -L::cx(i) : L::cx(i)),
                         (ax == 1 ? -L::cy(i) : L::cy(i)),
                         (ax == 2 ? -L::cz(i) : L::cz(i))};
    for (int j = 0; j < L::Q; ++j) {
      if (L::cx(j) == want[0] && L::cy(j) == want[1] && L::cz(j) == want[2]) {
        h[i] = in[j];
        break;
      }
    }
  }
}

}  // namespace lbm
