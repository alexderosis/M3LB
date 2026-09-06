#pragma once
//==============================================================================
//  Specular reflection: a free-slip (symmetry) wall.
//
//  A population arriving at the wall leaves with its WALL-NORMAL component
//  reversed and its tangential components untouched:
//
//      f_i^out = f_{m(i)}^in,     c_{m(i)} = c_i with component `axis` negated.
//
//  That imposes u.n = 0 and zero tangential stress, which is the symmetry-plane
//  condition. Compare bounce-back, f_i^out = f_{opp(i)}^in, which reverses ALL
//  components and imposes u = 0.
//
//  ================= WHY THIS WAS THOUGHT IMPOSSIBLE HERE ====================
//  This file exists because a claim in ehd_electroconvection.cpp and in
//  doc/m3lb.tex was WRONG, and the record is corrected here rather than quietly
//  dropped. The claim was that specular reflection could not be added to
//  Esoteric Pull because bounce-back is only cheap there thanks to opposite
//  directions occupying adjacent slots, and a mirror is a different permutation
//  landing in a different pair.
//
//  The second half is true and the conclusion does not follow. Read the storage
//  table at the top of memory/EsotericPull.hpp: at node n, even parity, for odd
//  i, the scheme READS A[n][i] and A[n+c_i][i+1] and WRITES those same two
//  slots, crossed. Over all pairs the read set and the write set are THE SAME Q
//  SLOTS. A node therefore holds its complete incoming population vector and
//  writes back only into slots it alone owns -- so ANY permutation applied
//  between the load and the store is race-free and needs no temporary buffer.
//  Specular reflection is exactly such a permutation. Nothing about the pairing
//  obstructs it.
//
//  WHAT IS ACTUALLY TRUE, and what the wrong claim was conflated with:
//  bounce-back on Esoteric Pull is the IDENTITY on the storage, so a Solid cell
//  is skipped outright -- no load, no store, no arithmetic (see that banner).
//  Specular is not the identity, so a free-slip cell has to run. That is a cost
//  and a cell type, not an obstruction.
//
//  ================= WHERE THE PLANE SITS ====================================
//  Halfway, exactly as for bounce-back, and for the same reason: the fluid node
//  emits toward the wall cell on one step and reads the reflected population
//  back on the next, so the reflecting plane lies midway between the last fluid
//  node and the wall cell. Measured in tests/test_specular.cpp against the
//  analytic half-channel; do not assume it, the EHD hydrostatic case is a
//  standing reminder that half a cell matters when you differentiate the field.
//
//  ================= THE TANGENTIAL DIRECTIONS ===============================
//  For c_i with a zero normal component m(i) == i, so those populations pass
//  through the wall cell unchanged. That looks alarming and is not: a fluid node
//  never reads them. A fluid node at the wall reads only the populations whose
//  velocity points INTO the fluid, i.e. those with a nonzero normal component,
//  and those are precisely the ones the mirror rewrites. The tangential
//  populations circulate along the wall column, are never collided and are never
//  observed. They are also bounded -- pure transport of whatever the initial
//  condition put there -- so they cannot poison the run.
//
//  ================= WHAT THIS DOES NOT DO ===================================
//  Axis-aligned normals only (NrmXp/Xm/Yp/Ym/Zp/Zm). An oblique free-slip
//  surface has no exact specular permutation on a lattice and is not faked. Two
//  specular walls meeting at an edge are not handled either: the corner cell
//  would need both mirrors composed, and it is left to the caller to keep them
//  apart. NrmCorner and the outflow codes are rejected at setup.
//==============================================================================
#include "boundary/Flags.hpp"
#include "boundary/Regularized.hpp"      // NormalCode, normal_of
#include "core/Types.hpp"
#include "lattice/Lattices.hpp"

namespace lbm {

//------------------------------------------------------------------------------
//  m[axis][i] -- the index of the direction that mirrors i about `axis`.
//
//  Built at COMPILE TIME. The alternative is a linear search over Q per
//  direction, i.e. Q^2 comparisons per wall cell per step (729 on D3Q27), and
//  more to the point it would put a runtime-indexed table where CLAUDE.md's
//  moment-index rule says not to.
//------------------------------------------------------------------------------
template <class L>
struct MirrorTable {
  static constexpr int Q = L::Q;
  int m[3][Q];

  constexpr MirrorTable() : m{} {
    for (int a = 0; a < 3; ++a)
      for (int i = 0; i < Q; ++i) {
        int c[3] = {cvel<L>(i, 0), cvel<L>(i, 1), cvel<L>(i, 2)};
        c[a] = -c[a];
        m[a][i] = i;                       // identity unless a partner exists
        for (int j = 0; j < Q; ++j)
          if (cvel<L>(j, 0) == c[0] && cvel<L>(j, 1) == c[1] &&
              cvel<L>(j, 2) == c[2]) { m[a][i] = j; break; }
      }
  }
};

template <class L>
inline constexpr MirrorTable<L> mirror_table{};

// Axis a wall normal acts on: 0 for +-x, 1 for +-y, 2 for +-z.
KOKKOS_INLINE_FUNCTION
constexpr int mirror_axis(std::uint8_t code) {
  int n[3]; normal_of(code, n);
  return n[0] ? 0 : (n[1] ? 1 : 2);
}

//------------------------------------------------------------------------------
//  THE ON-NODE ZERO-FLUX SCALAR WALL, and it is a DIFFERENT boundary from the
//  fluid's SpecWall above even though it shares this table.
//
//  SpecWall is a GHOST cell: the mirror plane falls half a cell outside the
//  last fluid node, the cell does not collide, and validation/specular.cpp
//  measures the plane at 16.5000. That is right for a fluid, whose no-slip
//  family is halfway anyway.
//
//  A scalar in this tree is often on-node -- ScalarMoment puts a Dirichlet
//  value exactly ON the node, because E = -grad phi is a DERIVATIVE of the
//  field carrying the boundary value and a halfway stencil never sees it
//  (CLAUDE.md records that measurement: 11.19 % -> 1.49 %). A zero-flux
//  condition half a cell away from an on-node Dirichlet one is the mixed-wall-
//  family trap, so this variant puts the mirror plane ON the node.
//
//  It works on the UNKNOWN directions only. After streaming, a wall node holds
//  real data in every direction that arrived from inside and garbage in every
//  direction that would have arrived from outside. Mirror symmetry about the
//  node says the missing one equals its normal-reflected partner, which IS one
//  of the arrived ones:
//
//      q(-1) = q(+1)   =>   g_{+n}(0, streamed) = g_{-n}(0, streamed) mirrored.
//
//  So only the unknowns are overwritten, everything else is left alone, and the
//  node then COLLIDES like any other. Three consequences, and each of them is
//  the reason one of the existing options fails here:
//
//    * it reports the REAL value at the node (Sum g_i is meaningful), whereas
//      ScalarAdiabatic reports zero -- fatal in a case that differentiates phi
//      and integrates q;
//    * it reverses only the NORMAL component, whereas bounce-back reverses all
//      of them. At omega -> 2 a full reversal re-injects an odd-even mode that
//      never damps (CLAUDE.md's ringing entry); a mirror does not;
//    * the normal flux Sum_i g_i c_i.n vanishes identically by the symmetry of
//      the mirrored set, so it is zero-flux exactly and not to O(dx).
//
//  Safe on Esoteric Pull for the same reason SpecWall is: the read set and the
//  write set at a node are the SAME Q slots, so any function of the loaded
//  values may be stored back.
//------------------------------------------------------------------------------
template <class L, class R>
KOKKOS_INLINE_FUNCTION void mirror_unknowns(R* g, std::uint8_t code) {
  int nv[3];
  normal_of(code, nv);                       // OUTWARD normal
  const int ax = nv[0] ? 0 : (nv[1] ? 1 : 2);
  const int sg = nv[ax];
  R h[L::Q];
  for (int i = 0; i < L::Q; ++i) h[i] = g[i];
  for (int i = 0; i < L::Q; ++i)
    if (cvel<L>(i, ax) * sg < 0)             // points INTO the domain: unknown
      g[i] = h[mirror_table<L>.m[ax][i]];
}

//------------------------------------------------------------------------------
//  Every lattice in this tree is symmetric under reflection in each axis, so the
//  mirror is a genuine permutation rather than a partial map. Assert it: a
//  lattice that failed this would silently reflect some directions onto
//  themselves and leak momentum through the wall.
//------------------------------------------------------------------------------
template <class L>
constexpr bool mirror_is_permutation() {
  for (int a = 0; a < (L::D == 3 ? 3 : 2); ++a) {
    for (int i = 0; i < L::Q; ++i) {
      const int j = mirror_table<L>.m[a][i];
      // involution, and the normal component really did flip
      if (mirror_table<L>.m[a][j] != i) return false;
      if (cvel<L>(j, a) != -cvel<L>(i, a)) return false;
      for (int b = 0; b < 3; ++b)
        if (b != a && cvel<L>(j, b) != cvel<L>(i, b)) return false;
    }
  }
  return true;
}

static_assert(mirror_is_permutation<D2Q9>(),  "D2Q9 mirror is not a permutation");
static_assert(mirror_is_permutation<D3Q27>(), "D3Q27 mirror is not a permutation");
static_assert(mirror_is_permutation<D3Q19>(), "D3Q19 mirror is not a permutation");
static_assert(mirror_is_permutation<D2Q5>(),  "D2Q5 mirror is not a permutation");
static_assert(mirror_is_permutation<D3Q7>(),  "D3Q7 mirror is not a permutation");

}  // namespace lbm
