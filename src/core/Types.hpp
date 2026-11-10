#pragma once
//==============================================================================
//  Scalar types, Kokkos aliases, and the memory-layout policy.
//==============================================================================
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <utility>

namespace lbm {

//------------------------------------------------------------------------------
// Precision. Selected at configure time via -DLBM_PRECISION=float|double.
// One run uses one precision; there is no mixing, so a typedef beats templating
// every kernel and keeps instantiation counts (and compile time) down.
//------------------------------------------------------------------------------
#if defined(LBM_SINGLE_PRECISION)
using Real = float;
#else
using Real = double;
#endif

inline constexpr const char* precision_name() {
  return sizeof(Real) == 4 ? "FP32" : "FP64";
}

// Node index. int32 covers 2.1e9 nodes (~1290^3); widen here if that is ever
// not enough. Deliberately signed: neighbour arithmetic goes negative.
using Index = std::int32_t;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using HostSpace = Kokkos::DefaultHostExecutionSpace;

//------------------------------------------------------------------------------
// LAYOUT POLICY -- deliberately the Kokkos default, do not pin it.
//
// Population fields are declared View<Real**> with extents (node, q). With the
// default layout Kokkos gives:
//    CUDA/HIP -> LayoutLeft  -> node index fastest -> SoA -> coalesced
//    OpenMP/Serial/Threads -> LayoutRight -> q fastest -> AoS -> cache-local
// which is the right answer on both. Hardcoding either one costs performance on
// the other backend.
//------------------------------------------------------------------------------
template <class T> using View1D = Kokkos::View<T*,  MemSpace>;
template <class T> using View2D = Kokkos::View<T**, MemSpace>;

// Kokkos 5 dropped View::HostMirror; derive the mirror type from the factory
// so this works across 4.x and 5.x.
template <class V> using MirrorOf = decltype(Kokkos::create_mirror_view(std::declval<V>()));
template <class T> using HostView1D = MirrorOf<View1D<T>>;
template <class T> using HostView2D = MirrorOf<View2D<T>>;

using Range = Kokkos::RangePolicy<ExecSpace>;

//------------------------------------------------------------------------------
// The macroscopic state a collision policy reduces the populations to, and the
// one type the collision operators and the solvers BOTH name.
//
// HERE, NOT IN collision/BGK.hpp, where it lived until 2026-11-10. FluidSolver
// declares `const Macro m = coll.macroscopic(f, n)` in three kernels, and Macro
// is not a dependent name, so that header failed to parse unless some collision
// header had already been included -- "alphabetical include order fails to
// compile". Measured that day by compiling every header in src/ on its own
// (tests/headers_alone.sh): 48 of 49 passed and the forty-ninth failed on
// exactly this, three times, nothing else; all 49 pass since.
// tests/header_fluidsolver_alone.cpp keeps this one compiling at every build.
//
// `dens` is rho for raw storage and rho - 1 for shifted storage, and is NOT
// normalised to rho in the shifted case -- see collision/BGK.hpp's banner.
//------------------------------------------------------------------------------
struct Macro {
  Real dens;              // rho (raw) or rho - 1 (shifted)
  Real ux, uy, uz;
};

}  // namespace lbm
