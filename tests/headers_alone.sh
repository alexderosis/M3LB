#!/bin/sh
#==============================================================================
#  DOES EVERY HEADER IN src/ COMPILE ON ITS OWN?
#
#  One translation unit per header, holding that include and nothing else,
#  through -fsyntax-only. A header that fails here leans on something a caller
#  has to remember to include first, and that is a rule nobody can see from the
#  header.
#
#  WHY IT EXISTS. Until 2026-11-10 CLAUDE.md, ONBOARDING.md and the new-case
#  skill all said "headers are not self-contained; alphabetical include order
#  fails to compile". Run that day, this gave 48 of 49 passing: the whole rule
#  was one header, solver/FluidSolver.hpp, failing three times on one name,
#  `Macro`, which lived in collision/BGK.hpp. Macro moved to core/Types.hpp and
#  the count has been 49 of 49 since. To see the old failure again, move the
#  struct back.
#
#  WHY A SCRIPT AND NOT A ctest CASE. It is 49 compiles of Kokkos_Core.hpp --
#  about two minutes on four cores -- to guard something that changes when a
#  header's include list does, not on every commit. The one header that HAD the
#  defect is guarded at every build by tests/header_fluidsolver_alone.cpp; this
#  re-tries the rest by hand, like tests/frame_check.sh. Run it after adding a
#  header or moving a type between two.
#
#  WHAT IT DOES NOT DO: instantiate anything. A template that names a missing
#  DEPENDENT entity parses cleanly and fails only where it is used, so a pass
#  here means "parses alone", not "usable alone". And it checks src/ only;
#  GPU/'s headers are a different tree with a different rule (core.cuh first).
#
#  USAGE
#      tests/headers_alone.sh [<kokkos build dir>]     # default: build
#  Exit status is the number of headers that failed.
#==============================================================================
BUILD=${1:-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1

K=$BUILD/_deps
if [ ! -d "$K/kokkos-src" ]; then
  echo "no Kokkos sources under $K -- pass a configured Kokkos build dir" >&2
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0
for h in $(find src -name '*.hpp' | sort); do
  printf '#include "%s"\n' "${h#src/}" > "$TMP/one.cpp"
  if ${CXX:-c++} -std=c++20 -fsyntax-only "$TMP/one.cpp" \
      -Isrc \
      -I"$K/kokkos-build" -I"$K/kokkos-build/core/src" -I"$K/kokkos-src/core/src" \
      -I"$K/kokkos-build/containers/src" -I"$K/kokkos-src/containers/src" \
      -I"$K/kokkos-build/algorithms/src" -I"$K/kokkos-src/algorithms/src" \
      -I"$K/kokkos-build/simd/src" -I"$K/kokkos-src/simd/src" \
      -I"$K/kokkos-src/tpls/desul/include" -I"$K/kokkos-src/tpls/mdspan/include" \
      > "$TMP/out.txt" 2>&1; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    printf 'FAIL  %s\n' "$h"
    grep 'error' "$TMP/out.txt" | head -3 | sed 's/^/        /'
  fi
done
printf '%d of %d headers in src/ compile on their own\n' "$pass" "$((pass + fail))"
exit "$fail"
